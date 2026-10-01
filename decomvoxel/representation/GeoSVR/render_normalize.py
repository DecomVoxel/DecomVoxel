"""
Normalized Voxel Renderer for TRELLIS Conditioning

This module provides functionality for:
1. Loading and normalizing GeoSVR voxel data to [-1, 1] range
2. Generating camera views from multiple angles (sphere sampling)
3. Rendering object-only images with transparent background

The rendered images serve as conditioning input for TRELLIS SDS completion.
"""

import os
import sys
import cv2
import math
import numpy as np
import torch
from typing import List, Tuple, Optional, Dict
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from decomvoxel.representation.GeoSVR.voxel_to_mesh import voxel_to_mesh

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from yacs.config import CfgNode

from src.config import cfg
from src.dataloader.data_pack import DataPack
from src.sparse_voxel_model import SparseVoxelModel
from src.utils import activation_utils, octree_utils
from src.cameras import Camera

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


DEBUG = True


def sphere_hammersley_sequence(i: int, n: int, offset: Tuple[float, float] = (0.0, 0.0)) -> Tuple[float, float]:
    """
    Generate Hammersley sequence points on a sphere for uniform camera sampling.
    
    This produces quasi-random points that are more evenly distributed than
    pure random sampling, which is better for multi-view rendering.
    
    Args:
        i: index of the current sample
        n: total number of samples
        offset: random offset for variation (yaw_offset, pitch_offset)
        
    Returns:
        tuple of (yaw, pitch) in radians
    """
    def radical_inverse_base2(bits):
        bits = (bits << 16) | (bits >> 16)
        bits = ((bits & 0x55555555) << 1) | ((bits & 0xAAAAAAAA) >> 1)
        bits = ((bits & 0x33333333) << 2) | ((bits & 0xCCCCCCCC) >> 2)
        bits = ((bits & 0x0F0F0F0F) << 4) | ((bits & 0xF0F0F0F0) >> 4)
        bits = ((bits & 0x00FF00FF) << 8) | ((bits & 0xFF00FF00) >> 8)
        return float(bits) * 2.3283064365386963e-10  # / 0x100000000

    # Yaw: uniform around the sphere
    yaw = 2 * math.pi * ((i / n + offset[0]) % 1.0)
    
    # Pitch: use Hammersley sequence for better distribution
    pitch_param = (radical_inverse_base2(i) + offset[1]) % 1.0
    # Map to [-pi/3, pi/3] for reasonable viewing angles (not too extreme)
    pitch = math.asin(2 * pitch_param - 1) * 0.6  # 0.6 limits extreme angles
    
    return yaw, pitch


def generate_camera_views(num_views: int = 24, 
                          radius: float = 2.5,
                          fov: float = 40.0,
                          random_offset: bool = True,
                          roll: float = -90.0) -> List[Dict]:
    """
    Generate camera view parameters for multi-view rendering.
    
    Args:
        num_views: number of views to generate
        radius: distance from camera to origin
        fov: field of view in degrees
        random_offset: if True, add random offset for variation
        roll: camera roll angle in degrees (rotation around viewing direction), default 90 for clockwise rotation
        
    Returns:
        list of camera parameter dicts with keys:
            - yaw, pitch: camera angles in radians
            - radius: distance from origin
            - fov: field of view in degrees
            - position: camera position (x, y, z)
            - look_at: point camera looks at
            - roll: camera roll angle in degrees
    """
    if random_offset:
        offset = (np.random.rand(), np.random.rand())
    else:
        offset = (0.0, 0.0)
    
    views = []
    for i in range(num_views):
        yaw, pitch = sphere_hammersley_sequence(i, num_views, offset)
        
        # Compute camera position on sphere
        # Use Y-up convention (Y is vertical axis) to match GeoSVR/Replica scenes
        x = radius * math.cos(pitch) * math.cos(yaw)
        z = radius * math.cos(pitch) * math.sin(yaw)  # z instead of y for horizontal
        y = radius * math.sin(pitch)  # y is the vertical axis
        
        views.append({
            'yaw': yaw,
            'pitch': pitch,
            'radius': radius,
            'fov': fov,
            'position': np.array([x, y, z]),
            'look_at': np.array([0.0, 0.0, 0.0]),
            'roll': roll
        })
    
    return views


class ObjectVoxelRenderer:
    """
    Renderer for single-object voxels with transparent background.
    
    This class:
    1. Loads object voxel data
    2. Creates a minimal SparseVoxelModel for the object
    3. Renders from multiple views with object-only visibility
    """
    
    def __init__(self, output_dir: str, device: torch.device = device):
        """
        Args:
            output_dir: directory to save rendered images
            device: torch device for computation
        """
        self.output_dir = output_dir
        self.device = device
        # self.normalizer = VoxelNormalizer()
        
        os.makedirs(output_dir, exist_ok=True)
    
    def create_camera(self, view: Dict, image_size: int = 512) -> Camera:
        """
        Create a Camera object from view parameters.
        
        Uses GeoSVR's standard Camera constructor signature.
        
        Args:
            view: camera view dict with position, look_at, fov
            image_size: output image resolution
            
        Returns:
            Camera object for rendering
        """
        position = torch.tensor(view['position'], dtype=torch.float32, device=self.device)
        look_at = torch.tensor(view['look_at'], dtype=torch.float32, device=self.device)
        
        # Compute camera extrinsics (world to camera)
        # GeoSVR camera convention (from camera_utils.py):
        # - Column 0: right
        # - Column 1: down (not up!)
        # - Column 2: lookat (camera looks along +Z, not -Z)
        lookat = look_at - position
        lookat = lookat / torch.norm(lookat)
        
        # World up vector (Y-up convention to match GeoSVR/Replica scenes)
        world_up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=self.device)
        
        # Handle case where lookat is parallel to world_up
        if abs(torch.dot(lookat, world_up)) > 0.99:
            world_up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=self.device)
        
        # Compute right and down vectors
        right = torch.linalg.cross(lookat, world_up)
        right = right / torch.norm(right)
        down = torch.linalg.cross(lookat, right)  # down = lookat x right
        
        # Apply roll rotation (rotation around viewing direction/lookat axis)
        # For clockwise roll when looking along +Z (from camera's perspective)
        if 'roll' in view and view['roll'] != 0:
            roll_rad = view['roll'] * math.pi / 180.0
            # Rotation matrix around lookat axis (z-axis of camera)
            # For clockwise rotation (when looking along +z): R = [cos(θ), sin(θ), 0; -sin(θ), cos(θ), 0; 0, 0, 1]
            cos_roll = math.cos(roll_rad)
            sin_roll = math.sin(roll_rad)
            # New right = cos(roll) * right + sin(roll) * down
            # New down = -sin(roll) * right + cos(roll) * down
            right_rotated = cos_roll * right + sin_roll * down
            down_rotated = -sin_roll * right + cos_roll * down
            right = right_rotated / torch.norm(right_rotated)
            down = down_rotated / torch.norm(down_rotated)
        
        # Build rotation matrix (camera to world)
        # Column 0: right, Column 1: down, Column 2: lookat
        c2w_rot = torch.stack([right, down, lookat], dim=1)  # (3, 3)
        
        # Build full c2w matrix
        c2w = torch.eye(4, dtype=torch.float32, device=self.device)
        c2w[:3, :3] = c2w_rot
        c2w[:3, 3] = position
        
        # Invert to get w2c
        w2c = torch.inverse(c2w)
        
        # Convert to numpy for Camera constructor (GeoSVR expects numpy)
        w2c_np = w2c.cpu().numpy()
        R = c2w_rot.T.cpu().numpy()  # Camera expects R transpose for w2c rotation
        T = position.cpu().numpy()   # T is camera position in world coords
        
        # Compute FOV
        fov_rad = view['fov'] * math.pi / 180.0
        
        # Create dummy image tensor (C, H, W) - required by Camera constructor
        dummy_image = torch.zeros((3, image_size, image_size), dtype=torch.float32)
        
        # Create camera using GeoSVR's standard Camera constructor
        # This matches the signature from segm_3d.py's usage
        camera = Camera(
            image_name="render",
            w2c=w2c_np,
            fovx=fov_rad,
            fovy=fov_rad,
            cx_p=0.5,
            cy_p=0.5,
            R=R,
            T=T,
            near=0.01,
            image=dummy_image,
            mask=None,
            depth=None
        )
        
        return camera
    
    def visualize_camera_setup(self, voxel_model: SparseVoxelModel, views: List[Dict], 
                              camera_look_at: np.ndarray, save_path: str):
        """
        Visualize camera positions, voxel center, and bbox in top-down and 3D views.
        
        Args:
            voxel_model: SparseVoxelModel instance
            views: list of camera view dicts
            camera_look_at: center point cameras are looking at
            save_path: path to save visualization image
        """
        with torch.no_grad():
            # Get voxel data
            vox_centers = voxel_model.vox_center.cpu().numpy()  # (N, 3)
            voxel_center = vox_centers.mean(axis=0)  # (3,)
            
            # Compute bbox
            bbox_min = vox_centers.min(axis=0)
            bbox_max = vox_centers.max(axis=0)
            
            # 8 corners of bbox
            bbox_corners = np.array([
                [bbox_min[0], bbox_min[1], bbox_min[2]],
                [bbox_max[0], bbox_min[1], bbox_min[2]],
                [bbox_max[0], bbox_max[1], bbox_min[2]],
                [bbox_min[0], bbox_max[1], bbox_min[2]],
                [bbox_min[0], bbox_min[1], bbox_max[2]],
                [bbox_max[0], bbox_min[1], bbox_max[2]],
                [bbox_max[0], bbox_max[1], bbox_max[2]],
                [bbox_min[0], bbox_max[1], bbox_max[2]],
            ])
            
            # Extract camera positions
            camera_positions = np.array([view['position'] for view in views])  # (N, 3)
            
        # Create figure with 2 subplots
        fig = plt.figure(figsize=(16, 8))
        
        # ========== Subplot 1: Top-down view (X-Y plane) ==========
        ax1 = fig.add_subplot(121)
        
        # Plot voxel center
        ax1.scatter(voxel_center[0], voxel_center[1], c='red', s=200, marker='*', 
                   label='Voxel Center', zorder=5, edgecolors='black', linewidths=2)
        
        # Plot camera look-at point
        ax1.scatter(camera_look_at[0], camera_look_at[1], c='orange', s=150, marker='x', 
                   label='Camera Look-At', zorder=5, linewidths=3)
        
        # Plot bbox corners (top face)
        bbox_xy = bbox_corners[:, :2]  # Project to XY
        # Draw bbox edges (top face)
        top_face = [0, 1, 2, 3, 0]  # indices for top face
        ax1.plot(bbox_corners[top_face, 0], bbox_corners[top_face, 1], 
                'g--', linewidth=2, label='BBox (top)', alpha=0.6)
        # Draw bbox edges (bottom face)
        bottom_face = [4, 5, 6, 7, 4]
        ax1.plot(bbox_corners[bottom_face, 0], bbox_corners[bottom_face, 1], 
                'g:', linewidth=2, label='BBox (bottom)', alpha=0.4)
        
        # Plot camera positions
        ax1.scatter(camera_positions[:, 0], camera_positions[:, 1], 
                   c='blue', s=80, marker='^', label='Cameras', zorder=4, alpha=0.7)
        
        # Draw lines from cameras to look-at point
        for i, cam_pos in enumerate(camera_positions):
            ax1.plot([cam_pos[0], camera_look_at[0]], 
                    [cam_pos[1], camera_look_at[1]], 
                    'b-', alpha=0.2, linewidth=0.5)
            # Label first few cameras
            if i < 3:
                ax1.text(cam_pos[0], cam_pos[1], f'  Cam{i}', 
                        fontsize=8, ha='left', va='bottom')
        
        ax1.set_xlabel('X', fontsize=12)
        ax1.set_ylabel('Y', fontsize=12)
        ax1.set_title('Top-Down View (X-Y Plane)', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')
        
        # ========== Subplot 2: 3D view ==========
        ax2 = fig.add_subplot(122, projection='3d')
        
        # Plot voxel center
        ax2.scatter(voxel_center[0], voxel_center[1], voxel_center[2], 
                   c='red', s=200, marker='*', label='Voxel Center', 
                   edgecolors='black', linewidths=2)
        
        # Plot camera look-at point
        ax2.scatter(camera_look_at[0], camera_look_at[1], camera_look_at[2], 
                   c='orange', s=150, marker='x', label='Camera Look-At', linewidths=3)
        
        # Draw bbox wireframe
        # Define the 12 edges of the bbox
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # bottom face
            [4, 5], [5, 6], [6, 7], [7, 4],  # top face
            [0, 4], [1, 5], [2, 6], [3, 7],  # vertical edges
        ]
        for edge in edges:
            points = bbox_corners[edge]
            ax2.plot3D(points[:, 0], points[:, 1], points[:, 2], 
                      'g-', linewidth=1.5, alpha=0.6)
        
        # Plot camera positions
        ax2.scatter(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2],
                   c='blue', s=80, marker='^', label='Cameras', alpha=0.7)
        
        # Draw lines from cameras to look-at point
        for i, cam_pos in enumerate(camera_positions):
            ax2.plot3D([cam_pos[0], camera_look_at[0]], 
                      [cam_pos[1], camera_look_at[1]], 
                      [cam_pos[2], camera_look_at[2]], 
                      'b-', alpha=0.2, linewidth=0.5)
        
        ax2.set_xlabel('X', fontsize=10)
        ax2.set_ylabel('Y', fontsize=10)
        ax2.set_zlabel('Z', fontsize=10)
        ax2.set_title('3D View', fontsize=14, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=9)
        
        # Set equal aspect ratio
        max_range = np.array([bbox_max[0]-bbox_min[0], 
                             bbox_max[1]-bbox_min[1], 
                             bbox_max[2]-bbox_min[2]]).max() / 2.0
        mid_x = (bbox_max[0] + bbox_min[0]) * 0.5
        mid_y = (bbox_max[1] + bbox_min[1]) * 0.5
        mid_z = (bbox_max[2] + bbox_min[2]) * 0.5
        ax2.set_xlim(mid_x - max_range*1.5, mid_x + max_range*1.5)
        ax2.set_ylim(mid_y - max_range*1.5, mid_y + max_range*1.5)
        ax2.set_zlim(mid_z - max_range*1.5, mid_z + max_range*1.5)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[Debug] Camera setup visualization saved to {save_path}")
    
    def render_views(self, voxel_model: SparseVoxelModel,
                     num_views: int = 24,
                     image_size: int = 512,
                     normalize: bool = True) -> List[np.ndarray]:
      
        # Compute voxel center before normalization (for camera positioning)
        with torch.no_grad():
            voxel_center = voxel_model.vox_center.mean(dim=0).cpu().numpy()  # (3,)
        
        # Normalize if requested
        # if normalize:
        #     voxel_model = self.normalizer.normalize(voxel_model)
        #     # After normalization, voxels are centered at origin
        #     camera_look_at = np.array([0.0, 0.0, 0.0])
        # else:
        #     # Without normalization, camera should look at voxel center
        camera_look_at = voxel_center
        
        voxel_model.freeze_vox_geo()
        
        # Compute appropriate camera radius based on object size
        with torch.no_grad():
            bbox_min = voxel_model.vox_center.min(dim=0).values
            bbox_max = voxel_model.vox_center.max(dim=0).values
            bbox_extent = bbox_max - bbox_min
            max_extent = bbox_extent.max().item()
            
            # Camera radius should be ~2-3x the object size for good framing
            # camera_radius = max_extent * 2.5
            camera_radius = max_extent * 1.5
            
            if DEBUG:
                print(f"\n[Debug] Camera Setup:")
                print(f"  - Object bbox extent: {bbox_extent.cpu().numpy()}")
                print(f"  - Max extent: {max_extent:.4f}")
                print(f"  - Camera radius: {camera_radius:.4f}")
                print(f"  - Camera look_at (voxel center): {camera_look_at}")
        
        views = generate_camera_views(num_views=num_views, radius=camera_radius, fov=45.0, random_offset=True, roll=-90.0)
        
        # If not normalized, offset all camera positions to look at voxel center
        if not normalize:
            for view in views:
                view['position'] = view['position'] + camera_look_at
                view['look_at'] = camera_look_at
        
        # Debug: visualize camera setup
        if DEBUG:
            vis_path = os.path.join(self.output_dir, 'camera_setup_visualization.png')
            self.visualize_camera_setup(voxel_model, views, camera_look_at, vis_path)
        
        rendered_images = []
        
        with torch.no_grad():
            for i, view in enumerate(tqdm(views, desc="Rendering views")):
                camera = self.create_camera(view, image_size)
                
                # Render with white background
                voxel_model.bg_color = torch.tensor([1.0, 1.0, 1.0], device=self.device)
                res = voxel_model.render(camera, color_mode=None, output_T=True)
                
                color = res['color']  # (3, H, W)
                transmittance = res.get('T', None)  # (1, H, W) if available
                
                # Debug: check render output for first view
                # if DEBUG and i == 0:
                #     print("\n[Debug] Render Output Check (View 0):")
                #     print(f"  - color shape: {color.shape}")
                #     print(f"  - color range: [{color.min().item():.4f}, {color.max().item():.4f}]")
                #     print(f"  - color mean: {color.mean().item():.4f}")
                #     if transmittance is not None:
                #         print(f"  - transmittance shape: {transmittance.shape}")
                #         print(f"  - transmittance range: [{transmittance.min().item():.4f}, {transmittance.max().item():.4f}]")
                #         print(f"  - transmittance mean: {transmittance.mean().item():.4f}")
                #         # Count non-background pixels
                #         alpha = 1.0 - transmittance
                #         visible_pixels = (alpha > 0.01).sum().item()
                #         total_pixels = alpha.numel()
                #         print(f"  - Visible pixels (alpha > 0.01): {visible_pixels} / {total_pixels} "
                #               f"({100*visible_pixels/total_pixels:.2f}%)")
                #         if visible_pixels == 0:
                #             print("  ⚠️ WARNING: No visible pixels! Rays are not hitting any voxels!")
                            
                #             # Additional diagnostics
                #             print("\n  Additional diagnostics:")
                #             cam_pos = camera.position
                #             print(f"    - Camera position (from camera): {cam_pos.cpu().numpy()}")
                #             print(f"    - Camera c2w[:3,3]: {camera.c2w[:3,3].cpu().numpy()}")
                #             print(f"    - Camera w2c[:3,3]: {camera.w2c[:3,3].cpu().numpy()}")
                            
                #             # Check distance from camera to voxels
                #             vox_centers = voxel_model.vox_center
                #             distances = torch.norm(vox_centers - cam_pos.view(1, 3), dim=1)
                #             print(f"    - Distance to voxels: min={distances.min().item():.4f}, max={distances.max().item():.4f}, mean={distances.mean().item():.4f}")
                            
                #             # Check if voxels are in front of camera
                #             cam_forward = torch.tensor(view['look_at'], device=self.device) - torch.tensor(view['position'], device=self.device)
                #             cam_forward = cam_forward / torch.norm(cam_forward)
                #             vox_dirs = vox_centers - cam_pos.view(1, 3)
                #             vox_dirs = vox_dirs / torch.norm(vox_dirs, dim=1, keepdim=True)
                #             dot_products = (vox_dirs * cam_forward.view(1, 3)).sum(dim=1)
                #             in_front = (dot_products > 0).sum().item()
                #             print(f"    - Voxels in front of camera: {in_front} / {len(vox_centers)}")
                            
                #             # Check frozen vox geo
                #             if hasattr(voxel_model, 'frozen_vox_geo'):
                #                 print(f"    - frozen_vox_geo shape: {voxel_model.frozen_vox_geo.shape}")
                #                 print(f"    - frozen_vox_geo range: [{voxel_model.frozen_vox_geo.min().item():.4f}, {voxel_model.frozen_vox_geo.max().item():.4f}]")
                #     else:
                #         print("  - transmittance: None (not returned)")
                    
                #     # Check camera params
                #     print(f"\n  Camera position: {view['position']}")
                #     print(f"  Camera look_at: {view['look_at']}")
                #     print(f"  Camera fov: {view['fov']}°")
                
                # Convert to numpy (H, W, 3)
                img_rgb = (color.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                
                # Create alpha channel from transmittance or color
                if transmittance is not None:
                    # Alpha = 1 - T (where T is transmittance)
                    alpha = ((1.0 - transmittance).permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                else:
                    # Estimate alpha from white background: pixels that are pure white are transparent
                    white_threshold = 250
                    is_white = np.all(img_rgb >= white_threshold, axis=2)
                    alpha = np.where(is_white, 0, 255).astype(np.uint8)
                    alpha = alpha[:, :, np.newaxis]
                
                # Combine to RGBA
                img_rgba = np.concatenate([img_rgb, alpha], axis=2)
                rendered_images.append(img_rgba)
                
                # Save image
                save_path = os.path.join(self.output_dir, f"view_{i:03d}.png")
                cv2.imwrite(save_path, cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGRA))
        
        return rendered_images


def create_object_voxel_model(voxel_path: str, 
                               model_path: str = None) -> SparseVoxelModel:
    cfg_model = CfgNode()
    cfg_model.model_path = model_path or os.path.dirname(voxel_path)
    cfg_model.vox_geo_mode = "triinterp1" # geometry interpolation mode
    cfg_model.density_mode = "exp_linear_11" # density activation 
    cfg_model.sh_degree = 3 # Spherical Harmonics degree
    cfg_model.ss = 1.5 # supersampling factor, control ray marching density
    cfg_model.outside_level = 5 # controls background region depth
    cfg_model.white_background = True 
    cfg_model.black_background = False
    
    voxel_model = SparseVoxelModel(cfg_model)
    voxel_model.load(voxel_path)
    
    return voxel_model


def render_normalized_voxel(voxel_path: str,
                            output_dir: str,
                            num_views: int = 24,
                            image_size: int = 512) -> List[str]:
    """
    Main function to render normalized voxel views.
    This is the primary interface for rendering conditioning images
    for TRELLIS SDS completion.
    """
    print("=" * 60)
    print("Normalized Voxel Renderer")
    print("=" * 60)
    print(f"Input: {voxel_path}")
    print(f"Output: {output_dir}")
    print(f"Views: {num_views}")
    print(f"Resolution: {image_size}x{image_size}")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create voxel model from saved data
    print("[Render] Loading voxel model...")
    voxel_model = create_object_voxel_model(voxel_path)
    
    # Create renderer and render views
    renderer = ObjectVoxelRenderer(output_dir=output_dir)
    rendered_images = renderer.render_views(
        voxel_model=voxel_model,
        num_views=num_views,
        image_size=image_size,
        normalize=False
    )
    
    # Get list of saved image paths
    image_paths = [
        os.path.join(output_dir, f"view_{i:03d}.png")
        for i in range(num_views)
    ]
    
    print(f"\n[Render] Saved {len(image_paths)} images to {output_dir}")
    
    return image_paths



if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Render normalized voxel views")
    parser.add_argument('--voxel_path', type=str, 
                        default='../../../datasets/replica_large/scan1/semantic_result/object_voxels/object_002_voxels.pt',
                        help='Path to object voxel .pt file')
    parser.add_argument('--output_dir', type=str,
                        default='../../../outputs/render_normalize',
                        help='Output directory for rendered images')
    parser.add_argument('--num_views', type=int, default=24,
                        help='Number of views to render')
    parser.add_argument('--image_size', type=int, default=512,
                        help='Output image resolution')
    
    args = parser.parse_args()
    
    # Make paths absolute
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(args.voxel_path):
        args.voxel_path = os.path.abspath(os.path.join(script_dir, args.voxel_path))
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.abspath(os.path.join(script_dir, args.output_dir))
    
    # Run rendering
    render_normalized_voxel(
        voxel_path=args.voxel_path,
        output_dir=args.output_dir,
        num_views=args.num_views,
        image_size=args.image_size
    )
