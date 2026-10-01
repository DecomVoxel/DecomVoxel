"""
HashGrid Visualization and Conversion Pipeline

This module provides utilities for:
1. Loading a trained HashGridVoxel model from checkpoint
2. Visualizing the hash grid as 2D images (slices and projections)
3. Converting HashGridVoxel back to dense voxel grid
4. Converting voxels to mesh using GeoSVR's voxel_to_mesh utilities

The visualization pipeline is designed to work with the DecomVoxel pipeline,
reusing existing methods from load_voxel.py, hash_grid.py, and voxel_to_mesh.py.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, Union, List
import matplotlib.pyplot as plt
from PIL import Image

# Add project paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

GEOSVR_ROOT = os.path.join(PROJECT_ROOT, 'decomvoxel', 'representation', 'GeoSVR')
if GEOSVR_ROOT not in sys.path:
    sys.path.insert(0, GEOSVR_ROOT)

TRELLIS_ROOT = os.path.join(PROJECT_ROOT, 'decomvoxel', 'model', 'TRELLIS')
if TRELLIS_ROOT not in sys.path:
    sys.path.insert(0, TRELLIS_ROOT)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HashGridVisualizer:
    """
    Visualizer for HashGridVoxel models.
    
    This class provides methods to:
    - Load HashGridVoxel model from checkpoint
    - Convert to dense voxel grid
    - Visualize as 2D images
    - Export to mesh format
    """
    
    def __init__(self, resolution: int = 64, device: torch.device = device):
        """
        Initialize the visualizer.
        
        Args:
            resolution: voxel grid resolution (default: 64)
            device: torch device for computation
        """
        self.resolution = resolution
        self.device = device
        self.model = None
    
    def load_model(self, checkpoint_path: str) -> nn.Module:
        """
        Load a HashGridVoxel model from checkpoint.
        
        Args:
            checkpoint_path: path to the .pth checkpoint file
            
        Returns:
            loaded HashGridVoxel model
        """
        from network.hash_grid import HashGridVoxel
        
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"[HashGridVisualizer] Loading model from: {checkpoint_path}")
        
        model = HashGridVoxel().to(self.device)
        state_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        
        self.model = model
        print(f"[HashGridVisualizer] Model loaded successfully")
        
        return model
    
    def to_dense_voxels(self, threshold: float = 0.5) -> torch.Tensor:
        """
        Convert HashGridVoxel to dense 64^3 voxel grid.
        
        Args:
            threshold: occupancy threshold (default: 0.5)
            
        Returns:
            (64, 64, 64) dense occupancy tensor
        """
        from network.hash_grid import generate_image_grid_3d
        
        if self.model is None:
            raise RuntimeError("No model loaded. Call load_model first.")
        
        print(f"[HashGridVisualizer] Converting to dense voxel grid...")
        
        with torch.no_grad():
            # Generate 3D grid coordinates
            grid = generate_image_grid_3d(self.resolution).to(self.device)
            grid = grid.view(-1, 3)
            
            # Query the model
            occupancy = self.model(grid)  # (N, 1) or (1, 64, 64, 64)
            
            # Reshape to grid
            if occupancy.dim() == 2:
                occupancy = occupancy.view(self.resolution, self.resolution, self.resolution)
            elif occupancy.dim() == 4:
                occupancy = occupancy.squeeze(0)
            
            # Binarize
            voxels = (occupancy > threshold).float()
            
            occupied = voxels.sum().item()
            total = self.resolution ** 3
            print(f"[HashGridVisualizer] Occupied voxels: {occupied:.0f} / {total} ({100*occupied/total:.2f}%)")
            
            return voxels
    
    def to_occupancy_grid(self) -> torch.Tensor:
        """
        Get raw occupancy values (continuous [0, 1]) without thresholding.
        
        Returns:
            (64, 64, 64) occupancy tensor with values in [0, 1]
        """
        from network.hash_grid import generate_image_grid_3d
        
        if self.model is None:
            raise RuntimeError("No model loaded. Call load_model first.")
        
        with torch.no_grad():
            grid = generate_image_grid_3d(self.resolution).to(self.device)
            grid = grid.view(-1, 3)
            
            occupancy = self.model(grid)
            
            if occupancy.dim() == 2:
                occupancy = occupancy.view(self.resolution, self.resolution, self.resolution)
            elif occupancy.dim() == 4:
                occupancy = occupancy.squeeze(0)
            
            return occupancy.squeeze()
    
    def visualize_slices(self, save_path: str, 
                         num_slices: int = 8,
                         axis: str = 'z') -> str:
        """
        Visualize voxel grid as 2D slices.
        
        Args:
            save_path: path to save the visualization image
            num_slices: number of slices to show
            axis: axis to slice along ('x', 'y', or 'z')
            
        Returns:
            path to saved image
        """
        voxels = self.to_occupancy_grid().cpu().numpy()
        
        # Get slice indices
        axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis]
        res = voxels.shape[axis_idx]
        slice_indices = np.linspace(0, res - 1, num_slices, dtype=int)
        
        # Create figure
        cols = min(4, num_slices)
        rows = (num_slices + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
        axes = np.atleast_2d(axes).flatten()
        
        for i, idx in enumerate(slice_indices):
            if axis == 'x':
                slice_data = voxels[idx, :, :]
            elif axis == 'y':
                slice_data = voxels[:, idx, :]
            else:
                slice_data = voxels[:, :, idx]
            
            axes[i].imshow(slice_data, cmap='hot', vmin=0, vmax=1)
            axes[i].set_title(f'{axis.upper()}={idx}')
            axes[i].axis('off')
        
        # Hide empty axes
        for i in range(len(slice_indices), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[HashGridVisualizer] Slice visualization saved to: {save_path}")
        return save_path
    
    def visualize_projections(self, save_path: str) -> str:
        """
        Visualize voxel grid as maximum intensity projections along each axis.
        
        Args:
            save_path: path to save the visualization image
            
        Returns:
            path to saved image
        """
        voxels = self.to_occupancy_grid().cpu().numpy()
        
        # Create projections
        proj_x = voxels.max(axis=0)  # YZ plane
        proj_y = voxels.max(axis=1)  # XZ plane
        proj_z = voxels.max(axis=2)  # XY plane
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        axes[0].imshow(proj_x, cmap='hot', vmin=0, vmax=1)
        axes[0].set_title('Max Projection (X-axis)')
        axes[0].axis('off')
        
        axes[1].imshow(proj_y, cmap='hot', vmin=0, vmax=1)
        axes[1].set_title('Max Projection (Y-axis)')
        axes[1].axis('off')
        
        axes[2].imshow(proj_z, cmap='hot', vmin=0, vmax=1)
        axes[2].set_title('Max Projection (Z-axis)')
        axes[2].axis('off')
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[HashGridVisualizer] Projection visualization saved to: {save_path}")
        return save_path
    
    def visualize_3d(self, save_path: str, threshold: float = 0.5) -> str:
        """
        Visualize voxel grid as 3D scatter plot using plotly.
        
        Args:
            save_path: path to save the visualization
            threshold: occupancy threshold for visualization
            
        Returns:
            path to saved image/html
        """
        from decomvoxel.utils.visualization import visualize_voxels
        
        voxels = self.to_dense_voxels(threshold=threshold).cpu().numpy()
        visualize_voxels(voxels, save_path)
        
        print(f"[HashGridVisualizer] 3D visualization saved to: {save_path}")
        return save_path
    
    def save_dense_voxels(self, save_path: str, threshold: float = 0.5) -> str:
        """
        Save dense voxel grid to a .pt file.
        
        Args:
            save_path: path to save the voxel grid
            threshold: occupancy threshold
            
        Returns:
            path to saved file
        """
        voxels = self.to_dense_voxels(threshold=threshold)
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(voxels, save_path)
        
        print(f"[HashGridVisualizer] Dense voxels saved to: {save_path}")
        return save_path


class HashGridToMeshConverter:
    """
    Converter from HashGridVoxel to mesh format.
    
    This class leverages the existing voxel_to_mesh utilities from GeoSVR
    to convert HashGridVoxel representations to mesh format.
    """
    
    def __init__(self, resolution: int = 64, device: torch.device = device):
        """
        Initialize the converter.
        
        Args:
            resolution: voxel grid resolution
            device: torch device
        """
        self.resolution = resolution
        self.device = device
        self.visualizer = HashGridVisualizer(resolution=resolution, device=device)
    
    def convert(self, hashgrid_path: str, 
                output_path: str,
                format: str = 'glb',
                threshold: float = 0.5,
                color: Tuple[float, float, float] = (1.0, 0.5, 0.0)) -> str:
        """
        Convert HashGridVoxel checkpoint to mesh.
        
        Args:
            hashgrid_path: path to HashGridVoxel .pth file
            output_path: path to save the mesh
            format: output format ('glb', 'obj', 'ply', 'stl')
            threshold: occupancy threshold
            color: RGB color for the mesh (default: orange)
            
        Returns:
            path to saved mesh
        """
        # Import mesh conversion utilities
        from voxel_to_mesh import voxels_to_mesh_vectorized, create_unit_cube
        
        try:
            import trimesh
        except ImportError:
            raise ImportError("Please install trimesh: pip install trimesh")
        
        print(f"[HashGridToMesh] Loading HashGrid from: {hashgrid_path}")
        
        # Load the model
        self.visualizer.load_model(hashgrid_path)
        
        # Get dense voxels
        voxels = self.visualizer.to_dense_voxels(threshold=threshold)
        
        # Get occupied voxel coordinates
        occupied_coords = torch.nonzero(voxels, as_tuple=False)  # (M, 3)
        n_occupied = len(occupied_coords)
        
        if n_occupied == 0:
            print("[HashGridToMesh] Warning: No occupied voxels found!")
            return None
        
        print(f"[HashGridToMesh] Converting {n_occupied} occupied voxels to mesh...")
        
        # Convert grid indices to normalized coordinates [-0.5, 0.5]
        centers = (occupied_coords.float() / self.resolution) - 0.5 + (0.5 / self.resolution)
        sizes = torch.ones(n_occupied, device=self.device) / self.resolution
        colors = torch.tensor([color], device=self.device).expand(n_occupied, 3)
        
        # Use vectorized conversion
        mesh = voxels_to_mesh_vectorized(
            centers=centers.cpu().numpy(),
            sizes=sizes.cpu().numpy(),
            colors=colors.cpu().numpy()
        )
        
        # Ensure output path has correct extension
        if not output_path.endswith(f'.{format}'):
            output_path = f"{os.path.splitext(output_path)[0]}.{format}"
        
        # Save mesh
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        mesh.export(output_path)
        
        # Get file size
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"[HashGridToMesh] Mesh saved to: {output_path} ({file_size:.2f} MB)")
        
        return output_path


def visualize_hashgrid(hashgrid_path: str,
                       output_dir: str,
                       threshold: float = 0.5,
                       export_mesh: bool = True,
                       mesh_format: str = 'glb') -> Dict[str, str]:
    """
    Full visualization pipeline for a HashGridVoxel model.
    
    Args:
        hashgrid_path: path to HashGridVoxel .pth checkpoint
        output_dir: directory to save all outputs
        threshold: occupancy threshold
        export_mesh: if True, also export to mesh format
        mesh_format: mesh output format
        
    Returns:
        dict with paths to all generated files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(hashgrid_path))[0]
    
    outputs = {}
    
    # Initialize visualizer
    visualizer = HashGridVisualizer(device=device)
    visualizer.load_model(hashgrid_path)
    
    # 1. Save slice visualization
    slice_path = os.path.join(output_dir, f"{base_name}_slices.png")
    visualizer.visualize_slices(slice_path, num_slices=8, axis='z')
    outputs['slices'] = slice_path
    
    # 2. Save projection visualization
    proj_path = os.path.join(output_dir, f"{base_name}_projections.png")
    visualizer.visualize_projections(proj_path)
    outputs['projections'] = proj_path
    
    # 3. Save 3D visualization
    viz3d_path = os.path.join(output_dir, f"{base_name}_3d.png")
    visualizer.visualize_3d(viz3d_path, threshold=threshold)
    outputs['visualization_3d'] = viz3d_path
    
    # 4. Save dense voxels
    voxel_path = os.path.join(output_dir, f"{base_name}_dense_voxels.pt")
    visualizer.save_dense_voxels(voxel_path, threshold=threshold)
    outputs['dense_voxels'] = voxel_path
    
    # 5. Export to mesh
    if export_mesh:
        converter = HashGridToMeshConverter(device=device)
        mesh_path = os.path.join(output_dir, f"{base_name}_mesh.{mesh_format}")
        converter.convert(hashgrid_path, mesh_path, format=mesh_format, threshold=threshold)
        outputs['mesh'] = mesh_path
    
    print(f"\n[HashGridVisualize] All outputs saved to: {output_dir}")
    for key, path in outputs.items():
        print(f"  - {key}: {os.path.basename(path)}")
    
    return outputs


def hashgrid_to_mesh(hashgrid_path: str,
                     output_path: str,
                     format: str = 'glb',
                     threshold: float = 0.5,
                     color: Tuple[float, float, float] = (1.0, 0.5, 0.0)) -> str:
    """
    Convenience function to convert HashGridVoxel to mesh.
    
    Args:
        hashgrid_path: path to HashGridVoxel .pth file
        output_path: path to save the mesh
        format: output format
        threshold: occupancy threshold
        color: RGB color for mesh
        
    Returns:
        path to saved mesh
    """
    converter = HashGridToMeshConverter(device=device)
    return converter.convert(hashgrid_path, output_path, format=format, 
                            threshold=threshold, color=color)


# ============================================================================
# Main entry point for command-line usage
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Visualize and convert HashGridVoxel models"
    )
    
    parser.add_argument('--hashgrid', type=str, required=True,
                        help='Path to HashGridVoxel .pth checkpoint')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: same as hashgrid)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Occupancy threshold (default: 0.5)')
    parser.add_argument('--export_mesh', action='store_true', default=True,
                        help='Export to mesh format')
    parser.add_argument('--no_mesh', action='store_false', dest='export_mesh',
                        help='Skip mesh export')
    parser.add_argument('--mesh_format', type=str, default='glb',
                        choices=['glb', 'obj', 'ply', 'stl'],
                        help='Mesh output format (default: glb)')
    
    args = parser.parse_args()
    
    # Set default output dir
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.hashgrid)
    
    # Make paths absolute
    if not os.path.isabs(args.hashgrid):
        args.hashgrid = os.path.join(PROJECT_ROOT, args.hashgrid)
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(PROJECT_ROOT, args.output_dir)
    
    # Run visualization pipeline
    outputs = visualize_hashgrid(
        hashgrid_path=args.hashgrid,
        output_dir=args.output_dir,
        threshold=args.threshold,
        export_mesh=args.export_mesh,
        mesh_format=args.mesh_format
    )
    
    print("\n[Done] Visualization complete!")
