"""
Voxel to Mesh Converter for Blender Visualization

This script converts SVRaster voxel checkpoints to 3D mesh format (OBJ/PLY/GLB)
that can be directly opened in Blender for visualization.

Features:
- Load voxel checkpoint and extract geometry
- Convert each voxel to a cube mesh
- Parallel processing for efficiency
- Export to Blender-compatible formats (OBJ, PLY, GLB)
- Support for voxel colors from SH coefficients

Usage:
    python dataset_tools/voxel_to_mesh.py --checkpoint outputs/Replica/scan1/checkpoints/iter030000_model.pt --output output.glb
    
TODO: Some functionality here can be moved into GeoSVR classes (e.g., add mesh export in io).
    
"""

import os
import sys
import argparse
import numpy as np
import torch
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from yacs.config import CfgNode
from src.sparse_voxel_model import SparseVoxelModel
from src.utils.activation_utils import shzero2rgb

try:
    import trimesh
except ImportError:
    print("Please install trimesh: pip install trimesh")
    sys.exit(1)


def create_unit_cube():
    """
    Create a unit cube mesh centered at origin with size 1.
    Returns vertices and faces arrays.
    """
    # 8 vertices of a unit cube centered at origin
    vertices = np.array([
        [-0.5, -0.5, -0.5],
        [ 0.5, -0.5, -0.5],
        [ 0.5,  0.5, -0.5],
        [-0.5,  0.5, -0.5],
        [-0.5, -0.5,  0.5],
        [ 0.5, -0.5,  0.5],
        [ 0.5,  0.5,  0.5],
        [-0.5,  0.5,  0.5],
    ], dtype=np.float32)
    
    # 12 triangular faces (2 per cube face)
    faces = np.array([
        # Bottom
        [0, 2, 1], [0, 3, 2],
        # Top
        [4, 5, 6], [4, 6, 7],
        # Front
        [0, 1, 5], [0, 5, 4],
        # Back
        [2, 3, 7], [2, 7, 6],
        # Left
        [0, 4, 7], [0, 7, 3],
        # Right
        [1, 2, 6], [1, 6, 5],
    ], dtype=np.int32)
    
    return vertices, faces


def transform_cube(unit_verts, center, size):
    """
    Transform unit cube vertices to target position and size.
    
    Args:
        unit_verts: (8, 3) unit cube vertices
        center: (3,) center position
        size: scalar or (3,) size
        
    Returns:
        (8, 3) transformed vertices
    """
    return unit_verts * size + center


def process_voxel_batch(args):
    """
    Process a batch of voxels and create mesh data.
    
    Args:
        args: tuple of (batch_centers, batch_sizes, batch_colors, unit_verts, unit_faces, start_idx)
        
    Returns:
        tuple of (vertices, faces, vertex_colors)
    """
    batch_centers, batch_sizes, batch_colors, unit_verts, unit_faces, start_idx = args
    
    n_voxels = len(batch_centers)
    n_verts_per_cube = 8
    n_faces_per_cube = 12
    
    # Preallocate arrays
    all_vertices = np.zeros((n_voxels * n_verts_per_cube, 3), dtype=np.float32)
    all_faces = np.zeros((n_voxels * n_faces_per_cube, 3), dtype=np.int32)
    all_colors = np.zeros((n_voxels * n_verts_per_cube, 4), dtype=np.uint8)
    
    for i in range(n_voxels):
        # Transform vertices
        verts = transform_cube(unit_verts, batch_centers[i], batch_sizes[i])
        
        # Store vertices
        v_start = i * n_verts_per_cube
        v_end = v_start + n_verts_per_cube
        all_vertices[v_start:v_end] = verts
        
        # Store faces with offset
        f_start = i * n_faces_per_cube
        f_end = f_start + n_faces_per_cube
        all_faces[f_start:f_end] = unit_faces + v_start
        
        # Store colors (RGBA, same for all 8 vertices of cube)
        color_rgb = batch_colors[i]
        color_rgba = np.array([
            int(color_rgb[0] * 255),
            int(color_rgb[1] * 255),
            int(color_rgb[2] * 255),
            255
        ], dtype=np.uint8)
        all_colors[v_start:v_end] = color_rgba
    
    return all_vertices, all_faces, all_colors


def voxels_to_mesh_parallel(centers, sizes, colors, num_workers=None, batch_size=10000):
    """
    Convert voxels to mesh using parallel processing.
    
    Args:
        centers: (N, 3) voxel centers
        sizes: (N,) or (N, 1) voxel sizes
        colors: (N, 3) RGB colors in [0, 1]
        num_workers: number of parallel workers
        batch_size: number of voxels per batch
        
    Returns:
        trimesh.Trimesh object
    """
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 1)
    
    n_voxels = len(centers)
    print(f"Converting {n_voxels} voxels to mesh using {num_workers} workers...")
    
    # Ensure numpy arrays
    if torch.is_tensor(centers):
        centers = centers.cpu().numpy()
    if torch.is_tensor(sizes):
        sizes = sizes.cpu().numpy()
    if torch.is_tensor(colors):
        colors = colors.cpu().numpy()
    
    # Flatten sizes if needed
    if sizes.ndim > 1:
        sizes = sizes.squeeze()
    
    # Clip colors to [0, 1]
    colors = np.clip(colors, 0, 1)
    
    # Create unit cube template
    unit_verts, unit_faces = create_unit_cube()
    
    # Split into batches
    n_batches = (n_voxels + batch_size - 1) // batch_size
    batches = []
    
    for i in range(n_batches):
        start = i * batch_size
        end = min(start + batch_size, n_voxels)
        batches.append((
            centers[start:end],
            sizes[start:end],
            colors[start:end],
            unit_verts,
            unit_faces,
            start
        ))
    
    # Process batches in parallel
    all_vertices = []
    all_faces = []
    all_colors = []
    
    vertex_offset = 0
    
    # Use ThreadPoolExecutor for I/O bound operations
    # For CPU-bound numpy operations, ProcessPoolExecutor might be slower due to serialization
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(
            executor.map(process_voxel_batch, batches),
            total=len(batches),
            desc="Processing batches"
        ))
    
    # Combine results
    print("Combining mesh data...")
    for verts, faces, colors in results:
        # Offset faces
        faces_offset = faces + vertex_offset
        
        all_vertices.append(verts)
        all_faces.append(faces_offset)
        all_colors.append(colors)
        
        vertex_offset += len(verts)
    
    # Concatenate all
    final_vertices = np.concatenate(all_vertices, axis=0)
    final_faces = np.concatenate(all_faces, axis=0)
    final_colors = np.concatenate(all_colors, axis=0)
    
    print(f"Created mesh with {len(final_vertices)} vertices and {len(final_faces)} faces")
    
    # Create trimesh object
    mesh = trimesh.Trimesh(
        vertices=final_vertices,
        faces=final_faces,
        vertex_colors=final_colors,
        process=False  # Skip processing for speed
    )
    
    return mesh


def voxels_to_mesh_vectorized(centers, sizes, colors):
    """
    Convert voxels to mesh using fully vectorized numpy operations.
    This is faster for moderate number of voxels.
    
    Args:
        centers: (N, 3) voxel centers
        sizes: (N,) or (N, 1) voxel sizes
        colors: (N, 3) RGB colors in [0, 1]
        
    Returns:
        trimesh.Trimesh object
    """
    n_voxels = len(centers)
    print(f"Converting {n_voxels} voxels to mesh (vectorized)...")
    
    # Ensure numpy arrays
    if torch.is_tensor(centers):
        centers = centers.cpu().numpy()
    if torch.is_tensor(sizes):
        sizes = sizes.cpu().numpy()
    if torch.is_tensor(colors):
        colors = colors.cpu().numpy()
    
    # Flatten sizes if needed
    if sizes.ndim > 1:
        sizes = sizes.squeeze()
    
    # Clip colors to [0, 1]
    colors = np.clip(colors, 0, 1)
    
    # Create unit cube template
    unit_verts, unit_faces = create_unit_cube()
    
    # Vectorized transformation
    # Broadcast: (N, 1, 3) * (1, 8, 3) + (N, 1, 3) -> (N, 8, 3)
    all_vertices = (
        sizes[:, None, None] * unit_verts[None, :, :] + 
        centers[:, None, :]
    ).reshape(-1, 3).astype(np.float32)
    
    # Create face indices with offsets
    # Each cube has 12 faces, offset by 8 vertices per cube
    offsets = np.arange(n_voxels)[:, None, None] * 8  # (N, 1, 1)
    all_faces = (unit_faces[None, :, :] + offsets).reshape(-1, 3).astype(np.int32)
    
    # Create vertex colors (RGBA)
    # Each cube has 8 vertices with same color
    colors_rgba = np.zeros((n_voxels, 8, 4), dtype=np.uint8)
    colors_rgba[:, :, :3] = (colors[:, None, :] * 255).astype(np.uint8)
    colors_rgba[:, :, 3] = 255
    all_colors = colors_rgba.reshape(-1, 4)
    
    print(f"Created mesh with {len(all_vertices)} vertices and {len(all_faces)} faces")
    
    # Create trimesh object
    mesh = trimesh.Trimesh(
        vertices=all_vertices,
        faces=all_faces,
        vertex_colors=all_colors,
        process=False  # Skip processing for speed
    )
    
    return mesh


def detect_checkpoint_type(checkpoint_path):
    """
    Detect the type of checkpoint file.
    
    Args:
        checkpoint_path: path to .pt file
        
    Returns:
        str: 'model' for full model checkpoint, 'object' for object voxel file
    """
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    # Full model checkpoints have 'active_sh_degree' key
    if 'active_sh_degree' in state_dict:
        return 'model', state_dict
    # Object voxel files have 'vox_center' key
    elif 'vox_center' in state_dict:
        return 'object', state_dict
    else:
        raise ValueError(f"Unknown checkpoint format: {checkpoint_path}")


def create_cfg_model(model_path=None):
    """Create cfg_model config node for SparseVoxelModel."""
    cfg_model = CfgNode()
    cfg_model.model_path = model_path if model_path else ""
    cfg_model.vox_geo_mode = "triinterp1"
    cfg_model.density_mode = "exp_linear_11"
    cfg_model.sh_degree = 3
    cfg_model.ss = 1.5
    cfg_model.outside_level = 5
    cfg_model.white_background = False
    cfg_model.black_background = False
    return cfg_model


def load_voxel_model(checkpoint_path):
    checkpoint_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    if os.path.basename(checkpoint_dir) == 'checkpoints':
        model_path = os.path.dirname(checkpoint_dir)
    else:
        model_path = checkpoint_dir
    
    cfg_model = create_cfg_model(model_path)
    voxel_model = SparseVoxelModel(cfg_model)
    voxel_model.load(checkpoint_path)
    return voxel_model


def load_object_voxels(state_dict):
    """
    Load voxel data from object voxel file (exported by replica_segm_3d.py).
    
    Args:
        state_dict: loaded state dict from object voxel .pt file
        
    Returns:
        tuple of (centers, sizes, colors, octlevel, num_voxels)
    """
    centers = state_dict['vox_center']  # (N, 3)
    sizes = state_dict['vox_size']      # (N, 1)
    sh0 = state_dict['sh0']             # (N, 1, 3)
    octlevel = state_dict.get('octlevel', None)  # (N,)
    
    # Convert to CUDA tensors if needed
    if not centers.is_cuda:
        centers = centers.cuda()
    if not sizes.is_cuda:
        sizes = sizes.cuda()
    if not sh0.is_cuda:
        sh0 = sh0.cuda()
    if octlevel is not None and not octlevel.is_cuda:
        octlevel = octlevel.cuda()
    
    # Extract color from SH0 coefficient
    colors = shzero2rgb(sh0.squeeze(1))  # (N, 3)
    colors = colors.clamp(0, 1)
    
    num_voxels = len(centers)
    
    return centers, sizes, colors, octlevel, num_voxels


def extract_voxel_data(voxel_model, use_density_threshold=False, density_threshold=0.5):
    """
    Extract voxel centers, sizes, colors, and octree levels from model.
    
    Args:
        voxel_model: SparseVoxelModel instance
        use_density_threshold: whether to filter by density
        density_threshold: minimum density to include voxel
        
    Returns:
        tuple of (centers, sizes, colors, octlevel)
    """
    with torch.no_grad():
        centers = voxel_model.vox_center  # (N, 3)
        sizes = voxel_model.vox_size      # (N, 1)
        octlevel = voxel_model.octlevel   # (N,)
        
        # Extract color from SH0 coefficient
        sh0 = voxel_model._sh0.data  # (N, 1, 3)
        colors = shzero2rgb(sh0.squeeze(1))  # (N, 3)
        colors = colors.clamp(0, 1)
        
        # Optional: filter by density
        if use_density_threshold:
            # Get mean density per voxel from geo_grid_pts
            # This is approximate - actual density depends on position within voxel
            geo = voxel_model._geo_grid_pts.data  # (N_grid_pts, 1)
            # For simplicity, we use all voxels for now
            # A more accurate method would compute density at voxel centers
            pass
        
    return centers, sizes, colors, octlevel


def print_voxel_statistics(octlevel, num_voxels):
    """
    Print statistics about voxel distribution across octree levels.
    
    Args:
        octlevel: (N,) tensor of octree levels for each voxel
        num_voxels: total number of voxels
    """
    print("\n" + "="*60)
    print("VOXEL STATISTICS")
    print("="*60)
    print(f"Total voxels: {num_voxels:,}")
    
    if octlevel is not None:
        # Convert to CPU for processing
        if torch.is_tensor(octlevel):
            octlevel_cpu = octlevel.cpu().numpy()
        else:
            octlevel_cpu = octlevel
        
        # Get unique levels and counts
        unique_levels, counts = np.unique(octlevel_cpu, return_counts=True)
        
        print(f"\nOctree levels: {len(unique_levels)}")
        print("\nVoxel distribution by level:")
        print("-" * 60)
        print(f"{'Level':<8} {'Count':<12} {'Percentage':<12} {'Bar'}")
        print("-" * 60)
        
        for level, count in zip(unique_levels, counts):
            percentage = (count / num_voxels) * 100
            bar_length = int(percentage / 2)  # Scale to max 50 chars
            bar = '█' * bar_length
            print(f"{level:<8} {count:<12,} {percentage:>6.2f}%      {bar}")
        
        print("-" * 60)
        print(f"Min level: {unique_levels.min()}  |  Max level: {unique_levels.max()}")
    else:
        print("\nOctree level information not available.")
    
    print("="*60 + "\n")


def voxel_to_mesh(checkpoint, output=None, format='glb', max_voxels=None, method='vectorized', workers=None, simplify=False, simplify_ratio=0.5, stats_only=False):
    # Check checkpoint file exists
    if not os.path.isfile(checkpoint):
        print(f"Error: Checkpoint file not found: {checkpoint}")
        return
    
    # Set output path
    if output is None:
        checkpoint_dir = os.path.dirname(os.path.abspath(checkpoint))
        output = os.path.join(checkpoint_dir, f'voxel_mesh.{format}')
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    
    print(f"Loading checkpoint from: {checkpoint}")
    
    # Detect checkpoint type and load accordingly
    checkpoint_type, state_dict = detect_checkpoint_type(checkpoint)
    
    # BUG 
    if checkpoint_type == 'model':
        print("Detected: Full model checkpoint")
        voxel_model = load_voxel_model(checkpoint)
        num_voxels = voxel_model.num_voxels
        print(f"Model loaded: {num_voxels} voxels")
        
        # Extract voxel data
        print("Extracting voxel data...")
        centers, sizes, colors, octlevel = extract_voxel_data(voxel_model)
    else:  # checkpoint_type == 'object'
        print("Detected: Object voxel file")
        centers, sizes, colors, octlevel, num_voxels = load_object_voxels(state_dict)
        print(f"Object loaded: {num_voxels} voxels")
        if 'original_id' in state_dict:
            print(f"  Original ID: {state_dict['original_id']}")
    
    # Print voxel statistics
    print(f"[Debug] Printing voxel statistics...")
    print_voxel_statistics(octlevel, num_voxels)
    
    # If only stats requested, exit here
    if stats_only:
        print("Statistics only mode - skipping mesh conversion.")
        return True
    
    # Limit voxels if requested
    if max_voxels is not None and max_voxels < len(centers):
        print(f"Limiting to {max_voxels} voxels for mesh export")
        # Random sampling
        indices = torch.randperm(len(centers))[:max_voxels]
        centers = centers[indices]
        sizes = sizes[indices]
        colors = colors[indices]
        if octlevel is not None:
            octlevel = octlevel[indices]
    
    # Convert to mesh
    print(f"[Debug] Converting voxels to mesh using '{method}' method...")
    if method == 'vectorized':
        mesh = voxels_to_mesh_vectorized(centers, sizes, colors)
    else:
        mesh = voxels_to_mesh_parallel(centers, sizes, colors, num_workers=workers)
    
    print(f"[Debug] Mesh created: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    # Optional simplification
    if simplify:
        print(f"Simplifying mesh (target ratio: {simplify_ratio})...")
        target_faces = int(len(mesh.faces) * simplify_ratio)
        mesh = mesh.simplify_quadric_decimation(target_faces)
        print(f"Simplified to {len(mesh.faces)} faces")
    
    # Export mesh
    print(f"[Debug] Exporting mesh to: {output}")
    mesh.export(output)
    
    # Verify file was created
    if not os.path.exists(output):
        print(f"[Error] Output file was not created: {output}")
        return False
    
    # Print file size
    file_size = os.path.getsize(output) / (1024 * 1024)
    print(f"Output file size: {file_size:.2f} MB")
    print(f"[Success] Mesh exported successfully!")
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert SVRaster voxel checkpoint to 3D mesh for Blender visualization"
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint .pt file (e.g., outputs/Replica/scan1/checkpoints/iter030000_model.pt)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path (default: same directory as checkpoint, named voxel_mesh.glb)')
    parser.add_argument('--format', type=str, choices=['glb', 'obj', 'ply', 'stl'], default='glb',
                        help='Output format (default: glb for Blender)')
    parser.add_argument('--max_voxels', type=int, default=None,
                        help='Maximum number of voxels to export (for testing)')
    parser.add_argument('--method', type=str, choices=['vectorized', 'parallel'], default='vectorized',
                        help='Conversion method (default: vectorized)')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers (default: CPU count - 1)')
    parser.add_argument('--simplify', action='store_true',
                        help='Simplify mesh to reduce file size')
    parser.add_argument('--simplify_ratio', type=float, default=0.5,
                        help='Target ratio for mesh simplification (default: 0.5)')
    parser.add_argument('--stats-only', action='store_true',
                        help='Only print statistics without converting to mesh')
    
    args = parser.parse_args()
    
    voxel_to_mesh(
        checkpoint=args.checkpoint,
        output=args.output,
        format=args.format,
        max_voxels=args.max_voxels,
        method=args.method,
        workers=args.workers,
        simplify=args.simplify,
        simplify_ratio=args.simplify_ratio,
        stats_only=args.stats_only
    )
