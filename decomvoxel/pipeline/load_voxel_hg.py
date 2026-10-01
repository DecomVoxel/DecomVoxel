"""
(deprecated history code)
Voxel Loader and Converter for DecomVoxel Pipeline

This module provides classes and functions for:
1. Loading GeoSVR voxel data from checkpoint files
2. Converting sparse voxel representation to dense 64^3 grid (SparseStructure)
3. SparseStructure class for TRELLIS-compatible representation
4. Visualization utilities for voxel and sparse structure data

The GeoSVR voxels are stored in a sparse format with:
- vox_center: (N, 3) center positions of each voxel
- vox_size: (N, 1) size of each voxel
- sh0: (N, 1, 3) spherical harmonics coefficients for color
- octlevel: (N,) octree level for each voxel

TRELLIS Sparse Structure:
- Dense occupancy grid: (1, 1, 64, 64, 64)
- Can be directly optimized via SDS
- Input/output format for TRELLIS diffusion models

# Definitions in freeart3d:
# spatial layout of dense tensor is [Z, Y, X]
# equivalent to [D, H, W], where D=Z, H=Y, W=X

# Definitions in this project:
# spatial layout of dense tensor is [X, Y, Z]
# equivalent to [W, H, D], where W=X, H=Y, D=Z

# The perceived 90-degree rotation may come from different coordinate conventions
# Swapping X and Y gives [Y, X, Z], [H, W, D]

"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from typing import Optional, Dict, Tuple, Union, List

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

# Import GeoSVR modules
from yacs.config import CfgNode
from src.sparse_voxel_model import SparseVoxelModel
from src.utils import activation_utils

DEBUG=True

class GeoSVRVoxelData:
    """
    Container class for GeoSVR voxel data.
    
    Attributes:
        centers: (N, 3) voxel center positions
        sizes: (N, 1) voxel sizes  
        colors: (N, 3) RGB colors in [0, 1]
        octlevel: (N,) octree levels
        num_voxels: total number of voxels
        bbox_min: (3,) minimum bounding box corner
        bbox_max: (3,) maximum bounding box corner
        original_id: original object ID from segmentation
    """
    
    def __init__(self, centers: torch.Tensor, sizes: torch.Tensor, 
                 colors: torch.Tensor, octlevel: Optional[torch.Tensor] = None,
                 original_id: Optional[int] = None):
        self.centers = centers
        self.sizes = sizes
        self.colors = colors
        self.octlevel = octlevel
        self.num_voxels = len(centers)
        self.original_id = original_id
        
        # Compute bounding box
        self.bbox_min = centers.min(dim=0).values
        self.bbox_max = centers.max(dim=0).values
        self.bbox_center = (self.bbox_min + self.bbox_max) / 2
        self.bbox_extent = self.bbox_max - self.bbox_min
    
    def to(self, device: torch.device) -> 'GeoSVRVoxelData':
        """Move all tensors to specified device."""
        self.centers = self.centers.to(device)
        self.sizes = self.sizes.to(device)
        self.colors = self.colors.to(device)
        if self.octlevel is not None:
            self.octlevel = self.octlevel.to(device)
        self.bbox_min = self.bbox_min.to(device)
        self.bbox_max = self.bbox_max.to(device)
        self.bbox_center = self.bbox_center.to(device)
        self.bbox_extent = self.bbox_extent.to(device)
        return self


class SparseStructure:
    """
    TRELLIS-compatible Sparse Structure representation.
    
    This is a 64^3 dense occupancy grid that serves as the input/output
    format for TRELLIS diffusion models. Despite the name, it's stored
    as a dense tensor for compatibility with TRELLIS encoder/decoder.
    
    Attributes:
        data: (1, 1, 64, 64, 64) dense occupancy grid
        resolution: grid resolution (default: 64)
        device: torch device
        occupancy_count: number of occupied voxels
        known_mask: optional mask for known regions (for SDS)
    """
    
    RESOLUTION = 64  # Fixed resolution for TRELLIS compatibility
    
    def __init__(self, 
                 data: torch.Tensor = None,
                 resolution: int = 64,
                 device: torch.device = None):
        """
        Initialize SparseStructure.
        
        Args:
            data: (1, 1, R, R, R) occupancy grid, or None to create empty
            resolution: grid resolution (default: 64)
            device: torch device
        """
        self.resolution = resolution
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if data is not None:
            self.data = data.to(self.device)
            # Ensure correct shape
            if self.data.dim() == 3:
                self.data = self.data.unsqueeze(0).unsqueeze(0)
            elif self.data.dim() == 4:
                self.data = self.data.unsqueeze(0)
        else:
            self.data = torch.zeros(1, 1, resolution, resolution, resolution, 
                                   device=self.device)
        
        self.known_mask = None
        self.transform_info = None  # Set by VoxelToSparseStructure.convert()
        self._update_stats()
    
    def _update_stats(self):
        """Update occupancy statistics."""
        self.occupancy_count = (self.data > 0.5).sum().item()
        self.occupancy_ratio = self.occupancy_count / (self.resolution ** 3)
    
    @classmethod
    def from_tensor(cls, tensor: torch.Tensor, 
                    device: torch.device = None) -> 'SparseStructure':
        """Create SparseStructure from a tensor."""
        return cls(data=tensor, device=device)
    
    @classmethod
    def from_file(cls, path: str, 
                  device: torch.device = None) -> 'SparseStructure':
        """Load SparseStructure from a .pt file."""
        data = torch.load(path, map_location='cpu', weights_only=True)
        return cls(data=data, device=device)
    
    @classmethod
    def empty(cls, resolution: int = 64, 
              device: torch.device = None) -> 'SparseStructure':
        """Create an empty SparseStructure."""
        return cls(resolution=resolution, device=device)
    
    def to(self, device: torch.device) -> 'SparseStructure':
        """Move to specified device."""
        self.device = device
        self.data = self.data.to(device)
        if self.known_mask is not None:
            self.known_mask = self.known_mask.to(device)
        return self
    
    def clone(self) -> 'SparseStructure':
        """Create a deep copy."""
        new_ss = SparseStructure(data=self.data.clone(), device=self.device)
        if self.known_mask is not None:
            new_ss.known_mask = self.known_mask.clone()
        return new_ss
    
    def get_occupancy_grid(self) -> torch.Tensor:
        """Return the raw occupancy tensor (1, 1, 64, 64, 64)."""
        return self.data
    
    def get_binary_grid(self, threshold: float = 0.5) -> torch.Tensor:
        """Return binarized occupancy tensor."""
        return (self.data > threshold).float()
    
    def set_known_mask(self, mask: torch.Tensor):
        """Set the known mask for SDS optimization."""
        if mask.dim() == 3:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.dim() == 4:
            mask = mask.unsqueeze(0)
        self.known_mask = mask.to(self.device)
    
    def get_occupied_coords(self, threshold: float = 0.5) -> torch.Tensor:
        """
        Get coordinates of occupied voxels.
        
        Returns:
            (N, 3) integer coordinates of occupied voxels
        """
        binary = self.data.squeeze() > threshold
        coords = torch.argwhere(binary)  # (N, 3)
        return coords
    
    def get_normalized_coords(self, threshold: float = 0.5) -> torch.Tensor:
        """
        Get normalized coordinates [-0.5, 0.5] of occupied voxels.
        
        Returns:
            (N, 3) normalized coordinates
        """
        coords = self.get_occupied_coords(threshold).float()
        coords_norm = coords / self.resolution - 0.5
        return coords_norm
    
    def save(self, path: str):
        """Save SparseStructure to file."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        torch.save(self.data, path)
        print(f"[SparseStructure] Saved to: {path}")

    def rotate_z(self, angle: float) -> 'SparseStructure':
        if angle == 0.0:
            return self

        # Normalize to [0, 360)
        angle_mod = angle % 360

        # Fast path for exact 90° multiples
        if angle_mod == 90.0:
            # CCW 90°: (x, y) -> (-y, x)  =>  transpose XY then flip X
            self.data = self.data.transpose(2, 3)
            self.data = self.data.flip(2)
        elif angle_mod == 180.0:
            # 180°: flip X and flip Y
            self.data = self.data.flip(2).flip(3)
        elif angle_mod == 270.0:
            # CW 90° (= CCW 270°): transpose XY then flip Y
            self.data = self.data.transpose(2, 3)
            self.data = self.data.flip(3)
        else:
            # Arbitrary angle – use scipy
            from scipy.ndimage import rotate as ndimage_rotate
            # data shape: (1, 1, X, Y, Z)
            # Rotate in XY plane (axes 2,3)
            vol = self.data.squeeze(0).squeeze(0).cpu().numpy()  # (X, Y, Z)
            rotated = ndimage_rotate(
                vol, angle, axes=(0, 1), reshape=False,
                order=1, mode='constant', cval=0.0
            )
            # Re-binarize after interpolation
            rotated = (rotated > 0.5).astype(np.float32)
            self.data = torch.from_numpy(rotated).unsqueeze(0).unsqueeze(0).to(self.device)

        self._update_stats()
        print(f"[SparseStructure] Rotated {angle}° around Z-axis")
        return self
    
    def __repr__(self) -> str:
        return (f"SparseStructure(resolution={self.resolution}, "
                f"occupancy={self.occupancy_count}/{self.resolution**3} "
                f"({self.occupancy_ratio*100:.1f}%), device={self.device})")


class VoxelToSparseStructure:
    def __init__(self, resolution: int = 64, device: torch.device = None):
        self.resolution = resolution
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.last_transform_info = None  # Stored after each convert() call
    
    def convert(self, voxel_data: GeoSVRVoxelData, 
                normalize: bool = True,
                padding: float = 0.3,
                robust_percentile: float = 0.02) -> SparseStructure:
        """
        Convert GeoSVR voxel data to SparseStructure.
        """
        res = self.resolution
        
        # Get voxel centers and sizes
        centers = voxel_data.centers.to(self.device)  # (N, 3)
        sizes = voxel_data.sizes.squeeze(-1).to(self.device)  # (N,)
        
        # if DEBUG:
        #     # Debug: print 10 random samples of centers and sizes
        #     n = len(centers)
        #     sample_idx = torch.randperm(n)[:min(10, n)]
        #     print(f"\n[VoxelToSparseStructure] === Debug: {n} voxels total, showing {len(sample_idx)} random samples ===")
        #     for idx in sample_idx:
        #         c = centers[idx].cpu().numpy()
        #         s = sizes[idx].cpu().item()
        #         print(f"  voxel {idx.item():>6d}: center=({c[0]:.4f}, {c[1]:.4f}, {c[2]:.4f}), size={s:.6f}")
        #     print(f"  centers range: min={centers.min(dim=0).values.cpu().numpy()}, max={centers.max(dim=0).values.cpu().numpy()}")
        #     print(f"  sizes  range: min={sizes.min().item():.6f}, max={sizes.max().item():.6f}, mean={sizes.mean().item():.6f}")
        
        if robust_percentile > 0:
            # Use percentile-based bbox to filter out outlier floating voxels
            lo = robust_percentile
            hi = 1.0 - robust_percentile
            bbox_min_robust = torch.quantile(centers, lo, dim=0)
            bbox_max_robust = torch.quantile(centers, hi, dim=0)
            bbox_extent_robust = bbox_max_robust - bbox_min_robust
            
            # Apply padding on the robust bbox
            bbox_min = bbox_min_robust - bbox_extent_robust * padding
            bbox_max = bbox_max_robust + bbox_extent_robust * padding
            bbox_extent = bbox_max - bbox_min
            max_extent = bbox_extent.max()
            
            print(f"[Robust normalize] percentile={lo:.2%}~{hi:.2%}")
            print(f"  robust bbox min: {bbox_min_robust.cpu().numpy()}, max: {bbox_max_robust.cpu().numpy()}")
            print(f"  padded bbox extent: {bbox_extent.cpu().numpy()}, max_extent: {max_extent.item():.6f}")
            
            # Use mean as center
            # mean_center = centers.mean(dim=0)
            
            # Filter out outliers, then compute mean as center
            # inlier_mask = ((centers >= bbox_min_robust) & (centers <= bbox_max_robust)).all(dim=1)
            # mean_center = centers[inlier_mask].mean(dim=0)
            
            # Use robust bbox center as the cube center
            robust_center = (bbox_min_robust + bbox_max_robust) / 2
            cube_min = robust_center - max_extent / 2
            
            centers_norm = (centers - cube_min) / max_extent
            sizes_norm = sizes / max_extent
            
            print(f"  robust center: {robust_center.cpu().numpy()}, cube_min: {cube_min.cpu().numpy()}")
            
            # Clamp outliers that fall outside [0, 1] after normalization
            # (they will be clamped to grid boundary later anyway)
        else:
            # Original: use min/max bbox
            bbox_min = voxel_data.bbox_min.to(self.device) - voxel_data.bbox_extent.to(self.device) * padding
            bbox_max = voxel_data.bbox_max.to(self.device) + voxel_data.bbox_extent.to(self.device) * padding
            bbox_extent = bbox_max - bbox_min
            max_extent = bbox_extent.max()  # Length of the longest side
            print("bbox extent:" + str(bbox_extent.cpu().numpy()) + ", max extent: " + str(max_extent.item()))
            print("bbox min:" + str(bbox_min.cpu().numpy()) + ", bbox max: " + str(bbox_max.cpu().numpy()))
            
            mean_center = centers.mean(dim=0)  # (3,) mean of all voxel centers
            cube_min = mean_center - max_extent / 2
            centers_norm = (centers - cube_min) / max_extent
            sizes_norm = sizes / max_extent
            
            print("mean center:" + str(mean_center.cpu().numpy()) + ", cube min: " + str(cube_min.cpu().numpy()))
        
        # Convert to grid indices [0, res-1]
        grid_coords = (centers_norm * res).long()
        grid_coords = grid_coords.clamp(0, res - 1)
        
        # Create dense occupancy grid
        dense_grid = torch.zeros(1, 1, res, res, res, device=self.device)
        
        # Main bottleneck: this step is relatively slow.
        # half_sizes = (sizes_norm * res / 2).long()  # (N,)
        half_sizes = (sizes_norm * res / 2).clamp(min=1).long()  # (N,)
        unique_hs = half_sizes.unique()
        
        for hs in unique_hs:
            mask = half_sizes == hs
            coords = grid_coords[mask]  # (M, 3)
            hs_val = hs.item()
            
            if hs_val == 0:
                # Single cell — simple index scatter
                dense_grid[0, 0, coords[:, 0], coords[:, 1], coords[:, 2]] = 1.0
            else:
                # Build (2h+1)^3 offset grid, broadcast over all centres
                offsets = torch.arange(-hs_val, hs_val + 1, device=self.device)
                ox, oy, oz = torch.meshgrid(offsets, offsets, offsets, indexing='ij')
                offset_grid = torch.stack([ox.reshape(-1), oy.reshape(-1), oz.reshape(-1)], dim=1)  # (K, 3)
                
                # Process in batches to avoid OOM when M * K is large
                # K = (2*hs_val+1)^3; keep each batch ≤ ~4 M elements
                K = offset_grid.shape[0]
                batch_size = max(1, 4_000_000 // K)
                for b_start in range(0, len(coords), batch_size):
                    b_coords = coords[b_start:b_start + batch_size]  # (B, 3)
                    # (B, 1, 3) + (1, K, 3) -> (B, K, 3)
                    bc = b_coords.unsqueeze(1) + offset_grid.unsqueeze(0)
                    bc = bc.reshape(-1, 3)
                    bc.clamp_(0, res - 1)
                    dense_grid[0, 0, bc[:, 0], bc[:, 1], bc[:, 2]] = 1.0
        
        if DEBUG:
            # Debug: verify grid filling statistics
            sizes_in_cells = sizes_norm * res  # voxel edge length in grid cells
            half_sizes = (sizes_in_cells / 2).long()
            footprints = (2 * half_sizes + 1)  # actual cell span per axis
            print(f"\n[VoxelToSparseStructure] === Grid filling debug ===")
            print(f"  voxel size in grid cells: min={sizes_in_cells.min().item():.2f}, "
                  f"max={sizes_in_cells.max().item():.2f}, mean={sizes_in_cells.mean().item():.2f}")
            print(f"  half_size: min={half_sizes.min().item()}, max={half_sizes.max().item()}")
            print(f"  actual footprint per axis: min={footprints.min().item()}, max={footprints.max().item()}")
            print(f"  grid_coords range: min={grid_coords.min(dim=0).values.cpu().numpy()}, "
                  f"max={grid_coords.max(dim=0).values.cpu().numpy()}")
            # Show 5 examples of the mapping
            sample_idx = torch.randperm(len(centers))[:min(5, len(centers))]
            for idx in sample_idx:
                gc = grid_coords[idx].cpu().numpy()
                hs = int(max(1, (sizes_norm[idx] * res / 2).item()))
                sc = sizes_in_cells[idx].item()
                print(f"  voxel {idx.item()}: grid=({gc[0]},{gc[1]},{gc[2]}), "
                      f"size_cells={sc:.2f}, half_size={hs}, footprint={2*hs+1}³")
            
            # Auto-save visualization
            debug_path = os.path.join(PROJECT_ROOT, "outputs", "debug", "grid_coords_debug.png")
            visualize_grid_coords(
                grid_coords, dense_grid=dense_grid, resolution=res,
                save_path=debug_path,
                title=f"VoxelToSparseStructure Debug (N={len(centers)}, res={res})",
                sizes_in_cells=sizes_in_cells
            )
        
        # Create SparseStructure
        ss = SparseStructure(data=dense_grid, device=self.device)
        
        # Set known mask (initially all occupied voxels are known)
        ss.set_known_mask(dense_grid.clone())
        
        # Store transform info for inverse conversion
        from decomvoxel.utils.converting import TransformInfo
        transform_info = TransformInfo(
            cube_min=cube_min.cpu(),
            max_extent=max_extent.item() if isinstance(max_extent, torch.Tensor) else float(max_extent),
            resolution=res,
        )
        ss.transform_info = transform_info
        self.last_transform_info = transform_info
        
        print(f"[VoxelToSparseStructure] Converted {voxel_data.num_voxels} voxels -> {ss}")
        print(f"[VoxelToSparseStructure] {transform_info}")
        
        return ss
    
    def convert_batch(self, voxel_data_list: List[GeoSVRVoxelData],
                      normalize: bool = True) -> List[SparseStructure]:
        """Convert multiple voxel data objects."""
        return [self.convert(vd, normalize) for vd in voxel_data_list]


def voxel_to_sparse_structure(voxel_data: GeoSVRVoxelData,
                              resolution: int = 64,
                              device: torch.device = None,
                              robust_percentile: float = 0.0) -> SparseStructure: # XXX
    converter = VoxelToSparseStructure(resolution=resolution, device=device)
    return converter.convert(voxel_data, robust_percentile=robust_percentile)


def visualize_grid_coords(grid_coords: torch.Tensor,
                          dense_grid: torch.Tensor = None,
                          resolution: int = 64,
                          save_path: str = None,
                          title: str = "Grid Coordinates Visualization",
                          sizes_in_cells: torch.Tensor = None):
    """
    Visualize grid coordinates with three orthogonal projection views + 3D scatter,
    combined into a single figure (2x2 layout).
    
    Args:
        grid_coords: (N, 3) integer grid coordinates of voxel centers
        dense_grid: optional (1, 1, R, R, R) dense occupancy grid to visualize 
                    instead of just centers. If provided, shows filled voxels.
        resolution: grid resolution (default: 64)
        save_path: path to save the figure. If None, shows interactively.
        title: figure title
        sizes_in_cells: optional (N,) voxel sizes in grid cells, used for coloring
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    coords = grid_coords.cpu().numpy() if isinstance(grid_coords, torch.Tensor) else grid_coords
    
    # If dense_grid is provided, extract occupied coords from it (shows actual filled result)
    if dense_grid is not None:
        occupied = (dense_grid.squeeze() > 0.5).cpu().numpy()
        occ_coords = np.argwhere(occupied)  # (M, 3)
        use_dense = True
    else:
        use_dense = False
    
    # Color by voxel size if available
    if sizes_in_cells is not None:
        sc = sizes_in_cells.cpu().numpy() if isinstance(sizes_in_cells, torch.Tensor) else sizes_in_cells
        # Normalize to [0, 1] for colormap
        sc_min, sc_max = sc.min(), sc.max()
        if sc_max > sc_min:
            sc_norm = (sc - sc_min) / (sc_max - sc_min)
        else:
            sc_norm = np.zeros_like(sc)
        colors_center = plt.cm.viridis(sc_norm)
    else:
        colors_center = 'dodgerblue'
    
    fig = plt.figure(figsize=(16, 14))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    res = resolution
    dot_size_center = max(1, 80 // (res // 16))
    dot_size_dense = max(0.3, 20 // (res // 16))
    
    # --- Top-left: XY projection (top view, looking down Z) ---
    ax1 = fig.add_subplot(2, 2, 1)
    if use_dense:
        # Project: for each (x, y), check if any z is occupied
        proj_xy = occupied.any(axis=2)  # (X, Y)
        ax1.imshow(proj_xy.T, origin='lower', cmap='Greys', aspect='equal',
                   extent=[0, res, 0, res], alpha=0.3)
    ax1.scatter(coords[:, 0], coords[:, 1], s=dot_size_center, c=colors_center, 
                alpha=0.6, edgecolors='none')
    ax1.set_xlim(0, res)
    ax1.set_ylim(0, res)
    ax1.set_xlabel('X (grid)')
    ax1.set_ylabel('Y (grid)')
    ax1.set_title(f'XY Projection (top view, along Z)\ncenters: {len(coords)}')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.2)
    
    # --- Top-right: XZ projection (front view, looking along Y) ---
    ax2 = fig.add_subplot(2, 2, 2)
    if use_dense:
        proj_xz = occupied.any(axis=1)  # (X, Z)
        ax2.imshow(proj_xz.T, origin='lower', cmap='Greys', aspect='equal',
                   extent=[0, res, 0, res], alpha=0.3)
    ax2.scatter(coords[:, 0], coords[:, 2], s=dot_size_center, c=colors_center,
                alpha=0.6, edgecolors='none')
    ax2.set_xlim(0, res)
    ax2.set_ylim(0, res)
    ax2.set_xlabel('X (grid)')
    ax2.set_ylabel('Z (grid)')
    ax2.set_title(f'XZ Projection (front view, along Y)')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.2)
    
    # --- Bottom-left: YZ projection (side view, looking along X) ---
    ax3 = fig.add_subplot(2, 2, 3)
    if use_dense:
        proj_yz = occupied.any(axis=0)  # (Y, Z)
        ax3.imshow(proj_yz.T, origin='lower', cmap='Greys', aspect='equal',
                   extent=[0, res, 0, res], alpha=0.3)
    ax3.scatter(coords[:, 1], coords[:, 2], s=dot_size_center, c=colors_center,
                alpha=0.6, edgecolors='none')
    ax3.set_xlim(0, res)
    ax3.set_ylim(0, res)
    ax3.set_xlabel('Y (grid)')
    ax3.set_ylabel('Z (grid)')
    ax3.set_title(f'YZ Projection (side view, along X)')
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.2)
    
    # --- Bottom-right: 3D scatter ---
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    if use_dense:
        # Subsample dense coords if too many for 3D plot
        if len(occ_coords) > 8000:
            idx = np.random.choice(len(occ_coords), 8000, replace=False)
            occ_sub = occ_coords[idx]
        else:
            occ_sub = occ_coords
        ax4.scatter(occ_sub[:, 0], occ_sub[:, 1], occ_sub[:, 2],
                    s=dot_size_dense, c='lightgray', alpha=0.15, edgecolors='none',
                    label=f'filled ({len(occ_coords)})')
    # Plot centers on top
    if len(coords) > 5000:
        idx = np.random.choice(len(coords), 5000, replace=False)
        coords_sub = coords[idx]
        c_sub = colors_center[idx] if isinstance(colors_center, np.ndarray) else colors_center
    else:
        coords_sub = coords
        c_sub = colors_center
    ax4.scatter(coords_sub[:, 0], coords_sub[:, 1], coords_sub[:, 2],
                s=dot_size_center, c=c_sub, alpha=0.5, edgecolors='none',
                label=f'centers ({len(coords)})')
    ax4.set_xlim(0, res)
    ax4.set_ylim(0, res)
    ax4.set_zlim(0, res)
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_zlabel('Z')
    ax4.set_title('3D View')
    ax4.legend(fontsize=8, loc='upper left')
    
    # Stats text
    if use_dense:
        occ_count = occupied.sum()
        occ_ratio = occ_count / (res ** 3) * 100
        fig.text(0.5, 0.01, 
                 f"Resolution: {res}³ | Centers: {len(coords)} | "
                 f"Filled cells: {occ_count} ({occ_ratio:.1f}%) | "
                 f"Grid range: X[{coords[:,0].min()}-{coords[:,0].max()}] "
                 f"Y[{coords[:,1].min()}-{coords[:,1].max()}] "
                 f"Z[{coords[:,2].min()}-{coords[:,2].max()}]",
                 ha='center', fontsize=9, style='italic')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[visualize_grid_coords] Saved to: {save_path}")
    else:
        plt.show()
    plt.close(fig)


def load_sparse_structure(path: str, device: torch.device = None) -> SparseStructure:
    """
    Load SparseStructure from a .pt file.
    
    Args:
        path: path to .pt file containing (1, 1, 64, 64, 64) tensor
        device: torch device
        
    Returns:
        SparseStructure object
    """
    return SparseStructure.from_file(path, device=device)


class VoxelLoader:
    """
    Loader for GeoSVR voxel data from checkpoint files.
    
    This class handles loading voxel data from .pt files exported by
    the 3D segmentation pipeline (segm_3d.py).
    """
    
    def __init__(self, device: torch.device = device):
        self.device = device
    
    def load(self, checkpoint_path: str) -> GeoSVRVoxelData:
        """
        Load voxel data from a checkpoint file using SparseVoxelModel.
        
        Args:
            checkpoint_path: path to .pt voxel file
            
        Returns:
            GeoSVRVoxelData object containing the loaded voxel data
        """
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        print(f"[VoxelLoader] Loading voxel data from: {checkpoint_path}")
        
        # Create a minimal config for SparseVoxelModel
        cfg_model = CfgNode()
        cfg_model.model_path = os.path.dirname(checkpoint_path)
        cfg_model.vox_geo_mode = "triinterp1"
        cfg_model.density_mode = "exp_linear_11"
        cfg_model.sh_degree = 3
        cfg_model.ss = 1.5
        cfg_model.outside_level = 5
        cfg_model.white_background = True
        cfg_model.black_background = False
        
        # Load using SparseVoxelModel
        voxel_model = SparseVoxelModel(cfg_model)
        voxel_model.load(checkpoint_path)
        
        # Extract voxel properties from the loaded model
        centers = voxel_model.vox_center.cpu()  # (N, 3)
        sizes = voxel_model.vox_size.cpu()      # (N, 1)
        octlevel = voxel_model.octlevel.cpu()   # (N, 1) or (N,)
        
        # Convert SH coefficients to RGB colors
        # _sh0 is (N, 3) in the new format
        sh0 = voxel_model._sh0.data.cpu()  # (N, 3)
        colors = self._sh0_to_rgb(sh0)     # (N, 3)
        colors = colors.clamp(0, 1)
        
        # Extract metadata if available
        original_id = None
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if '_metadata' in state_dict:
            original_id = state_dict['_metadata'].get('original_id', None)
        
        # Flatten octlevel if needed (convert (N, 1) to (N,))
        if octlevel.dim() > 1:
            octlevel = octlevel.squeeze(-1)
        
        # Create voxel data object
        voxel_data = GeoSVRVoxelData(
            centers=centers,
            sizes=sizes,
            colors=colors,
            octlevel=octlevel,
            original_id=original_id
        )
        
        print(f"[VoxelLoader] Loaded {voxel_data.num_voxels} voxels")
        print(f"[VoxelLoader] Bounding box: min={voxel_data.bbox_min.tolist()}, max={voxel_data.bbox_max.tolist()}")
        
        return voxel_data.to(self.device)
    
    def _sh0_to_rgb(self, sh0: torch.Tensor) -> torch.Tensor:
        """
        Convert SH0 (zeroth-order spherical harmonic) to RGB color.
        
        The SH0 coefficient represents the DC component of the color,
        which needs to be converted to actual RGB values.
        
        Args:
            sh0: (N, 3) SH0 coefficients
            
        Returns:
            (N, 3) RGB colors in [0, 1]
        """
        # SH0 to RGB conversion: rgb = sh0 * C0 + 0.5
        # where C0 = 0.28209479177387814 (normalization constant)
        C0 = 0.28209479177387814
        return sh0 * C0 + 0.5


class VoxelToHashGridConverter:
    """
    Converter from sparse GeoSVR voxels to HashGridVoxel implicit representation.
    
    This class handles:
    1. Converting sparse voxels to dense 64^3 occupancy grid
    2. Fitting a HashGridVoxel neural network to represent the voxels
    3. Providing the trained model for SDS optimization
    """
    
    def __init__(self, resolution: int = 64, device: torch.device = device):
        """
        Initialize the converter.
        
        Args:
            resolution: target voxel grid resolution (default: 64)
            device: torch device for computation
        """
        self.resolution = resolution
        self.device = device
        self.model = None
    
    def sparse_to_dense(self, voxel_data: GeoSVRVoxelData, 
                        normalize: bool = True) -> torch.Tensor:
        """
        Convert sparse voxel representation to dense 64^3 occupancy grid.
        
        Args:
            voxel_data: GeoSVRVoxelData object with sparse voxels
            normalize: if True, normalize coordinates to [-0.5, 0.5] range
            
        Returns:
            (1, 1, 64, 64, 64) dense occupancy grid
        """
        res = self.resolution
        
        # Get voxel centers and sizes
        centers = voxel_data.centers  # (N, 3)
        sizes = voxel_data.sizes.squeeze(-1)  # (N,)
        
        if normalize:
            # Normalize centers to [-0.5, 0.5] range based on bounding box
            # Add small padding to avoid boundary issues
            padding = 0.05
            bbox_min = voxel_data.bbox_min - voxel_data.bbox_extent * padding
            bbox_max = voxel_data.bbox_max + voxel_data.bbox_extent * padding
            bbox_extent = bbox_max - bbox_min
            
            # Normalize to [0, 1] then to [-0.5, 0.5]
            centers_norm = (centers - bbox_min) / bbox_extent - 0.5
            sizes_norm = sizes / bbox_extent.max()
        else:
            centers_norm = centers
            sizes_norm = sizes
        
        # Convert normalized coordinates to grid indices
        # Grid coordinates are in [-0.5, 0.5], map to [0, res-1]
        grid_coords = ((centers_norm + 0.5) * res).long()
        grid_coords = grid_coords.clamp(0, res - 1)
        
        # Create dense occupancy grid
        dense_grid = torch.zeros(1, 1, res, res, res, device=self.device)
        
        # Mark occupied voxels
        # Handle voxel sizes by filling multiple grid cells if needed
        for i in range(len(centers)):
            x, y, z = grid_coords[i]
            # Compute the number of grid cells this voxel spans
            half_size = int(max(1, (sizes_norm[i] * res / 2).item()))
            # half_size = int(round((sizes_norm[i] * res / 2).item()))
            
            x_min = max(0, x - half_size)
            x_max = min(res, x + half_size + 1)
            y_min = max(0, y - half_size)
            y_max = min(res, y + half_size + 1)
            z_min = max(0, z - half_size)
            z_max = min(res, z + half_size + 1)
            
            dense_grid[0, 0, x_min:x_max, y_min:y_max, z_min:z_max] = 1.0
        
        print(f"[VoxelToHashGrid] Dense grid occupancy: {dense_grid.sum().item():.0f} / {res**3}")
        
        return dense_grid
    
    def fit_hashgrid(self, dense_voxels: torch.Tensor,
                     num_iters: int = 10000,
                     lr: float = 0.001,
                     print_interval: int = 500) -> 'HashGridVoxel':
        """
        Fit a HashGridVoxel model to the dense voxel grid.
        
        This trains a neural implicit representation that can be used
        for SDS optimization with TRELLIS.
        
        Args:
            dense_voxels: (1, 1, 64, 64, 64) dense occupancy grid
            num_iters: number of training iterations
            lr: learning rate
            print_interval: interval for printing loss
            
        Returns:
            trained HashGridVoxel model
        """
        from network.hash_grid import HashGridVoxel, generate_image_grid_3d
        
        print(f"[VoxelToHashGrid] Fitting HashGridVoxel model...")
        
        # Initialize model
        model = HashGridVoxel().to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_iters, eta_min=lr * 0.01
        )
        
        # Target voxels
        target = dense_voxels.to(self.device)
        
        with torch.enable_grad():
            pbar = tqdm(range(1, num_iters + 1), desc="Fitting HashGrid")
            for i in pbar:
                # Generate sample grid with jitter for better generalization
                grid = generate_image_grid_3d(self.resolution).to(self.device)
                grid = grid + (torch.rand_like(grid) - 0.5) / self.resolution
                grid = grid.view(-1, 3)
                
                # Forward pass
                voxels_pred = model(grid).unsqueeze(0)  # (1, 1, 64, 64, 64)
                
                # MSE loss
                loss = F.mse_loss(voxels_pred, target)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                if i % print_interval == 0:
                    pbar.set_postfix(loss=f"{loss.item():.6f}")
        
        # Final evaluation
        with torch.no_grad():
            grid = generate_image_grid_3d(self.resolution).to(self.device)
            grid = grid.view(-1, 3)
            voxels_final = model(grid).unsqueeze(0)
            voxels_binary = (voxels_final > 0.5).float()
            
            iou = self._compute_iou(voxels_binary, target)
            print(f"[VoxelToHashGrid] Final IoU: {iou:.4f}")
        
        self.model = model
        return model
    
    def _compute_iou(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Compute Intersection over Union between predicted and target voxels."""
        pred_binary = pred > 0.5
        target_binary = target > 0.5
        
        intersection = (pred_binary & target_binary).sum().float()
        union = (pred_binary | target_binary).sum().float()
        
        return (intersection / (union + 1e-6)).item()
    
    def save_model(self, save_path: str):
        """Save the trained HashGridVoxel model."""
        if self.model is None:
            raise ValueError("No model to save. Call fit_hashgrid first.")
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.model.state_dict(), save_path)
        print(f"[VoxelToHashGrid] Model saved to: {save_path}")
    
    def load_model(self, load_path: str) -> 'HashGridVoxel':
        """Load a pre-trained HashGridVoxel model."""
        from network.hash_grid import HashGridVoxel
        
        model = HashGridVoxel().to(self.device)
        model.load_state_dict(torch.load(load_path, weights_only=True))
        self.model = model
        print(f"[VoxelToHashGrid] Model loaded from: {load_path}")
        return model


class ObjectVoxelProcessor:
    """
    High-level processor for object voxel data.
    
    This class provides a unified interface for:
    1. Loading segmented object voxels from GeoSVR
    2. Converting to HashGridVoxel representation
    3. Preparing data for SDS completion with TRELLIS
    """
    
    def __init__(self, output_dir: str, device: torch.device = device):
        """
        Initialize the processor.
        
        Args:
            output_dir: directory to save intermediate results
            device: torch device for computation
        """
        self.output_dir = output_dir
        self.device = device
        self.loader = VoxelLoader(device=device)
        self.converter = VoxelToHashGridConverter(device=device)
        
        os.makedirs(output_dir, exist_ok=True)
    
    def process(self, voxel_path: str, 
                fit_hashgrid: bool = True,
                num_iters: int = 10000) -> Dict:
        """
        Process an object voxel file.
        
        Args:
            voxel_path: path to the object voxel .pt file
            fit_hashgrid: if True, fit a HashGridVoxel model
            num_iters: number of iterations for HashGrid fitting
            
        Returns:
            dict containing:
                - 'voxel_data': GeoSVRVoxelData object
                - 'dense_grid': (1, 1, 64, 64, 64) dense occupancy
                - 'hashgrid_model': trained HashGridVoxel (if fit_hashgrid=True)
        """
        # Load voxel data
        voxel_data = self.loader.load(voxel_path)
        
        # Convert to dense grid
        dense_grid = self.converter.sparse_to_dense(voxel_data)
        
        # Save dense grid for visualization
        object_name = os.path.splitext(os.path.basename(voxel_path))[0]
        dense_path = os.path.join(self.output_dir, f"{object_name}_dense.pt")
        torch.save(dense_grid, dense_path)
        print(f"[ObjectProcessor] Dense grid saved to: {dense_path}")
        
        result = {
            'voxel_data': voxel_data,
            'dense_grid': dense_grid,
            'hashgrid_model': None
        }
        
        # Fit HashGridVoxel if requested
        if fit_hashgrid:
            model = self.converter.fit_hashgrid(dense_grid, num_iters=num_iters)
            
            # Save model
            model_path = os.path.join(self.output_dir, f"{object_name}_hashgrid.pth")
            self.converter.save_model(model_path)
            
            result['hashgrid_model'] = model
        
        return result
    
    def visualize(self, dense_grid: torch.Tensor, save_path: str):
        """
        Visualize the dense voxel grid.
        
        Args:
            dense_grid: (1, 1, 64, 64, 64) dense occupancy grid
            save_path: path to save the visualization
        """
        from decomvoxel.pipeline.visualize_sparse_structure import visualize_sparse_structure
        
        output_dir = os.path.dirname(save_path) if os.path.dirname(save_path) else '.'
        visualize_sparse_structure(dense_grid, output_dir=output_dir, mode='all')
        print(f"[ObjectProcessor] Visualization saved to: {output_dir}")


def load_object_voxel(voxel_path: str, device: torch.device = device) -> GeoSVRVoxelData:
    """
    Convenience function to load object voxel data.
    
    Args:
        voxel_path: path to the object voxel .pt file
        device: torch device
        
    Returns:
        GeoSVRVoxelData object
    """
    loader = VoxelLoader(device=device)
    return loader.load(voxel_path)


def load_and_convert_to_sparse_structure(voxel_path: str, 
                                          device: torch.device = device) -> SparseStructure:
    """
    Convenience function to load voxel and convert to SparseStructure in one step.
    
    Args:
        voxel_path: path to the object voxel .pt file
        device: torch device
        
    Returns:
        SparseStructure object
    """
    voxel_data = load_object_voxel(voxel_path, device=device)
    return voxel_to_sparse_structure(voxel_data, device=device)


def convert_to_hashgrid(voxel_data: GeoSVRVoxelData, 
                        num_iters: int = 10000,
                        device: torch.device = device) -> Tuple[torch.Tensor, 'HashGridVoxel']:
    """
    Convenience function to convert voxel data to HashGridVoxel.
    
    Args:
        voxel_data: GeoSVRVoxelData object
        num_iters: number of training iterations
        device: torch device
        
    Returns:
        tuple of (dense_grid, hashgrid_model)
    """
    converter = VoxelToHashGridConverter(device=device)
    dense_grid = converter.sparse_to_dense(voxel_data)
    model = converter.fit_hashgrid(dense_grid, num_iters=num_iters)
    return dense_grid, model


# ============================================================================
# Test code
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test voxel loading and conversion')
    parser.add_argument('--input', '-i', type=str, default=None,
                        help='Path to voxel .pt file')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output directory')
    parser.add_argument('--mode', '-m', type=str, default='sparse_structure',
                        choices=['sparse_structure', 'hashgrid', 'both'],
                        help='Conversion mode')
    parser.add_argument('--visualize', '-v', action='store_true',
                        help='Generate visualizations')
    args = parser.parse_args()
    
    # Default test path
    if args.input is None:
        test_voxel_path = os.path.join(
            PROJECT_ROOT, 
            "datasets/replica_large/scan1/semantic_result/object_voxels/object_002_voxels.pt"
        )
    else:
        test_voxel_path = args.input
    
    if not os.path.exists(test_voxel_path):
        print(f"File not found: {test_voxel_path}")
        print("Please provide a valid voxel file path with --input")
        sys.exit(1)
    
    # Output directory
    test_output_dir = args.output or os.path.join(PROJECT_ROOT, "outputs/test_voxel_loading")
    os.makedirs(test_output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Voxel Loading and Conversion Test")
    print("=" * 60)
    print(f"Input: {test_voxel_path}")
    print(f"Output: {test_output_dir}")
    print(f"Mode: {args.mode}")
    print("=" * 60)
    
    # Load voxel data
    loader = VoxelLoader()
    voxel_data = loader.load(test_voxel_path)
    
    print(f"\nLoaded voxel data:")
    print(f"  - Num voxels: {voxel_data.num_voxels}")
    print(f"  - Bbox min: {voxel_data.bbox_min.tolist()}")
    print(f"  - Bbox max: {voxel_data.bbox_max.tolist()}")
    
    # Test SparseStructure conversion
    if args.mode in ['sparse_structure', 'both']:
        print("\n" + "-" * 40)
        print("Testing SparseStructure conversion...")
        print("-" * 40)
        
        ss = voxel_to_sparse_structure(voxel_data)
        print(f"SparseStructure: {ss}")
        
        # Save
        ss_path = os.path.join(test_output_dir, "sparse_structure.pt")
        ss.save(ss_path)
        
        # Visualize
        if args.visualize:
            from decomvoxel.pipeline.visualize_sparse_structure import visualize_sparse_structure
            visualize_sparse_structure(ss.data, output_dir=test_output_dir, mode='all')
    
    # Test HashGrid conversion (legacy)
    if args.mode in ['hashgrid', 'both']:
        print("\n" + "-" * 40)
        print("Testing HashGrid conversion...")
        print("-" * 40)
        
        converter = VoxelToHashGridConverter()
        dense_grid = converter.sparse_to_dense(voxel_data)
        print(f"Dense grid shape: {dense_grid.shape}")
        print(f"Occupancy: {dense_grid.sum().item():.0f}")
        
        # Save dense grid
        dense_path = os.path.join(test_output_dir, "dense_grid.pt")
        torch.save(dense_grid, dense_path)
        print(f"Saved dense grid to: {dense_path}")
    
    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)
