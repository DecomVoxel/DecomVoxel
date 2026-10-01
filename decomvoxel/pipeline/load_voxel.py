"""
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
from decomvoxel.utils.visualization import visualize_grid_coords
from decomvoxel.utils.uncertainty import compute_voxel_certainty_base, compute_voxel_uncertainty_base, normalize_uncertainty, normalize_certainty, normalize_certainty_log
from decomvoxel.utils.converting import compute_ss_to_svr_ratio

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

        has_certainty = hasattr(self, 'certainty_grid') and self.certainty_grid is not None
        has_blank_unc = hasattr(self, 'blank_space_uncertainty') and self.blank_space_uncertainty is not None

        # Fast path for exact 90° multiples
        if angle_mod == 90.0:
            # CCW 90°: (x, y) -> (-y, x)  =>  transpose XY then flip X
            self.data = self.data.transpose(2, 3).flip(2)
            if has_certainty:
                self.certainty_grid = self.certainty_grid.transpose(2, 3).flip(2)
            if has_blank_unc:
                self.blank_space_uncertainty = self.blank_space_uncertainty.transpose(2, 3).flip(2)
        elif angle_mod == 180.0:
            # 180°: flip X and flip Y
            self.data = self.data.flip(2).flip(3)
            if has_certainty:
                self.certainty_grid = self.certainty_grid.flip(2).flip(3)
            if has_blank_unc:
                self.blank_space_uncertainty = self.blank_space_uncertainty.flip(2).flip(3)
        elif angle_mod == 270.0:
            # CW 90° (= CCW 270°): transpose XY then flip Y
            self.data = self.data.transpose(2, 3).flip(3)
            if has_certainty:
                self.certainty_grid = self.certainty_grid.transpose(2, 3).flip(3)
            if has_blank_unc:
                self.blank_space_uncertainty = self.blank_space_uncertainty.transpose(2, 3).flip(3)
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
            if has_certainty:
                cert_vol = self.certainty_grid.squeeze(0).squeeze(0).cpu().numpy()  # (X, Y, Z)
                cert_rotated = ndimage_rotate(
                    cert_vol, angle, axes=(0, 1), reshape=False,
                    order=1, mode='constant', cval=0.0
                )
                cert_rotated = cert_rotated.astype(np.float32)
                self.certainty_grid = torch.from_numpy(cert_rotated).unsqueeze(0).unsqueeze(0).to(self.device)
            if has_blank_unc:
                bsu_vol = self.blank_space_uncertainty.squeeze(0).squeeze(0).cpu().numpy()  # (X, Y, Z)
                bsu_rotated = ndimage_rotate(
                    bsu_vol, angle, axes=(0, 1), reshape=False,
                    order=1, mode='constant', cval=0.0
                )
                bsu_rotated = bsu_rotated.astype(np.float32)
                self.blank_space_uncertainty = torch.from_numpy(bsu_rotated).unsqueeze(0).unsqueeze(0).to(self.device)

        self._update_stats()
        print(f"[SparseStructure] Rotated {angle}° around Z-axis")
        return self
    
    def __repr__(self) -> str:
        return (f"SparseStructure(resolution={self.resolution}, "
                f"occupancy={self.occupancy_count}/{self.resolution**3} "
                f"({self.occupancy_ratio*100:.1f}%), device={self.device})")


def convert_voxel_to_sparse_structure(voxel_data: GeoSVRVoxelData,
                                       resolution: int = 64,
                                       device: torch.device = None,
                                       normalize: bool = True,
                                       padding: float = 0.3, # 0.4,
                                       robust_percentile: float = 0.02,
                                       uncertainty_threshold: float = 0.1,
                                       vis_grid=None,
                                       model_path: str = None,
                                       floor_min_z: float = None,
                                       bbox_modify: list = None) -> SparseStructure:
    """
    Convert GeoSVR voxel data to SparseStructure.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res = resolution

    # Get voxel centers and sizes
    centers = voxel_data.centers.to(device)  # (N, 3)
    sizes = voxel_data.sizes.squeeze(-1).to(device)  # (N,)
    mins = centers - sizes.unsqueeze(-1) / 2  # (N, 3)
    
    uncertainty_base = compute_voxel_uncertainty_base(voxel_data)
    uncertainty_base_norm = normalize_uncertainty(uncertainty_base)
    certainty_base = compute_voxel_certainty_base(voxel_data)
    certainty_base_norm = normalize_certainty(certainty_base)

    if DEBUG:
        n = len(centers)
        octlevel_dev = voxel_data.octlevel.to(device) if voxel_data.octlevel is not None else None

        # --- Per-level statistics ---
        print(f"\n[convert_voxel_to_sparse_structure] === Debug: {n} voxels total ===")
        print(f"  Per-octlevel summary:")
        if octlevel_dev is not None:
            for lvl in octlevel_dev.unique().sort().values:
                mask = octlevel_dev == lvl
                lvl_sizes = sizes[mask]
                print(f"    level {lvl.item():>3d}: count={mask.sum().item():>7d}, "
                      f"size min={lvl_sizes.min().item():.6f}, "
                      f"max={lvl_sizes.max().item():.6f}, "
                      f"mean={lvl_sizes.mean().item():.6f}")
        else:
            print("    (octlevel not available)")
        print(f"  sizes  range: min={sizes.min().item():.6f}, max={sizes.max().item():.6f}, mean={sizes.mean().item():.6f}")

        # --- 10 random voxels: uncertainty, size, octlevel ---
        sample_idx = torch.randperm(n)[:min(10, n)]
        print(f"\n  Random {len(sample_idx)} voxel samples (uncertainty / size / octlevel):")
        unc = uncertainty_base.to(device) if torch.is_tensor(uncertainty_base) else None
        unc_norm = uncertainty_base_norm.to(device) if torch.is_tensor(uncertainty_base_norm) else None
        for idx in sample_idx:
            s = sizes[idx].cpu().item()
            lvl = octlevel_dev[idx].cpu().item() if octlevel_dev is not None else 'N/A'
            u_raw = unc[idx].cpu().item() if unc is not None else float('nan')
            u_nrm = unc_norm[idx].cpu().item() if unc_norm is not None else float('nan')
            print(f"    voxel {idx.item():>6d}: size={s:.6f}, octlevel={lvl}, "
                  f"uncertainty={u_raw:.6f} (norm={u_nrm:.4f})")
    
    # Uncertainty-based filtering: remove high-uncertainty voxels before bbox calculation
    centers_certain = centers  # fallback: use all voxels
    if uncertainty_threshold is not None and uncertainty_threshold > 0.0:
        unc_norm_dev = uncertainty_base_norm.to(device) if torch.is_tensor(uncertainty_base_norm) else None
        if unc_norm_dev is not None:
            keep_mask = unc_norm_dev <= uncertainty_threshold
            n_before = len(centers)
            centers_certain = centers[keep_mask]
            print(f"[Uncertainty filter] threshold={uncertainty_threshold:.4f}: "
                  f"{keep_mask.sum().item()}/{n_before} voxels kept "
                  f"({keep_mask.sum().item()/n_before*100:.1f}%)")

    # Use percentile-based bbox to filter out outlier floating voxels
    lo = robust_percentile
    hi = 1.0 - robust_percentile
    bbox_min_robust = torch.quantile(centers_certain, lo, dim=0)
    bbox_max_robust = torch.quantile(centers_certain, hi, dim=0)
    bbox_extent_robust = bbox_max_robust - bbox_min_robust
    
    if floor_min_z is not None and floor_min_z < bbox_min_robust[2].item():
        bbox_floor_extent = bbox_max_robust[2] - floor_min_z
        shrink_factor = 1.0
        bbox_floor_extent_shrink = bbox_floor_extent * shrink_factor
        floor_min_z_shrink = bbox_max_robust[2] - bbox_floor_extent_shrink
        bbox_min_robust[2] = min(bbox_min_robust[2], floor_min_z_shrink)
        bbox_extent_robust[2] = bbox_max_robust[2] - bbox_min_robust[2]
    
    # Chair bbox modify: expand robust bbox toward nearest table in XY.
    # bbox_modify = [dx_min, dx_max, dy_min, dy_max, dz_min, dz_max], all >= 0.
    if bbox_modify is not None:
        dx_min, dx_max, dy_min, dy_max = bbox_modify[0], bbox_modify[1], bbox_modify[2], bbox_modify[3]
        if dx_min > 0:
            bbox_min_robust[0] = bbox_min_robust[0] - dx_min
        if dx_max > 0:
            bbox_max_robust[0] = bbox_max_robust[0] + dx_max
        if dy_min > 0:
            bbox_min_robust[1] = bbox_min_robust[1] - dy_min
        if dy_max > 0:
            bbox_max_robust[1] = bbox_max_robust[1] + dy_max
        bbox_extent_robust = bbox_max_robust - bbox_min_robust
        print(f"[bbox_modify] Applied chair bbox expand: {bbox_modify[:4]}")

    # Magic command kept from earlier experiments.
    # bbox_min_robust[0] = bbox_min_robust[0] - 0.5

    # Apply padding on the robust bbox
    bbox_min = bbox_min_robust - bbox_extent_robust * padding
    bbox_max = bbox_max_robust + bbox_extent_robust * padding
    bbox_extent = bbox_max - bbox_min
    max_extent = bbox_extent.max()

    print(f"[Robust normalize] percentile={lo:.2%}~{hi:.2%}")
    print(f"  robust bbox min: {bbox_min_robust.cpu().numpy()}, max: {bbox_max_robust.cpu().numpy()}")
    print(f"  padded bbox extent: {bbox_extent.cpu().numpy()}, max_extent: {max_extent.item():.6f}")

    # Filter out outliers, then compute mean as center (error)
    # inlier_mask = ((centers >= bbox_min_robust) & (centers <= bbox_max_robust)).all(dim=1)
    # mean_center = centers[inlier_mask].mean(dim=0)

    # Use robust bbox center as the cube center
    robust_center = (bbox_min_robust + bbox_max_robust) / 2 # object bbox center
    cube_min = robust_center - max_extent / 2 # object bbox min

    centers_norm = (centers - cube_min) / max_extent
    sizes_norm = sizes / max_extent
    mins_norm = centers_norm - sizes_norm.unsqueeze(-1) / 2

    print(f"  robust center: {robust_center.cpu().numpy()}, cube_min: {cube_min.cpu().numpy()}")
    
    
    # Code with center + half size
    # Convert to grid indices [0, res-1]
    grid_coords = (centers_norm * res).long()
    grid_coords = grid_coords.clamp(0, res - 1)
    
    if DEBUG:
        # print the first 10 centers:
        for i in range(min(10, len(centers))):
            print(f"   voxel {i}: center={centers[i].cpu().numpy()}, size={sizes[i].item():.6f}")
            print(f"   normalized: center={centers_norm[i].cpu().numpy()}, size={sizes_norm[i].item():.6f}")
            print(f"   grid coords: {grid_coords[i].cpu().numpy()}")

    # Create dense occupancy grid
    dense_grid = torch.zeros(1, 1, res, res, res, device=device)
    certainty_grid = torch.zeros(res, res, res, device=device)
    cert_norm_dev = certainty_base_norm.to(device) if torch.is_tensor(certainty_base_norm) else torch.ones(len(centers), device=device)

    # Main bottleneck: this step is relatively slow.
    # half_sizes = (sizes_norm * res / 2).long()  # (N,)
    half_sizes = (sizes_norm * res / 2).clamp(min=1).long()  # (N,)
    unique_hs = half_sizes.unique()

    for hs in unique_hs:
        mask = half_sizes == hs
        coords = grid_coords[mask]  # (M, 3)
        cert = cert_norm_dev[mask]  # (M,)
        hs_val = hs.item()
    
        # Build (2h+1)^3 offset grid, broadcast over all centres
        offsets = torch.arange(-hs_val, hs_val + 1, device=device)
        ox, oy, oz = torch.meshgrid(offsets, offsets, offsets, indexing='ij')
        offset_grid = torch.stack([ox.reshape(-1), oy.reshape(-1), oz.reshape(-1)], dim=1)  # (K, 3)

        # Process in batches to avoid OOM when M * K is large
        # K = (2*hs_val+1)^3; keep each batch ≤ ~4 M elements
        K = offset_grid.shape[0]
        batch_size = max(1, 4_000_000 // K)
        for b_start in range(0, len(coords), batch_size):
            b_coords = coords[b_start:b_start + batch_size]  # (B, 3)
            b_cert   = cert[b_start:b_start + batch_size]    # (B,)
            # (B, 1, 3) + (1, K, 3) -> (B, K, 3)
            bc = b_coords.unsqueeze(1) + offset_grid.unsqueeze(0)
            bc = bc.reshape(-1, 3)
            bc.clamp_(0, res - 1)
            dense_grid[0, 0, bc[:, 0], bc[:, 1], bc[:, 2]] = 1.0
            cert_vals = b_cert.unsqueeze(1).expand(-1, K).reshape(-1)
            current = certainty_grid[bc[:, 0], bc[:, 1], bc[:, 2]]
            certainty_grid[bc[:, 0], bc[:, 1], bc[:, 2]] = torch.maximum(current, cert_vals)


    # History code with mins and sizes
    # min_grid_coords = (mins_norm * res).long().clamp(0, res - 1)  # (N, 3)
    # size_cells = sizes_norm * res  # (N,)
    # size_cells_int = size_cells.long().clamp(min=1) # .long casts to int, clamp prevents out-of-range values

    # cert_norm_dev = certainty_base_norm.to(device) if torch.is_tensor(certainty_base_norm) else torch.ones(len(centers), device=device)
    
    # # Temporarily disabled because objects can be processed independently
    # ss_to_svr_ratio = compute_ss_to_svr_ratio(size_cells, voxel_data)

    # dense_grid = torch.zeros(1, 1, res, res, res, device=device)
    # certainty_grid = torch.zeros(res, res, res, device=device)

    # for sc in size_cells_int.unique():
    #     mask = size_cells_int == sc
    #     m_coords = min_grid_coords[mask]  # (M, 3)
    #     cert = cert_norm_dev[mask]        # (M,)
    #     sc_val = sc.item()

    #     offsets = torch.arange(0, sc_val, device=device)
    #     ox, oy, oz = torch.meshgrid(offsets, offsets, offsets, indexing='ij')
    #     offset_grid = torch.stack([ox.reshape(-1), oy.reshape(-1), oz.reshape(-1)], dim=1)  # (K, 3)
    #     K = offset_grid.shape[0]

    #     batch_size = max(1, 4_000_000 // K)
    #     for b_start in range(0, len(m_coords), batch_size):
    #         b_coords = m_coords[b_start:b_start + batch_size]   # (B, 3)
    #         b_cert   = cert[b_start:b_start + batch_size]       # (B,)
    #         bc = (b_coords.unsqueeze(1) + offset_grid.unsqueeze(0)).reshape(-1, 3)
    #         bc.clamp_(0, res - 1)
    #         dense_grid[0, 0, bc[:, 0], bc[:, 1], bc[:, 2]] = 1.0
    #         cert_vals = b_cert.unsqueeze(1).expand(-1, K).reshape(-1)
    #         current = certainty_grid[bc[:, 0], bc[:, 1], bc[:, 2]]
    #         certainty_grid[bc[:, 0], bc[:, 1], bc[:, 2]] = torch.maximum(current, cert_vals)



    # Certainty filter: remove grid cells whose certainty is below threshold
    certainty_threshold = 0.03
    low_cert_mask = certainty_grid < certainty_threshold  # (R, R, R)
    n_before = int(dense_grid[0, 0].sum().item())
    dense_grid[0, 0][low_cert_mask] = 0.0
    n_after = int(dense_grid[0, 0].sum().item())
    print(f"[Certainty filter] threshold={certainty_threshold}: "
          f"{n_before} -> {n_after} occupied cells "
          f"(removed {n_before - n_after})")


    if DEBUG:
        # Debug: verify grid filling statistics
        sizes_in_cells = sizes_norm * res  # voxel edge length in grid cells
        half_sizes = (sizes_in_cells / 2).long()
        footprints = (2 * half_sizes + 1)  # actual cell span per axis
        print(f"\n[convert_voxel_to_sparse_structure] === Grid filling debug ===")
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
        # At this point, visualized grid coords are already floored integers,
        # so they appear very regular and have heavy overlap.
        debug_path = os.path.join(PROJECT_ROOT, "outputs", "debug", "grid_coords_debug.png")
        visualize_grid_coords(
            grid_coords, dense_grid=dense_grid, resolution=res,
            save_path=debug_path,
            title=f"convert_voxel_to_sparse_structure Debug (N={len(centers)}, res={res})",
            sizes_in_cells=sizes_in_cells
        )
    

    # History code mins & centers debug
    # if DEBUG:
    #     sizes_in_cells = sizes_norm * res
    #     print(f"\n[convert_voxel_to_sparse_structure] === Grid filling debug ===")
    #     print(f"  voxel size in grid cells: min={sizes_in_cells.min().item():.2f}, "
    #           f"max={sizes_in_cells.max().item():.2f}, mean={sizes_in_cells.mean().item():.2f}")
    #     print(f"  size_cells (clamped): min={size_cells_int.min().item()}, max={size_cells_int.max().item()}")
    #     print(f"  min_grid_coords range: min={min_grid_coords.min(dim=0).values.cpu().numpy()}, "
    #           f"max={min_grid_coords.max(dim=0).values.cpu().numpy()}")
    #     sample_idx = torch.randperm(len(centers))[:min(5, len(centers))]
    #     for idx in sample_idx:
    #         gc = min_grid_coords[idx].cpu().numpy()
    #         sc_v = size_cells_int[idx].item()
    #         print(f"  voxel {idx.item()}: min_grid=({gc[0]},{gc[1]},{gc[2]}), size_cells={sc_v}")
    #     debug_path = os.path.join(PROJECT_ROOT, "outputs", "debug", "grid_coords_debug.png")
    #     visualize_grid_coords(
    #         min_grid_coords, dense_grid=dense_grid, resolution=res,
    #         save_path=debug_path,
    #         title=f"convert_voxel_to_sparse_structure Debug (N={len(centers)}, res={res})",
    #         sizes_in_cells=sizes_in_cells
    #     )
        
    

    # We take max at each cell, so no extra normalization is needed
    # because certainty is already in [0, 1].
    # certainty_grid = normalize_certainty(certainty_grid)

    # Create SparseStructure
    ss = SparseStructure(data=dense_grid, device=device)
    ss.certainty_grid = certainty_grid.unsqueeze(0).unsqueeze(0)  # (1, 1, R, R, R)

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

    print(f"[convert_voxel_to_sparse_structure] Converted {voxel_data.num_voxels} voxels -> {ss}")
    print(f"[convert_voxel_to_sparse_structure] {transform_info}")
    
    # Compute blank-space uncertainty from scene visibility grid
    if vis_grid is not None or model_path is not None:
        from decomvoxel.pipeline.blank_space_uncertainty import get_object_blank_space_uncertainty
        ss.blank_space_uncertainty = get_object_blank_space_uncertainty(
            transform_info=transform_info,
            sparse_structure=ss,
            vis_grid=vis_grid,
            model_path=model_path,
            device=str(device),
        )
    else:
        ss.blank_space_uncertainty = None

    return ss

def convert_voxel_batch(voxel_data_list: List[GeoSVRVoxelData], resolution: int = 64, device: torch.device = None, normalize: bool = True) -> List[SparseStructure]:
    return [convert_voxel_to_sparse_structure(vd, resolution=resolution, device=device, normalize=normalize) for vd in voxel_data_list]

# HERE ENTRY POINT 2
def voxel_to_sparse_structure(voxel_data: GeoSVRVoxelData, resolution: int = 64, device: torch.device = None, robust_percentile: float = 0.02, uncertainty_threshold: float = 0.2, vis_grid=None, model_path: str = None, floor_min_z: float = None, bbox_modify: list = None) -> SparseStructure: # XXX
    return convert_voxel_to_sparse_structure(voxel_data, resolution=resolution, device=device, robust_percentile=robust_percentile, uncertainty_threshold=uncertainty_threshold, vis_grid=vis_grid, model_path=model_path, floor_min_z=floor_min_z, bbox_modify=bbox_modify)


class VoxelLoader:
    def __init__(self, device: torch.device = device):
        self.device = device
    
    def load(self, checkpoint_path: str) -> GeoSVRVoxelData:
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
        # SH0 to RGB conversion: rgb = sh0 * C0 + 0.5
        # where C0 = 0.28209479177387814 (normalization constant)
        C0 = 0.28209479177387814
        return sh0 * C0 + 0.5

# HERE ENTRY POINT 1
def load_object_voxel(voxel_path: str, device: torch.device = device) -> GeoSVRVoxelData:
    loader = VoxelLoader(device=device)
    return loader.load(voxel_path)


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
    
    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)
