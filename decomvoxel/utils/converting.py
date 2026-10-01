'''
Convert between SVR voxels, Sparse Structure, and Mesh coordinate systems.

This module bridges three representations:
    1. SVR Voxels (GeoSVRVoxelData): sparse voxels in world coordinates
    2. Sparse Structure (64^3 grid): TRELLIS-compatible occupancy grid in [0, res] index space
    3. TRELLIS Mesh: mesh vertices in normalized [-0.5, 0.5] space (z-up),
       or in GLB space (y-up) after export rotation
'''

'''
Contents:
class: TransformInfo
func: 
    - compute_transform_info
    - compute_ss_to_svr_ratio
    - sparse_structure_to_svr_voxels
    - inverse_transform_mesh
    - inverse_transform_vertices
    - load_and_inverse_transform_glb
'''

import os
import sys
from typing import Optional, Union, Dict, Any
from dataclasses import dataclass, field

import torch
import numpy as np
import trimesh

# Add project paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


@dataclass
class TransformInfo:
    """
    Stores the forward transformation parameters used when converting
    SVR voxels → Sparse Structure (64^3 grid).

    Forward transform:
        p_norm  = (p_world - cube_min) / max_extent   # → [0, 1]
        p_grid  = p_norm * resolution                  # → [0, res]

    Inverse transform:
        p_world = p_norm * max_extent + cube_min

    Attributes:
        cube_min: (3,) world-space origin of the normalization cube.
        max_extent: scalar, edge length of the normalization cube.
        resolution: int, grid resolution (default 64).
        rotation_angle_z: rotation angle (degrees, CCW positive) applied to the
                          sparse structure around the Z-axis before SDS/mesh
                          generation. inverse_transform_mesh will undo it.
        rotated_z90: **deprecated** – kept for backward compatibility.
                     If True and rotation_angle_z == 0, treated as 90°.
    """
    cube_min: torch.Tensor          # (3,)
    max_extent: float               # scalar
    resolution: int = 64
    rotation_angle_z: float = 0.0   # Arbitrary rotation angle (degrees, CCW positive)
    rotated_z90: bool = False       # DEPRECATED – use rotation_angle_z instead

    @property
    def effective_rotation_z(self) -> float:
        """Return the effective rotation angle, respecting the legacy flag."""
        if self.rotation_angle_z != 0.0:
            return self.rotation_angle_z
        if self.rotated_z90:
            return 90.0
        return 0.0

    def save(self, path: str):
        """Save transform info to a .pt file."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        torch.save({
            'cube_min': self.cube_min.cpu(),
            'max_extent': float(self.max_extent),
            'resolution': self.resolution,
            'rotation_angle_z': float(self.rotation_angle_z),
            'rotated_z90': self.rotated_z90,  # keep for legacy loaders
        }, path)
        print(f"[TransformInfo] Saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'TransformInfo':
        """Load transform info from a .pt file."""
        d = torch.load(path, map_location='cpu')
        return cls(
            cube_min=d['cube_min'],
            max_extent=d['max_extent'],
            resolution=d.get('resolution', 64),
            rotation_angle_z=float(d.get('rotation_angle_z', 0.0)),
            rotated_z90=d.get('rotated_z90', False),
        )

    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dictionary."""
        return {
            'cube_min': self.cube_min.cpu().tolist(),
            'max_extent': float(self.max_extent),
            'resolution': int(self.resolution),
            'rotation_angle_z': float(self.rotation_angle_z),
            'rotated_z90': bool(self.rotated_z90),
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'TransformInfo':
        """Create TransformInfo from a dictionary (inverse of to_dict)."""
        return cls(
            cube_min=torch.tensor(d['cube_min'], dtype=torch.float32),
            max_extent=float(d['max_extent']),
            resolution=int(d.get('resolution', 64)),
            rotation_angle_z=float(d.get('rotation_angle_z', 0.0)),
            rotated_z90=bool(d.get('rotated_z90', False)),
        )

    def __repr__(self) -> str:
        return (f"TransformInfo(cube_min={self.cube_min.cpu().numpy()}, "
                f"max_extent={self.max_extent:.6f}, res={self.resolution}, "
                f"rotation_angle_z={self.effective_rotation_z}°)")


def compute_transform_info(
    voxel_data,
    resolution: int = 64,
    padding: float = 0.05,
    robust_percentile: float = 0.02,
    device: torch.device = None,
) -> TransformInfo:
    """
    Compute the normalization transform info from GeoSVRVoxelData.

    This mirrors the normalization logic in VoxelToSparseStructure.convert()
    so that the exact same cube_min / max_extent are produced.
    """
    raise NotImplementedError("This is the old verson code.")
    # device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # centers = voxel_data.centers.to(device)

    # if robust_percentile > 0:
    #     lo = robust_percentile
    #     hi = 1.0 - robust_percentile
    #     bbox_min_robust = torch.quantile(centers, lo, dim=0)
    #     bbox_max_robust = torch.quantile(centers, hi, dim=0)
    #     bbox_extent_robust = bbox_max_robust - bbox_min_robust

    #     bbox_extent = bbox_extent_robust * (1 + 2 * padding)
    #     max_extent = bbox_extent.max().item()

    #     median_center = torch.median(centers, dim=0).values
    #     cube_min = median_center - max_extent / 2
    # else:
    #     bbox_min = voxel_data.bbox_min.to(device) - voxel_data.bbox_extent.to(device) * padding
    #     bbox_max = voxel_data.bbox_max.to(device) + voxel_data.bbox_extent.to(device) * padding
    #     bbox_extent = bbox_max - bbox_min
    #     max_extent = bbox_extent.max().item()

    #     mean_center = centers.mean(dim=0)
    #     cube_min = mean_center - max_extent / 2

    # return TransformInfo(
    #     cube_min=cube_min.cpu(),
    #     max_extent=max_extent,
    #     resolution=resolution,
    # )


def compute_ss_to_svr_ratio(
    size_cells: torch.Tensor,
    voxel_data,
) -> float:
    """
    Compute optimal scaling ratio so that one SS grid cell aligns exactly
    with an SVR octree level size.

    SVR voxel sizes form a geometric series with common ratio 1/2
    (higher octree level → smaller size).  After forward conversion the
    SS unit cell (= 1 grid cell) typically falls between two consecutive
    SVR levels l_{i-1} and l_i (measured in grid-cell units, descending).

    Relationships among three adjacent levels:
        l_{i-1} = 2 · l_i = 4 · l_{i+1}

    Three candidate snap targets:
        • l_{i-1}       (coarser level)
        • 3 · l_{i+1}   (= 3/2 · l_i, between the two levels)
        • l_i            (finer level)

    For each target we compute  ss_unit / target  and return the ratio
    closest to 1.0.  Multiplying max_extent by this ratio during the
    forward conversion makes the SS grid cell match the chosen SVR level.

    Args:
        size_cells: (N,) SVR voxel sizes in SS grid-cell units
                    (= svr_world_size / max_extent * resolution).
        voxel_data: GeoSVRVoxelData with octlevel information.

    Returns:
        float – scaling ratio closest to 1.0, or 1.0 if no bracketing
        levels are found.
    """
    # --- Determine per-level sizes in grid cells ---
    # unique_levels: ascending octree level numbers (low = coarse, high = fine)
    # level_sizes:   corresponding sizes in grid cells, will be sorted descending
    #                (large = coarse first, small = fine last)
    # The two lists are kept in 1-to-1 correspondence after sorting.
    has_octlevel = voxel_data.octlevel is not None
    if has_octlevel:
        octlevel = voxel_data.octlevel
        if octlevel.device != size_cells.device:
            octlevel = octlevel.to(size_cells.device)
        unique_levels_asc = octlevel.unique().sort().values
        level_sizes_asc = []
        for lvl in unique_levels_asc:
            mask = octlevel == lvl
            mean_sz = size_cells[mask].float().mean().item()
            level_sizes_asc.append(mean_sz)
        # unique_levels_asc is ascending by level number (low = coarse = large size).
        # level_sizes_asc is therefore already in descending size order.
        # Keep both as-is so they stay in 1-to-1 correspondence (coarse first).
        unique_levels = unique_levels_asc.tolist()  # ascending level numbers, coarse first
        level_sizes = level_sizes_asc               # descending sizes, coarse first
    else:
        unique_sz = size_cells.float().unique().sort(descending=True).values
        level_sizes = unique_sz.tolist()
        # Synthesize pseudo-level numbers (0 = coarsest, 1, 2, …)
        unique_levels = list(range(len(level_sizes)))

    ss_unit = 1.0  # one grid cell

    # --- Find two consecutive levels bracketing ss_unit ---
    bracket_idx = None  # index such that level_sizes[bracket_idx] >= ss_unit >= level_sizes[bracket_idx+1]

    for idx in range(len(level_sizes) - 1):
        if level_sizes[idx] >= ss_unit >= level_sizes[idx + 1]:
            bracket_idx = idx
            break

    if bracket_idx is None:
        # ss_unit is outside the range spanned by existing SVR levels.
        # Extend the geometric series (ratio 1/2) and update both lists.
        largest = level_sizes[0]
        smallest = level_sizes[-1]

        if ss_unit > largest:
            # Extend toward coarser (prepend): double sizes, decrement level numbers
            coarsest_lvl = unique_levels[0]
            cur_size = largest
            while cur_size < ss_unit:
                cur_size *= 2.0
                coarsest_lvl = coarsest_lvl - 1  # one level coarser
                level_sizes.insert(0, cur_size)
                unique_levels.insert(0, coarsest_lvl)
            # Bracket is the first pair
            bracket_idx = 0
        elif ss_unit < smallest:
            # Extend toward finer (append): halve sizes, increment level numbers
            finest_lvl = unique_levels[-1]
            cur_size = smallest
            while cur_size > ss_unit:
                cur_size /= 2.0
                finest_lvl = finest_lvl + 1  # one level finer
                level_sizes.append(cur_size)
                unique_levels.append(finest_lvl)
            # Bracket is the last pair
            bracket_idx = len(level_sizes) - 2
        else:
            print(f"[compute_ss_to_svr_ratio] WARNING: unexpected state, returning 1.0")
            return 1.0

        # print(f"[compute_ss_to_svr_ratio] Extended geometric series, "
        #       f"levels now: {unique_levels}, sizes: {[f'{s:.4f}' for s in level_sizes]}")

    li_minus1 = level_sizes[bracket_idx]
    li = level_sizes[bracket_idx + 1]
    level_minus1 = unique_levels[bracket_idx]
    level_i = unique_levels[bracket_idx + 1]

    # One level finer than l_i
    li_plus1 = li / 2.0

    # --- Three candidate ratios: ss_unit / target ---
    candidates = {
        'l_{i-1}':     ss_unit / li_minus1,
        '3*l_{i+1}':   ss_unit / (3.0 * li_plus1),
        'l_i':         ss_unit / li,
    }

    best_name = min(candidates, key=lambda k: abs(candidates[k] - 1.0))
    best_ratio = candidates[best_name]

    print(f"[compute_ss_to_svr_ratio] SVR level sizes (grid cells): "
          f"{[f'{s:.4f}' for s in level_sizes]}")
    print(f"  Unique SVR levels: {unique_levels}")
    print(f"  Bracketing: l_{{i-1}}={li_minus1:.4f}, l_i={li:.4f}, "
          f"l_{{i+1}}={li_plus1:.4f}")
    print(f"  level_{{i-1}} = {level_minus1}, level_i = {level_i}")
    print(f"  Candidates: "
          f"{', '.join(f'{k}={v:.6f}' for k, v in candidates.items())}")
    print(f"  Best: {best_name} -> ratio={best_ratio:.6f}")

    return best_ratio


def sparse_structure_to_svr_voxels(
    sparse_structure: torch.Tensor,
    transform_info: TransformInfo,
    threshold: Optional[float] = 0.5,
    device: torch.device = None,
    outside_level: int = 5,
    sh_degree: int = 3,
    geo_init: float = 3.0,
    ss_value: float = 1.5,
    per_voxel_geo: 'torch.Tensor | None' = None,  # (N,), (R,R,R) or (1,1,R,R,R)
    use_all_voxels: bool = False,
):
    """
    Convert a completed sparse structure (64^3 occupancy grid) back to
    SVR-style voxels in world coordinates, represented as a SparseVoxelModel.

    Each occupied grid cell becomes one voxel whose center is the cell center
    and whose size equals one grid cell in world units.

    Inverse transform correctness:
        Forward: grid_coord = floor((center_world - cube_min) / max_extent * res)
        Inverse: center_world = (grid_coord + 0.5) / res * max_extent + cube_min
        The +0.5 places the sample point at the cell centre (unbiased).

    The resulting SparseVoxelModel can be saved via SVInOut.save_state_dict or
    model.save(path), and rendered/exported with standard GeoSVR tools.

    Args:
        sparse_structure: (1,1,R,R,R) or (R,R,R) occupancy grid.
        transform_info: forward-pass normalisation parameters.
        threshold: occupancy threshold for binarizing the grid
            (used only when use_all_voxels=False).
        device: target device (defaults to CUDA if available).
        outside_level: number of octree levels outside the inside region.
            Determines scene_extent = inside_extent * 2^outside_level.
            Default 5 matches the GeoSVR training config.
        sh_degree: spherical-harmonics degree for colour (default 3).
        geo_init: initial geo_grid_pts value (positive → occupied density).
        ss_value: `ss` field stored in the model (sampling-step multiplier).
        per_voxel_geo: optional per-voxel geometry values.
            - (N,) aligned with selected voxels
            - (R,R,R) or (1,1,R,R,R) dense grid; values are gathered by grid_coords
        use_all_voxels: if True, convert every grid cell (ignore threshold).
            This is useful for differentiable soft rendering where geometry is
            controlled by dense per_voxel_geo rather than hard occupancy.

    Returns:
        dict with keys:
            'voxel_model'  – SparseVoxelModel ready for rendering / saving
            'centers'      – (N, 3) world-space voxel centres
            'sizes'        – (N, 1) world-space voxel sizes
            'grid_coords'  – (N, 3) integer grid indices
            'num_voxels'   – int
    """
    import sys, os
    _PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    _GEOSVR_ROOT = os.path.join(_PROJ_ROOT, 'decomvoxel', 'representation', 'GeoSVR')
    for _p in (_PROJ_ROOT, _GEOSVR_ROOT):
        if _p not in sys.path:
            sys.path.insert(0, _p)

    from yacs.config import CfgNode
    from src.sparse_voxel_model import SparseVoxelModel
    from src.sparse_voxel_gears.io import SVInOut
    from src.utils import octree_utils
    from src.utils.activation_utils import rgb2shzero

    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ss = sparse_structure.to(device)

    # ------------------------------------------------------------------
    # 1. Extract occupied grid coordinates  (N, 3) — integers in [0, res-1]
    # ------------------------------------------------------------------
    if ss.dim() == 5:
        grid = ss.squeeze(0).squeeze(0)   # (R, R, R)
    elif ss.dim() == 3:
        grid = ss
    else:
        raise ValueError(f"Unexpected sparse_structure shape: {ss.shape}")

    if grid.shape[0] != grid.shape[1] or grid.shape[1] != grid.shape[2]:
        raise ValueError(f"sparse_structure must be cubic, got shape {tuple(grid.shape)}")

    dense_geo_grid = None
    if per_voxel_geo is not None:
        if per_voxel_geo.dim() == 5:
            if per_voxel_geo.shape[0] != 1 or per_voxel_geo.shape[1] != 1:
                raise ValueError(f"per_voxel_geo 5D shape must be (1,1,R,R,R), got {tuple(per_voxel_geo.shape)}")
            dense_geo_grid = per_voxel_geo.squeeze(0).squeeze(0)
        elif per_voxel_geo.dim() == 3:
            dense_geo_grid = per_voxel_geo

        if dense_geo_grid is not None and tuple(dense_geo_grid.shape) != tuple(grid.shape):
            raise ValueError(
                f"Dense per_voxel_geo shape {tuple(dense_geo_grid.shape)} must match sparse_structure grid shape {tuple(grid.shape)}"
            )

    if use_all_voxels:
        _coords = torch.arange(grid.shape[0], device=device, dtype=torch.long)
        gx, gy, gz = torch.meshgrid(_coords, _coords, _coords, indexing='ij')
        grid_coords = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)  # (R^3, 3)
    else:
        if threshold is None:
            raise ValueError("threshold must be provided when use_all_voxels=False")
        occupied = grid > threshold
        grid_coords = torch.argwhere(occupied)  # (N, 3) int64

    # Keep a copy before optional inverse rotation so dense per_voxel_geo values
    # stay aligned with their original grid-cell probabilities.
    grid_coords_geo = grid_coords.clone()

    if grid_coords.shape[0] == 0:
        print("[converting] WARNING: no occupied voxels found.")
        return {
            'voxel_model': None,
            'centers': torch.zeros(0, 3),
            'sizes': torch.zeros(0, 1),
            'grid_coords': torch.zeros(0, 3, dtype=torch.long),
            'num_voxels': 0,
        }

    res = transform_info.resolution
    if grid.shape[0] != res:
        raise ValueError(
            f"Grid resolution mismatch: sparse_structure has R={grid.shape[0]}, "
            f"but transform_info.resolution={res}"
        )
    cube_min = transform_info.cube_min.to(device)
    max_extent = float(transform_info.max_extent)

    # ------------------------------------------------------------------
    # 2. Undo optional Z rotation on grid coordinates
    # ------------------------------------------------------------------
    rot_angle = transform_info.effective_rotation_z
    if rot_angle != 0.0:
        cx = cy = (res - 1) / 2.0
        gc_f = grid_coords.float()
        x = gc_f[:, 0] - cx
        y = gc_f[:, 1] - cy
        z = gc_f[:, 2]
        rad = np.deg2rad(-rot_angle)  # inverse rotation
        cos_a, sin_a = float(np.cos(rad)), float(np.sin(rad))
        x_new = cos_a * x - sin_a * y + cx
        y_new = sin_a * x + cos_a * y + cy
        grid_coords = torch.stack([x_new.round().long().clamp(0, res - 1),
                                   y_new.round().long().clamp(0, res - 1),
                                   z.long()], dim=1)

    # ------------------------------------------------------------------
    # 3. Inverse transform: grid cell centre → world space
    #    Forward:  grid_coord = floor((p_world - cube_min) / max_extent * res)
    #    Inverse:  p_world = (grid_coord + 0.5) / res * max_extent + cube_min
    # ------------------------------------------------------------------
    centers_norm = (grid_coords.float() + 0.5) / res      # (N, 3) in (0, 1)
    centers_world = centers_norm * max_extent + cube_min   # (N, 3) world

    # Each grid cell has world-space edge length = max_extent / res
    voxel_size_world = max_extent / res
    sizes = torch.full((grid_coords.shape[0], 1), voxel_size_world, device=device)

    print(f"[converting] Sparse structure → {grid_coords.shape[0]} SVR voxels "
          f"(voxel_size_world={voxel_size_world:.6f})")
    print(
        f"[converting] DEBUG sparse_structure_to_svr_voxels:\n"
        f"  transform_info: cube_min={cube_min.cpu().numpy()}, max_extent={max_extent:.6f}, "
        f"res={res}, rot_angle={rot_angle}°\n"
        f"  grid_coords range: x=[{grid_coords[:,0].min().item()},{grid_coords[:,0].max().item()}], "
        f"y=[{grid_coords[:,1].min().item()},{grid_coords[:,1].max().item()}], "
        f"z=[{grid_coords[:,2].min().item()},{grid_coords[:,2].max().item()}]\n"
        f"  centers_norm range: x=[{centers_norm[:,0].min():.4f},{centers_norm[:,0].max():.4f}], "
        f"y=[{centers_norm[:,1].min():.4f},{centers_norm[:,1].max():.4f}], "
        f"z=[{centers_norm[:,2].min():.4f},{centers_norm[:,2].max():.4f}]\n"
        f"  centers_world range: x=[{centers_world[:,0].min():.4f},{centers_world[:,0].max():.4f}], "
        f"y=[{centers_world[:,1].min():.4f},{centers_world[:,1].max():.4f}], "
        f"z=[{centers_world[:,2].min():.4f},{centers_world[:,2].max():.4f}]"
    )

    # ------------------------------------------------------------------
    # 4. Build SparseVoxelModel from the extracted voxels
    #
    #    Scene geometry:
    #      inside_extent  = max_extent          (the normalised cube)
    #      scene_extent   = inside_extent * 2^outside_level
    #      scene_center   = cube_min + max_extent/2 * [1,1,1]
    #
    #    Octlevel for one SS grid cell:
    #      voxel_size = scene_extent * 2^(-octlevel)
    #      => octlevel = log2(scene_extent / voxel_size_world)
    #                  = log2(max_extent * 2^outside_level / (max_extent / res))
    #                  = log2(2^outside_level * res)
    #                  = outside_level + log2(res)
    #    e.g. outside_level=5, res=64 → octlevel = 5 + 6 = 11
    # ------------------------------------------------------------------
    inside_extent = max_extent
    scene_extent_val = inside_extent * float(2 ** outside_level)
    scene_center_val = cube_min + torch.tensor(
        [max_extent / 2, max_extent / 2, max_extent / 2], device=device)

    # octlevel must be an integer; res must be a power of 2
    log2_res = int(round(np.log2(res)))
    assert 2 ** log2_res == res, f"resolution {res} must be a power of 2"
    target_octlevel = outside_level + log2_res

    scene_center_t = scene_center_val.float().cuda()
    scene_extent_t = torch.tensor(scene_extent_val, dtype=torch.float32, device='cuda')
    inside_extent_t = torch.tensor(inside_extent, dtype=torch.float32, device='cuda')

    centers_cuda = centers_world.float().cuda()
    N = centers_cuda.shape[0]

    octlevel_t = torch.full((N, 1), target_octlevel, dtype=torch.int8, device='cuda')
    octpath_t = octree_utils.xyz_2_octpath(
        centers_cuda, octlevel_t, scene_center_t, scene_extent_t
    )  # (N, 1)

    grid_pts_key, vox_key = octree_utils.build_grid_pts_link(octpath_t, octlevel_t)
    M = grid_pts_key.shape[0]

    # geo_grid_pts: positive value → voxel is occupied (opaque)
    if per_voxel_geo is not None:
        # Gather dense per-voxel geo to selected coords, or validate explicit (N,) values.
        if dense_geo_grid is not None:
            _pvg = dense_geo_grid[
                grid_coords_geo[:, 0],
                grid_coords_geo[:, 1],
                grid_coords_geo[:, 2],
            ]
        else:
            _pvg = per_voxel_geo
            if _pvg.dim() == 2 and _pvg.shape[1] == 1:
                _pvg = _pvg.squeeze(1)
            if _pvg.dim() != 1:
                raise ValueError(
                    f"per_voxel_geo must be 1D (N,) or dense 3D/5D grid, got shape {tuple(per_voxel_geo.shape)}"
                )
            if _pvg.shape[0] != N:
                raise ValueError(
                    f"per_voxel_geo length mismatch: got {_pvg.shape[0]}, expected {N}"
                )

        _pvg = _pvg.float().to('cuda')  # (N,)

        # Differentiable corner averaging: voxel values -> shared grid-point values.
        _flat_idx = vox_key.reshape(-1)                     # (N*8,)
        _flat_vals = _pvg.repeat_interleave(8)             # (N*8,)
        _geo_grid_pts = torch.zeros(M, dtype=torch.float32, device='cuda')
        _geo_count = torch.zeros(M, dtype=torch.float32, device='cuda')
        _geo_grid_pts = _geo_grid_pts.index_add(0, _flat_idx, _flat_vals)
        _geo_count = _geo_count.index_add(0, _flat_idx, torch.ones_like(_flat_vals))
        _geo_grid_pts = (_geo_grid_pts / _geo_count.clamp(min=1.0)).unsqueeze(1)
    else:
        _geo_grid_pts = torch.full([M, 1], geo_init, dtype=torch.float32, device='cuda')

    # sh0 (DC colour): white placeholder  rgb=[1,1,1]
    _rgb = torch.ones(N, 3, dtype=torch.float32, device='cuda')
    _sh0 = rgb2shzero(_rgb)                               # (N, 1, 3)

    # higher-order SH coefficients: zero
    _shs = torch.zeros(N, (sh_degree + 1) ** 2 - 1, 3, dtype=torch.float32, device='cuda')

    state_dict = {
        'active_sh_degree': sh_degree,
        'ss': ss_value,
        'scene_center': scene_center_t.contiguous(),
        'inside_extent': inside_extent_t.contiguous(),
        'scene_extent': scene_extent_t.contiguous(),
        'octpath': octpath_t.contiguous(),
        'octlevel': octlevel_t.contiguous(),
        '_geo_grid_pts': _geo_grid_pts.contiguous(),
        '_sh0': _sh0.contiguous(),
        '_shs': _shs.contiguous(),
        'quantized': False,
    }

    # Build a minimal cfg_model so SparseVoxelModel.__init__ succeeds
    cfg_model = CfgNode()
    cfg_model.model_path = ''
    cfg_model.vox_geo_mode = 'triinterp1'
    cfg_model.density_mode = 'exp_linear_11'
    cfg_model.sh_degree = sh_degree
    cfg_model.ss = ss_value
    cfg_model.outside_level = outside_level
    cfg_model.white_background = False
    cfg_model.black_background = False

    voxel_model = SparseVoxelModel(cfg_model)
    voxel_model.load_from_state_dict(state_dict)

    print(f"[converting] SparseVoxelModel built: {N} voxels, "
          f"octlevel={target_octlevel}, scene_extent={scene_extent_val:.4f}")

    return {
        'voxel_model': voxel_model,
        'centers': centers_world,
        'sizes': sizes,
        'grid_coords': grid_coords.long(),
        'num_voxels': N,
    }


# Rotation matrix used by TRELLIS to_glb: z-up → y-up
_ZUP_TO_YUP = np.array([[1, 0, 0],
                         [0, 0, -1],
                         [0, 1, 0]], dtype=np.float64)

# Its inverse (transpose): y-up → z-up
_YUP_TO_ZUP = _ZUP_TO_YUP.T  # = [[1,0,0],[0,0,1],[0,-1,0]]


def inverse_transform_mesh(
    mesh,
    transform_info: TransformInfo,
    is_glb: bool = True,
    in_place: bool = False,
):
    """
    Apply the inverse of the forward normalization transform to bring
    a TRELLIS-generated mesh back to the original SVR world coordinates.

    Supports both trimesh.Trimesh and raw numpy vertices.

    Steps (in order):
        1. If is_glb, undo the z-up → y-up rotation (applied by to_glb export).
        2. If rotation_angle_z != 0, undo the Z-axis rotation that was applied
           to the sparse structure before mesh generation.
        3. Undo normalization:  p_world = (p_mesh + 0.5) * max_extent + cube_min
    """
    import trimesh as _trimesh

    raw_input = False
    if isinstance(mesh, tuple):
        vertices, faces = mesh
        raw_input = True
    elif isinstance(mesh, _trimesh.Trimesh):
        if not in_place:
            mesh = mesh.copy()
        vertices = mesh.vertices
        faces = mesh.faces
    else:
        raise TypeError(f"Unsupported mesh type: {type(mesh)}")

    vertices = np.asarray(vertices, dtype=np.float64)

    # --- Step 1: undo y-up rotation ---
    if is_glb:
        vertices = vertices @ _YUP_TO_ZUP

    # --- Step 2: undo Z rotation ---
    # The sparse structure was rotated by rotation_angle_z degrees (CCW positive)
    # around Z before mesh generation. To undo, rotate by -angle.
    rot_angle = transform_info.effective_rotation_z
    if rot_angle != 0.0:
        rad = np.deg2rad(-rot_angle)  # inverse rotation
        c, s = np.cos(rad), np.sin(rad)
        rot_inv = np.array([[ c, -s, 0],
                            [ s,  c, 0],
                            [ 0,  0, 1]], dtype=np.float64)
        vertices = vertices @ rot_inv.T

    # --- Step 3: undo normalization ---
    # In TRELLIS mesh space, vertex ≈ p_norm - 0.5 where p_norm ∈ [0, 1]
    # p_world = (p_mesh + 0.5) * max_extent + cube_min
    cube_min = transform_info.cube_min.cpu().numpy().astype(np.float64)
    max_extent = float(transform_info.max_extent)
    vertices = (vertices + 0.5) * max_extent + cube_min

    if raw_input:
        return vertices, faces
    else:
        mesh.vertices = vertices
        return mesh


def inverse_transform_vertices(
    vertices: Union[torch.Tensor, np.ndarray],
    transform_info: TransformInfo,
    is_glb: bool = True,
) -> Union[torch.Tensor, np.ndarray]:
    """
    Convenience wrapper: inverse-transform only vertices (no faces needed).
    """
    return_torch = isinstance(vertices, torch.Tensor)
    if return_torch:
        dev = vertices.device
        verts_np = vertices.detach().cpu().numpy()
    else:
        verts_np = np.asarray(vertices, dtype=np.float64)

    v_world, _ = inverse_transform_mesh(
        (verts_np, np.zeros((0, 3), dtype=np.int64)),
        transform_info=transform_info,
        is_glb=is_glb,
    )

    if return_torch:
        return torch.from_numpy(v_world).float().to(dev)
    return v_world


def load_and_inverse_transform_glb(
    glb_path: str,
    transform_info: TransformInfo,
) -> 'trimesh.Trimesh':
    """
    Load a GLB file exported by the pipeline and inverse-transform its
    vertices back to SVR world coordinates.

    Args:
        glb_path: path to the .glb file.
        transform_info: TransformInfo for the object.

    Returns:
        trimesh.Trimesh in world coordinates.
    """
    import trimesh
    scene = trimesh.load(glb_path)
    if isinstance(scene, trimesh.Scene):
        mesh = scene.dump(concatenate=True)
    else:
        mesh = scene
    # Capture pre-transform range before in-place modification
    _v = mesh.vertices
    _pre = (float(_v[:,0].min()), float(_v[:,0].max()),
            float(_v[:,1].min()), float(_v[:,1].max()),
            float(_v[:,2].min()), float(_v[:,2].max()))
    result = inverse_transform_mesh(mesh, transform_info, is_glb=True, in_place=True)
    _w = result.vertices
    print(
        f"[converting] load_and_inverse_transform_glb DEBUG:\n"
        f"  GLB vertices (y-up): x=[{_pre[0]:.4f},{_pre[1]:.4f}], "
        f"y=[{_pre[2]:.4f},{_pre[3]:.4f}], z=[{_pre[4]:.4f},{_pre[5]:.4f}]\n"
        f"  world vertices:      x=[{_w[:,0].min():.4f},{_w[:,0].max():.4f}], "
        f"y=[{_w[:,1].min():.4f},{_w[:,1].max():.4f}], "
        f"z=[{_w[:,2].min():.4f},{_w[:,2].max():.4f}]\n"
        f"  transform_info: {transform_info}"
    )
    return result


def validate_ss_geometry(
    sparse_structure: torch.Tensor,
    transform_info: 'TransformInfo',
    output_dir: str,
    tag: str,
    threshold: float = 0.5,
    device: torch.device = None,
) -> dict:
    """DEBUG: convert ``sparse_structure`` -> SVR voxels via
    :func:`sparse_structure_to_svr_voxels` and dump the resulting
    grid-cell centers as a point cloud (.ply) plus an .npz for inspection.

    Use this to verify, end-to-end, that an SS tensor at any pipeline stage
    (initial / loaded / post-SDS) maps to the expected world-space region.

    Files written under ``output_dir`` (created if needed):
        validate_<tag>.ply   – point cloud of voxel centers (world coords)
        validate_<tag>.npz   – centers + grid_coords + transform metadata

    Returns the raw output dict from sparse_structure_to_svr_voxels.
    """
    os.makedirs(output_dir, exist_ok=True)
    out = sparse_structure_to_svr_voxels(
        sparse_structure, transform_info,
        threshold=threshold, device=device,
    )
    centers = out['centers'].detach().cpu().numpy()
    grid_coords = out['grid_coords'].detach().cpu().numpy()
    n = centers.shape[0]

    ply_path = os.path.join(output_dir, f'validate_{tag}.ply')
    npz_path = os.path.join(output_dir, f'validate_{tag}.npz')

    if n > 0:
        pc = trimesh.PointCloud(centers)
        pc.export(ply_path)
    np.savez(
        npz_path,
        centers=centers,
        grid_coords=grid_coords,
        cube_min=transform_info.cube_min.cpu().numpy(),
        max_extent=float(transform_info.max_extent),
        resolution=int(transform_info.resolution),
        rotation_angle_z=float(transform_info.effective_rotation_z),
        threshold=float(threshold),
    )

    print(f"[validate_ss_geometry] tag='{tag}' threshold={threshold} N={n}")
    if n > 0:
        cmin = centers.min(axis=0)
        cmax = centers.max(axis=0)
        ccen = centers.mean(axis=0)
        print(
            f"  centers world: x=[{cmin[0]:.4f},{cmax[0]:.4f}], "
            f"y=[{cmin[1]:.4f},{cmax[1]:.4f}], "
            f"z=[{cmin[2]:.4f},{cmax[2]:.4f}], "
            f"centroid=({ccen[0]:.4f},{ccen[1]:.4f},{ccen[2]:.4f})"
        )
        gmin = grid_coords.min(axis=0)
        gmax = grid_coords.max(axis=0)
        print(
            f"  grid_coords:   x=[{gmin[0]},{gmax[0]}], "
            f"y=[{gmin[1]},{gmax[1]}], z=[{gmin[2]},{gmax[2]}]"
        )
    print(f"  saved -> {ply_path}\n           {npz_path}")
    return out


