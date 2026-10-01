"""
Error Code by my claude.
Voxel to Surface Mesh using Marching Cubes

As described in GeoSVR:
  "To extract a mesh, we implement Marching Cubes to extract the triangles
   of an isosurface over density from the sparse voxels."

The density field is defined by:
  - _geo_grid_pts: raw density values at octree grid point corners  [M, 1]
  - grid_pts_key:  integer coordinates of these grid points         [M, 3]
  - density_mode:  activation function (e.g., exp_linear_11)

Two extraction methods are provided:
  1. grid_points  – scatter grid-point densities into a dense volume, then MC.
     Fast and simple; works well when most voxels are at similar octree levels.
  2. per_voxel    – iterate over each leaf voxel (as an MC cell) and run the
     standard Marching-Cubes lookup.  Correctly handles multi-resolution
     octrees without resampling artefacts.

Usage:
    # From a full model checkpoint
    python voxel_to_mesh_surface.py --checkpoint outputs/Replica/scan1/checkpoints/iter030000_model.pt

    # From an object voxel file (exported by segm_3d.py)
    python voxel_to_mesh_surface.py --checkpoint semantic_result/object_voxels/object_001_voxels.pt

    # Control resolution / threshold
    python voxel_to_mesh_surface.py --checkpoint model.pt --resolution 512 --threshold 0.5

    # Use per-voxel MC (handles multi-res better)
    python voxel_to_mesh_surface.py --checkpoint model.pt --method per_voxel --threshold 1.0
"""

import os
import sys
import copy
import argparse
import numpy as np
import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Project setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from yacs.config import CfgNode
from svraster_cuda.meta import MAX_NUM_LEVELS

from src.sparse_voxel_model import SparseVoxelModel
from src.sparse_voxel_gears.io import SVInOut
from src.utils import octree_utils
from src.utils.activation_utils import (
    shzero2rgb,
    exp_linear_11,
    exp_linear_10,
    exp_linear_20,
    softplus,
)

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False

try:
    from skimage.measure import marching_cubes as skimage_marching_cubes
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

try:
    import mcubes
    MCUBES_AVAILABLE = True
except ImportError:
    MCUBES_AVAILABLE = False


# ========================================================================= #
#  Utility helpers                                                           #
# ========================================================================= #

def create_cfg_model(model_path=""):
    """Create a minimal cfg_model CfgNode for SparseVoxelModel."""
    cfg_model = CfgNode()
    cfg_model.model_path = model_path
    cfg_model.vox_geo_mode = "triinterp1"
    cfg_model.density_mode = "exp_linear_11"
    cfg_model.sh_degree = 3
    cfg_model.ss = 1.5
    cfg_model.outside_level = 5
    cfg_model.white_background = False
    cfg_model.black_background = False
    return cfg_model


def load_model(checkpoint_path):
    """Load a SparseVoxelModel from a full model checkpoint (.pt)."""
    checkpoint_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    if os.path.basename(checkpoint_dir) == "checkpoints":
        model_path = os.path.dirname(checkpoint_dir)
    else:
        model_path = checkpoint_dir

    cfg_model = create_cfg_model(model_path)
    model = SparseVoxelModel(cfg_model)
    model.load(checkpoint_path)
    return model


def detect_and_load(checkpoint_path):
    """
    Detect checkpoint type and load accordingly.
    Returns (voxel_model | None, state_dict | None, checkpoint_type).
    """
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "active_sh_degree" in state_dict:
        # Full model checkpoint – load into SparseVoxelModel
        model = load_model(checkpoint_path)
        return model, None, "model"
    else:
        # Object voxel file – we will build a model from the state_dict
        model = _build_model_from_state_dict(state_dict)
        return model, state_dict, "object"


def _build_model_from_state_dict(state_dict):
    """
    Build a SparseVoxelModel and populate it from a state_dict produced by
    SVInOut.create_state_dict_from_subset / SVInOut.save_state_dict.
    """
    cfg_model = create_cfg_model()
    model = SparseVoxelModel(cfg_model)
    model.load_from_state_dict(state_dict)
    return model


def apply_density_activation(raw, density_mode):
    """Apply the configured density activation to raw geo values."""
    if density_mode == "exp_linear_11":
        return exp_linear_11(raw)
    elif density_mode == "exp_linear_10":
        return exp_linear_10(raw)
    elif density_mode == "exp_linear_20":
        return exp_linear_20(raw)
    elif density_mode == "softplus":
        return softplus(raw)
    else:
        print(f"Warning: unknown density_mode '{density_mode}', using raw values")
        return raw


# ========================================================================= #
#  Vertex colouring                                                          #
# ========================================================================= #

def color_vertices_nearest(query_pts, ref_pts, ref_colors,
                           batch_size=4096, ref_batch_size=100000):
    """
    Assign colours to query points by nearest-neighbour lookup from ref_pts.

    Both query and reference points are batched to avoid OOM when either
    side is large (e.g. millions of voxel centres).

    Args:
        query_pts:      [Q, 3]  float  CUDA
        ref_pts:        [N, 3]  float  CUDA
        ref_colors:     [N, 3]  float  CUDA   (RGB in [0,1])
        batch_size:     number of query points per batch
        ref_batch_size: number of reference points per inner batch

    Returns:
        colors: [Q, 3] float CUDA
    """
    Q = len(query_pts)
    N = len(ref_pts)
    colors = torch.empty(Q, 3, device="cuda")

    for i in tqdm(range(0, Q, batch_size), desc="  Coloring vertices"):
        end_q = min(i + batch_size, Q)
        q_batch = query_pts[i:end_q]                          # [B, 3]
        B = end_q - i

        min_dists = torch.full((B,), float('inf'), device="cuda")
        min_indices = torch.zeros(B, dtype=torch.long, device="cuda")

        for j in range(0, N, ref_batch_size):
            end_r = min(j + ref_batch_size, N)
            # distance matrix [B, R] – bounded by batch_size * ref_batch_size
            dists = torch.cdist(q_batch.unsqueeze(0),
                                ref_pts[j:end_r].unsqueeze(0)).squeeze(0)
            batch_min_dists, batch_min_idx = dists.min(dim=1)  # [B]
            closer = batch_min_dists < min_dists
            min_dists[closer] = batch_min_dists[closer]
            min_indices[closer] = batch_min_idx[closer] + j

        colors[i:end_q] = ref_colors[min_indices]
    return colors


# ========================================================================= #
#  Method 1 – Grid-point scatter into dense volume + Marching Cubes          #
# ========================================================================= #

@torch.no_grad()
def extract_mesh_grid_points(
    voxel_model,
    resolution: int = 512,
    density_threshold: float = 0.5,
    max_memory_gb: float = 4.0,
    use_inside_region: bool = True,
    color_mesh: bool = True,
):
    """
    Build a dense 3D volume from the sparse grid-point density values,
    then run Marching Cubes.

    The grid_pts_key integer coordinates are mapped into a dense numpy array
    (down-sampled if the native range exceeds *resolution* or *max_memory_gb*).
    """
    assert SKIMAGE_AVAILABLE or MCUBES_AVAILABLE, \
        "Install scikit-image (`pip install scikit-image`) or PyMCubes (`pip install PyMCubes`)."

    # ------ 1. Gather grid-point data ------
    grid_pts_key = voxel_model.grid_pts_key                   # [M, 3] int64
    raw_geo = voxel_model._geo_grid_pts.data.squeeze(-1)      # [M]
    density = apply_density_activation(raw_geo, voxel_model.density_mode)
    grid_pts_xyz = voxel_model.grid_pts_xyz                   # [M, 3]

    scene_center = voxel_model.scene_center
    scene_extent = voxel_model.scene_extent

    # ------ 2. Optionally restrict to inside region ------
    if use_inside_region:
        inside_min = voxel_model.inside_min
        inside_max = voxel_model.inside_max
        inside_mask = ((grid_pts_xyz >= inside_min) &
                       (grid_pts_xyz <= inside_max)).all(dim=1)
        grid_pts_key = grid_pts_key[inside_mask]
        density = density[inside_mask]
        grid_pts_xyz = grid_pts_xyz[inside_mask]

    M = len(grid_pts_key)
    print(f"  Grid points to process: {M}")

    # ------ 3. Determine dense volume dimensions ------
    ijk_min = grid_pts_key.min(dim=0).values     # [3]
    ijk_max = grid_pts_key.max(dim=0).values     # [3]
    ijk_range = (ijk_max - ijk_min + 1).float()  # [3]

    scale = max(1, int(np.ceil(ijk_range.max().item() / resolution)))
    grid_size = ((ijk_range / scale).long() + 1).cpu().numpy()

    # Ensure memory limit
    while (int(grid_size[0]) * int(grid_size[1]) * int(grid_size[2]) * 4
           / (1024 ** 3) > max_memory_gb):
        scale *= 2
        grid_size = ((ijk_range / scale).long() + 1).cpu().numpy()

    gx, gy, gz = int(grid_size[0]), int(grid_size[1]), int(grid_size[2])
    total_cells = gx * gy * gz
    print(f"  Native ijk range : ({ijk_range[0]:.0f}, {ijk_range[1]:.0f}, {ijk_range[2]:.0f})")
    print(f"  Down-sample scale: {scale}")
    print(f"  Dense volume size: ({gx}, {gy}, {gz})  ({total_cells * 4 / 1024**3:.2f} GB)")

    # ------ 4. Scatter density into dense volume ------
    scaled_idx = ((grid_pts_key - ijk_min).float() / scale).long()
    scaled_idx[:, 0].clamp_(0, gx - 1)
    scaled_idx[:, 1].clamp_(0, gy - 1)
    scaled_idx[:, 2].clamp_(0, gz - 1)

    flat_idx = scaled_idx[:, 0] * (gy * gz) + scaled_idx[:, 1] * gz + scaled_idx[:, 2]

    volume_flat = torch.zeros(total_cells, dtype=torch.float32, device="cuda")
    count_flat  = torch.zeros(total_cells, dtype=torch.float32, device="cuda")
    volume_flat.scatter_add_(0, flat_idx, density.float())
    count_flat.scatter_add_(0, flat_idx, torch.ones_like(density, dtype=torch.float32))
    nz = count_flat > 0
    volume_flat[nz] /= count_flat[nz]

    volume_np = volume_flat.reshape(gx, gy, gz).cpu().numpy()
    occupied = (volume_np > 0).sum()
    print(f"  Density range : [{volume_np[volume_np > 0].min():.4f}, {volume_np.max():.4f}]"
          if occupied > 0 else "  (all zeros)")
    print(f"  Occupied cells: {occupied} / {total_cells} "
          f"({100.0 * occupied / total_cells:.1f}%)")

    # ------ 5. Marching Cubes ------
    print(f"  Running Marching Cubes  (threshold = {density_threshold}) ...")
    verts, faces = _run_marching_cubes(volume_np, density_threshold)
    if verts is None:
        return None
    print(f"  Extracted mesh: {len(verts)} vertices, {len(faces)} faces")

    # ------ 6. Map vertex coords back to world space ------
    finest_vs = octree_utils.level_2_vox_size(
        scene_extent,
        torch.tensor(MAX_NUM_LEVELS, dtype=torch.int64, device="cuda"),
    ).item()
    scene_min_np = (scene_center - 0.5 * scene_extent).cpu().numpy()
    ijk_min_np = ijk_min.cpu().numpy().astype(np.float64)

    world_verts = scene_min_np + (verts * scale + ijk_min_np) * finest_vs

    # ------ 7. Vertex colouring ------
    vertex_colors_np = None
    if color_mesh:
        vertex_colors_np = _color_mesh_vertices(
            world_verts, voxel_model)

    # ------ 8. Build trimesh ------
    return _build_trimesh(world_verts, faces, vertex_colors_np)


# ========================================================================= #
#  Method 2 – Per-voxel Marching Cubes                                       #
# ========================================================================= #

# Classic Marching-Cubes edge / triangle tables (Lorensen & Cline 1987).
# Each voxel cell has 8 corners; corner ordering follows the standard:
#   0:(0,0,0) 1:(1,0,0) 2:(1,1,0) 3:(0,1,0)
#   4:(0,0,1) 5:(1,0,0) 6:(1,1,1) 7:(0,1,1)
# fmt: off
_EDGE_VERTICES = np.array([
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 4], [1, 5], [2, 6], [3, 7],
], dtype=np.int32)

# Cube corner offsets (x, y, z) matching the standard MC corner layout
_CORNER_OFFSETS = np.array([
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
], dtype=np.float64)
# fmt: on

# The full 256-entry MC edge table and triangle table are large constants.
# We generate them lazily using a known algorithm or rely on scikit-image /
# PyMCubes to do the heavy lifting.  In this method we process one voxel at
# a time through skimage/mcubes by creating a tiny 2×2×2 volume.


@torch.no_grad()
def extract_mesh_per_voxel(
    voxel_model,
    density_threshold: float = 1.0,
    use_inside_region: bool = True,
    color_mesh: bool = True,
    batch_size: int = 50000,
):
    """
    Run Marching Cubes *per leaf voxel*.  Each voxel is treated as one MC
    cell whose 8 corners carry the activated density from ``_geo_grid_pts``.

    This correctly handles multi-resolution octree grids without resampling.
    """
    assert SKIMAGE_AVAILABLE or MCUBES_AVAILABLE, \
        "Install scikit-image or PyMCubes."

    # ------ gather per-voxel corner data ------
    vox_key = voxel_model.vox_key         # [N, 8]
    grid_pts_xyz = voxel_model.grid_pts_xyz  # [M, 3]
    raw_geo = voxel_model._geo_grid_pts.data.squeeze(-1)  # [M]
    density = apply_density_activation(raw_geo, voxel_model.density_mode)

    N = voxel_model.num_voxels
    print(f"  Total voxels: {N}")

    # IMPORTANT: the 8 corners in vox_key follow the octree corner order
    # (binary encoding of octants) which matches:
    #   idx  xyz
    #    0   000
    #    1   001  (z+1)
    #    2   010  (y+1)
    #    3   011  (y+1 z+1)
    #    4   100  (x+1)
    #    5   101  (x+1 z+1)
    #    6   110  (x+1 y+1)
    #    7   111  (x+1 y+1 z+1)
    #
    # The standard MC corner order is:
    #    0  (0,0,0)   1  (1,0,0)   2  (1,1,0)   3  (0,1,0)
    #    4  (0,0,1)   5  (1,0,0)   6  (1,1,1)   7  (0,1,1)
    #
    # We map octree order -> MC order:
    octree_to_mc = [0, 4, 6, 2, 1, 5, 7, 3]  # octree_idx -> mc_idx mapping
    mc_to_octree = [0, 4, 3, 7, 1, 5, 2, 6]  # mc_idx -> octree_idx mapping
    # So: corner_density_mc[mc_idx] = corner_density_octree[mc_to_octree[mc_idx]]

    # pre-fetch corner densities and positions (all on CUDA, process in batches)
    corner_density = density[vox_key]     # [N, 8]  (octree order)
    corner_xyz     = grid_pts_xyz[vox_key]  # [N, 8, 3]

    # Optionally restrict to inside region
    if use_inside_region:
        inside_min = voxel_model.inside_min  # [3]
        inside_max = voxel_model.inside_max  # [3]
        centers = voxel_model.vox_center     # [N, 3]
        inside_mask = ((centers >= inside_min) & (centers <= inside_max)).all(dim=1)
        keep = torch.where(inside_mask)[0]
        corner_density = corner_density[keep]
        corner_xyz = corner_xyz[keep]
        N = len(keep)
        print(f"  Voxels in inside region: {N}")

    # Move to CPU for MC processing
    corner_density_cpu = corner_density.cpu().numpy()   # [N, 8] octree order
    corner_xyz_cpu = corner_xyz.cpu().numpy()           # [N, 8, 3]

    all_verts = []
    all_faces = []
    vert_offset = 0

    print(f"  Running per-voxel Marching Cubes (threshold = {density_threshold}) ...")
    for start in tqdm(range(0, N, batch_size), desc="  MC batches"):
        end = min(start + batch_size, N)
        for vi in range(start, end):
            d_octree = corner_density_cpu[vi]           # [8] in octree order
            xyz_octree = corner_xyz_cpu[vi]             # [8, 3]

            # Reorder to MC order
            d_mc = d_octree[mc_to_octree]               # [8] in MC order
            xyz_mc = xyz_octree[mc_to_octree]           # [8, 3]

            # Build tiny 2×2×2 volume
            vol = np.zeros((2, 2, 2), dtype=np.float64)
            for ci, (ox, oy, oz) in enumerate(_CORNER_OFFSETS):
                vol[int(ox), int(oy), int(oz)] = d_mc[ci]

            # MC on the tiny volume
            v, f = _run_marching_cubes(vol, density_threshold)
            if v is None or len(v) == 0:
                continue

            # Map local [0,1]^3 vertices to world coordinates
            # Local coords are in [0, 1] range – we use the 8 MC-ordered corners
            # to trilinearly interpolate world positions.
            xyz_min = xyz_mc[0]           # corner (0,0,0) in MC order
            xyz_max = xyz_mc[6]           # corner (1,1,1) in MC order
            span = xyz_max - xyz_min
            world_v = xyz_min + v * span  # broadcast [V, 3]

            all_verts.append(world_v)
            all_faces.append(f + vert_offset)
            vert_offset += len(world_v)

    if vert_offset == 0:
        print("  No surface found at this threshold.")
        return None

    world_verts = np.concatenate(all_verts, axis=0).astype(np.float64)
    faces = np.concatenate(all_faces, axis=0)
    print(f"  Raw mesh: {len(world_verts)} vertices, {len(faces)} faces")

    # Merge duplicate vertices at shared voxel boundaries
    if TRIMESH_AVAILABLE:
        tmp = trimesh.Trimesh(vertices=world_verts, faces=faces, process=False)
        tmp.merge_vertices(merge_tex=True, merge_norm=True)
        world_verts = np.array(tmp.vertices)
        faces = np.array(tmp.faces)
        print(f"  After vertex merging: {len(world_verts)} vertices, {len(faces)} faces")

    # Colour
    vertex_colors_np = None
    if color_mesh:
        vertex_colors_np = _color_mesh_vertices(world_verts, voxel_model)

    return _build_trimesh(world_verts, faces, vertex_colors_np)


# ========================================================================= #
#  Shared helpers                                                            #
# ========================================================================= #

def _run_marching_cubes(volume_np, threshold):
    """
    Run Marching Cubes on a numpy volume.  Tries PyMCubes first (produces
    watertight meshes), then falls back to scikit-image.
    Returns (verts, faces) or (None, None) on failure.
    """
    try:
        if MCUBES_AVAILABLE:
            verts, faces = mcubes.marching_cubes(volume_np, threshold)
        elif SKIMAGE_AVAILABLE:
            verts, faces, _, _ = skimage_marching_cubes(volume_np, level=threshold)
        else:
            raise RuntimeError("No MC backend available.")
    except (ValueError, RuntimeError) as exc:
        print(f"  Marching Cubes failed: {exc}")
        return None, None

    if len(verts) == 0:
        return None, None
    return verts, faces


def _color_mesh_vertices(world_verts_np, voxel_model):
    """Assign per-vertex RGB colour via nearest-voxel-center SH0 lookup."""
    query = torch.from_numpy(world_verts_np).float().cuda()
    centers = voxel_model.vox_center                    # [N, 3]
    sh0 = voxel_model._sh0.data.squeeze(1)              # [N, 3]
    ref_colors = shzero2rgb(sh0).clamp(0, 1)            # [N, 3]
    vc = color_vertices_nearest(query, centers, ref_colors)
    return vc.cpu().numpy()


def _build_trimesh(verts_np, faces_np, vertex_colors_np=None):
    """Wrap arrays into a trimesh.Trimesh."""
    if not TRIMESH_AVAILABLE:
        print("  Warning: trimesh not installed – returning raw arrays.")
        return verts_np, faces_np, vertex_colors_np

    kwargs = dict(vertices=verts_np.astype(np.float32),
                  faces=faces_np.astype(np.int32),
                  process=False)

    if vertex_colors_np is not None:
        rgba = np.zeros((len(vertex_colors_np), 4), dtype=np.uint8)
        rgba[:, :3] = (vertex_colors_np * 255).clip(0, 255).astype(np.uint8)
        rgba[:, 3] = 255
        kwargs["vertex_colors"] = rgba

    return trimesh.Trimesh(**kwargs)


# ========================================================================= #
#  Post-processing                                                           #
# ========================================================================= #

def clean_mesh(mesh, min_faces=50):
    """Remove small connected components from a trimesh.Trimesh."""
    if not TRIMESH_AVAILABLE:
        return mesh
    cc = mesh.split(only_watertight=False)
    kept = [c for c in cc if len(c.faces) >= min_faces]
    if not kept:
        return mesh
    return trimesh.util.concatenate(kept)


# ========================================================================= #
#  Main entry point                                                          #
# ========================================================================= #

def voxel_to_mesh_surface(
    checkpoint: str,
    output: str | None = None,
    method: str = "grid_points",
    resolution: int = 512,
    density_threshold: float = 0.5,
    max_memory_gb: float = 4.0,
    use_inside: bool = True,
    color: bool = True,
    clean: bool = True,
    min_component_faces: int = 50,
    format: str = "ply",
):
    """
    End-to-end: load checkpoint ➜ extract surface mesh ➜ export.

    Args:
        checkpoint:  path to a model checkpoint or object voxel .pt file
        output:      output file path (auto-generated if None)
        method:      "grid_points" or "per_voxel"
        resolution:  max dense-grid resolution per axis (grid_points method)
        density_threshold:  isosurface threshold
        max_memory_gb:      memory budget for dense volume
        use_inside:  restrict extraction to the inside region
        color:       colour the mesh vertices from SH0
        clean:       remove small connected components
        min_component_faces:  minimum #faces to keep a component
        format:      output format (ply, glb, obj, stl)
    """
    if not os.path.isfile(checkpoint):
        print(f"Error: file not found – {checkpoint}")
        return False

    # --- Resolve output path ---
    base = os.path.splitext(os.path.basename(checkpoint))[0]
    if output is None:
        out_dir = os.path.dirname(os.path.abspath(checkpoint))
        output = os.path.join(out_dir, f"{base}_surface.{format}")
    elif os.path.isdir(output) or not os.path.splitext(output)[1]:
        # output is a directory (or has no extension) – append a filename
        output = os.path.join(output, f"{base}_surface.{format}")
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)

    # --- Load ---
    print(f"Loading: {checkpoint}")
    model, state_dict, ckpt_type = detect_and_load(checkpoint)
    print(f"  Checkpoint type : {ckpt_type}")
    print(f"  Voxels          : {model.num_voxels}")
    print(f"  Grid points     : {model.num_grid_pts}")

    # --- Extract mesh ---
    print(f"\nExtracting surface mesh (method={method}) ...")
    if method == "grid_points":
        mesh = extract_mesh_grid_points(
            model,
            resolution=resolution,
            density_threshold=density_threshold,
            max_memory_gb=max_memory_gb,
            use_inside_region=use_inside,
            color_mesh=color,
        )
    elif method == "per_voxel":
        mesh = extract_mesh_per_voxel(
            model,
            density_threshold=density_threshold,
            use_inside_region=use_inside,
            color_mesh=color,
        )
    else:
        print(f"Error: unknown method '{method}'. Use 'grid_points' or 'per_voxel'.")
        return False

    # --- Post-process ---
    if clean and TRIMESH_AVAILABLE and isinstance(mesh, trimesh.Trimesh):
        before = len(mesh.faces)
        if before > 500000:
            print(f"\n  Skipping clean_mesh: mesh too large ({before} faces). "
                  f"Use --no_clean or reduce --resolution to speed up.")
        else:
            print(f"\nCleaning mesh ({before} faces) ...")
            mesh = clean_mesh(mesh, min_faces=min_component_faces)
            after = len(mesh.faces)
            if before != after:
                print(f"  Cleaned: {before} → {after} faces")

    # --- Export ---
    print(f"\nExporting to: {output}")
    if TRIMESH_AVAILABLE and isinstance(mesh, trimesh.Trimesh):
        mesh.export(output)
    else:
        # Fallback: save raw arrays
        verts, faces, colors = mesh
        np.savez(output, vertices=verts, faces=faces, colors=colors)

    size_mb = os.path.getsize(output) / (1024 * 1024)
    print(f"  File size: {size_mb:.2f} MB")
    print("Done.")
    return True


# ========================================================================= #
#  CLI                                                                       #
# ========================================================================= #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract a surface mesh from GeoSVR sparse voxels via Marching Cubes.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint or object voxel .pt file")
    parser.add_argument("--output", type=str, default=None,
                        help="Output mesh path (default: <checkpoint>_surface.<format>)")
    parser.add_argument("--method", type=str, default="grid_points",
                        choices=["grid_points", "per_voxel"],
                        help="Extraction method (default: grid_points)")
    parser.add_argument("--resolution", type=int, default=512,
                        help="Max resolution per axis for dense volume (grid_points method)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Density isosurface threshold")
    parser.add_argument("--max_memory_gb", type=float, default=4.0,
                        help="Max memory for dense volume in GB")
    parser.add_argument("--no_inside", action="store_true",
                        help="Extract from entire scene (not just inside region)")
    parser.add_argument("--no_color", action="store_true",
                        help="Skip vertex colouring")
    parser.add_argument("--no_clean", action="store_true",
                        help="Skip removal of small connected components")
    parser.add_argument("--min_faces", type=int, default=50,
                        help="Minimum faces to keep a component (default: 50)")
    parser.add_argument("--format", type=str, default="ply",
                        choices=["ply", "glb", "obj", "stl"],
                        help="Output format (default: ply)")
    args = parser.parse_args()

    voxel_to_mesh_surface(
        checkpoint=args.checkpoint,
        output=args.output,
        method=args.method,
        resolution=args.resolution,
        density_threshold=args.threshold,
        max_memory_gb=args.max_memory_gb,
        use_inside=not args.no_inside,
        color=not args.no_color,
        clean=not args.no_clean,
        min_component_faces=args.min_faces,
        format=args.format,
    )
