"""
Extract mesh from object voxel files using the Direct Marching Cubes method.

This wraps the `direct_mc` approach from extract_mesh.py so it can run on
standalone object voxel checkpoint files (produced by segm_3d.py) without
needing training data, config.yaml, or DataPack.

The iso-surface threshold is computed automatically from alpha=0.5 using
the same logic as extract_mesh.py:
    iso = density_mode_inverse(alpha2density(0.5, vox_size))

Usage:
    # Basic usage (auto iso from alpha=0.5)
    python extract_mesh_object.py --checkpoint object_voxels/object_001_voxels.pt

    # Custom output path
    python extract_mesh_object.py --checkpoint object_001_voxels.pt --output meshes/obj1.ply

    # Custom alpha threshold for iso-surface
    python extract_mesh_object.py --checkpoint object_001_voxels.pt --alpha 0.3

    # Batch: all object voxels in a directory
    python extract_mesh_object.py --checkpoint_dir object_voxels/ --output_dir meshes/

    # Skip vertex colouring (faster)
    python extract_mesh_object.py --checkpoint object_001_voxels.pt --no_color
"""

import os
import sys
import time
import argparse
import glob
import numpy as np
import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Project setup – ensure GeoSVR source is on sys.path
# ---------------------------------------------------------------------------
GEOSVR_ROOT = os.path.abspath(os.path.dirname(__file__))
if GEOSVR_ROOT not in sys.path:
    sys.path.insert(0, GEOSVR_ROOT)

import trimesh
import svraster_cuda
from svraster_cuda.meta import MAX_NUM_LEVELS
from svraster_cuda.marching_cubes import torch_marching_cubes_grid

from yacs.config import CfgNode
from src.sparse_voxel_model import SparseVoxelModel
from src.utils import octree_utils
from src.utils import activation_utils
from src.utils.activation_utils import shzero2rgb


# ========================================================================= #
#  Model loading (reused from voxel_to_mesh_surface.py)                      #
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
    Returns (voxel_model, checkpoint_type).
    """
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "active_sh_degree" in state_dict:
        model = load_model(checkpoint_path)
        return model, "model"
    else:
        cfg_model = create_cfg_model()
        model = SparseVoxelModel(cfg_model)
        model.load_from_state_dict(state_dict)
        return model, "object"


# ========================================================================= #
#  Direct Marching Cubes (adapted from extract_mesh.py)                      #
# ========================================================================= #

@torch.no_grad()
def direct_mc(
    voxel_model,
    iso_alpha: float = 0.5,
    bbox_scale: float = 1.0,
    crop_bbox=None,
    final_lv: int | None = None,
):
    """
    Direct Marching Cubes on the density field stored in the voxel model.

    Automatically computes the iso-surface threshold from `iso_alpha`
    (default 0.5) using the same formula as extract_mesh.py::direct_mc:

        vox_size  = level_2_vox_size(scene_extent, outside_level + final_lv)
        density   = alpha2density(iso_alpha, vox_size)
        iso       = density_mode_inverse(density)

    Args:
        voxel_model:  loaded SparseVoxelModel
        iso_alpha:    alpha value for the iso-surface (0–1, default 0.5)
        bbox_scale:   scale factor for the inside bounding box
        crop_bbox:    optional [[xmin,ymin,zmin],[xmax,ymax,zmax]] crop box
        final_lv:     octree level for vox_size computation (auto-detected if None)

    Returns:
        trimesh.Trimesh
    """
    # --- Determine bounding box ---
    if crop_bbox is None:
        inside_min = voxel_model.scene_center - 0.5 * voxel_model.inside_extent * bbox_scale
        inside_max = voxel_model.scene_center + 0.5 * voxel_model.inside_extent * bbox_scale
    else:
        inside_min = torch.tensor(crop_bbox[0], dtype=torch.float32, device="cuda")
        inside_max = torch.tensor(crop_bbox[1], dtype=torch.float32, device="cuda")

    # --- Filter voxels to inside region ---
    inside_mask = (
        (inside_min <= voxel_model.grid_pts_xyz) &
        (voxel_model.grid_pts_xyz <= inside_max)
    ).all(-1)
    inside_mask = inside_mask[voxel_model.vox_key].any(-1)
    inside_idx = torch.where(inside_mask)[0]
    print(f"  Voxels in bbox: {len(inside_idx)} / {voxel_model.num_voxels}")

    # --- Auto-detect final_lv from the most common octree level ---
    if final_lv is None:
        level_counts = voxel_model.octlevel.flatten().bincount()
        final_lv = int(level_counts.argmax().item()) - voxel_model.outside_level
        print(f"  Auto-detected final_lv: {final_lv} "
              f"(octlevel={voxel_model.outside_level + final_lv})")

    # --- Compute iso threshold ---
    vox_level = torch.tensor(
        [voxel_model.outside_level + final_lv], device="cuda")
    vox_size = octree_utils.level_2_vox_size(
        voxel_model.scene_extent, vox_level).item()

    iso_alpha_t = torch.tensor(iso_alpha, device="cuda")
    iso_density = activation_utils.alpha2density(iso_alpha_t, vox_size)
    iso = getattr(
        activation_utils,
        f"{voxel_model.density_mode}_inverse"
    )(iso_density)
    sign = -1

    print(f"  density_mode : {voxel_model.density_mode}")
    print(f"  vox_size     : {vox_size:.6f}")
    print(f"  iso_alpha    : {iso_alpha}")
    print(f"  iso_density  : {iso_density.item():.4f}")
    print(f"  iso (raw)    : {iso.item():.4f}")

    # --- Run Marching Cubes ---
    print("  Running Marching Cubes ...")
    t0 = time.time()
    verts, faces = torch_marching_cubes_grid(
        grid_pts_val=sign * voxel_model._geo_grid_pts,
        grid_pts_xyz=voxel_model.grid_pts_xyz,
        vox_key=voxel_model.vox_key[inside_idx],
        iso=sign * iso,
    )
    dt = time.time() - t0
    print(f"  MC done in {dt:.2f}s: {len(verts)} verts, {len(faces)} faces")

    mesh = trimesh.Trimesh(verts.cpu().numpy(), faces.cpu().numpy())
    return mesh


# ========================================================================= #
#  Vertex colouring                                                          #
# ========================================================================= #

@torch.no_grad()
def color_mesh_vertices(mesh, voxel_model,
                        batch_size=4096, ref_batch_size=100000):
    """
    Assign per-vertex RGB colour from the nearest voxel center's SH0.
    Operates in batches on GPU to avoid OOM.
    """
    query = torch.from_numpy(
        np.array(mesh.vertices)).float().cuda()
    centers = voxel_model.vox_center                   # [N, 3]
    sh0 = voxel_model._sh0.data.squeeze(1)             # [N, 3]
    ref_colors = shzero2rgb(sh0).clamp(0, 1)           # [N, 3]

    Q = len(query)
    N = len(centers)
    colors = torch.empty(Q, 3, device="cuda")

    for i in tqdm(range(0, Q, batch_size), desc="  Coloring vertices"):
        end_q = min(i + batch_size, Q)
        q_batch = query[i:end_q]
        B = end_q - i

        min_dists = torch.full((B,), float('inf'), device="cuda")
        min_indices = torch.zeros(B, dtype=torch.long, device="cuda")

        for j in range(0, N, ref_batch_size):
            end_r = min(j + ref_batch_size, N)
            dists = torch.cdist(
                q_batch.unsqueeze(0),
                centers[j:end_r].unsqueeze(0),
            ).squeeze(0)
            batch_min_dists, batch_min_idx = dists.min(dim=1)
            closer = batch_min_dists < min_dists
            min_dists[closer] = batch_min_dists[closer]
            min_indices[closer] = batch_min_idx[closer] + j

        colors[i:end_q] = ref_colors[min_indices]

    vc_np = colors.cpu().numpy()
    rgba = np.zeros((Q, 4), dtype=np.uint8)
    rgba[:, :3] = (vc_np * 255).clip(0, 255).astype(np.uint8)
    rgba[:, 3] = 255
    mesh.visual.vertex_colors = rgba
    return mesh


# ========================================================================= #
#  Main entry point                                                          #
# ========================================================================= #

def extract_mesh_object(
    checkpoint: str,
    output: str | None = None,
    iso_alpha: float = 0.5,
    bbox_scale: float = 1.0,
    color: bool = True,
    format: str = "ply",
):
    """
    End-to-end: load checkpoint -> direct MC -> optional colouring -> export.
    """
    if not os.path.isfile(checkpoint):
        print(f"Error: file not found – {checkpoint}")
        return False

    # --- Resolve output path ---
    base = os.path.splitext(os.path.basename(checkpoint))[0]
    if output is None:
        out_dir = os.path.dirname(os.path.abspath(checkpoint))
        output = os.path.join(out_dir, f"{base}_mesh.{format}")
    elif os.path.isdir(output) or not os.path.splitext(output)[1]:
        output = os.path.join(output, f"{base}_mesh.{format}")
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)

    # --- Load ---
    print(f"Loading: {checkpoint}")
    model, ckpt_type = detect_and_load(checkpoint)
    print(f"  Checkpoint type : {ckpt_type}")
    print(f"  Voxels          : {model.num_voxels}")
    print(f"  Grid points     : {model.num_grid_pts}")

    # --- Extract mesh via direct MC ---
    print(f"\nExtracting mesh (direct MC, alpha={iso_alpha}) ...")
    mesh = direct_mc(model, iso_alpha=iso_alpha, bbox_scale=bbox_scale)

    if mesh is None or len(mesh.faces) == 0:
        print("Mesh extraction failed – try adjusting --alpha.")
        return False

    # --- Vertex colouring ---
    if color:
        print(f"\nColoring mesh vertices ...")
        mesh = color_mesh_vertices(mesh, model)

    # --- Export ---
    print(f"\nExporting to: {output}")
    mesh.export(output)
    size_mb = os.path.getsize(output) / (1024 * 1024)
    print(f"  {len(mesh.vertices)} verts, {len(mesh.faces)} faces, {size_mb:.2f} MB")
    print("Done.")
    return True


# ========================================================================= #
#  CLI                                                                       #
# ========================================================================= #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract mesh from object voxels using Direct Marching Cubes.")

    # Single file mode
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a single checkpoint or object voxel .pt file")
    parser.add_argument("--output", type=str, default=None,
                        help="Output mesh path (default: <checkpoint>_mesh.<format>)")

    # Batch mode
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Directory of object voxel .pt files (batch mode)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for batch mode")

    # Parameters
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Alpha threshold for iso-surface (default: 0.5)")
    parser.add_argument("--bbox_scale", type=float, default=1.0,
                        help="Scale factor for inside bounding box (default: 1.0)")
    parser.add_argument("--no_color", action="store_true",
                        help="Skip vertex colouring")
    parser.add_argument("--format", type=str, default="ply",
                        choices=["ply", "glb", "obj", "stl"],
                        help="Output format (default: ply)")

    args = parser.parse_args()

    if args.checkpoint_dir:
        # ---- Batch mode ----
        pt_files = sorted(glob.glob(os.path.join(args.checkpoint_dir, "*.pt")))
        if not pt_files:
            print(f"No .pt files found in {args.checkpoint_dir}")
            sys.exit(1)
        out_dir = args.output_dir or args.checkpoint_dir
        print(f"Batch mode: {len(pt_files)} files → {out_dir}\n")
        for pt_file in pt_files:
            print(f"{'='*60}")
            extract_mesh_object(
                checkpoint=pt_file,
                output=out_dir,
                iso_alpha=args.alpha,
                bbox_scale=args.bbox_scale,
                color=not args.no_color,
                format=args.format,
            )
            print()
    elif args.checkpoint:
        # ---- Single file mode ----
        extract_mesh_object(
            checkpoint=args.checkpoint,
            output=args.output,
            iso_alpha=args.alpha,
            bbox_scale=args.bbox_scale,
            color=not args.no_color,
            format=args.format,
        )
    else:
        parser.print_help()
        print("\nError: specify --checkpoint or --checkpoint_dir")
        sys.exit(1)
