"""
Background-only TSDF mesh extraction.

Uses instance masks to exclude object regions from TSDF fusion,
producing a mesh that contains only the background (walls, floor, ceiling, etc.).

Instance masks are expected at:
    <source_path>/instance_masks/<image_name>.png
where pixel value 255 = background, everything else = object.

The object pixels are masked out (depth set to 0) before TSDF integration,
so the resulting mesh only contains background surfaces.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
GEOSVR_ROOT = os.path.join(os.path.dirname(PROJECT_ROOT))  # one level up: .../GeoSVR
if GEOSVR_ROOT not in sys.path:
    sys.path.insert(0, GEOSVR_ROOT)

import numpy as np
from tqdm import tqdm
import open3d as o3d
import torch
import copy
from PIL import Image

from src.config import cfg, update_argparser, update_config
from src.dataloader.data_pack import DataPack
from src.sparse_voxel_model import SparseVoxelModel


def post_process_mesh(mesh, cluster_to_keep=1):
    """Post-process a mesh to filter out floaters and disconnected parts."""
    print(f"post processing the mesh to have {cluster_to_keep} clusters")
    mesh_0 = copy.deepcopy(mesh)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        triangle_clusters, cluster_n_triangles, cluster_area = (
            mesh_0.cluster_connected_triangles()
        )

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    n_cluster = np.sort(cluster_n_triangles.copy())[-cluster_to_keep]
    n_cluster = max(n_cluster, 50)
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    mesh_0.remove_unreferenced_vertices()
    mesh_0.remove_degenerate_triangles()
    print(f"num vertices raw {len(mesh.vertices)}")
    print(f"num vertices post {len(mesh_0.vertices)}")
    return mesh_0


def load_instance_mask(mask_dir: str, image_name: str, H: int, W: int) -> np.ndarray:
    """
    Load an instance mask and return a boolean array where True = background.

    Args:
        mask_dir: directory containing instance mask PNGs
        image_name: stem name of the image (e.g. "000000")
        H, W: expected height and width of the rendered depth

    Returns:
        bg_mask: (H, W) bool array, True for background pixels
    """
    image_uid = image_name.split("_")[0]
    mask_path = os.path.join(mask_dir, f"{image_uid}.png")
    if not os.path.isfile(mask_path):
        print(f"[WARNING] Instance mask not found: {mask_path}, treating entire frame as background")
        return np.ones((H, W), dtype=bool)

    mask_img = np.array(Image.open(mask_path))
    # If RGB, take the first channel (all channels are identical)
    if mask_img.ndim == 3:
        mask_img = mask_img[:, :, 0]

    # Resize if needed (mask resolution might differ from rendered resolution)
    if mask_img.shape[0] != H or mask_img.shape[1] != W:
        mask_pil = Image.fromarray(mask_img)
        mask_pil = mask_pil.resize((W, H), Image.NEAREST)
        mask_img = np.array(mask_pil)

    # 255 = background, everything else = object
    bg_mask = (mask_img == 255)
    return bg_mask


@torch.no_grad()
def render_set_bg_only(name, iteration, suffix, args, datapack, voxel_model, volume=None):
    """
    Render all training views and integrate only background pixels into TSDF.
    Object pixels (identified by instance masks) have their depth zeroed out.
    """
    views = datapack.get_train_cameras()

    # Locate instance masks directory
    source_path = cfg.data.source_path
    mask_dir = args.mask_dir if args.mask_dir else os.path.join(source_path, "instance_masks")
    if not os.path.isdir(mask_dir):
        raise FileNotFoundError(
            f"Instance mask directory not found: {mask_dir}\n"
            f"Expected masks at <source_path>/instance_masks/ or specify --mask_dir"
        )
    print(f"[BG-TSDF] Using instance masks from: {mask_dir}")

    # Dilation: optionally expand object mask to remove boundary artifacts
    dilate_px = args.dilate_object_mask
    if dilate_px > 0:
        import cv2
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1))
        print(f"[BG-TSDF] Dilating object mask by {dilate_px} pixels")

    tr_render_opt = {
        'track_max_w': False,
        'output_depth': True,
        'output_normal': True,
        'output_T': True,
    }

    depths_tsdf_fusion = []
    colors_fusion = []
    bg_masks = []

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        render_pkg = voxel_model.render(view, **tr_render_opt)
        rendering = render_pkg['color']
        _, H, W = rendering.shape

        depth2normal = (
            view.depth2normal(render_pkg['depth'][0])
            .reshape(3, -1).permute(1, 0)
            @ view.world_view_transform[:3, :3]
        ).permute(1, 0).reshape(3, H, W)
        depth2normal *= -1

        depth = render_pkg['depth'][0].squeeze()
        depth_tsdf = depth.clone()

        # Depth-normal angle filter
        if args.use_depth_filter:
            view_dir = torch.nn.functional.normalize(view.get_rays(), p=2, dim=-1)
            depth_normal = depth2normal.permute(1, 2, 0)
            depth_normal = torch.nn.functional.normalize(depth_normal, p=2, dim=-1)
            dot = torch.sum(view_dir * depth_normal, dim=-1).abs()
            angle = torch.acos(dot.clamp(-1, 1))
            mask_angle = angle > (80.0 / 180 * 3.14159)
            depth_tsdf[mask_angle] = 0

        # Load instance mask: True = background
        bg_mask = load_instance_mask(mask_dir, view.image_name, H, W)

        # Optionally dilate the object (non-background) region
        if dilate_px > 0:
            obj_mask_uint8 = (~bg_mask).astype(np.uint8) * 255
            obj_mask_dilated = cv2.dilate(obj_mask_uint8, kernel, iterations=1)
            bg_mask = (obj_mask_dilated == 0)

        depths_tsdf_fusion.append(depth_tsdf.squeeze().cpu())
        color_np = np.ascontiguousarray(
            (rendering.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        )
        colors_fusion.append(color_np)
        bg_masks.append(bg_mask)

    if volume is not None:
        depths_tsdf_fusion = torch.stack(depths_tsdf_fusion, dim=0)
        n_masked_total = 0
        n_pixels_total = 0

        for idx, view in enumerate(tqdm(views, desc="TSDF Fusion (BG only)")):
            ref_depth = depths_tsdf_fusion[idx].cuda()

            # Apply existing view mask if present
            if view.mask is not None:
                ref_depth[view.mask.squeeze() < 0.5] = 0
            ref_depth[ref_depth > args.max_depth] = 0

            # ---- KEY: zero out object pixels so they are not integrated ----
            bg_mask = bg_masks[idx]
            obj_mask_torch = torch.from_numpy(~bg_mask).cuda()  # True = object
            ref_depth[obj_mask_torch] = 0

            n_obj = obj_mask_torch.sum().item()
            n_masked_total += n_obj
            n_pixels_total += obj_mask_torch.numel()

            ref_depth = ref_depth.detach().cpu().numpy()

            pose = np.identity(4)
            pose[:3, :3] = view.R.transpose(-1, -2)
            pose[:3, 3] = view.T

            color = o3d.geometry.Image(colors_fusion[idx])
            depth = o3d.geometry.Image((ref_depth * 1000).astype(np.uint16))
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color, depth,
                depth_scale=1000.0, depth_trunc=args.max_depth,
                convert_rgb_to_intensity=False,
            )
            volume.integrate(
                rgbd,
                o3d.camera.PinholeCameraIntrinsic(W, H, view.Fx, view.Fy, view.Cx, view.Cy),
                pose,
            )

        pct = n_masked_total / max(n_pixels_total, 1) * 100
        print(f"[BG-TSDF] Masked out {n_masked_total} object pixels total "
              f"({pct:.1f}% of all pixels across {len(views)} views)")

    torch.cuda.synchronize()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Background-only TSDF mesh extraction using instance masks."
    )
    parser.add_argument("--checkpoint", required=True, type=str,
                        help="Path to the GeoSVR voxel checkpoint .pt file")
    parser.add_argument("--config", required=True, type=str,
                        help="Path to the GeoSVR config.yaml file")
    parser.add_argument("--output", required=True, type=str,
                        help="Output directory for the extracted mesh")
    parser.add_argument("--clear_res_down", action="store_true")
    parser.add_argument("--overwrite_ss", default=None, type=float)
    parser.add_argument("--overwrite_vox_geo_mode", default=None)

    # TSDF parameters
    parser.add_argument("--max_depth", default=5.0, type=float)
    parser.add_argument("--voxel_size", default=0.002, type=float)
    parser.add_argument("--sdf_trunc_scale", default=2.0, type=float)
    parser.add_argument("--num_cluster", default=1, type=int)
    parser.add_argument("--use_depth_filter", action="store_true")

    # Background mask parameters
    parser.add_argument("--mask_dir", default=None, type=str,
                        help="Path to instance mask directory. "
                             "Default: <source_path>/instance_masks/")
    parser.add_argument("--dilate_object_mask", default=0, type=int,
                        help="Dilate object mask by N pixels to remove boundary "
                             "artifacts (default: 0, no dilation)")

    args = parser.parse_args()
    print("=" * 60)
    print("Background-only TSDF Mesh Extraction")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Config:     {args.config}")
    print(f"Output:     {args.output}")

    # Load config
    update_config(args.config)

    if args.clear_res_down:
        cfg.data.res_downscale = 0
        cfg.data.res_width = 0

    # Load data
    data_pack = DataPack(cfg.data, cfg.model.white_background)

    # Set model_path to the directory containing the checkpoint
    # (needed internally by SparseVoxelModel.__init__)
    cfg.model.model_path = os.path.dirname(os.path.dirname(args.checkpoint))

    # Load model directly from checkpoint path
    voxel_model = SparseVoxelModel(cfg.model)
    voxel_model.load(args.checkpoint)

    suffix = ""

    if args.overwrite_ss:
        voxel_model.ss = args.overwrite_ss

    if args.overwrite_vox_geo_mode:
        voxel_model.vox_geo_mode = args.overwrite_vox_geo_mode

    voxel_model.freeze_vox_geo()

    # Create TSDF volume
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=args.voxel_size,
        sdf_trunc=args.sdf_trunc_scale * args.voxel_size,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    # Render and fuse (background only)
    render_set_bg_only(
        "train", 0, suffix, args,
        data_pack, voxel_model, volume,
    )

    # Extract mesh
    print("Extracting triangle mesh from TSDF...")
    mesh = volume.extract_triangle_mesh()

    os.makedirs(args.output, exist_ok=True)

    raw_path = os.path.join(args.output, "tsdf_fusion_bg.ply")
    o3d.io.write_triangle_mesh(
        raw_path, mesh,
        write_triangle_uvs=True, write_vertex_colors=True, write_vertex_normals=True,
    )
    print(f"[BG-TSDF] Raw mesh saved to: {raw_path}")

    # Post-process
    mesh = post_process_mesh(mesh, args.num_cluster)
    post_path = os.path.join(args.output, "tsdf_fusion_bg_post.ply")
    o3d.io.write_triangle_mesh(
        post_path, mesh,
        write_triangle_uvs=True, write_vertex_colors=True, write_vertex_normals=True,
    )
    print(f"[BG-TSDF] Post-processed mesh saved to: {post_path}")

    print("=" * 60)
    print("Done!")
    print("=" * 60)
