import copy
import json
import os
import sys
from argparse import ArgumentParser

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

try:
    import open3d as o3d
except ImportError as exc:
    raise ImportError(
        "open3d is required for tsdf_refine_bg.py. "
        "Install it with `pip install open3d`."
    ) from exc


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
GEOSVR_ROOT = os.path.dirname(PROJECT_ROOT)
if GEOSVR_ROOT not in sys.path:
    sys.path.insert(0, GEOSVR_ROOT)

from cam_util import load_selected_camera_params


def post_process_mesh(mesh, cluster_to_keep=1):
    """Post-process a mesh to filter out floaters and disconnected parts."""
    if len(mesh.triangles) == 0:
        return mesh

    print(f"Post processing mesh to keep {cluster_to_keep} clusters")
    mesh_0 = copy.deepcopy(mesh)

    with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug
    ):
        triangle_clusters, cluster_n_triangles, cluster_area = (
            mesh_0.cluster_connected_triangles()
        )

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)

    if cluster_n_triangles.size == 0:
        return mesh_0

    keep_k = min(max(int(cluster_to_keep), 1), len(cluster_n_triangles))
    n_cluster = np.sort(cluster_n_triangles.copy())[-keep_k]
    n_cluster = max(int(n_cluster), 50)

    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    mesh_0.remove_unreferenced_vertices()
    mesh_0.remove_degenerate_triangles()
    mesh_0.remove_duplicated_triangles()
    mesh_0.remove_duplicated_vertices()

    print(f"Num vertices raw  : {len(mesh.vertices)}")
    print(f"Num vertices post : {len(mesh_0.vertices)}")
    return mesh_0


def read_depth_npy(depth_path, width=None, height=None):
    depth = np.load(depth_path).astype(np.float32)
    if width is not None and height is not None and depth.shape != (height, width):
        depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST)
    return depth


def read_bool_mask_npy(mask_path, width=None, height=None):
    mask = np.load(mask_path).astype(bool)
    if width is not None and height is not None and mask.shape != (height, width):
        mask = cv2.resize(
            mask.astype(np.uint8),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        ) > 0
    return mask


def read_rgb_image(image_path, width=None, height=None):
    image = Image.open(image_path).convert("RGB")
    if width is not None and height is not None and image.size != (width, height):
        image = image.resize((width, height), Image.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def resolve_rgb_path(guidance_root, dn_root, frame_idx, camera_name):
    candidates = [
        os.path.join(dn_root, f"rgb_frame{frame_idx:06d}.png"),
        os.path.join(guidance_root, "raw_rgb", f"{camera_name}.png"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def camera_to_o3d_intrinsic(cam):
    width = int(cam.image_width)
    height = int(cam.image_height)

    fx = 0.5 * width / np.tan(float(cam.fovx) * 0.5)
    fy = 0.5 * height / np.tan(float(cam.fovy) * 0.5)
    cx = float(width) * float(cam.cx_p)
    cy = float(height) * float(cam.cy_p)

    return o3d.camera.PinholeCameraIntrinsic(
        width,
        height,
        fx,
        fy,
        cx,
        cy,
    )


def load_refined_rgbd_sequence(guidance_root, merge_root, dn_root, cameras, max_depth):
    colors = []
    depths = []
    frame_ids = []
    trusted_pixels = []
    rgb_sources = []

    for cam in cameras:
        frame_idx = int(cam.image_name)
        width = int(cam.image_width)
        height = int(cam.image_height)

        # depth_path = os.path.join(
        #     merge_root,
        #     f"plane_refined_depth_frame{frame_idx:06d}.npy",
        # )
        # conf_path = os.path.join(
        #     merge_root,
        #     f"plane_refined_conf_frame{frame_idx:06d}.npy",
        # )

        # use merge refined depth/conf
        depth_path = os.path.join(
            merge_root,
            f"merge_refined_depth_frame{frame_idx:06d}.npy",
        )
        conf_path = os.path.join(
            merge_root,
            f"merge_refined_conf_frame{frame_idx:06d}.npy",
        )

        if not os.path.exists(depth_path):
            raise FileNotFoundError(f"Refined depth not found: {depth_path}")
        if not os.path.exists(conf_path):
            raise FileNotFoundError(f"Refined confidence not found: {conf_path}")

        depth_np = read_depth_npy(depth_path, width=width, height=height)
        conf_np = read_bool_mask_npy(conf_path, width=width, height=height)

        valid = np.isfinite(depth_np) & (depth_np > 0) & conf_np
        if max_depth is not None and max_depth > 0:
            valid &= depth_np <= float(max_depth)

        depth_filtered = np.zeros_like(depth_np, dtype=np.float32)
        depth_filtered[valid] = depth_np[valid]

        rgb_path = resolve_rgb_path(
            guidance_root=guidance_root,
            dn_root=dn_root,
            frame_idx=frame_idx,
            camera_name=str(cam.image_name),
        )

        if rgb_path is None:
            print(
                f"[Warning] RGB not found for frame {frame_idx:06d}. "
                "Using a white image as fallback."
            )
            color_np = np.full((height, width, 3), 255, dtype=np.uint8)
            rgb_sources.append("fallback_white")
        else:
            color_np = read_rgb_image(rgb_path, width=width, height=height)
            rgb_sources.append(rgb_path)

        colors.append(np.ascontiguousarray(color_np))
        depths.append(depth_filtered)
        frame_ids.append(frame_idx)
        trusted_pixels.append(int(valid.sum()))

    return frame_ids, colors, depths, trusted_pixels, rgb_sources


@torch.no_grad()
def integrate_refined_tsdf(cameras, colors, depths, args):
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=float(args.voxel_size),
        sdf_trunc=float(args.sdf_trunc_scale) * float(args.voxel_size),
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    valid_pixels_total = 0
    pixels_total = 0
    fused_view_count = 0

    for cam, color_np, depth_np in tqdm(
        zip(cameras, colors, depths),
        total=len(cameras),
        desc="TSDF Fusion (plane refined)",
    ):
        height, width = depth_np.shape
        n_valid = int((depth_np > 0).sum())

        valid_pixels_total += n_valid
        pixels_total += height * width

        if n_valid == 0:
            continue

        fused_view_count += 1

        color = o3d.geometry.Image(np.ascontiguousarray(color_np))
        depth = o3d.geometry.Image(
            np.ascontiguousarray((depth_np * 1000.0).astype(np.uint16))
        )
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color,
            depth,
            depth_scale=1000.0,
            depth_trunc=float(args.max_depth),
            convert_rgb_to_intensity=False,
        )

        intrinsic = camera_to_o3d_intrinsic(cam)
        extrinsic = cam.w2c.detach().cpu().numpy().astype(np.float64)

        volume.integrate(rgbd, intrinsic, extrinsic)

    valid_ratio = 100.0 * valid_pixels_total / max(pixels_total, 1)
    print(
        f"[TSDF] Integrated {fused_view_count}/{len(cameras)} views, "
        f"trusted pixels={valid_pixels_total} ({valid_ratio:.2f}% of all pixels)"
    )
    return volume


def save_summary(
    output_root,
    guidance_root,
    merge_root,
    dn_root,
    frame_ids,
    trusted_pixels,
    rgb_sources,
    raw_mesh,
    post_mesh,
    args,
):
    summary = {
        "guidance_root": guidance_root,
        "merge_root": merge_root,
        "dn_root": dn_root,
        "num_views": len(frame_ids),
        "frame_ids": frame_ids,
        "avg_trusted_pixels": float(np.mean(trusted_pixels)) if trusted_pixels else 0.0,
        "max_depth": float(args.max_depth),
        "voxel_size": float(args.voxel_size),
        "sdf_trunc_scale": float(args.sdf_trunc_scale),
        "num_cluster": int(args.num_cluster),
        "raw_mesh_vertices": int(len(raw_mesh.vertices)),
        "raw_mesh_triangles": int(len(raw_mesh.triangles)),
        "post_mesh_vertices": int(len(post_mesh.vertices)),
        "post_mesh_triangles": int(len(post_mesh.triangles)),
        "rgb_sources": rgb_sources,
        "compat_checkpoint": args.checkpoint,
        "compat_config": args.config,
    }

    summary_path = os.path.join(output_root, "tsdf_refine_bg_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main(args):
    if args.merge_root is None:
        args.merge_root = os.path.join(args.guidance_root, "merge_3d_plane")
    if args.dn_root is None:
        args.dn_root = os.path.join(args.guidance_root, "see3d_mono_dn")
    if args.output_root is None:
        args.output_root = os.path.join(args.guidance_root, "tsdf_refine_bg")

    os.makedirs(args.output_root, exist_ok=True)

    camera_json_path = os.path.join(args.guidance_root, "camera_params.json")
    if not os.path.exists(camera_json_path):
        raise FileNotFoundError(f"camera_params.json not found: {camera_json_path}")

    cameras = load_selected_camera_params(camera_json_path)

    print("=" * 60)
    print("Plane Refined TSDF Background Mesh Extraction")
    print("=" * 60)
    print(f"Guidance root : {args.guidance_root}")
    print(f"Merge root    : {args.merge_root}")
    print(f"DN root       : {args.dn_root}")
    print(f"Output root   : {args.output_root}")
    print(f"Cameras       : {len(cameras)}")
    print(f"Voxel size    : {args.voxel_size}")
    print(f"SDF trunc     : {args.sdf_trunc_scale * args.voxel_size}")
    print(f"Max depth     : {args.max_depth}")

    frame_ids, colors, depths, trusted_pixels, rgb_sources = load_refined_rgbd_sequence(
        guidance_root=args.guidance_root,
        merge_root=args.merge_root,
        dn_root=args.dn_root,
        cameras=cameras,
        max_depth=args.max_depth,
    )

    volume = integrate_refined_tsdf(
        cameras=cameras,
        colors=colors,
        depths=depths,
        args=args,
    )

    print("Extracting triangle mesh from TSDF...")
    raw_mesh = volume.extract_triangle_mesh()
    raw_mesh.compute_vertex_normals()

    raw_path = os.path.join(args.output_root, "tsdf_refine_bg_raw.ply")
    o3d.io.write_triangle_mesh(
        raw_path,
        raw_mesh,
        write_triangle_uvs=True,
        write_vertex_colors=True,
        write_vertex_normals=True,
    )
    print(f"[TSDF] Raw mesh saved to: {raw_path}")

    post_mesh = post_process_mesh(raw_mesh, cluster_to_keep=args.num_cluster)
    post_mesh.compute_vertex_normals()

    post_path = os.path.join(args.output_root, "tsdf_refine_bg_post.ply")
    o3d.io.write_triangle_mesh(
        post_path,
        post_mesh,
        write_triangle_uvs=True,
        write_vertex_colors=True,
        write_vertex_normals=True,
    )
    print(f"[TSDF] Post-processed mesh saved to: {post_path}")

    save_summary(
        output_root=args.output_root,
        guidance_root=args.guidance_root,
        merge_root=args.merge_root,
        dn_root=args.dn_root,
        frame_ids=frame_ids,
        trusted_pixels=trusted_pixels,
        rgb_sources=rgb_sources,
        raw_mesh=raw_mesh,
        post_mesh=post_mesh,
        args=args,
    )

    print("=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Extract background mesh from plane refined depth/conf using Open3D TSDF."
    )

    parser.add_argument(
        "--guidance_root",
        required=True,
        type=str,
        help="Path to see3d_guidance_views containing camera_params.json",
    )
    parser.add_argument(
        "--merge_root",
        default=None,
        type=str,
        help="Directory containing plane_refined_depth/conf outputs. Default: <guidance_root>/merge_3d_plane",
    )
    parser.add_argument(
        "--dn_root",
        default=None,
        type=str,
        help="Directory containing rgb_frame*.png. Default: <guidance_root>/see3d_mono_dn",
    )
    parser.add_argument(
        "--output_root",
        default=None,
        type=str,
        help="Output directory. Default: <merge_root>/tsdf_refine_bg",
    )

    parser.add_argument(
        "--max_depth",
        default=5.0,
        type=float,
        help="Depth truncation threshold in meters",
    )
    parser.add_argument(
        "--voxel_size",
        default=0.01,
        type=float,
        help="Open3D TSDF voxel length in meters",
    )
    parser.add_argument(
        "--sdf_trunc_scale",
        default=2.0,
        type=float,
        help="SDF truncation scale relative to voxel_size",
    )
    parser.add_argument(
        "--num_cluster",
        default=1,
        type=int,
        help="Number of connected mesh clusters to keep in post-processing",
    )

    # Compatibility arguments kept so old shell commands do not break.
    parser.add_argument("--checkpoint", default=None, type=str)
    parser.add_argument("--config", default=None, type=str)

    args = parser.parse_args()
    main(args)