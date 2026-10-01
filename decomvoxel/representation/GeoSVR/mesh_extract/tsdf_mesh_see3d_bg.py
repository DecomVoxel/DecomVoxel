import os
import sys
import copy
import argparse

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
GEOSVR_ROOT = os.path.dirname(PROJECT_ROOT)
if GEOSVR_ROOT not in sys.path:
    sys.path.insert(0, GEOSVR_ROOT)

import numpy as np
import open3d as o3d
import torch
from tqdm import tqdm

from src.config import cfg, update_config
from src.sparse_voxel_model import SparseVoxelModel

import train_bg_refine as bg_train


def post_process_mesh(mesh, cluster_to_keep=1):
    """
    Post-process a mesh to filter out floaters and disconnected parts.
    """
    print(f"Post processing mesh to keep {cluster_to_keep} clusters")
    mesh_0 = copy.deepcopy(mesh)

    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug):
        triangle_clusters, cluster_n_triangles, cluster_area = mesh_0.cluster_connected_triangles()

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


def infer_guidance_root(model_path, guidance_root=None):
    if guidance_root is not None:
        return os.path.abspath(guidance_root)

    model_path = os.path.abspath(model_path)
    return os.path.dirname(model_path)


def build_bg_datapack(args):
    args.guidance_root = infer_guidance_root(args.model_path, args.guidance_root)

    if args.merge_root is None:
        args.merge_root = os.path.join(args.guidance_root, "merge_3d_plane")
    if args.dn_root is None:
        args.dn_root = os.path.join(args.guidance_root, "see3d_mono_dn")

    bg_train.BG_ROOTS["guidance_root"] = args.guidance_root
    bg_train.BG_ROOTS["merge_root"] = args.merge_root
    bg_train.BG_ROOTS["dn_root"] = args.dn_root

    return bg_train.BackgroundDataPack(cfg.data, cfg.model.white_background)


@torch.no_grad()
def render_set(iteration, suffix, args, datapack, voxel_model, volume=None):
    views = datapack.get_train_cameras()

    tr_render_opt = {
        "track_max_w": False,
        "output_depth": True,
        "output_normal": True,
        "output_T": True,
    }

    depths_tsdf_fusion = []
    colors_fusion = []

    for view in tqdm(views, desc="Rendering progress"):
        render_pkg = voxel_model.render(view, **tr_render_opt)

        rendering = render_pkg["color"]
        depth = render_pkg["depth"][2].squeeze()
        depth_tsdf = depth.clone()

        if args.use_depth_filter:
            _, H, W = rendering.shape
            depth2normal = (
                view.depth2normal(render_pkg["depth"][0]).reshape(3, -1).permute(1, 0)
                @ view.world_view_transform[:3, :3]
            ).permute(1, 0).reshape(3, H, W)
            depth2normal *= -1

            view_dir = torch.nn.functional.normalize(view.get_rays(), p=2, dim=-1)
            depth_normal = depth2normal.permute(1, 2, 0)
            depth_normal = torch.nn.functional.normalize(depth_normal, p=2, dim=-1)

            dot = torch.sum(view_dir * depth_normal, dim=-1).abs().clamp(-1.0, 1.0)
            angle = torch.acos(dot)
            mask = angle > (80.0 / 180.0 * np.pi)
            depth_tsdf[mask] = 0

        depths_tsdf_fusion.append(depth_tsdf.cpu())

        color_np = np.ascontiguousarray(
            (rendering.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        )
        colors_fusion.append(color_np)

    if volume is not None:
        depths_tsdf_fusion = torch.stack(depths_tsdf_fusion, dim=0)

        for idx, view in enumerate(tqdm(views, desc="TSDF Fusion progress")):
            ref_depth = depths_tsdf_fusion[idx].cuda()

            # For See3D background training, mask usually marks originally observed regions.
            # Applying it here would remove inpainted background areas from fusion.
            if args.apply_mask and view.mask is not None:
                ref_depth[view.mask.squeeze() < 0.5] = 0

            ref_depth[ref_depth > args.max_depth] = 0
            ref_depth = ref_depth.detach().cpu().numpy()

            pose = np.identity(4, dtype=np.float64)
            pose[:3, :3] = view.R.transpose(-1, -2)
            pose[:3, 3] = view.T

            height = int(view.image_height)
            width = int(view.image_width)

            color = o3d.geometry.Image(colors_fusion[idx])
            depth = o3d.geometry.Image((ref_depth * 1000.0).astype(np.uint16))
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color,
                depth,
                depth_scale=1000.0,
                depth_trunc=args.max_depth,
                convert_rgb_to_intensity=False,
            )

            intrinsic = o3d.camera.PinholeCameraIntrinsic(
                width,
                height,
                float(view.Fx),
                float(view.Fy),
                float(view.Cx),
                float(view.Cy),
            )

            volume.integrate(rgbd, intrinsic, pose)

    torch.cuda.synchronize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract TSDF mesh from See3D background voxel model."
    )
    parser.add_argument("model_path")
    parser.add_argument("--guidance_root", default=None, type=str)
    parser.add_argument("--merge_root", default=None, type=str)
    parser.add_argument("--dn_root", default=None, type=str)

    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--clear_res_down", action="store_true")
    parser.add_argument("--suffix", default="", type=str)
    parser.add_argument("--overwrite_ss", default=None, type=float)
    parser.add_argument("--overwrite_vox_geo_mode", default=None)

    parser.add_argument("--max_depth", default=5.0, type=float)
    parser.add_argument("--voxel_size", default=0.002, type=float)
    parser.add_argument("--sdf_trunc_scale", default=2.0, type=float)
    parser.add_argument("--num_cluster", default=1, type=int)
    parser.add_argument("--use_depth_filter", action="store_true")

    # Disabled by default for See3D background extraction.
    parser.add_argument("--apply_mask", action="store_true")

    args = parser.parse_args()
    print(f"Rendering {args.model_path}")

    update_config(os.path.join(args.model_path, "config.yaml"))

    if args.clear_res_down:
        cfg.data.res_downscale = 0
        cfg.data.res_width = 0

    data_pack = build_bg_datapack(args)

    cfg.model.model_path = args.model_path

    voxel_model = SparseVoxelModel(cfg.model)
    loaded_iter = voxel_model.load_iteration(args.iteration)

    suffix = args.suffix
    if not suffix:
        if cfg.data.res_downscale > 0:
            suffix += f"_r{cfg.data.res_downscale}"
        if cfg.data.res_width > 0:
            suffix += f"_w{cfg.data.res_width}"

    if args.overwrite_ss is not None:
        voxel_model.ss = args.overwrite_ss
        if not args.suffix:
            suffix += f"_ss{args.overwrite_ss:.2f}"

    if args.overwrite_vox_geo_mode:
        voxel_model.vox_geo_mode = args.overwrite_vox_geo_mode
        if not args.suffix:
            suffix += f"_{args.overwrite_vox_geo_mode}"

    print(f"Guidance root : {args.guidance_root}")
    print(f"Merge root    : {args.merge_root}")
    print(f"DN root       : {args.dn_root}")

    voxel_model.freeze_vox_geo()

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=args.voxel_size,
        sdf_trunc=args.sdf_trunc_scale * args.voxel_size,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    render_set(
        loaded_iter,
        suffix,
        args,
        data_pack,
        voxel_model,
        volume,
    )

    print("Extracting triangle mesh...")
    mesh = volume.extract_triangle_mesh()

    tsdf_path = os.path.join(args.model_path, "mesh", "tsdf")
    os.makedirs(tsdf_path, exist_ok=True)

    raw_mesh_path = os.path.join(tsdf_path, "tsdf_fusion.ply")
    o3d.io.write_triangle_mesh(
        raw_mesh_path,
        mesh,
        write_triangle_uvs=True,
        write_vertex_colors=True,
        write_vertex_normals=True,
    )
    print(f"Raw mesh saved to: {raw_mesh_path}")

    mesh = post_process_mesh(mesh, args.num_cluster)

    post_mesh_path = os.path.join(tsdf_path, "tsdf_fusion_post.ply")
    o3d.io.write_triangle_mesh(
        post_mesh_path,
        mesh,
        write_triangle_uvs=True,
        write_vertex_colors=True,
        write_vertex_normals=True,
    )
    print(f"Post mesh saved to: {post_mesh_path}")