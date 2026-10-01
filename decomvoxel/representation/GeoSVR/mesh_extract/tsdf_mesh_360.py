#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import time
import numpy as np
from tqdm import tqdm
from os import makedirs
import imageio
import open3d as o3d
import torch

from src.config import cfg, update_argparser, update_config

from src.dataloader.data_pack import DataPack
from src.sparse_voxel_model import SparseVoxelModel
from src.utils.image_utils import im_tensor2np, viz_tensordepth


def clean_mesh(mesh, min_len=1000):
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        triangle_clusters, cluster_n_triangles, cluster_area = (mesh.cluster_connected_triangles())
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < min_len
    mesh_0 = copy.deepcopy(mesh)
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    return mesh_0

def post_process_mesh(mesh, cluster_to_keep=1):
    """
    Post-process a mesh to filter out floaters and disconnected parts
    """
    import copy
    print("post processing the mesh to have {} clusterscluster_to_kep".format(cluster_to_keep))
    mesh_0 = copy.deepcopy(mesh)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            triangle_clusters, cluster_n_triangles, cluster_area = (mesh_0.cluster_connected_triangles())

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    n_cluster = np.sort(cluster_n_triangles.copy())[-cluster_to_keep]
    n_cluster = max(n_cluster, 50) # filter meshes smaller than 50
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    mesh_0.remove_unreferenced_vertices()
    mesh_0.remove_degenerate_triangles()
    print("num vertices raw {}".format(len(mesh.vertices)))
    print("num vertices post {}".format(len(mesh_0.vertices)))
    return mesh_0

@torch.no_grad()
def estimate_bounding_sphere(viewpoint_stack):
    """
    Estimate the bounding sphere given camera pose
    """
    from src.utils.render_utils import transform_poses_pca, focus_point_fn
    torch.cuda.empty_cache()
    c2ws = np.array([np.linalg.inv(np.asarray((cam.world_view_transform.T).cpu().numpy())) for cam in viewpoint_stack])
    poses = c2ws[:,:3,:] @ np.diag([1, -1, -1, 1])
    center = (focus_point_fn(poses))
    radius = np.linalg.norm(c2ws[:,:3,3] - center, axis=-1).min()
    center = torch.from_numpy(center).float().cuda()
    print(f"The estimated bounding radius is {radius:.2f}")
    return radius, center


@torch.no_grad()
def render_set(name, iteration, suffix, args, datapack, voxel_model, volume=None):
    views = datapack.get_train_cameras()
    render_path = os.path.join(voxel_model.model_path, name, f"ours_{iteration}{suffix}", "renders")

    tr_render_opt = {
        'track_max_w': False,
        'output_depth': True,
        'output_normal': True,
        'output_T': True,
    }
    depths_tsdf_fusion = []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        render_pkg = voxel_model.render(view, **tr_render_opt)

        rendering = render_pkg['color']
        _, H, W = rendering.shape
                
        depth2normal = (view.depth2normal(render_pkg['depth'][0]).reshape(3,-1).permute(1,0) @ view.world_view_transform[:3,:3]).permute(1,0).reshape(3, H, W)
        depth2normal *= -1
        depth = render_pkg['depth'][0].squeeze()
        depth_tsdf = depth.clone()
        
        
        if args.use_depth_filter:
            view_dir = torch.nn.functional.normalize(view.get_rays(), p=2, dim=-1)
            depth_normal = depth2normal.permute(1,2,0)
            depth_normal = torch.nn.functional.normalize(depth_normal, p=2, dim=-1)
            dot = torch.sum(view_dir*depth_normal, dim=-1).abs()
            angle = torch.acos(dot)
            mask = angle > (80.0 / 180 * 3.14159)
            depth_tsdf[mask] = 0
        depths_tsdf_fusion.append(depth_tsdf.squeeze().cpu())
        
    if volume is not None:
        depths_tsdf_fusion = torch.stack(depths_tsdf_fusion, dim=0)
        for idx, view in enumerate(tqdm(views, desc="TSDF Fusion progress")):
            ref_depth = depths_tsdf_fusion[idx].cuda()

            if view.mask is not None:
                ref_depth[view.mask.squeeze() < 0.5] = 0
            ref_depth[ref_depth > args.max_depth] = 0
            ref_depth = ref_depth.detach().cpu().numpy()
            
            pose = np.identity(4)
            pose[:3,:3] = view.R.transpose(-1,-2)
            pose[:3, 3] = view.T
            rgb_path = os.path.join(render_path, view.image_name + ".jpg")
            if not os.path.exists(rgb_path):
                rgb_path = os.path.join(render_path, view.image_name + ".png")
            color = o3d.io.read_image(rgb_path)
            depth = o3d.geometry.Image((ref_depth*1000).astype(np.uint16))
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color, depth, depth_scale=1000.0, depth_trunc=args.max_depth, convert_rgb_to_intensity=False)
            volume.integrate(
                rgbd,
                o3d.camera.PinholeCameraIntrinsic(W, H, view.Fx, view.Fy, view.Cx, view.Cy),
                pose)

        
            
            
            
    torch.cuda.synchronize()
    


if __name__ == "__main__":
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(
        description="Sparse voxels raster rendering.")
    parser.add_argument('model_path')
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--clear_res_down", action="store_true")
    parser.add_argument("--suffix", default="", type=str)
    parser.add_argument("--overwrite_ss", default=None, type=float)
    parser.add_argument("--overwrite_vox_geo_mode", default=None)
    parser.add_argument("--num_cluster", default=1, type=int)
    parser.add_argument("--radius_scale", default=2.0, type=float)
    parser.add_argument("--use_depth_filter", action="store_true")
    args = parser.parse_args()
    print("Rendering " + args.model_path)

    # Load config
    update_config(os.path.join(args.model_path, 'config.yaml'))

    if args.clear_res_down:
        cfg.data.res_downscale = 0
        cfg.data.res_width = 0

    # Load data
    data_pack = DataPack(cfg.data, cfg.model.white_background)

    cfg.model.model_path = args.model_path
    
    # Load model
    voxel_model = SparseVoxelModel(cfg.model)
    loaded_iter = voxel_model.load_iteration(args.iteration)

    # Output path suffix
    suffix = args.suffix
    if not args.suffix:
        if cfg.data.res_downscale > 0:
            suffix += f"_r{cfg.data.res_downscale}"
        if cfg.data.res_width > 0:
            suffix += f"_w{cfg.data.res_width}"

    if args.overwrite_ss:
        voxel_model.ss = args.overwrite_ss
        if not args.suffix:
            suffix += f"_ss{args.overwrite_ss:.2f}"
    
    if args.overwrite_vox_geo_mode:
        voxel_model.vox_geo_mode = args.overwrite_vox_geo_mode
        if not args.suffix:
            suffix += f"_{args.overwrite_vox_geo_mode}"

    voxel_model.freeze_vox_geo()
    
    radius, center = estimate_bounding_sphere(data_pack.get_train_cameras())
    
    args.max_depth = (radius * args.radius_scale)
    voxel_size = (args.max_depth / 1536)
    # voxel_size = (args.max_depth / 1024)
    sdf_trunc = 5.0 * voxel_size
    
    print(f"depth_trunc: {args.max_depth:.2f}, voxel_size: {voxel_size:.2f}, sdf_trunc: {sdf_trunc:.2f}")
    
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=voxel_size,
    sdf_trunc=sdf_trunc,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    render_set(
        "train", loaded_iter, suffix, args,
        data_pack, voxel_model, volume)
    print(f"extract_triangle_mesh")
    mesh = volume.extract_triangle_mesh()

    tsdf_path = os.path.join(args.model_path, "mesh", "tsdf")
    os.makedirs(tsdf_path, exist_ok=True)
    
    o3d.io.write_triangle_mesh(os.path.join(tsdf_path, "tsdf_fusion.ply"), mesh, 
                                write_triangle_uvs=True, write_vertex_colors=True, write_vertex_normals=True)
    
    mesh = post_process_mesh(mesh, args.num_cluster)
    o3d.io.write_triangle_mesh(os.path.join(tsdf_path, "tsdf_fusion_post.ply"), mesh, 
                                write_triangle_uvs=True, write_vertex_colors=True, write_vertex_normals=True)

