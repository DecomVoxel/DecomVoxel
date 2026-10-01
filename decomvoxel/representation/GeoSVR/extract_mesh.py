# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import math
import time
import numpy as np
from tqdm import tqdm
import trimesh
import sys

import torch
import svraster_cuda

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

GEOSVR_ROOT = os.path.join(PROJECT_ROOT, 'decomvoxel', 'representation', 'GeoSVR')
if GEOSVR_ROOT not in sys.path:
    sys.path.insert(0, GEOSVR_ROOT)

from yacs.config import CfgNode

from src.config import cfg, update_argparser, update_config
from src.utils import octree_utils
from src.utils import activation_utils
from svraster_cuda.marching_cubes import torch_marching_cubes_grid
from src.sparse_voxel_gears.adaptive import subdivide_by_interp, agg_voxel_into_grid_pts

from src.dataloader.data_pack import DataPack
from src.sparse_voxel_model import SparseVoxelModel


def _make_temp_vol(outside_level=0):
    """Create a temporary SparseVoxelModel for TSDF/MC with a minimal config."""
    vol_cfg = CfgNode(dict(
        model_path="",
        vox_geo_mode=cfg.model.vox_geo_mode,
        density_mode=cfg.model.density_mode,
        sh_degree=0,
        ss=cfg.model.ss,
        outside_level=outside_level,
        white_background=cfg.model.white_background,
        black_background=cfg.model.black_background,
    ))
    return SparseVoxelModel(vol_cfg)


def _init_vol_from_octpath(vol, scene_center, scene_extent, inside_extent, octpath, octlevel):
    """Initialise a temporary SparseVoxelModel from existing octpath/octlevel tensors
    by building a minimal state dict and delegating to load_from_state_dict."""
    N = len(octpath)
    # Need grid pts count to pre-allocate _geo_grid_pts with the correct shape
    grid_pts_key_tmp, _ = octree_utils.build_grid_pts_link(octpath, octlevel)
    G = len(grid_pts_key_tmp)
    vol.load_from_state_dict({
        'active_sh_degree': 0,
        'ss': vol.ss,
        'scene_center': scene_center,
        'inside_extent': inside_extent,
        'scene_extent': scene_extent,
        'octpath': octpath,
        'octlevel': octlevel,
        '_geo_grid_pts': torch.zeros([G, 1], dtype=torch.float32, device='cuda'),
        '_sh0': torch.zeros([N, 3], dtype=torch.float32, device='cuda'),
        '_shs': torch.zeros([N, 0, 3], dtype=torch.float32, device='cuda'),
        'quantized': False,
    })

from src.utils.fuser_utils import Fuser


def tsdf_fusion(
        cam_lst, depth_lst, alpha_lst,
        grid_pts_xyz, trunc_dist, crop_border, alpha_thres):

    assert len(cam_lst) == len(depth_lst)
    assert len(cam_lst) == len(alpha_lst)

    fuser = Fuser(
        xyz=grid_pts_xyz,
        bandwidth=trunc_dist,
        use_trunc=True,
        fuse_tsdf=True,
        feat_dim=0,
        alpha_thres=alpha_thres,
        crop_border=crop_border,
        normal_weight=False,
        depth_weight=False,
        border_weight=False,
        use_half=False)

    for cam, frame_depth, frame_alpha in zip(tqdm(cam_lst), depth_lst, alpha_lst):

        frame_depth = frame_depth.cuda()
        frame_alpha = frame_alpha.cuda()

        fuser.integrate(cam, frame_depth, alpha=frame_alpha)

    tsdf = fuser.tsdf.squeeze(1).contiguous()
    return tsdf


def extract_mesh_progressive(args, data_pack, voxel_model, init_lv, final_lv, crop_bbox):

    # Render depth and alpha for all training views
    cam_lst = data_pack.get_train_cameras()
    depth_lst = []
    alpha_lst = []
    for cam in tqdm(cam_lst, desc="Render training views"):
        render_pkg = voxel_model.render(cam, output_depth=True, output_T=True)
        if args.use_mean:
            frame_depth = render_pkg['raw_depth'][[0]]  # Use mean depth
        else:
            frame_depth = render_pkg['raw_depth'][[2]]  # Use median depth
        frame_alpha = 1 - render_pkg['raw_T']
        if args.save_gpu:
            frame_depth = frame_depth.cpu()
            frame_alpha = frame_alpha.cpu()
        depth_lst.append(frame_depth)
        alpha_lst.append(frame_alpha)

    # Determine bounding volume for marching cube
    if crop_bbox is None:
        inside_min = voxel_model.scene_center - 0.5 * voxel_model.inside_extent * args.bbox_scale
        inside_max = voxel_model.scene_center + 0.5 * voxel_model.inside_extent * args.bbox_scale
    else:
        inside_min = torch.tensor(crop_bbox[0], dtype=torch.float32, device="cuda")
        inside_max = torch.tensor(crop_bbox[1], dtype=torch.float32, device="cuda")

    # Initialize a dense grid
    vol = _make_temp_vol(outside_level=0)
    _cfg_init = CfgNode(dict(
        init_n_level=init_lv,
        sh_degree_init=0,
        geo_init=-10.0,
        sh0_init=0.5,
        shs_init=0.0,
        outside_mode="none",
        aabb_crop=False,
    ))
    vol.model_init(
        bounding=torch.stack([inside_min, inside_max]),
        cfg_init=_cfg_init,
    )

    # Run progressive TSDF fusion
    for lv in range(init_lv, final_lv+1):

        # Determine bandwidth
        now_voxel_size = vol.vox_size[0].item()
        bandwidth = args.bandwidth_vox * now_voxel_size
        print(f"Running lv={lv:2d}: #voxels={vol.num_voxels:9d}; vox_size={now_voxel_size}; band={bandwidth}")

        # Run tsdf fusion at current levels
        grid_tsdf = tsdf_fusion(
            cam_lst, depth_lst, alpha_lst,
            vol.grid_pts_xyz, bandwidth, args.crop_border, args.alpha_thres)

        # Progressive to next level
        if lv < final_lv:

            # Prune voxels
            vox_tsdf = grid_tsdf[vol.vox_key]  # Get the sd values of voxel corners [#vox, 8]
            thickness = 2 / args.bandwidth_vox  # Keep voxels touching a surface thickness of 2 voxels
            thickness = min(thickness, 0.99)
            prune_mask = vox_tsdf.isnan().any(-1) | \
                        (vox_tsdf.amax(1) < -thickness) | \
                        (vox_tsdf.amin(1) > thickness)
            vol.pruning(prune_mask)

            # Subdivide to next level
            vol.subdividing(torch.ones([vol.num_voxels], dtype=torch.bool))

    # Extract mesh from grid
    verts, faces = torch_marching_cubes_grid(
        grid_pts_val=grid_tsdf,
        grid_pts_xyz=vol.grid_pts_xyz,
        vox_key=vol.vox_key,
        iso=0)
    mesh = trimesh.Trimesh(verts.cpu().numpy(), faces.cpu().numpy())
    return mesh


def extract_mesh(args, data_pack, voxel_model, final_lv, crop_bbox, iso=0):

    # Render depth and alpha for all training views
    cam_lst = data_pack.get_train_cameras()
    depth_lst = []
    alpha_lst = []
    for cam in tqdm(cam_lst, desc="Render training views"):
        render_pkg = voxel_model.render(cam, output_depth=True, output_T=True)
        if args.use_mean:
            frame_depth = render_pkg['raw_depth'][[0]]  # Use mean depth
        else:
            frame_depth = render_pkg['raw_depth'][[2]]  # Use median depth
        frame_alpha = 1 - render_pkg['raw_T']
        if args.save_gpu:
            frame_depth = frame_depth.cpu()
            frame_alpha = frame_alpha.cpu()
        depth_lst.append(frame_depth)
        alpha_lst.append(frame_alpha)

    # Filter background voxels
    if crop_bbox is None:
        inside_min = voxel_model.scene_center - 0.5 * voxel_model.inside_extent * args.bbox_scale
        inside_max = voxel_model.scene_center + 0.5 * voxel_model.inside_extent * args.bbox_scale
    else:
        inside_min = torch.tensor(crop_bbox[0], dtype=torch.float32, device="cuda")
        inside_max = torch.tensor(crop_bbox[1], dtype=torch.float32, device="cuda")

    # Clamp levels
    target_lv = voxel_model.outside_level + final_lv
    octpath, octlevel = octree_utils.clamp_level(voxel_model.octpath, voxel_model.octlevel, target_lv)

    # Initialize from clamped adaptive sparse voxels
    vol = _make_temp_vol(outside_level=voxel_model.outside_level)
    _init_vol_from_octpath(
        vol,
        voxel_model.scene_center,
        voxel_model.scene_extent,
        voxel_model.inside_extent,
        octpath,
        octlevel,
    )

    # Prune voxel outside
    gridpts_outside = ((vol.grid_pts_xyz < inside_min) | (vol.grid_pts_xyz > inside_max)).any(-1)
    corners_outside = gridpts_outside[vol.vox_key]
    prune_mask = corners_outside.all(-1)
    vol.pruning(prune_mask)

    # Determine bandwidth
    bandwidth = args.bandwidth_vox * vol.vox_size.min().item()

    # Run TSDF fusion
    print(f"Running adaptive: #voxels={vol.num_voxels:9d} / band={bandwidth}")
    grid_tsdf = tsdf_fusion(
        cam_lst, depth_lst, alpha_lst,
        vol.grid_pts_xyz, bandwidth, args.crop_border, args.alpha_thres)

    # Extract mesh from grid
    verts, faces = torch_marching_cubes_grid(
        grid_pts_val=grid_tsdf,
        grid_pts_xyz=vol.grid_pts_xyz,
        vox_key=vol.vox_key,
        iso=iso)
    mesh = trimesh.Trimesh(verts.cpu().numpy(), faces.cpu().numpy())
    return mesh


def direct_mc(args, voxel_model, final_lv, crop_bbox):
    # Filter background voxels
    if crop_bbox is None:
        inside_min = voxel_model.scene_center - 0.5 * voxel_model.inside_extent * args.bbox_scale
        inside_max = voxel_model.scene_center + 0.5 * voxel_model.inside_extent * args.bbox_scale
    else:
        inside_min = torch.tensor(crop_bbox[0], dtype=torch.float32, device="cuda")
        inside_max = torch.tensor(crop_bbox[1], dtype=torch.float32, device="cuda")
    inside_mask = ((inside_min <= voxel_model.grid_pts_xyz) & (voxel_model.grid_pts_xyz <= inside_max)).all(-1)
    inside_mask = inside_mask[voxel_model.vox_key].any(-1)
    inside_idx = torch.where(inside_mask)[0]

    # Infer iso value for level set
    vox_level = torch.tensor([voxel_model.outside_level + final_lv], device="cuda")
    vox_size = octree_utils.level_2_vox_size(voxel_model.scene_extent, vox_level).item()
    iso_alpha = torch.tensor(0.5, device="cuda")
    iso_density = activation_utils.alpha2density(iso_alpha, vox_size)
    iso = getattr(activation_utils, f"{voxel_model.density_mode}_inverse")(iso_density)
    sign = -1

    verts, faces = torch_marching_cubes_grid(
        grid_pts_val=sign * voxel_model._geo_grid_pts,
        grid_pts_xyz=voxel_model.grid_pts_xyz,
        vox_key=voxel_model.vox_key[inside_idx],
        iso=sign * iso)
    mesh = trimesh.Trimesh(verts.cpu().numpy(), faces.cpu().numpy())
    return mesh


def colorize_pts(args, pts, data_pack):
    cloest_color = torch.full([len(pts), 3], 0.5, dtype=torch.float32, device="cuda")
    cloest_dist = torch.full([len(pts)], np.inf, dtype=torch.float32, device="cuda")

    cam_lst = data_pack.get_train_cameras()

    for cam in tqdm(cam_lst):

        render_pkg = voxel_model.render(cam, color_mode="sh0", output_depth=True, output_T=True)
        frame_color = render_pkg['color']
        if args.use_mean:
            frame_depth = render_pkg['raw_depth'][[0]]  # Use mean depth
        else:
            frame_depth = render_pkg['raw_depth'][[2]]  # Use median depth
        frame_alpha = 1 - render_pkg['raw_T']
        H, W = frame_depth.shape[-2:]

        # Project grid points to image
        pts_uv = cam.project(pts)

        # Filter points projected outside
        filter_idx = torch.where((pts_uv.abs() <= 1).all(-1))[0]
        valid_pts_idx = filter_idx
        valid_pts = pts[filter_idx]
        pts_uv = pts_uv[filter_idx]

        # Sample alpha and filter
        pts_frame_alpha = torch.nn.functional.grid_sample(
            frame_alpha.view(1,1,H,W),
            pts_uv.view(1,1,-1,2),
            mode='bilinear',
            align_corners=False).flatten()
        filter_idx = torch.where(pts_frame_alpha > args.alpha_thres)[0]
        valid_pts_idx = valid_pts_idx[filter_idx]
        valid_pts = valid_pts[filter_idx]
        pts_uv = pts_uv[filter_idx]

        # Compute projective sdf
        pts_frame_depth = torch.nn.functional.grid_sample(
            frame_depth.view(1,1,H,W),
            pts_uv.view(1,1,-1,2),
            mode='bilinear',
            align_corners=False).flatten()
        pts_depth = ((valid_pts - cam.position) @ cam.lookat)
        pts_dist = (pts_frame_depth - pts_depth).abs()

        filter_idx = torch.where(pts_dist < cloest_dist[valid_pts_idx])[0]
        valid_pts_idx = valid_pts_idx[filter_idx]
        pts_uv = pts_uv[filter_idx]
        pts_dist = pts_dist[filter_idx]
        pts_color = torch.nn.functional.grid_sample(
            frame_color[None],
            pts_uv.view(1,1,-1,2),
            mode='bilinear',
            align_corners=False).squeeze().T
        cloest_dist[valid_pts_idx] = pts_dist
        cloest_color[valid_pts_idx] = pts_color

    return cloest_color


if __name__ == "__main__":
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(
        description="Sparse voxels raster extract mesh.")
    parser.add_argument('model_path')
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--save_gpu", action='store_true')
    parser.add_argument("--overwrite_ss", default=None, type=float)
    parser.add_argument("--overwrite_n_samp_per_vox", default=None, type=str)

    # Output file name
    parser.add_argument("--mesh_fname", default=None, type=str)

    # Provide bounding box to extract mesh
    parser.add_argument("--bbox_path", default=None)
    parser.add_argument("--bbox_scale", default=1.0, type=float)
    
    # Algorithm
    parser.add_argument("--direct", action='store_true')
    parser.add_argument("--progressive", action='store_true')

    # Level to run MC
    parser.add_argument("--init_lv", default=8, type=int)
    parser.add_argument("--final_lv", default=10, type=int)

    # Truncation bandwidth in the unit of voxel size
    parser.add_argument("--bandwidth_vox", default=5.0, type=float)

    # TSDF fusion hyperparams
    parser.add_argument("--crop_border", default=0.01, type=float)
    parser.add_argument("--alpha_thres", default=0.5, type=float)

    # Other hyperparams
    parser.add_argument("--use_mean", action='store_true')
    parser.add_argument("--use_vert_color", action='store_true')
    parser.add_argument("--use_clean", action='store_true')
    parser.add_argument("--use_remesh", action='store_true')
    parser.add_argument("--remesh_len", default=-1, type=float)

    args = parser.parse_args()
    print("Rendering " + args.model_path)

    # Load config
    update_config(os.path.join(args.model_path, 'config.yaml'))
    cfg.model.model_path = args.model_path

    # Load data (not needed for --direct mode)
    data_pack = None
    if not args.direct:
        data_pack = DataPack(
            cfg_data=cfg.data,
            white_background=cfg.model.white_background,
            camera_params_only=True,
        )

    # Load model
    voxel_model = SparseVoxelModel(cfg_model=cfg.model)
    voxel_model.load_iteration(args.iteration)
    voxel_model.freeze_vox_geo()

    if args.overwrite_ss is not None:
        voxel_model.ss = args.overwrite_ss

    # Prepare output dir
    outdir = os.path.join(
        args.model_path, "mesh",
        f"iter{voxel_model.loaded_iter:06d}" if voxel_model.loaded_iter > 0 else "latest")
    os.makedirs(outdir, exist_ok=True)

    print(f'outdir: {outdir}')
    print(f'ss            =: {voxel_model.ss}')
    print(f'Voxel level distribution:')
    for lv, num in enumerate(voxel_model.octlevel.flatten().bincount().tolist()):
        if num > 0:
            size = octree_utils.level_2_vox_size(voxel_model.scene_extent, torch.tensor(lv)).item()
            percen = num / voxel_model.num_voxels * 100
            suffix = ""
            if lv == voxel_model.outside_level + args.final_lv:
                suffix = ">>> MC level <<<"
            elif lv == voxel_model.outside_level + args.init_lv and args.progressive:
                suffix = ">>> Init level <<<"
            elif lv > voxel_model.outside_level + args.final_lv and not args.progressive:
                suffix = ">>> Level clamped <<<"
            print(f'  level={lv:2d} (size={size:.6f}): {num:7d} ({percen:5.2f}%) {suffix}')

    # Read crop bbox
    if args.bbox_path:
        crop_bbox = np.loadtxt(args.bbox_path)
    else:
        crop_bbox = None

    # GOGO
    fname = 'mesh'
    eps_time = time.time()
    with torch.no_grad():
        if args.direct:
            fname += f'_lv{args.final_lv}_direct'
            mesh = direct_mc(args, voxel_model, args.final_lv, crop_bbox)
        elif args.progressive:
            fname += f'_lv{args.init_lv}-{args.final_lv}'
            mesh = extract_mesh_progressive(args, data_pack, voxel_model, args.init_lv, args.final_lv, crop_bbox)
        else:
            mesh = extract_mesh(args, data_pack, voxel_model, args.final_lv, crop_bbox)
            fname += f'_lv{args.final_lv}_adaptive'
    eps_time = time.time() - eps_time
    print(f"Extracted mesh in {eps_time:.3f} sec")

    if args.use_mean:
        fname += '_dmean'

    # Taking the biggest connected component
    if args.use_clean:
        fname += '_clean'
        print("Taking the biggest connected component")
        try:
            labels = trimesh.graph.connected_component_labels(mesh.face_adjacency)
            cc, cc_cnt = np.unique(labels, return_counts=True)
            cc_maxid = cc[cc_cnt.argmax()]
            mesh.update_faces(labels==cc_maxid)

            vmask = np.zeros([len(mesh.vertices)], dtype=bool)
            vmask[mesh.faces] = 1
            mesh.update_vertices(vmask)
        except:
            print("Failed to segment largest cc")

    # Remesh
    if args.use_remesh:
        from gpytoolbox import remesh_botsch
        avg_edge_len = mesh.edges_unique_length.mean()
        if args.remesh_len < 0:
            target_edge_len = min(avg_edge_len, voxel_model.inside_extent.item() / 1024)
        else:
            target_edge_len = args.remesh_len
        print(f"Remeshing: original avg_len={avg_edge_len}; target edge_len={target_edge_len}")
        try:
            eps_time = time.time()
            v, f = remesh_botsch(mesh.vertices, mesh.faces, i=5, h=target_edge_len)
            eps_time = time.time() - eps_time
            print(f"Remeshed in {eps_time:.3f} sec")
            mesh = trimesh.Trimesh(vertices=v, faces=f)
        except:
            print(f"Remesh failed.")

    # Colorize vertices
    # TODO: Unwrap and use high-res UV texture map
    verts_color = None
    if args.use_vert_color:
        if data_pack is None:
            print("Warning: --use_vert_color requires training data; skipped in --direct mode.")
        else:
            print("Colorizing vertices")
            with torch.no_grad():
                pts = torch.tensor(mesh.vertices, dtype=torch.float32, device="cuda")
                verts_color = colorize_pts(args, pts, data_pack)
                verts_color = verts_color.cpu().numpy()
            mesh = trimesh.Trimesh(mesh.vertices, mesh.faces, vertex_colors=verts_color)

    # Transform to world coordinate
    if data_pack is not None and data_pack.to_world_matrix is not None:
        mesh = mesh.apply_transform(data_pack.to_world_matrix)

    # Export mesh
    print(mesh)
    if args.mesh_fname is not None:
        fname = args.mesh_fname
    outpath = os.path.join(outdir, f'{fname}.ply')
    mesh.export(outpath)
    print('Save to', outpath)

    print("done!")
