# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch
from . import _C
from .meta import VoxGeoModes, DensityModes, CamModes

from typing import NamedTuple


class RasterSettings(NamedTuple):
    color_mode: str
    vox_geo_mode: str
    density_mode: str
    image_width: int
    image_height: int
    tanfovx: float
    tanfovy: float
    cx: float
    cy: float
    w2c_matrix: torch.Tensor
    c2w_matrix: torch.Tensor
    background: torch.Tensor
    cam_mode: str = "persp"
    near: float = 0.02
    need_depth: bool = False
    need_normal: bool = False
    track_max_w: bool = False
    lambda_R_concen: float = 0
    lambda_ascending: float = 0
    lambda_scaling_penalty: float = 0
    min_voxel_size: float = 0
    lambda_dist: float = 0
    # Optional gt color for color concnetration loss in backward pass.
    gt_color: torch.Tensor = torch.empty(0)
    # Optional voxel feature to render. Slower if there is.
    vox_feats: torch.Tensor = torch.empty([0, 0])
    debug: bool = False


def rasterize_voxels(
        raster_settings: RasterSettings,
        octree_paths: torch.Tensor,
        vox_centers: torch.Tensor,
        vox_lengths: torch.Tensor,
        vox_fn,
    ):

    # Some input checking
    if not isinstance(raster_settings, RasterSettings):
        raise Exception("Expect RasterSettings as first argument.")
    if raster_settings.vox_geo_mode not in VoxGeoModes:
        raise NotImplementedError("Unknown voxel geo mode.")
    if raster_settings.density_mode not in DensityModes:
        raise NotImplementedError("Unknown density mode.")
    if raster_settings.cam_mode not in CamModes:
        raise NotImplementedError("Unknow camera mode.")

    N = octree_paths.numel()
    device = octree_paths.device

    if vox_centers.shape[0] != N or vox_lengths.numel() != N:
        raise Exception("Size mismatched.")
    if len(vox_centers.shape) != 2 or vox_centers.shape[1] != 3:
        raise Exception("Expect vox_centers in shape [N, 3].")
    if raster_settings.w2c_matrix.device != device or \
            raster_settings.c2w_matrix.device != device or \
            raster_settings.background.device != device or \
            vox_centers.device != device or \
            vox_lengths.device != device:
        raise Exception("Device mismatch.")
    if raster_settings.vox_feats.requires_grad:
        raise NotImplementedError

    # Preprocess octree
    n_duplicates, geomBuffer = _C.rasterize_preprocess(
        raster_settings.image_width,
        raster_settings.image_height,
        raster_settings.tanfovx,
        raster_settings.tanfovy,
        raster_settings.cx,
        raster_settings.cy,
        raster_settings.w2c_matrix,
        raster_settings.c2w_matrix,
        CamModes[raster_settings.cam_mode],
        raster_settings.near,

        octree_paths,
        vox_centers,
        vox_lengths,

        raster_settings.debug,
    )
    in_frusts_idx = torch.where(n_duplicates > 0)[0]

    # Forward voxel parameters
    cam_pos = raster_settings.c2w_matrix[:3, 3]
    vox_params = vox_fn(in_frusts_idx, cam_pos, raster_settings.color_mode)
    geos = vox_params['geos']
    rgbs = vox_params['rgbs']
    subdiv_p = vox_params['subdiv_p']

    # Some voxel parameters checking
    if geos.shape != (N, 8):
        raise Exception(f"Expect geos in ({N}, 8) but got", geos.shape)
    if rgbs.shape[0] != N:
        raise Exception(f"Expect rgbs in ({N}, 3) but got", rgbs.shape)
    if subdiv_p.shape[0] != N:
        raise Exception(f"Expect subdiv_p in ({N}, 1) but got", subdiv_p.shape)

    if geos.device != device:
        raise Exception("Device mismatch: geos.")
    if rgbs.device != device:
        raise Exception("Device mismatch: rgbs.")
    if subdiv_p.device != device:
        raise Exception("Device mismatch: subdiv_p.")

    # Some checking for regularizations
    if raster_settings.lambda_R_concen > 0:
        if len(raster_settings.gt_color.shape) != 3 or \
                raster_settings.gt_color.shape[0] != 3 or \
                raster_settings.gt_color.shape[1] != raster_settings.image_height or \
                raster_settings.gt_color.shape[2] != raster_settings.image_width:
            raise Exception("Except gt_color in shape of [3, H, W]")
        if raster_settings.gt_color.device != device:
            raise Exception("Device mismatch.")

    # Involk differentiable voxels rasterization.
    return _RasterizeVoxels.apply(
        raster_settings,
        geomBuffer,
        octree_paths,
        vox_centers,
        vox_lengths,
        geos,
        rgbs,
        raster_settings.vox_feats,
        subdiv_p,
    )


class _RasterizeVoxels(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        raster_settings,
        geomBuffer,
        octree_paths,
        vox_centers,
        vox_lengths,
        geos,
        rgbs,
        vox_feats,
        subdiv_p,
    ):

        need_distortion = raster_settings.lambda_dist > 0

        args = (
            VoxGeoModes[raster_settings.vox_geo_mode],
            DensityModes[raster_settings.density_mode],
            raster_settings.image_width,
            raster_settings.image_height,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.cx,
            raster_settings.cy,
            raster_settings.w2c_matrix,
            raster_settings.c2w_matrix,
            raster_settings.background,
            CamModes[raster_settings.cam_mode],
            raster_settings.need_depth,
            need_distortion,
            raster_settings.need_normal,
            raster_settings.track_max_w,

            octree_paths,
            vox_centers,
            vox_lengths,
            geos,
            rgbs,
            vox_feats,

            geomBuffer,

            raster_settings.debug,
        )

        num_rendered, binningBuffer, imgBuffer, out_color, out_depth, out_normal, out_T, max_w, out_feat = _C.rasterize_voxels(*args)

        # In case you want some advanced debuging here
        # ranges, tile_last, n_contrib = _C.unpack_ImageState(raster_settings.image_width, raster_settings.image_height, imgBuffer)
        # ranges_x = ranges[:, :, 0]
        # ranges_y = ranges[:, :, 1]
        # tile_nonempty = (ranges_y - ranges_x > 0)
        # print(ranges_y[tile_nonempty] - tile_last[tile_nonempty])
        # import pdb; pdb.set_trace()

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(
            octree_paths, vox_centers, vox_lengths,
            geos, rgbs,
            geomBuffer, binningBuffer, imgBuffer, out_T, out_depth, out_normal)
        ctx.mark_non_differentiable(max_w)
        return out_color, out_depth, out_normal, out_T, max_w, out_feat

    @staticmethod
    def backward(ctx, dL_dout_color, dL_dout_depth, dL_dout_normal, dL_dout_T, dL_dmax_w, dL_dout_feat):
        # Restore necessary values from context
        raster_settings = ctx.raster_settings
        num_rendered = ctx.num_rendered
        octree_paths, vox_centers, vox_lengths, \
            geos, rgbs, \
            geomBuffer, binningBuffer, imgBuffer, out_T, out_depth, out_normal = ctx.saved_tensors

        if raster_settings.cam_mode == "ortho":
            raise NotImplementedError("Backward pass of orthographic projection is not implemented.")

        args = (
            num_rendered,
            VoxGeoModes[raster_settings.vox_geo_mode],
            DensityModes[raster_settings.density_mode],
            raster_settings.image_width,
            raster_settings.image_height,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.cx,
            raster_settings.cy,
            raster_settings.w2c_matrix,
            raster_settings.c2w_matrix,
            # CamModes[raster_settings.cam_mode],
            raster_settings.background,

            octree_paths,
            vox_centers,
            vox_lengths,
            geos,
            rgbs,

            geomBuffer,
            binningBuffer,
            imgBuffer,
            out_T,

            dL_dout_color,
            dL_dout_depth,
            dL_dout_normal,
            dL_dout_T,

            raster_settings.lambda_R_concen,
            raster_settings.gt_color,
            raster_settings.lambda_ascending,
            raster_settings.lambda_scaling_penalty,
            raster_settings.min_voxel_size,
            raster_settings.lambda_dist,
            raster_settings.need_depth,
            raster_settings.need_normal,
            out_depth,
            out_normal,

            raster_settings.debug,
        )

        dL_dgeos, dL_drgbs, subdiv_p_bw = _C.rasterize_voxels_backward(*args)
        dL_drgbs = dL_drgbs.view(rgbs.shape).contiguous()

        grads = (
            None, # => raster_settings
            None, # => geomBuffer
            None, # => octree_paths
            None, # => vox_centers
            None, # => vox_lengths
            dL_dgeos, # => geos
            dL_drgbs, # => rgbs
            None, # => vox_feats
            subdiv_p_bw, # => subdivision priority
        )

        return grads


class SH_eval(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        active_sh_degree,
        idx,
        vox_centers, # Use dir to vox center
        cam_pos,
        viewdir,     # Use given dir
        sh0,
        shs,
    ):

        if torch.is_tensor(vox_centers) and vox_centers.requires_grad:
            raise NotImplementedError
        if torch.is_tensor(cam_pos) and cam_pos.requires_grad:
            raise NotImplementedError
        if torch.is_tensor(viewdir) and viewdir.requires_grad:
            raise NotImplementedError

        if idx is None:
            idx = torch.empty(0, dtype=torch.int64)

        if viewdir is not None:
            vox_centers = viewdir
            cam_pos = torch.zeros_like(cam_pos)

        rgbs = _C.sh_compute(
            active_sh_degree,
            idx,
            vox_centers,
            cam_pos,
            sh0,
            shs,
        )

        ctx.active_sh_degree = active_sh_degree
        ctx.M = 1 + shs.shape[1]
        ctx.save_for_backward(idx, vox_centers, cam_pos, rgbs)
        return rgbs

    @staticmethod
    def backward(ctx, dL_drgbs):
        # Restore necessary values from context
        idx, vox_centers, cam_pos, rgbs = ctx.saved_tensors
        dL_dsh0, dL_dshs = _C.sh_compute_bw(
            ctx.active_sh_degree,
            ctx.M,
            idx,
            vox_centers,
            cam_pos,
            rgbs,
            dL_drgbs,
        )

        grads = (
            None, # => active_sh_degree
            None, # => idx
            None, # => vox_centers
            None, # => cam_pos
            None, # => viewdir
            dL_dsh0,
            dL_dshs,
        )

        return grads


class GatherGeoParams(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        vox_geo_mode,
        vox_key,
        vox_size_inv,
        care_idx,
        grid_pts,
    ):
        assert len(vox_key.shape) == 2 and vox_key.shape[1] == 8
        assert len(care_idx.shape) == 1
        assert grid_pts.shape[0] == grid_pts.numel()

        if vox_geo_mode.startswith("triinterp"):
            geo_params = _C.gather_triinterp_geo_params(vox_key, care_idx, grid_pts)
        else:
            raise NotImplementedError(f"Unknow vox_geo_mode: {vox_geo_mode}")

        ctx.vox_geo_mode = vox_geo_mode
        ctx.num_grid_pts = grid_pts.numel()
        ctx.save_for_backward(vox_key, vox_size_inv, care_idx)
        return geo_params

    @staticmethod
    def backward(ctx, dL_dgeo_params):
        # Restore necessary values from context
        vox_geo_mode = ctx.vox_geo_mode
        num_grid_pts = ctx.num_grid_pts
        vox_key, vox_size_inv, care_idx = ctx.saved_tensors

        if vox_geo_mode.startswith("triinterp"):
            dL_dgrid_pts = _C.gather_triinterp_geo_params_bw(vox_key, care_idx, num_grid_pts, dL_dgeo_params)
        else:
            raise NotImplementedError(f"Unknow vox_geo_mode: {vox_geo_mode}")

        return None, None, None, None, dL_dgrid_pts


class GatherFeatParams(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        vox_key,
        care_idx,
        grid_pts,
    ):
        assert len(vox_key.shape) == 2 and vox_key.shape[1] == 8
        assert len(care_idx.shape) == 1
        assert len(grid_pts.shape) == 2

        feat_params = _C.gather_triinterp_feat_params(vox_key, care_idx, grid_pts)

        ctx.num_grid_pts = len(grid_pts)
        ctx.save_for_backward(vox_key, care_idx)
        return feat_params

    @staticmethod
    def backward(ctx, dL_dfeat_params):
        # Restore necessary values from context
        num_grid_pts = ctx.num_grid_pts
        vox_key, care_idx = ctx.saved_tensors

        dL_dgrid_pts = _C.gather_triinterp_feat_params_bw(vox_key, care_idx, num_grid_pts, dL_dfeat_params)

        return None, None, dL_dgrid_pts


def mark_n_duplicates(
        image_width, image_height,
        tanfovx, tanfovy,
        cx, cy,
        w2c_matrix, c2w_matrix, near,
        octree_paths, vox_centers, vox_lengths,
        cam_mode="persp",
        return_buffer=False,
        debug=False):

    n_duplicates, geomBuffer = _C.rasterize_preprocess(
        image_width,
        image_height,
        tanfovx,
        tanfovy,
        cx,
        cy,
        w2c_matrix,
        c2w_matrix,
        CamModes[cam_mode],
        near,

        octree_paths,
        vox_centers,
        vox_lengths,

        debug,
    )
    if return_buffer:
        return n_duplicates, geomBuffer
    return n_duplicates


def mark_min_samp_rate(cameras, octree_paths, vox_centers, vox_lengths, near=0.02):
    MAX_RATE = 1e30
    min_samp_rate = torch.full([len(octree_paths)], MAX_RATE, dtype=torch.float32, device="cuda")
    for cam in cameras:
        n_duplicates = mark_n_duplicates(
            image_width=cam.image_width, image_height=cam.image_height,
            tanfovx=cam.tanfovx, tanfovy=cam.tanfovy,
            cx=cam.cx, cy=cam.cy,
            w2c_matrix=cam.w2c, c2w_matrix=cam.c2w, near=near,
            octree_paths=octree_paths, vox_centers=vox_centers, vox_lengths=vox_lengths)
        zdist = ((vox_centers - cam.position) * cam.lookat).sum(-1)
        vis_idx = torch.where((n_duplicates > 0) & (zdist > near))[0]
        zdist = zdist[vis_idx]
        samp_interval = zdist * cam.pix_size
        samp_rate = vox_lengths.squeeze(1)[vis_idx] / samp_interval
        min_samp_rate[vis_idx] = torch.minimum(min_samp_rate[vis_idx], samp_rate)
    min_samp_rate[min_samp_rate >= MAX_RATE] = 0
    return min_samp_rate


def mark_max_samp_rate(cameras, octree_paths, vox_centers, vox_lengths, near=0.02):
    max_samp_rate = torch.zeros([len(octree_paths)], dtype=torch.float32, device="cuda")
    for cam in cameras:
        n_duplicates = mark_n_duplicates(
            image_width=cam.image_width, image_height=cam.image_height,
            tanfovx=cam.tanfovx, tanfovy=cam.tanfovy,
            cx=cam.cx, cy=cam.cy,
            w2c_matrix=cam.w2c, c2w_matrix=cam.c2w, near=near,
            octree_paths=octree_paths, vox_centers=vox_centers, vox_lengths=vox_lengths)
        zdist = ((vox_centers - cam.position) * cam.lookat).sum(-1)
        vis_idx = torch.where((n_duplicates > 0) & (zdist > near))[0]
        zdist = zdist[vis_idx]
        samp_interval = zdist * cam.pix_size
        samp_rate = vox_lengths.squeeze(1)[vis_idx] / samp_interval
        max_samp_rate[vis_idx] = torch.maximum(max_samp_rate[vis_idx], samp_rate)
    return max_samp_rate


def mark_avg_samp_rate(cameras, octree_paths, vox_centers, vox_lengths, near=0.02):
    total_samp_rate = torch.zeros([len(octree_paths)], dtype=torch.float32, device="cuda")
    total_cnt = torch.zeros([len(octree_paths)], dtype=torch.float32, device="cuda")
    for cam in cameras:
        n_duplicates = mark_n_duplicates(
            image_width=cam.image_width, image_height=cam.image_height,
            tanfovx=cam.tanfovx, tanfovy=cam.tanfovy,
            cx=cam.cx, cy=cam.cy,
            w2c_matrix=cam.w2c, c2w_matrix=cam.c2w, near=near,
            octree_paths=octree_paths, vox_centers=vox_centers, vox_lengths=vox_lengths)
        zdist = ((vox_centers - cam.position) * cam.lookat).sum(-1)
        vis_idx = torch.where((n_duplicates > 0) & (zdist > near))[0]
        zdist = zdist[vis_idx]
        samp_interval = zdist * cam.pix_size
        samp_rate = vox_lengths.squeeze(1)[vis_idx] / samp_interval
        total_samp_rate[vis_idx] += samp_rate
        total_cnt[vis_idx] += 1
    avg_samp_rate = total_samp_rate / total_cnt.clamp_min(1)
    return avg_samp_rate


def mark_near(cameras, octree_paths, vox_centers, vox_lengths, near=0.2):
    is_near = torch.zeros([len(octree_paths)], dtype=torch.bool, device="cuda")
    for cam in cameras:
        n_duplicates = mark_n_duplicates(
            image_width=cam.image_width, image_height=cam.image_height,
            tanfovx=cam.tanfovx, tanfovy=cam.tanfovy,
            cx=cam.cx, cy=cam.cy,
            w2c_matrix=cam.w2c, c2w_matrix=cam.c2w, near=near,
            octree_paths=octree_paths, vox_centers=vox_centers, vox_lengths=vox_lengths)
        vis_idx = torch.where(n_duplicates > 0)[0]
        zdist = ((vox_centers[vis_idx] - cam.position) * cam.lookat).sum(-1)
        is_near[vis_idx] |= (zdist <= near + vox_lengths.squeeze(1)[vis_idx])
    return is_near
