# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np

import torch
import svraster_cuda

from src.utils.activation_utils import rgb2shzero
from src.utils import octree_utils

class SVConstructor:

    def model_init(self, bounding, cfg_init, cameras=None):
        # Define scene bound
        center = (bounding[0] + bounding[1]) * 0.5
        radius = (bounding[1] - bounding[0]) * 0.5
        self.scene_center = torch.tensor(center, dtype=torch.float32, device="cuda")
        self.inside_extent = 2 * torch.tensor(max(radius), dtype=torch.float32, device="cuda")
        self.scene_extent = self.inside_extent * (2 ** self.outside_level)

        # Init voxel layout.
        # The world is seperated into inside (main foreground) and outside (background) regions.
        init_inside_level = self.outside_level + cfg_init.init_n_level
        in_path, in_level, in_samp_rate = octlayout_inside_uniform(
            voxel_model=self,
            n_level=cfg_init.init_n_level,
            cameras=cameras,
            filter_zero_visiblity=True,
            filter_near=-1)

        if cfg_init.outside_mode == "none" or self.outside_level == 0:
            # Object centric bounded scenes
            ou_path = torch.empty([0, 1], dtype=in_path.dtype, device="cuda")
            ou_level = torch.empty([0, 1], dtype=in_level.dtype, device="cuda")
        elif cfg_init.outside_mode.startswith("uniform"):
            n_level_in_shell = int(cfg_init.outside_mode[7:])
            ou_path, ou_level, ou_avg_max_rate = octlayout_outside_uniform(
                voxel_model=self,
                n_level=n_level_in_shell,
                cameras=cameras,
                filter_zero_visiblity=True,
                filter_near=-1)
        elif cfg_init.outside_mode == "heuristic":
            min_num = len(in_path) * cfg_init.init_out_ratio
            ou_path, ou_level, ou_avg_max_rate = octlayout_outside_heuristic(
                voxel_model=self,
                cameras=cameras,
                min_num=min_num,
                max_level=init_inside_level,
                filter_near=-1)
        else:
            raise NotImplementedError
        self.octpath = torch.cat([ou_path, in_path])
        self.octlevel = torch.cat([ou_level, in_level])

        if cfg_init.aabb_crop:
            aabb_radius = torch.tensor(radius, dtype=torch.float32, device="cuda")
            aabb_min = self.scene_center - aabb_radius
            aabb_max = self.scene_center + aabb_radius
            self.octpath, self.octlevel = aabb_crop(
                octpath=self.octpath, octlevel=self.octlevel,
                scene_center=self.scene_center, scene_extent=self.scene_extent,
                aabb_min=aabb_min, aabb_max=aabb_max)

        self.vox_center, self.vox_size = octree_utils.octpath_decoding(
            self.octpath, self.octlevel, self.scene_center, self.scene_extent)

        # Show statistic
        inside_min = self.scene_center - 0.5 * self.inside_extent
        inside_max = self.scene_center + 0.5 * self.inside_extent
        print('Inside bound min:', inside_min.tolist())
        print('Inside bound max:', inside_max.tolist())
        if cameras is not None:
            max_rate = svraster_cuda.renderer.mark_max_samp_rate(
                cameras, self.octpath, self.vox_center, self.vox_size)
            min_rate = svraster_cuda.renderer.mark_min_samp_rate(
                cameras, self.octpath, self.vox_center, self.vox_size)
            avg_rate = svraster_cuda.renderer.mark_avg_samp_rate(
                cameras, self.octpath, self.vox_center, self.vox_size)
            inside_mask = ((inside_min <= self.vox_center) & (self.vox_center <= inside_max)).all(-1)
            in_avg_max_rate = max_rate[inside_mask].float().mean().item()
            ou_avg_max_rate = max_rate[~inside_mask].float().mean().item()
            in_avg_min_rate = min_rate[inside_mask].float().mean().item()
            ou_avg_min_rate = min_rate[~inside_mask].float().mean().item()
            in_avg_avg_rate = avg_rate[inside_mask].float().mean().item()
            ou_avg_avg_rate = avg_rate[~inside_mask].float().mean().item()
            print(f'Inside layout : {inside_mask.sum():8d} voxels, '
                  f'{in_avg_avg_rate:6.0f}/{in_avg_min_rate:6.0f}/{in_avg_max_rate:6.0f} '
                  f'avg avg/min/max samp-rate.')
            print(f'Outside layout: {(~inside_mask).sum():8d} voxels, '
                  f'{ou_avg_avg_rate:6.0f}/{ou_avg_min_rate:6.0f}/{ou_avg_max_rate:6.0f} '
                  f'avg avg/min/max samp-rate.')

        N = len(self.octpath)
        print("Number of points at initialisation:", N)

        self.grid_pts_key, self.vox_key = octree_utils.build_grid_pts_link(self.octpath, self.octlevel)
        print("grid_pts size at initialisation:", self.num_grid_pts)

        self.active_sh_degree = min(cfg_init.sh_degree_init, self.max_sh_degree)

        _geo_grid_pts = torch.full([self.num_grid_pts, 1], cfg_init.geo_init, dtype=torch.float32, device="cuda")

        _rgb = torch.full([N, 3], cfg_init.sh0_init, dtype=torch.float32, device="cuda")
        _shs = torch.full([N, (self.max_sh_degree+1)**2 - 1, 3], cfg_init.shs_init, dtype=torch.float32, device="cuda")

        _sh0 = rgb2shzero(_rgb)

        _subdiv_p = torch.full([N, 1], 1.0, dtype=torch.float32, device="cuda")

        self._geo_grid_pts = _geo_grid_pts.contiguous().requires_grad_()
        self._sh0 = _sh0.contiguous().requires_grad_()
        self._shs = _shs.contiguous().requires_grad_()
        self._subdiv_p = _subdiv_p.contiguous().requires_grad_()

        self.subdiv_meta = torch.zeros([N, 1], dtype=torch.float32, device="cuda")
        self.bg_color = torch.tensor(
            [1, 1, 1] if self.white_background else [0, 0, 0],
            dtype=torch.float32, device="cuda")


#################################################
# Initial Octree layout construction
#################################################
def octlayout_filtering(octpath, octlevel, voxel_model, cameras=None, samp_mode="max", filter_zero_visiblity=True, filter_near=-1):
    vox_center, vox_size = octree_utils.octpath_decoding(
        octpath, octlevel,
        voxel_model.scene_center, voxel_model.scene_extent)
    if cameras is not None:
        if samp_mode == "avg":
            rate = svraster_cuda.renderer.mark_avg_samp_rate(
                cameras, octpath, vox_center, vox_size)
        elif samp_mode == "min":
            rate = svraster_cuda.renderer.mark_min_samp_rate(
                cameras, octpath, vox_center, vox_size)
        elif samp_mode == "max":
            rate = svraster_cuda.renderer.mark_max_samp_rate(
                cameras, octpath, vox_center, vox_size)
        else:
            raise NotImplementedError
    else:
        rate = torch.ones([len(octpath)], device="cuda")

    # Filtering
    kept_mask = torch.ones([len(octpath)], dtype=torch.bool, device="cuda")
    if filter_zero_visiblity:
        kept_mask &= (rate > 0)
    if filter_near > 0:
        is_near = svraster_cuda.renderer.mark_near(
            cameras, octpath, vox_center, vox_size, near=filter_near)
        kept_mask &= (~is_near)
    kept_idx = torch.where(kept_mask)[0]
    octpath = octpath[kept_idx]
    octlevel = octlevel[kept_idx]
    avg_rate = rate[kept_idx].float().mean().item()
    return octpath, octlevel, avg_rate


def octlayout_inside_uniform(voxel_model, n_level, cameras=None, samp_mode="max", filter_zero_visiblity=True, filter_near=-1):
    octpath, octlevel = octree_utils.gen_octpath_dense(
        outside_level=voxel_model.outside_level,
        n_level_inside=n_level)

    octpath, octlevel, avg_rate = octlayout_filtering(
        octpath=octpath,
        octlevel=octlevel,
        voxel_model=voxel_model,
        cameras=cameras,
        samp_mode=samp_mode,
        filter_zero_visiblity=filter_zero_visiblity,
        filter_near=filter_near)
    return octpath, octlevel, avg_rate


def octlayout_outside_uniform(voxel_model, n_level, cameras=None, samp_mode="max", filter_zero_visiblity=True, filter_near=-1):
    octpath = []
    octlevel = []
    for lv in range(1, 1+voxel_model.outside_level):
        path, lv = octree_utils.gen_octpath_shell(
            shell_level=lv,
            n_level_inside=n_level)
        octpath.append(path)
        octlevel.append(lv)
    octpath = torch.cat(octpath)
    octlevel = torch.cat(octlevel)

    octpath, octlevel, avg_rate = octlayout_filtering(
        octpath=octpath,
        octlevel=octlevel,
        voxel_model=voxel_model,
        cameras=cameras,
        samp_mode=samp_mode,
        filter_zero_visiblity=filter_zero_visiblity,
        filter_near=filter_near)
    return octpath, octlevel, avg_rate


def octlayout_outside_heuristic(voxel_model, cameras, min_num, max_level, samp_mode="max", filter_near=-1):

    assert cameras is not None, "Cameras should provided in this mode."

    mark_samp_rate = None
    if samp_mode == "avg":
        mark_samp_rate = svraster_cuda.renderer.mark_avg_samp_rate
    elif samp_mode == "min":
        mark_samp_rate = svraster_cuda.renderer.mark_min_samp_rate
    elif samp_mode == "max":
        mark_samp_rate = svraster_cuda.renderer.mark_max_samp_rate
    else:
        raise NotImplementedError

    # Init by adding one sub-level in each shell level
    octpath = []
    octlevel = []
    for lv in range(1, 1+voxel_model.outside_level):
        path, lv = octree_utils.gen_octpath_shell(
            shell_level=lv,
            n_level_inside=1)
        octpath.append(path)
        octlevel.append(lv)
    octpath = torch.cat(octpath)
    octlevel = torch.cat(octlevel)

    # Check visibility and how many init subdivision
    while True:
        vox_center, vox_size = octree_utils.octpath_decoding(
            octpath, octlevel, voxel_model.scene_center, voxel_model.scene_extent)
        samp_rate = mark_samp_rate(cameras, octpath, vox_center, vox_size)
        kept_idx = torch.where((samp_rate > 0))[0]
        octpath = octpath[kept_idx]
        octlevel = octlevel[kept_idx]
        octlevel_mask = (octlevel.squeeze(1) < max_level)
        samp_rate = samp_rate[kept_idx] * octlevel_mask
        vox_size = vox_size[kept_idx]
        still_need_n = (min_num - len(octpath)) // 7
        still_need_n = min(len(octpath), round(still_need_n))
        if still_need_n <= 0:
            break
        rank = samp_rate * (octlevel.squeeze(1) < svraster_cuda.meta.MAX_NUM_LEVELS)
        subdiv_mask = (rank >= rank.sort().values[-still_need_n])
        subdiv_mask &= (octlevel.squeeze(1) < svraster_cuda.meta.MAX_NUM_LEVELS)
        subdiv_mask &= octlevel_mask
        samp_rate *= subdiv_mask
        subdiv_mask &= (samp_rate >= samp_rate.quantile(0.9))  # Subdivide only 10% each iteration
        if subdiv_mask.sum() == 0:
            break
        octpath_children, octlevel_children = octree_utils.gen_children(
            octpath[subdiv_mask], octlevel[subdiv_mask])
        octpath = torch.cat([octpath[~subdiv_mask], octpath_children])
        octlevel = torch.cat([octlevel[~subdiv_mask], octlevel_children])

    octpath, octlevel, avg_rate = octlayout_filtering(
        octpath=octpath,
        octlevel=octlevel,
        voxel_model=voxel_model,
        cameras=cameras,
        samp_mode=samp_mode,
        filter_zero_visiblity=True,
        filter_near=filter_near)
    return octpath, octlevel, avg_rate


def aabb_crop(octpath, octlevel, scene_center, scene_extent, aabb_min, aabb_max):
    vox_center, vox_size = octree_utils.octpath_decoding(octpath, octlevel, scene_center, scene_extent)
    vox_radius = 0.5 * vox_size
    vox_min = vox_center - vox_radius
    vox_max = vox_center + vox_radius
    aabb_center = (aabb_max + aabb_min) * 0.5
    aabb_radius = (aabb_max - aabb_min) * 0.5
    isin_aabb = torch.zeros([len(vox_size)], dtype=torch.bool, device="cuda")
    for i in range(8):
        shift = torch.tensor([(i&1)>0, (i&2)>0, (i&4)>0], dtype=torch.float32, device="cuda")
        vox_pt = vox_center + shift * vox_radius
        aabb_pt = aabb_center + shift * aabb_radius
        isin_aabb |= ((aabb_min <= vox_pt) & (vox_pt <= aabb_max)).all(-1)
        isin_aabb |= ((vox_min <= aabb_pt) & (aabb_pt <= vox_max)).all(-1)

    isin_idx = torch.where(isin_aabb)[0]
    octpath = octpath[isin_idx]
    octlevel = octlevel[isin_idx]
    return octpath, octlevel
