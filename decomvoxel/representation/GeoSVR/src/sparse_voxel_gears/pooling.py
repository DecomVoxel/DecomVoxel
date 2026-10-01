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

from src.utils import octree_utils


class SVPooling:

    def pooling_to_level(self, max_level, octpath=None, octlevel=None):
        octpath = self.octpath if octpath is None else octpath
        octlevel = self.octlevel if octlevel is None else octlevel

        num_bit_to_mask = 3 * max(0, svraster_cuda.meta.MAX_NUM_LEVELS - max_level)
        octpath = (octpath >> num_bit_to_mask) << num_bit_to_mask
        octlevel = octlevel.clamp_max(max_level)
        octpack, invmap = torch.stack([octpath, octlevel]).unique(sorted=True, dim=1, return_inverse=True)
        octpath, octlevel = octpack
        octlevel = octlevel.to(torch.int8)
        
        vox_center, vox_size = octree_utils.octpath_decoding(
            octpath, octlevel, self.scene_center, self.scene_extent)

        return dict(
            invmap=invmap,
            octpath=octpath,
            octlevel=octlevel,
            vox_center=vox_center,
            vox_size=vox_size,
        )

    def pooling_to_rate(self, cameras, max_rate, octpath=None, octlevel=None):
        octpath = self.octpath.clone() if octpath is None else octpath
        octlevel = self.octlevel.clone() if octlevel is None else octlevel
        invmap = torch.arange(len(octpath), device="cuda")

        for _ in range(svraster_cuda.meta.MAX_NUM_LEVELS):
            vox_center, vox_size = octree_utils.octpath_decoding(octpath, octlevel, self.scene_center, self.scene_extent)
            samp_rate = svraster_cuda.renderer.mark_max_samp_rate(cameras, octpath, vox_center, vox_size)
            pool_mask = (samp_rate < max_rate) & (octlevel.squeeze(1) > 1)
            if pool_mask.sum() == 0:
                break
            octlevel[pool_mask] = octlevel[pool_mask] - 1
            num_bit_to_mask = 3 * (svraster_cuda.meta.MAX_NUM_LEVELS - octlevel[pool_mask])
            octpath[pool_mask] = octpath[pool_mask] >> num_bit_to_mask << num_bit_to_mask

            octpack, cur_invmap = torch.stack([octpath, octlevel]).unique(sorted=True, dim=1, return_inverse=True)
            octpath, octlevel = octpack
            octlevel = octlevel.to(torch.int8)
            invmap = cur_invmap[invmap]

        vox_center, vox_size = octree_utils.octpath_decoding(
            octpath, octlevel, self.scene_center, self.scene_extent)

        return dict(
            invmap=invmap,
            octpath=octpath,
            octlevel=octlevel,
            vox_center=vox_center,
            vox_size=vox_size,
        )
