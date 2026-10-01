# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch
from . import _C


def total_variation(grid_pts, vox_key, weight, vox_size_inv, no_tv_s, tv_sparse, grid_pts_grad):
    assert grid_pts.shape == grid_pts_grad.shape
    assert len(vox_key.shape) == 2 and vox_key.shape[1] == 8
    assert vox_key.shape[0] == vox_size_inv.numel()
    _C.total_variation_bw(grid_pts, vox_key, weight, vox_size_inv, no_tv_s, tv_sparse, grid_pts_grad)
