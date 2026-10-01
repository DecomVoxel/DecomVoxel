# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch
from . import _C


def voxel_order_rank(octree_paths):
    # Compute the eight possible voxel rendering orders.
    order_ranks = _C.voxel_order_rank(octree_paths)
    return order_ranks


def is_in_cone(pts, cam):
    assert torch.is_tensor(pts)
    assert pts.device == cam.w2c.device
    assert len(pts.shape) == 2
    assert pts.shape[1] == 3
    return _C.is_in_cone(
        cam.tanfovx,
        cam.tanfovy,
        cam.near,
        cam.w2c,
        pts)


def compute_rd(width, height, cx, cy, tanfovx, tanfovy, c2w_matrix):
    assert torch.is_tensor(c2w_matrix)
    return _C.compute_rd(width, height, cx, cy, tanfovx, tanfovy, c2w_matrix)


def depth2pts(width, height, cx, cy, tanfovx, tanfovy, c2w_matrix, depth):
    assert torch.is_tensor(c2w_matrix)
    assert depth.device == c2w_matrix.device
    assert depth.numel() == width * height
    return _C.depth2pts(width, height, cx, cy, tanfovx, tanfovy, c2w_matrix, depth)


def ijk_2_octpath(ijk, octlevel):
    assert torch.is_tensor(ijk) and torch.is_tensor(octlevel)
    assert len(ijk.shape) == 2 and ijk.shape[1] == 3
    assert ijk.numel() == octlevel.numel() * 3
    assert ijk.dtype == torch.int64
    assert octlevel.dtype == torch.int8
    return _C.ijk_2_octpath(ijk, octlevel)


def octpath_2_ijk(octpath, octlevel):
    assert torch.is_tensor(octpath) and torch.is_tensor(octlevel)
    assert octpath.numel() == octlevel.numel()
    assert octpath.dtype == torch.int64
    assert octlevel.dtype == torch.int8
    return _C.octpath_2_ijk(octpath, octlevel)
