/*************************************************************************
Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto.  Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*************************************************************************/

#ifndef UTILS_H_INCLUDED
#define UTILS_H_INCLUDED

#include <torch/extension.h>

namespace UTILS {

torch::Tensor is_in_cone(
    const float tanfovx,
    const float tanfovy,
    const float near,
    const torch::Tensor& w2c_matrix,
    const torch::Tensor& pts);

torch::Tensor compute_rd(
    const int width, const int height,
    const float cx, const float cy,
    const float tanfovx, const float tanfovy,
    const torch::Tensor& c2w_matrix);

torch::Tensor depth2pts(
    const int width, const int height,
    const float cx, const float cy,
    const float tanfovx, const float tanfovy,
    const torch::Tensor& c2w_matrix,
    const torch::Tensor& depth);

torch::Tensor voxel_order_rank(
    const torch::Tensor& octree_paths);

torch::Tensor ijk_2_octpath(const torch::Tensor& ijk, const torch::Tensor& octlevel);

torch::Tensor octpath_2_ijk(const torch::Tensor& octpath, const torch::Tensor& octlevel);

}

#endif
