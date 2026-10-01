/*************************************************************************
Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto.  Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*************************************************************************/

#ifndef RASTERIZER_BACKWARD_H_INCLUDED
#define RASTERIZER_BACKWARD_H_INCLUDED

#include <torch/extension.h>

namespace BACKWARD
{

// Interface for python to run backward pass of voxel rasterization.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
rasterize_voxels_backward(
    const int R,
    const int vox_geo_mode,
    const int density_mode,
    const int image_width, const int image_height,
    const float tan_fovx, const float tan_fovy,
    const float cx, const float cy,
    const torch::Tensor& w2c_matrix,
    const torch::Tensor& c2w_matrix,
    const torch::Tensor& background,

    const torch::Tensor& octree_paths,
    const torch::Tensor& vox_centers,
    const torch::Tensor& vox_lengths,
    const torch::Tensor& geos,
    const torch::Tensor& rgbs,

    const torch::Tensor& geomBuffer,
    const torch::Tensor& binningBuffer,
    const torch::Tensor& imageBuffer,
    const torch::Tensor& out_T,

    const torch::Tensor& dL_dout_color,
    const torch::Tensor& dL_dout_depth,
    const torch::Tensor& dL_dout_normal,
    const torch::Tensor& dL_dout_T,

    const float lambda_R_concen,
    const torch::Tensor& gt_color,
    const float lambda_ascending,
    const float lambda_scaling_penalty,
    const float min_voxel_size,
    const float lambda_dist,
    const bool need_depth,
    const bool need_normal,
    const torch::Tensor& out_D,
    const torch::Tensor& out_N,

    const bool debug);

}

#endif
