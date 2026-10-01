/*************************************************************************
Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto.  Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*************************************************************************/

#ifndef RASTERIZER_FORWARD_H_INCLUDED
#define RASTERIZER_FORWARD_H_INCLUDED

#include <torch/extension.h>

namespace FORWARD {

// Interface for python to run forward rasterization.
std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
rasterize_voxels(
    const int vox_geo_mode,
    const int density_mode,
    const int image_width, const int image_height,
    const float tan_fovx, const float tan_fovy,
    const float cx, const float cy,
    const torch::Tensor& w2c_matrix,
    const torch::Tensor& c2w_matrix,
    const torch::Tensor& background,
    const int cam_mode,
    const bool need_depth,
    const bool need_distortion,
    const bool need_normal,
    const bool track_max_w,

    const torch::Tensor& octree_paths,
    const torch::Tensor& vox_centers,
    const torch::Tensor& vox_lengths,
    const torch::Tensor& geos,
    const torch::Tensor& rgbs,
    const torch::Tensor& feats,

    const torch::Tensor& geomBuffer,

    const bool debug);

}

#endif
