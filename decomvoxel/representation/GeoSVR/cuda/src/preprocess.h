/*************************************************************************
Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto.  Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*************************************************************************/

#ifndef RASTERIZER_PREPROCESS_H_INCLUDED
#define RASTERIZER_PREPROCESS_H_INCLUDED

#include <torch/extension.h>

namespace PREPROCESS {

// Interface for python to find the voxel to render and compute some init values.
std::tuple<torch::Tensor, torch::Tensor>
rasterize_preprocess(
    const int image_width, const int image_height,
    const float tan_fovx, const float tan_fovy,
    const float cx, const float cy,
    const torch::Tensor& w2c_matrix,
    const torch::Tensor& c2w_matrix,
    const int cam_mode,
    const float near,

    const torch::Tensor& octree_paths,
    const torch::Tensor& vox_centers,
    const torch::Tensor& vox_lengths,

    const bool debug);

}

#endif
