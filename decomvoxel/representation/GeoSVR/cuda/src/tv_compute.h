/*************************************************************************
Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto.  Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*************************************************************************/

#ifndef TV_COMPUTE_H_INCLUDED
#define TV_COMPUTE_H_INCLUDED

#include <torch/extension.h>

namespace TV_COMPUTE {

// Python interface to directly write the gradient of tv loss.
void total_variation_bw(
    const torch::Tensor& grid_pts,
    const torch::Tensor& vox_key,
    const float weight,
    const torch::Tensor& vox_size_inv,
    const bool no_tv_s,
    const bool tv_sparse,
    const torch::Tensor& grid_pts_grad);

}

#endif
