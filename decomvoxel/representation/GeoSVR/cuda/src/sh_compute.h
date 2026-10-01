/*************************************************************************
Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto.  Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*************************************************************************/

#ifndef SH_COMPUTE_H_INCLUDED
#define SH_COMPUTE_H_INCLUDED

#include <torch/extension.h>

namespace SH_COMPUTE {

// Python interface for spherical harmonic computation.
torch::Tensor sh_compute(
    const int D,
    const torch::Tensor& idx,
    const torch::Tensor& vox_centers,
    const torch::Tensor& cam_pos,
    const torch::Tensor& sh0,
    const torch::Tensor& shs);

std::tuple<torch::Tensor, torch::Tensor> sh_compute_bw(
    const int D, const int M,
    const torch::Tensor& idx,
    const torch::Tensor& vox_centers,
    const torch::Tensor& cam_pos,
    const torch::Tensor& rgbs,
    const torch::Tensor& dL_drgbs);

}

#endif
