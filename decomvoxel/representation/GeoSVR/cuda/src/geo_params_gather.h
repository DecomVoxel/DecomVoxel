/*************************************************************************
Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto.  Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*************************************************************************/

#ifndef GEO_PARAMS_GATHER_H_INCLUDED
#define GEO_PARAMS_GATHER_H_INCLUDED

#include <torch/extension.h>

namespace GEO_PARAMS_GATHER {

// Python interface for gather grid points value into each voxel.
torch::Tensor gather_triinterp_geo_params(
    const torch::Tensor& vox_key,
    const torch::Tensor& care_idx,
    const torch::Tensor& grid_pts);

torch::Tensor gather_triinterp_geo_params_bw(
    const torch::Tensor& vox_key,
    const torch::Tensor& care_idx,
    const int num_grid_pts,
    const torch::Tensor& dL_dgeo_params);

torch::Tensor gather_triinterp_feat_params(
    const torch::Tensor& vox_key,
    const torch::Tensor& care_idx,
    const torch::Tensor& grid_pts);

torch::Tensor gather_triinterp_feat_params_bw(
    const torch::Tensor& vox_key,
    const torch::Tensor& care_idx,
    const int num_grid_pts,
    const torch::Tensor& dL_dfeat_params);

}

#endif
