/*************************************************************************
Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto.  Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*************************************************************************/

#include "geo_params_gather.h"
#include "auxiliary.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

namespace GEO_PARAMS_GATHER {

__global__ void gather_triinterp_geo_params_cuda(
    const int n_care,
    const int64_t* __restrict__ vox_key,
    const int64_t* __restrict__ care_idx,
    const float* __restrict__ grid_pts,
    float* __restrict__ geo_params)
{
    auto tid = cg::this_grid().thread_rank();
    if (tid >= n_care)
        return;

    // Load from global mem
    const int idx = care_idx[tid];
    int key[8];
    for(int i=0; i<8; ++i)
        key[i] = vox_key[idx * 8 + i];

    float params[8];
    for(int i=0; i<8; ++i)
        params[i] = grid_pts[key[i]];

    // Write to voxel geo param
    for(int i=0; i<8; ++i)
        geo_params[idx * 8 + i] = params[i];
}

__global__ void gather_triinterp_geo_params_bw_cuda(
    const int n_care,
    const int64_t* __restrict__ vox_key,
    const int64_t* __restrict__ care_idx,
    const float* __restrict__ dL_dgeo_params,
    float* __restrict__ dL_dgrid_pts)
{
    auto tid = cg::this_grid().thread_rank();
    if (tid >= n_care)
        return;

    // Load from global mem
    const int idx = care_idx[tid];
    int key[8];
    for(int i=0; i<8; ++i)
        key[i] = vox_key[idx * 8 + i];

    float dL_dparams[8];
    for(int i=0; i<8; ++i)
        dL_dparams[i] = dL_dgeo_params[idx * 8 + i];

    // Write to voxel geo param
    for(int i=0; i<8; ++i)
        atomicAdd(dL_dgrid_pts + key[i], dL_dparams[i]);
}

template <int n_dim>
__global__ void gather_triinterp_feat_params_cuda(
    const int n_care,
    const int64_t* __restrict__ vox_key,
    const int64_t* __restrict__ care_idx,
    const float* __restrict__ grid_pts,
    float* __restrict__ feat_params)
{
    auto tid = cg::this_grid().thread_rank();
    if (tid >= n_care)
        return;

    // Load from global mem
    const int idx = care_idx[tid];
    int key[8];
    for(int i=0; i<8; ++i)
        key[i] = vox_key[idx * 8 + i];

    float params[8][n_dim];
    for(int i=0; i<8; ++i)
        for(int j=0; j<n_dim; ++j)
            params[i][j] = grid_pts[key[i] * n_dim + j];

    // Write to voxel geo param
    for(int i=0; i<8; ++i)
        for(int j=0; j<n_dim; ++j)
            feat_params[idx * 8 * n_dim + i * n_dim + j] = params[i][j];
}

template <int n_dim>
__global__ void gather_triinterp_feat_params_bw_cuda(
    const int n_care,
    const int64_t* __restrict__ vox_key,
    const int64_t* __restrict__ care_idx,
    const float* __restrict__ dL_dfeat_params,
    float* __restrict__ dL_dgrid_pts)
{
    auto tid = cg::this_grid().thread_rank();
    if (tid >= n_care)
        return;

    // Load from global mem
    const int idx = care_idx[tid];
    int key[8];
    for(int i=0; i<8; ++i)
        key[i] = vox_key[idx * 8 + i];

    float dL_dparams[8][n_dim];
    for(int i=0; i<8; ++i)
        for(int j=0; j<n_dim; ++j)
            dL_dparams[i][j] = dL_dfeat_params[idx * 8 * n_dim + i * n_dim + j];

    // Write to voxel geo param
    for(int i=0; i<8; ++i)
        for(int j=0; j<n_dim; ++j)
            atomicAdd(dL_dgrid_pts + key[i] * n_dim + j, dL_dparams[i][j]);
}


// Python interface for gather grid points value into each voxel.
torch::Tensor gather_triinterp_geo_params(
    const torch::Tensor& vox_key,
    const torch::Tensor& care_idx,
    const torch::Tensor& grid_pts)
{
    const int n_vox = vox_key.size(0);
    const int n_care = care_idx.size(0);
    torch::Tensor geo_params = torch::empty({n_vox, 8}, grid_pts.options());

    if (n_care > 0)
        gather_triinterp_geo_params_cuda <<<(n_care + 255) / 256, 256>>> (
            n_care,
            vox_key.contiguous().data_ptr<int64_t>(),
            care_idx.contiguous().data_ptr<int64_t>(),
            grid_pts.contiguous().data_ptr<float>(),
            geo_params.contiguous().data_ptr<float>());

    return geo_params;
}

torch::Tensor gather_triinterp_geo_params_bw(
    const torch::Tensor& vox_key,
    const torch::Tensor& care_idx,
    const int num_grid_pts,
    const torch::Tensor& dL_dgeo_params)
{
    const int n_vox = vox_key.size(0);
    const int n_care = care_idx.size(0);
    torch::Tensor dL_dgrid_pts = torch::zeros({num_grid_pts, 1}, dL_dgeo_params.options());

    if (n_care > 0)
        gather_triinterp_geo_params_bw_cuda <<<(n_care + 255) / 256, 256>>> (
            n_care,
            vox_key.contiguous().data_ptr<int64_t>(),
            care_idx.contiguous().data_ptr<int64_t>(),
            dL_dgeo_params.contiguous().data_ptr<float>(),
            dL_dgrid_pts.contiguous().data_ptr<float>());

    return dL_dgrid_pts;
}

torch::Tensor gather_triinterp_feat_params(
    const torch::Tensor& vox_key,
    const torch::Tensor& care_idx,
    const torch::Tensor& grid_pts)
{
    const int n_vox = vox_key.size(0);
    const int n_care = care_idx.size(0);
    const int n_dim = grid_pts.size(1);
    torch::Tensor feat_params = torch::empty({n_vox, 8, n_dim}, grid_pts.options());

    if (n_dim != 3)
        AT_ERROR("Only support n_dim=3 now.");

    if (n_care > 0)
        gather_triinterp_feat_params_cuda<3> <<<(n_care + 255) / 256, 256>>> (
            n_care,
            vox_key.contiguous().data_ptr<int64_t>(),
            care_idx.contiguous().data_ptr<int64_t>(),
            grid_pts.contiguous().data_ptr<float>(),
            feat_params.contiguous().data_ptr<float>());

    return feat_params;
}

torch::Tensor gather_triinterp_feat_params_bw(
    const torch::Tensor& vox_key,
    const torch::Tensor& care_idx,
    const int num_grid_pts,
    const torch::Tensor& dL_dfeat_params)
{
    const int n_vox = vox_key.size(0);
    const int n_care = care_idx.size(0);
    const int n_dim = dL_dfeat_params.size(2);
    torch::Tensor dL_dgrid_pts = torch::zeros({num_grid_pts, n_dim}, dL_dfeat_params.options());

    if (n_dim != 3)
        AT_ERROR("Only support n_dim=3");

    if (n_care > 0)
        gather_triinterp_feat_params_bw_cuda<3> <<<(n_care + 255) / 256, 256>>> (
            n_care,
            vox_key.contiguous().data_ptr<int64_t>(),
            care_idx.contiguous().data_ptr<int64_t>(),
            dL_dfeat_params.contiguous().data_ptr<float>(),
            dL_dgrid_pts.contiguous().data_ptr<float>());

    return dL_dgrid_pts;
}

}
