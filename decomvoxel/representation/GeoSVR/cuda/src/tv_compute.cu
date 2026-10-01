/*************************************************************************
Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto.  Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*************************************************************************/

#include "tv_compute.h"
#include "auxiliary.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

namespace TV_COMPUTE {

template <bool no_tv_s, bool tv_sparse>
__global__ void total_variation_bw_cuda(
    const int N, const int C, const int NC,
    const float* __restrict__ grid_pts,
    const int64_t* __restrict__ vox_key,
    const float weight,
    const float* __restrict__ vox_size_inv,
    float* __restrict__ grid_pts_grad)
{
    auto idx = cg::this_grid().thread_rank();
    if (idx >= NC)
        return;
    const int iN = idx / C;
    const int iC = idx % C;

    // Load from global memory.
    int i_book[8];
    #pragma unroll
    for (int i=0, k=iN*8; i<8; ++i, ++k)
        i_book[i] = vox_key[k];

    if (tv_sparse)
    {
        bool valid = false;
        for (int i=0; i<8; ++i)
            valid |= (grid_pts_grad[i_book[i] * C + iC] != 0.f);
        if (!valid)
            return;
    }

    float vlst[8];
    #pragma unroll
    for (int i=0; i<8; ++i)
        vlst[i] = grid_pts[i_book[i] * C + iC];

    float w = weight;
    if (!no_tv_s)
        w *= 0.01f * vox_size_inv[iN];

    // Compute gradient wrt total variation loss
    int glst[8] = {0};
    #pragma unroll
    for (int i=0; i<8; ++i)
    {
        glst[i] += (vlst[i] > vlst[i^0b001]) * 2 - 1;
        glst[i] += (vlst[i] > vlst[i^0b010]) * 2 - 1;
        glst[i] += (vlst[i] > vlst[i^0b100]) * 2 - 1;
    }

    float dtv_dgrid_pts[8];
    #pragma unroll
    for (int i=0; i<8; ++i)
        dtv_dgrid_pts[i] = w * ((float)glst[i]);

    // Write back
    #pragma unroll
    for (int i=0; i<8; ++i)
        atomicAdd(grid_pts_grad + i_book[i] * C + iC, dtv_dgrid_pts[i]);
}


// Python interface to directly write the gradient of tv loss.
void total_variation_bw(
    const torch::Tensor& grid_pts,
    const torch::Tensor& vox_key,
    const float weight,
    const torch::Tensor& vox_size_inv,
    const bool no_tv_s,
    const bool tv_sparse,
    const torch::Tensor& grid_pts_grad)
{
    const int N = vox_key.size(0);
    const int C = grid_pts.size(1);
    const int NC = N * C;

    auto tv_kernel =
        (no_tv_s & tv_sparse) ? total_variation_bw_cuda<true, true>   :
        (no_tv_s)             ? total_variation_bw_cuda<true, false>  :
        (tv_sparse)           ? total_variation_bw_cuda<false, true>  :
                                total_variation_bw_cuda<false, false> ;

    if (N > 0)
        tv_kernel <<<(NC + 255) / 256, 256>>> (
            N, C, NC,
            grid_pts.contiguous().data_ptr<float>(),
            vox_key.contiguous().data_ptr<int64_t>(),
            weight,
            vox_size_inv.contiguous().data_ptr<float>(),
            grid_pts_grad.contiguous().data_ptr<float>());
}

}
