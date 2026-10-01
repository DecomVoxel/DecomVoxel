/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

/*************************************************************************
Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto.  Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*************************************************************************/

#include "sh_compute.h"
#include "auxiliary.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

namespace SH_COMPUTE {

__device__ const float SH_C0 = 0.28209479177387814f;
__device__ const float SH_C1 = 0.4886025119029199f;
__device__ const float SH_C2[] = {
    1.0925484305920792f,
    -1.0925484305920792f,
    0.31539156525252005f,
    -1.0925484305920792f,
    0.5462742152960396f
};
__device__ const float SH_C3[] = {
    -0.5900435899266435f,
    2.890611442640554f,
    -0.4570457994644658f,
    0.3731763325901154f,
    -0.4570457994644658f,
    1.445305721320277f,
    -0.5900435899266435f
};

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
template <int deg>
__device__ float3 computeColorFromSH(
    int idx, int max_coeffs, const float3 vox_c, const float3 ro,
    const float3* sh0, const float3* shs)
{
    // The implementation is loosely based on code for
    // "Differentiable Point-Based Radiance Fields for
    // Efficient View Synthesis" by Zhang et al. (2022)
    float3 dir = vox_c - ro;
    dir = dir * rnorm3df(dir.x, dir.y, dir.z);

    const float3* sh = shs + idx * (max_coeffs - 1);
    float3 result = SH_C0 * sh0[idx];

    if (deg > 0)
    {
        float x = dir.x;
        float y = dir.y;
        float z = dir.z;
        result = result - SH_C1 * y * sh[0] + SH_C1 * z * sh[1] - SH_C1 * x * sh[2];

        if (deg > 1)
        {
            float xx = x * x, yy = y * y, zz = z * z;
            float xy = x * y, yz = y * z, xz = x * z;
            result = result +
                SH_C2[0] * xy * sh[3] +
                SH_C2[1] * yz * sh[4] +
                SH_C2[2] * (2.0f * zz - xx - yy) * sh[5] +
                SH_C2[3] * xz * sh[6] +
                SH_C2[4] * (xx - yy) * sh[7];

            if (deg > 2)
            {
                result = result +
                    SH_C3[0] * y * (3.0f * xx - yy) * sh[8] +
                    SH_C3[1] * xy * z * sh[9] +
                    SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[10] +
                    SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[11] +
                    SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[12] +
                    SH_C3[5] * z * (xx - yy) * sh[13] +
                    SH_C3[6] * x * (xx - 3.0f * yy) * sh[14];
            }
        }
    }
    result = result + 0.5f;

    // RGB colors are clamped to non-negative values.
    result.x *= (result.x > 0.0f);
    result.y *= (result.y > 0.0f);
    result.z *= (result.z > 0.0f);
    return result;
}


// Backward pass for spherical harmonics to RGB.
template <int deg>
__device__ void computeColorFromSH_bw(
    int idx, int max_coeffs, const float3 vox_c, const float3 ro,
    const float3 rgb, float3 dL_drgb, float3* dL_dsh0, float3* dL_dshs)
{
    float3 dir = vox_c - ro;
    dir = dir * rnorm3df(dir.x, dir.y, dir.z);

    // Check if the color was clampped in the forward pass.
    // dL_drgb.x *= (float)(rgb.x > 0);
    // dL_drgb.y *= (float)(rgb.y > 0);
    // dL_drgb.z *= (float)(rgb.z > 0);

    // Base address of the sh coefficients of the voxel.
    float3* dL_dsh = dL_dshs + idx * (max_coeffs - 1);

    // Adapt the implementation from 3DGS.
    float dRGBdsh0 = SH_C0;
    dL_dsh0[idx] = dRGBdsh0 * dL_drgb;
    if (deg > 0)
    {
        float x = dir.x;
        float y = dir.y;
        float z = dir.z;
        float dRGBdsh1 = -SH_C1 * y;
        float dRGBdsh2 = SH_C1 * z;
        float dRGBdsh3 = -SH_C1 * x;
        dL_dsh[0] = dRGBdsh1 * dL_drgb;
        dL_dsh[1] = dRGBdsh2 * dL_drgb;
        dL_dsh[2] = dRGBdsh3 * dL_drgb;

        if (deg > 1)
        {
            float xx = x * x, yy = y * y, zz = z * z;
            float xy = x * y, yz = y * z, xz = x * z;

            float dRGBdsh4 = SH_C2[0] * xy;
            float dRGBdsh5 = SH_C2[1] * yz;
            float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
            float dRGBdsh7 = SH_C2[3] * xz;
            float dRGBdsh8 = SH_C2[4] * (xx - yy);
            dL_dsh[3] = dRGBdsh4 * dL_drgb;
            dL_dsh[4] = dRGBdsh5 * dL_drgb;
            dL_dsh[5] = dRGBdsh6 * dL_drgb;
            dL_dsh[6] = dRGBdsh7 * dL_drgb;
            dL_dsh[7] = dRGBdsh8 * dL_drgb;

            if (deg > 2)
            {
                float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
                float dRGBdsh10 = SH_C3[1] * xy * z;
                float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
                float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
                float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
                float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
                float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
                dL_dsh[8] = dRGBdsh9 * dL_drgb;
                dL_dsh[9] = dRGBdsh10 * dL_drgb;
                dL_dsh[10] = dRGBdsh11 * dL_drgb;
                dL_dsh[11] = dRGBdsh12 * dL_drgb;
                dL_dsh[12] = dRGBdsh13 * dL_drgb;
                dL_dsh[13] = dRGBdsh14 * dL_drgb;
                dL_dsh[14] = dRGBdsh15 * dL_drgb;
            }
        }
    }
}


// Compute rgb from spherical harmonic.
__global__ void sh_compute_cuda(
    const int N, const int n_vox, const int D, const int M,
    const int64_t* __restrict__ indices,
    const float3* __restrict__ vox_centers,
    const float3* __restrict__ cam_pos,
    const float3* __restrict__ sh0,
    const float3* __restrict__ shs,
    float3* __restrict__ rgbs)
{
    auto tid = cg::this_grid().thread_rank();
    if ((N == 0 && tid >= n_vox) || (N != 0 && tid >= N))
        return;

    // Load from global memory.
    const int idx = (N != 0) ? indices[tid] : tid;
    const float3 vox_c = vox_centers[idx];
    const float3 ro = *(cam_pos);

    // Convert spherical harmonics coefficients to RGB color.
    auto sh_eval =
        (D == 0) ? computeColorFromSH<0> :
        (D == 1) ? computeColorFromSH<1> :
        (D == 2) ? computeColorFromSH<2> :
                   computeColorFromSH<3> ;
    float3 sh_result = sh_eval(idx, M, vox_c, ro, sh0, shs);

    // Write back the results.
    rgbs[idx] = sh_result;
}


// Backward pass of the preprocessing steps.
__global__ void sh_compute_bw_cuda(
    const int N, const int n_vox, const int D, const int M,
    const int64_t* __restrict__ indices,
    const float3* __restrict__ vox_centers,
    const float3* __restrict__ cam_pos,
    const float3* __restrict__ rgbs,
    const float3* __restrict__ dL_drgbs,
    float3* __restrict__ dL_dsh0,
    float3* __restrict__ dL_dshs)
{
    auto tid = cg::this_grid().thread_rank();
    if ((N == 0 && tid >= n_vox) || (N != 0 && tid >= N))
        return;

    // Load from global memory.
    const int idx = (N != 0) ? indices[tid] : tid;
    const float3 vox_c = vox_centers[idx];
    const float3 ro = *(cam_pos);
    const float3 rgb = rgbs[idx];
    const float3 dL_drgb = dL_drgbs[idx];

    // Compute gradient updates due to computing colors from SHs
    auto sh_eval =
        (D == 0) ? computeColorFromSH_bw<0> :
        (D == 1) ? computeColorFromSH_bw<1> :
        (D == 2) ? computeColorFromSH_bw<2> :
                   computeColorFromSH_bw<3> ;
    sh_eval(idx, M, vox_c, ro, rgb, dL_drgb, dL_dsh0, dL_dshs);
}


// Python interface for spherical harmonic computation.
torch::Tensor sh_compute(
    const int D,
    const torch::Tensor& indices,
    const torch::Tensor& vox_centers,
    const torch::Tensor& cam_pos,
    const torch::Tensor& sh0,
    const torch::Tensor& shs)
{
    const int P = vox_centers.size(0);
    const int N = indices.size(0);
    const int M = 1 + shs.size(1);
    torch::Tensor rgbs = torch::zeros({P, 3}, vox_centers.options());

    const int total_threads = N != 0 ? N : P;

    if (P > 0)
        sh_compute_cuda <<<(total_threads + 255) / 256, 256>>> (
            N, P, D, M,
            indices.contiguous().data_ptr<int64_t>(),
            (float3*)vox_centers.contiguous().data_ptr<float>(),
            (float3*)cam_pos.contiguous().data_ptr<float>(),
            (float3*)sh0.contiguous().data_ptr<float>(),
            (float3*)shs.contiguous().data_ptr<float>(),
            (float3*)rgbs.contiguous().data_ptr<float>());
    
    return rgbs;
}

std::tuple<torch::Tensor, torch::Tensor> sh_compute_bw(
    const int D, const int M,
    const torch::Tensor& indices,
    const torch::Tensor& vox_centers,
    const torch::Tensor& cam_pos,
    const torch::Tensor& rgbs,
    const torch::Tensor& dL_drgbs)
{
    const int P = vox_centers.size(0);
    const int N = indices.size(0);
    torch::Tensor dL_dsh0 = torch::zeros({P, 3}, vox_centers.options());
    torch::Tensor dL_dshs = torch::zeros({P, M-1, 3}, vox_centers.options());

    const int total_threads = N != 0 ? N : P;

    if (P > 0)
        sh_compute_bw_cuda <<<(total_threads + 255) / 256, 256>>> (
            N, P, D, M,
            indices.contiguous().data_ptr<int64_t>(),
            (float3*)vox_centers.contiguous().data_ptr<float>(),
            (float3*)cam_pos.contiguous().data_ptr<float>(),
            (float3*)rgbs.contiguous().data_ptr<float>(),
            (float3*)dL_drgbs.contiguous().data_ptr<float>(),
            (float3*)dL_dsh0.contiguous().data_ptr<float>(),
            (float3*)dL_dshs.contiguous().data_ptr<float>());

    return std::make_tuple(dL_dsh0, dL_dshs);
}

}
