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

#include "backward.h"
#include "raster_state.h"
#include "auxiliary.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

namespace BACKWARD {

// CUDA backward pass of sparse voxel rendering.
template <bool need_depth, bool need_distortion, bool need_normal,
          int n_samp>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
    const uint2* __restrict__ ranges,
    const uint32_t* __restrict__ vox_list,
    const int W, const int H,
    const float tan_fovx, const float tan_fovy,
    const float cx, const float cy,
    const float* __restrict__ c2w_matrix,
    const float3* __restrict__ background,

    const uint2* __restrict__ bboxes,
    const float3* __restrict__ vox_centers,
    const float* __restrict__ vox_lengths,
    const float* __restrict__ geos,
    const float3* __restrict__ rgbs,

    const float* __restrict__ out_T,
    const uint32_t* __restrict__ tile_last,
    const uint32_t* __restrict__ n_contrib,

    const float* __restrict__ dL_dout_color,
    const float* __restrict__ dL_dout_depth,
    const float* __restrict__ dL_dout_normal,
    const float* __restrict__ dL_dout_T,

    const float lambda_R_concen,
    const float* gt_color,
    const float lambda_ascending,
    const float lambda_scaling_penalty,
    const float min_voxel_size,
    const float lambda_dist,
    const float* out_D,
    const float* out_N,

    float* dL_dvox)
{
    // We rasterize again. Compute necessary block info.
    auto block = cg::this_thread_block();
    uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
    int thread_id = block.thread_rank();
    int tile_id = block.group_index().y * horizontal_blocks + block.group_index().x;
    uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
    uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };

    uint2 pix;
    uint32_t pix_id;
    float2 pixf;
    if (BLOCK_X % 8 == 0 && BLOCK_Y % 4 == 0)
    {
        // Pack the warp threads into a 4x8 macro blocks.
        // It could reduce idle warp threads as the voxels to render
        // are more coherent in 4x8 than 2x16 rectangle.
        int macro_x_num = BLOCK_X / 8;
        int macro_id = thread_id / 32;
        int macro_xid = macro_id % macro_x_num;
        int macro_yid = macro_id / macro_x_num;
        int micro_id = thread_id % 32;
        int micro_xid = micro_id % 8;
        int micro_yid = micro_id / 8;
        pix = { pix_min.x + macro_xid * 8 + micro_xid, pix_min.y + macro_yid * 4 + micro_yid};
        pix_id = W * pix.y + pix.x;
        pixf = { (float)pix.x, (float)pix.y };
    }
    else
    {
        pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
        pix_id = W * pix.y + pix.x;
        pixf = { (float)pix.x, (float)pix.y };
    }

    // Compute camera info.
    const float3 cam_rd = compute_ray_d(pixf, W, H, tan_fovx, tan_fovy, cx, cy);
    const float rd_norm = sqrtf(dot(cam_rd, cam_rd));
    const float rd_norm_inv = 1.f / rd_norm;
    const float3 ro = last_col_3x4(c2w_matrix);
    const float3 rd_raw = rotate_3x4(c2w_matrix, cam_rd);
    const float3 rd = rd_raw * rd_norm_inv;
    const float3 rd_inv = {1.f/ rd.x, 1.f / rd.y, 1.f / rd.z};
    uint32_t pix_quad_id = compute_ray_quadrant_id(rd);

    // Check if this thread is associated with a valid pixel or outside.
    bool inside = (pix.x < W) && (pix.y < H);
    // Done threads can help with fetching, but don't rasterize
    bool done = !inside;

    const uint2 range_raw = ranges[tile_id];
    const uint2 range = {range_raw.x, tile_last[tile_id]};
    if (range.y > range_raw.y)
    {
        // TODO: remove sanity check.
        printf("range.y > range_raw.y !???");
        __trap();
    }
    if (range.x > range.y)
    {
        // TODO: remove sanity check.
        printf("range.x > range.y !???");
        __trap();
    }

    const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

    int toDo = range.y - range.x;

    // Allocate storage for batches of collectively fetched data.
    // 3090Ti shared memory per-block statistic:
    //   total shared memory      = 49152 bytes
    //   shared memory per-thread = 49152/BLOCK_SIZE = 192 bytes
    //                            = 48 int or float
    __shared__ int collected_vox_id[BLOCK_SIZE];
    __shared__ int collected_quad_id[BLOCK_SIZE];
    __shared__ uint2 collected_bbox[BLOCK_SIZE];
    __shared__ float3 collected_vox_c[BLOCK_SIZE];
    __shared__ float collected_vox_l[BLOCK_SIZE];
    __shared__ float collected_geo_params[BLOCK_SIZE * 8];
    __shared__ float3 collected_rgb[BLOCK_SIZE];

    // In the forward, we stored the final value for T, the
    // product of all (1 - alpha) factors.
    const float T_final = inside ? out_T[pix_id] : 0.f;
    float T = T_final;

    // We start from the back.
    // The last contributing voxel ID of each pixel is known from the forward.
    uint32_t contributor = toDo;
    const int last_contributor = inside ? n_contrib[pix_id] : 0;

    // Init gradient from the last computation node.
    float3 dL_dpix;
    float dL_dD;
    float3 dL_dN;
    float last_dL_dT;
    if (inside)
    {
        dL_dpix.x = dL_dout_color[0 * H * W + pix_id];
        dL_dpix.y = dL_dout_color[1 * H * W + pix_id];
        dL_dpix.z = dL_dout_color[2 * H * W + pix_id];
        const float dL_dpix_T = dL_dout_T[pix_id];
        const float3 bg_color = background[0];
        last_dL_dT = dL_dpix_T + dot(bg_color, dL_dpix);

        dL_dD = dL_dout_depth[pix_id] * rd_norm_inv;
        dL_dN.x = dL_dout_normal[0 * H * W + pix_id];
        dL_dN.y = dL_dout_normal[1 * H * W + pix_id];
        dL_dN.z = dL_dout_normal[2 * H * W + pix_id];
    }

    // Compute regularization weights.
    const float WH_inv = 1.f / ((float)(W * H));
    const float weight_R_concen = lambda_R_concen * WH_inv;
    const float weight_ascending = lambda_ascending * WH_inv;
    const float weight_scaling_penalty = lambda_scaling_penalty * WH_inv;
    float3 gt_pix;
    if (lambda_R_concen > 0 && inside)
    {
        gt_pix.x = gt_color[0 * H * W + pix_id];
        gt_pix.y = gt_color[1 * H * W + pix_id];
        gt_pix.z = gt_color[2 * H * W + pix_id];
    }

    float3 pix_n;
    if (need_normal && inside)
    {
        pix_n.x = out_N[0 * H * W + pix_id];
        pix_n.y = out_N[1 * H * W + pix_id];
        pix_n.z = out_N[2 * H * W + pix_id];
        pix_n = safe_rnorm(pix_n) * pix_n;
    }

    const float weight_dist = lambda_dist * WH_inv;
    float prefix_wm, suffix_wm, prefix_w, suffix_w;
    if (lambda_dist > 0 && inside)
    {
        // See DVGOv2 for formula.
        prefix_wm = out_D[H * W + pix_id];
        suffix_wm = 0.f;
        prefix_w = 1.f - T_final;
        suffix_w = 0.f;
    }

    // For seam regularizaiton.
    int j_lst[BLOCK_SIZE];

    // Traverse all voxels.
    for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
    {
        // Load auxiliary data into shared memory, start in the BACK
        // and load them in revers order.
        block.sync();
        const int progress = i * BLOCK_SIZE + thread_id;
        if (range.x + progress < range.y)
        {
            uint32_t order_val = vox_list[range.y - progress - 1];
            uint32_t vox_id = decode_order_val_4_vox_id(order_val);
            uint32_t quad_id = decode_order_val_4_quadrant_id(order_val);
            collected_vox_id[thread_id] = vox_id;
            collected_quad_id[thread_id] = quad_id;
            collected_bbox[thread_id] = bboxes[vox_id];
            collected_vox_c[thread_id] = vox_centers[vox_id];
            collected_vox_l[thread_id] = vox_lengths[vox_id];
            for (int k=0; k<8; ++k)
                collected_geo_params[thread_id*8 + k] = geos[vox_id*8 + k];
            collected_rgb[thread_id] = rgbs[vox_id];
        }
        block.sync();

        // Iterate over voxels.
        const int end_j = min(BLOCK_SIZE, toDo);
        int j_lst_top = -1;
        for (int j = 0; !done && j < end_j; j++)
        {
            // Keep track of current voxel ID. Skip, if this one
            // is behind the last contributor for this pixel.
            contributor--;
            if (contributor >= last_contributor)
                continue;

            /**************************
            Below, we first compute blending values, as in the forward.
            **************************/

            // Check if the pixel in the projected bbox region.
            // Check if the quadrant id match the pixel.
            if (!pix_in_bbox(pix, collected_bbox[j]) || pix_quad_id != collected_quad_id[j])
                continue;

            // Compute ray aabb intersection.
            const float3 vox_c = collected_vox_c[j];
            const float vox_l = collected_vox_l[j];
            const float2 ab = ray_aabb(vox_c, vox_l, ro, rd_inv);
            const float a = ab.x;
            const float b = ab.y;
            if (a > b)
                continue;  // Skip if no intersection.

            j_lst_top += 1;
            j_lst[j_lst_top] = j;
        }

        for (int jj = 0; !done && jj <= j_lst_top; jj++)
        {
            int j = j_lst[jj];

            // Compute ray aabb intersection.
            const float3 vox_c = collected_vox_c[j];
            const float vox_l = collected_vox_l[j];
            const float2 ab = ray_aabb(vox_c, vox_l, ro, rd_inv);
            const float a = ab.x;
            const float b = ab.y;

            // Compute closed-form volume integral.
            float geo_params[8];
            for (int k=0; k<8; ++k)
                geo_params[k] = collected_geo_params[j*8 + k];
            float dL_dgeo_params[8] = {0.f};

            float vol_int = 0.f;
            float dI_dgeo_params[8] = {0.f};
            float each_dI_dgeo_params[n_samp][8];
            float interp_w[8];
            float local_alphas[n_samp];

            float vox_l_inv = 1.f / vox_l;
            const float step_sz = (b - a) * (1.f / n_samp);
            const float3 step = step_sz * rd;
            float3 pt = ro + (a + 0.5f * step_sz) * rd;
            float3 qt = (pt - (vox_c - 0.5f * vox_l)) * vox_l_inv;
            const float3 qt_step = step * vox_l_inv;

            #pragma unroll
            for (int k=0; k<n_samp; k++, qt=qt+qt_step)
            {
                tri_interp_weight(qt, interp_w);
                float d = 0.f;
                for (int iii=0; iii<8; ++iii)
                    d += geo_params[iii] * interp_w[iii];
                const float local_vol_int = STEP_SZ_SCALE * step_sz * exp_linear_11(d);
                vol_int += local_vol_int;

                if (need_depth && n_samp > 1)
                    local_alphas[k] = min(MAX_ALPHA, 1.f - expf(-local_vol_int));

                const float dd_dd = STEP_SZ_SCALE * step_sz * exp_linear_11_bw(d);
                for (int iii=0; iii<8; ++iii)
                {
                    float tmp = dd_dd * interp_w[iii];
                    dI_dgeo_params[iii] += tmp;
                    if (need_depth && n_samp > 1)
                        each_dI_dgeo_params[k][iii] = tmp;
                }
            }

            // Compute alpha from volume integral.
            // Follow 3DGS's alpha clamping to avoid numerical instabilities.
            const float exp_neg_vol_int = expf(-vol_int);
            float alpha = min(MAX_ALPHA, 1.f - exp_neg_vol_int);
            if (alpha < MIN_ALPHA)
                continue;

            // Recover the blending weight of this voxel.
            T = T / (1.f - alpha);
            const float pt_w = alpha * T;

            // Propagate gradients to per-voxel colors and keep
            // gradients w.r.t. voxel alpha.
            // Load from share memory.
            const int vox_id = collected_vox_id[j];

            const float3 c = collected_rgb[j];

            // The gradients w.r.t. voxel alpha.
            float dL_dpt_w = dot(dL_dpix, c);

            // Gradient from distortion loss
            if (need_distortion)
            {
                float adist = depth_contracted(a);
                float bdist = depth_contracted(b);

                const float now_m = 0.5f * (adist + bdist);
                const float now_wm = now_m * pt_w;
                prefix_wm -= now_wm;
                prefix_w -= pt_w;

                const float dist_grad_uni = 0.6666666f * pt_w * (bdist - adist);
                const float dist_grad_bi = 2.f * (now_m * (prefix_w - suffix_w) - (prefix_wm - suffix_wm));
                dL_dpt_w += weight_dist * (dist_grad_uni + dist_grad_bi);

                suffix_wm += now_wm;
                suffix_w += pt_w;
            }

            // Gradient from normal
            if (need_normal)
            {
                float N_grad_to_geo_params[8];
                const float lin_nx = (
                    (geo_params[0b100] + geo_params[0b101] + geo_params[0b110] + geo_params[0b111]) -
                    (geo_params[0b000] + geo_params[0b001] + geo_params[0b010] + geo_params[0b011]));
                const float lin_ny = (
                    (geo_params[0b010] + geo_params[0b011] + geo_params[0b110] + geo_params[0b111]) -
                    (geo_params[0b000] + geo_params[0b001] + geo_params[0b100] + geo_params[0b101]));
                const float lin_nz = (
                    (geo_params[0b001] + geo_params[0b011] + geo_params[0b101] + geo_params[0b111]) -
                    (geo_params[0b000] + geo_params[0b010] + geo_params[0b100] + geo_params[0b110]));
                const float3 lin_n = make_float3(lin_nx, lin_ny, lin_nz);
                const float r_lin = safe_rnorm(lin_n);
                const float3 surf_n = r_lin * lin_n;

                dL_dpt_w += dot(dL_dN, surf_n);
                float3 dL_dsurf_n = pt_w * dL_dN;

                // Gradient from normal concentration loss
                // dL_dsurf_n = dL_dsurf_n + weight_N_concen * pt_w * (1.f - dot(surf_n, pix_n)) * (-pix_n);

                // To be added later
                float3 dL_dlin_n = r_lin * (dL_dsurf_n - dot(dL_dsurf_n, lin_n) * r_lin * surf_n);
                N_grad_to_geo_params[0b000] =  -dL_dlin_n.x  -dL_dlin_n.y  -dL_dlin_n.z;
                N_grad_to_geo_params[0b001] =  -dL_dlin_n.x  -dL_dlin_n.y  +dL_dlin_n.z;
                N_grad_to_geo_params[0b010] =  -dL_dlin_n.x  +dL_dlin_n.y  -dL_dlin_n.z;
                N_grad_to_geo_params[0b011] =  -dL_dlin_n.x  +dL_dlin_n.y  +dL_dlin_n.z;
                N_grad_to_geo_params[0b100] =  +dL_dlin_n.x  -dL_dlin_n.y  -dL_dlin_n.z;
                N_grad_to_geo_params[0b101] =  +dL_dlin_n.x  -dL_dlin_n.y  +dL_dlin_n.z;
                N_grad_to_geo_params[0b110] =  +dL_dlin_n.x  +dL_dlin_n.y  -dL_dlin_n.z;
                N_grad_to_geo_params[0b111] =  +dL_dlin_n.x  +dL_dlin_n.y  +dL_dlin_n.z;

                for (int iii=0; iii<8; ++iii)
                    dL_dgeo_params[iii] += N_grad_to_geo_params[iii];
            }

            // Compute gradient accumulated to the alpha.
            const float dL_dalpha = T * (dL_dpt_w - last_dL_dT);

            // Update last_dL_dT for next iteration.
            last_dL_dT += alpha * (dL_dpt_w - last_dL_dT);

            /**************************
            Backprop from voxel volume integral to surface parameters.
            **************************/
            const float dL_dI = dL_dalpha * exp_neg_vol_int;

            /**************************
            Sum up the gradient from rendering below.
            **************************/
            for (int iii=0; iii<8; ++iii)
                dL_dgeo_params[iii] += dL_dI * dI_dgeo_params[iii];

            // Gradient from depth
            if (need_depth)
            {
                float dval;
                float dLdepth_dI[n_samp];
                if (n_samp == 3)
                {
                    float step_sz = 0.3333333f * (b - a);
                    float a0 = local_alphas[0], a1 = local_alphas[1], a2 = local_alphas[2];
                    float t0 = a + 0.5f * step_sz;
                    float t1 = a + 1.5f * step_sz;
                    float t2 = a + 2.5f * step_sz;
                    dval = a0*t0 + (1.f-a0)*a1*t1 + (1.f-a0)*(1.f-a1)*a2*t2;
                    dLdepth_dI[0] = dL_dD * T * (t0 + a1*a2*t2 - a1*t1 - a2*t2) * (1.f - a0);
                    dLdepth_dI[1] = dL_dD * T * (t1 + a0*a2*t2 - a0*t1 - a2*t2) * (1.f - a1);
                    dLdepth_dI[2] = dL_dD * T * (t2 + a0*a1*t2 - a0*t2 - a1*t2) * (1.f - a2);
                }
                else if (n_samp == 2)
                {
                    float step_sz = 0.5f * (b - a);
                    float a0 = local_alphas[0], a1 = local_alphas[1];
                    float t0 = a + 0.5f * step_sz;
                    float t1 = a + 1.5f * step_sz;
                    dval = a0*t0 + (1.f-a0)*a1*t1;
                    dLdepth_dI[0] = dL_dD * T * (t0 - a1*t1) * (1.f - a0);
                    dLdepth_dI[1] = dL_dD * T * (t1 - a0*t1) * (1.f - a1);
                }
                else
                {
                    float t0 = 0.5f * (a + b);
                    dval = alpha * t0;
                    dLdepth_dI[0] = dL_dD * T * t0 * (1.f - alpha);
                }

                last_dL_dT += dL_dD * dval;

                if (n_samp == 3)
                {
                    for (int iii=0; iii<8; ++iii)
                        dL_dgeo_params[iii] += dLdepth_dI[0] * each_dI_dgeo_params[0][iii] + \
                                            dLdepth_dI[1] * each_dI_dgeo_params[1][iii] + \
                                            dLdepth_dI[2] * each_dI_dgeo_params[2][iii];
                }
                else if (n_samp == 2)
                {
                    for (int iii=0; iii<8; ++iii)
                        dL_dgeo_params[iii] += dLdepth_dI[0] * each_dI_dgeo_params[0][iii] + \
                                            dLdepth_dI[1] * each_dI_dgeo_params[1][iii];
                }
                else
                {
                    for (int iii=0; iii<8; ++iii)
                        dL_dgeo_params[iii] += dLdepth_dI[0] * dI_dgeo_params[iii];
                }
            }

            if (lambda_ascending > 0)
            {
                const float3 pt_a = ro + a * rd;
                const float3 qt_a = (pt_a - (vox_c - 0.5f * vox_l)) * vox_l_inv;
                const float3 pt_b = ro + b * rd;
                const float3 qt_b = (pt_b - (vox_c - 0.5f * vox_l)) * vox_l_inv;

                float interp_w_a[8], interp_w_b[8];
                tri_interp_weight(qt_a, interp_w_a);
                tri_interp_weight(qt_b, interp_w_b);

                float d_a = 0.f, d_b = 0.f;
                for (int iii=0; iii<8; ++iii)
                {
                    d_a += geo_params[iii] * interp_w_a[iii];
                    d_b += geo_params[iii] * interp_w_b[iii];
                }

                // L = max(0, d_a - d_b)
                const float reg_w = weight_ascending * pt_w * static_cast<float>(d_a > d_b);
                for (int iii=0; iii<8; ++iii)
                    dL_dgeo_params[iii] += reg_w * (interp_w_a[iii] - interp_w_b[iii]);
            }

            // Surface Rectification
            if (lambda_ascending < 0)
            {
                const float3 pt_a = ro + a * rd;
                const float3 qt_a = (pt_a - (vox_c - 0.5f * vox_l)) * vox_l_inv;
                const float3 pt_b = ro + b * rd;
                const float3 qt_b = (pt_b - (vox_c - 0.5f * vox_l)) * vox_l_inv;

                float interp_w_a[8], interp_w_b[8];
                tri_interp_weight(qt_a, interp_w_a);
                tri_interp_weight(qt_b, interp_w_b);

                float d_a = 0.f, d_b = 0.f;
                for (int iii=0; iii<8; ++iii)
                {
                    d_a += geo_params[iii] * interp_w_a[iii];
                    d_b += geo_params[iii] * interp_w_b[iii];
                }

                // L = max(0, d_a - d_b)
                if (pt_w > 0.3){
                    const float reg_w_1 = -weight_ascending * pt_w * static_cast<float>(STEP_SZ_SCALE * (b - a) * exp_linear_11(d_a) < 0.69) * static_cast<float>(STEP_SZ_SCALE * (b - a) * exp_linear_11(d_b) > 0.69); // alpha < 0.5
                    const float trunc_a = static_cast<float>(STEP_SZ_SCALE * (b - a) * exp_linear_11(d_a) > 0.1); // alpha = 0.1
                    const float trunc_b = static_cast<float>(STEP_SZ_SCALE * (b - a) * exp_linear_11(d_b) < 4.6); // alpha = 0.99
                    for (int iii=0; iii<8; ++iii)
                        dL_dgeo_params[iii] += reg_w_1 * (interp_w_a[iii] * trunc_a - interp_w_b[iii] * trunc_b);
                
                    const float reg_w_2 = -weight_ascending * pt_w * static_cast<float>(d_a > d_b);
                    const float trunc_a_max = static_cast<float>(STEP_SZ_SCALE * (b - a) * exp_linear_11(d_a) < 2.3); // alpha = 0.9
                    for (int iii=0; iii<8; ++iii)
                        dL_dgeo_params[iii] += reg_w_2 * (-interp_w_a[iii] * trunc_a_max);
                }
            }

            if (lambda_scaling_penalty > 0)
            {
                const float3 pt_a = ro + a * rd;
                const float3 qt_a = (pt_a - (vox_c - 0.5f * vox_l)) * vox_l_inv;
                const float3 pt_b = ro + b * rd;
                const float3 qt_b = (pt_b - (vox_c - 0.5f * vox_l)) * vox_l_inv;

                // L = max(0, d_a - d_b)
                const float reg_w = weight_scaling_penalty * pt_w * max(0.0, log2((b - a) / min_voxel_size));
                for (int iii=0; iii<8; ++iii)
                    dL_dgeo_params[iii] += reg_w;
            }

            /**************************
            Compute regularization gradient below.
            **************************/
            float dL_drgb[3] = {pt_w * dL_dpix.x, pt_w * dL_dpix.y, pt_w * dL_dpix.z};
            if (lambda_R_concen > 0)
            {
                const float3 grad_R_concen = weight_R_concen * pt_w * 2.0f * (c - gt_pix);
                dL_drgb[0] += grad_R_concen.x;
                dL_drgb[1] += grad_R_concen.y;
                dL_drgb[2] += grad_R_concen.z;
            }

            /**************************
            Write back the gradient below.
            **************************/
            float grad_pack[12];
            #pragma unroll
            for (int iii=0; iii<8; ++iii)
                grad_pack[iii] = dL_dgeo_params[iii];
            grad_pack[8] = dL_drgb[0];
            grad_pack[9] = dL_drgb[1];
            grad_pack[10] = dL_drgb[2];
            grad_pack[11] = fabs(dL_dalpha * alpha);

            const int base_id = cg::this_grid().thread_rank();
            #pragma unroll
            for (int iii=0; iii<12; ++iii)
                atomicAdd(dL_dvox + vox_id * 12 + (base_id+iii)%12, grad_pack[(base_id+iii)%12]);
        }
    }
}

#ifndef BwRendFunc
// Dirty trick. The argument name must be aligned with BACKWARD::render.
#define BwRendFunc(...) \
    ( \
        (need_depth && need_distortion && need_normal) ?\
            renderCUDA<true, true, true, __VA_ARGS__> :\
        (need_depth && need_distortion && !need_normal) ?\
            renderCUDA<true, true, false, __VA_ARGS__> :\
        (need_depth && !need_distortion && need_normal) ?\
            renderCUDA<true, false, true, __VA_ARGS__> :\
        (need_depth && !need_distortion && !need_normal) ?\
            renderCUDA<true, false, false, __VA_ARGS__> :\
        (!need_depth && need_distortion && need_normal) ?\
            renderCUDA<false, true, true, __VA_ARGS__> :\
        (!need_depth && need_distortion && !need_normal) ?\
            renderCUDA<false, true, false, __VA_ARGS__> :\
        (!need_depth && !need_distortion && need_normal) ?\
            renderCUDA<false, false, true, __VA_ARGS__> :\
        (!need_depth && !need_distortion && !need_normal) ?\
            renderCUDA<false, false, false, __VA_ARGS__> :\
        (need_depth && need_distortion && need_normal) ?\
            renderCUDA<true, true, true, __VA_ARGS__> :\
        (need_depth && need_distortion && !need_normal) ?\
            renderCUDA<true, true, false, __VA_ARGS__> :\
        (need_depth && !need_distortion && need_normal) ?\
            renderCUDA<true, false, true, __VA_ARGS__> :\
        (need_depth && !need_distortion && !need_normal) ?\
            renderCUDA<true, false, false, __VA_ARGS__> :\
        (!need_depth && need_distortion && need_normal) ?\
            renderCUDA<false, true, true, __VA_ARGS__> :\
        (!need_depth && need_distortion && !need_normal) ?\
            renderCUDA<false, true, false, __VA_ARGS__> :\
        (!need_depth && !need_distortion && need_normal) ?\
            renderCUDA<false, false, true, __VA_ARGS__> :\
            renderCUDA<false, false, false, __VA_ARGS__> \
    )
#endif

// Lowest-level C interface for launching the CUDA.
void render(
    const dim3 tile_grid, const dim3 block,
    const uint2* ranges,
    const uint32_t* vox_list,
    const int vox_geo_mode,
    const int density_mode,
    const int W, const int H,
    const float tan_fovx, const float tan_fovy,
    const float cx, const float cy,
    const float* c2w_matrix,
    const float3* background,

    const uint2* bboxes,
    const float3* vox_centers,
    const float* vox_lengths,
    const float* geos,
    const float3* rgbs,

    const float* out_T,
    const uint32_t* tile_last,
    const uint32_t* n_contrib,

    const float* dL_dout_color,
    const float* dL_dout_depth,
    const float* dL_dout_normal,
    const float* dL_dout_T,

    const float lambda_R_concen,
    const float* gt_color,
    const float lambda_ascending,
    const float lambda_scaling_penalty,
    const float min_voxel_size,
    const float lambda_dist,
    const bool need_depth,
    const bool need_normal,
    const float* out_D,
    const float* out_N,

    float* dL_dvox)
{
    const bool need_distortion = (lambda_dist > 0);

    // The density_mode now is always EXP_LINEAR_11_MODE
    const auto kernel_func =
        (vox_geo_mode == VOX_TRIINTERP1_MODE) ?
            BwRendFunc(1) :
        (vox_geo_mode == VOX_TRIINTERP3_MODE) ?
            BwRendFunc(3) :
            BwRendFunc(2) ;

    kernel_func <<<tile_grid, block>>> (
        ranges,
        vox_list,
        W, H,
        tan_fovx, tan_fovy,
        cx, cy,
        c2w_matrix,
        background,

        bboxes,
        vox_centers,
        vox_lengths,
        geos,
        rgbs,

        out_T,
        tile_last,
        n_contrib,

        dL_dout_color,
        dL_dout_depth,
        dL_dout_normal,
        dL_dout_T,

        lambda_R_concen,
        gt_color,
        lambda_ascending,
        lambda_scaling_penalty,
        min_voxel_size,
        lambda_dist,
        out_D,
        out_N,

        dL_dvox);
}


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

    const bool debug)
{
    if (vox_centers.ndimension() != 2 || vox_centers.size(1) != 3)
        AT_ERROR("vox_centers must have dimensions (num_points, 3)");

    const int P = vox_centers.size(0);

    if (P == 0)
    {
        torch::Tensor dL_dgeos = torch::empty({0});
        torch::Tensor dL_drgbs = torch::empty({0});
        torch::Tensor subdiv_p_bw = torch::empty({0});
        return std::make_tuple(dL_dgeos, dL_drgbs, subdiv_p_bw);
    }

    torch::Tensor dL_dvox = torch::zeros({P, geos.size(1)+3+1}, vox_centers.options());
    dim3 tile_grid((image_width + BLOCK_X - 1) / BLOCK_X, (image_height + BLOCK_Y - 1) / BLOCK_Y, 1);
    dim3 block(BLOCK_X, BLOCK_Y, 1);

    // Retrive raster state from pytorch tensor
    char* geomB_ptr = reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr());
    char* binningB_ptr = reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr());
    char* imageB_ptr = reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr());
    RASTER_STATE::GeometryState geomState = RASTER_STATE::GeometryState::fromChunk(
        geomB_ptr,
        P);
    RASTER_STATE::BinningState binningState = RASTER_STATE::BinningState::fromChunk(
        binningB_ptr,
        R);
    RASTER_STATE::ImageState imgState = RASTER_STATE::ImageState::fromChunk(
        imageB_ptr,
        image_width * image_height,
        tile_grid.x * tile_grid.y);

    // Compute loss gradients w.r.t. surface property and voxel color.
    render(
        tile_grid, block,
        imgState.ranges,
        binningState.vox_list,
        vox_geo_mode,
        density_mode,
        image_width, image_height,
        tan_fovx, tan_fovy,
        cx, cy,
        c2w_matrix.contiguous().data_ptr<float>(),
        (float3*)(background.contiguous().data_ptr<float>()),

        geomState.bboxes,
        (float3*)(vox_centers.contiguous().data_ptr<float>()),
        vox_lengths.contiguous().data_ptr<float>(),
        geos.contiguous().data_ptr<float>(),
        (float3*)(rgbs.contiguous().data_ptr<float>()),

        out_T.contiguous().data_ptr<float>(),
        imgState.tile_last,
        imgState.n_contrib,

        dL_dout_color.contiguous().data_ptr<float>(),
        dL_dout_depth.contiguous().data_ptr<float>(),
        dL_dout_normal.contiguous().data_ptr<float>(),
        dL_dout_T.contiguous().data_ptr<float>(),

        lambda_R_concen,
        gt_color.contiguous().data_ptr<float>(),
        lambda_ascending,
        lambda_scaling_penalty,
        min_voxel_size,
        lambda_dist,
        need_depth,
        need_normal,
        out_D.contiguous().data_ptr<float>(),
        out_N.contiguous().data_ptr<float>(),

        dL_dvox.contiguous().data_ptr<float>());
    CHECK_CUDA(debug);

    std::vector<torch::Tensor> gradient_lst = dL_dvox.split({geos.size(1), 3, 1}, 1);
    torch::Tensor dL_dgeos = gradient_lst[0];
    torch::Tensor dL_drgbs = gradient_lst[1];
    torch::Tensor subdiv_p_bw = gradient_lst[2];

    return std::make_tuple(dL_dgeos, dL_drgbs, subdiv_p_bw);
}

}
