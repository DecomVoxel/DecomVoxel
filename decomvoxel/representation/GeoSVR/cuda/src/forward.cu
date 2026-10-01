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

#include "forward.h"
#include "raster_state.h"
#include "auxiliary.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>

#include <cooperative_groups.h>
namespace cg = cooperative_groups;


namespace FORWARD {

// CUDA sparse voxel rendering.
template <bool need_feat, bool need_depth, bool need_distortion, bool need_normal, bool track_max_w,
          int n_samp>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
    const uint2* __restrict__ ranges,
    const uint32_t* __restrict__ vox_list,
    int W, int H,
    const float tan_fovx, const float tan_fovy,
    const float cx, const float cy,
    const float* __restrict__ c2w_matrix,
    const float3* __restrict__ background,
    const int cam_mode,

    const uint2* __restrict__ bboxes,
    const float3* __restrict__ vox_centers,
    const float* __restrict__ vox_lengths,
    const float* __restrict__ geos,
    const float3* __restrict__ rgbs,

    uint32_t* __restrict__ tile_last,
    uint32_t* __restrict__ n_contrib,

    float* __restrict__ out_color,
    float* __restrict__ out_depth,
    float* __restrict__ out_normal,
    float* __restrict__ out_T,
    float* __restrict__ max_w,

    const int feat_dim,
    const float* __restrict__ feats,
    float* __restrict__ out_feat)
{
    // Identify current tile and associated min/max pixel range.
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
    float3 ro, rd, rd_inv;
    float rd_norm_inv;
    if (cam_mode == CAM_ORTHO)
    {
        const float3 lookat = third_col_3x4(c2w_matrix);
        rd = lookat;
        rd_inv = {1.f/ rd.x, 1.f / rd.y, 1.f / rd.z};
        rd_norm_inv = 1.f;

        const float3 cam_rd = compute_ray_d(pixf, W, H, tan_fovx, tan_fovy, cx, cy);
        const float3 cam_ro = make_float3(cam_rd.x, cam_rd.y, 0.f);
        ro = transform_3x4(c2w_matrix, cam_ro);
    }
    else
    {
        const float3 cam_rd = compute_ray_d(pixf, W, H, tan_fovx, tan_fovy, cx, cy);
        const float rd_norm = sqrtf(dot(cam_rd, cam_rd));
        const float3 rd_raw = rotate_3x4(c2w_matrix, cam_rd);
        rd_norm_inv = 1.f / rd_norm;
        ro = last_col_3x4(c2w_matrix);
        rd = rd_raw * rd_norm_inv;
        rd_inv = {1.f/ rd.x, 1.f / rd.y, 1.f / rd.z};
    }

    const uint32_t pix_quad_id = compute_ray_quadrant_id(rd);

    // Check if this thread is associated with a valid pixel or outside.
    bool inside = (pix.x < W) && (pix.y < H);
    // Done threads can help with fetching, but don't rasterize
    bool done = !inside;

    // Load start/end range of IDs to process in BinningState.
    uint2 range = ranges[tile_id];
    const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
    int toDo = range.y - range.x;

    // Init the last non-occluded range index of the tile.
    if (thread_id == 0)
        tile_last[tile_id] = range.x;

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

    // Initialize helper variables.
    float T = 1.f;
    uint32_t contributor = 0;
    uint32_t last_contributor = 0;
    float3 C = {0.f, 0.f, 0.f};
    float3 N = {0.f, 0.f, 0.f};
    float D = 0.f;
    int D_med_vox_id = -1;
    float D_med_T;
    float D_med = 0.f;
    float Ddist = 0.f;
    int j_lst[BLOCK_SIZE];

    float feat[MAX_FEAT_DIM] = {0.f};

    // Iterate over batches until all done or range is complete.
    for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
    {
        // End if entire block votes that it is done rasterizing.
        int num_done = __syncthreads_count(done);
        if (num_done == BLOCK_SIZE)
            break;

        // Collectively fetch batch of voxel data from global to shared.
        int progress = i * BLOCK_SIZE + thread_id;
        if (range.x + progress < range.y)
        {
            uint32_t order_val = vox_list[range.x + progress];
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

        // Iterate over current batch.
        const int end_j = min(BLOCK_SIZE, toDo);
        int j_lst_top = -1;
        for (int j = 0; !done && j < end_j; j++)
        {
            // Check if the pixel in the projected bbox region.
            // Check if the quadrant id match the pixel.
            if (!pix_in_bbox(pix, collected_bbox[j]) || pix_quad_id != collected_quad_id[j])
                continue;

            // Compute ray aabb intersection
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

        int contributor_inc = 0;
        for (int jj = 0; !done && jj <= j_lst_top; jj++)
        {
            int j = j_lst[jj];
            const int vox_id = collected_vox_id[j];

            // Keep track of current position in range.
            contributor_inc = j + 1;

            // Compute ray aabb intersection
            const float3 vox_c = collected_vox_c[j];
            const float vox_l = collected_vox_l[j];
            const float2 ab = ray_aabb(vox_c, vox_l, ro, rd_inv);
            const float a = ab.x;
            const float b = ab.y;

            float geo_params[8];
            for (int k=0; k<8; ++k)
                geo_params[k] = collected_geo_params[j*8 + k];

            // Compute volume density
            float vol_int = 0.f;
            float interp_w[8];
            float local_alphas[n_samp];

            // Quadrature integral from trilinear sampling.
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
            }

            // Compute alpha from volume integral.
            float alpha = min(MAX_ALPHA, 1.f - expf(-vol_int));
            if (alpha < MIN_ALPHA)
                continue;

            // Accumulate to the pixel.
            float pt_w = T * alpha;
            C = C + pt_w * collected_rgb[j];

            if (need_feat)
            {
                for (int k=0; k<feat_dim; ++k)
                    feat[k] += pt_w * feats[vox_id*feat_dim + k];
            }

            if (need_depth)
            {
                // Mean depth
                float dval;
                if (n_samp == 3)
                {
                    float step_sz = 0.3333333f * (b - a);
                    float a0 = local_alphas[0], a1 = local_alphas[1], a2 = local_alphas[2];
                    float t0 = a + 0.5f * step_sz;
                    float t1 = a + 1.5f * step_sz;
                    float t2 = a + 2.5f * step_sz;
                    dval = a0*t0 + (1.f-a0)*a1*t1 + (1.f-a0)*(1.f-a1)*a2*t2;
                }
                else if (n_samp == 2)
                {
                    float step_sz = 0.5f * (b - a);
                    float a0 = local_alphas[0], a1 = local_alphas[1];
                    float t0 = a + 0.5f * step_sz;
                    float t1 = a + 1.5f * step_sz;
                    dval = a0*t0 + (1.f-a0)*a1*t1;
                }
                else
                {
                    dval = alpha * 0.5f * (a + b);
                }
                D = D + T * dval;

                // Median depth
                if (T > 0.5f)
                {
                    D_med_vox_id = vox_id;
                    D_med_T = T;
                }
            }

            // Distortion depth
            if (need_distortion)
                Ddist = Ddist + pt_w * 0.5f * (depth_contracted(a) + depth_contracted(b));

            // Normal
            if (need_normal)
            {
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
                N = N + pt_w * r_lin * lin_n;
            }

            T *= (1.f - alpha);
            done |= (T < EARLY_STOP_T);

            // Keep track of last range entry to update this pixel.
            last_contributor = contributor + contributor_inc;

            // Keep track of the maxiumum importance weight of each voxel.
            if (track_max_w)
                atomicMax(((int*)max_w) + vox_id, *((int*)(&pt_w)));
        }
        contributor += done ? contributor_inc : end_j;
    }

    if (need_depth && inside && D_med_vox_id != -1)
    {
        // Finest sampling of median depth
        const int n_samp_dmed = 16;

        float3 vox_c = vox_centers[D_med_vox_id];
        float vox_l = vox_lengths[D_med_vox_id];
        float geo_params[8];
        for (int k=0; k<8; ++k)
            geo_params[k] = geos[D_med_vox_id*8 + k];
        const float2 ab = ray_aabb(vox_c, vox_l, ro, rd_inv);
        const float a = ab.x;
        const float b = ab.y;

        float vox_l_inv = 1.f / vox_l;
        const float step_sz = (b - a) * (1.f / n_samp_dmed);
        const float3 step = step_sz * rd;
        float3 pt = ro + (a + 0.5f * step_sz) * rd;
        float3 qt = (pt - (vox_c - 0.5f * vox_l)) * vox_l_inv;
        const float3 qt_step = step * vox_l_inv;

        D_med = a - 0.5f * step_sz;
        for (int k=0; k<n_samp_dmed && D_med_T > 0.5f; k++, qt=qt+qt_step)
        {
            D_med += step_sz;

            float interp_w[8];
            tri_interp_weight(qt, interp_w);
            float d = 0.f;
            for (int iii=0; iii<8; ++iii)
                d += geo_params[iii] * interp_w[iii];

            const float vol_int = STEP_SZ_SCALE * step_sz * exp_linear_11(d);

            D_med_T *= expf(-vol_int);
        }
    }

    // All threads that treat valid pixel write out their final
    // rendering data to the frame and auxiliary buffers.
    if (inside)
    {
        const float3 bg_color = *(background);

        n_contrib[pix_id] = last_contributor;
        out_color[0 * H * W + pix_id] = C.x + T * bg_color.x;
        out_color[1 * H * W + pix_id] = C.y + T * bg_color.y;
        out_color[2 * H * W + pix_id] = C.z + T * bg_color.z;
        out_T[pix_id] = T;  // Equal to (1 - alpha).
        if (need_feat)
        {
            for (int k=0; k<feat_dim; ++k)
                out_feat[k * H * W + pix_id] = feat[k];
        }
        if (need_depth)
        {
            out_depth[pix_id] = D * rd_norm_inv;
            out_depth[H * W * 2 + pix_id] = D_med * rd_norm_inv;
        }
        if (need_distortion)
        {
            out_depth[H * W + pix_id] = Ddist;
        }
        if (need_normal)
        {
            out_normal[0 * H * W + pix_id] = N.x;
            out_normal[1 * H * W + pix_id] = N.y;
            out_normal[2 * H * W + pix_id] = N.z;
        }
        atomicMax(tile_last + tile_id, range.x + last_contributor);
    }
}


#ifndef FwRendFunc
// Dirty trick. The argument name must be aligned with FORWARD::render.
#define FwRendFunc(...) \
    ( \
        (need_feat && need_depth && need_distortion && need_normal && track_max_w) ?\
            renderCUDA<true, true, true, true, true, __VA_ARGS__> :\
        (need_feat && need_depth && need_distortion && need_normal && !track_max_w) ?\
            renderCUDA<true, true, true, true, false, __VA_ARGS__> :\
        (need_feat && need_depth && need_distortion && !need_normal && track_max_w) ?\
            renderCUDA<true, true, true, false, true, __VA_ARGS__> :\
        (need_feat && need_depth && need_distortion && !need_normal && !track_max_w) ?\
            renderCUDA<true, true, true, false, false, __VA_ARGS__> :\
        (need_feat && need_depth && !need_distortion && need_normal && track_max_w) ?\
            renderCUDA<true, true, false, true, true, __VA_ARGS__> :\
        (need_feat && need_depth && !need_distortion && need_normal && !track_max_w) ?\
            renderCUDA<true, true, false, true, false, __VA_ARGS__> :\
        (need_feat && need_depth && !need_distortion && !need_normal && track_max_w) ?\
            renderCUDA<true, true, false, false, true, __VA_ARGS__> :\
        (need_feat && need_depth && !need_distortion && !need_normal && !track_max_w) ?\
            renderCUDA<true, true, false, false, false, __VA_ARGS__> :\
        (need_feat && !need_depth && need_distortion && need_normal && track_max_w) ?\
            renderCUDA<true, false, true, true, true, __VA_ARGS__> :\
        (need_feat && !need_depth && need_distortion && need_normal && !track_max_w) ?\
            renderCUDA<true, false, true, true, false, __VA_ARGS__> :\
        (need_feat && !need_depth && need_distortion && !need_normal && track_max_w) ?\
            renderCUDA<true, false, true, false, true, __VA_ARGS__> :\
        (need_feat && !need_depth && need_distortion && !need_normal && !track_max_w) ?\
            renderCUDA<true, false, true, false, false, __VA_ARGS__> :\
        (need_feat && !need_depth && !need_distortion && need_normal && track_max_w) ?\
            renderCUDA<true, false, false, true, true, __VA_ARGS__> :\
        (need_feat && !need_depth && !need_distortion && need_normal && !track_max_w) ?\
            renderCUDA<true, false, false, true, false, __VA_ARGS__> :\
        (need_feat && !need_depth && !need_distortion && !need_normal && track_max_w) ?\
            renderCUDA<true, false, false, false, true, __VA_ARGS__> :\
        (need_feat && !need_depth && !need_distortion && !need_normal && !track_max_w) ?\
            renderCUDA<true, false, false, false, false, __VA_ARGS__> :\
        (!need_feat && need_depth && need_distortion && need_normal && track_max_w) ?\
            renderCUDA<false, true, true, true, true, __VA_ARGS__> :\
        (!need_feat && need_depth && need_distortion && need_normal && !track_max_w) ?\
            renderCUDA<false, true, true, true, false, __VA_ARGS__> :\
        (!need_feat && need_depth && need_distortion && !need_normal && track_max_w) ?\
            renderCUDA<false, true, true, false, true, __VA_ARGS__> :\
        (!need_feat && need_depth && need_distortion && !need_normal && !track_max_w) ?\
            renderCUDA<false, true, true, false, false, __VA_ARGS__> :\
        (!need_feat && need_depth && !need_distortion && need_normal && track_max_w) ?\
            renderCUDA<false, true, false, true, true, __VA_ARGS__> :\
        (!need_feat && need_depth && !need_distortion && need_normal && !track_max_w) ?\
            renderCUDA<false, true, false, true, false, __VA_ARGS__> :\
        (!need_feat && need_depth && !need_distortion && !need_normal && track_max_w) ?\
            renderCUDA<false, true, false, false, true, __VA_ARGS__> :\
        (!need_feat && need_depth && !need_distortion && !need_normal && !track_max_w) ?\
            renderCUDA<false, true, false, false, false, __VA_ARGS__> :\
        (!need_feat && !need_depth && need_distortion && need_normal && track_max_w) ?\
            renderCUDA<false, false, true, true, true, __VA_ARGS__> :\
        (!need_feat && !need_depth && need_distortion && need_normal && !track_max_w) ?\
            renderCUDA<false, false, true, true, false, __VA_ARGS__> :\
        (!need_feat && !need_depth && need_distortion && !need_normal && track_max_w) ?\
            renderCUDA<false, false, true, false, true, __VA_ARGS__> :\
        (!need_feat && !need_depth && need_distortion && !need_normal && !track_max_w) ?\
            renderCUDA<false, false, true, false, false, __VA_ARGS__> :\
        (!need_feat && !need_depth && !need_distortion && need_normal && track_max_w) ?\
            renderCUDA<false, false, false, true, true, __VA_ARGS__> :\
        (!need_feat && !need_depth && !need_distortion && need_normal && !track_max_w) ?\
            renderCUDA<false, false, false, true, false, __VA_ARGS__> :\
        (!need_feat && !need_depth && !need_distortion && !need_normal && track_max_w) ?\
            renderCUDA<false, false, false, false, true, __VA_ARGS__> :\
            renderCUDA<false, false, false, false, false, __VA_ARGS__> \
    )
#endif


// Lowest-level C interface for launching the CUDA.
void render(
    const dim3 tile_grid, const dim3 block,
    const uint2* ranges,
    const uint32_t* vox_list,
    const int vox_geo_mode,
    const int density_mode,
    int W, int H,
    const float tan_fovx, const float tan_fovy,
    const float cx, const float cy,
    const float* c2w_matrix,
    const float3* background,
    const int cam_mode,
    const bool need_depth,
    const bool need_distortion,
    const bool need_normal,

    const uint2* bboxes,
    const float3* vox_centers,
    const float* vox_lengths,
    const float* geos,
    const float3* rgbs,

    uint32_t* tile_last,
    uint32_t* n_contrib,

    float* out_color,
    float* out_depth,
    float* out_normal,
    float* out_T,
    float* max_w,

    const int feat_dim,
    const float* feats,
    float* out_feat)
{
    const bool need_feat = (feat_dim > 0);
    const bool track_max_w = (max_w != nullptr);

    // The density_mode now is always EXP_LINEAR_11_MODE
    const auto kernel_func =
        (vox_geo_mode == VOX_TRIINTERP1_MODE) ?
            FwRendFunc(1) :
        (vox_geo_mode == VOX_TRIINTERP3_MODE) ?
            FwRendFunc(3) :
            FwRendFunc(2) ;

    kernel_func <<<tile_grid, block>>> (
        ranges,
        vox_list,
        W, H,
        tan_fovx, tan_fovy,
        cx, cy,
        c2w_matrix,
        background,
        cam_mode,

        bboxes,
        vox_centers,
        vox_lengths,
        geos,
        rgbs,

        tile_last,
        n_contrib,

        out_color,
        out_depth,
        out_normal,
        out_T,
        max_w,

        feat_dim,
        feats,
        out_feat);
}


// Helper function to find the next-highest bit of the MSB on the CPU.
uint32_t getHigherMsb(uint32_t n)
{
    uint32_t msb = sizeof(n) * 4;
    uint32_t step = msb;
    while (step > 1)
    {
        step /= 2;
        if (n >> msb)
            msb += step;
        else
            msb -= step;
    }
    if (n >> msb)
        msb++;
    return msb;
}

// Duplicate each voxel by #tiles x #cam_quadrant it touches.
__global__ void duplicateWithKeys(
    int P,
    const int64_t* octree_paths,
    const uint2* bboxes,
    const uint32_t* cam_quadrant_bitsets,
    const uint32_t* n_duplicates,
    const uint32_t* n_duplicates_scan,
    uint64_t* vox_list_keys_unsorted,
    uint32_t* vox_list_unsorted,
    dim3 grid)
{
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P || n_duplicates[idx] == 0)
        return;

    // Find this voxel's array offset in buffer for writing the key/value.
    uint32_t off = (idx == 0) ? 0 : n_duplicates_scan[idx - 1];
    uint2 tile_min, tile_max;
    getBboxTileRect(bboxes[idx], tile_min, tile_max, grid);

    // For each tile that the bounding rect overlaps, emit a key/value pair.
    // The key bit structure is [  tile ID  |  order_rank  ],
    // so the voxels are first sorted by tile and then by order_ranks.
    // The value bit structure is [  quadrant ID  |  voxel ID  ].
    const uint64_t octree_path = octree_paths[idx];
    uint32_t quadrant_bitsets = cam_quadrant_bitsets[idx];
    for (int quadrant_id = 0; quadrant_id < 8; quadrant_id++)
    {
        if ((quadrant_bitsets & (1 << quadrant_id)) == 0)
            continue;

        // Compute order_rank for the voxel in this quadrant.
        uint64_t order_rank = compute_order_rank(octree_path, quadrant_id);

        // Duplicate result to touched tiles.
        for (int y = tile_min.y; y <= tile_max.y; y++)
        {
            for (int x = tile_min.x; x <= tile_max.x; x++)
            {
                uint64_t tile_id = y * grid.x + x;
                vox_list_keys_unsorted[off] = encode_order_key(tile_id, order_rank);
                vox_list_unsorted[off] = encode_order_val(idx, quadrant_id);
                off++;
            }
        }
    }

    if (off != n_duplicates_scan[idx])
    {
        // TODO: remove sanity check.
        printf("Number of duplication mismatch !???");
        __trap();
    }
}

// The sorted vox_list_keys is now as:
//   [--sorted voxels for tile 1--  --sorted voxels for tile 2--  ...]
// We want to identify the start/end index of each tile from this list.
__global__ void identifyTileRanges(int L, uint64_t* vox_list_keys, uint2* ranges)
{
    auto idx = cg::this_grid().thread_rank();
    if (idx >= L)
        return;

    // Read tile ID from key. Update start/end of tile range if at limit.
    uint64_t key = vox_list_keys[idx];
    uint32_t currtile = key >> NUM_BIT_ORDER_RANK;
    if (idx == 0)
        ranges[currtile].x = 0;
    else
    {
        uint32_t prevtile = vox_list_keys[idx - 1] >> NUM_BIT_ORDER_RANK;
        if (currtile != prevtile)
        {
            ranges[prevtile].y = idx;
            ranges[currtile].x = idx;
        }
    }
    if (idx == L - 1)
        ranges[currtile].y = L;
}

// Mid-level C interface for the entire rasterization procedure.
int rasterize_voxels_procedure(
    char* geom_buffer,
    std::function<char* (size_t)> binningBuffer,
    std::function<char* (size_t)> imageBuffer,
    const int P,
    const int vox_geo_mode,
    const int density_mode,
    const int width, const int height,
    const float tan_fovx, const float tan_fovy,
    const float cx, float cy,
    const float* w2c_matrix,
    const float* c2w_matrix,
    const float* background,
    const int cam_mode,
    const bool need_depth,
    const bool need_distortion,
    const bool need_normal,

    const int64_t* octree_paths,
    const float* vox_centers,
    const float* vox_lengths,
    const float* geos,
    const float* rgbs,

    float* out_color,
    float* out_depth,
    float* out_normal,
    float* out_T,
    float* max_w,

    const int feat_dim,
    const float* feats,
    float* out_feat,

    bool debug)
{
    dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
    dim3 block(BLOCK_X, BLOCK_Y, 1);

    // Recover the preprocessing results.
    RASTER_STATE::GeometryState geomState = RASTER_STATE::GeometryState::fromChunk(geom_buffer, P);

    // Dynamically resize image-based auxiliary buffers during training.
    size_t img_chunk_size = RASTER_STATE::required<RASTER_STATE::ImageState>(width * height, tile_grid.x * tile_grid.y);
    char* img_chunkptr = imageBuffer(img_chunk_size);
    RASTER_STATE::ImageState imgState = RASTER_STATE::ImageState::fromChunk(img_chunkptr, width * height, tile_grid.x * tile_grid.y);

    // Compute prefix sum over full list of the number of voxel duplications.
    cub::DeviceScan::InclusiveSum(
        geomState.scanning_temp_space,
        geomState.scan_size,
        geomState.n_duplicates,
        geomState.n_duplicates_scan,
        P);
    CHECK_CUDA(debug);

    // Retrieve total number of voxels after duplication.
    int num_rendered;
    cudaMemcpy(
        &num_rendered,
        geomState.n_duplicates_scan + P - 1,
        sizeof(int),
        cudaMemcpyDeviceToHost);
    CHECK_CUDA(debug);

    size_t binning_chunk_size = RASTER_STATE::required<RASTER_STATE::BinningState>(num_rendered);
    char* binning_chunkptr = binningBuffer(binning_chunk_size);
    RASTER_STATE::BinningState binningState = RASTER_STATE::BinningState::fromChunk(binning_chunkptr, num_rendered);

    // For each voxel to be rendered, produce adequate [ tile ID | rank ] key
    // and the corresponding dublicated voxel [ quadrant ID | voxel ID ] to be sorted.
    duplicateWithKeys <<<(P + 255) / 256, 256>>> (
        P,
        octree_paths,
        geomState.bboxes,
        geomState.cam_quadrant_bitsets,
        geomState.n_duplicates,
        geomState.n_duplicates_scan,
        binningState.vox_list_keys_unsorted,
        binningState.vox_list_unsorted,
        tile_grid);
    CHECK_CUDA(debug);

    int bit = getHigherMsb(tile_grid.x * tile_grid.y);

    // Sort complete list of (duplicated) ID by keys.
    cub::DeviceRadixSort::SortPairs(
        binningState.list_sorting_space,
        binningState.sorting_size,
        binningState.vox_list_keys_unsorted, binningState.vox_list_keys,
        binningState.vox_list_unsorted, binningState.vox_list,
        num_rendered, 0, NUM_BIT_ORDER_RANK + bit);
    CHECK_CUDA(debug);

    cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2));
    CHECK_CUDA(debug);

    // Identify start and end of per-tile workloads in sorted list.
    if (num_rendered > 0)
    {
        identifyTileRanges <<<(num_rendered + 255) / 256, 256>>> (
            num_rendered,
            binningState.vox_list_keys,
            imgState.ranges);
        CHECK_CUDA(debug);
    }

    // Let each tile blend its range of voxels independently in parallel.
    render(
        tile_grid, block,
        imgState.ranges,
        binningState.vox_list,
        vox_geo_mode,
        density_mode,
        width, height,
        tan_fovx, tan_fovy,
        cx, cy,
        c2w_matrix,
        (float3*)background,
        cam_mode,
        need_depth,
        need_distortion,
        need_normal,

        geomState.bboxes,
        (float3*)vox_centers,
        vox_lengths,
        geos,
        (float3*)rgbs,

        imgState.tile_last,
        imgState.n_contrib,

        out_color,
        out_depth,
        out_normal,
        out_T,
        max_w,

        feat_dim,
        feats,
        out_feat);
    CHECK_CUDA(debug);

    return num_rendered;
}


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

    const bool debug)
{
    if (vox_centers.ndimension() != 2 || vox_centers.size(1) != 3)
        AT_ERROR("vox_centers must have dimensions (num_points, 3)");
    if (rgbs.ndimension() != 2 || rgbs.size(1) != 3)
        AT_ERROR("rgbs should be either (num_points, 3)");
    if (feats.ndimension() != 2)
        AT_ERROR("feats should be either (num_points, n_dim)");
    if (feats.size(1) > MAX_FEAT_DIM)
        AT_ERROR("feats dimension out of maximum");
    if (vox_centers.size(0) != rgbs.size(0))
        AT_ERROR("size mismatch");

    const int P = vox_centers.size(0);
    const int H = image_height;
    const int W = image_width;

    auto float_opts = vox_centers.options().dtype(torch::kFloat32);

    torch::Tensor out_color = torch::full({3, H, W}, 0.f, float_opts);
    torch::Tensor out_depth = need_depth || need_distortion ? torch::full({3, H, W}, 0.f, float_opts) : torch::empty({0});
    torch::Tensor out_normal = need_normal ? torch::full({3, H, W}, 0.f, float_opts) : torch::empty({0});
    torch::Tensor out_T = torch::full({1, H, W}, 0.f, float_opts);
    torch::Tensor max_w = track_max_w ? torch::full({P, 1}, 0.f, float_opts) : torch::empty({0});

    const int feat_dim = feats.size(1);
    torch::Tensor out_feat = torch::full({feat_dim, H, W}, 0.f, float_opts);

    torch::Device device(torch::kCUDA);
    torch::TensorOptions options(torch::kByte);
    torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
    torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
    std::function<char*(size_t)> binningFunc = RASTER_STATE::resizeFunctional(binningBuffer);
    std::function<char*(size_t)> imgFunc = RASTER_STATE::resizeFunctional(imgBuffer);

    float* max_w_pointer = nullptr;
    if (track_max_w)
        max_w_pointer = max_w.contiguous().data_ptr<float>();

    int rendered = 0;
    if(P != 0)
        rendered = rasterize_voxels_procedure(
            reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
            binningFunc,
            imgFunc,
            P,
            vox_geo_mode,
            density_mode,

            W, H,
            tan_fovx, tan_fovy,
            cx, cy,
            w2c_matrix.contiguous().data_ptr<float>(),
            c2w_matrix.contiguous().data_ptr<float>(),
            background.contiguous().data_ptr<float>(),
            cam_mode,
            need_depth,
            need_distortion,
            need_normal,

            octree_paths.contiguous().data_ptr<int64_t>(),
            vox_centers.contiguous().data_ptr<float>(),
            vox_lengths.contiguous().data_ptr<float>(),
            geos.contiguous().data_ptr<float>(),
            rgbs.contiguous().data_ptr<float>(),

            out_color.contiguous().data_ptr<float>(),
            out_depth.contiguous().data_ptr<float>(),
            out_normal.contiguous().data_ptr<float>(),
            out_T.contiguous().data_ptr<float>(),
            max_w_pointer,

            feat_dim,
            feats.contiguous().data_ptr<float>(),
            out_feat.contiguous().data_ptr<float>(),

            debug);

    return std::make_tuple(rendered, binningBuffer, imgBuffer, out_color, out_depth, out_normal, out_T, max_w, out_feat);
}

}
