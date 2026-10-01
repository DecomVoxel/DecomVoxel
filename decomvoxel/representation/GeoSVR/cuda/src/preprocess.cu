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

#include "preprocess.h"
#include "raster_state.h"
#include "auxiliary.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

namespace PREPROCESS {

// CUDA implementation of the preprocess step.
template <int cam_mode>
__global__ void preprocessCUDA(
    const int P,
    const int W, const int H,
    const float tan_fovx, const float tan_fovy,
    const float focal_x, const float focal_y,
    const float cx, const float cy,
    const float* __restrict__ w2c_matrix,
    const float* __restrict__ c2w_matrix,
    const float near,

    const float3* __restrict__ vox_centers,
    const float* __restrict__ vox_lengths,

    int* __restrict__ out_n_duplicates,
    uint32_t* __restrict__ n_duplicates,
    uint2* __restrict__ bboxes,
    uint32_t* __restrict__ cam_quadrant_bitsets,

    const dim3 tile_grid)
{
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P)
        return;

    // First things first.
    // Initialize the number of voxel duplication to 0.
    // We later can then skip rendering voxel with 0 duplication.
    out_n_duplicates[idx] = 0;
    n_duplicates[idx] = 0;

    // Load from global memory.
    const float3 vox_c = vox_centers[idx];
    const float vox_r = 0.5f * vox_lengths[idx];
    const float3 ro = last_col_3x4(c2w_matrix);
    float w2c[12];
    for (int i = 0; i < 12; i++)
        w2c[i] = w2c_matrix[i];

    // Near plane clipping (it's actually sphere)
    const float3 rel_pos = vox_c - ro;
    if (dot(rel_pos, rel_pos) < near * near)
        return;

    // Iterate the eight voxel corners and do the following:
    // 1. Compute bbox region of the projected voxel.
    // 2. Check if the voxel touch a camera quadrant.
    uint32_t quadrant_bitset = 0;
    float2 coord_min = {1e9f, 1e9f};
    float2 coord_max = {-1e9f, -1e9f};
    for (int i=0; i<8; ++i)
    {
        float3 shift = make_float3(
            (float)(((i&4)>>2) * 2 - 1),
            (float)(((i&2)>>1) * 2 - 1),
            (float)(((i&1)   ) * 2 - 1)
        );
        float3 world_corner = vox_c + vox_r * shift;
        float3 cam_corner = transform_3x4(w2c, world_corner);
        if (cam_corner.z < near)
            continue;

        float2 corner_coord;
        int quadrant_id;
        if (cam_mode == CAM_ORTHO)
        {
            const float3 lookat = third_col_3x4(c2w_matrix);
            corner_coord = make_float2(cam_corner.x, cam_corner.y);
            quadrant_id = compute_ray_quadrant_id(lookat);
        }
        else
        {
            const float inv_z = 1.0f / cam_corner.z;
            corner_coord = make_float2(cam_corner.x * inv_z, cam_corner.y * inv_z);
            quadrant_id = compute_corner_quadrant_id(world_corner, ro);
        }

        coord_min = min(coord_min, corner_coord);
        coord_max = max(coord_max, corner_coord);
        quadrant_bitset |= (1 << quadrant_id);
    }

    float cx_h = cx - 0.5f;
    float cy_h = cy - 0.5f;
    float2 bbox_min = {
        max(focal_x * coord_min.x + cx_h, 0.0f),
        max(focal_y * coord_min.y + cy_h, 0.0f)
    };
    float2 bbox_max = {
        min(focal_x * coord_max.x + cx_h, (float)W),
        min(focal_y * coord_max.y + cy_h, (float)H)
    };
    if (bbox_min.x > bbox_max.x || bbox_min.y > bbox_max.y)
        return; // Bbox outside image plane.

    // Squeeze bbox info into 2 uint.
    const uint2 bbox = {
        (((uint)lrintf(bbox_min.x)) << 16) | ((uint)lrintf(bbox_min.y)),
        (((uint)lrintf(bbox_max.x)) << 16) | ((uint)lrintf(bbox_max.y))
    };

    // Compute tile range.
    uint2 tile_min, tile_max;
    getBboxTileRect(bbox, tile_min, tile_max, tile_grid);
    int tiles_touched = (1 + tile_max.y - tile_min.y) * (1 + tile_max.x - tile_min.x);
    if (tiles_touched <= 0)
    {
        // TODO: remove sanity check.
        printf("tiles_touched <= 0 !???");
        __trap();
    }

    // Write back the results.
    const int quadrant_touched = __popc(quadrant_bitset);
    out_n_duplicates[idx] = tiles_touched * quadrant_touched;
    n_duplicates[idx] = tiles_touched * quadrant_touched;
    bboxes[idx] = bbox;
    cam_quadrant_bitsets[idx] = quadrant_bitset;
}


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

    const bool debug)
{
    if (vox_centers.ndimension() != 2 || vox_centers.size(1) != 3)
        AT_ERROR("vox_centers must have dimensions (num_points, 3)");

    const int P = vox_centers.size(0);

    auto t_opt_byte = torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA);
    auto t_opt_int32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);

    torch::Tensor geomBuffer = torch::empty({0}, t_opt_byte);
    torch::Tensor out_n_duplicates = torch::full({P}, 0, t_opt_int32);

    if (P == 0)
        return std::make_tuple(out_n_duplicates, geomBuffer);

    // Allocate GeometryState
    size_t chunk_size = RASTER_STATE::required<RASTER_STATE::GeometryState>(P);
    geomBuffer.resize_({(long long)chunk_size});
    char* chunkptr = reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr());
    RASTER_STATE::GeometryState geomState = RASTER_STATE::GeometryState::fromChunk(chunkptr, P);

    // Derive arguments
    dim3 tile_grid((image_width + BLOCK_X - 1) / BLOCK_X, (image_height + BLOCK_Y - 1) / BLOCK_Y, 1);
    const float focal_x = 0.5f * image_width / tan_fovx;
    const float focal_y = 0.5f * image_height / tan_fovy;

    // Lanching CUDA
    const auto kernel_func =
        (cam_mode == CAM_ORTHO) ?
            preprocessCUDA<CAM_ORTHO> :
            preprocessCUDA<CAM_PERSP> ;

    kernel_func <<<(P + 255) / 256, 256>>> (
        P,
        image_width, image_height,
        tan_fovx, tan_fovy,
        focal_x, focal_y,
        cx, cy,
        w2c_matrix.contiguous().data_ptr<float>(),
        c2w_matrix.contiguous().data_ptr<float>(),
        near,

        (float3*)(vox_centers.contiguous().data_ptr<float>()),
        vox_lengths.contiguous().data_ptr<float>(),

        out_n_duplicates.contiguous().data_ptr<int>(),
        geomState.n_duplicates,
        geomState.bboxes,
        geomState.cam_quadrant_bitsets,

        tile_grid);
    CHECK_CUDA(debug);

    return std::make_tuple(out_n_duplicates, geomBuffer);
}

}
