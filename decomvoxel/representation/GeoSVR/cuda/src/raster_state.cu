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

#include "raster_state.h"
#include "auxiliary.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>

#include <cooperative_groups.h>
namespace cg = cooperative_groups;


namespace RASTER_STATE {

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    // Helper function to allocate GeometryState, ImageState, and BinningState.
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
        return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

template <typename T>
static void obtain(char*& chunk, T*& ptr, std::size_t count, std::size_t alignment)
{
    std::size_t offset = (reinterpret_cast<std::uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1);
    ptr = reinterpret_cast<T*>(offset);
    chunk = reinterpret_cast<char*>(ptr + count);
}

template<typename T>
size_t required(size_t P)
{
    char* size = nullptr;
    T::fromChunk(size, P);
    return ((size_t)size) + 128;
}

template<typename T>
size_t required(size_t P, size_t Q)
{
    char* size = nullptr;
    T::fromChunk(size, P, Q);
    return ((size_t)size) + 128;
}

// Explicit template initialization
template size_t required<GeometryState>(size_t P);
template size_t required<ImageState>(size_t P, size_t Q);
template size_t required<BinningState>(size_t P);


// Given the pointer to the allocated memory,
// assign the starting address of each memory partition to their pointers.
GeometryState GeometryState::fromChunk(char*& chunk, size_t P)
{
    GeometryState geom;
    obtain(chunk, geom.n_duplicates, P, 128);
    obtain(chunk, geom.n_duplicates_scan, P, 128);
    obtain(chunk, geom.bboxes, P, 128);
    obtain(chunk, geom.cam_quadrant_bitsets, P, 128);
    // obtain(chunk, geom.rgbs, P, 128);

    // Prepare temporary space for scanning (prefix-sum).
    cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.n_duplicates, geom.n_duplicates, P);
    obtain(chunk, geom.scanning_temp_space, geom.scan_size, 128);
    return geom;
}

ImageState ImageState::fromChunk(char*& chunk, size_t N, size_t n_tiles)
{
    ImageState img;
    obtain(chunk, img.ranges, n_tiles, 128);
    obtain(chunk, img.tile_last, n_tiles, 128);
    obtain(chunk, img.n_contrib, N, 128);
    return img;
}

BinningState BinningState::fromChunk(char*& chunk, size_t P)
{
    BinningState binning;
    obtain(chunk, binning.vox_list_keys_unsorted, P, 128);
    obtain(chunk, binning.vox_list_keys, P, 128);
    obtain(chunk, binning.vox_list_unsorted, P, 128);
    obtain(chunk, binning.vox_list, P, 128);

    // Prepare temporary space for sorting.
    cub::DeviceRadixSort::SortPairs(
        nullptr, binning.sorting_size,
        binning.vox_list_keys_unsorted, binning.vox_list_keys,
        binning.vox_list_unsorted, binning.vox_list, P);
    obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
    return binning;
}

// Helper function for advance debugging.
void unpack_image_state(
    const int width, const int height,
    char* img_buffer,
    int* ranges,
    int* tile_last,
    int* n_contrib)
{
    const bool debug = true;

    dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);

    ImageState imgState = ImageState::fromChunk(img_buffer, width * height, tile_grid.x * tile_grid.y);

    cudaMemcpy(
        ranges,
        (int*)imgState.ranges,
        tile_grid.x * tile_grid.y * 2 * sizeof(int),
        cudaMemcpyDeviceToHost);
    CHECK_CUDA(debug)
    cudaMemcpy(
        tile_last,
        (int*)imgState.tile_last,
        tile_grid.x * tile_grid.y * sizeof(int),
        cudaMemcpyDeviceToHost);
    CHECK_CUDA(debug);
    cudaMemcpy(
        n_contrib,
        (int*)imgState.n_contrib,
        width * height * sizeof(int),
        cudaMemcpyDeviceToHost);
    CHECK_CUDA(debug);
}


// Export utility for debugging
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
unpack_ImageState(
    const int image_width, const int image_height,
    const torch::Tensor& imageBuffer)
{
    dim3 tile_grid((image_width + BLOCK_X - 1) / BLOCK_X, (image_height + BLOCK_Y - 1) / BLOCK_Y, 1);
    torch::Device device(torch::kCUDA);
    torch::TensorOptions options(torch::kInt32);
    torch::Tensor ranges = torch::full({tile_grid.y, tile_grid.x, 2}, 0, options.device(device));
    torch::Tensor tile_last = torch::full({tile_grid.y, tile_grid.x}, 0, options.device(device));
    torch::Tensor n_contrib = torch::full({image_height, image_width}, 0, options.device(device));

    unpack_image_state(
        image_width, image_height,
        reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
        ranges.contiguous().data_ptr<int>(),
        tile_last.contiguous().data_ptr<int>(),
        n_contrib.contiguous().data_ptr<int>()
    );
    return std::make_tuple(ranges, tile_last, n_contrib);
}

__global__ void filter_geomState_cuda(
    const int P,
    const int64_t* __restrict__ indices,
    const uint32_t* __restrict__ ori_n_duplicates,
    const uint2* __restrict__ ori_bboxes,
    const uint32_t* __restrict__ ori_cam_quadrant_bitsets,
    uint32_t* __restrict__ new_n_duplicates,
    uint2* __restrict__ new_bboxes,
    uint32_t* __restrict__ new_cam_quadrant_bitsets)
{
    const int tid = cg::this_grid().thread_rank();
    if (tid >= P)
        return;

    const int idx = indices[tid];
    new_n_duplicates[tid] = ori_n_duplicates[idx];
    new_bboxes[tid] = ori_bboxes[idx];
    new_cam_quadrant_bitsets[tid] = ori_cam_quadrant_bitsets[idx];
}

torch::Tensor filter_geomState(
    const int ori_P,
    const torch::Tensor& indices,
    const torch::Tensor& geomState)
{
    const int P = indices.size(0);

    auto t_opt_byte = torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA);
    torch::Tensor new_geomBuffer = torch::empty({0}, t_opt_byte);

    if (P == 0)
        return new_geomBuffer;

    size_t chunk_size = required<GeometryState>(P);
    new_geomBuffer.resize_({(long long)chunk_size});
    char* chunkptr = reinterpret_cast<char*>(new_geomBuffer.contiguous().data_ptr());
    GeometryState new_geomState = GeometryState::fromChunk(chunkptr, P);

    char* ori_chunkptr = reinterpret_cast<char*>(geomState.contiguous().data_ptr());
    GeometryState ori_geomState = GeometryState::fromChunk(ori_chunkptr, ori_P);

    filter_geomState_cuda <<<(P + 255) / 256, 256>>> (
        P,
        indices.contiguous().data_ptr<int64_t>(),
        ori_geomState.n_duplicates,
        ori_geomState.bboxes,
        ori_geomState.cam_quadrant_bitsets,
        new_geomState.n_duplicates,
        new_geomState.bboxes,
        new_geomState.cam_quadrant_bitsets
    );

    return new_geomBuffer;
}

}
