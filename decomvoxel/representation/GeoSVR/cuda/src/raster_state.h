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

#ifndef RASTER_STATE_H_INCLUDED
#define RASTER_STATE_H_INCLUDED

#include <cuda_runtime.h>
#include <torch/extension.h>

namespace RASTER_STATE {

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t);

template <typename T>
static void obtain(char*& chunk, T*& ptr, std::size_t count, std::size_t alignment);

template<typename T>
size_t required(size_t P);

template<typename T>
size_t required(size_t P, size_t Q);

struct GeometryState
{
    // Voxel duplication related variables.
    // A voxel is duplicated by the # of touched tile times the # of camera quadrants.
    // We need to calculate the prefix sum (scan) for organizing the BinningState.
    uint32_t* n_duplicates; // <==> tiles_touched
    uint32_t* n_duplicates_scan; // <==> point_offsets;
    size_t scan_size;
    char* scanning_temp_space;
    uint2* bboxes;  // The bbox region enclosing a projected voxel.

    // Voxel sorting related variables.
    // uint64_t* order_ranks; // <=> float* depths;  // The ranking of the rendering order.
    uint32_t* cam_quadrant_bitsets;  // The camera quadrants a voxel can reach.

    static GeometryState fromChunk(char*& chunk, size_t P);
};

struct ImageState
{
    uint2* ranges;
    uint32_t* tile_last;
    uint32_t* n_contrib;

    static ImageState fromChunk(char*& chunk, size_t N, size_t n_tiles);
};

struct BinningState
{
    size_t sorting_size;
    uint64_t* vox_list_keys_unsorted;
    uint64_t* vox_list_keys;
    uint32_t* vox_list_unsorted;
    uint32_t* vox_list;
    char* list_sorting_space;

    static BinningState fromChunk(char*& chunk, size_t P);
};

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
unpack_ImageState(
    const int image_width, const int image_height,
    const torch::Tensor& imageBuffer);

torch::Tensor filter_geomState(
    const int ori_P,
    const torch::Tensor& indices,
    const torch::Tensor& geomState);

}

#endif