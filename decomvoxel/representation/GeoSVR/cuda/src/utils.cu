/*************************************************************************
Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto.  Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*************************************************************************/

#include "utils.h"
#include "auxiliary.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

namespace UTILS {

// CUDA
__global__ void is_in_cone_kernel(
    const int N,
    const float tanfovx,
    const float tanfovy,
    const float near,
    const float* __restrict__ w2c_matrix,
    const float* __restrict__ pts,
    bool* __restrict__ is_in)
{
    auto idx = cg::this_grid().thread_rank();
    if (idx >= N)
        return;

    float w2c[12];
    for (int i = 0; i < 12; i++)
        w2c[i] = w2c_matrix[i];

    const float3 pt_cam = transform_3x4(w2c, ((float3*)pts)[idx]);
    is_in[idx] = (pt_cam.z > near) && 
                 (fabs(pt_cam.x) <= tanfovx * pt_cam.z) &&
                 (fabs(pt_cam.y) <= tanfovy * pt_cam.z);
}

__global__ void compute_rd_kernel(
    const int width, const int height,
    const float cx, const float cy,
    const float tanfovx, const float tanfovy,
    const float* __restrict__ c2w_matrix,
    float* rd)
{
    const int pix_id = cg::this_grid().thread_rank();
    const int n_pix = width * height;
    if (pix_id >= n_pix)
        return;

    float c2w[12];
    for (int i = 0; i < 12; i++)
        c2w[i] = c2w_matrix[i];

    const float yid = pix_id / width;
    const float xid = pix_id % width;
    const float3 cam_rd = make_float3(
        (xid + 0.5f - cx) * 2.f * tanfovx / static_cast<float>(width),
        (yid + 0.5f - cy) * 2.f * tanfovy / static_cast<float>(height),
        1.f
    );
    const float3 dir = rotate_3x4(c2w, cam_rd);
    rd[0 * n_pix + pix_id] = dir.x;
    rd[1 * n_pix + pix_id] = dir.y;
    rd[2 * n_pix + pix_id] = dir.z;
}

__global__ void depth2pts_kernel(
    const int width, const int height,
    const float cx, const float cy,
    const float tanfovx, const float tanfovy,
    const float* __restrict__ c2w_matrix,
    const float* __restrict__ depth,
    float* pts)
{
    const int pix_id = cg::this_grid().thread_rank();
    const int n_pix = width * height;
    if (pix_id >= n_pix)
        return;

    float c2w[12];
    for (int i = 0; i < 12; i++)
        c2w[i] = c2w_matrix[i];
    const float d = depth[pix_id];

    const float yid = pix_id / width;
    const float xid = pix_id % width;
    const float3 cam_rd = make_float3(
        (xid + 0.5f - cx) * 2.f * tanfovx / static_cast<float>(width),
        (yid + 0.5f - cy) * 2.f * tanfovy / static_cast<float>(height),
        1.f
    );
    const float3 dir = rotate_3x4(c2w, cam_rd);
    pts[0 * n_pix + pix_id] = c2w[3] + d * dir.x;
    pts[1 * n_pix + pix_id] = c2w[7] + d * dir.y;
    pts[2 * n_pix + pix_id] = c2w[11] + d * dir.z;
}

__global__ void voxel_order_rank_cuda(const int P, const int64_t* octree_paths, int64_t* order_ranks)
{
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P)
        return;

    const uint64_t octree_path = octree_paths[idx];

    const int64_t rank[8] = {
        compute_order_rank(octree_path, 0),
        compute_order_rank(octree_path, 1),
        compute_order_rank(octree_path, 2),
        compute_order_rank(octree_path, 3),
        compute_order_rank(octree_path, 4),
        compute_order_rank(octree_path, 5),
        compute_order_rank(octree_path, 6),
        compute_order_rank(octree_path, 7)
    };

    for (int i = 0; i < 8; i++)
        order_ranks[i*P + idx] = rank[i];
}

__global__ void ijk_2_octpath_cuda (
    const int P,
    const int64_t* __restrict__ ijk,
    const int8_t* __restrict__ octlevel,
    int64_t* __restrict__ octpath)
{
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P)
        return;

    const int lv = octlevel[idx];
    int i = ijk[idx * 3 + 0];
    int j = ijk[idx * 3 + 1];
    int k = ijk[idx * 3 + 2];

    int64_t path = 0;
    for (int l=0; l<MAX_NUM_LEVELS; ++l)
    {
        const int64_t bits = ((i & 1) << 2) | ((j & 1) << 1) | (k & 1);
        path |= bits << (3 * l);

        // i,j,k should all be 0 after lv iterations.
        i >>= 1;
        j >>= 1;
        k >>= 1;
    }
    path <<= (3 * (MAX_NUM_LEVELS - lv));

    octpath[idx] = path;
}

__global__ void octpath_2_ijk_cuda (
    const int P,
    const int64_t* __restrict__ octpath,
    const int8_t* __restrict__ octlevel,
    int64_t* __restrict__ ijk)
{
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P)
        return;

    const int lv = octlevel[idx];
    int64_t path = octpath[idx];
    int i = 0, j = 0, k = 0;

    path >>= (3 * (MAX_NUM_LEVELS - lv));
    for (int l=0; l<MAX_NUM_LEVELS; ++l)
    {
        const int bits = static_cast<int>(path & 0b111LL);
        i |= ((bits & 0b100) >> 2) << l;
        j |= ((bits & 0b010) >> 1) << l;
        k |= ((bits & 0b001)) << l;

        // path should be 0 after lv iterations.
        path >>= 3;
    }

    ijk[idx * 3 + 0] = i;
    ijk[idx * 3 + 1] = j;
    ijk[idx * 3 + 2] = k;
}

// Interface for python
torch::Tensor is_in_cone(
    const float tanfovx,
    const float tanfovy,
    const float near,
    const torch::Tensor& w2c_matrix,
    const torch::Tensor& pts)
{
    const int N = pts.size(0);
    torch::Tensor is_in = torch::zeros({N}, pts.options().dtype(torch::kBool));
    if (N > 0)
        is_in_cone_kernel <<<(N + 255) / 256, 256>>> (
            N,
            tanfovx,
            tanfovy,
            near,
            w2c_matrix.contiguous().data_ptr<float>(),
            pts.contiguous().data_ptr<float>(),
            is_in.contiguous().data_ptr<bool>());
    return is_in;
}

torch::Tensor compute_rd(
    const int width, const int height,
    const float cx, const float cy,
    const float tanfovx, const float tanfovy,
    const torch::Tensor& c2w_matrix)
{
    auto float_opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor rd = torch::empty({3, height, width}, float_opts);
    const int n_pix = width * height;
    compute_rd_kernel <<<(n_pix + 255) / 256, 256>>> (
        width, height,
        cx, cy,
        tanfovx, tanfovy,
        c2w_matrix.contiguous().data_ptr<float>(),
        rd.contiguous().data_ptr<float>());
    return rd;
}

torch::Tensor depth2pts(
    const int width, const int height,
    const float cx, const float cy,
    const float tanfovx, const float tanfovy,
    const torch::Tensor& c2w_matrix,
    const torch::Tensor& depth)
{
    auto float_opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor pts = torch::empty({3, height, width}, float_opts);
    const int n_pix = width * height;
    depth2pts_kernel <<<(n_pix + 255) / 256, 256>>> (
        width, height,
        cx, cy,
        tanfovx, tanfovy,
        c2w_matrix.contiguous().data_ptr<float>(),
        depth.contiguous().data_ptr<float>(),
        pts.contiguous().data_ptr<float>());
    return pts;
}

torch::Tensor voxel_order_rank(const torch::Tensor& octree_paths)
{
    const int P = octree_paths.size(0);

    torch::Tensor order_ranks = torch::empty({8, P}, octree_paths.options());

    if (P > 0)
        voxel_order_rank_cuda <<<(P + 255) / 256, 256>>> (
            P,
            octree_paths.contiguous().data_ptr<int64_t>(),
            order_ranks.contiguous().data_ptr<int64_t>());

    return order_ranks;
}

torch::Tensor ijk_2_octpath(const torch::Tensor& ijk, const torch::Tensor& octlevel)
{
    const int P = octlevel.size(0);

    const auto tensor_opt = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
    torch::Tensor octpath = torch::empty({P, 1}, tensor_opt);

    if (P > 0)
        ijk_2_octpath_cuda <<<(P + 255) / 256, 256>>> (
            P,
            ijk.contiguous().data_ptr<int64_t>(),
            octlevel.contiguous().data_ptr<int8_t>(),
            octpath.contiguous().data_ptr<int64_t>());

    return octpath;
}

torch::Tensor octpath_2_ijk(const torch::Tensor& octpath, const torch::Tensor& octlevel)
{
    const int P = octlevel.size(0);

    const auto tensor_opt = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
    torch::Tensor ijk = torch::empty({P, 3}, tensor_opt);

    if (P > 0)
        octpath_2_ijk_cuda <<<(P + 255) / 256, 256>>> (
            P,
            octpath.contiguous().data_ptr<int64_t>(),
            octlevel.contiguous().data_ptr<int8_t>(),
            ijk.contiguous().data_ptr<int64_t>());

    return ijk;
}

}
