/*************************************************************************
Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto.  Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*************************************************************************/

#include <torch/extension.h>
#include "src/config.h"
#include "src/raster_state.h"
#include "src/preprocess.h"
#include "src/forward.h"
#include "src/backward.h"
#include "src/sh_compute.h"
#include "src/tv_compute.h"
#include "src/geo_params_gather.h"
#include "src/utils.h"
#include "src/adam_step.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rasterize_preprocess", &PREPROCESS::rasterize_preprocess);
    m.def("rasterize_voxels", &FORWARD::rasterize_voxels);
    m.def("rasterize_voxels_backward", &BACKWARD::rasterize_voxels_backward);
    m.def("filter_geomState", &RASTER_STATE::filter_geomState);
    m.def("unpack_ImageState", &RASTER_STATE::unpack_ImageState);

    m.def("gather_triinterp_geo_params", &GEO_PARAMS_GATHER::gather_triinterp_geo_params);
    m.def("gather_triinterp_geo_params_bw", &GEO_PARAMS_GATHER::gather_triinterp_geo_params_bw);
    m.def("gather_triinterp_feat_params", &GEO_PARAMS_GATHER::gather_triinterp_feat_params);
    m.def("gather_triinterp_feat_params_bw", &GEO_PARAMS_GATHER::gather_triinterp_feat_params_bw);

    m.def("sh_compute", &SH_COMPUTE::sh_compute);
    m.def("sh_compute_bw", &SH_COMPUTE::sh_compute_bw);

    m.def("total_variation_bw", &TV_COMPUTE::total_variation_bw);

    m.def("is_in_cone", &UTILS::is_in_cone);
    m.def("compute_rd", &UTILS::compute_rd);
    m.def("depth2pts", &UTILS::depth2pts);
    m.def("voxel_order_rank", &UTILS::voxel_order_rank);
    m.def("ijk_2_octpath", &UTILS::ijk_2_octpath);
    m.def("octpath_2_ijk", &UTILS::octpath_2_ijk);

    m.def("unbiased_adam_step", &ADAM_STEP::unbiased_adam_step);
    m.def("biased_adam_step", &ADAM_STEP::biased_adam_step);

    // Some readonly constant
    m.attr("MAX_NUM_LEVELS") = pybind11::int_(MAX_NUM_LEVELS);
    m.attr("STEP_SZ_SCALE") = pybind11::float_(STEP_SZ_SCALE);

    m.attr("VOX_TRIINTERP_MODE") = pybind11::int_(VOX_TRIINTERP_MODE);
    m.attr("VOX_TRIINTERP1_MODE") = pybind11::int_(VOX_TRIINTERP1_MODE);
    m.attr("VOX_TRIINTERP3_MODE") = pybind11::int_(VOX_TRIINTERP3_MODE);

    m.attr("EXP_LINEAR_11_MODE") = pybind11::int_(EXP_LINEAR_11_MODE);

    m.attr("CAM_PERSP") = pybind11::int_(CAM_PERSP);
    m.attr("CAM_ORTHO") = pybind11::int_(CAM_ORTHO);
}
