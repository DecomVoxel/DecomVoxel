# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="svraster_cuda",
    packages=["svraster_cuda"],
    ext_modules=[
        CUDAExtension(
            name="svraster_cuda._C",
            sources=[
                "src/raster_state.cu",
                "src/preprocess.cu",
                "src/forward.cu",
                "src/backward.cu",
                "src/geo_params_gather.cu",
                "src/sh_compute.cu",
                "src/tv_compute.cu",
                "src/utils.cu",
                "src/adam_step.cu",
                "binding.cpp"
            ],
            # extra_compile_args={"nvcc": ["--use_fast_math"]},
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)
