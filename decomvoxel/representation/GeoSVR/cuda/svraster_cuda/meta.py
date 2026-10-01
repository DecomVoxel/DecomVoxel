# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch
from . import _C


MAX_NUM_LEVELS = _C.MAX_NUM_LEVELS
STEP_SZ_SCALE = _C.STEP_SZ_SCALE


VoxGeoModes = dict(
    triinterp=_C.VOX_TRIINTERP_MODE,
    triinterp1=_C.VOX_TRIINTERP1_MODE,
    triinterp3=_C.VOX_TRIINTERP3_MODE,
)

DensityModes = dict(
    exp_linear_11=_C.EXP_LINEAR_11_MODE,
)

CamModes = dict(
    persp=_C.CAM_PERSP,
    ortho=_C.CAM_ORTHO,
)
