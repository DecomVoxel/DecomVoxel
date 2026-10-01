# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch
from svraster_cuda.meta import STEP_SZ_SCALE

def softplus(x):
    return torch.nn.functional.softplus(x)

def exp_linear_10(x):
    return torch.where(x > 1, x, torch.exp(x - 1))

def exp_linear_11(x):
    return torch.where(x > 1.1, x, torch.exp(0.909090909091 * x - 0.904689820196))

def exp_linear_20(x):
    return torch.where(x > 2.0, x, torch.exp(0.5 * x - 0.30685281944))

def softplus_inverse(y):
    return y + torch.log(-torch.expm1(-y))

def exp_linear_10_inverse(y):
    return torch.where(y > 1, y, torch.log(y) + 1)

def exp_linear_11_inverse(y):
    return torch.where(y > 1.1, y, (torch.log(y) + 0.904689820196) / 0.909090909091)

def exp_linear_20_inverse(x):
    return torch.where(y > 2.0, y, (torch.log(y) + 0.30685281944) / 0.5)

def smooth_clamp_max(x, max_val):
    return max_val - torch.nn.functional.softplus(max_val - x)

def density2alpha(density, interval):
    return 1 - torch.exp(-STEP_SZ_SCALE * interval * density)

def alpha2density(alpha, interval):
    return torch.log(1 - alpha) / (-STEP_SZ_SCALE * interval)

def rgb2shzero(x):
    return (x - 0.5) / 0.28209479177387814

def shzero2rgb(x):
    return x * 0.28209479177387814 + 0.5
