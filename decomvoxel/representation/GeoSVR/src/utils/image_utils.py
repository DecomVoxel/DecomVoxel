# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import os
import cv2
import torch
import numpy as np


def im_tensor2np(x):
    if x.shape[0] == 1:
        x = x.squeeze(0)
    if len(x.shape) == 3:
        x = x.moveaxis(0, -1)
    return x.clamp(0, 1).mul(255).cpu().numpy().astype(np.uint8)

def im_pil2tensor(x):
    return torch.from_numpy(np.array(x).astype(np.float32)).moveaxis(-1, 0) / 255

def viz_tensordepth_histeq(x, alpha_mass=None):
    '''
    Use histogram equalization for better depth visulization.
    By doing so, each color scale will have similar amout of pixels.
    The depth order is maintained but the scale do not reflect any actual distance.
    '''
    if alpha_mass is not None:
        m = (alpha_mass>0.01) & (x>0)
    else:
        m = (x>0)

    x = x.cpu().numpy()
    m = m.cpu().numpy()
    n_valid = m.sum()
    if alpha_mass is not None:
        mass = alpha_mass.cpu().numpy()[m]
    else:
        mass = np.ones([n_valid])
    order = np.argsort(x[m])
    cdf = np.cumsum(mass[order]) / mass.sum()
    hist = np.empty([n_valid])
    hist[order] = 1 + 254 * (cdf ** 2)
    x[~m] = 0
    x[m] = np.clip(hist, 1, 255)
    viz = cv2.applyColorMap(x.astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    viz = cv2.cvtColor(viz, cv2.COLOR_BGR2RGB)
    viz[~m] = 0
    return viz

def viz_tensordepth_log(x, alpha_mass=None):
    if alpha_mass is not None:
        m = (alpha_mass>0.01) & (x>0)
    else:
        m = (x>0)

    x = x.cpu().numpy()
    m = m.cpu().numpy()
    dmin, dmax = np.quantile(x[m], q=[0.03, 0.97])
    x = np.log(np.clip(1 + x - dmin, 1, 1e9))
    x = x / np.log(1 + dmax - dmin)
    x = np.clip(x, 0, 1) * 255
    viz = cv2.applyColorMap(x.astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    viz = cv2.cvtColor(viz, cv2.COLOR_BGR2RGB)
    viz[~m] = 0
    return viz


def viz_tensordepth(x, alpha_mass=None, mode='log'):
    if mode == 'histeq':
        return viz_tensordepth_histeq(x, alpha_mass)
    elif mode == 'log':
        return viz_tensordepth_log(x)
    raise NotImplementedError

def resize_rendering(render, size, mode='bilinear', align_corners=False):
    return torch.nn.functional.interpolate(
        render[None], size=size, mode=mode, align_corners=align_corners, antialias=True)[0]
