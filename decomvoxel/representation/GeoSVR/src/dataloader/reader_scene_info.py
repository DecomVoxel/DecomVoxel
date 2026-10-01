#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import PIL
import numpy as np
from typing import List, NamedTuple

class CameraInfo(NamedTuple):
    image_name: str

    # Camera parameters
    w2c: np.array
    fovx: np.array
    fovy: np.array
    width: int
    height: int
    
    R: np.array = None
    T: np.array = None

    # (cx, cy) = (cx_p * width, cy_p * height)
    cx_p: float = None
    cy_p: float = None

    # Frame info
    image: PIL.Image.Image = None
    image_path: str = ""
    depth: PIL.Image.Image = None
    depth_path: str = ""
    mask: PIL.Image.Image = None
    mask_path: str = ""

    sparse_pt: np.array = None


class PointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array
    ply_path: str


class SceneInfo(NamedTuple):
    train_cam_infos: List[CameraInfo]
    test_cam_infos: List[CameraInfo]
    suggested_bounding: np.array
    point_cloud: PointCloud
