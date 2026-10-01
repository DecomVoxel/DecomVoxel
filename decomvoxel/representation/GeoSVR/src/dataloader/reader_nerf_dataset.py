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

# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import json
import numpy as np
from PIL import Image
from pathlib import Path

from src.utils.camera_utils import fov2focal, focal2fov
from .colmap_loader import fetchPly
from .reader_scene_info import CameraInfo, PointCloud, SceneInfo


def parse_principle_point(info, is_cx):
    key = "cx" if is_cx else "cy"
    key_res = "w" if is_cx else "h"
    if f"{key}_p" in info:
        return info[f"{key}_p"]
    if key in info and key_res in info:
        return info[key] / info[key_res]
    return None


def read_a_camera(frame, fovx, fovy, cx_p, cy_p, path, extension, points=None, correspondent=None):
    # Guess the rgb image path and load image
    image_path = os.path.join(path, frame["file_path"] + extension)
    if not os.path.exists(image_path):
        image_path = os.path.join(path, frame["file_path"])
    image_name = Path(image_path).stem
    image = Image.open(image_path)

    # Load per-frame camera parameters
    fovx = frame.get('camera_angle_x', fovx)
    cx_p = frame.get('cx_p', cx_p)
    cy_p = frame.get('cy_p', cy_p)

    if 'camera_angle_y' in frame:
        fovy = frame['camera_angle_y']
    elif fovy <= 0:
        fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])

    # Load camera-to-world
    c2w = np.array(frame["transform_matrix"])
    # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
    c2w[:3, 1:3] *= -1

    # Compute the world-to-camera transform
    w2c = np.linalg.inv(c2w).astype(np.float32)
    
    R = np.transpose(w2c[:3, :3])
    T = np.array(w2c[:3, 3])

    # Load depth if there is
    if "depth_path" in frame:
        depth_path = os.path.join(path, frame["depth_path"])
        depth = Image.open(depth_path)
    else:
        depth_path = ""
        depth = None

    # Load mask if there is
    if "mask_path" in frame:
        mask_path = os.path.join(path, frame["mask_path"])
        mask = Image.open(mask_path)
    else:
        mask_path = ""
        mask = None

    # Load sparse point
    if points is not None:
        key = f"{image_name}.{extension}"
        sparse_pt = points[correspondent[key]]
    else:
        sparse_pt = None

    return CameraInfo(
        image_name=image_name,
        w2c=w2c,
        fovx=fovx, fovy=fovy,
        width=image.size[0], height=image.size[1],
        R=R, T=T,
        cx_p=cx_p, cy_p=cy_p,
        image=image, image_path=image_path,
        depth=depth, depth_path=depth_path,
        mask=mask, mask_path=mask_path,
        sparse_pt=sparse_pt,
    )

def read_cameras_from_json(path, transformsfile, extension=".png", points=None, correspondent=None):

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)

    fovx = contents.get("camera_angle_x", 0)
    fovy = contents.get("camera_angle_y", 0)
    cx_p = parse_principle_point(contents, is_cx=True)
    cy_p = parse_principle_point(contents, is_cx=False)
    frames = contents["frames"]

    cam_infos = [
        read_a_camera(frame, fovx, fovy, cx_p, cy_p, path, extension, points, correspondent)
        for idx, frame in enumerate(frames)]

    return cam_infos

def read_nerf_dataset(path, extension, test_every, eval):
    # Read SfM sparse points if there is
    point_cloud = None
    correspondent = None

    ply_path = os.path.join(path, "points3D.ply")
    if os.path.exists(ply_path):
        points, colors, normals = fetchPly(ply_path)
        point_cloud = PointCloud(
            points=points,
            colors=colors,
            normals=normals,
            ply_path=ply_path)
    else:
        points = None

    cor_path = os.path.join(path, "points_correspondent.json")
    if os.path.exists(cor_path):
        assert point_cloud is not None
        with open(cor_path) as f:
            correspondent = json.load(f)

    # Load train/test camera info
    if os.path.exists(os.path.join(path, "transforms_train.json")):
        train_cam_infos = read_cameras_from_json(
            path, "transforms_train.json", extension, points, correspondent)
        test_cam_infos = read_cameras_from_json(
            path, "transforms_test.json", extension, points, correspondent)
        if not eval:
            train_cam_infos.extend(test_cam_infos)
            test_cam_infos = []
    else:
        train_cam_infos = read_cameras_from_json(
            path, "transforms.json", extension, points, correspondent)
        test_cam_infos = []
        if eval:
            test_cam_infos = [c for idx, c in enumerate(train_cam_infos) if idx % test_every == 0]
            train_cam_infos = [c for idx, c in enumerate(train_cam_infos) if idx % test_every != 0]

    # Parse main scene bound
    nerf_normalization_path = os.path.join(path, "nerf_normalization.json")
    if os.path.isfile(nerf_normalization_path):
        with open(nerf_normalization_path) as f:
            nerf_normalization = json.load(f)
        suggested_center = np.array(nerf_normalization["center"], dtype=np.float32)
        suggested_radius = np.array(nerf_normalization["radius"], dtype=np.float32)
        suggested_bounding = np.stack([
            suggested_center - suggested_radius,
            suggested_center + suggested_radius,
        ])
    else:
        # Use 3DGS's setup for synthetic blender scene bound
        suggested_bounding = np.array([
            [-1.5, -1.5, -1.5],
            [1.5, 1.5, 1.5],
        ], dtype=np.float32)

    # Pack scene info
    scene_info = SceneInfo(
        train_cam_infos=train_cam_infos,
        test_cam_infos=test_cam_infos,
        suggested_bounding=suggested_bounding,
        point_cloud=point_cloud)
    return scene_info
