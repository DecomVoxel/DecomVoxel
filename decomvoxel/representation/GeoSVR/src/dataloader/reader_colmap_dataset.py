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
import natsort
import numpy as np
from PIL import Image
from pathlib import Path

from src.utils.camera_utils import focal2fov
from .reader_scene_info import CameraInfo, PointCloud, SceneInfo
from .colmap_loader import read_extrinsics_text, read_intrinsics_text, \
                           read_extrinsics_binary, read_intrinsics_binary, \
                           read_points3D_binary, read_points3D_text, \
                           read_colmap_ply, qvec2rotmat


def read_cameras_from_colmap(cam_extrinsics, cam_intrinsics, images_folder, points, correspondent, depth_paths):

    print(f"images_folder={images_folder}")

    # Sort cameras
    keys = natsort.natsorted(
        cam_extrinsics.keys(),
        key = lambda i : cam_extrinsics[i].name)

    # Parse paths to depth map if given
    if depth_paths:
        depth_paths = natsort.natsorted(glob.glob(depth_paths))
        assert len(depth_paths) == len(keys), "Number of depth maps mismatched."

    cam_infos = []
    for idx, key in enumerate(keys):

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = qvec2rotmat(extr.qvec)
        w2c[:3, 3] = np.array(extr.tvec)
        
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            fovx = focal2fov(focal_length_x, width)
            fovy = focal2fov(focal_length_x, height)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            fovx = focal2fov(focal_length_x, width)
            fovy = focal2fov(focal_length_y, height)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        if not os.path.isfile(image_path) and (image_path.endswith('jpg') or image_path.endswith('JPG')):
            image_path = image_path[:-3] + 'png'
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        # No mask to load
        mask_path = ""
        mask = None

        # Load depth if there is
        if depth_paths:
            depth_path = depth_paths[idx]
            depth = Image.open(depth_path)
        else:
            depth_path = ""
            depth = None

        # Load sparse depth
        if extr.name in correspondent:
            sparse_pt = points[correspondent[extr.name]]
        else:
            sparse_pt = None

        # Pack eveything
        cam_info = CameraInfo(
                image_name=image_name,
                w2c=w2c,
                fovx=fovx, fovy=fovy,
                width=width, height=height,
                R=R, T=T,
                cx_p=None, cy_p=None,
                image=image, image_path=image_path,
                depth=depth, depth_path=depth_path,
                sparse_pt=sparse_pt,
                mask=mask, mask_path=mask_path)
        cam_infos.append(cam_info)
    return cam_infos


def read_colmap_dataset(path, images, eval, test_every=8, depth_paths="", n_sparse=-1):
    # Parse colmap meta data
    sparse_path = os.path.join(path, "sparse", "0")
    if not os.path.exists(sparse_path):
        sparse_path = os.path.join(path, "colmap", "sparse", "0")
    if not os.path.exists(sparse_path):
        sparse_path = os.path.join(path, "sparse")
    if not os.path.exists(sparse_path):
        raise Exception("Can not find COLMAP outcome.")

    # Parse cameras
    try:
        cameras_extrinsic_file = os.path.join(sparse_path, "images.bin")
        cameras_intrinsic_file = os.path.join(sparse_path, "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(sparse_path, "images.txt")
        cameras_intrinsic_file = os.path.join(sparse_path, "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    # Parse sparse points
    points, colors, normals, ply_path, correspondent = read_colmap_ply(sparse_path, cam_extrinsics)
    point_cloud = PointCloud(
        points=points,
        colors=colors,
        normals=normals,
        ply_path=ply_path)

    # Load cameras
    image_dir = "images" if images is None else images
    cam_infos = read_cameras_from_colmap(
        cam_extrinsics=cam_extrinsics,
        cam_intrinsics=cam_intrinsics,
        images_folder=os.path.join(path, image_dir),
        points=points,
        correspondent=correspondent,
        depth_paths=depth_paths)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % test_every != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % test_every == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []
    
    if n_sparse > 0:
        idx = list(range(len(train_cam_infos)))
        idx_train = np.linspace(0, len(train_cam_infos) - 1, n_sparse)
        idx_train = [round(i) for i in idx_train]
        train_cam_infos = [c for idx, c in enumerate(train_cam_infos) if idx in idx_train]

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
        suggested_bounding = None

    # Pack scene info
    scene_info = SceneInfo(
        train_cam_infos=train_cam_infos,
        test_cam_infos=test_cam_infos,
        suggested_bounding=suggested_bounding,
        point_cloud=point_cloud)
    return scene_info
