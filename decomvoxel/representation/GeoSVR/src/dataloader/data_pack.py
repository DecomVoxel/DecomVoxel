# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import random
import numpy as np
import json
from PIL import Image

import torch

from src.dataloader.reader_colmap_dataset import read_colmap_dataset
from src.dataloader.reader_nerf_dataset import read_nerf_dataset

from src.cameras import Camera, MiniCam

AUTO_MAX_N_PIXELS = 1024 ** 2
AUTO_WARN = False

class DataPack:

    def __init__(self, cfg_data, white_background=False, dataset_downscales=[1.0], camera_params_only=False):

        self._cameras = dict(train={}, test={})
        self.source_path = cfg_data.source_path

        sparse_path = os.path.join(cfg_data.source_path, "sparse")
        colmap_path = os.path.join(cfg_data.source_path, "colmap", "sparse")
        meta_path1 = os.path.join(cfg_data.source_path, "transforms_train.json")
        meta_path2 = os.path.join(cfg_data.source_path, "transforms.json")

        if os.path.exists(sparse_path) or os.path.exists(colmap_path):
            print("Read dataset in COLMAP format.")
            scene_info = read_colmap_dataset(
                path=cfg_data.source_path,
                images=cfg_data.images,
                depth_paths=cfg_data.depth_paths,
                test_every=cfg_data.test_every,
                eval=cfg_data.eval,
                n_sparse=cfg_data.n_sparse)
        elif os.path.exists(meta_path1) or os.path.exists(meta_path2):
            print("Read dataset in NeRF format.")
            scene_info = read_nerf_dataset(
                path=cfg_data.source_path,
                extension=cfg_data.extension,
                test_every=cfg_data.test_every,
                eval=cfg_data.eval)
        else:
            raise Exception("Unknown scene type!")

        self._save_test_split(cfg_data.source_path, scene_info.test_cam_infos)

        print(f"res_downscale={cfg_data.res_downscale}")
        print(f"res_width={cfg_data.res_width}")
        if camera_params_only:
            self._cameras['train'][1.0] = CameraList(
                scene_info.train_cam_infos, cfg_data, camera_params_only=True)
            self._cameras['test'][1.0] = CameraList(
                scene_info.test_cam_infos, cfg_data, camera_params_only=True)
        else:
            for dataset_downscale in dataset_downscales:
                self._cameras['train'][dataset_downscale] = CameraList(
                    scene_info.train_cam_infos, cfg_data, dataset_downscale)
                self._cameras['test'][dataset_downscale] = CameraList(
                    scene_info.test_cam_infos, cfg_data, dataset_downscale)
                
                print("computing nearest_id")
                multi_view_num = 8
                multi_view_max_angle = 30
                multi_view_min_dis = 0.01
                multi_view_max_dis = 1.5
                
                self.world_view_transforms = []
                camera_centers = []
                center_rays = []
                for id, cur_cam in enumerate(self._cameras['train'][dataset_downscale]):
                    self.world_view_transforms.append(cur_cam.world_view_transform)
                    camera_centers.append(cur_cam.camera_center)
                    R = torch.tensor(cur_cam.R).float().cuda()
                    T = torch.tensor(cur_cam.T).float().cuda()
                    center_ray = torch.tensor([0.0,0.0,1.0]).float().cuda()
                    center_ray = center_ray@R.transpose(-1,-2)
                    center_rays.append(center_ray)
                self.world_view_transforms = torch.stack(self.world_view_transforms)
                camera_centers = torch.stack(camera_centers, dim=0)
                center_rays = torch.stack(center_rays, dim=0)
                center_rays = torch.nn.functional.normalize(center_rays, dim=-1)
                diss = torch.norm(camera_centers[:,None] - camera_centers[None], dim=-1).detach().cpu().numpy()
                tmp = torch.sum(center_rays[:,None]*center_rays[None], dim=-1)
                angles = torch.arccos(tmp)*180/3.14159
                angles = angles.detach().cpu().numpy()
                with open(os.path.join(cfg_data.source_path, "multi_view.json"), 'w') as file:
                    for id, cur_cam in enumerate(self._cameras['train'][dataset_downscale]):
                        sorted_indices = np.lexsort((angles[id], diss[id]))
                        # sorted_indices = np.lexsort((diss[id], angles[id]))
                        mask = (angles[id][sorted_indices] < multi_view_max_angle) & \
                            (diss[id][sorted_indices] > multi_view_min_dis) & \
                            (diss[id][sorted_indices] < multi_view_max_dis)
                        sorted_indices = sorted_indices[mask]
                        multi_view_num = min(multi_view_num, len(sorted_indices))
                        json_d = {'ref_name' : cur_cam.image_name, 'nearest_name': []}
                        for index in sorted_indices[:multi_view_num]:
                            cur_cam.nearest_id.append(index)
                            cur_cam.nearest_names.append(self._cameras['train'][dataset_downscale][index].image_name)
                            json_d["nearest_name"].append(self._cameras['train'][dataset_downscale][index].image_name)
                        json_str = json.dumps(json_d, separators=(',', ':'))
                        file.write(json_str)
                        file.write('\n')
                        # print(f"frame {cur_cam.image_name}, neareast {cur_cam.nearest_names}, \
                        #       angle {angles[id][cur_cam.nearest_id]}, diss {diss[id][cur_cam.nearest_id]}")
            

        self.has_depth = False
        self.has_mask = False
        for cams_split in self._cameras.values():
            for cams in cams_split.values():
                for cam in cams:
                    self.has_depth |= cam.depth is not None
                    self.has_mask |= cam.mask is not None

        if self.has_mask and cfg_data.blend_mask:
            bg_color = torch.tensor([float(white_background)]*3, dtype=torch.float32)
            self.composite_bg_color(bg_color)

        self.suggested_bounding = scene_info.suggested_bounding

        self.to_world_matrix = None
        to_world_path = os.path.join(cfg_data.source_path, 'to_world_matrix.txt')
        if os.path.isfile(to_world_path):
            self.to_world_matrix = np.loadtxt(to_world_path)

        self.point_cloud = scene_info.point_cloud

    def get_train_cameras(self, scale=1.0):
        return self._cameras['train'][scale]

    def get_test_cameras(self, scale=1.0):
        return self._cameras['test'][scale]

    @staticmethod
    def _save_test_split(source_path, test_cam_infos):
        test_names = [c.image_name for c in test_cam_infos]
        out_path = os.path.join(source_path, "test_split.json")
        with open(out_path, "w") as f:
            json.dump({"num_test": len(test_names), "test_images": test_names}, f, indent=2)
        print(f"[DataPack] Test split ({len(test_names)} images) -> {out_path}")

    def composite_bg_color(self, bg_color):
        for cams_split in self._cameras.values():
            for cams in cams_split.values():
                for cam in cams:
                    cam.composite_bg_color(bg_color)


def compute_iter_idx(num_data, num_iter):
    tr_iter_idx = []
    while len(tr_iter_idx) < num_iter:
        lst = list(range(num_data))
        random.shuffle(lst)
        tr_iter_idx.extend(lst)
    return tr_iter_idx[:num_iter]


class CameraList:
    def __init__(self, cam_infos, cfg_data, dataset_downscale=1.0, camera_params_only=False):
        if camera_params_only:
            self.camera_list = [
                instantiate_a_minicamera(cam_info, cfg_data, dataset_downscale)
                for cam_info in cam_infos
            ]
        else:
            self.camera_list = [
                instantiate_a_camera(cam_info, cfg_data, dataset_downscale)
                for cam_info in cam_infos
            ]
            for i in range(len(cam_infos)):
                self.camera_list[i] = self.camera_list[i].to(cfg_data.data_device)

    def __len__(self):
        return len(self.camera_list)

    def __getitem__(self, idx):
        return self.camera_list[idx]


def instantiate_a_camera(cam_info, cfg_data, dataset_downscale):
    # Determine target resolution
    W, H = cam_info.image.size
    if cfg_data.res_downscale > 0:
        global_downscale = cfg_data.res_downscale
    elif cfg_data.res_width > 0:
        global_downscale = W / cfg_data.res_width
    elif W * H > AUTO_MAX_N_PIXELS and cfg_data.images == "images":
        global_downscale = (AUTO_MAX_N_PIXELS / (W * H)) ** -0.5
        global AUTO_WARN
        if not AUTO_WARN:
            AUTO_WARN = True
            print(f"[WARN] Source images are too large ({W}x{H}). ")
            print(f"       Auto downscale gt by {global_downscale}. ")
            print(f"       Use `--images`, `--res_downscale`, or `--res_width` to prevent it.")
    else:
        global_downscale = 1

    target_downscale = float(global_downscale * dataset_downscale)
    target_resolution = (round(W / target_downscale), round(H / target_downscale))

    # Resize image if needed
            
    if len(cam_info.image.split()) > 3:
        pil_image = Image.fromarray(np.stack([im for im in cam_info.image.split()[:3]], axis=-1))
        mask = Image.fromarray(np.array(cam_info.image.split()[-1]).astype(np.uint8))
    else:
        pil_image = cam_info.image
        mask = cam_info.mask
    
    if (W, H) != target_resolution:
        pil_image = pil_image.resize(target_resolution)
        if mask is not None:
            mask = mask.resize(target_resolution)
    
    ncc_scale = cfg_data.ncc_scale
    # if ncc_scale != 1.0:
    #     ncc_resolution = (int(target_resolution[0]/ncc_scale), int(target_resolution[1]/ncc_scale))
    #     ncc_pil_image = pil_image.resize(ncc_resolution)

    # Read color image
    gt_image = torch.tensor(np.array(pil_image), dtype=torch.float32).moveaxis(-1, 0) / 255.0
    if mask is not None:
        mask = torch.tensor(np.array(mask), dtype=torch.float32) / 255.0    


    # Load mask if exist
    if cam_info.mask is not None:
        if mask is not None:
            raise NotImplementedError("Duplicated mask from RGBA and given mask path !?")
        if cam_info.mask.size != target_resolution:
            pil_mask = cam_info.mask.resize(target_resolution)
        else:
            pil_mask = cam_info.mask
        mask = torch.tensor(np.array(pil_mask), dtype=torch.float32) / 255.0
        if len(mask.shape) == 3:
            mask = mask.mean(-1)
        mask = mask.unsqueeze(0).contiguous()

    # Load depth if exist
    depth = None
    if cam_info.depth is not None:
        if cam_info.depth.size != target_resolution:
            pil_depth = cam_info.depth.resize(target_resolution, Image.Resampling.NEAREST)
        else:
            pil_depth = cam_info.depth
        depth = torch.tensor(np.array(pil_depth) / cfg_data.depth_scale, dtype=torch.float32)
        depth = depth.unsqueeze(0).contiguous()

    return Camera(w2c=cam_info.w2c,
                  fovx=cam_info.fovx, fovy=cam_info.fovy,
                  cx_p=cam_info.cx_p, cy_p=cam_info.cy_p,
                  R=cam_info.R, T=cam_info.T,
                  image=gt_image, mask=mask, depth=depth,
                  sparse_pt=cam_info.sparse_pt,
                  ncc_scale=ncc_scale,
                  image_name=cam_info.image_name)


def instantiate_a_minicamera(cam_info, cfg_data, dataset_downscale=1):
    # Determine target resolution
    W, H = cam_info.image.size
    if cfg_data.res_downscale > 0:
        global_downscale = cfg_data.res_downscale
    elif cfg_data.res_width > 0:
        global_downscale = W / cfg_data.res_width
    elif W * H > AUTO_MAX_N_PIXELS and cfg_data.images == "images":
        global_downscale = (AUTO_MAX_N_PIXELS / (W * H)) ** -0.5
        global AUTO_WARN
        if not AUTO_WARN:
            AUTO_WARN = True
            print(f"[WARN] Auto downscale gt by {global_downscale}. ")
            print(f"       Use `--res_downscale` or `--res_width` to prevent it.")
    else:
        global_downscale = 1

    target_downscale = float(global_downscale * dataset_downscale)
    target_resolution = (round(W / target_downscale), round(H / target_downscale))

    return MiniCam(
        c2w=np.linalg.inv(cam_info.w2c),
        fovx=cam_info.fovx,
        fovy=cam_info.fovy,
        width=target_resolution[0],
        height=target_resolution[1],
        cx_p=cam_info.cx_p, cy_p=cam_info.cy_p)
