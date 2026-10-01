# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

'''
Reference: KinectFusion algorithm.
'''

import numpy as np

import torch


class Fuser:
    def __init__(self,
            xyz,
            bandwidth,
            use_trunc=True, # Whether to truncate and filter occluded points.
            fuse_tsdf=True,
            feat_dim=0, # Feature dimension: 0=no features, 3=RGB, N=categories.
            alpha_thres=0.5,
            crop_border=0.0,
            normal_weight=False,
            depth_weight=False,
            border_weight=False,
            max_norm_dist=10.,
            use_half=False):
        assert len(xyz.shape) == 2
        assert xyz.shape[1] == 3
        self.xyz = xyz
        self.bandwidth = bandwidth
        self.use_trunc = use_trunc
        self.fuse_tsdf = fuse_tsdf
        self.feat_dim = feat_dim
        self.alpha_thres = alpha_thres
        self.crop_border = crop_border
        self.normal_weight = normal_weight
        self.depth_weight = depth_weight
        self.border_weight = border_weight
        self.max_norm_dist = max_norm_dist

        self.dtype = torch.float16 if use_half else torch.float32
        self.weight = torch.zeros([len(xyz), 1], dtype=self.dtype, device="cuda")
        self.feat = torch.zeros([len(xyz), feat_dim], dtype=self.dtype, device="cuda")
        if self.fuse_tsdf:
            self.sd_val = torch.zeros([len(xyz), 1], dtype=self.dtype, device="cuda")
        else:
            self.sd_val = None

    def integrate(self, cam, depth, feat=None, alpha=None, front_only=True):
        # Project grid points to image.
        xyz_uv, xyz_depth = cam.project(self.xyz, return_depth=True)

        inside_mask = (xyz_uv.abs() <= 1 - self.crop_border).all(-1)
        if front_only:
            front_mask = xyz_depth[:, 0] > max(float(cam.near), 1e-6)
            filter_idx = torch.where(inside_mask & front_mask)[0]
        else:
            filter_idx = torch.where(inside_mask)[0]

        valid_idx = filter_idx
        valid_xyz = self.xyz[filter_idx]
        valid_uv = xyz_uv[filter_idx]
        
        # Compute projective sdf
        valid_frame_depth = torch.nn.functional.grid_sample(
            depth.view(1,1,*depth.shape[-2:]),
            valid_uv.view(1,1,-1,2),
            mode='bilinear',
            align_corners=False).flatten()
        valid_xyz_depth = (valid_xyz - cam.position) @ cam.lookat
        valid_sdf = valid_frame_depth - valid_xyz_depth

        if torch.is_tensor(self.bandwidth):
            bandwidth = self.bandwidth[valid_idx]
        else:
            bandwidth = self.bandwidth

        valid_sdf *= (1 / bandwidth)

        if self.use_trunc:
            # Filter occluded
            filter_idx = torch.where(valid_sdf >= -1)[0]
            valid_idx = valid_idx[filter_idx]
            valid_uv = valid_uv[filter_idx]
            valid_frame_depth = valid_frame_depth[filter_idx]
            valid_sdf = valid_sdf[filter_idx]
            valid_sdf = valid_sdf.clamp_(-1, 1)

            # Init weighting
            w = torch.ones_like(valid_frame_depth)
        else:
            norm_dist = valid_sdf.abs()
            w = torch.exp(-norm_dist.clamp_max(self.max_norm_dist))

        # Alpha filtering
        if alpha is not None:
            valid_alpha = torch.nn.functional.grid_sample(
                alpha.view(1,1,*alpha.shape[-2:]),
                valid_uv.view(1,1,-1,2),
                mode='bilinear',
                align_corners=False).flatten()
            w *= valid_alpha

            filter_idx = torch.where(valid_alpha >= self.alpha_thres)[0]
            valid_idx = valid_idx[filter_idx]
            valid_uv = valid_uv[filter_idx]
            valid_frame_depth = valid_frame_depth[filter_idx]
            valid_sdf = valid_sdf[filter_idx]
            w = w[filter_idx]

        # Compute geometric weighting
        if self.depth_weight:
            w *= 1 / valid_frame_depth.clamp_min(0.1)

        if self.normal_weight:
            normal = cam.depth2normal(depth)
            rd = torch.nn.functional.normalize(cam.depth2pts(depth) - cam.position.view(3,1,1), dim=0)
            cos_theta = (normal * rd).sum(0).clamp_min(0)
            valid_cos_theta = torch.nn.functional.grid_sample(
                cos_theta.view(1,1,*cos_theta.shape[-2:]),
                valid_uv.view(1,1,-1,2),
                mode='bilinear',
                align_corners=False).flatten()
            w *= valid_cos_theta

        if self.border_weight:
            # The image center get 1.0; corners get 0.1
            w *= 1 / (1 + 9/np.sqrt(2) * valid_uv.square().sum(1).sqrt())
        
        # Reshape integration weight
        w = w.unsqueeze(-1).to(self.dtype)

        # Integrate weight
        self.weight[valid_idx] += w

        # Integrate tsdf
        if self.fuse_tsdf:
            valid_sdf = valid_sdf.unsqueeze(-1).to(self.dtype)
            self.sd_val[valid_idx] += w * valid_sdf

        # Sample feature
        if self.feat_dim > 0:
            valid_feat = torch.nn.functional.grid_sample(
                feat.view(1,self.feat_dim,*feat.shape[-2:]).to(self.dtype),
                valid_uv.view(1,1,-1,2).to(self.dtype),
                mode='bilinear',
                align_corners=False)[0,:,0].T
            self.feat[valid_idx] += w * valid_feat

    @property
    def feature(self):
        return self.feat / self.weight

    @property
    def tsdf(self):
        return self.sd_val / self.weight


@torch.no_grad()
def rgb_fusion(voxel_model, cameras):

    from .octree_utils import level_2_vox_size

    # Define volume integrator
    finest_vox_size = level_2_vox_size(voxel_model.scene_extent, voxel_model.octlevel.max()).item()
    feat_volume = Fuser(
        xyz=voxel_model.vox_center,
        bandwidth=10 * finest_vox_size,
        use_trunc=False,
        fuse_tsdf=False,
        feat_dim=3,
        crop_border=0.,
        normal_weight=False,
        depth_weight=False,
        border_weight=False,
        use_half=True)

    # Run semantic maps fusion
    for cam in cameras:
        render_pkg = voxel_model.render(cam, color_mode="dontcare", output_depth=True)
        depth = render_pkg['depth'][2]
        feat_volume.integrate(cam=cam, feat=cam.image.cuda(), depth=depth)

    return feat_volume.feature.nan_to_num_(0.5).float()
