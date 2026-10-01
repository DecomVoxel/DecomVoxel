# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import re
import torch
from svraster_cuda.meta import MAX_NUM_LEVELS

from src.utils import octree_utils

class SVInOut:

    @staticmethod
    def create_state_dict_from_subset(
        voxel_model,
        voxel_mask,
        recenter=True,
        outside_level=None,
    ):
        indices = torch.where(voxel_mask)[0]
        
        if len(indices) == 0:
            return None
        
        # Extract per-voxel attributes
        obj_vox_center = voxel_model.vox_center[indices]
        obj_vox_size = voxel_model.vox_size[indices]
        obj_octlevel = voxel_model.octlevel[indices]
        obj_sh0 = voxel_model._sh0.data[indices].clone()
        obj_shs = voxel_model._shs.data[indices].clone()
        
        # Extract grid points for the subset of voxels
        # vox_key[indices] gives us the indices into grid_pts_key for these voxels
        obj_vox_key = voxel_model.vox_key[indices]  # [N_obj, 8]
        
        # Get unique grid point indices used by this subset
        unique_grid_indices, inverse_indices = obj_vox_key.flatten().unique(return_inverse=True)
        
        # Extract the geo values for these grid points
        obj_geo_grid_pts = voxel_model._geo_grid_pts.data[unique_grid_indices].clone()
        
        # Compute scene bounds for the object
        bbox_min = obj_vox_center.min(dim=0).values
        bbox_max = obj_vox_center.max(dim=0).values
        
        # # Recenter
        # # Compute new scene bounds centered on the object
        # bbox_center = (bbox_min + bbox_max) / 2
        # bbox_radius = (bbox_max - bbox_min) / 2
        
        # # Use the max radius with some padding to ensure all voxels fit
        # max_radius = bbox_radius.max() * 1.1  # 10% padding
        
        # # The inside_extent covers the object
        # new_inside_extent = 2 * max_radius
        
        # # Use the original outside_level or provided one
        # if outside_level is None:
        #     outside_level = voxel_model.outside_level
        
        # new_scene_extent = new_inside_extent * (2 ** outside_level)
        # new_scene_center = bbox_center
        
        # # Recalculate octpath for new scene bounds
        # new_octpath = octree_utils.xyz_2_octpath(
        #     obj_vox_center,
        #     obj_octlevel,
        #     new_scene_center,
        #     new_scene_extent
        # )
        
        if recenter:
            # Compute new scene bounds centered on the object
            bbox_center = (bbox_min + bbox_max) / 2
            bbox_radius = (bbox_max - bbox_min) / 2

            # Use the max radius with some padding to ensure all voxels fit
            max_radius = bbox_radius.max() * 1.1  # 10% padding

            # The inside_extent covers the object
            new_inside_extent = 2 * max_radius
            new_scene_extent = new_inside_extent * (2 ** outside_level)
            new_scene_center = bbox_center

            # Recalculate octpath for new scene bounds
            new_octpath = octree_utils.xyz_2_octpath(
                obj_vox_center,
                obj_octlevel,
                new_scene_center,
                new_scene_extent
            )
        else:
            # Keep original scene bounds; octpath is just the subset of the original
            new_scene_center = voxel_model.scene_center
            new_scene_extent = voxel_model.scene_extent
            new_inside_extent = voxel_model.inside_extent
            new_octpath = voxel_model.octpath[indices]
       
        # Rebuild grid points link for the new octree structure
        new_grid_pts_key, new_vox_key = octree_utils.build_grid_pts_link(
            new_octpath, obj_octlevel
        )
        
        # Now we need to map the old geo values to the new grid point structure
        # The new grid points are at different integer coordinates
        # We need to map by matching the world-space positions
        
        # _geo_grid_pts stores density values at grid points, usually as [M, 1]
        # (some code paths may use [M, 8]).
        # vox_key is an integer [N, 8] tensor mapping each voxel's eight corners
        # to indices in grid_pts_key.
        # grid_pts_key is an integer [M, 3] tensor containing the coordinate
        # indices of all unique grid points (voxel corners) in the scene.
        
        
        # Compute world-space positions for old grid points
        old_grid_pts_xyz = octree_utils.compute_gridpoints_xyz(
            voxel_model.grid_pts_key[unique_grid_indices],
            voxel_model.scene_center,
            voxel_model.scene_extent
        )
        
        # Compute world-space positions for new grid points
        new_grid_pts_xyz = octree_utils.compute_gridpoints_xyz(
            new_grid_pts_key,
            new_scene_center,
            new_scene_extent
        )
        
        # Find closest matches using voxel corner correspondence
        # Since we're dealing with the same voxels, just recentered,
        # we can use the vox_key mapping
        
        # Build mapping: for each new grid point, find corresponding old geo value
        # Use the inverse of vox_key relationships
        
        # Method: for each new grid point at index i, find which old grid point
        # corresponds to the same physical location by checking voxel corners
        
        # Simpler approach: since grid points are shared at voxel corners,
        # we can trace through voxel corners
        
        # For each voxel, its 8 corners in old and new systems correspond 1-to-1
        # old_vox_key[v, c] -> old grid point index
        # new_vox_key[v, c] -> new grid point index
        # These should have the same geo value
        
        old_vox_key_flat = obj_vox_key.flatten()  # Indices into original grid_pts
        new_vox_key_flat = new_vox_key.flatten()  # Indices into new grid_pts
        
        # Map old grid indices to subset indices
        old_to_subset = torch.zeros(voxel_model.num_grid_pts, dtype=torch.long, device=obj_vox_key.device)
        old_to_subset[unique_grid_indices] = torch.arange(len(unique_grid_indices), device=obj_vox_key.device)
        subset_indices = old_to_subset[old_vox_key_flat]
        
        # Create new geo grid pts array
        new_geo_grid_pts = torch.zeros(len(new_grid_pts_key), 1, dtype=obj_geo_grid_pts.dtype, device=obj_geo_grid_pts.device)
        new_geo_grid_pts[new_vox_key_flat] = obj_geo_grid_pts[subset_indices]
        
        state_dict = {
            'active_sh_degree': voxel_model.active_sh_degree,
            'ss': voxel_model.ss,
            'scene_center': new_scene_center.contiguous(),
            'inside_extent': new_inside_extent.contiguous() if torch.is_tensor(new_inside_extent) else torch.tensor(new_inside_extent, device='cuda'),
            'scene_extent': new_scene_extent.contiguous() if torch.is_tensor(new_scene_extent) else torch.tensor(new_scene_extent, device='cuda'),
            'octpath': new_octpath.contiguous(),
            'octlevel': obj_octlevel.contiguous(),
            '_geo_grid_pts': new_geo_grid_pts.contiguous(),
            '_sh0': obj_sh0.contiguous(),
            '_shs': obj_shs.contiguous(),
            'quantized': False,
        }
    
        
        # Add metadata for reference
        state_dict['_metadata'] = {
            'original_indices': indices.cpu(),
            'bbox_min': bbox_min.cpu(),
            'bbox_max': bbox_max.cpu(),
            'num_voxels': len(indices),
            'num_grid_pts': len(new_grid_pts_key),
        }
        
        return state_dict

    @staticmethod  
    def save_state_dict(state_dict, path, quantize=False):
        """
        Save a state_dict to file, optionally with quantization.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        save_dict = {k: v for k, v in state_dict.items() if not k.startswith('_metadata')}
        
        if quantize:
            quantize_state_dict(save_dict)
            save_dict['quantized'] = True
        else:
            save_dict['quantized'] = False
        
        for k, v in save_dict.items():
            if torch.is_tensor(v):
                save_dict[k] = v.cpu()
        
        torch.save(save_dict, path)
        
        return path

    def load_from_state_dict(self, state_dict):
        if state_dict.get('quantized', False):
            dequantize_state_dict(state_dict)

        self.active_sh_degree = state_dict['active_sh_degree']
        self.ss = state_dict['ss']

        self.scene_center = state_dict['scene_center'].cuda() if state_dict['scene_center'].device.type != 'cuda' else state_dict['scene_center']
        self.inside_extent = state_dict['inside_extent'].cuda() if state_dict['inside_extent'].device.type != 'cuda' else state_dict['inside_extent']
        self.scene_extent = state_dict['scene_extent'].cuda() if state_dict['scene_extent'].device.type != 'cuda' else state_dict['scene_extent']

        self.octpath = state_dict['octpath'].cuda() if state_dict['octpath'].device.type != 'cuda' else state_dict['octpath']
        self.octlevel = (state_dict['octlevel'].cuda() if state_dict['octlevel'].device.type != 'cuda' else state_dict['octlevel']).to(torch.int8)
        
        self.vox_center, self.vox_size = octree_utils.octpath_decoding(
            self.octpath, self.octlevel, self.scene_center, self.scene_extent)
        self.grid_pts_key, self.vox_key = octree_utils.build_grid_pts_link(self.octpath, self.octlevel)

        self._geo_grid_pts = (state_dict['_geo_grid_pts'].cuda() if state_dict['_geo_grid_pts'].device.type != 'cuda' else state_dict['_geo_grid_pts']).requires_grad_()

        self._sh0 = (state_dict['_sh0'].cuda() if state_dict['_sh0'].device.type != 'cuda' else state_dict['_sh0']).requires_grad_()
        self._shs = (state_dict['_shs'].cuda() if state_dict['_shs'].device.type != 'cuda' else state_dict['_shs']).requires_grad_()

        N = len(self.octpath)
        self._subdiv_p = torch.full([N, 1], 1.0, dtype=torch.float32, device="cuda").requires_grad_()
        self.subdiv_meta = torch.zeros([N, 1], dtype=torch.float32, device="cuda")

        self.bg_color = torch.tensor(
            [1, 1, 1] if self.white_background else [0, 0, 0],
            dtype=torch.float32, device="cuda")

    def save(self, path, quantize=False):
        '''
        Save the necessary attributes and parameters for reproducing rendering.
        '''
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state_dict = {
            'active_sh_degree': self.active_sh_degree,
            'ss': self.ss,
            'scene_center': self.scene_center.data.contiguous(),
            'inside_extent': self.inside_extent.data.contiguous(),
            'scene_extent': self.scene_extent.data.contiguous(),
            'octpath': self.octpath.data.contiguous(),
            'octlevel': self.octlevel.data.contiguous(),
            '_geo_grid_pts': self._geo_grid_pts.data.contiguous(),
            '_sh0': self._sh0.data.contiguous(),
            '_shs': self._shs.data.contiguous(),
        }

        if quantize:
            quantize_state_dict(state_dict)
            state_dict['quantized'] = True
        else:
            state_dict['quantized'] = False

        for k, v in state_dict.items():
            if torch.is_tensor(v):
                state_dict[k] = v.cpu()
        torch.save(state_dict, path)
        self.latest_save_path = path

    def load(self, path):
        '''
        Load the saved models.
        '''
        state_dict = torch.load(path, map_location="cpu", weights_only=False)

        if state_dict.get('quantized', False):
            dequantize_state_dict(state_dict)

        self.active_sh_degree = state_dict['active_sh_degree']
        self.ss = state_dict['ss']

        self.scene_center = state_dict['scene_center'].cuda()
        self.inside_extent = state_dict['inside_extent'].cuda()
        self.scene_extent = state_dict['scene_extent'].cuda()

        # Octree path encodes each voxel's position and parent-child relations.
        self.octpath = state_dict['octpath'].cuda()
        # Deeper octree levels contain smaller voxels at higher resolution.
        self.octlevel = state_dict['octlevel'].cuda().to(torch.int8)
        self.vox_center, self.vox_size = octree_utils.octpath_decoding(
            self.octpath, self.octlevel, self.scene_center, self.scene_extent)
        self.grid_pts_key, self.vox_key = octree_utils.build_grid_pts_link(self.octpath, self.octlevel) 
        # grid_pts_key is an integer [M, 3] tensor of unique grid-point
        # coordinates. Sharing these entries avoids duplicate corner storage
        # and supports fast lookup from world coordinates.
        # vox_key maps the eight corners of each voxel to grid_pts_key indices.
        # _geo_grid_pts stores the density value at each grid point.

        self._geo_grid_pts = state_dict['_geo_grid_pts'].cuda().requires_grad_()

        self._sh0 = state_dict['_sh0'].cuda().requires_grad_()
        self._shs = state_dict['_shs'].cuda().requires_grad_()

        N = len(self.octpath)
        self._subdiv_p = torch.full([N, 1], 1.0, dtype=torch.float32, device="cuda").requires_grad_()
        self.subdiv_meta = torch.zeros([N, 1], dtype=torch.float32, device="cuda")

        self.bg_color = torch.tensor(
            [1, 1, 1] if self.white_background else [0, 0, 0],
            dtype=torch.float32, device="cuda")

        self.loaded_path = path

    def save_iteration(self, iteration, quantize=False):
        path = os.path.join(self.model_path, "checkpoints", f"iter{iteration:06d}_model.pt")
        self.save(path, quantize=quantize)
        self.latest_save_iter = iteration

    def load_iteration(self, iteration=-1):
        if iteration == -1:
            # Find the maximum iteration if it is -1.
            fnames = os.listdir(os.path.join(self.model_path, "checkpoints"))
            loaded_iter = max(int(re.sub("[^0-9]", "", fname)) for fname in fnames)
        else:
            loaded_iter = iteration

        path = os.path.join(self.model_path, "checkpoints", f"iter{loaded_iter:06d}_model.pt")
        self.load(path)

        self.loaded_iter = iteration

        return loaded_iter


# Quantization utilities to reduce size when saving model.
# It can reduce ~70% model size with minor PSNR drop.
def quantize_state_dict(state_dict):
    state_dict['_geo_grid_pts'] = quantization(state_dict['_geo_grid_pts'])
    state_dict['_sh0'] = [quantization(v) for v in state_dict['_sh0'].split(1, dim=1)]
    state_dict['_shs'] = [quantization(v) for v in state_dict['_shs'].split(1, dim=1)]

def dequantize_state_dict(state_dict):
    state_dict['_geo_grid_pts'] = dequantization(state_dict['_geo_grid_pts'])
    state_dict['_sh0'] = torch.cat(
        [dequantization(v) for v in state_dict['_sh0']], dim=1)
    state_dict['_shs'] = torch.cat(
        [dequantization(v) for v in state_dict['_shs']], dim=1)

def quantization(src_tensor, max_iter=10):
    src_shape = src_tensor.shape
    src_vals = src_tensor.flatten().contiguous()
    order = src_vals.argsort()
    quantile_ind = (torch.linspace(0,1,257) * (len(order) - 1)).long().clamp_(0, len(order)-1)
    codebook = src_vals[order[quantile_ind]].contiguous()
    codebook[0] = -torch.inf
    ind = torch.searchsorted(codebook, src_vals)

    codebook = codebook[1:]
    ind = (ind - 1).clamp_(0, 255)

    diff_l = (src_vals - codebook[ind-1]).abs()
    diff_m = (src_vals - codebook[ind]).abs()
    ind = ind - 1 + (diff_m < diff_l)
    ind.clamp_(0, 255)

    for _ in range(max_iter):
        codebook = torch.zeros_like(codebook).index_reduce_(
            dim=0,
            index=ind,
            source=src_vals,
            reduce='mean',
            include_self=False)
        diff_l = (src_vals - codebook[ind-1]).abs()
        diff_r = (src_vals - codebook[(ind+1).clamp_max_(255)]).abs()
        diff_m = (src_vals - codebook[ind]).abs()
        upd_mask = torch.minimum(diff_l, diff_r) < diff_m
        if upd_mask.sum() == 0:
            break
        shift = (diff_r < diff_l) * 2 - 1
        ind[upd_mask] += shift[upd_mask]
        ind.clamp_(0, 255)

    codebook = torch.zeros_like(codebook).index_reduce_(
        dim=0,
        index=ind,
        source=src_vals,
        reduce='mean',
        include_self=False)

    return dict(
        index=ind.reshape(src_shape).to(torch.uint8),
        codebook=codebook,
    )

def dequantization(quant_dict):
    return quant_dict['codebook'][quant_dict['index'].long()]
