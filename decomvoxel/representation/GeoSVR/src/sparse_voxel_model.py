# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import numpy as np

import torch
import svraster_cuda

from src.sparse_voxel_gears.constructor import SVConstructor
from src.sparse_voxel_gears.properties import SVProperties
from src.sparse_voxel_gears.renderer import SVRenderer
from src.sparse_voxel_gears.adaptive import SVAdaptive
from src.sparse_voxel_gears.optimizer import SVOptimizer
from src.sparse_voxel_gears.io import SVInOut
from src.sparse_voxel_gears.pooling import SVPooling


class SparseVoxelModel(SVConstructor, SVProperties, SVRenderer, SVAdaptive, SVOptimizer, SVInOut, SVPooling):

    def __init__(self, cfg_model):
        '''
        Setup of the model. The config is defined by `cfg.model` in `src/config.py`.
        After the initial setup. There are two ways to instantiate the models:

        1. `model_load` defined in `src/sparse_voxel_gears/io.py`.
           Load the saved models from a given path.

        2. `model_init` defined in `src/sparse_voxel_gears/constructor.py`.
           Heuristically initial the sparse grid layout and parameters from the training datas.
        '''
        super().__init__()
        self.model_path = cfg_model.model_path
        self.vox_geo_mode = cfg_model.vox_geo_mode
        self.density_mode = cfg_model.density_mode
        self.active_sh_degree = cfg_model.sh_degree
        self.max_sh_degree = cfg_model.sh_degree
        self.ss = cfg_model.ss
        self.white_background = cfg_model.white_background
        self.black_background = cfg_model.black_background

        assert cfg_model.outside_level <= svraster_cuda.meta.MAX_NUM_LEVELS
        self.outside_level = cfg_model.outside_level
        self.inside_level = svraster_cuda.meta.MAX_NUM_LEVELS - self.outside_level

        # List the variable names
        self.per_voxel_attr_lst = [
            'octpath', 'octlevel',
            'vox_center', 'vox_size',
            'subdiv_meta',
        ]
        self.per_voxel_param_lst = [
            '_sh0', '_shs', '_subdiv_p',
        ]
        self.grid_pts_param_lst = [
            '_geo_grid_pts',
        ]
        self.state_attr_names = ['exp_avg', 'exp_avg_sq']
