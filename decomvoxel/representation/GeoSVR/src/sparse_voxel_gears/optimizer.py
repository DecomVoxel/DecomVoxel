# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import torch
from svraster_cuda.sparse_adam import SparseAdam

class SVOptimizer:

    def optimizer_init(self, cfg_optimizer):
        param_lst = [
            {'params': [self._geo_grid_pts], 'lr': cfg_optimizer.geo_lr, "name": "_geo_grid_pts"},
            {'params': [self._sh0], 'lr': cfg_optimizer.sh0_lr, "name": "_sh0"},
            {'params': [self._shs], 'lr': cfg_optimizer.shs_lr, "name": "_shs"},
            {'params': [self._subdiv_p], 'lr': 0.0, "name": "_subdiv_p"},
        ]

        self.optimizer = SparseAdam(
            param_lst, lr=0.0,
            betas=(cfg_optimizer.optim_beta1, cfg_optimizer.optim_beta2),
            eps=cfg_optimizer.optim_eps)

    def optimizer_save(self, path):
        ''' Save the other properties for resuming training. '''
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state_dict = {'optimizer': self.optimizer.state_dict()}
        torch.save(state_dict, path)

    def optimizer_load(self, path):
        state_dict = torch.load(path, map_location="cuda")
        self.optimizer.load_state_dict(state_dict['optimizer'])

    def optimizer_save_iteration(self, iteration):
        path = os.path.join(self.model_path, "checkpoints", f"iter{iteration:06d}_optim.pt")
        self.optimizer_save(path)

    def optimizer_load_iteration(self, iteration):
        path = os.path.join(self.model_path, "checkpoints", f"iter{iteration:06d}_optim.pt")
        self.optimizer_load(path)
