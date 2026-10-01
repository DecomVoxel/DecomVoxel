# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch
from . import _C


class SparseAdam(torch.optim.Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-15, biased=False, sparse=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super(SparseAdam, self).__init__(params, defaults)
        self.biased = biased
        self.sparse = sparse

    def __setstate__(self, state):
        super(SparseAdam, self).__setstate__(state)

    @torch.no_grad()
    def step(self):

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']

            for param in group['params']:
                if param.grad is not None:
                    state = self.state[param]
                    # Lazy state initialization
                    if len(state) == 0:
                        # Number of time each param is visited
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(param, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(param, memory_format=torch.preserve_format)

                    state['step'] += 1

                    if self.biased:
                        _C.biased_adam_step(
                            self.sparse,
                            param,
                            param.grad,
                            state['exp_avg'],
                            state['exp_avg_sq'],
                            lr, beta1, beta2, eps
                        )
                    else:
                        _C.unbiased_adam_step(
                            self.sparse,
                            param,
                            param.grad,
                            state['exp_avg'],
                            state['exp_avg_sq'],
                            state['step'],
                            lr, beta1, beta2, eps
                        )
