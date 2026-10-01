import torch
import lpips
from torch import nn

def lpips_loss(pred, target, lpips_fun):
    return lpips_fun(pred, target).flatten()

class LPIPSLoss(nn.Module):

    def __init__(self,
                 net='vgg',
                 lpips_list=None,
                 normalize_inputs=True,
                 loss_weight=1.0):
        super().__init__()
        self.net = net
        self.lpips = [] if (lpips_list is None or lpips_list[0].pnet_type != net) else lpips_list  # use a list to avoid registering the LPIPS model in state_dict
        self.normalize_inputs = normalize_inputs
        self.loss_weight = loss_weight

    def forward(self, pred, target):
        dtype = pred.dtype
        cdtype = torch.bfloat16
        if len(self.lpips) == 0:
            lpips_eval = lpips.LPIPS(
                net=self.net, eval_mode=True, pnet_tune=False).to(
                device=pred.device, dtype=cdtype)
            # with torch.no_grad():
            #     lpips_eval = torch.jit.trace(lpips_eval, (pred.to(cdtype), target.to(cdtype)))
                # lpips_eval = torch.jit.optimize_for_inference(lpips_eval)
            self.lpips.append(lpips_eval)
        if self.normalize_inputs:
            pred = pred * 2 - 1
            target = target * 2 - 1
        with torch.jit.optimized_execution(False):
            return lpips_loss(
                pred.to(cdtype), target.to(cdtype), lpips_fun=self.lpips[0]
            ).to(dtype) * self.loss_weight