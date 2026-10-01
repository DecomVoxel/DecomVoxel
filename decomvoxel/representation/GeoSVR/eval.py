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
from tqdm import trange
from pathlib import Path

import torch

from src.utils.image_utils import im_pil2tensor
from src.utils.loss_utils import psnr, ssim, lpips_loss, correct_lpips_loss


def read_pairs(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        if fname.endswith("jpg"): continue
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(im_pil2tensor(render).unsqueeze(0))
        gts.append(im_pil2tensor(gt).unsqueeze(0))
        image_names.append(fname)
    return renders, gts, image_names


def evaluate(model_paths, split):
    full_dict = {}
    per_view_dict = {}
    print(split)

    for scene_dir in model_paths:
        print("Scene:", scene_dir)
        full_dict[scene_dir] = {}
        per_view_dict[scene_dir] = {}
        

        test_dir = scene_dir / split

        for method in os.listdir(test_dir):
            method_dir = test_dir / method
            if not method_dir.is_dir():
                continue
            print("Method:", method)

            renders, gts, image_names = read_pairs(
                renders_dir=method_dir / "renders",
                gt_dir=method_dir/ "gt")

            ssims = []
            psnrs = []
            lpipss = []
            correct_lpipss = []

            for idx in trange(len(renders)):
                im_render = renders[idx].cuda()
                im_gt = gts[idx].cuda()
                # ssims.append(ssim_score(im_render, im_gt).item())
                ssims.append(ssim(im_render, im_gt).item())
                psnrs.append(psnr(im_render, im_gt).item())
                lpipss.append(lpips_loss(im_render, im_gt).item())
                correct_lpipss.append(correct_lpips_loss(im_render, im_gt).item())
                del im_render, im_gt
                torch.cuda.empty_cache()

            avg_ssim = np.mean(ssims)
            avg_psnr = np.mean(psnrs)
            avg_lpips = np.mean(lpipss)
            avg_correct_lpips = np.mean(correct_lpipss)

            print(f"  SSIM : {avg_ssim:>12.7f}")
            print(f"  PSNR : {avg_psnr:>12.7f}")
            print(f"  LPIPS: {avg_lpips:>12.7f}")
            print(f"  LPIPS: {avg_correct_lpips:>12.7f} (corrected)")
            print("")

            full_dict[scene_dir][method] = {
                "SSIM": avg_ssim,
                "PSNR": avg_psnr,
                "LPIPS": avg_lpips,
                "LPIPS-corrected": avg_correct_lpips,
            }
            per_view_dict[scene_dir][method] = {
                "SSIM": {name: ssim for ssim, name in zip(ssims, image_names)},
                "PSNR": {name: psnr for psnr, name in zip(psnrs, image_names)},
                "LPIPS": {name: lp for lp, name in zip(lpipss, image_names)},
                "LPIPS-corrected": {name: lp for lp, name in zip(correct_lpipss, image_names)},
            }

        with open(scene_dir / "results.json", 'w') as f:
            json.dump(full_dict[scene_dir], f, indent=True)
        with open(scene_dir / "per_view.json", 'w') as f:
            json.dump(per_view_dict[scene_dir], f, indent=True)
        print("Saved to", scene_dir / "results.json")
        print("Saved to", scene_dir / "per_view.json")


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="Quantitative evaluation of the rendered images.")
    parser.add_argument('--split', type=str, default="test")
    parser.add_argument('model_paths', nargs=argparse.REMAINDER, type=Path)
    args = parser.parse_args()

    assert len(args.model_paths) > 0
    evaluate(args.model_paths, args.split)
