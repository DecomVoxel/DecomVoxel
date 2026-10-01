# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import sys
import cv2
import tqdm
import torch
import numpy as np
import imageio.v2 as iio
from PIL import Image


def depth_path(depth_root, cam):
    return os.path.join(depth_root, f"{cam.image_name}.png")

def codebook_path(depth_root, cam):
    return os.path.join(depth_root, f"{cam.image_name}.npy")

def gather_todo_list(depth_root, cameras, force_rerun=False):
    # Gather list of camera to estimate depth
    todo_indices = []
    for i, cam in enumerate(cameras):
        if not os.path.exists(depth_path(depth_root, cam)) or force_rerun:
            todo_indices.append(i)
    return todo_indices

def load_depth_to_camera(depth_root, cameras, depth_name):
    for cam in tqdm.tqdm(cameras):
        depth_np = iio.imread(depth_path(depth_root, cam))
        codebook = np.load(codebook_path(depth_root, cam))
        setattr(cam, depth_name, torch.tensor(codebook[depth_np]))

def save_quantize_depth(depth_root, cam, depth):
    # Quantize depth map to 16 bit
    codebook = depth.quantile(torch.linspace(0, 1, 65536).cuda(), interpolation='nearest')
    depth_idx = torch.searchsorted(codebook, depth, side='right').clamp_max_(65535)
    depth_idx[(depth - codebook[depth_idx-1]).abs() < (depth - codebook[depth_idx]).abs()] -= 1
    assert depth_idx.max() <= 65535
    assert depth_idx.min() >= 0

    # Save result
    depth_np = depth_idx.cpu().numpy().astype(np.uint16)
    iio.imwrite(depth_path(depth_root, cam), depth_np)
    np.save(codebook_path(depth_root, cam), codebook.cpu().numpy().astype(np.float32))

def resize_maxres_divisible(im, len, divisible):
    max_res = max(im.shape[-2:])
    target_size = (
        divisible * round(len * im.shape[-2] / max_res / divisible),
        divisible * round(len * im.shape[-1] / max_res / divisible))
    im = torch.nn.functional.interpolate(im, size=target_size, mode='bilinear', antialias=True)
    return im


@torch.no_grad()
def prepare_depthanythingv2(cameras, source_path, force_rerun=False):

    depth_root = os.path.join(source_path, "mono_priors", "depthanythingv2")
    os.makedirs(depth_root, exist_ok=True)

    todo_indices = gather_todo_list(depth_root, cameras, force_rerun=force_rerun)
    
    if len(todo_indices):
        print(f"Infer depth for {len(todo_indices)} images. Saved to {depth_root}.")

        # Load model
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        image_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Large-hf")
        model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Large-hf").cuda()

    for i in tqdm.tqdm(todo_indices):
        cam = cameras[i]

        # Inference depth
        inputs = image_processor(images=cam.image, return_tensors="pt", do_rescale=False)
        inputs['pixel_values'] = inputs['pixel_values'].cuda()
        outputs = model(**inputs)
        depth = outputs['predicted_depth'].squeeze()

        # Save result
        save_quantize_depth(depth_root, cam, depth)

    # Load the estimated depth
    print("Load the estimated depths to cameras.")
    load_depth_to_camera(depth_root, cameras, 'depthanythingv2')




@torch.no_grad()
def prepare_depthanythingv1(cameras, source_path, force_rerun=False):

    depth_root = os.path.join(source_path, "mono_priors", "depthanythingv1")
    os.makedirs(depth_root, exist_ok=True)

    todo_indices = gather_todo_list(depth_root, cameras, force_rerun=force_rerun)
    
    if len(todo_indices):
        print(f"Infer depth for {len(todo_indices)} images. Saved to {depth_root}.")

        # Load model
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        image_processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-large-hf")
        model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-large-hf").cuda()

    for i in tqdm.tqdm(todo_indices):
        cam = cameras[i]

        # Inference depth
        inputs = image_processor(images=cam.image, return_tensors="pt", do_rescale=False)
        inputs['pixel_values'] = inputs['pixel_values'].cuda()
        outputs = model(**inputs)
        depth = outputs['predicted_depth'].squeeze()

        # Save result
        save_quantize_depth(depth_root, cam, depth)

    # Load the estimated depth
    print("Load the estimated depths to cameras.")
    load_depth_to_camera(depth_root, cameras, 'depthanythingv2') # fixed



@torch.no_grad()
def prepare_depthpro(cameras, source_path, force_rerun=False):
    from transformers import DepthProImageProcessorFast, DepthProForDepthEstimation
    depth_root = os.path.join(source_path, "mono_priors", "depthpro")
    os.makedirs(depth_root, exist_ok=True)

    todo_indices = gather_todo_list(depth_root, cameras, force_rerun=force_rerun)
    
    if len(todo_indices):
        print(f"Infer depth for {len(todo_indices)} images. Saved to {depth_root}.")

        # Load model
        # from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        # image_processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-large-hf")
        # model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-large-hf").cuda()
        from transformers import DepthProImageProcessorFast, DepthProForDepthEstimation
        image_processor = DepthProImageProcessorFast.from_pretrained("apple/DepthPro-hf")
        model = DepthProForDepthEstimation.from_pretrained("apple/DepthPro-hf", use_fov_model=True).cuda()

    for i in tqdm.tqdm(todo_indices):
        cam = cameras[i]

        # Inference depth
        inputs = image_processor(images=cam.image, return_tensors="pt", do_rescale=False)
        inputs['pixel_values'] = inputs['pixel_values'].cuda()
        outputs = model(**inputs)
        outputs = image_processor.post_process_depth_estimation(
            outputs, target_sizes=[(cam.image.shape[-2], cam.image.shape[-1])],
        )
        
        depth = outputs[0]['predicted_depth'].squeeze()
                
        depth = 1/depth

        # Save result
        save_quantize_depth(depth_root, cam, depth)

    # Load the estimated depth
    print("Load the estimated depths to cameras.")
    load_depth_to_camera(depth_root, cameras, 'depthanythingv2') # fixed





@torch.no_grad()
def _prepare_mast3r_metric_depth(cameras, depth_root, mast3r_repo_path):
    print(f"Infer depth for {len(cameras)} images. Saved to {depth_root}.")

    assert os.path.exists(mast3r_repo_path), "mast3r repo path: `{mast3r_repo_path}` not found"
    sys.path.insert(0, mast3r_repo_path)

    # Load model
    from mast3r.model import AsymmetricMASt3R
    from dust3r.inference import inference
    from dust3r.cloud_opt.pair_viewer import PairViewer

    model = AsymmetricMASt3R.from_pretrained('naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric').cuda()

    depth_prev = None

    for i in tqdm.trange(len(cameras) - 1):
        cam1 = cameras[i]
        cam2 = cameras[i+1]

        # Inference depth
        image_1 = resize_maxres_divisible(cam1.image[None].cuda(), len=512, divisible=16)
        image_2 = resize_maxres_divisible(cam2.image[None].cuda(), len=512, divisible=16)
        true_shape = np.array([image_1.shape[-2:]], dtype=np.int32)

        image_dicts = [
            dict(img=image_1 * 2 - 1, true_shape=true_shape, idx=0, instance='0'),
            dict(img=image_2 * 2 - 1, true_shape=true_shape, idx=1, instance='1')
        ]
        symmetric_pair = [(image_dicts[0], image_dicts[1]), (image_dicts[1], image_dicts[0])]
        output = inference(symmetric_pair, model, 'cuda', batch_size=1, verbose=False)

        view1, pred1 = output['view1'], output['pred1']
        view2, pred2 = output['view2'], output['pred2']
        pair = PairViewer(view1, view2, pred1, pred2, verbose=False).cuda()
        depth1, depth2 = pair.get_depthmaps()
        pose1, pose2 = pair.get_im_poses()

        rescale = (cam1.position - cam2.position).norm() / (pose1[:3, 3] - pose2[:3, 3]).norm()
        depth1 = (depth1 * rescale).detach().squeeze()
        depth2 = (depth2 * rescale).detach().squeeze()

        # Average with previous result
        if depth_prev is not None:
            depth1 = (depth_prev + depth1) * 0.5

        depth_prev = depth2

        # Save result
        save_quantize_depth(depth_root, cam1, depth1)

    # Save last frame
    save_quantize_depth(depth_root, cam2, depth2)


@torch.no_grad()
def prepare_mast3r_metric_depth(cameras, source_path, mast3r_repo_path):

    depth_root = os.path.join(source_path, "mono_priors", "mast3r_metric_depth")
    os.makedirs(depth_root, exist_ok=True)

    n_todo = len(gather_todo_list(depth_root, cameras, force_rerun=False))
    if n_todo > 0:
        _prepare_mast3r_metric_depth(cameras, depth_root, mast3r_repo_path)

    # Load the estimated depth
    print("Load the estimated depths to cameras.")
    load_depth_to_camera(depth_root, cameras, 'mast3r_metric_depth')
