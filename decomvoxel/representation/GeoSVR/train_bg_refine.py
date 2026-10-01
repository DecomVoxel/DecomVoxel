import os
import sys
import json
import time
import uuid
import imageio
import datetime
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import torchvision
from PIL import Image
import numpy as np

from src.config import cfg, update_argparser, update_config

from src.utils.system_utils import seed_everything
from src.utils.image_utils import im_tensor2np, viz_tensordepth
from src.utils.bounding_utils import decide_main_bounding
from src.utils import loss_utils
from src.utils.graphics_utils import render_normal_func

from src.dataloader.data_pack import compute_iter_idx
from src.sparse_voxel_model import SparseVoxelModel
from src.cameras import Camera
from bg_inpaint.cam_util import load_selected_camera_params
import svraster_cuda


BG_ROOTS = {}


def read_rgb_tensor(image_path, width=None, height=None):
    image = Image.open(image_path).convert("RGB")
    if width is not None and height is not None and image.size != (width, height):
        image = image.resize((width, height), Image.BILINEAR)
    image_np = np.asarray(image, dtype=np.float32) / 255.0
    image_t = torch.from_numpy(image_np).permute(2, 0, 1).contiguous()
    return image_t


def read_mask_tensor(mask_path, width=None, height=None):
    mask = Image.open(mask_path).convert("L")
    if width is not None and height is not None and mask.size != (width, height):
        mask = mask.resize((width, height), Image.NEAREST)
    mask_np = np.asarray(mask, dtype=np.float32) / 255.0
    mask_t = torch.from_numpy(mask_np).unsqueeze(0).contiguous()
    return mask_t


def _resize_2d_tensor(tensor_2d, size_hw, mode):
    if tuple(tensor_2d.shape[-2:]) == tuple(size_hw):
        return tensor_2d
    x = tensor_2d.float()[None, None]
    if mode == "nearest":
        x = F.interpolate(x, size=size_hw, mode=mode)
    else:
        x = F.interpolate(x, size=size_hw, mode=mode, align_corners=False)
    return x[0, 0]


def read_depth_npy(depth_path, width=None, height=None):
    depth = np.load(depth_path).astype(np.float32)
    if width is not None and height is not None and depth.shape != (height, width):
        depth_t = torch.from_numpy(depth)
        depth = _resize_2d_tensor(depth_t, (height, width), mode="nearest").cpu().numpy()
    return depth


def read_bool_mask_npy(mask_path, width=None, height=None):
    mask = np.load(mask_path).astype(bool)
    if width is not None and height is not None and mask.shape != (height, width):
        mask_t = torch.from_numpy(mask.astype(np.float32))
        mask = (_resize_2d_tensor(mask_t, (height, width), mode="nearest") > 0.5).cpu().numpy()
    return mask


def resolve_rgb_path(guidance_root, dn_root, frame_idx):
    candidates = [
        os.path.join(dn_root, f"rgb_frame{frame_idx:06d}.png"),
        os.path.join(guidance_root, "see3d_inpaint_output", f"predict_warp_frame{frame_idx:06d}.png"),
        os.path.join(guidance_root, "raw_rgb", f"{frame_idx}.png"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Cannot find RGB for frame {frame_idx:06d} in {candidates}")


def compute_nearest_views(
    cameras,
    multi_view_num=8,
    multi_view_max_angle=30.0,
    multi_view_min_dis=0.01,
    multi_view_max_dis=1.5,
):
    if len(cameras) <= 1:
        return

    camera_centers = []
    center_rays = []

    for cam in cameras:
        cam.nearest_id = []
        cam.nearest_names = []

        camera_centers.append(cam.camera_center)

        R = torch.tensor(cam.R, dtype=torch.float32, device="cuda")
        center_ray = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device="cuda")
        center_ray = center_ray @ R.transpose(-1, -2)
        center_rays.append(center_ray)

    camera_centers = torch.stack(camera_centers, dim=0)
    center_rays = torch.stack(center_rays, dim=0)
    center_rays = F.normalize(center_rays, dim=-1)

    diss = torch.norm(camera_centers[:, None] - camera_centers[None], dim=-1).detach().cpu().numpy()
    dots = torch.sum(center_rays[:, None] * center_rays[None], dim=-1).clamp(-1.0, 1.0)
    angles = (torch.arccos(dots) * 180.0 / np.pi).detach().cpu().numpy()

    for idx, cur_cam in enumerate(cameras):
        sorted_indices = np.lexsort((angles[idx], diss[idx]))
        mask = (
            (angles[idx][sorted_indices] < multi_view_max_angle)
            & (diss[idx][sorted_indices] > multi_view_min_dis)
            & (diss[idx][sorted_indices] < multi_view_max_dis)
        )
        sorted_indices = sorted_indices[mask]

        for nbr_idx in sorted_indices[:multi_view_num]:
            if nbr_idx == idx:
                continue
            cur_cam.nearest_id.append(int(nbr_idx))
            cur_cam.nearest_names.append(cameras[nbr_idx].image_name)


def build_background_cameras():
    guidance_root = BG_ROOTS["guidance_root"]
    merge_root = BG_ROOTS["merge_root"]
    dn_root = BG_ROOTS["dn_root"]

    camera_json_path = os.path.join(guidance_root, "camera_params.json")
    if not os.path.exists(camera_json_path):
        raise FileNotFoundError(f"camera_params.json not found: {camera_json_path}")

    camera_specs = load_selected_camera_params(camera_json_path, keep_metadata=True)

    cameras = []
    for spec in camera_specs:
        frame_idx = int(spec["camera_name"])
        width = int(spec["width"])
        height = int(spec["height"])

        rgb_path = resolve_rgb_path(guidance_root, dn_root, frame_idx)
        image = read_rgb_tensor(rgb_path, width=width, height=height)

        depth_path = os.path.join(
            merge_root,
            f"plane_refined_depth_frame{frame_idx:06d}.npy",
        )
        conf_path = os.path.join(
            merge_root,
            f"plane_refined_conf_frame{frame_idx:06d}.npy",
        )

        if not os.path.exists(depth_path):
            raise FileNotFoundError(f"Refined depth not found: {depth_path}")
        if not os.path.exists(conf_path):
            raise FileNotFoundError(f"Refined conf not found: {conf_path}")

        plane_depth = read_depth_npy(depth_path, width=width, height=height)
        plane_conf = read_bool_mask_npy(conf_path, width=width, height=height)

        mask_path = os.path.join(dn_root, f"mask_frame{frame_idx:06d}.png")
        mask = None
        if os.path.exists(mask_path):
            mask = read_mask_tensor(mask_path, width=width, height=height)

        c2w = np.asarray(spec["c2w"], dtype=np.float32)
        w2c = np.linalg.inv(c2w).astype(np.float32)
        R = np.transpose(w2c[:3, :3]).astype(np.float32)
        T = np.asarray(w2c[:3, 3], dtype=np.float32)

        cam = Camera(
            image_name=str(spec["camera_name"]),
            w2c=w2c,
            fovx=float(spec["fovx"]),
            fovy=float(spec["fovy"]),
            cx_p=float(spec["cx_p"]),
            cy_p=float(spec["cy_p"]),
            R=R,
            T=T,
            near=float(spec["near"]),
            image=image,
            mask=mask,
            depth=None,
            sparse_pt=None,
            ncc_scale=cfg.data.ncc_scale,
        )

        cam.source_view_index = spec["source_view_index"]
        cam.source_image_name = spec["source_image_name"]
        cam.rgb_path = rgb_path
        cam.plane_refined_depth = torch.from_numpy(plane_depth).float().cpu()
        cam.plane_refined_conf = torch.from_numpy(plane_conf.astype(np.float32)).cpu()

        cameras.append(cam)

    compute_nearest_views(cameras)
    return cameras


class BackgroundDataPack:
    def __init__(self, cfg_data, white_background=False, dataset_downscales=None, camera_params_only=False):
        self.source_path = BG_ROOTS["guidance_root"]
        self._train_cameras = build_background_cameras()

        # Reuse train views as test views so base training_report keeps working.
        self._test_cameras = list(self._train_cameras)

        self.has_depth = True
        self.has_mask = any(cam.mask is not None for cam in self._train_cameras)
        self.suggested_bounding = None
        self.to_world_matrix = None
        self.point_cloud = None

    def get_train_cameras(self, scale=1.0):
        return self._train_cameras

    def get_test_cameras(self, scale=1.0):
        return self._test_cameras


def get_plane_depth_and_mask(cam, target_hw):
    if not hasattr(cam, "plane_refined_depth") or not hasattr(cam, "plane_refined_conf"):
        empty_depth = torch.zeros(target_hw, dtype=torch.float32, device="cuda")
        empty_mask = torch.zeros(target_hw, dtype=torch.bool, device="cuda")
        return empty_depth, empty_mask

    plane_depth = cam.plane_refined_depth.cuda().float()
    plane_conf = cam.plane_refined_conf.cuda().float()

    plane_depth = _resize_2d_tensor(plane_depth, target_hw, mode="nearest")
    plane_conf = _resize_2d_tensor(plane_conf, target_hw, mode="nearest") > 0.5

    valid_mask = plane_conf & torch.isfinite(plane_depth) & (plane_depth > cam.near)
    return plane_depth, valid_mask


def plane_refined_depth_l1_loss(cam, render_pkg):
    alpha = (1.0 - render_pkg["raw_T"]).clamp_min(1e-4)
    metric_depth = render_pkg["raw_depth"][0] / alpha[0]

    plane_depth, plane_mask = get_plane_depth_and_mask(cam, metric_depth.shape[-2:])
    plane_mask = plane_mask & torch.isfinite(metric_depth)

    # save plane_depth and plane_mask as img
    # im = np.concatenate([
    #             viz_tensordepth(plane_depth),
    #             im_tensor2np(plane_mask)[...,None].repeat(3, axis=-1),
    #         ], axis=1)
    # os.makedirs(os.path.join(cfg.model.model_path, "plane_depth"), exist_ok=True)
    # imageio.imwrite(
    #     os.path.join(cfg.model.model_path, "plane_depth", cam.image_name + ".jpg"),
    #     im
    # )

    if plane_mask.sum() == 0:
        return metric_depth.new_zeros(())

    return torch.log(1 + torch.abs(metric_depth[plane_mask] - plane_depth[plane_mask])).mean()


def training(args):
    # Init and load data pack
    data_pack = BackgroundDataPack(cfg.data, cfg.model.white_background)

    # Instantiate data loader
    tr_cams = data_pack.get_train_cameras()
    n_bg_iters = (cfg.procedure.n_iter + 1) // 2
    tr_cam_indices = compute_iter_idx(len(tr_cams), n_bg_iters)

    for cam in tr_cams:
        cam.is_origin_data = False

    if cfg.auto_exposure.enable:
        for cam in tr_cams:
            cam.auto_exposure_init()

    # Decide main (inside) region bounding box
    bounding = decide_main_bounding(
        cfg_bounding=cfg.bounding,
        tr_cams=tr_cams,
        pcd=data_pack.point_cloud,  # Not used
        suggested_bounding=data_pack.suggested_bounding,  # Can be None
    )

    # Init voxel model
    voxel_model = SparseVoxelModel(cfg.model)

    if args.load_iteration:
        loaded_iter = voxel_model.load_iteration(args.load_iteration) # SparseVoxelModel inherits SVInOut, so load_iteration directly assigns state in SVInOut.
    else:
        loaded_iter = None
        voxel_model.model_init(
            bounding=bounding,
            cfg_init=cfg.init,
            cameras=tr_cams)

    first_iter = loaded_iter if loaded_iter else 1
    print(f"Start optmization from iters={first_iter}.")

    # Init optimizer
    voxel_model.optimizer_init(cfg.optimizer)
    if loaded_iter and args.load_optimizer:
        voxel_model.optimizer_load_iteration(loaded_iter)

    # Init lr warmup scheduler
    if first_iter <= cfg.optimizer.n_warmup:
        rate = max(first_iter - 1, 0) / cfg.optimizer.n_warmup
        for param_group in voxel_model.optimizer.param_groups:
            param_group["base_lr"] = param_group["lr"]
            param_group["lr"] = rate * param_group["base_lr"]

    # Init subdiv
    remain_subdiv_times = sum(
        (i >= first_iter)
        for i in range(
            cfg.procedure.subdivide_from, cfg.procedure.subdivide_until+1,
            cfg.procedure.subdivide_every
        )
    )
    subdivide_scale = cfg.procedure.subdivide_target_scale ** (1 / remain_subdiv_times)
    subdivide_prop = max(0, (subdivide_scale - 1) / 7)
    print(f"Subdiv: times={remain_subdiv_times:2d} scale-each-time={subdivide_scale*100:.1f}% prop={subdivide_prop*100:.1f}%")

    # Some other initialization
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    elapsed = 0

    tr_render_opt = {
        'track_max_w': True,
        'lambda_R_concen': cfg.regularizer.lambda_R_concen,
        'output_T': False,
        'output_depth': False,
        'ss': 1.0,  # disable supersampling at first
        'rand_bg': cfg.regularizer.rand_bg,
        'use_auto_exposure': cfg.auto_exposure.enable,
    }

    nd_loss = loss_utils.NormalDepthConsistencyLoss(
        iter_from=cfg.regularizer.n_dmean_from,
        iter_end=cfg.regularizer.n_dmean_end,
        ks=cfg.regularizer.n_dmean_ks,
        tol_deg=cfg.regularizer.n_dmean_tol_deg)
    nmed_loss = loss_utils.NormalMedianConsistencyLoss(
        iter_from=cfg.regularizer.n_dmed_from,
        iter_end=cfg.regularizer.n_dmed_end)
    
    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0
    ema_photo_for_log = 0.0
    iter_rng = range(first_iter, cfg.procedure.n_iter+1)
    progress_bar = tqdm(iter_rng, desc="Training", ascii=True)
    for iteration in iter_rng:

        # Start processing time tracking of this iteration
        iter_start.record()

        # Increase the degree of SH by one up to a maximum degree
        if iteration % 1000 == 0:
            voxel_model.sh_degree_add1()

        # Recompute sh from cameras
        if iteration in cfg.procedure.reset_sh_ckpt:
            print("Reset sh0 from cameras.")
            print("Reset shs to zero.")
            voxel_model.reset_sh_from_cameras(tr_cams)
            torch.cuda.empty_cache()

        # Use default super-sampling option
        if iteration > 1000:
            if cfg.regularizer.ss_aug_max > 1:
                tr_render_opt['ss'] = np.random.uniform(1, cfg.regularizer.ss_aug_max)
            elif 'ss' in tr_render_opt:
                tr_render_opt.pop('ss')  # Use default ss

        need_abs_depth = cfg.regularizer.lambda_abs_depth > 0
        need_nd_loss = cfg.regularizer.lambda_normal_dmean > 0 and nd_loss.is_active(iteration)
        need_nmed_loss = cfg.regularizer.lambda_normal_dmed > 0 and nmed_loss.is_active(iteration)
        tr_render_opt['output_T'] = cfg.regularizer.lambda_T_concen > 0 or cfg.regularizer.lambda_T_inside > 0 or need_nd_loss
        tr_render_opt['output_normal'] = need_nd_loss or need_nmed_loss
        tr_render_opt['output_depth'] = need_abs_depth or need_nd_loss or need_nmed_loss

        if iteration >= cfg.regularizer.dist_from and cfg.regularizer.lambda_dist:
            tr_render_opt['lambda_dist'] = cfg.regularizer.lambda_dist

        if iteration >= cfg.regularizer.rectifiy_from and cfg.regularizer.lambda_rectify:
            # Lazy implementation. Change to rectify mode.
            tr_render_opt['lambda_ascending'] = 0
            
        if iteration > cfg.regularizer.scaling_penalty_from and cfg.regularizer.lambda_scaling_penalty:
            tr_render_opt['lambda_scaling_penalty'] = cfg.regularizer.lambda_scaling_penalty
            tr_render_opt['min_voxel_size'] = voxel_model.vox_size.min()
        if iteration > cfg.regularizer.scaling_penalty_end:
            tr_render_opt['lambda_scaling_penalty'] = 0

        # Update auto exposure
        if cfg.auto_exposure.enable and iteration in cfg.auto_exposure.auto_exposure_upd_ckpt:
            for cam in tr_cams:
                with torch.no_grad():
                    ref = voxel_model.render(cam, ss=1.0)['color']
                cam.auto_exposure_update(ref, cam.image.cuda())

        bg_iter_id = (iteration - 1) // 2
        cam = tr_cams[tr_cam_indices[bg_iter_id]]

        # Get gt image
        gt_image = cam.image.cuda()
        
        if cfg.regularizer.lambda_R_concen > 0:
            tr_render_opt['gt_color'] = gt_image
        
        if 'vox_feats' in tr_render_opt:
            tr_render_opt.pop("vox_feats")

        # Render
        render_pkg = voxel_model.render(cam, **tr_render_opt)
        render_image = render_pkg['color']

        mse = loss_utils.l2_loss(render_image, gt_image)
        photo_loss = mse
        loss = 0
        
        abs_loss = plane_refined_depth_l1_loss(cam, render_pkg)
        loss += cfg.regularizer.lambda_abs_depth * abs_loss
        if cfg.regularizer.lambda_T_concen:
            loss += cfg.regularizer.lambda_T_concen * loss_utils.prob_concen_loss(render_pkg['raw_T'])
        if cfg.regularizer.lambda_T_inside:
            loss += cfg.regularizer.lambda_T_inside * render_pkg['raw_T'].square().mean()
        if need_nd_loss:
            loss += cfg.regularizer.lambda_normal_dmean * nd_loss(cam, render_pkg, iteration)
        if need_nmed_loss:
            loss += cfg.regularizer.lambda_normal_dmed * nmed_loss(cam, render_pkg, iteration)
            
            
        if iteration > cfg.regularizer.W_enlarge_from:
            max_w = render_pkg["max_w"]
            if max_w.max() > 0.5:
                loss += 0.2 * torch.log((1 - max_w[max_w > 0.5]) + 1).mean()
        

        # Backward to get gradient of current iteration
        voxel_model.optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Grid-level regularization
        grid_reg_interval = iteration >= cfg.regularizer.tv_from and iteration <= cfg.regularizer.tv_until
        if cfg.regularizer.lambda_tv_density and grid_reg_interval:
            lambda_tv_mult = cfg.regularizer.tv_decay_mult ** (iteration // cfg.regularizer.tv_decay_every)
            svraster_cuda.grid_loss_bw.total_variation(
                grid_pts=voxel_model._geo_grid_pts,
                vox_key=voxel_model.vox_key,
                weight=cfg.regularizer.lambda_tv_density * lambda_tv_mult,
                vox_size_inv=voxel_model.vox_size_inv,
                no_tv_s=True,
                tv_sparse=cfg.regularizer.tv_sparse,
                grid_pts_grad=voxel_model._geo_grid_pts.grad)

        # Optimizer step
        voxel_model.optimizer.step()

        # Learning rate warmup scheduler step
        if iteration <= cfg.optimizer.n_warmup:
            rate = iteration / cfg.optimizer.n_warmup
            for param_group in voxel_model.optimizer.param_groups:
                param_group["lr"] = rate * param_group["base_lr"]

        if iteration in cfg.optimizer.lr_decay_ckpt:
            for param_group in voxel_model.optimizer.param_groups:
                ori_lr = param_group["lr"]
                param_group["lr"] *= cfg.optimizer.lr_decay_mult
                print(f'LR decay of {param_group["name"]}: {ori_lr} => {param_group["lr"]}')

        ######################################################
        # Gradient statistic should happen before adaptive op
        ######################################################

        need_stat = (
            iteration >= 500 and \
            iteration <= cfg.procedure.subdivide_until)
        if need_stat:
            voxel_model.subdiv_meta += voxel_model._subdiv_p.grad

        ######################################################
        # Start adaptive voxels pruning and subdividing
        ######################################################

        need_pruning = (
            iteration % cfg.procedure.prune_every == 0 and \
            iteration >= cfg.procedure.prune_from and \
            iteration <= cfg.procedure.prune_until)
        need_subdividing = (
            iteration % cfg.procedure.subdivide_every == 0 and \
            iteration >= cfg.procedure.subdivide_from and \
            iteration <= cfg.procedure.subdivide_until and \
            voxel_model.num_voxels < cfg.procedure.subdivide_max_num)

        # Do nothing in last 500 iteration
        need_pruning &= (iteration <= cfg.procedure.n_iter-1500)
        need_subdividing &= (iteration <= cfg.procedure.n_iter-1500)

        if need_pruning or need_subdividing:
            stat_pkg = voxel_model.compute_training_stat(camera_lst=tr_cams)
            torch.cuda.empty_cache()
        
        if need_pruning:
            ori_n = voxel_model.num_voxels

            # Compute pruning threshold
            prune_all_iter = max(1, cfg.procedure.prune_until - cfg.procedure.prune_every)
            prune_now_iter = max(0, iteration - cfg.procedure.prune_every)
            prune_iter_rate = max(0, min(1, prune_now_iter / prune_all_iter))
            thres_inc = max(0, cfg.procedure.prune_thres_final - cfg.procedure.prune_thres_init)
            prune_thres = cfg.procedure.prune_thres_init + thres_inc * prune_iter_rate

            # Prune voxels
            prune_mask = (stat_pkg['max_w'] < prune_thres).squeeze(1)

            voxel_model.pruning(prune_mask)

            # Prune statistic (for the following subdivision)
            kept_idx = (~prune_mask).argwhere().squeeze(1)
            for k, v in stat_pkg.items():
                stat_pkg[k] = v[kept_idx]

            new_n = voxel_model.num_voxels
            print(f'[PRUNING]     {ori_n:7d} => {new_n:7d} (x{new_n/ori_n:.2f};  thres={prune_thres:.4f})')
            torch.cuda.empty_cache()

        if need_subdividing:
            # Exclude some voxels
            size_thres = stat_pkg['min_samp_interval'] * cfg.procedure.subdivide_samp_thres
            large_enough_mask = (voxel_model.vox_size * 0.5 > size_thres).squeeze(1)
            non_finest_mask = voxel_model.octlevel.squeeze(1) < svraster_cuda.meta.MAX_NUM_LEVELS
            valid_mask = large_enough_mask & non_finest_mask

            # Get some statistic for subdivision priority
            priority = voxel_model.subdiv_meta.squeeze(1) * valid_mask

            # Compute priority rank (larger value has higher priority)
            rank = torch.zeros_like(priority)
            rank[priority.argsort()] = torch.arange(len(priority), dtype=torch.float32, device="cuda")

            # Determine the number of voxels to subdivided
            if iteration <= cfg.procedure.subdivide_all_until:
                thres = -1
            else:
                thres = rank.quantile(1 - subdivide_prop)

            # Compute subdivision mask
            subdivide_mask = (rank > thres) & valid_mask

            # In case the number of voxels over the threshold
            max_n_subdiv = round((cfg.procedure.subdivide_max_num - voxel_model.num_voxels) / 7)
            if subdivide_mask.sum() > max_n_subdiv:
                n_removed = subdivide_mask.sum() - max_n_subdiv
                subdivide_mask &= (rank > rank[subdivide_mask].sort().values[n_removed-1])

            # Subdivision
            ori_n = voxel_model.num_voxels
            if subdivide_mask.sum() > 0:
                voxel_model.subdividing(subdivide_mask, cfg.procedure.subdivide_save_gpu)
            new_n = voxel_model.num_voxels
            in_p = voxel_model.inside_mask.float().mean().item()
            print(f'[SUBDIVIDING] {ori_n:7d} => {new_n:7d} (x{new_n/ori_n:.2f}; inside={in_p*100:.1f}%)')

            voxel_model.subdiv_meta.zero_()  # reset subdiv meta
            remain_subdiv_times -= 1
            torch.cuda.empty_cache()

        ######################################################
        # End of adaptive voxels procedure
        ######################################################
        
                
        # End processing time tracking of this iteration
        iter_end.record()
        torch.cuda.synchronize()
        elapsed += iter_start.elapsed_time(iter_end)

        # Logging
        with torch.no_grad():
            # Metric
            loss = loss.item()
            psnr = -10 * np.log10(mse.item())

            # Progress bar
            # ema_p = max(0.01, 1 / (iteration - first_iter + 1))
            ema_loss_for_log = 0.6 * ema_loss_for_log + 0.4 * loss
            ema_psnr_for_log = 0.6 * ema_psnr_for_log + 0.4 * psnr
            ema_photo_for_log = 0.6 * ema_photo_for_log + 0.4 * photo_loss.item()
            if iteration % 10 == 0:
                pb_text = {
                    "Loss": f"{ema_loss_for_log:.5f}",
                    "photo": f"{ema_photo_for_log:.4f}",
                    "psnr": f"{ema_psnr_for_log:.2f}",
                }
                progress_bar.set_postfix(pb_text)
                progress_bar.update(10)
            if iteration == cfg.procedure.n_iter:
                progress_bar.close()

            # Log and save
            training_report(
                data_pack=data_pack,
                voxel_model=voxel_model,
                iteration=iteration,
                loss=loss,
                psnr=psnr,
                elapsed=elapsed,
                ema_psnr=ema_psnr_for_log,
                pg_view_every=args.pg_view_every,
                test_iterations=args.test_iterations)

            if iteration in args.checkpoint_iterations or iteration == cfg.procedure.n_iter:
                voxel_model.save_iteration(iteration, quantize=args.save_quantized)
                if args.save_optimizer:
                    voxel_model.optimizer_save_iteration(iteration)
                print(f"[SAVE] path={voxel_model.latest_save_path}")
    
    # ── Final render pass: save RGB / normal / depth for all train views ──
    print("[FINAL RENDER] Rendering all train views...")
    torch.cuda.empty_cache()
    voxel_model.freeze_vox_geo()

    rgb_dir    = os.path.join(voxel_model.model_path, "rgbs")
    normal_dir = os.path.join(voxel_model.model_path, "normals")
    depth_dir  = os.path.join(voxel_model.model_path, "depths")
    for d in (rgb_dir, normal_dir, depth_dir):
        os.makedirs(d, exist_ok=True)

    with torch.no_grad():
        for cam in tqdm(tr_cams, desc="Final render", ascii=True):
            render_pkg = voxel_model.render(
                cam,
                output_depth=True,
                output_normal=True,
                output_T=True,
            )
            fname = cam.image_name
            alpha = 1 - render_pkg['T'][0]

            # RGB
            imageio.imwrite(
                os.path.join(rgb_dir, fname + ".jpg"),
                im_tensor2np(render_pkg['color'])
            )

            # Normal (world-space, flipped convention matching render.py)
            render_normal = render_pkg['normal'] * -1
            imageio.imwrite(
                os.path.join(normal_dir, fname + ".jpg"),
                im_tensor2np(render_normal * 0.5 + 0.5)
            )

            # Depth-derived normal
            depth2normal = render_normal_func(cam, render_pkg['depth'][0].squeeze())
            imageio.imwrite(
                os.path.join(normal_dir, fname + "_depth2normal.jpg"),
                im_tensor2np(depth2normal * 0.5 + 0.5)
            )

            # Depth visualisation (skip views with no visible voxels)
            if (render_pkg['depth'][0] > 0).any():
                imageio.imwrite(
                    os.path.join(depth_dir, fname + ".jpg"),
                    viz_tensordepth(render_pkg['depth'][0], alpha)
                )

    voxel_model.unfreeze_vox_geo()
    print(f"[FINAL RENDER] Done. Saved to {voxel_model.model_path}/{{rgbs,normals,depths}}/")


def training_report(data_pack, voxel_model, iteration, loss, psnr, elapsed, ema_psnr, pg_view_every, test_iterations):

    voxel_model.freeze_vox_geo()

    # Progress view
    if pg_view_every > 0 and (iteration % pg_view_every == 0 or iteration == 1):
        torch.cuda.empty_cache()
        test_cameras = data_pack.get_test_cameras()
        if len(test_cameras) == 0:
            test_cameras = data_pack.get_train_cameras()
        pg_idx = iteration // pg_view_every
        # pg_idx = 0
        # pg_idx = 131
        view = test_cameras[pg_idx % len(test_cameras)]
        
        tr_render_opt = {"vox_feats": voxel_model.octlevel * 1.0}
        
        render_pkg = voxel_model.render(view, output_depth=True, output_normal=True, output_T=True, **tr_render_opt)
        render_image = render_pkg['color']
        render_depth = render_pkg['depth'][0]
        render_depth_med = render_pkg['depth'][2]
        render_normal = render_pkg['normal']
        render_alpha = 1 - render_pkg['T'][0]
        
        render_normal_from_points = render_normal_func(view, render_depth.squeeze())
        render_normal_med_from_points = render_normal_func(view, render_depth_med.squeeze())
        
        render_vox_level = render_pkg['feat']/(1-render_pkg['T'][0]).clamp(min=0.1).squeeze().detach()
        level_weight = (render_vox_level.max()-render_vox_level.min())/(render_vox_level-render_vox_level.min()).clamp(min=1.0)
                
        im = np.concatenate([
            np.concatenate([
                im_tensor2np(render_image),
                im_tensor2np(render_image),
                im_tensor2np(render_image),
                im_tensor2np(level_weight / level_weight.max())[...,None].repeat(3, axis=-1),
            ], axis=1),
            np.concatenate([
                viz_tensordepth(render_depth, render_alpha),
                im_tensor2np(render_normal * 0.5 + 0.5),
                im_tensor2np(render_normal_from_points * 0.5 + 0.5),
                im_tensor2np(render_vox_level / render_vox_level.max())[...,None].repeat(3, axis=-1),
            ], axis=1),
            np.concatenate([
                im_tensor2np(view.depth2normal(render_depth) * 0.5 + 0.5),
                im_tensor2np(view.depth2normal(render_depth_med) * 0.5 + 0.5),
                viz_tensordepth(render_depth_med, render_alpha),
                im_tensor2np(render_vox_level / render_vox_level.max())[...,None].repeat(3, axis=-1),
            ], axis=1),
        ], axis=0)
        torch.cuda.empty_cache()

        outdir = os.path.join(voxel_model.model_path, "pg_view")
        outpath = os.path.join(outdir, f"iter{iteration:06d}.jpg")
        os.makedirs(outdir, exist_ok=True)

        imageio.imwrite(outpath, im)

        eps_file = os.path.join(voxel_model.model_path, "pg_view", "eps.txt")
        with open(eps_file, 'a') as f:
            f.write(f"{iteration},{elapsed/1000:.1f}\n")

    # Report test and samples of training set
    if iteration in test_iterations:
        print(f"[EVAL] running...")
        torch.cuda.empty_cache()
        test_cameras = data_pack.get_test_cameras()
        save_every = max(1, len(test_cameras) // 8)
        outdir = os.path.join(voxel_model.model_path, "test_view")
        os.makedirs(outdir, exist_ok=True)
        psnr_lst = []
        video = []
        max_w = torch.zeros([voxel_model.num_voxels, 1], dtype=torch.float32, device="cuda")
        for idx, camera in enumerate(test_cameras):
            render_pkg = voxel_model.render(camera, output_normal=True, track_max_w=True)
            render_image = render_pkg['color']
            im = im_tensor2np(render_image)
            gt = im_tensor2np(camera.image)
            video.append(im)
            if idx % save_every == 0:
                outpath = os.path.join(outdir, f"idx{idx:04d}_iter{iteration:06d}.jpg")
                cat = np.concatenate([gt, im], axis=1)
                imageio.imwrite(outpath, cat)

                outpath = os.path.join(outdir, f"idx{idx:04d}_iter{iteration:06d}_normal.jpg")
                render_normal = render_pkg['normal']
                render_normal = im_tensor2np(render_normal * 0.5 + 0.5)
                imageio.imwrite(outpath, render_normal)
            mse = np.square(im/255 - gt/255).mean()
            psnr_lst.append(-10 * np.log10(mse))
            max_w = torch.maximum(max_w, render_pkg['max_w'])
        avg_psnr = np.mean(psnr_lst)
        imageio.mimwrite(
            os.path.join(outdir, f"video_iter{iteration:06d}.mp4"),
            video, fps=30)
        torch.cuda.empty_cache()

        fps = time.time()
        for idx, camera in enumerate(test_cameras):
            voxel_model.render(camera, track_max_w=False)
        torch.cuda.synchronize()
        fps = len(test_cameras) / (time.time() - fps)
        torch.cuda.empty_cache()

        # Sample training views to render
        train_cameras = data_pack.get_train_cameras()
        for idx in range(0, len(train_cameras), max(1, len(train_cameras)//8)):
            camera = train_cameras[idx]
            render_pkg = voxel_model.render(
                camera, output_normal=True, track_max_w=True,
                use_auto_exposure=cfg.auto_exposure.enable)
            render_image = render_pkg['color']
            im = im_tensor2np(render_image)
            gt = im_tensor2np(camera.image)
            outpath = os.path.join(outdir, f"train_idx{idx:04d}_iter{iteration:06d}.jpg")
            cat = np.concatenate([gt, im], axis=1)
            imageio.imwrite(outpath, cat)

            outpath = os.path.join(outdir, f"train_idx{idx:04d}_iter{iteration:06d}_normal.jpg")
            render_normal = render_pkg['normal']
            render_normal = im_tensor2np(render_normal * 0.5 + 0.5)
            imageio.imwrite(outpath, render_normal)

        print(f"[EVAL] iter={iteration:6d}  psnr={avg_psnr:.2f}  fps={fps:.0f}")

        outdir = os.path.join(voxel_model.model_path, "test_stat")
        outpath = os.path.join(outdir, f"iter{iteration:06d}.json")
        os.makedirs(outdir, exist_ok=True)
        with open(outpath, 'w') as f:
            q = torch.linspace(0,1,5, device="cuda")
            max_w_q = max_w.quantile(q).tolist()
            peak_mem = torch.cuda.memory_stats()["allocated_bytes.all.peak"] / 1024 ** 3
            stat = {
                'psnr': avg_psnr,
                'ema_psnr': ema_psnr,
                'elapsed': elapsed,
                'fps': fps,
                'n_voxels': voxel_model.num_voxels,
                'max_w_q': max_w_q,
                'peak_mem': peak_mem,
            }
            json.dump(stat, f, indent=4)

    voxel_model.unfreeze_vox_geo()



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Background sparse voxel training with See3D RGB views and "
            "hybrid depth supervision from plane refined depth + DepthAnythingV2."
        )
    )

    parser.add_argument("--guidance_root", required=True, type=str)
    parser.add_argument("--merge_root", default=None, type=str)
    parser.add_argument("--dn_root", default=None, type=str)
    parser.add_argument("--mono_cache_root", default=None, type=str)

    parser.add_argument("--cfg_files", default=[], nargs="*")
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--test_iterations", nargs="*", type=int, default=[-1])
    parser.add_argument("--pg_view_every", type=int, default=200)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--load_iteration", type=int, default=None)
    parser.add_argument("--load_optimizer", action="store_true")
    parser.add_argument("--save_optimizer", action="store_true")
    parser.add_argument("--save_quantized", action="store_true")
    parser.add_argument("--exp_name", type=str, default="train_bg_voxel_model")

    args, cmd_lst = parser.parse_known_args()

    if args.merge_root is None:
        args.merge_root = os.path.join(args.guidance_root, "merge_3d_plane")
    if args.dn_root is None:
        args.dn_root = os.path.join(args.guidance_root, "see3d_mono_dn")
    if args.mono_cache_root is None:
        args.mono_cache_root = os.path.join(args.guidance_root, "da2_mono_cache")

    update_config(args.cfg_files, cmd_lst)

    # Use the dedicated See3D mono cache so relative depth supervision
    # is computed on inpainted background RGB views.
    cfg.data.source_path = args.mono_cache_root

    BG_ROOTS["guidance_root"] = args.guidance_root
    BG_ROOTS["merge_root"] = args.merge_root
    BG_ROOTS["dn_root"] = args.dn_root


    seed_everything(cfg.procedure.seed)
    torch.cuda.set_device(torch.device("cuda:0"))
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    cfg.model.model_path = os.path.join(args.guidance_root, args.exp_name)
    os.makedirs(cfg.model.model_path, exist_ok=True)
    with open(os.path.join(cfg.model.model_path, "config.yaml"), "w") as f:
        f.write(cfg.dump())

    print(f"Output folder: {cfg.model.model_path}")
    print(f"Guidance root : {args.guidance_root}")
    print(f"Merge root    : {args.merge_root}")
    print(f"DN root       : {args.dn_root}")
    print(f"Mono cache    : {args.mono_cache_root}")

    if cfg.procedure.sche_mult != 1:
        sche_mult = cfg.procedure.sche_mult

        cfg.optimizer.n_warmup = round(cfg.optimizer.n_warmup * sche_mult)
        cfg.optimizer.lr_decay_ckpt = [
            round(v * sche_mult) if v > 0 else v
            for v in cfg.optimizer.lr_decay_ckpt
        ]

        for key in [
            "dist_from",
            "tv_from",
            "tv_until",
            "n_dmean_from",
            "n_dmean_end",
            "n_dmed_from",
            "n_dmed_end",
        ]:
            cfg.regularizer[key] = round(cfg.regularizer[key] * sche_mult)

        for key in [
            "n_iter",
            "prune_from",
            "prune_every",
            "prune_until",
            "subdivide_from",
            "subdivide_every",
            "subdivide_until",
        ]:
            cfg.procedure[key] = round(cfg.procedure[key] * sche_mult)

        cfg.procedure.reset_sh_ckpt = [
            round(v * sche_mult) if v > 0 else v
            for v in cfg.procedure.reset_sh_ckpt
        ]

    for i in range(len(args.test_iterations)):
        if args.test_iterations[i] < 0:
            args.test_iterations[i] += cfg.procedure.n_iter + 1

    for i in range(len(args.checkpoint_iterations)):
        if args.checkpoint_iterations[i] < 0:
            args.checkpoint_iterations[i] += cfg.procedure.n_iter + 1

    training(args)
    print("Everything done.")