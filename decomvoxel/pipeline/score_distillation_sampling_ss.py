"""
Score Distillation Sampling (SDS) for Sparse Structure Completion
Original version: exp_single_obj, exp_0218, exp_0222
Optimize sparse structure voxel values directly with sigmoid activation
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from datetime import datetime
from typing import Optional, Dict, Tuple, List
from PIL import Image
from dataclasses import dataclass, field

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for server
except ImportError:
    print("[Warning] matplotlib not available, skipping plot generation")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

TRELLIS_ROOT = os.path.join(PROJECT_ROOT, 'decomvoxel', 'model', 'TRELLIS')
if TRELLIS_ROOT not in sys.path:
    sys.path.insert(0, TRELLIS_ROOT)
    
from decomvoxel.pipeline.visualize_sparse_structure import visualize_sparse_structure

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEBUG = False

@dataclass
class SDSConfig:
    rotate_initial_structure: bool = True  # Whether to rotate the initial sparse structure 
    rotation_angle_z: float = 90.0  # Rotation angle in degrees (CCW positive) around Z-axis. Only used when rotate_initial_structure=True. Legacy default 90° matches old rotate_z_90(clockwise=False).
    total_iters: int = 5000
    lr: float = 0.01
    noise_start: float = 0.02
    noise_end: float = 0.98
    weighting_strategy: str = "snr"  # "snr" or "uniform"
    sds_weight: float = 1.0
    mask_weight: float = 0.5  # Weight for preserving known regions (lower = more aggressive completion)
    mask_weight_schedule: str = "cosine"  # Schedule for mask_weight over training: "const", "linear", "cosine", "poly", "exponential", "logarithmic" (strong early, weak late)
    mask_poly_power: float = 2.0  # Exponent for "poly" schedule (2~5 recommended; larger = steeper late-stage decay)
    mask_exp_gamma: float = 5.0   # Decay rate for "exponential" schedule (3~8 recommended; larger = faster decay)
    mask_end_iter: int = -1       # Iteration after which mask_weight is forced to 0 (-1 = use total_iters, i.e. no early cutoff)
    # Blank Region Preservation Loss:
    # Penalizes grid values rising in voxels that were EMPTY in the initial sparse structure.
    # Per-voxel weight = (1 - blank_space_uncertainty) = blank certainty.
    # High certainty (certain it should be empty) → high penalty; high uncertainty → low penalty.
    # Requires blank_space_uncertainty tensor passed to complete() / run_sds_completion_ss().
    use_blank_region_loss: bool = False
    blank_region_weight: float = 1.0       # Overall scalar weight for this loss term
    blank_region_weight_schedule: str = "cosine"  # Schedule over training: "const", "linear", "cosine", "cos_power", "poly", "exponential", "logarithmic"
    blank_region_poly_power: float = 2.0   # Exponent for "poly" / "cos_power" schedules
    blank_region_exp_gamma: float = 5.0    # Decay rate for "exponential" schedule
    blank_region_end_iter: int = -1        # Iteration after which blank_region_weight is forced to 0 (-1 = use total_iters)
    # SDS weight scheduler based on x_pred:
    # Each iteration, decode TRELLIS x_0_pred to voxel space and measure how well it agrees
    # with the initial_ss in the known/certain region. High agreement → prediction is trustworthy
    # → large sds_schedule_w coefficient; low agreement → suppress SDS loss for this step.
    use_sds_x_pred_schedule: bool = False  # Whether to gate SDS loss by x_0_pred agreement with initial_ss
    # Known Region Preservation Sampling Loss:
    # Each iteration, decode both the current latents (grad path) and TRELLIS x_0_pred (no grad) to voxel space.
    # Loss = MSE(decoded(latents), initial_ss) weighted by: certainty × (1 - |decoded(x_0_pred) - initial_ss|)
    # Weight is high where TRELLIS agrees with the initial structure (trustworthy step) →
    # strongly push ss_param to reconstruct the known structure there.
    # Gradient path: loss → decoded(latents) → decoder (frozen ops) → latents → encoder (frozen ops) → ss_param
    use_known_region_sampling_loss: bool = False
    known_region_sampling_weight: float = 0.5  # Overall weight for this loss term
    sampling_weight_schedule: str = "const"  # Schedule for known_region_sampling_weight: "const", "linear", "cosine", "poly", "exponential", "logarithmic"
    sampling_poly_power: float = 2.0  # Exponent for "poly" schedule
    sampling_exp_gamma: float = 5.0   # Decay rate for "exponential" schedule
    sampling_end_iter: int = -1       # Iteration after which sampling loss weight is forced to 0 (-1 = use total_iters)
    cfg_strength: float = 3.0  # Classifier-free guidance strength (higher = stronger conditioning)
    sds_loss_type: str = 'classic'  # 'classic' for gradient-based SDS, 'mse' for simplified MSE loss
    refine_enabled: bool = True
    refine_noise: float = 0.3
    refine_steps: int = 25
    print_interval: int = 100
    save_interval: int = 500
    timestep_schedule: str = "uniform" # "uniform", "linear", "bell", "progressive"
    linear_anneal_iter: int = 1000  # for "linear": iteration at which t_low reaches noise_start
    warmup_iters: int = 0 # 500
    warmup_t_min: float = 0.5  # During warmup, sample t from [warmup_t_min → noise_start, noise_end]
    use_rescaled_t: bool = False  # True: apply TRELLIS rescale formula; False: use original t ∈ [0,1]
    rescale_t: float = 3.0  # Rescale factor for t (TRELLIS default=3.0). Formula: t' = r*t/(1+(r-1)*t), keeps t' ∈ [0,1]
    init_noise_scale: float = 1.0  # Std of normal noise added to empty regions in logit space (0 = no noise, keep logit(0.01)≈-4.6)
    empty_init_value: float = 0.2  # Initial probability value for empty voxels in [0,1]. logit(empty_init_value) sets starting point
    # TV regularization: penalize high-frequency noise by encouraging local smoothness
    use_tv_loss: bool = True
    tv_weight: float = 0.001  # Weight for 3D total variation loss (0.0001~0.01)
    # Entropy regularization: gently push toward binary in late stage only
    use_entropy_loss: bool = False
    entropy_weight: float = 1e-5  # Max weight for entropy loss (very small to avoid killing gradients)
    entropy_start_ratio: float = 0.6  # Start entropy regularization at this fraction of total_iters (0.0~1.0)
    # Periodic smoothing: apply 3D Gaussian blur or median filter to suppress isolated noise
    use_periodic_smooth: bool = False  # Enable periodic smoothing of ss_param
    smooth_interval: int = 500  # Apply smoothing every N iterations
    smooth_method: str = 'gaussian'  # 'gaussian' or 'median'
    smooth_sigma: float = 0.5  # Gaussian blur sigma (only for gaussian method)
    smooth_kernel_size: int = 3  # Kernel size for median filter (only for median method, must be odd)
    smooth_strength: float = 0.3  # Blending factor: new = (1-s)*original + s*smoothed, 0=no effect, 1=full replace
    smooth_start_ratio: float = 0.1  # Only start smoothing after this fraction of total_iters
    smooth_preserve_known: bool = True  # If True, do not smooth known (occupied) voxels
    # Periodic local smoothing: uncertainty-adaptive smoothing based on accumulated x0_pred density
    # Each iter, x0_pred (latent) is accumulated. At smoothing time, decode the average to get a
    # per-voxel density estimate. Low density → high uncertainty → stronger smoothing.
    use_periodic_local_smooth: bool = False
    local_smooth_interval: int = 500  
    local_smooth_method: str = 'gaussian'  # 'gaussian' or 'median' 
    local_smooth_sigma: float = 0.5  # Gaussian sigma for smoothing (only for gaussian)
    local_smooth_kernel_size: int = 3  # Kernel size for median filter (only for median, must be odd)
    local_smooth_density_sigma: float = 2.0  # Sigma for heavy smoothing of the density estimate (higher = smoother density map)
    local_smooth_max_strength: float = 0.5  # Max blending factor at low-density (uncertain) voxels
    local_smooth_min_strength: float = 0.05  # Min blending factor at high-density (confident) voxels
    local_smooth_start_ratio: float = 0.1  # Only start local smoothing after this fraction of total_iters
    local_smooth_end_ratio: float = 1.0  # Stop local smoothing after this fraction of total_iters (1.0 = no early stop)
    local_smooth_preserve_known: bool = True  
    local_smooth_accum_decay: float = 0.0  # Exponential decay for accumulator (0 = no decay, simple average; >0 = more recent iters weighted higher)
    # Periodic pruning: suppress "certainly blank" voxels during optimization.
    # Protected region  : certainty_grid > 0  (known occupied)  → never pruned.
    # Prunable region   : blank_space_uncertainty == 0  (certainly empty, outside certainty_grid).
    # Requires blank_space_uncertainty to be passed to complete() / run_sds_completion_ss().
    use_periodic_prune: bool = False
    prune_interval: int = 500          # Apply pruning every N iterations within [prune_start_iter, prune_end_iter]
    prune_start_iter: int = 0          # First iteration at which pruning is applied (inclusive)
    prune_end_iter: int = -1           # Last  iteration at which pruning is applied (-1 = total_iters)
    prune_strength: float = 1.0        # 1.0 = hard-set logit to prune_empty_logit; <1.0 = lerp toward it
    prune_empty_logit: float = -7.0    # Target logit for pruned voxels (sigmoid(-7) ≈ 0.001 ≈ 0)
    # Redistribute mode: instead of collapsing pruned voxels to a single low value, keep the
    # overall mean roughly intact (or nudge it slightly downward) while spreading the values into
    # a uniform distribution. This prevents the optimizer from seeing an abrupt hard-zero signal
    # while still discouraging occupancy in certainly-blank regions.
    #   prune_mode = "suppress"     → original behaviour (lerp / hard-set to prune_empty_logit)
    #   prune_mode = "redistribute" → sample Uniform(target_mean ± prune_redistribute_uniform_range)
    prune_mode: str = "suppress"                   # "suppress" or "redistribute"
    prune_redistribute_mean_factor: float = 0.1    # How far to shift the mean toward prune_empty_logit
                                                   # 0.0 = exact current mean, 1.0 = full shift to prune_empty_logit
    prune_redistribute_uniform_range: float = 2.0  # Half-width of the uniform distribution (logit units)
    # Schedule for prune_redistribute_mean_factor over the prune window [prune_start_iter, prune_end_iter].
    # Progress p ∈ [0,1] is normalised to that window; schedule w(p) ∈ [1→0].
    # effective_mean_factor = prune_redistribute_mean_factor × w(p).
    # "const" → always at the configured value; decreasing schedules (cosine/linear/…) taper it to 0.
    prune_redistribute_schedule: str = "const"    # "const", "linear", "cosine", "cos_power", "poly", "exponential", "logarithmic"
    prune_redistribute_poly_power: float = 2.0    # Exponent for "poly" / "cos_power" schedules
    prune_redistribute_exp_gamma: float = 5.0     # Decay rate for "exponential" schedule
    # Blank-region noise fill: re-inject noise into certainly-blank voxels (_prune_mask) to
    # prevent them from being permanently locked out of completion after pruning.
    # Noise std = max(current_logit_std_in_mask, blank_noise_min_scale) * blank_noise_scale.
    use_blank_noise_fill: bool = False
    blank_noise_start_iter: int = 0       # First iteration at which noise fill is applied (inclusive)
    blank_noise_end_iter: int = -1        # Last  iteration at which noise fill is applied (-1 = total_iters)
    blank_noise_interval: int = 500       # Apply noise fill every N iterations within [start, end]
    blank_noise_scale: float = 1.0        # Multiplier on the computed logit std within the mask
    blank_noise_min_scale: float = 0.5    # Minimum noise std in logit space (fallback when std ≈ 0 after hard prune)
    # Distance-decay prior loss: penalize voxels that are far from the current known structure.
    # Encourages completion to grow near existing occupied regions rather than scattered far away.
    # Prior weight map = exp(-dist / sigma), so far voxels have higher penalty when activated.
    use_dist_decay_loss: bool = False
    dist_decay_weight: float = 0.01  # Overall weight for the distance-decay loss
    dist_decay_sigma: float = 5.0  # Decay length scale in voxels (larger = allow farther completion)
    dist_decay_on_known: bool = False  # If True, also penalize known voxels (usually False)
    # Boundary penalty loss: penalize voxels near the spatial boundary of the 3D volume.
    # Prevents optimization from producing spurious occupancy at grid edges.
    # Prior weight map = 1 - smoothstep from boundary inward by `boundary_margin` voxels.
    use_boundary_loss: bool = False
    boundary_weight: float = 0.01  # Overall weight for the boundary penalty loss
    boundary_margin: int = 4  # Width (in voxels) of the penalized boundary band
    boundary_sigma: float = 2.0  # Gaussian falloff sigma within the boundary band (softer edge)
    # -----------------------------------------------------------------------
    # Rendering loss: compare soft-rendered depth (from decoded latents) with
    # monocular GT depth (DepthAnythingV2) masked to the object region.
    # Gradient path:
    #   loss → rendered_depth → occ_probs = sigmoid(decoder(latents))
    #        → decoder (frozen backward) → latents
    #        → encoder (frozen backward) → ss_activated → ss_param
    # -----------------------------------------------------------------------
    use_rendering_loss: bool = False
    rendering_loss_weight: float = 0.01     # Overall weight for the rendering depth loss
    rendering_loss_schedule: str = "const"  # Weight schedule: "const", "linear", "cosine"
    rendering_start_iter: int = 0           # Start applying rendering loss after this iteration
    rendering_interval: int = 10            # Compute every N iters (0 = every iter)
    # Source occupancy for rendering loss:
    #   "current" -> render sigmoid(decoder(latents))
    #   "pred"    -> render sigmoid(decoder(x_0_pred)) with differentiable x_t->x_0_pred path
    rendering_ss_source: str = "current"   # "current" or "pred"
    rendering_source_path: str = ""         # Dataset root (e.g. datasets/Replica/scan2)
    rendering_obj_id: int = -1              # Object original_id from segm_3d instance masks
    rendering_depth_dir: str = ""           # Depth map dir; default: <source_path>/mono_priors/depthanythingv2
    rendering_mask_dir: str = ""            # Mask dir; default: <source_path>/instance_masks
    rendering_bg_id: int = 255              # Background pixel value in instance masks
    rendering_min_obj_pixels: int = 50      # Min object pixels to consider a camera valid
    rendering_depth_loss_type: str = "l1"   # "l1" or "l2" on median-normalised depth
    rendering_soft_depth_eps: float = 0.01  # Min accumulated weight for a rendered pixel to be "occupied"
    rendering_gt_depth_invert: bool = False  # Invert GT depth (True when closer=larger in the file)
    # Depth rendering occupancy policy:
    #   True  -> hard threshold (only occ_prob > rendering_occ_threshold contributes)
    #   False -> soft dense rendering (all voxels can contribute with soft geo)
    rendering_strict_occ_threshold: bool = True
    rendering_occ_threshold: float = 0.5
    rendering_soft_gate_alpha: float = 10.0
    rendering_soft_empty_geo: float = -10.0
    # Visualize intermediate t=0.4 denoised prediction at each save_interval.
    # When True, adds a fresh forward pass at t=0.4 on the current latents and
    # saves + visualizes the decoded target (target_ss_04_iter{i:06d}.pt).
    vis_mid: bool = False


class SparseStructureCompleter:
    def __init__(self, device: torch.device = device):
        self.device = device
        self.encoder = None
        self.decoder = None
        self.diffusion = None
        self.pipe = None
        self.is_initialized = False
    
    @staticmethod
    def _gaussian_smooth_3d(x: torch.Tensor, sigma: float) -> torch.Tensor:
        """
        Apply 3D Gaussian blur to a (B, C, D, H, W) tensor using separable convolutions.
        """
        # Determine kernel radius from sigma (cover 3*sigma in each direction)
        radius = max(int(round(3.0 * sigma)), 1)
        size = 2 * radius + 1
        coords = torch.arange(size, dtype=torch.float32, device=x.device) - radius
        kernel_1d = torch.exp(-0.5 * (coords / sigma) ** 2)
        kernel_1d = kernel_1d / kernel_1d.sum()
        
        # Apply separable 1D convolutions along each spatial axis
        # Conv along D (dim=2)
        k_d = kernel_1d.view(1, 1, size, 1, 1)
        # Conv along H (dim=3)
        k_h = kernel_1d.view(1, 1, 1, size, 1)
        # Conv along W (dim=4)
        k_w = kernel_1d.view(1, 1, 1, 1, size)
        
        pad = radius
        out = F.conv3d(x, k_d.expand(x.shape[1], 1, -1, -1, -1), padding=(pad, 0, 0), groups=x.shape[1])
        out = F.conv3d(out, k_h.expand(x.shape[1], 1, -1, -1, -1), padding=(0, pad, 0), groups=x.shape[1])
        out = F.conv3d(out, k_w.expand(x.shape[1], 1, -1, -1, -1), padding=(0, 0, pad), groups=x.shape[1])
        return out
    
    @staticmethod
    def _median_smooth_3d(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
        """
        Apply 3D median filter to a (B, C, D, H, W) tensor using unfold.
        """
        pad = kernel_size // 2
        x_padded = F.pad(x, (pad, pad, pad, pad, pad, pad), mode='replicate')
        B, C, D, H, W = x.shape
        # Unfold all 3 spatial dimensions to get local patches
        # Result shape: (B, C, D, H, W, k, k, k)
        unfolded = x_padded.unfold(2, kernel_size, 1).unfold(3, kernel_size, 1).unfold(4, kernel_size, 1)
        # Reshape to (B, C, D, H, W, k^3) then take median
        unfolded = unfolded.contiguous().view(B, C, D, H, W, -1)
        return unfolded.median(dim=-1).values


    def _preload_rendering_data(self, cfg: 'SDSConfig') -> Optional[dict]:
        """
        Pre-load cameras, object masks, and depth maps from the dataset
        directory for the rendering loss.  Returns None if data is missing.

        Valid cameras = cameras where the object (cfg.rendering_obj_id)
        appears in the instance mask with at least cfg.rendering_min_obj_pixels
        pixels AND a matching depth file exists.
        """
        if not cfg.rendering_source_path or not os.path.isdir(cfg.rendering_source_path):
            print(f"[RenderingLoss] source_path not found: '{cfg.rendering_source_path}'")
            return None
        if cfg.rendering_obj_id < 0:
            print("[RenderingLoss] rendering_obj_id not set – skipping rendering loss.")
            return None
        
        print(f"[RenderingLoss] Pre-loading rendering data from '{cfg.rendering_source_path}' for object id {cfg.rendering_obj_id}...")

        source_path = cfg.rendering_source_path # XXX
        mask_dir = cfg.rendering_mask_dir or os.path.join(source_path, 'instance_masks')
        depth_dir = cfg.rendering_depth_dir or os.path.join(
            source_path, 'mono_priors', 'depthanythingv2')
        bg_id = cfg.rendering_bg_id

        # ---------- build segm_3d id↔class mapping ----------
        # Mapping rule used by segm_3d:
        #   BG_ID (e.g. 255) -> class 0
        #   sorted non-BG pixel ids -> class 1, 2, 3, ...
        
        import glob as _glob
        _mask_files = sorted(_glob.glob(os.path.join(mask_dir, '*')))
        _unique_pixel_ids = set()
        for _mf in _mask_files:
            import cv2 as _cv2_scan
            _m = _cv2_scan.imread(_mf, _cv2_scan.IMREAD_UNCHANGED)
            if _m is None:
                continue
            if _m.ndim > 2:
                _m = _m[:, :, 0]
            _unique_pixel_ids.update(np.unique(_m).tolist())
        _object_pixel_ids = sorted([uid for uid in _unique_pixel_ids if uid != bg_id])
        _id_to_class = {bg_id: 0}
        for _i, _oid in enumerate(_object_pixel_ids):
            _id_to_class[_oid] = _i + 1
        _class_to_id = {v: k for k, v in _id_to_class.items()}
        print(f"[RenderingLoss] segm_3d mapping: {_id_to_class}")

        _valid_obj_ids = sorted([_id for _id in _id_to_class.keys() if _id != bg_id])
        _valid_obj_classes = sorted([_cls for _cls in _class_to_id.keys() if _cls != 0])
        _requested_obj_id = int(cfg.rendering_obj_id)

        if _requested_obj_id in _id_to_class:
            obj_id = _requested_obj_id
            obj_class = _id_to_class[obj_id]
            print(f"[RenderingLoss] rendering_obj_id={_requested_obj_id} "
                  f"(original mask pixel id) -> class index={obj_class}")
        else:
            print(f"[RenderingLoss] rendering_obj_id={_requested_obj_id} not found in mask mapping "
                  f"(valid original mask pixel ids: {_valid_obj_ids}; "
                  f"valid class indices: {_valid_obj_classes}). Skipping.")
            return None

        if obj_id == bg_id:
            print(f"[RenderingLoss] rendering_obj_id resolves to background id={bg_id}; "
                  f"please provide an object id. Skipping.")
            return None

        # ---------- load cameras via GeoSVR DataPack ----------
        try:
            import cv2
            _PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            _GEOSVR_ROOT = os.path.join(_PROJ_ROOT, 'decomvoxel', 'representation', 'GeoSVR')
            for _p in (_PROJ_ROOT, _GEOSVR_ROOT, os.path.join(_GEOSVR_ROOT, 'src')):
                if _p not in sys.path:
                    sys.path.insert(0, _p)
            from yacs.config import CfgNode as _CfgNode
            from src.dataloader.data_pack import DataPack as _DataPack
            cfg_data = _CfgNode()
            cfg_data.source_path = source_path
            cfg_data.images = "images"
            cfg_data.res_downscale = 0.
            cfg_data.res_width = 0
            cfg_data.extension = ".png"
            cfg_data.blend_mask = True
            cfg_data.depth_paths = ""
            cfg_data.depth_scale = 1.0
            cfg_data.data_device = "cpu"
            cfg_data.eval = False
            cfg_data.test_every = 8
            cfg_data.n_sparse = -1
            cfg_data.ncc_scale = 1.0
            data_pack = _DataPack(cfg_data)
            cameras = data_pack.get_train_cameras()
            print(f"[RenderingLoss] Loaded {len(cameras)} cameras from {source_path}")
        except Exception as e:
            print(f"[RenderingLoss] Failed to load cameras: {e}")
            return None

        # ---------- find valid cameras ----------
        valid_cameras, valid_depths, valid_obj_masks = [], [], []

        # DEBUG counters
        _dbg_no_mask = 0
        _dbg_few_pixels = 0
        _dbg_no_depth = 0
        _dbg_printed = 0  # print details for first 3 cameras

        for cam in cameras:
            # Build base name for mask/depth lookup
            img_name = cam.image_name
            basename = os.path.splitext(img_name)[0]
            for suffix in ('_rgb', '_color', '_image'):
                if basename.endswith(suffix):
                    basename = basename[:-len(suffix)]
                    break

            # -- mask --
            mask_path = None
            for ext in ('.png', '.jpg', '.jpeg'):
                p = os.path.join(mask_dir, basename + ext)
                if os.path.exists(p):
                    mask_path = p
                    break
            if mask_path is None:
                _dbg_no_mask += 1
                if _dbg_printed < 3:
                    print(f"[RenderingLoss DEBUG] cam '{img_name}': no mask found, "
                          f"tried {mask_dir}/{basename}{{.png,.jpg,.jpeg}}")
                    _dbg_printed += 1
                continue
            mask_cv = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            if mask_cv is None:
                _dbg_no_mask += 1
                continue
            if mask_cv.ndim > 2:
                mask_cv = mask_cv[:, :, 0]
            unique_ids = np.unique(mask_cv)
            obj_mask_np = (mask_cv == obj_id)
            n_pixels = int(obj_mask_np.sum())
            if _dbg_printed < 3:
                print(f"[RenderingLoss DEBUG] cam '{img_name}': mask={mask_path}, "
                      f"unique_ids={unique_ids.tolist()}, obj_id={obj_id}, "
                      f"obj_pixels={n_pixels}, min_required={cfg.rendering_min_obj_pixels}")
                _dbg_printed += 1
            if n_pixels < cfg.rendering_min_obj_pixels:
                _dbg_few_pixels += 1
                continue

            # -- depth --
            depth_path = None
            for candidate_name in (basename, os.path.splitext(os.path.basename(img_name))[0]):
                for ext in ('.png', '.jpg', '.npy'):
                    p = os.path.join(depth_dir, candidate_name + ext)
                    if os.path.exists(p):
                        depth_path = p
                        break
                if depth_path:
                    break
            if depth_path is None:
                _dbg_no_depth += 1
                continue

            if depth_path.endswith('.npy'):
                # ndarray (shape=(65536,), dtype=float32) min: 4.912, max: 621.8, mean: 172.1
                depth_arr = np.load(depth_path).astype(np.float32)
                if depth_arr.ndim != 2:
                    # 1-D codebook without its index PNG is unusable
                    _dbg_no_depth += 1
                    continue
            else:
                depth_img = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
                if depth_img is None:
                    _dbg_no_depth += 1
                    continue
                if depth_img.ndim > 2:
                    depth_img = depth_img[:, :, 0]
                # GeoSVR quantized format: PNG is a uint16 index image; .npy is the codebook
                # (saved by mono_utils.save_quantize_depth).  Decode: depth = codebook[index].
                # Real-world depth, 0~65535
                _codebook_path = os.path.splitext(depth_path)[0] + '.npy'
                if os.path.exists(_codebook_path):
                    _codebook = np.load(_codebook_path).astype(np.float32)  # (65536,)
                    depth_arr = _codebook[depth_img.astype(np.int32)]       # (H, W) actual depth
                else:
                    depth_arr = depth_img.astype(np.float32)

            # resize to camera resolution if needed
            H, W = cam.image_height, cam.image_width
            if depth_arr.shape[0] != H or depth_arr.shape[1] != W:
                depth_arr = cv2.resize(depth_arr, (W, H), interpolation=cv2.INTER_LINEAR)
                obj_mask_np = cv2.resize(
                    obj_mask_np.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST
                ).astype(bool)

            valid_cameras.append(cam)
            valid_depths.append(torch.from_numpy(depth_arr).float())
            valid_obj_masks.append(torch.from_numpy(obj_mask_np).bool())
            if DEBUG:
                print(f"[RenderingLoss] Valid camera: {cam.image_name}, depth: {depth_path}, mask: {mask_path}, obj pixels: {obj_mask_np.sum()}")

        print(f"[RenderingLoss DEBUG] Filter summary: "
              f"no_mask={_dbg_no_mask}, too_few_pixels={_dbg_few_pixels}, no_depth={_dbg_no_depth}, "
              f"valid={len(valid_cameras)}")

        if not valid_cameras:
            print(f"[RenderingLoss] No valid cameras found for object pixel_id={obj_id} (class={obj_class}) "
                  f"in {mask_dir} / {depth_dir}")
            return None

        print(f"[RenderingLoss] {len(valid_cameras)} valid views for object pixel_id={obj_id} (class={obj_class})")
        return {
            'cameras': valid_cameras,
            'depths': valid_depths,      # list of (H, W) float32 tensors
            'obj_masks': valid_obj_masks, # list of (H, W) bool tensors
        }

    def _soft_depth_render(
        self,
        occ_probs: torch.Tensor,   # (1,1,R,R,R) – occupancy
        transform_info,             # TransformInfo
        camera,                     # GeoSVR Camera
        rot_angle_deg: float = 0.0,  # kept for API compat; rotation is read from transform_info
        output_dir: str = None,      # if set, save depth viz to <output_dir>/depths/
        iteration: int = None,       # used for depth image filename
        strict_occ_threshold: bool = True,
        occ_threshold: float = 0.5,
        soft_gate_alpha: float = 10.0,
        soft_empty_geo: float = -10.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Render depth using the GeoSVR/svraster voxel rasterizer.

        Follows the same pattern as run_inverse_convert:
          1. sparse_structure_to_svr_voxels → SparseVoxelModel
             (Z-rotation un-doing is handled automatically via transform_info)
          2. voxel_model.render(camera, output_depth=True, output_T=True)
          3. Return depth[0] and alpha = 1 - T[0]

                     Two rendering modes:
                          1) strict_occ_threshold=True:
                              depth only uses voxels with occ_probs > occ_threshold.
                              This matches binary occupancy visualisation semantics.
                          2) strict_occ_threshold=False:
                              soft dense rendering where all voxels can contribute.

        Depth images are saved to <output_dir>/depths/ every call.

        Returns:
            depth_map  (H, W): camera-space depth from the rasterizer
            weight_map (H, W): alpha = 1 − transmittance (occupied → 1)
        """
        import imageio
        from decomvoxel.utils.converting import sparse_structure_to_svr_voxels

        _GEOSVR_SRC = os.path.join(PROJECT_ROOT, 'decomvoxel', 'representation', 'GeoSVR', 'src')
        if _GEOSVR_SRC not in sys.path:
            sys.path.insert(0, _GEOSVR_SRC)
        from src.utils.image_utils import viz_tensordepth

        device = occ_probs.device
        H_img = camera.image_height
        W_img = camera.image_width

        # occ_probs: (1,1,R,R,R)
        _grid = occ_probs.squeeze(0).squeeze(0)  # (R, R, R)

        eps = 1e-4
        if strict_occ_threshold:
            # Hard occupancy policy: only p > threshold voxels are rendered.
            # This makes depth behaviour consistent with binary voxel visualisation.
            _occupied = _grid > occ_threshold
            if _occupied.sum() == 0:
                out = {
                    'voxel_model': None,
                    'num_voxels': 0,
                }
            else:
                # Pass dense geo and let converting.py gather values by selected
                # occupied coordinates to avoid any index-order mismatch.
                _per_voxel_geo = torch.logit(_grid.clamp(eps, 1.0 - eps))
                out = sparse_structure_to_svr_voxels(
                    occ_probs,
                    transform_info,
                    threshold=occ_threshold,
                    device=device,
                    per_voxel_geo=_per_voxel_geo,
                    use_all_voxels=False,
                )
        else:
            # Soft occupancy gate around occ_threshold.
            soft_mask = torch.sigmoid(soft_gate_alpha * (_grid - occ_threshold))

            # Stable logit encoding while preserving gradients in the interior.
            p_safe = _grid.clamp(eps, 1.0 - eps)
            g_occ = torch.logit(p_safe)

            # Anchor empty space, then smoothly interpolate toward occupancy-driven geo.
            weighted_geo = soft_empty_geo + soft_mask * (g_occ - soft_empty_geo)

            out = sparse_structure_to_svr_voxels(
                occ_probs,
                transform_info,
                threshold=None,
                device=device,
                per_voxel_geo=weighted_geo,
                use_all_voxels=True,
            )

        voxel_model = out['voxel_model']
        if voxel_model is None or out['num_voxels'] == 0:
            depth_map = torch.zeros(H_img, W_img, device=device)
            weight_map = torch.zeros(H_img, W_img, device=device)
            # Still save a blank image so the depths/ folder stays consistent
            if output_dir is not None:
                depths_dir = os.path.join(output_dir, 'depths')
                os.makedirs(depths_dir, exist_ok=True)
                iter_str = f"{iteration:05d}" if iteration is not None else "final"
                cam_name = getattr(camera, 'image_name', 'cam')
                imageio.imwrite(
                    os.path.join(depths_dir, f"iter{iter_str}_{cam_name}.depth_viz.jpg"),
                    np.zeros((H_img, W_img, 3), dtype=np.uint8)
                )
            return depth_map, weight_map

        render_pkg = voxel_model.render(
            camera, output_depth=True, output_normal=False, output_T=True
        )

        depth_map = render_pkg['depth'][0]       # (H, W)
        weight_map = 1.0 - render_pkg['T'][0]    # alpha = 1 - transmittance, (H, W)

        # Save depth visualisation for inspection
        if output_dir is not None and iteration is not None and iteration % 20 == 0:
            _depth_detached = depth_map.detach()
            print(f"[RenderingLoss] Depth map statistics: max: {_depth_detached.max().item():.2f}, min: {_depth_detached.min().item():.2f}, mean: {_depth_detached.mean().item():.2f}")
            # [RenderingLoss] Depth map statistics: max: 3.08, min: 0.00, mean: 0.41
            # print(f"[RenderingLoss] Saving depth viz for camera '{camera.image_name}' at iteration {iteration} to {output_dir}/depths/")
            depths_dir = os.path.join(output_dir, 'depths')
            os.makedirs(depths_dir, exist_ok=True)
            iter_str = f"{iteration:05d}" if iteration is not None else "final"
            cam_name = getattr(camera, 'image_name', 'cam')
            _viz_path = os.path.join(depths_dir, f"iter{iter_str}_{cam_name}.depth_viz.jpg")
            if (_depth_detached > 0).any():
                imageio.imwrite(_viz_path, viz_tensordepth(_depth_detached, weight_map.detach()))
            else:
                imageio.imwrite(_viz_path, np.zeros((H_img, W_img, 3), dtype=np.uint8))

        return depth_map, weight_map

    def initialize_models(self):
        if self.is_initialized:
            return
        
        import trellis.models as models
        from trellis.pipelines import TrellisImageTo3DPipeline
        
        print("[SparseStructureCompleter] Loading TRELLIS models...")
        
        # Load encoder
        self.encoder = models.from_pretrained(
            "JeffreyXiang/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16"
        ).to(self.device)
        self.encoder.eval()
        self.encoder.requires_grad_(False)  # Freeze encoder: save memory & avoid fp16 grad underflow
        
        # Load pipeline (includes decoder and diffusion)
        self.pipe = TrellisImageTo3DPipeline.from_pretrained(
            "JeffreyXiang/TRELLIS-image-large"
        )
        self.pipe.cuda()
        
        self.decoder = self.pipe.models['sparse_structure_decoder']
        self.decoder.eval()
        self.decoder.requires_grad_(False)  # Decoder only used for visualization, no grad needed
        self.diffusion = self.pipe.models['sparse_structure_flow_model']
        self.diffusion.eval()
        self.diffusion.requires_grad_(False)  # Diffusion model is the "teacher", no grad needed
        
        self.is_initialized = True
        print("[SparseStructureCompleter] Models initialized")
    
    
    def prepare_condition(self, image: str | Image.Image) -> Dict:
        """
        Prepare conditioning from an input image.
        """
        if not self.is_initialized:
            self.initialize_models()
        
        if isinstance(image, str):
            image = Image.open(image)
        image = self.pipe.preprocess_image(image)
        cond = self.pipe.get_cond([image])
        return cond
    
    
    def plot_training_history(self, history: Dict, output_path: str, cfg: SDSConfig = None, json_path: Optional[str] = None):
        if json_path is not None:
            with open(json_path, 'r') as f:
                history=json.load(f)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('SDS Training History', fontsize=16, fontweight='bold')
        
        iterations = list(range(1, len(history['loss_total']) + 1))
        
        # Plot 1: Total Loss
        ax1 = axes[0, 0]
        ax1.plot(iterations, history['loss_total'], linewidth=2, color='#2E86AB', label='Total Loss')
        ax1.set_xlabel('Iteration', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Total Loss', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)
        
        # Plot 2: Timestep Visualization
        ax2 = axes[0, 1]
        if 't_history' in history:
            ax2.plot(iterations, history['t_history'], linewidth=2, color='#A23B72', label='t (timestep)')
            ax2.set_xlabel('Iteration', fontsize=12)
            ax2.set_ylabel('t', fontsize=12)
            ax2.set_title('Sampled t per Iteration', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.legend(fontsize=10)
        else:
            ax2.text(0.5, 0.5, 'No t_history found', fontsize=14, ha='center', va='center')
        
        # Plot 3: Occupancy (Voxel Count)
        ax3 = axes[1, 0]
        ax3.plot(iterations, history['occupancy'], linewidth=2, color='#C73E1D', label='Occupancy')
        ax3.axhline(y=history['occupancy'][0], color='gray', linestyle='--', alpha=0.5, label='Initial')
        ax3.set_xlabel('Iteration', fontsize=12)
        ax3.set_ylabel('Occupied Voxels', fontsize=12)
        ax3.set_title('Voxel Occupancy', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=10)
        
        # Plot 4: Gradient Norm (diagnose dead gradients)
        ax4 = axes[1, 1]
        if 'grad_norm' in history and len(history['grad_norm']) > 0:
            ax4.plot(iterations[:len(history['grad_norm'])], history['grad_norm'][:len(iterations)], 
                     linewidth=2, color='#6A994E', label='Grad Norm')
            ax4.set_xlabel('Iteration', fontsize=12)
            ax4.set_ylabel('||∇θ||', fontsize=12)
            ax4.set_title('Gradient Norm (ss_param)', fontsize=14, fontweight='bold')
            ax4.set_yscale('log')
            ax4.grid(True, alpha=0.3)
            ax4.legend(fontsize=10)
        else:
            ax4.text(0.5, 0.5, 'No grad_norm data', fontsize=14, ha='center', va='center')
        
        # Add configuration info as text
        if cfg:
            info_text = (
                f"Configuration:\n"
                f"Loss Type: {cfg.sds_loss_type}\n"
                f"SDS Weight: {cfg.sds_weight}\n"
                f"Mask Weight: {cfg.mask_weight}\n"
                f"CFG Strength: {cfg.cfg_strength}\n"
                f"LR: {cfg.lr}"
            )
            fig.text(0.02, 0.02, info_text, fontsize=9, family='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.96])
        
        # Save figure
        plot_path = output_path
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # print(f"[SparseStructureCompleter] Training history plot saved to: {plot_path}")

    # History Code
    # def sample_timestep(self, iteration: int, cfg: SDSConfig) -> torch.Tensor:
    #     # Sample random t from [0, 1]
    #     t = torch.rand(1, device=self.device)
        
    #     # Apply warmup: gradually expand sampling range from [warmup_t_min, noise_end] to [noise_start, noise_end]
    #     if iteration <= cfg.warmup_iters:
    #         warmup_progress = iteration / cfg.warmup_iters  # 0 → 1
    #         # Start with t_min=warmup_t_min (e.g., 0.5), gradually decrease to noise_start (e.g., 0.02)
    #         # e.g. wp=0, t_min=0.5; wp=0.5, t_min=0.25; wp=1, t_min=0.02
    #         t_min = cfg.warmup_t_min * (1 - warmup_progress) + cfg.noise_start * warmup_progress
    #         # Map t from [0,1] to [t_min, noise_end]
    #         t = t * (cfg.noise_end - t_min) + t_min
    #     else:
    #         # After warmup, sample from full range [noise_start, noise_end]
    #         t = t * (cfg.noise_end - cfg.noise_start) + cfg.noise_start
        
    #     # Apply TRELLIS rescaling if enabled
    #     # Formula: t' = r*t / (1 + (r-1)*t) where r = rescale_t
    #     # This keeps t' ∈ [0,1] but shifts distribution toward higher values
    #     if cfg.use_rescaled_t:
    #         r = cfg.rescale_t
    #         t = r * t / (1 + (r - 1) * t)
        
    #     return t
    
    def sample_timestep(self, iteration: int, cfg: SDSConfig) -> torch.Tensor:
        """
        Sample a diffusion timestep t ∈ [noise_start, noise_end] according to
        cfg.timestep_schedule:
        """
        progress = iteration / max(cfg.total_iters - 1, 1)  # 0 → 1
        t_low  = cfg.noise_start
        t_high = cfg.noise_end

        schedule = getattr(cfg, 'timestep_schedule', 'uniform')

        if schedule == 'linear':
            anneal_iter = getattr(cfg, 'linear_anneal_iter', cfg.total_iters)
            p = min(iteration / max(anneal_iter, 1), 1.0)
            # lower bound falls linearly from noise_end * 0.8 → noise_start
            t_low  = cfg.noise_end * (1.0 - p) * 0.8 + cfg.noise_start * p
            t_high = cfg.noise_end

        elif schedule == 'bell':
            # bell(p) = 4p(1−p) ∈ [0, 1], peaks at p = 0.5
            bell   = 4.0 * progress * (1.0 - progress)
            t_high = cfg.noise_start + bell * (cfg.noise_end - cfg.noise_start)
            t_high = max(t_high, cfg.noise_start + 1e-4)
            # t_low follows the same bell curve but capped at 80% of t_high
            t_low  = cfg.noise_start + 0.8 * bell * (cfg.noise_end - cfg.noise_start)
            t_low  = min(t_low, t_high - 1e-4)

        elif schedule == 'progressive':
            # t_high grows linearly from noise_start to noise_end
            t_high = cfg.noise_start + progress * (cfg.noise_end - cfg.noise_start)
            t_high = max(t_high, cfg.noise_start + 1e-4)
            # t_low also grows linearly, reaching 80% of t_high at the end
            t_low  = cfg.noise_start + 0.8 * progress * (cfg.noise_end - cfg.noise_start)
            t_low  = min(t_low, t_high - 1e-4)

        # Sample uniformly within the current [t_low, t_high] window
        t = torch.rand(1, device=self.device) * (t_high - t_low) + t_low

        # Optionally apply TRELLIS rescaling: t' = r·t / (1 + (r−1)·t)
        if cfg.use_rescaled_t:
            r = cfg.rescale_t
            t = r * t / (1 + (r - 1) * t)

        return t
    
    
    @staticmethod
    def initialize_ss(initial_ss, fill=False, empty_init_value=0.2):
        # Initialize in logit space:
        #   - Occupied regions (value=1): logit(0.99) ≈ 4.6 → sigmoid ≈ 0.99
        #   - Empty regions (value=0): N(0, init_noise_scale) → sigmoid ≈ 0.5 ± spread
        # This gives empty regions a neutral starting point so the optimizer can
        # push them toward either 0 or 1, instead of being stuck at logit(0.01)≈-4.6.
        occupied_mask = (initial_ss > 0.5).float()  # 1 for occupied, 0 for empty
        
        # Occupied voxels: clamp to 0.99 then logit → ≈ 4.6
        # logit(p) = ln(p / (1-p))
        occupied_logits = torch.logit(torch.tensor(0.99)) * torch.ones_like(initial_ss)
        # Empty voxels: initialize near decision boundary so small gradients can flip them
        # logit(0.5) = 0 → sigmoid(0) = 0.5, right at the threshold
        if fill:
            empty_logits = torch.randn_like(initial_ss) 
            # * cfg.init_noise_scale # 1.0
        else:
            # OLD: logit(0.1) = -2.2, too far from boundary for SDS gradients to push across
            # NEW: logit(0.45) ≈ -0.2, close to boundary so small gradients can flip voxels
            # empty_logits = torch.logit(torch.tensor(0.2)) * torch.ones_like(initial_ss)
            empty_logits = torch.logit(torch.tensor(empty_init_value)) * torch.ones_like(initial_ss)
        
        # Combine: occupied regions keep strong positive logits, empty regions get noise
        init_logits = occupied_mask * occupied_logits + (1 - occupied_mask) * empty_logits
        return init_logits
    
    
    def generate_pure_output(self, condition, latent=None, num_samples=1):
        """
        Debug function: Generate directly from noise using TRELLIS pipeline.
        """
        if not self.is_initialized:
            self.initialize_models()
            
        print("[Debug] Generating pure output from noise (Standard Inference)...")
        z_noise = torch.randn(num_samples, 8, 16, 16, 16, device=self.device)
        
        # Run standard sampling
        if latent is None:
            samples = self.pipe.sparse_structure_sampler.sample(
                self.diffusion,
                z_noise,
                **condition,
                cfg_strength=7.5,
                steps=50
            ).samples
        
        else:
            samples = self.pipe.sparse_structure_sampler.sample(
                self.diffusion,
                latent,
                **condition,
                cfg_strength=7.5,
                steps=50
            ).samples
        
        output_ss = self.decoder(samples)
        return torch.sigmoid(output_ss)
    
    
    def complete(self,
                 sparse_structure: torch.Tensor,
                 condition: Dict,
                 cfg: SDSConfig = None,
                 output_dir: str = None,
                 known_mask: Optional[torch.Tensor] = None,
                 certainty_grid: Optional[torch.Tensor] = None,
                 transform_info=None,
                 blank_space_uncertainty: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        if not self.is_initialized:
            self.initialize_models()
        
        cfg = cfg or SDSConfig()
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        print(f"[SparseStructureCompleter] Starting SDS optimization")
        print(f"  - Iterations: {cfg.total_iters}")
        
        # Make sparse structure a learnable parameter
        initial_ss = sparse_structure.clone().to(self.device)
        init_logits = self.initialize_ss(initial_ss, fill=False, empty_init_value=cfg.empty_init_value)
        ss_param = nn.Parameter(init_logits.to(self.device))
        
        print(f"  - Initial known occupancy: {(initial_ss > 0.5).sum().item():.0f} voxels")
        print(f"  - Empty voxels initialized with logit({cfg.empty_init_value}) ≈ {torch.logit(torch.tensor(cfg.empty_init_value)):.2f}")
        print(f"  - ss_param range: [{ss_param.min().item():.2f}, {ss_param.max().item():.2f}]")
        
        if known_mask is not None:
            known_mask = known_mask.to(self.device)
        
        if certainty_grid is not None:
            certainty_grid = certainty_grid.to(self.device)
        
        if blank_space_uncertainty is not None:
            blank_space_uncertainty = blank_space_uncertainty.to(self.device)
        
        # Setup optimizer
        optimizer = torch.optim.Adam([ss_param], lr=cfg.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.total_iters, eta_min=cfg.lr * 0.01
        )
        
        # Training history
        history = {
            'loss_total': [], 
            'loss_sds': [], 
            'loss_mask': [],
            'loss_tv': [],
            'loss_entropy': [],
            'loss_dist_decay': [],
            'loss_boundary': [],
            'loss_mask_sampling': [],
            'loss_rendering': [],
            'loss_blank_region': [],
            'occupancy': [],
            'sds_schedule_w': [],
            't_history': [],
            'grad_norm': [],
            'param_min': [],
            'param_max': [],
        }
        
        # Get sampler reference
        sampler = self.pipe.sparse_structure_sampler # FlowEulerGuidanceIntervalSampler

        # -----------------------------------------------------------------
        # Pre-load rendering loss data (cameras, masks, depth maps)
        # -----------------------------------------------------------------
        rendering_data = None
        if cfg.use_rendering_loss:
            if transform_info is None:
                print("[RenderingLoss] WARNING: transform_info not provided – "
                      "rendering loss will be skipped.")
            else:
                rendering_data = self._preload_rendering_data(cfg)
                # The rotation that was applied to the SS grid (used to undo it during rendering)
                _rendering_rot_angle = transform_info.effective_rotation_z if hasattr(transform_info, 'effective_rotation_z') else getattr(cfg, 'rotation_angle_z', 0.0)


        
        # -----------------------------------------------------------------
        # Precompute static prior weight maps (shape: (1,1,D,H,W), float32)
        # -----------------------------------------------------------------
        
        # Distance-decay prior map: high weight far from initial occupied voxels
        # Used to penalize newly created occupancy that is far from the seed.
        dist_decay_map = None  # Dynamically updated each iteration from x_0_pred_accum
        if cfg.use_dist_decay_loss:
            # with torch.no_grad():
            #     occupied_bin = (initial_ss > 0.5).float()  # (1,1,D,H,W)
            #     # 3D distance transform in voxel units via Gaussian proxy:
            #     # smooth the binary mask with a large Gaussian → proximal voxels get high values.
            #     # Then distance weight = 1 - smoothed (far = 1, near = 0) clamped to [0,1].
            #     proximity = self._gaussian_smooth_3d(occupied_bin, sigma=cfg.dist_decay_sigma)
            #     proximity = proximity / (proximity.max() + 1e-8)  # normalize to [0,1]
            #     dist_decay_map = (1.0 - proximity).clamp(0, 1)  # far from seed → high penalty
            #     print(f"  - dist_decay_map range: [{dist_decay_map.min():.3f}, {dist_decay_map.max():.3f}]")
            print(f"  - dist_decay_map: will be dynamically computed from x_0_pred_accum (sigma={cfg.dist_decay_sigma}, update every {cfg.print_interval} iters)")
        
        # Boundary penalty map: high weight near grid edges
        boundary_map = None
        if cfg.use_boundary_loss:
            with torch.no_grad():
                D, H, W = initial_ss.shape[2], initial_ss.shape[3], initial_ss.shape[4]
                # Build a 3D coordinate grid in [0, dim-1]
                dz = torch.arange(D, dtype=torch.float32, device=self.device)
                dy = torch.arange(H, dtype=torch.float32, device=self.device)
                dx = torch.arange(W, dtype=torch.float32, device=self.device)
                gz, gy, gx = torch.meshgrid(dz, dy, dx, indexing='ij')  # each (D,H,W)
                # Distance of each voxel to the nearest boundary face
                dist_to_boundary = torch.stack([
                    gz, (D - 1) - gz,
                    gy, (H - 1) - gy,
                    gx, (W - 1) - gx,
                ], dim=0).min(dim=0).values  # (D,H,W)
                # Soft penalty: 1 at border, falls off with Gaussian inside boundary_margin
                inside = (dist_to_boundary / (cfg.boundary_margin + 1e-8)).clamp(0, 1)
                penalty = torch.exp(-0.5 * (inside * cfg.boundary_margin / (cfg.boundary_sigma + 1e-8)) ** 2)
                penalty = (1.0 - inside) * (1.0 - penalty) + penalty  # keep outer = 1, ramp inward
                # Simpler formulation: gaussian from boundary, clamp to [0,1]
                boundary_map = torch.exp(
                    -0.5 * (dist_to_boundary.clamp(0, cfg.boundary_margin) / (cfg.boundary_sigma + 1e-8)) ** 2
                ).unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
                boundary_map = boundary_map.clamp(0, 1)
                print(f"  - boundary_map range: [{boundary_map.min():.3f}, {boundary_map.max():.3f}]")
        
        # x0_pred accumulator for density estimation (used by periodic local smoothing)
        # Accumulated in latent space (B, 8, 16, 16, 16) for efficiency; decoded only at smoothing time
        x_0_pred_accum = None  # Will be initialized on first iteration (to match latent shape)
        x_0_pred_count = 0
        
        # Blank Region Preservation Loss: precompute static maps from initial_ss + blank_space_uncertainty.
        blank_region_mask = None
        blank_region_certainty = None
        if cfg.use_blank_region_loss and blank_space_uncertainty is not None:
            with torch.no_grad():
                blank_region_mask = (initial_ss < 0.5).float()  # (1,1,D,H,W)
                # certainty = 1 - uncertainty; blank_space_uncertainty shape=(D,H,W)
                _bsu = blank_space_uncertainty
                if _bsu.dim() == 3:
                    _bsu = _bsu.unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
                blank_region_certainty = (1.0 - _bsu).clamp(0.0, 1.0)  # (1,1,D,H,W)
                print(f"  - blank_region_mask: {blank_region_mask.sum().long().item()} blank voxels")
                print(f"  - blank_region_certainty range: [{blank_region_certainty.min():.3f}, {blank_region_certainty.max():.3f}]")

        # Periodic Pruning: precompute static mask (blank_space_uncertainty==0 AND certainty_grid<=0)
        _prune_mask = None
        if cfg.use_periodic_prune and blank_space_uncertainty is not None:
            with torch.no_grad():
                _bsu_p = blank_space_uncertainty
                if _bsu_p.dim() == 3:
                    _bsu_p = _bsu_p.unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
                # Certainly empty: uncertainty == 0
                _prune_mask = (_bsu_p == 0)                        # (1,1,D,H,W) bool
                # Protect voxels where certainty_grid > 0 (known occupied regions)
                if certainty_grid is not None:
                    _prune_mask = _prune_mask & (certainty_grid <= 0)
                print(f"  - prune_mask: {_prune_mask.sum().long().item()} prunable voxels "
                      f"(blank_space_uncertainty==0 & certainty_grid<=0)")
        elif cfg.use_periodic_prune:
            print("  [Prune] WARNING: use_periodic_prune=True but blank_space_uncertainty not provided — pruning disabled.")

        pbar = tqdm(range(1, cfg.total_iters + 1), desc="SDS Optimization")
        for i in pbar:
            # Forward Pass
            # Apply sigmoid to keep values in [0, 1]
            # Why sigmoid: keep the raw ss value between 0 and 1!!!
            ss_activated = torch.sigmoid(ss_param)
            
            # Encode to Latent Space
            # Z_0 = Encoder(X_0), shape: (B, 8, 16, 16, 16)
            latents = self.encoder(ss_activated, sample_posterior=False)
            
            # Sample timestep with warmup and optional rescaling
            t = self.sample_timestep(i, cfg)
            history['t_history'].append(t.item())
            t_broadcast = t[:, None, None, None, None]
            assert 0 <= t.item() <= 1, f"Sampled t out of range: {t.item():.4f}"
            
            # Add Noise (Flow Matching)
            # x_t = (1 - t) * x_0 + t * ε,  where t ∈ [0, 1]
            noise = torch.randn_like(latents)
            x_t = (1 - t_broadcast) * latents + t_broadcast * noise
            
            # Diffusion Model Prediction
            # Predict denoised state using classifier-free guidance
            with torch.no_grad():
                # sample_once_eps returns (pred_x_0, pred_v) where pred_v is VELOCITY
                # In flow matching: v = ε - x_0, model predicts velocity
                # Reference: flow_euler.py _v_to_xstart_eps, sample_once_eps
                
                # x_0_pred = x_t - t * v_pred
                # eps_pred = x_t + (1 - t) * v_pred
                x_0_pred, eps_pred, v_pred = sampler.sample_once_eps_all(
                    self.diffusion, x_t, t.squeeze(),
                    **condition,
                    cfg_strength=cfg.cfg_strength,
                    cfg_interval=[0.5, 1.0]
                )
                
            # Accumulate x0_pred for density estimation (periodic local smoothing + dist_decay_map)
            # TODO
            # if cfg.use_periodic_local_smooth:
            with torch.no_grad():
                if x_0_pred_accum is None:
                    x_0_pred_accum = x_0_pred.detach().clone()
                else:
                    if cfg.local_smooth_accum_decay > 0:
                        # Exponential moving average: accum = decay * accum + x0_pred
                        x_0_pred_accum = cfg.local_smooth_accum_decay * x_0_pred_accum + x_0_pred.detach()
                    else:
                        # Simple cumulative sum
                        x_0_pred_accum = x_0_pred_accum + x_0_pred.detach()
                x_0_pred_count += 1
            
            # Dynamically update dist_decay_map from accumulated x0 predictions.
            # Penalty is high where the diffusion model predicts NO occupancy (far from predicted object).
            if cfg.use_dist_decay_loss and x_0_pred_count > 0 and i % cfg.print_interval == 0:
                with torch.no_grad():
                    # Compute average x0 latent
                    if cfg.local_smooth_accum_decay > 0:
                        ema_effective = (1 - cfg.local_smooth_accum_decay ** x_0_pred_count) / (1 - cfg.local_smooth_accum_decay)
                        avg_x0_latent = x_0_pred_accum / ema_effective
                    else:
                        avg_x0_latent = x_0_pred_accum / x_0_pred_count
                    # Decode to voxel space: (1, 1, 64, 64, 64) predicted occupancy density
                    density_ss = torch.sigmoid(self.decoder(avg_x0_latent))
                    # Smooth density → proximity map (near predicted object = high value)
                    proximity = self._gaussian_smooth_3d(density_ss, sigma=cfg.dist_decay_sigma)
                    proximity = proximity / (proximity.max() + 1e-8)  # normalize to [0, 1]
                    # Invert: far from predicted object → high penalty weight
                    dist_decay_map = (1.0 - proximity).clamp(0, 1)
            
            if cfg.weighting_strategy == "snr":
                w_t = 1.0 - t_broadcast
            elif cfg.weighting_strategy == "uniform":
                w_t = 1.0
            else:
                raise ValueError(f"Unknown weighting strategy: {cfg.weighting_strategy}")
            
            # Compute SDS Loss
            if cfg.sds_loss_type == 'classic': # reparam
                # Classic SDS gradient: ∇L = w(t)(ε̂ - ε) ∂z_t/∂θ
                # Noise residual: (ε̂ - ε)
                
                # Compute gradient direction
                # (ε̂ - ε)
                # = x_t + (1-t) * v_pred - noise
                # = x_0 * (1-t) + t * noise + (1-t) * v_pred - noise
                # = (1-t) * (x_0 + v_pred) - (1-t) * noise
                # = (1-t) * (x_0 + v_pred - noise)
                # = (1-t) * (v_pred - v_true)
                # This shows that noise residual and velocity residual are equivalent in essence
                grad_direction = w_t * (eps_pred - noise)
                
                # Build the loss using the reparameterization trick
                # Target: target = latents - grad_direction
                # Loss:   0.5 * ||latents - target||^2
                # Equivalent to: 0.5 * ||grad_direction||^2
                # Differentiating yields gradient direction = grad_direction
                target = (latents - grad_direction).detach()
                
                # Compute loss
                # Use MSE with scaling factor 0.5
                loss_sds = 0.5 * F.mse_loss(latents, target, reduction='mean')
                
                # Flow Matching Step
                # pred_x_prev = x_t - (t - t_prev) * pred_v 
                

            elif cfg.sds_loss_type == 'flow_matching': # ERROR
                # Flow Matching SDS gradient: ∇L = w(t)(v̂ - v) ∂z/∂θ
                # True velocity: v_true = noise - latents (because v = ε - x_0)
                # Predicted velocity: v_pred (from the sampler)
                
                # Compute true velocity
                v_true = noise - latents
                
                # Compute gradient direction
                grad_direction = w_t * (v_pred - v_true)
                
                # 3. Build loss via reparameterization
                target = (latents - grad_direction).detach()
                loss_sds = 0.5 * F.mse_loss(latents, target, reduction='mean')

            elif cfg.sds_loss_type == 'x0_matching': # ERROR
                # Variant using x_0 prediction
                # Gradient direction: w(t)(x_0_pred - latents)
                
                # Compute gradient direction
                grad_direction = w_t * (x_0_pred - latents)
                
                # Build loss via reparameterization
                target = (latents - grad_direction).detach()
                loss_sds = 0.5 * F.mse_loss(latents, target, reduction='mean')
                
            elif cfg.sds_loss_type == 'mse':
                # Simplified MSE loss: MSE(z_0, z_0_pred)
                loss_sds = cfg.sds_weight * F.mse_loss(latents, x_0_pred.detach())
                
            else:
                raise ValueError(f"Unknown SDS loss type: {cfg.sds_loss_type}")

            loss_sds = torch.nan_to_num(loss_sds)
            
            # XXX Not Test Yet. Seems doesn't work.
            # SDS weight scheduler based on x_pred
            # Decode x_0_pred to voxel space and measure agreement with initial_ss in the known/certain region.
            # Agreement = 1 - normalised MSE in that region (∈ [0, 1]).
            # High agreement → x_0_pred is consistent with what we know → trust the SDS gradient → w ≈ 1
            # Low  agreement → x_0_pred contradicts the known structure → distrust SDS gradient → w ≈ 0
            # NOTE: decoder is already frozen (requires_grad_(False)); this block adds zero graph overhead.
            sds_schedule_w = 1.0
            if cfg.use_sds_x_pred_schedule and known_mask is not None:
                with torch.no_grad():
                    # Decode predicted clean latent → probability map in voxel space
                    pred_ss_decoded = torch.sigmoid(self.decoder(x_0_pred))  # (1,1,D,H,W)
                    # Certainty-weighted region mask
                    cert_w = certainty_grid if certainty_grid is not None else torch.ones_like(known_mask)
                    region = known_mask * cert_w  # (1,1,D,H,W); zero outside known
                    n_region = region.sum().clamp(min=1.0)
                    # Weighted MSE: how much x_0_pred deviates from initial_ss in the certain known region
                    mse_region = ((pred_ss_decoded - initial_ss) ** 2 * region).sum() / n_region
                    # Map MSE → similarity score in [0, 1]; clamp mse to [0,1] before inverting
                    sds_schedule_w = (1.0 - mse_region.clamp(0.0, 1.0)).item()
            
            # Known Region Preservation Sampling Loss
            # Gradient path: loss → sigmoid(decoder(latents)) → latents → encoder (frozen ops) → ss_param
            # latents already has requires_grad=True (derived from ss_param via frozen encoder). Decoder parameters have requires_grad=False so no weight updates occur, but the gradient w.r.t. the input `latents` is still computed and propagated to ss_param.
            # A werid Per-voxel weight = certainty × similarity(decoded_x0pred, initial_ss): high where TRELLIS agrees with the known structure → strongly preserve that region, low where TRELLIS is inconsistent → don't force the correction this step
            loss_mask_sampling = torch.tensor(0.0, device=self.device)
            if cfg.use_known_region_sampling_loss and known_mask is not None:
                current_ss_decoded = torch.sigmoid(self.decoder(latents))
                with torch.no_grad():
                    pred_ss_decoded = torch.sigmoid(self.decoder(x_0_pred)) 
                    cert_w = certainty_grid 
                    # Similarity: 1 where x_0_pred ≈ initial_ss, 0 where they diverge
                    # similarity = (1.0 - (pred_ss_decoded - initial_ss).abs()).clamp(0.0, 1.0)  # (1,1,D,H,W)
                    # sampling_weight = cert_w * similarity * known_mask
                    sampling_weight = cert_w * known_mask
                    n_region = sampling_weight.sum().clamp(min=1.0)
                # Push decoded(latents) toward initial_ss, weighted by sampling_weight
                loss_mask_sampling = ((current_ss_decoded - initial_ss) ** 2 * sampling_weight).sum() / n_region
                loss_mask_sampling = torch.nan_to_num(loss_mask_sampling)
            
            # Known Region Preservation Loss
            # Normalize by number of known voxels (not total voxels) to avoid dilution
            # when known region is sparse. initial_ss is binary (0/1), ss_activated in (0,1).
            # certainty_grid (optional): per-cell weight in [0,1]; high-certainty cells
            # contribute proportionally more to the loss when they deviate from initial_ss.
            loss_mask = torch.tensor(0.0, device=self.device)
            if known_mask is not None:
                n_known = known_mask.sum().clamp(min=1.0)
                cert_weight = certainty_grid if certainty_grid is not None else torch.ones_like(known_mask)
                loss_mask = ((ss_activated - initial_ss) ** 2 * known_mask * cert_weight).sum() / n_known
                loss_mask = torch.nan_to_num(loss_mask)
                
            
            # Blank Region Preservation Loss:
            # Penalizes ss_activated rising in voxels that were empty at initialisation.
            # Per-voxel penalty weight = blank certainty = (1 - blank_space_uncertainty).
            loss_blank_region = torch.tensor(0.0, device=self.device)
            blank_schedule_w = 0.0
            if cfg.use_blank_region_loss and blank_region_mask is not None:
                blank_end = cfg.blank_region_end_iter if cfg.blank_region_end_iter > 0 else cfg.total_iters
                if i > blank_end:
                    blank_schedule_w = 0.0
                else:
                    blank_progress = i / blank_end
                    if cfg.blank_region_weight_schedule == "cosine":
                        blank_schedule_w = 0.5 * (1.0 + np.cos(np.pi * blank_progress))
                    elif cfg.blank_region_weight_schedule == "cos_power":
                        cos_base = 0.5 * (1.0 + np.cos(np.pi * blank_progress))
                        blank_schedule_w = cos_base ** cfg.blank_region_poly_power
                    elif cfg.blank_region_weight_schedule == "linear":
                        blank_schedule_w = 1.0 - blank_progress
                    elif cfg.blank_region_weight_schedule == "poly":
                        blank_schedule_w = (1.0 - blank_progress) ** cfg.blank_region_poly_power
                    elif cfg.blank_region_weight_schedule == "exponential":
                        blank_schedule_w = np.exp(-cfg.blank_region_exp_gamma * blank_progress)
                    elif cfg.blank_region_weight_schedule == "logarithmic":
                        blank_schedule_w = 1.0 - np.log(blank_progress * (np.e - 1) + 1)
                    else:  # "const"
                        blank_schedule_w = 1.0

                if blank_schedule_w > 0.0:
                    # Mean activation in blank region, weighted by per-voxel certainty.
                    # Penalises voxels becoming occupied where we are certain they should stay empty.
                    n_blank = blank_region_mask.sum().clamp(min=1.0)
                    loss_blank_region = (ss_activated * blank_region_mask * blank_region_certainty).sum() / n_blank
                    loss_blank_region = torch.nan_to_num(loss_blank_region)

            # TV Regularization: penalize high-frequency noise via 3D total variation
            loss_tv = torch.tensor(0.0, device=self.device)
            if cfg.use_tv_loss and cfg.tv_weight > 0:
                diff_x = (ss_activated[:, :, 1:, :, :] - ss_activated[:, :, :-1, :, :]).abs().mean()
                diff_y = (ss_activated[:, :, :, 1:, :] - ss_activated[:, :, :, :-1, :]).abs().mean()
                diff_z = (ss_activated[:, :, :, :, 1:] - ss_activated[:, :, :, :, :-1]).abs().mean()
                loss_tv = (diff_x + diff_y + diff_z) / 3.0
            
            # Entropy Regularization: gently push toward binary in late stage
            loss_entropy = torch.tensor(0.0, device=self.device)
            progress = i / cfg.total_iters

            # Mask weight schedule: stronger at the start, decays to 0 by end_iter; 0 afterwards
            mask_end_iter = cfg.mask_end_iter if cfg.mask_end_iter > 0 else cfg.total_iters
            if i > mask_end_iter:
                mask_schedule_w = 0.0
            else:
                mask_progress = i / mask_end_iter  # remapped to [0,1] within [0, mask_end_iter]
                if cfg.mask_weight_schedule == "cosine":
                    mask_schedule_w = 0.5 * (1.0 + np.cos(np.pi * mask_progress))  # 1.0 → 0.0
                elif cfg.mask_weight_schedule == "cos_power":
                    power = 3.0
                    cos_base = 0.5 * (1.0 + np.cos(np.pi * mask_progress))
                    mask_schedule_w = cos_base ** power
                elif cfg.mask_weight_schedule == "linear":
                    mask_schedule_w = 1.0 - mask_progress  # 1.0 → 0.0
                elif cfg.mask_weight_schedule == "poly":
                    mask_schedule_w = (1.0 - mask_progress) ** cfg.mask_poly_power
                elif cfg.mask_weight_schedule == "exponential":
                    mask_schedule_w = np.exp(-cfg.mask_exp_gamma * mask_progress)
                elif cfg.mask_weight_schedule == "logarithmic":
                    mask_schedule_w = 1.0 - np.log(mask_progress * (np.e - 1) + 1)
                else:  # "const"
                    mask_schedule_w = 1.0

            # Sampling weight schedule: same options as mask_weight_schedule
            sampling_end_iter = cfg.sampling_end_iter if cfg.sampling_end_iter > 0 else cfg.total_iters
            if i > sampling_end_iter:
                sampling_schedule_w = 0.0
            else:
                sampling_progress = i / sampling_end_iter
                if cfg.sampling_weight_schedule == "cosine":
                    sampling_schedule_w = 0.5 * (1.0 + np.cos(np.pi * sampling_progress))
                elif cfg.sampling_weight_schedule == "linear":
                    sampling_schedule_w = 1.0 - sampling_progress
                elif cfg.sampling_weight_schedule == "poly":
                    sampling_schedule_w = (1.0 - sampling_progress) ** cfg.sampling_poly_power
                elif cfg.sampling_weight_schedule == "exponential":
                    sampling_schedule_w = np.exp(-cfg.sampling_exp_gamma * sampling_progress)
                elif cfg.sampling_weight_schedule == "logarithmic":
                    sampling_schedule_w = 1.0 - np.log(sampling_progress * (np.e - 1) + 1)
                else:  # "const"
                    sampling_schedule_w = 1.0

            entropy_w_effective = 0.0
            if cfg.use_entropy_loss and cfg.entropy_weight > 0 and progress > cfg.entropy_start_ratio:
                # Linearly ramp from 0 to entropy_weight over [entropy_start_ratio, 1.0]
                ramp = (progress - cfg.entropy_start_ratio) / (1.0 - cfg.entropy_start_ratio)
                entropy_w_effective = cfg.entropy_weight * ramp
                loss_entropy = torch.mean(ss_activated * (1 - ss_activated))
            
            # Distance-decay prior loss: penalize occupancy that is far from the known structure.
            # loss = mean( dist_decay_map * ss_activated )  ← far voxels with high activation are penalized
            loss_dist_decay = torch.tensor(0.0, device=self.device)
            if cfg.use_dist_decay_loss and dist_decay_map is not None:
                target_region = ss_activated
                # dist decay on known: also penalize known voxels that are far from seed (of course not)
                if not cfg.dist_decay_on_known and known_mask is not None:
                    target_region = ss_activated * (1 - known_mask)  # skip known voxels
                loss_dist_decay = (dist_decay_map * target_region).mean()
                loss_dist_decay = torch.nan_to_num(loss_dist_decay)
            
            # Boundary penalty loss: penalize occupancy near spatial grid boundaries.
            # loss = mean( boundary_map * ss_activated )  ← edge voxels with high activation are penalized
            loss_boundary = torch.tensor(0.0, device=self.device)
            if cfg.use_boundary_loss and boundary_map is not None:
                loss_boundary = (boundary_map * ss_activated).mean()
                loss_boundary = torch.nan_to_num(loss_boundary)

            # Rendering loss
            loss_rendering = torch.tensor(0.0, device=self.device)
            rendering_schedule_w = 0.0
            if (cfg.use_rendering_loss
                    and rendering_data is not None
                    and i >= cfg.rendering_start_iter
                    and (cfg.rendering_interval <= 0 or i % cfg.rendering_interval == 0)):

                # -- weight schedule --
                _rl_end = cfg.total_iters
                _rl_progress = max(0.0, (i - cfg.rendering_start_iter) / max(_rl_end - cfg.rendering_start_iter, 1))
                if cfg.rendering_loss_schedule == "linear":
                    rendering_schedule_w = 1.0 - _rl_progress
                elif cfg.rendering_loss_schedule == "cosine":
                    rendering_schedule_w = 0.5 * (1.0 + np.cos(np.pi * _rl_progress))
                else:  # "const"
                    rendering_schedule_w = 1.0

                # -- pick a random valid camera --
                _n_cams = len(rendering_data['cameras'])
                _cam_idx = int(torch.randint(0, _n_cams, (1,)).item())
                _cam = rendering_data['cameras'][_cam_idx]
                _gt_depth = rendering_data['depths'][_cam_idx].to(self.device)       # (H, W) float32
                _gt_obj_mask = rendering_data['obj_masks'][_cam_idx].to(self.device) # (H, W) bool

                # -- choose occupancy source for rendering loss (differentiable) --
                if cfg.rendering_ss_source == "pred":
                    # sample_once_eps_all is wrapped by no_grad; use internal
                    # prediction path here to keep gradients to latents alive.
                    _x0_pred_render, _, _ = sampler._get_model_prediction(
                        self.diffusion, x_t, t.squeeze(),
                        **condition,
                        cfg_strength=cfg.cfg_strength,
                        cfg_interval=[0.5, 1.0]
                    )
                    _render_ss = torch.sigmoid(self.decoder(_x0_pred_render))  # (1,1,R,R,R)
                elif cfg.rendering_ss_source == "current":
                    _render_ss = torch.sigmoid(self.decoder(latents))  # (1,1,R,R,R)
                else:
                    raise ValueError(
                        f"Unknown rendering_ss_source: {cfg.rendering_ss_source}. "
                        "Use 'current' or 'pred'."
                    )
                
                # -- soft depth render --
                _depth_map, _weight_map = self._soft_depth_render(
                    _render_ss, transform_info, _cam,
                    rot_angle_deg=_rendering_rot_angle,
                    output_dir=output_dir,
                    iteration=i,
                    strict_occ_threshold=cfg.rendering_strict_occ_threshold,
                    occ_threshold=cfg.rendering_occ_threshold,
                    soft_gate_alpha=cfg.rendering_soft_gate_alpha,
                    soft_empty_geo=cfg.rendering_soft_empty_geo,
                )

                # -- build rendered object mask --
                _render_obj_mask = _weight_map > cfg.rendering_soft_depth_eps
                _combined_mask = _render_obj_mask & _gt_obj_mask  # (H, W) bool

                if _combined_mask.sum() >= 10:
                    # Mean rendered depth (normalise by accumulated weight)
                    _rd = _depth_map[_combined_mask] / (_weight_map[_combined_mask] + 1e-8)
                    _gd = _gt_depth[_combined_mask]

                    # Affine-invariant depth comparison (matching GeoSVR DepthAnythingv2Loss):
                    #   DepthAnythingV2 outputs relative depth where larger = closer (disparity-like).
                    #   Rendered camera-space Z has larger = farther.
                    #   → convert rendered Z to inverse-depth (1/Z), so both have larger = closer.
                    #   → affinely align GT mono to rendered invdepth via MAD-based scale+shift.
                    _rd_inv = 1.0 / _rd.clamp(min=1e-3)  # invdepth: larger = closer
                    with torch.no_grad():
                        _gd_med = _gd.median()
                        _gd_mad = (_gd - _gd_med).abs().mean().clamp(min=1e-8)
                        _ri_med = _rd_inv.detach().median()
                        _ri_mad = (_rd_inv.detach() - _ri_med).abs().mean().clamp(min=1e-8)
                        _gd_aligned = (_gd - _gd_med) * (_ri_mad / _gd_mad) + _ri_med

                    if cfg.rendering_depth_loss_type == "l2":
                        loss_rendering = ((_rd_inv - _gd_aligned.detach()) ** 2).mean()
                    else:  # "l1"
                        loss_rendering = (_rd_inv - _gd_aligned.detach()).abs().mean()

                    loss_rendering = torch.nan_to_num(loss_rendering)

            # Total Loss
            loss = (cfg.sds_weight * sds_schedule_w * loss_sds
                    + cfg.mask_weight * mask_schedule_w * loss_mask
                    + cfg.known_region_sampling_weight * sampling_schedule_w * loss_mask_sampling
                    + cfg.tv_weight * loss_tv
                    + entropy_w_effective * loss_entropy
                    + cfg.dist_decay_weight * loss_dist_decay
                    + cfg.boundary_weight * loss_boundary
                    + cfg.rendering_loss_weight * rendering_schedule_w * loss_rendering
                    + cfg.blank_region_weight * blank_schedule_w * loss_blank_region)

            # Record history
            history['loss_total'].append(loss.item())
            history['loss_sds'].append(loss_sds.item())
            history['loss_mask'].append(loss_mask.item())
            history['loss_tv'].append(loss_tv.item())
            history['loss_entropy'].append(loss_entropy.item())
            history['loss_dist_decay'].append(loss_dist_decay.item())
            history['loss_boundary'].append(loss_boundary.item())
            history['loss_mask_sampling'].append(loss_mask_sampling.item())
            history['loss_rendering'].append(loss_rendering.item())
            history['loss_blank_region'].append(loss_blank_region.item())
            history['occupancy'].append((ss_activated > 0.5).sum().item())
            history['sds_schedule_w'].append(sds_schedule_w)

            # Backward and Optimize
            optimizer.zero_grad()
            loss.backward()

            # Track gradient info
            grad_norm = ss_param.grad.norm().item() if ss_param.grad is not None else 0.0
            history['grad_norm'].append(grad_norm)
            history['param_min'].append(ss_param.min().item())
            history['param_max'].append(ss_param.max().item())
            
            optimizer.step()
            scheduler.step()
            
            # Periodic Smoothing (Global): apply 3D blur/median to ss_param to suppress isolated noise
            if (cfg.use_periodic_smooth and cfg.smooth_interval > 0 and i % cfg.smooth_interval == 0 and progress >= cfg.smooth_start_ratio):
                with torch.no_grad():
                    ss_smooth_input = torch.sigmoid(ss_param)  # [0, 1]
                    if cfg.smooth_method == 'gaussian':
                        ss_smoothed = self._gaussian_smooth_3d(ss_smooth_input, cfg.smooth_sigma)
                    elif cfg.smooth_method == 'median':
                        ss_smoothed = self._median_smooth_3d(ss_smooth_input, cfg.smooth_kernel_size)
                    else:
                        raise ValueError(f"Unknown smooth method: {cfg.smooth_method}")
                    
                    # Blend: new = (1-s)*original + s*smoothed
                    ss_blended = (1 - cfg.smooth_strength) * ss_smooth_input + cfg.smooth_strength * ss_smoothed
                    ss_blended = ss_blended.clamp(1e-6, 1 - 1e-6)  # Avoid logit(0) or logit(1) = ±inf
                    
                    # Preserve known voxels: restore original logits for occupied regions
                    new_logits = torch.logit(ss_blended)
                    if cfg.smooth_preserve_known and known_mask is not None:
                        new_logits = known_mask * ss_param.data + (1 - known_mask) * new_logits
                    
                    ss_param.data.copy_(new_logits)
                    
                    if i % cfg.print_interval == 0:
                        print(f"  [Smooth] iter {i}: {cfg.smooth_method} (sigma={cfg.smooth_sigma}, strength={cfg.smooth_strength})")
            
            # Periodic Local Smoothing (Adaptive): uncertainty-guided smoothing based on accumulated x0_pred density
            # Low density (uncertain) regions get stronger smoothing; high density (confident) regions get weaker smoothing
            if (cfg.use_periodic_local_smooth and cfg.local_smooth_interval > 0
                    and i % cfg.local_smooth_interval == 0
                    and progress >= cfg.local_smooth_start_ratio
                    and progress < cfg.local_smooth_end_ratio
                    and x_0_pred_count > 0):
                with torch.no_grad():
                    # 1. Compute average x0_pred in latent space
                    if cfg.local_smooth_accum_decay > 0:
                        # For EMA, effective count is geometric series sum
                        ema_effective = (1 - cfg.local_smooth_accum_decay ** x_0_pred_count) / (1 - cfg.local_smooth_accum_decay)
                        avg_x0_latent = x_0_pred_accum / ema_effective
                    else:
                        avg_x0_latent = x_0_pred_accum / x_0_pred_count
                    
                    # 2. Decode to voxel space to get density estimate
                    density_ss = torch.sigmoid(self.decoder(avg_x0_latent))  # (1, 1, 64, 64, 64) in [0, 1]
                    
                    # 3. Apply heavy smoothing to density estimate for a stable uncertainty map
                    density_smoothed = self._gaussian_smooth_3d(density_ss, cfg.local_smooth_density_sigma)
                    density_smoothed = density_smoothed.clamp(0, 1)
                    
                    # 4. Compute per-voxel adaptive smooth strength
                    # Low density → high smooth strength (uncertain/empty regions → more denoising)
                    # High density → low smooth strength (confident occupied regions → preserve)
                    adaptive_strength = (cfg.local_smooth_max_strength * (1 - density_smoothed)
                                         + cfg.local_smooth_min_strength * density_smoothed)
                    
                    # 5. Smooth ss_param
                    ss_smooth_input = torch.sigmoid(ss_param)  # [0, 1]
                    if cfg.local_smooth_method == 'gaussian':
                        ss_smoothed = self._gaussian_smooth_3d(ss_smooth_input, cfg.local_smooth_sigma)
                    elif cfg.local_smooth_method == 'median':
                        ss_smoothed = self._median_smooth_3d(ss_smooth_input, cfg.local_smooth_kernel_size)
                    else:
                        raise ValueError(f"Unknown local smooth method: {cfg.local_smooth_method}")
                    
                    # 6. Adaptive blending: new = (1 - strength) * original + strength * smoothed
                    ss_blended = (1 - adaptive_strength) * ss_smooth_input + adaptive_strength * ss_smoothed
                    ss_blended = ss_blended.clamp(1e-6, 1 - 1e-6)
                    
                    # 7. Convert back to logit space
                    new_logits = torch.logit(ss_blended)
                    
                    # 8. Preserve known voxels
                    if cfg.local_smooth_preserve_known and known_mask is not None:
                        new_logits = known_mask * ss_param.data + (1 - known_mask) * new_logits
                    
                    ss_param.data.copy_(new_logits)
                    
                    if i % cfg.print_interval == 0:
                        print(f"  [LocalSmooth] iter {i}: density [{density_smoothed.min():.3f}, {density_smoothed.max():.3f}], "
                              f"strength [{adaptive_strength.min():.3f}, {adaptive_strength.max():.3f}]")
            
            # Periodic Pruning: zero-out or suppress "certainly blank" voxels outside the certainty region
            _prune_end_iter = cfg.prune_end_iter if cfg.prune_end_iter >= 0 else cfg.total_iters
            if (cfg.use_periodic_prune
                    and _prune_mask is not None
                    and cfg.prune_interval > 0
                    and cfg.prune_start_iter <= i <= _prune_end_iter
                    and (i - cfg.prune_start_iter) % cfg.prune_interval == 0):
                with torch.no_grad():
                    target_logit = cfg.prune_empty_logit
                    current_logits = ss_param.data[_prune_mask]
                    if current_logits.numel() == 0:
                        pass  # Nothing to prune
                    elif cfg.prune_mode == 'redistribute':
                        # Redistribute: keep mean roughly intact (shifted slightly toward target)
                        # but replace the collapsed distribution with a uniform spread.
                        # Schedule: effective_mean_factor = mean_factor * w(progress in prune window)
                        _p_end = _prune_end_iter
                        _p_start = cfg.prune_start_iter
                        _prune_progress = ((i - _p_start) / max(_p_end - _p_start, 1)) if _p_end > _p_start else 1.0
                        _prune_progress = float(np.clip(_prune_progress, 0.0, 1.0))
                        _sched = cfg.prune_redistribute_schedule
                        if _sched == 'cosine':
                            _redist_w = 0.5 * (1.0 + np.cos(np.pi * _prune_progress))
                        elif _sched == 'cos_power':
                            _redist_w = (0.5 * (1.0 + np.cos(np.pi * _prune_progress))) ** cfg.prune_redistribute_poly_power
                        elif _sched == 'linear':
                            _redist_w = 1.0 - _prune_progress
                        elif _sched == 'poly':
                            _redist_w = (1.0 - _prune_progress) ** cfg.prune_redistribute_poly_power
                        elif _sched == 'exponential':
                            _redist_w = np.exp(-cfg.prune_redistribute_exp_gamma * _prune_progress)
                        elif _sched == 'logarithmic':
                            _redist_w = 1.0 - np.log(_prune_progress * (np.e - 1) + 1)
                        elif _sched == 'u_shape':
                            # U-shape: strong at start and end, weak in the middle
                            # w(p) = |2p - 1|^power  →  w(0)=1, w(0.5)=0, w(1)=1
                            _redist_w = abs(2.0 * _prune_progress - 1.0) ** cfg.prune_redistribute_poly_power
                            # _redist_w = _redist_w * 0.1
                        else:  # 'const'
                            _redist_w = 1.0
                        effective_mean_factor = cfg.prune_redistribute_mean_factor * _redist_w
                        current_mean = current_logits.mean().item()
                        target_mean = ((1.0 - effective_mean_factor) * current_mean
                                       + effective_mean_factor * target_logit)
                        half_r = cfg.prune_redistribute_uniform_range
                        new_logits = torch.empty_like(current_logits).uniform_(
                            target_mean - half_r, target_mean + half_r
                        )
                        ss_param.data[_prune_mask] = new_logits
                        if i % cfg.print_interval == 0:
                            n_pruned = _prune_mask.sum().item()
                            print(f"  [Prune/Redistrib] iter {i}: {n_pruned} voxels redistributed, "
                                  f"mean_factor={effective_mean_factor:.3f} (sched_w={_redist_w:.3f}), "
                                  f"mean {current_mean:.2f} → {target_mean:.2f}, "
                                  f"range=[{target_mean - half_r:.2f}, {target_mean + half_r:.2f}]")
                    else:
                        # Suppress (original behaviour)
                        if cfg.prune_strength >= 1.0:
                            # Hard prune: snap prunable logits directly to target
                            ss_param.data[_prune_mask] = target_logit
                        else:
                            # Soft prune: lerp current logit toward target
                            ss_param.data[_prune_mask] = (
                                (1.0 - cfg.prune_strength) * current_logits
                                + cfg.prune_strength * target_logit
                            )
                        if i % cfg.print_interval == 0:
                            n_pruned = _prune_mask.sum().item()
                            print(f"  [Prune] iter {i}: {n_pruned} voxels pruned "
                                  f"(strength={cfg.prune_strength:.2f}, target_logit={target_logit:.1f})")
            
            # Blank-region noise fill: re-inject noise into certainly-blank voxels to prevent
            # them from being permanently locked at prune_empty_logit after periodic pruning.
            # Noise std is derived from the current logit distribution within the prunable mask,
            # ensuring it stays proportional to the pruned values' scale.
            _blank_noise_end_iter = cfg.blank_noise_end_iter if cfg.blank_noise_end_iter >= 0 else cfg.total_iters
            if (cfg.use_blank_noise_fill
                    and _prune_mask is not None
                    and cfg.blank_noise_interval > 0
                    and cfg.blank_noise_start_iter <= i <= _blank_noise_end_iter
                    and (i - cfg.blank_noise_start_iter) % cfg.blank_noise_interval == 0):
                with torch.no_grad():
                    current_logits = ss_param.data[_prune_mask]
                    logit_std = current_logits.std().item()
                    noise_std = max(logit_std, cfg.blank_noise_min_scale) * cfg.blank_noise_scale
                    noise = torch.randn_like(current_logits) * noise_std
                    ss_param.data[_prune_mask] = ss_param.data[_prune_mask] + noise
                    if i % cfg.print_interval == 0:
                        n_fill = _prune_mask.sum().item()
                        print(f"  [BlankNoise] iter {i}: {n_fill} voxels filled, "
                              f"noise_std={noise_std:.3f} (logit_std={logit_std:.3f})")

            # Logging
            if i % cfg.print_interval == 0:
                occ = (ss_activated > 0.5).sum().item()
                # Count how many empty voxels are near the decision boundary
                with torch.no_grad():
                    empty_mask = (initial_ss < 0.5)
                    empty_logits = ss_param[empty_mask]
                    near_boundary = ((empty_logits > -1.0) & (empty_logits < 1.0)).sum().item()
                    new_occupied = (empty_logits > 0).sum().item()
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    sds=f"{loss_sds.item():.4f}",
                    sds_w=f"{sds_schedule_w:.3f}",
                    occ=f"{occ:.0f}",
                    new=f"{new_occupied:.0f}",
                    grad=f"{grad_norm:.2e}",
                    t=f"{t.item():.2f}"
                )
            
            # Save Intermediate Results
            if output_dir and cfg.save_interval > 0 and i % cfg.save_interval == 0:
                with torch.no_grad():
                    # Store actual values here for later analysis/visualization
                    # inter_ss = (torch.sigmoid(ss_param) > 0.5).float()
                    inter_ss = (torch.sigmoid(ss_param)).float()
                    torch.save(inter_ss, os.path.join(output_dir, f'ss_iter{i:06d}.pt'))

                    if getattr(cfg, 'vis_mid', False):
                        # Run a fresh forward pass at t=0.4 on the current latents
                        # to get the diffusion model's denoised prediction.
                        _latents_mid = self.encoder(inter_ss, sample_posterior=False)
                        _t_mid = torch.tensor([0.4], device=self.device)
                        _t_mid_bc = _t_mid[:, None, None, None, None]
                        _noise_mid = torch.randn_like(_latents_mid)
                        _x_t_mid = (1 - _t_mid_bc) * _latents_mid + _t_mid_bc * _noise_mid
                        _x_0_pred_04, _, _ = sampler.sample_once_eps_all(
                            self.diffusion, _x_t_mid, _t_mid.squeeze(),
                            **condition,
                            cfg_strength=cfg.cfg_strength,
                            cfg_interval=[0.5, 1.0]
                        )
                        _target_ss_04 = (torch.sigmoid(self.decoder(_x_0_pred_04)) > 0.5).float()
                        _fname_04 = os.path.join(output_dir, f'target_ss_04_iter{i:06d}.pt')
                        torch.save(_target_ss_04, _fname_04)
                        visualize_sparse_structure(
                            data=_fname_04,
                            output_dir=output_dir,
                            mode='3d',
                            threshold=0.5,
                            value_colored=False,
                            show=False
                        )

                    if DEBUG:
                        # Debug: Save what the diffusion model predicts (The "Teacher")
                        target_ss = self.decoder(x_0_pred)
                        target_ss = (torch.sigmoid(target_ss) > 0.5).float()
                        torch.save(target_ss, os.path.join(output_dir, f'target_iter{i:06d}_t{t.item():.2f}.pt'))
                    
                        visualize_sparse_structure(
                            data=os.path.join(output_dir, f'ss_iter{i:06d}.pt'),
                            output_dir=output_dir,
                            mode='3d',
                            threshold=0.5,
                            value_colored=False,
                            show=False
                        )
                        visualize_sparse_structure(
                            data=os.path.join(output_dir, f'target_iter{i:06d}_t{t.item():.2f}.pt'),
                            output_dir=output_dir,
                            mode='3d',
                            threshold=0.5,
                            value_colored=False,
                            show=False
                        )
                    
            # Save training history
            if output_dir and i % cfg.print_interval == 0:
                output_path = os.path.join(output_dir, 'training_history.png')
                self.plot_training_history(history, output_path, cfg, json_path=None)
        
        # Get final trellis target: run one forward pass at t=0.4 to get the diffusion model's prediction
        with torch.no_grad():
            ss_activated_final = torch.sigmoid(ss_param)
            latents_final = self.encoder(ss_activated_final, sample_posterior=False)
            t_final = torch.tensor([0.4], device=self.device)
            t_final_broadcast = t_final[:, None, None, None, None]
            noise_final = torch.randn_like(latents_final)
            x_t_final = (1 - t_final_broadcast) * latents_final + t_final_broadcast * noise_final
            x_0_pred_final, _, _ = sampler.sample_once_eps_all(
                self.diffusion, x_t_final, t_final.squeeze(),
                **condition,
                cfg_strength=cfg.cfg_strength,
                cfg_interval=[0.5, 1.0]
            )
            # Decode to voxel space
            final_target_logits = self.decoder(x_0_pred_final)
            final_target = (torch.sigmoid(final_target_logits) > 0.5).float()
            print(f"[SparseStructureCompleter] Final trellis target occupancy: {final_target.sum().item():.0f} / {64**3}")
        
        # Save and visualize final target
        torch.save(final_target, os.path.join(output_dir, 'final_target.pt'))
        visualize_sparse_structure(
            data=os.path.join(output_dir, 'final_target.pt'),
            output_dir=output_dir,
            mode='3d',
            threshold=0.5,
            value_colored=False,
            show=False
        )
        print(f"[SparseStructureCompleter] Final target saved to: {os.path.join(output_dir, 'final_target.pt')}")

        # Save and visualize ss_activated_final (optimized sparse structure before binarisation)
        torch.save(ss_activated_final, os.path.join(output_dir, 'ss_activated_final.pt'))
        visualize_sparse_structure(
            data=os.path.join(output_dir, 'ss_activated_final.pt'),
            output_dir=output_dir,
            mode='3d',
            threshold=0.5,
            value_colored=False,
            show=False
        )
        print(f"[SparseStructureCompleter] ss_activated_final saved to: {os.path.join(output_dir, 'ss_activated_final.pt')}")

        # Get final result
        with torch.no_grad():
            final_ss = torch.sigmoid(ss_param)
            final_ss_binary = (final_ss > 0.5).float()
        
        print(f"[SparseStructureCompleter] Optimization complete")
        print(f"  - Final occupancy: {final_ss_binary.sum().item():.0f} / {64**3}")
        
        # Save training history
        if output_dir:
            with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
                json.dump(history, f)
            output_path = os.path.join(output_dir, 'training_history.png')
            self.plot_training_history(history, output_path, cfg, json_path=None)
        
        return final_ss, history, final_target
    
    
    def refine(self,
               sparse_structure: torch.Tensor,
               condition: Dict,
               noise_level: float = 0.3,
               num_steps: int = 25) -> torch.Tensor:
        """
        Refine sparse structure using partial diffusion denoising.
        
        Args:
            sparse_structure: (1, 1, 64, 64, 64) sparse structure
            condition: conditioning dict
            noise_level: noise level for refinement (0 to 1)
            num_steps: number of denoising steps
            
        Returns:
            refined sparse structure
        """
        if not self.is_initialized:
            self.initialize_models()
        
        print(f"[SparseStructureCompleter] Refining with {num_steps} diffusion steps")
        
        # Encode to latent
        with torch.no_grad():
            latents = self.encoder(sparse_structure, sample_posterior=False)
            
            # Add noise at specified level
            t = torch.tensor([noise_level], device=self.device)
            t_rescaled = 3 * t / (1 + 2 * t)
            t_broadcast = t_rescaled[:, None, None, None, None]
            
            noise = torch.randn_like(latents)
            x_t = (1 - t_broadcast) * latents + t_broadcast * noise
            
            # Denoise using sampler
            refined_latents = self.pipe.sparse_structure_sampler.sample(
                self.diffusion,
                x_t,
                **condition,
                cfg_strength=3.0,
                cfg_interval=[0.5, 1.0],
                steps=num_steps,
                verbose=False
            ).samples
            
            # Decode back to sparse structure
            refined_ss = self.decoder(refined_latents)
            refined_ss = torch.sigmoid(refined_ss)
        
        return refined_ss

# def log_config()

def run_sds_completion_ss(sparse_structure: torch.Tensor,
                          image_path: str,
                          output_dir: str,
                          cfg: SDSConfig = None,
                          known_mask: Optional[torch.Tensor] = None,
                          certainty_grid: Optional[torch.Tensor] = None,
                          refine: bool = False,
                          transform_info=None,
                          blank_space_uncertainty: Optional[torch.Tensor] = None) -> Dict:
    """
    Run the full SDS completion pipeline on sparse structure.
    Args:
        sparse_structure: (1, 1, 64, 64, 64) initial occupancy
        image_path: path to conditioning image
        output_dir: output directory
        cfg: SDS configuration
        known_mask: (1, 1, 64, 64, 64) binary mask of known regions
        certainty_grid: (1, 1, 64, 64, 64) per-cell certainty weights in [0, 1];
                        higher certainty amplifies the mask preservation loss
        refine: whether to run diffusion refinement
        transform_info: TransformInfo for converting SS grid to world coordinates
                        (required when cfg.use_rendering_loss=True)
    """
    cfg = cfg or SDSConfig()
    os.makedirs(output_dir, exist_ok=True)
    
    completer = SparseStructureCompleter()
    
    if known_mask is None:
        known_mask = (sparse_structure > 0.5).float()
        print(f"[run_sds_completion_ss] Created known mask: {known_mask.sum().item():.0f} voxels")
    
    log_info = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config': {
            'rotate_initial_structure': cfg.rotate_initial_structure,
            'rotation_angle_z': cfg.rotation_angle_z,
            'total_iters': cfg.total_iters,
            'lr': cfg.lr,
            'noise_start': cfg.noise_start,
            'noise_end': cfg.noise_end,
            'weighting_strategy': cfg.weighting_strategy,
            'sds_weight': cfg.sds_weight,
            'mask_weight': cfg.mask_weight,
            'mask_weight_schedule': cfg.mask_weight_schedule,
            'mask_poly_power': cfg.mask_poly_power,
            'mask_exp_gamma': cfg.mask_exp_gamma,
            'mask_end_iter': cfg.mask_end_iter,
            'use_blank_region_loss': cfg.use_blank_region_loss,
            'blank_region_weight': cfg.blank_region_weight,
            'blank_region_weight_schedule': cfg.blank_region_weight_schedule,
            'blank_region_poly_power': cfg.blank_region_poly_power,
            'blank_region_exp_gamma': cfg.blank_region_exp_gamma,
            'blank_region_end_iter': cfg.blank_region_end_iter,
            'use_sds_x_pred_schedule': cfg.use_sds_x_pred_schedule,
            'use_known_region_sampling_loss': cfg.use_known_region_sampling_loss,
            'known_region_sampling_weight': cfg.known_region_sampling_weight,
            'cfg_strength': cfg.cfg_strength,
            'sds_loss_type': cfg.sds_loss_type,
            'refine_enabled': cfg.refine_enabled,
            'refine_noise': cfg.refine_noise,
            'refine_steps': cfg.refine_steps,
            'print_interval': cfg.print_interval,
            'save_interval': cfg.save_interval,
            'timestep_schedule': cfg.timestep_schedule,
            'linear_anneal_iter': cfg.linear_anneal_iter,
            'warmup_iters': cfg.warmup_iters,
            'warmup_t_min': cfg.warmup_t_min,
            'use_rescaled_t': cfg.use_rescaled_t,
            'rescale_t': cfg.rescale_t,
            'init_noise_scale': cfg.init_noise_scale,
            'empty_init_value': cfg.empty_init_value,
            'use_tv_loss': cfg.use_tv_loss,
            'tv_weight': cfg.tv_weight,
            'use_entropy_loss': cfg.use_entropy_loss,
            'entropy_weight': cfg.entropy_weight,
            'entropy_start_ratio': cfg.entropy_start_ratio,
            'use_periodic_smooth': cfg.use_periodic_smooth,
            'smooth_interval': cfg.smooth_interval,
            'smooth_method': cfg.smooth_method,
            'smooth_sigma': cfg.smooth_sigma,
            'smooth_kernel_size': cfg.smooth_kernel_size,
            'smooth_strength': cfg.smooth_strength,
            'smooth_start_ratio': cfg.smooth_start_ratio,
            'smooth_preserve_known': cfg.smooth_preserve_known,
            'use_dist_decay_loss': cfg.use_dist_decay_loss,
            'dist_decay_weight': cfg.dist_decay_weight,
            'dist_decay_sigma': cfg.dist_decay_sigma,
            'dist_decay_on_known': cfg.dist_decay_on_known,
            'use_boundary_loss': cfg.use_boundary_loss,
            'boundary_weight': cfg.boundary_weight,
            'boundary_margin': cfg.boundary_margin,
            'boundary_sigma': cfg.boundary_sigma,
            'use_periodic_local_smooth': cfg.use_periodic_local_smooth,
            'local_smooth_interval': cfg.local_smooth_interval,
            'local_smooth_method': cfg.local_smooth_method,
            'local_smooth_sigma': cfg.local_smooth_sigma,
            'local_smooth_kernel_size': cfg.local_smooth_kernel_size,
            'local_smooth_density_sigma': cfg.local_smooth_density_sigma,
            'local_smooth_max_strength': cfg.local_smooth_max_strength,
            'local_smooth_min_strength': cfg.local_smooth_min_strength,
            'local_smooth_start_ratio': cfg.local_smooth_start_ratio,
            'local_smooth_end_ratio': cfg.local_smooth_end_ratio,
            'local_smooth_preserve_known': cfg.local_smooth_preserve_known,
            'local_smooth_accum_decay': cfg.local_smooth_accum_decay,
            'use_rendering_loss': cfg.use_rendering_loss,
            'rendering_loss_weight': cfg.rendering_loss_weight,
            'rendering_ss_source': cfg.rendering_ss_source,
            'rendering_source_path': cfg.rendering_source_path,
            'rendering_obj_id': cfg.rendering_obj_id,
            'rendering_interval': cfg.rendering_interval,
            'rendering_depth_loss_type': cfg.rendering_depth_loss_type,
            'rendering_strict_occ_threshold': cfg.rendering_strict_occ_threshold,
            'rendering_occ_threshold': cfg.rendering_occ_threshold,
            'rendering_soft_gate_alpha': cfg.rendering_soft_gate_alpha,
            'rendering_soft_empty_geo': cfg.rendering_soft_empty_geo,
            'use_periodic_prune': cfg.use_periodic_prune,
            'prune_interval': cfg.prune_interval,
            'prune_start_iter': cfg.prune_start_iter,
            'prune_end_iter': cfg.prune_end_iter,
            'prune_strength': cfg.prune_strength,
            'prune_empty_logit': cfg.prune_empty_logit,
            'prune_mode': cfg.prune_mode,
            'prune_redistribute_mean_factor': cfg.prune_redistribute_mean_factor,
            'prune_redistribute_uniform_range': cfg.prune_redistribute_uniform_range,
            'prune_redistribute_schedule': cfg.prune_redistribute_schedule,
            'prune_redistribute_poly_power': cfg.prune_redistribute_poly_power,
            'prune_redistribute_exp_gamma': cfg.prune_redistribute_exp_gamma,
            'use_blank_noise_fill': cfg.use_blank_noise_fill,
            'blank_noise_start_iter': cfg.blank_noise_start_iter,
            'blank_noise_end_iter': cfg.blank_noise_end_iter,
            'blank_noise_interval': cfg.blank_noise_interval,
            'blank_noise_scale': cfg.blank_noise_scale,
            'blank_noise_min_scale': cfg.blank_noise_min_scale,
            'vis_mid': cfg.vis_mid
        },
        'input_info': {
            'image_path': image_path,
            'output_dir': output_dir,
            'sparse_structure_shape': list(sparse_structure.shape),
            'initial_occupancy': int((sparse_structure > 0.5).sum().item()),
            'known_mask_occupancy': int(known_mask.sum().item()),
            'unknown_voxels': int((1 - known_mask).sum().item()),
            'total_voxels': int(np.prod(sparse_structure.shape)),
            'refine_enabled': refine,
        },
        'input_info': {
            'image_path': image_path,
            'output_dir': output_dir,
            'sparse_structure_shape': list(sparse_structure.shape),
            'initial_occupancy': int((sparse_structure > 0.5).sum().item()),
            'known_mask_occupancy': int(known_mask.sum().item()),
            'unknown_voxels': int((1 - known_mask).sum().item()),
            'total_voxels': int(np.prod(sparse_structure.shape)),
            'refine_enabled': refine,
        },
        'system_info': {
            'device': str(device),
            'cuda_available': torch.cuda.is_available(),
            'cuda_device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A',
        }
    }
    
    # Save log to file
    log_path = os.path.join(output_dir, 'sds_config_log.json')
    with open(log_path, 'w') as f:
        json.dump(log_info, f, indent=2)
    
    print(f"[run_sds_completion_ss] Configuration logged to: {log_path}")
    print(f"[run_sds_completion_ss] Initial occupancy: {log_info['input_info']['initial_occupancy']} / {log_info['input_info']['total_voxels']}")
    print(f"[run_sds_completion_ss] Known voxels: {log_info['input_info']['known_mask_occupancy']}, Unknown voxels: {log_info['input_info']['unknown_voxels']}")
    
    # Prepare conditioning
    print(f"[run_sds_completion_ss] Preparing conditioning from: {image_path}")
    condition = completer.prepare_condition(image_path)
    
    # Run SDS optimization
    print("[run_sds_completion_ss] Starting SDS optimization...")
    
    # if DEBUG:
    #     with torch.no_grad():
    #         print("[Debug] Generating pure reference from image...")
    #         pure_gen = completer.generate_pure_output(condition)
    #         pure_gen_binary = (pure_gen > 0.5).float()
    #         torch.save(pure_gen_binary, os.path.join(output_dir, 'debug_pure_generation.pt'))
    #         print(f"[Debug] Saved pure generation reference to {os.path.join(output_dir, 'debug_pure_generation.pt')}")
    #         visualize_sparse_structure(
    #             data=os.path.join(output_dir, 'debug_pure_generation.pt'),
    #             output_dir=output_dir,
    #             mode='3d',
    #             threshold=0.2,
    #             value_colored=False,
    #             show=False
    #         )
            
    #         # not fill, with noise
    #         initial_ss = sparse_structure.clone().to(device)
    #         init_logits = SparseStructureCompleter.initialize_ss(initial_ss, fill=False, empty_init_value=cfg.empty_init_value)
    #         latents = completer.encoder(torch.sigmoid(init_logits), sample_posterior=False)
    #         noise = torch.randn_like(latents)
    #         x_t = 0.5 * latents + 0.5 * noise
    #         pure_gen = completer.generate_pure_output(condition, latent=x_t)
    #         pure_gen_binary = (pure_gen > 0.5).float()
    #         torch.save(pure_gen_binary, os.path.join(output_dir, 'debug_unfill_generation.pt'))
    #         print(f"[Debug] Saved pure generation reference to {os.path.join(output_dir, 'debug_unfill_generation.pt')}")
    #         visualize_sparse_structure(
    #             data=os.path.join(output_dir, 'debug_unfill_generation.pt'),
    #             output_dir=output_dir,
    #             mode='3d',
    #             threshold=0.2,
    #             value_colored=False,
    #             show=False
    #         )
            
    completed_ss, history, final_target = completer.complete(
        sparse_structure=sparse_structure,
        condition=condition,
        cfg=cfg,
        output_dir=output_dir,
        known_mask=known_mask,
        certainty_grid=certainty_grid,
        transform_info=transform_info,
        blank_space_uncertainty=blank_space_uncertainty,
    )
    
    refined_ss = None
    if refine and cfg.refine_enabled:
        print("[run_sds_completion_ss] Running diffusion refinement...")
        refined_ss = completer.refine(
            sparse_structure=completed_ss,
            condition=condition,
            noise_level=cfg.refine_noise,
            num_steps=cfg.refine_steps
        )
    
    torch.save(completed_ss, os.path.join(output_dir, 'completed_ss.pt'))
    if refined_ss is not None:
        torch.save(refined_ss, os.path.join(output_dir, 'refined_ss.pt'))
    if final_target is not None:
        torch.save(final_target, os.path.join(output_dir, 'final_target.pt'))
    torch.save(known_mask, os.path.join(output_dir, 'known_mask.pt'))
    torch.save(sparse_structure, os.path.join(output_dir, 'initial_ss.pt'))
    
    print(f"[run_sds_completion_ss] Results saved to: {output_dir}")
    
    return {
        'completed': completed_ss,
        'refined': refined_ss,
        'final_target': final_target,
        'history': history,
        'known_mask': known_mask,
        'initial': sparse_structure
    }

SparseStructureSDSConfig = SDSConfig

if __name__ == "__main__":
    print("=" * 60)
    print("Sparse Structure SDS Completion Module")
    print("=" * 60)
    
    # Test config
    cfg = SDSConfig(total_iters=100, print_interval=10)
    print(f"Config: {cfg}")
    
    # Create dummy sparse structure for testing
    dummy_ss = torch.zeros(1, 1, 64, 64, 64)
    dummy_ss[0, 0, 20:40, 20:40, 20:40] = 1.0  # A cube
    print(f"Dummy sparse structure shape: {dummy_ss.shape}")
    print(f"Dummy occupancy: {dummy_ss.sum().item():.0f}")
    
    print("\nModule loaded successfully. Full test requires TRELLIS installation.")
