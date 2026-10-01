"""Multi-view DINO feature projection to Sparse Structure → Sparse Latents
(appearance SDS pipeline helpers)

Workflow
--------
Phase 1 – Feature Aggregation  (MultiViewDINOProjectionDataset)
  1. Load all training cameras from a scan via GeoSVR DataPack.
  2. Scan instance_masks to build the pixel_id → class_index mapping
     (same logic as segm_3d / _preload_rendering_data).
  3. Resolve the given obj_id (pixel value) to a class index and locate
     the corresponding object mask pixel.
  4. For each valid camera view (target object visible ≥ min_obj_pixels):
       a. Apply the instance mask to the RGB image (non-object pixels → 0).
       b. Resize to dino_image_size × dino_image_size; apply ImageNet
          normalisation.
       c. Run DINOv2 → patch tokens  (B, C, n_patch, n_patch).
       d. Project sparse voxel 3-D positions → 2-D UV via perspective
          projection  (utils3d.torch.project_cv).
       e. Sample patch features at the projected UV  (F.grid_sample,
          bilinear, padding_mode='zeros').
  5. Mean-aggregate sampled features across all valid views
     → (N, C) per-voxel feature tensor.

Phase 2 – Latent Encoding  (encode_object_slat)
  6. Build  SparseTensor  from (N, C) features + (N, 3) voxel indices.
  7. Run  SLatEncoder  → Sparse Latents.
  8. Save latents to the specified .npz path.

References
----------
- dataset_toolkits/extract_feature.py   (DINO extraction + projection)
- dataset_toolkits/encode_latent.py     (SLatEncoder encoding)
- pipeline/score_distillation_sampling_ss.py  _preload_rendering_data
  (camera loading, id→class mapping)
"""

import os
import sys
import glob
import json
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from torchvision import transforms
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

TRELLIS_ROOT = os.path.join(PROJECT_ROOT, 'decomvoxel', 'model', 'TRELLIS')
if TRELLIS_ROOT not in sys.path:
    sys.path.insert(0, TRELLIS_ROOT)

_GEOSVR_ROOT = os.path.join(PROJECT_ROOT, 'decomvoxel', 'representation', 'GeoSVR')


def _ensure_geosvr_path() -> None:
    for p in (_GEOSVR_ROOT, os.path.join(_GEOSVR_ROOT, 'src')):
        if p not in sys.path:
            sys.path.insert(0, p)


import utils3d.torch as utils3d_torch
import trellis.models as trellis_models
import trellis.modules.sparse as sp

DEBUG=True

# ===========================================================================
#  Phase 1 – Multi-view DINO feature aggregation onto sparse voxels
# ===========================================================================

class MultiViewDINOProjectionDataset:

    def __init__(
        self,
        source_path: str,
        obj_id: int,
        voxel_coords: torch.Tensor,
        voxel_resolution: int = 64,
        voxel_centers_world: Optional[torch.Tensor] = None,
        dino_model_name: str = 'dinov2_vitl14_reg',
        dino_image_size: int = 518,
        min_obj_pixels: int = 50,
        bg_id: int = 255,
        mask_dir: Optional[str] = None,
        batch_size: int = 8,
        device: Optional[torch.device] = None,
        debug_output_dir: Optional[str] = None,
    ):
        self.source_path = source_path
        self.obj_id = int(obj_id)
        self.voxel_coords = voxel_coords.long()          # (N, 3)
        self.voxel_resolution = voxel_resolution
        self.dino_model_name = dino_model_name
        self.dino_image_size = dino_image_size
        self.n_patch = dino_image_size // 14             # e.g. 518//14 = 37
        self.min_obj_pixels = min_obj_pixels
        self.bg_id = bg_id
        if mask_dir is not None:
            self.mask_dir = mask_dir
        else:
            # Support both 'instance_masks' and 'instance_mask' directory names
            _cands = [
                os.path.join(source_path, 'instance_masks'),
                os.path.join(source_path, 'instance_mask'),
            ]
            self.mask_dir = next(
                (p for p in _cands if os.path.isdir(p)),
                _cands[0],  # fallback so error message is meaningful
            )
        self.batch_size = batch_size
        self.debug_output_dir = debug_output_dir
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

        # Voxel positions for projection onto camera images.
        # IMPORTANT: must be in the SAME coordinate space as the camera extrinsics.
        # GeoSVR cameras use real world-space w2c matrices — pass voxel_centers_world.
        # Fallback uses TRELLIS-normalised [-0.5, 0.5] which is WRONG for GeoSVR cameras.
        if voxel_centers_world is not None:
            self.voxel_positions: torch.Tensor = voxel_centers_world.float().to(self.device)
            if DEBUG:
                _vp = self.voxel_positions
                print(
                    f"[MultiViewDINO] Voxel positions (world-space): "
                    f"x=[{_vp[:,0].min():.3f},{_vp[:,0].max():.3f}], "
                    f"y=[{_vp[:,1].min():.3f},{_vp[:,1].max():.3f}], "
                    f"z=[{_vp[:,2].min():.3f},{_vp[:,2].max():.3f}]"
                )
        else:
            self.voxel_positions = (
                self.voxel_coords.float().to(self.device) + 0.5
            ) / voxel_resolution - 0.5
            if DEBUG:
                print(
                    f"[MultiViewDINO] WARNING: voxel_centers_world not provided -- "
                    f"using TRELLIS-normalised positions "
                    f"[{self.voxel_positions.min():.3f}, {self.voxel_positions.max():.3f}]. "
                    f"Projection will be WRONG with GeoSVR world-space cameras!"
                )

        # ImageNet normalisation for DINOv2
        self._dino_transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        # Populated by constructor helpers
        self._dino_model = None
        self._cameras: List = []
        self._obj_masks: List[torch.Tensor] = []   # per-camera (H, W) bool
        self._obj_pixel_id: int = obj_id           # pixel value in masks
        self._obj_class: int = -1                  # segm_3d class index
        self._id_to_class: Dict[int, int] = {}

        # Initialise
        self._build_id_class_mapping()
        self._load_cameras_and_masks()

    # -------------------------------------------------------------------
    # Initialisation helpers
    # -------------------------------------------------------------------

    def _build_id_class_mapping(self) -> None:
        """Scan all mask files and build pixel_id → class_index mapping.

        Convention mirrors segm_3d / _preload_rendering_data:
          bg_id        → class 0
          sorted non-bg pixel ids → class 1, 2, 3, …
        """
        import cv2

        _IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
        mask_files = sorted(
            p for p in glob.glob(os.path.join(self.mask_dir, '*'))
            if os.path.splitext(p)[1].lower() in _IMG_EXTS
        )
        if not mask_files:
            raise FileNotFoundError(
                f"No mask files found in '{self.mask_dir}'"
            )

        unique_ids: set = set()
        for mf in mask_files:
            m = cv2.imread(mf, cv2.IMREAD_UNCHANGED)
            if m is None:
                continue
            if m.ndim > 2:
                m = m[:, :, 0]
            unique_ids.update(np.unique(m).tolist())

        if self.bg_id not in unique_ids and 0 in unique_ids:
            print(
                f"[MultiViewDINO] WARNING: bg_id={self.bg_id} not found in masks, "
                f"but id 0 exists. If your background is black, set bg_id=0."
            )

        object_pixel_ids = sorted(
            int(uid) for uid in unique_ids if int(uid) != self.bg_id
        )
        self._id_to_class = {self.bg_id: 0}
        for i, oid in enumerate(object_pixel_ids):
            self._id_to_class[oid] = i + 1
        self._class_to_id: Dict[int, int] = {
            v: k for k, v in self._id_to_class.items()
        }

        if self.obj_id not in self._id_to_class:
            raise ValueError(
                f"obj_id={self.obj_id} not found in instance masks under "
                f"'{self.mask_dir}'. "
                f"Valid pixel ids: {object_pixel_ids}"
            )

        self._obj_pixel_id = self.obj_id
        self._obj_class = self._id_to_class[self.obj_id]
        print(
            f"[MultiViewDINO] obj_id={self.obj_id} → class={self._obj_class} "
            f"(scan contains {len(object_pixel_ids)} objects)"
        )

    def _load_cameras_and_masks(self) -> None:
        """Load GeoSVR train cameras; keep only views where the target
        object is visible with ≥ min_obj_pixels pixels."""
        _ensure_geosvr_path()
        import cv2
        from yacs.config import CfgNode
        from src.dataloader.data_pack import DataPack

        cfg_data = CfgNode()
        cfg_data.source_path = self.source_path
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

        data_pack = DataPack(cfg_data)
        all_cameras = data_pack.get_train_cameras()
        print(
            f"[MultiViewDINO] Loaded {len(all_cameras)} training cameras "
            f"from '{self.source_path}'"
        )

        n_no_mask = 0
        n_few_pixels = 0
        valid_cameras: List = []
        valid_masks: List[torch.Tensor] = []

        for cam in all_cameras:
            # Derive basename for mask file lookup
            # (strip trailing _rgb, _color, _image suffixes)
            img_name = cam.image_name
            basename = os.path.splitext(img_name)[0]
            for suffix in ('_rgb', '_color', '_image'):
                if basename.endswith(suffix):
                    basename = basename[:-len(suffix)]
                    break

            # Locate the instance mask file (case-insensitive extension search)
            mask_path = None
            for ext in ('.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG',
                        '.bmp', '.BMP', '.tif', '.TIF', '.tiff', '.TIFF'):
                p = os.path.join(self.mask_dir, basename + ext)
                if os.path.exists(p):
                    mask_path = p
                    break
            if mask_path is None:
                n_no_mask += 1
                continue

            mask_cv = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            if mask_cv is None:
                n_no_mask += 1
                continue
            if mask_cv.ndim > 2:
                mask_cv = mask_cv[:, :, 0]

            obj_mask_np = (mask_cv == self._obj_pixel_id)
            if int(obj_mask_np.sum()) < self.min_obj_pixels:
                n_few_pixels += 1
                continue

            # Resize mask to camera resolution if needed
            H, W = cam.image_height, cam.image_width
            if obj_mask_np.shape[0] != H or obj_mask_np.shape[1] != W:
                obj_mask_np = cv2.resize(
                    obj_mask_np.astype(np.uint8), (W, H),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)

            valid_cameras.append(cam)
            valid_masks.append(torch.from_numpy(obj_mask_np).bool())  # (H, W)

        print(
            f"[MultiViewDINO] Camera filter: valid={len(valid_cameras)}, "
            f"no_mask={n_no_mask}, too_few_pixels={n_few_pixels}"
        )

        if not valid_cameras:
            raise ValueError(
                f"No valid views found for obj_id={self.obj_id} "
                f"(pixel_id={self._obj_pixel_id}, class={self._obj_class}) "
                f"in '{self.mask_dir}'"
            )

        self._cameras = valid_cameras
        self._obj_masks = valid_masks

        if DEBUG and self.debug_output_dir is not None and len(valid_cameras) > 0:
            import torchvision
            n_save = min(5, len(valid_cameras))
            dbg_dir = os.path.join(self.debug_output_dir, f'debug_masked_obj{self.obj_id}')
            os.makedirs(dbg_dir, exist_ok=True)
            for k in range(n_save):
                cam = valid_cameras[k]
                mk = valid_masks[k]
                raw = cam.image[:3].clone().float().cpu().clamp(0, 1)
                torchvision.utils.save_image(raw, os.path.join(dbg_dir, f'view{k:03d}_original.png'))
                masked = raw.clone()
                masked[:, ~mk] = 0.0
                torchvision.utils.save_image(masked, os.path.join(dbg_dir, f'view{k:03d}_masked.png'))
                torchvision.utils.save_image(mk.float().unsqueeze(0), os.path.join(dbg_dir, f'view{k:03d}_mask.png'))
            print(f"[MultiViewDINO] DEBUG: saved {n_save} sample masked views -> {dbg_dir}")

    # -------------------------------------------------------------------
    # Per-view processing helpers
    # -------------------------------------------------------------------

    def _load_dino_model(self) -> None:
        """Load DINOv2 model on first call (lazy)."""
        if self._dino_model is None:
            print(f"[MultiViewDINO] Loading DINOv2 model: {self.dino_model_name}")
            model = torch.hub.load(
                'facebookresearch/dinov2', self.dino_model_name
            )
            model.eval().to(self.device)
            self._dino_model = model
            print("[MultiViewDINO] DINOv2 ready.")

    def _cam_to_extrinsics_intrinsics(
        self, cam
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert a GeoSVR Camera to utils3d extrinsics + normalised
        intrinsics (both on self.device).

        Returns
        -------
        extrinsics : (4, 4) world-to-camera matrix.
        intrinsics : (3, 3) normalised camera intrinsics in [0, 1] range,
                     as expected by utils3d.torch.project_cv.
        """
        extrinsics = cam.w2c.to(self.device)  # (4, 4) float32
        # Build normalised intrinsics from FoV angles (radians)
        intrinsics = utils3d_torch.intrinsics_from_fov_xy(
            torch.tensor(cam.fovx, dtype=torch.float32, device=self.device),
            torch.tensor(cam.fovy, dtype=torch.float32, device=self.device),
        )  # (3, 3)
        return extrinsics, intrinsics

    def _prepare_masked_image(
        self, cam, obj_mask: torch.Tensor
    ) -> torch.Tensor:
        """Return a normalised (3, dino_image_size, dino_image_size) tensor:
        the camera's RGB image with non-object pixels zeroed out, resized for
        DINOv2 input.

        Parameters
        ----------
        cam      : GeoSVR Camera  (cam.image is (C, H, W) float in [0, 1] on CPU)
        obj_mask : (H, W) bool tensor selecting the target object pixels.
        """
        # Take RGB channels only (3, H, W), ensure float in [0, 1]
        img = cam.image[:3].clone().float()  # (3, H, W)

        # Zero-out pixels that do not belong to the target object
        img[:, ~obj_mask] = 0.0

        # Resize to DINOv2 input resolution
        img = F.interpolate(
            img.unsqueeze(0),  # (1, 3, H, W)
            size=(self.dino_image_size, self.dino_image_size),
            mode='bilinear',
            align_corners=False,
        ).squeeze(0)  # (3, S, S)

        # Apply ImageNet normalisation
        img = self._dino_transform(img)
        return img  # (3, S, S) float32

    # -------------------------------------------------------------------
    # Main public API
    # -------------------------------------------------------------------

    def compute_aggregated_features(
        self,
        use_non_zero_init: bool = True,
        non_zero_init_k: int = 8,
        return_init_info: bool = False,
    ):
        """Extract DINOv2 features from all valid views, project onto voxel
        positions, and return the mean per-voxel feature vector.

        Returns
        -------
        patchtokens_agg : (N, feat_dim) float32 Tensor  – per-voxel features
                          averaged across all valid views.
        voxel_indices   : (N, 3) int32 Tensor           – voxel grid indices.
        """
        self._load_dino_model()

        N = self.voxel_positions.shape[0]
        n_views = len(self._cameras)
        n_patch = self.n_patch

        print(
            f"[MultiViewDINO] Processing {n_views} views "
            f"for {N} voxels (batch_size={self.batch_size}) ..."
        )

        # Pre-compute masked images and camera matrices for all views
        images_list: List[torch.Tensor] = []
        extrinsics_list: List[torch.Tensor] = []
        intrinsics_list: List[torch.Tensor] = []
        obj_mask_resized_list: List[torch.Tensor] = []
        for cam, obj_mask in zip(self._cameras, self._obj_masks):
            images_list.append(self._prepare_masked_image(cam, obj_mask))
            ext, intr = self._cam_to_extrinsics_intrinsics(cam)
            extrinsics_list.append(ext)
            intrinsics_list.append(intr)
            obj_mask_resized = F.interpolate(
                obj_mask.float().unsqueeze(0).unsqueeze(0),
                size=(self.dino_image_size, self.dino_image_size),
                mode='nearest',
            ).squeeze(0).squeeze(0)
            obj_mask_resized_list.append(obj_mask_resized)

        # Streaming accumulation (avoids keeping all feature maps in memory)
        patchtokens_sum: Optional[torch.Tensor] = None
        voxel_vis_count: Optional[torch.Tensor] = None  # per-voxel in-frame view count

        with torch.no_grad():
            for i in range(0, n_views, self.batch_size):
                # Batch of images
                batch_imgs = torch.stack(
                    images_list[i: i + self.batch_size]
                ).to(self.device)  # (B, 3, S, S)

                # Camera matrices (already on self.device from init)
                batch_ext = torch.stack(
                    extrinsics_list[i: i + self.batch_size]
                )  # (B, 4, 4)
                batch_intr = torch.stack(
                    intrinsics_list[i: i + self.batch_size]
                )  # (B, 3, 3)
                batch_obj_mask = torch.stack(
                    obj_mask_resized_list[i: i + self.batch_size]
                ).to(self.device)  # (B, S, S)
                B = batch_imgs.shape[0]

                # ---- DINOv2 forward pass --------------------------------
                feats = self._dino_model(batch_imgs, is_training=True)
                # Drop [CLS] token and all register tokens
                # patch_seq: (B, n_tokens, C)
                patch_seq = feats['x_prenorm'][
                    :, self._dino_model.num_register_tokens + 1:
                ]
                # Reshape to spatial grid: (B, C, n_patch, n_patch)
                pt = patch_seq.permute(0, 2, 1).reshape(
                    B, -1, n_patch, n_patch
                )  # (B, C, n_patch, n_patch)

                # ---- Perspective projection: 3-D voxels → 2-D UV -------
                uv_01, depth = utils3d_torch.project_cv(
                    self.voxel_positions,  # (N, 3)
                    batch_ext,             # (B, 4, 4)
                    batch_intr,            # (B, 3, 3)
                )  # uv_01: (B, N, 2) in [0,1],  depth: (B, N)
                uv = uv_01 * 2.0 - 1.0    # (B, N, 2) -> grid_sample range [-1, 1]

                # Visibility: voxel is in front of camera AND projects within frame
                in_frame = (
                    (uv[..., 0].abs() <= 1.0) &
                    (uv[..., 1].abs() <= 1.0) &
                    (depth > 0)
                )  # (B, N) bool

                # Sample resized object mask at projected UV and require the
                # projected voxel to lie on object pixels (not only in-frame).
                sampled_mask = F.grid_sample(
                    batch_obj_mask.unsqueeze(1),
                    uv.unsqueeze(1),
                    mode='nearest',
                    align_corners=False,
                    padding_mode='zeros',
                ).squeeze(1).squeeze(1)  # (B, N)
                obj_visible = in_frame & (sampled_mask > 0.5)

                if DEBUG and i == 0:
                    print(
                        f"[MultiViewDINO] DEBUG batch0 UV(0-1): "
                        f"x=[{uv_01[...,0].min():.3f},{uv_01[...,0].max():.3f}], "
                        f"y=[{uv_01[...,1].min():.3f},{uv_01[...,1].max():.3f}], "
                        f"depth=[{depth.min():.3f},{depth.max():.3f}], "
                        f"in_frame_rate={in_frame.float().mean():.2%}, "
                        f"obj_visible_rate={obj_visible.float().mean():.2%}"
                    )

                # ---- Bilinear feature sampling --------------------------
                # F.grid_sample:
                #   input (B, C, H_in, W_in),  grid (B, H_out, W_out, 2)
                # padding_mode='zeros': voxels outside image or behind camera -> 0
                sampled = F.grid_sample(
                    pt,                    # (B, C, n_patch, n_patch)
                    uv.unsqueeze(1),       # (B, 1, N, 2)
                    mode='bilinear',
                    align_corners=False,
                    padding_mode='zeros',
                )  # (B, C, 1, N)
                sampled = sampled.squeeze(2).permute(0, 2, 1)  # (B, N, C)
                sampled = sampled * obj_visible.unsqueeze(-1).float()

                # Accumulate sum and per-voxel visibility count
                batch_sum = sampled.sum(dim=0)           # (N, C)
                batch_vis = obj_visible.float().sum(dim=0)  # (N,): number of views where voxel is visible on object mask
                patchtokens_sum = (
                    batch_sum if patchtokens_sum is None
                    else patchtokens_sum + batch_sum
                )
                voxel_vis_count = (
                    batch_vis if voxel_vis_count is None
                    else voxel_vis_count + batch_vis
                )

        if patchtokens_sum is None or voxel_vis_count is None:
            raise RuntimeError("No views were processed.")

        # Per-voxel mean: divide only by number of views where each voxel was visible
        safe_vis = voxel_vis_count.clamp(min=1.0).unsqueeze(1)  # (N, 1)
        patchtokens_agg = patchtokens_sum / safe_vis             # (N, C)
        voxel_indices = self.voxel_coords.int()                  # (N, 3)

        unseen_mask = (voxel_vis_count == 0)
        filled_mask = torch.zeros_like(unseen_mask)
        if use_non_zero_init:
            if unseen_mask.any():
                seen_mask = ~unseen_mask
                non_zero_mask = patchtokens_agg.abs().sum(dim=1) > 1e-8
                src_mask = seen_mask & non_zero_mask
                if not src_mask.any():
                    src_mask = seen_mask
                if src_mask.any():
                    src_coords = self.voxel_coords[src_mask].float().to(self.device)
                    src_feats = patchtokens_agg[src_mask]
                    dst_coords = self.voxel_coords[unseen_mask].float().to(self.device)
                    k = min(max(int(non_zero_init_k), 1), src_coords.shape[0])
                    dists = torch.cdist(dst_coords, src_coords)
                    knn_idx = dists.topk(k=k, largest=False).indices
                    filled_feats = src_feats[knn_idx].mean(dim=1)
                    patchtokens_agg = patchtokens_agg.clone()
                    patchtokens_agg[unseen_mask] = filled_feats
                    filled_mask = unseen_mask.clone()
                    print(
                        f"[MultiViewDINO] Non-zero init filled {int(unseen_mask.sum().item())} unseen voxels "
                        f"with neighbor mean (k={k})."
                    )

        if DEBUG:
            n_zero = (voxel_vis_count == 0).sum().item()
            print(
                f"[MultiViewDINO] Aggregation done: shape={tuple(patchtokens_agg.shape)}, "
                f"feat_mean={patchtokens_agg.mean().item():.4f}, "
                f"feat_std={patchtokens_agg.std().item():.4f}, "
                f"voxels_never_visible={n_zero}/{N} ({n_zero/N:.1%}), "
                f"mean_vis_count={voxel_vis_count.mean().item():.1f} views"
            )
        else:
            print(
                f"[MultiViewDINO] Aggregation done: shape={tuple(patchtokens_agg.shape)}, "
                f"mean={patchtokens_agg.mean().item():.4f}, "
                f"std={patchtokens_agg.std().item():.4f}"
            )
        if return_init_info:
            init_info = {
                'voxel_indices': voxel_indices.detach().cpu(),
                'unseen_mask': unseen_mask.detach().cpu(),
                'filled_mask': filled_mask.detach().cpu(),
            }
            return patchtokens_agg, voxel_indices, init_info

        return patchtokens_agg, voxel_indices


# ===========================================================================
#  Phase 2 – SLatEncoder encoding → save .npz
# ===========================================================================

def encode_object_slat(
    source_path: str,
    obj_id: int,
    voxel_coords: torch.Tensor,
    output_path: str,
    enc_pretrained: str = (
        'microsoft/TRELLIS-image-large/ckpts/slat_enc_swin8_B_64l8_fp16'
    ),
    voxel_resolution: int = 64,
    voxel_centers_world: Optional[torch.Tensor] = None,
    dino_model_name: str = 'dinov2_vitl14_reg',
    dino_image_size: int = 518,
    min_obj_pixels: int = 50,
    bg_id: int = 255,
    mask_dir: Optional[str] = None,
    batch_size: int = 8,
    use_non_zero_init: bool = True,
    non_zero_init_k: int = 8,
    device: Optional[torch.device] = None,
    debug_output_dir: Optional[str] = None,
) -> sp.SparseTensor:
    """End-to-end pipeline: multi-view DINO aggregation → SLatEncoder →
    save latents as ``.npz``.

    Parameters
    ----------
    source_path : str
        Dataset root (e.g. ``datasets/Replica/scan0``).
    obj_id : int
        Object pixel-value in instance_masks (original id, NOT class index).
    voxel_coords : Tensor (N, 3) int
        Integer voxel grid indices of the sparse structure in
        ``[0, voxel_resolution - 1]``.
    output_path : str
        Destination ``.npz`` file path.  Parent directories are created
        automatically.
    enc_pretrained : str
        HuggingFace repo / local path for the pretrained SLatEncoder.
    voxel_resolution : int
        Sparse grid resolution (default 64).
    dino_model_name : str
        DINOv2 variant.
    dino_image_size : int
        Input size for DINOv2 (default 518).
    min_obj_pixels : int
        Minimum object pixels per view.
    bg_id : int
        Background pixel value in instance_masks.
    mask_dir : str | None
        Custom instance mask directory.
    batch_size : int
        DINOv2 forward-pass batch size.
    use_non_zero_init : bool
        Replace never-observed voxel features using nearby non-zero features.
    non_zero_init_k : int
        Number of nearest non-zero neighbors used for feature averaging.
    device : torch.device | None
        Computation device.

    Returns
    -------
    latent : SparseTensor
        The encoded sparse latent (also persisted to ``output_path``).

    Saved .npz keys
    ---------------
    feats  : (N, latent_dim) float32 ndarray
    coords : (N, 3) uint8 ndarray  – voxel grid indices (batch dim stripped)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Default debug output dir to the parent of the output .npz
    if debug_output_dir is None:
        debug_output_dir = os.path.dirname(os.path.abspath(output_path))

    if DEBUG:
        print(
            f"[encode_object_slat] obj_id={obj_id}, "
            f"voxel_coords shape={tuple(voxel_coords.shape)}, "
            f"voxel_centers_world={'provided' if voxel_centers_world is not None else 'NOT provided (WARNING: TRELLIS-space fallback)'}"
        )

    # ------------------------------------------------------------------
    # Phase 1: multi-view DINO feature aggregation
    # ------------------------------------------------------------------
    dataset = MultiViewDINOProjectionDataset(
        source_path=source_path,
        obj_id=obj_id,
        voxel_coords=voxel_coords,
        voxel_resolution=voxel_resolution,
        voxel_centers_world=voxel_centers_world,
        dino_model_name=dino_model_name,
        dino_image_size=dino_image_size,
        min_obj_pixels=min_obj_pixels,
        bg_id=bg_id,
        mask_dir=mask_dir,
        batch_size=batch_size,
        device=device,
        debug_output_dir=debug_output_dir,
    )
    patchtokens, voxel_indices, init_info = dataset.compute_aggregated_features(
        use_non_zero_init=use_non_zero_init,
        non_zero_init_k=non_zero_init_k,
        return_init_info=True,
    )

    if use_non_zero_init:
        from decomvoxel.utils.vis_slat import visualize_non_zero_init
        visualize_non_zero_init(
            voxel_indices=init_info['voxel_indices'],
            unseen_mask=init_info['unseen_mask'],
            filled_mask=init_info['filled_mask'],
            output_dir=os.path.join(os.path.dirname(os.path.abspath(output_path)), 'slat_visualization'),
            prefix='non_zero_init',
            grid_resolution=voxel_resolution,
        )

    # Free DINOv2 GPU memory before loading the encoder
    dataset._dino_model = None
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Load SLatEncoder
    # ------------------------------------------------------------------
    print(f"[encode_object_slat] Loading SLatEncoder: {enc_pretrained}")
    encoder = trellis_models.from_pretrained(enc_pretrained).eval().to(device)

    # ------------------------------------------------------------------
    # Phase 2: build SparseTensor and encode
    # ------------------------------------------------------------------
    N = patchtokens.shape[0]

    # Prepend a batch index of 0 to each voxel coordinate → (N, 4)
    coords = torch.cat(
        [
            torch.zeros(N, 1, dtype=torch.int32, device=device),
            voxel_indices.int().to(device),          # (N, 3)
        ],
        dim=1,
    )  # (N, 4): [batch_idx=0, x, y, z]

    feats_sparse = sp.SparseTensor(
        feats=patchtokens.float(),
        coords=coords,
    )

    print(f"[encode_object_slat] Encoding {N} voxels with SLatEncoder ...")
    with torch.no_grad():
        latent = encoder(feats_sparse, sample_posterior=False)

    if not torch.isfinite(latent.feats).all():
        raise RuntimeError(
            "Non-finite values detected in the encoded latent. "
            "Check input features / encoder weights."
        )

    if DEBUG:
        print(
            f"[encode_object_slat] Latent stats: "
            f"feats shape={tuple(latent.feats.shape)}, "
            f"coords xyz range=[{latent.coords[:,1:].min().item()},{latent.coords[:,1:].max().item()}], "
            f"feat_mean={latent.feats.float().mean().item():.4f}, "
            f"feat_std={latent.feats.float().std().item():.4f}, "
            f"feat_abs_max={latent.feats.float().abs().max().item():.4f}"
        )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    pack = {
        'feats':  latent.feats.cpu().numpy().astype(np.float32),
        # Strip the leading batch dimension from coords
        'coords': latent.coords[:, 1:].cpu().numpy().astype(np.uint8),
    }
    np.savez_compressed(output_path, **pack)
    print(f"[encode_object_slat] Latents saved → {output_path}")

    return latent


# ===========================================================================
#  Phase 3 – SLatMeshDecoder: structured latents → mesh
# ===========================================================================

def decode_slat_to_mesh(
    latent: sp.SparseTensor,
    output_path: str,
    pretrained: str = 'JeffreyXiang/TRELLIS-image-large',
    simplify_ratio: float = 0.95,
    texture_size: int = 1024,
    device: Optional[torch.device] = None,
) -> str:
    from decomvoxel.pipeline.sparse_structure_to_mesh import TrellisModelManager, MeshExporter

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    manager = TrellisModelManager()
    manager.load(device=str(device))

    with torch.no_grad():
        decoded = manager.pipeline.decode_slat(latent, formats=['mesh', 'gaussian'])

    mesh_results = decoded.get('mesh', [])
    gaussian_results = decoded.get('gaussian', [])

    mesh_result = mesh_results[0] if mesh_results else None
    gaussian_result = gaussian_results[0] if gaussian_results else None

    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)

    MeshExporter.export_glb(
        mesh_result,
        gaussian_result,
        output_path=output_path,
        simplify=simplify_ratio,
        texture_size=texture_size,
    )
    print(f"[decode_slat_to_mesh] Mesh saved → {output_path}")
    return output_path


@dataclass
class AppearanceSDSConfig:
    """Configuration for SLAT-space SDS optimization (appearance stage).

    These keys live under the ``appearance:`` section of the YAML config so
    they do **not** conflict with any ``sds:`` keys used by the geometry SDS
    stage, even when both sections share the same key name.
    """
    total_iters: int = 200
    lr: float = 1e-3
    cfg_strength: float = 3.0
    # cfg_interval is stored as two separate scalars so it round-trips
    # cleanly via plain YAML (e.g. ``cfg_interval_lo: 0.5``).
    cfg_interval_lo: float = 0.5
    cfg_interval_hi: float = 1.0
    noise_start: float = 0.02
    noise_end: float = 0.98
    weighting_strategy: str = 'snr'   # 'snr' (w_t = 1-t) or 'uniform'
    print_interval: int = 50
    save_interval: int = 50   # Save history plot+JSON every N iterations (0 = only at end)
    use_non_zero_init: bool = True
    non_zero_init_k: int = 8
    # Timestep sampling schedule (mirrors SDSConfig.timestep_schedule)
    timestep_schedule: str = 'uniform'  # 'uniform', 'linear', 'bell', 'progressive'
    linear_anneal_iter: int = 0         # For 'linear': iter at which t_low reaches noise_start (0 = total_iters)
    use_rescaled_t: bool = False        # Apply TRELLIS t-rescaling: t' = r·t/(1+(r-1)·t)
    rescale_t: float = 3.0             # Rescaling factor r (only used when use_rescaled_t=True)
    use_certainty_mask_loss: bool = True
    certainty_mask_weight: float = 0.3
    certainty_mask_weight_schedule: str = 'uniform'
    certainty_mask_end_iter: int = -1
    certainty_mask_poly_power: float = 2.0
    certainty_mask_exp_gamma: float = 5.0
    certainty_blur_sigma: float = 1.0
    certainty_mask_power: float = 2.0


def _sample_appearance_timestep(iteration: int, cfg: 'AppearanceSDSConfig', device: torch.device) -> float:
    """Sample a diffusion timestep t in [noise_start, noise_end] according to
    cfg.timestep_schedule.  Mirrors SparseStructureCompleter.sample_timestep."""
    progress = iteration / max(cfg.total_iters - 1, 1)  # 0 → 1
    t_low  = cfg.noise_start
    t_high = cfg.noise_end
    schedule = cfg.timestep_schedule

    if schedule == 'linear':
        anneal_iter = cfg.linear_anneal_iter if cfg.linear_anneal_iter > 0 else cfg.total_iters
        p = min(iteration / max(anneal_iter, 1), 1.0)
        t_low  = cfg.noise_end * (1.0 - p) * 0.8 + cfg.noise_start * p
        t_high = cfg.noise_end

    elif schedule == 'bell':
        bell   = 4.0 * progress * (1.0 - progress)
        t_high = cfg.noise_start + bell * (cfg.noise_end - cfg.noise_start)
        t_high = max(t_high, cfg.noise_start + 1e-4)
        t_low  = cfg.noise_start + 0.8 * bell * (cfg.noise_end - cfg.noise_start)
        t_low  = min(t_low, t_high - 1e-4)

    elif schedule == 'progressive':
        t_high = cfg.noise_start + progress * (cfg.noise_end - cfg.noise_start)
        t_high = max(t_high, cfg.noise_start + 1e-4)
        t_low  = cfg.noise_start + 0.8 * progress * (cfg.noise_end - cfg.noise_start)
        t_low  = min(t_low, t_high - 1e-4)

    t = torch.rand(1, device=device) * (t_high - t_low) + t_low

    if cfg.use_rescaled_t:
        r = cfg.rescale_t
        t = r * t / (1 + (r - 1) * t)

    return t.item()


def _schedule_weight(
    iteration: int,
    total_iters: int,
    end_iter: int,
    schedule: str,
    poly_power: float,
    exp_gamma: float,
) -> float:
    final_iter = end_iter if end_iter > 0 else total_iters
    if iteration > final_iter:
        return 0.0
    progress = iteration / max(final_iter, 1)
    if schedule == 'cosine':
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    if schedule == 'cos_power':
        cos_base = 0.5 * (1.0 + np.cos(np.pi * progress))
        return cos_base ** poly_power
    if schedule == 'linear':
        return 1.0 - progress
    if schedule == 'poly':
        return (1.0 - progress) ** poly_power
    if schedule == 'exponential':
        return float(np.exp(-exp_gamma * progress))
    if schedule == 'logarithmic':
        return 1.0 - float(np.log(progress * (np.e - 1) + 1.0))
    return 1.0


def _gaussian_kernel_1d(sigma: float, device: torch.device) -> torch.Tensor:
    radius = max(1, int(3.0 * sigma))
    x = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def _gaussian_blur_3d(volume: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return volume
    k = _gaussian_kernel_1d(sigma, volume.device).to(volume.dtype)
    pad = k.numel() // 2
    v = volume.unsqueeze(0).unsqueeze(0)
    kx = k.view(1, 1, -1, 1, 1)
    ky = k.view(1, 1, 1, -1, 1)
    kz = k.view(1, 1, 1, 1, -1)
    v = F.conv3d(v, kx, padding=(pad, 0, 0))
    v = F.conv3d(v, ky, padding=(0, pad, 0))
    v = F.conv3d(v, kz, padding=(0, 0, pad))
    return v.squeeze(0).squeeze(0)


def plot_appearance_history(
    history: Dict,
    output_path: str,
    cfg: Optional['AppearanceSDSConfig'] = None,
) -> None:
    """Render a 2×2 training-history figure and save it as a PNG.

    Subplots:
        (0,0) SDS loss per iteration
        (0,1) Sampled noise timestep *t* per iteration
        (1,0) Feature magnitude: mean(|feats|) and std(feats) over iterations
        (1,1) Gradient norm of feats_param (log scale)
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    iters = list(range(1, len(history['loss_sds']) + 1))

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Appearance SDS Training History', fontsize=16, fontweight='bold')

    # (0,0) SDS Loss
    ax1 = axes[0, 0]
    ax1.plot(iters, history['loss_sds'], linewidth=2, color='#2E86AB', label='SDS Loss')
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('SDS Loss', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)

    # (0,1) Sampled t per iteration
    ax2 = axes[0, 1]
    if 't_history' in history and len(history['t_history']) > 0:
        n = min(len(history['t_history']), len(iters))
        ax2.plot(iters[:n], history['t_history'][:n], linewidth=2, color='#A23B72', label='t (timestep)')
        ax2.set_xlabel('Iteration', fontsize=12)
        ax2.set_ylabel('t', fontsize=12)
        ax2.set_title('Sampled t per Iteration', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10)
    else:
        ax2.text(0.5, 0.5, 'No t_history data', fontsize=14, ha='center', va='center',
                 transform=ax2.transAxes)

    # (1,0) Feature magnitude: mean(|feats|) and std(feats)
    ax3 = axes[1, 0]
    if 'feats_mean' in history and len(history['feats_mean']) > 0:
        n = min(len(history['feats_mean']), len(iters))
        ax3.plot(iters[:n], history['feats_mean'][:n], linewidth=2, color='#C73E1D', label='mean(|feats|)')
        if 'feats_std' in history and len(history['feats_std']) > 0:
            ax3.plot(iters[:n], history['feats_std'][:n], linewidth=2, color='#FF9F1C',
                     linestyle='--', label='std(feats)')
        ax3.set_xlabel('Iteration', fontsize=12)
        ax3.set_ylabel('Value', fontsize=12)
        ax3.set_title('Feature Magnitude', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=10)
    else:
        ax3.text(0.5, 0.5, 'No feats stats', fontsize=14, ha='center', va='center',
                 transform=ax3.transAxes)

    # (1,1) Gradient norm (log scale)
    ax4 = axes[1, 1]
    if 'grad_norm' in history and len(history['grad_norm']) > 0:
        n = min(len(history['grad_norm']), len(iters))
        ax4.plot(iters[:n], history['grad_norm'][:n], linewidth=2, color='#6A994E', label='Grad Norm')
        ax4.set_xlabel('Iteration', fontsize=12)
        ax4.set_ylabel('||∇||', fontsize=12)
        ax4.set_title('Gradient Norm (feats_param)', fontsize=14, fontweight='bold')
        ax4.set_yscale('log')
        ax4.grid(True, alpha=0.3)
        ax4.legend(fontsize=10)
    else:
        ax4.text(0.5, 0.5, 'No grad_norm data', fontsize=14, ha='center', va='center',
                 transform=ax4.transAxes)

    # Config info text box
    if cfg is not None:
        info_text = (
            f"Configuration:\n"
            f"total_iters: {cfg.total_iters}\n"
            f"lr: {cfg.lr}\n"
            f"cfg_strength: {cfg.cfg_strength}\n"
            f"weighting: {cfg.weighting_strategy}\n"
            f"t: [{cfg.noise_start}, {cfg.noise_end}]"
        )
        fig.text(0.02, 0.02, info_text, fontsize=9, family='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def slat_sds(
    latent,
    cond_image: str,
    output_dir: str,
    cfg: Optional['AppearanceSDSConfig'] = None,
    certainty_grid: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
):
    """
    SDS optimization in SLAT (SparseTensor) feature space.

    Keeps voxel coordinates fixed and optimizes ``latent.feats`` via SDS
    loss using the Sparse Flow Transformer as the diffusion prior.

    Args:
        latent: SparseTensor produced by :func:`encode_object_slat` (unnormalized).
        cond_image: Path to the conditioning image (RGBA or RGB).
        output_dir: Directory for loss history / debug outputs.
        cfg: :class:`AppearanceSDSConfig` instance.  Defaults are used when
            ``None``.
        certainty_grid: Per-voxel certainty prior from the initial sparse structure.
        device: Torch device; defaults to CUDA if available.

    Returns:
        SparseTensor with refined feats (same coordinate layout as input).
    """
    if cfg is None:
        cfg = AppearanceSDSConfig()
    from decomvoxel.pipeline.sparse_structure_to_mesh import TrellisModelManager, ImageConditioner
    from trellis.modules import sparse as sp

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load TRELLIS models
    # ------------------------------------------------------------------
    manager = TrellisModelManager()
    manager.load(device=str(device))

    flow_model = manager.get_slat_flow_model()
    sampler    = manager.get_slat_sampler()
    normalization = manager.get_slat_normalization()

    flow_model.eval()
    for p in flow_model.parameters():
        p.requires_grad_(False)

    print(f"[slat_sds] Flow model loaded. in_channels={flow_model.in_channels}")

    # ------------------------------------------------------------------
    # 2. Conditioning from cond_image
    # ------------------------------------------------------------------
    conditioner = ImageConditioner(manager)
    condition = conditioner.prepare(cond_image, preprocess=True)
    # Move to device; 'cond' and 'neg_cond' are already on pipeline device,
    # but ensure consistency.
    condition = {k: v.to(device) for k, v in condition.items()}
    print(f"[slat_sds] Condition prepared. cond shape: {condition['cond'].shape}")

    # ------------------------------------------------------------------
    # 3. Normalization constants (flow model operates on normalised feats)
    # ------------------------------------------------------------------
    mean = torch.tensor(normalization['mean'], device=device, dtype=torch.float32).unsqueeze(0)  # (1, C)
    std  = torch.tensor(normalization['std'],  device=device, dtype=torch.float32).unsqueeze(0)  # (1, C)

    # ------------------------------------------------------------------
    # 4. Optimisation setup — feats_param is in *unnormalised* space
    # ------------------------------------------------------------------
    coords = latent.coords.detach().to(device)          # (N, 4), fixed
    init_feats = latent.feats.detach().clone().to(device)  # (N, C)

    feats_param = torch.nn.Parameter(init_feats)
    optimizer   = torch.optim.Adam([feats_param], lr=cfg.lr)
    scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.total_iters, eta_min=cfg.lr * 0.1
    )

    print(
        f"[slat_sds] Starting SDS loop: iters={cfg.total_iters}, lr={cfg.lr}, "
        f"cfg_strength={cfg.cfg_strength}, t=[{cfg.noise_start},{cfg.noise_end}], "
        f"cfg_interval=[{cfg.cfg_interval_lo},{cfg.cfg_interval_hi}], "
        f"weighting={cfg.weighting_strategy}, "
        f"certainty_mask={cfg.use_certainty_mask_loss}, "
        f"certainty_mask_weight={cfg.certainty_mask_weight}, "
        f"certainty_mask_weight_schedule={cfg.certainty_mask_weight_schedule}, "
        f"certainty_mask_end_iter={cfg.certainty_mask_end_iter}, "
        f"certainty_blur_sigma={cfg.certainty_blur_sigma}, "
        f"certainty_mask_power={cfg.certainty_mask_power}"
    )

    certainty_weights = None
    if cfg.use_certainty_mask_loss and certainty_grid is not None:
        certainty_prior = certainty_grid.float().to(device).squeeze()
        certainty_prior = _gaussian_blur_3d(certainty_prior, cfg.certainty_blur_sigma)
        xyz = coords[:, 1:].long()
        certainty_weights = certainty_prior[xyz[:, 0], xyz[:, 1], xyz[:, 2]].clamp(0.0, 1.0)
        certainty_weights = certainty_weights.pow(cfg.certainty_mask_power)
        from decomvoxel.utils.vis_slat import visualize_certainty_weights
        certainty_vis_stats = visualize_certainty_weights(
            voxel_coords=xyz,
            certainty_weights=certainty_weights,
            output_dir=os.path.join(output_dir, 'slat_visualization'),
            prefix='certainty_weights',
            grid_resolution=int(certainty_prior.shape[0]),
        )
        print(
            f"[slat_sds] certainty prior ready: "
            f"mean={certainty_weights.mean().item():.4f}, "
            f"max={certainty_weights.max().item():.4f}, "
            f"vis={certainty_vis_stats['files']['weight_png']}"
        )

    history: Dict = {
        'loss_sds':    [],
        'loss_total':  [],
        'loss_mask':   [],
        'certainty_mask_schedule_w': [],
        't_history':   [],
        'grad_norm':   [],
        'feats_mean':  [],
        'feats_std':   [],
        'w_t_history': [],
    }

    # ------------------------------------------------------------------
    # 5. SDS loop
    # ------------------------------------------------------------------
    _iter = tqdm(range(cfg.total_iters), desc='[slat_sds]', dynamic_ncols=True)

    for i in _iter:
        optimizer.zero_grad()

        # Normalise feats for flow-model space (differentiable w.r.t. feats_param)
        feats_norm = (feats_param - mean) / std  # (N, C), in computation graph

        # Sample timestep according to cfg.timestep_schedule
        t_val: float = _sample_appearance_timestep(i, cfg, device)

        # Flow-matching forward: x_t = (1 − t) x_0 + t ε
        noise_feats = torch.randn_like(feats_norm.detach())          # (N, C), no grad
        x_t_feats   = (1.0 - t_val) * feats_norm + t_val * noise_feats  # in graph

        # Build noisy SparseTensor for the flow model
        x_t_sparse = sp.SparseTensor(feats=x_t_feats.detach(), coords=coords)

        # Velocity prediction — no gradient through the model
        with torch.no_grad():
            x_0_pred, eps_pred, v_pred = sampler.sample_once_eps_all(
                flow_model,
                x_t_sparse,
                t_val,
                **condition,
                cfg_strength=cfg.cfg_strength,
                cfg_interval=[cfg.cfg_interval_lo, cfg.cfg_interval_hi],
            )

        # SDS weight
        w_t = (1.0 - t_val) if cfg.weighting_strategy == 'snr' else 1.0

        # Classic SDS gradient: w_t * (ε̂ − ε), reparametrised as MSE loss
        grad_direction = w_t * (eps_pred.feats - noise_feats)         # (N, C)
        target         = (feats_norm - grad_direction).detach()        # (N, C)
        loss_sds       = 0.5 * F.mse_loss(feats_norm, target, reduction='mean')
        loss_sds = torch.nan_to_num(loss_sds)

        if certainty_weights is not None:
            voxel_delta = (feats_param - init_feats).pow(2).mean(dim=1)
            loss_mask = (certainty_weights * voxel_delta).mean()
        else:
            loss_mask = torch.zeros((), device=device, dtype=loss_sds.dtype)

        certainty_mask_schedule_w = _schedule_weight(
            i,
            cfg.total_iters,
            cfg.certainty_mask_end_iter,
            cfg.certainty_mask_weight_schedule,
            cfg.certainty_mask_poly_power,
            cfg.certainty_mask_exp_gamma,
        )
        loss_total = loss_sds + cfg.certainty_mask_weight * certainty_mask_schedule_w * loss_mask

        loss_total.backward()

        # Capture grad norm BEFORE optimizer.step() clears the gradients
        grad_norm = feats_param.grad.norm().item() if feats_param.grad is not None else 0.0

        optimizer.step()
        scheduler.step()

        loss_sds_val = loss_sds.item()
        loss_mask_val = loss_mask.item()
        loss_total_val = loss_total.item()

        # Record per-iteration statistics
        history['loss_sds'].append(loss_sds_val)
        history['loss_total'].append(loss_total_val)
        history['loss_mask'].append(loss_mask_val)
        history['certainty_mask_schedule_w'].append(certainty_mask_schedule_w)
        history['t_history'].append(t_val)
        history['grad_norm'].append(grad_norm)
        with torch.no_grad():
            history['feats_mean'].append(feats_param.abs().mean().item())
            history['feats_std'].append(feats_param.std().item())
        history['w_t_history'].append(w_t)

        if i % cfg.print_interval == 0 or i == cfg.total_iters - 1:
            print(
                f"[slat_sds] iter {i:4d}/{cfg.total_iters}  "
                f"loss_total={loss_total_val:.6f}  "
                f"loss_sds={loss_sds_val:.6f}  "
                f"loss_mask={loss_mask_val:.6f}  "
                f"mask_w={certainty_mask_schedule_w:.3f}  t={t_val:.3f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )
        if hasattr(_iter, 'set_postfix'):
            _iter.set_postfix(loss=f'{loss_total_val:.4f}', t=f'{t_val:.3f}',
                              lr=f'{scheduler.get_last_lr()[0]:.1e}')

        # Periodic history save (JSON + PNG)
        if output_dir and (
            (cfg.save_interval > 0 and i % cfg.save_interval == 0)
            or i == cfg.total_iters - 1
        ):
            with open(os.path.join(output_dir, 'appearance_history.json'), 'w') as _f:
                json.dump(history, _f)
            plot_appearance_history(
                history,
                os.path.join(output_dir, 'appearance_history.png'),
                cfg,
            )

    # ------------------------------------------------------------------
    # 6. Return refined latent (unnormalised feats, original coords)
    # ------------------------------------------------------------------
    refined_feats  = feats_param.detach()
    refined_latent = sp.SparseTensor(feats=refined_feats, coords=coords)
    print(f"[slat_sds] Done. feats: {refined_feats.shape}, coords: {coords.shape}")
    return refined_latent






# ===========================================================================
#  Standalone test entry point
# ===========================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Appearance SDS: encode slat + decode to mesh')
    parser.add_argument('--source_path', type=str, required=True,
                        help='Dataset scan root (contains images/, instance_masks/)')
    parser.add_argument('--obj_id', type=int, required=True,
                        help='Object pixel ID in instance masks')
    parser.add_argument('--voxel_path', type=str, required=True,
                        help='Path to object_XXX_voxels.pt')
    parser.add_argument('--output_dir', type=str, default='outputs/appearance_sds_test')
    parser.add_argument('--voxel_resolution', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--min_obj_pixels', type=int, default=50)
    args = parser.parse_args()

    _ensure_geosvr_path()
    sys.path.insert(0, os.path.join(PROJECT_ROOT, 'decomvoxel', 'representation', 'GeoSVR', 'src'))

    from decomvoxel.pipeline.load_voxel import load_object_voxel, voxel_to_sparse_structure
    from decomvoxel.utils.converting import sparse_structure_to_svr_voxels

    _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    voxel_data = load_object_voxel(args.voxel_path, device=_device)
    sparse_structure_obj = voxel_to_sparse_structure(voxel_data, device=_device)
    sparse_structure = sparse_structure_obj.get_occupancy_grid()

    out = sparse_structure_to_svr_voxels(
        sparse_structure, sparse_structure_obj.transform_info,
        threshold=0.2, device=_device,
    )
    grid_coords = out['grid_coords']  # (N, 3) int tensor

    os.makedirs(args.output_dir, exist_ok=True)
    slat_path = os.path.join(args.output_dir, f'obj{args.obj_id}_slat.npz')

    latent = encode_object_slat(
        source_path=args.source_path,
        obj_id=args.obj_id,
        voxel_coords=grid_coords,
        output_path=slat_path,
        voxel_resolution=args.voxel_resolution,
        batch_size=args.batch_size,
        min_obj_pixels=args.min_obj_pixels,
        device=_device,
    )

    mesh_path = os.path.join(args.output_dir, f'obj{args.obj_id}_mesh.glb')
    decode_slat_to_mesh(latent, output_path=mesh_path, device=_device)
    print(f'Done. slat={slat_path}  mesh={mesh_path}')
