"""
Background-only TSDF mesh extraction.

Uses instance masks to exclude object regions from TSDF fusion,
producing a mesh that contains only the background (walls, floor, ceiling, etc.).

Instance masks are expected at:
    <source_path>/instance_masks/<image_name>.png
where pixel value 255 = background, everything else = object.

The object pixels are masked out (depth set to 0) before TSDF integration,
so the resulting mesh only contains background surfaces.
"""

import os
import sys
import glob
import cv2
import json

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
GEOSVR_ROOT = os.path.join(os.path.dirname(PROJECT_ROOT))  # one level up: .../GeoSVR
if GEOSVR_ROOT not in sys.path:
    sys.path.insert(0, GEOSVR_ROOT)

import numpy as np
from tqdm import tqdm
import open3d as o3d
import torch
import copy
from PIL import Image
import matplotlib.pyplot as plt

from src.config import cfg, update_argparser, update_config
from src.dataloader.data_pack import DataPack
from src.sparse_voxel_model import SparseVoxelModel
from src.cameras import MiniCam

from src.utils.fuser_utils import Fuser
from src.utils.octree_utils import level_2_vox_size
from src.utils import activation_utils
from src.utils.render_utils import save_img_f32

import math
FACE_ORDER = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"]


def save_depth_vis(depth, save_path):
    if isinstance(depth, torch.Tensor):
        depth_np = depth.detach().cpu().numpy()
    else:
        depth_np = depth
    plt.imsave(save_path, depth_np, cmap="viridis")

def _normalize_np(x, eps=1e-8):
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    norm = np.clip(norm, eps, None)
    return x / norm

def _get_face_fov_deg(face_name, side_fov_deg, z_fov_deg):
    if face_name in {"+Z", "-Z"}:
        return float(z_fov_deg)
    return float(side_fov_deg)

def _build_cube_face_metadata(
    cube_size,
    cube_side_fov_deg,
    cube_z_fov_deg,
    cube_center=None,
    side_pitch_deg=0.0,
):
    cx = cube_size * 0.5
    cy = cube_size * 0.5

    # Explicit Z-up cube-face convention.
    # Camera local axes are:
    #   +x = image right
    #   +y = image down
    #   +z = camera forward
    #
    # We specify each face by world-space right/down/forward directions.
    face_axes = {
        "+X": {
            "right":   np.array([0.0, -1.0,  0.0], dtype=np.float32),
            "down":    np.array([0.0,  0.0, -1.0], dtype=np.float32),
            "forward": np.array([1.0,  0.0,  0.0], dtype=np.float32),
        },
        "-X": {
            "right":   np.array([0.0,  1.0,  0.0], dtype=np.float32),
            "down":    np.array([0.0,  0.0, -1.0], dtype=np.float32),
            "forward": np.array([-1.0, 0.0,  0.0], dtype=np.float32),
        },
        "+Y": {
            "right":   np.array([1.0,  0.0,  0.0], dtype=np.float32),
            "down":    np.array([0.0,  0.0, -1.0], dtype=np.float32),
            "forward": np.array([0.0,  1.0,  0.0], dtype=np.float32),
        },
        "-Y": {
            "right":   np.array([-1.0, 0.0,  0.0], dtype=np.float32),
            "down":    np.array([0.0,  0.0, -1.0], dtype=np.float32),
            "forward": np.array([0.0, -1.0,  0.0], dtype=np.float32),
        },
        "+Z": {
            "right":   np.array([1.0,  0.0,  0.0], dtype=np.float32),
            "down":    np.array([0.0,  1.0,  0.0], dtype=np.float32),
            "forward": np.array([0.0,  0.0,  1.0], dtype=np.float32),
        },
        "-Z": {
            "right":   np.array([1.0,  0.0,  0.0], dtype=np.float32),
            "down":    np.array([0.0, -1.0,  0.0], dtype=np.float32),
            "forward": np.array([0.0,  0.0, -1.0], dtype=np.float32),
        },
    }

    pitch_rad = math.radians(float(side_pitch_deg))
    side_faces = {"+X", "-X", "+Y", "-Y"}

    face_meta = {}
    for face_name in FACE_ORDER:
        right = face_axes[face_name]["right"]
        down = face_axes[face_name]["down"]
        forward = face_axes[face_name]["forward"]

        if face_name in side_faces and abs(pitch_rad) > 1e-8:
            base_down = down
            base_forward = forward

            # Pitch side faces downward around the local right axis.
            forward = _normalize_np(
                np.cos(pitch_rad) * base_forward +
                np.sin(pitch_rad) * base_down
            ).astype(np.float32)

            down = _normalize_np(
                np.cos(pitch_rad) * base_down -
                np.sin(pitch_rad) * base_forward
            ).astype(np.float32)

        face_fov_deg = _get_face_fov_deg(
            face_name=face_name,
            side_fov_deg=cube_side_fov_deg,
            z_fov_deg=cube_z_fov_deg,
        )
        tan_half = math.tan(math.radians(face_fov_deg * 0.5))
        focal = cube_size / (2.0 * tan_half)

        # Row-vector convention: dir_face = dir_world @ R
        rot_world_to_face = np.stack([right, down, forward], axis=1).astype(np.float32)

        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, 0] = right
        c2w[:3, 1] = down
        c2w[:3, 2] = forward
        if cube_center is not None:
            c2w[:3, 3] = cube_center.astype(np.float32)

        face_meta[face_name] = {
            "name": face_name,
            "R": rot_world_to_face,
            "c2w": c2w,
            "Fx": focal,
            "Fy": focal,
            "Cx": cx,
            "Cy": cy,
            "W": cube_size,
            "H": cube_size,
            "fov_deg": face_fov_deg,
        }

    return face_meta

def _get_cubemap_center(datapack):
    if getattr(datapack, "suggested_bounding", None) is not None:
        return np.asarray(datapack.suggested_bounding, dtype=np.float32).mean(axis=0)

    views = datapack.get_train_cameras()
    cam_centers = []
    for view in views:
        cam_centers.append(view.camera_center.detach().cpu().numpy())
    return np.stack(cam_centers, axis=0).mean(axis=0).astype(np.float32)


@torch.no_grad()
def fuse_instance_ids_to_voxels(datapack, voxel_model, args, bg_id=255):
    """
    Fuse per-view 2D instance-id masks into per-voxel labels.

    Returns:
        voxel_labels: (num_voxels,) long tensor, class index per voxel
        id_to_class: dict, raw instance id -> fused class index
        class_to_id: dict, fused class index -> raw instance id
    """
    views = datapack.get_train_cameras()

    source_path = cfg.data.source_path
    if args.mask_dir:
        mask_dir = args.mask_dir
    else:
        # Support both naming conventions: Replica uses "instance_masks",
        # ScanNetpp uses "instance_mask".
        for candidate in ("instance_masks", "instance_mask"):
            mask_dir = os.path.join(source_path, candidate)
            if os.path.isdir(mask_dir):
                break
    if not os.path.isdir(mask_dir):
        raise FileNotFoundError(
            f"Instance mask directory not found under {source_path}\n"
            f"Expected 'instance_masks' or 'instance_mask', or specify --mask_dir"
        )

    print(f"[Instance Fusion] Using instance-id masks from: {mask_dir}")

    mask_cache = {}
    unique_ids = {bg_id}

    print("[Instance Fusion] Scanning instance ids...")
    for view in tqdm(views, desc="Scan masks"):
        H = int(view.image_height)
        W = int(view.image_width)
        mask_np = load_instance_id_mask(mask_dir, view.image_name, H, W, bg_id=bg_id)
        mask_cache[view.image_name] = mask_np
        unique_ids.update(np.unique(mask_np).tolist())

    all_ids = sorted(unique_ids)
    object_ids = [uid for uid in all_ids if uid != bg_id]

    id_to_class = {bg_id: 0}
    for i, oid in enumerate(object_ids):
        id_to_class[oid] = i + 1
    class_to_id = {v: k for k, v in id_to_class.items()}
    num_classes = len(id_to_class)

    max_id_val = max(all_ids) if all_ids else bg_id
    lut = torch.zeros(max_id_val + 1, dtype=torch.int64, device="cuda")
    for uid, cidx in id_to_class.items():
        if 0 <= uid <= max_id_val:
            lut[uid] = cidx

    finest_vox_size = level_2_vox_size(
        voxel_model.scene_extent,
        voxel_model.octlevel.max()
    ).item()

    fuser = Fuser(
        xyz=voxel_model.vox_center,
        bandwidth=50 * finest_vox_size,
        use_trunc=False,
        fuse_tsdf=False,
        feat_dim=num_classes,
        crop_border=0.0,
        normal_weight=False,
        depth_weight=False,
        border_weight=False,
        use_half=True,
    )

    print("[Instance Fusion] Fusing 2D masks into voxel labels...")
    for view in tqdm(views, desc="Fuse instance ids"):
        mask_np = mask_cache[view.image_name]
        mask_tensor = torch.from_numpy(mask_np).long().cuda()

        class_mask = lut[mask_tensor.clamp(0, max_id_val)]
        probs = torch.nn.functional.one_hot(class_mask, num_classes=num_classes)
        probs = probs.permute(2, 0, 1).float()

        render_pkg = voxel_model.render(view, output_depth=True)
        depth = render_pkg["depth"][2]

        fuser.integrate(cam=view, feat=probs, depth=depth)

    feature_vol = fuser.feature.nan_to_num_(0)
    voxel_labels = feature_vol.argmax(dim=1).long()

    print(
        f"[Instance Fusion] Done. "
        f"num_classes={num_classes} "
        f"(background=1, objects={num_classes - 1})"
    )

    return voxel_labels, id_to_class, class_to_id

@torch.no_grad()
def render_cube_object_masks_from_voxel_labels(
    datapack,
    voxel_model,
    voxel_labels,
    args,
    cube_size=1024,
    cube_side_fov_deg=100.0,
    cube_z_fov_deg=120.0,
    alpha_thresh=1e-3,
    side_pitch_deg=0.0,
):
    """
    Render cube-face object masks from fused voxel labels.

    Returns:
        object_masks: dict[face_name] -> HxW uint8, 255 means object region
        debug_label_rgbs: dict[face_name] -> HxWx3 uint8
    """
    cube_center = _get_cubemap_center(datapack)
    face_meta = _build_cube_face_metadata(
        cube_size=cube_size,
        cube_side_fov_deg=cube_side_fov_deg,
        cube_z_fov_deg=cube_z_fov_deg,
        cube_center=cube_center,
        side_pitch_deg=side_pitch_deg,
    )

    ori_sh0 = voxel_model.sh0.data.clone()
    ori_shs = voxel_model.shs.data.clone()

    try:
        # Background -> white, objects -> black
        voxel_colors = torch.ones((voxel_model.num_voxels, 3), dtype=torch.float32, device="cuda")
        voxel_colors[voxel_labels > 0] = 0.0

        voxel_model.sh0.data = activation_utils.rgb2shzero(voxel_colors)
        voxel_model.shs.data.fill_(0.0)

        object_masks = {}
        debug_label_rgbs = {}

        for face_name in FACE_ORDER:
            face = face_meta[face_name]
            cube_cam = MiniCam(
                c2w=face["c2w"],
                fovx=np.deg2rad(face["fov_deg"]),
                fovy=np.deg2rad(face["fov_deg"]),
                width=cube_size,
                height=cube_size,
                near=0.02,
                cx_p=0.5,
                cy_p=0.5,
            )

            render_pkg = voxel_model.render(
                cube_cam,
                track_max_w=False,
                output_T=True,
            )

            label_rgb = render_pkg["color"].permute(1, 2, 0).detach().cpu().numpy()
            label_rgb = np.clip(label_rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)

            trans = render_pkg["T"].squeeze(0).detach().cpu().numpy()
            alpha = 1.0 - np.clip(trans, 0.0, 1.0)

            # Only classify visible rendered regions.
            visible = alpha >= alpha_thresh

            # Background is rendered as white; objects as black.
            # Use a small tolerance for numerical/resampling noise.
            bg_dist = np.linalg.norm(label_rgb.astype(np.float32) - 255.0, axis=-1)
            object_mask = visible & (bg_dist > 5.0)

            object_masks[face_name] = object_mask.astype(np.uint8) * 255
            debug_label_rgbs[face_name] = label_rgb

        debug_root = os.path.join(args.output, "cubemap_inpaint", "label_vis")
        os.makedirs(debug_root, exist_ok=True)
        for face_name in FACE_ORDER:
            Image.fromarray(debug_label_rgbs[face_name]).save(
                os.path.join(debug_root, f"{face_name}.png")
            )

        print(f"[Cube Label] Saved cube label visualization to: {debug_root}")
        return object_masks, debug_label_rgbs

    finally:
        voxel_model.sh0.data = ori_sh0
        voxel_model.shs.data = ori_shs

def _render_tensor_to_uint8(image_tensor):
    image = image_tensor.detach().cpu().permute(1, 2, 0).numpy()
    return np.clip(image * 255.0 + 0.5, 0, 255).astype(np.uint8)


def _select_evenly_spaced_indices(num_total, num_select):

    num_select = min(num_total, num_select)

    # Sample the center of each interval to cover the full sequence.
    edges = np.linspace(0.0, float(num_total), num_select + 1, dtype=np.float32)
    centers = 0.5 * (edges[:-1] + edges[1:])
    indices = np.floor(centers).astype(np.int32)
    indices = np.clip(indices, 0, num_total - 1)

    # Keep order and uniqueness.
    out = []
    used = set()
    for idx in indices.tolist():
        if idx not in used:
            out.append(idx)
            used.add(idx)

    # Fill any missing slots if rounding created duplicates.
    if len(out) < num_select:
        for idx in range(num_total):
            if idx not in used:
                out.append(idx)
                used.add(idx)
            if len(out) == num_select:
                break

    return out


def build_selected_training_minicams(
    datapack,
    num_views=50,
    img_w=512,
    img_h=512,
    fov_deg=60.0,
    near=0.02,
):
    train_views = datapack.get_train_cameras()
    selected_indices = _select_evenly_spaced_indices(len(train_views), num_views)

    fov_rad = np.deg2rad(float(fov_deg))
    camera_specs = []

    for rank, view_idx in enumerate(selected_indices):
        src_view = train_views[view_idx]
        c2w = src_view.c2w.detach().cpu().numpy().astype(np.float32)

        cam = MiniCam(
            c2w=c2w,
            fovx=fov_rad,
            fovy=fov_rad,
            width=img_w,
            height=img_h,
            near=near,
            cx_p=0.5,
            cy_p=0.5,
        )

        cam.image_name = f"{rank:03d}"

        camera_specs.append({
            "camera": cam,
            "camera_name": cam.image_name,
            "source_view_index": int(view_idx),
            "source_image_name": src_view.image_name,
            "c2w": c2w,
            "fov_deg": float(fov_deg),
            "fovx": float(fov_rad),
            "fovy": float(fov_rad),
            "width": int(img_w),
            "height": int(img_h),
            "near": float(near),
            "cx_p": 0.5,
            "cy_p": 0.5,
        })

    return camera_specs


def _lookat_c2w_from_position_and_target(position_np: np.ndarray, lookat_np: np.ndarray, up_np: np.ndarray) -> np.ndarray:
    """Build 4x4 c2w using the same view basis as G4Splat cam_utils.generate_see3d_camera_by_lookat_object_centric."""
    # lookdir = position_np.astype(np.float64) - lookat_np.astype(np.float64)
    lookdir = lookat_np.astype(np.float64) - position_np.astype(np.float64)
    vec2 = _normalize_np(lookdir.reshape(1, -1)).reshape(-1).astype(np.float32)
    vec1 = _normalize_np(up_np.reshape(1, -1)).reshape(-1).astype(np.float32)
    vec0 = _normalize_np(np.cross(vec1, vec2)).astype(np.float32)
    vec1 = _normalize_np(np.cross(vec2, vec0)).astype(np.float32)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = vec0
    c2w[:3, 1] = vec1
    c2w[:3, 2] = vec2
    c2w[:3, 3] = position_np.astype(np.float32)
    return c2w


def get_traj_see3d_views(
    datapack,
    num_views=40,
    img_w=512,
    img_h=512,
    fov_deg=60.0,
    near=0.02,
    name_offset=0,
    traj_center=None,
):
    """
    Extra See3D cameras on a circular ego trajectory around the scene extent (G4Splat object-centric style).
    """
    if num_views <= 0:
        return []

    train_views = datapack.get_train_cameras()
    if len(train_views) == 0:
        return []

    train_centers = torch.stack([v.camera_center for v in train_views], dim=0).float()
    device = train_centers.device

    half_extent = torch.stack([
        (train_centers[:, 0].max() - train_centers[:, 0].min()).clamp(min=1e-6) * 0.5,
        (train_centers[:, 1].max() - train_centers[:, 1].min()).clamp(min=1e-6) * 0.5,
        (train_centers[:, 2].max() - train_centers[:, 2].min()).clamp(min=1e-6) * 0.5,
    ])

    if traj_center is None:
        traj_center_t = train_centers.mean(dim=0)
    else:
        traj_center_t = torch.as_tensor(traj_center, dtype=torch.float32, device=device).reshape(3)

    x_range_scale = (0.9, 1.1)
    y_range_scale = (0.9, 1.1)
    z_range_scale = (0.9, 1.1)

    n_plus = num_views + 1
    theta = torch.linspace(0, 2.0 * math.pi, n_plus, device=device)
    rx = torch.rand(n_plus, device=device) * (x_range_scale[1] - x_range_scale[0]) + x_range_scale[0]
    ry = torch.rand(n_plus, device=device) * (y_range_scale[1] - y_range_scale[0]) + y_range_scale[0]
    rz = torch.rand(n_plus, device=device) * (z_range_scale[1] - z_range_scale[0]) + z_range_scale[0]

    novel_cam_centers = torch.stack([
        rx * half_extent[0] * torch.cos(theta) + traj_center_t[0],
        ry * half_extent[1] * torch.sin(theta) + traj_center_t[1],
        rz * half_extent[2] * torch.cos(theta) + traj_center_t[2],
    ], dim=-1)
    novel_cam_centers = novel_cam_centers[:-1]

    max_z = train_centers[:, 2].max()
    novel_cam_centers[:, 2] = max_z

    lookat_points = traj_center_t.unsqueeze(0).expand_as(novel_cam_centers).clone()
    min_z = train_centers[:, 2].min()
    lookat_points[:, 2] = min_z

    up = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    fov_rad = np.deg2rad(float(fov_deg))
    traj_specs = []

    novel_np = novel_cam_centers.detach().cpu().numpy()
    look_np = lookat_points.detach().cpu().numpy()

    for i in range(novel_np.shape[0]):
        c2w = _lookat_c2w_from_position_and_target(novel_np[i], look_np[i], up)
        cam = MiniCam(
            c2w=c2w,
            fovx=fov_rad,
            fovy=fov_rad,
            width=img_w,
            height=img_h,
            near=near,
            cx_p=0.5,
            cy_p=0.5,
        )
        rank = name_offset + i
        cam.image_name = f"{rank:03d}"
        traj_specs.append({
            "camera": cam,
            "camera_name": cam.image_name,
            "source_view_index": -1,
            "source_image_name": "lookat_traj",
            "c2w": c2w,
            "fov_deg": float(fov_deg),
            "fovx": float(fov_rad),
            "fovy": float(fov_rad),
            "width": int(img_w),
            "height": int(img_h),
            "near": float(near),
            "cx_p": 0.5,
            "cy_p": 0.5,
        })

    print(f"[See3D] Added {len(traj_specs)} look-at trajectory cameras (names from {name_offset:03d}).")
    return traj_specs


def save_selected_camera_params(camera_specs, save_path):
    payload = {
        "camera_model": "MiniCam",
        "count": len(camera_specs),
        "frames": [],
    }

    for spec in camera_specs:
        payload["frames"].append({
            "camera_name": spec["camera_name"],
            "source_view_index": spec["source_view_index"],
            "source_image_name": spec["source_image_name"],
            "width": spec["width"],
            "height": spec["height"],
            "near": spec["near"],
            "cx_p": spec["cx_p"],
            "cy_p": spec["cy_p"],
            "fov_deg": spec["fov_deg"],
            "fovx": spec["fovx"],
            "fovy": spec["fovy"],
            "c2w": spec["c2w"].tolist(),
        })

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _build_semantic_palette(num_classes):
    rng = np.random.default_rng(42)
    palette = rng.integers(0, 256, size=(num_classes, 3), dtype=np.uint8)
    palette[0] = np.array([255, 255, 255], dtype=np.uint8)
    return palette


@torch.no_grad()
def render_selected_training_views_for_see3d(
    datapack,
    voxel_model,
    voxel_labels,
    class_to_id,
    output_root,
    num_views=50,
    traj_num_views=40,
    img_w=512,
    img_h=512,
    fov_deg=60.0,
    near=0.02,
    alpha_thresh=1e-3,
    dilate_object_mask=0,
):
    """
    Render selected training-aligned guidance views for See3D.

    Outputs:
        <output_root>/raw_rgb/*.png
        <output_root>/masked_raw_rgb/*.png
        <output_root>/inpaint_mask/*.png
        <output_root>/instance_id/*.png or *.npy
        <output_root>/instance_vis/*.png
        <output_root>/camera_params.json
    """
    os.makedirs(output_root, exist_ok=True)

    raw_rgb_dir = os.path.join(output_root, "raw_rgb")
    raw_depth_dir = os.path.join(output_root, "raw_depth")
    warp_images_dir = os.path.join(output_root, "warp_images")
    instance_id_dir = os.path.join(output_root, "instance_id")
    instance_vis_dir = os.path.join(output_root, "instance_vis")

    os.makedirs(raw_rgb_dir, exist_ok=True)
    os.makedirs(raw_depth_dir, exist_ok=True)
    os.makedirs(warp_images_dir, exist_ok=True)
    os.makedirs(instance_id_dir, exist_ok=True)
    os.makedirs(instance_vis_dir, exist_ok=True)

    train_camera_specs = build_selected_training_minicams(
        datapack=datapack,
        num_views=num_views,
        img_w=img_w,
        img_h=img_h,
        fov_deg=fov_deg,
        near=near,
    )

    traj_see3d_cam_specs = get_traj_see3d_views(
        datapack=datapack,
        num_views=traj_num_views,
        img_w=img_w,
        img_h=img_h,
        fov_deg=fov_deg,
        near=near,
        name_offset=len(train_camera_specs),
    )

    camera_specs = train_camera_specs + traj_see3d_cam_specs
    print(f'total see3d views: {len(camera_specs)}, include train views: {len(train_camera_specs)}, traj views: {len(traj_see3d_cam_specs)}')

    save_selected_camera_params(
        camera_specs,
        os.path.join(output_root, "camera_params.json"),
    )

    render_opt = {
        "track_max_w": False,
        "output_T": True,
        "output_depth": True,
    }

    empty_masks = {}
    raw_rgbs = {}

    print(f"[See3D] Rendering {len(camera_specs)} selected views for raw RGB...")
    for spec in tqdm(camera_specs, desc="Render raw RGB"):
        cam = spec["camera"]
        name = spec["camera_name"]

        render_pkg = voxel_model.render(cam, **render_opt)
        raw_rgb = _render_tensor_to_uint8(render_pkg["color"])

        trans = render_pkg["T"].squeeze(0).detach().cpu().numpy()
        alpha = 1.0 - np.clip(trans, 0.0, 1.0)
        empty_mask = (alpha < alpha_thresh)

        raw_rgbs[name] = raw_rgb
        empty_masks[name] = empty_mask
        Image.fromarray(raw_rgb).save(os.path.join(raw_rgb_dir, f"{name}.png"))

        render_depth = render_pkg["depth"][0].detach().cpu().numpy().astype(np.float32)
        save_img_f32(render_depth, os.path.join(raw_depth_dir, f"depth_frame{int(name):06d}.tiff"))
        save_depth_vis(render_depth, os.path.join(raw_depth_dir, f"depth_frame{int(name):06d}.png"))

    num_classes = int(max(class_to_id.keys())) + 1
    palette = _build_semantic_palette(num_classes)
    palette_f32 = torch.from_numpy(palette.astype(np.float32) / 255.0).cuda()

    class_to_raw = np.zeros(num_classes, dtype=np.int32)
    for cidx, raw_id in class_to_id.items():
        class_to_raw[int(cidx)] = int(raw_id)

    bg_raw_id = int(class_to_id[0])
    max_raw_id = int(class_to_raw.max())
    save_instance_as_u16_png = max_raw_id <= np.iinfo(np.uint16).max

    ori_sh0 = voxel_model.sh0.data.clone()
    ori_shs = voxel_model.shs.data.clone()

    try:
        voxel_colors = palette_f32[voxel_labels.long()]
        voxel_model.sh0.data = activation_utils.rgb2shzero(voxel_colors)
        voxel_model.shs.data.fill_(0.0)

        kernel = None
        if dilate_object_mask > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (dilate_object_mask * 2 + 1, dilate_object_mask * 2 + 1),
            )

        print(f"[See3D] Rendering instance maps and inpaint masks...")
        palette_f32_np = palette.astype(np.float32)

        for spec in tqdm(camera_specs, desc="Render labels"):
            cam = spec["camera"]
            name = spec["camera_name"]

            render_pkg = voxel_model.render(cam, **render_opt)
            label_rgb = _render_tensor_to_uint8(render_pkg["color"])

            trans = render_pkg["T"].squeeze(0).detach().cpu().numpy()
            alpha = 1.0 - np.clip(trans, 0.0, 1.0)
            visible = (alpha >= alpha_thresh)

            # Decode the rendered semantic color by nearest palette color.
            diff = label_rgb.astype(np.float32)[:, :, None, :] - palette_f32_np[None, None, :, :]
            dist2 = np.sum(diff * diff, axis=-1)
            class_map = dist2.argmin(axis=-1).astype(np.int32)

            instance_id_map = np.full((img_h, img_w), bg_raw_id, dtype=np.int32)
            instance_id_map[visible] = class_to_raw[class_map[visible]]

            object_mask = visible & (class_map > 0)

            # Internal convention: hole_mask == 255 means region to inpaint.
            hole_mask = np.logical_or(empty_masks[name], object_mask).astype(np.uint8) * 255

            if kernel is not None:
                hole_mask = cv2.dilate(hole_mask, kernel, iterations=1)

            # See3D mask convention:
            #   black (0)   = hole to inpaint
            #   white (255) = trusted / known region
            see3d_inpaint_mask = np.where(hole_mask > 0, 0, 255).astype(np.uint8)

            masked_raw_rgb = raw_rgbs[name].copy()
            masked_raw_rgb[hole_mask > 0] = 0

            Image.fromarray(label_rgb).save(os.path.join(instance_vis_dir, f"{name}.png"))
            Image.fromarray(see3d_inpaint_mask).save(os.path.join(warp_images_dir, f"mask_frame{int(name):06d}.png"))
            Image.fromarray(masked_raw_rgb).save(os.path.join(warp_images_dir, f"warp_frame{int(name):06d}.png"))

            if save_instance_as_u16_png:
                Image.fromarray(instance_id_map.astype(np.uint16)).save(
                    os.path.join(instance_id_dir, f"{name}.png")
                )
            else:
                np.save(
                    os.path.join(instance_id_dir, f"{name}.npy"),
                    instance_id_map.astype(np.int32),
                )

    finally:
        voxel_model.sh0.data = ori_sh0
        voxel_model.shs.data = ori_shs

    print(f"[See3D] Saved selected-view guidance set to: {output_root}")
    return camera_specs

@torch.no_grad()
def project_views_to_redundant_cubemap(
    datapack,
    voxel_model,
    voxel_labels,
    args,
    cube_size=1024,
    cube_side_fov_deg=100.0,
    cube_z_fov_deg=120.0,
    alpha_thresh=1e-3,
    side_pitch_deg=0.0,
):
    """
    Step 1:
    Render 6 cube faces from the pretrained voxel model and build inpaint masks
    from both empty regions and object-labeled voxels.

    Returns:
        cube_rgbs:  dict[face_name] -> HxWx3 uint8
        cube_masks: dict[face_name] -> HxW uint8, 255 means hole to inpaint
        coverage:   dict[face_name] -> stats
    """
    cube_center = _get_cubemap_center(datapack)
    face_meta = _build_cube_face_metadata(
        cube_size=cube_size,
        cube_side_fov_deg=cube_side_fov_deg,
        cube_z_fov_deg=cube_z_fov_deg,
        cube_center=cube_center,
        side_pitch_deg=side_pitch_deg,
    )

    cube_rgbs = {}
    cube_masks = {}
    coverage = {}
    empty_masks = {}

    render_opt = {
        "track_max_w": False,
        "output_depth": False,
        "output_normal": False,
        "output_T": True,
    }

    for face_name in FACE_ORDER:
        face = face_meta[face_name]

        cube_cam = MiniCam(
            c2w=face["c2w"],
            fovx=np.deg2rad(face["fov_deg"]),
            fovy=np.deg2rad(face["fov_deg"]),
            width=cube_size,
            height=cube_size,
            near=0.02,
            cx_p=0.5,
            cy_p=0.5,
        )

        render_pkg = voxel_model.render(cube_cam, **render_opt)

        color = render_pkg["color"].permute(1, 2, 0).detach().cpu().numpy()
        color = np.clip(color * 255.0 + 0.5, 0, 255).astype(np.uint8)

        trans = render_pkg["T"].squeeze(0).detach().cpu().numpy()
        alpha = 1.0 - np.clip(trans, 0.0, 1.0)

        empty_mask = (alpha < alpha_thresh)
        empty_masks[face_name] = empty_mask

        cube_rgbs[face_name] = color

    object_masks, debug_label_rgbs = render_cube_object_masks_from_voxel_labels(
        datapack=datapack,
        voxel_model=voxel_model,
        voxel_labels=voxel_labels,
        args=args,
        cube_size=cube_size,
        cube_side_fov_deg=cube_side_fov_deg,
        cube_z_fov_deg=cube_z_fov_deg,
        alpha_thresh=alpha_thresh,
        side_pitch_deg=side_pitch_deg,
    )

    dilate_px = args.dilate_object_mask
    kernel = None
    if dilate_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (dilate_px * 2 + 1, dilate_px * 2 + 1),
        )
        print(f"[Cube Render] Dilating cube mask by {dilate_px} pixels")

    for face_name in FACE_ORDER:
        empty_mask = empty_masks[face_name]
        object_mask = object_masks[face_name] > 0

        # Final hole mask: unseen region OR object region
        hole_mask = np.logical_or(empty_mask, object_mask).astype(np.uint8) * 255

        # Dilate the final cube mask to keep trusted pixels conservative.
        if kernel is not None:
            hole_mask = cv2.dilate(hole_mask, kernel, iterations=1)

        cube_masks[face_name] = hole_mask

        coverage[face_name] = {
            "empty_ratio": float(empty_mask.mean()),
            "object_ratio": float(object_mask.mean()),
            "hole_ratio": float((hole_mask > 0).mean()),
            "fov_deg": float(face_meta[face_name]["fov_deg"]),
        }

        print(
            f"[Cube Render] {face_name}: "
            f"fov_deg={coverage[face_name]['fov_deg']:.1f}, "
            f"empty_ratio={coverage[face_name]['empty_ratio']:.3f}, "
            f"object_ratio={coverage[face_name]['object_ratio']:.3f}, "
            f"hole_ratio={coverage[face_name]['hole_ratio']:.3f}"
        )

    cube_root = os.path.join(args.output, "cubemap_inpaint")
    rgb_dir = os.path.join(cube_root, "rgb")
    mask_save_dir = os.path.join(cube_root, "mask")
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(mask_save_dir, exist_ok=True)

    for face_name in FACE_ORDER:
        Image.fromarray(cube_rgbs[face_name]).save(os.path.join(rgb_dir, f"{face_name}.png"))
        Image.fromarray(cube_masks[face_name]).save(os.path.join(mask_save_dir, f"{face_name}.png"))

    print(f"[Cube Render] Saved rendered cube RGBs to: {rgb_dir}")
    print(f"[Cube Render] Saved rendered cube masks to: {mask_save_dir}")

    return True

def inpaint_cubemap_faces(work_dir, retry_times=10):
    """
    Step 2:
    Run single-view inpainting on the 6 cube faces sequentially.

    Returns:
        cube_inpainted: dict[face_name] -> HxWx3 uint8
    """

    cube_root = os.path.join(work_dir, "cubemap_inpaint")
    rgb_dir = os.path.join(cube_root, "rgb")
    mask_dir = os.path.join(cube_root, "mask")
    out_dir = os.path.join(cube_root, "out")
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    runner_script = 'decomvoxel/representation/GeoSVR/bg_inpaint/PixelHacker/infer_pixelhacker.py'
    inpaint_cmd = f'python {runner_script} --image_dir {rgb_dir} --mask_dir {mask_dir} --output_dir {out_dir} --retry_times {retry_times}'
    print("[Cube Inpaint]", inpaint_cmd)
    os.system(inpaint_cmd)

    return True

def _resolve_instance_mask_path(mask_dir: str, image_name: str):
    basename = os.path.splitext(image_name)[0]

    for suffix in ["_rgb", "_color", "_image"]:
        if basename.endswith(suffix):
            basename = basename[:-len(suffix)]
            break

    candidates = [basename, image_name.split("_")[0]]
    exts = [".png", ".jpg", ".jpeg", ".tif", ".bmp"]

    for stem in candidates:
        for ext in exts:
            path = os.path.join(mask_dir, stem + ext)
            if os.path.exists(path):
                return path
    return None

def load_instance_id_mask(mask_dir: str, image_name: str, H: int, W: int, bg_id: int) -> np.ndarray:
    """
    Load an instance-id mask.
    Returns:
        mask_img: (H, W) uint16/uint8 array, each pixel is an instance id.
    """
    mask_path = _resolve_instance_mask_path(mask_dir, image_name)
    if mask_path is None:
        print(f"[WARNING] Instance-id mask not found for {image_name}, using bg_id={bg_id}")
        return np.full((H, W), bg_id, dtype=np.int32)

    mask_img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if mask_img is None:
        print(f"[WARNING] Failed to read mask: {mask_path}, using bg_id={bg_id}")
        return np.full((H, W), bg_id, dtype=np.int32)

    if mask_img.ndim == 3:
        mask_img = mask_img[:, :, 0]

    if mask_img.shape[0] != H or mask_img.shape[1] != W:
        mask_img = cv2.resize(mask_img, (W, H), interpolation=cv2.INTER_NEAREST)

    return mask_img.astype(np.int32)

def load_instance_mask(mask_dir: str, image_name: str, H: int, W: int) -> np.ndarray:
    """
    Load an instance mask and return a boolean array where True = background.

    Args:
        mask_dir: directory containing instance mask PNGs
        image_name: stem name of the image (e.g. "000000")
        H, W: expected height and width of the rendered depth

    Returns:
        bg_mask: (H, W) bool array, True for background pixels
    """
    image_uid = image_name.split("_")[0]
    mask_path = os.path.join(mask_dir, f"{image_uid}.png")
    if not os.path.isfile(mask_path):
        print(f"[WARNING] Instance mask not found: {mask_path}, treating entire frame as background")
        return np.ones((H, W), dtype=bool)

    mask_img = np.array(Image.open(mask_path))
    # If RGB, take the first channel (all channels are identical)
    if mask_img.ndim == 3:
        mask_img = mask_img[:, :, 0]

    # Resize if needed (mask resolution might differ from rendered resolution)
    if mask_img.shape[0] != H or mask_img.shape[1] != W:
        mask_pil = Image.fromarray(mask_img)
        mask_pil = mask_pil.resize((W, H), Image.NEAREST)
        mask_img = np.array(mask_pil)

    # 255 = background, everything else = object
    bg_mask = (mask_img == 255)
    return bg_mask


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Background-only TSDF mesh extraction using instance masks."
    )
    parser.add_argument("--checkpoint", required=True, type=str,
                        help="Path to the GeoSVR voxel checkpoint .pt file")
    parser.add_argument("--config", required=True, type=str,
                        help="Path to the GeoSVR config.yaml file")
    parser.add_argument("--output", required=True, type=str,
                        help="Output directory for the extracted mesh")
    parser.add_argument("--clear_res_down", action="store_true")
    parser.add_argument("--overwrite_ss", default=None, type=float)
    parser.add_argument("--overwrite_vox_geo_mode", default=None)

    # Cube map parameters
    parser.add_argument("--cube_size", default=1024, type=int)
    parser.add_argument("--cube_side_fov_deg", default=100.0, type=float)
    parser.add_argument("--cube_z_fov_deg", default=120.0, type=float)
    parser.add_argument("--cube_alpha_thresh", default=1e-3, type=float)
    parser.add_argument(
        "--cube_side_pitch_deg",
        default=30.0,
        type=float,
        help="Pitch +X/-X/+Y/-Y cube faces downward by this many degrees to cover more floor",
    )

    # Background mask parameters
    parser.add_argument("--bg_id", default=255, type=int,
                        help="Background id in instance-id masks")
    parser.add_argument("--mask_dir", default=None, type=str,
                        help="Path to instance mask directory. "
                             "Default: <source_path>/instance_masks/")
    parser.add_argument("--dilate_object_mask", default=0, type=int,
                        help="Dilate object mask by N pixels to remove boundary "
                             "artifacts (default: 0, no dilation)")
    parser.add_argument("--retry_times", default=10, type=int,
                        help="Retry times for inpainting (default: 10)")

    # See3D parameters
    parser.add_argument("--train_num_views", default=40, type=int,
                        help="Number of views to render for See3D (default: 30)")
    parser.add_argument("--traj_num_views", default=40, type=int,
                        help="Number of views to render for trajectory (default: 40)")
    parser.add_argument("--see3d_fov_deg", default=60.0, type=float,
                        help="Field of view for See3D (default: 60.0)")

    args = parser.parse_args()
    print("=" * 60)
    print("BG Inpaint")
    print("=" * 60)
    print(f"Config:     {args.config}")
    print(f"Output:     {args.output}")

    # Load config
    update_config(args.config)

    if args.clear_res_down:
        cfg.data.res_downscale = 0
        cfg.data.res_width = 0

    # Load data
    data_pack = DataPack(cfg.data, cfg.model.white_background)
    print("[BG Inpaint] Loaded data pack")

    # Load pretrained voxel model
    cfg.model.model_path = os.path.dirname(os.path.dirname(args.checkpoint))

    voxel_model = SparseVoxelModel(cfg.model)
    voxel_model.load(args.checkpoint)

    if args.overwrite_ss:
        voxel_model.ss = args.overwrite_ss

    if args.overwrite_vox_geo_mode:
        voxel_model.vox_geo_mode = args.overwrite_vox_geo_mode

    voxel_model.freeze_vox_geo()
    print("[BG Inpaint] Loaded pretrained voxel model")

    voxel_labels, id_to_class, class_to_id = fuse_instance_ids_to_voxels(
        datapack=data_pack,
        voxel_model=voxel_model,
        args=args,
        bg_id=args.bg_id,
    )
    print("[BG Inpaint] Fused 2D instance ids into voxel labels")

    render_selected_training_views_for_see3d(
        datapack=data_pack,
        voxel_model=voxel_model,
        voxel_labels=voxel_labels,
        class_to_id=class_to_id,
        output_root=os.path.join(args.output, "see3d_guidance_views"),
        num_views=args.train_num_views,
        traj_num_views=args.traj_num_views,
        img_w=512,
        img_h=512,
        fov_deg=args.see3d_fov_deg,
        near=0.02,
        alpha_thresh=args.cube_alpha_thresh,
        dilate_object_mask=args.dilate_object_mask,
    )
    print("[BG Inpaint] Rendered selected guidance views for See3D")

    project_views_to_redundant_cubemap(
        datapack=data_pack,
        voxel_model=voxel_model,
        voxel_labels=voxel_labels,
        args=args,
        cube_size=args.cube_size,
        cube_side_fov_deg=args.cube_side_fov_deg,
        cube_z_fov_deg=args.cube_z_fov_deg,
        alpha_thresh=args.cube_alpha_thresh,
        side_pitch_deg=args.cube_side_pitch_deg,
    )
    print("[BG Inpaint] Projected views to redundant cubemap")

    inpaint_cubemap_faces(
        work_dir=args.output,
        retry_times=args.retry_times,
    )
    print("[BG Inpaint] Inpainted cubemap faces")

    print("=" * 60)
    print("Done!")
    print("=" * 60)
