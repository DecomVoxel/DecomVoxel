import json
import os
import sys
from argparse import ArgumentParser

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import colorsys
import trimesh

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
GEOSVR_ROOT = os.path.dirname(PROJECT_ROOT)
if GEOSVR_ROOT not in sys.path:
    sys.path.insert(0, GEOSVR_ROOT)

from cam_util import load_selected_camera_params
from src.config import cfg, update_config
from src.sparse_voxel_model import SparseVoxelModel
from src.utils.fuser_utils import Fuser
from src.utils.octree_utils import level_2_vox_size
from src.utils.render_utils import save_img_f32


def normalize_np(x, eps=1e-8):
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(norm, eps, None)


def read_depth_tiff(depth_path, width=None, height=None):
    depth = np.asarray(Image.open(depth_path), dtype=np.float32)
    if width is not None and height is not None and depth.shape[:2] != (height, width):
        depth = np.array(
            Image.fromarray(depth).resize((width, height), Image.NEAREST),
            dtype=np.float32,
        )
    return depth


def read_mask_png(mask_path, width=None, height=None):
    mask = Image.open(mask_path).convert("L")
    if width is not None and height is not None and mask.size != (width, height):
        mask = mask.resize((width, height), Image.NEAREST)
    mask_np = np.asarray(mask, dtype=np.uint8) > 127
    return mask_np


def read_rgb_image(image_path, width=None, height=None):
    image = Image.open(image_path).convert("RGB")
    if width is not None and height is not None and image.size != (width, height):
        image = image.resize((width, height), Image.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def save_depth_vis(depth, save_path):
    if isinstance(depth, torch.Tensor):
        depth_np = depth.detach().cpu().numpy()
    else:
        depth_np = depth
    plt.imsave(save_path, depth_np, cmap="viridis")


def save_bool_mask(mask, save_path):
    Image.fromarray((mask.astype(np.uint8) * 255)).save(save_path)


def save_index_mask(label_map, save_path):
    max_val = int(np.max(label_map)) if label_map.size > 0 else 0
    if max_val <= 255:
        Image.fromarray(label_map.astype(np.uint8)).save(save_path)
    else:
        Image.fromarray(label_map.astype(np.uint16)).save(save_path)


def label_to_color(label):
    hashed = (int(label) * 2654435761) % (2 ** 32)
    hue = hashed / float(2 ** 32)
    sat = 0.75
    val = 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
    return np.array(
        [int(r * 255), int(g * 255), int(b * 255)],
        dtype=np.uint8,
    )


def overlay_label_map(image_rgb, label_map, alpha=0.55):
    image_rgb = np.asarray(image_rgb).astype(np.uint8)
    label_map = np.asarray(label_map)

    if label_map.size == 0:
        return image_rgb.copy()

    unique_labels = np.unique(label_map)
    unique_labels = unique_labels[unique_labels > 0]

    if unique_labels.size == 0:
        return image_rgb.copy()

    out = image_rgb.astype(np.float32).copy()
    for label in unique_labels.tolist():
        mask = label_map == label
        color = label_to_color(label).astype(np.float32)
        out[mask] = (1.0 - alpha) * out[mask] + alpha * color

    return np.clip(out, 0, 255).astype(np.uint8)


def overlay_binary_mask(image_rgb, mask, color, alpha=0.55):
    image_rgb = np.asarray(image_rgb).astype(np.uint8)
    mask = np.asarray(mask).astype(bool)

    out = image_rgb.astype(np.float32).copy()
    color = np.asarray(color, dtype=np.float32)
    out[mask] = (1.0 - alpha) * out[mask] + alpha * color
    return np.clip(out, 0, 255).astype(np.uint8)


def save_tensor_as_pcd(points, path, pcd_colors=None):
    points = np.asarray(points, dtype=np.float32)
    pcd = trimesh.PointCloud(points)

    if pcd_colors is not None:
        pcd_colors = np.asarray(pcd_colors)
        if pcd_colors.dtype != np.uint8:
            pcd_colors = np.clip(pcd_colors, 0, 255).astype(np.uint8)
        pcd.colors = pcd_colors

    pcd.export(path)

def create_unit_cube():
    vertices = np.array(
        [
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5],
        ],
        dtype=np.float32,
    )
    faces = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [2, 3, 7],
            [2, 7, 6],
            [0, 4, 7],
            [0, 7, 3],
            [1, 2, 6],
            [1, 6, 5],
        ],
        dtype=np.int32,
    )
    return vertices, faces

def voxels_to_mesh(centers, sizes, colors):
    centers = np.asarray(centers, dtype=np.float32)
    sizes = np.asarray(sizes, dtype=np.float32).reshape(-1)
    colors = np.asarray(colors, dtype=np.float32)
    colors = np.clip(colors, 0.0, 1.0)

    unit_verts, unit_faces = create_unit_cube()
    num_voxels = centers.shape[0]

    all_vertices = (
        sizes[:, None, None] * unit_verts[None, :, :] + centers[:, None, :]
    ).reshape(-1, 3).astype(np.float32)
    all_faces = (
        unit_faces[None, :, :] + (np.arange(num_voxels)[:, None, None] * 8)
    ).reshape(-1, 3).astype(np.int32)

    colors_rgba = np.zeros((num_voxels, 8, 4), dtype=np.uint8)
    colors_rgba[:, :, :3] = (colors[:, None, :] * 255).astype(np.uint8)
    colors_rgba[:, :, 3] = 255
    all_colors = colors_rgba.reshape(-1, 4)

    return trimesh.Trimesh(
        vertices=all_vertices,
        faces=all_faces,
        vertex_colors=all_colors,
        process=False,
    )

def generate_points_on_plane(
    plane_points,
    plane_normal,
    mean_plane_points=None,
    num_points=20000,
):
    plane_points = np.asarray(plane_points, dtype=np.float32)
    plane_normal = normalize_np(np.asarray(plane_normal, dtype=np.float32).reshape(1, 3))[0]

    if plane_points.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)

    if mean_plane_points is None:
        mean_plane_points = plane_points.mean(axis=0).astype(np.float32)
    else:
        mean_plane_points = np.asarray(mean_plane_points, dtype=np.float32)

    tangent = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if abs(float(plane_normal[0])) > 0.9:
        tangent = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    u = np.cross(plane_normal, tangent)
    u = normalize_np(u.reshape(1, 3))[0]
    v = np.cross(plane_normal, u)
    v = normalize_np(v.reshape(1, 3))[0]

    centered = plane_points - mean_plane_points[None]
    proj_u = centered @ u
    proj_v = centered @ v

    u_min, u_max = float(proj_u.min()), float(proj_u.max())
    v_min, v_max = float(proj_v.min()), float(proj_v.max())

    rng = np.random.default_rng(42)
    rand_u = rng.random(num_points).astype(np.float32) * (u_max - u_min) + u_min
    rand_v = rng.random(num_points).astype(np.float32) * (v_max - v_min) + v_min

    sampled = (
        mean_plane_points[None]
        + rand_u[:, None] * u[None]
        + rand_v[:, None] * v[None]
    )
    return sampled.astype(np.float32)


def fit_plane_svd(points, prior_normal=None):
    points = np.asarray(points, dtype=np.float64)
    center = points.mean(axis=0)
    centered = points - center

    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    normal = vt[-1]
    normal = normal / (np.linalg.norm(normal) + 1e-8)

    if prior_normal is not None:
        prior_normal = normalize_np(prior_normal.reshape(1, 3))[0]
        if np.dot(normal, prior_normal) < 0:
            normal = -normal

    d = -np.dot(normal, center)
    return normal.astype(np.float32), center.astype(np.float32), float(d)


def fit_plane_ransac_np(
    points,
    threshold=0.01,
    max_trials=512,
    prior_normal=None,
    min_inliers=64,
    seed=42,
):
    points = np.asarray(points, dtype=np.float64)
    if points.shape[0] < 3:
        normal, center, d = fit_plane_svd(points, prior_normal=prior_normal)
        return normal, center, d, np.ones(points.shape[0], dtype=bool)

    rng = np.random.default_rng(seed)
    best_inliers = None
    best_count = 0
    best_error = np.inf

    for _ in range(max_trials):
        sample_idx = rng.choice(points.shape[0], size=3, replace=False)
        sample = points[sample_idx]

        v1 = sample[1] - sample[0]
        v2 = sample[2] - sample[0]
        normal = np.cross(v1, v2)
        normal_norm = np.linalg.norm(normal)
        if normal_norm < 1e-8:
            continue

        normal = normal / normal_norm
        if prior_normal is not None and np.dot(normal, prior_normal) < 0:
            normal = -normal

        d = -np.dot(normal, sample[0])
        dist = np.abs(points @ normal + d)
        inliers = dist < threshold
        count = int(inliers.sum())

        if count < min_inliers and count < best_count:
            continue

        error = float(dist[inliers].mean()) if count > 0 else np.inf
        if count > best_count or (count == best_count and error < best_error):
            best_inliers = inliers
            best_count = count
            best_error = error

    if best_inliers is None or best_count < 3:
        normal, center, d = fit_plane_svd(points, prior_normal=prior_normal)
        return normal, center, d, np.ones(points.shape[0], dtype=bool)

    normal, center, d = fit_plane_svd(points[best_inliers], prior_normal=prior_normal)
    dist = np.abs(points @ normal + d)
    final_inliers = dist < threshold
    return normal, center, d, final_inliers


def oriented_average_normals(normals):
    if len(normals) == 0:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)

    normals = [normalize_np(np.asarray(n, dtype=np.float32).reshape(1, 3))[0] for n in normals]
    ref = normals[0]
    aligned = []
    for n in normals:
        if np.dot(n, ref) < 0:
            n = -n
        aligned.append(n)

    mean_n = np.mean(np.stack(aligned, axis=0), axis=0)
    mean_n = normalize_np(mean_n.reshape(1, 3))[0]
    return mean_n.astype(np.float32)


def average_plane_normal_from_mask(normal_map, plane_mask, trusted_mask=None):
    if trusted_mask is not None:
        use_mask = plane_mask & trusted_mask
        if int(use_mask.sum()) < 32:
            use_mask = plane_mask
    else:
        use_mask = plane_mask

    values = normal_map[use_mask]
    if values.size == 0:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)

    finite = np.isfinite(values).all(axis=-1)
    values = values[finite]
    if values.shape[0] == 0:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)

    norms = np.linalg.norm(values, axis=-1)
    values = values[norms > 1e-6]
    if values.shape[0] == 0:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)

    return oriented_average_normals(values)


def voxel_overlap_score(a, b):
    if a.size == 0 or b.size == 0:
        return 0.0, 0.0, 0.0

    inter = np.intersect1d(a, b, assume_unique=True).size
    if inter == 0:
        return 0.0, 0.0, 0.0

    score_a = inter / float(max(len(a), 1))
    score_b = inter / float(max(len(b), 1))
    score = max(score_a, score_b)
    return score, score_a, score_b


class SparseVoxelPlaneMerger:
    def __init__(
        self,
        voxel_model,
        guidance_root,
        dn_root,
        plane_root,
        output_root,
        trace_bandwidth_mul=50.0,
        trace_conf_thresh=0.35,
        voxel_iou_thresh=0.5,
        normal_angle_thresh=30.0,
        min_global_view_rate=0.02,
        plane_fit_thresh=0.01,
        plane_fit_trials=512,
        max_plane_fit_points=200000,
        min_voxels_per_plane=8,
        device="cuda",
    ):
        self.device = device
        self.voxel_model = voxel_model
        self.guidance_root = guidance_root
        self.dn_root = dn_root
        self.plane_root = plane_root
        self.output_root = output_root

        self.raw_depth_dir = os.path.join(guidance_root, "raw_depth")
        self.warp_images_dir = os.path.join(guidance_root, "warp_images")
        self.raw_rgb_dir = os.path.join(guidance_root, "raw_rgb")
        self.camera_json_path = os.path.join(guidance_root, "camera_params.json")

        self.trace_conf_thresh = float(trace_conf_thresh)
        self.voxel_iou_thresh = float(voxel_iou_thresh)
        self.normal_angle_thresh = float(normal_angle_thresh)
        self.min_global_view_rate = float(min_global_view_rate)
        self.plane_fit_thresh = float(plane_fit_thresh)
        self.plane_fit_trials = int(plane_fit_trials)
        self.max_plane_fit_points = int(max_plane_fit_points)
        self.min_voxels_per_plane = int(min_voxels_per_plane)

        finest_vox_size = level_2_vox_size(
            voxel_model.scene_extent,
            voxel_model.octlevel.max(),
        ).item()
        self.trace_bandwidth = float(trace_bandwidth_mul) * float(finest_vox_size)

        self.camera_specs = load_selected_camera_params(
            self.camera_json_path,
            keep_metadata=True,
        )
        self.cameras = [spec["camera"] for spec in self.camera_specs]
        self.num_views = len(self.cameras)

        self.frame_ids = []
        self.rgb_images = []
        self.raw_depth_maps = []
        self.base_depth_maps = []
        self.trusted_masks = []
        self.plane_mask_maps = []
        self.world_normal_maps = []
        self.local_plane_normals = {}
        self.local_planes = []
        self.local_plane2global_plane_idx = {}
        self.global_planes = {}
        self.global_plane_info = {}
        self._world_points_cache = {}
        self._view_plane_depth_cache = {}

        self._load_view_data()

    def _load_view_data(self):
        for view_idx, spec in enumerate(self.camera_specs):
            frame_idx = int(spec["camera_name"])
            width = int(spec["width"])
            height = int(spec["height"])

            rgb_path = os.path.join(self.dn_root, f"rgb_frame{frame_idx:06d}.png")
            if not os.path.exists(rgb_path):
                rgb_path = os.path.join(self.raw_rgb_dir, f"{spec['camera_name']}.png")

            raw_depth_path = os.path.join(self.raw_depth_dir, f"depth_frame{frame_idx:06d}.tiff")
            trusted_mask_path = os.path.join(self.warp_images_dir, f"mask_frame{frame_idx:06d}.png")
            plane_mask_path = os.path.join(self.plane_root, f"plane_mask_frame{frame_idx:06d}.npy")
            world_normal_path = os.path.join(self.dn_root, f"mono_normal_world_frame{frame_idx:06d}.npy")
            aligned_depth_path = os.path.join(self.dn_root, f"aligned_depth_frame{frame_idx:06d}.tiff")

            rgb = read_rgb_image(rgb_path, width=width, height=height)
            raw_depth = read_depth_tiff(raw_depth_path, width=width, height=height)
            trusted_mask = read_mask_png(trusted_mask_path, width=width, height=height)
            plane_mask = np.load(plane_mask_path).astype(np.int32)
            if plane_mask.shape != (height, width):
                plane_mask = np.array(
                    Image.fromarray(plane_mask).resize((width, height), Image.NEAREST),
                    dtype=np.int32,
                )
            world_normal = np.load(world_normal_path).astype(np.float32)
            if world_normal.shape[:2] != (height, width):
                world_normal = cv2.resize(world_normal, (width, height), interpolation=cv2.INTER_LINEAR)

            if os.path.exists(aligned_depth_path):
                base_depth = read_depth_tiff(aligned_depth_path, width=width, height=height)
            else:
                raise ValueError(f"Aligned depth not found: {aligned_depth_path}")

            self.frame_ids.append(frame_idx)
            self.rgb_images.append(rgb)
            self.raw_depth_maps.append(raw_depth)
            self.base_depth_maps.append(base_depth)
            self.trusted_masks.append(trusted_mask)
            self.plane_mask_maps.append(plane_mask)
            self.world_normal_maps.append(world_normal)

            max_plane_id = int(plane_mask.max())
            for plane_id in range(1, max_plane_id + 1):
                local_mask = plane_mask == plane_id
                avg_normal = average_plane_normal_from_mask(
                    world_normal,
                    local_mask,
                    trusted_mask=trusted_mask,
                )
                self.local_plane_normals[(view_idx, plane_id)] = avg_normal

    def _get_world_points_map(self, view_idx):
        if view_idx not in self._world_points_cache:
            cam = self.cameras[view_idx]
            depth_t = torch.from_numpy(self.raw_depth_maps[view_idx]).float().to(self.device)
            pts = cam.depth2pts(depth_t).permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)
            self._world_points_cache[view_idx] = pts
        return self._world_points_cache[view_idx]

    @torch.no_grad()
    def _trace_view_planes_to_voxels(self, view_idx):
        cam = self.cameras[view_idx]
        plane_mask = self.plane_mask_maps[view_idx]
        raw_depth = self.raw_depth_maps[view_idx]
        trusted_mask = self.trusted_masks[view_idx]

        known_mask = trusted_mask & (raw_depth > 0)
        num_planes = int(plane_mask.max())
        if num_planes <= 0:
            return {}

        class_mask = plane_mask.copy()
        class_mask[~known_mask] = 0

        depth_t = torch.from_numpy(raw_depth).float().to(self.device)
        alpha_t = torch.from_numpy(known_mask.astype(np.float32)).float().to(self.device)
        class_t = torch.from_numpy(class_mask.astype(np.int64)).long().to(self.device)

        feat = torch.nn.functional.one_hot(
            class_t,
            num_classes=num_planes + 1,
        ).permute(2, 0, 1).float()

        fuser = Fuser(
            xyz=self.voxel_model.vox_center,
            bandwidth=self.trace_bandwidth,
            use_trunc=False,
            fuse_tsdf=False,
            feat_dim=num_planes + 1,
            crop_border=0.0,
            normal_weight=False,
            depth_weight=False,
            border_weight=False,
            use_half=True,
        )
        fuser.integrate(
            cam=cam,
            feat=feat,
            depth=depth_t,
            alpha=alpha_t,
            front_only=True,            # NOTE: only integrate front-facing voxels
        )

        feature = fuser.feature.nan_to_num_(0).float()
        weight = fuser.weight.squeeze(-1).float()

        conf, owner = feature.max(dim=1)
        owner[(weight <= 0) | (conf < self.trace_conf_thresh)] = 0

        plane_to_voxels = {}
        for plane_id in range(1, num_planes + 1):
            voxel_idx = torch.where(owner == plane_id)[0].detach().cpu().numpy().astype(np.int32)
            if voxel_idx.size >= self.min_voxels_per_plane:
                plane_to_voxels[plane_id] = np.unique(voxel_idx)

        return plane_to_voxels

    def _save_traced_plane_voxels_for_view(self, view_idx, plane_to_voxels, max_save_views=5):
        if view_idx >= max_save_views:
            return
        if len(plane_to_voxels) == 0:
            return

        frame_idx = self.frame_ids[view_idx]
        view_dir = os.path.join(
            self.output_root,
            "debug_trace_voxels",
            f"view_{view_idx:03d}_frame{frame_idx:06d}",
        )
        os.makedirs(view_dir, exist_ok=True)

        voxel_device = self.voxel_model.vox_center.device
        summary = []

        for plane_id, voxel_ids in sorted(plane_to_voxels.items()):
            if voxel_ids.size == 0:
                continue

            voxel_idx_t = torch.from_numpy(voxel_ids.astype(np.int64)).to(voxel_device)

            centers = (
                self.voxel_model.vox_center[voxel_idx_t]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            sizes = (
                self.voxel_model.vox_size[voxel_idx_t]
                .detach()
                .cpu()
                .numpy()
                .reshape(-1)
                .astype(np.float32)
            )

            color_u8 = label_to_color(plane_id)
            color_f32 = color_u8.astype(np.float32) / 255.0
            point_colors = np.repeat(color_u8[None, :], centers.shape[0], axis=0)
            mesh_colors = np.repeat(color_f32[None, :], centers.shape[0], axis=0)

            stem = f"plane_{plane_id:03d}"

            np.save(os.path.join(view_dir, f"{stem}_voxel_ids.npy"), voxel_ids.astype(np.int32))
            np.save(os.path.join(view_dir, f"{stem}_voxel_centers.npy"), centers)
            np.save(os.path.join(view_dir, f"{stem}_voxel_sizes.npy"), sizes)

            save_tensor_as_pcd(
                centers,
                os.path.join(view_dir, f"{stem}_centers.ply"),
                pcd_colors=point_colors,
            )

            mesh = voxels_to_mesh(centers, sizes, mesh_colors)
            mesh.export(os.path.join(view_dir, f"{stem}.glb"))

            summary.append(
                {
                    "view_idx": int(view_idx),
                    "frame_idx": int(frame_idx),
                    "plane_id": int(plane_id),
                    "num_voxels": int(voxel_ids.size),
                    "source_view_index": int(self.camera_specs[view_idx]["source_view_index"]),
                    "source_image_name": self.camera_specs[view_idx]["source_image_name"],
                    "mesh_path": f"{stem}.glb",
                    "points_path": f"{stem}_centers.ply",
                }
            )

        with open(os.path.join(view_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    @torch.no_grad()
    def trace_local_planes(self):
        self.local_planes = []

        for view_idx in tqdm(range(self.num_views), desc="Trace local planes to voxels"):
            plane_to_voxels = self._trace_view_planes_to_voxels(view_idx)

            # # ###### NOTE: just for debug
            # self._save_traced_plane_voxels_for_view(
            #     view_idx=view_idx,
            #     plane_to_voxels=plane_to_voxels,
            #     max_save_views=5,
            # )

            plane_mask = self.plane_mask_maps[view_idx]
            known_mask = self.trusted_masks[view_idx] & (self.raw_depth_maps[view_idx] > 0)

            max_plane_id = int(plane_mask.max())
            for plane_id in range(1, max_plane_id + 1):
                local_mask = plane_mask == plane_id
                known_area = int((local_mask & known_mask).sum())
                full_area = int(local_mask.sum())
                voxel_ids = plane_to_voxels.get(plane_id, np.zeros((0,), dtype=np.int32))

                self.local_planes.append(
                    {
                        "node": (view_idx, plane_id),
                        "view_idx": view_idx,
                        "frame_idx": self.frame_ids[view_idx],
                        "plane_id": plane_id,
                        "source_view_index": self.camera_specs[view_idx]["source_view_index"],
                        "source_image_name": self.camera_specs[view_idx]["source_image_name"],
                        "normal": self.local_plane_normals[(view_idx, plane_id)],
                        "known_area": known_area,
                        "full_area": full_area,
                        "voxel_ids": voxel_ids,
                    }
                )

    def merge_planes(self):
        normal_thresh = float(np.cos(np.deg2rad(self.normal_angle_thresh)))
        temp_global_planes = []

        sortable_local_planes = sorted(
            self.local_planes,
            key=lambda item: (item["view_idx"], -len(item["voxel_ids"]), -item["known_area"]),
        )

        for local_plane in tqdm(sortable_local_planes, desc="Merge local planes"):
            voxel_ids = local_plane["voxel_ids"]
            if voxel_ids.size < self.min_voxels_per_plane:
                continue

            best_idx = -1
            best_score = -1.0

            for global_idx, global_plane in enumerate(temp_global_planes):
                if local_plane["view_idx"] in global_plane["view_set"]:
                    continue

                cos_sim = float(np.abs(np.dot(global_plane["normal"], local_plane["normal"])))
                if cos_sim < normal_thresh:
                    continue

                overlap_score, _, _ = voxel_overlap_score(global_plane["voxel_ids"], voxel_ids)
                if overlap_score < self.voxel_iou_thresh:
                    continue

                if overlap_score > best_score:
                    best_score = overlap_score
                    best_idx = global_idx

            if best_idx >= 0:
                global_plane = temp_global_planes[best_idx]
                merged_normal = oriented_average_normals(
                    [global_plane["normal"], local_plane["normal"]]
                )
                global_plane["normal"] = merged_normal
                global_plane["voxel_ids"] = np.union1d(global_plane["voxel_ids"], voxel_ids)
                global_plane["nodes"].append(local_plane["node"])
                global_plane["view_set"].add(local_plane["view_idx"])
            else:
                temp_global_planes.append(
                    {
                        "normal": local_plane["normal"].copy(),
                        "voxel_ids": voxel_ids.copy(),
                        "nodes": [local_plane["node"]],
                        "view_set": {local_plane["view_idx"]},
                    }
                )

        merged_flags = [False] * len(temp_global_planes)
        final_global_planes = []

        for i in range(len(temp_global_planes)):
            if merged_flags[i]:
                continue

            curr = temp_global_planes[i]

            for j in range(i + 1, len(temp_global_planes)):
                if merged_flags[j]:
                    continue

                other = temp_global_planes[j]
                cos_sim = float(np.abs(np.dot(curr["normal"], other["normal"])))
                if cos_sim < normal_thresh:
                    continue

                overlap_score, _, _ = voxel_overlap_score(curr["voxel_ids"], other["voxel_ids"])
                if overlap_score < self.voxel_iou_thresh:
                    continue

                curr["normal"] = oriented_average_normals([curr["normal"], other["normal"]])
                curr["voxel_ids"] = np.union1d(curr["voxel_ids"], other["voxel_ids"])
                curr["nodes"].extend(other["nodes"])
                curr["view_set"] = curr["view_set"].union(other["view_set"])
                merged_flags[j] = True

            final_global_planes.append(curr)
            merged_flags[i] = True

        min_view_count = max(1, int(np.ceil(self.num_views * self.min_global_view_rate)))

        self.global_planes = {}
        self.local_plane2global_plane_idx = {}

        global_plane_idx = 1
        for global_plane in final_global_planes:
            if len(global_plane["view_set"]) < min_view_count:
                continue

            self.global_planes[global_plane_idx] = global_plane
            for node in global_plane["nodes"]:
                self.local_plane2global_plane_idx[node] = global_plane_idx
            global_plane_idx += 1

        print(f"[Merge] Kept {len(self.global_planes)} global planes")

    def _collect_local_plane_points(self, view_idx, plane_id):
        plane_mask = self.plane_mask_maps[view_idx] == plane_id
        known_mask = self.trusted_masks[view_idx] & (self.raw_depth_maps[view_idx] > 0)
        valid_mask = plane_mask & known_mask

        if int(valid_mask.sum()) == 0:
            return np.zeros((0, 3), dtype=np.float32)

        world_points_map = self._get_world_points_map(view_idx)
        points = world_points_map[valid_mask]
        finite = np.isfinite(points).all(axis=-1)
        points = points[finite]
        return points.astype(np.float32)

    def fit_global_planes(self):
        self.global_plane_info = {}

        for global_idx, global_plane in tqdm(self.global_planes.items(), desc="Fit global planes"):
            all_points = []
            prior_normals = []

            for node in global_plane["nodes"]:
                view_idx, plane_id = node
                points = self._collect_local_plane_points(view_idx, plane_id)
                if points.shape[0] == 0:
                    continue
                all_points.append(points)
                prior_normals.append(self.local_plane_normals[node])

            if len(all_points) == 0:
                continue

            all_points = np.concatenate(all_points, axis=0)
            if all_points.shape[0] > self.max_plane_fit_points:
                rng = np.random.default_rng(42 + global_idx)
                sample_idx = rng.choice(all_points.shape[0], size=self.max_plane_fit_points, replace=False)
                all_points = all_points[sample_idx]

            prior_normal = oriented_average_normals(prior_normals)
            normal, center, d, inlier_mask = fit_plane_ransac_np(
                all_points,
                threshold=self.plane_fit_thresh,
                max_trials=self.plane_fit_trials,
                prior_normal=prior_normal,
                min_inliers=64,
                seed=42 + global_idx,
            )

            if np.dot(normal, prior_normal) < 0:
                normal = -normal
                d = -d

            inlier_points = all_points[inlier_mask] if inlier_mask is not None else all_points
            if inlier_points.shape[0] == 0:
                inlier_points = all_points

            self.global_plane_info[global_idx] = {
                "normal": normal.astype(np.float32),
                "mean": center.astype(np.float32),
                "plane_params": np.concatenate([normal.astype(np.float32), np.array([d], dtype=np.float32)], axis=0),
                "corners": np.stack(
                    [
                        inlier_points.min(axis=0),
                        inlier_points.max(axis=0),
                    ],
                    axis=0,
                ).astype(np.float32),
                "num_nodes": len(global_plane["nodes"]),
                "num_points": int(inlier_points.shape[0]),
            }

        print(f"[Fit] Fitted {len(self.global_plane_info)} global planes")

    @torch.no_grad()
    def _compute_global_plane_depth_map(self, view_idx, global_idx):
        cache_key = (view_idx, global_idx)
        if cache_key in self._view_plane_depth_cache:
            return self._view_plane_depth_cache[cache_key]

        cam = self.cameras[view_idx]
        plane_params = self.global_plane_info[global_idx]["plane_params"]
        normal = torch.from_numpy(plane_params[:3]).float().to(self.device)
        d = float(plane_params[3])

        rd = cam.compute_rd()
        cam_origin = cam.position.view(3, 1, 1)

        denom = (normal.view(3, 1, 1) * rd).sum(dim=0)
        numer = -((normal.view(3, 1, 1) * cam_origin).sum(dim=0) + d)

        denom_safe = torch.where(
            denom.abs() < 1e-8,
            torch.ones_like(denom),
            denom,
        )
        depth = numer / denom_safe
        valid = (denom.abs() >= 1e-8) & (depth > 0)
        depth = depth.clamp_min(0)
        depth[~valid] = 0

        depth_np = depth.detach().cpu().numpy().astype(np.float32)
        self._view_plane_depth_cache[cache_key] = depth_np
        return depth_np

    def compute_refined_depth_maps(self):
        refined_depth_maps = [depth.copy() for depth in self.base_depth_maps]
        refined_conf_maps = [
            np.zeros_like(depth, dtype=bool) for depth in self.base_depth_maps
        ]
        global_plane_mask_maps = [
            np.zeros_like(mask, dtype=np.int32) for mask in self.plane_mask_maps
        ]

        for view_idx in tqdm(range(self.num_views), desc="Compute refined depth maps"):
            plane_mask_map = self.plane_mask_maps[view_idx]
            max_plane_id = int(plane_mask_map.max())

            for plane_id in range(1, max_plane_id + 1):
                node = (view_idx, plane_id)
                if node not in self.local_plane2global_plane_idx:
                    continue

                global_idx = self.local_plane2global_plane_idx[node]
                if global_idx not in self.global_plane_info:
                    continue

                plane_depth = self._compute_global_plane_depth_map(view_idx, global_idx)
                local_mask = plane_mask_map == plane_id
                valid_fill = local_mask & (plane_depth > 0)

                refined_depth_maps[view_idx][valid_fill] = plane_depth[valid_fill]
                refined_conf_maps[view_idx][valid_fill] = True
                global_plane_mask_maps[view_idx][valid_fill] = global_idx

        return refined_depth_maps, refined_conf_maps, global_plane_mask_maps

    def _build_merge_only_global_plane_mask_maps(self):
        merge_only_maps = [
            np.zeros_like(mask_map, dtype=np.int32)
            for mask_map in self.plane_mask_maps
        ]

        for view_idx in range(self.num_views):
            plane_mask_map = self.plane_mask_maps[view_idx]
            max_plane_id = int(plane_mask_map.max())

            for plane_id in range(1, max_plane_id + 1):
                node = (view_idx, plane_id)
                if node not in self.local_plane2global_plane_idx:
                    continue

                global_idx = self.local_plane2global_plane_idx[node]
                local_mask = plane_mask_map == plane_id
                merge_only_maps[view_idx][local_mask] = global_idx

        return merge_only_maps

    def save_debug_visualizations(self, projected_global_plane_mask_maps):
        debug_root = os.path.join(self.output_root, "debug_merge")
        os.makedirs(debug_root, exist_ok=True)

        merge_only_maps = self._build_merge_only_global_plane_mask_maps()
        local_plane_lookup = {
            (item["view_idx"], item["plane_id"]): item
            for item in self.local_planes
        }

        for view_idx in range(self.num_views):
            frame_idx = self.frame_ids[view_idx]
            rgb = self.rgb_images[view_idx]
            plane_mask_map = self.plane_mask_maps[view_idx]
            merge_only_map = merge_only_maps[view_idx]
            projected_map = projected_global_plane_mask_maps[view_idx]

            np.save(
                os.path.join(debug_root, f"merge_only_global_plane_mask_frame{frame_idx:06d}.npy"),
                merge_only_map.astype(np.int32),
            )
            save_index_mask(
                merge_only_map,
                os.path.join(debug_root, f"merge_only_global_plane_mask_frame{frame_idx:06d}.png"),
            )
            Image.fromarray(
                overlay_label_map(rgb, merge_only_map, alpha=0.55)
            ).save(os.path.join(debug_root, f"merge_only_global_plane_vis_frame{frame_idx:06d}.png"))

            projection_drop_mask = (merge_only_map > 0) & (projected_map == 0)
            save_bool_mask(
                projection_drop_mask,
                os.path.join(debug_root, f"projection_drop_mask_frame{frame_idx:06d}.png"),
            )
            Image.fromarray(
                overlay_binary_mask(
                    rgb,
                    projection_drop_mask,
                    color=np.array([255, 64, 64], dtype=np.uint8),
                    alpha=0.65,
                )
            ).save(os.path.join(debug_root, f"projection_drop_vis_frame{frame_idx:06d}.png"))

            unmerged_local_mask = np.zeros_like(plane_mask_map, dtype=bool)
            local_plane_records = []

            max_plane_id = int(plane_mask_map.max())
            for plane_id in range(1, max_plane_id + 1):
                node = (view_idx, plane_id)
                local_mask = plane_mask_map == plane_id
                merged_global_idx = self.local_plane2global_plane_idx.get(node, 0)
                if merged_global_idx == 0:
                    unmerged_local_mask |= local_mask

                local_info = local_plane_lookup.get(node, None)
                record = {
                    "view_idx": int(view_idx),
                    "frame_idx": int(frame_idx),
                    "plane_id": int(plane_id),
                    "global_plane_idx": int(merged_global_idx),
                    "full_area": int(local_mask.sum()),
                }
                if local_info is not None:
                    record["known_area"] = int(local_info["known_area"])
                    record["voxel_count"] = int(len(local_info["voxel_ids"]))
                    record["source_view_index"] = int(local_info["source_view_index"])
                    record["source_image_name"] = local_info["source_image_name"]
                local_plane_records.append(record)

            save_bool_mask(
                unmerged_local_mask,
                os.path.join(debug_root, f"unmerged_local_plane_mask_frame{frame_idx:06d}.png"),
            )
            Image.fromarray(
                overlay_binary_mask(
                    rgb,
                    unmerged_local_mask,
                    color=np.array([128, 128, 128], dtype=np.uint8),
                    alpha=0.65,
                )
            ).save(os.path.join(debug_root, f"unmerged_local_plane_vis_frame{frame_idx:06d}.png"))

            with open(
                os.path.join(debug_root, f"local_plane_status_frame{frame_idx:06d}.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(local_plane_records, f, indent=2)

    def vis_global_plane_masks(self, save_root_path):
        os.makedirs(save_root_path, exist_ok=True)

        for global_plane_idx, global_plane in self.global_planes.items():
            global_plane_dir = os.path.join(
                save_root_path,
                f"global_plane_{global_plane_idx:03d}",
            )
            os.makedirs(global_plane_dir, exist_ok=True)

            global_color = label_to_color(global_plane_idx)

            for node in global_plane["nodes"]:
                view_idx, plane_id = node
                frame_idx = self.frame_ids[view_idx]
                rgb = self.rgb_images[view_idx]
                local_mask = self.plane_mask_maps[view_idx] == plane_id

                vis = overlay_binary_mask(
                    rgb,
                    local_mask,
                    color=global_color,
                    alpha=0.60,
                )
                save_path = os.path.join(
                    global_plane_dir,
                    f"frame{frame_idx:06d}_view{view_idx:03d}_plane{plane_id:03d}.png",
                )
                Image.fromarray(vis).save(save_path)

    def vis_each_view_local_planes(self, save_root_path):
        os.makedirs(save_root_path, exist_ok=True)
        local_plane_lookup = {
            (item["view_idx"], item["plane_id"]): item
            for item in self.local_planes
        }

        for view_idx in range(self.num_views):
            frame_idx = self.frame_ids[view_idx]
            rgb = self.rgb_images[view_idx]
            plane_mask_map = self.plane_mask_maps[view_idx]
            max_plane_id = int(plane_mask_map.max())

            vis = rgb.astype(np.float32).copy()
            local_records = []

            for plane_id in range(1, max_plane_id + 1):
                node = (view_idx, plane_id)
                local_mask = plane_mask_map == plane_id
                global_plane_idx = self.local_plane2global_plane_idx.get(node, 0)

                if global_plane_idx > 0:
                    color = label_to_color(global_plane_idx).astype(np.float32)
                else:
                    color = np.array([140, 140, 140], dtype=np.float32)

                vis[local_mask] = 0.45 * vis[local_mask] + 0.55 * color

                local_info = local_plane_lookup.get(node, None)
                record = {
                    "view_idx": int(view_idx),
                    "frame_idx": int(frame_idx),
                    "plane_id": int(plane_id),
                    "global_plane_idx": int(global_plane_idx),
                    "full_area": int(local_mask.sum()),
                }
                if local_info is not None:
                    record["known_area"] = int(local_info["known_area"])
                    record["voxel_count"] = int(len(local_info["voxel_ids"]))
                    record["source_view_index"] = int(local_info["source_view_index"])
                    record["source_image_name"] = local_info["source_image_name"]
                local_records.append(record)

            Image.fromarray(np.clip(vis, 0, 255).astype(np.uint8)).save(
                os.path.join(save_root_path, f"local_plane_vis_frame{frame_idx:06d}.png")
            )

            with open(
                os.path.join(save_root_path, f"local_plane_vis_frame{frame_idx:06d}.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(local_records, f, indent=2)

    def vis_global_3Dplanes(self, save_root_path, num_points=20000):
        os.makedirs(save_root_path, exist_ok=True)

        for global_plane_idx, global_plane in self.global_planes.items():
            if global_plane_idx not in self.global_plane_info:
                continue

            support_points = []
            for node in global_plane["nodes"]:
                view_idx, plane_id = node
                points = self._collect_local_plane_points(view_idx, plane_id)
                if points.shape[0] > 0:
                    support_points.append(points)

            if len(support_points) == 0:
                continue

            support_points = np.concatenate(support_points, axis=0).astype(np.float32)
            info = self.global_plane_info[global_plane_idx]

            sampled_plane_points = generate_points_on_plane(
                plane_points=support_points,
                plane_normal=info["normal"],
                mean_plane_points=info["mean"],
                num_points=num_points,
            )

            color = label_to_color(global_plane_idx)
            sampled_colors = np.repeat(color[None], sampled_plane_points.shape[0], axis=0)
            support_colors = np.repeat(color[None], support_points.shape[0], axis=0)

            save_tensor_as_pcd(
                sampled_plane_points,
                os.path.join(save_root_path, f"global_plane_{global_plane_idx:03d}.ply"),
                pcd_colors=sampled_colors,
            )
            save_tensor_as_pcd(
                support_points,
                os.path.join(save_root_path, f"global_plane_{global_plane_idx:03d}_support.ply"),
                pcd_colors=support_colors,
            )

    def save_outputs(self, refined_depth_maps, refined_conf_maps, global_plane_mask_maps):
        os.makedirs(self.output_root, exist_ok=True)

        global_plane_num = max(self.global_plane_info.keys(), default=0)
        global_plane_params = np.zeros((global_plane_num + 1, 4), dtype=np.float32)
        for global_idx, info in self.global_plane_info.items():
            global_plane_params[global_idx] = info["plane_params"]
        np.save(os.path.join(self.output_root, "global_plane_params.npy"), global_plane_params)

        global_plane_meta = {}
        for global_idx, global_plane in self.global_planes.items():
            nodes_payload = []
            for view_idx, plane_id in global_plane["nodes"]:
                spec = self.camera_specs[view_idx]
                nodes_payload.append(
                    {
                        "view_idx": int(view_idx),
                        "frame_idx": int(self.frame_ids[view_idx]),
                        "plane_id": int(plane_id),
                        "camera_name": spec["camera_name"],
                        "source_view_index": int(spec["source_view_index"]),
                        "source_image_name": spec["source_image_name"],
                    }
                )

            info = self.global_plane_info.get(global_idx, None)
            global_plane_meta[str(global_idx)] = {
                "nodes": nodes_payload,
                "num_views": len(global_plane["view_set"]),
                "num_voxels": int(len(global_plane["voxel_ids"])),
                "normal": info["normal"].tolist() if info is not None else None,
                "mean": info["mean"].tolist() if info is not None else None,
                "plane_params": info["plane_params"].tolist() if info is not None else None,
            }

        with open(os.path.join(self.output_root, "global_plane_info.json"), "w", encoding="utf-8") as f:
            json.dump(global_plane_meta, f, indent=2)

        local_to_global = {}
        for (view_idx, plane_id), global_idx in self.local_plane2global_plane_idx.items():
            local_to_global[f"{view_idx}:{plane_id}"] = int(global_idx)
        with open(os.path.join(self.output_root, "local_plane_to_global_plane.json"), "w", encoding="utf-8") as f:
            json.dump(local_to_global, f, indent=2)

        for view_idx in range(self.num_views):
            frame_idx = self.frame_ids[view_idx]
            rgb = self.rgb_images[view_idx]

            refined_depth = refined_depth_maps[view_idx]
            refined_conf = refined_conf_maps[view_idx]
            global_plane_mask = global_plane_mask_maps[view_idx]

            np.save(
                os.path.join(self.output_root, f"plane_refined_depth_frame{frame_idx:06d}.npy"),
                refined_depth.astype(np.float32),
            )
            save_img_f32(
                refined_depth.astype(np.float32),
                os.path.join(self.output_root, f"plane_refined_depth_frame{frame_idx:06d}.tiff"),
            )
            save_depth_vis(
                refined_depth,
                os.path.join(self.output_root, f"plane_refined_depth_frame{frame_idx:06d}.png"),
            )

            np.save(
                os.path.join(self.output_root, f"plane_refined_conf_frame{frame_idx:06d}.npy"),
                refined_conf.astype(bool),
            )
            save_bool_mask(
                refined_conf,
                os.path.join(self.output_root, f"plane_refined_conf_frame{frame_idx:06d}.png"),
            )

            trusted = self.trusted_masks[view_idx]
            raw_depth = self.raw_depth_maps[view_idx]
            merge_refined_depth = np.where(trusted, raw_depth, refined_depth).astype(np.float32)
            merge_refined_conf = (trusted | refined_conf).astype(bool)

            np.save(
                os.path.join(self.output_root, f"merge_refined_depth_frame{frame_idx:06d}.npy"),
                merge_refined_depth,
            )
            save_img_f32(
                merge_refined_depth,
                os.path.join(self.output_root, f"merge_refined_depth_frame{frame_idx:06d}.tiff"),
            )
            save_depth_vis(
                merge_refined_depth,
                os.path.join(self.output_root, f"merge_refined_depth_frame{frame_idx:06d}.png"),
            )

            np.save(
                os.path.join(self.output_root, f"merge_refined_conf_frame{frame_idx:06d}.npy"),
                merge_refined_conf,
            )
            save_bool_mask(
                merge_refined_conf,
                os.path.join(self.output_root, f"merge_refined_conf_frame{frame_idx:06d}.png"),
            )

            np.save(
                os.path.join(self.output_root, f"global_plane_mask_frame{frame_idx:06d}.npy"),
                global_plane_mask.astype(np.int32),
            )
            save_index_mask(
                global_plane_mask,
                os.path.join(self.output_root, f"global_plane_mask_frame{frame_idx:06d}.png"),
            )
            Image.fromarray(
                overlay_label_map(rgb, global_plane_mask, alpha=0.55)
            ).save(os.path.join(self.output_root, f"global_plane_vis_frame{frame_idx:06d}.png"))

    # def run(self):
    #     self.trace_local_planes()
    #     self.merge_planes()
    #     self.fit_global_planes()
    #     refined_depth_maps, refined_conf_maps, global_plane_mask_maps = self.compute_refined_depth_maps()
    #     self.save_outputs(refined_depth_maps, refined_conf_maps, global_plane_mask_maps)

    def run(self):
        self.trace_local_planes()
        self.merge_planes()
        self.fit_global_planes()

        refined_depth_maps, refined_conf_maps, global_plane_mask_maps = self.compute_refined_depth_maps()
        self.save_outputs(refined_depth_maps, refined_conf_maps, global_plane_mask_maps)

        self.save_debug_visualizations(global_plane_mask_maps)
        self.vis_global_plane_masks(os.path.join(self.output_root, "vis_global_plane_masks"))
        self.vis_each_view_local_planes(os.path.join(self.output_root, "vis_local_planes"))
        self.vis_global_3Dplanes(os.path.join(self.output_root, "vis_global_3d_planes"))


def load_voxel_model(args):
    update_config(args.config)
    cfg.model.model_path = os.path.dirname(os.path.dirname(args.checkpoint))

    voxel_model = SparseVoxelModel(cfg.model)
    voxel_model.load(args.checkpoint)

    if args.overwrite_ss is not None:
        voxel_model.ss = args.overwrite_ss

    if args.overwrite_vox_geo_mode is not None:
        voxel_model.vox_geo_mode = args.overwrite_vox_geo_mode

    voxel_model.freeze_vox_geo()
    return voxel_model


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--config", required=True, type=str)

    parser.add_argument("--guidance_root", required=True, type=str,
                        help="Path to see3d_guidance_views")
    parser.add_argument("--dn_root", default=None, type=str,
                        help="Path to see3d_mono_dn")
    parser.add_argument("--plane_root", default=None, type=str,
                        help="Path to plane extraction outputs")
    parser.add_argument("--output_root", default=None, type=str,
                        help="Directory to save merged 3D plane outputs")

    parser.add_argument("--overwrite_ss", default=None, type=float)
    parser.add_argument("--overwrite_vox_geo_mode", default=None, type=str)

    parser.add_argument("--trace_bandwidth_mul", default=50.0, type=float)
    parser.add_argument("--trace_conf_thresh", default=0.35, type=float)
    parser.add_argument("--voxel_iou_thresh", default=0.50, type=float)
    parser.add_argument("--normal_angle_thresh", default=30.0, type=float)
    parser.add_argument("--min_global_view_rate", default=0.02, type=float)
    parser.add_argument("--plane_fit_thresh", default=0.01, type=float)
    parser.add_argument("--plane_fit_trials", default=512, type=int)
    parser.add_argument("--max_plane_fit_points", default=200000, type=int)
    parser.add_argument("--min_voxels_per_plane", default=8, type=int)

    args = parser.parse_args()

    if args.dn_root is None:
        args.dn_root = os.path.join(args.guidance_root, "see3d_mono_dn")
    if args.plane_root is None:
        args.plane_root = args.dn_root
    if args.output_root is None:
        args.output_root = os.path.join(args.guidance_root, "merge_3d_plane")

    print("=" * 60)
    print("Merge 3D Plane With Sparse Voxels")
    print("=" * 60)
    print(f"Checkpoint:   {args.checkpoint}")
    print(f"Config:       {args.config}")
    print(f"Guidance:     {args.guidance_root}")
    print(f"DepthNormal:  {args.dn_root}")
    print(f"PlaneRoot:    {args.plane_root}")
    print(f"Output:       {args.output_root}")

    voxel_model = load_voxel_model(args)

    merger = SparseVoxelPlaneMerger(
        voxel_model=voxel_model,
        guidance_root=args.guidance_root,
        dn_root=args.dn_root,
        plane_root=args.plane_root,
        output_root=args.output_root,
        trace_bandwidth_mul=args.trace_bandwidth_mul,
        trace_conf_thresh=args.trace_conf_thresh,
        voxel_iou_thresh=args.voxel_iou_thresh,
        normal_angle_thresh=args.normal_angle_thresh,
        min_global_view_rate=args.min_global_view_rate,
        plane_fit_thresh=args.plane_fit_thresh,
        plane_fit_trials=args.plane_fit_trials,
        max_plane_fit_points=args.max_plane_fit_points,
        min_voxels_per_plane=args.min_voxels_per_plane,
        device="cuda",
    )
    merger.run()

    print("=" * 60)
    print("Done!")
    print("=" * 60)