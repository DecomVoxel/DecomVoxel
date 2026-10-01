import gc
import os
from argparse import ArgumentParser
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from PIL import Image

from sam_util import (
    SAMConfig,
    build_sam_mask_generator,
    generate_sam_masks,
    sam_masks_to_instance_map,
    overlay_label_map,
)


@dataclass
class PlaneExtractorConfig:
    min_size_ratio: float = 0.01
    n_init_normal_clusters: int = 8
    n_normal_clusters: int = 6
    normal_merge_cos_thresh: float = 0.965
    min_sam_overlap: float = 0.35
    min_normal_overlap: float = 0.25
    candidate_iou_thresh: float = 0.90
    sam_max_area_ratio: float = 0.95
    soft_bg_plane_det: bool = False
    soft_bg_hole_overlap_thresh: float = 0.8


def normalize_np(x, eps=1e-6):
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(norm, eps, None)


def list_frame_indices(data_root):
    frame_indices = []
    for file_name in os.listdir(data_root):
        if not file_name.startswith("rgb_frame") or not file_name.endswith(".png"):
            continue
        stem = os.path.splitext(file_name)[0]
        frame_idx = int(stem.split("frame")[-1])
        frame_indices.append(frame_idx)
    frame_indices.sort()
    return frame_indices


def load_rgb(data_root, frame_idx):
    path = os.path.join(data_root, f"rgb_frame{frame_idx:06d}.png")
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def load_normals(data_root, frame_idx, use_world_normal, target_hw):
    prefix = "mono_normal_world_frame" if use_world_normal else "mono_normal_frame"
    path = os.path.join(data_root, f"{prefix}{frame_idx:06d}.npy")
    normals = np.load(path).astype(np.float32)

    target_h, target_w = target_hw
    if normals.shape[:2] != (target_h, target_w):
        normals = cv2.resize(normals, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    finite = np.isfinite(normals).all(axis=-1)
    lengths = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = normals / np.clip(lengths, 1e-6, None)
    normals[~finite] = 0.0
    return normals

def load_inpaint_hole_mask(data_root, frame_idx, target_hw):
    target_h, target_w = target_hw
    mask_path = os.path.join(data_root, f"mask_frame{frame_idx:06d}.png")

    if not os.path.exists(mask_path):
        return np.zeros((target_h, target_w), dtype=bool)

    mask = Image.open(mask_path).convert("L")
    if mask.size != (target_w, target_h):
        mask = mask.resize((target_w, target_h), Image.NEAREST)

    # mask_frame convention:
    #   255 = trusted / visible region
    #   0   = hole / inpaint region
    visible_mask = np.asarray(mask, dtype=np.uint8) > 127
    hole_mask = ~visible_mask
    return hole_mask

def split_connected_components(mask, min_area):
    mask_u8 = mask.astype(np.uint8)
    if int(mask_u8.sum()) == 0:
        return []

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    components = []

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < int(min_area):
            continue
        components.append(labels == label)

    return components


def merge_similar_clusters(labels, centers, cos_thresh):
    labels = labels.astype(np.int32).copy()
    centers = normalize_np(centers.astype(np.float32))

    num_centers = centers.shape[0]
    parents = np.arange(num_centers, dtype=np.int32)

    def find(x):
        while parents[x] != x:
            parents[x] = parents[parents[x]]
            x = parents[x]
        return x

    def union(a, b):
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parents[root_b] = root_a

    for i in range(num_centers):
        for j in range(i + 1, num_centers):
            if float(np.dot(centers[i], centers[j])) >= float(cos_thresh):
                union(i, j)

    root_to_new = {}
    next_label = 0
    merged = np.empty_like(labels)

    for cluster_id in np.unique(labels):
        root = find(int(cluster_id))
        if root not in root_to_new:
            root_to_new[root] = next_label
            next_label += 1
        merged[labels == cluster_id] = root_to_new[root]

    return merged


def cluster_normal_regions(normals, config, min_area):
    image_h, image_w = normals.shape[:2]

    finite = np.isfinite(normals).all(axis=-1)
    lengths = np.linalg.norm(normals, axis=-1)
    valid = finite & (lengths > 1e-6)

    if int(valid.sum()) < max(16, int(min_area)):
        return []

    normals_unit = np.zeros_like(normals, dtype=np.float32)
    normals_unit[valid] = normals[valid] / np.clip(lengths[valid, None], 1e-6, None)

    samples = normals_unit[valid].astype(np.float32)
    num_clusters = max(1, min(int(config.n_init_normal_clusters), samples.shape[0]))

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)
    _compactness, labels, centers = cv2.kmeans(
        samples,
        num_clusters,
        None,
        criteria,
        3,
        cv2.KMEANS_PP_CENTERS,
    )

    labels = labels.reshape(-1)
    labels = merge_similar_clusters(
        labels=labels,
        centers=centers,
        cos_thresh=config.normal_merge_cos_thresh,
    )

    unique_ids, counts = np.unique(labels, return_counts=True)
    sorted_ids = unique_ids[np.argsort(counts)[::-1]]
    sorted_ids = sorted_ids[: min(int(config.n_normal_clusters), len(sorted_ids))]

    label_map = np.full((image_h, image_w), -1, dtype=np.int32)
    label_map[valid] = labels

    normal_masks = []
    for cluster_id in sorted_ids:
        cluster_mask = label_map == int(cluster_id)
        normal_masks.extend(split_connected_components(cluster_mask, min_area=min_area))

    return normal_masks


def mask_iou(mask_a, mask_b):
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def average_normal(normals, mask):
    values = normals[mask]
    if values.size == 0:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)

    avg = values.mean(axis=0)
    avg = normalize_np(avg[None])[0]
    return avg.astype(np.float32)

def mask_overlap_ratio(mask, ref_mask):
    mask = np.asarray(mask, dtype=bool)
    ref_mask = np.asarray(ref_mask, dtype=bool)

    area = int(mask.sum())
    if area == 0:
        return 0.0

    overlap = int((mask & ref_mask).sum())
    return float(overlap) / float(area)


def masks_to_label_map(masks, image_shape):
    image_h, image_w = image_shape[:2]
    label_map = np.zeros((image_h, image_w), dtype=np.uint16)

    for mask_idx, mask in enumerate(masks, start=1):
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (image_h, image_w):
            raise ValueError(
                f"Mask shape {mask.shape} does not match image shape {(image_h, image_w)}"
            )
        label_map[mask] = mask_idx

    return label_map

def extract_plane_instances(
    normals,
    image_shape,
    sam_masks,
    normal_masks,
    config,
    min_area,
    inpaint_hole_mask=None,
):
    image_h, image_w = image_shape[:2]

    if inpaint_hole_mask is None:
        inpaint_hole_mask = np.zeros((image_h, image_w), dtype=bool)
    else:
        inpaint_hole_mask = np.asarray(inpaint_hole_mask, dtype=bool)
        if inpaint_hole_mask.shape != (image_h, image_w):
            raise ValueError(
                f"inpaint_hole_mask shape {inpaint_hole_mask.shape} "
                f"does not match image shape {(image_h, image_w)}"
            )

    soft_bg_plane_det = bool(getattr(config, "soft_bg_plane_det", False))
    soft_bg_hole_overlap_thresh = float(
        getattr(config, "soft_bg_hole_overlap_thresh", 0.8)
    )

    raw_candidates_by_normal = [[] for _ in range(len(normal_masks))]

    for sam_item in sam_masks:
        sam_components = split_connected_components(sam_item["mask"], min_area=min_area)
        for sam_mask in sam_components:
            sam_area = int(sam_mask.sum())
            if sam_area < int(min_area):
                continue

            for normal_idx, normal_mask in enumerate(normal_masks):
                inter = sam_mask & normal_mask
                inter_components = split_connected_components(inter, min_area=min_area)

                for inter_mask in inter_components:
                    area = int(inter_mask.sum())
                    if area < int(min_area):
                        continue

                    normal_area = int(normal_mask.sum())
                    sam_ratio = float(area) / float(max(sam_area, 1))
                    normal_ratio = float(area) / float(max(normal_area, 1))

                    if sam_ratio < float(config.min_sam_overlap):
                        continue

                    # ###### NOTE: NOT use normal ratio for now
                    # if normal_ratio < float(config.min_normal_overlap):
                    #     continue

                    raw_candidates_by_normal[normal_idx].append(
                        {
                            "mask": inter_mask,
                            "area": area,
                            "score": float(area) * (0.5 * sam_ratio + 0.5 * normal_ratio),
                            "normal_idx": normal_idx,
                            "hole_overlap_ratio": mask_overlap_ratio(
                                inter_mask,
                                inpaint_hole_mask,
                            ),
                            "candidate_type": "sam_normal_intersection",
                        }
                    )

    candidates = []

    for normal_idx, raw_candidates in enumerate(raw_candidates_by_normal):
        if len(raw_candidates) == 0:
            continue

        if not soft_bg_plane_det:
            candidates.extend(raw_candidates)
            continue

        soft_items = [
            item
            for item in raw_candidates
            if item["hole_overlap_ratio"] >= soft_bg_hole_overlap_thresh
        ]
        hard_items = [
            item
            for item in raw_candidates
            if item["hole_overlap_ratio"] < soft_bg_hole_overlap_thresh
        ]

        # No strongly hole-overlapping region: keep the original SAM separation.
        if len(soft_items) == 0:
            candidates.extend(raw_candidates)
            continue

        # Pick the anchor plane inside this normal region.
        # Prefer the largest hard candidate so the soft hole-related pieces merge
        # into an existing visible plane instead of collapsing the whole normal region.
        if len(hard_items) > 0:
            anchor = max(hard_items, key=lambda item: (item["area"], item["score"]))
            remaining_items = [item for item in hard_items if item is not anchor]
        else:
            # If every candidate in this normal region is hole-dominated,
            # fall back to the largest candidate as the merge anchor.
            anchor = max(raw_candidates, key=lambda item: (item["area"], item["score"]))
            remaining_items = []

        merged_mask = anchor["mask"].copy()
        merged_score = float(anchor["score"])

        for item in soft_items:
            if item is anchor:
                continue
            merged_mask |= item["mask"]
            merged_score += float(item["score"])

        merged_area = int(merged_mask.sum())
        candidates.append(
            {
                "mask": merged_mask,
                "area": merged_area,
                "score": merged_score,
                "normal_idx": normal_idx,
                "candidate_type": "soft_bg_merged_anchor",
            }
        )

        candidates.extend(remaining_items)

    candidates.sort(key=lambda item: (item["score"], item["area"]), reverse=True)

    plane_label_map = np.zeros(image_shape, dtype=np.uint16)
    occupied = np.zeros(image_shape, dtype=bool)

    accepted_masks = []
    accepted_normals = []
    accepted_areas = []

    for cand in candidates:
        if any(
            mask_iou(cand["mask"], prev) >= float(config.candidate_iou_thresh)
            for prev in accepted_masks
        ):
            continue

        mask = cand["mask"] & (~occupied)
        area = int(mask.sum())
        if area < int(min_area):
            continue

        plane_id = len(accepted_masks) + 1
        plane_label_map[mask] = plane_id
        occupied[mask] = True

        accepted_masks.append(mask)
        accepted_areas.append(area)
        accepted_normals.append(average_normal(normals, mask))

    if accepted_normals:
        plane_normals = np.stack(accepted_normals, axis=0).astype(np.float32)
    else:
        plane_normals = np.zeros((0, 3), dtype=np.float32)

    plane_areas = np.asarray(accepted_areas, dtype=np.int32)
    return plane_label_map, accepted_masks, plane_normals, plane_areas


def save_index_mask(label_map, save_path):
    if int(label_map.max()) <= 255:
        Image.fromarray(label_map.astype(np.uint8)).save(save_path)
    else:
        Image.fromarray(label_map.astype(np.uint16)).save(save_path)


@torch.no_grad()
def extract_plane_masks_for_directory(
    data_root,
    sam_checkpoint,
    output_root=None,
    sam_model_type="vit_h",
    use_world_normal=False,
    config=None,
    sam_points_per_side=32,
    sam_pred_iou_thresh=0.86,
    sam_stability_score_thresh=0.92,
    sam_crop_n_layers=0,
):
    if config is None:
        config = PlaneExtractorConfig()

    if output_root is None:
        output_root = data_root
    os.makedirs(output_root, exist_ok=True)

    frame_indices = list_frame_indices(data_root)
    if len(frame_indices) == 0:
        raise FileNotFoundError(f"No rgb_frame*.png found in {data_root}")

    sam_config = SAMConfig(
        checkpoint_path=sam_checkpoint,
        model_type=sam_model_type,
        points_per_side=sam_points_per_side,
        pred_iou_thresh=sam_pred_iou_thresh,
        stability_score_thresh=sam_stability_score_thresh,
        crop_n_layers=sam_crop_n_layers,
        min_mask_region_area=0,
    )
    sam_generator = build_sam_mask_generator(sam_config)

    for frame_idx in frame_indices:
        rgb = load_rgb(data_root, frame_idx)
        image_h, image_w = rgb.shape[:2]
        min_area = max(1, int(round(image_h * image_w * float(config.min_size_ratio))))

        normals = load_normals(
            data_root=data_root,
            frame_idx=frame_idx,
            use_world_normal=use_world_normal,
            target_hw=(image_h, image_w),
        )

        normal_masks = cluster_normal_regions(
            normals=normals,
            config=config,
            min_area=min_area,
        )

        if bool(getattr(config, "soft_bg_plane_det", False)):
            inpaint_hole_mask = load_inpaint_hole_mask(
                data_root=data_root,
                frame_idx=frame_idx,
                target_hw=(image_h, image_w),
            )
        else:
            inpaint_hole_mask = np.zeros((image_h, image_w), dtype=bool)

        normal_mask_map = masks_to_label_map(
            normal_masks,
            image_shape=(image_h, image_w),
        )

        # np.save(
        #     os.path.join(output_root, f"normal_mask_map_frame{frame_idx:06d}.npy"),
        #     normal_mask_map.astype(np.int32),
        # )

        # save_index_mask(
        #     normal_mask_map,
        #     os.path.join(output_root, f"normal_mask_map_frame{frame_idx:06d}.png"),
        # )

        Image.fromarray(
            overlay_label_map(rgb, normal_mask_map, alpha=0.55, seed=50000 + frame_idx)
        ).save(os.path.join(output_root, f"normal_mask_vis_frame{frame_idx:06d}.png"))

        sam_masks = generate_sam_masks(
            image_rgb=rgb,
            mask_generator=sam_generator,
            min_area=min_area,
            max_area_ratio=float(config.sam_max_area_ratio),
            sort_small_to_large=True,
        )

        sam_instance_map = sam_masks_to_instance_map(
            sam_masks=sam_masks,
            image_shape=(image_h, image_w),
            keep_small_first=True,
        )

        # np.save(
        #     os.path.join(output_root, f"sam_instance_map_frame{frame_idx:06d}.npy"),
        #     sam_instance_map.astype(np.int32),
        # )

        # save_index_mask(
        #     sam_instance_map,
        #     os.path.join(output_root, f"sam_instance_map_frame{frame_idx:06d}.png"),
        # )

        Image.fromarray(
            overlay_label_map(rgb, sam_instance_map, alpha=0.55, seed=100000 + frame_idx)
        ).save(os.path.join(output_root, f"sam_instance_vis_frame{frame_idx:06d}.png"))

        plane_label_map, plane_masks, plane_normals, plane_areas = extract_plane_instances(
            normals=normals,
            image_shape=(image_h, image_w),
            sam_masks=sam_masks,
            normal_masks=normal_masks,
            config=config,
            min_area=min_area,
            inpaint_hole_mask=inpaint_hole_mask,
        )

        np.save(
            os.path.join(output_root, f"plane_mask_frame{frame_idx:06d}.npy"),
            plane_label_map.astype(np.int32),
        )
        # save_index_mask(
        #     plane_label_map,
        #     os.path.join(output_root, f"plane_mask_frame{frame_idx:06d}.png"),
        # )
        Image.fromarray(
            overlay_label_map(rgb, plane_label_map, alpha=0.55, seed=frame_idx)
        ).save(os.path.join(output_root, f"plane_vis_frame{frame_idx:06d}.png"))

        np.savez(
            os.path.join(output_root, f"plane_info_frame{frame_idx:06d}.npz"),
            plane_normals=plane_normals.astype(np.float32),
            plane_areas=plane_areas.astype(np.int32),
        )

        print(
            f"frame {frame_idx:06d}: "
            f"sam_masks={len(sam_masks)}, "
            f"normal_clusters={len(normal_masks)}, "
            f"planes={len(plane_masks)}"
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    print("All frames done!")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--data_root",
        required=True,
        type=str,
        help="Directory containing rgb_frame*.png and mono_normal_frame*.npy",
    )
    parser.add_argument(
        "--sam_checkpoint",
        type=str,
        default="decomvoxel/representation/GeoSVR/bg_inpaint/checkpoint/segment-anything/sam_vit_h_4b8939.pth",
        help="Path to the official SAM checkpoint",
    )
    parser.add_argument(
        "--output_root",
        default=None,
        type=str,
        help="Directory to save plane masks and visualizations",
    )
    parser.add_argument(
        "--soft_bg_plane_det",
        action="store_true",
        help="Allow normal regions touching inpaint holes to bypass SAM splitting.",
    )
    parser.add_argument(
        "--soft_bg_hole_overlap_thresh",
        default=0.8,
        type=float,
        help="Merge SAM-normal regions into the largest plane of the same normal region when their hole overlap ratio is above this threshold.",
    )
    parser.add_argument(
        "--sam_model_type",
        default="vit_h",
        type=str,
        help="SAM model type, such as vit_h, vit_l, or vit_b",
    )
    parser.add_argument(
        "--use_world_normal",
        action="store_true",
        help="Use mono_normal_world_frame*.npy instead of mono_normal_frame*.npy",
    )
    parser.add_argument("--min_size_ratio", default=0.01, type=float)
    parser.add_argument("--n_init_normal_clusters", default=8, type=int)
    parser.add_argument("--n_normal_clusters", default=6, type=int)
    parser.add_argument("--normal_merge_cos_thresh", default=0.965, type=float)
    parser.add_argument("--min_sam_overlap", default=0.35, type=float)
    parser.add_argument("--min_normal_overlap", default=0.25, type=float)
    parser.add_argument("--candidate_iou_thresh", default=0.90, type=float)
    parser.add_argument("--sam_max_area_ratio", default=0.95, type=float)
    parser.add_argument("--sam_points_per_side", default=32, type=int)
    parser.add_argument("--sam_pred_iou_thresh", default=0.86, type=float)
    parser.add_argument("--sam_stability_score_thresh", default=0.92, type=float)
    parser.add_argument("--sam_crop_n_layers", default=0, type=int)
    args = parser.parse_args()

    config = PlaneExtractorConfig(
        min_size_ratio=args.min_size_ratio,
        n_init_normal_clusters=args.n_init_normal_clusters,
        n_normal_clusters=args.n_normal_clusters,
        normal_merge_cos_thresh=args.normal_merge_cos_thresh,
        min_sam_overlap=args.min_sam_overlap,
        min_normal_overlap=args.min_normal_overlap,
        candidate_iou_thresh=args.candidate_iou_thresh,
        sam_max_area_ratio=args.sam_max_area_ratio,
        soft_bg_plane_det=args.soft_bg_plane_det,
        soft_bg_hole_overlap_thresh=args.soft_bg_hole_overlap_thresh,
    )

    extract_plane_masks_for_directory(
        data_root=args.data_root,
        sam_checkpoint=args.sam_checkpoint,
        output_root=args.output_root,
        sam_model_type=args.sam_model_type,
        use_world_normal=args.use_world_normal,
        config=config,
        sam_points_per_side=args.sam_points_per_side,
        sam_pred_iou_thresh=args.sam_pred_iou_thresh,
        sam_stability_score_thresh=args.sam_stability_score_thresh,
        sam_crop_n_layers=args.sam_crop_n_layers,
    )