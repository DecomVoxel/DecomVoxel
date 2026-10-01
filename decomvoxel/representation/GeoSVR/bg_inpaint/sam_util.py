import numpy as np
import torch
from dataclasses import dataclass


@dataclass
class SAMConfig:
    checkpoint_path: str
    model_type: str = "vit_h"
    points_per_side: int = 32
    pred_iou_thresh: float = 0.86
    stability_score_thresh: float = 0.92
    crop_n_layers: int = 0
    crop_n_points_downscale_factor: int = 1
    min_mask_region_area: int = 0


def build_sam_mask_generator(config: SAMConfig, device: str = None):
    try:
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    except ImportError as exc:
        raise ImportError(
            "segment_anything is not installed. "
            "Install it with `pip install git+https://github.com/facebookresearch/segment-anything.git`."
        ) from exc

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    sam = sam_model_registry[config.model_type](checkpoint=config.checkpoint_path)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=int(config.points_per_side),
        pred_iou_thresh=float(config.pred_iou_thresh),
        stability_score_thresh=float(config.stability_score_thresh),
        crop_n_layers=int(config.crop_n_layers),
        crop_n_points_downscale_factor=int(config.crop_n_points_downscale_factor),
        min_mask_region_area=int(config.min_mask_region_area),
    )
    return mask_generator


def generate_sam_masks(
    image_rgb: np.ndarray,
    mask_generator,
    min_area: int = 0,
    max_area_ratio: float = 1.0,
    sort_small_to_large: bool = True,
):
    image_rgb = np.asarray(image_rgb)
    if image_rgb.dtype != np.uint8:
        image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)

    anns = mask_generator.generate(image_rgb)
    image_area = int(image_rgb.shape[0] * image_rgb.shape[1])

    outputs = []
    for ann in anns:
        mask = ann["segmentation"].astype(bool)
        area = int(mask.sum())

        if area < int(min_area):
            continue
        if area > int(max_area_ratio * image_area):
            continue

        outputs.append(
            {
                "mask": mask,
                "area": area,
                "bbox": ann.get("bbox"),
                "predicted_iou": float(ann.get("predicted_iou", 0.0)),
                "stability_score": float(ann.get("stability_score", 0.0)),
            }
        )

    outputs.sort(key=lambda item: item["area"], reverse=not sort_small_to_large)
    return outputs


def overlay_label_map(
    image_rgb: np.ndarray,
    label_map: np.ndarray,
    alpha: float = 0.55,
    seed: int = 0,
):
    image_rgb = np.asarray(image_rgb).astype(np.uint8)
    label_map = np.asarray(label_map)
    max_label = int(label_map.max())

    if max_label <= 0:
        return image_rgb.copy()

    rng = np.random.default_rng(seed)
    colors = rng.integers(0, 256, size=(max_label + 1, 3), dtype=np.uint8)
    colors[0] = 0

    out = image_rgb.astype(np.float32).copy()
    for label in range(1, max_label + 1):
        mask = label_map == label
        if not np.any(mask):
            continue
        out[mask] = (1.0 - alpha) * out[mask] + alpha * colors[label]

    return np.clip(out, 0, 255).astype(np.uint8)

def sam_masks_to_instance_map(
    sam_masks,
    image_shape,
    keep_small_first: bool = True,
):
    image_h, image_w = image_shape[:2]
    label_map = np.zeros((image_h, image_w), dtype=np.uint16)
    occupied = np.zeros((image_h, image_w), dtype=bool)

    mask_items = sam_masks if keep_small_first else list(reversed(sam_masks))
    next_label = 1

    for item in mask_items:
        mask = np.asarray(item["mask"], dtype=bool)
        if mask.shape != (image_h, image_w):
            raise ValueError(
                f"Mask shape {mask.shape} does not match image shape {(image_h, image_w)}"
            )

        # Keep earlier instances stable and avoid overwriting by later masks.
        mask = mask & (~occupied)
        if not np.any(mask):
            continue

        label_map[mask] = next_label
        occupied[mask] = True
        next_label += 1

    return label_map

