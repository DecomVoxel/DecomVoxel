"""Mask application + PSNR/SSIM/LPIPS computation for rendered evaluation."""

from __future__ import annotations

import os
from glob import glob

import cv2
import numpy as np
import torch


def find_instance_mask_dir(scene_data_dir: str) -> str:
    for name in ("instance_masks", "instance_mask"):
        cand = os.path.join(scene_data_dir, name)
        if os.path.isdir(cand):
            return cand
    raise FileNotFoundError(f"No instance_masks/ or instance_mask/ under {scene_data_dir}")


def _read_image_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"cv2 failed to read: {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def _read_mask(path: str, target_hw: tuple[int, int]) -> np.ndarray:
    """Return uint8 binary mask (H,W); 1 = foreground (keep original color),
    0 = background (will be whitened). Background convention: pixel value > 250.
    """
    m = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(f"cv2 failed to read mask: {path}")
    if m.ndim == 3:
        m = m[..., 0]
    if (m.shape[0], m.shape[1]) != target_hw:
        m = cv2.resize(m, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_NEAREST)
    return (m <= 250).astype(np.uint8)


def apply_mask_to_image(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """White-out pixels where mask==0; keep pixels where mask>0."""
    out = rgb.copy()
    out[mask == 0] = 255
    return out


def _resolve_mask_path(mask_dir: str, image_name: str) -> str | None:
    """Try multiple mask filename conventions:
       1. <image_name>.<ext>
       2. <image_name before first '_'>.<ext>
       Supported extensions: png, jpg, jpeg, JPG, JPEG, PNG.
    """
    exts = ("png", "PNG", "jpg", "JPG", "jpeg", "JPEG")
    candidates = [image_name]
    if "_" in image_name:
        candidates.append(image_name.split("_", 1)[0])
    for stem in candidates:
        for ext in exts:
            p = os.path.join(mask_dir, f"{stem}.{ext}")
            if os.path.isfile(p):
                return p
    return None


def build_masked_pairs(
    pred_dir: str,
    gt_dir: str,
    mask_dir: str,
    out_pred_dir: str,
    out_gt_dir: str,
    image_names: list[str],
) -> list[tuple[str, str]]:
    """Apply mask to both pred & gt renders, save as PNG, return list of (pred_path, gt_path)."""
    os.makedirs(out_pred_dir, exist_ok=True)
    os.makedirs(out_gt_dir, exist_ok=True)
    pairs = []
    for name in image_names:
        pred_path = os.path.join(pred_dir, f"{name}.png")
        gt_path = os.path.join(gt_dir, f"{name}.png")
        if not (os.path.isfile(pred_path) and os.path.isfile(gt_path)):
            print(f"[WARN] missing pred or gt for {name}, skipping")
            continue

        pred = _read_image_rgb(pred_path)
        gt = _read_image_rgb(gt_path)
        if pred.shape[:2] != gt.shape[:2]:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_AREA)
        H, W = gt.shape[:2]

        mask_path = _resolve_mask_path(mask_dir, name)
        if mask_path is None:
            print(f"[WARN] missing mask for {name} under {mask_dir}, skipping")
            continue
        mask = _read_mask(mask_path, (H, W))

        pred_m = apply_mask_to_image(pred, mask)
        gt_m = apply_mask_to_image(gt, mask)

        out_pred = os.path.join(out_pred_dir, f"{name}.png")
        out_gt = os.path.join(out_gt_dir, f"{name}.png")
        cv2.imwrite(out_pred, cv2.cvtColor(pred_m, cv2.COLOR_RGB2BGR))
        cv2.imwrite(out_gt, cv2.cvtColor(gt_m, cv2.COLOR_RGB2BGR))
        pairs.append((out_pred, out_gt))
    return pairs


def _to_tensor(img: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0


def compute_metrics(pairs: list[tuple[str, str]], device: str | None = None) -> dict:
    if not pairs:
        raise ValueError("No image pairs to evaluate.")
    from piq import psnr as piq_psnr, ssim as piq_ssim
    import lpips as lpips_lib

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    lpips_net = lpips_lib.LPIPS(net="vgg").to(device).eval()

    psnr_vals, ssim_vals, lpips_vals = [], [], []
    with torch.no_grad():
        for pred_path, gt_path in pairs:
            pred = _to_tensor(_read_image_rgb(pred_path)).to(device)
            gt = _to_tensor(_read_image_rgb(gt_path)).to(device)
            psnr_vals.append(piq_psnr(pred, gt, data_range=1.0).item())
            ssim_vals.append(piq_ssim(pred, gt, data_range=1.0).item())
            # lpips expects range [-1,1]
            lpips_vals.append(lpips_net(pred * 2 - 1, gt * 2 - 1).item())

    return {
        "PSNR": float(np.mean(psnr_vals)),
        "SSIM": float(np.mean(ssim_vals)),
        "LPIPS": float(np.mean(lpips_vals)),
        "Count": len(pairs),
    }
