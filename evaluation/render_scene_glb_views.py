"""render_scene_glb_views.py

Render test-split views of a single scene.glb file via Blender, then compute
PSNR / SSIM / LPIPS against masked GT images.

This is a variant of render_svr_views.py that works with a single scene.glb
file containing multiple objects (instead of separate object_*.glb files).

Pipeline:
  1. Load COLMAP cameras from scan_path (test split).
  2. Load a single scene.glb file.
  3. Blender renders each camera (excluding 'background' mesh).
  4. Composite transparent regions to white (pure Python, no Blender).
  5. Apply instance mask to GT images only → mask_gt/.
  6. Compute metrics: pred (white-bg, no mask) vs masked-GT.

Output files:
    <exp_path>/render/<name>.png       white-background pred renders
    <exp_path>/gt_render/<name>.png    copied GT images
    <exp_path>/mask_gt/<name>.png      GT masked to instance regions
    <exp_path>/metrics_render_scene.txt  PSNR / SSIM / LPIPS

Usage::

    python evaluation/render_scene_glb_views.py \\
        --scan_path  datasets/Replica/scan1 \\
        --exp_path   exps/Replica-scene-render/scan1 \\
        --scene_glb  exps/Replica-scene-render/scan1/scene.glb
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

import cv2
import imageio
import numpy as np

# ── path setup ──────────────────────────────────────────────────────────────
_THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
sys.path.insert(0, os.path.join(_THIS_DIR, "utils"))

from colmap_io import load_test_cameras          # noqa: E402
from blender_invoke import run_blender_worker     # noqa: E402
from render_metrics import (                      # noqa: E402
    find_instance_mask_dir,
    compute_metrics,
    _read_image_rgb,
    _read_mask,
    _resolve_mask_path,
    apply_mask_to_image,
)

BLENDER_WORKER = os.path.join(_THIS_DIR, "utils", "blender_scene_glb_render_worker.py")


# ── helpers ──────────────────────────────────────────────────────────────────

def _composite_white_bg(rgba_path: str, out_path: str) -> None:
    """Read RGBA PNG, composite over white, save RGB PNG."""
    rgba = imageio.imread(rgba_path)         # H,W,4  uint8
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
    rgb   = rgba[:, :, :3].astype(np.float32)
    white = np.full_like(rgb, 255.0)
    result = (rgb * alpha + white * (1.0 - alpha)).clip(0, 255).astype(np.uint8)
    imageio.imwrite(out_path, result)


def _write_metrics_txt(path: str, metrics: dict, meta: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for k, v in meta.items():
            f.write(f"# {k}: {v}\n")
        for k in ("PSNR", "SSIM", "LPIPS", "Count"):
            if k in metrics:
                f.write(f"{k}: {metrics[k]}\n")


def _parse_metrics_txt(path: str) -> dict:
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            k, v = line.split(":", 1)
            try:
                out[k.strip()] = float(v.strip())
            except ValueError:
                out[k.strip()] = v.strip()
    return out


# ── GT + mask handling ───────────────────────────────────────────────────────

def copy_gt_images(test_cams: list[dict], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for cam in test_cams:
        src = cam["image_path"]
        if not os.path.isfile(src):
            print(f"[WARN] Missing GT image: {src}")
            continue
        dst = os.path.join(out_dir, f"{cam['image_name']}.png")
        if src.lower().endswith(".png"):
            shutil.copy2(src, dst)
        else:
            img = cv2.imread(src, cv2.IMREAD_UNCHANGED)
            cv2.imwrite(dst, img)


# ── overview ─────────────────────────────────────────────────────────────────

def generate_overview_md(
    exps_root: str,
    filename: str = "metrics_render_scene.txt",
    output_name: str = "overview_metrics_render_scene.md",
) -> str | None:
    from glob import glob
    files = sorted(glob(os.path.join(exps_root, "*", filename)))
    if not files:
        return None

    rows = [(os.path.basename(os.path.dirname(fp)), _parse_metrics_txt(fp)) for fp in files]
    keys = ["PSNR", "SSIM", "LPIPS", "Count"]
    out_path = os.path.join(exps_root, output_name)
    avg: dict[str, float] = {k: 0.0 for k in keys}
    avg_n: dict[str, int]  = {k: 0   for k in keys}

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Replica Scene GLB Render Eval Overview\n\n")
        f.write("| scene | " + " | ".join(keys) + " |\n")
        f.write("|" + "---|" * (len(keys) + 1) + "\n")
        for scene, m in rows:
            cells = []
            for k in keys:
                v = m.get(k, "-")
                if isinstance(v, float):
                    cells.append(f"{int(v)}" if k == "Count" else f"{v:.4f}")
                    avg[k] += v; avg_n[k] += 1
                else:
                    cells.append(str(v))
            f.write(f"| {scene} | " + " | ".join(cells) + " |\n")

        avg_cells = []
        for k in keys:
            if avg_n[k] > 0 and k != "Count":
                avg_cells.append(f"{avg[k]/avg_n[k]:.4f}")
            elif k == "Count" and avg_n[k] > 0:
                avg_cells.append(f"{int(avg[k])}")
            else:
                avg_cells.append("-")
        f.write(f"| **mean** | " + " | ".join(avg_cells) + " |\n")

    print(f"[INFO] Wrote overview -> {out_path}")
    return out_path


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a single scene.glb via Blender and compute masked PSNR/SSIM/LPIPS."
    )
    parser.add_argument("--scan_path", required=True,
                        help="Dataset scene dir, e.g. datasets/Replica/scan1")
    parser.add_argument("--exp_path", required=True,
                        help="Exp dir, e.g. exps/Replica-scene-render/scan1")
    parser.add_argument("--scene_glb", required=True,
                        help="Path to scene.glb file containing multiple meshes")
    parser.add_argument("--exclude_mesh_names", type=str, nargs="*", default=["background"],
                        help="Mesh names to exclude from rendering (default: ['background']).")
    parser.add_argument("--test_every", type=int, default=8,
                        help="Fallback test-camera stride when no test_split.json exists.")
    parser.add_argument("--images_subdir", type=str, default="images")
    parser.add_argument("--engine", type=str, default="CYCLES",
                        choices=["CYCLES", "EEVEE"])
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--device", type=str, default="GPU", choices=["GPU", "CPU"])
    parser.add_argument("--blender_install_root", type=str, default=None)
    parser.add_argument("--exps_root", default=None,
                        help="Root dir for overview markdown (defaults to parent of exp_path).")
    parser.add_argument("--update_overview", action="store_true", default=True)
    parser.add_argument("--no_update_overview", action="store_false", dest="update_overview")
    parser.add_argument("--skip_render", action="store_true",
                        help="Skip Blender rendering; reuse existing render/ dir.")
    parser.add_argument("--force_rerun", action="store_true", default=True)
    parser.add_argument("--no_force_rerun", action="store_false", dest="force_rerun")
    parser.add_argument("--mask_pred", action="store_true",
                        help="Also apply instance mask to predicted renders before computing metrics. "
                             "Skips rendering entirely (reuses existing render/ dir). "
                             "Adds '_mask' suffix to output metric filenames.")
    args = parser.parse_args()

    # mask_pred mode: never touch existing render directories
    if args.mask_pred:
        args.force_rerun = False
        args.skip_render = True

    scan_name  = os.path.basename(os.path.normpath(args.scan_path))
    exps_root  = os.path.abspath(args.exps_root or os.path.dirname(args.exp_path))

    _mask_suffix  = "_mask" if args.mask_pred else ""
    render_dir    = os.path.join(args.exp_path, "render")
    gt_dir        = os.path.join(args.exp_path, "gt_render")
    mask_gt_dir   = os.path.join(args.exp_path, "mask_gt")
    mask_pred_dir = os.path.join(args.exp_path, "mask_pred")
    metrics_path  = os.path.join(args.exp_path, f"metrics_render_scene{_mask_suffix}.txt")
    overview_filename = f"overview_metrics_render_scene{_mask_suffix}.md"

    # ── check scene GLB ───────────────────────────────────────────────────────
    if not os.path.isfile(args.scene_glb):
        raise FileNotFoundError(
            f"Scene GLB not found: {args.scene_glb}"
        )

    # ── optionally clear stale outputs ───────────────────────────────────────
    if args.force_rerun and not args.skip_render:
        for d in [render_dir, gt_dir, mask_gt_dir]:
            if os.path.isdir(d):
                shutil.rmtree(d)
                print(f"[force_rerun] Cleared {d}")

    # ── COLMAP cameras ───────────────────────────────────────────────────────
    print(f"[INFO] Loading COLMAP cameras from {args.scan_path}")
    test_cams, total = load_test_cameras(
        args.scan_path,
        images_subdir=args.images_subdir,
        test_every=args.test_every,
    )
    print(f"[INFO] {len(test_cams)} / {total} cameras selected as test.")
    image_names = [c["image_name"] for c in test_cams]

    # ── Blender render ───────────────────────────────────────────────────────
    if not args.skip_render:
        os.makedirs(render_dir, exist_ok=True)

        cam_payload = [
            {"image_name": c["image_name"],
             "W": c["W"], "H": c["H"],
             "K": c["K"], "w2c": c["w2c"]}
            for c in test_cams
        ]
        blender_cfg = {
            "jobs": [{
                "scene_glb":         os.path.abspath(args.scene_glb),
                "exclude_mesh_names": args.exclude_mesh_names,
                "output_dir":        os.path.abspath(render_dir),
                "engine":            args.engine,
                "samples":           args.samples,
                "device":            args.device,
                "cameras":           cam_payload,
            }]
        }
        rc = run_blender_worker(
            BLENDER_WORKER, blender_cfg,
            install_root=args.blender_install_root,
        )
        if rc != 0:
            raise RuntimeError(f"Blender worker exited with code {rc}")

        # ── transparent → white compositing ──────────────────────────────────
        print("[INFO] Compositing transparent background → white ...")
        for name in image_names:
            rgba_path = os.path.join(render_dir, f"{name}.png")
            if not os.path.isfile(rgba_path):
                print(f"[WARN] Missing render: {rgba_path}")
                continue
            _composite_white_bg(rgba_path, rgba_path)   # in-place

        # ── copy GT images ────────────────────────────────────────────────────
        print(f"[INFO] Copying GT images -> {gt_dir}")
        copy_gt_images(test_cams, gt_dir)

    else:
        print("[INFO] --skip_render: reusing existing render/ and gt_render/.")

    # ── mask GT only, build pairs ────────────────────────────────────────────
    mask_dir = find_instance_mask_dir(args.scan_path)
    print(f"[INFO] Using instance masks from {mask_dir}")

    os.makedirs(mask_gt_dir, exist_ok=True)
    if args.mask_pred:
        os.makedirs(mask_pred_dir, exist_ok=True)

    pairs = []
    for name in image_names:
        pred_path = os.path.join(render_dir, f"{name}.png")
        gt_path   = os.path.join(gt_dir,     f"{name}.png")
        if not os.path.isfile(pred_path):
            print(f"[WARN] Missing pred render for {name}, skipping")
            continue
        if not os.path.isfile(gt_path):
            print(f"[WARN] Missing GT image for {name}, skipping")
            continue

        gt    = _read_image_rgb(gt_path)
        H, W  = gt.shape[:2]
        mask_path = _resolve_mask_path(mask_dir, name)
        if mask_path is None:
            print(f"[WARN] Missing mask for {name}, skipping")
            continue
        mask = _read_mask(mask_path, (H, W))

        # Mask GT
        gt_m   = apply_mask_to_image(gt, mask)
        out_gt = os.path.join(mask_gt_dir, f"{name}.png")
        cv2.imwrite(out_gt, cv2.cvtColor(gt_m, cv2.COLOR_RGB2BGR))

        # Optionally mask pred
        if args.mask_pred:
            pred_img = _read_image_rgb(pred_path)
            pred_m   = apply_mask_to_image(pred_img, mask)
            out_pred = os.path.join(mask_pred_dir, f"{name}.png")
            cv2.imwrite(out_pred, cv2.cvtColor(pred_m, cv2.COLOR_RGB2BGR))
            pairs.append((out_pred, out_gt))
        else:
            pairs.append((pred_path, out_gt))
    if not pairs:
        raise RuntimeError("No valid pred/masked-gt pairs found; cannot compute metrics.")

    # ── metrics ──────────────────────────────────────────────────────────────
    metrics = compute_metrics(pairs)
    print(f"[METRICS] {metrics}")

    _write_metrics_txt(
        metrics_path,
        metrics,
        meta={
            "scene":            scan_name,
            "engine":           args.engine,
            "samples":          args.samples,
            "n_views":          len(image_names),
            "exclude_meshes":   args.exclude_mesh_names,
        },
    )
    print(f"[INFO] Wrote {metrics_path}")

    if args.update_overview:
        generate_overview_md(exps_root,
                             filename=f"metrics_render_scene{_mask_suffix}.txt",
                             output_name=overview_filename)


if __name__ == "__main__":
    main()
