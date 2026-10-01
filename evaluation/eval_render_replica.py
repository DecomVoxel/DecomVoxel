"""Replica rendering evaluation (PSNR / SSIM / LPIPS) for predicted scene GLBs.

Predicted scene layout:
    <exp_root>/<scene>/scene_combined.glb

Two GT modes:
    --gt_mode mesh    : render <dataset>/GTmesh/<scene>/scene_mesh_texture.glb in Blender
    --gt_mode image   : copy real test images from <dataset>/<scene>/images (sub-sampled)

Two render modes (apply to BOTH pred and gt-mesh renders):
    --render_mode light     : top area light + Principled BSDF base color
    --render_mode emission  : replace shaders with Emission(base_color)

Cameras:
    Read from COLMAP at <dataset>/<scene>/sparse[/0]; pick test cameras as every
    `--test_every`-th sorted view (matches GeoSVR DataPack default of 8).

Outputs (per scene):
    <exp_root>/<scene>/render/<name>.png              predicted renders
    <exp_root>/<scene>/gt_render/<name>.png           [gt_mode=image] copied gt images
    <exp_root>/<scene>/gt_combine_render/<name>.png   [gt_mode=mesh]  rendered gt scene mesh
    <exp_root>/<scene>/mask_pred/<name>.png           masked predicted (white outside instance mask)
    <exp_root>/<scene>/mask_gt/<name>.png             [gt_mode=image] masked gt image
    <exp_root>/<scene>/mask_gt_combine/<name>.png     [gt_mode=mesh]  masked gt mesh render
    <exp_root>/<scene>/metrics_render_<gt_mode>_<render_mode>.txt   scalar metrics

Plus an aggregated report:
    <exps_root>/overview_metrics_render_<gt_mode>_<render_mode>.md
"""

from __future__ import annotations

import argparse
import glob as _glob
import os
import shutil
import sys
from glob import glob

import cv2
import imageio
import numpy as np
import trimesh

# Local utils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.colmap_io import load_test_cameras  # noqa: E402
from utils.blender_invoke import run_blender_worker  # noqa: E402
from utils.render_metrics import (  # noqa: E402
    build_masked_pairs,
    compute_metrics,
    find_instance_mask_dir,
    _read_image_rgb,
    _read_mask,
    _resolve_mask_path,
    apply_mask_to_image,
)


WORKER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "utils", "blender_render_worker.py")
MERGE_WORKER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "utils", "blender_merge_ply_worker.py")

DEBUG=True


# --------------------------------------------------------------------------
def _composite_white_bg(rgba_path: str, out_path: str) -> None:
    """Read RGBA PNG, composite over white background, write RGB PNG."""
    rgba = imageio.imread(rgba_path)           # H,W,4  uint8
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
    rgb   = rgba[:, :, :3].astype(np.float32)
    white = np.full_like(rgb, 255.0)
    result = (rgb * alpha + white * (1.0 - alpha)).clip(0, 255).astype(np.uint8)
    imageio.imwrite(out_path, result)


# --------------------------------------------------------------------------
def build_render_jobs(
    pred_glb: str,
    pred_out_dir: str,
    gt_glb: str | None,
    gt_out_dir: str | None,
    cameras: list[dict],
    render_mode: str,
    add_floor: bool,
    engine: str,
    samples: int,
    device: str,
) -> list[dict]:
    jobs = []
    if pred_glb is not None:
        jobs.append({
            "tag": "pred",
            "glb_path": os.path.abspath(pred_glb),
            "output_dir": os.path.abspath(pred_out_dir),
            "render_mode": render_mode,
            "add_floor": add_floor,
            "engine": engine,
            "samples": samples,
            "device": device,
            "cameras": cameras,
        })
    if gt_glb is not None and gt_out_dir is not None:
        jobs.append({
            "tag": "gt_render",
            "glb_path": os.path.abspath(gt_glb),
            "output_dir": os.path.abspath(gt_out_dir),
            "render_mode": render_mode,
            "add_floor": add_floor,
            "engine": engine,
            "samples": samples,
            "device": device,
            "cameras": cameras,
        })
    return jobs


def build_gt_scene_color_glb(gt_scene_dir: str, output_glb_path: str,
                             pattern: str = "obj_*_colored.ply",
                             blender_install_root: str | None = None) -> str:
    """Combine all `obj_*_colored.ply` under gt_scene_dir into a single GLB,
    using Blender to preserve per-vertex color attributes (trimesh's concatenate
    drops them on heterogeneous inputs).

    Skips if the output already exists.
    Returns the output path.
    """
    if os.path.isfile(output_glb_path):
        print(f"[INFO] Reusing existing combined GT GLB: {output_glb_path}")
        return output_glb_path

    ply_paths = sorted(_glob.glob(os.path.join(gt_scene_dir, pattern)))
    if not ply_paths:
        raise FileNotFoundError(
            f"No '{pattern}' files found under {gt_scene_dir}; cannot build combined GT GLB."
        )

    print(f"[INFO] Merging {len(ply_paths)} GT object meshes via Blender -> {output_glb_path}")
    os.makedirs(os.path.dirname(output_glb_path), exist_ok=True)
    cfg = {
        "ply_paths": [os.path.abspath(p) for p in ply_paths],
        "output_glb": os.path.abspath(output_glb_path),
    }
    rc = run_blender_worker(MERGE_WORKER_SCRIPT, cfg, install_root=blender_install_root)
    if rc != 0:
        raise RuntimeError(f"Blender merge worker exited with code {rc}")
    if not os.path.isfile(output_glb_path):
        raise RuntimeError(f"Merge worker finished but output missing: {output_glb_path}")
    print(f"[INFO] Wrote {output_glb_path}")
    return output_glb_path


def copy_gt_images(test_cams: list[dict], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for cam in test_cams:
        src = cam["image_path"]
        if not os.path.isfile(src):
            print(f"[WARN] missing source image: {src}")
            continue
        dst = os.path.join(out_dir, f"{cam['image_name']}.png")
        # Re-encode if not png to keep extension uniform
        if src.lower().endswith(".png"):
            shutil.copy2(src, dst)
        else:
            import cv2
            img = cv2.imread(src, cv2.IMREAD_UNCHANGED)
            cv2.imwrite(dst, img)


def write_metrics_txt(path: str, metrics: dict, meta: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for k, v in meta.items():
            f.write(f"# {k}: {v}\n")
        for k in ("PSNR", "SSIM", "LPIPS", "Count"):
            if k in metrics:
                f.write(f"{k}: {metrics[k]}\n")


# ---------------------------- overview --------------------------------------
def _parse_metrics_txt(path: str) -> dict:
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            k, v = line.split(":", 1)
            k = k.strip(); v = v.strip()
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
    return out


def generate_overview_md(
    exps_root: str,
    filename: str = "metrics_render.txt",
    output_name: str = "overview_metrics_render.md",
) -> str | None:
    files = sorted(glob(os.path.join(exps_root, "*", filename)))
    if not files:
        print(f"[WARN] No '{filename}' found under {exps_root}")
        return None

    rows = []
    for fp in files:
        scene = os.path.basename(os.path.dirname(fp))
        m = _parse_metrics_txt(fp)
        rows.append((scene, m))

    keys = ["PSNR", "SSIM", "LPIPS", "Count"]
    out_path = os.path.join(exps_root, output_name)
    avg = {k: 0.0 for k in keys}
    avg_n = {k: 0 for k in keys}
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# Replica Render Eval Overview ({filename})\n\n")
        f.write("| scene | " + " | ".join(keys) + " |\n")
        f.write("|" + "---|" * (len(keys) + 1) + "\n")
        for scene, m in rows:
            cells = []
            for k in keys:
                v = m.get(k, "-")
                if isinstance(v, float):
                    cells.append(f"{v:.4f}" if k != "Count" else f"{int(v)}")
                    avg[k] += v
                    avg_n[k] += 1
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


# ------------------------------- main ---------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Replica rendering eval (PSNR/SSIM/LPIPS).")
    parser.add_argument("--scan_path", type=str, required=True,
                        help="Path to dataset scene dir, e.g. datasets/Replica/scan1")
    parser.add_argument("--exp_path", type=str, required=True,
                        help="Path to predicted exp dir, e.g. exps/Replica-ab-app-0428/scan1")
    parser.add_argument("--pred_glb_name", type=str, default="scene_combined.glb")
    parser.add_argument("--gt_mode", type=str, default="image", choices=["mesh", "image"])
    parser.add_argument("--gt_mesh_root", type=str, default=None,
                        help="Root containing <gt_mesh_root>/<scene>/{obj_*_colored.ply, scene_mesh_color.glb}. "
                             "Defaults to <scan_path>/../GTmesh")
    parser.add_argument("--gt_mesh_name", type=str, default="scene_mesh_color.glb",
                        help="Combined GT scene GLB filename (built from obj_*_colored.ply if missing).")
    parser.add_argument("--gt_obj_pattern", type=str, default="obj_*_colored.ply",
                        help="Glob pattern for per-object GT PLY files used to build the combined GLB.")
    parser.add_argument("--render_mode", type=str, default="light", choices=["light", "emission"])
    parser.add_argument("--add_floor", action="store_true", default=True)
    parser.add_argument("--no_add_floor", action="store_false", dest="add_floor")
    parser.add_argument("--engine", type=str, default="CYCLES", choices=["CYCLES", "EEVEE"])
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--device", type=str, default="GPU", choices=["GPU", "CPU"])
    parser.add_argument("--test_every", type=int, default=8)
    parser.add_argument("--images_subdir", type=str, default="images")
    parser.add_argument("--blender_install_root", type=str, default=None)
    parser.add_argument("--exps_root", type=str, default="exps/Replica",
                        help="Root over which overview_metrics_render.md is generated.")
    parser.add_argument("--update_overview", action="store_true", default=True)
    parser.add_argument("--no_update_overview", action="store_false", dest="update_overview")
    parser.add_argument("--skip_render", action="store_true",
                        help="Skip Blender rendering; reuse existing render/ + gt_render/ dirs.")
    parser.add_argument("--force_rerun", action="store_true", default=True,
                        help="Clear all image output dirs before running to avoid stale results (default: on).")
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

    scan_name = os.path.basename(os.path.normpath(args.scan_path))
    pred_glb = os.path.join(args.exp_path, args.pred_glb_name)
    if not os.path.isfile(pred_glb):
        raise FileNotFoundError(f"Predicted GLB not found: {pred_glb}")

    # Output directories under exp_path. GT-mesh path uses *_combine suffix to keep
    # gt-image and gt-mesh outputs side-by-side without overwriting each other.
    pred_out_dir = os.path.join(args.exp_path, "render")
    if args.gt_mode == "mesh":
        gt_out_dir = os.path.join(args.exp_path, "gt_combine_render")
        mask_gt_dir = os.path.join(args.exp_path, "mask_gt_combine")
    else:
        gt_out_dir = os.path.join(args.exp_path, "gt_render")
        mask_gt_dir = os.path.join(args.exp_path, "mask_gt")
    mask_pred_dir = os.path.join(args.exp_path, "mask_pred")  # kept for compat; not used in metrics
    _mask_suffix = "_mask" if args.mask_pred else ""
    metrics_filename = f"metrics_render_{args.gt_mode}_{args.render_mode}{_mask_suffix}.txt"
    overview_filename = f"overview_metrics_render_{args.gt_mode}_{args.render_mode}{_mask_suffix}.md"
    metrics_path = os.path.join(args.exp_path, metrics_filename)

    # Clear stale image output dirs so partial/old renders never pollute metrics.
    if args.force_rerun and not args.skip_render:
        _image_dirs = [
            pred_out_dir, gt_out_dir,
            os.path.join(args.exp_path, "gt_render"),
            os.path.join(args.exp_path, "gt_combine_render"),
            mask_pred_dir,
            os.path.join(args.exp_path, "mask_pred"),
            os.path.join(args.exp_path, "mask_gt"),
            os.path.join(args.exp_path, "mask_gt_combine"),
        ]
        for d in dict.fromkeys(_image_dirs):  # deduplicate, preserve order
            if os.path.isdir(d):
                shutil.rmtree(d)
                print(f"[force_rerun] Cleared {d}")

    # Cameras
    print(f"[INFO] Loading COLMAP cameras from {args.scan_path}")
    test_cams, total = load_test_cameras(args.scan_path, images_subdir=args.images_subdir,
                                         test_every=args.test_every)
    print(f"[INFO] {len(test_cams)} / {total} cameras selected as test (every {args.test_every}).")
    cam_payload = [
        {
            "image_name": c["image_name"],
            "W": c["W"], "H": c["H"],
            "K": c["K"], "w2c": c["w2c"],
        } for c in test_cams
    ]

    # GT source
    gt_glb = None
    if args.gt_mode == "mesh":
        gt_root = args.gt_mesh_root or os.path.join(os.path.dirname(os.path.normpath(args.scan_path)), "GTmesh")
        gt_scene_dir = os.path.join(gt_root, scan_name)
        gt_glb = os.path.join(gt_scene_dir, args.gt_mesh_name)
        if not os.path.isfile(gt_glb):
            # Get scene_mesh_color.glb
            gt_glb = build_gt_scene_color_glb(gt_scene_dir, gt_glb,
                                              pattern=args.gt_obj_pattern,
                                              blender_install_root=args.blender_install_root)
            
    # Render
    if not args.skip_render:
        jobs = build_render_jobs(
            pred_glb=pred_glb,
            pred_out_dir=pred_out_dir,
            gt_glb=gt_glb,
            gt_out_dir=gt_out_dir if args.gt_mode == "mesh" else None,
            cameras=cam_payload,
            render_mode=args.render_mode,
            add_floor=args.add_floor,
            engine=args.engine,
            samples=args.samples,
            device=args.device,
        )
        rc = run_blender_worker(WORKER_SCRIPT, {"jobs": jobs}, install_root=args.blender_install_root)
        if rc != 0:
            raise RuntimeError(f"Blender worker exited with code {rc}")

        # ── transparent → white compositing ──────────────────────────────
        print("[INFO] Compositing transparent background -> white ...")
        for cam in test_cams:
            for out_dir_c in [pred_out_dir] + ([gt_out_dir] if args.gt_mode == "mesh" else []):
                p = os.path.join(out_dir_c, f"{cam['image_name']}.png")
                if os.path.isfile(p):
                    _composite_white_bg(p, p)

        if args.gt_mode == "image":
            print(f"[INFO] Copying GT images -> {gt_out_dir}")
            copy_gt_images(test_cams, gt_out_dir)
    else:
        print("[INFO] --skip_render set; reusing existing renders.")

    # Mask GT only; pred is used as-is (raw render, no mask).
    mask_dir = find_instance_mask_dir(args.scan_path)
    print(f"[INFO] Using instance masks from {mask_dir}")
    image_names = [c["image_name"] for c in test_cams]

    os.makedirs(mask_gt_dir, exist_ok=True)
    if args.mask_pred:
        os.makedirs(mask_pred_dir, exist_ok=True)
    pairs = []
    for name in image_names:
        pred_path = os.path.join(pred_out_dir, f"{name}.png")
        gt_path   = os.path.join(gt_out_dir,   f"{name}.png")
        if not os.path.isfile(pred_path):
            print(f"[WARN] Missing pred render for {name}, skipping")
            continue
        if not os.path.isfile(gt_path):
            print(f"[WARN] Missing GT image for {name}, skipping")
            continue

        gt  = _read_image_rgb(gt_path)
        H, W = gt.shape[:2]
        mask_path = _resolve_mask_path(mask_dir, name)
        if mask_path is None:
            print(f"[WARN] Missing mask for {name}, skipping")
            continue
        mask = _read_mask(mask_path, (H, W))

        # Mask GT
        gt_m = apply_mask_to_image(gt, mask)
        out_gt = os.path.join(mask_gt_dir, f"{name}.png")
        cv2.imwrite(out_gt, cv2.cvtColor(gt_m, cv2.COLOR_RGB2BGR))

        # Optionally mask pred
        if args.mask_pred:
            pred_img = _read_image_rgb(pred_path)
            pred_m = apply_mask_to_image(pred_img, mask)
            out_pred = os.path.join(mask_pred_dir, f"{name}.png")
            cv2.imwrite(out_pred, cv2.cvtColor(pred_m, cv2.COLOR_RGB2BGR))
            pairs.append((out_pred, out_gt))
        else:
            pairs.append((pred_path, out_gt))
    if not pairs:
        raise RuntimeError("No valid pred/gt/mask triples found; cannot compute metrics.")
    metrics = compute_metrics(pairs)
    print(f"[METRICS] {metrics}")
    write_metrics_txt(
        metrics_path,
        metrics,
        meta={
            "scene": scan_name,
            "gt_mode": args.gt_mode,
            "render_mode": args.render_mode,
            "engine": args.engine,
            "samples": args.samples,
            "test_every": args.test_every,
            "n_views": len(image_names),
        },
    )
    print(f"[INFO] Wrote {metrics_path}")

    if args.update_overview:
        generate_overview_md(args.exps_root, filename=metrics_filename, output_name=overview_filename)


if __name__ == "__main__":
    main()
