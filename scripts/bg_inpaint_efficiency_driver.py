"""bg_inpaint_efficiency_driver.py

Runs the bg-inpaint pipeline for a single scene, recording per-stage timing
via PipelineLogger and background GPU usage via GpuMonitor (both built into
decomvoxel/utils/).

Invoked by bg_inpaint_replica_para.sh. Mirrors run_one_scene() from
bg_inpaint_replica.sh but does NOT modify any of those original files.

Logs are written to <out_dir>/log/bg_inpaint_<timestamp>.{log,json}
GPU usage JSON + plot go to the same log sub-directory.
"""

import argparse
import os
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from decomvoxel.utils.pipeline_logger import PipelineLogger


def _run(cmd: list, env: dict) -> None:
    """Run a subprocess command; raise RuntimeError on non-zero exit."""
    print(f"[CMD] {' '.join(str(c) for c in cmd)}", flush=True)
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed (rc={result.returncode}): {' '.join(str(c) for c in cmd)}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="bg-inpaint pipeline with timing + GPU monitoring."
    )
    ap.add_argument("--scene",    default="scan1",
                    help="Scene name, e.g. scan1 (default: %(default)s)")
    ap.add_argument("--gpu",      default="0",
                    help="CUDA_VISIBLE_DEVICES value to pass to subprocesses (default: %(default)s)")
    ap.add_argument("--ckpt_root", default="outputs/bg_replica",
                    help="Root directory holding trained checkpoints (default: %(default)s)")
    ap.add_argument("--out_root",  default="outputs/bg_replica_efficiency",
                    help="Root output directory (default: %(default)s)")
    ap.add_argument("--exp_name",  default="bg_training",
                    help="Experiment name passed to train_bg_refine (default: %(default)s)")
    ap.add_argument("--gpu_sample_interval", type=float, default=10.0,
                    help="GPU sampling interval in seconds (default: %(default)s)")
    args = ap.parse_args()

    ckpt_dir = os.path.join(args.ckpt_root, args.scene)
    out_dir  = os.path.join(args.out_root,  args.scene)
    os.makedirs(out_dir, exist_ok=True)

    # Subprocesses inherit the full environment but with GPU pinned.
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu

    py = sys.executable  # same interpreter as caller

    logger = PipelineLogger(
        model_path=out_dir,
        cli_args=vars(args),
        tag="bg_inpaint",
        gpu_sample_interval=args.gpu_sample_interval,
    )

    try:
        guidance_root = os.path.join(out_dir, "see3d_guidance_views")

        # ── Stage 1: cubemap reference views ────────────────────────────────
        logger.start_step("Stage 1: get_cubemap_ref_views")
        _run([
            py,
            "decomvoxel/representation/GeoSVR/bg_inpaint/get_cubemap_ref_views.py",
            "--checkpoint",        os.path.join(ckpt_dir, "checkpoints", "iter020000_model.pt"),
            "--config",            os.path.join(ckpt_dir, "config.yaml"),
            "--output",            out_dir,
            "--dilate_object_mask", "5",
            "--retry_times",        "20",
            "--see3d_fov_deg",      "80.0",
        ], env)
        logger.end_step()

        # ── Stage 2.1: inpaint see3d views ──────────────────────────────────
        logger.start_step("Stage 2.1: infer_see3d")
        _run([
            py,
            "decomvoxel/representation/GeoSVR/bg_inpaint/See3D/infer_see3d.py",
            "--root_dir", out_dir,
        ], env)
        logger.end_step()

        # ── Stage 2.2: depth + normal maps ──────────────────────────────────
        logger.start_step("Stage 2.2: dn_util")
        _run([
            py,
            "decomvoxel/representation/GeoSVR/bg_inpaint/dn_util.py",
            "--guidance_root", guidance_root,
        ], env)
        logger.end_step()

        # ── Stage 2.3: 2D plane masks ────────────────────────────────────────
        logger.start_step("Stage 2.3: plane_util")
        _run([
            py,
            "decomvoxel/representation/GeoSVR/bg_inpaint/plane_util.py",
            "--data_root",       os.path.join(guidance_root, "see3d_mono_dn"),
            "--soft_bg_plane_det",
        ], env)
        logger.end_step()

        # ── Stage 2.4: merge 3D planes ───────────────────────────────────────
        logger.start_step("Stage 2.4: merge_3D_plane")
        _run([
            py,
            "decomvoxel/representation/GeoSVR/bg_inpaint/merge_3D_plane.py",
            "--checkpoint",    os.path.join(ckpt_dir, "checkpoints", "iter020000_model.pt"),
            "--config",        os.path.join(ckpt_dir, "config.yaml"),
            "--guidance_root", guidance_root,
        ], env)
        logger.end_step()

        # ── Stage 2.5.1: train bg voxel model ───────────────────────────────
        logger.start_step("Stage 2.5.1: train_bg_refine")
        _run([
            py,
            "decomvoxel/representation/GeoSVR/train_bg_refine.py",
            "--cfg_files",     "configs/config_bg.yaml",
            "--guidance_root", guidance_root,
            "--exp_name",      args.exp_name,
        ], env)
        logger.end_step()

        # ── Stage 2.5.2: extract bg mesh ────────────────────────────────────
        logger.start_step("Stage 2.5.2: tsdf_mesh_see3d_bg")
        _run([
            py,
            "decomvoxel/representation/GeoSVR/mesh_extract/tsdf_mesh_see3d_bg.py",
            os.path.join(guidance_root, args.exp_name),
            "--use_depth_filter",
            "--voxel_size",       "0.01",
            "--max_depth",        "20",
            "--sdf_trunc_scale",  "4.0",
        ], env)
        logger.end_step()

    finally:
        # Stops GPU monitor, prints mean/peak util+mem, flushes all logs.
        logger.finalize()


if __name__ == "__main__":
    main()
