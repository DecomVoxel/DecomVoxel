"""
Evaluate background mesh geometry for Replica scenes.

Pred: <pred_root>/<scene>/bg.ply          (default: exps/Replica-bg)
GT:   <gtmesh_root>/<scene>/bg_repair.ply (default: datasets/Replica/GTmesh)

Usage:
    # Evaluate all scenes and write overview
    python evaluation/eval_bg_geo.py

    # Single scene
    python evaluation/eval_bg_geo.py --scene scan1

    # Custom roots
    python evaluation/eval_bg_geo.py \
        --pred_root exps/Replica-bg \
        --gtmesh_root datasets/Replica/GTmesh
"""

import argparse
import glob
import os

import numpy as np
import open3d as o3d
import trimesh
from sklearn.neighbors import KDTree


# ---------------------------------------------------------------------------
# Core evaluation helpers (shared with eval_geo_replica.py)
# ---------------------------------------------------------------------------

def nn_correspondance(verts1, verts2):
    if len(verts1) == 0 or len(verts2) == 0:
        return [], []
    kdtree = KDTree(verts1)
    distances, indices = kdtree.query(verts2)
    return distances.reshape(-1), indices


def evaluate(mesh_pred, mesh_trgt, threshold=0.05, down_sample=0.02):
    pcd_trgt = o3d.geometry.PointCloud()
    pcd_pred = o3d.geometry.PointCloud()
    pcd_trgt.points = o3d.utility.Vector3dVector(mesh_trgt.vertices[:, :3])
    pcd_pred.points = o3d.utility.Vector3dVector(mesh_pred.vertices[:, :3])

    if down_sample:
        pcd_pred = pcd_pred.voxel_down_sample(down_sample)
        pcd_trgt = pcd_trgt.voxel_down_sample(down_sample)

    verts_pred = np.asarray(pcd_pred.points)
    verts_trgt = np.asarray(pcd_trgt.points)

    dist1, _ = nn_correspondance(verts_pred, verts_trgt)
    dist2, _ = nn_correspondance(verts_trgt, verts_pred)

    precision = np.mean((dist2 < threshold).astype(float))
    recal     = np.mean((dist1 < threshold).astype(float))
    den = precision + recal
    fscore = 2 * precision * recal / den if den > 0 else 0.0

    n = 200000
    pointcloud_pred, idx = mesh_pred.sample(n, return_index=True)
    pointcloud_pred = pointcloud_pred.astype(np.float32)
    normal_pred = mesh_pred.face_normals[idx]

    pointcloud_trgt, idx = mesh_trgt.sample(n, return_index=True)
    pointcloud_trgt = pointcloud_trgt.astype(np.float32)
    normal_trgt = mesh_trgt.face_normals[idx]

    _, index1 = nn_correspondance(pointcloud_pred, pointcloud_trgt)
    _, index2 = nn_correspondance(pointcloud_trgt, pointcloud_pred)

    normal_acc         = np.abs((normal_pred * normal_trgt[index2.reshape(-1)]).sum(axis=-1)).mean()
    normal_comp        = np.abs((normal_trgt * normal_pred[index1.reshape(-1)]).sum(axis=-1)).mean()
    normal_consistency = 0.5 * (normal_acc + normal_comp)

    return {
        "Acc":                np.mean(dist2),
        "Comp":               np.mean(dist1),
        "Chamfer-L1":         0.5 * (np.mean(dist2) + np.mean(dist1)),
        "Prec":               precision,
        "Recal":              recal,
        "F-score":            fscore,
        "Normal-Acc":         normal_acc,
        "Normal-Comp":        normal_comp,
        "Normal-Consistency": normal_consistency,
    }


def load_mesh(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Mesh file not found: {path}")
    mesh = trimesh.load(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    return mesh


# ---------------------------------------------------------------------------
# Overview table
# ---------------------------------------------------------------------------

def _fmt(v):
    return f"{v:.6f}" if isinstance(v, (float, np.floating)) else str(v)


def generate_overview_md(pred_root, metric_filename="metrics_bg.txt",
                         output_name="overview_metrics_bg.md"):
    pattern = os.path.join(pred_root, "*", metric_filename)
    files = sorted(glob.glob(pattern))
    if not files:
        return None

    priority = [
        "Acc", "Comp", "Chamfer-L1", "Prec", "Recal", "F-score",
        "Normal-Acc", "Normal-Comp", "Normal-Consistency",
    ]

    rows = []
    all_keys = set()
    for fp in files:
        scene = os.path.basename(os.path.dirname(fp))
        metrics = {}
        with open(fp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or ":" not in line:
                    continue
                k, v = line.split(":", 1)
                try:
                    metrics[k.strip()] = float(v.strip())
                except ValueError:
                    metrics[k.strip()] = v.strip()
        rows.append((scene, metrics))
        all_keys.update(metrics.keys())

    ordered_keys = [k for k in priority if k in all_keys] + sorted(
        k for k in all_keys if k not in priority
    )

    def _sort_key(x):
        num = "".join(ch for ch in x[0] if ch.isdigit())
        return int(num) if num else 10**9

    rows.sort(key=_sort_key)

    avg = {}
    for k in ordered_keys:
        vals = [float(m[k]) for _, m in rows if isinstance(m.get(k), (int, float, np.floating))]
        if vals:
            avg[k] = float(np.mean(vals))

    overview_path = os.path.join(pred_root, output_name)
    with open(overview_path, "w", encoding="utf-8") as f:
        f.write("# Replica Background Geometry Evaluation\n\n")
        f.write(f"Scanned files: `{pattern}`\n\n")

        headers = ["Scan"] + ordered_keys
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for scene, metrics in rows:
            vals = [scene] + [_fmt(metrics.get(k, "")) for k in ordered_keys]
            f.write("| " + " | ".join(vals) + " |\n")
        avg_vals = ["Dataset-Average"] + [_fmt(avg.get(k, "")) for k in ordered_keys]
        f.write("| " + " | ".join(avg_vals) + " |\n")

    return overview_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Background mesh geometry evaluation for Replica")
    parser.add_argument("--pred_root",   type=str, default="exps/Replica-bg",
                        help="Root with <scene>/bg.ply predicted meshes.")
    parser.add_argument("--gtmesh_root", type=str, default="datasets/Replica/GTmesh",
                        help="Root with <scene>/bg_repair.ply GT meshes.")
    parser.add_argument("--scene",       type=str, default=None, nargs="+",
                        help="Evaluate only these scene(s) (e.g. scan1 scan2). Default: all scenes.")
    parser.add_argument("--pred_name",   type=str, default="bg.ply",
                        help="Predicted mesh filename inside <pred_root>/<scene>/.")
    parser.add_argument("--gt_name",     type=str, default="bg_repair.ply",
                        help="GT mesh filename inside <gtmesh_root>/<scene>/.")
    parser.add_argument("--threshold",   type=float, default=0.05)
    parser.add_argument("--down_sample", type=float, default=0.02)
    parser.add_argument("--metric_file", type=str, default="metrics_bg.txt",
                        help="Per-scene metric output filename.")
    parser.add_argument("--overview",    type=str, default="overview_metrics_bg.md")
    args = parser.parse_args()

    pred_root   = os.path.abspath(args.pred_root)
    gtmesh_root = os.path.abspath(args.gtmesh_root)

    if args.scene:
        scenes = list(args.scene)
    else:
        scenes = sorted(
            d for d in os.listdir(pred_root)
            if os.path.isdir(os.path.join(pred_root, d))
        )

    print(f"Pred root  : {pred_root}")
    print(f"GT root    : {gtmesh_root}")
    print(f"Scenes     : {scenes}\n")

    success, skipped = 0, 0
    for scene in scenes:
        pred_path = os.path.join(pred_root, scene, args.pred_name)
        gt_path   = os.path.join(gtmesh_root, scene, args.gt_name)
        out_txt   = os.path.join(pred_root, scene, args.metric_file)

        print(f"[{scene}]")
        try:
            mesh_pred = load_mesh(pred_path)
            mesh_gt   = load_mesh(gt_path)
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
            skipped += 1
            continue

        metrics = evaluate(mesh_pred, mesh_gt,
                           threshold=args.threshold, down_sample=args.down_sample)
        print("  " + "  ".join(f"{k}: {v:.4f}" for k, v in metrics.items()))

        with open(out_txt, "w", encoding="utf-8") as f:
            for k, v in metrics.items():
                f.write(f"{k}: {v}\n")
        print(f"  -> {out_txt}")
        success += 1

    print(f"\nDone. {success} scene(s) evaluated, {skipped} skipped.")

    overview_path = generate_overview_md(
        pred_root=pred_root,
        metric_filename=args.metric_file,
        output_name=args.overview,
    )
    if overview_path:
        print(f"Overview   : {overview_path}")


if __name__ == "__main__":
    main()
