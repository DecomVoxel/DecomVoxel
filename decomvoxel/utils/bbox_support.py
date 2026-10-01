"""
bbox_support.py – Bounding-box utility helpers.
"""

import os
import glob
import sys
from typing import Optional
import torch


def get_min_z(voxel_dir: str, device: str = "cpu",
              uncertainty_threshold: float = 0.005) -> Optional[float]:
    _PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    _GEOSVR_ROOT = os.path.join(_PROJECT_ROOT, 'decomvoxel', 'representation', 'GeoSVR')
    for _p in [_PROJECT_ROOT, _GEOSVR_ROOT, os.path.join(_GEOSVR_ROOT, 'src')]:
        if _p not in sys.path:
            sys.path.insert(0, _p)

    pattern = os.path.join(voxel_dir, "*_voxels.pt")
    voxel_files = sorted(glob.glob(pattern))

    if not voxel_files:
        print(f"[get_min_z] No '*_voxels.pt' files found in: {voxel_dir}")
        return None

    print(f"[get_min_z] Scanning {len(voxel_files)} voxel file(s) for global min-z …")

    global_min_z = float("inf")

    from yacs.config import CfgNode
    from decomvoxel.representation.GeoSVR.src.sparse_voxel_model import SparseVoxelModel

    for voxel_path in voxel_files:
        # Skip background object (object_255)
        basename = os.path.basename(voxel_path)
        if "object_255" in basename:
            print(f"  Skipping background object: {basename}")
            continue
        try:
            cfg_model = CfgNode()
            cfg_model.model_path       = os.path.dirname(voxel_path)
            cfg_model.vox_geo_mode     = "triinterp1"
            cfg_model.density_mode     = "exp_linear_11"
            cfg_model.sh_degree        = 3
            cfg_model.ss               = 1.5
            cfg_model.outside_level    = 5
            cfg_model.white_background = True
            cfg_model.black_background = False

            voxel_model = SparseVoxelModel(cfg_model)
            voxel_model.load(voxel_path)

            centers = voxel_model.vox_center.float()   # (N, 3)

            # Uncertainty-based filtering (mirrors convert_voxel_to_sparse_structure)
            centers_certain = centers
            if uncertainty_threshold is not None and uncertainty_threshold > 0.0:
                sizes  = voxel_model.vox_size.float().squeeze(-1)   # (N,)
                levels = voxel_model.octlevel.float()
                if levels.dim() > 1:
                    levels = levels.squeeze(-1)
                u_base = sizes / (1.0 * (levels + 1.0))             # (N,)
                u_min, u_max = u_base.min(), u_base.max()
                u_norm = (u_base - u_min) / (u_max - u_min + 1e-8)  # (N,) in [0,1]
                keep_mask = u_norm <= uncertainty_threshold
                n_before = len(centers)
                centers_certain = centers[keep_mask]
                print(f"    uncertainty filter: {keep_mask.sum().item()}/{n_before} voxels kept "
                      f"(threshold={uncertainty_threshold})")

            min_z_obj = centers_certain[:, 2].min().item()
            global_min_z = min(global_min_z, min_z_obj)
            print(f"  {os.path.basename(voxel_path)}: min_z = {min_z_obj:.4f}")
        except Exception as e:
            print(f"  [get_min_z] WARNING: failed to load {voxel_path}: {e}")

    if global_min_z == float("inf"):
        print("[get_min_z] Could not determine min_z (all files failed to load).")
        return None

    print(f"[get_min_z] Global min_z = {global_min_z:.4f}")
    return global_min_z


def chair_bbox_xy_modify(
    voxel_dir: str,
    categories_json: str,
    uncertainty_threshold: float = 0.1,
    device: str = "cpu",
    robust_percentile: float = 0.02,
    chair_proximity_padding: float = 0.5,
    table_proximity_padding: float = 0.2,
    expand_ratio: float = 0.2,
    absolute_gap_threshold: float = 0.3,
    vis_output_dir: str = None,
) -> dict:
    """
    For each chair / dining-chair in *voxel_dir*, find the nearest
    table / dining-table.  If their XY-projected bboxes overlap, or if the
    chair bbox padded by *chair_proximity_padding* overlaps the table bbox
    padded by *table_proximity_padding*, expand the chair's bbox toward the
    table by *expand_ratio* of the table's max XY extent.

    Returns
    -------
    dict
        ``{obj_id_str: [dx_min, dx_max, dy_min, dy_max, 0.0, 0.0]}``
        All six values are non-negative magnitudes.
        dx_min  > 0  →  move x_min further left  by that amount.
        dx_max  > 0  →  move x_max further right by that amount.
        z values are always 0.  Non-chair objects map to all-zeros.
    """
    import json as _json

    _PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    _GEOSVR_ROOT = os.path.join(_PROJECT_ROOT, 'decomvoxel', 'representation', 'GeoSVR')
    for _p in [_PROJECT_ROOT, _GEOSVR_ROOT, os.path.join(_GEOSVR_ROOT, 'src')]:
        if _p not in sys.path:
            sys.path.insert(0, _p)

    def _norm_id(k: str) -> str:
        """Convert any of '2', '002', 'object_002' → '2'."""
        k = k.strip()
        if k.startswith("object_"):
            k = k[len("object_"):]
        return str(int(k.lstrip("0") or "0"))

    CHAIR_CATS = {"chair", "dining chair", "dining_chair", "A chair", "a chair"}
    TABLE_CATS = {"table", "dining table", "dining_table", "coffee table", "coffee_table", "A table", "a table"}

    # ── 1. Load categories ────────────────────────────────────────────────
    with open(categories_json) as f:
        raw_cats = _json.load(f)
    categories = {_norm_id(k): v.strip().lower() for k, v in raw_cats.items()}

    # ── 2. Load voxels and compute robust XY bboxes ───────────────────────
    from yacs.config import CfgNode
    from decomvoxel.representation.GeoSVR.src.sparse_voxel_model import SparseVoxelModel

    voxel_files = sorted(glob.glob(os.path.join(voxel_dir, "*_voxels.pt")))
    obj_bboxes = {}   # obj_id_str → {"min": Tensor(3,), "max": Tensor(3,)}

    print(f"[chair_bbox_xy_modify] Loading {len(voxel_files)} voxel file(s) …")
    for voxel_path in voxel_files:
        basename = os.path.basename(voxel_path)
        if "object_255" in basename:
            continue
        raw_id = basename.replace("object_", "").replace("_voxels.pt", "")
        obj_id = str(int(raw_id.lstrip("0") or "0"))
        try:
            cfg_model = CfgNode()
            cfg_model.model_path       = os.path.dirname(voxel_path)
            cfg_model.vox_geo_mode     = "triinterp1"
            cfg_model.density_mode     = "exp_linear_11"
            cfg_model.sh_degree        = 3
            cfg_model.ss               = 1.5
            cfg_model.outside_level    = 5
            cfg_model.white_background = True
            cfg_model.black_background = False

            voxel_model = SparseVoxelModel(cfg_model)
            voxel_model.load(voxel_path)

            centers = voxel_model.vox_center.float()   # (N, 3)

            # Uncertainty filter
            centers_certain = centers
            if uncertainty_threshold is not None and uncertainty_threshold > 0.0:
                sizes  = voxel_model.vox_size.float().squeeze(-1)
                levels = voxel_model.octlevel.float()
                if levels.dim() > 1:
                    levels = levels.squeeze(-1)
                u_base = sizes / (1.0 * (levels + 1.0))
                u_norm = (u_base - u_base.min()) / (u_base.max() - u_base.min() + 1e-8)
                keep = u_norm <= uncertainty_threshold
                if keep.sum() > 0:
                    centers_certain = centers[keep]

            # Robust bbox
            lo, hi = robust_percentile, 1.0 - robust_percentile
            bbox_min = torch.quantile(centers_certain, lo, dim=0).cpu()
            bbox_max = torch.quantile(centers_certain, hi, dim=0).cpu()
            obj_bboxes[obj_id] = {"min": bbox_min, "max": bbox_max}
            print(f"  obj {obj_id} ({categories.get(obj_id, '?')}): "
                  f"xy=[{bbox_min[0]:.3f},{bbox_max[0]:.3f}] x [{bbox_min[1]:.3f},{bbox_max[1]:.3f}]")
        except Exception as e:
            print(f"  [chair_bbox_xy_modify] WARNING: failed to process {voxel_path}: {e}")

    # ── 3. Initialise result dict (all zeros) ─────────────────────────────
    result = {oid: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] for oid in obj_bboxes}

    chair_ids = [oid for oid in obj_bboxes if categories.get(oid, "") in CHAIR_CATS]
    table_ids = [oid for oid in obj_bboxes if categories.get(oid, "") in TABLE_CATS]
    print(f"[chair_bbox_xy_modify] Chairs: {chair_ids}")
    print(f"[chair_bbox_xy_modify] Tables: {table_ids}")

    if not chair_ids or not table_ids:
        print("[chair_bbox_xy_modify] No chairs or no tables found — no modifications.")
        return result

    # ── 4. Per-chair proximity check & bbox expansion ─────────────────────
    for chair_id in chair_ids:
        c_min = obj_bboxes[chair_id]["min"]   # (3,)
        c_max = obj_bboxes[chair_id]["max"]   # (3,)
        c_ext = c_max - c_min                 # (3,)
        c_cx, c_cy = ((c_min + c_max) / 2)[:2].tolist()

        # Padded XY bbox for proximity test
        pad_x = c_ext[0].item() * chair_proximity_padding
        pad_y = c_ext[1].item() * chair_proximity_padding
        cp_xmin = c_min[0].item() - pad_x
        cp_xmax = c_max[0].item() + pad_x
        cp_ymin = c_min[1].item() - pad_y
        cp_ymax = c_max[1].item() + pad_y

        # Nearest table by XY centre distance
        best_table_id, best_dist = None, float("inf")
        for table_id in table_ids:
            t_min = obj_bboxes[table_id]["min"]
            t_max = obj_bboxes[table_id]["max"]
            t_cx, t_cy = ((t_min + t_max) / 2)[:2].tolist()
            dist = ((t_cx - c_cx) ** 2 + (t_cy - c_cy) ** 2) ** 0.5
            if dist < best_dist:
                best_dist, best_table_id = dist, table_id

        if best_table_id is None:
            continue

        t_min = obj_bboxes[best_table_id]["min"]
        t_max = obj_bboxes[best_table_id]["max"]
        t_xmin, t_xmax = t_min[0].item(), t_max[0].item()
        t_ymin, t_ymax = t_min[1].item(), t_max[1].item()
        t_cx, t_cy = ((t_min + t_max) / 2)[:2].tolist()
        t_ext = t_max - t_min  # (3,)

        # Padded table bbox for overlap test
        t_pad_x = t_ext[0].item() * table_proximity_padding
        t_pad_y = t_ext[1].item() * table_proximity_padding
        tp_xmin = t_xmin - t_pad_x
        tp_xmax = t_xmax + t_pad_x
        tp_ymin = t_ymin - t_pad_y
        tp_ymax = t_ymax + t_pad_y

        # XY overlap check (padded chair bbox vs padded table bbox)
        overlap = (cp_xmin <= tp_xmax and cp_xmax >= tp_xmin and
                   cp_ymin <= tp_ymax and cp_ymax >= tp_ymin)

        # Also trigger if the raw bbox gap on BOTH XY axes is within absolute_gap_threshold.
        # Using AND (both axes) avoids false positives for chairs that are far in one axis
        # but happen to be aligned in the other (e.g. a chair directly behind the table).
        gap_x = max(0.0, max(t_xmin - c_max[0].item(), c_min[0].item() - t_xmax))
        gap_y = max(0.0, max(t_ymin - c_max[1].item(), c_min[1].item() - t_ymax))
        # close_enough = gap_x < absolute_gap_threshold and gap_y < absolute_gap_threshold
        close_enough = False  
        

        if not overlap and not close_enough:
            print(f"  chair {chair_id}: nearest table {best_table_id}, "
                  f"dist={best_dist:.3f}, bbox_gap=({gap_x:.3f},{gap_y:.3f}) — no proximity, skip")
            continue

        print(f"  chair {chair_id}: near table {best_table_id}, "
              f"dist={best_dist:.3f}, bbox_gap=({gap_x:.3f},{gap_y:.3f}) — modifying bbox")

        # Direction: from chair centre toward the overlap-region centroid (or nearest
        # point on table bbox when there is no geometric overlap).
        if overlap:
            # Centroid of the padded-bbox intersection rectangle
            ix_min = max(cp_xmin, tp_xmin)
            ix_max = min(cp_xmax, tp_xmax)
            iy_min = max(cp_ymin, tp_ymin)
            iy_max = min(cp_ymax, tp_ymax)
            target_x = (ix_min + ix_max) / 2
            target_y = (iy_min + iy_max) / 2
        else:
            raise NotImplementedError("close_enough without overlap is not implemented yet")
            # Nearest point on raw table bbox to chair centre
            target_x = max(t_xmin, min(c_cx, t_xmax))
            target_y = max(t_ymin, min(c_cy, t_ymax))

        dx = target_x - c_cx
        dy = target_y - c_cy
        d_norm = (dx * dx + dy * dy) ** 0.5
        if d_norm < 1e-6:
            # Fallback to table centre if target coincides with chair centre
            dx = t_cx - c_cx
            dy = t_cy - c_cy
            d_norm = (dx * dx + dy * dy) ** 0.5
        if d_norm < 1e-6:
            print(f"  chair {chair_id}: direction undefined — skip")
            continue

        # Unit direction vector toward the overlap region
        ux, uy = dx / d_norm, dy / d_norm

        # Project expand_dist onto x and y axes; expand the face in the direction of travel
        t_max_xy_extent = max(t_ext[0].item(), t_ext[1].item())
        expand_x = abs(ux) * t_max_xy_extent * expand_ratio
        expand_y = abs(uy) * t_max_xy_extent * expand_ratio

        if ux > 0:
            result[chair_id][1] = expand_x   # expand x_max
        else:
            result[chair_id][0] = expand_x   # expand x_min

        if uy > 0:
            result[chair_id][3] = expand_y   # expand y_max
        else:
            result[chair_id][2] = expand_y   # expand y_min

        print(f"    → direction=({ux:.3f},{uy:.3f}), "
              f"expand x={'max' if ux>0 else 'min'} by {expand_x:.4f}, "
              f"expand y={'max' if uy>0 else 'min'} by {expand_y:.4f}")

    # ── 5. (Optional) Visualise results ──────────────────────────────────
    if vis_output_dir is not None:
        os.makedirs(vis_output_dir, exist_ok=True)
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            from matplotlib.lines import Line2D

            fig, ax = plt.subplots(1, 1, figsize=(12, 12))

            # Padded table bboxes (dashed blue)
            for tid in table_ids:
                t_min = obj_bboxes[tid]["min"]
                t_max = obj_bboxes[tid]["max"]
                t_ext = t_max - t_min
                t_pad_x = t_ext[0].item() * table_proximity_padding
                t_pad_y = t_ext[1].item() * table_proximity_padding
                ax.add_patch(patches.Rectangle(
                    (t_min[0].item() - t_pad_x, t_min[1].item() - t_pad_y),
                    t_ext[0].item() + 2 * t_pad_x, t_ext[1].item() + 2 * t_pad_y,
                    linewidth=1, edgecolor='royalblue', facecolor='none',
                    linestyle='--', alpha=0.6,
                ))

            # Solid table bboxes
            for tid in table_ids:
                t_min = obj_bboxes[tid]["min"]
                t_max = obj_bboxes[tid]["max"]
                t_ext = t_max - t_min
                t_cx = ((t_min[0] + t_max[0]) / 2).item()
                t_cy = ((t_min[1] + t_max[1]) / 2).item()
                ax.add_patch(patches.Rectangle(
                    (t_min[0].item(), t_min[1].item()),
                    t_ext[0].item(), t_ext[1].item(),
                    linewidth=2, edgecolor='royalblue', facecolor='lightblue', alpha=0.5,
                ))
                ax.text(t_cx, t_cy, f'T{tid}\n({categories.get(tid, "?")[:8]})',
                        ha='center', va='center', fontsize=8, color='royalblue', fontweight='bold')

            # Per-chair: padded bbox, solid bbox, expanded bbox, arrow
            for cid in chair_ids:
                c_min = obj_bboxes[cid]["min"]
                c_max = obj_bboxes[cid]["max"]
                c_ext = c_max - c_min
                c_cx = ((c_min[0] + c_max[0]) / 2).item()
                c_cy = ((c_min[1] + c_max[1]) / 2).item()

                # Padded chair bbox (dashed green)
                pad_x = c_ext[0].item() * chair_proximity_padding
                pad_y = c_ext[1].item() * chair_proximity_padding
                ax.add_patch(patches.Rectangle(
                    (c_min[0].item() - pad_x, c_min[1].item() - pad_y),
                    c_ext[0].item() + 2 * pad_x, c_ext[1].item() + 2 * pad_y,
                    linewidth=1, edgecolor='forestgreen', facecolor='none',
                    linestyle='--', alpha=0.6,
                ))

                # Solid chair bbox
                ax.add_patch(patches.Rectangle(
                    (c_min[0].item(), c_min[1].item()),
                    c_ext[0].item(), c_ext[1].item(),
                    linewidth=2, edgecolor='forestgreen', facecolor='lightgreen', alpha=0.5,
                ))
                ax.text(c_cx, c_cy, f'C{cid}',
                        ha='center', va='center', fontsize=8, color='forestgreen', fontweight='bold')

                # Expanded bbox + expansion arrow
                mod = result.get(cid, [0.0] * 6)
                dx_min_v, dx_max_v, dy_min_v, dy_max_v = mod[0], mod[1], mod[2], mod[3]
                if dx_min_v > 0 or dx_max_v > 0 or dy_min_v > 0 or dy_max_v > 0:
                    new_xmin = c_min[0].item() - dx_min_v
                    new_xmax = c_max[0].item() + dx_max_v
                    new_ymin = c_min[1].item() - dy_min_v
                    new_ymax = c_max[1].item() + dy_max_v
                    ax.add_patch(patches.Rectangle(
                        (new_xmin, new_ymin), new_xmax - new_xmin, new_ymax - new_ymin,
                        linewidth=2, edgecolor='red', facecolor='none', linestyle='-', alpha=0.9,
                    ))
                    # Arrow: signed net displacement
                    arrow_dx = dx_max_v - dx_min_v
                    arrow_dy = dy_max_v - dy_min_v
                    ax.annotate(
                        '', xy=(c_cx + arrow_dx, c_cy + arrow_dy),
                        xytext=(c_cx, c_cy),
                        arrowprops=dict(arrowstyle='->', color='red', lw=2),
                    )

            ax.set_aspect('equal', adjustable='datalim')
            ax.autoscale_view()
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            margin = 0.5
            ax.set_xlim(xlim[0] - margin, xlim[1] + margin)
            ax.set_ylim(ylim[0] - margin, ylim[1] + margin)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title('Chair–Table Bbox Modifications (XY top-down view)')
            ax.grid(True, alpha=0.3)
            legend_elements = [
                patches.Patch(facecolor='lightblue', edgecolor='royalblue', label='Table bbox'),
                patches.Patch(facecolor='none', edgecolor='royalblue', linestyle='--', label='Table padded bbox'),
                patches.Patch(facecolor='lightgreen', edgecolor='forestgreen', label='Chair bbox'),
                patches.Patch(facecolor='none', edgecolor='forestgreen', linestyle='--', label='Chair padded bbox'),
                patches.Patch(facecolor='none', edgecolor='red', label='Chair expanded bbox'),
                Line2D([0], [0], color='red', lw=2, marker='>', markersize=8, label='Expand direction'),
            ]
            ax.legend(handles=legend_elements, loc='best', fontsize=8)

            out_path = os.path.join(vis_output_dir, "chair_bbox_modifications.png")
            plt.savefig(out_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"[chair_bbox_xy_modify] Visualization saved to {out_path}")
        except Exception as e:
            import traceback
            print(f"[chair_bbox_xy_modify] WARNING: visualization failed: {e}")
            traceback.print_exc()

    return result