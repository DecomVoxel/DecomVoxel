"""Minimal COLMAP camera reader for evaluation.

Reuses the COLMAP binary loader from
`decomvoxel/representation/GeoSVR/src/dataloader/colmap_loader.py`.

Returned camera dicts contain everything the Blender worker needs:
    - image_name : str (basename without extension)
    - image_path : str (absolute)
    - W, H       : int
    - K          : 3x3 intrinsics (list-of-list, fx, fy, cx, cy)
    - w2c        : 4x4 OpenCV-convention world->camera (list-of-list)
"""

from __future__ import annotations

import json
import os
import sys
import natsort
import numpy as np


def _ensure_geosvr_src_on_path() -> None:
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    geosvr_src = os.path.join(repo_root, "decomvoxel", "representation", "GeoSVR", "src")
    if geosvr_src not in sys.path:
        sys.path.insert(0, geosvr_src)


def _find_sparse_dir(source_path: str) -> str:
    for sub in (("sparse", "0"), ("colmap", "sparse", "0"), ("sparse",)):
        candidate = os.path.join(source_path, *sub)
        if not os.path.isdir(candidate):
            continue
        if os.path.isfile(os.path.join(candidate, "cameras.bin")) or \
           os.path.isfile(os.path.join(candidate, "cameras.txt")):
            return candidate
    raise FileNotFoundError(f"Cannot find COLMAP sparse dir under {source_path}")


def load_test_cameras(source_path: str, images_subdir: str = "images", test_every: int = 8):
    """Return (test_cams, all_count). Test cameras are every `test_every`-th sorted image."""
    _ensure_geosvr_src_on_path()
    from dataloader.colmap_loader import (  # type: ignore
        read_extrinsics_binary,
        read_intrinsics_binary,
        read_extrinsics_text,
        read_intrinsics_text,
        qvec2rotmat,
    )

    sparse_dir = _find_sparse_dir(source_path)
    if os.path.isfile(os.path.join(sparse_dir, "cameras.bin")):
        cam_extr = read_extrinsics_binary(os.path.join(sparse_dir, "images.bin"))
        cam_intr = read_intrinsics_binary(os.path.join(sparse_dir, "cameras.bin"))
    else:
        cam_extr = read_extrinsics_text(os.path.join(sparse_dir, "images.txt"))
        cam_intr = read_intrinsics_text(os.path.join(sparse_dir, "cameras.txt"))

    keys = natsort.natsorted(cam_extr.keys(), key=lambda i: cam_extr[i].name)
    images_dir = os.path.join(source_path, images_subdir)

    cams = []
    for idx, key in enumerate(keys):
        extr = cam_extr[key]
        intr = cam_intr[extr.camera_id]

        W, H = intr.width, intr.height
        if intr.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL"):
            fx = fy = float(intr.params[0])
            cx, cy = float(intr.params[1]), float(intr.params[2])
        elif intr.model == "PINHOLE":
            fx, fy = float(intr.params[0]), float(intr.params[1])
            cx, cy = float(intr.params[2]), float(intr.params[3])
        else:
            raise NotImplementedError(f"COLMAP camera model not supported: {intr.model}")

        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :3] = qvec2rotmat(extr.qvec)
        w2c[:3, 3] = np.asarray(extr.tvec, dtype=np.float64)

        image_name = os.path.splitext(os.path.basename(extr.name))[0]
        image_path = os.path.join(images_dir, os.path.basename(extr.name))
        if not os.path.isfile(image_path):
            # Try png fallback
            alt = os.path.splitext(image_path)[0] + ".png"
            if os.path.isfile(alt):
                image_path = alt

        cams.append({
            "idx": idx,
            "image_name": image_name,
            "image_path": image_path,
            "W": int(W), "H": int(H),
            "K": [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            "w2c": w2c.tolist(),
        })

    # Prefer test_split.json written by DataPack during training to guarantee
    # exact alignment with the training/test split that was actually used.
    split_json = os.path.join(source_path, "test_split.json")
    if os.path.isfile(split_json):
        with open(split_json) as f:
            split_data = json.load(f)
        test_names_set = set(split_data["test_images"])
        test_cams = [c for c in cams if c["image_name"] in test_names_set]
        print(f"[colmap_io] Using test_split.json: {len(test_cams)} test cameras")
    else:
        # Fallback: every test_every-th frame starting from idx=0
        test_cams = [c for c in cams if c["idx"] % test_every == 0]
        print(f"[colmap_io] No test_split.json found; using idx%{test_every}==0: {len(test_cams)} test cameras")
    return test_cams, len(cams)