"""
Dump test_split.json for one scene without running full training.

Reads COLMAP metadata only (no images loaded), applies the same natsort + modulo
split used by DataPack / reader_colmap_dataset.py.

Usage:
    python evaluation/dump_test_split.py --source_path datasets/Replica/scan1
    python evaluation/dump_test_split.py --source_path datasets/Replica/scan1 --test_every 8
"""

import argparse
import json
import os
import sys

import natsort

GEOSVR = os.path.join(os.path.dirname(__file__), "..", "decomvoxel", "representation", "GeoSVR")
sys.path.insert(0, os.path.abspath(GEOSVR))

from src.dataloader.colmap_loader import (
    read_extrinsics_binary,
    read_extrinsics_text,
)


def _find_sparse(source_path):
    for subdir in ("sparse/0", "sparse", "colmap/sparse/0", "colmap/sparse"):
        p = os.path.join(source_path, subdir)
        if os.path.isdir(p):
            return p
    return None


def get_sorted_image_names(source_path):
    sparse_dir = _find_sparse(source_path)
    if sparse_dir is None:
        raise FileNotFoundError(f"No COLMAP sparse directory found under {source_path}")

    bin_path = os.path.join(sparse_dir, "images.bin")
    txt_path = os.path.join(sparse_dir, "images.txt")

    if os.path.exists(bin_path):
        cam_extrinsics = read_extrinsics_binary(bin_path)
    elif os.path.exists(txt_path):
        cam_extrinsics = read_extrinsics_text(txt_path)
    else:
        raise FileNotFoundError(f"No images.bin or images.txt in {sparse_dir}")

    # Same sort as reader_colmap_dataset.py:read_cameras_from_colmap
    keys = natsort.natsorted(cam_extrinsics.keys(), key=lambda i: cam_extrinsics[i].name)
    # image_name strips extension, matching reader_colmap_dataset.py line 79
    names = [os.path.basename(cam_extrinsics[k].name).rsplit(".", 1)[0] for k in keys]
    return names


def main():
    parser = argparse.ArgumentParser(description="Dump test_split.json for a COLMAP scene")
    parser.add_argument("--source_path", required=True, help="Dataset scene root directory")
    parser.add_argument("--test_every", type=int, default=8,
                        help="Every Nth image is test (default: 8)")
    parser.add_argument("--output", default="test_split.json",
                        help="Output filename inside source_path (default: test_split.json)")
    args = parser.parse_args()

    source_path = os.path.abspath(args.source_path)
    all_names = get_sorted_image_names(source_path)

    test_names  = [n for i, n in enumerate(all_names) if i % args.test_every == 0]
    train_names = [n for i, n in enumerate(all_names) if i % args.test_every != 0]

    result = {
        "source_path": source_path,
        "test_every": args.test_every,
        "num_total": len(all_names),
        "num_test": len(test_names),
        "num_train": len(train_names),
        "test_images": test_names,
    }

    out_path = os.path.join(source_path, args.output)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Saved: {out_path}  ({len(test_names)} test / {len(train_names)} train / {len(all_names)} total)")


if __name__ == "__main__":
    main()
