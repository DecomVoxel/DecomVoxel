"""
Replace fully-black pixels (all channels == 0) with white (255) in all images
under a given directory, and save results to a sibling folder `mask_replace`.

Usage:
    python decomvoxel/utils/black_to_white.py <input_dir>
"""

import argparse
import os

import numpy as np
from PIL import Image


def process_image(img_path: str, out_path: str) -> None:
    img = Image.open(img_path).convert("RGB")
    arr = np.array(img)

    # Mask: pixels where ALL channels are 0 (fully black)
    black_mask = np.all(arr == 0, axis=-1)

    arr[black_mask] = 255
    Image.fromarray(arr).save(out_path)


def process_dir(input_dir: str) -> None:
    input_dir = os.path.abspath(input_dir)
    parent_dir = os.path.dirname(input_dir)
    out_dir = os.path.join(parent_dir, "mask_replace")
    os.makedirs(out_dir, exist_ok=True)

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
    img_files = sorted(
        f for f in os.listdir(input_dir)
        if os.path.splitext(f)[1].lower() in exts
    )

    if not img_files:
        print(f"No image files found in {input_dir}")
        return

    for fname in img_files:
        src = os.path.join(input_dir, fname)
        dst = os.path.join(out_dir, fname)
        process_image(src, dst)
        print(f"  {fname} -> {dst}")

    print(f"\nDone. {len(img_files)} images saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replace black pixels with white.")
    parser.add_argument("input_dir", type=str, help="Directory containing images.")
    args = parser.parse_args()
    process_dir(args.input_dir)
