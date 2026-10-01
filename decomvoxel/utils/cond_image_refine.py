"""
decomvoxel/utils/cond_image_refine.py

Standalone utility: whiten background + recenter/pad a single conditioning image.
Processing logic is identical to `pad_and_recenter_cond_img` in
decomvoxel/pipeline/cond_image_generate.py (imported directly to avoid duplication).

CLI usage:
    python decomvoxel/utils/cond_image_refine.py --input path/to/img.png --output path/to/out.png

    # Overwrite in-place:
    python decomvoxel/utils/cond_image_refine.py --input path/to/img.png

    # Batch: process every PNG in a directory:
    python decomvoxel/utils/cond_image_refine.py --input dir/ --output dir_refined/

Python API:
    from decomvoxel.utils.cond_image_refine import refine_cond_image
    refine_cond_image("in.png", "out.png")
"""

from __future__ import annotations

import argparse
import os
import sys

from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from decomvoxel.pipeline.cond_image_generate import pad_and_recenter_cond_img


def refine_cond_image(
    input_path: str,
    output_path: str | None = None,
    white_threshold: int = 20,
    target_ratio: float = 0.40,
    output_size: int = None,
) -> str:
    """
    Load *input_path*, apply background-whitening + recentering/padding,
    and save the result to *output_path* (defaults to overwriting *input_path*).

    Returns the absolute path of the saved file.
    """
    if output_path is None:
        output_path = input_path

    img = Image.open(input_path)
    refined = pad_and_recenter_cond_img(
        img,
        white_threshold=white_threshold,
        target_ratio=target_ratio,
        output_size=output_size,
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    refined.save(output_path)
    return os.path.abspath(output_path)


def _process_dir(input_dir: str, output_dir: str, **kwargs) -> None:
    import glob
    png_files = sorted(glob.glob(os.path.join(input_dir, "*.png")))
    if not png_files:
        print(f"[cond_image_refine] No PNG files found in: {input_dir}")
        return
    os.makedirs(output_dir, exist_ok=True)
    for src in png_files:
        dst = os.path.join(output_dir, os.path.basename(src))
        out = refine_cond_image(src, dst, **kwargs)
        print(f"[cond_image_refine] {src} -> {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Whiten background + recenter/pad a conditioning image."
    )
    parser.add_argument("--input", required=True,
                        help="Input image file (.png) or directory of PNGs.")
    parser.add_argument("--output", default=None,
                        help="Output file or directory (default: overwrite input).")
    parser.add_argument("--white_threshold", type=int, default=20,
                        help="Pixels with all channels >= (255 - white_threshold) are treated as background (default: 20).")
    parser.add_argument("--target_ratio", type=float, default=0.40,
                        help="Target object extent as a fraction of output image size (default: 0.40).")
    parser.add_argument("--output_size", type=int, default=None,
                        help="Output image size in pixels (default: max(H, W) of input).")
    args = parser.parse_args()

    kwargs = dict(
        white_threshold=args.white_threshold,
        target_ratio=args.target_ratio,
        output_size=args.output_size,
    )

    if os.path.isdir(args.input):
        output_dir = args.output if args.output else args.input
        _process_dir(args.input, output_dir, **kwargs)
    else:
        out = refine_cond_image(args.input, args.output, **kwargs)
        print(f"[cond_image_refine] Saved to: {out}")
