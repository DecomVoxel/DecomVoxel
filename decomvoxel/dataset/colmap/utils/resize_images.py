#!/usr/bin/env python3
"""
Resize dataset images for COLMAP preprocessing.

Usage examples:
    python resize_images.py datasets/Blender scene1 --width 800 --height 600
    python resize_images.py datasets/Blender --width 800 --height 600

Behavior:
- If dataset_name is provided: process only <root_dir>/<dataset_name>/images
- Otherwise: process every direct child dataset that contains an images folder.
- Images are resized in place and keep original filenames.
"""

import argparse
import shutil
import sys
from pathlib import Path

try:
    from PIL import Image, ImageOps
except ImportError:
    print("Error: Pillow is required. Install with: pip install pillow", file=sys.stderr)
    sys.exit(1)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".JPG", ".JPEG", ".PNG", ".BMP", ".TIF", ".TIFF", ".WEBP"}


def find_image_dirs(root_dir: Path, dataset_name: str | None) -> list[Path]:
    image_dirs: list[Path] = []

    if dataset_name:
        image_dir = root_dir / dataset_name / "images"
        if image_dir.is_dir():
            image_dirs.append(image_dir)
        return image_dirs

    # Include root_dir/images if root_dir itself is a single dataset directory.
    root_images = root_dir / "images"
    if root_images.is_dir():
        image_dirs.append(root_images)

    # Include each direct child dataset with an images folder.
    for child in sorted(root_dir.iterdir()):
        if child.is_dir():
            child_images = child / "images"
            if child_images.is_dir():
                image_dirs.append(child_images)

    return image_dirs


def collect_images(image_dir: Path) -> list[Path]:
    files = [p for p in image_dir.rglob("*") if p.is_file() and p.suffix in IMAGE_SUFFIXES]
    return sorted(files)


def ensure_backup_and_reset_output(image_dir: Path) -> tuple[Path, Path]:
    dataset_dir = image_dir.parent
    source_dir = dataset_dir / "images_origin"
    output_dir = dataset_dir / "images"

    if not source_dir.exists():
        print(f"  Creating backup: {source_dir}")
        shutil.copytree(output_dir, source_dir)
    else:
        print(f"  Using existing backup: {source_dir}")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    return source_dir, output_dir


def validate_fixed_aspect_ratio(images: list[Path], width: int, height: int) -> None:
    target_ratio = f"{width}:{height}"
    for image_path in images:
        try:
            with Image.open(image_path) as img:
                img = ImageOps.exif_transpose(img)
                orig_w, orig_h = img.size
        except Exception as exc:
            raise RuntimeError(f"Cannot read image size: {image_path} ({exc})") from exc

        if orig_w <= 0 or orig_h <= 0:
            raise RuntimeError(f"Invalid image size: {image_path} ({orig_w}x{orig_h})")

        # Strict aspect ratio check: no crop/pad allowed.
        if orig_w * height != orig_h * width:
            raise RuntimeError(
                "Aspect ratio mismatch detected. "
                f"Image {image_path} is {orig_w}x{orig_h} (ratio {orig_w}:{orig_h}), "
                f"but target ratio is {target_ratio}."
            )


def resize_one_image(src_path: Path, dst_path: Path, width: int, height: int) -> tuple[bool, str]:
    try:
        with Image.open(src_path) as img:
            # Respect EXIF orientation before resizing.
            img = ImageOps.exif_transpose(img)
            resized = img.resize((width, height), Image.Resampling.LANCZOS)

            save_kwargs = {}
            if dst_path.suffix.lower() in {".jpg", ".jpeg"}:
                save_kwargs["quality"] = 95
                save_kwargs["optimize"] = True

            dst_path.parent.mkdir(parents=True, exist_ok=True)
            resized.save(dst_path, **save_kwargs)
        return True, ""
    except Exception as exc:
        return False, str(exc)


def process_dataset(image_dir: Path, width: int, height: int) -> tuple[int, int]:
    print(f"\nProcessing: {image_dir.parent}")
    source_dir, output_dir = ensure_backup_and_reset_output(image_dir)

    images = collect_images(source_dir)
    if not images:
        print(f"[WARN] No images found in: {source_dir}")
        return 0, 0

    print(f"  Found {len(images)} image(s)")
    print(f"  Target size: {width}x{height}")

    validate_fixed_aspect_ratio(images, width, height)

    ok_count = 0
    fail_count = 0

    for idx, src_path in enumerate(images, start=1):
        rel_path = src_path.relative_to(source_dir)
        dst_path = output_dir / rel_path

        ok, err = resize_one_image(src_path, dst_path, width, height)
        if ok:
            ok_count += 1
        else:
            fail_count += 1
            print(f"  [ERROR] {src_path}: {err}")

        if idx % 100 == 0 or idx == len(images):
            print(f"  Progress: {idx}/{len(images)}")

    print(f"  Done: success={ok_count}, failed={fail_count}")
    return ok_count, fail_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resize dataset images for COLMAP")
    parser.add_argument("root_dir", type=str, help="Root dataset directory")
    parser.add_argument("dataset_name", nargs="?", default=None, help="Optional dataset name under root_dir")
    parser.add_argument("--width", type=int, required=True, help="Target width in pixels")
    parser.add_argument("--height", type=int, required=True, help="Target height in pixels")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.width <= 0 or args.height <= 0:
        print("Error: --width and --height must be positive integers", file=sys.stderr)
        return 1

    root_dir = Path(args.root_dir)
    if not root_dir.exists():
        print(f"Error: root_dir does not exist: {root_dir}", file=sys.stderr)
        return 1

    image_dirs = find_image_dirs(root_dir, args.dataset_name)
    if not image_dirs:
        if args.dataset_name:
            print(
                f"Error: images directory not found at: {root_dir / args.dataset_name / 'images'}",
                file=sys.stderr,
            )
        else:
            print(f"Error: no images directories found under: {root_dir}", file=sys.stderr)
        return 1

    print(f"Found {len(image_dirs)} dataset(s) to resize")

    total_ok = 0
    total_fail = 0
    for image_dir in image_dirs:
        try:
            ok, fail = process_dataset(image_dir, args.width, args.height)
            total_ok += ok
            total_fail += fail
        except RuntimeError as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            return 1

    print("\nSummary")
    print(f"  Resized: {total_ok}")
    print(f"  Failed:  {total_fail}")

    return 0 if total_ok > 0 and total_fail == 0 else 1 if total_ok == 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
