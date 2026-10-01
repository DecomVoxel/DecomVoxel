#!/usr/bin/env python3
"""Split images into train/test by moving sampled files to sibling images_test.

Default behavior:
- Input: <parent>/images
- Output: <parent>/images_test
- Move about 1/10 of images, sampled uniformly across sorted filenames.
"""

import argparse
import shutil
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(images_dir: Path):
	return sorted(
		[p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
	)


def sample_indices(total: int, move_count: int):
	if move_count <= 0:
		return []
	if move_count >= total:
		return list(range(total))
	step = total / float(move_count)
	return sorted({int(i * step) for i in range(move_count)})


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--images_dir", type=str, required=True, help="Path to images directory")
	parser.add_argument(
		"--test_ratio",
		type=float,
		default=0.1,
		help="Fraction of images to move to images_test (default: 0.1)",
	)
	parser.add_argument(
		"--move_count",
		type=int,
		default=None,
		help="Override number of images to move; if set, test_ratio is ignored",
	)
	parser.add_argument("--dry_run", action="store_true", help="Print planned moves without moving")
	return parser.parse_args()


def main():
	args = parse_args()

	images_dir = Path(args.images_dir).expanduser().resolve()
	if not images_dir.exists() or not images_dir.is_dir():
		raise ValueError(f"Invalid images_dir: {images_dir}")

	test_dir = images_dir.parent / "images_test"
	test_dir.mkdir(parents=True, exist_ok=True)

	images = list_images(images_dir)
	total = len(images)
	if total == 0:
		print(f"No images found in {images_dir}")
		return

	if args.move_count is not None:
		move_count = max(0, min(args.move_count, total))
	else:
		move_count = max(1, int(total * args.test_ratio))

	indices = sample_indices(total, move_count)
	selected = [images[i] for i in indices if i < total]

	print(f"images_dir: {images_dir}")
	print(f"images_test: {test_dir}")
	print(f"total images: {total}")
	print(f"move count: {len(selected)}")

	for src in selected:
		dst = test_dir / src.name
		if args.dry_run:
			print(f"[DRY RUN] {src} -> {dst}")
		else:
			shutil.move(str(src), str(dst))
			print(f"Moved: {src.name}")

	if args.dry_run:
		print("Dry run completed.")
	else:
		print("Done.")


if __name__ == "__main__":
	main()
