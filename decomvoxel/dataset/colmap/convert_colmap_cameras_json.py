#!/usr/bin/env python3
"""
Convert a colmap_cameras.json file to COLMAP model files:
  - cameras.txt
  - cameras.bin
  - images.txt
  - images.bin

The input JSON is expected to contain:
  {
    "cameras": [{"camera_id", "model", "width", "height", "params"}, ...],
    "images": [{"image_id", "camera_id", "name", "qvec", "tvec"}, ...]
  }

Example:
  python decomvoxel/dataset/colmap/convert_colmap_cameras_json.py \
      --input_json datasets/Blender/scene1/colmap_cameras.json \
      --output_dir datasets/Blender/scene1/sparse/0
"""

import argparse
import json
import struct
from collections import namedtuple
from pathlib import Path

import numpy as np


CameraModel = namedtuple("CameraModel", ["model_id", "model_name", "num_params"])
Camera = namedtuple("Camera", ["id", "model", "width", "height", "params"])
ImageEntry = namedtuple("ImageEntry", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])

CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12),
}

CAMERA_MODEL_NAMES = {model.model_name: model for model in CAMERA_MODELS}


def write_next_bytes(fid, data, format_char_sequence, endian_character="<"):
    if isinstance(data, (list, tuple)):
        data_bytes = struct.pack(endian_character + format_char_sequence, *data)
    else:
        data_bytes = struct.pack(endian_character + format_char_sequence, data)
    fid.write(data_bytes)


def write_cameras_text(cameras, path):
    header = [
        "# Camera list with one line of data per camera:\n"
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        f"# Number of cameras: {len(cameras)}\n"
    ]

    with open(path, "w", encoding="utf-8") as fid:
        fid.write("".join(header))
        for camera_id in sorted(cameras):
            cam = cameras[camera_id]
            line = " ".join(map(str, [cam.id, cam.model, cam.width, cam.height, *cam.params]))
            fid.write(line + "\n")


def write_cameras_binary(cameras, path):
    with open(path, "wb") as fid:
        write_next_bytes(fid, len(cameras), "Q")
        for camera_id in sorted(cameras):
            cam = cameras[camera_id]
            model_id = CAMERA_MODEL_NAMES[cam.model].model_id
            write_next_bytes(fid, [cam.id, model_id, cam.width, cam.height], "iiQQ")
            for p in cam.params:
                write_next_bytes(fid, float(p), "d")


def write_images_text(images, path):
    mean_observations = 0.0 if len(images) == 0 else sum(len(img.point3D_ids) for img in images.values()) / len(images)
    header = [
        "# Image list with two lines of data per image:\n"
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
        "#   POINTS2D[] as (X, Y, POINT3D_ID)\n"
        f"# Number of images: {len(images)}, mean observations per image: {mean_observations}\n"
    ]

    with open(path, "w", encoding="utf-8") as fid:
        fid.write("".join(header))
        for image_id in sorted(images):
            img = images[image_id]
            first_line = " ".join(map(str, [img.id, *img.qvec, *img.tvec, img.camera_id, img.name]))
            fid.write(first_line + "\n")

            if len(img.point3D_ids) == 0:
                fid.write("\n")
            else:
                points_strings = []
                for xy, point3d_id in zip(img.xys, img.point3D_ids):
                    points_strings.append(" ".join(map(str, [*xy, point3d_id])))
                fid.write(" ".join(points_strings) + "\n")


def write_images_binary(images, path):
    with open(path, "wb") as fid:
        write_next_bytes(fid, len(images), "Q")
        for image_id in sorted(images):
            img = images[image_id]
            write_next_bytes(fid, img.id, "i")
            write_next_bytes(fid, img.qvec.tolist(), "dddd")
            write_next_bytes(fid, img.tvec.tolist(), "ddd")
            write_next_bytes(fid, img.camera_id, "i")

            for ch in img.name:
                write_next_bytes(fid, ch.encode("utf-8"), "c")
            write_next_bytes(fid, b"\x00", "c")

            write_next_bytes(fid, len(img.point3D_ids), "Q")
            for xy, p3d_id in zip(img.xys, img.point3D_ids):
                write_next_bytes(fid, [float(xy[0]), float(xy[1]), int(p3d_id)], "ddq")


def _validate_camera_params(model_name, params):
    if model_name not in CAMERA_MODEL_NAMES:
        raise ValueError(f"Unsupported camera model: {model_name}")
    expected = CAMERA_MODEL_NAMES[model_name].num_params
    if len(params) != expected:
        raise ValueError(
            f"Camera model {model_name} expects {expected} params, got {len(params)}"
        )


def load_colmap_json(input_json):
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "cameras" not in data or "images" not in data:
        raise ValueError("Input JSON must contain both 'cameras' and 'images' fields")

    cameras = {}
    for cam in data["cameras"]:
        camera_id = int(cam["camera_id"])
        model = str(cam["model"]).upper()
        width = int(cam["width"])
        height = int(cam["height"])
        params = np.asarray(cam["params"], dtype=np.float64)
        _validate_camera_params(model, params)

        cameras[camera_id] = Camera(
            id=camera_id,
            model=model,
            width=width,
            height=height,
            params=params,
        )

    images = {}
    for item in data["images"]:
        image_id = int(item["image_id"])
        camera_id = int(item["camera_id"])
        if camera_id not in cameras:
            raise ValueError(f"Image {image_id} references unknown camera_id={camera_id}")

        qvec = np.asarray(item["qvec"], dtype=np.float64)
        tvec = np.asarray(item["tvec"], dtype=np.float64)
        if qvec.shape != (4,):
            raise ValueError(f"Image {image_id} qvec must have 4 elements, got {qvec.shape}")
        if tvec.shape != (3,):
            raise ValueError(f"Image {image_id} tvec must have 3 elements, got {tvec.shape}")

        name = item.get("name") or Path(item.get("image_path", f"{image_id:06d}.png")).name

        images[image_id] = ImageEntry(
            id=image_id,
            qvec=qvec,
            tvec=tvec,
            camera_id=camera_id,
            name=name,
            xys=np.empty((0, 2), dtype=np.float64),
            point3D_ids=np.empty((0,), dtype=np.int64),
        )

    if not cameras:
        raise ValueError("No camera found in input JSON")
    if not images:
        raise ValueError("No image found in input JSON")

    return cameras, images


def convert(input_json, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    cameras, images = load_colmap_json(input_json)

    write_cameras_text(cameras, output_dir / "cameras.txt")
    write_cameras_binary(cameras, output_dir / "cameras.bin")
    write_images_text(images, output_dir / "images.txt")
    write_images_binary(images, output_dir / "images.bin")

    print(f"Wrote cameras/images model files to: {output_dir}")
    print(f"  cameras: {len(cameras)}")
    print(f"  images : {len(images)}")


def main():
    parser = argparse.ArgumentParser(description="Convert colmap_cameras.json to COLMAP txt/bin model files")
    parser.add_argument("--input_json", type=Path, required=True, help="Path to colmap_cameras.json")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory for cameras/images txt+bin (default: <input_json_dir>/sparse/0)",
    )
    args = parser.parse_args()

    if not args.input_json.is_file():
        raise FileNotFoundError(f"Input JSON does not exist: {args.input_json}")

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = args.input_json.parent / "sparse" / "0"

    convert(args.input_json, output_dir)


if __name__ == "__main__":
    main()