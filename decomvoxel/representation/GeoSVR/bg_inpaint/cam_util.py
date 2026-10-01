import os
import json
import sys
import collections

import struct
import numpy as np
import torch

from PIL import Image as PILImage

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
GEOSVR_ROOT = os.path.dirname(PROJECT_ROOT)
if GEOSVR_ROOT not in sys.path:
    sys.path.insert(0, GEOSVR_ROOT)

from src.cameras import MiniCam

CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"]
)
Camera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"]
)
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"]
)


class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)


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
CAMERA_MODEL_IDS = dict(
    [(camera_model.model_id, camera_model) for camera_model in CAMERA_MODELS]
)
CAMERA_MODEL_NAMES = dict(
    [(camera_model.model_name, camera_model) for camera_model in CAMERA_MODELS]
)


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def write_next_bytes(fid, data, format_char_sequence, endian_character="<"):
    """pack and write to a binary file.
    :param fid:
    :param data: data to send, if multiple elements are sent at the same time,
    they should be encapsuled either in a list or a tuple
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    should be the same length as the data list or tuple
    :param endian_character: Any of {@, =, <, >, !}
    """
    if isinstance(data, (list, tuple)):
        bytes = struct.pack(endian_character + format_char_sequence, *data)
    else:
        bytes = struct.pack(endian_character + format_char_sequence, data)
    fid.write(bytes)


def read_cameras_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasText(const std::string& path)
        void Reconstruction::ReadCamerasText(const std::string& path)
    """
    cameras = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = Camera(
                    id=camera_id, model=model, width=width, height=height, params=params
                )
    return cameras


def read_cameras_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for camera_line_index in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ"
            )
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(
                fid, num_bytes=8 * num_params, format_char_sequence="d" * num_params
            )
            cameras[camera_id] = Camera(
                id=camera_id,
                model=model_name,
                width=width,
                height=height,
                params=np.array(params),
            )
        assert len(cameras) == num_cameras
    return cameras


def write_cameras_text(cameras, path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasText(const std::string& path)
        void Reconstruction::ReadCamerasText(const std::string& path)
    """
    HEADER = [
        "# Camera list with one line of data per camera:\n"
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        "# Number of cameras: {}\n".format(len(cameras))
    ]
    with open(path, "w") as fid:
        fid.write("".join(HEADER))
        for _, cam in cameras.items():
            to_write = [cam.id, cam.model, cam.width, cam.height, *cam.params]
            line = " ".join([str(elem) for elem in to_write])
            fid.write(line + "\n")


def write_cameras_binary(cameras, path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    with open(path_to_model_file, "wb") as fid:
        write_next_bytes(fid, len(cameras), "Q")
        for _, cam in cameras.items():
            model_id = CAMERA_MODEL_NAMES[cam.model].model_id
            camera_properties = [cam.id, model_id, cam.width, cam.height]
            write_next_bytes(fid, camera_properties, "iiQQ")
            for p in cam.params:
                write_next_bytes(fid, float(p), "d")
    return cameras


def read_images_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesText(const std::string& path)
        void Reconstruction::WriteImagesText(const std::string& path)
    """
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack(
                    [tuple(map(float, elems[0::3])), tuple(map(float, elems[1::3]))]
                )
                point3D_ids = np.array(tuple(map(int, elems[2::3])))
                images[image_id] = Image(
                    id=image_id,
                    qvec=qvec,
                    tvec=tvec,
                    camera_id=camera_id,
                    name=image_name,
                    xys=xys,
                    point3D_ids=point3D_ids,
                )
    return images


def read_images_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for image_index in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi"
            )
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":  # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[
                0
            ]
            x_y_id_s = read_next_bytes(
                fid,
                num_bytes=24 * num_points2D,
                format_char_sequence="ddq" * num_points2D,
            )
            xys = np.column_stack(
                [tuple(map(float, x_y_id_s[0::3])), tuple(map(float, x_y_id_s[1::3]))]
            )
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=image_name,
                xys=xys,
                point3D_ids=point3D_ids,
            )
    return images


def write_images_text(images, path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesText(const std::string& path)
        void Reconstruction::WriteImagesText(const std::string& path)
    """
    if len(images) == 0:
        mean_observations = 0
    else:
        mean_observations = sum(
            (len(img.point3D_ids) for _, img in images.items())
        ) / len(images)
    HEADER = [
        "# Image list with two lines of data per image:\n"
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
        "#   POINTS2D[] as (X, Y, POINT3D_ID)\n"
        "# Number of images: {}, mean observations per image: {}\n".format(
            len(images), mean_observations
        )
    ]

    with open(path, "w") as fid:
        fid.write("".join(HEADER))
        for _, img in images.items():
            image_header = [img.id, *img.qvec, *img.tvec, img.camera_id, img.name]
            first_line = " ".join(map(str, image_header))
            fid.write(first_line + "\n")

            points_strings = []
            for xy, point3D_id in zip(img.xys, img.point3D_ids):
                points_strings.append(" ".join(map(str, [*xy, point3D_id])))
            fid.write(" ".join(points_strings) + "\n")


def write_images_binary(images, path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    with open(path_to_model_file, "wb") as fid:
        write_next_bytes(fid, len(images), "Q")
        for _, img in images.items():
            write_next_bytes(fid, img.id, "i")
            write_next_bytes(fid, img.qvec.tolist(), "dddd")
            write_next_bytes(fid, img.tvec.tolist(), "ddd")
            write_next_bytes(fid, img.camera_id, "i")
            for char in img.name:
                write_next_bytes(fid, char.encode("utf-8"), "c")
            write_next_bytes(fid, b"\x00", "c")
            write_next_bytes(fid, len(img.point3D_ids), "Q")
            for xy, p3d_id in zip(img.xys, img.point3D_ids):
                write_next_bytes(fid, [*xy, p3d_id], "ddq")


def read_points3D_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DText(const std::string& path)
        void Reconstruction::WritePoints3DText(const std::string& path)
    """
    points3D = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                point3D_id = int(elems[0])
                xyz = np.array(tuple(map(float, elems[1:4])))
                rgb = np.array(tuple(map(int, elems[4:7])))
                error = float(elems[7])
                image_ids = np.array(tuple(map(int, elems[8::2])))
                point2D_idxs = np.array(tuple(map(int, elems[9::2])))
                points3D[point3D_id] = Point3D(
                    id=point3D_id,
                    xyz=xyz,
                    rgb=rgb,
                    error=error,
                    image_ids=image_ids,
                    point2D_idxs=point2D_idxs,
                )
    return points3D


def read_points3d_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """
    points3D = {}
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for point_line_index in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd"
            )
            point3D_id = binary_point_line_properties[0]
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[
                0
            ]
            track_elems = read_next_bytes(
                fid,
                num_bytes=8 * track_length,
                format_char_sequence="ii" * track_length,
            )
            image_ids = np.array(tuple(map(int, track_elems[0::2])))
            point2D_idxs = np.array(tuple(map(int, track_elems[1::2])))
            points3D[point3D_id] = Point3D(
                id=point3D_id,
                xyz=xyz,
                rgb=rgb,
                error=error,
                image_ids=image_ids,
                point2D_idxs=point2D_idxs,
            )
    return points3D


def write_points3D_text(points3D, path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DText(const std::string& path)
        void Reconstruction::WritePoints3DText(const std::string& path)
    """
    if len(points3D) == 0:
        mean_track_length = 0
    else:
        mean_track_length = sum(
            (len(pt.image_ids) for _, pt in points3D.items())
        ) / len(points3D)
    HEADER = [
        "# 3D point list with one line of data per point:\n"
        "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
        "# Number of points: {}, mean track length: {}\n".format(
            len(points3D), mean_track_length
        )
    ]

    with open(path, "w") as fid:
        fid.write("".join(HEADER))
        for _, pt in points3D.items():
            point_header = [pt.id, *pt.xyz, *pt.rgb, pt.error]
            fid.write(" ".join(map(str, point_header)) + " ")
            track_strings = []
            for image_id, point2D in zip(pt.image_ids, pt.point2D_idxs):
                track_strings.append(" ".join(map(str, [image_id, point2D])))
            fid.write(" ".join(track_strings) + "\n")


def write_points3d_binary(points3D, path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """
    with open(path_to_model_file, "wb") as fid:
        write_next_bytes(fid, len(points3D), "Q")
        for _, pt in points3D.items():
            write_next_bytes(fid, pt.id, "Q")
            write_next_bytes(fid, pt.xyz.tolist(), "ddd")
            write_next_bytes(fid, pt.rgb.tolist(), "BBB")
            write_next_bytes(fid, pt.error, "d")
            track_length = pt.image_ids.shape[0]
            write_next_bytes(fid, track_length, "Q")
            for image_id, point2D_id in zip(pt.image_ids, pt.point2D_idxs):
                write_next_bytes(fid, [image_id, point2D_id], "ii")


def read_model(path, ext):
    if ext == ".txt":
        cameras = read_cameras_text(os.path.join(path, "cameras" + ext))
        images = read_images_text(os.path.join(path, "images" + ext))
        points3D = read_points3D_text(os.path.join(path, "points3D") + ext)
    else:
        cameras = read_cameras_binary(os.path.join(path, "cameras" + ext))
        images = read_images_binary(os.path.join(path, "images" + ext))
        points3D = read_points3d_binary(os.path.join(path, "points3D") + ext)
    return cameras, images, points3D


def write_model(cameras, images, points3D, path, ext):
    if ext == ".txt":
        write_cameras_text(cameras, os.path.join(path, "cameras" + ext))
        write_images_text(images, os.path.join(path, "images" + ext))
        write_points3D_text(points3D, os.path.join(path, "points3D") + ext)
    else:
        write_cameras_binary(cameras, os.path.join(path, "cameras" + ext))
        write_images_binary(images, os.path.join(path, "images" + ext))
        write_points3d_binary(points3D, os.path.join(path, "points3D") + ext)
    return cameras, images, points3D


def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = (
        np.array(
            [
                [Rxx - Ryy - Rzz, 0, 0, 0],
                [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
                [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
                [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
            ]
        )
        / 3.0
    )
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def get_camera_matrix(camera_params, camera_model):
    """Get camera matrix and distortion coefficients from camera parameters in COLMAP format
    Arguments
        camera_params   - Camera parameters in COLMAP format
        camera_model    - Camera model
    Return
        K               - [3, 3] Camera matrix
        dc              - [12,] Distortion coefficients
    """
    camera_params = np.asarray(camera_params)
    K = np.zeros([3, 3])
    K[2, 2] = 1
    dc = np.zeros(
        [
            12,
        ]
    )
    if str.upper(camera_model) == "SIMPLE_PINHOLE":
        K[0, 0] = camera_params[0]
        K[1, 1] = camera_params[0]
        K[0, 2] = camera_params[1]
        K[1, 2] = camera_params[2]
    elif str.upper(camera_model) == "PINHOLE":
        K[0, 0] = camera_params[0]
        K[1, 1] = camera_params[1]
        K[0, 2] = camera_params[2]
        K[1, 2] = camera_params[3]
    elif str.upper(camera_model) in ("SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE"):
        K[0, 0] = camera_params[0]
        K[1, 1] = camera_params[0]
        K[0, 2] = camera_params[1]
        K[1, 2] = camera_params[2]
        dc[0] = camera_params[3]
    elif str.upper(camera_model) in ("RADIAL", "RADIAL_FISHEYE"):
        K[0, 0] = camera_params[0]
        K[1, 1] = camera_params[0]
        K[0, 2] = camera_params[1]
        K[1, 2] = camera_params[2]
        dc[0:2] = camera_params[3:5]
    elif str.upper(camera_model) == "OPENCV":
        K[0, 0] = camera_params[0]
        K[1, 1] = camera_params[1]
        K[0, 2] = camera_params[2]
        K[1, 2] = camera_params[3]
        dc[0:4] = camera_params[4:8]
    elif str.upper(camera_model) == "FULL_OPENCV":
        K[0, 0] = camera_params[0]
        K[1, 1] = camera_params[1]
        K[0, 2] = camera_params[2]
        K[1, 2] = camera_params[3]
        dc[0:8] = camera_params[4:12]
    elif str.upper(camera_model) == "OPENCV_FISHEYE":
        K[0, 0] = camera_params[0]
        K[1, 1] = camera_params[1]
        K[0, 2] = camera_params[2]
        K[1, 2] = camera_params[3]
        dc[0:2] = camera_params[4:6]
        dc[4:6] = camera_params[6:8]
    else:
        raise ValueError("Unsupported camera model: " + camera_model)
    return K, dc



def closed_form_pose_inverse(
    pose_matrices, rotation_matrices=None, translation_vectors=None
):
    """
    Compute the inverse of each 4x4 (or 3x4) SE3 pose matrices in a batch.

    If `rotation_matrices` and `translation_vectors` are provided, they must correspond to the rotation and translation
    components of `pose_matrices`. Otherwise, they will be extracted from `pose_matrices`.

    Args:
        pose_matrices: Nx4x4 or Nx3x4 array or tensor of SE3 matrices.
        rotation_matrices (optional): Nx3x3 array or tensor of rotation matrices.
        translation_vectors (optional): Nx3x1 array or tensor of translation vectors.

    Returns:
        Inverted SE3 matrices with the same type and device as input `pose_matrices`.

    Shapes:
        pose_matrices: (N, 4, 4)
        rotation_matrices: (N, 3, 3)
        translation_vectors: (N, 3, 1)
    """
    # Check if pose_matrices is a numpy array or a torch tensor
    is_numpy = isinstance(pose_matrices, np.ndarray)

    # Validate shapes
    if pose_matrices.shape[-2:] != (4, 4) and pose_matrices.shape[-2:] != (3, 4):
        raise ValueError(
            f"pose_matrices must be of shape (N,4,4), got {pose_matrices.shape}."
        )

    # Extract rotation_matrices and translation_vectors if not provided
    if rotation_matrices is None:
        rotation_matrices = pose_matrices[:, :3, :3]
    if translation_vectors is None:
        translation_vectors = pose_matrices[:, :3, 3:]

    # Compute the inverse of input SE3 matrices
    if is_numpy:
        rotation_transposed = np.transpose(rotation_matrices, (0, 2, 1))
        new_translation = -np.matmul(rotation_transposed, translation_vectors)
        inverted_matrix = np.tile(np.eye(4), (len(rotation_matrices), 1, 1))
    else:
        rotation_transposed = rotation_matrices.transpose(1, 2)
        new_translation = -torch.bmm(rotation_transposed, translation_vectors)
        inverted_matrix = torch.eye(4, 4)[None].repeat(len(rotation_matrices), 1, 1)
        inverted_matrix = inverted_matrix.to(rotation_matrices.dtype).to(
            rotation_matrices.device
        )
    inverted_matrix[:, :3, :3] = rotation_transposed
    inverted_matrix[:, :3, 3:] = new_translation

    return inverted_matrix

def load_colmap_data(colmap_path, stride=1, verbose=False, return_raw_data=False, device='cuda'):
    """
    Load COLMAP format data.

    Expected folder structure:
    colmap_path/
      images/
        img1.jpg
        img2.jpg
        ...
      sparse/
        cameras.bin/txt
        images.bin/txt
        points3D.bin/txt

    Args:
        colmap_path (str): Path to the main folder containing images/ and sparse/ subfolders
        stride (int): Load every nth image (default: 50)
        verbose (bool): Print progress messages
        return_raw_data (bool): Return raw COLMAP data (cameras, images, points3D)

    Returns:
        list: List of view dictionaries
    """
    # Define paths
    images_folder = os.path.join(colmap_path, "images")
    sparse_folder = os.path.join(colmap_path, "sparse")

    # Check that required folders exist
    if not os.path.exists(images_folder):
        raise ValueError(f"Required folder 'images' not found at: {images_folder}")
    if not os.path.exists(sparse_folder):
        sparse_folder = os.path.join(colmap_path, "all-sparse")
    if os.path.exists(os.path.join(sparse_folder, "0")):
        sparse_folder = os.path.join(sparse_folder, "0")

    if verbose:
        print(f"Loading COLMAP data from: {colmap_path}")
        print(f"Images folder: {images_folder}")
        print(f"Sparse folder: {sparse_folder}")

    # Read COLMAP model
    images_file = os.path.join(sparse_folder, 'images.bin')
    if os.path.exists(images_file):
        cameras, images_colmap, points3D = read_model(sparse_folder, ext='.bin')
    else:
        images_file = os.path.join(sparse_folder, "images" + '.txt')
        if os.path.exists(images_file):
            cameras, images_colmap, points3D = read_model(sparse_folder, ext='.txt')
        else:
            raise ValueError(f"Failed to read COLMAP model from {sparse_folder}: {e}")

    if verbose:
        print(
            f"Loaded COLMAP model with {len(cameras)} cameras, {len(images_colmap)} images, {len(points3D)} 3D points"
        )

    # Get list of available image files
    available_images = set()
    for f in os.listdir(images_folder):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            available_images.add(f)

    if not available_images:
        raise ValueError(f"No image files found in {images_folder}")

    views_example = []
    processed_count = 0

    # Get a list of all colmap image names
    colmap_image_names = set(img_info.name for img_info in images_colmap.values())
    # Find the unposed images (in images/ but not in colmap)
    unposed_images = available_images - colmap_image_names

    if verbose:
        print(f"Found {len(unposed_images)} images without COLMAP poses")

    # Process images in COLMAP order
    for img_id, img_info in images_colmap.items():
        # Apply stride
        if processed_count % stride != 0:
            processed_count += 1
            continue

        img_name = img_info.name

        # Check if image file exists
        image_path = os.path.join(images_folder, img_name)
        if not os.path.exists(image_path):
            if verbose:
                print(f"Warning: Image file not found for {img_name}, skipping")
            processed_count += 1
            continue

        try:
            # Load image
            image = PILImage.open(image_path).convert("RGB")
            image_array = np.array(image).astype(np.uint8)  # (H, W, 3) - [0, 255]

            # Get camera info
            cam_info = cameras[img_info.camera_id]
            cam_params = cam_info.params

            # Get intrinsic matrix
            K, _ = get_camera_matrix(
                camera_params=cam_params, camera_model=cam_info.model
            )

            # Get pose (COLMAP provides world2cam, we need cam2world)
            # COLMAP: world2cam rotation and translation
            C_R_G, C_t_G = qvec2rotmat(img_info.qvec), img_info.tvec

            # Create 4x4 world2cam pose matrix
            world2cam_matrix = np.eye(4)
            world2cam_matrix[:3, :3] = C_R_G
            world2cam_matrix[:3, 3] = C_t_G

            # Convert to cam2world using closed form pose inverse
            pose_matrix = closed_form_pose_inverse(world2cam_matrix[None, :, :])[0]

            # Convert to tensors
            image_tensor = torch.from_numpy(image_array).to(device)  # (H, W, 3)
            intrinsics_tensor = torch.from_numpy(K.astype(np.float32)).to(device)  # (3, 3)
            pose_tensor = torch.from_numpy(pose_matrix.astype(np.float32)).to(device)  # (4, 4)

            # Create view dictionary for MapAnything inference
            view = {
                "img": image_tensor,  # (H, W, 3) - [0, 255]
                "img_name": img_name,
                "img_id": img_info.id,
                "intrinsics": intrinsics_tensor,  # (3, 3)
                "cam2world": pose_tensor,  # (4, 4) in OpenCV cam2world convention
                "world2cam": torch.from_numpy(world2cam_matrix.astype(np.float32)).to(device),  # (4, 4)
            }

            views_example.append(view)
            processed_count += 1

            if verbose:
                print(
                    f"Loaded view {len(views_example) - 1}: {img_name} (shape: {image_array.shape})"
                )
                print(f"  - Camera ID: {img_info.camera_id}")
                print(f"  - Camera Model: {cam_info.model}")
                print(f"  - Image ID: {img_id}")

        except Exception as e:
            if verbose:
                print(f"Warning: Failed to load data for {img_name}: {e}")
            processed_count += 1
            continue
    
    # process unposed images (without COLMAP poses)
    for img_name in unposed_images:
        # Apply stride
        if processed_count % stride != 0:
            processed_count += 1
            continue

        image_path = os.path.join(images_folder, img_name)

        try:
            # Load image
            image = Image.open(image_path).convert("RGB")
            image_array = np.array(image).astype(np.uint8)  # (H, W, 3) - [0, 255]

            # Convert to tensor
            image_tensor = torch.from_numpy(image_array).to(device)  # (H, W, 3)

            view = {
                "img": image_tensor,  # (H, W, 3) - [0, 255]
                # No intrinsics or pose available
            }

            views_example.append(view)
            processed_count += 1

            if verbose:
                print(
                    f"Loaded unposed view {len(views_example) - 1}: {img_name} (shape: {image_array.shape})"
                )

        except Exception as e:
            if verbose:
                print(f"Warning: Failed to load data for {img_name}: {e}")
            processed_count += 1
            continue


    if not views_example:
        raise ValueError("No valid images found")

    # Sort views_example by img_name
    views_example = sorted(views_example, key=lambda x: x["img_name"])

    if verbose:
        print(f"Successfully loaded {len(views_example)} views with stride={stride}")

    if return_raw_data:
        return views_example, cameras, images_colmap, points3D
    else:
        return views_example

def load_selected_camera_params(json_path, return_dict=False, keep_metadata=False):
    """
    Load camera_params.json saved by save_selected_camera_params().

    Args:
        json_path: Path to camera_params.json.
        return_dict: If True, return a dict keyed by camera_name.
        keep_metadata: If True, return camera spec dicts containing both the
            MiniCam object and the original metadata. Otherwise return MiniCam objects only.

    Returns:
        If keep_metadata is False:
            list[MiniCam] or dict[str, MiniCam]
        If keep_metadata is True:
            list[dict] or dict[str, dict]
    """
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    camera_model = payload.get("camera_model")
    if camera_model not in (None, "MiniCam"):
        raise ValueError(f"Unsupported camera_model: {camera_model}")

    frames = payload.get("frames", [])
    camera_specs = []

    for idx, frame in enumerate(frames):
        c2w = np.asarray(frame["c2w"], dtype=np.float32)
        if c2w.shape != (4, 4):
            raise ValueError(
                f"Frame {idx} has invalid c2w shape: {c2w.shape}, expected (4, 4)"
            )

        fovx = float(frame["fovx"])
        fovy = float(frame["fovy"])
        width = int(frame["width"])
        height = int(frame["height"])
        near = float(frame.get("near", 0.02))
        cx_p = float(frame.get("cx_p", 0.5))
        cy_p = float(frame.get("cy_p", 0.5))
        camera_name = frame.get("camera_name", f"{idx:03d}")

        cam = MiniCam(
            c2w=c2w,
            fovx=fovx,
            fovy=fovy,
            width=width,
            height=height,
            near=near,
            cx_p=cx_p,
            cy_p=cy_p,
        )

        # Keep names and provenance for downstream use.
        cam.image_name = camera_name
        cam.source_view_index = frame.get("source_view_index")
        cam.source_image_name = frame.get("source_image_name")

        spec = {
            "camera": cam,
            "camera_name": camera_name,
            "source_view_index": frame.get("source_view_index"),
            "source_image_name": frame.get("source_image_name"),
            "width": width,
            "height": height,
            "near": near,
            "cx_p": cx_p,
            "cy_p": cy_p,
            "fov_deg": float(frame.get("fov_deg", np.rad2deg(fovx))),
            "fovx": fovx,
            "fovy": fovy,
            "c2w": c2w,
        }
        camera_specs.append(spec)

    if keep_metadata:
        if return_dict:
            return {spec["camera_name"]: spec for spec in camera_specs}
        return camera_specs

    cameras = [spec["camera"] for spec in camera_specs]
    if return_dict:
        return {cam.image_name: cam for cam in cameras}
    return cameras