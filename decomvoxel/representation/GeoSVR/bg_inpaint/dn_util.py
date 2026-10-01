import os
import sys
import shutil
from argparse import ArgumentParser

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
GEOSVR_ROOT = os.path.dirname(PROJECT_ROOT)
if GEOSVR_ROOT not in sys.path:
    sys.path.insert(0, GEOSVR_ROOT)

from cam_util import load_selected_camera_params
from src.utils import mono_utils
from src.utils.render_utils import save_img_f32, save_img_u8


def read_rgb_tensor(image_path, width=None, height=None):
    image = Image.open(image_path).convert("RGB")
    if width is not None and height is not None and image.size != (width, height):
        image = image.resize((width, height), Image.BILINEAR)
    image_np = np.asarray(image, dtype=np.float32) / 255.0
    image_t = torch.from_numpy(image_np).permute(2, 0, 1).contiguous()
    return image_t


def read_mask_as_visible(mask_path, width=None, height=None):
    mask = Image.open(mask_path).convert("L")
    if width is not None and height is not None and mask.size != (width, height):
        mask = mask.resize((width, height), Image.NEAREST)
    mask_np = np.asarray(mask, dtype=np.uint8)
    visible = mask_np > 127
    return visible


def read_depth_tiff(depth_path, width=None, height=None):
    depth = np.asarray(Image.open(depth_path), dtype=np.float32)
    if width is not None and height is not None and depth.shape[:2] != (height, width):
        depth = np.array(
            Image.fromarray(depth).resize((width, height), Image.NEAREST),
            dtype=np.float32,
        )
    return depth


def align_mono_to_metric_depth(mono_disp, render_depth, visible_mask, near=1e-3):
    """
    Align DepthAnythingV2 output to rendered metric depth.

    DepthAnythingV2 output is treated as disparity-like:
    larger value means closer. We align it to rendered inverse depth
    with median/MAD normalization, then invert back to metric depth.
    """
    mono_disp = mono_disp.float()
    render_depth = render_depth.float()
    visible_mask = visible_mask.bool()

    valid = (
        visible_mask
        & torch.isfinite(mono_disp)
        & torch.isfinite(render_depth)
        & (render_depth > near)
    )

    if valid.sum() < 16:
        aligned_inv = mono_disp.clamp(min=near)
        return 1.0 / aligned_inv

    mono_valid = mono_disp[valid]
    render_inv_valid = 1.0 / render_depth[valid].clamp(min=near)

    mono_med = mono_valid.median()
    mono_mad = (mono_valid - mono_med).abs().mean().clamp(min=1e-6)

    render_med = render_inv_valid.median()
    render_mad = (render_inv_valid - render_med).abs().mean().clamp(min=1e-6)

    aligned_inv = (mono_disp - mono_med) * (render_mad / mono_mad) + render_med
    aligned_inv = aligned_inv.clamp(min=near)

    aligned_depth = 1.0 / aligned_inv
    return aligned_depth


def compute_world_and_camera_normals(cam, depth, ks=3):
    world_normal = cam.depth2normal(depth, ks=ks)

    camera_normal = (
        world_normal.reshape(3, -1).permute(1, 0) @ cam.world_view_transform[:3, :3]
    ).permute(1, 0).reshape_as(world_normal)

    return world_normal, camera_normal


def save_depth_vis(depth, save_path):
    depth_np = depth.detach().cpu().numpy()
    plt.imsave(save_path, depth_np, cmap="viridis")


@torch.no_grad()
def extract_see3d_mono_depth_and_normal(
    guidance_root,
    inpaint_root,
    save_root,
    force_rerun=False,
):
    os.makedirs(save_root, exist_ok=True)

    camera_json_path = os.path.join(guidance_root, "camera_params.json")
    warp_root_dir = os.path.join(guidance_root, "warp_images")
    raw_rgb_dir = os.path.join(guidance_root, "raw_rgb")
    raw_depth_dir = os.path.join(guidance_root, "raw_depth")

    cameras = load_selected_camera_params(camera_json_path)

    # Attach inpainted RGB to cameras so we can directly reuse mono_utils.prepare_depthanythingv2().
    for cam in cameras:
        frame_idx = int(cam.image_name)

        inpaint_img_path = os.path.join(
            inpaint_root,
            f"predict_warp_frame{frame_idx:06d}.png",
        )
        if os.path.exists(inpaint_img_path):
            rgb_path = inpaint_img_path
        else:
            raise ValueError(f"Inpaint image not found: {inpaint_img_path}")

        cam.image = read_rgb_tensor(
            rgb_path,
            width=int(cam.image_width),
            height=int(cam.image_height),
        )

    # Use a dedicated cache root to avoid colliding with training-set mono priors.
    mono_cache_root = os.path.join(guidance_root, "da2_mono_cache")
    os.makedirs(mono_cache_root, exist_ok=True)
    mono_utils.prepare_depthanythingv2(
        cameras=cameras,
        source_path=mono_cache_root,
        force_rerun=force_rerun,
    )

    for cam in cameras:
        frame_idx = int(cam.image_name)

        rgb_save_path = os.path.join(save_root, f"rgb_frame{frame_idx:06d}.png")
        inpaint_img_path = os.path.join(
            inpaint_root,
            f"predict_warp_frame{frame_idx:06d}.png",
        )
        if os.path.exists(inpaint_img_path):
            shutil.copy(inpaint_img_path, rgb_save_path)

        mask_path = os.path.join(warp_root_dir, f"mask_frame{frame_idx:06d}.png")
        depth_path = os.path.join(raw_depth_dir, f"depth_frame{frame_idx:06d}.tiff")

        if os.path.exists(mask_path):
            shutil.copy(mask_path, os.path.join(save_root, f"mask_frame{frame_idx:06d}.png"))

        visible_mask_np = read_mask_as_visible(
            mask_path,
            width=int(cam.image_width),
            height=int(cam.image_height),
        )
        render_depth_np = read_depth_tiff(
            depth_path,
            width=int(cam.image_width),
            height=int(cam.image_height),
        )

        visible_mask = torch.from_numpy(visible_mask_np).to(device="cuda")
        render_depth = torch.from_numpy(render_depth_np).to(device="cuda")

        # DepthAnythingV2 output stored by GeoSVR is disparity-like: larger = closer.
        mono_disp = cam.depthanythingv2.cuda().float()
        if mono_disp.shape != render_depth.shape:
            mono_disp = torch.nn.functional.interpolate(
                mono_disp[None, None],
                size=render_depth.shape,
                mode="bilinear",
                align_corners=False,
            )[0, 0]

        mono_depth = 1.0 / mono_disp.clamp(min=1e-6)
        aligned_depth = align_mono_to_metric_depth(
            mono_disp=mono_disp,
            render_depth=render_depth,
            visible_mask=visible_mask,
        )

        world_normal, camera_normal = compute_world_and_camera_normals(cam, aligned_depth, ks=3)

        world_normal_hwc = world_normal.permute(1, 2, 0).detach().cpu().numpy()
        camera_normal_hwc = camera_normal.permute(1, 2, 0).detach().cpu().numpy()

        save_img_f32(
            mono_depth.detach().cpu().numpy(),
            os.path.join(save_root, f"mono_depth_frame{frame_idx:06d}.tiff"),
        )
        save_depth_vis(
            mono_depth,
            os.path.join(save_root, f"mono_depth_frame{frame_idx:06d}.png"),
        )

        save_img_f32(
            aligned_depth.detach().cpu().numpy(),
            os.path.join(save_root, f"aligned_depth_frame{frame_idx:06d}.tiff"),
        )
        save_depth_vis(
            aligned_depth,
            os.path.join(save_root, f"aligned_depth_frame{frame_idx:06d}.png"),
        )

        np.save(
            os.path.join(save_root, f"mono_normal_world_frame{frame_idx:06d}.npy"),
            world_normal_hwc,
        )
        save_img_u8(
            world_normal_hwc * 0.5 + 0.5,
            os.path.join(save_root, f"mono_normal_world_frame{frame_idx:06d}.png"),
        )

        np.save(
            os.path.join(save_root, f"mono_normal_frame{frame_idx:06d}.npy"),
            camera_normal_hwc,
        )
        save_img_u8(
            camera_normal_hwc * 0.5 + 0.5,
            os.path.join(save_root, f"mono_normal_frame{frame_idx:06d}.png"),
        )

        print(f"frame {frame_idx:06d} done!")

    print("All frames done!")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--guidance_root", required=True, type=str,
                        help="Path to see3d_guidance_views root")
    parser.add_argument("--inpaint_root", type=str, default=None,
                        help="Path to See3D inpaint output directory")
    parser.add_argument("--save_root", type=str, default=None,
                        help="Directory to save depth and normal outputs")
    parser.add_argument("--force_rerun", action="store_true")
    args = parser.parse_args()

    if args.inpaint_root is None:
        args.inpaint_root = os.path.join(args.guidance_root, "see3d_inpaint_output")
    if args.save_root is None:
        args.save_root = os.path.join(args.guidance_root, "see3d_mono_dn")

    extract_see3d_mono_depth_and_normal(
        guidance_root=args.guidance_root,
        inpaint_root=args.inpaint_root,
        save_root=args.save_root,
        force_rerun=args.force_rerun,
    )