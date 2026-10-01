import numpy as np


def decide_main_bounding(cfg_bounding, tr_cams, pcd, suggested_bounding):
    if cfg_bounding.bound_mode == "default" and suggested_bounding is not None:
        print("Use suggested bounding")
        center = suggested_bounding.mean(0)
        radius = (suggested_bounding[1] - suggested_bounding[0]) * 0.5
    elif cfg_bounding.bound_mode in ["camera_max", "camera_median"]:
        center, radius = main_scene_bound_camera_heuristic(
            cams=tr_cams, bound_mode=cfg_bounding.bound_mode)
    elif cfg_bounding.bound_mode == "forward":
        center, radius = main_scene_bound_forward_heuristic(
            cams=tr_cams, forward_dist_scale=cfg_bounding.forward_dist_scale)
    elif cfg_bounding.bound_mode == "pcd":
        center, radius = main_scene_bound_pcd_heuristic(
            pcd=pcd, pcd_density_rate=cfg_bounding.pcd_density_rate)
    elif cfg_bounding.bound_mode == "default":
        cam_lookats = np.stack([cam.lookat.tolist() for cam in tr_cams])
        lookat_dots = (cam_lookats[:,None] * cam_lookats).sum(-1)
        is_forward_facing = lookat_dots.min() > 0

        if is_forward_facing:
            center, radius = main_scene_bound_forward_heuristic(
                cams=tr_cams, forward_dist_scale=cfg_bounding.forward_dist_scale)
        else:
            center, radius = main_scene_bound_camera_heuristic(
                cams=tr_cams, bound_mode="camera_median")
    else:
        raise NotImplementedError

    radius = radius * cfg_bounding.bound_scale

    bounding = np.array([
        center - radius,
        center + radius,
    ], dtype=np.float32)
    return bounding


def main_scene_bound_camera_heuristic(cams, bound_mode):
    print("Heuristic bounding:", bound_mode)
    cam_positions = np.stack([cam.position.tolist() for cam in cams])
    center = cam_positions.mean(0)
    dists = np.linalg.norm(cam_positions - center, axis=1)
    if bound_mode == "camera_max":
        radius = np.max(dists)
    elif bound_mode == "camera_median":
        radius = np.median(dists)
    else:
        raise NotImplementedError
    return center, radius


def main_scene_bound_forward_heuristic(cams, forward_dist_scale):
    print("Heuristic bounding: forward")
    positions = np.stack([cam.position.tolist() for cam in cams])
    cam_center = positions.mean(0)
    cam_lookat = np.stack([cam.lookat.tolist() for cam in cams]).mean(0)
    cam_lookat /= np.linalg.norm(cam_lookat)
    cam_extent = 2 * np.linalg.norm(positions - cam_center, axis=1).max()

    center = cam_center + forward_dist_scale * cam_extent * cam_lookat
    radius = 0.8 * forward_dist_scale * cam_extent

    return center, radius


def main_scene_bound_pcd_heuristic(pcd, pcd_density_rate):
    print("Heuristic bounding: pcd")
    center = np.median(pcd.points, axis=0)
    dist = np.abs(pcd.points - center).max(axis=1)
    dist = np.sort(dist)
    density = (1 + np.arange(len(dist))) * (dist > 0) / ((2 * dist) ** 3 + 1e-6)

    # Should cover at least 5% of the point
    begin_idx = round(len(density) * 0.05)

    # Find the radius with maximum point density
    max_idx = begin_idx + density[begin_idx:].argmax()

    # Find the smallest radius with point density equal to pcd_density_rate of maximum
    target_density = pcd_density_rate * density[max_idx]
    target_idx = max_idx + np.where(density[max_idx:] < target_density)[0][0]

    radius = dist[target_idx]

    return center, radius
