import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import einops as ein


def depthmap_to_camera_frame(depthmap, intrinsics):
    """
    Convert depth image to a pointcloud in camera frame.

    Args:
        - depthmap: HxW or BxHxW torch tensor
        - intrinsics: 3x3 or Bx3x3 torch tensor

    Returns:
        pointmap in camera frame (HxWx3 or BxHxWx3 tensor), and a mask specifying valid pixels.
    """
    # Add batch dimension if not present
    if depthmap.dim() == 2:
        depthmap = depthmap.unsqueeze(0)
        intrinsics = intrinsics.unsqueeze(0)
        squeeze_batch_dim = True
    else:
        squeeze_batch_dim = False

    batch_size, height, width = depthmap.shape
    device = depthmap.device

    # Compute 3D point in camera frame associated with each pixel
    x_grid, y_grid = torch.meshgrid(
        torch.arange(width, device=device).float(),
        torch.arange(height, device=device).float(),
        indexing="xy",
    )
    x_grid = x_grid.unsqueeze(0).expand(batch_size, -1, -1)
    y_grid = y_grid.unsqueeze(0).expand(batch_size, -1, -1)

    fx = intrinsics[:, 0, 0].view(-1, 1, 1)
    fy = intrinsics[:, 1, 1].view(-1, 1, 1)
    cx = intrinsics[:, 0, 2].view(-1, 1, 1)
    cy = intrinsics[:, 1, 2].view(-1, 1, 1)

    depth_z = depthmap
    xx = (x_grid - cx) * depth_z / fx
    yy = (y_grid - cy) * depth_z / fy
    pts3d_cam = torch.stack((xx, yy, depth_z), dim=-1)

    # Compute mask of valid non-zero depth pixels
    valid_mask = depthmap > 0.0

    # Remove batch dimension if it was added
    if squeeze_batch_dim:
        pts3d_cam = pts3d_cam.squeeze(0)
        valid_mask = valid_mask.squeeze(0)

    return pts3d_cam, valid_mask


def depthmap_to_world_frame(depthmap, intrinsics, camera_pose=None):
    """
    Convert depth image to a pointcloud in world frame.

    Args:
        - depthmap: HxW or BxHxW torch tensor
        - intrinsics: 3x3 or Bx3x3 torch tensor
        - camera_pose: 4x4 or Bx4x4 torch tensor

    Returns:
        pointmap in world frame (HxWx3 or BxHxWx3 tensor), and a mask specifying valid pixels.
    """
    pts3d_cam, valid_mask = depthmap_to_camera_frame(depthmap, intrinsics)

    if camera_pose is not None:
        # Add batch dimension if not present
        if camera_pose.dim() == 2:
            camera_pose = camera_pose.unsqueeze(0)
            pts3d_cam = pts3d_cam.unsqueeze(0)
            squeeze_batch_dim = True
        else:
            squeeze_batch_dim = False

        # Convert points from camera frame to world frame
        pts3d_cam_homo = torch.cat(
            [pts3d_cam, torch.ones_like(pts3d_cam[..., :1])], dim=-1
        )
        pts3d_world = ein.einsum(
            camera_pose, pts3d_cam_homo, "b i k, b h w k -> b h w i"
        )
        pts3d_world = pts3d_world[..., :3]

        # Remove batch dimension if it was added
        if squeeze_batch_dim:
            pts3d_world = pts3d_world.squeeze(0)
    else:
        pts3d_world = pts3d_cam

    return pts3d_world, valid_mask


def transform_pts3d(pts3d, transformation):
    """
    Transform 3D points using a 4x4 transformation matrix.

    Args:
        - pts3d: HxWx3 or BxHxWx3 torch tensor
        - transformation: 4x4 or Bx4x4 torch tensor

    Returns:
        transformed points (HxWx3 or BxHxWx3 tensor)
    """
    # Add batch dimension if not present
    if pts3d.dim() == 3:
        pts3d = pts3d.unsqueeze(0)
        transformation = transformation.unsqueeze(0)
        squeeze_batch_dim = True
    else:
        squeeze_batch_dim = False

    # Convert points to homogeneous coordinates
    pts3d_homo = torch.cat([pts3d, torch.ones_like(pts3d[..., :1])], dim=-1)

    # Transform points
    transformed_pts3d = ein.einsum(
        transformation, pts3d_homo, "b i k, b h w k -> b h w i"
    )
    transformed_pts3d = transformed_pts3d[..., :3]

    # Remove batch dimension if it was added
    if squeeze_batch_dim:
        transformed_pts3d = transformed_pts3d.squeeze(0)

    return transformed_pts3d


def project_pts3d_to_image(pts3d, intrinsics, return_z_dim):
    """
    Project 3D points to image plane (assumes pinhole camera model with no distortion).

    Args:
        - pts3d: HxWx3 or BxHxWx3 torch tensor
        - intrinsics: 3x3 or Bx3x3 torch tensor
        - return_z_dim: bool, whether to return the third dimension of the projected points

    Returns:
        projected points (HxWx2)
    """
    if pts3d.dim() == 3:
        pts3d = pts3d.unsqueeze(0)
        intrinsics = intrinsics.unsqueeze(0)
        squeeze_batch_dim = True
    else:
        squeeze_batch_dim = False

    # Project points to image plane
    projected_pts2d = ein.einsum(intrinsics, pts3d, "b i k, b h w k -> b h w i")
    
    # Avoid inplace operation for gradient safety
    z = projected_pts2d[..., 2:3]
    xy = projected_pts2d[..., :2] / torch.clamp(z, min=1e-6)
    projected_pts2d = torch.cat([xy, z], dim=-1)

    # Remove the z dimension if not required
    if not return_z_dim:
        projected_pts2d = projected_pts2d[..., :2]

    # Remove batch dimension if it was added
    if squeeze_batch_dim:
        projected_pts2d = projected_pts2d.squeeze(0)

    return projected_pts2d


def compute_world_points(depth_tensor, view_dict):
    intrinsics = view_dict["intrinsics"].to(torch.float32)
    cam2world = view_dict["cam2world"].to(torch.float32)

    pts_world, valid_mask = depthmap_to_world_frame(depth_tensor, intrinsics, cam2world)
    return pts_world, valid_mask


def warp_points_to_view(
    src_pts_world,
    src_valid_mask,
    tgt_view,
    tgt_depth_tensor,
    depth_threshold=0.01,
):
    world2cam = tgt_view["world2cam"].to(torch.float32)
    intrinsics = tgt_view["intrinsics"].to(torch.float32)

    pts_in_tgt = transform_pts3d(src_pts_world, world2cam)
    proj = project_pts3d_to_image(pts_in_tgt, intrinsics, return_z_dim=True)

    proj_xy = proj[..., :2]
    proj_depth = proj[..., 2]

    H_tgt, W_tgt = tgt_view["img"].shape[0:2]
    in_image = (
        (proj_xy[..., 0] >= 0.0)
        & (proj_xy[..., 0] <= W_tgt - 1)
        & (proj_xy[..., 1] >= 0.0)
        & (proj_xy[..., 1] <= H_tgt - 1)
    )
    depth_positive = proj_depth > 0.0

    u_int = torch.clamp(torch.round(proj_xy[..., 0]).long(), 0, W_tgt - 1)
    v_int = torch.clamp(torch.round(proj_xy[..., 1]).long(), 0, H_tgt - 1)
    tgt_depth_sampled = tgt_depth_tensor[v_int, u_int]

    # in_image_valid_mask: whether the projected point is in the image
    in_image_valid_mask = (depth_positive & in_image & src_valid_mask)

    # depth_valid_mask: whether the depth difference is less than the depth threshold, avoid occlusion
    depth_diff = torch.abs(proj_depth - tgt_depth_sampled)
    relative_diff = depth_diff / (proj_depth + 1e-6)
    depth_valid_mask = relative_diff < depth_threshold

    return in_image_valid_mask, depth_valid_mask, u_int, v_int, proj_depth


def depth2normal_parallel(depth, intrinsics, cam2world):
    _, H, W = depth.shape        # [B, H, W]
    points, _ = depthmap_to_world_frame(depth, intrinsics, cam2world)
    points = points.reshape(-1, H, W, 3)
    output = torch.zeros_like(points)
    dx = points[:, 2:, 1:-1] - points[:, :-2, 1:-1]
    dy = points[:, 1:-1, 2:] - points[:, 1:-1, :-2]
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[:, 1:-1, 1:-1, :] = normal_map
    output[:, 0, 1:-1, :] = output[:, 1, 1:-1, :]
    output[:, -1, 1:-1, :] = output[:, -2, 1:-1, :]
    output[:, :, 0, :] = output[:, :, 1, :]
    output[:, :, -1, :] = output[:, :, -2, :]
    return output


def pnts2normal_parallel(points):
    # points: [B, H, W, 3]
    output = torch.zeros_like(points)
    dx = points[:, 2:, 1:-1] - points[:, :-2, 1:-1]
    dy = points[:, 1:-1, 2:] - points[:, 1:-1, :-2]
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[:, 1:-1, 1:-1, :] = normal_map
    output[:, 0, 1:-1, :] = output[:, 1, 1:-1, :]
    output[:, -1, 1:-1, :] = output[:, -2, 1:-1, :]
    output[:, :, 0, :] = output[:, :, 1, :]
    output[:, :, -1, :] = output[:, :, -2, :]
    return output


def normal2curv_parallel(normal, mask=None):
    """Reworked. Originally comes from Gaussian Surfels.
    
    Args:
        normal (torch.Tensor): (n_camera, H, W, 3) or (n_camera, 3, H, W)
        mask (torch.Tensor): (n_camera, H, W, 1) or (n_camera, 1, H, W)
    """
    # normal = normal.detach()
    if mask is None:
        mask = torch.ones_like(normal[..., 0:1])
    if normal.shape[-1] != 3:
        n = normal.permute(0, -2, -1, -3)
        m = mask.permute(0, -2, -1, -3)
    else:
        n = normal.clone()
        m = mask.clone()
    n = torch.nn.functional.pad(n[:, None], [0, 0, 1, 1, 1, 1], mode='replicate')
    m = torch.nn.functional.pad(m[:, None].to(torch.float32), [0, 0, 1, 1, 1, 1], mode='replicate').to(torch.bool)
    n_c = (n[..., 1:-1, 1:-1, :]      ) * m[..., 1:-1, 1:-1, :]
    n_u = (n[...,  :-2, 1:-1, :] - n_c) * m[...,  :-2, 1:-1, :]
    n_l = (n[..., 1:-1,  :-2, :] - n_c) * m[..., 1:-1,  :-2, :]
    n_b = (n[..., 2:  , 1:-1, :] - n_c) * m[..., 2:  , 1:-1, :]
    n_r = (n[..., 1:-1, 2:  , :] - n_c) * m[..., 1:-1, 2:  , :]
    curv = (n_u + n_l + n_b + n_r)[:, 0]
    curv = curv.norm(p=1, dim=-1, keepdim=True)
    if normal.shape[-1] != 3:
        curv = curv.permute(0, -1, -3, -2) * mask
    else:
        curv = curv * mask
    return curv

def pts2dist_map(pts_world_i):
    """ Compute the distance between adjacent pixels, used to initialize Gaussian scale
    Args:
        pts_world_i: [H, W, 3]
    Returns:
        min_dist: [H, W]
    """
    
    # 1. Compute horizontal distance between adjacent points [H, W-1]
    # dist_h[i, j] = distance(pts[i, j], pts[i, j+1])
    dist_h = torch.norm(pts_world_i[:, 1:, :] - pts_world_i[:, :-1, :], dim=-1)
    
    # 2. Compute vertical distance between adjacent points [H-1, W]
    # dist_v[i, j] = distance(pts[i, j], pts[i+1, j])
    dist_v = torch.norm(pts_world_i[1:, :, :] - pts_world_i[:-1, :, :], dim=-1)
    
    # 3. Construct distance maps for four directions [H, W]
    # For boundary pixels, use the logic: if there is no neighbor in that direction, 
    # use the distance to the neighbor in the opposite direction.
    
    # Right distance: [H, W] -> Last column copies the second to last column
    dist_r = torch.cat([dist_h, dist_h[:, -1:]], dim=1)
    # Left distance: [H, W] -> First column copies the second column
    dist_l = torch.cat([dist_h[:, :1], dist_h], dim=1)
    # Down distance: [H, W] -> Last row copies the second to last row
    dist_d = torch.cat([dist_v, dist_v[-1:, :]], dim=0)
    # Up distance: [H, W] -> First row copies the second row
    dist_u = torch.cat([dist_v[:1, :], dist_v], dim=0)
    
    # 4. Take the minimum value among four directions
    # Stack the four [H, W] matrices and take the minimum along dim=0
    stacked_dists = torch.stack([dist_r, dist_l, dist_d, dist_u], dim=0)
    min_dist, _ = torch.min(stacked_dists, dim=0)
    
    return min_dist

