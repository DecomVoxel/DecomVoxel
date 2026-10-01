import torch

def get_points_from_depth(fov_camera, depth, scale=1):
    st = int(max(int(scale/2)-1,0))
    depth_view = depth.squeeze()[st::scale,st::scale]
    W, H = int(fov_camera.image_width/scale), int(fov_camera.image_height/scale)
    ix, iy = torch.meshgrid(
        torch.arange(W), torch.arange(H), indexing='xy')
    rays_d = torch.stack(
                [(ix-fov_camera.Cx/scale) / fov_camera.Fx * scale,
                (iy-fov_camera.Cy/scale) / fov_camera.Fy * scale,
                torch.ones_like(ix)], -1).float().cuda()
    depth_view = depth_view[:rays_d.shape[0], :rays_d.shape[1]]
    pts = (rays_d * depth_view[..., None]).reshape(-1,3)
    R = torch.tensor(fov_camera.R).float().cuda()
    T = torch.tensor(fov_camera.T).float().cuda()
    pts = (pts-T)@R.transpose(-1,-2)
    return pts

def get_points_depth_in_depth_map(fov_camera, depth, points_in_camera_space, scale=1):
    st = max(int(scale/2)-1,0)
    depth_view = depth[None,:,st::scale,st::scale]
    W, H = int(fov_camera.image_width/scale), int(fov_camera.image_height/scale)
    depth_view = depth_view[:H, :W]
    pts_projections = torch.stack(
                    [points_in_camera_space[:,0] * fov_camera.Fx / points_in_camera_space[:,2] + fov_camera.Cx,
                        points_in_camera_space[:,1] * fov_camera.Fy / points_in_camera_space[:,2] + fov_camera.Cy], -1).float()/scale
    mask = (pts_projections[:, 0] > 0) & (pts_projections[:, 0] < W) &\
            (pts_projections[:, 1] > 0) & (pts_projections[:, 1] < H) & (points_in_camera_space[:,2] > 0.1)

    pts_projections[..., 0] /= ((W - 1) / 2)
    pts_projections[..., 1] /= ((H - 1) / 2)
    pts_projections -= 1
    pts_projections = pts_projections.view(1, -1, 1, 2)
    map_z = torch.nn.functional.grid_sample(input=depth_view,
                                            grid=pts_projections,
                                            mode='bilinear',
                                            padding_mode='border',
                                            align_corners=True
                                            )[0, :, :, 0]
    return map_z, mask