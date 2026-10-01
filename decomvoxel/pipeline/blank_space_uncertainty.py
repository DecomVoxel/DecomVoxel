def get_scene_vis_grid(
    model_path: str,
    resolution: int = 256,
    certainty_threshold: int = 3,
    checkpoint_name: str = "iter020000_model.pt",
    device: str = "cuda",
):
    from decomvoxel.utils.vis_grid import VisibilityGrid

    return VisibilityGrid.from_model_path(
        model_path=model_path,
        resolution=resolution,
        certainty_threshold=certainty_threshold,
        checkpoint_name=checkpoint_name,
        device=device,
    )


def _trilinear_interp(
    grid: "torch.Tensor",
    coords: "torch.Tensor",
) -> "torch.Tensor":
    """
    Trilinear interpolation on a 3D grid.

    Parameters
    ----------
    grid   : (nx, ny, nz) float32 tensor on some device.
    coords : (N, 3) float tensor – **continuous voxel-centre indices**.
             coord[i] == j means the sample falls exactly on voxel-centre j.
             Computed as  ``(world - bbox_min) / voxel_size - 0.5``.

    Returns
    -------
    values : (N,) float32 tensor.
    """
    import torch

    nx, ny, nz = grid.shape
    device = grid.device
    coords = coords.to(device)

    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

    x0 = x.floor().long().clamp(0, nx - 2)
    y0 = y.floor().long().clamp(0, ny - 2)
    z0 = z.floor().long().clamp(0, nz - 2)
    x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1

    wx = (x - x0.float()).clamp(0.0, 1.0)
    wy = (y - y0.float()).clamp(0.0, 1.0)
    wz = (z - z0.float()).clamp(0.0, 1.0)

    return (
        grid[x0, y0, z0] * (1 - wx) * (1 - wy) * (1 - wz)
        + grid[x0, y0, z1] * (1 - wx) * (1 - wy) * wz
        + grid[x0, y1, z0] * (1 - wx) * wy * (1 - wz)
        + grid[x0, y1, z1] * (1 - wx) * wy * wz
        + grid[x1, y0, z0] * wx * (1 - wy) * (1 - wz)
        + grid[x1, y0, z1] * wx * (1 - wy) * wz
        + grid[x1, y1, z0] * wx * wy * (1 - wz)
        + grid[x1, y1, z1] * wx * wy * wz
    )


def get_object_blank_space_uncertainty(
    transform_info,
    sparse_structure,
    vis_grid=None,
    model_path: str = None,
    device: str = "cuda",
):
    import os
    import torch
    from decomvoxel.utils.converting import inverse_transform_vertices

    # ---- 1. Obtain visibility certainty grid ----------------------------
    if vis_grid is not None:
        certainty_grid = vis_grid.certainty_grid.to(device)  # (nx, ny, nz)
        bbox_min = vis_grid.bbox_min.to(device)              # (3,)
        voxel_size = vis_grid.voxel_size.to(device)          # scalar
        nx, ny, nz = vis_grid.nx, vis_grid.ny, vis_grid.nz
    elif model_path is not None:
        vis_path = os.path.join(model_path, "vis_grid", "vis_grid.pt")
        d = torch.load(vis_path, map_location="cpu", weights_only=True)
        certainty_grid = d["certainty_grid"].to(device)
        bbox_min       = d["bbox_min"].to(device)
        voxel_size     = d["voxel_size"].to(device)
        nx, ny, nz     = int(d["nx"]), int(d["ny"]), int(d["nz"])
    else:
        raise ValueError(
            "Either `vis_grid` or `model_path` must be provided to "
            "get_object_blank_space_uncertainty."
        )

    # ---- 2. Find blank voxels ------------------------------------------
    data = sparse_structure.data.squeeze()   # (res, res, res)
    blank_mask    = data < 0.5               # True where empty
    blank_indices = torch.argwhere(blank_mask).float()  # (M, 3)

    if blank_indices.shape[0] == 0:
        return torch.zeros_like(data)

    # ---- 3. Blank-voxel centres → mesh space [-0.5, 0.5] ----------------
    res = transform_info.resolution
    # (i + 0.5) / res gives the normalised [0, 1] coord of the cell centre;
    # subtracting 0.5 shifts to the [-0.5, 0.5] mesh / TRELLIS space.
    mesh_coords = (blank_indices + 0.5) / res - 0.5   # (M, 3)

    # ---- 4. Inverse transform → world space (handles Z-rotation) --------
    world_coords = inverse_transform_vertices(
        mesh_coords.cpu(), transform_info, is_glb=False
    )  # returns numpy array or torch Tensor
    if not isinstance(world_coords, torch.Tensor):
        world_coords = torch.from_numpy(world_coords).float()
    world_coords = world_coords.to(device)   # (M, 3)

    # ---- 5. Trilinear interpolation on the certainty grid ---------------
    # Continuous voxel-centre index: 0.0 → centre of voxel 0, 1.0 → voxel 1, …
    grid_coords = (world_coords - bbox_min) / voxel_size - 0.5   # (M, 3)
    certainty_values = _trilinear_interp(certainty_grid, grid_coords)  # (M,)

    # ---- 6. Assemble output map -----------------------------------------
    uncertainty_map = torch.zeros_like(data)
    uncertainty_map[blank_mask] = (1.0 - certainty_values).clamp(0.0, 1.0)

    n_blank = int(blank_mask.sum().item())
    mean_unc = float(uncertainty_map[blank_mask].mean().item())
    print(
        f"[get_object_blank_space_uncertainty] "
        f"{n_blank} blank voxels, mean uncertainty = {mean_unc:.4f}"
    )
    return uncertainty_map