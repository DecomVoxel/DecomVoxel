import json
import os

import numpy as np
import torch


def _to_dense_stat_grids(latent, grid_resolution: int = 64):
    coords = latent.coords.detach().cpu().long()
    if coords.shape[1] == 4:
        coords = coords[:, 1:]
    feats = latent.feats.detach().float().cpu()

    voxel_mean = feats.mean(dim=1)
    voxel_var = feats.var(dim=1, unbiased=False)

    mean_grid = torch.full((grid_resolution, grid_resolution, grid_resolution), float('nan'))
    var_grid = torch.full((grid_resolution, grid_resolution, grid_resolution), float('nan'))
    mask_grid = torch.zeros((grid_resolution, grid_resolution, grid_resolution), dtype=torch.bool)

    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    mean_grid[x, y, z] = voxel_mean
    var_grid[x, y, z] = voxel_var
    mask_grid[x, y, z] = True
    return mean_grid.numpy(), var_grid.numpy(), mask_grid.numpy()


def _to_dense_non_zero_init_grids(
    voxel_indices: torch.Tensor,
    unseen_mask: torch.Tensor,
    filled_mask: torch.Tensor,
    grid_resolution: int = 64,
):
    coords = voxel_indices.detach().cpu().long()
    unseen = unseen_mask.detach().cpu().bool()
    filled = filled_mask.detach().cpu().bool()

    sparse_grid = torch.zeros((grid_resolution, grid_resolution, grid_resolution), dtype=torch.bool)
    unseen_grid = torch.zeros((grid_resolution, grid_resolution, grid_resolution), dtype=torch.bool)
    filled_grid = torch.zeros((grid_resolution, grid_resolution, grid_resolution), dtype=torch.bool)

    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    sparse_grid[x, y, z] = True
    unseen_grid[x[unseen], y[unseen], z[unseen]] = True
    filled_grid[x[filled], y[filled], z[filled]] = True
    return sparse_grid.numpy(), unseen_grid.numpy(), filled_grid.numpy()


def _to_dense_certainty_weight_grid(
    voxel_coords: torch.Tensor,
    certainty_weights: torch.Tensor,
    grid_resolution: int = 64,
):
    coords = voxel_coords.detach().cpu().long()
    if coords.shape[1] == 4:
        coords = coords[:, 1:]
    weights = certainty_weights.detach().float().cpu().flatten()

    if coords.shape[0] != weights.shape[0]:
        raise ValueError(
            f"coords/weights size mismatch: {coords.shape[0]} vs {weights.shape[0]}"
        )

    weight_grid = torch.full((grid_resolution, grid_resolution, grid_resolution), float('nan'))
    sparse_grid = torch.zeros((grid_resolution, grid_resolution, grid_resolution), dtype=torch.bool)

    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    weight_grid[x, y, z] = weights
    sparse_grid[x, y, z] = True
    return weight_grid.numpy(), sparse_grid.numpy()


def _mean_projection(grid: np.ndarray, mask: np.ndarray, axis: int) -> np.ndarray:
    sum_v = np.sum(np.where(mask, grid, 0.0), axis=axis)
    cnt = np.sum(mask, axis=axis)
    proj = np.divide(sum_v, np.maximum(cnt, 1))
    proj[cnt == 0] = np.nan
    return proj


def _masked_slice(grid: np.ndarray, mask: np.ndarray, axis: int, idx: int) -> np.ndarray:
    if axis == 0:
        sl = grid[idx, :, :]
        mk = mask[idx, :, :]
    elif axis == 1:
        sl = grid[:, idx, :]
        mk = mask[:, idx, :]
    else:
        sl = grid[:, :, idx]
        mk = mask[:, :, idx]
    return np.where(mk, sl, np.nan)


def _plot_stat_grid(grid: np.ndarray, mask: np.ndarray, title: str, cmap: str, save_path: str) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    res = grid.shape[0]
    mid = res // 2

    images = [
        _mean_projection(grid, mask, axis=2),
        _mean_projection(grid, mask, axis=1),
        _mean_projection(grid, mask, axis=0),
        _masked_slice(grid, mask, axis=2, idx=mid),
        _masked_slice(grid, mask, axis=1, idx=mid),
        _masked_slice(grid, mask, axis=0, idx=mid),
    ]
    names = [
        'Proj XY (mean over Z)',
        'Proj XZ (mean over Y)',
        'Proj YZ (mean over X)',
        f'Slice XY @ Z={mid}',
        f'Slice XZ @ Y={mid}',
        f'Slice YZ @ X={mid}',
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle(title, fontsize=14)

    for ax, img, name in zip(axes.ravel(), images, names):
        im = ax.imshow(img.T, cmap=cmap, origin='lower')
        ax.set_title(name)
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)

    plt.tight_layout(rect=[0, 0.01, 1, 0.96])
    plt.savefig(save_path, dpi=160, bbox_inches='tight')
    plt.close(fig)


def visualize_slat(latent, output_dir: str, prefix: str = 'slat', grid_resolution: int = 64) -> dict:
    os.makedirs(output_dir, exist_ok=True)

    mean_grid, var_grid, mask_grid = _to_dense_stat_grids(latent, grid_resolution=grid_resolution)

    mean_pt = os.path.join(output_dir, f'{prefix}_mean_grid.pt')
    var_pt = os.path.join(output_dir, f'{prefix}_var_grid.pt')
    torch.save(torch.from_numpy(mean_grid), mean_pt)
    torch.save(torch.from_numpy(var_grid), var_pt)

    mean_png = os.path.join(output_dir, f'{prefix}_mean.png')
    var_png = os.path.join(output_dir, f'{prefix}_var.png')
    _plot_stat_grid(mean_grid, mask_grid, f'{prefix}: SLAT Mean', 'viridis', mean_png)
    _plot_stat_grid(var_grid, mask_grid, f'{prefix}: SLAT Variance', 'magma', var_png)

    stats = {
        'prefix': prefix,
        'n_voxels': int(mask_grid.sum()),
        'mean_min': float(np.nanmin(mean_grid)),
        'mean_max': float(np.nanmax(mean_grid)),
        'mean_avg': float(np.nanmean(mean_grid)),
        'var_min': float(np.nanmin(var_grid)),
        'var_max': float(np.nanmax(var_grid)),
        'var_avg': float(np.nanmean(var_grid)),
        'files': {
            'mean_grid': mean_pt,
            'var_grid': var_pt,
            'mean_png': mean_png,
            'var_png': var_png,
        },
    }

    stats_path = os.path.join(output_dir, f'{prefix}_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"[vis_slat] Saved SLAT stats for '{prefix}' -> {output_dir}")
    return stats


def visualize_non_zero_init(
    voxel_indices: torch.Tensor,
    unseen_mask: torch.Tensor,
    filled_mask: torch.Tensor,
    output_dir: str,
    prefix: str = 'non_zero_init',
    grid_resolution: int = 64,
) -> dict:
    os.makedirs(output_dir, exist_ok=True)

    sparse_grid, unseen_grid, filled_grid = _to_dense_non_zero_init_grids(
        voxel_indices=voxel_indices,
        unseen_mask=unseen_mask,
        filled_mask=filled_mask,
        grid_resolution=grid_resolution,
    )

    unseen_pt = os.path.join(output_dir, f'{prefix}_unseen_grid.pt')
    filled_pt = os.path.join(output_dir, f'{prefix}_filled_grid.pt')
    torch.save(torch.from_numpy(unseen_grid), unseen_pt)
    torch.save(torch.from_numpy(filled_grid), filled_pt)

    unseen_png = os.path.join(output_dir, f'{prefix}_unseen.png')
    filled_png = os.path.join(output_dir, f'{prefix}_filled.png')
    _plot_stat_grid(unseen_grid.astype(np.float32), sparse_grid, f'{prefix}: Unseen Mask', 'Reds', unseen_png)
    _plot_stat_grid(filled_grid.astype(np.float32), sparse_grid, f'{prefix}: Filled Mask', 'Greens', filled_png)

    n_sparse = int(sparse_grid.sum())
    n_unseen = int(unseen_grid.sum())
    n_filled = int(filled_grid.sum())

    stats = {
        'prefix': prefix,
        'n_sparse': n_sparse,
        'n_unseen': n_unseen,
        'n_filled': n_filled,
        'fill_ratio_in_unseen': float(n_filled / max(n_unseen, 1)),
        'files': {
            'unseen_grid': unseen_pt,
            'filled_grid': filled_pt,
            'unseen_png': unseen_png,
            'filled_png': filled_png,
        },
    }

    stats_path = os.path.join(output_dir, f'{prefix}_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(
        f"[vis_slat] Saved non-zero init stats for '{prefix}' -> {output_dir} "
        f"(unseen={n_unseen}, filled={n_filled})"
    )
    return stats


def visualize_certainty_weights(
    voxel_coords: torch.Tensor,
    certainty_weights: torch.Tensor,
    output_dir: str,
    prefix: str = 'certainty_weights',
    grid_resolution: int = 64,
) -> dict:
    os.makedirs(output_dir, exist_ok=True)

    weight_grid, sparse_grid = _to_dense_certainty_weight_grid(
        voxel_coords=voxel_coords,
        certainty_weights=certainty_weights,
        grid_resolution=grid_resolution,
    )

    weight_pt = os.path.join(output_dir, f'{prefix}_grid.pt')
    torch.save(torch.from_numpy(weight_grid), weight_pt)

    weight_png = os.path.join(output_dir, f'{prefix}.png')
    _plot_stat_grid(weight_grid, sparse_grid, f'{prefix}: Certainty Weights', 'cividis', weight_png)

    valid = weight_grid[sparse_grid]
    stats = {
        'prefix': prefix,
        'n_voxels': int(sparse_grid.sum()),
        'weight_min': float(valid.min()) if valid.size > 0 else float('nan'),
        'weight_max': float(valid.max()) if valid.size > 0 else float('nan'),
        'weight_avg': float(valid.mean()) if valid.size > 0 else float('nan'),
        'files': {
            'weight_grid': weight_pt,
            'weight_png': weight_png,
        },
    }

    stats_path = os.path.join(output_dir, f'{prefix}_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"[vis_slat] Saved certainty weights for '{prefix}' -> {output_dir}")
    return stats
