
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from typing import Optional, Dict, Tuple, Union, List

# Add project paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

GEOSVR_ROOT = os.path.join(PROJECT_ROOT, 'decomvoxel', 'representation', 'GeoSVR')
if GEOSVR_ROOT not in sys.path:
    sys.path.insert(0, GEOSVR_ROOT)

TRELLIS_ROOT = os.path.join(PROJECT_ROOT, 'decomvoxel', 'model', 'TRELLIS')
if TRELLIS_ROOT not in sys.path:
    sys.path.insert(0, TRELLIS_ROOT)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Import GeoSVR modules
from yacs.config import CfgNode

DEBUG=True


def visualize_grid_coords(grid_coords: torch.Tensor,
                          dense_grid: torch.Tensor = None,
                          resolution: int = 64,
                          save_path: str = None,
                          title: str = "Grid Coordinates Visualization",
                          sizes_in_cells: torch.Tensor = None):
    """
    Visualize grid coordinates with three orthogonal projection views + 3D scatter,
    combined into a single figure (2x2 layout).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    coords = grid_coords.cpu().numpy() if isinstance(grid_coords, torch.Tensor) else grid_coords
    
    # If dense_grid is provided, extract occupied coords from it (shows actual filled result)
    if dense_grid is not None:
        occupied = (dense_grid.squeeze() > 0.5).cpu().numpy()
        occ_coords = np.argwhere(occupied)  # (M, 3)
        use_dense = True
    else:
        use_dense = False
    
    # Color by voxel size if available
    if sizes_in_cells is not None:
        sc = sizes_in_cells.cpu().numpy() if isinstance(sizes_in_cells, torch.Tensor) else sizes_in_cells
        # Normalize to [0, 1] for colormap
        sc_min, sc_max = sc.min(), sc.max()
        if sc_max > sc_min:
            sc_norm = (sc - sc_min) / (sc_max - sc_min)
        else:
            sc_norm = np.zeros_like(sc)
        colors_center = plt.cm.viridis(sc_norm)
    else:
        colors_center = 'dodgerblue'
    
    fig = plt.figure(figsize=(16, 14))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    res = resolution
    dot_size_center = max(1, 80 // (res // 16))
    dot_size_dense = max(0.3, 20 // (res // 16))
    
    # --- Top-left: XY projection (top view, looking down Z) ---
    ax1 = fig.add_subplot(2, 2, 1)
    if use_dense:
        # Project: for each (x, y), check if any z is occupied
        proj_xy = occupied.any(axis=2)  # (X, Y)
        ax1.imshow(proj_xy.T, origin='lower', cmap='Greys', aspect='equal',
                   extent=[0, res, 0, res], alpha=0.3)
    ax1.scatter(coords[:, 0], coords[:, 1], s=dot_size_center, c=colors_center, 
                alpha=0.6, edgecolors='none')
    ax1.set_xlim(0, res)
    ax1.set_ylim(0, res)
    ax1.set_xlabel('X (grid)')
    ax1.set_ylabel('Y (grid)')
    ax1.set_title(f'XY Projection (top view, along Z)\ncenters: {len(coords)}')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.2)
    
    # --- Top-right: XZ projection (front view, looking along Y) ---
    ax2 = fig.add_subplot(2, 2, 2)
    if use_dense:
        proj_xz = occupied.any(axis=1)  # (X, Z)
        ax2.imshow(proj_xz.T, origin='lower', cmap='Greys', aspect='equal',
                   extent=[0, res, 0, res], alpha=0.3)
    ax2.scatter(coords[:, 0], coords[:, 2], s=dot_size_center, c=colors_center,
                alpha=0.6, edgecolors='none')
    ax2.set_xlim(0, res)
    ax2.set_ylim(0, res)
    ax2.set_xlabel('X (grid)')
    ax2.set_ylabel('Z (grid)')
    ax2.set_title(f'XZ Projection (front view, along Y)')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.2)
    
    # --- Bottom-left: YZ projection (side view, looking along X) ---
    ax3 = fig.add_subplot(2, 2, 3)
    if use_dense:
        proj_yz = occupied.any(axis=0)  # (Y, Z)
        ax3.imshow(proj_yz.T, origin='lower', cmap='Greys', aspect='equal',
                   extent=[0, res, 0, res], alpha=0.3)
    ax3.scatter(coords[:, 1], coords[:, 2], s=dot_size_center, c=colors_center,
                alpha=0.6, edgecolors='none')
    ax3.set_xlim(0, res)
    ax3.set_ylim(0, res)
    ax3.set_xlabel('Y (grid)')
    ax3.set_ylabel('Z (grid)')
    ax3.set_title(f'YZ Projection (side view, along X)')
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.2)
    
    # --- Bottom-right: 3D scatter ---
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    if use_dense:
        # Subsample dense coords if too many for 3D plot
        if len(occ_coords) > 8000:
            idx = np.random.choice(len(occ_coords), 8000, replace=False)
            occ_sub = occ_coords[idx]
        else:
            occ_sub = occ_coords
        ax4.scatter(occ_sub[:, 0], occ_sub[:, 1], occ_sub[:, 2],
                    s=dot_size_dense, c='lightgray', alpha=0.15, edgecolors='none',
                    label=f'filled ({len(occ_coords)})')
    # Plot centers on top
    if len(coords) > 5000:
        idx = np.random.choice(len(coords), 5000, replace=False)
        coords_sub = coords[idx]
        c_sub = colors_center[idx] if isinstance(colors_center, np.ndarray) else colors_center
    else:
        coords_sub = coords
        c_sub = colors_center
    ax4.scatter(coords_sub[:, 0], coords_sub[:, 1], coords_sub[:, 2],
                s=dot_size_center, c=c_sub, alpha=0.5, edgecolors='none',
                label=f'centers ({len(coords)})')
    ax4.set_xlim(0, res)
    ax4.set_ylim(0, res)
    ax4.set_zlim(0, res)
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_zlabel('Z')
    ax4.set_title('3D View')
    ax4.legend(fontsize=8, loc='upper left')
    
    # Stats text
    if use_dense:
        occ_count = occupied.sum()
        occ_ratio = occ_count / (res ** 3) * 100
        fig.text(0.5, 0.01, 
                 f"Resolution: {res}³ | Centers: {len(coords)} | "
                 f"Filled cells: {occ_count} ({occ_ratio:.1f}%) | "
                 f"Grid range: X[{coords[:,0].min()}-{coords[:,0].max()}] "
                 f"Y[{coords[:,1].min()}-{coords[:,1].max()}] "
                 f"Z[{coords[:,2].min()}-{coords[:,2].max()}]",
                 ha='center', fontsize=9, style='italic')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[visualize_grid_coords] Saved to: {save_path}")
    else:
        plt.show()
    plt.close(fig)
