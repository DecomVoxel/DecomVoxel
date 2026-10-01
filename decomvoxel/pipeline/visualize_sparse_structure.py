"""
Sparse Structure Visualization Module

This module provides visualization utilities for TRELLIS sparse structures
(64^3 occupancy grids) with multiple output formats:
- 2D slice visualizations
- 3D voxel renderings
- Interactive HTML visualizations
- Comparison visualizations (before/after)

Usage:
    python visualize_sparse_structure.py --input path/to/ss.pt --output output_dir
    python visualize_sparse_structure.py --input path/to/ss.pt --mode 3d
    python visualize_sparse_structure.py --compare before.pt after.pt --output comparison.png
"""

import os
import sys
import argparse
import numpy as np
import torch
from typing import Optional, Tuple, List, Union
from pathlib import Path

# Add project paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class SparseStructureVisualizer:
    """
    Visualization utilities for sparse structure (64^3 occupancy grid).
    
    Supports multiple visualization modes:
    - slices: 2D orthogonal slices through the volume
    - projections: max projections along each axis
    - 3d: 3D voxel rendering using matplotlib
    - interactive: interactive HTML using plotly
    """
    
    def __init__(self, resolution: int = 64):
        """
        Initialize visualizer.
        
        Args:
            resolution: expected grid resolution (default: 64)
        """
        self.resolution = resolution
    
    @staticmethod
    def ensure_cpu_tensor(tensor):
        """
        Ensure the tensor is on CPU and detached from gradients.
        
        Args:
            tensor: any tensor
            
        Returns:
            tensor on CPU
        """
        if isinstance(tensor, torch.Tensor):
            if tensor.is_cuda:
                tensor = tensor.cpu()
            if tensor.requires_grad:
                tensor = tensor.detach()
        return tensor
    
    def load(self, path: str) -> torch.Tensor:
        """
        Load sparse structure from file.
        
        Args:
            path: path to .pt file
            
        Returns:
            (1, 1, R, R, R) or (R, R, R) tensor
        """
        data = torch.load(path, map_location='cpu', weights_only=True)
        
        # Normalize shape to (R, R, R)
        if data.dim() == 5:
            data = data[0, 0]
        elif data.dim() == 4:
            data = data[0]
        
        return data
    
    def visualize_slices(self, 
                         data: torch.Tensor,
                         save_path: str = None,
                         num_slices: int = 8,
                         threshold: float = 0.5,
                         show: bool = False) -> Optional[np.ndarray]:
        """
        Visualize orthogonal slices through the volume.
        
        Args:
            data: (R, R, R) occupancy tensor
            save_path: path to save image
            num_slices: number of slices per axis
            threshold: occupancy threshold for binary display
            show: whether to display the plot
            
        Returns:
            visualization image as numpy array if save_path is None
        """
        import matplotlib.pyplot as plt
        
        # Ensure tensor is on CPU
        data = self.ensure_cpu_tensor(data)
        
        if data.dim() != 3:
            data = data.squeeze()
        
        binary = (data > threshold).float().numpy()
        res = binary.shape[0]
        
        # Create figure with 3 rows (X, Y, Z slices)
        fig, axes = plt.subplots(3, num_slices, figsize=(num_slices * 2, 6))
        
        slice_indices = np.linspace(0, res - 1, num_slices, dtype=int)
        
        for i, idx in enumerate(slice_indices):
            # X slices (YZ plane)
            axes[0, i].imshow(binary[idx, :, :], cmap='gray', vmin=0, vmax=1)
            axes[0, i].set_title(f'X={idx}')
            axes[0, i].axis('off')
            
            # Y slices (XZ plane)
            axes[1, i].imshow(binary[:, idx, :], cmap='gray', vmin=0, vmax=1)
            axes[1, i].set_title(f'Y={idx}')
            axes[1, i].axis('off')
            
            # Z slices (XY plane)
            axes[2, i].imshow(binary[:, :, idx], cmap='gray', vmin=0, vmax=1)
            axes[2, i].set_title(f'Z={idx}')
            axes[2, i].axis('off')
        
        plt.suptitle(f'Sparse Structure Slices (occupancy: {binary.sum():.0f}/{res**3})')
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[Visualizer] Slices saved to: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return None
    
    def visualize_projections(self,
                              data: torch.Tensor,
                              save_path: str = None,
                              threshold: float = 0.5,
                              show: bool = False) -> Optional[np.ndarray]:
        """
        Visualize max projections along each axis.
        
        Args:
            data: (R, R, R) occupancy tensor
            save_path: path to save image
            threshold: occupancy threshold
            show: whether to display
            
        Returns:
            visualization image as numpy array if save_path is None
        """
        import matplotlib.pyplot as plt
        
        # Ensure tensor is on CPU
        data = self.ensure_cpu_tensor(data)
        
        if data.dim() != 3:
            data = data.squeeze()
        
        binary = (data > threshold).float().numpy()
        
        # Max projections
        proj_x = binary.max(axis=0)
        proj_y = binary.max(axis=1)
        proj_z = binary.max(axis=2)
        
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        axes[0].imshow(proj_x, cmap='viridis', vmin=0, vmax=1)
        axes[0].set_title('X Projection (YZ)')
        axes[0].axis('off')
        
        axes[1].imshow(proj_y, cmap='viridis', vmin=0, vmax=1)
        axes[1].set_title('Y Projection (XZ)')
        axes[1].axis('off')
        
        axes[2].imshow(proj_z, cmap='viridis', vmin=0, vmax=1)
        axes[2].set_title('Z Projection (XY)')
        axes[2].axis('off')
        
        plt.suptitle(f'Max Projections (occupancy: {binary.sum():.0f})')
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[Visualizer] Projections saved to: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return None
    
    def visualize_3d(self,
                     data: torch.Tensor,
                     save_path: str = None,
                     threshold: float = 0.5,
                     downsample: int = 1,
                     color: str = 'viridis',
                     value_colored: bool = False,
                     show: bool = False,
                     use_scatter: bool = True):
        """
        Create 3D voxel visualization using matplotlib.

        Two rendering modes are available via `use_scatter`:

        * use_scatter=True  (default, fast): renders each occupied voxel as a
          single scatter point via ax.scatter().  O(N) complexity — typically
          10-50× faster than the voxel mode, suitable for saving many frames.

        * use_scatter=False (slow, detailed): renders solid voxel cubes via
          ax.voxels().  Each occupied voxel produces 6 quad patches + edge
          lines, giving a true block appearance at the cost of O(N×6) patch
          overhead that becomes very slow for dense 64³ grids.

        Args:
            data: (R, R, R) occupancy tensor
            save_path: path to save image
            threshold: occupancy threshold
            downsample: downsample factor for large grids
            color: colormap name (used when value_colored=True)
            value_colored: whether to color voxels by their density values (default: False)
            show: whether to display
            use_scatter: if True (default) use fast scatter; if False use slow ax.voxels
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – registers 3d projection
        import matplotlib.cm as cm
        from matplotlib.colors import Normalize

        # Ensure tensor is on CPU
        data = self.ensure_cpu_tensor(data)

        if data.dim() != 3:
            data = data.squeeze()

        binary = (data > threshold).float().numpy()
        data_np = data.numpy()

        # Downsample if needed
        if downsample > 1:
            try:
                from scipy.ndimage import zoom
                binary = zoom(binary, 1/downsample, order=0)
                data_np = zoom(data_np, 1/downsample, order=1)
            except ImportError:
                print("[Visualizer] scipy not installed for downsampling. Using original resolution.")

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        if use_scatter:
            # ── Fast path: scatter plot ──────────────────────────────────────
            xs, ys, zs = np.where(binary > 0)
            n_occ = len(xs)

            if n_occ == 0:
                print("[Visualizer] No occupied voxels above threshold.")
            elif value_colored:
                vals = data_np[binary > 0]
                cmap = cm.get_cmap(color)
                norm = Normalize(vmin=0, vmax=1)
                rgba = cmap(norm(vals))          # (N, 4)
                ax.scatter(xs, ys, zs, c=rgba, s=4, depthshade=True, linewidths=0)
                mappable = cm.ScalarMappable(norm=norm, cmap=color)
                mappable.set_array([])
                plt.colorbar(mappable, ax=ax, shrink=0.5, label='Density')
            else:
                ax.scatter(xs, ys, zs, c='#3399cc', s=4, alpha=0.7, depthshade=True, linewidths=0)
        else:
            # ── Slow path: solid voxel cubes (ax.voxels) ────────────────────
            colors = np.zeros(binary.shape + (4,))

            if value_colored:
                cmap = cm.get_cmap(color)
                norm = Normalize(vmin=0, vmax=1)
                rgba = cmap(norm(data_np))       # (R, R, R, 4)
                colors = rgba
                colors[binary == 0] = [0, 0, 0, 0]
            else:
                colors[binary > 0] = [0.2, 0.6, 0.8, 0.8]  # Blue-ish with transparency

            ax.voxels(binary, facecolors=colors, edgecolor='k', linewidth=0.1)

            if value_colored:
                mappable = cm.ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap=color)
                mappable.set_array([])
                plt.colorbar(mappable, ax=ax, shrink=0.5, label='Density')

        ax.set_box_aspect(binary.shape)
        ax.set_xlim([0, binary.shape[0]])
        ax.set_ylim([0, binary.shape[1]])
        ax.set_zlim([0, binary.shape[2]])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Sparse Structure 3D (occupancy: {(data > threshold).sum().item():.0f})')

        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[Visualizer] 3D visualization saved to: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()
    
    def visualize_interactive(self,
                              data: torch.Tensor,
                              save_path: str = None,
                              threshold: float = 0.2,
                              title: str = "Sparse Structure"):
        """
        Create interactive 3D visualization using plotly.
        
        Args:
            data: (R, R, R) occupancy tensor
            save_path: path to save HTML file
            threshold: occupancy threshold
            title: plot title
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            print("[Visualizer] plotly not installed. Install with: pip install plotly")
            return
        
        # Ensure tensor is on CPU
        data = self.ensure_cpu_tensor(data)
        
        if data.dim() != 3:
            data = data.squeeze()
        
        binary = data > threshold
        coords = torch.argwhere(binary).float()
        
        if len(coords) == 0:
            print("[Visualizer] No occupied voxels found")
            return
        
        # Get continuous values for coloring
        values = data[binary].numpy()
        
        fig = go.Figure(data=[go.Scatter3d(
            x=coords[:, 0].numpy(),
            y=coords[:, 1].numpy(),
            z=coords[:, 2].numpy(),
            mode='markers',
            marker=dict(
                size=4,
                color=values,
                colorscale='Viridis',
                opacity=0.8,
                cmin=0,  # Fix color scale minimum to 0
                cmax=1,  # Fix color scale maximum to 1
                colorbar=dict(title='Occupancy')
            )
        )])
        
        fig.update_layout(
            title=f'{title} (occupancy: {len(coords)}/{data.numel()})',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='data'
            ),
            margin=dict(l=0, r=0, t=40, b=0)
        )
        
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            fig.write_html(save_path)
            print(f"[Visualizer] Interactive visualization saved to: {save_path}")
        else:
            fig.show()
    
    def compare(self,
                data1: torch.Tensor,
                data2: torch.Tensor,
                save_path: str = None,
                labels: Tuple[str, str] = ('Before', 'After'),
                threshold: float = 0.5,
                show: bool = False):
        """
        Create side-by-side comparison visualization.
        
        Args:
            data1: first sparse structure
            data2: second sparse structure
            save_path: path to save image
            labels: labels for the two structures
            threshold: occupancy threshold
            show: whether to display
        """
        import matplotlib.pyplot as plt
        
        # Ensure tensor is on CPU
        data1 = self.ensure_cpu_tensor(data1)
        data2 = self.ensure_cpu_tensor(data2)
        
        if data1.dim() != 3:
            data1 = data1.squeeze()
        if data2.dim() != 3:
            data2 = data2.squeeze()
        
        binary1 = (data1 > threshold).float().numpy()
        binary2 = (data2 > threshold).float().numpy()
        
        res = binary1.shape[0]
        mid = res // 2
        
        # Create comparison figure
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        # First row: data1
        axes[0, 0].imshow(binary1[mid, :, :], cmap='gray', vmin=0, vmax=1)
        axes[0, 0].set_title(f'{labels[0]} - X={mid}')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(binary1[:, mid, :], cmap='gray', vmin=0, vmax=1)
        axes[0, 1].set_title(f'{labels[0]} - Y={mid}')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(binary1[:, :, mid], cmap='gray', vmin=0, vmax=1)
        axes[0, 2].set_title(f'{labels[0]} - Z={mid}')
        axes[0, 2].axis('off')
        
        # Max projection
        axes[0, 3].imshow(binary1.max(axis=2), cmap='viridis', vmin=0, vmax=1)
        axes[0, 3].set_title(f'{labels[0]} - Projection')
        axes[0, 3].axis('off')
        
        # Second row: data2
        axes[1, 0].imshow(binary2[mid, :, :], cmap='gray', vmin=0, vmax=1)
        axes[1, 0].set_title(f'{labels[1]} - X={mid}')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(binary2[:, mid, :], cmap='gray', vmin=0, vmax=1)
        axes[1, 1].set_title(f'{labels[1]} - Y={mid}')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(binary2[:, :, mid], cmap='gray', vmin=0, vmax=1)
        axes[1, 2].set_title(f'{labels[1]} - Z={mid}')
        axes[1, 2].axis('off')
        
        axes[1, 3].imshow(binary2.max(axis=2), cmap='viridis', vmin=0, vmax=1)
        axes[1, 3].set_title(f'{labels[1]} - Projection')
        axes[1, 3].axis('off')
        
        occ1 = binary1.sum()
        occ2 = binary2.sum()
        plt.suptitle(f'Comparison: {labels[0]} ({occ1:.0f} voxels) vs {labels[1]} ({occ2:.0f} voxels)')
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[Visualizer] Comparison saved to: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()

    def visualize_certainty(self,
                            certainty: torch.Tensor,
                            occupancy: torch.Tensor = None,
                            save_path: str = None,
                            threshold: float = 0.0,
                            show: bool = False):
        """
        Visualize certainty grid as color-mapped max-projections and 3-D scatter.
        """
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        from matplotlib.colors import Normalize
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        certainty = self.ensure_cpu_tensor(certainty).squeeze()
        cert_np = certainty.numpy().astype(np.float32)
        vmax = float(cert_np.max()) if cert_np.max() > 0 else 1.0

        # --- max-projection panel (3 axes) ---
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for ax, proj, title in zip(
            axes,
            [cert_np.max(axis=0), cert_np.max(axis=1), cert_np.max(axis=2)],
            ['X-proj (YZ)', 'Y-proj (XZ)', 'Z-proj (XY)']
        ):
            im = ax.imshow(proj, cmap='viridis', vmin=0, vmax=vmax)
            ax.set_title(title)
            ax.axis('off')
            plt.colorbar(im, ax=ax, shrink=0.8, label='certainty')
        plt.suptitle(f'Certainty projections  (max={vmax:.4f})')
        plt.tight_layout()
        if save_path:
            p = save_path.replace('.png', '_projections.png')
            os.makedirs(os.path.dirname(p) if os.path.dirname(p) else '.', exist_ok=True)
            plt.savefig(p, dpi=150, bbox_inches='tight')
            print(f"[Visualizer] Certainty projections saved to: {p}")
        if show:
            plt.show()
        else:
            plt.close()

        # # --- 3-D scatter coloured by certainty ---
        # if occupancy is not None:
        #     occ_np = self.ensure_cpu_tensor(occupancy).squeeze().numpy()
        #     mask = (occ_np > 0.5) & (cert_np > threshold)
        # else:
        #     mask = cert_np > threshold
        # xs, ys, zs = np.where(mask)
        # if len(xs) == 0:
        #     print("[Visualizer] No certainty voxels above threshold — skipping 3D.")
        #     return
        # vals = cert_np[mask]
        # cmap = cm.get_cmap('viridis')
        # norm = Normalize(vmin=0, vmax=vmax)
        # fig2 = plt.figure(figsize=(9, 9))
        # ax3  = fig2.add_subplot(111, projection='3d')
        # ax3.scatter(xs, ys, zs, c=cmap(norm(vals)), s=4, depthshade=True, linewidths=0)
        # mappable = cm.ScalarMappable(norm=norm, cmap='viridis')
        # mappable.set_array([])
        # plt.colorbar(mappable, ax=ax3, shrink=0.5, label='certainty')
        # ax3.set_box_aspect(cert_np.shape)
        # ax3.set_xlabel('X'); ax3.set_ylabel('Y'); ax3.set_zlabel('Z')
        # ax3.set_title(f'Certainty 3D  ({len(xs)} voxels, max={vmax:.4f})')
        # if save_path:
        #     p = save_path.replace('.png', '_3d.png')
        #     os.makedirs(os.path.dirname(p) if os.path.dirname(p) else '.', exist_ok=True)
        #     plt.savefig(p, dpi=150, bbox_inches='tight')
        #     print(f"[Visualizer] Certainty 3D saved to: {p}")
        # if show:
        #     plt.show()
        # else:
        #     plt.close()

    def visualize_blank_certainty(
        self,
        blank_uncertainty: torch.Tensor,
        occupancy: torch.Tensor = None,
        save_path: str = None,
        show: bool = False,
    ):
        """
        Visualise the blank-space uncertainty distribution for one object.

        ``blank_uncertainty`` is the tensor stored in
        ``SparseStructure.blank_space_uncertainty`` (shape ``(res, res, res)``):

        * 0 for **occupied** voxels (uncertainty not defined there).
        * ``1 - scene_certainty`` for **blank** voxels (range [0, 1]).

        The method produces a two-panel figure per axis (max-projection and
        mid-slice) with a *hot* colour map, plus a histogram of uncertainty
        values over the blank voxels.  Saves to:

        * ``<save_path>_projections.png`` — 2×3 spatial panel.
        * ``<save_path>_histogram.png``   — uncertainty histogram.
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        unc = self.ensure_cpu_tensor(blank_uncertainty).squeeze().float().numpy()

        # Determine blank-voxel mask: non-zero uncertainty (occupied have unc==0)
        blank_mask = unc > 0
        n_blank = int(blank_mask.sum())
        n_total = int(unc.size)

        # ---- 2×3 spatial panel (max-proj + mid-slice per axis) --------
        nx, ny, nz = unc.shape
        mid_x, mid_y, mid_z = nx // 2, ny // 2, nz // 2

        fig, axes = plt.subplots(2, 3, figsize=(14, 9))
        panels = [
            # row 0: max projections
            (axes[0, 0], unc.max(axis=0),        'Max-proj X (YZ)'),
            (axes[0, 1], unc.max(axis=1),        'Max-proj Y (XZ)'),
            (axes[0, 2], unc.max(axis=2),        'Max-proj Z (XY)'),
            # row 1: mid slices
            (axes[1, 0], unc[mid_x, :, :],       f'Mid-slice X={mid_x}'),
            (axes[1, 1], unc[:, mid_y, :],       f'Mid-slice Y={mid_y}'),
            (axes[1, 2], unc[:, :, mid_z],       f'Mid-slice Z={mid_z}'),
        ]
        for ax, img, title in panels:
            im = ax.imshow(img, cmap='hot', vmin=0.0, vmax=1.0)
            ax.set_title(title, fontsize=9)
            ax.axis('off')
            plt.colorbar(im, ax=ax, shrink=0.75, label='uncertainty')

        mean_unc = float(unc[blank_mask].mean()) if n_blank > 0 else 0.0
        fig.suptitle(
            f'Blank-space uncertainty  |  blank voxels: {n_blank}/{n_total} '
            f'({100*n_blank/max(n_total,1):.1f}%)  |  mean={mean_unc:.4f}',
            fontsize=10,
        )
        plt.tight_layout()

        if save_path:
            p = save_path.replace('.png', '_projections.png')
            os.makedirs(os.path.dirname(p) if os.path.dirname(p) else '.', exist_ok=True)
            plt.savefig(p, dpi=150, bbox_inches='tight')
            print(f'[Visualizer] Blank-certainty projections saved to: {p}')
        if show:
            plt.show()
        else:
            plt.close()

        # ---- Histogram of blank-voxel uncertainty values ---------------
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        if n_blank > 0:
            vals = unc[blank_mask].flatten()
            ax2.hist(vals, bins=50, range=(0.0, 1.0), color='tomato',
                     edgecolor='black', linewidth=0.4)
            ax2.axvline(mean_unc, color='navy', linestyle='--',
                        linewidth=1.4, label=f'mean={mean_unc:.4f}')
            ax2.legend(fontsize=9)
        ax2.set_xlabel('Uncertainty  (1 − scene certainty)', fontsize=10)
        ax2.set_ylabel('Voxel count', fontsize=10)
        ax2.set_title(
            f'Blank-space uncertainty histogram  '
            f'({n_blank} blank voxels)',
            fontsize=10,
        )
        ax2.set_xlim(0.0, 1.0)
        plt.tight_layout()

        if save_path:
            p = save_path.replace('.png', '_histogram.png')
            os.makedirs(os.path.dirname(p) if os.path.dirname(p) else '.', exist_ok=True)
            plt.savefig(p, dpi=150, bbox_inches='tight')
            print(f'[Visualizer] Blank-certainty histogram saved to: {p}')
        if show:
            plt.show()
        else:
            plt.close()


def visualize_sparse_structure(data: Union[str, torch.Tensor],
                               output_dir: str = None,
                               mode: str = 'all',
                               threshold: float = 0.2,
                               value_colored: bool = False,
                               use_scatter: bool = True,
                               show: bool = False,
                               certainty: torch.Tensor = None,
                               blank_uncertainty: torch.Tensor = None) -> None:
    """
    Convenience function to visualize a sparse structure.
    """
    vis = SparseStructureVisualizer()
    
    if isinstance(data, str):
        tensor = vis.load(data)
        name = Path(data).stem
    else:
        tensor = data
        name = 'sparse_structure'
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    if mode in ['slices', 'all']:
        save_path = os.path.join(output_dir, f'{name}_slices.png') if output_dir else None
        vis.visualize_slices(tensor, save_path, threshold=threshold, show=show)
    
    if mode in ['projections', 'all']:
        save_path = os.path.join(output_dir, f'{name}_projections.png') if output_dir else None
        vis.visualize_projections(tensor, save_path, threshold=threshold, show=show)
    
    if mode in ['3d', 'all']:
        save_path = os.path.join(output_dir, f'{name}_3d.png') if output_dir else None
        vis.visualize_3d(tensor, save_path, threshold=threshold, downsample=1, value_colored=value_colored, show=show, use_scatter=use_scatter) # XXX
    
    if mode in ['interactive', 'all']:
        save_path = os.path.join(output_dir, f'{name}_interactive.html') if output_dir else None
        vis.visualize_interactive(tensor, save_path, threshold=threshold, title=name)

    if certainty is not None and mode in ['certainty', 'all']:
        save_path = os.path.join(output_dir, f'{name}_certainty.png') if output_dir else None
        vis.visualize_certainty(certainty, occupancy=tensor, save_path=save_path, show=show)

    if blank_uncertainty is not None and mode in ['blank_certainty', 'all']:
        save_path = os.path.join(output_dir, f'{name}_blank_certainty.png') if output_dir else None
        vis.visualize_blank_certainty(blank_uncertainty, occupancy=tensor, save_path=save_path, show=show)


def compare_sparse_structures(path1: str,
                              path2: str,
                              output_path: str = None,
                              labels: Tuple[str, str] = ('Before', 'After'),
                              show_diff: bool = True,
                              show: bool = False) -> None:
    """
    Compare two sparse structures.
    
    Args:
        path1: path to first .pt file
        path2: path to second .pt file
        output_path: path to save comparison image
        labels: labels for the two structures
        show_diff: whether to also show difference visualization
        show: whether to display plots
    """
    vis = SparseStructureVisualizer()
    
    data1 = vis.load(path1)
    data2 = vis.load(path2)
    
    vis.compare(data1, data2, output_path, labels=labels, show=show)
    
    if show_diff and output_path:
        diff_path = output_path.replace('.png', '_diff.png')
        vis.visualize_difference(data1, data2, diff_path, show=show)


if __name__ == "__main__":
    """Command-line interface for sparse structure visualization."""
    parser = argparse.ArgumentParser(description='Visualize TRELLIS sparse structures')
    
    parser.add_argument('--input', '-i', type=str, help='Path to sparse structure .pt file')
    parser.add_argument('--output', '-o', type=str, default='./outputs/ss_visualization',
                        help='Output directory or file path')
    parser.add_argument('--mode', '-m', type=str, default='all',
                        choices=['slices', 'projections', '3d', 'interactive', 'all'],
                        help='Visualization mode')
    parser.add_argument('--threshold', '-t', type=float, default=0.5,
                        help='Occupancy threshold for binarization')
    parser.add_argument('--value-colored', action='store_true',
                        help='Color 3D voxels by density values (default: single color)')
    parser.add_argument('--show', action='store_true', help='Display plots interactively')
    
    # Comparison mode
    parser.add_argument('--compare', nargs=2, type=str, metavar=('BEFORE', 'AFTER'),
                        help='Compare two sparse structures')
    parser.add_argument('--labels', nargs=2, type=str, default=['Before', 'After'],
                        help='Labels for comparison')
    
    args = parser.parse_args()
    
    if args.compare:
        # Comparison mode
        output_path = args.output if args.output.endswith('.png') else os.path.join(args.output, 'comparison.png')
        compare_sparse_structures(
            args.compare[0], args.compare[1],
            output_path=output_path,
            labels=tuple(args.labels),
            show=args.show
        )
    elif args.input:
        # Single visualization mode
        visualize_sparse_structure(
            args.input,
            output_dir=args.output,
            mode=args.mode,
            threshold=args.threshold,
            value_colored=args.value_colored,
            show=args.show
        )
    else:
        # Demo mode with synthetic data
        print("No input provided. Running demo with synthetic data...")
        
        # Create synthetic sparse structure
        demo_ss = torch.zeros(64, 64, 64)
        
        # Add a sphere with gradient values based on distance
        center = torch.tensor([32, 32, 32])
        for x in range(64):
            for y in range(64):
                for z in range(64):
                    dist = ((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2) ** 0.5
                    if dist < 20:
                        # Gradient from 1.0 at center to 0.5 at edge
                        demo_ss[x, y, z] = 1.0 - (dist / 40)
        
        # Add a cube cutout
        demo_ss[25:40, 25:40, 25:40] = 0.0
        
        print(f"Demo sparse structure: {demo_ss.shape}, occupancy: {demo_ss.sum().item():.0f}")
        
        visualize_sparse_structure(
            demo_ss,
            output_dir=args.output,
            mode=args.mode,
            threshold=args.threshold,
            value_colored=args.value_colored,
            show=args.show
        )
        
        print(f"\nDemo visualizations saved to: {args.output}")


