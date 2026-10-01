"""
vis_grid.py – Truncated-uncertainty visibility grid for a reconstructed scene.

Each voxel in the 256^3 grid records a certainty value:

    certainty = clamp(visible_count / certainty_threshold, 0.0, 1.0)

With the default threshold of 3:
  - seen by ≥ 3 cameras  → certainty = 1.0
  - never seen            → certainty = 0.0
  - 1 or 2 cameras        → linearly interpolated

Usage
-----
# Build directly from a GeoSVR model path (renders depth maps if not cached):
    vis_grid = VisibilityGrid.from_model_path(
        model_path  = "/path/to/model",
        resolution  = 256,
    )
    vis_grid.visualize("/path/to/vis_output")

# Or pass pre-computed cameras / depths:
    vis_grid = VisibilityGrid(
        bbox_min=..., bbox_max=..., resolution=256,
        cameras=[...], depth_maps=[...],
    )
"""

import os
import sys
import numpy as np
import torch
from typing import List
from tqdm import tqdm

# ─── Resolve project / GeoSVR roots so `src.*` imports work ─────────────────
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..'))
_GEOSVR_ROOT = os.path.join(
    _PROJECT_ROOT, 'decomvoxel', 'representation', 'GeoSVR')
for _p in [_PROJECT_ROOT, _GEOSVR_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


class VisibilityGrid:
    """
    A 3D visibility grid that counts how many input cameras see each voxel
    and converts the count into a per-voxel certainty in [0, 1].

    Certainty formula (truncated linear / "tranc uncertainty"):

        certainty = clamp(visible_count / certainty_threshold, 0.0, 1.0)

    Parameters
    ----------
    bbox_min, bbox_max : torch.Tensor  shape (3,)
        World-space axis-aligned bounding box of the grid.
    resolution : int
        Number of voxels along the longest scene axis.
        Other axes are scaled with the same cubic voxel size.
    cameras : list
        List of Camera objects (GeoSVR ``src.cameras.Camera``).
    depth_maps : list[torch.Tensor]
        One (H, W) float32 z-depth tensor per camera (0 = no surface hit).
    certainty_threshold : int
        Number of cameras that must see a voxel for certainty == 1.
    device : str
        Torch device for grid tensors ('cuda' recommended).
    """

    # ────────────────────────────────────────────────────────── constructor ──

    def __init__(
        self,
        bbox_min: torch.Tensor,
        bbox_max: torch.Tensor,
        resolution: int,
        cameras: list,
        depth_maps: List[torch.Tensor],
        certainty_threshold: int = 3,
        device: str = "cuda",
    ):
        self.device = device
        self.bbox_min = bbox_min.to(device)
        self.bbox_max = bbox_max.to(device)
        self.resolution = resolution
        self.cameras = cameras
        self.depth_maps = depth_maps
        self.certainty_threshold = certainty_threshold

        # Cubic voxel size: longest edge / resolution
        edge_lengths = self.bbox_max - self.bbox_min
        scene_center = 0.5 * (self.bbox_min + self.bbox_max)
        self.voxel_size = edge_lengths.max() / resolution
        ns = torch.ceil(edge_lengths / self.voxel_size).long().clamp(min=1)
        aligned_lengths = ns.to(edge_lengths.dtype) * self.voxel_size
        self.bbox_min = scene_center - 0.5 * aligned_lengths
        self.bbox_max = scene_center + 0.5 * aligned_lengths
        self.nx, self.ny, self.nz = ns[0].item(), ns[1].item(), ns[2].item()

        # Will be populated by _build_grid()
        self.vis_count_grid = torch.zeros(
            (self.nx, self.ny, self.nz),
            dtype=torch.float32, device=device)
        self.certainty_grid = torch.zeros(
            (self.nx, self.ny, self.nz),
            dtype=torch.float32, device=device)

        self._build_grid()

    # ─────────────────────────────────────────────────────── grid building ──

    def _build_grid(self):
        """
        Iterate over all cameras and accumulate per-voxel visibility counts,
        then compute certainty = clamp(count / threshold, 0, 1).
        """
        nx, ny, nz = self.nx, self.ny, self.nz
        N = nx * ny * nz
        print(f"[VisibilityGrid] Building {nx}\u00d7{ny}\u00d7{nz} visibility grid "
              f"({N:,} voxels, {len(self.cameras)} cameras, "
              f"voxel_size={self.voxel_size.item():.4f}) ...")

        # Pre-compute all grid-centre world coordinates once
        X, Y, Z = torch.meshgrid(
            torch.arange(nx, device=self.device),
            torch.arange(ny, device=self.device),
            torch.arange(nz, device=self.device),
            indexing='ij')
        grid_centers = torch.stack([
            self.bbox_min[0] + (X + 0.5) * self.voxel_size,
            self.bbox_min[1] + (Y + 0.5) * self.voxel_size,
            self.bbox_min[2] + (Z + 0.5) * self.voxel_size,
        ], dim=-1)
        grid_flat = grid_centers.reshape(-1, 3)
        N = grid_flat.shape[0]

        vis_count_flat = torch.zeros(N, dtype=torch.float32, device=self.device)

        # Process cameras; inner loop over grid-point batches for memory safety
        BATCH = 500_000
        n_batches = (N + BATCH - 1) // BATCH

        for cam, depth_map in tqdm(
                zip(self.cameras, self.depth_maps),
                total=len(self.cameras),
                desc="[VisGrid] counting visibility"):

            depth_map = depth_map.to(self.device)
            cam_vis = torch.zeros(N, dtype=torch.bool, device=self.device)

            for b in range(n_batches):
                s = b * BATCH
                e = min(s + BATCH, N)
                cam_vis[s:e] = self._check_visibility_for_camera(
                    cam, depth_map, grid_flat[s:e])

            vis_count_flat += cam_vis.float()

        self.vis_count_grid = vis_count_flat.reshape(nx, ny, nz)
        self.certainty_grid = (
            self.vis_count_grid / self.certainty_threshold
        ).clamp(0.0, 1.0)

        ever_visible = (self.vis_count_grid > 0).sum().item()
        fully_certain = (self.certainty_grid >= 1.0).sum().item()
        print(f"[VisibilityGrid] Done — "
              f"{ever_visible:,}/{N:,} ({100*ever_visible/N:.1f}%) ever-visible, "
              f"{fully_certain:,}/{N:,} ({100*fully_certain/N:.1f}%) fully-certain.")

    # ───────────────────────────────────────────────────────── device move ──

    def to(self, device) -> "VisibilityGrid":
        """Return a shallow clone of this grid with tensors moved to *device*.

        The returned instance shares no tensor storage with the original
        (so worker threads on different GPUs can read/modify it
        independently), but it skips the expensive ``_build_grid()``
        pass. Cameras / depth maps (heavy Python objects) are NOT deep
        copied — they are kept as references.
        """
        device = str(device) if not isinstance(device, str) else device
        if str(self.device) == device:
            return self
        new = self.__class__.__new__(self.__class__)
        # ── carry over scalars / lists / references unchanged ──
        new.device = device
        new.resolution = self.resolution
        new.cameras = self.cameras
        new.depth_maps = self.depth_maps
        new.certainty_threshold = self.certainty_threshold
        new.nx, new.ny, new.nz = self.nx, self.ny, self.nz
        # ── move tensors ──
        new.bbox_min = self.bbox_min.to(device)
        new.bbox_max = self.bbox_max.to(device)
        new.voxel_size = (
            self.voxel_size.to(device) if torch.is_tensor(self.voxel_size) else self.voxel_size
        )
        new.vis_count_grid = self.vis_count_grid.to(device)
        new.certainty_grid = self.certainty_grid.to(device)
        return new

    # ─────────────────────────────────────────────── per-camera visibility ──

    @staticmethod
    def _check_visibility_for_camera(
        camera,
        depth_map: torch.Tensor,
        pts: torch.Tensor,
        rel_depth_tol: float = 0.05,
    ) -> torch.Tensor:
        """
        Check whether 3D world-space points are visible from a single camera.

        A point is considered visible if ALL of the following hold:

        1. Its projection falls inside the image  (normalised coords in [-1, 1]).
        2. Camera-space z-depth > 0  (point in front of the camera).
        3. The rendered depth at that pixel is valid (> 0).
        4. Point depth ≤ rendered_depth × (1 + rel_depth_tol)
           (the point is on or in front of the reconstructed surface).

        Parameters
        ----------
        camera : Camera
            GeoSVR Camera object; its ``.project()`` method is used.
        depth_map : Tensor  (H, W)
            Expected z-depth from the rendered reconstruction; 0 = no hit.
        pts : Tensor  (N, 3)
            World-space query points.
        rel_depth_tol : float
            Relative depth tolerance (default 5 %).

        Returns
        -------
        visible : BoolTensor  (N,) on the same device as ``depth_map``.
        """
        device = depth_map.device
        pts = pts.to(device)

        # Project points → normalised image coords + camera-space z-depth
        cam_uv, cam_depth = camera.project(pts, return_depth=True)
        # cam_uv   : (N, 2)  normalised coords in [-1, 1]
        # cam_depth: (N, 1)  z-component in camera space
        cam_depth = cam_depth.squeeze(-1)        # (N,)

        H, W = depth_map.shape

        # ── 1) In-frustum check ─────────────────────────────────────────────
        in_frustum = (
            (cam_uv[:, 0] >= -1.0) & (cam_uv[:, 0] <= 1.0) &
            (cam_uv[:, 1] >= -1.0) & (cam_uv[:, 1] <= 1.0) &
            (cam_depth > 0.0)
        )

        # ── 2) Map normalised coords to pixel indices ────────────────────────
        # cam_uv == -1  →  pixel 0;  cam_uv == +1  →  pixel W (or H)
        px = ((cam_uv[:, 0] + 1.0) * 0.5 * W).long().clamp(0, W - 1)
        py = ((cam_uv[:, 1] + 1.0) * 0.5 * H).long().clamp(0, H - 1)

        # ── 3) Depth at projected pixel ─────────────────────────────────────
        rendered_depth = depth_map[py, px]          # (N,)
        valid_depth    = rendered_depth > 1e-6

        # ── 4) Occlusion test ────────────────────────────────────────────────
        not_occluded = cam_depth <= rendered_depth * (1.0 + rel_depth_tol)

        return in_frustum & valid_depth & not_occluded

    # ──────────────────────────────────────── grid-index / sampling helpers ──

    def _world_to_grid_indices(self, points: torch.Tensor) -> torch.Tensor:
        """
        Convert world-space coordinates to integer voxel indices (clamped to
        valid range [0, resolution − 1]).

        Parameters
        ----------
        points : Tensor  (..., 3)

        Returns
        -------
        indices : LongTensor  (..., 3)
        """
        indices = ((points - self.bbox_min) / self.voxel_size).long()
        indices[..., 0] = indices[..., 0].clamp(0, self.nx - 1)
        indices[..., 1] = indices[..., 1].clamp(0, self.ny - 1)
        indices[..., 2] = indices[..., 2].clamp(0, self.nz - 1)
        return indices

    def _sample_certainty_at_points(
        self,
        points: torch.Tensor,
        max_batch: int = 100_000,
    ) -> torch.Tensor:
        """
        Nearest-neighbour certainty sampling at arbitrary world-space points.

        Parameters
        ----------
        points : Tensor  (..., 3)
        max_batch : int
            Maximum number of points processed per GPU call.

        Returns
        -------
        certainty : Tensor  (...,)  values in [0, 1].
        """
        orig_shape = points.shape[:-1]
        pts_flat   = points.reshape(-1, 3)
        N          = pts_flat.shape[0]

        if N <= max_batch:
            idx = self._world_to_grid_indices(pts_flat)
            return self.certainty_grid[
                idx[:, 0], idx[:, 1], idx[:, 2]
            ].reshape(orig_shape)

        chunks = []
        for s in range(0, N, max_batch):
            e   = min(s + max_batch, N)
            idx = self._world_to_grid_indices(pts_flat[s:e])
            chunks.append(
                self.certainty_grid[idx[:, 0], idx[:, 1], idx[:, 2]])
        return torch.cat(chunks).reshape(orig_shape)

    def update_certainty_from_points(self, points: torch.Tensor) -> int:
        pts = points.to(self.device).reshape(-1, 3)
        if pts.shape[0] == 0:
            return 0

        x = (pts[:, 0] - self.bbox_min[0]) / self.voxel_size - 0.5
        y = (pts[:, 1] - self.bbox_min[1]) / self.voxel_size - 0.5
        z = (pts[:, 2] - self.bbox_min[2]) / self.voxel_size - 0.5

        valid = (
            (x >= -0.5) & (x <= self.nx - 0.5)
            & (y >= -0.5) & (y <= self.ny - 0.5)
            & (z >= -0.5) & (z <= self.nz - 0.5)
        )
        if valid.sum().item() == 0:
            return 0

        x = x[valid]
        y = y[valid]
        z = z[valid]

        x0 = x.floor().long().clamp(0, self.nx - 2)
        y0 = y.floor().long().clamp(0, self.ny - 2)
        z0 = z.floor().long().clamp(0, self.nz - 2)
        x1 = x0 + 1
        y1 = y0 + 1
        z1 = z0 + 1

        wx = (x - x0.float()).clamp(0.0, 1.0)
        wy = (y - y0.float()).clamp(0.0, 1.0)
        wz = (z - z0.float()).clamp(0.0, 1.0)

        ix = torch.stack([x0, x0, x0, x0, x1, x1, x1, x1], dim=1)
        iy = torch.stack([y0, y0, y1, y1, y0, y0, y1, y1], dim=1)
        iz = torch.stack([z0, z1, z0, z1, z0, z1, z0, z1], dim=1)

        w = torch.stack([
            (1 - wx) * (1 - wy) * (1 - wz),
            (1 - wx) * (1 - wy) * wz,
            (1 - wx) * wy * (1 - wz),
            (1 - wx) * wy * wz,
            wx * (1 - wy) * (1 - wz),
            wx * (1 - wy) * wz,
            wx * wy * (1 - wz),
            wx * wy * wz,
        ], dim=1)

        active = w > 0
        lin = (
            ix[active] * (self.ny * self.nz)
            + iy[active] * self.nz
            + iz[active]
        ).unique()

        cert_flat = self.certainty_grid.reshape(-1)
        vis_flat = self.vis_count_grid.reshape(-1)
        cert_flat[lin] = 1.0
        vis_flat[lin] = torch.maximum(
            vis_flat[lin],
            torch.full_like(vis_flat[lin], float(self.certainty_threshold)),
        )

        return int(lin.numel())

    # ─────────────────────────────────────────────────── public interfaces ──

    def check_valid_camera_center(self, cam_centers: torch.Tensor) -> torch.Tensor:
        """
        Return whether camera centres are in well-observed (certain) regions.

        Parameters
        ----------
        cam_centers : Tensor  (N, 3)  world-space positions.

        Returns
        -------
        valid_mask : BoolTensor  (N,);  True when certainty ≥ 0.5.
        """
        certainty = self._sample_certainty_at_points(cam_centers)
        return certainty >= 0.5

    def vis_invisible_pnts(self, save_path: str):
        import trimesh
        nx, ny, nz = self.nx, self.ny, self.nz
        X, Y, Z = torch.meshgrid(
            torch.arange(nx, device=self.device),
            torch.arange(ny, device=self.device),
            torch.arange(nz, device=self.device),
            indexing='ij')
        grid_centers = torch.stack([
            self.bbox_min[0] + (X + 0.5) * self.voxel_size,
            self.bbox_min[1] + (Y + 0.5) * self.voxel_size,
            self.bbox_min[2] + (Z + 0.5) * self.voxel_size,
        ], dim=-1)
        points = grid_centers[self.certainty_grid < 1.0].detach().cpu().numpy()
        trimesh.PointCloud(points).export(save_path)
        print(f"[VisibilityGrid] Invisible points saved \u2192 {save_path}")

    # ──────────────────────────────────────────────────────── persistence ──

    def save(self, model_path: str) -> str:
        """
        Save the visibility grid to ``<model_path>/vis_grid/vis_grid.pt``.

        The saved file is a plain ``torch.save`` dict containing:
        - ``bbox_min``, ``bbox_max``  – world-space bounding box (CPU tensors)
        - ``resolution``              – int, voxels on longest axis
        - ``voxel_size``              – scalar voxel edge length
        - ``nx``, ``ny``, ``nz``      – per-axis grid counts
        - ``certainty_threshold``     – int
        - ``vis_count_grid``          – (nx, ny, nz) float32 CPU tensor
        - ``certainty_grid``          – (nx, ny, nz) float32 CPU tensor

        Parameters
        ----------
        model_path : str
            Root directory of the GeoSVR run; the file is written to
            ``<model_path>/vis_grid/vis_grid.pt``.

        Returns
        -------
        save_path : str
            Absolute path of the saved file.
        """
        save_dir = os.path.join(model_path, "vis_grid")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "vis_grid.pt")
        torch.save({
            'bbox_min':            self.bbox_min.cpu(),
            'bbox_max':            self.bbox_max.cpu(),
            'resolution':          self.resolution,
            'voxel_size':          self.voxel_size.cpu(),
            'nx':                  self.nx,
            'ny':                  self.ny,
            'nz':                  self.nz,
            'certainty_threshold': self.certainty_threshold,
            'vis_count_grid':      self.vis_count_grid.cpu(),
            'certainty_grid':      self.certainty_grid.cpu(),
        }, save_path)
        print(f"[VisibilityGrid] Saved → {save_path}")
        return save_path

    @classmethod
    def load(cls, model_path: str, device: str = "cuda") -> "VisibilityGrid":
        """
        Load a previously saved ``VisibilityGrid`` from
        ``<model_path>/vis_grid/vis_grid.pt``.

        Bypasses ``__init__`` (no camera / depth-map processing) and
        restores all grid attributes directly from the saved dict.

        Parameters
        ----------
        model_path : str
            Root directory of the GeoSVR run.
        device : str
            Torch device to move the grid tensors to (default ``'cuda'``).

        Returns
        -------
        VisibilityGrid
        """
        save_path = os.path.join(model_path, "vis_grid", "vis_grid.pt")
        d = torch.load(save_path, map_location="cpu", weights_only=True)

        obj = object.__new__(cls)
        obj.device             = device
        obj.resolution         = int(d["resolution"])
        obj.certainty_threshold = int(d["certainty_threshold"])
        obj.nx                 = int(d["nx"])
        obj.ny                 = int(d["ny"])
        obj.nz                 = int(d["nz"])
        obj.voxel_size         = d["voxel_size"].to(device)
        obj.bbox_min           = d["bbox_min"].to(device)
        obj.bbox_max           = d["bbox_max"].to(device)
        obj.vis_count_grid     = d["vis_count_grid"].to(device)
        obj.certainty_grid     = d["certainty_grid"].to(device)
        obj.cameras            = []
        obj.depth_maps         = []
        print(f"[VisibilityGrid] Loaded from {save_path}")
        return obj

    # ───────────────────────────────────────────────────── visualisation ────

    def visualize(self, save_dir: str):
        """
        Write a PNG summarising the certainty grid to *save_dir*.

        The figure is laid out as 2 rows × 3 columns:

        +-------------------+-------------------+-------------------+
        | Max-proj along X  | Max-proj along Y  | Max-proj along Z  |
        | (YZ  plane)       | (XZ  plane)       | (XY  plane)       |
        +-------------------+-------------------+-------------------+
        | Mid-slice at X/2  | Mid-slice at Y/2  | Mid-slice at Z/2  |
        | (YZ  plane)       | (XZ  plane)       | (XY  plane)       |
        +-------------------+-------------------+-------------------+

        A *plasma* colour map is used; 0 = black (unseen), 1 = yellow (certain).
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        os.makedirs(save_dir, exist_ok=True)

        cert = self.certainty_grid.cpu().float().numpy()
        nx, ny, nz = self.nx, self.ny, self.nz
        mid_x, mid_y, mid_z = nx // 2, ny // 2, nz // 2

        # Max-projections (collapsed along each axis)
        proj_x = cert.max(axis=0)   # YZ plane
        proj_y = cert.max(axis=1)   # XZ plane
        proj_z = cert.max(axis=2)   # XY plane

        # Mid-plane slices
        slice_x = cert[mid_x, :, :]   # YZ slice at X = mid
        slice_y = cert[:, mid_y, :]   # XZ slice at Y = mid
        slice_z = cert[:, :, mid_z]   # XY slice at Z = mid

        rows = [
            [proj_x,  proj_y,  proj_z],
            [slice_x, slice_y, slice_z],
        ]
        row_labels = ["Max-projection", "Mid-plane slice"]
        col_labels = [
            "Along X  (YZ plane)",
            "Along Y  (XZ plane)",
            "Along Z  (XY plane)",
        ]

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        for r_i, (imgs, r_lbl) in enumerate(zip(rows, row_labels)):
            for c_i, (img, c_lbl) in enumerate(zip(imgs, col_labels)):
                ax = axes[r_i, c_i]
                im = ax.imshow(
                    img.T, origin='lower',
                    cmap='plasma', vmin=0.0, vmax=1.0, aspect='equal')
                ax.set_title(f"{r_lbl}\n{c_lbl}", fontsize=9)
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        fig.suptitle(
            f"Certainty Grid  ({nx}\u00d7{ny}\u00d7{nz},  "
            f"voxel_size={self.voxel_size.item():.4f},  threshold={self.certainty_threshold})",
            fontsize=12)
        plt.tight_layout()
        out_path = os.path.join(save_dir, "certainty_vis.png")
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"[VisibilityGrid] Visualisation saved → {out_path}")

    # ─────────────────────────────────────────── factory: load from disk ────

    @classmethod
    def from_model_path(
        cls,
        model_path: str,
        resolution: int = 256,
        certainty_threshold: int = 3,
        checkpoint_name: str = "iter020000_model.pt",
        device: str = "cuda",
    ) -> "VisibilityGrid":
        """
        Build a VisibilityGrid by loading a trained GeoSVR scene.

        Steps
        -----
        1. Parse ``model_path/config.yaml`` to obtain the dataset source path.
        2. Load **SparseVoxelModel** from the checkpoint using
           ``SVInOut.load()`` (the official io.py interface).
          3. Derive the scene bounding box from loaded voxels
              (``vox_center`` + ``vox_size``).
        4. Load training cameras via ``DataPack``.
        5. Load cached raw depth maps from ``model_path/depths_npy/``
           (float32 .npy files).  If any are missing the model is used to
           render and cache them (mirrors the "Final render pass" in train.py).
        6. Instantiate and return the ``VisibilityGrid``.

        Parameters
        ----------
        model_path : str
            Root directory of the GeoSVR training run (contains config.yaml
            and checkpoints/).
        resolution : int
            Grid resolution on the longest axis (default 256).
        certainty_threshold : int
            Truncation threshold for the certainty formula (default 3).
        checkpoint_name : str
            Filename inside ``model_path/checkpoints/``.
        device : str
            Target device for grid tensors.

        Returns
        -------
        VisibilityGrid
        """
        import yaml
        from yacs.config import CfgNode
        from src.sparse_voxel_model import SparseVoxelModel
        from src.dataloader.data_pack import DataPack

        # ── 1. Read training config to find source_path ─────────────────────
        cfg_yaml_path = os.path.join(model_path, "config.yaml")
        if not os.path.isfile(cfg_yaml_path):
            raise FileNotFoundError(
                f"Training config not found: {cfg_yaml_path}")
        with open(cfg_yaml_path, 'r') as f:
            train_cfg = yaml.safe_load(f)

        source_path_raw = train_cfg['data']['source_path']
        source_path = (source_path_raw
                       if os.path.isabs(source_path_raw)
                       else os.path.join(_PROJECT_ROOT, source_path_raw))

        # ── 2. Load voxel model via SVInOut.load() ──────────────────────────
        checkpoint_path = os.path.join(
            model_path, "checkpoints", checkpoint_name)
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}")

        cfg_model = CfgNode()
        cfg_model.model_path       = model_path
        cfg_model.vox_geo_mode     = "triinterp1"
        cfg_model.density_mode     = "exp_linear_11"
        cfg_model.sh_degree        = int(train_cfg.get('model', {}).get('sh_degree', 3))
        cfg_model.ss               = float(train_cfg.get('model', {}).get('ss', 1.5))
        cfg_model.outside_level    = int(train_cfg.get('model', {}).get('outside_level', 5))
        cfg_model.white_background = bool(train_cfg.get('model', {}).get('white_background', False))
        cfg_model.black_background = bool(train_cfg.get('model', {}).get('black_background', False))

        print(f"[VisibilityGrid] Loading model from {checkpoint_path} …")
        voxel_model = SparseVoxelModel(cfg_model)
        voxel_model.load(checkpoint_path)   # uses SVInOut.load() interface

        # ── 3. Derive scene bounding box from voxel centers/sizes ───────────
        vox_center = voxel_model.vox_center.float()
        vox_half = 0.5 * voxel_model.vox_size.float()
        bbox_min = (vox_center - vox_half).amin(dim=0).cpu()
        bbox_max = (vox_center + vox_half).amax(dim=0).cpu()
        scene_center = 0.5 * (bbox_min + bbox_max)
        scene_extent = (bbox_max - bbox_min).max().item()
        print(f"[VisibilityGrid] Scene bbox from voxels: min={bbox_min.tolist()}, "
              f"max={bbox_max.tolist()}, center={scene_center.tolist()}, "
              f"max_extent={scene_extent:.4f}")

        # ── 4. Load cameras via DataPack ────────────────────────────────────
        cfg_data = CfgNode()
        cfg_data.source_path   = source_path
        cfg_data.images        = train_cfg['data'].get('images', 'images')
        cfg_data.res_downscale = float(train_cfg['data'].get('res_downscale', 0.0))
        cfg_data.res_width     = int(train_cfg['data'].get('res_width', 0))
        cfg_data.extension     = train_cfg['data'].get('extension', '.png')
        cfg_data.depth_paths   = train_cfg['data'].get('depth_paths', '')
        cfg_data.depth_scale   = float(train_cfg['data'].get('depth_scale', 1.0))
        cfg_data.data_device   = 'cpu'
        cfg_data.eval          = bool(train_cfg['data'].get('eval', True))
        cfg_data.test_every    = int(train_cfg['data'].get('test_every', 8))
        cfg_data.n_sparse      = int(train_cfg['data'].get('n_sparse', -1))
        cfg_data.blend_mask    = bool(train_cfg['data'].get('blend_mask', True))
        cfg_data.ncc_scale     = float(train_cfg['data'].get('ncc_scale', 1.0))

        print(f"[VisibilityGrid] Loading cameras from {source_path} …")
        data_pack = DataPack(cfg_data, white_background=cfg_model.white_background)
        cameras   = data_pack.get_train_cameras()
        print(f"[VisibilityGrid] {len(cameras)} training cameras loaded.")

        # ── 5. Load or render+cache depth maps ──────────────────────────────
        depth_dir  = os.path.join(model_path, "depths_npy")
        depth_maps = cls._load_or_render_depths(
            cameras, depth_dir, voxel_model)

        # ── 6. Build and return the grid ────────────────────────────────────
        return cls(
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            resolution=resolution,
            cameras=cameras,
            depth_maps=depth_maps,
            certainty_threshold=certainty_threshold,
            device=device,
        )

    @staticmethod
    def _load_or_render_depths(
        cameras,
        depth_dir: str,
        voxel_model,
    ) -> List[torch.Tensor]:
        """
        Load per-camera depth maps from ``depth_dir`` (float32 .npy files).

        For any camera whose cache file is absent, render the expected z-depth
        using ``voxel_model`` (same approach as train.py's "Final render pass")
        and save the result as ``depth_dir/<image_name>.npy``.

        Returns
        -------
        depth_maps : list[Tensor]
            One (H, W) CPU float32 tensor per camera.
        """
        os.makedirs(depth_dir, exist_ok=True)

        depth_maps  = []
        need_render = []          # image_names that need rendering

        for cam in cameras:
            npy_path = os.path.join(depth_dir, f"{cam.image_name}.npy")
            if os.path.isfile(npy_path):
                depth_maps.append(
                    torch.from_numpy(np.load(npy_path)))
            else:
                depth_maps.append(None)
                need_render.append(cam.image_name)

        if need_render:
            print(f"[VisibilityGrid] Rendering {len(need_render)} missing "
                  f"depth maps and caching to {depth_dir} …")
            # Build name → (list-index, camera) lookup for fast access
            cam_lookup = {cam.image_name: (i, cam)
                          for i, cam in enumerate(cameras)}

            voxel_model.freeze_vox_geo()
            with torch.no_grad():
                for cam_name in tqdm(need_render, desc="[VisGrid] rendering"):
                    cam_idx, cam = cam_lookup[cam_name]
                    render_pkg = voxel_model.render(
                        cam, output_depth=True, output_T=True)
                    # depth[0] = expected (mean) z-depth, shape (H, W)
                    depth = render_pkg['depth'][0].detach().cpu()
                    npy_path = os.path.join(depth_dir, f"{cam_name}.npy")
                    np.save(npy_path, depth.numpy().astype(np.float32))
                    depth_maps[cam_idx] = depth
            voxel_model.unfreeze_vox_geo()
            print(f"[VisibilityGrid] Depth maps cached to {depth_dir}")

        return depth_maps

    
    
    