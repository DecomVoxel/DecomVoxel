"""
Voxel Geometric Uncertainty for DecomVoxel.

Formula (from paper):

    U_base(l)  = w_s / (beta * (l + l0))
    U_geom(v)  = U_base(l) * (1 - exp(-v_geo))

where:
    w_s    – voxel size (GeoSVRVoxelData.sizes)
    l      – octree level (GeoSVRVoxelData.octlevel)
    beta   – global scaling factor (default 1.0)
    l0     – starting-level offset (default 1.0, prevents division by zero at l=0)
    v_geo  – per-voxel geometry/density value.
             If not provided explicitly, colour luminance (BT.709) is used as a proxy,
             since it correlates with the SH0-based density representation in SVRaster.

All operations are fully vectorised over the N voxels.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional
import torch

if TYPE_CHECKING:
    from decomvoxel.pipeline.load_voxel import GeoSVRVoxelData

def compute_voxel_uncertainty(
    voxel_data: "GeoSVRVoxelData",
    beta: float = 1.0,
    l0: float = 1.0,
    v_geo: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute per-voxel geometric uncertainty for a GeoSVRVoxelData object.
    """
    dev = voxel_data.centers.device

    # w_s: (N,) – voxel edge length
    sizes = voxel_data.sizes.squeeze(-1).float().to(dev)   # (N,)
    # l: (N,) – octree level
    levels = voxel_data.octlevel.float().to(dev)           # (N,)

    # v_geo: (N,) – per-voxel geometry / density proxy
    if v_geo is None:
        # BT.709 luminance as a proxy for density; fully vectorised
        lum_w = torch.tensor([0.2126, 0.7152, 0.0722], dtype=torch.float32, device=dev)
        v_geo = (voxel_data.colors.float().to(dev) * lum_w).sum(dim=1)  # (N,)
    else:
        v_geo = v_geo.float().to(dev)

    # U_base(l) = w_s / (beta * (l + l0))
    u_base = sizes / (beta * (levels + l0))                # (N,)
    # U_geom(v) = U_base(l) * (1 - exp(-v_geo))
    u_geom = u_base * (1.0 - torch.exp(-v_geo))           # (N,)

    return u_geom

def compute_voxel_uncertainty_base(
    voxel_data: "GeoSVRVoxelData",
    beta: float = 1.0,
    l0: float = 1.0,
) -> torch.Tensor:
    dev = voxel_data.centers.device
    sizes = voxel_data.sizes.squeeze(-1).float().to(dev)   # (N,)
    levels = voxel_data.octlevel.float().to(dev)           # (N,)
    return sizes / (beta * (levels + l0))                  # (N,)

def compute_voxel_certainty_base(
    voxel_data: "GeoSVRVoxelData",
    beta: float = 1.0,
    l0: float = 1.0,
) -> torch.Tensor:
    return 1.0 / compute_voxel_uncertainty_base(voxel_data, beta=beta, l0=l0)

def normalize_uncertainty(u: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Min-max normalise uncertainty values to [0, 1]."""
    u_min = u.min()
    u_max = u.max()
    return (u - u_min) / (u_max - u_min + eps)

def normalize_certainty(c: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Min-max normalise certainty values to [0, 1]."""
    c_min = c.min()
    c_max = c.max()
    return (c - c_min) / (c_max - c_min + eps)

def normalize_certainty_log(certainty: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    log_cert = torch.log(certainty + 1)          # Add 1 to avoid log(0)
    log_min = log_cert.min()
    log_max = log_cert.max()
    if log_max - log_min < eps:
        return torch.zeros_like(certainty)
    return (log_cert - log_min) / (log_max - log_min)