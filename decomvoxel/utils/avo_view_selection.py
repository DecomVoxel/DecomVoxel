"""
AVO (Adaptive Visibility Optimization) inspired view selection for rendered object images.

The AVO optimization objective from SimRecon/optimize_by_avo.py maximizes the sum of
foreground alpha values within a CENTER REGION of the rendered image. This module
implements the same scoring criterion on pre-rendered PNG images (white-background
object renders), allowing view selection without running gradient-descent optimization.

Score definition
----------------
  avo_score(img) = number of foreground pixels inside the central (center_ratio × center_ratio)
                   crop of the image.

This is distinct from the two criteria already used in select_cond_img:
  - "area"        : total foreground pixels (entire image)
  - "center_dist" : distance of foreground centroid to image center

The AVO score rewards views where the object fills the central portion of the frame,
which directly mirrors the AVO loss function.
"""

from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np
from PIL import Image


def compute_avo_score(
    img_path: str,
    center_ratio: float = 0.60,
    white_threshold: int = 5,
) -> float:
    """
    Compute AVO-inspired visibility score for a single rendered image.

    The score equals the number of foreground pixels (non-white pixels) inside
    a centered rectangular crop of size ``(H * center_ratio) × (W * center_ratio)``.
    This mirrors the objective of SimRecon's optimize_instance_visibility which
    maximises ``alpha[center_h:, center_w:].sum()`` in camera space.

    Args:
        img_path:        Absolute or relative path to a white-background PNG image.
        center_ratio:    Fraction of each dimension used as the central crop.
                         0.60 means the central 60 % × 60 % region.
        white_threshold: Pixels with ALL channels >= (255 - white_threshold) are
                         treated as background.  Default 5 is conservative.

    Returns:
        Float score (≥ 0).  Returns 0.0 if the image cannot be loaded or is
        fully white.
    """
    try:
        img = np.array(Image.open(img_path).convert("RGB"))
    except Exception:
        return 0.0

    H, W = img.shape[:2]

    # Foreground mask: any channel below the white threshold
    white_lo = 255 - white_threshold
    fg_mask = np.any(img < white_lo, axis=-1)  # (H, W) bool

    if not fg_mask.any():
        return 0.0

    # Central crop boundaries
    ch = int(H * center_ratio)
    cw = int(W * center_ratio)
    start_h = (H - ch) // 2
    start_w = (W - cw) // 2
    end_h = start_h + ch
    end_w = start_w + cw

    central_fg = fg_mask[start_h:end_h, start_w:end_w]
    return float(central_fg.sum())


def select_best_avo_view(
    candidates: List[Tuple[str, int, float]],
    center_ratio: float = 0.60,
    white_threshold: int = 5,
) -> Tuple[str, int, float, float]:
    """
    Select the candidate view with the highest AVO score.

    Args:
        candidates:      List of ``(img_path, area, center_dist)`` tuples,
                         as produced inside ``select_cond_img``.
        center_ratio:    Passed through to ``compute_avo_score``.
        white_threshold: Passed through to ``compute_avo_score``.

    Returns:
        ``(img_path, area, center_dist, avo_score)`` for the best candidate.
        Returns the first candidate (with score 0) if the list is empty.
    """
    if not candidates:
        raise ValueError("candidates list is empty")

    best: Tuple[str, int, float, float] | None = None
    best_score = -1.0

    for img_path, area, center_dist in candidates:
        score = compute_avo_score(img_path, center_ratio=center_ratio,
                                  white_threshold=white_threshold)
        if score > best_score:
            best_score = score
            best = (img_path, area, center_dist, score)

    if best is None:
        img_path, area, center_dist = candidates[0]
        best = (img_path, area, center_dist, 0.0)

    return best
