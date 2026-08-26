"""Edge-aware refinement of predicted depth using the source image as a guide.

Monocular models trained on natural photographs place depth edges only
approximately when applied to nadir imagery: building outlines come back soft
and displaced, which is what reconstructs them as rounded mounds rather than
blocks. The source image already contains the exact boundary, so it is used to
pull the depth edges onto it.

This is a guided filter (He et al., "Guided Image Filtering"). It behaves like a
smoothing filter inside regions of similar guide intensity and preserves
transitions across them, so flat ground denoises while roof-to-street steps
sharpen. Implemented with cumulative-sum box filters, so it needs only NumPy and
runs in well under a second on a full-resolution tile.
"""

from __future__ import annotations

import numpy as np


DEFAULT_RADIUS = 8
DEFAULT_EPSILON = 1e-4


def _box_filter(image: np.ndarray, radius: int) -> np.ndarray:
    """Sum over a (2*radius+1) square window, via cumulative sums."""

    height, width = image.shape
    padded = np.cumsum(image, axis=0)
    output = np.empty_like(image)
    output[: radius + 1] = padded[radius : 2 * radius + 1]
    output[radius + 1 : height - radius] = (
        padded[2 * radius + 1 :] - padded[: height - 2 * radius - 1]
    )
    output[height - radius :] = (
        padded[-1][None, :] - padded[height - 2 * radius - 1 : height - radius - 1]
    )

    padded = np.cumsum(output, axis=1)
    output = np.empty_like(image)
    output[:, : radius + 1] = padded[:, radius : 2 * radius + 1]
    output[:, radius + 1 : width - radius] = (
        padded[:, 2 * radius + 1 :] - padded[:, : width - 2 * radius - 1]
    )
    output[:, width - radius :] = (
        padded[:, -1][:, None] - padded[:, width - 2 * radius - 1 : width - radius - 1]
    )
    return output


def guided_filter(
    guide: np.ndarray,
    source: np.ndarray,
    radius: int = DEFAULT_RADIUS,
    epsilon: float = DEFAULT_EPSILON,
) -> np.ndarray:
    """Filter ``source`` so its edges follow ``guide``. Both in [0, 1]."""

    guide = np.asarray(guide, dtype=np.float32)
    source = np.asarray(source, dtype=np.float32)
    if guide.shape != source.shape:
        raise ValueError("Guide and source must share a shape")
    # A window wider than the image makes the box filter indexing degenerate.
    radius = max(1, min(int(radius), (min(guide.shape) - 1) // 2))

    counts = _box_filter(np.ones_like(guide), radius)
    mean_guide = _box_filter(guide, radius) / counts
    mean_source = _box_filter(source, radius) / counts
    variance = _box_filter(guide * guide, radius) / counts - mean_guide * mean_guide
    covariance = _box_filter(guide * source, radius) / counts - mean_guide * mean_source

    scale = covariance / (variance + epsilon)
    offset = mean_source - scale * mean_guide
    return (
        _box_filter(scale, radius) / counts * guide + _box_filter(offset, radius) / counts
    ).astype(np.float32, copy=False)


def refine_depth_with_image(
    depth: np.ndarray,
    rgb: np.ndarray,
    radius: int = DEFAULT_RADIUS,
    epsilon: float = DEFAULT_EPSILON,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Snap predicted depth edges onto the imagery, preserving depth units.

    Filtering happens in a normalized space so ``epsilon`` has a consistent
    meaning across scenes, and the original scale and offset are restored
    afterwards, leaving downstream calibration and metrics unchanged.
    """

    depth = np.asarray(depth, dtype=np.float32)
    finite = np.isfinite(depth)
    if valid_mask is not None:
        finite &= np.asarray(valid_mask, dtype=bool)
    if not finite.any():
        return depth

    low = float(depth[finite].min())
    high = float(depth[finite].max())
    span = high - low
    if span <= np.finfo(np.float32).eps:
        return depth

    # Non-finite pixels would poison every window they touch, so fill them with
    # the median before filtering and restore them afterwards.
    filled = np.where(finite, depth, np.median(depth[finite])).astype(np.float32)
    normalized = (filled - low) / span

    guide = np.asarray(rgb, dtype=np.float32)
    if guide.ndim == 3:
        # Rec. 601 luma: matches perceived edge contrast better than a flat mean.
        guide = guide[..., 0] * 0.299 + guide[..., 1] * 0.587 + guide[..., 2] * 0.114
    if guide.shape != depth.shape:
        raise ValueError("Guide image and depth must share a shape")
    guide = guide / 255.0 if guide.max() > 1.5 else guide

    refined = guided_filter(guide, normalized, radius=radius, epsilon=epsilon)
    output = refined * span + low
    return np.where(finite, output, depth).astype(np.float32, copy=False)
