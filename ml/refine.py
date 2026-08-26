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


DEFAULT_FLATTEN_ITERATIONS = 12
DEFAULT_FLATTEN_KAPPA = 0.03


def flatten_surfaces(
    depth: np.ndarray,
    iterations: int = DEFAULT_FLATTEN_ITERATIONS,
    kappa: float = DEFAULT_FLATTEN_KAPPA,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Drive the surface toward piecewise-flat regions with sharp boundaries.

    Perona-Malik anisotropic diffusion. Each step moves a pixel toward its
    neighbours in proportion to how similar they already are, so the interior of
    a rooftop levels out while the step down to the street is left alone. This
    is the difference between a rooftop that reads as a flat plane and one that
    reads as a mound.

    ``kappa`` is expressed as a fraction of the height spread, so the edge
    threshold adapts to the scene rather than assuming a unit scale.
    """

    array = np.asarray(depth, dtype=np.float32)
    finite = np.isfinite(array)
    if valid_mask is not None:
        finite &= np.asarray(valid_mask, dtype=bool)
    if not finite.any() or iterations <= 0:
        return array

    low = float(array[finite].min())
    high = float(array[finite].max())
    span = high - low
    if span <= np.finfo(np.float32).eps:
        return array

    working = np.where(finite, array, np.median(array[finite])).astype(np.float32)
    threshold = max(span * float(kappa), np.finfo(np.float32).eps)

    for _ in range(int(iterations)):
        # Forward differences in each direction; edges are zero-padded so the
        # border neither gains nor loses height.
        north = np.zeros_like(working)
        south = np.zeros_like(working)
        west = np.zeros_like(working)
        east = np.zeros_like(working)
        north[1:, :] = working[:-1, :] - working[1:, :]
        south[:-1, :] = working[1:, :] - working[:-1, :]
        west[:, 1:] = working[:, :-1] - working[:, 1:]
        east[:, :-1] = working[:, 1:] - working[:, :-1]

        # Conduction falls off with gradient magnitude, so flow effectively
        # stops at a roof edge while continuing across a flat roof.
        def conduct(difference: np.ndarray) -> np.ndarray:
            ratio = difference / threshold
            return np.exp(-(ratio * ratio)).astype(np.float32)

        # 0.25 is the stability limit for 4-neighbour explicit diffusion.
        working = working + 0.25 * (
            conduct(north) * north
            + conduct(south) * south
            + conduct(west) * west
            + conduct(east) * east
        )

    return np.where(finite, working, array).astype(np.float32, copy=False)
