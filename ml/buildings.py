"""Segment buildings from a height field and level their roofs.

A monocular depth model applied to nadir imagery returns a smooth surface, so
buildings arrive as mounds. Two properties are missing and neither can be
recovered by filtering: a roof is *flat*, and the drop to the street is
*vertical*. This module supplies the first by segmenting building regions and
fitting each one a plane; the renderer supplies the second by extruding walls
at the segment boundaries.

Deliberately no SciPy or OpenCV: the packaged desktop build already carries
Torch and rasterio, and a morphological opening plus connected-component
labelling are short enough in NumPy to not be worth another dependency.
"""

from __future__ import annotations

import numpy as np


DEFAULT_GROUND_RADIUS = 48
DEFAULT_HEIGHT_THRESHOLD = 0.06
DEFAULT_MIN_AREA = 24


def _sliding_reduce(array: np.ndarray, radius: int, reducer) -> np.ndarray:
    """Apply a separable square-window reduction (min or max)."""

    size = 2 * radius + 1
    padded = np.pad(array, radius, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, size, axis=0)
    rows = reducer(windows, axis=-1)
    windows = np.lib.stride_tricks.sliding_window_view(rows, size, axis=1)
    return reducer(windows, axis=-1)


def estimate_ground(heights: np.ndarray, radius: int = DEFAULT_GROUND_RADIUS) -> np.ndarray:
    """Approximate the bare-earth surface by a grayscale morphological opening.

    Erosion removes anything narrower than the structuring element, so buildings
    disappear while the terrain they stand on survives; the dilation restores
    the terrain's scale. The element must be wider than the widest building or
    that building is mistaken for ground.
    """

    array = np.asarray(heights, dtype=np.float32)
    eroded = _sliding_reduce(array, radius, np.min)
    return _sliding_reduce(eroded, radius, np.max).astype(np.float32, copy=False)


def label_regions(mask: np.ndarray, min_area: int = DEFAULT_MIN_AREA) -> tuple[np.ndarray, int]:
    """Label 4-connected regions, dropping those below ``min_area``.

    Two-pass union-find. Small regions are discarded because at two metres per
    cell a handful of cells is a tree crown or a depth artefact, and levelling
    those would flatten real detail.
    """

    binary = np.asarray(mask, dtype=bool)
    height, width = binary.shape
    labels = np.zeros((height, width), dtype=np.int32)
    parent: list[int] = [0]

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a: int, b: int) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[max(root_a, root_b)] = min(root_a, root_b)

    next_label = 1
    for row in range(height):
        row_mask = binary[row]
        for column in range(width):
            if not row_mask[column]:
                continue
            north = labels[row - 1, column] if row > 0 else 0
            west = labels[row, column - 1] if column > 0 else 0
            if north and west:
                labels[row, column] = min(north, west)
                union(north, west)
            elif north or west:
                labels[row, column] = north or west
            else:
                labels[row, column] = next_label
                parent.append(next_label)
                next_label += 1

    if next_label == 1:
        return labels, 0

    resolved = np.array([find(index) for index in range(next_label)], dtype=np.int32)
    labels = resolved[labels]

    counts = np.bincount(labels.reshape(-1))
    keep = np.zeros(counts.shape[0], dtype=np.int32)
    surviving = 0
    for index in range(1, counts.shape[0]):
        if counts[index] >= min_area:
            surviving += 1
            keep[index] = surviving
    return keep[labels], surviving


def flatten_building_roofs(
    heights: np.ndarray,
    ground_radius: int = DEFAULT_GROUND_RADIUS,
    height_threshold: float = DEFAULT_HEIGHT_THRESHOLD,
    min_area: int = DEFAULT_MIN_AREA,
    valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Level each detected roof onto its own fitted plane.

    A plane rather than a constant, so pitched and stepped roofs stay believable
    instead of being forced flat. Cells outside any building keep their original
    height, so terrain is untouched.
    """

    array = np.asarray(heights, dtype=np.float32)
    finite = np.isfinite(array)
    if valid_mask is not None:
        finite &= np.asarray(valid_mask, dtype=bool)
    report: dict[str, object] = {"buildings": 0, "coverage": 0.0}
    if not finite.any():
        return array, report

    fill = float(np.median(array[finite]))
    working = np.where(finite, array, fill).astype(np.float32)

    spread = float(working.max() - working.min())
    if spread <= np.finfo(np.float32).eps:
        return array, report

    ground = estimate_ground(working, ground_radius)
    normalized = working - ground
    labels, count = label_regions(normalized > spread * height_threshold, min_area)
    report["buildings"] = int(count)
    if count == 0:
        return array, report

    output = working.copy()
    rows, columns = np.indices(working.shape)
    for index in range(1, count + 1):
        selection = labels == index
        cells = int(selection.sum())
        if cells < min_area:
            continue
        y = rows[selection].astype(np.float64)
        x = columns[selection].astype(np.float64)
        z = working[selection].astype(np.float64)
        # Least squares plane; fall back to the median when the region is
        # degenerate, such as a single row of cells.
        design = np.column_stack([x, y, np.ones_like(x)])
        try:
            coefficients, *_ = np.linalg.lstsq(design, z, rcond=None)
            fitted = design @ coefficients
        except np.linalg.LinAlgError:
            fitted = np.full_like(z, np.median(z))
        # A plane fit through a mound tilts toward the mound; blending keeps a
        # genuinely sloped roof from being over-corrected.
        output[selection] = (0.94 * fitted + 0.06 * z).astype(np.float32)

    report["coverage"] = float((labels > 0).mean())
    return np.where(finite, output, array).astype(np.float32, copy=False), report
