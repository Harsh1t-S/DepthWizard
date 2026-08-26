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
    return_regions: bool = False,
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
    if return_regions:
        # Private, in-process fields used to build display-only solid geometry.
        # They are deliberately not included in metrics or exported DSM data.
        report["_labels"] = labels
        report["_ground"] = ground
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


def make_building_footprints(
    labels: np.ndarray,
    roof_heights: np.ndarray,
    ground: np.ndarray,
    *,
    rgb: np.ndarray | None = None,
    max_buildings: int = 160,
    minimum_fill: float = 0.42,
) -> list[dict[str, object]]:
    """Create compact oriented roof footprints for display-only extrusion.

    Each connected component is approximated by a PCA-oriented rectangle. The
    result is intentionally small JSON rather than a dense segmentation mask,
    and gives the browser actual roof polygons and vertical wall locations.
    Irregular, extremely thin components are rejected because they are usually
    roads, trees, or monocular-depth artefacts rather than buildings.
    """

    regions = np.asarray(labels, dtype=np.int32)
    roofs = np.asarray(roof_heights, dtype=np.float32)
    bare_ground = np.asarray(ground, dtype=np.float32)
    if regions.shape != roofs.shape or regions.shape != bare_ground.shape:
        raise ValueError("Labels, roof heights, and ground must share a pixel grid")
    if regions.ndim != 2 or not regions.size or regions.max(initial=0) <= 0:
        return []

    image = None if rgb is None else np.asarray(rgb)
    if image is not None and (image.ndim != 3 or image.shape[:2] != regions.shape or image.shape[2] < 3):
        raise ValueError("RGB guidance must share the labels' pixel grid")

    height, width = regions.shape
    scene_spread = float(np.nanmax(roofs) - np.nanmin(roofs))
    minimum_relief = max(scene_spread * 0.035, np.finfo(np.float32).eps)
    counts = np.bincount(regions.reshape(-1))
    ordered_labels = np.argsort(counts[1:])[::-1] + 1
    footprints: list[dict[str, object]] = []

    for label in ordered_labels:
        if len(footprints) >= max_buildings:
            break
        rows, columns = np.nonzero(regions == int(label))
        area = int(rows.size)
        if area < DEFAULT_MIN_AREA:
            continue

        points = np.column_stack((columns, rows)).astype(np.float64)
        centre = points.mean(axis=0)
        centred = points - centre
        covariance = centred.T @ centred / max(1, area)
        _, eigenvectors = np.linalg.eigh(covariance)
        major = eigenvectors[:, -1]
        if major[0] < 0:
            major = -major
        minor = np.array([-major[1], major[0]], dtype=np.float64)
        axes = np.column_stack((major, minor))
        projected = centred @ axes

        lower = projected.min(axis=0) - 0.5
        upper = projected.max(axis=0) + 0.5
        spans = np.maximum(upper - lower, 1.0)
        rectangle_area = float(spans[0] * spans[1])
        fill = float(area / rectangle_area)
        aspect = float(max(spans) / max(1.0, min(spans)))
        if fill < minimum_fill or aspect > 12.0:
            continue
        # A huge, loosely rectangular region is normally a hill, lake, or tree
        # canopy merged by the relative-depth threshold rather than one roof.
        if rectangle_area / float(height * width) > 0.018 and fill < 0.72:
            continue

        green_fraction = 0.0
        water_fraction = 0.0
        if image is not None:
            colours = image[rows, columns, :3].astype(np.float32)
            red, green, blue = colours[:, 0], colours[:, 1], colours[:, 2]
            green_excess = green - 0.5 * (red + blue)
            green_fraction = float(np.mean((green_excess > 10.0) & (green > 45.0)))
            water_fraction = float(
                np.mean((blue - red > 14.0) & (blue >= green * 0.92) & (green < 165.0))
            )
            if green_fraction > 0.28 or water_fraction > 0.46:
                continue

        roof_height = float(np.nanmedian(roofs[rows, columns]))
        base_height = float(np.nanmedian(bare_ground[rows, columns]))
        if not np.isfinite(roof_height) or not np.isfinite(base_height):
            continue
        if roof_height - base_height < minimum_relief:
            continue

        projected_corners = np.array(
            [
                [lower[0], lower[1]],
                [upper[0], lower[1]],
                [upper[0], upper[1]],
                [lower[0], upper[1]],
            ],
            dtype=np.float64,
        )
        corners = projected_corners @ axes.T + centre
        corners[:, 0] = np.clip(corners[:, 0], 0, max(0, width - 1))
        corners[:, 1] = np.clip(corners[:, 1], 0, max(0, height - 1))
        normalized_corners = [
            [
                float(column / max(1, width - 1)),
                float(row / max(1, height - 1)),
            ]
            for column, row in corners
        ]

        footprints.append(
            {
                "points": normalized_corners,
                "roof_height": roof_height,
                "base_height": base_height,
                "area_pixels": area,
                "confidence": fill * (1.0 - 0.65 * green_fraction - 0.45 * water_fraction),
            }
        )

    return footprints
