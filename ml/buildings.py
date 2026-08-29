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
from rasterio.features import shapes as raster_shapes

DEFAULT_GROUND_RADIUS = 32
DEFAULT_HEIGHT_THRESHOLD = 0.035
DEFAULT_MIN_AREA = 16


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


def _simplify_polyline(points: np.ndarray, tolerance: float) -> np.ndarray:
    """Simplify an open polyline with iterative Ramer-Douglas-Peucker."""

    if len(points) <= 2:
        return points

    keep = np.zeros(len(points), dtype=bool)
    keep[0] = keep[-1] = True
    pending = [(0, len(points) - 1)]
    tolerance_squared = tolerance * tolerance

    while pending:
        start, end = pending.pop()
        if end <= start + 1:
            continue
        origin = points[start]
        segment = points[end] - origin
        segment_squared = float(segment @ segment)
        candidates = points[start + 1 : end]
        if segment_squared <= np.finfo(np.float64).eps:
            distances_squared = np.sum((candidates - origin) ** 2, axis=1)
        else:
            projections = np.clip(((candidates - origin) @ segment) / segment_squared, 0.0, 1.0)
            closest = origin + projections[:, None] * segment
            distances_squared = np.sum((candidates - closest) ** 2, axis=1)
        relative = int(np.argmax(distances_squared))
        if float(distances_squared[relative]) <= tolerance_squared:
            continue
        index = start + 1 + relative
        keep[index] = True
        pending.append((start, index))
        pending.append((index, end))

    return points[keep]


def _signed_polygon_area(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    x = points[:, 0]
    y = points[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def _simplify_closed_ring(points: np.ndarray, max_vertices: int = 32) -> np.ndarray | None:
    """Simplify a closed raster outline while preserving its winding."""

    ring = np.asarray(points, dtype=np.float64)
    if len(ring) > 1 and np.allclose(ring[0], ring[-1]):
        ring = ring[:-1]
    if len(ring) < 3:
        return None

    consecutive = np.ones(len(ring), dtype=bool)
    consecutive[1:] = np.any(np.abs(ring[1:] - ring[:-1]) > 1e-9, axis=1)
    ring = ring[consecutive]
    if len(ring) < 3:
        return None

    original_area = _signed_polygon_area(ring)
    if abs(original_area) <= np.finfo(np.float64).eps:
        return None

    opposite = int(np.argmax(np.sum((ring - ring[0]) ** 2, axis=1)))
    if opposite <= 0:
        return None

    first_half = ring[: opposite + 1]
    second_half = np.vstack((ring[opposite:], ring[:1]))
    simplified = ring
    tolerance = 0.65
    for _ in range(12):
        first = _simplify_polyline(first_half, tolerance)
        second = _simplify_polyline(second_half, tolerance)
        candidate = np.vstack((first[:-1], second[:-1]))
        if len(candidate) >= 3 and abs(_signed_polygon_area(candidate)) > 1e-6:
            simplified = candidate
        if len(simplified) <= max_vertices:
            break
        tolerance *= 1.45

    if len(simplified) > max_vertices:
        return None
    if _signed_polygon_area(simplified) * original_area < 0:
        simplified = simplified[::-1]
    return simplified


def _vectorize_region_outlines(labels: np.ndarray) -> dict[int, np.ndarray]:
    """Return a compact exterior polygon for each connected label."""

    regions = np.asarray(labels, dtype=np.int32)
    outlines: dict[int, np.ndarray] = {}
    outline_areas: dict[int, float] = {}
    for geometry, raw_label in raster_shapes(
        regions,
        mask=regions > 0,
        connectivity=4,
    ):
        label = int(raw_label)
        if label <= 0 or geometry.get("type") != "Polygon":
            continue
        coordinates = geometry.get("coordinates", [])
        if not coordinates:
            continue
        outline = _simplify_closed_ring(np.asarray(coordinates[0], dtype=np.float64))
        if outline is None:
            continue
        area = abs(_signed_polygon_area(outline))
        if area > outline_areas.get(label, 0.0):
            outlines[label] = outline
            outline_areas[label] = area
    return outlines


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
    max_buildings: int = 500,
    minimum_fill: float = 0.28,
) -> list[dict[str, object]]:
    """Create compact mask-derived roof footprints for display-only extrusion.

    Each connected component is vectorized into a simplified exterior polygon.
    Its PCA-oriented rectangle is still used to reject irregular, extremely
    thin components, which are usually roads, trees, or monocular-depth
    artefacts rather than buildings. The result stays compact JSON rather than
    a dense segmentation mask and gives the browser true wall locations.
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
    # Adaptive relief cutoff to detect both low suburban residential roofs
    # and tall commercial structures without leaving mounds.
    minimum_relief = max(scene_spread * 0.012, np.finfo(np.float32).eps)
    counts = np.bincount(regions.reshape(-1))
    ordered_labels = np.argsort(counts[1:])[::-1] + 1
    vectorized_outlines = _vectorize_region_outlines(regions)
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
        # Reject thin needle columns, fence lines, and non-building spikes
        if fill < minimum_fill or aspect > 6.5 or min(spans) < 3.5:
            continue
        # Reject huge, amorphous regions (hills, large forests, rivers) while
        # accepting large complexes, schools, malls, and L/U-shaped structures.
        if rectangle_area / float(height * width) > 0.40 and fill < 0.22:
            continue

        green_fraction = 0.0
        water_fraction = 0.0
        if image is not None:
            colours = image[rows, columns, :3].astype(np.float32)
            red, green, blue = colours[:, 0], colours[:, 1], colours[:, 2]
            green_excess = green - 0.5 * (red + blue)
            green_fraction = float(np.mean((green_excess > 12.0) & (green > 45.0)))
            water_fraction = float(
                np.mean((blue - red > 14.0) & (blue >= green * 0.92) & (green < 165.0))
            )
            region_fraction = rectangle_area / float(height * width)
            # True tree canopies are dominated by green. Houses with trees/lawns nearby
            # remain accepted unless the region is primarily foliage.
            if green_fraction > 0.65:
                continue
            if region_fraction > 0.02 and green_fraction > 0.48:
                continue
            if region_fraction > 0.02 and water_fraction > 0.50:
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
        outline = vectorized_outlines.get(int(label), corners)
        # Raster polygon coordinates describe pixel edges in [0, width/height].
        # Shift them onto the sample-centre coordinate system used by the mesh.
        outline = outline.copy()
        if int(label) in vectorized_outlines:
            outline -= 0.5
        outline[:, 0] = np.clip(outline[:, 0], 0, max(0, width - 1))
        outline[:, 1] = np.clip(outline[:, 1], 0, max(0, height - 1))
        normalized_outline = [
            [
                float(column / max(1, width - 1)),
                float(row / max(1, height - 1)),
            ]
            for column, row in outline
        ]

        footprints.append(
            {
                "_label": int(label),
                "points": normalized_outline,
                "roof_height": roof_height,
                "base_height": base_height,
                "area_pixels": area,
                "confidence": fill * (1.0 - 0.65 * green_fraction - 0.45 * water_fraction),
            }
        )

    return footprints
