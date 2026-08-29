import numpy as np

from ml.buildings import flatten_building_roofs, make_building_footprints


def test_building_footprints_create_an_oriented_solid() -> None:
    labels = np.zeros((48, 64), dtype=np.int32)
    labels[14:30, 18:42] = 1
    ground = np.full(labels.shape, 0.2, dtype=np.float32)
    roofs = ground.copy()
    roofs[labels == 1] = 0.72

    footprints = make_building_footprints(labels, roofs, ground)

    assert len(footprints) == 1
    footprint = footprints[0]
    assert footprint["_label"] == 1
    assert len(footprint["points"]) == 4
    assert np.isclose(footprint["roof_height"], 0.72)
    assert np.isclose(footprint["base_height"], 0.2)
    assert 0.0 <= min(point[0] for point in footprint["points"]) <= 1.0
    assert 0.0 <= max(point[1] for point in footprint["points"]) <= 1.0


def test_building_footprints_preserve_a_concave_roof_outline() -> None:
    labels = np.zeros((512, 512), dtype=np.int32)
    labels[70:118, 72:88] = 1
    labels[102:118, 72:136] = 1
    ground = np.full(labels.shape, 0.15, dtype=np.float32)
    roofs = ground.copy()
    roofs[labels == 1] = 0.76

    footprints = make_building_footprints(labels, roofs, ground)

    assert len(footprints) == 1
    points = np.asarray(footprints[0]["points"], dtype=np.float64)
    assert len(points) >= 6
    x, y = points[:, 0], points[:, 1]
    polygon_area = abs(0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))
    bounding_area = float(np.ptp(x) * np.ptp(y))
    assert polygon_area < bounding_area * 0.75


def test_flatten_can_return_private_segmentation_for_rendering() -> None:
    heights = np.zeros((96, 96), dtype=np.float32)
    heights[34:58, 30:62] = 0.8

    _, report = flatten_building_roofs(
        heights,
        ground_radius=18,
        min_area=24,
        return_regions=True,
    )

    assert report["buildings"] == 1
    assert np.asarray(report["_labels"]).shape == heights.shape
    assert np.asarray(report["_ground"]).shape == heights.shape


def test_building_footprints_reject_green_canopy_regions() -> None:
    labels = np.zeros((64, 64), dtype=np.int32)
    labels[8:28, 8:28] = 1
    labels[36:56, 36:56] = 2
    ground = np.full(labels.shape, 0.1, dtype=np.float32)
    roofs = ground.copy()
    roofs[labels > 0] = 0.7
    rgb = np.full((*labels.shape, 3), 150, dtype=np.uint8)
    rgb[labels == 1] = np.array([45, 150, 50], dtype=np.uint8)

    footprints = make_building_footprints(labels, roofs, ground, rgb=rgb)

    assert len(footprints) == 1
    centre_u = np.mean([point[0] for point in footprints[0]["points"]])
    assert centre_u > 0.5
