from __future__ import annotations

import numpy as np

import labelimage_tools as lit


def four_label_meeting():
    return np.array(
        [
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [3, 3, 4, 4],
            [3, 3, 4, 4],
        ],
        dtype=np.int64,
    )


def test_junction_pixels_and_clusters():
    labels = four_label_meeting()
    mask, labels_at_pixel = lit.junction_pixels_with_labels(labels)
    assert mask.any()
    assert labels_at_pixel
    label_image, junctions = lit.cluster_junctions_with_labels(mask, labels_at_pixel)
    assert label_image.max() == len(junctions)
    assert junctions[0].id == 1
    assert junctions[0].labels == frozenset({1, 2, 3, 4})
    assert np.isfinite(junctions[0].yx).all()
    assert np.issubdtype(junctions[0].pixel_coords.dtype, np.integer)


def test_junctions_from_three_label_meeting_excluding_background():
    labels = np.array([[1, 1, 2], [1, 3, 2], [3, 3, 2]], dtype=np.int64)
    label_image, junctions = lit.junctions_from_labels(labels, background=0)
    assert label_image.max() >= 1
    assert any(j.labels == frozenset({1, 2, 3}) for j in junctions)


def test_merge_close_junctions():
    j1 = lit.Junction(1, np.array([0.0, 0.0]), np.array([[0, 0]]), frozenset({1, 2, 3}))
    j2 = lit.Junction(2, np.array([0.2, 0.2]), np.array([[0, 1]]), frozenset({3, 4, 5}))
    merged = lit.merge_close_junctions([j1, j2], epsilon=1.0)
    assert len(merged) == 1
    assert merged[0].labels == frozenset({1, 2, 3, 4, 5})
