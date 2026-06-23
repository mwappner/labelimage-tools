from __future__ import annotations

import numpy as np

import labelimage_tools as lit


def test_erode_and_dilate_nonconsecutive_labels(nonconsecutive_labels):
    eroded = lit.erode_labels(nonconsecutive_labels)
    dilated = lit.dilate_labels(nonconsecutive_labels)
    assert set(np.unique(eroded)) <= {0, 5, 10}
    assert {5, 10} <= set(np.unique(dilated))
    assert np.array_equal(lit.dialate_labels(nonconsecutive_labels), dilated)


def test_shuffle_preserves_background(nonconsecutive_labels):
    shuffled = lit.shuffle_labels(nonconsecutive_labels, seed=1, background=0)
    assert np.all(shuffled[nonconsecutive_labels == 0] == 0)
    assert set(np.unique(shuffled)) == {0, 5, 10}


def test_fill_internal_gaps_and_sentinel():
    labels = np.full((9, 9), 5, dtype=np.int64)
    labels[3:6, 3:6] = 0
    filled = lit.fill_internal_gaps_edt(labels, max_distance=None)
    assert np.all(filled == 5)
    sentinel = lit.fill_internal_gaps_edt(labels, max_distance=0.5, fill_value=10000)
    assert 10000 in np.unique(sentinel)


def test_skeletonize_variants(nonconsecutive_labels):
    assert lit.skeletonize_dilate(nonconsecutive_labels).shape == nonconsecutive_labels.shape
    assert lit.skeletonize_erode(nonconsecutive_labels).shape == nonconsecutive_labels.shape
    assert np.array_equal(
        lit.skeletonize_labels(nonconsecutive_labels),
        lit.skeletonize_erode(nonconsecutive_labels),
    )


def test_find_and_remove_non_self_connected_bits():
    labels = np.zeros((8, 8), dtype=np.int64)
    labels[1:3, 1:3] = 5
    labels[5:7, 5:7] = 5
    bad = lit.find_non_self_connected_labels(labels)
    assert 5 in bad
    cleaned = lit.remove_non_self_connected_bits(labels)
    assert 5 not in lit.find_non_self_connected_labels(cleaned)
    assert np.sum(cleaned == 5) == 4


def test_crop_to_foreground_bbox(nonconsecutive_labels):
    cropped, slices = lit.crop_to_foreground_bbox(nonconsecutive_labels, padding=0)
    assert cropped.shape == (6, 6)
    assert slices == (slice(1, 7), slice(1, 7))


def test_load_image_pipeline(sample_path):
    labels = lit.load_image_pipeline(sample_path, crop_to_foreground=True, shuffle=False)
    assert labels.ndim == 2
    assert labels.max() > 0
