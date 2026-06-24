from __future__ import annotations

import numpy as np

from labelimage_tools._bbox import label_slices


def _slice_bounds(slc):
    return (slc[0].start, slc[0].stop, slc[1].start, slc[1].stop)


def test_label_slices_ordinary_labels():
    labels = np.array([[0, 1, 1], [2, 2, 0]], dtype=np.int64)
    slices = label_slices(labels)
    assert set(slices) == {1, 2}
    assert _slice_bounds(slices[1]) == (0, 1, 1, 3)
    assert _slice_bounds(slices[2]) == (1, 2, 0, 2)


def test_label_slices_nonconsecutive_positive_labels():
    labels = np.zeros((6, 6), dtype=np.int64)
    labels[1:3, 1:4] = 5
    labels[4:6, 2:6] = 10
    slices = label_slices(labels)
    assert set(slices) == {5, 10}
    assert _slice_bounds(slices[5]) == (1, 3, 1, 4)
    assert _slice_bounds(slices[10]) == (4, 6, 2, 6)


def test_label_slices_negative_labels():
    labels = np.zeros((4, 5), dtype=np.int64)
    labels[1:3, 2:4] = -3
    labels[0, 0] = 1
    slices = label_slices(labels)
    assert set(slices) == {-3, 1}
    assert _slice_bounds(slices[-3]) == (1, 3, 2, 4)


def test_label_slices_huge_sparse_labels():
    labels = np.zeros((5, 5), dtype=np.int64)
    labels[1:4, 1:3] = 1
    labels[3:5, 3:5] = 100_001
    slices = label_slices(labels)
    assert set(slices) == {1, 100_001}
    assert _slice_bounds(slices[100_001]) == (3, 5, 3, 5)


def test_label_slices_excludes_nonzero_background_by_default():
    labels = np.full((4, 4), 99, dtype=np.int64)
    labels[1:3, 1:3] = -3
    slices = label_slices(labels, background=99)
    assert set(slices) == {-3}


def test_label_slices_can_include_nonzero_background():
    labels = np.full((4, 4), 99, dtype=np.int64)
    labels[1:3, 1:3] = -3
    slices = label_slices(labels, background=99, include_background=True)
    assert set(slices) == {-3, 99}
    assert _slice_bounds(slices[99]) == (0, 4, 0, 4)


def test_label_slices_padding_expands_and_clips_to_image():
    labels = np.zeros((5, 5), dtype=np.int64)
    labels[0, 0] = 7
    labels[3, 3] = 8
    slices = label_slices(labels, padding=2)
    assert _slice_bounds(slices[7]) == (0, 3, 0, 3)
    assert _slice_bounds(slices[8]) == (1, 5, 1, 5)


def test_label_slices_compacts_many_problematic_labels():
    labels = np.zeros((4, 4), dtype=np.int64)
    for idx, label in enumerate([-1, -2, -3, -4]):
        labels[idx, idx] = label
    slices = label_slices(labels, max_manual_labels=2)
    assert set(slices) == {-1, -2, -3, -4}
    assert _slice_bounds(slices[-4]) == (3, 4, 3, 4)
