from __future__ import annotations

import numpy as np

import labelimage_tools as lit


def test_ordered_contour_from_mask_and_labels(nonconsecutive_labels):
    contour = lit.ordered_contour_from_mask(nonconsecutive_labels == 5)
    assert contour.shape[1] == 2
    contours = lit.ordered_contours_from_labels(nonconsecutive_labels)
    assert set(contours) == {5, 10}
    assert all(value.shape[1] == 2 for value in contours.values())


def test_ordered_contours_from_labels_uses_global_coordinates_after_windowing():
    labels = np.zeros((20, 20), dtype=np.int64)
    labels[10:15, 12:17] = 100_001
    contour = lit.ordered_contours_from_labels(labels)[100_001]
    assert contour.shape[1] == 2
    assert contour[:, 0].min() >= 9
    assert contour[:, 0].max() <= 15
    assert contour[:, 1].min() >= 11
    assert contour[:, 1].max() <= 17
