from __future__ import annotations

import labelimage_tools as lit


def test_ordered_contour_from_mask_and_labels(nonconsecutive_labels):
    contour = lit.ordered_contour_from_mask(nonconsecutive_labels == 5)
    assert contour.shape[1] == 2
    contours = lit.ordered_contours_from_labels(nonconsecutive_labels)
    assert set(contours) == {5, 10}
    assert all(value.shape[1] == 2 for value in contours.values())
