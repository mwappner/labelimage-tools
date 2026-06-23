from __future__ import annotations

import numpy as np
import pytest

import labelimage_tools as lit


def test_validate_label_image_accepts_2d_integer_arrays():
    labels = np.array([[0, 5], [10, 5]])
    assert np.array_equal(lit.validate_label_image(labels), labels)


def test_validate_label_image_rejects_non_2d():
    with pytest.raises(ValueError, match="2-D"):
        lit.validate_label_image(np.zeros((2, 2, 2), dtype=int))


def test_validate_label_image_rejects_non_integer_floats():
    with pytest.raises(ValueError, match="integers"):
        lit.validate_label_image(np.array([[0.0, 1.5]]))


def test_unique_labels_respects_background():
    labels = np.array([[0, 5, 10]])
    assert lit.unique_labels(labels).tolist() == [5, 10]
    assert lit.unique_labels(labels, include_background=True).tolist() == [0, 5, 10]
