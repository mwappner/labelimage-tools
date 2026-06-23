from __future__ import annotations

import numpy as np

import labelimage_tools as lit


def test_load_label_image_preserves_integer_labels(sample_path):
    labels = lit.load_label_image(sample_path)
    assert labels.ndim == 2
    assert np.issubdtype(labels.dtype, np.integer)
    assert labels.max() > 0
    assert np.array_equal(labels, lit.load_img(sample_path))
