from __future__ import annotations

import numpy as np


def validate_label_image(labels, *, background=0) -> np.ndarray:
    """Return ``labels`` as an array after validating it is a 2-D integer label image."""
    array = np.asarray(labels)
    if array.ndim != 2:
        raise ValueError(f"label image must be 2-D, got shape {array.shape}")
    if not np.issubdtype(array.dtype, np.integer):
        if np.issubdtype(array.dtype, np.floating) and np.all(np.isfinite(array)):
            rounded = np.rint(array)
            if np.allclose(array, rounded):
                array = rounded.astype(np.int64)
            else:
                raise ValueError("label image values must be integers")
        else:
            raise ValueError("label image values must be integers")
    if not np.isfinite(background):
        raise ValueError("background label must be finite")
    return array


def unique_labels(labels, *, background=0, include_background: bool = False) -> np.ndarray:
    """Return sorted unique labels, optionally excluding the background label."""
    array = validate_label_image(labels, background=background)
    values = np.unique(array)
    if not include_background:
        values = values[values != background]
    return values
