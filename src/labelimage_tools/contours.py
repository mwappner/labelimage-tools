from __future__ import annotations

import numpy as np
from skimage import measure

from .validation import unique_labels, validate_label_image


def ordered_contour_from_mask(mask) -> np.ndarray:
    """Return the longest ordered contour of a boolean mask in ``(y, x)`` coordinates."""
    mask = np.asarray(mask, dtype=bool)
    contours = measure.find_contours(mask.astype(float), 0.5)
    if not contours:
        return np.empty((0, 2), dtype=float)
    return max(contours, key=len).astype(float)


def ordered_contours_from_labels(labels, *, background=0) -> dict[int, np.ndarray]:
    """Return longest ordered contour for each non-background label."""
    labels = validate_label_image(labels, background=background)
    return {
        int(label): ordered_contour_from_mask(labels == label)
        for label in unique_labels(labels, background=background)
    }
