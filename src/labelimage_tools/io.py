from __future__ import annotations

from os import PathLike

import numpy as np
from PIL import Image


def load_img(path: str | PathLike) -> np.ndarray:
    """Load an image file into a NumPy array, preserving stored label values."""
    with Image.open(path) as image:
        return np.asarray(image)


def load_label_image(path: str | PathLike) -> np.ndarray:
    """Clearer alias for :func:`load_img`."""
    return load_img(path)
