from __future__ import annotations

from os import PathLike

import numpy as np
from PIL import Image


def load_img(path: str | PathLike) -> np.ndarray:
    """
    Load a labeled image from disk as a NumPy array.

    Parameters
    ----------
    path : str or os.PathLike
        Path to an image file readable by Pillow.

    Returns
    -------
    np.ndarray
        Image contents as stored in the file. Integer label values are preserved;
        no normalization, rescaling, or relabeling is applied.

    Notes
    -----
    This is the PIL-based loader extracted from
    ``segmentation_processing.img_treatment.load_img``. It is intentionally
    lightweight so callers can decide how to validate or preprocess the array.
    """
    with Image.open(path) as image:
        return np.asarray(image)


def load_label_image(path: str | PathLike) -> np.ndarray:
    """
    Load a labeled image from disk.

    This is a clearer public alias for :func:`load_img`. It has the same
    behavior: preserve integer labels and avoid unnecessary image normalization.
    """
    return load_img(path)
