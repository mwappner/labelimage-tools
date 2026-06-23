from __future__ import annotations

import numpy as np


def validate_label_image(labels, *, background=0) -> np.ndarray:
    """
    Validate and return a 2-D integer label image as a NumPy array.

    Parameters
    ----------
    labels : array-like
        Candidate labeled image. The array must be two-dimensional and contain
        integer label values. Floating arrays are accepted only when every finite
        value is exactly integer-like, in which case they are cast to ``int64``.
    background : int, optional
        Label value used as background. The value is validated for sanity but the
        image is not required to contain it. Default is ``0``.

    Returns
    -------
    np.ndarray
        The validated label image. Integer inputs are returned as arrays without
        relabeling; integer-like floating inputs are returned as ``int64``.

    Raises
    ------
    ValueError
        If the input is not 2-D, contains non-integer values, or uses a non-finite
        background label.

    Notes
    -----
    Labels do not need to be consecutive. Values such as ``0, 5, 10`` are valid
    and are preserved.
    """
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
    """
    Return sorted unique labels from a 2-D label image.

    Parameters
    ----------
    labels : array-like
        Candidate labeled image accepted by :func:`validate_label_image`.
    background : int, optional
        Label value to treat as background. Default is ``0``.
    include_background : bool, optional
        If ``False`` (default), remove ``background`` from the returned values.
        If ``True``, include it when present.

    Returns
    -------
    np.ndarray
        Sorted unique label values. The original integer label values are
        preserved; no consecutiveness is assumed.
    """
    array = validate_label_image(labels, background=background)
    values = np.unique(array)
    if not include_background:
        values = values[values != background]
    return values
