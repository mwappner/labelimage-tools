from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from .validation import unique_labels


def _clip_padded_slice(
    slc: tuple[slice, slice],
    shape: tuple[int, ...],
    padding: int,
) -> tuple[slice, slice]:
    """Pad a 2-D slice tuple, clipping the result to image bounds."""
    return (
        slice(max(0, slc[0].start - padding), min(shape[0], slc[0].stop + padding)),
        slice(max(0, slc[1].start - padding), min(shape[1], slc[1].stop + padding)),
    )


def _slice_from_coords(coords: np.ndarray) -> tuple[slice, slice]:
    """Build the tight 2-D bounding-box slice around coordinate rows."""
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1
    return (slice(int(mins[0]), int(maxs[0])), slice(int(mins[1]), int(maxs[1])))


def _compact_problematic_labels(
    labels: np.ndarray,
    problematic: list[int],
) -> tuple[np.ndarray, list[int]]:
    """
    Remap sparse/problematic labels to dense positive ids in one image pass.

    ``ndi.find_objects`` is fast for dense positive labels, but cannot directly
    handle negative labels or very large sparse labels without creating a huge
    list. This helper creates a compact temporary image whose values are
    ``1..M`` only where the original image contains one of the problematic
    labels. The output order is sorted and mapped back by the caller.
    """
    sorted_labels = sorted(problematic)
    lookup = np.asarray(sorted_labels, dtype=labels.dtype)
    compact = np.zeros(labels.shape, dtype=np.int32)

    indices = np.searchsorted(lookup, labels)
    in_range = indices < lookup.size
    matches = np.zeros(labels.shape, dtype=bool)
    matches[in_range] = lookup[indices[in_range]] == labels[in_range]
    compact[matches] = indices[matches] + 1
    return compact, sorted_labels


def label_slices(
    labels,
    *,
    background=0,
    include_background: bool = False,
    padding: int = 0,
    max_direct_label: int = 100_000,
    max_manual_labels: int = 100,
) -> dict[int, tuple[slice, slice]]:
    """
    Return global bounding-box slices for labels in a 2-D label image.

    Parameters
    ----------
    labels : np.ndarray
        2-D integer label image.
    background : int, optional
        Background label value. Default is ``0``.
    include_background : bool, optional
        If ``False`` (default), the background label is excluded. If ``True``,
        the background is included when present in the image.
    padding : int, optional
        Number of pixels to add around each bounding box. Padding is clipped to
        the image boundary.
    max_direct_label : int, optional
        Largest positive label handled by the direct ``ndi.find_objects`` fast
        path. Larger labels are handled by the sparse-label fallback.
    max_manual_labels : int, optional
        Maximum number of sparse/problematic labels to handle with bounded
        per-label scans. When there are more, labels are compacted in one pass
        and processed with ``ndi.find_objects``.

    Returns
    -------
    dict[int, tuple[slice, slice]]
        Mapping from original label value to global ``(row_slice, col_slice)``.

    Notes
    -----
    Ordinary positive labels use ``scipy.ndimage.find_objects`` directly. This
    preserves the windowed design of the original tools, where expensive
    per-label work happens only inside local crops. Negative labels, zero-valued
    foreground labels, and very large sparse labels are still supported without
    requiring a label-indexed list of length ``max_label + 1``.
    """
    labels = np.asarray(labels)
    padding = int(padding)
    max_direct_label = int(max_direct_label)
    max_manual_labels = int(max_manual_labels)
    if labels.ndim != 2:
        raise ValueError("labels must be a 2-D array")
    if padding < 0:
        raise ValueError("padding must be non-negative")
    if max_direct_label < 1:
        raise ValueError("max_direct_label must be positive")
    if max_manual_labels < 0:
        raise ValueError("max_manual_labels must be non-negative")

    values = [
        int(label)
        for label in unique_labels(
            labels,
            background=background,
            include_background=include_background,
        )
    ]
    if not values:
        return {}

    shape = labels.shape
    slices: dict[int, tuple[slice, slice]] = {}

    direct = [label for label in values if 0 < label <= max_direct_label]
    problematic = [label for label in values if label <= 0 or label > max_direct_label]

    if direct:
        objects = ndi.find_objects(labels, max_label=max_direct_label)
        for label in direct:
            slc = objects[label - 1]
            if slc is not None:
                slices[label] = _clip_padded_slice(slc, shape, padding)

    if len(problematic) <= max_manual_labels:
        for label in problematic:
            coords = np.argwhere(labels == label)
            if coords.size:
                slices[label] = _clip_padded_slice(_slice_from_coords(coords), shape, padding)
    elif problematic:
        compact, original_labels = _compact_problematic_labels(labels, problematic)
        objects = ndi.find_objects(compact, max_label=len(original_labels))
        for compact_id, label in enumerate(original_labels, start=1):
            slc = objects[compact_id - 1]
            if slc is not None:
                slices[label] = _clip_padded_slice(slc, shape, padding)

    return {label: slices[label] for label in values if label in slices}
