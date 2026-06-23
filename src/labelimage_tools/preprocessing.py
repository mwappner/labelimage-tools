from __future__ import annotations

from os import PathLike

import numpy as np
from scipy import ndimage as ndi

from .io import load_img
from .validation import unique_labels, validate_label_image


def _structure(structure=None) -> np.ndarray:
    if structure is None:
        return np.ones((3, 3), dtype=bool)
    if isinstance(structure, int):
        return np.ones((structure, structure), dtype=bool)
    return np.asarray(structure, dtype=bool)


def _bbox_for_label(labels: np.ndarray, label: int, pad_y: int = 0, pad_x: int = 0):
    coords = np.argwhere(labels == label)
    if coords.size == 0:
        return None
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return (
        slice(max(0, y0 - pad_y), min(labels.shape[0], y1 + pad_y)),
        slice(max(0, x0 - pad_x), min(labels.shape[1], x1 + pad_x)),
    )


def erode_labels(im, structure=None, background=0) -> np.ndarray:
    """Erode each non-background label independently."""
    labels = validate_label_image(im, background=background)
    structure = _structure(structure)
    out = np.full(labels.shape, background, dtype=labels.dtype)
    for label in unique_labels(labels, background=background):
        slc = _bbox_for_label(labels, int(label))
        if slc is None:
            continue
        sub = labels[slc] == label
        out[slc][ndi.binary_erosion(sub, structure=structure)] = label
    return out


def dilate_labels(im, structure=None, background=0, background_only: bool = True) -> np.ndarray:
    """Dilate each label independently.

    If ``background_only`` is true, labels expand only into background pixels. Otherwise labels
    may overwrite each other in ascending label order, matching the behavior of the source
    implementation.
    """
    labels = validate_label_image(im, background=background)
    structure = _structure(structure)
    rowpad, colpad = structure.shape[0] // 2, structure.shape[1] // 2
    out = labels.copy()
    bg_mask = labels == background
    for label in unique_labels(labels, background=background):
        slc = _bbox_for_label(labels, int(label), rowpad, colpad)
        if slc is None:
            continue
        sub = labels[slc] == label
        dilated = ndi.binary_dilation(sub, structure=structure)
        if background_only:
            out[slc][dilated & bg_mask[slc]] = label
        else:
            out[slc][dilated] = label
    return out


def dialate_labels(im, structure=None, background=0, background_only: bool = True) -> np.ndarray:
    """Deprecated spelling kept as a behavior-preserving alias for ``dilate_labels``."""
    return dilate_labels(im, structure=structure, background=background, background_only=background_only)


def shuffle_labels(im, seed=None, background=None) -> np.ndarray:
    """Randomly permute label values, optionally preserving the background label."""
    labels = validate_label_image(im, background=background if background is not None else 0)
    rng = np.random.default_rng(seed)
    values = np.unique(labels)
    to_shuffle = values if background is None else values[values != background]
    shuffled = to_shuffle.copy()
    rng.shuffle(shuffled)
    mapping = {old: new for old, new in zip(to_shuffle, shuffled, strict=True)}
    if background is not None:
        mapping[background] = background
    return np.vectorize(mapping.get, otypes=[labels.dtype])(labels)


def fill_internal_gaps_edt(
    labels,
    background=0,
    max_distance=None,
    fill_value: int = 10_000,
) -> np.ndarray:
    """Fill internal background holes with nearest labels using Euclidean distance.

    Internal gaps are connected background components fully enclosed by foreground. If
    ``max_distance`` is provided, far pixels inside a hole receive sentinel labels starting at
    ``fill_value`` instead of a neighboring real label.
    """
    labels = validate_label_image(labels, background=background)
    fg = labels != background
    out = labels.copy()
    max_label = int(labels.max()) if labels.size else 0
    if fill_value <= max_label:
        raise ValueError("fill_value must be larger than all existing labels")

    filled_fg = ndi.binary_fill_holes(fg)
    holes = filled_fg & (~fg)
    if not np.any(holes):
        return labels.copy()

    distances, inds = ndi.distance_transform_edt(~fg, return_distances=True, return_indices=True)
    assign_all_bg = labels[tuple(inds)]
    cc, n_cc = ndi.label(holes)
    if n_cc == 0:
        return labels.copy()
    if max_distance is None:
        out[holes] = assign_all_bg[holes]
        return out

    for idx, slc in enumerate(ndi.find_objects(cc), start=1):
        if slc is None:
            continue
        hole_mask = cc[slc] == idx
        sub_dist = distances[slc]
        sub_assign = assign_all_bg[slc]
        far = hole_mask & (sub_dist > max_distance)
        close = hole_mask & ~far
        if np.any(close):
            out[slc][close] = sub_assign[close]
        if np.any(far):
            out[slc][far] = fill_value
            fill_value += 1
    return out


def skeletonize_dilate(labels, background=0) -> np.ndarray:
    """Return one-pixel exterior borders labeled by the adjacent object label."""
    labels = validate_label_image(labels, background=background)
    out = np.full(labels.shape, background, dtype=labels.dtype)
    struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    for label in unique_labels(labels, background=background):
        slc = _bbox_for_label(labels, int(label), 1, 1)
        if slc is None:
            continue
        sub = labels[slc] == label
        border = ndi.binary_dilation(sub, structure=struct) & (~sub)
        out[slc][border] = label
    return out


def skeletonize_erode(labels, background=0) -> np.ndarray:
    """Return one-pixel interior borders labeled by the owning object label."""
    labels = validate_label_image(labels, background=background)
    out = np.full(labels.shape, background, dtype=labels.dtype)
    struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    for label in unique_labels(labels, background=background):
        slc = _bbox_for_label(labels, int(label))
        if slc is None:
            continue
        sub = labels[slc] == label
        border = (~ndi.binary_erosion(sub, structure=struct)) & sub
        out[slc][border] = label
    return out


def skeletonize_labels(labels, background=0, kind: str = "interior") -> np.ndarray:
    """Skeletonize labels with ``kind='interior'`` or ``kind='exterior'``."""
    if kind == "interior":
        return skeletonize_erode(labels, background=background)
    if kind == "exterior":
        return skeletonize_dilate(labels, background=background)
    raise ValueError("kind must be 'interior' or 'exterior'")


def find_non_self_connected_labels(im, background=0, connectivity: int = 1) -> dict[int, np.ndarray]:
    """Return labels with multiple disconnected components and their component centroids."""
    labels = validate_label_image(im, background=background)
    structure = ndi.generate_binary_structure(labels.ndim, connectivity)
    bad = {}
    for label in unique_labels(labels, background=background):
        slc = _bbox_for_label(labels, int(label), 1, 1)
        if slc is None:
            continue
        sub = labels[slc] == label
        cc_labels, n_cc = ndi.label(sub, structure=structure)
        if n_cc > 1:
            centers = ndi.center_of_mass(sub, cc_labels, index=range(1, n_cc + 1))
            bad[int(label)] = np.asarray(
                [(cy + slc[0].start, cx + slc[1].start) for cy, cx in centers], dtype=float
            )
    return bad


def remove_non_self_connected_bits(im, background=0, connectivity: int = 1) -> np.ndarray:
    """Keep the largest connected component of each label and set smaller bits to background."""
    labels = validate_label_image(im, background=background)
    structure = ndi.generate_binary_structure(labels.ndim, connectivity)
    cleaned = labels.copy()
    for label in unique_labels(labels, background=background):
        slc = _bbox_for_label(labels, int(label), 1, 1)
        if slc is None:
            continue
        sub = labels[slc] == label
        cc_labels, n_cc = ndi.label(sub, structure=structure)
        if n_cc > 1:
            sizes = ndi.sum(sub, cc_labels, index=range(1, n_cc + 1))
            largest = int(np.argmax(sizes)) + 1
            cleaned[slc][(cc_labels != largest) & (cc_labels != 0)] = background
    return cleaned


def crop_to_foreground_bbox(im, background=0, padding: int = 20) -> tuple[np.ndarray, tuple[slice, slice]]:
    """Crop to non-background foreground plus padding and return ``(cropped, slices)``."""
    labels = validate_label_image(im, background=background)
    rows = np.any(labels != background, axis=1)
    cols = np.any(labels != background, axis=0)
    if not np.any(rows) or not np.any(cols):
        return labels, (slice(0, labels.shape[0]), slice(0, labels.shape[1]))
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    srow = slice(max(0, rmin - padding), min(labels.shape[0], rmax + padding + 1))
    scol = slice(max(0, cmin - padding), min(labels.shape[1], cmax + padding + 1))
    return labels[srow, scol], (srow, scol)


def load_image_pipeline(
    path: str | PathLike,
    seed=None,
    background=0,
    connectivity: int = 1,
    crop_to_foreground: bool = True,
    remove_small_bits: bool = True,
    fill_holes: bool = True,
    dilate_borders: bool = False,
    shuffle: bool = False,
) -> np.ndarray:
    """Load and optionally crop, clean, hole-fill, dilate, and shuffle a label image."""
    im = load_img(path)
    if crop_to_foreground:
        im, _ = crop_to_foreground_bbox(im, background=background, padding=5)
    if remove_small_bits:
        im = remove_non_self_connected_bits(im, background=background, connectivity=connectivity)
    if fill_holes:
        im = fill_internal_gaps_edt(im, background=background, max_distance=3)
    if dilate_borders:
        im = dilate_labels(im, background=background)
        im = fill_internal_gaps_edt(im, background=background, max_distance=3)
    if shuffle:
        im = shuffle_labels(im, seed=seed, background=background)
    return im
