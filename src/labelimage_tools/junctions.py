from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree
from skimage.measure import label as cc_label

from .validation import validate_label_image

try:  # pragma: no cover - fallback exists for environments without numba
    from numba import njit, prange
except Exception:  # pragma: no cover
    njit = None
    prange = range


@dataclass(frozen=True)
class Junction:
    """Clustered label-image junction in image coordinates."""

    id: int
    yx: np.ndarray
    pixel_coords: np.ndarray
    labels: frozenset[int]


def _python_junction_mask_core(padded: np.ndarray, h: int, w: int, min_labels: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    for y in range(h):
        yy = y + 1
        for x in range(w):
            if np.unique(padded[yy - 1 : yy + 2, x : x + 3]).size >= min_labels:
                mask[y, x] = 1
    return mask


if njit is not None:

    @njit(cache=True, parallel=True)
    def _numba_junction_mask_core(padded, h, w, min_labels):  # pragma: no cover
        mask = np.zeros((h, w), np.uint8)
        for y in prange(h):
            yy = y + 1
            for x in range(w):
                vals = np.empty(9, padded.dtype)
                nunique = 0
                done = False
                for dy in range(3):
                    if done:
                        break
                    for dx in range(3):
                        value = padded[yy - 1 + dy, x + dx]
                        seen = False
                        for k in range(nunique):
                            if vals[k] == value:
                                seen = True
                                break
                        if not seen:
                            vals[nunique] = value
                            nunique += 1
                            if nunique >= min_labels:
                                mask[y, x] = 1
                                done = True
                                break
        return mask

else:
    _numba_junction_mask_core = None


def junction_pixels_with_labels(
    labels,
    *,
    background=None,
    min_labels: int = 3,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Find pixels whose 3×3 neighborhood contains at least ``min_labels`` labels.

    If ``background`` is not ``None``, that label is excluded before counting. The default
    preserves the source behavior and lets background participate in junction detection.
    """
    labels = validate_label_image(labels, background=0 if background is None else background)
    h, w = labels.shape
    padded = np.pad(labels, 1, mode="edge")
    if background is None and _numba_junction_mask_core is not None and min_labels <= 9:
        mask = _numba_junction_mask_core(padded, h, w, int(min_labels)).astype(bool)
    else:
        mask = np.zeros((h, w), dtype=bool)
        for y in range(h):
            yy = y + 1
            for x in range(w):
                vals = np.unique(padded[yy - 1 : yy + 2, x : x + 3])
                if background is not None:
                    vals = vals[vals != background]
                if vals.size >= min_labels:
                    mask[y, x] = True

    labels_at_pixel: dict[int, np.ndarray] = {}
    ys, xs = np.nonzero(mask)
    for y, x in zip(ys, xs, strict=True):
        yy = y + 1
        vals = np.unique(padded[yy - 1 : yy + 2, x : x + 3])
        if background is not None:
            vals = vals[vals != background]
        if vals.size >= min_labels:
            labels_at_pixel[int(y) * w + int(x)] = vals.astype(np.int64)
    return mask, labels_at_pixel


def cluster_junctions_with_labels(
    junction_mask,
    labels_at_pixel: dict[int, np.ndarray],
    *,
    connectivity: int = 2,
    start_id: int = 1,
) -> tuple[np.ndarray, list[Junction]]:
    """Cluster junction pixels and union the label sets in each cluster."""
    mask = np.asarray(junction_mask, dtype=bool)
    if not np.any(mask):
        return np.zeros(mask.shape, dtype=np.int64), []
    component_labels = cc_label(mask, connectivity=connectivity)
    h, w = mask.shape
    junction_label_image = np.zeros(mask.shape, dtype=np.int64)
    junctions: list[Junction] = []
    for offset, component_id in enumerate(np.unique(component_labels[component_labels > 0])):
        jid = start_id + offset
        coords = np.argwhere(component_labels == component_id)
        junction_label_image[component_labels == component_id] = jid
        label_set: set[int] = set()
        for yy, xx in coords:
            vals = labels_at_pixel.get(int(yy) * w + int(xx))
            if vals is not None:
                label_set.update(int(value) for value in vals)
        junctions.append(
            Junction(
                id=jid,
                yx=coords.mean(axis=0).astype(float),
                pixel_coords=coords.astype(np.int64),
                labels=frozenset(label_set),
            )
        )
    return junction_label_image, junctions


def merge_close_junctions(
    junctions,
    *,
    epsilon: float,
    start_id: int = 1,
) -> list[Junction]:
    """Merge junctions within ``epsilon`` pixels by averaging coordinates and unioning labels."""
    junctions = list(junctions)
    if epsilon <= 0 or len(junctions) <= 1:
        return [
            Junction(start_id + i, np.asarray(j.yx, dtype=float), j.pixel_coords, frozenset(j.labels))
            for i, j in enumerate(junctions)
        ]
    positions = np.asarray([j.yx for j in junctions], dtype=float)
    tree = cKDTree(positions)
    pairs = tree.query_pairs(epsilon)
    parent = np.arange(len(junctions))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in pairs:
        union(a, b)
    groups: dict[int, list[int]] = {}
    for i in range(len(junctions)):
        groups.setdefault(find(i), []).append(i)
    merged = []
    for offset, group in enumerate(groups.values()):
        coords = np.vstack([junctions[i].pixel_coords for i in group])
        labs = frozenset().union(*(junctions[i].labels for i in group))
        merged.append(
            Junction(
                id=start_id + offset,
                yx=positions[group].mean(axis=0),
                pixel_coords=coords.astype(np.int64),
                labels=frozenset(int(v) for v in labs),
            )
        )
    return merged


def junctions_from_labels(
    labels,
    *,
    background=None,
    min_labels: int = 3,
    connectivity: int = 2,
    merge_epsilon: float = 0.0,
    start_id: int = 1,
) -> tuple[np.ndarray, list[Junction]]:
    """Detect, cluster, and optionally merge junctions in a label image."""
    mask, labels_at_pixel = junction_pixels_with_labels(
        labels, background=background, min_labels=min_labels
    )
    label_image, junctions = cluster_junctions_with_labels(
        mask, labels_at_pixel, connectivity=connectivity, start_id=start_id
    )
    if merge_epsilon > 0:
        junctions = merge_close_junctions(junctions, epsilon=merge_epsilon, start_id=start_id)
        label_image = np.zeros_like(label_image)
        for junction in junctions:
            label_image[tuple(junction.pixel_coords.T)] = junction.id
    return label_image, junctions
