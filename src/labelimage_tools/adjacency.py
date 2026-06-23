from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from .typing import Cont, Neig, Node
from .validation import unique_labels, validate_label_image


def adjacency_with_unique_from_labels(
    im,
    background=0,
    eight: bool = True,
    allow_background_contacts: bool = False,
) -> tuple[Neig, np.ndarray]:
    """Return neighbor mapping and unique undirected touching-label pairs."""
    labels = validate_label_image(im, background=background)
    h, w = labels.shape
    pairs_chunks = []
    empty = np.empty((0, 2), dtype=np.int64)

    def acc(a: np.ndarray, b: np.ndarray) -> None:
        mask = a != b if allow_background_contacts else (a != b) & (a != background) & (b != background)
        if np.any(mask):
            aa = a[mask].ravel()
            bb = b[mask].ravel()
            pairs_chunks.append(np.stack([np.minimum(aa, bb), np.maximum(aa, bb)], axis=1))

    if h == 0 or w == 0:
        return {}, empty
    acc(labels[:-1, :], labels[1:, :])
    acc(labels[:, :-1], labels[:, 1:])
    if eight and h > 1 and w > 1:
        acc(labels[:-1, :-1], labels[1:, 1:])
        acc(labels[:-1, 1:], labels[1:, :-1])
    if not pairs_chunks:
        return {}, empty

    pairs = np.concatenate(pairs_chunks, axis=0)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    uniq = np.unique(pairs.astype(np.int64), axis=0)
    adj: dict[int, list[int]] = {}
    for a, b in uniq:
        adj.setdefault(int(a), []).append(int(b))
        adj.setdefault(int(b), []).append(int(a))
    return {k: np.asarray(v, dtype=np.int64) for k, v in adj.items()}, uniq


def adjacency_from_labels(
    im,
    background=0,
    eight: bool = True,
    allow_background_contacts: bool = False,
) -> Neig:
    """Return adjacency mapping for touching labels."""
    neighbors, _ = adjacency_with_unique_from_labels(
        im,
        background=background,
        eight=eight,
        allow_background_contacts=allow_background_contacts,
    )
    return neighbors


def adjacency_pairs_from_labels(
    im,
    background=0,
    eight: bool = True,
    allow_background_contacts: bool = False,
) -> np.ndarray:
    """Return unique undirected touching-label pairs as an ``(n, 2)`` array."""
    _, pairs = adjacency_with_unique_from_labels(
        im,
        background=background,
        eight=eight,
        allow_background_contacts=allow_background_contacts,
    )
    return pairs


def adjacency_with_contact_from_labels(
    im,
    background=0,
    eight: bool = True,
    diag_weight: float | None = None,
    allow_background_contacts: bool = False,
) -> tuple[Neig, Cont]:
    """Return neighbors and pixel-neighborhood contact counts.

    Contact values count neighboring pixel pairs. They are useful weights but should not be
    interpreted as exact geometric contact lengths.
    """
    labels = validate_label_image(im, background=background)
    h, w = labels.shape
    if h == 0 or w == 0:
        return {}, {}
    totals: dict[tuple[int, int], float] = {}

    def acc(a: np.ndarray, b: np.ndarray, weight: float) -> None:
        mask = a != b if allow_background_contacts else (a != b) & (a != background) & (b != background)
        if not np.any(mask):
            return
        aa = a[mask].ravel()
        bb = b[mask].ravel()
        pairs = np.stack([np.minimum(aa, bb), np.maximum(aa, bb)], axis=1)
        uniq, counts = np.unique(pairs, axis=0, return_counts=True)
        for (pa, pb), count in zip(uniq, counts, strict=True):
            key = (int(pa), int(pb))
            totals[key] = totals.get(key, 0.0) + float(count) * weight

    acc(labels[:-1, :], labels[1:, :], 1.0)
    acc(labels[:, :-1], labels[:, 1:], 1.0)
    if eight and h > 1 and w > 1:
        wdiag = 1.0 if diag_weight is None else float(diag_weight)
        acc(labels[:-1, :-1], labels[1:, 1:], wdiag)
        acc(labels[:-1, 1:], labels[1:, :-1], wdiag)

    adj: dict[int, list[int]] = {}
    cont: dict[int, list[float]] = {}
    for (a, b), count in totals.items():
        adj.setdefault(a, []).append(b)
        cont.setdefault(a, []).append(count)
        adj.setdefault(b, []).append(a)
        cont.setdefault(b, []).append(count)
    return (
        {k: np.asarray(v, dtype=np.int64) for k, v in adj.items()},
        {k: np.asarray(cont[k]) for k in cont},
    )


def label_is_border(neighbors: Neig, label: Node, background: Node = 0) -> bool:
    """Return whether ``label`` has ``background`` among its neighbors."""
    return bool(background in neighbors.get(label, np.array([], dtype=np.int64)))


def border_labels(neighbors: Neig, background: Node = 0) -> np.ndarray:
    """Return labels that touch the background in an adjacency mapping."""
    return np.asarray(
        [int(label) for label in neighbors if label != background and label_is_border(neighbors, label, background)],
        dtype=np.int64,
    )


def get_centroids(im, background=0) -> dict[int, np.ndarray]:
    """Return label centroids in image coordinates ``(y, x)``."""
    labels = validate_label_image(im, background=background)
    values = unique_labels(labels, background=background)
    cms = ndi.center_of_mass(np.ones_like(labels), labels=labels, index=values)
    return {int(value): np.asarray(cm, dtype=float) for value, cm in zip(values, cms, strict=True)}


centroids_from_labels = get_centroids
