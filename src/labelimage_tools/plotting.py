from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from skimage.segmentation import find_boundaries

from .adjacency import adjacency_from_labels, get_centroids
from .coloring import show_map_with_colors
from .contours import ordered_contours_from_labels
from .junctions import Junction, junctions_from_labels
from .typing import Cont, Neig


def _fig_ax(ax: Axes | None):
    if ax is None:
        return plt.subplots()
    return ax.figure, ax


def draw_graph(
    im,
    neighbors: Neig,
    contacts: Cont | None = None,
    ax: Axes | None = None,
    show_labels: bool = True,
    show_centroids: bool = False,
    centroids: Mapping[int, np.ndarray] | None = None,
    autoadjust_ax: bool = False,
    lw_scaling: tuple[str, Any] = ("sqrt", 0.5),
    line_args=None,
    text_args=None,
    dot_args=None,
) -> tuple[LineCollection, Axes]:
    """Draw an adjacency/contact graph between label centroids."""
    line_args = {"color": "w", **(line_args or {})}
    text_args = {"color": "k", "ha": "center", "va": "center", **(text_args or {})}
    dot_args = {"color": "white", "markersize": 10, "marker": "o", **(dot_args or {})}
    if im is not None:
        centroid_dict = get_centroids(im)
    elif centroids is not None:
        centroid_dict = centroids
    else:
        raise ValueError("Either im or centroids must be provided")
    _, ax = _fig_ax(ax)
    kind, factor = lw_scaling
    if kind == "sqrt":
        scaler = lambda value: np.sqrt(value) * factor
    elif kind == "linear":
        scaler = lambda value: value * factor
    else:
        raise ValueError(f"unknown line width scaling kind: {kind}")

    lines = []
    widths = []
    for label, nbrs in neighbors.items():
        cy, cx = centroid_dict[int(label)]
        weights = contacts[int(label)] if contacts is not None else np.ones_like(nbrs)
        for nbr, weight in zip(nbrs, weights, strict=True):
            ny, nx = centroid_dict[int(nbr)]
            lines.append([(cx, cy), (nx, ny)])
            widths.append(scaler(weight))
        if show_labels:
            ax.text(cx, cy, str(label), **text_args)
        if show_centroids:
            ax.plot(cx, cy, **dot_args)
    collection = LineCollection(lines, linewidths=widths, **line_args)
    ax.add_collection(collection)
    if autoadjust_ax:
        ax.autoscale()
        ax.set_ylim(ax.get_ylim()[::-1])
        ax.set_aspect("equal")
    return collection, ax


def label_map(im, ax: Axes | None = None, background=0, **text_args) -> Axes:
    """Annotate each non-background label at its centroid."""
    _, ax = _fig_ax(ax)
    text_args = {"color": "k", "ha": "center", "va": "center", **text_args}
    for label, (cy, cx) in get_centroids(im, background=background).items():
        ax.text(cx, cy, str(label), **text_args)
    return ax


def plot_label_image(
    labels,
    *,
    ax: Axes | None = None,
    background=0,
    use_graph_coloring: bool = True,
    K: int = 8,
    seed: int | None = None,
    title: str | None = None,
    show_colorbar: bool = False,
    interpolation: str = "nearest",
    **imshow_kwargs,
):
    """Plot a label image, using graph coloring by default."""
    fig, ax = _fig_ax(ax)
    if use_graph_coloring:
        image, _, ax = show_map_with_colors(
            labels,
            ax=ax,
            K=K,
            seed=seed,
            interpolation=interpolation,
            **imshow_kwargs,
        )
    else:
        image = ax.imshow(labels, interpolation=interpolation, **imshow_kwargs)
    if show_colorbar:
        fig.colorbar(image, ax=ax)
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    return fig, ax


def plot_label_boundaries(
    labels,
    *,
    ax: Axes | None = None,
    background=0,
    color="white",
    linewidth: float = 0.5,
    title: str | None = None,
):
    """Plot boundaries between labels over the current axes."""
    fig, ax = _fig_ax(ax)
    boundaries = find_boundaries(np.asarray(labels), mode="thick")
    ys, xs = np.nonzero(boundaries)
    ax.plot(xs, ys, ".", color=color, markersize=linewidth)
    if title is not None:
        ax.set_title(title)
    ax.set_aspect("equal")
    return fig, ax


def plot_contours(
    labels,
    *,
    ax: Axes | None = None,
    background=0,
    color="white",
    linewidth: float = 0.8,
    title: str | None = None,
):
    """Plot ordered contours for each non-background label."""
    fig, ax = _fig_ax(ax)
    for contour in ordered_contours_from_labels(labels, background=background).values():
        if len(contour):
            ax.plot(contour[:, 1], contour[:, 0], color=color, linewidth=linewidth)
    if title is not None:
        ax.set_title(title)
    ax.set_aspect("equal")
    return fig, ax


def plot_junctions(
    labels=None,
    junctions: list[Junction] | None = None,
    *,
    junction_mask=None,
    ax: Axes | None = None,
    background=0,
    show_junction_ids: bool = False,
    title: str | None = None,
):
    """Plot junction coordinates over an optional label image."""
    fig, ax = _fig_ax(ax)
    if labels is not None:
        ax.imshow(labels, cmap="gray", interpolation="nearest")
    if junctions is None:
        if labels is None and junction_mask is None:
            raise ValueError("provide labels, junctions, or junction_mask")
        if labels is not None:
            _, junctions = junctions_from_labels(labels, background=background)
        else:
            coords = np.argwhere(junction_mask)
            junctions = [
                Junction(1, coords.mean(axis=0), coords.astype(np.int64), frozenset())
            ] if len(coords) else []
    if junction_mask is not None:
        ys, xs = np.nonzero(junction_mask)
        ax.plot(xs, ys, ".", color="yellow", markersize=1)
    for junction in junctions:
        y, x = junction.yx
        ax.plot(x, y, "o", color="red", markersize=4)
        if show_junction_ids:
            ax.text(x, y, str(junction.id), color="white", ha="center", va="center")
    if title is not None:
        ax.set_title(title)
    ax.set_aspect("equal")
    return fig, ax


def plot_adjacency_graph(labels, *, ax: Axes | None = None, background=0, eight: bool = True):
    """Convenience wrapper: show label image and overlay its adjacency graph."""
    fig, ax = plot_label_image(labels, ax=ax, background=background)
    neighbors = adjacency_from_labels(labels, background=background, eight=eight)
    draw_graph(labels, neighbors, ax=ax)
    return fig, ax
