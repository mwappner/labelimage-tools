from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, Normalize
from matplotlib.image import AxesImage

from .adjacency import adjacency_from_labels
from .typing import Adj, Node


def dsatur_color(adj: Adj, seed: int | None = None) -> dict[Node, int]:
    """Color an adjacency graph with NetworkX's DSATUR greedy heuristic."""
    graph = nx.Graph()
    for node, neighbors in adj.items():
        graph.add_node(node)
        for neighbor in neighbors:
            if node != neighbor:
                graph.add_edge(node, neighbor)
    return nx.algorithms.coloring.greedy_color(graph, strategy="saturation_largest_first")


def refine_to_K_colors(
    base_color: Mapping[Node, int],
    K: int,
    seed: int | None = None,
    balance: str = "proportional",
) -> dict[Node, int]:
    """Split independent color classes to use exactly ``K`` colors."""
    rng = np.random.default_rng(seed)
    classes: dict[int, list[Node]] = defaultdict(list)
    for node, color in base_color.items():
        classes[int(color)].append(node)
    c_count = len(classes)
    if K < c_count:
        raise ValueError(f"Requested K={K} < base colors {c_count}")
    sizes = np.array([len(classes[c]) for c in range(c_count)], dtype=float)
    if balance == "proportional" and sizes.sum() > 0:
        target = K * (sizes / sizes.sum())
        variants = np.floor(target).astype(int)
        variants = np.maximum(variants, 1)
        while variants.sum() < K:
            variants[int(np.argmax(target - variants))] += 1
        while variants.sum() > K:
            variants[int(np.argmax(variants))] -= 1
    else:
        q, r = divmod(K, c_count)
        variants = np.array([q + (1 if i < r else 0) for i in range(c_count)], dtype=int)

    refined = {}
    next_idx = 0
    for base in range(c_count):
        palette = list(range(next_idx, next_idx + int(variants[base])))
        next_idx += int(variants[base])
        nodes = list(classes[base])
        rng.shuffle(nodes)
        for j, node in enumerate(nodes):
            refined[node] = palette[j % len(palette)]
    if refined and len(set(refined.values())) != K:
        raise RuntimeError("could not refine coloring to exactly K colors")
    return refined


def rebalance_K_colors(
    adj: Adj,
    color: Mapping[Node, int],
    K: int,
    *,
    seed: int | None = None,
    max_rounds: int = 10,
    tolerance: float = 0.1,
    protect_singletons: bool = True,
) -> dict[Node, int]:
    """Heuristically balance K color class sizes without creating conflicts."""
    if not color:
        return dict(color)
    rng = np.random.default_rng(seed)
    new = dict(color)
    ideal = len(new) / float(K)
    max_diff = max(1, int(np.ceil(tolerance * ideal)))
    for _ in range(max_rounds):
        counts = np.bincount(np.fromiter(new.values(), dtype=int), minlength=K)
        smallest = int(np.argmin(counts))
        largest = int(np.argmax(counts))
        if counts[largest] - counts[smallest] <= max_diff:
            break
        candidates = [node for node, col in new.items() if col == largest]
        rng.shuffle(candidates)
        moved = False
        for node in candidates:
            if protect_singletons and counts[largest] <= 1:
                break
            if any(new.get(neighbor) == smallest for neighbor in adj.get(node, ())):
                continue
            new[node] = smallest
            counts[largest] -= 1
            counts[smallest] += 1
            moved = True
            break
        if not moved:
            break
    return new


def apply_color_lut_int(
    map_array: np.ndarray,
    color_mapping: dict[Node, int] | None = None,
    lut: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Map integer labels to color indices, returning NaN for unmapped labels."""
    array = np.asarray(map_array)
    if array.size == 0:
        return np.empty_like(array, dtype=np.float32), np.empty((0,), dtype=np.float32)
    if np.any(array < 0):
        raise ValueError("apply_color_lut_int expects non-negative labels")
    if lut is None:
        if color_mapping is None:
            raise ValueError("Either color_mapping or lut must be provided")
        max_label = int(max(int(array.max()), *(int(label) for label in color_mapping)))
        lut = np.full(max_label + 1, np.nan, dtype=np.float32)
        for label, color in color_mapping.items():
            if int(label) >= 0:
                lut[int(label)] = float(color)
    out = np.full(array.shape, np.nan, dtype=np.float32)
    in_range = array <= (len(lut) - 1)
    out[in_range] = lut[array[in_range]]
    return out, lut


def color_planar_with_variety(
    adj: Adj,
    K: int = 8,
    seed: int | None = None,
    balance: str = "proportional",
    rebalance: bool = True,
) -> dict[Node, int]:
    """Produce a conflict-free coloring that uses exactly K colors when feasible."""
    base = dsatur_color(adj, seed=seed)
    if K > len(base):
        raise ValueError(f"Requested K={K} colors for only {len(base)} labels")
    used = len(set(base.values()))
    if used == K:
        colored = dict(base)
    elif used < K:
        colored = refine_to_K_colors(base, K, seed=seed, balance=balance)
    else:
        raise ValueError(f"Base coloring used {used} colors; choose K >= {used}")
    if rebalance:
        colored = rebalance_K_colors(adj, colored, K, seed=seed)
    return colored


def show_map_with_colors(
    map_array: np.ndarray,
    ax: Axes | None = None,
    cmap: str | Colormap = "tab20",
    cyclic_cmap: bool = False,
    adj: Adj | None = None,
    lut: np.ndarray | None = None,
    K: int = 8,
    seed: int | None = None,
    balance: str = "proportional",
    rebalance: bool = True,
    holes_separate: bool = True,
    hole_color: Any = "0.3",
    **imshow_kwargs,
) -> tuple[AxesImage, np.ndarray, Axes]:
    """Display a label image with graph-based colors for adjacent labels."""
    if ax is None:
        _, ax = plt.subplots()
    if lut is None:
        if adj is None:
            adj = adjacency_from_labels(map_array, background=0)
        K_effective = min(K, max(1, len(adj)))
        color_mapping = color_planar_with_variety(
            adj, K=K_effective, seed=seed, balance=balance, rebalance=rebalance
        )
    else:
        color_mapping = None
        K_effective = K
    mapped, lut = apply_color_lut_int(map_array, color_mapping, lut=lut)
    if holes_separate:
        cmap = plt.get_cmap(cmap).copy()
        cmap.set_over(hole_color)
        mapped[np.asarray(map_array) > 9999] = K + 1
    norm = Normalize(vmin=0, vmax=(K_effective if cyclic_cmap else K_effective - 1))
    image = ax.imshow(mapped, cmap=cmap, norm=norm, **imshow_kwargs)
    return image, lut, ax
