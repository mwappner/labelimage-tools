"""Graph-coloring and adjacency plotting example.

Run this file from the repository root with:

    python examples/01_graph_coloring.py

The script writes PNG files to ``examples/plots``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

import labelimage_tools as lit  # noqa: E402

SAMPLE_PATH = REPOSITORY / "samples" / "test_cells2D.tif"
PLOT_DIR = REPOSITORY / "examples" / "plots"
BACKGROUND = 0
SEED = 4
K = 8
DPI = 150


def save_figure(fig, filename: str) -> Path:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOT_DIR / filename
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> list[Path]:
    labels = lit.load_label_image(SAMPLE_PATH)
    labels, _ = lit.crop_to_foreground_bbox(labels, background=BACKGROUND, padding=10)

    fig, ax = lit.plot_label_image(
        labels,
        background=BACKGROUND,
        use_graph_coloring=True,
        K=K,
        seed=SEED,
        cmap="managua",
        cyclic_cmap=True,
        title="Graph-colored label image",
    )
    graph_colored = save_figure(fig, "graph_colored_managua.png")

    fig, ax = lit.plot_label_image(
        labels,
        background=BACKGROUND,
        use_graph_coloring=True,
        K=K,
        seed=SEED,
        cmap="managua",
        cyclic_cmap=True,
        title="Label boundaries",
    )
    lit.plot_label_boundaries(labels, ax=ax, color="black", linewidth=0.7)
    boundaries = save_figure(fig, "label_boundaries.png")

    neighbors, contacts = lit.adjacency_with_contact_from_labels(labels, background=BACKGROUND)
    fig, ax = lit.plot_label_image(
        labels,
        background=BACKGROUND,
        use_graph_coloring=True,
        K=K,
        seed=SEED,
        cmap="managua",
        cyclic_cmap=True,
        title="Adjacency graph over labels",
    )
    lit.draw_graph(
        labels,
        neighbors,
        contacts=contacts,
        ax=ax,
        show_labels=False,
        lw_scaling=("sqrt", 0.15),
        line_args={"color": "white", "alpha": 0.75},
    )
    adjacency_graph = save_figure(fig, "adjacency_graph.png")

    print(f"Loaded {SAMPLE_PATH}")
    print(f"Labels: {len(lit.unique_labels(labels, background=BACKGROUND))}")
    print(f"Adjacency edges: {len(lit.adjacency_pairs_from_labels(labels, background=BACKGROUND))}")
    print("Wrote:")
    for path in [graph_colored, boundaries, adjacency_graph]:
        print(f"  {path}")
    return [graph_colored, boundaries, adjacency_graph]


if __name__ == "__main__":
    main()
