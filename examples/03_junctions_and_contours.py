"""Junction and contour extraction example.

The script detects clustered junctions, extracts ordered contours, overlays both
on the sample label image, and writes PNG files to ``examples/plots``.
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
    labels = lit.load_image_pipeline(SAMPLE_PATH)
    junction_label_image, junctions = lit.junctions_from_labels(
        labels,
        background=BACKGROUND,
        min_labels=3,
        connectivity=2,
    )
    contours = lit.ordered_contours_from_labels(labels, background=BACKGROUND)

    fig, ax = lit.plot_label_image(
        labels,
        background=BACKGROUND,
        use_graph_coloring=True,
        K=K,
        seed=SEED,
        cmap="managua",
        cyclic_cmap=True,
        title="Detected junctions",
    )
    lit.plot_junctions(junctions=junctions, junction_mask=junction_label_image > 0, ax=ax)
    junctions_path = save_figure(fig, "junctions.png")

    fig, ax = lit.plot_label_image(
        labels,
        background=BACKGROUND,
        use_graph_coloring=True,
        K=K,
        seed=SEED,
        cmap="managua",
        cyclic_cmap=True,
        title="Ordered contours",
    )
    lit.plot_contours(labels, ax=ax, background=BACKGROUND, color="black", linewidth=0.6)
    contours_path = save_figure(fig, "contours.png")

    fig, ax = lit.plot_label_image(
        labels,
        background=BACKGROUND,
        use_graph_coloring=True,
        K=K,
        seed=SEED,
        cmap="managua",
        cyclic_cmap=True,
        title="Contours and junctions",
    )
    lit.plot_contours(labels, ax=ax, background=BACKGROUND, color="black", linewidth=0.5)
    lit.plot_junctions(junctions=junctions, ax=ax)
    overlay_path = save_figure(fig, "junctions_contours_overlay.png")

    print(f"Detected junctions: {len(junctions)}")
    print(f"Extracted contours: {len(contours)}")
    print("Wrote:")
    for path in [junctions_path, contours_path, overlay_path]:
        print(f"  {path}")
    return [junctions_path, contours_path, overlay_path]


if __name__ == "__main__":
    main()
