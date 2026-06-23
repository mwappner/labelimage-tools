from __future__ import annotations

import matplotlib.pyplot as plt

import labelimage_tools as lit


def test_plot_label_image_and_boundaries(simple_labels):
    fig, ax = lit.plot_label_image(simple_labels, K=4, seed=1)
    assert ax.figure is fig
    fig2, ax2 = lit.plot_label_boundaries(simple_labels, ax=ax)
    assert fig2 is fig
    assert ax2 is ax
    plt.close(fig)


def test_plot_junctions_and_contours(simple_labels):
    _, junctions = lit.junctions_from_labels(simple_labels)
    fig, ax = lit.plot_junctions(simple_labels, junctions=junctions)
    assert ax.figure is fig
    fig2, ax2 = lit.plot_contours(simple_labels, ax=ax)
    assert fig2 is fig
    assert ax2 is ax
    plt.close(fig)


def test_draw_graph_and_label_map(simple_labels):
    neighbors, contacts = lit.adjacency_with_contact_from_labels(simple_labels)
    collection, ax = lit.draw_graph(simple_labels, neighbors, contacts=contacts)
    assert collection.axes is ax
    assert lit.label_map(simple_labels, ax=ax) is ax
    plt.close(ax.figure)


def test_plot_adjacency_graph(simple_labels):
    fig, ax = lit.plot_adjacency_graph(simple_labels)
    assert ax.figure is fig
    plt.close(fig)
