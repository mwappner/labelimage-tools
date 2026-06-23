from __future__ import annotations

import numpy as np

import labelimage_tools as lit


def test_dsatur_coloring_has_no_adjacent_conflicts(simple_labels):
    adj = lit.adjacency_from_labels(simple_labels)
    color = lit.dsatur_color(adj)
    for label, neighbors in adj.items():
        for neighbor in neighbors:
            assert color[label] != color[neighbor]


def test_color_planar_with_variety_uses_K_when_feasible():
    adj = {1: np.array([2, 4]), 2: np.array([1, 3]), 3: np.array([2, 4]), 4: np.array([1, 3])}
    color = lit.color_planar_with_variety(adj, K=3, seed=1)
    assert set(color) == {1, 2, 3, 4}
    assert len(set(color.values())) == 3


def test_apply_color_lut_int_maps_unmapped_to_nan():
    labels = np.array([[0, 5, 10]], dtype=np.int64)
    mapped, lut = lit.apply_color_lut_int(labels, {5: 1, 10: 2})
    assert np.isnan(mapped[0, 0])
    assert mapped[0, 1] == 1
    assert mapped[0, 2] == 2
    mapped2, _ = lit.apply_color_lut_int(labels, lut=lut)
    assert np.array_equal(np.nan_to_num(mapped), np.nan_to_num(mapped2))


def test_show_map_with_colors_runs(simple_labels):
    image, lut, ax = lit.show_map_with_colors(simple_labels, K=4, seed=1)
    assert image.axes is ax
    assert lut.size > 0


def test_show_map_with_colors_handles_isolated_or_empty_foreground():
    isolated = np.zeros((5, 5), dtype=np.int64)
    isolated[2, 2] = 5
    image, lut, ax = lit.show_map_with_colors(isolated, K=8, seed=1)
    assert image.axes is ax
    assert lut[5] == 0

    empty = np.zeros((5, 5), dtype=np.int64)
    image, lut, ax = lit.show_map_with_colors(empty, K=8, seed=1)
    assert image.axes is ax
    assert lut.size >= 1
