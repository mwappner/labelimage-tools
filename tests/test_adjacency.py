from __future__ import annotations

import numpy as np

import labelimage_tools as lit


def _four_labels_around_hole(size=3):
    """Return four side labels surrounding a square internal hole."""
    width = size + 6
    labels = np.zeros((width, width), dtype=np.int64)
    labels[:3, :] = 1
    labels[-3:, :] = 3
    labels[3:-3, :3] = 4
    labels[3:-3, -3:] = 2
    return labels


def _contact_lookup(neighbors, contacts):
    return {
        tuple(sorted((int(label), int(neighbor)))): float(contact)
        for label, values in neighbors.items()
        for neighbor, contact in zip(values, contacts[label], strict=True)
        if int(label) < int(neighbor)
    }


def test_adjacency_expected_pairs_and_contacts(simple_labels):
    neighbors, pairs = lit.adjacency_with_unique_from_labels(simple_labels, eight=False)
    assert {tuple(row) for row in pairs.tolist()} == {(1, 2), (1, 3), (2, 3)}
    assert set(neighbors[1]) == {2, 3}
    neighbors2, contacts = lit.adjacency_with_contact_from_labels(simple_labels, eight=False)
    assert set(neighbors2[1]) == {2, 3}
    assert all(np.all(values > 0) for values in contacts.values())


def test_four_vs_eight_neighborhood_differs():
    labels = np.array([[1, 0], [0, 2]], dtype=np.int64)
    assert lit.adjacency_from_labels(labels, eight=False) == {}
    adj8 = lit.adjacency_from_labels(labels, eight=True)
    assert set(adj8[1]) == {2}


def test_centroids_and_background_border_behavior(simple_labels):
    labels = np.pad(simple_labels, 1)
    neighbors = lit.adjacency_from_labels(labels, allow_background_contacts=True)
    assert lit.label_is_border(neighbors, 1)
    assert set(lit.border_labels(neighbors)) == {1, 2, 3}
    centroids = lit.get_centroids(simple_labels)
    assert np.allclose(centroids[1], [0.5, 1.0])


def test_adjacency_pairs_alias(simple_labels):
    pairs = lit.adjacency_pairs_from_labels(simple_labels)
    assert pairs.shape[1] == 2


def test_label_pixel_counts_preserves_labels_and_background_option():
    labels = np.array(
        [
            [0, 5, 5],
            [10, 10, 5],
            [0, 10, 42],
        ],
        dtype=np.int64,
    )
    assert lit.label_pixel_counts(labels) == {5: 3, 10: 3, 42: 1}
    assert lit.label_pixel_counts(labels, include_background=True) == {0: 2, 5: 3, 10: 3, 42: 1}


def test_graph_from_labels_returns_aligned_graph_data(simple_labels):
    neighbors, contacts, centroids, pixel_counts = lit.graph_from_labels(
        simple_labels,
        eight=False,
    )

    assert centroids is not None
    assert pixel_counts is not None
    assert set(neighbors) == {1, 2, 3}
    assert set(centroids) == {1, 2, 3}
    assert set(pixel_counts) == {1, 2, 3}
    assert pixel_counts[1] == int(np.count_nonzero(simple_labels == 1))

    for label, nbrs in neighbors.items():
        assert label in contacts
        assert len(contacts[label]) == len(nbrs)
        for nbr, contact in zip(nbrs, contacts[label], strict=True):
            reverse_index = list(neighbors[int(nbr)]).index(label)
            assert np.isclose(contact, contacts[int(nbr)][reverse_index])


def test_hole_bridging_is_optional_and_builds_uncertainty_clique():
    labels = _four_labels_around_hole()
    ordinary = {tuple(pair) for pair in lit.adjacency_pairs_from_labels(labels, eight=False)}
    bridged = {
        tuple(pair)
        for pair in lit.adjacency_pairs_from_labels(labels, eight=False, bridge_holes=True)
    }

    assert ordinary == {(1, 2), (1, 4), (2, 3), (3, 4)}
    assert bridged == ordinary | {(1, 3), (2, 4)}


def test_hole_bridging_skips_external_and_oversized_background():
    labels = _four_labels_around_hole()
    labels[3, :4] = 0  # Open the hole to the image's background exterior.
    assert not lit.image_has_holes(labels)[0]
    assert np.array_equal(
        lit.adjacency_pairs_from_labels(labels, eight=False),
        lit.adjacency_pairs_from_labels(labels, eight=False, bridge_holes=True),
    )

    closed = _four_labels_around_hole()
    ordinary = lit.adjacency_pairs_from_labels(closed, eight=False)
    skipped = lit.adjacency_pairs_from_labels(
        closed,
        eight=False,
        bridge_holes=True,
        max_hole_area=8,
    )
    assert np.array_equal(skipped, ordinary)


def test_hole_pair_distance_filters_opposite_labels():
    labels = _four_labels_around_hole()
    pairs = {
        tuple(pair)
        for pair in lit.adjacency_pairs_from_labels(
            labels,
            eight=False,
            bridge_holes=True,
            max_hole_distance=1,
        )
    }
    assert (1, 3) not in pairs
    assert (2, 4) not in pairs


def test_hole_border_labels_always_use_four_connectivity():
    labels = _four_labels_around_hole(size=1)
    labels[2, 2] = 9  # This label is only diagonal to the one-pixel hole.
    neighbors = lit.adjacency_from_labels(labels, eight=True, bridge_holes=True)

    assert 3 not in set(neighbors[9])


def test_hole_contact_fallback_is_symmetric_and_aligned():
    labels = _four_labels_around_hole()
    neighbors, contacts = lit.adjacency_with_contact_from_labels(
        labels,
        eight=False,
        bridge_holes=True,
        inferred_contact=2.5,
    )
    lookup = _contact_lookup(neighbors, contacts)

    assert lookup[(1, 3)] == 2.5
    assert lookup[(2, 4)] > 0  # The deterministic EDT fill makes this pair meet.
    for label, values in neighbors.items():
        assert len(values) == len(contacts[label])
        for neighbor, contact in zip(values, contacts[label], strict=True):
            reverse = list(neighbors[int(neighbor)]).index(label)
            assert contact == contacts[int(neighbor)][reverse]


def test_hole_contacts_respect_diagonal_weight():
    labels = _four_labels_around_hole()
    neighbors4, contacts4 = lit.adjacency_with_contact_from_labels(
        labels,
        eight=False,
        bridge_holes=True,
    )
    neighbors8, contacts8 = lit.adjacency_with_contact_from_labels(
        labels,
        eight=True,
        diag_weight=0.5,
        bridge_holes=True,
    )

    lookup4 = _contact_lookup(neighbors4, contacts4)
    lookup8 = _contact_lookup(neighbors8, contacts8)
    assert any(lookup8[pair] != lookup4[pair] for pair in lookup4)


def test_hole_contact_contributions_accumulate_across_holes():
    labels = np.full((5, 13), 4, dtype=np.int64)
    labels[:2, :] = 1
    labels[3:, :] = 3
    labels[2, 4:] = 2
    labels[2, 3] = 0
    labels[2, 9] = 0
    labels[2, 8] = 4

    neighbors, contacts = lit.adjacency_with_contact_from_labels(
        labels,
        eight=False,
        bridge_holes=True,
        inferred_contact=2.0,
    )
    lookup = _contact_lookup(neighbors, contacts)

    assert lookup[(1, 3)] == 4.0


def test_hole_bridging_is_independent_of_background_contacts():
    labels = np.pad(_four_labels_around_hole(), 1)
    neighbors = lit.adjacency_from_labels(
        labels,
        eight=False,
        allow_background_contacts=True,
        bridge_holes=True,
    )

    assert 0 in neighbors
    assert {1, 2, 3, 4} <= set(neighbors[0])
    assert 3 in set(neighbors[1])


def test_hole_bridging_supports_custom_background_and_high_labels():
    labels = _four_labels_around_hole() + 20_000
    labels[labels == 20_000] = -1
    pairs = lit.adjacency_pairs_from_labels(
        labels,
        background=-1,
        eight=False,
        bridge_holes=True,
    )
    assert {tuple(pair) for pair in pairs} >= {(20_001, 20_003), (20_002, 20_004)}


def test_hole_bridge_option_validation(simple_labels):
    for kwargs in (
        {"max_hole_area": -1},
        {"max_hole_area": 1.5},
        {"max_hole_distance": -1},
        {"inferred_contact": 0},
    ):
        with np.testing.assert_raises(ValueError):
            lit.adjacency_from_labels(simple_labels, bridge_holes=True, **kwargs)
