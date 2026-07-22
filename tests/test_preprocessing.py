from __future__ import annotations

import numpy as np
import pytest

import labelimage_tools as lit


def test_erode_and_dilate_nonconsecutive_labels(nonconsecutive_labels):
    eroded = lit.erode_labels(nonconsecutive_labels)
    dilated = lit.dilate_labels(nonconsecutive_labels)
    assert set(np.unique(eroded)) <= {0, 5, 10}
    assert {5, 10} <= set(np.unique(dilated))
    assert np.array_equal(lit.dialate_labels(nonconsecutive_labels), dilated)


def test_shuffle_preserves_background(nonconsecutive_labels):
    shuffled = lit.shuffle_labels(nonconsecutive_labels, seed=1, background=0)
    assert np.all(shuffled[nonconsecutive_labels == 0] == 0)
    assert set(np.unique(shuffled)) == {0, 5, 10}


def test_fill_internal_gaps_and_sentinel():
    labels = np.full((9, 9), 5, dtype=np.int64)
    labels[3:6, 3:6] = 0
    filled = lit.fill_internal_gaps_edt(labels, max_distance=None)
    assert np.all(filled == 5)
    sentinel = lit.fill_internal_gaps_edt(labels, max_distance=0.5, fill_value=10000)
    assert 10000 in np.unique(sentinel)


def test_fill_internal_gaps_can_leave_far_pixels_as_background():
    labels = np.full((9, 9), 5, dtype=np.int64)
    labels[2:7, 2:7] = 0

    filled = lit.fill_internal_gaps_edt(
        labels,
        max_distance=1,
        fill_value="background",
    )

    assert filled.dtype == labels.dtype
    assert filled[2, 2] == 5
    assert filled[4, 4] == 0
    assert np.all(filled[labels == 5] == 5)


def test_fill_internal_gaps_background_respects_custom_background():
    labels = np.full((7, 7), 3, dtype=np.int64)
    labels[2:5, 2:5] = -1

    filled = lit.fill_internal_gaps_edt(
        labels,
        background=-1,
        max_distance=1,
        fill_value="background",
    )

    assert filled[3, 3] == -1
    assert filled[2, 2] == 3


def test_fill_internal_gaps_without_threshold_allows_high_labels():
    labels = np.full((5, 5), 20_000, dtype=np.int64)
    labels[2, 2] = 0
    assert np.all(lit.fill_internal_gaps_edt(labels, max_distance=None) == 20_000)


def test_replace_labels_maps_background_and_handles_missing_labels():
    labels = np.array([[0, 5, 10], [10, 5, 0]], dtype=np.uint8)
    original = labels.copy()

    replaced = lit.replace_labels(labels, {0: 9, 5: 50}, default_missing=-1)

    np.testing.assert_array_equal(replaced, [[9, 50, -1], [-1, 50, 9]])
    assert np.issubdtype(replaced.dtype, np.signedinteger)
    np.testing.assert_array_equal(labels, original)


def test_replace_labels_can_preserve_unmapped_labels():
    labels = np.array([[0, 5, 10]], dtype=np.int16)

    replaced = lit.replace_labels(labels, {5: 7}, default_missing=None)

    np.testing.assert_array_equal(replaced, [[0, 7, 10]])


@pytest.mark.parametrize(
    ("mapping", "default_missing"),
    [({5: 1.5}, 0), ({5: 1}, 2.5)],
)
def test_replace_labels_rejects_non_integer_values(mapping, default_missing):
    with pytest.raises(ValueError, match="integer-like"):
        lit.replace_labels(np.array([[0, 5]]), mapping, default_missing=default_missing)


def test_image_has_holes_distinguishes_enclosed_and_border_background():
    labels = np.ones((7, 7), dtype=np.int64)
    labels[2, 2] = 0
    labels[4, :3] = 0

    has_holes, holes = lit.image_has_holes(labels)

    assert has_holes
    assert holes.dtype == np.bool_
    assert holes[2, 2]
    assert not np.any(holes[4, :3])


def test_image_has_holes_supports_custom_background_and_no_holes():
    labels = np.full((5, 5), 4, dtype=np.int64)
    labels[:, 0] = -1
    assert lit.image_has_holes(labels, background=-1)[0] is False

    labels[2, 2] = -1
    has_holes, holes = lit.image_has_holes(labels, background=-1)
    assert has_holes
    assert holes.sum() == 1


def test_image_has_hole_sentinel_with_explicit_and_inferred_threshold():
    labels = np.array([[0, 1, 2], [10_000, 10_001, 2]], dtype=np.int64)

    explicit_has_sentinel, explicit_mask = lit.image_has_hole_sentinel(labels)
    inferred_has_sentinel, inferred_mask = lit.image_has_hole_sentinel(
        labels,
        hole_sentinel=None,
    )

    assert explicit_has_sentinel and inferred_has_sentinel
    np.testing.assert_array_equal(explicit_mask, inferred_mask)
    np.testing.assert_array_equal(explicit_mask, labels >= 10_000)


def test_image_has_hole_sentinel_reports_absence_and_failed_inference():
    labels = np.array([[0, 1], [2, 3]], dtype=np.int64)
    has_sentinel, mask = lit.image_has_hole_sentinel(labels)
    assert has_sentinel is False
    assert not np.any(mask)

    with pytest.raises(ValueError, match="continuous run"):
        lit.image_has_hole_sentinel(labels, hole_sentinel=None)


def test_skeletonize_variants(nonconsecutive_labels):
    assert lit.skeletonize_dilate(nonconsecutive_labels).shape == nonconsecutive_labels.shape
    assert lit.skeletonize_erode(nonconsecutive_labels).shape == nonconsecutive_labels.shape
    assert np.array_equal(
        lit.skeletonize_labels(nonconsecutive_labels),
        lit.skeletonize_erode(nonconsecutive_labels),
    )


def test_find_and_remove_non_self_connected_bits():
    labels = np.zeros((8, 8), dtype=np.int64)
    labels[1:3, 1:3] = 5
    labels[5:7, 5:7] = 5
    bad = lit.find_non_self_connected_labels(labels)
    assert 5 in bad
    cleaned = lit.remove_non_self_connected_bits(labels)
    assert 5 not in lit.find_non_self_connected_labels(cleaned)
    assert np.sum(cleaned == 5) == 4


def test_crop_to_foreground_bbox(nonconsecutive_labels):
    cropped, slices = lit.crop_to_foreground_bbox(nonconsecutive_labels, padding=0)
    assert cropped.shape == (6, 6)
    assert slices == (slice(1, 7), slice(1, 7))


def test_load_image_pipeline(sample_path):
    labels = lit.load_image_pipeline(sample_path, crop_to_foreground=True, shuffle=False)
    assert labels.ndim == 2
    assert labels.max() > 0
