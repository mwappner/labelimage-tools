# labelimage-tools

`labelimage-tools` is a small utility package for 2-D labeled tissue
segmentation images. It provides loading, validation, preprocessing, adjacency
and contact extraction, junction detection, graph coloring, contours, and
matplotlib plotting helpers.

## Conventions

- Label images are 2-D NumPy arrays.
- Image coordinates are represented as `(y, x)`.
- The default background label is `0`.
- Labels do not need to be consecutive.
- Integer label values are preserved unless a function explicitly documents a
  relabeling operation.

## Installation

From this checkout:

```bash
python -m pip install -e .
```

For tests:

```bash
python -m pip install -e '.[test]'
python -m pytest
```

## Load and preprocess labels

Use these helpers directly from scripts or notebooks.

```python
import labelimage_tools as lit

labels = lit.load_label_image("segmentation.tif")
labels = lit.dilate_labels(labels, structure=3, background=0, background_only=True)
labels = lit.fill_internal_gaps_edt(labels, background=0, max_distance=3)
labels, crop_slices = lit.crop_to_foreground_bbox(labels, background=0, padding=20)
```

## Adjacency and contact graph

Adjacency is computed by vectorized neighbor scanning. Contact values are counts
of neighboring pixel pairs, useful as weights but not exact geometric lengths.

```python
neighbors, pairs = lit.adjacency_with_unique_from_labels(
    labels,
    background=0,
    eight=True,
    allow_background_contacts=True,
)

neighbors, contacts = lit.adjacency_with_contact_from_labels(labels)
centroids = lit.get_centroids(labels)
```

## Junction detection

Junction pixels are pixels whose 3×3 neighborhood contains at least three
distinct labels. Connected junction pixels are clustered into `Junction`
objects with subpixel `(y, x)` centroids and the set of labels that meet there.

```python
junction_label_image, junctions = lit.junctions_from_labels(
    labels,
    background=None,
    min_labels=3,
    connectivity=2,
)

for junction in junctions:
    print(junction.id, junction.yx, sorted(junction.labels))
```

## Graph-colored plotting

Plotting helpers return matplotlib objects and never call `plt.show()`, so they
compose cleanly in notebooks.

```python
fig, ax = lit.plot_label_image(
    labels,
    use_graph_coloring=True,
    K=8,
    seed=1,
    title="Graph-colored labels",
)

fig, ax = lit.plot_junctions(labels, junctions=junctions, ax=ax)
```

You can also use the lower-level coloring helper:

```python
image, lut, ax = lit.show_map_with_colors(labels, K=8, seed=1)
```
