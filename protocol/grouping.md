# Grouping

## Question
How are images of a dataset whose groups are given from outside (for example one folder per
fabric roll or capture run) split into train, val and test so that no group spans two parts,
the part sizes are as close to the target shares as whole groups allow, and the test part stays
locked?

## Inputs
- `group_of`: a mapping from image name to group. A group is a set of images that must stay in
  one part because they share a source (FabriD: a capture session from
  `fabrid.sessions.assign_sessions`; the second dataset: its folder). Group keys may be strings
  or integers. How groups are read from the second dataset is not fixed here.
- `shares`: target shares for train, val and test, default 0.70, 0.15, 0.15, non-negative and
  summing to 1.
- `strata` (optional): a mapping from group to one stratum label, for example the class a group
  was collected for.
- No pixels and no labels are read. This step only assigns names to parts.

## Rule
`fabrid.groups.grouped_split` is deterministic: the same inputs always give the same mapping,
and no random generator is used.

1. Every group gets its size, the number of its images. The cost of an assignment is the sum over
   the three parts of the squared difference between the part's image count and its target
   count (share times total images).
2. With strata, for every stratum its largest group (ties by group key) is placed in train
   first, so every stratum appears in train. The search below then runs on the other groups
   with those images already counted in train. With strata the result is the closest fit
   among splits that keep those groups in train, which can be farther from the shares than the
   closest fit without strata.
3. Exact search: the groups, largest first and ties by group key, are added one at a time to a
   table of every reachable pair of train and val image counts (test takes the rest). Each pair
   remembers the first group that made it reachable and whether that group went to train or
   val; train is preferred when both reach a pair at the same step. The pair with the lowest
   cost is chosen, ties going to more train images and then more val images, and the groups are
   read back from the table. Groups not read back go to test. This finds a split with the
   lowest possible cost over all assignments of whole groups. Memory grows with the square of
   the number of images (about 40 MB and 7 seconds at 2407 images in 183 groups on the test machine).
4. The achieved shares (image count of each part over the total) are reported next to the
   targets by `fabrid.groups.achieved_shares`. They differ from the targets when whole groups do
   not allow an exact fit.

Group keys are ordered by their text form, then by type name, so integers and strings sort
without error.

## Random arm
`fabrid.groups.random_arm` mirrors the random arm of `fabrid.yolo.arm_mapping` for FabriD: the
pool is every image not in the test part, sorted by name; the train count is the grouped split's;
the pool is permuted with `numpy.random.default_rng(seed).permutation`; the first train count
images of the permutation go to train and the rest to val. The result holds train and val only,
and test images are neither moved nor listed. The only difference from the FabriD function is
the input: it takes an existing grouped mapping instead of building the session split itself.
For the same mapping and seed the two give the same result.

## Test lock
- `fabrid.groups.check_groups` raises if a group spans two parts or an image has a group but no
  part.
- `fabrid.groups.check_view` raises if a training or validation view holds an image that the
  grouped split puts in test, an image the split does not know, or any part name other than
  train and val.
- Any trainer that uses a grouped split calls `check_view` on every view it builds, before
  reading pixels.

## Checks
Tested with synthetic groups only: no group is ever split; the cost equals the brute-force
optimum over all assignments for small cases; the same input gives the same output; every stratum is in
train; the lock raises on a test image in a view; the random arm keeps the train and val counts
and never lists a test image.

## Pixel maps
For datasets labelled with per-pixel class maps, `fabrid.pixelmap` turns a map into instances and
instances into label lines:
- `instances_from_map` returns one boolean mask per connected component of each class value,
  4 or 8 connected, background excluded, components under `min_pixels` dropped, ordered by
  class value and then by the first pixel of the component in row-major order. Touching
  components of different class values are always separate instances.
- `polygons_from_mask` traces the boundary along pixel edges, so a pixel is inside the polygon
  exactly when it is in the mask. Where two mask pixels touch only at a corner, the trace passes
  through that corner and keeps them in one outline, so each 8-connected component gives one
  outer polygon. Holes are not represented: the YOLO-seg format has no holes, so they are
  filled. Collinear points are removed; with a tolerance above 0 the outline is simplified with
  the Douglas-Peucker method. Coordinates are divided by width and height as in `labels.py`.
- Round trip: rasterizing the polygons with `fabrid.geometry.rasterize` gives back the mask
  exactly (IoU 1) at tolerance 0 for components without holes, one pixel wide lines included.
  With holes, the IoU is the mask area over the area with holes filled.

## Outputs
None by itself. A split built with these functions is written by the step that uses it, as
`split.csv`, next to that step's results.

## Notes
