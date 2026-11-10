# Similarity

## Question
How often does a held-out image have a near identical view of the same fabric in the training
part, under three ways of splitting the data?

## Inputs
- All 2407 images pooled from the export, read through `fabrid.labels.load_index`.
- One embedding per image from DINOv2 ViT-S/14 (`vit_small_patch14_dinov2.lvd142m` in timm),
  pretrained weights, no training or fine tuning, on CPU. Before embedding, every valid defect
  polygon is filled with the mean color of its image, so the vector describes the fabric rather
  than the defect. The image is then resized to 448 by 448, normalized with the ImageNet mean and
  standard deviation, and the class token is L2 normalized. Similarity is the cosine between two
  such vectors.
- Sessions from `fabrid.sessions.assign_sessions` with a 30 second gap.

No model is trained, and no detector or segmenter predicts on or is scored on any image. The
locked test part of the session split is embedded only because the export and random arms place
some of its images in their training part. Locked test images appear in the outputs only inside
aggregate counts and shares, never by name, in a pair list or in a contact sheet, and the code
raises if one would.

## Threshold
The threshold T is fixed before any arm is evaluated. Take the images of the session split's
train and val parts, sort them by capture time, and keep each pair of consecutive images that are
on the same day and at most 3 seconds apart. T is the 5th percentile (numpy linear
interpolation) of the cosine similarity over these pairs. It is written to `threshold.json`
before any arm share is computed.

## Arms
- export: the export's own folders, train against valid (val part) and test (test part).
- random: `fabrid.sessions.random_split(seed 0)` over all images.
- session: `fabrid.sessions.session_split`, checked with `check_disjoint`.

For every held-out image of an arm, its score is the highest cosine similarity to any image in
that arm's train part.

## Endpoint
For each arm, the share of its val images whose score is at or above T. The same share is also
reported at T minus 0.05 and T plus 0.05, and for the test part in aggregate, as description only.

## Rule
The session split passes when its val share at T is at most half of the random split's val share
at T. It fails otherwise. The export arm is descriptive and has no rule.

## Outputs
`results/audit/similarity/`: `config.yaml`, `threshold.json`, `leak_share.csv`,
`session_links.csv` (pairs of different sessions outside the locked test part whose highest cross
similarity is at or above T), `session_sizes.csv`, `summary.md`. Embeddings and contact sheets of
the 60 strongest cross session pairs go to `/kaggle/working/runs/audit/similarity/`.

## Notes
