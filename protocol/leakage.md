# Leakage

## Question
Does a random per-image split report a higher validation mask AP50 than the session split for
the same model and settings?

## Inputs
- All 2407 images pooled from the export, read through `fabrid.labels.load_index`.
- The session split from `fabrid.sessions.session_split` with a 30 second gap. Its test part is
  locked and excluded from both arms, so both arms draw from the same pool: the session split's
  train and val images.

## Arms
- session: train and val exactly as assigned by `session_split`.
- random: the same pool reassigned per image, ignoring sessions, by a permutation from
  `numpy.random.default_rng(0)` over the sorted file names. The first n names go to train, where
  n is the session arm's train count, and the rest to val, so both arms have identical train and
  val sizes. This is `random_split(seed 0)` restricted to the unlocked pool. The full
  `random_split` over all images is not used because its train part would contain locked test
  images.

Each model (YOLOv8m-seg, YOLO11m-seg) is trained on both arms with identical settings, those of
`protocol/baselines.md`, session arm first.

## Endpoint
Validation mask AP50 at the best validation epoch, defined as in `protocol/baselines.md`, per
model and arm. The difference is random arm minus session arm, per model.

## Rule
Inflation is claimed for a model only if the difference is positive and its 95% bootstrap
interval, computed later from the saved validation predictions of both arms, excludes zero.
Otherwise no inflation is claimed for that model. The resampling details are fixed in the leakage
analysis step before it reads any predictions.

## Notes
