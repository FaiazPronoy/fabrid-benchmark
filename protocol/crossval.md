# Cross-validation

## Question
Is Mask R-CNN with augmentation (`maskrcnn_r50_aug`) higher than its parent (`maskrcnn_r50`) on
mask AP50 when every image of the train plus val pool is validated once?

This step was written after the single-split comparison was seen. On the session split's val
part (359 images), `results/evaluation/compare_maskrcnn_aug/` gave a difference of 0.055 in mask
AP50 with a 95% interval from -0.010 to 0.087, verdict "not shown". That result is the reason for
this step: one val part of that size cannot resolve a gain of 0.03 to 0.05. Nothing in this
protocol was tuned on any cross-validation output; none existed when it was written.

## Inputs
- All 2407 images of the Roboflow YOLO-seg export, pooled across its train, valid and test
  folders, read through `fabrid.labels.load_index` from `FABRID_DATA`.
- The session split from `fabrid.sessions.session_split` and sessions from
  `fabrid.sessions.assign_sessions`, both with a 30 second gap.
- The pool is the session split's train and val parts. The test part is locked: no image from it
  is placed in a fold, trained on, predicted on, scored or used as a paste donor, and the code
  raises if one would be.

## Folds
`fabrid.folds` assigns every pool image to one of 5 folds, without randomness:
1. A session is a group in the sense of `protocol/grouping.md` and is never split. Every
   session's vector is `fabrid.sessions.record_vector` summed over its images: image count,
   instance count per class, and empty image count.
2. Sessions are taken by image count, largest first, ties by session key in the order of
   `fabrid.groups.group_order`. Each goes to the fold that gives the lowest total cost after it is
   placed, where the total cost is the sum over all 5 folds and over every dimension that is
   nonzero in the pool of (fold total over pool total minus 1/5) squared; ties go to the lower
   fold number.
3. A fold is not allowed for a session if the fold would then hold every session of a class that
   occurs in at least two pool sessions. At most one fold per class is excluded this way, so with
   4 classes and 5 folds a fold is always allowed. As a result every class found in at least two
   pool sessions is present in every training side. A class found in a single session cannot be
   in the training side of that session's fold; such a case is reported, not hidden.
4. For fold k, the validation part is the images of fold k and the training part is every other
   pool image. `fabrid.groups.check_view` and `fabrid.groups.check_groups` are run on every view
   before pixels are read: a test image, an image outside the pool, or a session on both sides
   stops the run.

Sizes and class counts of every fold are written to `results/crossval/folds/`.

## Arms
- `maskrcnn_r50`: every setting of `configs/maskrcnn_r50.yaml` (`protocol/baselines.md`, note
  of 2026-11-10).
- `maskrcnn_r50_aug`: every setting of `configs/maskrcnn_r50_aug.yaml`
  (`protocol/generalization.md`, remedy `aug`).

Seed 0, one run per arm and fold, 10 runs in all. The only differences from the single split are
the training and validation parts, which come from the fold, the run name suffix `_fold<k>`, and
the step `crossval`. Paste donors are not used by either arm.

Epoch choice inside a fold works as in the single split: the best epoch is the first with the
highest mask AP50 on that fold's validation part, and its predictions are the ones scored. The
validation part therefore picks the epoch it is then scored on, so each arm's absolute pooled
score is optimistic. Both arms get the same selection on the same images, so the bias is of the
same kind for both, but it is not shown to be of the same size. Absolute pooled scores are
reported for reference only and are not compared with the single-split scores; only the paired
difference is the endpoint.

## Endpoint
The difference, `maskrcnn_r50_aug` minus `maskrcnn_r50`, in pooled mask AP50:
- Every fold run's best epoch predictions on its own validation part are scored by
  `fabrid.evaluate` exactly as in `protocol/evaluation.md` (same matching, same AP, same ground
  truth).
- The scored validation parts of the 5 folds of one arm are joined into one validation set that
  holds every pool image once, and mask AP50 is computed on that set as if it were one run: per
  class over all pooled predictions, ties by image name and then file order, then the mean over
  classes with ground truth.
- The 95% interval comes from 1000 resamples of all pool sessions with replacement, from
  `numpy.random.default_rng([0, crc32("crossval_<comparison name>")])`. Both arms are rescored
  on the same resamples, and the interval is the 2.5 and 97.5 percentiles of the resampled
  differences, ignoring resamples where either value is undefined.

The same paired table is reported for tolerant AP50 and per class, and the per fold scores of
both arms are listed next to it. All of these are descriptive.

## Rule
For the mean pooled mask AP50 difference: "higher" if the lower end of the interval is above
zero, "lower" if the upper end is below zero, else "not shown". The evaluator refuses to compute
the endpoint if a fold run of either arm is missing, if an arm's folds do not cover every pool
image exactly once, or if the two arms did not validate the same images in a fold. This is a
seed 0 result and is reported as one.

## Outputs
- `results/crossval/folds/`: `config.yaml`, `folds.csv` (image, session, fold), `sizes.csv`
  (per fold: sessions, images, empty images, instances per class in the validation part, and
  the classes missing from the training part).
- `results/crossval/<model>_session_seed0_fold<k>/`: as for the single-split runs.
- `results/evaluation/<model>_session_seed0_fold<k>/`: per fold evaluation and stored score
  tables.
- `results/evaluation/crossval_<name>/`: `config.yaml`, `comparison.json`, `comparison.csv`.

## Notes
