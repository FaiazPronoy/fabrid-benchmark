# Evaluation

## Question
How do the trained models compare when their saved validation predictions are scored by one
evaluator, against one ground truth, with one interval method?

## Inputs
- Per run: a COCO json of validation predictions (box, score, category id, and a mask as
  compressed RLE, uncompressed RLE or polygons) and the run's `split.csv`. Runs are listed in
  `configs/evaluate.yaml` by name and predictions path; the files are searched for under
  `/kaggle/working/runs` and `/kaggle/input`, and `split.csv` under `results/` first.
- Category ids are class id plus one, as written by both trainers. A prediction's image is
  found by its `file_name`, by an `image_id` equal to a file name or file stem, or by an integer
  `image_id` looked up in the `images` list of the trainer's ground truth file next to the
  predictions. That file is used only to map ids to file names.
- Ground truth is rebuilt here for every run in the same way: each valid polygon from
  `fabrid.labels.load_index` is scaled to the image size and rasterized with
  `fabrid.geometry.rasterize`. Polygons that rasterize to no pixel are left out and counted.
- Sessions from `fabrid.sessions.assign_sessions` with a 30 second gap.
- Test lock: `split.csv` must hold only train and val images, none of them in the session
  split's test part. Only val images are opened and scored. A prediction for an image that is
  not in the run's train or val part, or that is in the locked test part, stops the run with an
  error. Predictions for train images are dropped and counted.

## Arms
Every run listed in the config, for example YOLOv8m-seg, YOLO11m-seg and Mask R-CNN R50, each
on the session arm and, where trained, the random arm. Every prediction in the file is used,
with no score floor and no cap on detections per image.

## Endpoint
Primary: validation mask AP50.
1. Per image and class, predictions are taken in descending score order and each is matched to
   the unmatched ground truth instance of that class with the highest mask IoU, if that IoU is
   at least 0.5 (`fabrid.metrics.match_instances` rule). Matched predictions are true
   positives, the rest false positives.
2. Per class, all val predictions are pooled and sorted by descending score, ties by image
   name and then file order. Precision and recall are accumulated, precision is made
   monotone by taking the running maximum from the high recall end, and AP is the mean of
   that precision at the 101 recall levels 0, 0.01, ..., 1, taking 0 where the recall level is
   not reached.
3. The endpoint is the mean of the per class AP over classes with at least one ground truth
   instance.

Secondary, fixed now:
- Tolerant AP50: the same procedure with `fabrid.metrics.tolerant_iou` at tol_px 2 in place of
  mask IoU.
- Per class precision and recall at the score threshold that maximizes the class's validation
  F1, under the primary matching. Thresholds are the distinct prediction scores; a tie in F1
  goes to the higher threshold.
- AP50 by size bucket (`fabrid.metrics.size_bucket` on the bounding box area in pixels of the
  rasterized ground truth mask or the decoded predicted mask). Matching is the primary
  matching on all instances. For a bucket, the ground truth count is the instances in it; a
  prediction counts as a true positive when matched to an instance in the bucket, as a false
  positive when unmatched and its own box is in the bucket, and is left out otherwise. Per
  class, then the mean over classes with ground truth in that bucket.
- Missing yarn: the mean `fabrid.metrics.cldice` at tol_px 1 over the primary matched pairs,
  and the share of missing yarn ground truth instances that are matched.

## Intervals
Sessions of the run's val images are resampled with replacement, in the style of
`fabrid.metrics.group_bootstrap`: 1000 resamples, each drawing as many sessions as there are,
with every metric recomputed per resample from the fixed per image matches, an image weighted
by how often its session was drawn. The generator for a run is
`numpy.random.default_rng([0, crc32(run name)])`, base seed 0, so runs are resampled
independently. Intervals are the 2.5 and 97.5 percentiles of the resampled values, ignoring
resamples where a metric is undefined. The point value is the metric on the full val split.

## Rule
Descriptive comparison of models; no pass or fail. The leakage rule of `protocol/leakage.md`
is applied with this evaluator's primary endpoint on each run's best epoch predictions: per
model, the difference is random arm minus session arm. The two arms have different val images,
so sessions are resampled independently within each arm, the 1000 resampled differences are
paired by resample index, and the interval is their 2.5 and 97.5 percentiles. Inflation is
claimed for a model only if the difference is positive and the interval excludes zero.

## Outputs
`results/evaluation/<run name>/metrics.json` and `per_class.csv`,
`results/evaluation/leakage.csv`, `results/evaluation/summary.md`, and `config.yaml` next to
each.

## Notes
- 2026-11-10, before any prediction file was scored: output layout only. The shared
  `results/evaluation/leakage.csv` and `summary.md` are written per model instead, as
  `results/evaluation/summary_<model>/leakage.csv` and `summary.md`, so that runs evaluated from
  different accounts never write the same file. Metrics, intervals and rules are unchanged.
- 2026-11-10, before any comparison was computed: paired comparisons. The config may list
  comparisons, each naming a baseline run and a candidate run. Both must have the same
  validation images, or the evaluator stops with an error. One set of 1000 session resamples
  per comparison, from `numpy.random.default_rng([0, crc32(comparison name)])`, rescores both
  runs on the same resampled sessions. For the primary endpoint and tolerant AP50, per class
  and mean, the difference is candidate minus baseline, with the 2.5 and 97.5 percentiles of
  the resampled differences as its interval, ignoring resamples where either value is
  undefined. Verdict: "higher" if the difference is positive and the interval excludes zero,
  "lower" if negative and the interval excludes zero, else "not shown". The top level verdict
  of a comparison is the one for the mean primary endpoint. Outputs go to
  `results/evaluation/compare_<name>/` (`comparison.json`, `comparison.csv`, `config.yaml`).
  The unpaired leakage rule is unchanged. A run entry may also name its step (default
  baselines), which sets its default predictions path `<step>/<run>/val_predictions.json` and
  split path `results/<step>/<run>/split.csv`; ablation runs use step ablation.
- 2026-11-11, after the Mask R-CNN parent was scored (0.172901) and before any YOLO or ablation
  run was scored: two changes, neither touching a metric, an interval or a rule. First, a
  defect: image ids in a ground truth file were read as integers, so the YOLO files, whose ids
  are file stems, stopped the evaluator with an error before anything was scored. Ids are now
  compared as text. Second, every scored run also writes its per-detection table
  (`scored_images.csv`, `scored_predictions.csv`, `scored_truth.csv`: image, class, score, match
  flags, size bucket, and per ground truth instance its match flag and clDice) next to
  `metrics.json`. When a listed run has no prediction file in reach but has these stored tables
  in the repository, the evaluator reads them instead, checks that they reproduce the stored
  primary endpoint exactly and that no image is from the locked test part, and uses them for
  the leakage table and the paired comparisons. This lets a comparison be computed when its two
  runs were trained on different accounts. Such a run is not rescored and its files are not
  rewritten. Runs under `results/generalization/` and their comparisons against the parent
  (`protocol/generalization.md`) are added to the config.
- 2026-11-12, after the single-split `maskrcnn_aug` comparison was seen and before any fold run
  existed: pooled cross-validation comparisons (`protocol/crossval.md`). The config may list
  `crossval` comparisons, each naming a baseline and a candidate run; their fold runs
  `<run>_fold0` to `<run>_fold4` (step crossval) are added to the runs and scored one by one
  exactly as above, from prediction files or from stored score tables. The evaluator then
  checks that both arms validated the same images in every fold, that these match the fold
  assignment of `fabrid.folds`, and that an arm's folds cover every pool image exactly once;
  any failure stops the comparison. The five scored folds of an arm are joined into one
  validation set over the pool, with images in name order, and scored as one run. One set of
  1000 resamples of all pool sessions, from
  `numpy.random.default_rng([0, crc32("crossval_<name>")])`, rescores both arms; differences,
  intervals and per class entries are formed as in the paired comparisons above. The verdict
  here is "higher" if the lower end is above zero, "lower" if the upper end is below zero, else
  "not shown". Per fold mask AP50 of both arms is listed for reference. Outputs go to
  `results/evaluation/crossval_<name>/` (`comparison.json`, `comparison.csv`, `config.yaml`).
  Nothing above this note changes.
