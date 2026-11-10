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
