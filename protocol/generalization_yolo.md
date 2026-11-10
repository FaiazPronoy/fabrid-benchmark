# Generalization of YOLOv8m

## Question
Does stronger augmentation with copy-paste raise the validation mask AP50 of YOLOv8m-seg on the
session split?

## Inputs
- All 2407 images pooled, read through `fabrid.labels.load_index`, and the session split from
  `fabrid.sessions.session_split` with a 30 second gap. Only its train and val parts are used;
  the test part stays locked, as in `protocol/baselines.md`.
- Saved validation predictions of both arms, scored by `fabrid.evaluate`.

## Arms
- Baseline: `yolov8m_session_seed0`, the existing run from `configs/yolov8m_seg.yaml`. It is
  not rerun.
- Candidate: `yolov8m_aug_session_seed0` from `configs/yolov8m_seg_aug.yaml`. Every setting
  equals the baseline's (Ultralytics, `yolov8m-seg.pt`, imgsz 640, batch 16, 100 epochs,
  patience 20, seed 0, deterministic, single GPU, albumentations CLAHE switched off) except
  these Ultralytics train arguments: degrees 15, scale 0.5, hsv_v 0.6, hsv_s 0.8, flipud 0.5,
  copy_paste 0.5. Scale 0.5 equals the Ultralytics default and is listed only so the full
  augmentation is written in one place. All other augmentation stays at its default.

## Endpoint
Validation mask AP50 of each arm from `fabrid.evaluate` (`protocol/evaluation.md`), on the best
epoch predictions, and its difference, candidate minus baseline.

## Rule
The paired comparison of `protocol/evaluation.md`: both runs share the same validation images,
are rescored on the same 1000 session resamples, and the verdict is "higher" when the mean
difference is positive and its 95% interval excludes zero, "lower" when negative and the
interval excludes zero, and "not shown" otherwise.

This is one seed. A "higher" verdict is not claimed on its own: both arms are then trained again
with seeds 1 and 2, otherwise unchanged, and the improvement is claimed only if the paired
verdict is "higher" for each of the three seeds. A "lower" or "not shown" verdict is reported
as it is.

## Outputs
`results/baselines/yolov8m_aug_session_seed0/` from training, and
`results/evaluation/yolov8m_aug_session_seed0/` and `results/evaluation/compare_aug/` from the
evaluator.

## Notes
