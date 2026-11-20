# Pilot on new fabrics

## Question
Do the newly collected photos help? On a held-out set of new fabrics, is YOLOv8m-seg trained on
the new photos, alone or together with the old FabriD training data, better on mask AP50 than
the existing model trained on the old data only?

## Inputs
- New data: the Roboflow YOLOv11 segmentation export `Fabri-new.v2i.yolov11.zip`, 535 images
  resized to fit within 1280 px, no augmentation, Kaggle dataset `fabrid-new`. Roboflow's own
  train, valid and test assignment is ignored and all 535 images are pooled.
- Fabric split list `configs/pilot/fabrics.csv` with columns `image,split,test_fabric`: 427
  `train`, 93 `test` and 15 `excluded` images. Excluded images look too similar to a test fabric
  and are used nowhere. Every test image names its fabric; the 93 test images come from 5
  fabrics (black_white_check 13, blue_check 16, maroon_floral 11, olive 21, royal_blue 32). The
  job stops with an error if an export image is missing from the list or the list names an image
  that is not in the export.
- Old data: the FabriD v9 export (Kaggle dataset `fabrid-v9`) read through
  `fabrid.labels.load_index`, and its session split from `fabrid.yolo.arm_mapping` with arm
  `session`, seed 0 and a 30 second gap, unchanged. Only its train and val parts are used; the
  locked test part is never linked, trained on, predicted on or scored.
- Old model: `yolov8m_seg_aug.pt` from the Kaggle dataset `fabrid-weights`, the model behind the
  demo app, trained on the old session split train part.

## Labels
- Class mapping by name, case-insensitive, ignoring a trailing " defect", to the old FabriD
  order needle mark 0, hole 1, missing yarn 2, stain 3 (`fabrid.labels.CLASSES`). The new
  export's `hole`, `missing yarn`, `needle mark`, `stain` become 1, 2, 0, 3. An unknown name stops
  the job. The same mapping is applied to the class names stored in a model, so predictions of
  every arm are read in this order.
- Needle mark merging. The labelers drew one polygon per needle hole (1104 tiny polygons in 81
  images) while the rule is one shape per dotted line. Per image, needle mark polygons are
  clustered by single linkage on the minimum distance between polygons in pixels (0 when they
  touch or overlap), with threshold `needle_link` = 0.05 of the image diagonal. Each cluster of
  two or more polygons becomes one polygon: the minimum-area rotated rectangle around all their
  vertices, with each side widened to at least `needle_min_width` = 0.006 of the image diagonal
  about its centre, clipped to the image. Singletons stay as drawn, as do all other classes. The
  merged polygon takes the place of the cluster's first polygon in the label file. The same step
  is applied to train and test labels of the new data, never to the old data. Polygon counts
  before and after are recorded per image, and a contact sheet of up to 24 merged clusters is
  written as a Kaggle output for visual review.

## Splits
- Pilot test: all 93 `split=test` images. Never used for training or model selection.
- Training pool: the 427 `split=train` images. Validation for checkpoint selection is carved
  from this pool: images are sorted by the photo number in the file name (the digits after
  `IMG_`, ties by name; names without it go last in name order), cut into consecutive blocks of
  10, and round(0.15 x number of blocks) blocks are drawn for validation with
  `numpy.random.default_rng(0)`.
- Merged pool: the training pool above plus the old session split train part. Old validation
  stays as the old split defines it and is added to the new validation.

Every built dataset is a YOLO-seg folder with its own `data.yaml` and `manifest.csv` (image,
source, split, test_fabric, needle counts before and after). The build checks that no pilot test
or excluded image is in any training or validation folder and stops otherwise.

## Arms
All YOLOv8m-seg, all predicting the same pilot test with the same settings.
- A `pilot_old`: the old model, no training.
- B `pilot_new`: trained on the new training pool only.
- C `pilot_merged`: trained on the merged pool.

B and C use the model, weights, image size, batch, epochs, patience, seed and augmentation of
`configs/yolov8m_seg_aug.yaml`, trained by `fabrid.yolo.train`, so the only difference from A is
the training data. The checkpoint is Ultralytics `best.pt` chosen on the arm's own validation
part. Inference: Ultralytics predict from the image file with imgsz 640, conf 0.001, iou 0.7,
max_det 100, retina_masks true; a mask pixel counts when the mask value is above 0.5.

## Endpoint
Single endpoint: mask AP50 over the four classes on the pilot test, scored by `fabrid.evaluate`
(`fabrid.evaluate.score_images` and `fabrid.evaluate.Scorer`, matching and AP as in
`protocol/evaluation.md`) with the merged pilot test labels as ground truth.

Intervals: 95% bootstrap over test images, 2000 resamples from
`numpy.random.default_rng([0, crc32("pilot")])`. The same resamples are used for every arm and
for the paired differences B minus A, C minus A and C minus B. The interval is over images of only
5 fabrics, so it does not describe variation between fabrics.

Secondary, descriptive: per class mask AP50, box AP50 (the same scorer with ground truth and
predictions replaced by their axis-aligned boxes), and mask AP50 per test fabric.

## Rule
B or C counts as better than A on new fabrics only if the lower end of the paired 95 percent
interval of its mask AP50 difference is above 0. Anything else is reported as not shown. C minus
B is reported with the same interval and the same wording.

## Outputs
- `results/pilot/prep/`: `config.yaml`, `counts.json`, `needles.csv`, and the manifest of every
  built dataset. The needle contact sheet `needles.jpg` stays in
  `/kaggle/working/runs/pilot/prep/` and is not committed.
- `results/pilot/<arm>/`: `config.yaml`, `metrics.json`, the scored tables of mask matching and,
  in `box/`, of box matching, and for B and C `curves.csv` and `manifest.csv`. Weights and
  prediction files stay in `/kaggle/working/runs/pilot/<arm>/`.
- `results/pilot/evaluation/`: `config.yaml`, `metrics.json`, `per_class.csv`, `per_fabric.csv`,
  `comparisons.csv`, `comparisons.json`, `summary.md`.

## Limitations
- The fabric grouping of the new photos was done by eye from colour and photo order.
- The needle merge is an automatic relabelling, not a labeler's decision.
- About 15 to 20 percent of new images may contain small labeling errors, seen in a 20 image spot
  check.
- 5 test fabrics give wide intervals, and the interval does not cover new fabrics beyond them.
- The old FabriD images were exported at 640 px with Roboflow auto-contrast; the new photos
  have no auto-contrast. Arm A therefore sees test inputs prepared differently from its training
  data, so part of any difference from A may come from preprocessing rather than from the data.
- In arm C, checkpoint selection uses a validation set that is mostly old images (359 of 419).
- This pilot does not touch the locked FabriD test and is not the paper's main result.

## Notes
