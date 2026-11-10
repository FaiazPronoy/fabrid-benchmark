# Baselines

## Question
What validation mask AP50 do standard instance segmenters reach on the session split?

## Inputs
- All 2407 images of the Roboflow YOLO-seg export, pooled across its train, valid and test
  folders, read through `fabrid.labels.load_index` from `FABRID_DATA`.
- The session split from `fabrid.sessions.session_split` with a 30 second gap. Only its train and
  val parts are used. The test part is locked: no image from it is linked into a training view,
  predicted on or scored, and the code raises if one would be.

## Arms
- YOLOv8m-seg, COCO pretrained weights `yolov8m-seg.pt`.
- YOLO11m-seg, COCO pretrained weights `yolo11m-seg.pt`.
- Mask R-CNN, specified by its own config.

YOLO settings: Ultralytics, imgsz 640, batch 16, 100 epochs, patience 20, seed 0, deterministic,
single GPU, default augmentation with the albumentations CLAHE transform switched off, no CLAHE
or Gabor preprocessing.

## Endpoint
Validation mask AP50 at the best validation epoch. The best epoch is the one Ultralytics keeps
as `best.pt`: the first epoch with the highest fitness, where fitness is
0.1 AP50 + 0.9 AP50-95 summed over box and mask. The reported value comes from evaluating
`best.pt` on the validation split after training ends. Box and mask precision, recall, AP50 and
AP50-95, overall and per class, are reported alongside.

## Rule
None. This step is descriptive and makes no pass or fail claim.

## Outputs
`results/baselines/<model>_<arm>_seed0/`: `config.yaml`, `metrics.json`, `curves.csv`,
`split.csv`. Weights, validation predictions (COCO json) and validation ground truth (COCO json)
stay in `/kaggle/working/runs/baselines/<model>_<arm>_seed0/`.

## Notes
- 2026-11-10, before the first Mask R-CNN run: the Mask R-CNN arm is torchvision
  `maskrcnn_resnet50_fpn` with COCO weights and heads replaced for 4 classes, session arm only,
  input 640, batch 4, mixed precision, AdamW with learning rate 1e-4 and weight decay 1e-4,
  2 warm-up epochs then cosine decay, 40 epochs, horizontal and vertical flip with probability
  0.5, seed 0, no CLAHE or Gabor preprocessing. Its best epoch is the one with the highest
  validation mask AP50 computed with pycocotools.
- 2026-11-10: the YOLO and Mask R-CNN trainers score validation with different evaluators and
  different ground truth rasterization, so the numbers they write are not compared with each
  other directly. Any comparison between models uses one evaluator on the saved validation
  predictions in a later step.
