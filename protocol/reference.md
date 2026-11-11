# Reference packs

## Question
What exactly does a trained model of this repository predict on a few fixed validation images,
so that a separate inference app can show it reproduces those predictions? No metric is
computed and nothing is compared here.

## Inputs
- All images pooled through `fabrid.labels.load_index` and the session split from
  `fabrid.sessions.session_split` with a 30 second gap. Only its val part is used.
- Six val images chosen without randomness: in sorted file name order, the first image that
  contains each of the four classes (needle mark, hole, missing yarn, stain), skipping one
  already chosen, then the first two images with an empty label file. A chosen image that is not
  in the session split's val part, or that the run's own `split.csv` does not list as val, stops
  the job with an error. The locked test part is never read.
- The weights of one finished run, found under `/kaggle/working/runs` and `/kaggle/input`.

## Arms
One pack per model:
- `yolov8m_seg_aug`: run `baselines/yolov8m_aug_session_seed0`, weights `train/weights/best.pt`.
  Ultralytics predicts from the image file path with imgsz 640, conf 0.05, iou 0.7, max_det
  100, retina_masks true, on CPU. A mask pixel counts when the returned mask value is above 0.5.
- `fabrimssa_net`: run `ablation/hybrid_anchors_mssa_aux_session_seed0`, weights
  `checkpoints/best.pth`. The model is built as training built it, by
  `fabrid.hybrid.build_hybrid` from the run's committed config with the COCO weights loaded, so
  the frozen batch norm eps matches, then the state dict is loaded strictly. Eval mode, CPU,
  float32, no autocast. The image is read with PIL as RGB, passed through the run's preprocessing
  when the run used it, and divided by 255, as `fabrid.maskrcnn.Samples` does. Masks are
  thresholded at the run's `mask_threshold`.

Detections with score at least 0.05 are kept. Class names come from `fabrid.labels.CLASSES`;
Ultralytics numbers classes from 0 and the Mask R-CNN side from 1.

## Endpoint
None. The pack is a record of outputs, not a measurement.

## Rule
None, beyond the test lock above.

## Outputs
`results/reference/<model_key>/`: `config.yaml`, `detections.json` (model key, sha256 of the
weights file, minimum score, and per image the file name, width, height and detections with
label, score, box as x0 y0 x1 y1 in pixels, and mask area in pixels), and `images/` with copies
of the six dataset images. The job also prints, per image, how many detections score at least
0.30.

## Notes
