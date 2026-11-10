# Semantic

## Question
What per-class pixel Dice does a standard semantic segmentation model reach on validation?

## Inputs
- All 2407 images pooled from the export, read through `fabrid.labels.load_index`.
- The session split from `fabrid.sessions.session_split` with a 30 second gap, through
  `fabrid.yolo.arm_mapping` with arm session. Only its train and val parts are used. The test
  part is locked: no image from it is trained on, predicted on or scored, and the code raises if
  one would be.
- Targets: one class map per image, background 0 and classes 1 to 4 for needle mark, hole,
  missing yarn and stain. Each valid polygon is rasterized with `fabrid.geometry.rasterize` in
  label line order, so where polygons of different classes overlap the later line wins. The
  number of pixels covered by more than one class is counted and reported per split.

## Arms
One arm. SegFormer-B0 from Hugging Face transformers, weights
`nvidia/segformer-b0-finetuned-ade-512-512`, with the classifier re-initialized for background
plus 4 classes. Each image is padded at the bottom and right to 640 by 640; padded pixels are
ignored by the loss and by scoring. Normalization with the ImageNet mean and standard deviation,
no CLAHE or Gabor preprocessing. Horizontal and vertical flip, each with probability 0.5. AdamW,
learning rate 6e-5, weight decay 0.01, 2 warm-up epochs then cosine decay per step, 60 epochs,
batch 8, mixed precision, seed 0. Loss is cross entropy plus soft Dice, with equal weight; the
soft Dice is averaged over the 4 defect classes and pooled over the batch with smoothing 1.
Logits are upsampled bilinearly to the input size before the loss and before scoring, and the
predicted class of a pixel is the argmax.

Before training, an overfit check trains a fresh model on 8 train images that contain a defect
for 150 iterations; it passes when the mean loss of the last 10 iterations is below half the
first loss. A batch of defect-free images must also give a finite loss. Training does not start
unless both pass.

## Endpoint
Mean over the 4 defect classes of validation Dice, where each class's Dice is
2 TP / (2 TP + FP + FN) with TP, FP and FN pooled over all validation pixels. A class with no
true and no predicted pixel has no Dice and is left out of the mean. The reported value is taken
at the best epoch by this same number; the first such epoch wins a tie.

Reported alongside, at the best epoch: per-class Dice, IoU, precision and recall, and the
image-level false alarm rate on defect-free validation images, where an image is an alarm when
any defect class has at least 20 predicted pixels.

## Rule
None. This step is descriptive and makes no pass or fail claim.

## Outputs
`results/baselines/segformer_b0_session_seed0/`: `config.yaml`, `sanity.json`, `metrics.json`,
`curves.csv`, `split.csv`. Checkpoints and the validation class maps of the best epoch (one
compressed npz) stay in `/kaggle/working/runs/baselines/segformer_b0_session_seed0/`.

## Notes
