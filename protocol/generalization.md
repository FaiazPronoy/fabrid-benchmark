# Generalization

## Question
Does each training-only remedy raise validation mask AP50 over the Mask R-CNN parent on the
session split?

The parent overfits: in `results/baselines/maskrcnn_r50_session_seed0/curves.csv` the mean
training loss falls from 1.237 at epoch 1 to 0.171 at epoch 40, while validation mask AP50 stays
between 0.11 and 0.17 from epoch 3 on (best 0.173 at epoch 13, 0.150 at epoch 40).

## Inputs
- All 2407 images of the Roboflow YOLO-seg export, pooled across its train, valid and test
  folders, read through `fabrid.labels.load_index` from `FABRID_DATA`.
- The session split from `fabrid.sessions.session_split` with a 30 second gap. Only its train and
  val parts are used. The test part is locked: no image from it is trained on, predicted on,
  scored or used as a paste donor, and the code raises if one would be.
- Everything else as in `configs/maskrcnn_r50.yaml`: torchvision Mask R-CNN ResNet-50 FPN with
  COCO weights, input 640, batch 4, AMP, AdamW lr 1e-4 and weight decay 1e-4, 2 warm-up epochs
  then cosine, 40 epochs, horizontal and vertical flip with probability 0.5 each, seed 0, the
  same target builder, the same two sanity checks (run without the remedies, as in the parent),
  and the same pycocotools evaluator on the val part after every epoch.

## Arms
The parent is the existing `maskrcnn_r50_session_seed0` run and is not rerun. The remedies are
applied to training images only, after the targets are built and before the existing flips.
Validation images are never transformed. Each remedy draws from its own generator,
`numpy.random.default_rng([seed, epoch, crc32(image name), k])` with k 1 for aug and 2 for
paste, so a given seed, image and epoch always get the same transform. When both are on, paste
runs first and aug second, so pasted instances follow the same photometric and geometric change.

1. `aug`:
   - Photometric, in this order: brightness (pixels times b), contrast (pixels pulled to or
     pushed from the image's mean gray by c), saturation (pixels pulled to or pushed from their
     own gray by s), with b, c and s each uniform in 0.6 to 1.4, then hue shifted by h of a full
     turn with h uniform in -0.05 to 0.05, in OpenCV HSV; values clipped to 0 to 255.
   - Geometric, one affine map: scale uniform in 0.6 to 1.4 about the image centre, then a
     translation uniform within the range that crops a larger image or places a smaller one
     inside the frame (up to half of the size change in each axis), then rotation uniform in -15
     to 15 degrees about the centre. The output keeps the original size. Pixels with no source
     are filled with the per-channel mean of the jittered image. The image is resampled
     bilinearly; each instance mask goes through the same map with nearest-neighbour
     resampling.
   - Instances whose mask has fewer than 4 pixels afterwards are dropped, and boxes are
     recomputed from the masks.
2. `paste`: with probability 0.5, paste 1 to 3 instances (uniform count). Each donor is an
   instance chosen uniformly from all instances of training images whose capture session
   differs from the destination's. The donor polygon is rasterized at its image size
   (`fabrid.geometry.rasterize`) and cut with a 5 pixel margin around its mask; the cut is
   scaled by a factor uniform in 0.75 to 1.25 (bilinear for pixels, nearest for the mask),
   flipped horizontally and vertically with probability 0.5 each, and placed at a uniform
   integer position fully inside the destination. Its colour is shifted per channel by the
   destination's mean minus the donor's mean, both taken over the pixels under the donor mask.
   It is blended with the mask blurred by a Gaussian of sigma 1.5 as the weight. The pasted mask
   is added as a new instance with the donor's class and is removed from every mask it covers,
   including earlier pastes. Instances left with fewer than 4 pixels are dropped, and boxes are
   recomputed from the masks. A donor that does not fit inside the destination is skipped.
3. `aug_paste`: both.

## Endpoint
Validation mask AP50 at the best epoch, where the best epoch is the first epoch with the highest
validation mask AP50, read from that epoch's evaluation. The training loss and validation mask
AP50 at the last epoch are recorded next to it in `metrics.json`, so that the gap between
training and validation can be compared across arms; they are descriptive.

## Rule
A remedy is said to help only if the paired session-bootstrap comparison from `fabrid.evaluate`
with the parent as baseline and the remedy's arm as candidate, both scored on the same val
images from their best epoch predictions, gives the top level verdict "higher" (the verdict for
the evaluator's mean primary endpoint, validation mask AP50). A verdict of "lower" or "not
shown" means the remedy is not shown to help. This step runs seed 0 only. Any such result is
confirmed with two more seeds before a claim is made; until then it is reported as a single
seed result.

## Outputs
`results/generalization/maskrcnn_r50_<arm>_session_seed0/`: `config.yaml`, `sanity.json`,
`metrics.json`, `curves.csv`, `split.csv`. Weights and validation predictions (COCO json) at the
best epoch stay in `/kaggle/working/runs/generalization/maskrcnn_r50_<arm>_session_seed0/`.

## Notes
