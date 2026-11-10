# Ablation

## Question
Does each component of the thesis model, added to the Mask R-CNN parent, raise validation mask
AP50 over the arm before it?

## Inputs
- All 2407 images of the Roboflow YOLO-seg export, pooled across its train, valid and test
  folders, read through `fabrid.labels.load_index` from `FABRID_DATA`.
- The session split from `fabrid.sessions.session_split` with a 30 second gap. Only its train and
  val parts are used. The test part is locked: no image from it is trained on, predicted on or
  scored, and the code raises if one would be.
- Everything else as in `configs/maskrcnn_r50.yaml`: torchvision Mask R-CNN ResNet-50 FPN, input
  640 (min and max size), batch 4, AMP, AdamW lr 1e-4 and weight decay 1e-4, 2 warm-up epochs
  then cosine, 40 epochs, horizontal and vertical flip with probability 0.5 each, seed 0, the
  same target builder, the same two sanity checks, and the same pycocotools evaluator run on the
  val part after every epoch.
- Weights: every tensor of the COCO Mask R-CNN checkpoint whose name and shape match is loaded,
  as in the parent, with frozen batch norm epsilon set to 0 as torchvision does for these
  weights. The box and mask predictors are replaced for 4 classes plus background, as in the
  parent. Tensors with no match start from their default initialization: the RPN objectness and
  box regression layers when the anchor count changes, and every module added below.

## Arms
The parent is the existing `maskrcnn_r50_session_seed0` run and is not rerun. Each arm adds one
component to the arm before it. Components are copied from the thesis notebook `final.ipynb`,
cells marked CELL 15, 16, 21, 22, 23, 24, 25, 26, 27 and 28.

1. `anchors` (thesis CELL 28 anchor set): the RPN uses the four FPN levels at strides 4, 8, 16
   and 32 with anchor sizes (8, 16, 24), (16, 32, 48), (32, 64, 96), (64, 128, 192) and aspect
   ratios 0.25, 0.5, 1, 2, 4, giving 15 anchors per location, and the stride 64 pooled level is
   removed. RoI pooling uses the strides 4 to 32 levels. The parent already uses the stride 4
   level, both in the RPN (32 px anchors) and in RoI pooling, so this arm does not add a level:
   it changes the anchor set and removes the stride 64 level.
2. `anchors_mssa` (CELL 21 and 22): after each of the four FPN levels a multi-scale strip
   attention block: six parallel branches, each a full convolution from 256 to 32 channels with
   kernel 1x3, 3x1, 1x5, 5x1, 1x7 or 7x1, BatchNorm and SiLU; concatenation; 1x1 convolution to
   256 channels, BatchNorm and SiLU; channel attention (global average pool, 1x1 to 32 channels,
   SiLU, 1x1 to 256, sigmoid); spatial attention (channel mean and max maps, 7x7 convolution
   without bias, sigmoid); 1x1 convolution and BatchNorm; output SiLU of input plus that result.
   One block per level, 378,626 parameters each.
3. `anchors_mssa_aux` (CELL 23 to 28): two training-only heads on the stride 4 level after its
   strip attention block. Semantic head: 3x3 convolution 256 to 128, BatchNorm, SiLU, 1x1
   convolution to 5 outputs (background and 4 classes). Boundary head: 3x3 convolution 256 to 64,
   BatchNorm, SiLU, 1x1 convolution to 1 output. Targets are built per image from the instance
   masks after the model's own resize and padding: a label map where later instances overwrite
   earlier ones, and the boundary as the radius 1 morphological gradient (3x3 elliptical kernel)
   of the foreground of that map; both are brought to the stride 4 grid by nearest
   interpolation. Losses, computed in fp32: multiclass focal loss (gamma 2) on the semantic head
   with weight 1.0, multiclass foreground Dice loss skipping absent classes with weight 1.0, and
   the boundary loss 0.5 BCE plus 0.5 Dice on the boundary head with weight 0.5. The heads are not
   used at inference.
4. `anchors_mssa_aux_preprocess` (CELL 15 and 16): when each image is loaded, before flips, at
   training and at validation: CLAHE on the L channel of Lab (clip limit 2.0, 8 x 8 tiles), then
   Gabor on the grayscale of the CLAHE output with orientations 0, 45, 90 and 135 degrees,
   frequency 0.18 (wavelength 1 / 0.18), sigma 3.0, gamma 0.5, phase 0 and kernel size 19; the
   absolute responses are reduced by the maximum over orientations and min-max normalized per
   image; the image is 0.85 times the CLAHE RGB image plus 0.15 times 255 times that map, clipped
   to 0 to 255 and stored as 8 bit.

## Deviations from the thesis model
- Settings stay at the parent's values, not the thesis values: input 640 instead of 1024, 100
  detections per image instead of 300, torchvision default RPN and box sampler sizes, the
  parent's optimizer and schedule, and the parent's flips (vertical flip 0.5 where the thesis
  used 0.05, and no other augmentation).
- The backbone and FPN start from COCO Mask R-CNN weights with frozen batch norm and the
  stem and the first ResNet stage frozen, as in the parent; the thesis started from ImageNet ResNet-50 weights.
- The aux losses are logged as three weighted terms, `loss_aux_focal`, `loss_aux_dice` and
  `loss_aux_boundary`; their sum equals the thesis `loss_aux_hybrid`.
- Bringing the aux targets to the stride 4 grid by max pooling instead of nearest interpolation
  would keep 1 px defects; it changes the thesis design and is left as a candidate for a later
  arm.

## Endpoint
Validation mask AP50 at the best epoch, where the best epoch is the first epoch with the highest
validation mask AP50, read from that epoch's evaluation.

## Rule
A component is said to help only if its arm's endpoint minus the endpoint of the arm before it is
positive and the 95% session-bootstrap interval of that difference, computed by the evaluation
step from the saved validation predictions, excludes zero. Otherwise it is reported as not shown
to help. This step runs one seed; the winning arm and the parent get two more seeds later.

## Outputs
`results/ablation/hybrid_<arm>_session_seed0/`: `config.yaml`, `sanity.json`, `metrics.json` (with
parameter counts of the arm and of the parent architecture), `curves.csv`, `split.csv`. Weights
and validation predictions (COCO json) at the best epoch stay in
`/kaggle/working/runs/ablation/hybrid_<arm>_session_seed0/`.

## Notes
