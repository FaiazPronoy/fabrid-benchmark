# Foundation

## Question
Does a segmentation head trained on a frozen self-supervised backbone (DINOv2 ViT-S/14) reach a
higher mean validation Dice over the 4 defect classes than a fine-tuned SegFormer-B0, on the
session split, where validation fabric comes from capture sessions never seen in training?

## Inputs
- The same pool, session split, targets, test lock and scoring as `protocol/semantic.md`. Only
  the train and val parts of the session split are used; the test part is never trained on,
  predicted on or scored, and the code raises if one would be.
- For the comparison: the validation class maps of the best epoch saved by each run
  (`val_maps.npz` in `/kaggle/working/runs/baselines/<run>/`, or under `/kaggle/input`), each
  run's `results/baselines/<run>/split.csv`, targets rebuilt from the labels exactly as in
  training, and sessions from `fabrid.sessions.assign_sessions` with a 30 second gap.

## Arms
- `segformer_b0_session_seed0`: as fixed in `protocol/semantic.md`, unchanged.
- `dinov2_s14_session_seed0`: timm `vit_small_patch14_dinov2.lvd142m` with pretrained weights,
  frozen: no parameter receives a gradient and the backbone stays in eval mode throughout
  training. Each image is padded at the bottom and right to 644 by 644, a multiple of 14, with
  the padded pixels ignored by loss and scoring. The patch tokens of the last four blocks,
  passed through the backbone's final layer norm, are reshaped to maps and concatenated; a 1x1
  convolution projects them to 256 channels, followed by two blocks of 3x3 convolution, batch
  norm and ReLU, bilinear upsampling to the input size, and a 1x1 classifier to background plus
  4 classes. Only this head trains. Loss, flips, warm-up and cosine schedule shape, sanity
  checks, checkpoints, deadline and outputs are those of the SegFormer arm; learning rate 1e-3,
  weight decay 0.01, 60 epochs, batch 8, mixed precision, seed 0.

## Endpoint
The endpoint of `protocol/semantic.md`: the mean over the 4 defect classes of validation Dice,
each class pooled over all validation pixels, at each run's best epoch by that number. The
compared quantity is the difference, DINOv2 arm minus SegFormer arm.

## Rule
The DINOv2 arm is said to be higher only if the difference is positive and its 95% session
bootstrap interval excludes zero. The interval: the validation sessions are resampled with
replacement, 2000 resamples, each drawing as many sessions as there are, with
`numpy.random.default_rng(0)`. In each resample both models are rescored on the same drawn
sessions, every image weighted by how often its session was drawn, and the difference is
recomputed; the interval is the 2.5 and 97.5 percentiles of the resampled differences,
ignoring a resample where either mean is undefined. Both runs must cover the same validation
images or the comparison stops with an error.

This holds for seed 0 only. No claim is made until the same comparison, with seeds 1 and 2
for both arms, also meets the rule.

## Outputs
Each run writes the outputs of `protocol/semantic.md` under
`results/baselines/<run>/`. The comparison, `python -m fabrid.semantic --compare <a> <b>`,
writes `results/evaluation/compare_semantic_<a>_vs_<b>/` with `config.yaml`,
`comparison.json`, `per_class.csv` and `summary.md`.

## Notes
