# Evaluation

Validation only, one evaluator for every run. Intervals: 95% session bootstrap.

| run | val images | sessions | mask AP50 | tolerant AP50 | missing yarn clDice |
|---|---|---|---|---|---|
| maskrcnn_r50_aug_session_seed0 | 359 | 33 | 0.228 [0.154, 0.303] | 0.277 [0.202, 0.353] | 0.892 [0.850, 0.919] |

## Leakage

No model has both arms evaluated.
