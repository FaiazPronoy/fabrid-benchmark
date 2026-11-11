# Evaluation

Validation only, one evaluator for every run. Intervals: 95% session bootstrap.

| run | val images | sessions | mask AP50 | tolerant AP50 | missing yarn clDice |
|---|---|---|---|---|---|
| maskrcnn_r50_aug_paste_session_seed0 | 359 | 33 | 0.184 [0.118, 0.274] | 0.229 [0.153, 0.324] | 0.905 [0.862, 0.925] |

## Leakage

No model has both arms evaluated.
