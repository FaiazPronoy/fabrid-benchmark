# Evaluation

Validation only, one evaluator for every run. Intervals: 95% session bootstrap.

| run | val images | sessions | mask AP50 | tolerant AP50 | missing yarn clDice |
|---|---|---|---|---|---|
| maskrcnn_r50_paste_session_seed0 | 359 | 33 | 0.200 [0.115, 0.312] | 0.254 [0.154, 0.372] | 0.898 [0.872, 0.915] |

## Leakage

No model has both arms evaluated.
