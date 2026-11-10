# Evaluation

Validation only, one evaluator for every run. Intervals: 95% session bootstrap.

| run | val images | sessions | mask AP50 | tolerant AP50 | missing yarn clDice |
|---|---|---|---|---|---|
| maskrcnn_r50_session_seed0 | 359 | 33 | 0.173 [0.099, 0.280] | 0.227 [0.141, 0.338] | 0.886 [0.857, 0.910] |

## Leakage

No model has both arms evaluated.
