# Evaluation

Validation only, one evaluator for every run. Intervals: 95% session bootstrap.

| run | val images | sessions | mask AP50 | tolerant AP50 | missing yarn clDice |
|---|---|---|---|---|---|
| yolov8m_session_seed0 | 359 | 33 | 0.160 [0.094, 0.239] | 0.206 [0.133, 0.286] | 0.855 [0.846, 0.869] |
| yolov8m_random_seed0 | 359 | 73 | 0.369 [0.273, 0.431] | 0.442 [0.330, 0.503] | 0.871 [0.836, 0.911] |

## Leakage

| model | session AP50 | random AP50 | difference | 95% interval | inflation |
|---|---|---|---|---|---|
| yolov8m | 0.160 | 0.369 | 0.210 | [0.080, 0.304] | yes |

Runs listed but not found: yolov8m_aug_session_seed0.
