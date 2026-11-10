# Evaluation

Validation only, one evaluator for every run. Intervals: 95% session bootstrap.

| run | val images | sessions | mask AP50 | tolerant AP50 | missing yarn clDice |
|---|---|---|---|---|---|
| yolo11m_session_seed0 | 359 | 33 | 0.145 [0.089, 0.223] | 0.215 [0.138, 0.307] | 0.866 [0.844, 0.880] |
| yolo11m_random_seed0 | 359 | 73 | 0.388 [0.262, 0.455] | 0.467 [0.332, 0.531] | 0.863 [0.836, 0.889] |

## Leakage

| model | session AP50 | random AP50 | difference | 95% interval | inflation |
|---|---|---|---|---|---|
| yolo11m | 0.145 | 0.388 | 0.243 | [0.094, 0.321] | yes |
