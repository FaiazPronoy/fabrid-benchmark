# Evaluation

Validation only, one evaluator for every run. Intervals: 95% session bootstrap.

| run | val images | sessions | mask AP50 | tolerant AP50 | missing yarn clDice |
|---|---|---|---|---|---|
| hybrid_anchors_session_seed0 | 359 | 33 | 0.182 [0.114, 0.272] | 0.228 [0.144, 0.320] | 0.890 [0.854, 0.922] |

## Leakage

No model has both arms evaluated.

Runs listed but not found: hybrid_anchors_mssa_session_seed0, hybrid_anchors_mssa_aux_preprocess_session_seed0.
