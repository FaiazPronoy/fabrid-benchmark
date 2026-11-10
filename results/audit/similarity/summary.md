# Similarity audit

Verdict for the session split: FAIL

- Threshold T: 0.6336, the 5th percentile of cosine similarity over 448 consecutive pairs at most 3 seconds apart on the same day, from 2031 images of the session split train and val
- Session val share at T: 0.9666, limit 0.5 times the random val share 0.9889 = 0.4945
- Export val share at T: 0.9945 (descriptive)

| Arm | Part | Images | At or above T | Share at T | Share at T - 0.05 | Share at T + 0.05 |
| --- | --- | --- | --- | --- | --- | --- |
| export | val | 364 | 362 | 0.9945 | 0.9973 | 0.9725 |
| export | test | 358 | 357 | 0.9972 | 1.0000 | 0.9916 |
| random | val | 361 | 357 | 0.9889 | 0.9972 | 0.9695 |
| random | test | 361 | 360 | 0.9972 | 1.0000 | 0.9751 |
| session | val | 359 | 347 | 0.9666 | 0.9805 | 0.8747 |
| session | test | 376 | 363 | 0.9654 | 0.9920 | 0.8910 |

## Sessions

- Sessions: 129
- Pairs of different sessions with highest cross similarity at or above T: 4030, of which 1820 involve a locked test session and are counted only here; the other 2210 are listed in `session_links.csv`
