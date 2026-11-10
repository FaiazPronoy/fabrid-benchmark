# Integrity audit

Verdict: PASS

| Check | Result |
| --- | --- |
| No exact duplicate images | pass (0 groups, 0 files) |
| Every label line parses | pass (0 unparsed of 2502) |

## Files

- Images: 2407 (test 358, train 1685, valid 364)
- Image sizes: 360x640 (379), 480x640 (433), 534x640 (1), 565x640 (1), 577x640 (1), 579x640 (1), 640x360 (269), 640x480 (737), 640x535 (1), 640x613 (1), 640x640 (583)
- Capture time range: 2026-05-05T22:31:38 to 2026-08-03T12:46:24
- Capture days: 5, sessions: 129
- Missing label files: 0

## Duplicates

- Near duplicate pairs (pHash or dHash distance up to 8): 590, of which 109 fall in the same session
- Stems shared by more than one file: 109 stems, 326 files, 109 stems with differing pixels

## Labels

- Label lines: 2502, valid polygons: 2502
- Parsed but invalid polygons (under 3 points or zero area): 0
- Polygons with a point outside the unit square: 0
- Empty label files: 613, images without a valid polygon: 613

| Class | Lines | Valid polygons | Images |
| --- | --- | --- | --- |
| needle mark | 636 | 636 | 477 |
| hole | 568 | 568 | 459 |
| missing yarn | 580 | 580 | 511 |
| stain | 718 | 718 | 429 |
