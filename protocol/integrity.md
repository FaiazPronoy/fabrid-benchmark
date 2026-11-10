# Integrity

## Question
Is the file set consistent with what is documented: 2407 JPG images at 640 px in a Roboflow
YOLO-seg export, one label file per image, label lines of a class id from 0 to 3 followed by
normalized polygon x y pairs, an empty label file for a defect-free image, and file names that
start with IMG_YYYYMMDD_HHMMSS?

## Inputs
Every image and label file under the export's train, valid and test folders, located through
`fabrid.labels.load_index` from `FABRID_DATA`. The audit reads files only to compute checksums,
perceptual hashes, image sizes and label counts. It uses no model and no split, so it does not
train, predict or score on the locked test split.

## Arms
None. A single audit of the whole file set.

## Endpoint
One pass or fail verdict made of two checks:
1. No exact duplicate images. Two images are exact duplicates when their decoded pixels are
   identical, which includes byte-identical files.
2. Every non-empty label line parses: an integer class id in 0 to 3 followed by an even number,
   at least four, of finite numbers.

## Rule
Pass when both checks hold, fail otherwise. Everything else is descriptive and makes no claim:
near duplicates by pHash and dHash with Hamming distance up to 8, stems shared by more than one
file and whether their pixels differ, polygons that parse but have fewer than 3 points or zero
area, coordinates outside the unit square, image sizes, label counts per class, and counts per
capture day.

## Outputs
`results/audit/integrity/`: `config.yaml`, `manifest.csv`, `duplicates_exact.csv`,
`duplicates_near.csv`, `stems_repeated.csv`, `label_counts.json`, `day_class_table.csv`,
`summary.md`.

## Notes
