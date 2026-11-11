# External data: TSFabrics

## Question
What does the public TSFabrics dataset (Scientific Data 2026, "TSFabrics: A Time-Series Fabric
Dataset for Real-Time Defect Detection on Circular Knitting Machines", Figshare DOI
10.6084/m9.figshare.29573381) actually contain, and does it support a leak-resistant grouped
split with pixel masks and defect classes? Nothing about its file layout is assumed; the job
discovers and reports it.

## Inputs
- The Figshare record found by searching the public Figshare API for the DOI, without
  credentials, and the files it lists.
- The files themselves, downloaded with resume and retries into the heavy runs directory, never
  committed, and extracted there. Archives found inside the extracted tree are extracted too, up
  to the depth set in the config.

## Arms
None. A single description of one dataset.

## What is measured
- The record as Figshare states it: title, version, DOI, license, file names, sizes and md5.
- Every folder: file count, extensions, total bytes.
- Every folder with images: image count and the distinct width x height x mode values.
- Every CSV file: its columns and, for a column whose name mentions label, class, defect or
  type, the count of each value, per folder.
- Every folder whose name or content marks it as masks (a single channel image with few distinct
  values): mask count, distinct pixel values with their pixel counts, connected components of
  non-zero pixels per mask (8-connectivity, summarized), the share of masks that are all zero,
  and the share of masks with more than one non-zero value.
- Pairing: how many mask file stems match image file stems and in which folders, how many CSV
  values name existing image files, and mask sizes that differ from their paired image.
- Folder name fields: names split on underscore, hyphen and space, with the distinct values per
  field position at each depth, and file name patterns with digits replaced by `#`.
- Anything inconsistent: unreadable or empty files, mixed sizes or modes within a folder, masks
  without an image and images without a mask in an annotated folder, CSV references to missing
  files, nested archives not extracted.

From these the summary states plainly: whether masks encode the defect class per pixel or only
foreground, whether separate defects are separable as instances, how images pair with masks and
labels, what the folder name fields are, and what is inconsistent. Whether a leak-resistant
grouped split is possible is read from the folder fields and pairing found, and stated as what
the files show, not assumed.

## Rule
Pass when every downloaded file matches the md5 the Figshare API publishes for it. A mismatch
stops the job with an error before anything is extracted. Everything else is descriptive.

## Outputs
`results/external/tsfabrics_fetch/`: `config.yaml`, `record.json`, `tree.csv`, `images.csv`,
`labels.csv`, `masks.csv`, `summary.md`. Downloads and extracted files stay in the runs
directory set by the config.

## Notes
