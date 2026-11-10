# Geometry

## Question
How thin, small and sparse are the annotated instances of each class?

## Inputs
- Every valid polygon of all 2407 images of the Roboflow YOLO-seg export, pooled across its
  train, valid and test folders, read through `fabrid.labels.load_index` from `FABRID_DATA`. A
  polygon is valid when it has at least 3 points and non-zero area.
- Image width and height, read from each image header without decoding pixels.
- Pixels are decoded only for the contact sheets, and only from images in the train and val parts
  of the session split (`fabrid.sessions.session_split`, 30 second gap). The code raises if a
  contact sheet would use an image from the locked test part. No model is trained, run or scored.

Each polygon is scaled to pixel units and rasterized at image size: a pixel belongs to the mask
when its centre lies inside the polygon (even-odd rule). Polygons that cover no pixel centre are
counted and kept in `instances.csv` with empty measures.

## Measures
Per rasterized instance:
- area: number of mask pixels.
- bbox width and height: extent of the mask pixels in columns and rows.
- aspect ratio and long side of the minimum-area rectangle around the mask pixel squares.
- median width: twice the median, over the skeleton pixels, of the distance from the pixel centre
  to the mask boundary. The boundary distance is the Euclidean distance transform to the nearest
  background pixel centre minus 0.5, with the area outside the image counted as background, so a
  straight line k pixels wide (k odd) has median width k.
- solidity: mask area over the area of the convex hull of the mask pixel squares.
- extent: mask area over bbox area.

## Arms
None. A single description of all annotations.

## Endpoint
Per class, the median of the median width in pixels.

## Rule
None. This step is descriptive and makes no pass or fail claim.

## Outputs
`results/audit/geometry/`: `config.yaml`, `instances.csv`, `per_class.csv` (count and the 5, 25,
50, 75 and 95 percentiles of each measure, per class and over all classes), `size_strata.csv`
(share of instances by bbox area under 32 x 32, 32 x 32 to 96 x 96, over 96 x 96, per class),
`width_strata.csv` (share with median width up to 2, over 2 to 4, over 4 to 8, over 8 px, per
class), `summary.md`. Four contact sheets per class of 24 instances each, sampled with seed 0 and
drawn with their polygons, go to `/kaggle/working/runs/audit/geometry/` and are not committed.

## Notes
