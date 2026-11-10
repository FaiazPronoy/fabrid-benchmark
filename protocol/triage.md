# Triage of empty-label images

## Question
What share of the images with an empty label file do two independent reviewers both judge
clean, that is, free of any needle mark, hole, missing yarn or stain?

## Inputs
- Every image whose label file is empty (no label lines), from `fabrid.labels.load_index`
  over all 2407 pooled images.
- The session split from `fabrid.sessions.session_split` with a 30 second gap, used only to
  tell which images are in the locked test part.
- The best session-arm YOLOv8m-seg weights,
  `baselines/yolov8m_session_seed0/train/weights/best.pt`, searched for under
  `/kaggle/working/runs` and `/kaggle/input`.

## Arms
None. One review of every empty-label image.

Step "sheets" prepares the review:
- Train and val images: the YOLOv8m weights are run at confidence 0.05 and the images are
  ordered by their highest detection score, highest first, ties by file name, images without a
  detection last. Predicted boxes are drawn on their tiles.
- Locked test images: no model is run on them and nothing is predicted or scored. They are
  ordered by a permutation from `numpy.random.default_rng(0)` over their sorted file names, and
  shown without boxes, on sheets of their own after the train and val sheets.
- Contact sheets of 12 images, 320 px tiles, each file name under its tile, stay in the runs
  directory. `candidates.csv` and a blank `review_form.csv` are committed.

The model only orders the train and val images for the reviewers and decides nothing.

Review: two reviewers each fill in their own copy of the form without seeing the other's
answers, writing clean, defect or unsure for every image after looking at the full image, not
only the tile. The owner then merges the two columns into one `review_form.csv`.

Step "collect" reads the merged form. A blank or unknown answer, or a form whose names differ
from `candidates.csv`, stops the step with an error.

## Endpoint
The share of all empty-label images that both reviewers judged clean.

## Rule
An image enters the verified defect-free set only if both reviewers say clean. Images with any
defect or unsure answer are excluded from use as negatives until they are relabeled.
Agreement is reported descriptively: counts per answer pair, Cohen's kappa over the three
answers, the share where both said clean and the share where at least one said defect, overall
and per split.

## Outputs
`results/audit/triage/`: `config.yaml`, `candidates.csv`, `review_form.csv`.
`results/audit/triage/review/`: `config.yaml`, `agreement.json`, `verified_clean.csv`.
Contact sheets in `/kaggle/working/runs/audit/triage/sheets/`.

## Notes
