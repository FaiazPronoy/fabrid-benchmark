# fabrid-benchmark

Instance segmentation benchmark for fabric defects (needle mark, hole, missing yarn, stain) on a
2407 image Roboflow export. The export's own train, valid and test folders are ignored: every
image is pooled and re-split by capture session, where a session is a run of photos taken less
than 30 seconds apart on the same day, so that near identical frames never land on both sides of
a split. The test split is locked and no code in this repository trains, predicts or scores on
it. Each step has a protocol in `protocol/` written before its code, a config in `configs/`, and
writes small citable results with their exact config to `results/`. Data is read from
`FABRID_DATA` (default `/kaggle/input/fabrid-v9`).

1. Integrity audit of the file set: `python -m fabrid.integrity --config configs/integrity.yaml`
2. YOLOv8m-seg baseline, session and random split arms: `python -m fabrid.yolo --config configs/yolov8m_seg.yaml`
3. YOLO11m-seg baseline, session and random split arms: `python -m fabrid.yolo --config configs/yolo11m_seg.yaml`
4. Mask R-CNN baseline: `python -m fabrid.maskrcnn --config configs/maskrcnn_r50.yaml`
5. Geometry study: `python -m fabrid.geometry --config configs/geometry.yaml`
6. Similarity audit of the export, random and session splits: `python -m fabrid.similarity --config configs/similarity.yaml`

On Kaggle, open `kaggle/run.ipynb`, set `JOB` to a name from `kaggle/jobs.csv`, and add the
secrets `GITHUB_TOKEN`, `GIT_USER_NAME`, `GIT_USER_EMAIL` and `REPO_URL` so finished runs are
committed back. Locally: `pip install -e ".[dev]"`, then `pytest` and `ruff check`.
