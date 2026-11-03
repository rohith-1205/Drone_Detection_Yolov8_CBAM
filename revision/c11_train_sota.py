"""
C11 - State-of-the-art comparison on the SAME scene split and settings.

Models (all COCO-pretrained, then fine-tuned on drones for 32 epochs):
  * YOLOv5x   (Ultralytics anchor-free 'yolov5xu.pt' - the original anchor-based
               YOLOv5x is a different code base)
  * YOLO11x   ('yolo11x.pt')
  * RT-DETR-X ('rtdetr-x.pt', transformer detector; AdamW 1e-4, see train_utils.py)
  * YOLOv8x baseline and YOLOv8x-CBAM (seed 0) are re-used from c12 so they are not trained twice.

Reports test-split P, R, mAP@0.5, mAP@0.5:0.95, params, GFLOPs, FPS.

Output: revision/results/c11_sota.csv and c11_sota.json
Run:    python revision/c11_train_sota.py            (GPU)
        python revision/c11_train_sota.py --smoke    (1 tiny epoch on CPU, pipeline test only)
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS_DIR, banner, save_json  # noqa: E402
from train_utils import run_experiment  # noqa: E402

smoke = "--smoke" in sys.argv
CONFIGS = [
    # (row name,      run name,            weights,         attention, family)
    ("YOLOv5x (u)",   "yolov5xu_s0",       "yolov5xu.pt",   None,      "yolo"),
    ("YOLO11x",       "yolo11x_s0",        "yolo11x.pt",    None,      "yolo"),
    ("RT-DETR-X",     "rtdetr-x_s0",       "rtdetr-x.pt",   None,      "rtdetr"),
    ("YOLOv8x",       "yolov8x_none_s0",   "yolov8x.pt",    None,      "yolo"),
    ("YOLOv8x-CBAM",  "yolov8x_cbam_s0",   "yolov8x.pt",    "cbam",    "yolo"),
]
if smoke:
    CONFIGS = [c for c in CONFIGS if c[1] in ("yolov8x_cbam_s0",)]

banner("C11  SOTA comparison" + ("  [SMOKE TEST]" if smoke else ""))
rows = []
for row_name, run_name, weights, attention, family in CONFIGS:
    r = run_experiment(run_name, weights, attention=attention, seed=0, family=family, smoke=smoke)
    row = {"model": row_name, "run": run_name, "status": r.get("status")}
    if r.get("status") == "done":
        row.update(**{k: r["test"][k] for k in ("precision", "recall", "mAP50", "mAP50_95")},
                   params=r["params"], GFLOPs=r.get("GFLOPs"), FPS=r["latency"]["fps_from_median"],
                   latency_ms=r["latency"]["median_ms"], device=r["latency"]["device"])
    else:
        row["reason"] = r.get("reason")
    rows.append(row)
    print(row)

name = "c11_sota_smoke" if smoke else "c11_sota"
fields = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
with open(RESULTS_DIR / f"{name}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
save_json(name, {"rows": rows, "split": "splits/scene_split (test)"})
