"""
C13 - WHERE to put attention, and WHICH attention block.

Placement (CBAM):        layer 12 only  |  layer 15 only  |  layers 12 + 15 (the paper)
Other attention blocks:  SE @ 12+15     |  Coordinate Attention @ 12+15
All on the scene split, seed 0, same settings.  The '12 + 15 CBAM' and
'baseline' rows are re-used from c12 (seed 0).

Output: revision/results/c13_ablation.csv and c13_ablation.json
Run:    python revision/c13_placement_ablation.py     (GPU, 4 new runs)
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS_DIR, banner, save_json  # noqa: E402
from train_utils import run_experiment  # noqa: E402

CONFIGS = [
    # (label,                 run name,               attention, layers)
    ("None (baseline)",       "yolov8x_none_s0",      None,      (12, 15)),
    ("CBAM @ L12",            "yolov8x_cbam_L12_s0",  "cbam",    (12,)),
    ("CBAM @ L15",            "yolov8x_cbam_L15_s0",  "cbam",    (15,)),
    ("CBAM @ L12+L15",        "yolov8x_cbam_s0",      "cbam",    (12, 15)),
    ("SE @ L12+L15",          "yolov8x_se_s0",        "se",      (12, 15)),
    ("CoordAtt @ L12+L15",    "yolov8x_coord_s0",     "coord",   (12, 15)),
]

banner("C13  Placement and attention-type ablation")
rows = []
for label, run_name, attention, layers in CONFIGS:
    r = run_experiment(run_name, "yolov8x.pt", attention=attention, layers=layers, seed=0)
    row = {"config": label, "run": run_name, "status": r.get("status"), **(r.get("test") or {}),
           "params": r.get("params"), "GFLOPs": r.get("GFLOPs"), "reason": r.get("reason")}
    rows.append(row)
    print(row)

with open(RESULTS_DIR / "c13_ablation.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
save_json("c13_ablation", {"rows": rows, "seed": 0, "split": "splits/scene_split (test)"})
