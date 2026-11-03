"""
C12 - Repeat baseline YOLOv8x and YOLOv8x-CBAM with 3 seeds; report mean +- std.

A single run can be lucky.  Training each model three times with different
random seeds (0, 1, 2) and reporting the average and the spread shows whether
the CBAM gain is larger than the run-to-run noise.

Output: revision/results/c12_seeds.csv (one row per run) and c12_seeds.json (mean +- std)
Run:    python revision/c12_train_seeds.py          (GPU, 6 runs)
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS_DIR, banner, save_json  # noqa: E402
from train_utils import mean_std, run_experiment  # noqa: E402

SEEDS = (0, 1, 2)
MODES = ("none", "cbam")
METRICS = ("precision", "recall", "mAP50", "mAP50_95")

banner("C12  Seeds 0/1/2 for baseline and CBAM")
rows, summary = [], {}
for mode in MODES:
    done = []
    for seed in SEEDS:
        r = run_experiment(f"yolov8x_{mode}_s{seed}", "yolov8x.pt", attention=None if mode == "none" else mode, seed=seed)
        row = {"mode": mode, "seed": seed, "status": r.get("status"), **(r.get("test") or {}), "reason": r.get("reason")}
        rows.append(row)
        if r.get("status") == "done":
            done.append(r["test"])
    if len(done) == len(SEEDS):
        summary[mode] = {m: dict(zip(("mean", "std"), mean_std([d[m] for d in done]))) for m in METRICS}
    else:
        summary[mode] = {"status": "incomplete", "finished_runs": len(done), "needed": len(SEEDS)}
    print(mode, summary[mode])

with open(RESULTS_DIR / "c12_seeds.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["mode", "seed", "status", *METRICS, "reason"])
    w.writeheader()
    w.writerows(rows)
save_json("c12_seeds", {"seeds": SEEDS, "summary": summary, "std": "sample standard deviation (n-1)"})
