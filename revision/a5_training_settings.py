"""
A5 - Which settings did the runs ACTUALLY use?

Sources read (nothing is guessed):
  1. colab_ablation.py                  - the CBAM ablation script in this repo
  2. evidence/baseline_2023_run/        - copied from github.com/rohith-1205/Drone-Detection-1
       args.yaml, results.csv, drone_detection_yolov8x.ipynb (training log)

It also checks whether the paper's baseline row (M1) and Fig. 4 come from
that 2023 run.

Output: revision/results/a5_training_settings.json
Run:    python revision/a5_training_settings.py
"""
import ast
import csv
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import PROJECT_DIR, REVISION_DIR, banner, save_json  # noqa: E402

EVIDENCE = REVISION_DIR / "evidence" / "baseline_2023_run"
out = {}

# ----------------------------------------------------------------------
banner("1. Settings written in colab_ablation.py (CBAM ablation)")
# ----------------------------------------------------------------------
source = (PROJECT_DIR / "colab_ablation.py").read_text()
tree = ast.parse(source)
train_calls, val_calls = [], []
for node in ast.walk(tree):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ("train", "val"):
        kwargs = {}
        for kw in node.keywords:
            try:
                kwargs[kw.arg] = ast.literal_eval(kw.value)
            except Exception:
                kwargs[kw.arg] = ast.unparse(kw.value)
        (train_calls if node.func.attr == "train" else val_calls).append(kwargs)
out["colab_ablation_train_calls"] = train_calls
out["colab_ablation_val_calls"] = val_calls
out["colab_ablation_checkpoint_evaluated"] = "best.pt" if "best.pt" in source else "unknown"
gflops_line = [l.strip() for l in source.splitlines() if "'gflops'" in l]
out["colab_ablation_gflops_is_hardcoded"] = gflops_line
for c in train_calls:
    print("train():", c)
for c in val_calls:
    print("val():  ", c)
print("GFLOPs line:", gflops_line)

# ----------------------------------------------------------------------
banner("2. The 2023 plain-YOLOv8x run (args.yaml + training log)")
# ----------------------------------------------------------------------
args = yaml.safe_load((EVIDENCE / "args.yaml").read_text())
keep = ["model", "epochs", "batch", "imgsz", "optimizer", "lr0", "lrf", "momentum", "weight_decay", "seed",
        "conf", "iou", "max_det", "mosaic", "close_mosaic", "fliplr", "hsv_h", "hsv_s", "hsv_v", "scale", "translate"]
out["baseline_2023_args"] = {k: args.get(k) for k in keep}
log = (EVIDENCE / "drone_detection_yolov8x.ipynb").read_text()
version = re.findall(r"Ultralytics YOLOv8\.[0-9.]+", log)
optim = re.findall(r"(AdamW|SGD)\(lr=([0-9.]+), momentum=([0-9.]+)\)", log)
gpu = re.findall(r"CUDA:0 \(([^,]+), (\d+)MiB\)", log)
out["baseline_2023_ultralytics_version"] = sorted(set(version))
out["baseline_2023_optimizer_actually_used"] = sorted(set(optim))
out["baseline_2023_gpu"] = sorted(set(gpu))
print(out["baseline_2023_args"])
print("version:", out["baseline_2023_ultralytics_version"], "| optimizer used:", out["baseline_2023_optimizer_actually_used"],
      "| GPU:", out["baseline_2023_gpu"])

# ----------------------------------------------------------------------
banner("3. results.csv: epochs, best epoch, and match with paper Table III (M1)")
# ----------------------------------------------------------------------
with open(EVIDENCE / "results.csv") as f:
    rows = [{k.strip(): float(v) for k, v in r.items()} for r in csv.DictReader(f)]
n_epochs = len(rows)
# Ultralytics picks best.pt by fitness = 0.1*mAP50 + 0.9*mAP50-95
fitness = [0.1 * r["metrics/mAP50(B)"] + 0.9 * r["metrics/mAP50-95(B)"] for r in rows]
best_idx = max(range(n_epochs), key=lambda i: fitness[i])
last = rows[-1]
paper_m1 = {"P": 86.82, "R": 83.90, "mAP50": 89.22, "mAP50_95": 53.76}
csv_last = {"P": last["metrics/precision(B)"] * 100, "R": last["metrics/recall(B)"] * 100,
            "mAP50": last["metrics/mAP50(B)"] * 100, "mAP50_95": last["metrics/mAP50-95(B)"] * 100}
match = all(abs(round(csv_last[k], 2) - paper_m1[k]) < 0.006 for k in paper_m1)
out["baseline_2023_epochs_in_results_csv"] = n_epochs
out["baseline_2023_best_epoch_by_fitness"] = int(rows[best_idx]["epoch"])
out["baseline_2023_last_epoch_metrics_percent"] = csv_last
out["paper_table3_M1"] = paper_m1
out["paper_M1_equals_2023_run_last_epoch"] = match
print(f"epochs logged: {n_epochs} | best epoch (fitness): {out['baseline_2023_best_epoch_by_fitness']} | last: {int(last['epoch'])}")
print("last-epoch metrics (%):", {k: round(v, 3) for k, v in csv_last.items()}, "| paper M1:", paper_m1, "| match:", match)

# Learning-rate check: the logged lr drops by lr0*(1-lrf)/epochs every epoch.
step = rows[-2]["lr/pg0"] - rows[-1]["lr/pg0"]
lr0_inferred = step * n_epochs / (1 - args["lrf"])
out["lr0_inferred_from_logged_schedule"] = lr0_inferred
print(f"lr0 implied by the logged schedule: {lr0_inferred:.5f}  (Table II says 0.01)")

# First-epoch values, to compare with the curves drawn in the paper's Fig. 4.
first = rows[0]
out["results_csv_epoch0"] = {k: first[k] for k in ("train/box_loss", "train/cls_loss", "metrics/precision(B)",
                                                    "metrics/recall(B)", "metrics/mAP50(B)", "val/box_loss")}

# ----------------------------------------------------------------------
banner("4. Paper statements vs facts")
# ----------------------------------------------------------------------
out["findings"] = [
    f"Training length: colab_ablation.py uses epochs={train_calls[-1].get('epochs')} and the 2023 run logged "
    f"{n_epochs} epochs. Nothing ran for 60 epochs -> the Fig. 4 caption '60' is wrong; it should be 32.",
    "Checkpoint: colab_ablation.py validates weights/best.pt (Ultralytics best = max 0.1*mAP50+0.9*mAP50-95).",
    f"Validation thresholds: conf={val_calls[-1].get('conf')}, NMS IoU={val_calls[-1].get('iou')} (Ultralytics defaults "
    "for mAP). Confusion matrix in Ultralytics uses conf=0.25 and IoU-match 0.45 (see a6).",
    f"Paper M1 row equals the LAST epoch of the 2023 run: {match}. That run used Ultralytics "
    f"{out['baseline_2023_ultralytics_version']} with optimizer=auto -> {out['baseline_2023_optimizer_actually_used']}, "
    "NOT 'SGD momentum 0.937, lr0 0.01' as Table II states.",
    "colab_ablation.py hard-codes GFLOPs (258.1 baseline / 258.5 attention) instead of measuring them.",
]
for f_ in out["findings"]:
    print(" -", f_)
save_json("a5_training_settings", out)
