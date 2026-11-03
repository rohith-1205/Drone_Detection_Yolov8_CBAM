"""
A6 - Re-validate the baseline and CBAM models, and explain the confusion matrix.

PART 1 (needs your trained drone checkpoints)
    Runs Ultralytics validation on drone_dataset/valid for BASELINE_WEIGHTS and
    CBAM_WEIGHTS (see common.py) and records P, R, mAP@0.5, mAP@0.5:0.95,
    the max-F1 confidence, and the confusion-matrix counts at conf 0.25.
    If a checkpoint is missing, that model is skipped (and logged).

PART 2 (always runs)
    Uses the RAW outputs that exist (the 2023 run's confusion_matrix.png counts,
    its training log, and the Ultralytics 8.0.141 source) to explain why the
    confusion matrix does not give the same P/R as Table III.

Output: revision/results/a6_validation.json
Run:    CBAM_WEIGHTS=path/to/cbam_best.pt BASELINE_WEIGHTS=path/to/base_best.pt python revision/a6_validate.py
"""
import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BASELINE_WEIGHTS, CBAM_WEIGHTS, DATASET_DIR, RESULTS_DIR, banner, save_json  # noqa: E402

out = {"validation_runs": {}}


def data_yaml():
    """Write a data.yaml with absolute paths (the one in drone_dataset/ uses '/drone_dataset')."""
    path = RESULTS_DIR / "_drone_valid.yaml"
    path.write_text(f"path: {DATASET_DIR}\ntrain: train/images\nval: valid/images\nnames:\n  0: drone\n")
    return str(path)


# ----------------------------------------------------------------------
banner("PART 1  Validation of the trained checkpoints")
# ----------------------------------------------------------------------
for name, weights in (("baseline", BASELINE_WEIGHTS), ("cbam", CBAM_WEIGHTS)):
    if not Path(weights).exists():
        msg = f"checkpoint not found at {weights} (set {name.upper()}_WEIGHTS=...)"
        print(f"[SKIPPED] {name}: {msg}")
        out["validation_runs"][name] = {"status": "skipped", "reason": msg}
        continue
    import attention_modules  # noqa: F401  (must be importable to un-pickle CBAM checkpoints)
    from ultralytics import YOLO

    model = YOLO(str(weights))
    metrics = model.val(data=data_yaml(), imgsz=640, batch=16, conf=0.001, iou=0.7, plots=True,
                        project=str(RESULTS_DIR / "a6_runs"), name=name, exist_ok=True)
    box = metrics.box
    from ultralytics.utils.metrics import smooth

    f1_curve = smooth(box.f1_curve.mean(0), 0.1)  # F1 at 1000 confidence steps, smoothed exactly as Ultralytics does
    conf_grid = np.linspace(0, 1, len(f1_curve))
    # Ultralytics >= 8.3 builds the confusion matrix at the validation conf (0.001), which floods it with
    # low-confidence boxes.  Ultralytics 8.0.141 (your paper) used conf 0.25, so we run a second pass at 0.25
    # ONLY to draw the matrix; the mAP numbers above come from the conf=0.001 pass.
    cm_metrics = model.val(data=data_yaml(), imgsz=640, batch=16, conf=0.25, iou=0.7, plots=True,
                           project=str(RESULTS_DIR / "a6_runs"), name=f"{name}_cm_conf0.25", exist_ok=True)
    cm = cm_metrics.confusion_matrix.matrix       # rows = predicted, cols = true; last = background
    tp, fp, fn = float(cm[0, 0]), float(cm[0, 1]), float(cm[1, 0])
    out["validation_runs"][name] = {
        "weights": str(weights),
        "precision_at_maxF1": float(box.mp), "recall_at_maxF1": float(box.mr),
        "max_F1_confidence": float(conf_grid[f1_curve.argmax()]),
        "mAP50": float(box.map50), "mAP50_95": float(box.map),
        "confusion_matrix_at_conf_0.25": {"TP": tp, "FP": fp, "FN": fn},
        "P_from_matrix": tp / (tp + fp) if tp + fp else None,
        "R_from_matrix": tp / (tp + fn) if tp + fn else None,
    }
    print(name, out["validation_runs"][name])

# ----------------------------------------------------------------------
banner("PART 2  Why the confusion matrix and Table III disagree")
# ----------------------------------------------------------------------
# Counts printed on the confusion-matrix PNG Ultralytics produced for the 2023 run
# (revision/evidence/baseline_2023_run/confusion_matrix.png) - identical to paper Fig. 5(a).
TP, FP, FN = 320, 48, 49
n_val_boxes = sum(len([l for l in open(f).read().splitlines() if l.strip()])
                  for f in glob.glob(str(DATASET_DIR / "valid" / "labels" / "*.txt"))
                  if not f.endswith("classes.txt"))
p_cm, r_cm = TP / (TP + FP), TP / (TP + FN)
out["confusion_matrix_analysis"] = {
    "counts_on_paper_fig5a": {"TP": TP, "FP": FP, "FN": FN},
    "validation_boxes_in_dataset": n_val_boxes,
    "TP_plus_FN": TP + FN,
    "precision_from_matrix_percent": 100 * p_cm,
    "recall_from_matrix_percent": 100 * r_cm,
    "paper_table3_M4_cbam": {"P": 91.30, "R": 88.75},
    "paper_table3_M1_baseline": {"P": 86.82, "R": 83.90},
    "ultralytics_8.0.141_confusion_matrix_threshold": {"conf": 0.25, "iou_match": 0.45},
    "ultralytics_P_R_operating_point": "P and R in the metrics table are read at the confidence that maximises the mean F1 "
                                       "over the validation run (validator conf=0.001, NMS IoU=0.7).",
    "finding_1": "The matrix (conf 0.25) and the table (max-F1 confidence) use different operating points, so their "
                 "P/R legitimately differ.",
    "finding_2": "More important: the Fig. 5(a) counts are IDENTICAL to the confusion_matrix.png of the 2023 plain "
                 "YOLOv8x run (no CBAM), and Fig. 5(b)'s legend reads 'all classes 0.892 mAP@0.5' = the baseline's "
                 "89.2 %, not CBAM's claimed 92.93 %. Fig. 5 therefore shows the BASELINE model.",
}
for k, v in out["confusion_matrix_analysis"].items():
    print(f"{k}: {v}")
save_json("a6_validation", out)
