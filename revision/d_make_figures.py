"""
D - Regenerate the paper figures with LARGE, readable fonts (Reviewer 1 & 2).

  Fig. 2  training-label statistics          (from drone_dataset/train/labels - always possible)
  Fig. 4  training curves                    (from any Ultralytics results.csv)
  Fig. 5  confusion matrix + PR curve        (counts / curves from a validation run)
  Fig. 6  qualitative detections, NO track IDs (uses model.predict, needs CBAM weights)

Usage
  python revision/d_make_figures.py                      # Fig. 2, Fig. 4 and Fig. 5(a) from the raw files we have
  CBAM_WEIGHTS=best.pt python revision/d_make_figures.py # also Fig. 5 from a fresh val run and Fig. 6

Outputs go to revision/figures/*.pdf (vector, for LaTeX) and *.png (preview),
and a provenance file revision/results/d_figures.json says which data each figure used.
"""
import csv
import glob
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CBAM_WEIGHTS, DATASET_DIR, FIGURES_DIR, IMGSZ, REVISION_DIR, banner, save_json  # noqa: E402

# One place to control how big the text is.  IEEE column = 3.5 in wide,
# so 9-10 pt text after scaling is comfortably readable.
plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13, "xtick.labelsize": 12,
                     "ytick.labelsize": 12, "legend.fontsize": 12, "font.family": "DejaVu Sans",
                     "axes.spines.top": False, "axes.spines.right": False})
COLORS = {"main": "#1f6fb4", "accent": "#e4572e", "soft": "#9cc3e6"}
provenance = {}


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES_DIR / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] revision/figures/{name}.pdf")


# ----------------------------------------------------------------------
def fig2_label_statistics():
    """Fig. 2: (a) instances, (b) overlaid box shapes, (c) box centres, (d) width vs height."""
    from PIL import Image

    boxes, side_640 = [], []
    for img_path in sorted((DATASET_DIR / "train" / "images").glob("*")):
        label = DATASET_DIR / "train" / "labels" / f"{img_path.stem}.txt"
        if not label.exists():
            continue
        w_img, h_img = Image.open(img_path).size
        s = IMGSZ / max(w_img, h_img)  # letter-box scale to 640
        for line in label.read_text().splitlines():
            if line.strip():
                xc, yc, w, h = (float(v) for v in line.split()[1:5])
                boxes.append([xc, yc, w, h])
                side_640.append(np.sqrt(w * w_img * s * h * h_img * s))  # sqrt(area) in 640-px image
    b, side = np.array(boxes), np.array(side_640)
    fig, ax = plt.subplots(2, 2, figsize=(9, 8))
    ax[0, 0].bar(["drone"], [len(b)], color=COLORS["main"], width=0.5)
    ax[0, 0].set_title("(a) Training instances")
    ax[0, 0].set_ylabel("Number of boxes")
    ax[0, 0].text(0, len(b), f"{len(b):,}", ha="center", va="bottom", fontsize=14, fontweight="bold")
    # (b) box size at 640 px with the COCO small / medium / large borders
    ax[0, 1].hist(side, bins=40, color=COLORS["main"])
    for border in (32, 96):
        ax[0, 1].axvline(border, color=COLORS["accent"], ls="--", lw=2)
    n_s, n_m = int((side < 32).sum()), int(((side >= 32) & (side < 96)).sum())
    ax[0, 1].set(title=f"(b) Box size at 640 px\nS={n_s}  M={n_m}  L={len(side) - n_s - n_m}",
                 xlabel=r"$\sqrt{\mathrm{area}}$ (pixels)", ylabel="Number of boxes")
    h2 = ax[1, 0].hist2d(b[:, 0], b[:, 1], bins=30, cmap="Blues")
    ax[1, 0].set(title="(c) Box centres", xlabel="x (normalised)", ylabel="y (normalised)")
    fig.colorbar(h2[3], ax=ax[1, 0], label="boxes")
    h3 = ax[1, 1].hist2d(b[:, 2], b[:, 3], bins=30, cmap="Blues")
    ax[1, 1].set(title="(d) Box width vs height", xlabel="width (normalised)", ylabel="height (normalised)")
    fig.colorbar(h3[3], ax=ax[1, 1], label="boxes")
    fig.tight_layout()
    save(fig, "fig2_label_statistics")
    provenance["fig2"] = {"source": "drone_dataset/train/labels/*.txt", "n_boxes": len(b),
                          "train_boxes_small_medium_large_at_640": [n_s, n_m, len(side) - n_s - n_m]}


# ----------------------------------------------------------------------
def fig4_training_curves(runs):
    """Fig. 4: losses and validation metrics per epoch.  runs = [(label, results.csv path), ...]"""
    panels = [("train/box_loss", "Train box loss"), ("train/cls_loss", "Train cls loss"),
              ("val/box_loss", "Val box loss"), ("metrics/precision(B)", "Precision"),
              ("metrics/recall(B)", "Recall"), ("metrics/mAP50-95(B)", "mAP@0.5:0.95")]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
    for (label, path), color in zip(runs, (COLORS["main"], COLORS["accent"])):
        with open(path) as f:
            rows = [{k.strip(): float(v) for k, v in r.items()} for r in csv.DictReader(f)]
        epochs = [int(r["epoch"]) + 1 for r in rows]  # 1-based for readers
        for axis, (key, title) in zip(axes.flat, panels):
            axis.plot(epochs, [r[key] for r in rows], marker="o", ms=3, lw=2, color=color, label=label)
            axis.set(title=title, xlabel="Epoch")
            axis.grid(alpha=0.3)
        provenance.setdefault("fig4", []).append({"label": label, "source": str(path), "epochs": len(rows)})
    axes[0, 0].legend(frameon=False)
    fig.tight_layout()
    save(fig, "fig4_training_curves")


# ----------------------------------------------------------------------
def fig5a_confusion(tp, fp, fn, title, source):
    """Fig. 5(a): 2x2 confusion matrix with big numbers."""
    m = np.array([[tp, fp], [fn, 0]])
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(m, cmap="Blues")
    for (i, j), v in np.ndenumerate(m):
        if (i, j) != (1, 1):
            ax.text(j, i, f"{int(v)}", ha="center", va="center", fontsize=26, fontweight="bold",
                    color="white" if v > m.max() / 2 else "#0b3d91")
    ax.set_xticks([0, 1], ["drone", "background"])
    ax.set_yticks([0, 1], ["drone", "background"], rotation=90, va="center")
    ax.set(xlabel="True", ylabel="Predicted", title=title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    save(fig, "fig5a_confusion_matrix")
    provenance["fig5a"] = {"TP": tp, "FP": fp, "FN": fn, "source": source}


def fig5_from_validation(weights):
    """Fig. 5(a)+(b) from a fresh validation run of a checkpoint."""
    import attention_modules  # noqa: F401
    from ultralytics import YOLO

    data = REVISION_DIR / "results" / "_drone_valid.yaml"
    data.write_text(f"path: {DATASET_DIR}\ntrain: train/images\nval: valid/images\nnames:\n  0: drone\n")
    model = YOLO(str(weights))
    m = model.val(data=str(data), imgsz=IMGSZ, conf=0.001, iou=0.7, plots=False, verbose=False)
    recall_grid = m.box.px if hasattr(m.box, "px") else np.linspace(0, 1, 1000)
    prec = m.box.prec_values[0] if hasattr(m.box, "prec_values") else None
    cm = model.val(data=str(data), imgsz=IMGSZ, conf=0.25, iou=0.7, plots=False, verbose=False).confusion_matrix.matrix
    fig5a_confusion(int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), "YOLOv8-CBAM (conf 0.25)", str(weights))
    if prec is not None:
        fig, ax = plt.subplots(figsize=(5.2, 4.6))
        ax.plot(recall_grid, prec, lw=3, color=COLORS["main"], label=f"drone  mAP@0.5 = {m.box.map50:.3f}")
        ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Recall", ylabel="Precision", title="Precision-Recall")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower left", frameon=False)
        fig.tight_layout()
        save(fig, "fig5b_pr_curve")
        provenance["fig5b"] = {"source": str(weights), "mAP50": float(m.box.map50)}


# ----------------------------------------------------------------------
def fig6_qualitative(weights, images):
    """Fig. 6: detections WITHOUT track IDs (predict, not track)."""
    import attention_modules  # noqa: F401
    from ultralytics import YOLO

    model = YOLO(str(weights))
    for k, img in enumerate(images):
        r = model.predict(img, imgsz=IMGSZ, conf=0.25, iou=0.7, verbose=False)[0]
        plotted = r.plot(line_width=3, font_size=22)[:, :, ::-1]  # BGR -> RGB, labels = class + confidence only
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.imshow(plotted)
        ax.axis("off")
        save(fig, f"fig6{'ab'[k]}_detection")
        provenance[f"fig6{'ab'[k]}"] = {"image": str(img), "weights": str(weights),
                                        "confidences": [float(c) for c in r.boxes.conf]}


if __name__ == "__main__":
    banner("D  Regenerating figures")
    fig2_label_statistics()

    runs = [("YOLOv8x baseline (M1)", REVISION_DIR / "evidence" / "baseline_2023_run" / "results.csv")]
    cbam_csv = Path(CBAM_WEIGHTS).parent.parent / "results.csv"
    if cbam_csv.exists():
        runs.append(("YOLOv8x-CBAM (M4)", cbam_csv))
    fig4_training_curves(runs)

    if Path(CBAM_WEIGHTS).exists():
        fig5_from_validation(CBAM_WEIGHTS)
        fig6_qualitative(CBAM_WEIGHTS, sorted((DATASET_DIR / "valid" / "images").glob("*"))[:2])
    else:
        # Only raw output available: the 2023 BASELINE confusion matrix (same counts as the paper's Fig. 5a).
        fig5a_confusion(320, 48, 49, "Baseline YOLOv8x (conf 0.25)",
                        "evidence/baseline_2023_run/confusion_matrix.png (Ultralytics 8.0.141 output)")
        provenance["fig5b"] = provenance["fig6"] = {"status": "skipped",
                                                    "reason": f"CBAM checkpoint not found at {CBAM_WEIGHTS}"}
    save_json("d_figures", provenance)
