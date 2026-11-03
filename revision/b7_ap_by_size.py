"""
B7 - AP for small / medium / large drones (COCO rules) at 640 px.

COCO size buckets use the box AREA in pixels:
    small  : area <  32*32
    medium : 32*32 <= area < 96*96
    large  : area >= 96*96
The reviewers asked for sizes "after resizing", so every box is measured in the
640x640 letter-boxed image (scale = 640 / max(width, height)).

Steps
  PART 1 (always): count how many validation boxes fall in each bucket.
  PART 2 (needs trained checkpoints): run Ultralytics val with save_json=True,
         rescale ground truth and predictions to the 640 frame, and run pycocotools
         to get AP, AP50, AP_small/medium/large and AR_small/medium/large.

Output: revision/results/b7_ap_by_size.json
Run:    CBAM_WEIGHTS=... BASELINE_WEIGHTS=... python revision/b7_ap_by_size.py
"""
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BASELINE_WEIGHTS, CBAM_WEIGHTS, DATASET_DIR, IMGSZ, RESULTS_DIR, banner, save_json  # noqa: E402

SMALL, MEDIUM = 32**2, 96**2


def bucket(area):
    return "small" if area < SMALL else "medium" if area < MEDIUM else "large"


def image_id_of(stem):
    """Same rule Ultralytics uses for image ids in predictions.json."""
    return int(stem) if stem.isnumeric() else stem


def build_ground_truth(split_dir):
    """COCO-format ground truth, already rescaled to the 640 letter-box frame.

    pycocotools needs integer ids, but Ultralytics names images by their file
    stem (an int for '0001', a string for 'pic_031').  We therefore give each
    image a simple number 0, 1, 2, ... and remember how to translate.
    """
    images, annotations, scale_of, id_map = [], [], {}, {}
    for number, img_path in enumerate(sorted((split_dir / "images").glob("*"))):
        w, h = Image.open(img_path).size
        s = IMGSZ / max(w, h)
        iid = number
        id_map[image_id_of(img_path.stem)] = number
        scale_of[iid] = s
        images.append({"id": iid, "file_name": img_path.name, "width": w * s, "height": h * s})
        label = split_dir / "labels" / f"{img_path.stem}.txt"
        for line in (label.read_text().splitlines() if label.exists() else []):
            if not line.strip():
                continue
            _, xc, yc, bw, bh = map(float, line.split()[:5])
            bw_px, bh_px = bw * w * s, bh * h * s
            x0, y0 = xc * w * s - bw_px / 2, yc * h * s - bh_px / 2
            annotations.append({"id": len(annotations) + 1, "image_id": iid, "category_id": 0,
                                "bbox": [x0, y0, bw_px, bh_px], "area": bw_px * bh_px, "iscrowd": 0})
    gt = {"images": images, "annotations": annotations, "categories": [{"id": 0, "name": "drone"}]}
    return gt, scale_of, id_map


banner("PART 1  Size distribution of validation boxes at 640 px")
gt, scale_of, id_map = build_ground_truth(DATASET_DIR / "valid")
counts = {"small": 0, "medium": 0, "large": 0}
for a in gt["annotations"]:
    counts[bucket(a["area"])] += 1
total = sum(counts.values())
out = {"val_boxes_per_bucket_at_640": counts,
       "val_boxes_percent": {k: 100 * v / total for k, v in counts.items()},
       "thresholds_px2": {"small_lt": SMALL, "large_ge": MEDIUM}, "models": {}}
print(out["val_boxes_per_bucket_at_640"], {k: round(v, 1) for k, v in out["val_boxes_percent"].items()})

banner("PART 2  COCO AP by size (needs trained checkpoints)")
gt_path = RESULTS_DIR / "_b7_gt_valid_640.json"
gt_path.write_text(json.dumps(gt))
for name, weights in (("baseline", BASELINE_WEIGHTS), ("cbam", CBAM_WEIGHTS)):
    if not Path(weights).exists():
        out["models"][name] = {"status": "skipped", "reason": f"checkpoint not found: {weights}"}
        print(f"[SKIPPED] {name}: checkpoint not found ({weights})")
        continue
    import attention_modules  # noqa: F401  (needed to un-pickle CBAM checkpoints)
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from ultralytics import YOLO

    data_yaml = RESULTS_DIR / "_drone_valid.yaml"
    data_yaml.write_text(f"path: {DATASET_DIR}\ntrain: train/images\nval: valid/images\nnames:\n  0: drone\n")
    v = YOLO(str(weights)).val(data=str(data_yaml), imgsz=IMGSZ, batch=16, conf=0.001, iou=0.7, save_json=True,
                               project=str(RESULTS_DIR / "b7_runs"), name=name, exist_ok=True, plots=False)
    preds = json.loads((Path(v.save_dir) / "predictions.json").read_text())
    for p in preds:  # move predictions into the same 640 frame as the ground truth
        p["image_id"] = id_map[p["image_id"]]
        s = scale_of[p["image_id"]]
        p["bbox"] = [c * s for c in p["bbox"]]
        p["category_id"] = 0
    coco_gt = COCO(str(gt_path))
    coco_dt = coco_gt.loadRes(preds) if preds else None
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.evaluate(), ev.accumulate(), ev.summarize()
    names = ["AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
             "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large"]
    out["models"][name] = {"weights": str(weights), **{n: float(x) for n, x in zip(names, ev.stats)}}
    print(name, out["models"][name])

out["note"] = "AP/AR = -1 means there are no ground-truth boxes in that bucket (pycocotools convention)."
save_json("b7_ap_by_size", out)
