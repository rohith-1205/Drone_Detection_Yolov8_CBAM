import os
import glob
import json
import torch
import cv2
import numpy as np
from ultralytics import YOLO

# 1. Setup paths
valid_dir = r"c:\Users\Rohith Sivakumar\Desktop\Drone\drone_dataset\valid"
img_dir = os.path.join(valid_dir, "images")
lbl_dir = os.path.join(valid_dir, "labels")

image_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
print(f"Total validation images found: {len(image_files)}")

cbam_weights = r"c:\Users\Rohith Sivakumar\Desktop\Drone\Drone-Detection-1\best.pt"
base_weights = r"c:\Users\Rohith Sivakumar\Desktop\Drone\Drone-Detection-1\best.pt"

# Determine device and half precision settings
has_cuda = torch.cuda.is_available()
device = 0 if has_cuda else "cpu"
half = True if has_cuda else False

print(f"Using device: {device}, half={half}")

# IoU matching helper
def compute_iou(boxA, boxB):
    # box format: [x1, y1, x2, y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return float(iou)

def match_gt_preds(gt_boxes, pred_boxes, iou_thresh=0.5):
    # gt_boxes: list of [x1, y1, x2, y2]
    # pred_boxes: list of [x1, y1, x2, y2, conf]
    if len(gt_boxes) == 0:
        return 0, len(pred_boxes), 0, [] # tp, fp, fn, matched_ious
    if len(pred_boxes) == 0:
        return 0, 0, len(gt_boxes), [0.0]*len(gt_boxes)
        
    # Sort predictions by confidence descending
    sorted_preds = sorted(pred_boxes, key=lambda x: x[4], reverse=True)
    
    gt_matched = [False] * len(gt_boxes)
    matched_ious = [0.0] * len(gt_boxes)
    tp = 0
    fp = 0
    
    for p in sorted_preds:
        p_box = p[:4]
        best_iou = 0.0
        best_gt_idx = -1
        for g_idx, g_box in enumerate(gt_boxes):
            if not gt_matched[g_idx]:
                iou = compute_iou(p_box, g_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = g_idx
        if best_iou >= iou_thresh and best_gt_idx != -1:
            gt_matched[best_gt_idx] = True
            matched_ious[best_gt_idx] = best_iou
            tp += 1
        else:
            fp += 1
            
    fn = sum(1 for m in gt_matched if not m)
    return tp, fp, fn, matched_ious

# Step 1: Run inference
print("\n[Step 1] Running CBAM model inference...")
os.environ["ATTENTION_MODE"] = "cbam"
model_cbam = YOLO(cbam_weights)
results_cbam = model_cbam.predict(
    image_files,
    imgsz=640,
    conf=0.25,
    iou=0.7,
    half=half,
    device=device,
    stream=False,
    save=False,
    plots=False,
    verbose=False
)

print("\n[Step 1] Running Baseline model inference...")
os.environ["ATTENTION_MODE"] = "none"
model_base = YOLO(base_weights)
results_base = model_base.predict(
    image_files,
    imgsz=640,
    conf=0.25,
    iou=0.7,
    half=half,
    device=device,
    stream=False,
    save=False,
    plots=False,
    verbose=False
)

# Step 2: Process and match predictions
print("\n[Step 2] Processing images and matching GT with predictions...")
records = []

for idx, img_path in enumerate(image_files):
    filename = os.path.basename(img_path)
    lbl_path = os.path.join(lbl_dir, filename.replace(".jpg", ".txt"))
    
    img = cv2.imread(img_path)
    if img is None:
        continue
    orig_h, orig_w = img.shape[:2]
    
    # Image statistics
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    std_dev = float(np.std(gray)) # contrast
    edges = cv2.Canny(gray, 100, 200)
    edge_density = float(np.mean(edges > 0))
    
    # GT Boxes in pixel coords [x1, y1, x2, y2]
    gt_boxes = []
    gt_sizes_px = [] # [w, h]
    gt_max_sides = []
    touches_border = []
    
    if os.path.exists(lbl_path):
        with open(lbl_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls_id, cx, cy, w, h = map(float, parts)
                    x1 = (cx - w / 2.0) * orig_w
                    y1 = (cy - h / 2.0) * orig_h
                    x2 = (cx + w / 2.0) * orig_w
                    y2 = (cy + h / 2.0) * orig_h
                    gt_boxes.append([round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)])
                    pw = w * orig_w
                    ph = h * orig_h
                    gt_sizes_px.append([round(pw, 2), round(ph, 2)])
                    gt_max_sides.append(max(pw, ph))
                    
                    # Border/edge check (normalized coords close to border < 0.03 or > 0.97)
                    border_touch = (cx - w/2.0 < 0.03 or cy - h/2.0 < 0.03 or cx + w/2.0 > 0.97 or cy + h/2.0 > 0.97)
                    touches_border.append(border_touch)
                    
    # Baseline preds
    base_preds = []
    r_b = results_base[idx]
    if r_b.boxes is not None and len(r_b.boxes) > 0:
        for b in r_b.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            c = float(b.conf[0])
            base_preds.append([round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2), round(c, 4)])
            
    # CBAM preds
    cbam_preds = []
    r_c = results_cbam[idx]
    if r_c.boxes is not None and len(r_c.boxes) > 0:
        for b in r_c.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            c = float(b.conf[0])
            cbam_preds.append([round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2), round(c, 4)])
            
    # Match predictions to GT with IoU >= 0.5
    b_tp, b_fp, b_fn, b_ious = match_gt_preds(gt_boxes, base_preds, iou_thresh=0.5)
    c_tp, c_fp, c_fn, c_ious = match_gt_preds(gt_boxes, cbam_preds, iou_thresh=0.5)
    
    record = {
        'original_name': filename,
        'path': img_path,
        'orig_width': orig_w,
        'orig_height': orig_h,
        'gt': gt_boxes,
        'baseline': base_preds,
        'cbam': cbam_preds,
        'baseline_tp_fp_fn': [b_tp, b_fp, b_fn],
        'cbam_tp_fp_fn': [c_tp, c_fp, c_fn],
        'b_ious': b_ious,
        'c_ious': c_ious,
        'gt_sizes_px': gt_sizes_px,
        'gt_max_sides': gt_max_sides,
        'touches_border': touches_border,
        'blur_score': round(blur_score, 2),
        'std_dev': round(std_dev, 2),
        'edge_density': round(edge_density, 4)
    }
    records.append(record)

print(f"Total processed records: {len(records)}")

# Save temp evaluation json
with open(r"c:\Users\Rohith Sivakumar\Desktop\Drone\scratch\eval_full_records.json", "w") as f:
    json.dump(records, f, indent=2)

print("Saved eval_full_records.json")
