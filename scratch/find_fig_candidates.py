import os
import glob
import torch
import cv2
import numpy as np
from ultralytics import YOLO

valid_dir = r"c:\Users\Rohith Sivakumar\Desktop\Drone\drone_dataset\valid"
img_dir = os.path.join(valid_dir, "images")
lbl_dir = os.path.join(valid_dir, "labels")

image_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
print(f"Total validation images found: {len(image_files)}")

# Weights
cbam_weights = r"c:\Users\Rohith Sivakumar\Desktop\Drone\Drone-Detection-1\best.pt"
base_weights = r"c:\Users\Rohith Sivakumar\Desktop\Drone\Drone-Detection-1\yolov8x.pt"

# Load models
os.environ["ATTENTION_MODE"] = "cbam"
model_cbam = YOLO(cbam_weights)

os.environ["ATTENTION_MODE"] = "none"
model_base = YOLO(base_weights)

results_summary = []

for img_path in image_files:
    filename = os.path.basename(img_path)
    lbl_path = os.path.join(lbl_dir, filename.replace(".jpg", ".txt"))
    
    # Read GT bounding boxes
    gt_boxes = []
    if os.path.exists(lbl_path):
        with open(lbl_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls, cx, cy, w, h = map(float, parts)
                    gt_boxes.append((cx, cy, w, h))
                    
    # CBAM inference
    os.environ["ATTENTION_MODE"] = "cbam"
    res_cbam = model_cbam(img_path, conf=0.25, verbose=False)[0]
    boxes_cbam = res_cbam.boxes
    
    # Base inference
    os.environ["ATTENTION_MODE"] = "none"
    res_base = model_base(img_path, conf=0.25, verbose=False)[0]
    boxes_base = res_base.boxes
    
    n_gt = len(gt_boxes)
    n_cbam = len(boxes_cbam)
    n_base = len(boxes_base)
    
    results_summary.append({
        'filename': filename,
        'path': img_path,
        'gt_count': n_gt,
        'gt_boxes': gt_boxes,
        'cbam_count': n_cbam,
        'cbam_boxes': boxes_cbam,
        'base_count': n_base,
        'base_boxes': boxes_base
    })

print("Inference completed for all validation images.")

# Analyze differences
print("\n--- Summary of Differences ---")
missed_by_base_found_by_cbam = []
false_pos_base_suppressed_cbam = []
different_confidence = []
no_detections_both = []

for item in results_summary:
    fn = item['filename']
    gt = item['gt_count']
    c_cbam = item['cbam_count']
    c_base = item['base_count']
    
    if gt > 0 and c_base == 0 and c_cbam > 0:
        missed_by_base_found_by_cbam.append(item)
    elif c_base > gt and c_cbam <= gt:
        false_pos_base_suppressed_cbam.append(item)
    elif gt > 0 and c_base == 0 and c_cbam == 0:
        no_detections_both.append(item)
    elif c_base > 0 and c_cbam > 0:
        conf_cbam = float(item['cbam_boxes'].conf[0]) if len(item['cbam_boxes'].conf) > 0 else 0
        conf_base = float(item['base_boxes'].conf[0]) if len(item['base_boxes'].conf) > 0 else 0
        different_confidence.append((item, conf_base, conf_cbam))

print(f"1. GT present, missed by base, detected by CBAM: {len(missed_by_base_found_by_cbam)}")
for x in missed_by_base_found_by_cbam[:10]:
    conf_c = [round(float(c), 3) for c in x['cbam_boxes'].conf]
    print(f"   - {x['filename']}: GT={x['gt_count']}, CBAM confs={conf_c}")

print(f"\n2. False positive in Base, suppressed in CBAM: {len(false_pos_base_suppressed_cbam)}")
for x in false_pos_base_suppressed_cbam[:10]:
    conf_b = [round(float(c), 3) for c in x['base_boxes'].conf]
    conf_c = [round(float(c), 3) for c in x['cbam_boxes'].conf]
    print(f"   - {x['filename']}: GT={x['gt_count']}, Base confs={conf_b}, CBAM confs={conf_c}")

print(f"\n3. Hard cases missed by both (candidates for Fig 7 failure cases): {len(no_detections_both)}")
for x in no_detections_both[:10]:
    gt_sizes = [round(w*h*640*640, 1) for (cx, cy, w, h) in x['gt_boxes']]
    print(f"   - {x['filename']}: GT={x['gt_count']}, GT box area (px^2)={gt_sizes}")
