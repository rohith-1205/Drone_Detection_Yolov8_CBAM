import os
import glob
import json
import torch
import cv2
import numpy as np
from ultralytics import YOLO

valid_dir = r"c:\Users\Rohith Sivakumar\Desktop\Drone\drone_dataset\valid"
img_dir = os.path.join(valid_dir, "images")
lbl_dir = os.path.join(valid_dir, "labels")

image_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
print(f"Total validation images found: {len(image_files)}")

cbam_weights = r"c:\Users\Rohith Sivakumar\Desktop\Drone\Drone-Detection-1\best.pt"
base_weights = r"c:\Users\Rohith Sivakumar\Desktop\Drone\Drone-Detection-1\yolov8x.pt"

# Check GPU availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# 1. Run CBAM model predictions
print("\n--- Running CBAM Model Predictions ---")
os.environ["ATTENTION_MODE"] = "cbam"
model_cbam = YOLO(cbam_weights)
results_cbam = model_cbam.predict(image_files, conf=0.15, device=device, verbose=False, batch=16)

# 2. Run Baseline model predictions
print("\n--- Running Baseline YOLOv8x Model Predictions ---")
os.environ["ATTENTION_MODE"] = "none"
model_base = YOLO(base_weights)
results_base = model_base.predict(image_files, conf=0.15, device=device, verbose=False, batch=16)

analyzed_data = []

for idx, img_path in enumerate(image_files):
    fn = os.path.basename(img_path)
    lbl_path = os.path.join(lbl_dir, fn.replace(".jpg", ".txt"))
    
    # Read Image & Compute Image Stats
    img = cv2.imread(img_path)
    h_img, w_img = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(np.mean(gray))
    std_contrast = float(np.std(gray))
    
    # Edge density for foliage/clutter measure
    edges = cv2.Canny(gray, 100, 200)
    edge_density = float(np.mean(edges > 0))
    
    # Read GT Bounding Boxes (normalized cx, cy, w, h)
    gt_boxes = []
    if os.path.exists(lbl_path):
        with open(lbl_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls, cx, cy, w, h = map(float, parts)
                    gt_boxes.append({
                        'cx': cx, 'cy': cy, 'w': w, 'h': h,
                        'px_w': w * w_img, 'px_h': h * h_img,
                        'area_px': w * w_img * h * h_img
                    })
                    
    # CBAM Detections
    res_c = results_cbam[idx]
    cbam_dets = []
    if res_c.boxes is not None and len(res_c.boxes) > 0:
        for b in res_c.boxes:
            box_xywhn = b.xywhn[0].tolist()
            box_xyxy = b.xyxy[0].tolist()
            conf = float(b.conf[0])
            cbam_dets.append({
                'xywhn': box_xywhn,
                'xyxy': box_xyxy,
                'conf': conf,
                'area_px': box_xywhn[2] * w_img * box_xywhn[3] * h_img
            })
            
    # Baseline Detections
    res_b = results_base[idx]
    base_dets = []
    if res_b.boxes is not None and len(res_b.boxes) > 0:
        for b in res_b.boxes:
            box_xywhn = b.xywhn[0].tolist()
            box_xyxy = b.xyxy[0].tolist()
            conf = float(b.conf[0])
            base_dets.append({
                'xywhn': box_xywhn,
                'xyxy': box_xyxy,
                'conf': conf,
                'area_px': box_xywhn[2] * w_img * box_xywhn[3] * h_img
            })
            
    entry = {
        'filename': fn,
        'path': img_path,
        'width': w_img,
        'height': h_img,
        'brightness': mean_brightness,
        'contrast': std_contrast,
        'edge_density': edge_density,
        'gt': gt_boxes,
        'gt_count': len(gt_boxes),
        'cbam_dets': cbam_dets,
        'cbam_count': len(cbam_dets),
        'base_dets': base_dets,
        'base_count': len(base_dets)
    }
    analyzed_data.append(entry)

out_json = r"c:\Users\Rohith Sivakumar\Desktop\Drone\scratch\validation_analysis.json"
with open(out_json, "w") as f:
    # Convert numpy floats/ints before serializing
    json.dump(analyzed_data, f, indent=2)

print(f"\nSaved analysis for {len(analyzed_data)} images to {out_json}")

# Categorize and Print Top Candidates for Figures 6 and 7

print("\n==========================================")
print("  FIG 6(a) CANDIDATES: Small Drone Missed by Base, Detected by CBAM")
print("==========================================")
fig6a = []
for d in analyzed_data:
    if d['gt_count'] > 0 and d['base_count'] == 0 and d['cbam_count'] > 0:
        min_gt_area = min([g['area_px'] for g in d['gt']])
        fig6a.append((d, min_gt_area))

fig6a.sort(key=lambda x: x[1])  # Smallest targets first
for d, area in fig6a[:10]:
    conf_c = [round(b['conf'], 3) for b in d['cbam_dets']]
    print(f"File: {d['filename']} | GT Count: {d['gt_count']} | Target Area: {area:.1f} px^2 | CBAM Confs: {conf_c}")

print("\n==========================================")
print("  FIG 6(b) CANDIDATES: False Positive in Base, Suppressed in CBAM")
print("==========================================")
fig6b = []
for d in analyzed_data:
    # GT is 0 or base has extra detections on background clutter, while CBAM has none or matches GT
    if d['base_count'] > d['gt_count'] and d['cbam_count'] <= d['gt_count']:
        base_confs = [round(b['conf'], 3) for b in d['base_dets']]
        cbam_confs = [round(b['conf'], 3) for b in d['cbam_dets']]
        fig6b.append((d, base_confs, cbam_confs))

for d, b_conf, c_conf in fig6b[:10]:
    print(f"File: {d['filename']} | GT: {d['gt_count']} | Base Dets: {d['base_count']} ({b_conf}) | CBAM Dets: {d['cbam_count']} ({c_conf}) | EdgeDensity: {d['edge_density']:.3f}")

print("\n==========================================")
print("  FIG 6(c) CANDIDATES: Foliage/Clutter Scene (CBAM Better Localization/Confidence)")
print("==========================================")
fig6c = []
for d in analyzed_data:
    if d['gt_count'] > 0 and d['base_count'] > 0 and d['cbam_count'] > 0:
        if d['edge_density'] > 0.05:  # High edge density = foliage/clutter
            max_c = max([b['conf'] for b in d['cbam_dets']])
            max_b = max([b['conf'] for b in d['base_dets']])
            diff = max_c - max_b
            fig6c.append((d, diff, max_b, max_c))

fig6c.sort(key=lambda x: x[1], reverse=True)
for d, diff, max_b, max_c in fig6c[:10]:
    print(f"File: {d['filename']} | GT: {d['gt_count']} | EdgeDensity: {d['edge_density']:.3f} | Base Conf: {max_b:.3f} | CBAM Conf: {max_c:.3f} (diff: {diff:+.3f})")

print("\n==========================================")
print("  FIG 6(d) CANDIDATES: Low Contrast Overcast Sky Scene")
print("==========================================")
fig6d = []
for d in analyzed_data:
    if d['gt_count'] > 0 and d['cbam_count'] > 0:
        # Low contrast & high brightness = overcast sky
        if d['contrast'] < 45 and d['brightness'] > 120:
            max_c = max([b['conf'] for b in d['cbam_dets']])
            max_b = max([b['conf'] for b in d['base_dets']]) if d['base_count'] > 0 else 0.0
            fig6d.append((d, d['contrast'], max_b, max_c))

fig6d.sort(key=lambda x: x[1])  # Lowest contrast first
for d, contrast, max_b, max_c in fig6d[:10]:
    print(f"File: {d['filename']} | GT: {d['gt_count']} | Brightness: {d['brightness']:.1f} | Contrast: {contrast:.1f} | Base Conf: {max_b:.3f} | CBAM Conf: {max_c:.3f}")

print("\n==========================================")
print("  FIG 7(a) CANDIDATES: Sub-pixel Micro Target (<8x8 px area <64 px^2)")
print("==========================================")
fig7a = []
for d in analyzed_data:
    if d['gt_count'] > 0:
        for g in d['gt']:
            if g['area_px'] < 100:  # extremely small target
                fig7a.append((d, g['area_px'], g['px_w'], g['px_h']))

fig7a.sort(key=lambda x: x[1])
for d, area, w, h in fig7a[:10]:
    print(f"File: {d['filename']} | Target Area: {area:.1f} px^2 ({w:.1f}x{h:.1f} px) | CBAM Count: {d['cbam_count']} | Base Count: {d['base_count']}")

print("\n==========================================")
print("  FIG 7 FAILURE CASE CANDIDATES (Missed or Low Conf by CBAM)")
print("==========================================")
fig7_missed = []
for d in analyzed_data:
    if d['gt_count'] > 0 and d['cbam_count'] == 0:
        min_gt_area = min([g['area_px'] for g in d['gt']])
        fig7_missed.append((d, min_gt_area))

fig7_missed.sort(key=lambda x: x[1])
print(f"Total GT images missed by CBAM: {len(fig7_missed)}")
for d, area in fig7_missed[:15]:
    gt_wh = [(round(g['px_w'],1), round(g['px_h'],1)) for g in d['gt']]
    print(f"File: {d['filename']} | GT Count: {d['gt_count']} | GT Target Dims (px): {gt_wh} | EdgeDensity: {d['edge_density']:.3f} | Contrast: {d['contrast']:.1f}")

