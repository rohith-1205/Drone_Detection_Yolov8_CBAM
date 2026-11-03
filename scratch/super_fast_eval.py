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
print(f"Total validation images: {len(image_files)}", flush=True)

cbam_weights = r"c:\Users\Rohith Sivakumar\Desktop\Drone\Drone-Detection-1\best.pt"
base_weights = r"c:\Users\Rohith Sivakumar\Desktop\Drone\Drone-Detection-1\yolov8x.pt"

# 1. Run CBAM model predictions in one single batch pass
print("\n[Step 1/3] Running CBAM model inference...", flush=True)
os.environ["ATTENTION_MODE"] = "cbam"
model_cbam = YOLO(cbam_weights)
results_cbam = model_cbam.predict(image_files, conf=0.15, verbose=False, batch=32)

# 2. Run Baseline model predictions in one single batch pass
print("\n[Step 2/3] Running Baseline model inference...", flush=True)
os.environ["ATTENTION_MODE"] = "none"
model_base = YOLO(base_weights)
results_base = model_base.predict(image_files, conf=0.15, verbose=False, batch=32)

# 3. Analyze results
print("\n[Step 3/3] Analyzing images and predictions...", flush=True)
all_data = []

for idx, img_path in enumerate(image_files):
    fn = os.path.basename(img_path)
    lbl_path = os.path.join(lbl_dir, fn.replace(".jpg", ".txt"))
    
    # Read image dimensions & compute image stats
    img = cv2.imread(img_path)
    h_img, w_img = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    edges = cv2.Canny(gray, 100, 200)
    edge_density = float(np.mean(edges > 0))
    
    # Ground Truths
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
                        'area': w * w_img * h * h_img
                    })
                    
    # CBAM Detections
    rc = results_cbam[idx]
    cbam_dets = []
    if rc.boxes is not None and len(rc.boxes) > 0:
        for b in rc.boxes:
            cbam_dets.append({
                'xywhn': b.xywhn[0].tolist(),
                'xyxy': b.xyxy[0].tolist(),
                'conf': float(b.conf[0])
            })
            
    # Base Detections
    rb = results_base[idx]
    base_dets = []
    if rb.boxes is not None and len(rb.boxes) > 0:
        for b in rb.boxes:
            base_dets.append({
                'xywhn': b.xywhn[0].tolist(),
                'xyxy': b.xyxy[0].tolist(),
                'conf': float(b.conf[0])
            })
            
    item = {
        'fn': fn,
        'path': img_path,
        'w': w_img, 'h': h_img,
        'brightness': brightness,
        'contrast': contrast,
        'edge_density': edge_density,
        'gt': gt_boxes,
        'cbam': cbam_dets,
        'base': base_dets
    }
    all_data.append(item)

# Save JSON
out_path = r"c:\Users\Rohith Sivakumar\Desktop\Drone\scratch\validation_analysis.json"
with open(out_path, "w") as f:
    json.dump(all_data, f, indent=2)

print(f"\nAnalysis saved to {out_path}", flush=True)

# -------------------------------------------------------------
# PRINT CATEGORIZED CANDIDATES FOR FIG 6 AND FIG 7
# -------------------------------------------------------------

print("\n" + "="*60, flush=True)
print("  FIG 6(a) CANDIDATES: Baseline missed small target, CBAM detected", flush=True)
print("="*60, flush=True)
cand_6a = []
for d in all_data:
    if len(d['gt']) > 0 and len(d['base']) == 0 and len(d['cbam']) > 0:
        min_area = min([g['area'] for g in d['gt']])
        conf = max([b['conf'] for b in d['cbam']])
        cand_6a.append((d['fn'], min_area, conf, d['gt'][0]['px_w'], d['gt'][0]['px_h']))

cand_6a.sort(key=lambda x: x[1])
for fn, area, conf, pw, ph in cand_6a[:10]:
    print(f"File: {fn} | Target Area: {area:.1f} px^2 ({pw:.1f}x{ph:.1f} px) | CBAM Conf: {conf:.3f}", flush=True)


print("\n" + "="*60, flush=True)
print("  FIG 6(b) CANDIDATES: False Positive in Baseline, Suppressed in CBAM", flush=True)
print("="*60, flush=True)
cand_6b = []
for d in all_data:
    if len(d['base']) > len(d['gt']) and len(d['cbam']) <= len(d['gt']):
        b_confs = [round(b['conf'], 3) for b in d['base']]
        c_confs = [round(b['conf'], 3) for b in d['cbam']]
        cand_6b.append((d['fn'], len(d['gt']), len(d['base']), b_confs, len(d['cbam']), c_confs, d['edge_density']))

for fn, gt_c, b_cnt, b_confs, c_cnt, c_confs, ed in cand_6b[:10]:
    print(f"File: {fn} | GT: {gt_c} | Base Dets: {b_cnt} {b_confs} | CBAM Dets: {c_cnt} {c_confs} | EdgeDensity: {ed:.3f}", flush=True)


print("\n" + "="*60, flush=True)
print("  FIG 6(c) CANDIDATES: High-clutter foliage/background scene", flush=True)
print("="*60, flush=True)
cand_6c = []
for d in all_data:
    if len(d['gt']) > 0 and len(d['cbam']) > 0 and len(d['base']) > 0:
        if d['edge_density'] > 0.04:
            max_c = max([b['conf'] for b in d['cbam']])
            max_b = max([b['conf'] for b in d['base']])
            cand_6c.append((d['fn'], d['edge_density'], max_b, max_c, max_c - max_b))

cand_6c.sort(key=lambda x: x[4], reverse=True)
for fn, ed, max_b, max_c, diff in cand_6c[:10]:
    print(f"File: {fn} | EdgeDensity: {ed:.3f} | Base Conf: {max_b:.3f} | CBAM Conf: {max_c:.3f} (diff: {diff:+.3f})", flush=True)


print("\n" + "="*60, flush=True)
print("  FIG 6(d) CANDIDATES: Low-contrast overcast sky scene", flush=True)
print("="*60, flush=True)
cand_6d = []
for d in all_data:
    if len(d['gt']) > 0 and len(d['cbam']) > 0:
        if d['contrast'] < 45 and d['brightness'] > 110:
            max_c = max([b['conf'] for b in d['cbam']])
            max_b = max([b['conf'] for b in d['base']]) if len(d['base']) > 0 else 0.0
            cand_6d.append((d['fn'], d['contrast'], d['brightness'], max_b, max_c))

cand_6d.sort(key=lambda x: x[1])
for fn, contrast, brightness, max_b, max_c in cand_6d[:10]:
    print(f"File: {fn} | Contrast: {contrast:.1f} | Brightness: {brightness:.1f} | Base Conf: {max_b:.3f} | CBAM Conf: {max_c:.3f}", flush=True)


print("\n" + "="*60, flush=True)
print("  FIG 7(a) CANDIDATES: Sub-pixel target (<8x8 px area < 64 px^2)", flush=True)
print("="*60, flush=True)
cand_7a = []
for d in all_data:
    if len(d['gt']) > 0:
        for g in d['gt']:
            if g['area'] < 100:
                c_conf = max([b['conf'] for b in d['cbam']]) if len(d['cbam']) > 0 else 0.0
                cand_7a.append((d['fn'], g['area'], g['px_w'], g['px_h'], len(d['cbam']), c_conf))

cand_7a.sort(key=lambda x: x[1])
for fn, area, pw, ph, c_cnt, c_conf in cand_7a[:10]:
    print(f"File: {fn} | Target: {pw:.1f}x{ph:.1f} px ({area:.1f} px^2) | CBAM Dets: {c_cnt} (Conf: {c_conf:.3f})", flush=True)


print("\n" + "="*60, flush=True)
print("  FIG 7 FAILURE CASES (Missed by CBAM)", flush=True)
print("="*60, flush=True)
cand_7_fail = []
for d in all_data:
    if len(d['gt']) > 0 and len(d['cbam']) == 0:
        min_area = min([g['area'] for g in d['gt']])
        pw = d['gt'][0]['px_w']
        ph = d['gt'][0]['px_h']
        cand_7_fail.append((d['fn'], min_area, pw, ph, d['edge_density'], d['contrast']))

cand_7_fail.sort(key=lambda x: x[1])
print(f"Total validation images with GT missed by CBAM: {len(cand_7_fail)}", flush=True)
for fn, area, pw, ph, ed, contrast in cand_7_fail[:15]:
    print(f"File: {fn} | Target: {pw:.1f}x{ph:.1f} px ({area:.1f} px^2) | EdgeDensity: {ed:.3f} | Contrast: {contrast:.1f}", flush=True)

