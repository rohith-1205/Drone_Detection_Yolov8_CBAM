import os
import glob
import json
import torch
import cv2
import numpy as np

# Maximize PyTorch CPU parallelism
num_cores = os.cpu_count() or 8
torch.set_num_threads(num_cores)
print(f"PyTorch configured to use {num_cores} CPU threads.", flush=True)

from ultralytics import YOLO

valid_dir = r"c:\Users\Rohith Sivakumar\Desktop\Drone\drone_dataset\valid"
img_dir = os.path.join(valid_dir, "images")
lbl_dir = os.path.join(valid_dir, "labels")

image_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
print(f"Total validation images found: {len(image_files)}", flush=True)

cbam_weights = r"c:\Users\Rohith Sivakumar\Desktop\Drone\Drone-Detection-1\best.pt"
base_weights = r"c:\Users\Rohith Sivakumar\Desktop\Drone\Drone-Detection-1\yolov8x.pt"

# Load models once
os.environ["ATTENTION_MODE"] = "cbam"
model_cbam = YOLO(cbam_weights)

os.environ["ATTENTION_MODE"] = "none"
model_base = YOLO(base_weights)

print("Models loaded successfully. Starting fast candidate search...", flush=True)

results = []

for idx, img_path in enumerate(image_files):
    fn = os.path.basename(img_path)
    lbl_path = os.path.join(lbl_dir, fn.replace(".jpg", ".txt"))
    
    # Read Ground Truths
    gt_boxes = []
    if os.path.exists(lbl_path):
        with open(lbl_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls, cx, cy, w, h = map(float, parts)
                    gt_boxes.append({'cx': cx, 'cy': cy, 'w': w, 'h': h})
                    
    # We are interested in images with ground truth or images with baseline predictions
    # Perform inference per image
    os.environ["ATTENTION_MODE"] = "cbam"
    res_c = model_cbam(img_path, conf=0.20, verbose=False)[0]
    
    os.environ["ATTENTION_MODE"] = "none"
    res_b = model_base(img_path, conf=0.20, verbose=False)[0]
    
    img = cv2.imread(img_path)
    h_img, w_img = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    edges = cv2.Canny(gray, 100, 200)
    edge_density = float(np.mean(edges > 0))
    
    # Extract bounding boxes
    cbam_boxes = []
    if res_c.boxes is not None and len(res_c.boxes) > 0:
        for box in res_c.boxes:
            cbam_boxes.append({
                'xywhn': box.xywhn[0].tolist(),
                'conf': float(box.conf[0])
            })
            
    base_boxes = []
    if res_b.boxes is not None and len(res_b.boxes) > 0:
        for box in res_b.boxes:
            base_boxes.append({
                'xywhn': box.xywhn[0].tolist(),
                'conf': float(box.conf[0])
            })
            
    rec = {
        'fn': fn,
        'path': img_path,
        'w': w_img, 'h': h_img,
        'brightness': brightness,
        'contrast': contrast,
        'edge_density': edge_density,
        'gt': gt_boxes,
        'cbam': cbam_boxes,
        'base': base_boxes
    }
    results.append(rec)
    
    if (idx + 1) % 25 == 0 or (idx + 1) == len(image_files):
        print(f"Processed {idx + 1}/{len(image_files)} images...", flush=True)

out_json = r"c:\Users\Rohith Sivakumar\Desktop\Drone\scratch\validation_analysis.json"
with open(out_json, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nCompleted! Results saved to {out_json}", flush=True)

# ----------------------------
# CATEGORIZE CANDIDATES
# ----------------------------

print("\n" + "="*50, flush=True)
print("  FIGURE 6(a) CANDIDATES (Small target missed by Base, detected by CBAM)", flush=True)
print("="*50, flush=True)
candidate_6a = []
for r in results:
    if len(r['gt']) > 0 and len(r['base']) == 0 and len(r['cbam']) > 0:
        # Measure target size
        min_gt_w = min([g['w'] for g in r['gt']]) * r['w']
        min_gt_h = min([g['h'] for g in r['gt']]) * r['h']
        area = min_gt_w * min_gt_h
        conf = max([b['conf'] for b in r['cbam']])
        candidate_6a.append((r['fn'], area, min_gt_w, min_gt_h, conf))

candidate_6a.sort(key=lambda x: x[1])
for fn, area, w, h in candidate_6a[:10]:
    print(f"File: {fn} | Target Dims: {w:.1f}x{h:.1f} px | Area: {area:.1f} px^2 | CBAM Conf: {conf:.3f}", flush=True)


print("\n" + "="*50, flush=True)
print("  FIGURE 6(b) CANDIDATES (False Positive in Base, Suppressed by CBAM)", flush=True)
print("="*50, flush=True)
candidate_6b = []
for r in results:
    if len(r['base']) > len(r['gt']) and len(r['cbam']) <= len(r['gt']):
        b_confs = [round(b['conf'], 3) for b in r['base']]
        c_confs = [round(b['conf'], 3) for b in r['cbam']]
        candidate_6b.append((r['fn'], len(r['gt']), len(r['base']), b_confs, len(r['cbam']), c_confs))

for fn, gt_c, base_c, b_confs, cbam_c, c_confs in candidate_6b[:10]:
    print(f"File: {fn} | GT: {gt_c} | Base Dets: {base_c} {b_confs} | CBAM Dets: {cbam_c} {c_confs}", flush=True)


print("\n" + "="*50, flush=True)
print("  FIGURE 6(c) CANDIDATES (Foliage/Tree Clutter Scene)", flush=True)
print("="*50, flush=True)
candidate_6c = []
for r in results:
    if len(r['gt']) > 0 and len(r['cbam']) > 0 and len(r['base']) > 0:
        if r['edge_density'] > 0.04:  # dense clutter
            max_c = max([b['conf'] for b in r['cbam']])
            max_b = max([b['conf'] for b in r['base']])
            candidate_6c.append((r['fn'], r['edge_density'], max_b, max_c, max_c - max_b))

candidate_6c.sort(key=lambda x: x[4], reverse=True)
for fn, ed, max_b, max_c, diff in candidate_6c[:10]:
    print(f"File: {fn} | EdgeDensity: {ed:.3f} | Base Conf: {max_b:.3f} | CBAM Conf: {max_c:.3f} | Diff: {diff:+.3f}", flush=True)


print("\n" + "="*50, flush=True)
print("  FIGURE 6(d) CANDIDATES (Low Contrast Overcast Sky Scene)", flush=True)
print("="*50, flush=True)
candidate_6d = []
for r in results:
    if len(r['gt']) > 0 and len(r['cbam']) > 0:
        if r['contrast'] < 45 and r['brightness'] > 110:
            max_c = max([b['conf'] for b in r['cbam']])
            max_b = max([b['conf'] for b in r['base']]) if len(r['base']) > 0 else 0.0
            candidate_6d.append((r['fn'], r['contrast'], r['brightness'], max_b, max_c))

candidate_6d.sort(key=lambda x: x[1])
for fn, contrast, brightness, max_b, max_c in candidate_6d[:10]:
    print(f"File: {fn} | Contrast: {contrast:.1f} | Brightness: {brightness:.1f} | Base Conf: {max_b:.3f} | CBAM Conf: {max_c:.3f}", flush=True)


print("\n" + "="*50, flush=True)
print("  FIGURE 7(a) CANDIDATES (Extreme sub-pixel distance target < 8x8 px)", flush=True)
print("="*50, flush=True)
candidate_7a = []
for r in results:
    if len(r['gt']) > 0:
        for g in r['gt']:
            tw = g['w'] * r['w']
            th = g['h'] * r['h']
            if tw < 12 and th < 12:
                c_conf = max([b['conf'] for b in r['cbam']]) if len(r['cbam']) > 0 else 0.0
                candidate_7a.append((r['fn'], tw*th, tw, th, len(r['cbam']), c_conf))

candidate_7a.sort(key=lambda x: x[1])
for fn, area, tw, th, c_count, c_conf in candidate_7a[:10]:
    print(f"File: {fn} | Target Dims: {tw:.1f}x{th:.1f} px ({area:.1f} px^2) | CBAM Dets: {c_count} (Conf: {c_conf:.3f})", flush=True)


print("\n" + "="*50, flush=True)
print("  FIGURE 7 FAILURE CASES (Missed by CBAM or difficult target)", flush=True)
print("="*50, flush=True)
candidate_7_fail = []
for r in results:
    if len(r['gt']) > 0 and len(r['cbam']) == 0:
        min_gt_w = min([g['w'] for g in r['gt']]) * r['w']
        min_gt_h = min([g['h'] for g in r['gt']]) * r['h']
        area = min_gt_w * min_gt_h
        candidate_7_fail.append((r['fn'], area, min_gt_w, min_gt_h, r['edge_density'], r['contrast']))

candidate_7_fail.sort(key=lambda x: x[1])
print(f"Total validation images with GT missed by CBAM: {len(candidate_7_fail)}", flush=True)
for fn, area, w, h, ed, contrast in candidate_7_fail[:15]:
    print(f"File: {fn} | Target Dims: {w:.1f}x{h:.1f} px ({area:.1f} px^2) | EdgeDensity: {ed:.3f} | Contrast: {contrast:.1f}", flush=True)

