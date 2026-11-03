import json
import os
import glob
import cv2
import numpy as np

json_path = r"c:\Users\Rohith Sivakumar\Desktop\Drone\scratch\validation_analysis.json"
with open(json_path, "r") as f:
    data = json.load(f)

print(f"Loaded {len(data)} evaluated image records from validation_analysis.json.")

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

print("\n" + "="*70)
print("  FIGURE 6(a) CANDIDATES (Baseline Missed, CBAM Detected Small Drone)")
print("="*70)
cand_6a = []
for d in data:
    if len(d['gt']) > 0 and len(d['base']) == 0 and len(d['cbam']) > 0:
        gt0 = d['gt'][0]
        cbam0 = d['cbam'][0]
        iou_c = compute_iou(gt0['xyxy'], cbam0['xyxy'])
        cand_6a.append((d, gt0['area'], gt0['px_w'], gt0['px_h'], cbam0['conf'], iou_c))

cand_6a.sort(key=lambda x: x[1]) # Smallest target area first
for d, area, pw, ph, conf, iou in cand_6a[:10]:
    print(f"File: {d['fn']} | Res: {d['w']}x{d['h']} | Target: {pw:.1f}x{ph:.1f} px ({area:.1f} px^2) | CBAM Conf: {conf:.4f} | IoU: {iou:.4f}")

print("\n" + "="*70)
print("  FIGURE 6(b) CANDIDATES (Baseline False Positive, Suppressed by CBAM)")
print("="*70)
cand_6b = []
for d in data:
    if len(d['base']) > len(d['gt']) and len(d['cbam']) <= len(d['gt']):
        # Find which base box is FP (iou with all gt < 0.1)
        fps = []
        for b in d['base']:
            max_gt_iou = max([compute_iou(b['xyxy'], g['xyxy']) for g in d['gt']], default=0.0)
            if max_gt_iou < 0.1:
                fps.append(b)
        if len(fps) > 0:
            cand_6b.append((d, fps, d['base'], d['cbam']))

for d, fps, base, cbam in cand_6b[:10]:
    fp_confs = [fp['conf'] for fp in fps]
    c_confs = [c['conf'] for c in cbam]
    print(f"File: {d['fn']} | GT Count: {len(d['gt'])} | Base Dets: {len(base)} (FP confs: {fp_confs}) | CBAM Dets: {len(cbam)} (Confs: {c_confs}) | EdgeDensity: {d['edge_density']:.4f}")

print("\n" + "="*70)
print("  FIGURE 6(c) CANDIDATES (High-clutter Foliage/Trees Background)")
print("="*70)
cand_6c = []
for d in data:
    if len(d['gt']) > 0 and len(d['base']) > 0 and len(d['cbam']) > 0:
        if d['edge_density'] > 0.045: # Complex dense texture/trees
            gt0 = d['gt'][0]
            b0 = d['base'][0]
            c0 = d['cbam'][0]
            iou_b = compute_iou(gt0['xyxy'], b0['xyxy'])
            iou_c = compute_iou(gt0['xyxy'], c0['xyxy'])
            cand_6c.append((d, d['edge_density'], b0['conf'], c0['conf'], iou_b, iou_c))

cand_6c.sort(key=lambda x: (x[3] - x[2]), reverse=True) # Max CBAM conf advantage
for d, ed, b_conf, c_conf, iou_b, iou_c in cand_6c[:10]:
    print(f"File: {d['fn']} | EdgeDensity: {ed:.4f} | Base Conf: {b_conf:.4f} (IoU: {iou_b:.4f}) | CBAM Conf: {c_conf:.4f} (IoU: {iou_c:.4f}) | Diff: {c_conf-b_conf:+.4f}")

print("\n" + "="*70)
print("  FIGURE 6(d) CANDIDATES (Low-contrast Overcast Sky Scene)")
print("="*70)
cand_6d = []
for d in data:
    if len(d['gt']) > 0 and len(d['cbam']) > 0:
        if d['contrast'] < 42 and d['brightness'] > 115:
            gt0 = d['gt'][0]
            c0 = d['cbam'][0]
            iou_c = compute_iou(gt0['xyxy'], c0['xyxy'])
            b0 = d['base'][0] if len(d['base']) > 0 else None
            iou_b = compute_iou(gt0['xyxy'], b0['xyxy']) if b0 else 0.0
            b_conf = b0['conf'] if b0 else 0.0
            cand_6d.append((d, d['contrast'], d['brightness'], b_conf, c0['conf'], iou_b, iou_c))

cand_6d.sort(key=lambda x: x[1]) # Lowest contrast first
for d, contrast, bright, b_conf, c_conf, iou_b, iou_c in cand_6d[:10]:
    print(f"File: {d['fn']} | Contrast: {contrast:.2f} | Brightness: {bright:.2f} | Base Conf: {b_conf:.4f} (IoU: {iou_b:.4f}) | CBAM Conf: {c_conf:.4f} (IoU: {iou_c:.4f})")

print("\n" + "="*70)
print("  FIGURE 7(a) CANDIDATES (Extreme sub-pixel distance: target area < 64 px^2)")
print("="*70)
cand_7a = []
for d in data:
    if len(d['gt']) > 0:
        for gt in d['gt']:
            if gt['area'] < 100: # <10x10 px target
                cbam_iou = max([compute_iou(gt['xyxy'], c['xyxy']) for c in d['cbam']], default=0.0)
                cbam_conf = max([c['conf'] for c in d['cbam']], default=0.0)
                cand_7a.append((d, gt, gt['area'], len(d['cbam']), cbam_conf, cbam_iou))

cand_7a.sort(key=lambda x: x[2]) # Smallest target first
for d, gt, area, c_cnt, c_conf, c_iou in cand_7a[:10]:
    print(f"File: {d['fn']} | Target Dims: {gt['px_w']:.1f}x{gt['px_h']:.1f} px ({area:.1f} px^2) | CBAM Dets: {c_cnt} | Conf: {c_conf:.4f} | IoU: {c_iou:.4f}")

print("\n" + "="*70)
print("  FIGURE 7 FAILURE CASES (Missed or Failed by CBAM)")
print("="*70)
cand_7_fails = []
for d in data:
    if len(d['gt']) > 0:
        for gt in d['gt']:
            iou_c = max([compute_iou(gt['xyxy'], c['xyxy']) for c in d['cbam']], default=0.0)
            if iou_c < 0.2: # Missed or poor localization
                cand_7_fails.append((d, gt, gt['area'], d['edge_density'], d['contrast'], d['blur_laplacian'], iou_c))

cand_7_fails.sort(key=lambda x: x[2])
for d, gt, area, ed, contrast, blur, iou_c in cand_7_fails[:15]:
    print(f"File: {d['fn']} | Target Area: {area:.1f} px^2 ({gt['px_w']:.1f}x{gt['px_h']:.1f} px) | EdgeDensity: {ed:.4f} | Contrast: {contrast:.2f} | Blur (LapVar): {blur:.1f} | CBAM IoU: {iou_c:.4f}")
