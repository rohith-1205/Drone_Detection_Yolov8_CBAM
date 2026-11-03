import os
import glob
import json
import torch
import cv2
import numpy as np

# Configure PyTorch CPU threads
num_cores = os.cpu_count() or 8
torch.set_num_threads(num_cores)
print(f"PyTorch using {num_cores} CPU threads.")

from ultralytics import YOLO

valid_dir = r"c:\Users\Rohith Sivakumar\Desktop\Drone\drone_dataset\valid"
img_dir = os.path.join(valid_dir, "images")
lbl_dir = os.path.join(valid_dir, "labels")

image_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
print(f"Total validation images found: {len(image_files)}")

cbam_weights = r"c:\Users\Rohith Sivakumar\Desktop\Drone\Drone-Detection-1\best.pt"
base_weights = r"c:\Users\Rohith Sivakumar\Desktop\Drone\Drone-Detection-1\yolov8x.pt"

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

# Run inference in chunks to monitor progress
os.environ["ATTENTION_MODE"] = "cbam"
model_cbam = YOLO(cbam_weights)

os.environ["ATTENTION_MODE"] = "none"
model_base = YOLO(base_weights)

print("Starting inference loop across all validation images...")

dataset_results = []
json_out_path = r"c:\Users\Rohith Sivakumar\Desktop\Drone\scratch\validation_analysis.json"

for idx, img_path in enumerate(image_files):
    fn = os.path.basename(img_path)
    lbl_path = os.path.join(lbl_dir, fn.replace(".jpg", ".txt"))
    
    img = cv2.imread(img_path)
    if img is None:
        continue
    h_img, w_img = img.shape[:2]
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    edges = cv2.Canny(gray, 100, 200)
    edge_density = float(np.mean(edges > 0))
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    
    # Read Ground Truths
    gt_boxes = []
    if os.path.exists(lbl_path):
        with open(lbl_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls_id, cx, cy, w, h = map(float, parts)
                    x_min = (cx - w / 2.0) * w_img
                    y_min = (cy - h / 2.0) * h_img
                    x_max = (cx + w / 2.0) * w_img
                    y_max = (cy + h / 2.0) * h_img
                    gt_boxes.append({
                        'class': int(cls_id),
                        'cx': cx, 'cy': cy, 'w': w, 'h': h,
                        'xyxy': [round(x_min, 2), round(y_min, 2), round(x_max, 2), round(y_max, 2)],
                        'px_w': round(w * w_img, 2),
                        'px_h': round(h * h_img, 2),
                        'area': round(w * w_img * h * h_img, 2)
                    })
                    
    # CBAM Model Inference
    os.environ["ATTENTION_MODE"] = "cbam"
    res_cbam = model_cbam(img_path, conf=0.15, verbose=False)[0]
    cbam_dets = []
    if res_cbam.boxes is not None and len(res_cbam.boxes) > 0:
        for b in res_cbam.boxes:
            box_xyxy = [round(x, 2) for x in b.xyxy[0].tolist()]
            conf = float(b.conf[0])
            cbam_dets.append({
                'xyxy': box_xyxy,
                'conf': round(conf, 4)
            })
            
    # Baseline Model Inference
    os.environ["ATTENTION_MODE"] = "none"
    res_base = model_base(img_path, conf=0.15, verbose=False)[0]
    base_dets = []
    if res_base.boxes is not None and len(res_base.boxes) > 0:
        for b in res_base.boxes:
            box_xyxy = [round(x, 2) for x in b.xyxy[0].tolist()]
            conf = float(b.conf[0])
            base_dets.append({
                'xyxy': box_xyxy,
                'conf': round(conf, 4)
            })
            
    # Calculate best IoUs for each GT
    for gt in gt_boxes:
        gt_b = gt['xyxy']
        gt['iou_base'] = max([compute_iou(gt_b, bd['xyxy']) for bd in base_dets], default=0.0)
        gt['iou_cbam'] = max([compute_iou(gt_b, cd['xyxy']) for cd in cbam_dets], default=0.0)

    record = {
        'fn': fn,
        'path': img_path,
        'lbl_path': lbl_path,
        'w': w_img,
        'h': h_img,
        'brightness': round(brightness, 2),
        'contrast': round(contrast, 2),
        'edge_density': round(edge_density, 4),
        'blur_laplacian': round(laplacian_var, 2),
        'gt': gt_boxes,
        'cbam': cbam_dets,
        'base': base_dets
    }
    dataset_results.append(record)
    
    if (idx + 1) % 25 == 0 or (idx + 1) == len(image_files):
        print(f"Processed {idx + 1}/{len(image_files)} images...")
        with open(json_out_path, "w") as f:
            json.dump(dataset_results, f, indent=2)

print(f"Dataset evaluation finished! Total records: {len(dataset_results)}")
