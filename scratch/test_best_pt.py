import os
import glob
from ultralytics import YOLO

best_pt = r"c:\Users\Rohith Sivakumar\Desktop\Drone\Drone-Detection-1\best.pt"
valid_dir = r"c:\Users\Rohith Sivakumar\Desktop\Drone\drone_dataset\valid\images"
images = sorted(glob.glob(os.path.join(valid_dir, "*.jpg")))[:30]

print("Testing root best.pt (136 MB YOLOv8x weights)...")

for mode in ["cbam", "none", "channel", "spatial"]:
    os.environ["ATTENTION_MODE"] = mode
    model = YOLO(best_pt)
    total_dets = 0
    confs = []
    for img_p in images:
        res = model(img_p, conf=0.01, verbose=False)[0] # conf=0.01 to see raw predictions
        if res.boxes is not None and len(res.boxes) > 0:
            total_dets += len(res.boxes)
            confs.extend([float(b.conf[0]) for b in res.boxes])
    max_c = max(confs) if confs else 0.0
    avg_c = sum(confs)/len(confs) if confs else 0.0
    print(f"Mode: {mode:7s} | Total Dets (conf>0.01): {total_dets:3d} | Max Conf: {max_c:.4f} | Avg Conf: {avg_c:.4f}")
