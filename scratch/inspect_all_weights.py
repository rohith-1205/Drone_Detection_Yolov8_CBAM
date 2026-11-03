import os
import glob
from ultralytics import YOLO

weights_list = [
    (r"c:\Users\Rohith Sivakumar\Desktop\Drone\Drone-Detection-1\best.pt", "cbam", "root_best.pt"),
    (r"c:\Users\Rohith Sivakumar\Desktop\Drone\Drone-Detection-1\yolov8x.pt", "none", "yolov8x.pt"),
    (r"c:\Users\Rohith Sivakumar\Desktop\Drone\Drone-Detection-1\runs\detect\train\weights\best.pt", "cbam", "runs_train_best.pt"),
    (r"c:\Users\Rohith Sivakumar\Desktop\Drone\Drone-Detection-1\runs\detect\yolov8n_drone_cbam\weights\best.pt", "cbam", "runs_yolov8n_cbam.pt"),
]

valid_dir = r"c:\Users\Rohith Sivakumar\Desktop\Drone\drone_dataset\valid\images"
test_images = sorted(glob.glob(os.path.join(valid_dir, "*.jpg")))[:20]

for path, mode, label in weights_list:
    if not os.path.exists(path):
        continue
    os.environ["ATTENTION_MODE"] = mode
    model = YOLO(path)
    total_dets = 0
    max_conf = 0.0
    for img_p in test_images:
        res = model(img_p, conf=0.10, verbose=False)[0]
        if res.boxes is not None and len(res.boxes) > 0:
            total_dets += len(res.boxes)
            confs = [float(b.conf[0]) for b in res.boxes]
            max_conf = max(max_conf, max(confs))
    print(f"Weight: {label} ({path}) | Total detections in 20 imgs: {total_dets} | Max conf: {max_conf:.4f}")
