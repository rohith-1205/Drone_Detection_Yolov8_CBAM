import os
from ultralytics import YOLO

cbam_weights = r"c:\Users\Rohith Sivakumar\Desktop\Drone\Drone-Detection-1\best.pt"
base_weights = r"c:\Users\Rohith Sivakumar\Desktop\Drone\Drone-Detection-1\yolov8x.pt"

os.environ["ATTENTION_MODE"] = "cbam"
model_c = YOLO(cbam_weights)

os.environ["ATTENTION_MODE"] = "none"
model_b = YOLO(base_weights)

img_dir = r"c:\Users\Rohith Sivakumar\Desktop\Drone\drone_dataset\valid\images"
images = [
    os.path.join(img_dir, "0001.jpg"),
    os.path.join(img_dir, "0005.jpg"),
    os.path.join(img_dir, "0010.jpg"),
    os.path.join(img_dir, "0015.jpg"),
    os.path.join(img_dir, "foto00204.jpg"),
    os.path.join(img_dir, "foto00233.jpg")
]

for img_p in images:
    if not os.path.exists(img_p):
        continue
    fn = os.path.basename(img_p)
    res_c = model_c(img_p, conf=0.1, verbose=False)[0]
    res_b = model_b(img_p, conf=0.1, verbose=False)[0]
    
    c_confs = [round(float(b.conf[0]), 3) for b in res_c.boxes] if res_c.boxes is not None else []
    b_confs = [round(float(b.conf[0]), 3) for b in res_b.boxes] if res_b.boxes is not None else []
    
    print(f"Image: {fn}")
    print(f"  CBAM confs: {c_confs}")
    print(f"  Base confs: {b_confs}")
