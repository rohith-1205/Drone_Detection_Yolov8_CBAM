"""
common.py - small helpers shared by every revision script.

Keeping paths in ONE place means you only edit this file when you move
the project to another computer (for example Google Colab with a GPU).
"""
import json
import os
import platform
import sys
from datetime import datetime
from pathlib import Path

# ----------------------------------------------------------------------
# 1. Folder layout
# ----------------------------------------------------------------------
REVISION_DIR = Path(__file__).resolve().parent          # .../revision
PROJECT_DIR = REVISION_DIR.parent                       # repo root
RESULTS_DIR = REVISION_DIR / "results"                  # every script writes here
FIGURES_DIR = REVISION_DIR / "figures"                  # regenerated paper figures
TABLES_DIR = REVISION_DIR / "tables"                    # auto-generated LaTeX tables
DATASET_DIR = PROJECT_DIR / "drone_dataset"             # train/ and valid/ folders

for folder in (RESULTS_DIR, FIGURES_DIR, TABLES_DIR):
    folder.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# 2. Weights.  Override any of these with environment variables, e.g.
#       CBAM_WEIGHTS=/content/drive/MyDrive/best.pt python a6_validate.py
# ----------------------------------------------------------------------
WEIGHTS_DIR = Path(os.environ.get("WEIGHTS_DIR", "/home/user/weights"))
PRETRAINED_YOLOV8X = Path(os.environ.get("PRETRAINED_YOLOV8X", WEIGHTS_DIR / "yolov8x.pt"))

# The *trained drone* checkpoints used in the paper.  They were NOT in the
# repository when this revision was prepared (see CHANGELOG.md), so the
# default paths below usually do not exist.  Point them at your files.
BASELINE_WEIGHTS = Path(os.environ.get("BASELINE_WEIGHTS", WEIGHTS_DIR / "drone_baseline_best.pt"))
CBAM_WEIGHTS = Path(os.environ.get("CBAM_WEIGHTS", WEIGHTS_DIR / "drone_cbam_best.pt"))

# The two neck layers the paper attaches CBAM to.
TARGET_LAYERS = (12, 15)

# Image size used everywhere in the paper.
IMGSZ = 640


def save_json(name, data):
    """Save `data` to revision/results/<name>.json and also print it."""
    data = dict(data)
    data.setdefault("_meta", {})
    data["_meta"].update(
        {
            "script": Path(sys.argv[0]).name,
            "created": datetime.now().isoformat(timespec="seconds"),
            "python": platform.python_version(),
            "machine": platform.processor() or platform.machine(),
        }
    )
    try:  # library versions are useful for the paper ("Ultralytics x.y.z")
        import torch
        import ultralytics

        data["_meta"]["torch"] = torch.__version__
        data["_meta"]["ultralytics"] = ultralytics.__version__
        data["_meta"]["device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    except Exception:
        pass

    # RESULTS_TAG=pipeline_test keeps test runs apart from the real results
    tag = os.environ.get("RESULTS_TAG")
    path = RESULTS_DIR / ("smoke" if tag else "") / (f"{name}_{tag}.json" if tag else f"{name}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))
    print(f"\n[saved] {path}")
    return path


def load_json(name):
    """Read revision/results/<name>.json, or return None if it does not exist."""
    path = RESULTS_DIR / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


def skip(name, reason, **extra):
    """Record that a task was skipped (and why) instead of crashing."""
    print(f"\n[SKIPPED] {name}: {reason}")
    return save_json(name, {"status": "skipped", "reason": reason, **extra})


def build_drone_yolov8x(mode="none", target_layers=TARGET_LAYERS, pretrained=True):
    """Build the 1-class (drone) YOLOv8x exactly the way Ultralytics training does:

    1. create the network from yolov8x.yaml with nc=1 (one class: drone),
    2. copy in the COCO-pretrained yolov8x.pt weights (all layers whose shapes match),
    3. optionally wrap `target_layers` with attention (mode = cbam/channel/spatial/se/coord).

    The *values* of the weights do not change shapes, parameter counts, GFLOPs
    or latency, so this model is valid for those measurements even without
    your trained drone checkpoints.
    """
    from ultralytics.nn.tasks import DetectionModel
    from attention_modules import apply_attention_to_yolo

    net = DetectionModel("yolov8x.yaml", nc=1, verbose=False)
    if pretrained:
        from ultralytics.nn.tasks import load_checkpoint

        coco_model, _ = load_checkpoint(str(PRETRAINED_YOLOV8X))
        net.load(coco_model, verbose=False)
    return apply_attention_to_yolo(net, mode=mode, target_layers=target_layers)


def make_split_yaml(split_name="scene_split"):
    """Create an Ultralytics data YAML for a split made by c10_make_split.py.

    The split files list paths RELATIVE to drone_dataset/ (portable between
    computers).  Ultralytics needs absolute paths, so we write absolute copies
    for THIS computer into revision/results/_runtime/ and return the YAML path.
    """
    src = REVISION_DIR / "splits" / split_name
    runtime = RESULTS_DIR / "_runtime" / split_name
    runtime.mkdir(parents=True, exist_ok=True)
    for part in ("train", "val", "test"):
        lines = [l for l in (src / f"{part}.txt").read_text().splitlines() if l.strip()]
        (runtime / f"{part}.txt").write_text("\n".join(str(DATASET_DIR / l) for l in lines) + "\n")
    yaml_path = runtime / "data.yaml"
    yaml_path.write_text(
        f"path: {runtime}\ntrain: {runtime / 'train.txt'}\nval: {runtime / 'val.txt'}\n"
        f"test: {runtime / 'test.txt'}\nnames:\n  0: drone\n"
    )
    return str(yaml_path)


def count_params(module):
    """Total number of parameters (weights + biases) in a PyTorch module."""
    return sum(p.numel() for p in module.parameters())


def banner(text):
    """Print a big, easy-to-spot heading in the terminal."""
    line = "=" * 70
    print(f"\n{line}\n  {text}\n{line}")
