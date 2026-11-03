"""
A2 - What are the REAL tensor shapes at every layer for a 640x640 input?

The paper (Table I, Sec. IV-D) says layer 12 = 512x40x40 and layer 15 = 256x80x80.
YOLOv8x multiplies channels by 1.25 (width multiple), so we measure it.

Output: revision/results/a2_tensor_shapes.json  (+ .csv with every layer)
Run:    python revision/a2_tensor_shapes.py
"""
import csv
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS_DIR, banner, build_drone_yolov8x, save_json  # noqa: E402

banner("A2  Tensor-shape trace of YOLOv8x (1 class) on a 1x3x640x640 input")

net = build_drone_yolov8x("cbam").eval()
shapes = {}


def remember_shape(index):
    """Return a small function that stores the output shape of layer `index`."""

    def hook(module, inputs, output):
        out = output[0] if isinstance(output, (list, tuple)) else output
        shapes[index] = {"module": type(module).__name__, "output_shape": list(out.shape)}

    return hook


# A forward hook is used here ONLY to read shapes; it does not change anything.
for i, layer in enumerate(net.model):
    layer.register_forward_hook(remember_shape(i))

with torch.no_grad():
    net(torch.zeros(1, 3, 640, 640))

rows = []
for i, info in shapes.items():
    inner = net.model[i].layer if info["module"] == "AttentionLayerWrapper" else net.model[i]
    rows.append({"layer": i, "module": info["module"], "inner": type(inner).__name__,
                 "from": str(net.model[i].f), "output_shape": "x".join(map(str, info["output_shape"][1:]))})
    print(f"layer {i:2d}  {rows[-1]['inner']:10s} from={rows[-1]['from']:>10s}  ->  {rows[-1]['output_shape']}")

with open(RESULTS_DIR / "a2_tensor_shapes.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

# The Detect head (layer 22) returns a list, so record what it is fed instead.
save_json(
    "a2_tensor_shapes",
    {
        "input": "1x3x640x640",
        "layer_12_output_CxHxW": rows[12]["output_shape"],
        "layer_15_output_CxHxW": rows[15]["output_shape"],
        "paper_claims": {"layer_12": "512x40x40", "layer_15": "256x80x80"},
        "detect_head_inputs_from_layers": net.model[22].f,
        "all_layers": rows,
    },
)
