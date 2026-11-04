"""
C13b - Cost of each CBAM PLACEMENT (no training needed).

Reviewer 1 asked why both fusion points (Layer 12 AND Layer 15) are needed.
Accuracy for L12-only / L15-only needs GPU training (c13_placement_ablation.py),
but the COST of each placement depends only on the architecture, so it is
measured here on CPU:

    placements:  none | CBAM @ L12 | CBAM @ L15 | CBAM @ L12+L15
    measured:    parameters, GFLOPs (thop, 640x640, 2xMACs), CPU latency (interleaved)

Output: revision/results/c13b_placement_cost.json
Run:    python revision/c13b_placement_cost.py
"""
import statistics
import sys
from copy import deepcopy
from pathlib import Path

import thop
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import IMGSZ, banner, build_drone_yolov8x, count_params, save_json  # noqa: E402
from b8_latency import benchmark  # noqa: E402

PLACEMENTS = {"none": ("none", (12, 15)), "L12": ("cbam", (12,)), "L15": ("cbam", (15,)), "L12+L15": ("cbam", (12, 15))}

banner("C13b  Parameters / GFLOPs / latency for each CBAM placement")
nets, out = {}, {"placements": {}}
for name, (mode, layers) in PLACEMENTS.items():
    net = build_drone_yolov8x(mode, target_layers=layers)
    macs, _ = thop.profile(deepcopy(net).eval(), inputs=(torch.zeros(1, 3, IMGSZ, IMGSZ),), verbose=False)
    out["placements"][name] = {"attention": mode, "layers": list(layers) if mode != "none" else [],
                               "params": count_params(net), "GFLOPs": 2 * macs / 1e9}
    net.fuse(verbose=False)
    nets[name] = net

base = out["placements"]["none"]
for row in out["placements"].values():
    row["added_params"] = row["params"] - base["params"]
    row["added_GFLOPs"] = row["GFLOPs"] - base["GFLOPs"]

# CPU latency, all four models timed interleaved so machine noise hits them equally.
gpu = torch.cuda.is_available()
lat, times = benchmark(nets, warmup=50 if gpu else 3, runs=300 if gpu else 40, half=gpu)
for name in PLACEMENTS:
    row = out["placements"][name]
    row["median_ms"] = lat[name]["median_ms"]
    row["std_ms"] = lat[name]["std_ms"]
    row["overhead_ms_paired_median"] = statistics.median(c - b for b, c in zip(times["none"], times[name]))
out["latency_device"] = torch.cuda.get_device_name(0) if gpu else f"CPU, {torch.get_num_threads()} threads, FP32"

for name, r in out["placements"].items():
    print(f"{name:8s} params {r['params']:,} (+{r['added_params']:,})  GFLOPs {r['GFLOPs']:.4f} (+{r['added_GFLOPs']:.4f})  "
          f"median {r['median_ms']:.2f} ms (paired +{r['overhead_ms_paired_median']:.2f} ms, std {r['std_ms']:.1f})")
out["accuracy"] = "needs GPU training: python revision/c13_placement_ablation.py"
save_json("c13b_placement_cost", out)
