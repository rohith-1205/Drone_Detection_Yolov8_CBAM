"""
A4 - Unrounded GFLOPs for the baseline and every attention variant.

Two independent tools are used so the numbers can be cross-checked:
  * thop    (what Ultralytics uses internally)  -> counts MACs (multiply-adds)
  * fvcore  (Facebook)                           -> also counts MACs  [optional]

CONVENTION (write this in the paper):
  Ultralytics reports "GFLOPs" = 2 x MACs / 1e9   (1 MAC = 1 multiply + 1 add).
  thop does NOT count element-wise ops (sigmoid, multiply by the attention map,
  pooling).  We therefore ALSO count those by hand so the attention cost is not
  under-reported.

Output: revision/results/a4_gflops.json
Run:    python revision/a4_gflops.py
"""
import sys
from copy import deepcopy
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import IMGSZ, TARGET_LAYERS, banner, build_drone_yolov8x, save_json  # noqa: E402

import thop  # noqa: E402

banner(f"A4  GFLOPs at {IMGSZ}x{IMGSZ}, batch 1")


def thop_macs(net):
    """Multiply-accumulate operations counted by thop for one 640x640 image."""
    x = torch.zeros(1, 3, IMGSZ, IMGSZ)
    macs, _ = thop.profile(deepcopy(net).eval(), inputs=(x,), verbose=False)
    return macs


def elementwise_ops(net):
    """Hand count of the element-wise work thop ignores, inside the attention blocks only."""
    total = 0
    for idx in TARGET_LAYERS:
        layer = net.model[idx]
        if not hasattr(layer, "attn_block"):
            continue
        att = layer.attn_block.attention
        c = layer.layer.cv2.conv.out_channels
        h = w = IMGSZ // (16 if idx == 12 else 8)
        n = c * h * w
        if hasattr(att, "channel") and not isinstance(att.channel, torch.nn.Identity):
            total += 2 * n + n  # avg+max pooling reads, then x * Mc
        if hasattr(att, "spatial") and not isinstance(att.spatial, torch.nn.Identity):
            total += 2 * n + n  # channel avg+max, then x * Ms
        total += 3 * n  # residual mix: x + gamma*(a - x)
    return total


results = {"convention": "GFLOPs = 2 * MACs / 1e9 (Ultralytics convention); thop ignores element-wise ops",
           "input": f"1x3x{IMGSZ}x{IMGSZ}", "variants": {}}
base_macs = None
for mode in ("none", "channel", "spatial", "cbam", "se", "coord"):
    net = build_drone_yolov8x(mode, pretrained=False)
    macs = thop_macs(net)
    if mode == "none":
        base_macs = macs
    extra_elem = elementwise_ops(net)
    row = {
        "thop_MACs": int(macs),
        "GFLOPs_ultralytics_convention": 2 * macs / 1e9,
        "added_GFLOPs_vs_baseline": 2 * (macs - base_macs) / 1e9,
        "added_percent_vs_baseline": 100 * (macs - base_macs) / base_macs,
        "elementwise_ops_not_in_thop": extra_elem,
        "GFLOPs_including_elementwise": (2 * macs + extra_elem) / 1e9,
    }
    results["variants"][mode] = row
    print(f"{mode:8s} {row['GFLOPs_ultralytics_convention']:.4f} GFLOPs  "
          f"(+{row['added_GFLOPs_vs_baseline']:.4f}, +{row['added_percent_vs_baseline']:.4f} %)  "
          f"| incl. element-wise: {row['GFLOPs_including_elementwise']:.4f}")

# Cross-check with Ultralytics' own model.info() helper.
from ultralytics.utils.torch_utils import get_flops  # noqa: E402

results["ultralytics_get_flops_baseline"] = get_flops(build_drone_yolov8x("none", pretrained=False), IMGSZ)
results["ultralytics_get_flops_cbam"] = get_flops(build_drone_yolov8x("cbam", pretrained=False), IMGSZ)
results["ultralytics_get_flops_note"] = (
    "Ultralytics get_flops()/model.info() profiles a small stride-sized image (e.g. 32x32) and multiplies by "
    "(640/32)^2 = 400. The CBAM channel MLP works on 1x1 pooled vectors, so its cost does NOT grow with image size; "
    "the x400 extrapolation therefore over-counts it (~+0.1 GFLOPs). The direct 640x640 thop/fvcore numbers above "
    "are the correct ones. This is the likely origin of the paper's '258.5 -> 258.6'."
)

# Optional second tool.
try:
    from fvcore.nn import FlopCountAnalysis

    x = torch.zeros(1, 3, IMGSZ, IMGSZ)
    for mode in ("none", "cbam"):
        fl = FlopCountAnalysis(build_drone_yolov8x(mode, pretrained=False).eval(), x)
        fl.unsupported_ops_warnings(False)
        results["variants"][mode]["fvcore_MACs"] = int(fl.total())
except ImportError:
    results["fvcore"] = "not installed (pip install fvcore) - thop result used"

results["paper_claims"] = {"baseline": 258.5, "cbam": 258.6, "increase_percent": 0.0387}
save_json("a4_gflops", results)
