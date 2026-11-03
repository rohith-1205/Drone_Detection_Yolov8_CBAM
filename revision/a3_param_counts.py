"""
A3 - Exact parameter counts: baseline, each attention block, and the total.

Also checks that the channel-attention MLP has a ReLU between W0 and W1,
and works the numbers out by hand so you can see where they come from.

Output: revision/results/a3_param_counts.json
Run:    python revision/a3_param_counts.py
"""
import sys
from pathlib import Path

import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import TARGET_LAYERS, banner, build_drone_yolov8x, count_params, save_json  # noqa: E402

banner("A3  Parameter counts (1-class YOLOv8x, as trained on the drone dataset)")

baseline = count_params(build_drone_yolov8x("none", pretrained=False))
print(f"Baseline YOLOv8x (nc=1): {baseline:,} parameters")

out = {"baseline_params": baseline, "variants": {}}
for mode in ("channel", "spatial", "cbam", "se", "coord"):
    net = build_drone_yolov8x(mode, pretrained=False)
    per_layer = {}
    for idx in TARGET_LAYERS:
        block = net.model[idx].attn_block
        channels = net.model[idx].layer.cv2.conv.out_channels
        per_layer[f"layer_{idx}"] = {
            "channels": channels,
            "attention_params": count_params(block.attention),
            "gamma_params": block.gamma.numel(),
            "total": count_params(block),
        }
    total = count_params(net)
    out["variants"][mode] = {"per_layer": per_layer, "total_params": total, "added_params": total - baseline,
                             "added_percent": 100 * (total - baseline) / baseline}
    print(f"{mode:8s}: total {total:,}  (+{total - baseline:,}, +{100 * (total - baseline) / baseline:.4f} %)  {per_layer}")

# ---------- ReLU check -------------------------------------------------
cbam_net = build_drone_yolov8x("cbam", pretrained=False)
mlp = cbam_net.model[12].attn_block.attention.channel.mlp
layer_types = [type(m).__name__ for m in mlp]
out["channel_mlp_layers"] = layer_types
out["channel_mlp_has_relu_between_W0_W1"] = isinstance(mlp[1], nn.ReLU)
print("Channel MLP layers:", layer_types)

# ---------- Hand calculation (r = 16, no MLP bias, 7x7 conv on 2 maps, no bias) ----------
def hand(c, r=16, k=7):
    return {"W0": c * (c // r), "W1": (c // r) * c, "spatial_conv": 2 * k * k, "gamma": 1,
            "total": 2 * c * (c // r) + 2 * k * k + 1}

out["hand_calculation"] = {
    "actual_channels_640_320": {"layer_12": hand(640), "layer_15": hand(320)},
    "paper_channels_512_256": {"layer_12": hand(512), "layer_15": hand(256)},
}
out["hand_calculation"]["actual_total"] = sum(v["total"] for v in out["hand_calculation"]["actual_channels_640_320"].values())
out["hand_calculation"]["paper_channel_total"] = sum(v["total"] for v in out["hand_calculation"]["paper_channels_512_256"].values())
print("Hand check, real channels:", out["hand_calculation"]["actual_total"],
      "| with the paper's 512/256:", out["hand_calculation"]["paper_channel_total"])

out["note"] = ("Counts are for the reference implementation in attention_modules.py (bias-free MLP, bias-free 7x7 conv, "
               "one gamma per block). If your original cbam.py used MLP biases, add C/r + C per block.")
save_json("a3_param_counts", out)
