"""
A1 - How is CBAM attached, and does it really train / save / export?

Answers four yes/no questions with real tests (no guessing):
  Q1  Are the original YOLOv8x weights frozen, or trained together with CBAM?
  Q2  Are the CBAM parameters inside the optimizer?
  Q3  Is CBAM saved in, and loaded back from, a checkpoint?
  Q4  Does the CBAM model still export to ONNX (and give the same output)?
plus one extra trap we found:
  Q0  Does stock `YOLO.train()` keep the wrapper, or rebuild the network and drop it?

Output: revision/results/a1_attachment.json
Run:    python revision/a1_attachment_check.py      (CPU is fine, ~2 minutes)
"""
import sys
import tempfile
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import TARGET_LAYERS, banner, build_drone_yolov8x, save_json  # noqa: E402
from attention_modules import AttentionLayerWrapper, apply_attention_to_yolo  # noqa: E402

results = {}
dummy = torch.randn(1, 3, 640, 640)

# ----------------------------------------------------------------------
banner("Q0  Does stock YOLO.train() keep the CBAM wrapper?")
# ----------------------------------------------------------------------
# YOLO.train() calls  trainer.get_model(cfg=model.yaml, weights=model).
# We call that SAME function directly (with a tiny stand-in trainer object).
from ultralytics.models.yolo.detect import DetectionTrainer  # noqa: E402

wrapped = build_drone_yolov8x("cbam")
stand_in_trainer = SimpleNamespace(data={"nc": 1, "channels": 3}, set_model_names_for_load=lambda m: m)
rebuilt = DetectionTrainer.get_model(stand_in_trainer, cfg=wrapped.yaml, weights=wrapped, verbose=False)
kept = isinstance(rebuilt.model[12], AttentionLayerWrapper)
print(f"Wrapper still present after the stock rebuild? {kept}")
results["Q0_stock_train_keeps_wrapper"] = kept
results["Q0_explanation"] = (
    "Stock Ultralytics rebuilds the network from yolov8x.yaml at the start of train(), so a wrapper added "
    "beforehand is dropped unless the trainer is customised (your fork must patch this; "
    "attention_modules.make_attention_trainer does it here)."
    if not kept
    else "Wrapper survived the rebuild."
)

# ----------------------------------------------------------------------
banner("Q1  Frozen or trained together?")
# ----------------------------------------------------------------------
# Ultralytics' trainer freezes only layers listed in `freeze=` (your scripts do
# not pass it) plus the fixed '.dfl' layer.  We apply that same rule.
net = build_drone_yolov8x("cbam")
freeze_names = [".dfl"]  # default when freeze=None
for name, p in net.named_parameters():
    p.requires_grad = not any(f in name for f in freeze_names)
base_params = [(n, p) for n, p in net.named_parameters() if "attn_block" not in n]
cbam_params = [(n, p) for n, p in net.named_parameters() if "attn_block" in n]
trainable_base = sum(p.numel() for n, p in base_params if p.requires_grad)
frozen_base = sum(p.numel() for n, p in base_params if not p.requires_grad)
print(f"Original YOLOv8x params that will be trained: {trainable_base:,}")
print(f"Original YOLOv8x params that stay frozen     : {frozen_base:,} (only the fixed DFL conv)")
results["Q1_base_trainable_params"] = trainable_base
results["Q1_base_frozen_params"] = frozen_base
results["Q1_answer"] = "trained together (fine-tuned end-to-end); only the fixed 16-weight DFL conv is frozen"

# ----------------------------------------------------------------------
banner("Q2  Are CBAM parameters in the optimizer?")
# ----------------------------------------------------------------------
# (a) The CURRENT Ultralytics grouping rule (version installed here).
stand_in = SimpleNamespace(args=SimpleNamespace(lr0=0.01, momentum=0.937, nbs=64, batch=16, epochs=32, cos_lr=False))
opt = DetectionTrainer.build_optimizer(stand_in, net, name="SGD", lr=0.01, momentum=0.937, decay=5e-4)
in_opt = {id(p) for g in opt.param_groups for p in g["params"]}
missing_now = [n for n, p in cbam_params if id(p) not in in_opt]
print(f"CBAM tensors: {len(cbam_params)}  missing from optimizer (current Ultralytics): {missing_now}")

# (b) The OLD rule used by Ultralytics 8.0.x (the version of your 2023 baseline run).
#     It only collected attributes literally called `.weight` / `.bias`,
#     so a parameter called `gamma` would be left out.
old_groups = []
for m in net.modules():
    if hasattr(m, "bias") and isinstance(m.bias, nn.Parameter):
        old_groups.append(m.bias)
    if isinstance(m, nn.BatchNorm2d):
        old_groups.append(m.weight)
    elif hasattr(m, "weight") and isinstance(m.weight, nn.Parameter):
        old_groups.append(m.weight)
old_ids = {id(p) for p in old_groups}
missing_old = [n for n, p in cbam_params if id(p) not in old_ids]
print(f"Missing under the OLD 8.0.x rule: {missing_old}")
results["Q2_cbam_param_tensors"] = [n for n, _ in cbam_params]
results["Q2_missing_current_ultralytics"] = missing_now
results["Q2_missing_old_8_0_rule"] = missing_old

# ----------------------------------------------------------------------
banner("Q3  Saved in, and loaded from, a checkpoint?")
# ----------------------------------------------------------------------
net.eval()
with torch.no_grad():
    net.model[12].attn_block.gamma.fill_(0.5)  # give gamma a recognisable value
    net.model[15].attn_block.gamma.fill_(0.25)
    before = net(dummy)[0]
tmp = Path(tempfile.mkdtemp()) / "cbam_ckpt.pt"
torch.save({"model": deepcopy(net).half(), "train_args": {}}, tmp)  # same format Ultralytics uses
from ultralytics import YOLO  # noqa: E402

reloaded = YOLO(str(tmp)).model.float().eval()
with torch.no_grad():
    after = reloaded(dummy)[0]
results["Q3_wrapper_after_reload"] = isinstance(reloaded.model[12], AttentionLayerWrapper) and isinstance(
    reloaded.model[15], AttentionLayerWrapper
)
results["Q3_gamma_after_reload"] = [reloaded.model[12].attn_block.gamma.item(), reloaded.model[15].attn_block.gamma.item()]
results["Q3_max_output_diff_fp16_roundtrip"] = (before - after).abs().max().item()
print({k: v for k, v in results.items() if k.startswith("Q3")})

# ----------------------------------------------------------------------
banner("Q4  ONNX export")
# ----------------------------------------------------------------------
try:
    import onnxruntime as ort

    onnx_path = tmp.with_suffix(".onnx")
    net_fp32 = deepcopy(net).float().eval()
    torch.onnx.export(net_fp32, dummy, str(onnx_path), opset_version=17, input_names=["images"], dynamo=False)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {"images": dummy.numpy()})[0]
    with torch.no_grad():
        torch_out = net_fp32(dummy)[0].numpy()
    diff = float(abs(onnx_out - torch_out).max())
    results["Q4_onnx_export_ok"] = True
    results["Q4_onnx_vs_torch_max_abs_diff"] = diff
    results["Q4_onnx_file_MB"] = round(onnx_path.stat().st_size / 1e6, 2)
    print(f"ONNX export OK, max |onnx - torch| = {diff:.3e}")
except Exception as e:  # keep going even if export fails
    results["Q4_onnx_export_ok"] = False
    results["Q4_error"] = repr(e)
    print("ONNX export failed:", e)

results["how_attached"] = (
    "Module replacement, not a PyTorch forward hook: model.model[12] and model.model[15] (C2f blocks) are each "
    "replaced by AttentionLayerWrapper(layer, ResidualAttentionBlock(CBAM)). Forward = layer(x), then "
    "y + gamma*(CBAM(y) - y) with gamma initialised to 0 (identity at start)."
)
results["note"] = (
    "Run on the reference re-implementation in revision/attention_modules.py, because your original "
    "ultralytics/nn/modules/cbam.py was not in the repository."
)
save_json("a1_attachment", results)
