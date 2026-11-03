"""
train_utils.py - ONE function that trains a model on the scene split and
measures everything the reviewers asked for.  Used by c11, c12 and c13.

    run_experiment(name="yolov8x_cbam_s0", weights="yolov8x.pt", attention="cbam", seed=0)

What it does, step by step:
  1. trains on  splits/scene_split/train.txt  (best.pt is chosen on val.txt)
  2. evaluates best.pt ONCE on the untouched  test.txt
  3. counts parameters and GFLOPs (thop, direct 640x640 profiling)
  4. times the network (batch 1; FP16 on GPU)
  5. saves everything to  revision/results/c_runs/<name>.json

Settings are the ones in colab_ablation.py / Table II, so every model is trained
under the same conditions: 32 epochs, 640 px, batch 16, SGD (lr0 0.01,
momentum 0.937, weight decay 5e-4), default Ultralytics augmentation, AMP.
Exception: RT-DETR uses AdamW lr0 1e-4 (its standard recipe), because SGD at
0.01 is known to diverge for transformer detectors; this is logged in its JSON.
"""
import json
import sys
from copy import deepcopy
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import IMGSZ, RESULTS_DIR, WEIGHTS_DIR, make_split_yaml  # noqa: E402

RUNS_DIR = RESULTS_DIR / "c_runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_SETTINGS = dict(epochs=32, imgsz=IMGSZ, batch=16, optimizer="SGD", lr0=0.01, momentum=0.937,
                      weight_decay=0.0005, amp=True, deterministic=True, plots=True)

# Tiny settings for a quick "does the pipeline run?" test on a CPU.
SMOKE_SETTINGS = dict(epochs=1, imgsz=320, batch=2, fraction=0.02, workers=0, plots=False)


def result_path(name, smoke=False):
    folder = RESULTS_DIR / "smoke" if smoke else RUNS_DIR
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{name}.json"


def run_experiment(name, weights, attention=None, layers=(12, 15), seed=0, family="yolo", smoke=False):
    """Train + test one configuration.  Returns the result dict (or a 'skipped' dict)."""
    out_file = result_path(name, smoke)
    if out_file.exists():  # re-use only FINISHED runs; a 'skipped' record is retried
        cached = json.loads(out_file.read_text())
        if cached.get("status") == "done":
            print(f"[cached] {name} -> {out_file}")
            return cached

    record = {"name": name, "weights": str(weights), "attention": attention or "none", "layers": list(layers),
              "seed": seed, "family": family, "smoke_test": smoke}

    # ---- 0. refuse to burn days of CPU time on a real run -----------------
    if not torch.cuda.is_available() and not smoke:
        record.update(status="skipped", reason="no CUDA GPU available; a 32-epoch x-scale detector run "
                      "(about 26 h each, extrapolated from the measured CPU smoke test: 1.5 s per iteration at 320 px, batch 2 -> x32 work at 640 px, batch 16, 61 iterations x 32 epochs) is not feasible on CPU")
        print(f"[SKIPPED] {name}: {record['reason']}")
        out_file.write_text(json.dumps(record, indent=2))
        return record

    weights_path = Path(weights)
    if not weights_path.exists():
        weights_path = WEIGHTS_DIR / weights_path.name
    data = make_split_yaml("scene_split")
    settings = dict(TRAIN_SETTINGS)
    if family == "rtdetr":
        settings.update(optimizer="AdamW", lr0=0.0001)
    if smoke:
        settings.update(SMOKE_SETTINGS)
    record["train_settings"] = settings

    # ---- 1. train ------------------------------------------------------
    import attention_modules  # noqa: F401  (makes CBAM checkpoints loadable)
    from ultralytics import RTDETR, YOLO

    model = RTDETR(str(weights_path)) if family == "rtdetr" else YOLO(str(weights_path))
    trainer = None
    if attention and attention != "none":
        trainer = attention_modules.make_attention_trainer(attention, tuple(layers))
    model.train(data=data, seed=seed, project=str(RESULTS_DIR / "c_train_runs"), name=name, exist_ok=True,
                trainer=trainer, **settings)
    save_dir = Path(model.trainer.save_dir)
    best = save_dir / "weights" / "best.pt"
    record["best_checkpoint"] = str(best)

    # ---- 2. test (only once, on the untouched test split) ----------------
    tested = (RTDETR if family == "rtdetr" else YOLO)(str(best))
    if attention and attention != "none":  # prove the attention really is inside the trained checkpoint
        record["attention_in_checkpoint"] = all(
            isinstance(tested.model.model[i], attention_modules.AttentionLayerWrapper) for i in layers)
        record["gamma_after_training"] = [float(tested.model.model[i].attn_block.gamma) for i in layers]
    m = tested.val(data=data, split="test", imgsz=settings["imgsz"], batch=16, conf=0.001, iou=0.7,
                   project=str(RESULTS_DIR / "c_test_runs"), name=name, exist_ok=True, plots=not smoke)
    record["test"] = {"precision": float(m.box.mp), "recall": float(m.box.mr),
                      "mAP50": float(m.box.map50), "mAP50_95": float(m.box.map)}

    # ---- 3. size and cost ------------------------------------------------
    import thop

    net = deepcopy(tested.model).float().eval()
    record["params"] = sum(p.numel() for p in net.parameters())
    try:
        macs, _ = thop.profile(deepcopy(net), inputs=(torch.zeros(1, 3, IMGSZ, IMGSZ),), verbose=False)
        record["GFLOPs"] = 2 * macs / 1e9
    except Exception as e:
        record["GFLOPs_error"] = repr(e)

    # ---- 4. latency ------------------------------------------------------
    from b8_latency import benchmark

    gpu = torch.cuda.is_available()
    lat, _ = benchmark({"m": net}, warmup=50 if gpu else 2, runs=300 if gpu else 5, half=gpu)
    record["latency"] = {**lat["m"], "device": torch.cuda.get_device_name(0) if gpu else "CPU",
                         "precision": "FP16" if gpu else "FP32"}

    record["status"] = "done"
    out_file.write_text(json.dumps(record, indent=2, default=str))
    print(f"[saved] {out_file}")
    return record


def mean_std(values):
    """Mean and (sample) standard deviation of a list of numbers."""
    import statistics

    return (statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0)
