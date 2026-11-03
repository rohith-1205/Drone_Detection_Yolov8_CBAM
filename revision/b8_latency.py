"""
B8 - Runtime benchmark: ms per image, FPS, peak memory, for baseline vs CBAM.

Protocol (write this in the paper):
  * batch size 1, input 1x3x640x640, network forward pass only (no NMS, no pre/post-processing),
    Conv+BN fused as Ultralytics does for inference
  * GPU: FP16, 50 warm-up runs per model, then 300 timed runs, torch.cuda.synchronize() around each run
  * CPU: FP32 (FP16 is not a meaningful CPU setting), 5 warm-up + 60 timed runs
  * INTERLEAVED timing: baseline, CBAM, baseline, CBAM, ... so that any slow-down of the
    machine affects both models equally (a first sequential attempt on the shared cloud CPU
    gave +13 % in one run and -4 % in the next, i.e. pure noise)
  * we report the MEDIAN (robust to hiccups), mean +- std, and the median of the PAIRED differences
  * Latency depends on the architecture, not the weight values, so the drone-sized (nc=1)
    YOLOv8x built from yolov8x.pt is used.

Output: revision/results/b8_latency.json   (one entry per device, so CPU and GPU runs are both kept)
Run:    python revision/b8_latency.py
"""
import platform
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import IMGSZ, banner, build_drone_yolov8x, load_json, save_json  # noqa: E402


def time_once(net, x, is_gpu):
    """Milliseconds for ONE forward pass."""
    if is_gpu:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    net(x)
    if is_gpu:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000


def peak_gpu_memory_mb(net, x):
    """Peak GPU memory (MB) for one forward pass of a single model."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        net(x)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 2**20


def stats(times):
    median = statistics.median(times)
    return {"median_ms": median, "mean_ms": statistics.mean(times), "std_ms": statistics.pstdev(times),
            "fps_from_median": 1000 / median, "runs": len(times)}


def benchmark(models, warmup, runs, half):
    """models: dict name -> nn.Module.  Returns per-model stats + paired differences."""
    is_gpu = torch.cuda.is_available()
    device = torch.device("cuda:0" if is_gpu else "cpu")
    x = torch.zeros(1, 3, IMGSZ, IMGSZ, device=device)
    x = x.half() if half else x
    for name in models:
        models[name] = (models[name].half() if half else models[name]).to(device).eval()
    times = {name: [] for name in models}
    with torch.inference_mode():
        for _ in range(warmup):
            for net in models.values():
                net(x)
        for _ in range(runs):  # interleaved
            for name, net in models.items():
                times[name].append(time_once(net, x, is_gpu))
    result = {name: stats(t) for name, t in times.items()}
    if is_gpu:
        for name, net in models.items():
            result[name]["peak_gpu_memory_MB"] = peak_gpu_memory_mb(net, x)
    return result, times


if __name__ == "__main__":
    use_gpu = torch.cuda.is_available()
    device_name = (torch.cuda.get_device_name(0) if use_gpu
                   else f"CPU ({platform.processor() or platform.machine()}, {torch.get_num_threads()} threads)")
    half = use_gpu
    warmup, runs = (50, 300) if use_gpu else (5, 60)
    banner(f"B8  Latency on {device_name}  ({'FP16' if half else 'FP32'}, batch 1, {IMGSZ}x{IMGSZ}, interleaved)")

    models = {}
    for mode in ("none", "cbam"):
        net = build_drone_yolov8x(mode)
        net.fuse(verbose=False)
        models[mode] = net
    result, times = benchmark(models, warmup, runs, half)

    paired = [c - b for b, c in zip(times["none"], times["cbam"])]
    entry = {
        "device": device_name, "precision": "FP16" if half else "FP32", "batch": 1, "imgsz": IMGSZ,
        "scope": "network forward only (no NMS), Conv+BN fused, interleaved timing",
        "torch": torch.__version__, "models": result,
        "cbam_overhead_ms_median_paired": statistics.median(paired),
        "cbam_overhead_percent_of_baseline_median": 100 * statistics.median(paired) / result["none"]["median_ms"],
    }
    for name, s in result.items():
        print(f"{name:5s}: median {s['median_ms']:.2f} ms ({s['fps_from_median']:.2f} FPS), "
              f"mean {s['mean_ms']:.2f} +- {s['std_ms']:.2f}, peak GPU MB {s.get('peak_gpu_memory_MB')}")
    print(f"CBAM overhead (median of paired differences): {entry['cbam_overhead_ms_median_paired']:+.2f} ms "
          f"({entry['cbam_overhead_percent_of_baseline_median']:+.2f} %)")

    previous = load_json("b8_latency") or {}
    previous.pop("_meta", None)
    previous = {k: v for k, v in previous.items() if isinstance(v, dict) and "models" in v and "interleaved" in v.get("scope", "")}
    previous[device_name] = entry
    save_json("b8_latency", previous)
