#!/usr/bin/env bash
# =====================================================================
# run_all_gpu.sh - run EVERY revision step on a CUDA GPU (e.g. Google Colab).
#
# Usage (from the repository root):
#     export BASELINE_WEIGHTS=/path/to/baseline/best.pt   # optional, your trained M1 checkpoint
#     export CBAM_WEIGHTS=/path/to/cbam/best.pt           # optional, your trained M4 checkpoint
#     bash revision/run_all_gpu.sh
#
# Steps that already have a finished result are skipped automatically, so
# you can stop and restart this script (e.g. when a Colab session ends).
# Time on one A100: about 15 training runs x ~1 h.  On a T4: x3-4.
# =====================================================================
set -e
cd "$(dirname "$0")/.."

pip install -q ultralytics ultralytics-thop fvcore pycocotools imagehash onnx onnxruntime

# Pretrained weights for all detectors (downloaded once).
export WEIGHTS_DIR="${WEIGHTS_DIR:-$PWD/weights}"
mkdir -p "$WEIGHTS_DIR"
for m in yolov8x yolov5xu yolo11x rtdetr-x; do
  [ -f "$WEIGHTS_DIR/$m.pt" ] || curl -sSL -o "$WEIGHTS_DIR/$m.pt" \
      "https://github.com/ultralytics/assets/releases/download/v8.3.0/$m.pt"
done

echo "== Part A: facts check ==";               for s in a1_attachment_check a2_tensor_shapes a3_param_counts a4_gflops a5_training_settings a6_validate; do python revision/$s.py; done
echo "== Part B: existing weights ==";          for s in b7_ap_by_size b8_latency b9_near_duplicates; do python revision/$s.py; done
echo "== Part C: split + retraining ==";        python revision/c10_make_split.py
python revision/c12_train_seeds.py          # baseline + CBAM, seeds 0/1/2   (6 runs)
python revision/c13b_placement_cost.py      # params/GFLOPs per placement (no training)
python revision/c13_placement_ablation.py   # L12, L15, SE, CoordAtt        (4 new runs)
python revision/c11_train_sota.py           # YOLOv5x, YOLO11x, RT-DETR-X   (3 new runs)
echo "== Part D: figures, tables, paper ==";
# Use the new seed-0 CBAM run for the figures if you have no CBAM_WEIGHTS of your own.
export CBAM_WEIGHTS="${CBAM_WEIGHTS:-$PWD/revision/results/c_train_runs/yolov8x_cbam_s0/weights/best.pt}"
python revision/d_make_figures.py
python revision/d_fill_tables.py
(cd revision && pdflatex -interaction=nonstopmode paper_revised.tex >/dev/null && pdflatex -interaction=nonstopmode paper_revised.tex >/dev/null) \
  && echo "Compiled revision/paper_revised.pdf" || echo "pdflatex not available - compile paper_revised.tex on Overleaf"
