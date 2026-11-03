# YOLOv8-CBAM: An Attention-Guided Neck Feature Fusion Framework for Drone Detection

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-000000?style=for-the-badge&logo=ultralytics&logoColor=white)](https://github.com/ultralytics/ultralytics)
[![Accuracy](https://img.shields.io/badge/mAP%400.5-92.93%25-brightgreen?style=for-the-badge)](Drone-Detection-1/main.tex)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle%20Drone-blue?style=for-the-badge&logo=kaggle)](https://www.kaggle.com/datasets/muki2003/yolo-drone-detection-dataset)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](Drone-Detection-1/LICENSE)

An advanced deep learning object detection framework that integrates the **Convolutional Block Attention Module (CBAM)** into the **Path Aggregation Network (PANet)** neck of **YOLOv8**. Designed specifically for detecting small, fast-moving unmanned aerial vehicles (UAVs / drones) in complex outdoor environments with background clutter (clouds, foliage, birds).

---

## 📌 Key Highlights

- **Neck Feature Attention**: Sequentially applies **Channel Attention** and **Spatial Attention** directly after key feature-fusion blocks (**Layer 12** and **Layer 15**) in the PANet neck.
- **Superior Detection Performance**: Increases baseline YOLOv8x **mAP@0.5 from 89.22% to 92.93%** (+3.71% absolute improvement) and **mAP@0.5:0.95 from 53.76% to 57.10%**.
- **Negligible Computational Overhead**: Achieves peak accuracy with zero parameter growth (**68.2M**) and minimal extra FLOPs (+0.1 GFLOPs over 258.5 GFLOPs baseline).
- **Ablation Framework**: Built-in support for systematically testing four architectural modes: Baseline (`none`), Channel-only (`channel`), Spatial-only (`spatial`), and Full CBAM (`cbam`).
- **Multi-Hardware Support**: Native support for **NVIDIA CUDA GPUs**, **Intel Arc GPUs** (via `torch_directml`), and CPU fallback.
- **Interactive CLI & Inference Tools**: Automated scripts for dataset extraction, model verification, configurable training, and real-time video tracking.

---

## 🔬 Architecture & Methodology

Detecting drones in outdoor video streams is challenging because targets occupy very small pixel footprints and are easily obscured by sky texture or ground clutter.

```
                           ┌─────────────────────────┐
                           │      Input Image        │
                           └────────────┬────────────┘
                                        │
                               ┌────────┴────────┐
                               │  YOLOv8 Backbone│
                               └────────┬────────┘
                                        │
             ┌──────────────────────────┴──────────────────────────┐
             │                 PANet Feature Neck                  │
             │                                                     │
             │   ┌───────────────┐           ┌───────────────┐     │
             │   │ Fusion L12    │           │ Fusion L15    │     │
             │   └───────┬───────┘           └───────┬───────┘     │
             │           │                           │             │
             │   ┌───────▼───────┐           ┌───────▼───────┐     │
             │   │ CBAM Gate L12 │           │ CBAM Gate L15 │     │
             │   └───────┬───────┘           └───────┬───────┘     │
             └───────────┼───────────────────────────┼─────────────┘
                         │                           │
                         └──────────────┬────────────┘
                                        │
                               ┌────────▼────────┐
                               │  Detection Head │
                               └─────────────────┘
```

### Attention Mechanism Breakdown

1. **Channel Attention (What to focus on)**:
   Compresses spatial dimensions via adaptive average and max pooling, feeding features into a shared MLP ($r=16$) to recalibrate feature channels. This suppresses background noise (e.g., cloud edges, trees) while boosting drone visual cues.
2. **Spatial Attention (Where to focus on)**:
   Aggregates channel details via channel-wise pooling and applies a $7 \times 7$ convolution gate. This directs feature attention onto small candidate bounding locations across the spatial grid.
3. **Sequential CBAM Integration**:
   Combines Channel Attention followed by Spatial Attention at key feature-fusion stages (Layer 12 & Layer 15) to refine multi-scale feature maps before passing them to the detection head.

---

## 📊 Ablation Study & Performance Results

Evaluated on the validation set of the **YOLO Drone Detection Dataset** (640x640 resolution, batch size 16, 32 epochs):

| Model ID | Configuration | Parameters | GFLOPs | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **M1** | Baseline YOLOv8x | 68.2M | 258.5 | 86.82% | 83.90% | 89.22% | 53.76% |
| **M2** | Channel-Only Neck | 68.2M | 258.5 | 88.54% | 85.12% | 90.71% | 54.92% |
| **M3** | Spatial-Only Neck | 68.2M | 258.5 | 89.15% | 86.40% | 91.25% | 55.43% |
| **M4** | **YOLOv8-CBAM (Ours)** | **68.2M** | **258.6** | **91.30%** | **88.75%** | **92.93%** | **57.10%** |

### Key Observations
- **Channel Attention (M2)** improves mAP@0.5 by **+1.49%** by filtering non-drone feature channels.
- **Spatial Attention (M3)** improves mAP@0.5 by **+2.03%** by pinpointing tiny spatial targets.
- **Full CBAM (M4)** delivers the highest precision (**91.30%**) and recall (**88.75%**), achieving **92.93% mAP@0.5** (+3.71% boost over baseline).

---

## 📁 Repository Structure

```
Drone/
├── drone_dataset/                   # Unpacked dataset directory (train/ valid/)
└── Drone-Detection-1/
    ├── setup_dataset.py             # Automates zip extraction & folder validation
    ├── test_attention.py            # Diagnostic script to test all 4 attention modes
    ├── train.py                     # Local trainer with hardware auto-detection & CLI
    ├── test_my_model.py             # Interactive CLI tool for video inference & tracking
    ├── run_inference.py             # Automated script to download weights & run tracking
    ├── drone_detection_yolov8x.py   # Google Colab / standalone python training script
    ├── drone_detection_yolov8x.ipynb# Jupyter/Colab notebook for cloud training
    ├── coco.yaml                    # Dataset YAML configuration (1 class: drone)
    ├── main.tex                     # LaTeX source with abstract & ablation tables
    ├── best.pt                      # Pre-trained / finest model checkpoint
    ├── test/                        # Sample video assets for inference testing
    ├── results/                     # Metric plots (PR curve, F1 curve, confusion matrix)
    ├── LICENSE                      # Project license file
    └── README.md                    # Project documentation
```

---

## 🚀 Quick Start Guide

### 1. Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/rohith-1205/Drone_Detection_Yolov8_CBAM.git
cd Drone_Detection_Yolov8_CBAM/Drone-Detection-1

pip install ultralytics torch torchvision opencv-python matplotlib pandas
```

*Optional (for Intel Arc GPU acceleration):*
```bash
pip install torch-directml
```

---

### 2. Dataset Setup

Download the [YOLO Drone Detection Dataset](https://www.kaggle.com/datasets/muki2003/yolo-drone-detection-dataset) from Kaggle as `archive.zip` to your `Downloads` folder, then execute:

```bash
python setup_dataset.py
```

This script automatically copies `archive.zip`, extracts it to `drone_dataset/` at the project parent level, and verifies the image/label split counts.

---

### 3. Verify Attention Module Architecture

Run the verification test suite to ensure the PyTorch submodules (`attn12`, `attn15`) and forward tensor passes function cleanly across all 4 modes:

```bash
python test_attention.py
```

Expected output:
```text
--- Testing Model with Attention Mode: CBAM ---
Model attention_mode attribute: cbam
Has attn12 submodule: True
Has attn15 submodule: True
Success: Forward pass for mode cbam completed successfully without shape errors!
```

---

### 4. Training

#### Option A: Local Training via Interactive CLI
Run the local trainer script to select model size (`yolov8n` to `yolov8x`), epochs, and attention mode:

```bash
python Drone-Detection-1/train.py
```

Or pass CLI arguments directly:
```bash
python Drone-Detection-1/train.py --attention-mode cbam
```

#### Option B: Google Colab Training
Upload `Drone-Detection-1/drone_detection_yolov8x.ipynb` or run `Drone-Detection-1/drone_detection_yolov8x.py` in Google Colab with GPU runtime enabled.

---

### 5. Real-Time Video Detection & Tracking

Run tracking on your own video or the included sample video:

```bash
python Drone-Detection-1/test_my_model.py
```

The script will detect available `.pt` weight files, configure the matching attention mode, prompt for video input, and export annotated tracking results to `runs/detect/predict/`.

Alternatively, to download default weights and run inference instantly:

```bash
python Drone-Detection-1/run_inference.py
```

---

## 📈 Evaluation Visualizations

The training and validation evaluation metrics are stored in `Drone-Detection-1/results/`:
- `results.png` & `results.csv`: Epoch-by-epoch loss, precision, recall, and mAP metrics.
- `confusion_matrix.png`: Normalized detection confusion matrix.
- `PR_curve.png` & `F1_curve.png`: Precision-Recall and F1 curves across confidence thresholds.

---

## 📜 Citation & Research Paper

If you use this repository or model architecture in your research, please cite the paper using the following LaTeX / BibTeX entry (see `Drone-Detection-1/main.tex`):

```bibtex
@article{yolov8_cbam_drone_2026,
  title={YOLOv8-CBAM: An Attention-Guided Neck Feature Fusion Framework for Drone Detection},
  author={Sivakumar, Rohith},
  journal={Computer Vision and Pattern Recognition Letters},
  year={2026}
}
```

---

## 📄 License

Distributed under the [MIT License](Drone-Detection-1/LICENSE).

---

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for the foundational object detection framework.
- [Kaggle YOLO Drone Detection Dataset](https://www.kaggle.com/datasets/muki2003/yolo-drone-detection-dataset) for the annotated dataset.
- Woo et al. for the original CBAM paper (*"CBAM: Convolutional Block Attention Module"*, ECCV 2018).
