# Data to enter in the revised paper (by reviewer comment)

Every value below was measured by a script in `revision/` (source file in the last column).
✅ = measured, ready to use · ⏳ = not measured yet (needs GPU / your trained checkpoints).

---

## 1. Parameter overhead (R1 main point, R1 minor 1, R2-5) ✅

| Quantity | Old (paper) | **New value** | Source |
|---|---|---|---|
| Baseline YOLOv8x (1 class) params | 68.2 M | **68,153,571** | `results/a3_param_counts.json` |
| CBAM model params | 68.2 M | **68,217,769** | 〃 |
| Added by CBAM | "< 0.01 M", "no parameter growth" | **+64,198 (+0.094 %)** | 〃 |
| CBAM-1 (Layer 12, C = 640) | – | **51,299** = 51,200 MLP + 98 conv + 1 γ | 〃 |
| CBAM-2 (Layer 15, C = 320) | – | **12,899** = 12,800 MLP + 98 conv + 1 γ | 〃 |
| Channel-only (M2) total / added | 68.2 M | **68,217,573 / +64,002** | 〃 |
| Spatial-only (M3) total / added | 68.2 M | **68,153,769 / +198** | 〃 |
| SE at L12+L15 (added) | – | **+65,022** | 〃 |
| Coordinate Attention at L12+L15 (added) | – | **+50,012** | 〃 |

**Paste-ready (§VI-C):** "The two CBAM modules add 64,198 parameters (51,299 at Layer 12 and 12,899 at Layer 15),
raising the 1-class YOLOv8x from 68,153,571 to 68,217,769 parameters (+0.094%)."

---

## 2. GFLOPs overhead (R1-8, R2-4, R2-5) ✅

Convention: GFLOPs = 2 × MACs / 10⁹, profiled directly at 640×640, batch 1 (thop; cross-checked with fvcore).

| Config | Old (paper) | **New GFLOPs** | Added | Source |
|---|---|---|---|---|
| M1 Baseline | 258.5 | **258.1123** | – | `results/a4_gflops.json` |
| M2 Channel-only | 258.5 | **258.1126** | +0.0003 | 〃 |
| M3 Spatial-only | 258.5 | **258.1139** | +0.0016 | 〃 |
| M4 CBAM | 258.6 | **258.1141** | **+0.0018 (+0.0007 %)** | 〃 |
| SE @ L12+L15 | – | 258.1124 | +0.0001 | 〃 |
| CoordAtt @ L12+L15 | – | 258.1185 | +0.0062 | 〃 |
| CBAM incl. element-wise ops (pooling, sigmoid, gating) | – | 258.1418 (+0.028 extra) | | 〃 |
| fvcore cross-check (baseline → CBAM) | – | 257.7562 → 257.7580 (same +0.0018) | | 〃 |

- **Delete** Eq. (7) and the phrase "0.0387 % rise in computation" (abstract, contributions, conclusion).
- **Explanation for the old "+0.1":** Ultralytics `model.info()` profiles a 32×32 image and multiplies by 400. The
  CBAM MLP cost does not depend on image size, so this inflates it (`model.info()` gives 258.1123 → 258.2163).

---

## 3. Architecture / Table I / hook implementation (R1-1) ✅

| Item | Old (paper) | **New value** | Source |
|---|---|---|---|
| Layer 12 output | 512 × 40 × 40 | **640 × 40 × 40** | `results/a2_tensor_shapes.json` |
| Layer 15 output | 256 × 80 × 80 | **320 × 80 × 80** | 〃 |
| P3 / P4 / P5 | 320 / 640 / 640 ch | unchanged (correct) | 〃 |
| Detect head inputs | – | layers 15, 18, 21 | 〃 |

Full corrected Table I (layer · module · channels · resolution):
`0 Conv 80 320² · 1 Conv 160 160² · 2 C2f 160 160² · 3 Conv 320 80² · 4 C2f 320 80² · 5 Conv 640 40² ·
6 C2f 640 40² · 7 Conv 640 20² · 8 C2f 640 20² · 9 SPPF 640 20² · 12 C2f 640 40² · 15 C2f 320 80² · 22 Detect`

Implementation facts to write in §IV-E (`results/a1_attachment.json`):

| Question | **Answer** |
|---|---|
| How attached? | **Module wrapping** of `model[12]` and `model[15]` (not a PyTorch forward hook) |
| Gate | learnable scalar γ, initialised to 0: Y = F + γ(F″ − F) (⚠ verify this formula against your `cbam.py`) |
| Original weights frozen? | **No.** All 68,153,555 parameters are fine-tuned; only the 16-weight DFL conv is fixed |
| CBAM params in optimizer? | **Yes**, all 8 tensors incl. both γ (Ultralytics 8.4.201) |
| Saved / loaded from checkpoint? | **Yes**; wrapper and γ values restored |
| ONNX export? | **Yes** (opset 17), max deviation vs PyTorch **9.7 × 10⁻⁴** |
| Caveat | Stock `YOLO.train()` rebuilds the model from YAML and drops the wrapper → apply it in a custom trainer |

**Eq. (3) with ReLU:** M_c(F) = σ( W₁ δ(W₀ F_avg) + W₁ δ(W₀ F_max) ). The MLP is Conv1×1 → ReLU → Conv1×1, no bias.

---

## 4. Training duration and settings (R1-4, R2-2) ✅

| Item | Old (paper) | **New value** | Source |
|---|---|---|---|
| Epochs | 32 (Table II) / **60 (Fig. 4 caption)** | **32** everywhere | `results/a5_training_settings.json` |
| Checkpoint reported | not stated | best.pt (max 0.1·mAP50 + 0.9·mAP50-95 on val) | 〃 |
| Baseline (M1) optimizer | "SGD 0.937, lr 0.01" | **AdamW, lr 0.002, momentum 0.9** (`optimizer=auto`) | 〃 + training log |
| Baseline (M1) software / GPU | – | **Ultralytics 8.0.141**, PyTorch 2.0.1, **Tesla T4** | 〃 |
| M2–M4 script settings | – | SGD lr 0.01, momentum 0.937, wd 0.0005, batch 16, seed 0, AMP | `colab_ablation.py` |
| Augmentation | "mosaic, flip, scale, HSV" | mosaic 1.0 (off last 10 epochs), fliplr 0.5, scale ±0.5, translate ±0.1, HSV (0.015, 0.7, 0.4), warm-up 3 epochs, linear LR to 0.01·η₀ | `evidence/baseline_2023_run/args.yaml` |
| Loss weights | not stated | λ_box 7.5, λ_cls 0.5, λ_dfl 1.5 | 〃 |

---

## 5. Thresholds / NMS / versions (R1 minor 3) ✅

| Setting | **Value** |
|---|---|
| Validation confidence threshold | **0.001** |
| NMS IoU | **0.7** (class-aware) |
| Max detections / image | **300** |
| P and R reported at | the confidence that maximises F1 |
| Confusion matrix drawn at | **conf 0.25, IoU match 0.45** |
| Ultralytics version | **8.0.141** (M1); **8.4.201** (revision scripts); your fork for M2–M4: ⚠ fill in |
| PyTorch | 2.0.1 (M1); 2.15.0 (revision scripts) |

---

## 6. Confusion matrix vs. table (R1-5) ✅

| Item | **Value** | Source |
|---|---|---|
| Matrix counts | TP 320, FP 48, FN 49 | `results/a6_validation.json` |
| Val boxes | 369 (= TP + FN ✔) | 〃 |
| P / R from matrix (conf 0.25) | **86.96 % / 86.72 %** | 〃 |
| P / R in Table III (max-F1 conf) | 86.82 / 83.90 (M1) | 〃 |
| ⚠ Whose matrix? | **Baseline (M1)**, not CBAM. Fig. 5(b) legend "0.892" = baseline mAP@0.5 | `evidence/` |

**Paste-ready:** "The confusion matrix is drawn at confidence 0.25 with IoU matching at 0.45 (TP 320, FP 48, FN 49;
P = 86.96%, R = 86.72%), whereas Table III reports P and R at the confidence that maximises F1."

---

## 7. Dataset overlap / independent test set (R1-2, R1-3, R2-3) ✅

Near-duplicate audit of the original split (`results/b9_near_duplicates.json`):

| Item | **Value** |
|---|---|
| Val images with near-duplicate in train (pHash ≤ 4 bits) | **33 / 347 (9.5 %)**, of which 30 identical |
| Val images with same-scene match in train (≤ 10 bits) | **48 / 347 (13.8 %)** |
| Duplicate pairs inside train / inside val | 23 / 38 |
| Sources (file-name families) | train: pic 852, scene 71, yoto 89 · val: foto 133, numeric 186, pic 28 |
| Drone-free (negative) images | **0** in train, **0** in val |

New scene-grouped split (`results/c10_split.json`), 979 groups, per-source assignment:

| Split | Images | Boxes | % |
|---|---|---|---|
| Train | **975** | 1,042 | 71.7 |
| Val | **197** | 235 | 14.5 |
| Test | **187** | 209 | 13.8 |
| Cross-split pairs ≤ 10 bits | **0** | | |

---

## 8. Small-object evidence (R1-6, R2-5)

Box sizes after letter-boxing to 640 px (COCO: S < 32², M 32²–96², L ≥ 96²) ✅ (`results/b7_ap_by_size.json`, `results/d_figures.json`):

| Split | Small | Medium | Large |
|---|---|---|---|
| Train (1,117 boxes) | **13 (1.2 %)** | 53 (4.7 %) | 1,051 (94.1 %) |
| Val (369 boxes) | **25 (6.8 %)** | 42 (11.4 %) | 302 (81.8 %) |

AP by size ⏳ (needs trained checkpoints; run `b7_ap_by_size.py`):

| Model | AP_S | AP_M | AP_L | AR_S | AR_M | AR_L |
|---|---|---|---|---|---|---|
| Baseline | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| YOLOv8x-CBAM | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |

---

## 9. Runtime (R1-8, R2-4)

✅ CPU measurement (`results/b8_latency.json`): batch 1, 640×640, FP32, forward only, Conv+BN fused, 60 interleaved runs.

| Model | Device | Median ms | Mean ± std ms | FPS |
|---|---|---|---|---|
| Baseline | CPU, 4 threads | **604.59** | 607.38 ± 57.90 | 1.65 |
| CBAM | CPU, 4 threads | **606.91** | 611.13 ± 49.76 | 1.65 |
| Overhead (paired median) | | **+7.67 ms (+1.27 %)** | within noise | |

⏳ GPU FP16 (run `b8_latency.py` on a GPU): ms/img, FPS, peak memory (MB), GPU name, for both models.

---

## 10. Experiments still to run ⏳ (all need a GPU: `bash revision/run_all_gpu.sh`)

| Reviewer ask | Table in revised paper | Rows | Columns |
|---|---|---|---|
| SOTA comparison (R1-7, R1 minor 6, R2-4, both Q8) | Table VII | YOLOv5x, YOLO11x, RT-DETR-X, YOLOv8x, YOLOv8x-CBAM | Params (M), GFLOPs, FPS, P, R, mAP@0.5, mAP@0.5:0.95 |
| Repeated runs (R1-7) | Table VI | YOLOv8x, YOLOv8x-CBAM | mean ± std over seeds 0/1/2 of P, R, mAP@0.5, mAP@0.5:0.95 |
| Placement / attention ablation (R1-7) | Table V | none, CBAM@L12, CBAM@L15, CBAM@L12+15, SE, CoordAtt | added params ✅, mAP@0.5 ⏳, mAP@0.5:0.95 ⏳ |
| Size AP (R1-6) | Table IX | baseline, CBAM | see §8 |
| GPU runtime (R1-8, R2-4) | Table VIII | baseline, CBAM | see §9 |
| Figures (R1 Q3, R2-1, R1 minor 4–5) | Figs. 4, 5, 6 | CBAM curves, CBAM confusion matrix / PR curve, Fig. 6 without track IDs | – |

---

## 11. Text-only changes (no new data)

| Reviewer ask | What to write |
|---|---|
| Table III gain row (R1) | Use **pp**: +4.48 pp, +4.85 pp, +3.71 pp, +3.34 pp |
| Background rejection (R1 minor 7) | Remove "effective background rejection"; state that the dataset has **0 drone-free images** |
| Track IDs in Fig. 6 (R1 minor 4) | "The frames come from a video-tracking demo; the leading number is the tracker ID, not part of the method." |
| Failure cases (R1-6, R2-5) | Motion blur, drones < 8×8 px, occlusion by buildings |
| Limitations (R2-3) | Single small dataset; few small targets (6.8 % of val); no negative or bird images; other datasets left as future work |
| Novelty (R1-7) | Moderate wording ("incremental, targeted placement") until Tables V–VII are filled |

All of the ✅ items are already entered in `revision/paper_revised.tex`.
