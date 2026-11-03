# Revision SUMMARY – paper #633

## ⚠ Read first: three findings that affect the paper's claims

1. **Figs. 4 and 5 show the baseline, not CBAM.** The "YOLOv8-CBAM" confusion matrix (320/48/49) and PR curve
   (legend *mAP@0.5 = 0.892*) are the outputs of the 2023 **plain YOLOv8x** run, and Fig. 4's curves are its
   `results.csv`. Evidence: `evidence/baseline_2023_run/` vs `evidence/paper_pdf_figures/`.
2. **The M1 baseline was not trained under the stated protocol.** Its numbers (86.82 / 83.90 / 89.22 / 53.76) are
   exactly the last epoch of that 2023 run: Ultralytics 8.0.141, `optimizer=auto` → **AdamW, lr 0.002**, while
   Table II says SGD 0.01. **No raw output for M2–M4 exists** in either repository. The +3.71 pp gain is therefore not
   a matched comparison. *Recommendation:* run `c12_train_seeds.py` on a GPU before resubmitting.
3. **Train/val leakage.** 33 of 347 val images (9.5%) have a near-identical picture in train (30 are pixel-identical
   copies under other names).

## Inputs that were missing

`Drone-Detection-1/` is a submodule pointer to commit `12c69f7`, which was **never pushed**. Your `cbam.py`, `main.tex`
and all trained checkpoints were therefore unavailable. CBAM was re-implemented in `attention_modules.py`, and the
.tex was rebuilt from the PDF. **Push that commit** (or upload `cbam.py`, `main.tex`, `best.pt`) to close the gap.

## Findings and new results

| # | Item | Result | File |
|---|---|---|---|
| A1 | How CBAM is attached | Module **wrapping** of layers 12/15 + zero-init gate γ (not hooks). Weights **fine-tuned, not frozen** | `results/a1_attachment.json` |
| A1 | CBAM in optimizer? | Yes (all 8 tensors) in Ultralytics 8.4. Under the **8.0.x rule the two γ would be excluded** | 〃 |
| A1 | Checkpoint save/load | Yes (wrapper + γ restored) | 〃 |
| A1 | ONNX export | Yes, max diff 9.7e-4 vs PyTorch | 〃 |
| A1 | Stock `YOLO.train()` | **Drops the wrapper** (rebuilds from YAML) → custom trainer needed | 〃 |
| A2 | Shapes at L12 / L15 | **640×40×40 / 320×80×80** (paper: 512 / 256) | `results/a2_tensor_shapes.json` |
| A3 | Parameters | 68,153,571 → **68,217,769 (+64,198, +0.094%)**; ReLU present in MLP | `results/a3_param_counts.json` |
| A3 | Other blocks (added params) | channel 64,002 · spatial 198 · SE 65,022 · CoordAtt 50,012 | 〃 |
| A4 | GFLOPs (2×MACs, 640²) | 258.1123 → **258.1141 (+0.0018, +0.0007%)**; fvcore same Δ; +0.028 element-wise | `results/a4_gflops.json` |
| A4 | Paper's "+0.1 GFLOPs" | Artifact of `model.info()` ×400 extrapolation; script also **hard-coded** 258.1/258.5 | 〃 |
| A5 | Epochs | **32** everywhere; "60" caption is wrong | `results/a5_training_settings.json` |
| A5 | Checkpoint / thresholds | best.pt (fitness); val conf 0.001, NMS IoU 0.7, max_det 300 | 〃 |
| A5 | Ultralytics version (M1) | 8.0.141, Tesla T4, AdamW 0.002 | 〃 |
| A6 | Confusion matrix vs table | Matrix at conf 0.25 / IoU 0.45 (P 86.96, R 86.72) vs table at max-F1. Matrix is the **baseline's** | `results/a6_validation.json` |
| B7 | Val boxes S / M / L at 640 | **25 / 42 / 302** (6.8 / 11.4 / 81.8%); train 13 / 53 / 1,051 | `results/b7_ap_by_size.json` |
| B8 | CPU latency (FP32, bs 1) | 604.59 vs 606.91 ms median; paired **+7.67 ms (+1.27%)**, within noise (std ≈ 50–58 ms) | `results/b8_latency.json` |
| B9 | Near-duplicates train↔val | **33** val images ≤ 4 bits (9.5%), **48** ≤ 10 bits (13.8%) | `results/b9_near_duplicates.json` |
| C10 | New split | 975 / 197 / 187 images, **0** cross-split pairs ≤ 10 bits | `results/c10_split.json` |
| C11–13 | Training pipeline | CPU smoke test passed (pipeline only, numbers not usable) | `results/smoke/` |
| D | Revised paper | `paper_revised.pdf` compiles (8 pages, 0 warnings). All Part D edits applied | `paper_revised.tex` |
| E | Response letter | Every R1/R2 comment answered with location | `response_to_reviewers.md` |

## Skipped, and why

| Item | Why | How to finish |
|---|---|---|
| A6 re-validation of M1/M4 | No trained drone checkpoints in the repo | `BASELINE_WEIGHTS=… CBAM_WEIGHTS=… python revision/a6_validate.py` |
| B7 AP_S/M/L for both models | Same | `… python revision/b7_ap_by_size.py` |
| B8 GPU FP16 latency, FPS, peak memory | No GPU | `python revision/b8_latency.py` on a GPU |
| C11 YOLOv5x / YOLO11x / RT-DETR-X | No GPU (≈ 26 h per run on this CPU) | `bash revision/run_all_gpu.sh` |
| C12 3 seeds × baseline / CBAM | No GPU | 〃 |
| C13 CBAM @L12, @L15, SE, CoordAtt | No GPU | 〃 |
| Fig. 5(b), Fig. 6 (no track IDs), CBAM curve in Fig. 4 | Need CBAM checkpoint | `CBAM_WEIGHTS=… python revision/d_make_figures.py` |
| Diff against your original `main.tex` | File not in repo | Compare before submitting |
| Ultralytics version of your fork; exact γ formula | `cbam.py` / fork not in repo | Fill the `% TODO` / `% VERIFY` lines |

Also note: the paper grew from **6 to 8 pages**. Check the conference page limit.
