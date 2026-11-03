# Revision CHANGELOG

Running log of what was checked, what was found, and what was changed for the
revision of *"An Attention-Guided Neck Feature Fusion Framework for Drone
Detection Using YOLOv8 and CBAM"* (IConSCEPT 2026, paper ID 633).

Every number quoted below comes from a file in `revision/results/` or
`revision/evidence/`, and the file is named next to it.

---

## 0. Environment and missing inputs

| Item | Status |
|---|---|
| Machine | Cloud container, **CPU only** (4 cores, 15 GB RAM, **no GPU**) |
| Software | Python 3.13, torch 2.15.0, **Ultralytics 8.4.201**, thop (ultralytics-thop 2.2.4), fvcore, pycocotools, imagehash, pdflatex (TeX Live) |
| `Drone-Detection-1/` | **Empty.** It is a git *submodule pointer* (gitlink) to commit `12c69f7` of `github.com/rohith-1205/Drone-Detection-1`, but there is no `.gitmodules` and that commit **was never pushed** (`git fetch` → "not our ref"). The public repo only has the Aug-2025 `main` (`8a97eaa`), which contains the 2023 plain-YOLOv8x run but no CBAM code. |
| Your `cbam.py` (hook code) | **Not available.** Re-implemented in `revision/attention_modules.py` following the paper and the API used by `colab_ablation.py` / `scratch/verify_cbam_refactor.py` (`apply_attention_to_yolo`, `AttentionLayerWrapper`, `attn_block.gamma`). |
| Your `main.tex` | **Not available** (README links to `Drone-Detection-1/main.tex`). `revision/paper_revised.tex` was rebuilt from the submitted PDF. Please diff it against your original before submitting. |
| Trained drone checkpoints (`best.pt` for M1–M4) | **Not available.** Only the public COCO `yolov8x.pt`, `yolov5xu.pt`, `yolo11x.pt` and `rtdetr-l.pt` were downloaded (to `/home/user/weights`, not committed). |
| `val_metrics.json` / `summary.md` from the Colab ablation | **Not available.** No raw output for the M2–M4 rows of Table III could be found anywhere. |

Consequences: any measurement that depends only on the **architecture**
(shapes, parameters, GFLOPs, latency, ONNX, data checks) was run here. Anything
that depends on **trained drone weights** or **training** (re-validation, AP by size,
retraining, seeds, ablations, Fig. 5/6) has a ready-to-run script, but was
**skipped** and is listed in `SUMMARY.md`.

---

## Part A – Facts check

### A1 How CBAM is attached (`a1_attachment_check.py` → `results/a1_attachment.json`)
- **Mechanism:** this is *module replacement*, not a PyTorch `register_forward_hook`. `model.model[12]` and `model.model[15]` (C2f blocks) are each wrapped in `AttentionLayerWrapper(layer, ResidualAttentionBlock(CBAM))`. The forward pass computes `y = layer(x)`, then `y + γ·(CBAM(y) − y)`, with a learnable scalar `γ` that starts at 0 (so at step 0 the model is exactly the pretrained one). The paper's Eq. (6) does not mention `γ`.
- **Q0 trap:** stock `YOLO.train()` **rebuilds the network from `yolov8x.yaml` and silently drops the wrapper** (`Q0_stock_train_keeps_wrapper: false`). Your fork must patch this; the revision code uses a custom trainer (`make_attention_trainer`).
- **Q1 Frozen?** No. All 68,153,555 original parameters are **fine-tuned together** with CBAM. Only the fixed 16-weight DFL conv is frozen. → The paper's "without modifying the original model parameters" is wrong.
- **Q2 Optimizer?** Yes in Ultralytics 8.4.201 (all 8 CBAM tensors are present). **But** under the Ultralytics 8.0.x grouping rule (the version of your baseline run), `model.12.attn_block.gamma` and `model.15.attn_block.gamma` would be **left out of the optimizer**.
- **Q3 Checkpoint?** Yes. The wrapper and γ survive an Ultralytics-style fp16 save/reload (γ = 0.5/0.25 restored; max output diff 1.5e-3 from fp16).
- **Q4 ONNX?** Yes. Export (opset 17) works, and ONNX Runtime matches PyTorch to 9.7e-4.

### A2 Tensor shapes (`a2_tensor_shapes.py` → `results/a2_tensor_shapes.{json,csv}`)
- Layer 12 output = **640×40×40** and layer 15 output = **320×80×80** (not 512 / 256). The paper's numbers are the un-scaled YAML values; YOLOv8x multiplies them by 1.25.
- Table I's other rows are correct. The Detect head takes layers [15, 18, 21].

### A3 Parameter counts (`a3_param_counts.py` → `results/a3_param_counts.json`)
- Baseline 1-class YOLOv8x: **68,153,571**.
- CBAM at layers 12+15: **+64,198** (51,299 + 12,899) → **68,217,769** (+0.0942 %). The paper's "< 0.01 M / no parameter growth" is wrong. Even with the paper's 512/256 channels it would be 41,158.
- Channel-only +64,002; spatial-only +198; SE +65,022; Coordinate Attention +50,012.
- The channel MLP is `Conv2d → ReLU → Conv2d` (the ReLU δ is present). Eq. (3) is missing it.

### A4 GFLOPs (`a4_gflops.py` → `results/a4_gflops.json`)
- Convention: GFLOPs = 2 × MACs / 1e9 (Ultralytics). Profiled directly at 640×640, batch 1.
- thop: baseline **258.1123**, CBAM **258.1141** (+0.0018 GFLOPs, +0.0007 %). fvcore gives the same +0.0018 GFLOPs on a 257.756 base. The element-wise ops that thop ignores add about 0.028 GFLOPs more (258.1418 total).
- **Origin of the paper's "+0.1"**: Ultralytics `get_flops()` / `model.info()` profiles a 32×32 image and multiplies by 400. The CBAM MLP runs on 1×1 pooled vectors, so it does not scale with image size, and the ×400 inflates it (`get_flops` gives 258.1123 → 258.2163). The paper's 258.5/258.6 matches neither number. `colab_ablation.py` also **hard-codes** `258.1 / 258.5`.
- Fixed a tooling issue: the old `thop` 0.1.1 package had overwritten `ultralytics-thop`, which made `get_flops` return 0. It was uninstalled and the numbers above were re-run.

### A5 Settings actually used (`a5_training_settings.py` → `results/a5_training_settings.json`)
- `colab_ablation.py`: `epochs=32`, batch 16, SGD lr0 0.01, momentum 0.937, wd 5e-4, seed 0, AMP. It evaluates `weights/best.pt` with `conf=0.001, iou=0.7`.
- 2023 baseline run (`evidence/baseline_2023_run/`): **Ultralytics 8.0.141, Tesla T4, 32 epochs, `optimizer=auto` → AdamW(lr=0.002, momentum=0.9)**. The logged LR schedule also implies lr0 = 0.00200.
- **The paper's M1 row (86.82 / 83.90 / 89.22 / 53.76) is exactly the last epoch (31) of this 2023 run**. That is also the best epoch by fitness. So the baseline was **not** trained with the Table II settings (SGD, 0.01) or with the `colab_ablation.py` pipeline.
- No run used 60 epochs. The Fig. 4 caption is wrong.

### A6 Validation / confusion matrix (`a6_validate.py` → `results/a6_validation.json`)
- Re-validation: **skipped**, because no trained drone checkpoints are available. The script runs as soon as `BASELINE_WEIGHTS` / `CBAM_WEIGHTS` are set.
- The val set has 369 boxes, and TP + FN = 320 + 49 = 369, so the matrix is on this val set.
- Ultralytics 8.0.141 draws the confusion matrix at **conf 0.25, IoU-match 0.45** (source saved in `evidence/ultralytics_8.0.141_confusion_matrix_defaults.txt`). Table P/R are read at the **max-F1** confidence. These are different operating points, which explains 86.96/86.72 vs. the table values.
- **CRITICAL:** Fig. 5(a)'s counts are identical to the 2023 **plain YOLOv8x** `confusion_matrix.png`, and Fig. 5(b)'s legend reads **"all classes 0.892 mAP@0.5"** (= baseline 89.2 %, not CBAM's 92.93 %). The PR curve is the same curve as the 2023 run's `PR_curve.png`. Fig. 4's curves match the 2023 `results.csv` point for point (epoch 0: box loss 1.615, P 0.680, R 0.553, mAP50 0.606). **Figs. 4 and 5, captioned as YOLOv8-CBAM, show the baseline model.** (Visual comparison; files are in `evidence/paper_pdf_figures/` and `evidence/baseline_2023_run/`.)

---

## Part B – Experiments with existing weights

### B7 AP by object size (`b7_ap_by_size.py` → `results/b7_ap_by_size.json`)
- **Measured now:** sizes of the validation ground-truth boxes after letter-boxing to 640 px, using COCO buckets: **small 25 (6.8 %), medium 42 (11.4 %), large 302 (81.8 %)** of 369 boxes. The dataset is dominated by large drones, so the "small-object benefit" cannot be shown convincingly on it. The paper's wording was softened accordingly.
- **Skipped:** AP_S/M/L for baseline and CBAM need the trained checkpoints. The script rescales GT and predictions to the 640 frame and runs pycocotools as soon as `BASELINE_WEIGHTS` / `CBAM_WEIGHTS` exist.

### B8 Latency (`b8_latency.py` → `results/b8_latency.json`)
- Protocol: batch 1, 640×640, forward only, Conv+BN fused, interleaved timing, median reported.
- **CPU only** (no GPU here): FP32, 4 threads. Baseline median 604.59 ms (1.65 FPS), CBAM median 606.91 ms (1.65 FPS); paired-median overhead **+7.67 ms (+1.27 %)**, inside the ±50–58 ms run-to-run std.
- A first *sequential* protocol gave +13 % and then −4 % on two runs. That is noise from the shared CPU, so the script was rewritten to time the two models interleaved. Only the interleaved result is kept.
- **Skipped:** GPU FP16 latency and peak GPU memory. The same script produces them automatically on a CUDA machine.

### B9 Near duplicates (`b9_near_duplicates.py` → `results/b9_*.{json,csv}`, `figures/b9_duplicate_examples.jpg`)
- pHash (64-bit), Hamming distance ≤ 4 = near-duplicate, ≤ 10 = same scene.
- **Leakage found:** **33 of 347 validation images (9.5 %) have a near-duplicate in train** (30 at distance 0, i.e. the same picture), and 48 (13.8 %) have a same-scene match. The example sheet confirms they are the same stock photos saved under different names (e.g. `train/pic_045.jpg` = `valid/0029.jpg` = `valid/0063.jpg`).
- Split structure: train = `pic` 852, `scene` 71, `yoto` 89; val = `foto` 133, numeric 186, `pic` 28. The video-frame families (`foto`, `yoto`, `scene`) are already separated by split. The leaks come from the web-image families (`pic` ↔ numeric/`foto`/`pic`).
- Within-split redundancy: 23 near-duplicate pairs inside train, 38 inside val.

---

## Part C – Retraining experiments

### C10 Leakage-free split (`c10_make_split.py` → `splits/scene_split/*.txt`, `results/c10_split.json`)
- B9 found leakage, so a **scene-grouped 70/15/15 split** was built from all 1,359 images. Images are linked if their pHash distance is ≤ 10 or if they are video frames < 100 apart in the same `foto`/`yoto`/`scene` family (979 groups). Whole groups are assigned **per source family**, so val/test also contain real video scenes.
- Result: train **975** images / 1,042 boxes, val **197** / 235, test **187** / 209 (71.7 / 14.5 / 13.8 %; the 71-frame `scene` clip cannot be divided). **0 cross-split pairs within 10 bits.**
- A first version without per-family assignment put every video frame in train (val/test were only web photos). It was rejected and replaced.
- Images are not copied. The split files list paths relative to `drone_dataset/`, and `common.make_split_yaml()` writes a machine-specific data YAML.

### C11–C13 Training (`c11_train_sota.py`, `c12_train_seeds.py`, `c13_placement_ablation.py`, shared `train_utils.py`)
- One function trains on the scene split's train part, selects best.pt on val, tests **once** on test, and records params, GFLOPs (thop at 640) and latency. Settings match `colab_ablation.py`: 32 epochs, 640 px, batch 16, SGD 0.01 / 0.937 / 5e-4, AMP. RT-DETR-X uses AdamW 1e-4 (its standard recipe; SGD 0.01 diverges for DETR-type models). YOLOv5x means Ultralytics' anchor-free `yolov5xu`.
- Attention is applied through `make_attention_trainer()`, because stock `train()` would drop it (A1 Q0).
- **CPU smoke test passed** (`results/smoke/yolov8x_cbam_s0.json`, `c11_sota_smoke.json`): 1 epoch, 2 % of data, 320 px. Wrappers were present in best.pt, and γ moved from 0 to −1.8e-4 / 3.1e-5, so CBAM receives gradients. Test-split evaluation, params (68,217,769), GFLOPs (258.1141) and latency all ran. **These numbers are a pipeline test only and must not be used.**
- The smoke test also caught three bugs, all fixed: (1) pycocotools crashed on mixed int/str image ids (B7); (2) Ultralytics 8.4 draws the confusion matrix at the val conf of 0.001 (FP = 103,755), so A6 now runs a separate conf-0.25 pass like 8.0.141; (3) "skipped" results were treated as cached, so they would never retrain on a GPU.
- **Skipped: all 15 real training runs** (YOLOv8x/CBAM × 3 seeds, CBAM @L12 / @L15, SE, CoordAtt, YOLOv5x, YOLO11x, RT-DETR-X). There is no GPU. Extrapolating the measured smoke-test speed (1.5 s/iteration at 320 px, batch 2) to 640 px, batch 16 gives about 26 h per run on this CPU. Each run writes a `status: skipped` record with the reason in `results/c_runs/`. `bash revision/run_all_gpu.sh` runs them all on a GPU.

---

## Part D – Paper (`paper_revised.tex` → `paper_revised.pdf`, 8 pages, compiles with 0 errors / 0 warnings)

`paper_revised.tex` was **rebuilt from the submitted PDF** (no .tex in the repo). Revised text is blue (`\revisionmarksfalse` turns it off). Tables are generated from the JSON results by `d_fill_tables.py`. Table numbering I–III is unchanged; new tables are IV–IX.

| Change | Where | Source of numbers |
|---|---|---|
| Fig. 4 caption 60 → **32 epochs**; it now says the curves are the **baseline** | Fig. 4, §VII-A | a5 |
| Table I channels 512/256 → **640/320**; table generated from the shape trace | Table I, §IV-B, §IV-D | a2 |
| Eq. (3): ReLU δ added, MLP bias-free stated | §IV-C | a3 |
| New Eq. (7): zero-initialised gate γ (**verify against your cbam.py**) | §IV-C | A1 |
| §IV-E rewritten: module wrapping (not hooks), weights fine-tuned not frozen, optimizer/checkpoint/ONNX verified | §IV-E | a1 |
| Exact params **68,153,571 → 68,217,769 (+64,198, +0.094 %)** and GFLOPs **258.1123 → 258.1141 (+0.0018)**; "0.0387 %" and "no parameter growth" removed; `model.info()` artifact explained | Abstract, contributions, §VI-C, Table III, Conclusion | a3, a4 |
| Table III gain row now in **pp**, measured Params/GFLOPs columns, footnote on M1's provenance (AdamW, 8.0.141) and missing raw outputs for M2–M4 | Table III | a3, a4, a5 |
| Table II: footnote that M1 used AdamW 0.002; rows for checkpoint and seed added; augmentation listed | Table II, §V-A | a5 |
| Evaluation protocol: conf 0.001, NMS IoU 0.7, max_det 300, P/R at max F1, CM at conf 0.25 / IoU 0.45, Ultralytics 8.0.141 / 8.4.201, GFLOP convention | §V-B | a5, a6 |
| Dataset provenance, near-duplicate audit and new split | §III-A, Table IV | b9, c10 |
| Size statistics (train S/M/L = 13/53/1,051, val 25/42/302) | §III-A, Fig. 2(b), Table IX | b7, d_figures |
| Runtime table (CPU now, GPU pending) | §VI-C, Table VIII | b8 |
| New tables: placement/attention (V), seeds (VI), SOTA (VII), AP by size (IX), with `--` cells until GPU runs | §VI, §VIII-A | c11–c13, b7 |
| Fig. 5 caption: it is the **baseline**; CM-vs-table P/R explained; "effective background rejection" removed | §VII-B, Fig. 5 | a6 |
| Limitations paragraph (single dataset, few small targets, no drone-free/bird negatives, no background-rejection claim) | §VIII-A | b7, b9 |
| Fig. 6 caption explains the track IDs (regeneration with `predict()` is pending) | Fig. 6 | – |
| Failure-cases paragraph; unmeasured wording removed | §VII-D | – |
| RT-DETR reference added | Refs | – |

**Figures regenerated** (`d_make_figures.py`, large fonts, vector PDF): Fig. 2 from the labels, with panel (b) replaced by a size histogram because the old box-overlay panel was unreadable; Fig. 4 from the 2023 `results.csv`; Fig. 5(a) from the recorded baseline counts. **Not regenerated:** Fig. 5(b) and Fig. 6 need the CBAM checkpoint, so they are kept from the PDF.

**Left for you (marked `% TODO` / `% VERIFY` in the .tex):** Ultralytics version of your fork; the γ gate formula; interpretation sentences for Tables V–VII once filled; CBAM versions of Figs. 4–6; a failure-case figure. **Page count grew from 6 to 8.** Check the conference page limit; Tables V/VI/IX can be merged if needed.

---

## Housekeeping
- Ultralytics rewrote `drone_dataset/{train,valid}/labels.cache` during validation. Both were restored to the committed version with `git checkout -- <file>`.
- Large outputs (checkpoints, run folders, runtime YAMLs) are excluded by `revision/.gitignore`.
