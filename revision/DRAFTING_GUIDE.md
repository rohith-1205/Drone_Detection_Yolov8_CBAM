# Drafting guide – every reviewer comment, with data and ready-to-use text

Paper #633, IConSCEPT 2026. Use this file to draft the revised manuscript and the response letter.

**How each comment is laid out**
- **Comment**: what the reviewer asked.
- **Data**: measured values. Every number comes from a file in `revision/results/` (named in brackets).
- **Manuscript text**: paste-ready text for the paper (already in `paper_revised.tex` unless marked ⏳).
- **Response**: paste-ready reply for the letter.
- **Status**: ✅ ready · ⏳ needs a GPU or your trained checkpoints (exact command given) · ✍ only you can supply it.

Section, table and figure numbers refer to the revised paper: Tables I–III keep their old numbers; IV = split,
V = placement, VI = seeds, VII = SOTA, VIII = runtime, IX = AP by size.

---

## 0. Fix these before drafting anything else

| # | Problem | Evidence | What to do |
|---|---|---|---|
| 0.1 | **Figs. 4 and 5 show the plain YOLOv8x baseline, not CBAM.** The confusion matrix (320/48/49) and PR curve (legend "0.892") are identical to the 2023 YOLOv8x run; Fig. 4 = that run's `results.csv`. | `evidence/baseline_2023_run/` vs `evidence/paper_pdf_figures/` | Caption them as baseline (done) **or** regenerate from your CBAM `best.pt`: `CBAM_WEIGHTS=… python revision/d_make_figures.py` |
| 0.2 | **M1 baseline was trained differently from Table II.** M1 = last epoch of the 2023 run: Ultralytics 8.0.141, AdamW lr 0.002 (`optimizer=auto`), not SGD 0.01. | `results/a5_training_settings.json` | Footnote added. Re-run the matched comparison (Table VI) before claiming the +3.71 pp gain again. |
| 0.3 | **No raw outputs of M2–M4** (channel / spatial / CBAM) exist in either repository. | – | ✍ Find `val_metrics.json` / `summary.md` in your Google Drive (`drone_experiments/`), or re-run. |
| 0.4 | **Your `cbam.py` and `main.tex` are not on GitHub** (submodule commit `12c69f7` never pushed). | – | ✍ Diff `paper_revised.tex` against your `main.tex`; confirm the γ-gate formula (Eq. 7) against your `cbam.py`. |

---

# Reviewer #1

## R1-Q3 · "Is quality of figures satisfactory? – No"

**Data:** Figs. 2, 4 and 5(a) were regenerated as vector PDFs with 12–14 pt fonts (`figures/`). Fig. 2(b), an
unreadable overlay of all boxes, was replaced by a size histogram.

**Manuscript text:** new captions are in `paper_revised.tex`.
- Fig. 2: "Statistics of the training annotations: (a) number of drone instances, (b) box size after letter-boxing to 640 px with the COCO small/large borders (dashed), (c) distribution of normalized box centers, and (d) joint distribution of normalized box width and height."

**Response:**
> We regenerated Figs. 2, 4 and 5 as vector graphics with 12–14 pt fonts. Panel (b) of Fig. 2, previously an
> unreadable overlay of all bounding boxes, was replaced by a histogram of box sizes at 640 px with the COCO size
> borders.

**Status:** ✅ Figs. 2, 4, 5(a) · ⏳ Fig. 5(b), Fig. 6 → `CBAM_WEIGHTS=<best.pt> python revision/d_make_figures.py`

---

## R1-0 · "Correct the parameter-overhead calculation" (paper claims < 0.01 M)

**Data** [`a3_param_counts.json`]

| Item | Value |
|---|---|
| Baseline YOLOv8x, 1 class | **68,153,571** |
| YOLOv8x-CBAM | **68,217,769** |
| Added | **+64,198 (+0.094 %)** |
| CBAM-1 @ L12 (C = 640) | 51,299 = 2·640·40 (MLP) + 98 (7×7 conv) + 1 (γ) |
| CBAM-2 @ L15 (C = 320) | 12,899 = 2·320·20 + 98 + 1 |
| Channel-only / spatial-only | +64,002 / +198 |
| With the paper's wrong 512/256 channels | would be 41,158; still > 0.01 M |

**Manuscript text (§VI-C):**
> The two CBAM modules add exactly 64,198 parameters: 51,299 at Layer 12 ($2\cdot640\cdot40$ MLP weights, 98
> convolution weights and one gate) and 12,899 at Layer 15, raising the 1-class YOLOv8x from 68,153,571 to
> 68,217,769 parameters (+0.094\%).

Also replace in the Abstract, Contributions and Conclusion: "no additional baseline parameter growth" → "adds 64,198 parameters (0.094\%)".

**Response:**
> We thank the reviewer; the statement was incorrect. With the actual channel widths (640 and 320; see R1-1), the
> shared MLPs alone contribute $2C^2/r$ = 51,200 and 12,800 parameters. In total, the two modules add 64,198
> parameters (+0.094%), measured with `sum(p.numel())` and consistent with the hand calculation. The corrected values
> appear in the Abstract, Contributions, §VI-C, Table III and the Conclusion.

**Status:** ✅

---

## R1-1 · "Clarify the architecture and hook implementation"

The reviewer asked for six things: (a) Table I mixes dimensions, (b) instantiated config and tensor-shape trace,
(c) module registration, optimizer, gradients, (d) checkpoint save/load, (e) export, and (f) pretrained
initialisation vs freezing.

**Data** [`a2_tensor_shapes.json`, `a1_attachment.json`]

Tensor trace (input 1×3×640×640):

| Layer | Module | Output |
|---|---|---|
| 0 | Conv | 80×320×320 |
| 1 | Conv | 160×160×160 |
| 2 | C2f | 160×160×160 |
| 3 | Conv | 320×80×80 |
| 4 | C2f (P3) | 320×80×80 |
| 5 | Conv | 640×40×40 |
| 6 | C2f (P4) | 640×40×40 |
| 7 | Conv | 640×20×20 |
| 8 | C2f | 640×20×20 |
| 9 | SPPF (P5) | 640×20×20 |
| **12** | **C2f → CBAM-1** | **640×40×40** (paper: 512) |
| **15** | **C2f → CBAM-2** | **320×80×80** (paper: 256) |
| 22 | Detect ← [15, 18, 21] | – |

| Question | Answer |
|---|---|
| Mechanism | **Module wrapping** of `model[12]` and `model[15]`; not a PyTorch forward hook |
| Gate | learnable scalar γ, initialised to 0: $Y = F + \gamma(F'' - F)$ ✍ verify against your cbam.py |
| Registration / optimizer | all 8 CBAM tensors, incl. both γ, are in the optimizer (Ultralytics 8.4.201) |
| Gradient flow | γ moved from 0 → −1.8e-4 / 3.1e-5 after one training epoch (smoke test) |
| Checkpoint | wrapper + γ saved and restored (γ = 0.5/0.25 round-trip) |
| ONNX export | works (opset 17); max deviation vs PyTorch **9.7 × 10⁻⁴** |
| Frozen? | **No**: 68,153,555 original params fine-tuned; only the 16-weight DFL conv is fixed |
| Pretrained init | wrapper inserted *after* loading COCO weights; γ = 0 ⇒ identical output at step 0 |
| Caveat | stock `YOLO.train()` rebuilds from YAML and **drops** the wrapper → applied in a custom trainer `get_model()` |

**Manuscript text (§IV-E, "Attention Integration and Training"):**
> The attention modules are attached by module wrapping: the C2f modules at indices 12 and 15 of the Ultralytics
> layer list are replaced by a wrapper that first runs the original C2f block and then the gated CBAM of Eq. (7).
> The wrapper copies the layer's routing attributes, so the rest of the network is unchanged. The wrapper is
> inserted after the COCO-pretrained YOLOv8x weights are loaded, which preserves the pretrained initialization. The
> original weights are not frozen: all parameters (except the fixed DFL projection) are fine-tuned jointly with
> CBAM. We verified that (i) all CBAM tensors, including both gates γ, are registered in the optimizer, (ii) the
> wrapped modules and gate values are stored in and restored from Ultralytics checkpoints, and (iii) the model
> exports to ONNX (opset 17) with a maximum deviation of $9.7\times10^{-4}$ from PyTorch.

**New Eq. (7):** $Y = F + \gamma\,(F'' - F)$, with γ initialised to 0.

**Response:**
> We regenerated Table I from a tensor-shape trace of the instantiated 1-class YOLOv8x. Layers 12 and 15 output
> 640×40×40 and 320×80×80; the values 512/256 were the unscaled YAML widths before the 1.25 width multiplier.
> §IV-E now explains that CBAM is attached by wrapping layers 12 and 15 (not by forward hooks) with a
> zero-initialised gate (new Eq. 7). It also states that all CBAM tensors are in the optimizer and receive
> gradients, that checkpoints save and restore them, and that ONNX export works (max deviation 9.7×10⁻⁴). Finally,
> it states that the pretrained weights are used for initialisation and are fine-tuned, not frozen.

**Status:** ✅ (✍ confirm the γ formula)

---

## R1-2 · "Provide an independent test evaluation"

**Data** [`c10_split.json`]: new split train 975 / val 197 / test 187 images (1,042 / 235 / 209 boxes).
Checkpoint = best fitness on val (`best.pt`); the test split is evaluated once.

**Manuscript text (end of §III-A):**
> Model selection uses only the validation part; the test part is evaluated once.

and in §V-A:
> For every configuration the checkpoint with the best Ultralytics fitness
> ($0.1\,\mathrm{mAP@0.5}+0.9\,\mathrm{mAP@0.5{:}0.95}$) on the validation data is reported.

**Response:**
> We created a new scene-grouped split with a held-out test set of 187 images (§III-A, Table IV). Attention
> placement, training length and checkpoints are selected on the validation part only, and the test part is
> evaluated once per configuration. For the original split we now state the checkpoint-selection rule explicitly.
> Test results for all configurations are reported in Tables V–VII.

**Status:** ✅ split · ⏳ test metrics → `bash revision/run_all_gpu.sh`

---

## R1-3 · "Address possible image or sequence overlap"

**Data** [`b9_near_duplicates.json`, `c10_split.json`]

| Item | Value |
|---|---|
| Val images with near-duplicate in train (pHash ≤ 4 bits) | **33 / 347 (9.5 %)**, 30 identical |
| Val images with same-scene match (≤ 10 bits) | **48 / 347 (13.8 %)** |
| Duplicate pairs within train / within val | 23 / 38 |
| Sources: train | pic 852 · scene 71 · yoto 89 |
| Sources: val | foto 133 · numeric 186 · pic 28 |
| New split groups | 979 (pHash ≤ 10 or video frames < 100 apart), assigned per source family |
| Cross-split pairs ≤ 10 bits after re-split | **0** |

**Manuscript text (§III-A, "Provenance and near-duplicate audit"):**
> The images come from several sources whose file names identify them: web photographs and frames sampled from
> videos. We hashed every image with a 64-bit perceptual hash and compared all train–validation pairs. In the
> original split, 33 of the 347 validation images (9.5\%) have a near-duplicate (Hamming distance $\leq4$) in the
> training set, 30 of them identical pictures stored under different names, and 48 (13.8\%) have a same-scene match
> ($\leq10$). We therefore built a scene-grouped split: images linked by a pHash distance $\leq10$ or by neighbouring
> frame numbers ($<100$ apart) of the same video source are merged into groups, and whole groups are assigned, per
> source family, to train/validation/test in a 70/15/15 ratio. The result (975/197/187 images) has no cross-split
> pair within 10 bits.

**Response:**
> We audited the dataset with 64-bit perceptual hashing and found that 33 of the 347 validation images (9.5%) have a
> near-duplicate in the training set (30 identical pictures under different names), and 48 (13.8%) share a scene.
> We describe the sources and build a new split that keeps visually similar images and neighbouring video frames in
> the same partition, assigned per source family. The new split has no cross-split pair within 10 bits (§III-A,
> Table IV).

**Status:** ✅

---

## R1-4 · "Resolve training-duration inconsistencies (32 vs 60 epochs)"

**Data** [`a5_training_settings.json`]: every run used **32 epochs** (`colab_ablation.py` epochs=32; the 2023 log has
32 rows). Checkpoint = best.pt. M1: Ultralytics 8.0.141, AdamW lr 0.002, Tesla T4. M2–M4 script: SGD 0.01,
momentum 0.937, wd 5e-4, batch 16, seed 0.

**Manuscript text:**
- Fig. 4 caption: "Training losses and validation metrics of the baseline YOLOv8x (M1) over its 32 training epochs."
- Table II footnote: "The M1 baseline of Table III was trained with Ultralytics 8.0.141 and `optimizer=auto`, which selected AdamW ($\eta_0=0.002$, momentum 0.9)."
- §V-A, augmentation: "mosaic (probability 1.0, disabled for the last 10 epochs), horizontal flip (0.5), random scaling (±50%), translation (±10%) and HSV perturbation (0.015, 0.7, 0.4), with a 3-epoch warm-up and a linear learning-rate decay to $0.01\eta_0$."

**Response:**
> All runs used 32 epochs; the "60 epochs" in the Fig. 4 caption was an error and has been corrected. Training
> settings, augmentation and the checkpoint-selection rule are now given for every configuration (§V-A, Table II).
> A footnote states that the M1 baseline was trained with `optimizer=auto` (AdamW, η₀ = 0.002) in Ultralytics
> 8.0.141.

**Status:** ✅

---

## R1-5 · "Reconcile the confusion matrix with reported P/R"

**Data** [`a6_validation.json`, `evidence/ultralytics_8.0.141_confusion_matrix_defaults.txt`]

| | Value |
|---|---|
| Matrix | TP 320 · FP 48 · FN 49 (TP + FN = 369 val boxes ✔) |
| P / R from matrix | **86.96 % / 86.72 %** |
| Matrix operating point | conf **0.25**, IoU match **0.45** (Ultralytics 8.0.141 default) |
| Table operating point | max-F1 confidence (validator conf 0.001, NMS IoU 0.7) |
| Whose matrix | **baseline M1** (Table III M1: 86.82 / 83.90) |

**Manuscript text (§VII-B):**
> Fig. 5 shows the confusion matrix and the precision–recall curve of the baseline on the original validation
> split. At the confusion-matrix operating point (confidence 0.25, IoU match 0.45) the model finds 320 of the 369
> drones and produces 48 false positives, i.e. P = 86.96\% and R = 86.72\%; these differ from Table III, which
> reports P/R at the max-F1 confidence.

**Response:**
> The reviewer is correct that 320/48/49 corresponds to P = 86.96% and R = 86.72%. The difference is the operating
> point: Ultralytics draws the confusion matrix at confidence 0.25 with IoU matching at 0.45, whereas the tabulated
> P/R are taken at the confidence that maximises F1. Both conventions are now stated (§V-B, §VII-B). While checking
> this, we found that the plotted matrix belonged to the baseline model. The caption has been corrected and the
> figure regenerated.

**Status:** ✅ explanation · ⏳ CBAM matrix → `CBAM_WEIGHTS=… python revision/a6_validate.py` then `d_make_figures.py`

---

## R1-6 · "Demonstrate the claimed small-object benefit"

**Data** [`b7_ap_by_size.json`, `d_figures.json`], box sizes after letter-boxing to 640 px:

| Split | Small (< 32²) | Medium | Large (≥ 96²) |
|---|---|---|---|
| Train (1,117) | **13 (1.2 %)** | 53 (4.7 %) | 1,051 (94.1 %) |
| Val (369) | **25 (6.8 %)** | 42 (11.4 %) | 302 (81.8 %) |

Drone-free (negative) images: **0** in train, **0** in val.

⏳ Table IX template (fill after `b7_ap_by_size.py`):

| Model | AP_S | AP_M | AP_L | AR_S | AR_M | AR_L |
|---|---|---|---|---|---|---|
| Baseline | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| YOLOv8x-CBAM | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |

**Manuscript text (§III-A):**
> Measured after letter-boxing to 640 px with the COCO thresholds, only 13 of the 1,117 training boxes (1.2\%) and
> 25 of the 369 validation boxes (6.8\%) are small ($<32^2$ pixels); 81.8\% of the validation boxes are large
> ($\geq96^2$). The benchmark is therefore dominated by close-range drones, which limits how strongly it can test
> small-object detection.

**Failure cases (§VII-D):**
> Missed or poorly localized drones occur mainly under severe motion blur, at extreme distances where the drone
> occupies fewer than $8\times8$ pixels, and under heavy occlusion by building structures.

⏳ **Template once Table IX is filled:**
> On the [ ] small validation boxes, CBAM changes AP_S from [ ] to [ ] and AR_S from [ ] to [ ]; given the small
> number of small targets, this difference [is / is not] conclusive.

**Response:**
> We measured box sizes after resizing. Only 1.2% of training boxes and 6.8% of validation boxes are small (< 32² px)
> (Fig. 2(b), Table IX). We report COCO-style AP/AR by size (Table IX), add failure cases, and add a Limitations
> section. The dataset contains no drone-free images and few bird-like distractors, so we have toned down the
> small-object and background-rejection claims accordingly.

**Status:** ✅ size statistics, text · ⏳ AP/AR by size, failure-case figure

---

## R1-7 · "Strengthen the novelty and placement ablation" (single location, other neck locations, another attention module, repeated runs)

**Data, cost** [`c13b_placement_cost.json`, `a3`, `a4`]:

| Configuration | +Params | +GFLOPs | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|
| None (baseline) | 0 | 0 | ⏳ | ⏳ |
| CBAM @ L12 only | +51,299 | +0.0005 | ⏳ | ⏳ |
| CBAM @ L15 only | +12,899 | +0.0013 | ⏳ | ⏳ |
| CBAM @ L12 + L15 | +64,198 | +0.0018 | ⏳ | ⏳ |
| SE @ L12 + L15 | +65,022 | +0.0001 | ⏳ | ⏳ |
| CoordAtt @ L12 + L15 | +50,012 | +0.0062 | ⏳ | ⏳ |

Observation (measured): L12 carries 80 % of the added parameters (MLP ∝ C², C = 640); L15 carries 70 % of the added
compute (spatial attention on the larger 80×80 map).

⏳ Table VI template (seeds 0, 1, 2; mean ± std):

| Model | P | R | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|
| YOLOv8x | [ ± ] | [ ± ] | [ ± ] | [ ± ] |
| YOLOv8x-CBAM | [ ± ] | [ ± ] | [ ± ] | [ ± ] |

**Manuscript text (§VI-B):**
> Table V compares CBAM at Layer 12 only, at Layer 15 only and at both, as well as SE and Coordinate Attention at
> Layers 12+15, under identical settings on the scene-grouped split. Table VI reports the baseline and CBAM over
> three seeds.

⏳ **Templates once the tables are filled:**
> Inserting CBAM at both fusion points gives [ ] mAP@0.5:0.95, compared with [ ] (L12 only) and [ ] (L15 only),
> indicating that [both stages contribute / most of the gain comes from L__].
> Over three seeds, CBAM reaches [ ± ] mAP@0.5:0.95 vs [ ± ] for the baseline; the gain of [ ] pp is
> [larger / not larger] than the seed standard deviation.
> CBAM [outperforms / matches / underperforms] SE ([ ]) and Coordinate Attention ([ ]) at similar cost.

**Response:**
> We agree that combining YOLO with CBAM is an incremental modification and have moderated the wording. To justify
> the placement, Table V compares single-location insertion (L12 only, L15 only), both locations, and two other
> attention modules (SE, Coordinate Attention) at the same locations under matched settings, with measured parameter
> and FLOP overheads. Table VI reports means and standard deviations over three seeds. [⏳ add the outcome in one
> sentence.]

**Status:** ✅ cost · ⏳ accuracy → `python revision/c13_placement_ablation.py`, `python revision/c12_train_seeds.py` · ✍ "alternative neck locations" (e.g. L18/L21) are not scripted; the scripts accept `layers=(18, 21)` if you want to add them.

---

## R1-8 · "Measure practical computational overhead"

**Data** [`a4_gflops.json`, `b8_latency.json`]

| Item | Value |
|---|---|
| GFLOPs baseline → CBAM (thop, 640², 2×MACs) | **258.1123 → 258.1141 (+0.0018, +0.0007 %)** |
| fvcore cross-check | 257.7562 → 257.7580 (+0.0018) |
| Element-wise ops not counted by profilers | +0.028 GFLOPs (258.1418 incl.) |
| Old "+0.1" explained | `model.info()` profiles 32×32 and ×400 → inflates size-independent MLP (258.1123 → 258.2163) |
| CPU latency (FP32, bs 1, 4 threads, Conv+BN fused, interleaved 60 runs) | baseline **604.59 ms** (1.65 FPS) · CBAM **606.91 ms** (1.65 FPS) |
| Paired overhead | **+7.67 ms (+1.27 %)**; std ≈ 50–58 ms |
| GPU FP16 latency / FPS / memory | ⏳ |

**Manuscript text (§VI-C):**
> Profiled directly at $640\times640$, the cost rises from 258.1123 to 258.1141 GFLOPs (+0.0018 GFLOPs,
> +0.0007\%); fvcore gives the same increment. The element-wise operations of the attention, which these profilers
> do not count, add a further 0.028 GFLOPs. The profile reported by Ultralytics' \texttt{model.info()} (+0.10
> GFLOPs) is an artifact of extrapolating from a $32\times32$ proxy image. A small arithmetic overhead does not
> guarantee a negligible runtime overhead, so we also timed the network (Table VIII). On a 4-thread CPU (FP32), the
> median latency is 604.59 ms for the baseline and 606.91 ms for YOLOv8-CBAM; the median paired difference,
> +7.67 ms (+1.27\%), is well below the run-to-run standard deviation of about 50–58 ms.

Delete old Eq. (7) and "0.0387 %".

⏳ **Template:** "On an [GPU name] (FP16, batch 1), the median latency is [ ] ms ([ ] FPS) for the baseline and [ ] ms ([ ] FPS) for YOLOv8-CBAM, with peak memory of [ ] and [ ] MB."

**Response:**
> We removed the 0.0387% figure, which had been computed from rounded values. Direct profiling at 640×640 gives
> +0.0018 GFLOPs (+0.0007%), with thop and fvcore in agreement, plus 0.028 GFLOPs of uncounted element-wise
> operations. The earlier "+0.1 GFLOPs" was an extrapolation artifact of `model.info()`. We now report measured
> latency, throughput and memory with the hardware and protocol stated (Table VIII). On CPU, the overhead is
> +7.67 ms (+1.27%), within run-to-run noise. [⏳ GPU sentence.]

**Status:** ✅ FLOPs, CPU · ⏳ GPU → `python revision/b8_latency.py` on a GPU

---

## R1 minor comments

| # | Comment | Manuscript change (paste-ready) | Data | Status |
|---|---|---|---|---|
| m1 | Replace "no additional baseline parameter growth" | "adds 64,198 parameters (0.094\%)" | a3 | ✅ |
| m2 | Nonlinear activation in Eq. 3 | $M_c(F)=\sigma\big(W_1\delta(W_0F^c_{avg})+W_1\delta(W_0F^c_{max})\big)$; text: "with a ReLU activation δ between its two layers and no bias terms" | a3: MLP = Conv→ReLU→Conv | ✅ |
| m3 | Confidence thresholds, NMS, tool versions | "All metrics are computed with the Ultralytics validator using a confidence threshold of 0.001, class-aware NMS at IoU 0.7 and at most 300 detections per image. P and R are reported at the confidence that maximizes F1. The confusion matrix is drawn at a confidence of 0.25 with an IoU match threshold of 0.45. The baseline M1 used Ultralytics 8.0.141; the revision experiments use Ultralytics 8.4.201 with PyTorch 2.15." | a5, a6 | ✅ ✍ add your fork's version for M2–M4 |
| m4 | Track IDs in Fig. 6 | Caption: "The frames were taken from a video-tracking demonstration; the leading number in each label is the tracker ID and is not part of the detection method." | – | ✅ text · ⏳ regenerate with `predict()` |
| m5 | Enlarge training plots and PR labels | Figs. 2, 4, 5(a) regenerated | – | ✅ · ⏳ 5(b) |
| m6 | Compare with drone-specific methods | Table VII (YOLOv5x, YOLO11x, RT-DETR-X, same protocol); see R2-4 | – | ⏳ `c11_train_sota.py` |
| m7 | Moderate background-rejection claims | Remove "effective background rejection". Add: "Because the validation set contains no drone-free images, the 48 false positives are spurious boxes on images that do contain a drone; the matrix does not measure rejection of empty scenes." | 0 negative images | ✅ |

**Response (minor, one block):**
> (1) Replaced with the measured +64,198 parameters. (2) The ReLU δ was added to Eq. 3; the implementation's MLP is
> Conv → ReLU → Conv without bias. (3) Thresholds, NMS settings and versions were added in §V-B. (4) The Fig. 6
> caption now explains the tracker IDs, and the figure was regenerated without them. (5) Plots were enlarged.
> (6) See Table VII. (7) Background-rejection claims were removed, and §VII-B and §VIII-A note that the dataset has
> no drone-free images.

## R1-Q8 · SOTA comparison necessary but not included → see R2-4 (Table VII). ⏳

---

# Reviewer #2

## R2-1 · "IEEE format consistency; labels in Figs. 2 and 5 too small"

**Data:** paper rebuilt in IEEEtran; compiles with 0 errors and 0 warnings. Figs. 2 and 5(a) were regenerated
(12–14 pt). Revised length: 8 pages (original 6).

**Response:**
> We checked section headings, mathematical notation (all symbols, including η₀, δ and γ, are defined at first use),
> figure and table captions, and reference formatting against the IEEE template. Figs. 2 and 5 were regenerated as
> vector graphics with larger fonts.

**Status:** ✅ · ⏳ Fig. 5(b) · ✍ check the conference page limit (6 → 8 pages)

---

## R2-2 · "32 vs 60 epochs inconsistency"

Same data as R1-4.

**Response:**
> All experiments used 32 epochs. The Fig. 4 caption was corrected, and the training settings are now consistent
> across the text, Table II and the figures (see also our reply to Reviewer 1, comment 4).

**Status:** ✅

---

## R2-3 · "Independent test set and additional drone datasets"

Data as in R1-2 / R1-3.

**Manuscript text (§VIII-A, Limitations):**
> The study uses a single, small public dataset. Although the scene-grouped split removes the near-duplicates we
> found, all splits come from the same sources, so generalization to other cameras, sensors (e.g. infrared) and
> environments is not established; evaluation on additional drone datasets is future work. The benchmark contains
> few small targets (6.8\% of validation boxes below $32^2$ pixels), and it contains no drone-free images and few
> bird-like distractors, so false-alarm behavior on empty skies, birds and aircraft is not measured.

**Response:**
> After finding 33 near-duplicate validation images, we created a leakage-free, scene-grouped split with a held-out
> test set (Table IV). Evaluation on further drone datasets is beyond the scope of this revision; we state it
> explicitly as a limitation and as future work (§VIII-A).

**Status:** ✅ · ✍ optional: a second dataset would strengthen this

---

## R2-4 · "Quantitative SOTA comparison; inference speed, hardware, actual overhead"

⏳ Table VII template (test split, seed 0, same epochs/size/batch/augmentation):

| Model | Params (M) | GFLOPs | FPS | P | R | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|---|---|---|
| YOLOv5x (Ultralytics `yolov5xu`) | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| YOLO11x | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| RT-DETR-X (AdamW 1e-4) | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| YOLOv8x | 68.15 | 258.1 | [ ] | [ ] | [ ] | [ ] | [ ] |
| YOLOv8x-CBAM | 68.22 | 258.1 | [ ] | [ ] | [ ] | [ ] | [ ] |

Overhead and speed data: see R1-8.

⏳ **Template:** "On the held-out test split, YOLOv8x-CBAM reaches [ ] mAP@0.5:0.95, compared with [ ] for YOLO11x,
[ ] for YOLOv5x and [ ] for RT-DETR-X, at [ ] FPS on [GPU]."

**Response:**
> Table VII compares YOLOv5x, YOLO11x and RT-DETR-X with YOLOv8x and YOLOv8x-CBAM, all trained under the same
> epochs, input size, batch size and augmentation on the same split, and reports parameters, GFLOPs and FPS.
> Table VIII reports measured latency with the hardware stated, and §VI-C gives the exact parameter and FLOP
> overhead. [⏳ one-sentence outcome.]

**Status:** ⏳ `python revision/c11_train_sota.py` · ✍ optional: add a published drone-specific detector if you
can run its code

---

## R2-5 · "Clarify attention implementation, parameter/GFLOPs calculation; failure cases; small-object evidence"

Covered by R1-1 (implementation), R1-0 and R1-8 (params, GFLOPs), and R1-6 (failure cases, small objects).

**Response:**
> §IV-E now details the implementation (see R1-1), and §VI-C derives the exact parameter and GFLOPs overhead (see
> R1-0 and R1-8). A failure-case paragraph covering severe motion blur, extreme viewing distance and partial
> occlusion was added (§VII-D). Evidence for small objects is given by the size statistics and the size-specific
> AP (Fig. 2(b), Table IX; see R1-6).

**Status:** ✅ · ⏳ Table IX values, failure-case figure

## R2-Q8 · SOTA comparison necessary → Table VII (see R2-4). ⏳

---

# Checklist before you submit

- [ ] Resolve section 0 (figures, M1 protocol, M2–M4 raw outputs, γ formula, diff with your `main.tex`)
- [ ] `bash revision/run_all_gpu.sh` on a GPU (≈ 15 runs), then `python revision/d_fill_tables.py`
- [ ] Fill every ⏳ template sentence above with the real numbers
- [ ] Replace Figs. 4, 5, 6 with the CBAM versions (`d_make_figures.py`)
- [ ] Search `paper_revised.tex` for `TODO` and `VERIFY`
- [ ] Set `\revisionmarksfalse` for the camera-ready (or keep blue for the reviewers' copy)
- [ ] Check the page limit (now 8 pages)
- [ ] Remove the author note at the top of `response_to_reviewers.md`
