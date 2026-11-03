# Response to Reviewers – Paper ID 633

**"An Attention-Guided Neck Feature Fusion Framework for Drone Detection Using YOLOv8 and CBAM"**, IConSCEPT 2026

We thank both reviewers for their careful reading and constructive comments. We have revised the
manuscript accordingly. All revised text is shown in **blue** in the revised PDF. Section, table, figure and equation
numbers below refer to the **revised** manuscript (Tables I–III keep their original numbers; Tables IV–IX are new).
Every new number was produced by a script in our released `revision/` folder, which also stores the raw outputs.

> **Note for the authors (delete before sending).** Items marked **[GPU]** depend on experiments that were scripted
> but not yet run, because the revision environment had no GPU and no trained checkpoints. Run
> `bash revision/run_all_gpu.sh`, check that Tables V–IX are filled, and replace each **[GPU]** sentence with the
> actual outcome. **Do not send this letter while any [GPU] marker remains.** Also resolve the two
> **⚠ integrity items** at the end of this note first.
>
> ⚠ **Integrity items found during the fact check (see `CHANGELOG.md`, A5/A6):**
> 1. The original Figs. 4 and 5, captioned "YOLOv8-CBAM", are the outputs of the **plain YOLOv8x baseline** run of
>    2023 (identical confusion-matrix counts; the PR legend reads mAP@0.5 = 0.892 = baseline). The revised captions
>    say so. Before submission, replace them with the CBAM run's figures (`d_make_figures.py` with `CBAM_WEIGHTS`).
> 2. The M1 baseline row equals the last epoch of a separate 2023 run (Ultralytics 8.0.141, `optimizer=auto` →
>    AdamW, lr 0.002), not the SGD/0.01 protocol of Table II used for M2–M4, and no raw outputs of M2–M4 were found.
>    The revised Table III footnote states this. The matched comparison (Table VI) must be completed before the gains
>    are claimed again.

---

## Reviewer #1

**Q3. Quality of figures is not satisfactory.**
> Figs. 2 and 4 were regenerated as vector graphics with 12–14 pt fonts, and Fig. 5(a) was redrawn with large
> labels. Fig. 2(b), the unreadable overlay of all boxes, was replaced by a histogram of box sizes at 640 px with the
> COCO small/large borders. **[GPU]** Fig. 5(b) and Fig. 6 will be regenerated from the CBAM checkpoint with the same
> style.

**R1-0. Correct the parameter-overhead calculation.**
> We agree; the statement was wrong. With the actual channel widths (640 and 320, see R1-1), the two CBAM modules
> add **64,198 parameters** (51,299 + 12,899: shared MLP $2C^2/r$, 98 weights for the 7×7 convolution, and one gate
> each). This raises the 1-class YOLOv8x from 68,153,571 to 68,217,769 parameters (+0.094%). The value was measured
> with `sum(p.numel())` and matches the hand calculation. "Fewer than 0.01 M" and "no additional parameter growth"
> were removed (Abstract, Contributions, §VI-C, Table III, Conclusion).

**R1-1. Clarify the architecture and hook implementation.**
> - *Channel dimensions:* a tensor-shape trace of a 640×640 input shows that Layer 12 outputs **640×40×40** and
>   Layer 15 outputs **320×80×80**. The values 512/256 were the unscaled YAML widths before YOLOv8x's 1.25 width
>   multiplier. Table I is now generated directly from this trace, and §IV-B/§IV-D were corrected.
> - *Implementation:* §IV-E was rewritten. The attention is attached by **module wrapping** (the C2f modules at
>   indices 12 and 15 are replaced by a wrapper that runs the original block and then CBAM), not by PyTorch forward
>   hooks. A zero-initialised scalar gate γ blends CBAM's output with its input (new Eq. 7), so training starts exactly
>   from the pretrained network.
> - *Registration, optimizer, gradients:* we verified that all eight CBAM tensors, including both γ, are in the
>   optimizer parameter groups. A short training run moved γ away from zero, which confirms gradient flow.
> - *Checkpoints and export:* wrapped modules and γ values are saved in and restored from Ultralytics checkpoints,
>   and the model exports to ONNX (opset 17) with a maximum deviation of 9.7×10⁻⁴ from PyTorch.
> - *Pretrained initialisation vs. freezing:* we now state explicitly that the wrapper is inserted *after* loading the
>   COCO weights (preserving initialisation) and that the original weights are **not frozen**; all parameters except
>   the fixed DFL projection are fine-tuned jointly. "Without modifying the original model parameters" was removed.

**R1-2. Provide an independent test evaluation.**
> We created a new scene-grouped split with a held-out **test set of 187 images** (§III-A, Table IV). Model and
> checkpoint selection use only the validation part, and the test part is evaluated once per configuration. For the
> original split we now state that the reported checkpoint is the best-fitness checkpoint on the validation data
> (§V-A). **[GPU]** Test-set results for all configurations are reported in Tables V–VII.

**R1-3. Address possible image or sequence overlap.**
> We audited the dataset with 64-bit perceptual hashing. In the original split, **33 of 347 validation images
> (9.5%) have a near-duplicate in the training set** (30 identical pictures under different file names), and 48
> (13.8%) have a same-scene match (Table IV). We describe the sources (web photographs and frames from several
> videos, identifiable from the file names) and built a new split in which images linked by visual similarity or by
> neighbouring video frames are kept in the same partition, assigned per source family in a 70/15/15 ratio. The
> resulting split has **no cross-split pair within 10 bits** (§III-A, Table IV).

**R1-4. Resolve training-duration inconsistencies.**
> Every run in this work used **32 epochs**. The "60 epochs" in the Fig. 4 caption was an error and is corrected.
> Training settings and the checkpoint-selection rule (best Ultralytics fitness on validation) are now given for
> every configuration (§V-A, Table II). The footnote to Table II states that the M1 baseline was trained with
> Ultralytics 8.0.141 and `optimizer=auto` (AdamW, η₀ = 0.002), unlike the other runs.

**R1-5. Reconcile the confusion matrix with reported precision and recall.**
> The reviewer is right that 320/48/49 gives P = 86.96% and R = 86.72%. The difference comes from the operating
> point. Ultralytics draws the confusion matrix at **confidence 0.25 with an IoU match of 0.45**, whereas the P/R in
> the table are taken at the **confidence that maximises F1** (validator confidence 0.001, NMS IoU 0.7). Both
> conventions are now stated in §V-B and §VII-B. While checking this we also found that the matrix and PR curve in
> the original Fig. 5 belonged to the **baseline** model (its PR legend reads mAP@0.5 = 0.892). The caption is
> corrected, and **[GPU]** the figure will be replaced with the CBAM model's.

**R1-6. Demonstrate the claimed small-object benefit.**
> We measured box sizes after letter-boxing to 640 px. Only **13 of 1,117 training boxes (1.2%) and 25 of 369
> validation boxes (6.8%) are small** (< 32² px), and 81.8% of the validation boxes are large (Fig. 2(b), Table IX).
> We added COCO-style AP_S/M/L and AR_S/M/L (Table IX) and a failure-case paragraph (§VII-D). Because so few targets
> are small and the dataset has no drone-free images, we toned down the small-object claims and added a Limitations
> section (§VIII-A). **[GPU]** AP/AR by size for both models, and a figure with failure cases and bird-like
> distractors, will be added.

**R1-7. Strengthen the novelty and placement ablation.**
> Table V compares CBAM at Layer 12 only, at Layer 15 only and at both, plus SE and Coordinate Attention at
> Layers 12+15, all under identical settings. Table VI reports the baseline and CBAM over **three seeds (mean ±
> std)**. The measured parameter overheads of these blocks are already included: CBAM +64,198, SE +65,022,
> CoordAtt +50,012. **[GPU]** Accuracy values and their interpretation will be added. We also agree that combining
> YOLO with CBAM is an incremental change and have moderated the wording accordingly.

**R1-8. Measure practical computational overhead.**
> The "0.0387%" figure, computed from rounded values, was removed. Profiling directly at 640×640 (GFLOPs = 2 × MACs,
> thop and fvcore) gives **258.1123 → 258.1141 GFLOPs (+0.0018, +0.0007%)**. The element-wise attention operations,
> which these tools do not count, add 0.028 GFLOPs. The earlier "+0.1 GFLOPs" came from Ultralytics' `model.info()`,
> which extrapolates from a 32×32 proxy image and so inflates the size-independent MLP cost by (640/32)². We also
> measured runtime (Table VIII): on a 4-thread CPU (FP32, batch 1, interleaved timing) the median latency is
> 604.59 ms vs. 606.91 ms, a paired difference of +7.67 ms (+1.27%) that lies within the run-to-run std of
> ≈ 50–58 ms. **[GPU]** FP16 GPU latency, throughput and peak memory, with the GPU model, will be added to Table VIII.

**Minor comments**
> - *"No additional baseline parameter growth"* → replaced by the measured +64,198 parameters.
> - *Nonlinear activation in Eq. 3* → ReLU δ added; the implementation's MLP is Conv → ReLU → Conv without bias.
> - *Confidence thresholds, NMS, tool versions* → added in §V-B (conf 0.001, NMS IoU 0.7, max 300 detections;
>   confusion matrix at conf 0.25 / IoU 0.45; Ultralytics 8.0.141 for M1 and 8.4.201 for the revision runs; PyTorch 2.15).
> - *Track IDs in Fig. 6* → the caption now explains that the frames come from a tracking demonstration and that the
>   ID is not part of the method. **[GPU]** The figure will be regenerated with plain detection (`predict`) and no IDs.
> - *Enlarge training plots and PR labels* → done for Figs. 2, 4 and 5(a); **[GPU]** 5(b).
> - *Comparisons with drone-specific detection methods* → **[GPU]** Table VII compares YOLOv5x, YOLO11x and
>   RT-DETR-X trained under the same protocol on the same split.
> - *Moderate background-rejection claims* → "effective background rejection" was removed. §VII-B now notes that the
>   validation set contains no drone-free images, and §VIII-A states that background rejection is not claimed.

**Q8. Comparison with state-of-the-art methods.**
> See R1-7 and the minor comment above: Table VII (state-of-the-art comparison) was added. **[GPU]**

---

## Reviewer #2

**R2-1. Formatting consistency; Figs. 2 and 5 labels too small.**
> We checked section headings, notation (e.g. η₀, δ, γ defined at first use), and figure/table captions and
> references in IEEE style. Figs. 2 and 5(a) were regenerated as vector graphics with large fonts, and Fig. 2(b) was
> redesigned. **[GPU]** Fig. 5(b) will follow.

**R2-2. 32 vs. 60 training epochs.**
> All runs used 32 epochs. The Fig. 4 caption was corrected, and settings are consistent across the text, Table II
> and the figures (see R1-4).

**R2-3. Independent test set and additional datasets.**
> A leakage-free, scene-grouped split with a held-out test set was created after we found 33 near-duplicate
> validation images (see R1-2 and R1-3; Table IV). Evaluation on additional drone datasets is outside the scope of
> this revision. It is named explicitly as a limitation and as future work (§VIII-A).

**R2-4. Quantitative SOTA comparison, inference speed, hardware, actual overhead.**
> Table VII compares YOLOv5x, YOLO11x and RT-DETR-X with YOLOv8x and YOLOv8x-CBAM under the same epochs, input size,
> batch size and augmentation, and reports parameters, GFLOPs and FPS. Table VIII reports measured latency with the
> hardware stated, and §VI-C gives the exact parameter and FLOP overhead (see R1-8). **[GPU]**

**R2-5. Implementation of the attention modules, parameter/GFLOPs calculation, failure cases, small-object
evidence.**
> §IV-E now describes the implementation in detail (see R1-1), and §VI-C gives the exact parameter and GFLOPs
> derivation (see R1-0, R1-8). A failure-case paragraph covering motion blur, extreme distance and occlusion was
> added (§VII-D). The small-object evidence is addressed by the size statistics and size-specific AP (Fig. 2(b),
> Table IX; see R1-6).

**Q8. Comparison with state-of-the-art methods.**
> Added in Table VII (see R2-4). **[GPU]**

---

We hope that the revised manuscript addresses the reviewers' concerns, and we thank them again for helping us
improve the paper.
