# `revision/` – reviewer-response work for paper #633

This folder holds everything that was done for the revision. Your original files
(`colab_ablation.py`, `scratch/`, `drone_dataset/`) were **not changed**.

| Start here | What it is |
|---|---|
| [`SUMMARY.md`](SUMMARY.md) | One-page table of every finding and result, plus the list of skipped items |
| [`CHANGELOG.md`](CHANGELOG.md) | Detailed log: what was checked, what was found, what was changed |
| [`response_to_reviewers.md`](response_to_reviewers.md) | Point-by-point answer to Reviewer 1 and Reviewer 2 |
| [`paper_revised.tex`](paper_revised.tex) / `paper_revised.pdf` | Revised paper (revised text in **blue**; switch off with `\revisionmarksfalse`) |

## Scripts (one task each, simple and commented)

| Script | Task | Needs | Output |
|---|---|---|---|
| `a1_attachment_check.py` | How CBAM is attached; frozen? optimizer? checkpoint? ONNX? | CPU | `results/a1_attachment.json` |
| `a2_tensor_shapes.py` | Real tensor shape at every layer | CPU | `results/a2_tensor_shapes.{json,csv}` |
| `a3_param_counts.py` | Exact parameter counts, ReLU check | CPU | `results/a3_param_counts.json` |
| `a4_gflops.py` | Unrounded GFLOPs (thop + fvcore) | CPU | `results/a4_gflops.json` |
| `a5_training_settings.py` | Settings the runs really used | CPU | `results/a5_training_settings.json` |
| `a6_validate.py` | Re-validation + confusion-matrix explanation | trained weights | `results/a6_validation.json` |
| `b7_ap_by_size.py` | COCO AP small/medium/large at 640 px | trained weights | `results/b7_ap_by_size.json` |
| `b8_latency.py` | ms/img, FPS, peak memory (FP16 on GPU) | GPU for FP16 | `results/b8_latency.json` |
| `b9_near_duplicates.py` | pHash duplicate check train ↔ val | CPU | `results/b9_*.{json,csv}` |
| `c10_make_split.py` | Leakage-free scene-grouped 70/15/15 split | CPU | `splits/scene_split/*.txt` |
| `c11_train_sota.py` | YOLOv5x, YOLO11x, RT-DETR-X comparison | GPU | `results/c11_sota.{csv,json}` |
| `c12_train_seeds.py` | Baseline + CBAM × 3 seeds, mean ± std | GPU | `results/c12_seeds.{csv,json}` |
| `c13_placement_ablation.py` | CBAM @12 / @15 / both, SE, CoordAtt | GPU | `results/c13_ablation.{csv,json}` |
| `c13b_placement_cost.py` | Params / GFLOPs of CBAM @ L12, @ L15, @ both | CPU | `results/c13b_placement_cost.json` |
| `d_make_figures.py` | Figs. 2, 4, 5, 6 with large fonts | (weights for 5/6) | `figures/*.pdf` |
| `d_fill_tables.py` | LaTeX tables straight from the JSON results | – | `tables/*.tex` |

Shared code: `common.py` (paths and helpers), `attention_modules.py` (CBAM / SE /
CoordAtt and the layer wrapper), `train_utils.py` (one train-and-test function).

## Finishing the GPU experiments

```bash
export CBAM_WEIGHTS=/path/to/your/cbam/best.pt         # optional
export BASELINE_WEIGHTS=/path/to/your/baseline/best.pt # optional
bash revision/run_all_gpu.sh
```

Finished runs are cached, so you can restart the script after a Colab disconnect.
The tables in the paper fill in automatically, and the blue `--` cells disappear.
Then search `paper_revised.tex` for `TODO(GPU)` and write 2–3 sentences per new table.

## Folders

- `results/`: raw outputs of every script. `results/smoke/` holds **pipeline tests only** (1 epoch, 2 % of data, 320 px), never use these numbers.
- `evidence/`: files copied from `github.com/rohith-1205/Drone-Detection-1` (the 2023 baseline run) and the figures extracted from the submitted PDF, kept as proof for the findings.
- `figures/`, `tables/`: generated for the paper.
- `splits/scene_split/`: the new train/val/test lists (paths relative to `drone_dataset/`).
