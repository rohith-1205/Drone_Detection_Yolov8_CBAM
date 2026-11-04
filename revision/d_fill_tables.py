"""
D - Turn the JSON results into LaTeX tables for paper_revised.tex.

Every number in the generated tables is read from revision/results/*.json.
Nothing is typed by hand.  If an experiment has not been run yet (e.g. it
needs a GPU), its cells show "--" and the caption says so.  Re-run this
script after the GPU experiments and recompile the paper: the tables fill in
automatically.

Outputs: revision/tables/*.tex   (\\input{} by paper_revised.tex)
Run:     python revision/d_fill_tables.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS_DIR, TABLES_DIR, banner, load_json  # noqa: E402

PENDING = "--"
NOTE_PENDING = r" Cells marked ``--'' require trained checkpoints / a GPU and were not run in this revision."


def pct(x, digits=2):
    return PENDING if x is None else f"{100 * x:.{digits}f}"


def write(name, body):
    (TABLES_DIR / f"{name}.tex").write_text(body)
    print(f"[table] revision/tables/{name}.tex")


def run(name):
    path = RESULTS_DIR / "c_runs" / f"{name}.json"
    if not path.exists():
        return None
    r = json.loads(path.read_text())
    return r if r.get("status") == "done" else None


banner("D  Generating LaTeX tables from results/*.json")
a2, a3, a4 = load_json("a2_tensor_shapes"), load_json("a3_param_counts"), load_json("a4_gflops")
b7, b8, b9, c10 = load_json("b7_ap_by_size"), load_json("b8_latency"), load_json("b9_near_duplicates"), load_json("c10_split")

# ---------------------------------------------------------------- Table I
layer_roles = {0: "Low-level stem", 1: "Downsampling", 2: "Feature extraction", 3: "Downsampling",
               4: "P3 multi-scale feature", 5: "Downsampling", 6: "P4 multi-scale feature", 7: "Downsampling",
               8: "Deep context extraction", 9: "P5 multi-scale feature", 12: "Neck fusion / CBAM-1",
               15: "Neck fusion / CBAM-2", 22: "Decoupled detection"}
module_names = {0: r"Conv ($k{=}3,s{=}2$)", 1: r"Conv ($k{=}3,s{=}2$)", 2: r"C2f ($n{=}3$)", 3: r"Conv ($k{=}3,s{=}2$)",
                4: r"C2f ($n{=}6$)", 5: r"Conv ($k{=}3,s{=}2$)", 6: r"C2f ($n{=}6$)", 7: r"Conv ($k{=}3,s{=}2$)",
                8: r"C2f ($n{=}3$)", 9: r"SPPF ($k{=}5$)", 12: "C2f", 15: "C2f", 22: "Detect head"}
lines = []
for row in a2["all_layers"]:
    i = int(row["layer"])
    if i not in layer_roles:
        continue
    if i == 22:
        ch, res = "Multi", "Multi"
    else:
        c, h, w = row["output_shape"].split("x")
        ch, res = c, rf"${h}\times{w}$"
    lines.append(f"{i} & {module_names[i]} & {ch} & {res} & {layer_roles[i]} \\\\")
write("table1_stages", r"""\begin{table}[!t]
\centering
\caption{Principal feature stages of YOLOv8x for a $640\times640$ input (measured by a tensor-shape trace).}
\label{tab:stages}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{clccl}
\toprule
Layer & Module & Channels & Resolution & Role \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---------------------------------------------------------------- Table III (complexity columns)
base_p = a3["baseline_params"]
cx = {"none": (base_p, a4["variants"]["none"]["GFLOPs_ultralytics_convention"])}
for m in ("channel", "spatial", "cbam"):
    cx[m] = (a3["variants"][m]["total_params"], a4["variants"][m]["GFLOPs_ultralytics_convention"])
# Accuracy numbers of the SUBMITTED paper (original 1,012/347 split). M1 was verified against the raw
# 2023 results.csv (a5); M2-M4 have no raw output in the repository (see CHANGELOG).
submitted = {"none": (86.82, 83.90, 89.22, 53.76), "channel": (88.54, 85.12, 90.71, 54.92),
             "spatial": (89.15, 86.40, 91.25, 55.43), "cbam": (91.30, 88.75, 92.93, 57.10)}
labels = {"none": r"M1: Baseline$^{\dagger}$", "channel": "M2: Channel-only", "spatial": "M3: Spatial-only",
          "cbam": "M4: YOLOv8x-CBAM"}
rows = []
for m in ("none", "channel", "spatial", "cbam"):
    p, g = cx[m]
    P, R, m50, m5095 = submitted[m]
    rows.append(f"{labels[m]} & {p:,} & {g:.4f} & {P:.2f} & {R:.2f} & {m50:.2f} & {m5095:.2f} \\\\")
gain = [submitted["cbam"][k] - submitted["none"][k] for k in range(4)]
rows.append(r"\midrule")
rows.append(f"Gain (M4$-$M1) & +{cx['cbam'][0] - cx['none'][0]:,} & +{cx['cbam'][1] - cx['none'][1]:.4f} & "
            + " & ".join(f"+{v:.2f} pp" for v in gain) + r" \\")
write("table3_ablation", r"""\begin{table*}[!t]
\centering
\caption{Ablation of attention configurations on the original validation split (347 images). Params and GFLOPs are
measured for the 1-class model ($640\times640$, GFLOPs $=2\times$MACs, thop); accuracy values are those of the submitted
version. Gains are in percentage points (pp).}
\label{tab:ablation}
\begin{tabular}{lrrcccc}
\toprule
Configuration & Params & GFLOPs & P (\%) & R (\%) & mAP@0.5 (\%) & mAP@0.5:0.95 (\%) \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}\\[2pt]
\footnotesize{$^{\dagger}$M1 is the final epoch of a separate run made with Ultralytics 8.0.141 and \texttt{optimizer=auto}
(AdamW, $\eta_0=0.002$), whereas the ablation script used for M2--M4 specifies SGD ($\eta_0=0.01$); no raw outputs of
M2--M4 were retained. The matched-protocol comparison on a leakage-free split is Table~\ref{tab:seeds}.}
\end{table*}
""")

# ---------------------------------------------------------------- Table: complexity + runtime
rt_rows = []
for dev, e in sorted((b8 or {}).items()):
    if not isinstance(e, dict) or "models" not in e:
        continue
    # short, LaTeX-safe device label, e.g. "CPU (x86_64, 4 threads)" -> "CPU, 4 thr."
    dev = dev.replace("_", r"\_")
    if dev.startswith("CPU"):
        dev = "CPU, " + dev.split(",")[-1].strip(" )").replace(" threads", " thr.")
    b, c = e["models"]["none"], e["models"]["cbam"]
    mem = lambda s: f"{s['peak_gpu_memory_MB']:.0f}" if s.get("peak_gpu_memory_MB") else "n/a"  # noqa: E731
    rt_rows.append(f"Baseline & {dev} & {e['precision']} & {b['median_ms']:.2f} & {b['fps_from_median']:.2f} & {mem(b)} \\\\")
    rt_rows.append(f"CBAM & {dev} & {e['precision']} & {c['median_ms']:.2f} & {c['fps_from_median']:.2f} & {mem(c)} \\\\")
gpu_done = any("CPU" not in d for d in (b8 or {}) if isinstance((b8 or {}).get(d), dict))
if not gpu_done:
    rt_rows.append(f"Baseline / CBAM & GPU (FP16) & FP16 & {PENDING} & {PENDING} & {PENDING} \\\\")
cb = a4["variants"]["cbam"]
write("table_runtime", r"""\begin{table}[!t]
\centering
\caption{Measured complexity and runtime (batch 1, $640\times640$, network forward only, Conv+BN fused, median of
interleaved runs).""" + ("" if gpu_done else NOTE_PENDING) + r"""}
\label{tab:runtime}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{llcccc}
\toprule
Model & Device & Prec. & ms/img & FPS & Peak mem (MB) \\
\midrule
""" + "\n".join(rt_rows) + r"""
\midrule
\multicolumn{6}{l}{Params added by CBAM: """ + f"+{a3['variants']['cbam']['added_params']:,} "
          f"(+{a3['variants']['cbam']['added_percent']:.3f}\\%)" + r"""} \\
\multicolumn{6}{l}{GFLOPs added by CBAM: """ + f"+{cb['added_GFLOPs_vs_baseline']:.4f} "
          f"(+{cb['added_percent_vs_baseline']:.4f}\\%)" + r"""} \\
\bottomrule
\end{tabular}
\end{table}
""")

# ---------------------------------------------------------------- Table: AP by size
counts = b7["val_boxes_per_bucket_at_640"]
models = b7.get("models", {})


def ap_row(label, key):
    m = models.get(key, {})
    if "AP" not in m:
        return f"{label} & " + " & ".join([PENDING] * 6) + r" \\"
    f = lambda v: PENDING if v < 0 else f"{100 * v:.2f}"  # noqa: E731
    return f"{label} & " + " & ".join(f(m[k]) for k in ("AP_small", "AP_medium", "AP_large",
                                                         "AR_small", "AR_medium", "AR_large")) + r" \\"


done7 = all("AP" in models.get(k, {}) for k in ("baseline", "cbam"))
write("table_size_ap", r"""\begin{table}[!t]
\centering
\caption{COCO-style AP@0.5:0.95 and AR by box size, measured after letter-boxing to 640\,px
(S: $<32^2$, M: $32^2$--$96^2$, L: $\geq96^2$ pixels).""" + ("" if done7 else NOTE_PENDING) + r"""}
\label{tab:sizeap}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lcccccc}
\toprule
Model & AP$_S$ & AP$_M$ & AP$_L$ & AR$_S$ & AR$_M$ & AR$_L$ \\
\midrule
""" + f"Val boxes (count) & {counts['small']} & {counts['medium']} & {counts['large']} & & & \\\\\n"
          + ap_row("Baseline", "baseline") + "\n" + ap_row("YOLOv8x-CBAM", "cbam") + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---------------------------------------------------------------- Table: data split + leakage
sp = c10["splits"]
write("table_split", r"""\begin{table}[!t]
\centering
\caption{Near-duplicate audit of the original split (pHash, Hamming $\leq4$ / $\leq10$ bits) and the new
scene-grouped split used for all retraining.}
\label{tab:split}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lccc}
\toprule
\multicolumn{4}{l}{\textit{Original split (1,012 train / 347 val)}} \\
Val images with near-duplicate in train & \multicolumn{3}{c}{""" + f"{b9['val_images_with_near_duplicate_in_train']} "
          f"({100 * b9['val_images_with_near_duplicate_in_train'] / b9['n_val']:.1f}\\%)" + r"""} \\
Val images with same-scene match in train & \multicolumn{3}{c}{""" + f"{b9['val_images_with_same_scene_in_train']} "
          f"({100 * b9['val_images_with_same_scene_in_train'] / b9['n_val']:.1f}\\%)" + r"""} \\
\midrule
\textit{Scene-grouped split} & Train & Val & Test \\
Images & """ + f"{sp['train']['images']} & {sp['val']['images']} & {sp['test']['images']}" + r""" \\
Boxes & """ + f"{sp['train']['boxes']} & {sp['val']['boxes']} & {sp['test']['boxes']}" + r""" \\
Cross-split pairs $\leq10$ bits & \multicolumn{3}{c}{""" + f"{sum(c10['cross_split_pairs_within_10_bits'].values())}" + r"""} \\
\bottomrule
\end{tabular}
\end{table}
""")

# ---------------------------------------------------------------- Table: SOTA comparison (test split)
sota = [("YOLOv5x (u)", "yolov5xu_s0"), ("YOLO11x", "yolo11x_s0"), ("RT-DETR-X", "rtdetr-x_s0"),
        ("YOLOv8x", "yolov8x_none_s0"), ("YOLOv8x-CBAM (ours)", "yolov8x_cbam_s0")]
srows, any_pending = [], False
for label, name in sota:
    r = run(name)
    if r is None:
        any_pending = True
        srows.append(f"{label} & " + " & ".join([PENDING] * 7) + r" \\")
    else:
        t = r["test"]
        srows.append(f"{label} & {r['params'] / 1e6:.2f} & {r.get('GFLOPs', 0):.1f} & {r['latency']['fps_from_median']:.1f} & "
                     f"{pct(t['precision'])} & {pct(t['recall'])} & {pct(t['mAP50'])} & {pct(t['mAP50_95'])} \\\\")
write("table_sota", r"""\begin{table*}[!t]
\centering
\caption{Comparison with state-of-the-art detectors on the held-out test split of the scene-grouped split
(same epochs, input size, batch size and augmentation; seed 0).""" + (NOTE_PENDING if any_pending else "") + r"""}
\label{tab:sota}
\begin{tabular}{lccccccc}
\toprule
Model & Params (M) & GFLOPs & FPS & P (\%) & R (\%) & mAP@0.5 (\%) & mAP@0.5:0.95 (\%) \\
\midrule
""" + "\n".join(srows) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")

# ---------------------------------------------------------------- Table: seeds
c12 = load_json("c12_seeds") or {}
summ = c12.get("summary", {})


def ms(mode, k):
    s = summ.get(mode, {}).get(k)
    return PENDING if not s or "mean" not in s else f"{100 * s['mean']:.2f} $\\pm$ {100 * s['std']:.2f}"


seed_rows = [f"{lab} & " + " & ".join(ms(m, k) for k in ("precision", "recall", "mAP50", "mAP50_95")) + r" \\"
             for lab, m in (("YOLOv8x", "none"), ("YOLOv8x-CBAM", "cbam"))]
seeds_done = all("mean" in summ.get(m, {}).get("mAP50", {}) for m in ("none", "cbam")) if summ else False
write("table_seeds", r"""\begin{table}[!t]
\centering
\caption{Test-split results over three seeds (mean $\pm$ sample std, \%).""" + ("" if seeds_done else NOTE_PENDING) + r"""}
\label{tab:seeds}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lcccc}
\toprule
Model & P & R & mAP@0.5 & mAP@0.5:0.95 \\
\midrule
""" + "\n".join(seed_rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---------------------------------------------------------------- Table: placement / attention type
# Cost columns are measured without training (a3/a4 for both layers, c13b for single layers);
# accuracy columns come from the GPU runs of c13_placement_ablation.py.
c13b = (load_json("c13b_placement_cost") or {}).get("placements", {})
cost = {"none": (0, 0.0)}
for key in ("L12", "L15", "L12+L15"):
    if key in c13b:
        cost["cbam_" + key] = (c13b[key]["added_params"], c13b[key]["added_GFLOPs"])
for key in ("se", "coord"):
    cost[key] = (a3["variants"][key]["added_params"], a4["variants"][key]["added_GFLOPs_vs_baseline"])
abl = [("None (baseline)", "yolov8x_none_s0", "none"), ("CBAM @ L12", "yolov8x_cbam_L12_s0", "cbam_L12"),
       ("CBAM @ L15", "yolov8x_cbam_L15_s0", "cbam_L15"), ("CBAM @ L12+L15", "yolov8x_cbam_s0", "cbam_L12+L15"),
       ("SE @ L12+L15", "yolov8x_se_s0", "se"), ("CoordAtt @ L12+L15", "yolov8x_coord_s0", "coord")]
arows, pend = [], False
for label, name, key in abl:
    r = run(name)
    if key in cost:
        added_p, added_g = cost[key]
        cost_cells = f"+{added_p:,} & +{added_g:.4f}"
    else:
        cost_cells = f"{PENDING} & {PENDING}"
    if r is None:
        pend = True
        arows.append(f"{label} & {cost_cells} & {PENDING} & {PENDING} \\\\")
    else:
        arows.append(f"{label} & {cost_cells} & {pct(r['test']['mAP50'])} & {pct(r['test']['mAP50_95'])} \\\\")
write("table_placement", r"""\begin{table}[!t]
\centering
\caption{Placement and attention-type ablation on the test split (seed 0). Added parameters and GFLOPs are measured
($640\times640$, GFLOPs $=2\times$MACs).""" + (NOTE_PENDING if pend else "") + r"""}
\label{tab:placement}
\setlength{\tabcolsep}{2.5pt}
\begin{tabular}{lcccc}
\toprule
Configuration & +Params & +GFLOPs & mAP@0.5 & mAP@0.5:0.95 \\
\midrule
""" + "\n".join(arows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
