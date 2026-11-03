"""
B9 - Are there (near-)duplicate or same-scene images across train and val?

Method (simple and standard):
  * Compute a 64-bit perceptual hash (pHash) of every image with `imagehash`.
    Two images that look alike have hashes that differ in only a few bits.
  * The "Hamming distance" = number of differing bits (0 = identical look).
        distance <= 4   -> near-duplicate (same frame, maybe re-encoded/resized)
        distance <= 10  -> very likely the same scene / neighbouring video frame
  * We also look at FILE NAMES: the dataset mixes several sources
    (pic_###, foto#####, yoto#####, scene#####, ####) and the 5-digit ones are
    video frame numbers, i.e. consecutive frames of the same clip.

Outputs:
  revision/results/b9_near_duplicates.json        summary numbers
  revision/results/b9_cross_split_pairs.csv       every train-val pair with distance <= 10
  revision/results/b9_image_hashes.csv            hash of every image (re-used by c10_make_split.py)
  revision/figures/b9_duplicate_examples.jpg      side-by-side examples to check by eye
Run:    python revision/b9_near_duplicates.py      (~1 minute on CPU)
"""
import csv
import re
import sys
from collections import Counter
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATASET_DIR, FIGURES_DIR, RESULTS_DIR, banner, save_json  # noqa: E402

NEAR_DUP, SAME_SCENE = 4, 10


def source_of(filename):
    """Group a file name into its source family, e.g. 'foto01234.jpg' -> 'foto'."""
    stem = Path(filename).stem
    if re.fullmatch(r"\d+", stem):
        return "numeric"
    return re.match(r"[A-Za-z]+", stem).group(0).rstrip("_")


banner("B9  Perceptual hashing of every image")
records = []
for split in ("train", "valid"):
    for img_path in sorted((DATASET_DIR / split / "images").glob("*")):
        with Image.open(img_path) as im:
            h = imagehash.phash(im.convert("RGB"))
        records.append({"split": split, "file": img_path.name, "source": source_of(img_path.name), "phash": str(h)})
print(f"hashed {len(records)} images")

with open(RESULTS_DIR / "b9_image_hashes.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=records[0].keys())
    w.writeheader()
    w.writerows(records)

# Turn every hash into 64 bits so all distances can be computed at once with numpy.
bits = np.array([[int(b) for b in bin(int(r["phash"], 16))[2:].zfill(64)] for r in records], dtype=np.uint8)
split_arr = np.array([r["split"] for r in records])
tr, va = np.where(split_arr == "train")[0], np.where(split_arr == "valid")[0]
dist = (bits[tr, None, :] != bits[None, va, :]).sum(-1)  # train x val Hamming distances

pairs = [(int(tr[i]), int(va[j]), int(dist[i, j])) for i, j in zip(*np.where(dist <= SAME_SCENE))]
pairs.sort(key=lambda p: p[2])
with open(RESULTS_DIR / "b9_cross_split_pairs.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["train_file", "val_file", "hamming_distance"])
    for a, b, d in pairs:
        w.writerow([records[a]["file"], records[b]["file"], d])

nearest_train = dist.min(axis=0)  # for each val image: distance to its closest train image
summary = {
    "n_train": int(len(tr)), "n_val": int(len(va)),
    "thresholds": {"near_duplicate": NEAR_DUP, "same_scene": SAME_SCENE},
    "cross_split_pairs_near_duplicate": int((dist <= NEAR_DUP).sum()),
    "cross_split_pairs_same_scene": int((dist <= SAME_SCENE).sum()),
    "val_images_with_near_duplicate_in_train": int((nearest_train <= NEAR_DUP).sum()),
    "val_images_with_same_scene_in_train": int((nearest_train <= SAME_SCENE).sum()),
    "val_nearest_train_distance_histogram": {str(k): int(v) for k, v in sorted(Counter(nearest_train.tolist()).items())},
}

# Sources per split (file-name families).
fam = Counter((r["split"], r["source"]) for r in records)
summary["images_per_source_and_split"] = {f"{s}/{src}": n for (s, src), n in sorted(fam.items())}
summary["sources_in_both_splits"] = sorted({r["source"] for r in records if r["split"] == "train"}
                                           & {r["source"] for r in records if r["split"] == "valid"})

# Within-split near duplicates (redundant frames inflate both splits).
for split, idx in (("train", tr), ("valid", va)):
    d = (bits[idx, None, :] != bits[None, idx, :]).sum(-1)
    iu = np.triu_indices(len(idx), 1)
    summary[f"within_{split}_pairs_near_duplicate"] = int((d[iu] <= NEAR_DUP).sum())
    summary[f"within_{split}_pairs_same_scene"] = int((d[iu] <= SAME_SCENE).sum())

for k, v in summary.items():
    print(f"{k}: {v}")

# ------------- visual check: the 8 closest train/val pairs -------------
tile = 220
sheet = Image.new("RGB", (2 * tile + 10, 8 * (tile // 2 + 22)), "white")
draw = ImageDraw.Draw(sheet)
for row, (a, b, d) in enumerate(pairs[:8]):
    y = row * (tile // 2 + 22)
    for col, k in enumerate((a, b)):
        r = records[k]
        im = Image.open(DATASET_DIR / r["split"] / "images" / r["file"]).convert("RGB")
        im.thumbnail((tile, tile // 2))
        sheet.paste(im, (col * (tile + 10), y + 18))
        draw.text((col * (tile + 10), y + 2), f"{r['split']}/{r['file']}  d={d}", fill="black")
sheet.save(FIGURES_DIR / "b9_duplicate_examples.jpg", quality=90)
summary["example_sheet"] = "revision/figures/b9_duplicate_examples.jpg"
save_json("b9_near_duplicates", summary)
