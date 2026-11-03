"""
C10 - Build a leakage-free, scene-based 70 / 15 / 15 train / val / test split.

B9 found duplicate pictures shared by the original train and val sets, so we
cannot just split images at random.  Instead:

  1. Put every image (train + valid together, 1,359 images) in its own group.
  2. MERGE groups that must stay together:
       a) images whose perceptual hashes differ by <= 10 bits (same picture / same scene),
       b) frames of the same video clip: same source family (foto / yoto / scene)
          and frame numbers less than 100 apart.
  3. Hand out whole GROUPS (largest first) to whichever split is furthest below
     its 70 / 15 / 15 target, so that no group is ever cut in two.
  4. Check that no train-val, train-test or val-test pair is within 10 bits.

The images are NOT copied.  We only write three text files that list image
paths relative to drone_dataset/ (e.g. "train/images/pic_031.jpg").
`common.make_split_yaml()` turns them into an Ultralytics data YAML.

Outputs:
  revision/splits/scene_split/{train,val,test}.txt
  revision/results/c10_split.json
Run:    python revision/c10_make_split.py        (needs results/b9_image_hashes.csv from b9)
"""
import csv
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATASET_DIR, RESULTS_DIR, REVISION_DIR, banner, save_json  # noqa: E402

SAME_SCENE_BITS = 10
CLIP_GAP = 100
TARGETS = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED = 0
SPLIT_DIR = REVISION_DIR / "splits" / "scene_split"

banner("C10  Scene-based split")
rows = list(csv.DictReader(open(RESULTS_DIR / "b9_image_hashes.csv")))
n = len(rows)

# ---------- union-find: a tiny helper to merge groups -----------------
parent = list(range(n))


def find(i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def union(i, j):
    parent[find(i)] = find(j)


# (a) visually similar images
bits = np.array([[int(b) for b in bin(int(r["phash"], 16))[2:].zfill(64)] for r in rows], dtype=np.uint8)
dist = (bits[:, None, :] != bits[None, :, :]).sum(-1)
for i, j in zip(*np.where(np.triu(dist <= SAME_SCENE_BITS, 1))):
    union(int(i), int(j))

# (b) neighbouring frames of the same video clip
frames = defaultdict(list)
for i, r in enumerate(rows):
    m = re.fullmatch(r"(foto|yoto|scene)(\d+)", Path(r["file"]).stem)
    if m:
        frames[m.group(1)].append((int(m.group(2)), i))
for family, items in frames.items():
    items.sort()
    for (fa, ia), (fb, ib) in zip(items, items[1:]):
        if fb - fa < CLIP_GAP:
            union(ia, ib)

groups = defaultdict(list)
for i in range(n):
    groups[find(i)].append(i)
group_list = sorted(groups.values(), key=len, reverse=True)
print(f"{n} images -> {len(group_list)} groups (largest {len(group_list[0])} images)")

# ---------- greedy assignment of whole groups, per source family ----------
# Doing it per family (pic / numeric / foto / yoto / scene) keeps real video
# scenes in val and test too, instead of only web photos.
random.seed(SEED)
family_of_group = lambda g: Counter(rows[i]["source"] for i in g).most_common(1)[0][0]  # noqa: E731
by_family = defaultdict(list)
for g in group_list:
    by_family[family_of_group(g)].append(g)

assigned = {s: [] for s in TARGETS}
for family, fam_groups in sorted(by_family.items()):
    fam_total = sum(len(g) for g in fam_groups)
    have = {s: 0 for s in TARGETS}
    # largest groups first; shuffle groups of equal size so file order does not decide
    for g in sorted(fam_groups, key=lambda g: (-len(g), random.random())):
        # the split with the largest shortfall relative to its target (within this family)
        target_split = max(TARGETS, key=lambda s: TARGETS[s] * fam_total - have[s])
        assigned[target_split].extend(g)
        have[target_split] += len(g)

# ---------- checks ------------------------------------------------------
split_of = {i: s for s, idx in assigned.items() for i in idx}
leaks = Counter()
for i, j in zip(*np.where(np.triu(dist <= SAME_SCENE_BITS, 1))):
    if split_of[int(i)] != split_of[int(j)]:
        leaks[tuple(sorted((split_of[int(i)], split_of[int(j)])))] += 1


def n_boxes(i):
    r = rows[i]
    lbl = DATASET_DIR / r["split"] / "labels" / (Path(r["file"]).stem + ".txt")
    return len([l for l in lbl.read_text().splitlines() if l.strip()]) if lbl.exists() else 0


SPLIT_DIR.mkdir(parents=True, exist_ok=True)
summary = {"seed": SEED, "targets": TARGETS, "same_scene_bits": SAME_SCENE_BITS, "clip_gap_frames": CLIP_GAP,
           "n_images": n, "n_groups": len(group_list), "largest_group": len(group_list[0]),
           "cross_split_pairs_within_10_bits": {f"{a}-{b}": c for (a, b), c in leaks.items()}, "splits": {}}
for s, idx in assigned.items():
    idx = sorted(idx, key=lambda i: (rows[i]["split"], rows[i]["file"]))
    (SPLIT_DIR / f"{s}.txt").write_text("\n".join(f"{rows[i]['split']}/images/{rows[i]['file']}" for i in idx) + "\n")
    summary["splits"][s] = {
        "images": len(idx), "fraction": len(idx) / n, "boxes": sum(n_boxes(i) for i in idx),
        "sources": dict(Counter(rows[i]["source"] for i in idx)),
        "from_original_split": dict(Counter(rows[i]["split"] for i in idx)),
    }
    print(s, summary["splits"][s])
print("cross-split pairs within 10 bits (must be empty):", summary["cross_split_pairs_within_10_bits"])
summary["files"] = [str((SPLIT_DIR / f"{s}.txt").relative_to(REVISION_DIR)) for s in TARGETS]
save_json("c10_split", summary)
