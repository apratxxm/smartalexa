#!/usr/bin/env python3
"""Fixed 70/30 split of the user's enrollment session (stratified by label).
Positives: per label, round(0.3*n) clips held out (seeded). neg_near: 2 of 8 held out.
neg_free_talk.wav (60 s): 0-42 s train, 44-60 s held out (2 s gap)."""
import json, random
from pathlib import Path
D = Path(r"A:\wake-data\enroll\20261002-224305")
man = json.loads((D / "manifest.json").read_text(encoding="utf-8"))
rng = random.Random(1337)
out = {"positives": [], "neg_near": [], "free_talk": {"train": [0.0, 42.0], "holdout": [44.0, 60.0]}}
by = {}
for e in man:
    by.setdefault(e["label"], []).append(e)
for lab, es in sorted(by.items()):
    if lab == "neg_free":
        continue
    es = sorted(es, key=lambda e: e["file"]); rng.shuffle(es)
    n_hold = max(1, round(0.3 * len(es))) if lab.startswith("pos") else 2
    for i, e in enumerate(es):
        item = {"path": str(D / e["file"]), "label": lab, "music": bool(e.get("music")), "split": "holdout" if i < n_hold else "train"}
        out["positives" if lab.startswith("pos") else "neg_near"].append(item)
(Path(__file__).parent / "enroll_split.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
from collections import Counter
print("pos", Counter(i["split"] for i in out["positives"]), "near", Counter(i["split"] for i in out["neg_near"]))
print(Counter((i["label"], i["split"]) for i in out["positives"]))
