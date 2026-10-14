"""Read-only probe: score the enrollment session with the CURRENT models (no training)."""
import json, sys, wave, collections
import numpy as np
from audio_io import read_wav_mono16k, SR
import wakelib as W
from eval import load_model, clip_scores, run_starts
D = r"A:\wake-data\enroll\20261002-224305"
man = json.load(open(D + r"\manifest.json"))
print(len(man), collections.Counter((e["label"], e.get("music")) for e in man))
ft = read_wav_mono16k(D + r"\neg_free_talk.wav"); print("free talk sec", len(ft) / SR)
W.ROOM_BANK.extend(W.build_room_bank([ft])); print("bank", len(W.ROOM_BANK))
rng = np.random.default_rng(1)
models = {"stock": r"A:\wake-data\models\alexa_v0.1.onnx", "ensemble5": r"A:\pisha\memoir\assistant\models\wake\alexa_user.onnx"}
pos = [e for e in man if e["label"].startswith("pos")]
neg = [e for e in man if e["label"].startswith("neg") and e["file"] != "neg_free_talk.wav"]
res = {}
for mn, mp in models.items():
    m, key = load_model(mp)
    by = collections.defaultdict(list)
    for e in pos:
        x = read_wav_mono16k(D + "\\" + e["file"])
        y = np.concatenate([W.room_pad(2 * SR, rng) if W.ROOM_BANK else np.zeros(2 * SR, np.float32), x, np.zeros(0, np.float32)])
        by[(e["label"], bool(e.get("music")))].append(clip_scores(m, key, y).max())
    nm = [clip_scores(m, key, np.concatenate([W.room_pad(2 * SR, rng), read_wav_mono16k(D + "\\" + e["file"])])).max() for e in neg]
    s = clip_scores(m, key, np.concatenate([W.room_pad(2 * SR, rng), ft]))
    print("==", mn)
    allp = [v for l in by.values() for v in l]
    print(" ALL positives n=%d det@.15/.3/.5/.7:" % len(allp), [round(float(np.mean(np.array(allp) >= t)), 2) for t in (.15, .3, .5, .7)])
    for k, l in sorted(by.items()):
        print("  ", k, "n=%d" % len(l), [round(float(np.mean(np.array(l) >= t)), 2) for t in (.15, .3, .5, .7)])
    print(" near-word negatives n=%d false-trigger@.15/.3/.5/.7:" % len(nm), [round(float(np.mean(np.array(nm) >= t)), 2) for t in (.15, .3, .5, .7)])
    print(" free talk %.0fs activations@.15/.3/.5/.7:" % (len(ft) / SR), [len(run_starts(s, t, 1)) for t in (.15, .3, .5, .7)])
