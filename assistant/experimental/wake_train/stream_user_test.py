#!/usr/bin/env python3
"""Held-out streaming test on the user's 40 s recording (19 'Alexa' over music).
NEVER used for training. Reports, per model and threshold, spans detected
(score >= thr within [span_start, span_end + 1.5 s]) and false activations
outside every such window."""
import json, sys, wave
import numpy as np
from audio_io import SR
import wakelib as W
from eval import load_model, clip_scores, run_starts, REFRACT

D = r"A:\wake-data\user_recordings"
meta = json.load(open(D + r"\alexas_with_music_20261002.json"))
with wave.open(D + r"\alexas_with_music_20261002.wav") as w:
    x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
spans = meta["alexa_spans_s"]
THR = (0.15, 0.3, 0.5, 0.7)
MODELS = {"stock": r"A:\wake-data\models\alexa_v0.1.onnx",
          "ensemble5 (final)": r"A:\pisha\memoir\assistant\models\wake\alexa_user.onnx"}
for n in ("m1", "m2", "m3", "s1", "s2"):
    MODELS["member_" + n] = rf"A:\wake-data\scratch\{n}.onnx"
variants = {"lead=file tail 2s (music context)": x[-2 * SR:], "lead=digital zeros": np.zeros(2 * SR, np.float32)}
out = {}
for vn, lead in variants.items():
    stream = np.concatenate([lead, x]).astype(np.float32)
    off = len(lead)
    print("\n###", vn)
    for mn, mp in MODELS.items():
        m, key = load_model(mp)
        s = clip_scores(m, key, stream)
        t_chunk = ((np.arange(len(s)) + 1) * 1280 - SR - off) / SR  # audio time (s) of chunk end
        for pat in (1, 3):
            row = []
            for t in THR:
                det = 0
                for a, b in spans:
                    sel = (t_chunk >= a) & (t_chunk <= b + 1.5)
                    if sel.any() and run_starts(s[sel], t, pat):
                        det += 1
                # false activations: runs (refractory 2 s) whose start is outside every span window
                fa, last = 0, -10 ** 9
                for i in run_starts(s, t, pat):
                    if i - last > REFRACT:
                        tt = t_chunk[i]
                        if not any(a <= tt <= b + 1.5 for a, b in spans) and tt >= 0:
                            fa += 1
                    last = i
                row.append((det, fa))
            print(f"{mn:22s} patience={pat}  " + "  ".join(f"thr{t}: {d}/19 det, {f} FA" for t, (d, f) in zip(THR, row)))
            out.setdefault(vn, {}).setdefault(mn, {})[f"patience_{pat}"] = {str(t): {"detected_of_19": d, "false_activations": f} for t, (d, f) in zip(THR, row)}
json.dump(out, open(r"A:\pisha\memoir\assistant\models\wake\alexa_user_stream_test.json", "w"), indent=1)
