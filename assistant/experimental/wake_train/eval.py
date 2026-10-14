#!/usr/bin/env python3
"""Held-out evaluation: custom alexa_user.onnx vs stock openWakeWord alexa_v0.1.

Everything is scored with the production pipeline (openwakeword.model.Model,
onnx, streaming 80 ms chunks) except the generic-negative set, which only
exists as precomputed embeddings and is scored directly through each
classifier ONNX (identical to what Model does after its front-end).

Only HOLDOUT items are used (split_manifest.json, fixed): held-out stt
"Alexa" clips, the held-out AEC/music recording, held-out false-accept /
retry clips, command-speech tails of held-out positives, held-out Piper
near-miss words, and 30% of the generic negative feature blocks.

Positive clips are evaluated as: [2 s real room-noise lead] + first 2.5 s of
the clip + [1 s room-noise tail]; the lead noise comes from quiet segments
of HELD-OUT clips only.  A harsher "digital-zero lead" variant and music-
mixed variants (held-out music recording at +10/+5/0 dB SNR) are also run.

Negative audio is scored as ONE continuous stream per group (a single
Model.reset), so start-up transients are not counted per file.
Activation counting: above-threshold chunks within a 2 s refractory window
count as ONE activation.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from audio_io import read_wav_mono16k, SR
import wakelib as W

ROOT = Path(__file__).parent
MODELS = Path(r"A:\wake-data\models")
CUSTOM = Path(r"A:\pisha\memoir\assistant\models\wake\alexa_user.onnx")
STOCK = MODELS / "alexa_v0.1.onnx"
FEATURES = Path(r"A:\wake-data\features")
SYNTH = Path(r"A:\wake-data\synthetic")
WAKE_DIR = Path(r"A:\pisha\memoir\assistant\models\wake")
OUT = Path(os.environ.get("EVAL_OUT", str(WAKE_DIR / "alexa_user_results.json")))
import os
THR = tuple(float(v) for v in os.environ.get('EVAL_THR', '0.15,0.3,0.5,0.7').split(','))
PATS = (1, 2, 3)
REFRACT = 25  # chunks (2 s)
SEED = 2026


def load_model(path):
    from openwakeword.model import Model
    m = Model(wakeword_models=[str(path)], inference_framework="onnx",
              melspec_model_path=str(MODELS / "melspectrogram.onnx"),
              embedding_model_path=str(MODELS / "embedding_model.onnx"))
    return m, list(m.models.keys())[0]


def clip_scores(m, key, x_float):
    m.reset()
    preds = m.predict_clip(W.to_i16(x_float))  # adds 1 s digital-zero padding each end
    return np.array([p[key] for p in preds], dtype=np.float32)


def run_starts(s, thr, p):
    """start indices of runs of >= p consecutive chunks with score >= thr."""
    above = s >= thr
    starts, i, n = [], 0, len(s)
    while i < n:
        if above[i]:
            j = i
            while j < n and above[j]:
                j += 1
            if j - i >= p:
                starts.append(i)
            i = j
        else:
            i += 1
    return starts


def n_activations(s, thr, p=1):
    n, last = 0, -10 ** 9
    for i in run_starts(s, thr, p):
        if i - last > REFRACT:
            n += 1
        last = i
    return n


def detected(s, thr, p=1):
    return len(run_starts(s, thr, p)) > 0


def holdout_audio():
    sm = json.loads((ROOT / "split_manifest.json").read_text(encoding="utf-8"))
    pos = [p for p in sm["positives"] if p["split"] == "holdout"]
    pos_clips, tails, raw_all = [], [], []
    for p in pos:
        x = read_wav_mono16k(p["path"])
        raw_all.append(x)
        pos_clips.append(x[: int(2.5 * SR)])
        if p["transcript"].lower().count("alexa") <= 1:
            on, we = W.estimate_word(x)
            t = x[we + int(1.5 * SR):]
            if len(t) > int(1.0 * SR):
                tails.append(t)
    neg = {"aec_music_room": [], "wake_false_accept": [], "retry_speech": [], "command_speech_tails": tails}
    for n in sm["negatives_long"]:
        if n["split"] == "holdout":
            neg["aec_music_room"].append(read_wav_mono16k(n["path"]))
    for n in sm["negatives_short"]:
        if n["split"] != "holdout":
            continue
        x = read_wav_mono16k(n["path"])
        raw_all.append(x)
        neg["wake_false_accept" if n["source"] == "wake_false_accept" else "retry_speech"].append(x)
    en = json.loads((ROOT / "enroll_split.json").read_text(encoding="utf-8"))
    enroll_pos = [(p["label"], p["music"], read_wav_mono16k(p["path"])) for p in en["positives"] if p["split"] == "holdout"]
    neg["enroll_near_words"] = [read_wav_mono16k(p["path"]) for p in en["neg_near"] if p["split"] == "holdout"]
    ft = read_wav_mono16k("A:/wake-data/enroll/20261002-224305/neg_free_talk.wav")
    a0, a1 = en["free_talk"]["holdout"]
    neg["enroll_free_talk"] = [ft[int(a0 * SR):int(a1 * SR)]]
    raw_all += [e[2] for e in enroll_pos if not e[1]] + neg["enroll_free_talk"]
    return pos, pos_clips, neg, raw_all, enroll_pos


def stream_of(files, rng, lead=1.5, gap=0.5):
    parts = [W.room_pad(int(lead * SR), rng)]
    spans = []
    pos = len(parts[0])
    for x in files:
        parts.append(x); spans.append((pos, pos + len(x))); pos += len(x)
        g = W.room_pad(int(gap * SR), rng); parts.append(g); pos += len(g)
    return np.concatenate(parts).astype(np.float32), spans


def main():
    rng = np.random.default_rng(SEED)
    pos_meta, pos_clips, neg, raw_all, enroll_pos = holdout_audio()
    # room-noise bank from HELD-OUT clips only
    W.ROOM_BANK.clear()
    W.ROOM_BANK.extend(W.build_room_bank(raw_all))
    print("holdout room bank segments:", len(W.ROOM_BANK))
    music = neg["aec_music_room"][0]
    adv = [read_wav_mono16k(str(p)) for p in sorted((SYNTH / "negatives_holdout").glob("*.wav"))]
    print("holdout positives:", len(pos_clips), {k: len(v) for k, v in neg.items()}, "adversarial:", len(adv))

    # ---- positive test streams (identical for both models) ----
    def room_wrap(x):
        return np.concatenate([W.room_pad(2 * SR, rng), x, W.room_pad(SR, rng)]).astype(np.float32)
    pos_sets = {"room_lead": [room_wrap(x) for x in pos_clips],
                "zero_lead": [np.concatenate([np.zeros(2 * SR, np.float32), x, np.zeros(SR, np.float32)]) for x in pos_clips]}
    for snr in (10, 5, 0):
        lst = []
        for x in pos_clips:
            on, we = W.estimate_word(x)
            ref = W.rms(x[on:max(we, on + 1600)])
            y = room_wrap(x)
            lst.append(W.mix_snr(y, W.noise_segment([music], len(y), rng), snr, ref))
        pos_sets[f"room_lead_music_snr_{snr}dB"] = lst

    pos_sets["enroll_holdout_all"] = [room_wrap(e[2]) for e in enroll_pos]
    pos_sets["enroll_holdout_quiet"] = [room_wrap(e[2]) for e in enroll_pos if not e[1]]
    pos_sets["enroll_holdout_music"] = [room_wrap(e[2]) for e in enroll_pos if e[1]]
    neg_streams = {g: stream_of(v, rng) for g, v in neg.items()}
    # (c) user's 40 s music recording (spans + JSON labels), never trained on
    import wave as _wave
    rec_dir = "A:/wake-data/user_recordings/"
    rec_meta = json.loads(open(rec_dir + "alexas_with_music_20261002.json").read())
    with _wave.open(rec_dir + "alexas_with_music_20261002.wav") as _w:
        rec_x = np.frombuffer(_w.readframes(_w.getnframes()), np.int16).astype(np.float32) / 32768
    rec_lead = rec_x[-2 * SR:]
    rec_stream = np.concatenate([rec_lead, rec_x]).astype(np.float32)
    rec_off = len(rec_lead)
    adv_stream, adv_spans = stream_of(adv, rng, lead=1.5, gap=1.0)

    results = {"data": {
        "holdout_positive_clips": len(pos_clips),
        "holdout_negative_files": {k: len(v) for k, v in neg.items()},
        "holdout_negative_hours": {k: float(sum(len(x) for x in v) / SR / 3600) for k, v in neg.items()},
        "adversarial_holdout_clips": len(adv),
        "thresholds": list(THR),
        "refractory_s": 2.0,
    }}
    neg_total_h = sum(results["data"]["holdout_negative_hours"].values())
    results["data"]["holdout_negative_hours_total_local"] = neg_total_h
    G = np.load(FEATURES / "generic_holdout_blocks.npy")
    gen_hours = G.shape[0] * G.shape[1] * 0.08 / 3600
    results["data"]["generic_holdout_hours"] = gen_hours

    for name, path in ((("custom", CUSTOM),) if os.environ.get("EVAL_CUSTOM_ONLY") else (("custom", CUSTOM), ("stock", STOCK))):
        t0 = time.time()
        m, key = load_model(path)
        r = {}
        pos_scores = {sn: [clip_scores(m, key, y) for y in clips] for sn, clips in pos_sets.items()}
        neg_scores = {g: clip_scores(m, key, stream) for g, (stream, _) in neg_streams.items()}
        adv_scores = clip_scores(m, key, adv_stream)
        rec_scores = clip_scores(m, key, rec_stream)
        t_chunk = ((np.arange(len(rec_scores)) + 1) * 1280 - SR - rec_off) / SR
        r["max_score_per_clip_room_lead"] = [round(float(s.max()), 3) for s in pos_scores["room_lead"]]
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        iname = sess.get_inputs()[0].name
        fixed1 = sess.get_inputs()[0].shape[0] == 1
        gen_scores = []
        for blk in G:
            Wn = W.windows(blk, range(W.N_FRAMES - 1, len(blk)))
            if fixed1:
                sc = np.array([sess.run(None, {iname: Wn[i:i + 1]})[0].reshape(-1)[0] for i in range(len(Wn))], np.float32)
            else:
                sc = np.concatenate([sess.run(None, {iname: Wn[i:i + 4096]})[0].reshape(-1)
                                     for i in range(0, len(Wn), 4096)])
            gen_scores.append(sc)
        for pat in PATS:  # patience = consecutive 80 ms chunks required above threshold
            R = {}
            for sn, sl in pos_scores.items():
                R[f"detection_{sn}"] = {str(t): float(np.mean([detected(s, t, pat) for s in sl])) for t in THR}
            fa = {}
            for grp, s in neg_scores.items():
                fa[grp] = {"hours": results["data"]["holdout_negative_hours"][grp],
                           "activations": {str(t): int(n_activations(s, t, pat)) for t in THR}}
            tot = {str(t): sum(fa[g]["activations"][str(t)] for g in fa) for t in THR}
            fa["ALL_LOCAL"] = {"hours": neg_total_h, "activations": tot,
                               "per_hour": {k: v / neg_total_h for k, v in tot.items()}}
            R["local_negatives"] = fa
            amx = []
            for (a_, b_) in adv_spans:
                c0 = max(0, int((a_ + SR) / 1280)); c1 = int((b_ + SR + 0.64 * SR) / 1280) + 1
                amx.append({t: detected(adv_scores[c0:c1], t, pat) for t in THR})
            R["adversarial_word_false_trigger_rate"] = {str(t): float(np.mean([a[t] for a in amx])) for t in THR}
            cnt = {str(t): int(sum(n_activations(sc, t, pat) for sc in gen_scores)) for t in THR}
            R["generic_negatives"] = {"hours": gen_hours, "activations": cnt,
                                      "per_hour": {k: v / gen_hours for k, v in cnt.items()}}
            spans = rec_meta["alexa_spans_s"]
            Rr = {}
            for t in THR:
                det = 0
                for a_, b_ in spans:
                    sel = (t_chunk >= a_) & (t_chunk <= b_ + 1.5)
                    if sel.any() and run_starts(rec_scores[sel], t, pat):
                        det += 1
                fa_, last = 0, -10 ** 9
                for i in run_starts(rec_scores, t, pat):
                    if i - last > REFRACT:
                        tt = t_chunk[i]
                        if tt >= 0 and not any(a_ <= tt <= b_ + 1.5 for a_, b_ in spans):
                            fa_ += 1
                    last = i
                Rr[str(t)] = {"detected_of_19": det, "false_activations": fa_}
            R["music_recording_40s"] = Rr
            r[f"patience_{pat}"] = R
        r["eval_time_s"] = time.time() - t0
        results[name] = r
        print(name, json.dumps({k: v for k, v in r.items() if k != "max_score_per_clip_room_lead"}, indent=1))

    # ---- attach training provenance ----
    prov = {}
    prov["train_meta_members"] = [
        {k: v for k, v in json.loads(p.read_text(encoding="utf-8")).items() if k != "history"}
        for p in sorted(WAKE_DIR.glob("train_meta_member*.json"))]
    p = WAKE_DIR / "ensemble_members.json"
    if p.exists():
        prov["ensemble_members"] = json.loads(p.read_text(encoding="utf-8"))
    ex = FEATURES / "extract_stats.json"
    if ex.exists():
        prov["feature_extraction"] = json.loads(ex.read_text(encoding="utf-8"))
    results["provenance"] = prov
    OUT.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
