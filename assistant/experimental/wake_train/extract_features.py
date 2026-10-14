#!/usr/bin/env python3
"""Build (16, 96) embedding windows for training (TRAIN split only) and the
held-out generic-negative stream for eval.

Positives (real, TRAIN clips + golden): each clip is expanded into 1 clean +
N_AUG augmented streams (speed 0.9-1.1, gain +-6 dB, music/noise mixed at
0..+10 dB SNR using TRAIN-side recordings only). In every stream the word
end is located with an energy heuristic and the positive windows are the
ones whose last embedding frame lies 1..8 frames (80-640 ms) after it.
Windows ending >= 26 frames after the word end (command speech) become
negatives, as do windows ending before speech onset.

Other training negatives: train AEC/music recordings (x gain/speed
variants), wake false-accept / hotkey / retry / follow-up clips (+ augment),
Piper adversarial words and generic phrases, 63% of the official
openWakeWord validation_set_features (blocks of 60 s).  The remaining
generic blocks (30% of blocks) are saved as a stream for eval only.

Run: A:\\pisha\\memoir\\.venv-wake\\Scripts\\python.exe extract_features.py
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

try:
    from .active_learning import require_supported_training
except ImportError:
    from active_learning import require_supported_training

# Fail before importing inference dependencies or allocating/writing features.
if __name__ == "__main__":
    try:
        require_supported_training("feature extraction")
    except ValueError as exc:
        raise SystemExit(str(exc))

import numpy as np

from audio_io import read_wav_mono16k, SR
from generic_negatives import load_generic_windows  # noqa: F401  (kept for reference)
import wakelib as W

ROOT = Path(__file__).parent
SYNTH = Path(r"A:\wake-data\synthetic")
NPY_GENERIC = Path(r"A:\wake-data\negatives\validation_set_features.npy")
OUT = Path(r"A:\wake-data\features")
AEC = Path(r"A:\pisha\memoir\assistant\logs\aec")

SEED = 1337
N_AUG_REAL = 12
N_AUG_SYNTH = 3
GEN_BLOCK = 750            # rows (80 ms each) = 60 s
GEN_HOLDOUT_PCT = 30
GEN_STRIDE = 4
K_POS = range(1, 9)        # window end = word-end frame + k
TAIL_MIN = 26              # frames after word end before command speech is used as negative

KINDS = {"real_pos": 0, "synth_pos": 1, "generic_neg": 2, "local_neg": 3,
         "synth_neg": 4, "tail_neg": 5, "pad_neg": 6, "music_neg": 7, "startup_neg": 8}


def hpct(key: str) -> int:
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % 100


class Acc:
    def __init__(self):
        self.X, self.y, self.w, self.k, self.val, self.g = [], [], [], [], [], []

    def add(self, Xw, label, weight, kind, is_val, group):
        if len(Xw) == 0:
            return
        n = len(Xw)
        self.X.append(Xw.astype(np.float16))
        self.y.append(np.full(n, label, np.float32))
        self.w.append(np.full(n, weight, np.float32))
        self.k.append(np.full(n, KINDS[kind], np.int8))
        self.val.append(np.full(n, is_val, bool))
        self.g.append(np.full(n, group, np.int32))

    def counts(self):
        k = np.concatenate(self.k)
        y = np.concatenate(self.y)
        v = np.concatenate(self.val)
        return {name: int((k == i).sum()) for name, i in KINDS.items()}, int((y == 1).sum()), int(v.sum())


def make_noise_pool():
    pool = []
    for sub, names in (("live-223244", ["raw.wav", "clean.wav", "ref.wav"]),):
        for n in names:
            p = AEC / sub / n
            if p.exists():
                try:
                    pool.append(read_wav_mono16k(str(p)))
                except Exception as e:  # noqa
                    print("noise pool skip", p, e)
    return pool


def augment_stream(y, ref_rms, pool, rng, p_noise=0.8):
    """gain + optional music (70%) / colored noise (30%) at 0..10 dB SNR."""
    g = 10 ** (rng.uniform(-6, 6) / 20)
    y = (y * g).astype(np.float32)
    ref = ref_rms * g
    if rng.random() < p_noise:
        snr = rng.uniform(0, 10)
        if pool and rng.random() < 0.7:
            nz = W.noise_segment(pool, len(y), rng)
            if rng.random() < 0.5:  # time-reverse / speed variety
                nz = nz[::-1]
        else:
            nz = W.colored_noise(len(y), rng)
        y = W.mix_snr(y, nz, snr, ref)
    return y


def pos_stream(x, rng, pool, f, aug, synth=False, override=None):
    x = W.speed_perturb(x, f)
    if override is not None:
        on, we = min(int(override[0] / f), len(x) - 1), min(int(override[1] / f), len(x))
    elif synth:
        on, we = 0, max(1, len(x) - int(0.08 * SR))
    else:
        on, we = W.estimate_word(x)
    keep = min(len(x), we + int(2.6 * SR))
    body = x[:keep]
    lead = int(rng.uniform(2.4, 3.2) * SR) + int(rng.integers(0, W.HOP))
    tail_n = max(int(1.0 * SR), int(1.6 * SR) - (len(body) - we))
    lead_pad = W.noise_floor_pad(lead, x, rng)
    tail_pad = W.noise_floor_pad(tail_n, x, rng)
    y = np.concatenate([lead_pad, body, tail_pad])
    ref_rms = W.rms(x[on:max(we, on + 1600)])
    if aug:
        y = augment_stream(y, ref_rms, pool, rng)
    return y, lead + on, lead + we


def align_enroll(paths, pool):
    """Bootstrapped word-end estimate for the (cued, often music-laden) enrollment clips:
    stream each clean clip through the v1 ensemble; the peak chunk time minus 4.5 frames ~ word end
    (windows used for training are te+1..te+8). None when the v1 model does not fire (>0.3)."""
    from eval import load_model, clip_scores
    m, key = load_model("A:/pisha/memoir/assistant/models/wake/alexa_user_v1.onnx")
    rng_a = np.random.default_rng(3)
    out = {}
    for p in paths:
        x = read_wav_mono16k(p)
        lead = int(2.0 * SR)
        y = np.concatenate([W.room_pad(lead, rng_a), x, W.room_pad(SR, rng_a)])
        s = clip_scores(m, key, y)
        c = int(np.argmax(s))
        if s[c] < 0.3:
            out[p] = None
            continue
        t_peak = (c + 1) * 1280 - SR - lead  # audio sample (in clip) at the end of the peak window
        we = int(t_peak - 4.5 * 1280)
        we = max(int(0.3 * SR), min(we, len(x)))
        out[p] = (max(0, we - int(0.6 * SR)), we)
    return out


def neg_windows_from_audio(af, x, rng, stride):
    if len(x) < int(3.0 * SR):
        x = np.concatenate([W.noise_floor_pad(int(3.0 * SR) - len(x), x, rng), x])
    E = W.embed(af, x)
    return W.windows(E, range(W.N_FRAMES - 1, len(E), stride))


def main():
    require_supported_training("feature extraction")
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    af = W.get_af()
    sm = json.loads((ROOT / "split_manifest.json").read_text(encoding="utf-8"))
    pool = make_noise_pool()
    print("noise pool recordings:", len(pool))
    bank_src = [read_wav_mono16k(p["path"]) for p in sm["positives"] if p["split"] == "train"]
    bank_src += [read_wav_mono16k(n["path"]) for n in sm["negatives_short"]
                 if n["split"] == "train" and n["source"] in ("retry_audio_path", "follow_up_audio_path", "hotkey")]
    W.ROOM_BANK.extend(W.build_room_bank(bank_src))
    print("room-noise bank segments:", len(W.ROOM_BANK),
          "total sec:", round(sum(len(b) for b in W.ROOM_BANK) / SR, 1))
    del bank_src
    af_s = W.get_af()
    acc = Acc()
    gid = 0
    stats = {}

    # ---------------- real positives (TRAIN only) ----------------
    pos_items = [dict(p, weight=1.0, align=None) for p in sm["positives"] if p["split"] == "train"]
    enroll = json.loads((ROOT / "enroll_split.json").read_text(encoding="utf-8"))
    align = align_enroll([p["path"] for p in enroll["positives"] if p["split"] == "train"], pool)
    for p in enroll["positives"]:
        if p["split"] == "train":
            pos_items.append({"path": p["path"], "transcript": "alexa", "weight": 1.5, "align": align.get(p["path"]),
                              "source": "enroll_" + p["label"]})
    stats["enroll_train_positives"] = sum(1 for p in enroll["positives"] if p["split"] == "train")
    stats["enroll_aligned_by_model"] = sum(1 for v in align.values() if v is not None)
    order = rng.permutation(len(pos_items))
    n_val_pos = max(2, int(round(0.15 * len(pos_items))))
    val_set = set(order[:n_val_pos].tolist())
    print(f"train positives: {len(pos_items)} (inner-val clips: {n_val_pos})")
    n_streams = 0
    for i, p in enumerate(pos_items):
        x = read_wav_mono16k(p["path"])
        is_val = i in val_set
        single_alexa = p["transcript"].lower().count("alexa") <= 1
        gid += 1
        for a in range(N_AUG_REAL + 1):
            aug = a > 0
            f = float(rng.choice([0.9, 0.95, 1.0, 1.05, 1.1])) if aug else 1.0
            y, on_s, we_s = pos_stream(x, rng, pool, f, aug, override=p.get("align"))
            E = W.embed(af, y)
            te = W.frame_of_sample(we_s)
            acc.add(W.windows(E, [te + k for k in K_POS]), 1, p["weight"], "real_pos", is_val, gid)
            if single_alexa:
                acc.add(W.windows(E, range(te + TAIL_MIN, len(E), 2)), 0, 2.0, "tail_neg", is_val, gid)
            t_on = W.frame_of_sample(on_s) - 2
            if t_on >= W.N_FRAMES and p.get("align") is None:
                acc.add(W.windows(E, range(W.N_FRAMES - 1, t_on, 6)), 0, 1.0, "pad_neg", is_val, gid)
            n_streams += 1
    stats["real_pos_streams"] = n_streams
    print("real positives done", time.time() - t0)

    # ---------------- synthetic positives ----------------
    sp = sorted((SYNTH / "positives").glob("*.wav"))
    ns = 0
    for p in sp:
        x = read_wav_mono16k(str(p))
        gid += 1
        is_val = hpct("synthpos" + p.stem) < 15
        for a in range(N_AUG_SYNTH + 1):
            f = float(rng.choice([0.9, 0.95, 1.0, 1.05, 1.1, 1.15])) if a else 1.0
            y, on_s, we_s = pos_stream(x, rng, pool, f, aug=a > 0, synth=True)
            E = W.embed(af, y)
            te = W.frame_of_sample(we_s)
            acc.add(W.windows(E, [te + k for k in K_POS]), 1, 0.5, "synth_pos", is_val, gid)
            ns += 1
    stats["synth_pos_streams"] = ns
    print("synthetic positives done", ns, time.time() - t0)

    # ---------------- synthetic negatives (train list) ----------------
    sn = sorted((SYNTH / "negatives_train").glob("*.wav"))
    for p in sn:
        x = read_wav_mono16k(str(p))
        gid += 1
        is_val = hpct("synthneg" + p.stem) < 15
        for a in range(3):
            xx = W.speed_perturb(x, float(rng.choice([0.92, 1.0, 1.08]))) if a else x
            lead = W.noise_floor_pad(int(rng.uniform(1.5, 2.5) * SR), xx, rng)
            tail = W.noise_floor_pad(int(1.2 * SR), xx, rng)
            y = np.concatenate([lead, xx, tail])
            if a == 2:
                y = augment_stream(y, W.rms(xx), pool, rng, p_noise=1.0)
            E = W.embed(af, y)
            acc.add(W.windows(E, range(W.N_FRAMES - 1, len(E), 2)), 0, 1.0, "synth_neg", is_val, gid)
    stats["synth_neg_clips"] = len(sn)
    print("synthetic negatives done", len(sn), time.time() - t0)

    # ---------------- local negatives: short clips (train split) ----------------
    shorts = [n for n in sm["negatives_short"] if n["split"] == "train"]
    for n in shorts:
        x = read_wav_mono16k(n["path"])
        gid += 1
        is_val = hpct("short" + n["path"]) < 15
        for a in range(8):
            xx = x
            if a:
                xx = W.speed_perturb(x, float(rng.choice([0.92, 1.0, 1.08])))
                xx = augment_stream(xx, W.rms(xx), pool, rng, p_noise=0.6)
            acc.add(neg_windows_from_audio(af, xx, rng, 2), 0, 3.0, "local_neg", is_val, gid)
    stats["local_short_neg_clips"] = len(shorts)
    print("local short negatives done", len(shorts), time.time() - t0)

    # ---------------- enrollment negatives (near-words, free talk), TRAIN parts ----------------
    ent = [n for n in enroll["neg_near"] if n["split"] == "train"]
    for i, n in enumerate(ent):
        x = read_wav_mono16k(n["path"])
        gid += 1
        for a in range(12):
            xx = x
            if a:
                xx = W.speed_perturb(x, float(rng.choice([0.92, 0.96, 1.0, 1.04, 1.08])))
                xx = augment_stream(xx, W.rms(xx), pool, rng, p_noise=0.6)
            acc.add(neg_windows_from_audio(af, xx, rng, 1), 0, 4.0, "local_neg", (i == 0 and a == 0) or False, gid)
    ft = read_wav_mono16k("A:/wake-data/enroll/20261002-224305/neg_free_talk.wav")
    a0, a1 = enroll["free_talk"]["train"]
    ft = ft[int(a0 * SR):int(a1 * SR)]
    segs = [ft[j:j + 6 * SR] for j in range(0, len(ft) - 3 * SR, 6 * SR)]
    for si, sg in enumerate(segs):
        gid += 1
        is_val = si % 5 == 4
        for a in range(1 if is_val else 8):
            xx = sg
            if a:
                xx = W.speed_perturb(sg, float(rng.choice([0.92, 0.96, 1.0, 1.04, 1.08])))
                xx = augment_stream(xx, W.rms(xx), pool, rng, p_noise=0.6)
            acc.add(neg_windows_from_audio(af, xx, rng, 2), 0, 4.0, "local_neg", is_val, gid)
    stats["enroll_neg_near_train"] = len(ent); stats["enroll_free_talk_train_s"] = float(len(ft) / SR)
    print("enrollment negatives done", time.time() - t0)

    # ---------------- local negatives: long AEC/music recordings (train split) ----------------
    longs = [n for n in sm["negatives_long"] if n["split"] == "train"]
    long_paths = [Path(n["path"]) for n in longs]
    extra = []
    for lp in long_paths:  # same-session processed variants (train side only)
        for nm in ("clean.wav", "ref.wav"):
            if (lp.parent / nm).exists():
                extra.append(lp.parent / nm)
    n_long = 0
    for lp in long_paths + extra:
        x = read_wav_mono16k(str(lp))
        for a in range(6):
            xx = x
            if a:
                xx = W.speed_perturb(x, float(rng.choice([0.9, 0.95, 1.0, 1.05, 1.1])))
                xx = (xx * 10 ** (rng.uniform(-8, 6) / 20)).astype(np.float32)
                xx = np.clip(xx, -0.99, 0.99)
            gid += 1
            # time-split each long recording: last 20% of windows to inner-val
            Wn = neg_windows_from_audio(af, xx, rng, 2)
            cut = int(len(Wn) * 0.8)
            acc.add(Wn[:cut], 0, 1.0, "music_neg", False, gid)
            acc.add(Wn[cut:], 0, 1.0, "music_neg", True, gid)
            n_long += 1
    stats["long_neg_variants"] = n_long
    print("long/music negatives done", n_long, time.time() - t0)

    # ---------------- extra music/room negatives: random excerpts of the TRAIN-side pool ----------------
    n_ex = 0
    for j in range(400):
        L = int(rng.uniform(4.0, 7.0) * SR)
        seg = W.noise_segment(pool, L, rng).copy()
        if rng.random() < 0.5:
            seg = seg[::-1].copy()
        seg = W.speed_perturb(seg, float(rng.uniform(0.85, 1.15)))
        seg = np.clip(seg * 10 ** (rng.uniform(-14, 6) / 20), -0.99, 0.99).astype(np.float32)
        if rng.random() < 0.3:
            seg = (seg + W.colored_noise(len(seg), rng) * rng.uniform(0.002, 0.02)).astype(np.float32)
        gid += 1
        E = W.embed(af, seg)
        acc.add(W.windows(E, range(W.N_FRAMES - 1, len(E), 1)), 0, 1.0, "music_neg",
                hpct(f"musicx{j}") < 15, gid)
        n_ex += 1
    stats["extra_music_excerpts"] = n_ex
    print("extra music negatives done", n_ex, time.time() - t0)

    # ---------------- startup-transient negatives (fresh streaming front-end) ----------------
    speech_pool = [read_wav_mono16k(n["path"]) for n in shorts if n["source"] != "wake_false_accept"]
    syn_pool = [read_wav_mono16k(str(p)) for p in sorted((SYNTH / "negatives_train").glob("*.wav"))]
    n_su = 0
    for j in range(1000):
        src = j % 5
        L = int(2.2 * SR)
        if src == 0:
            seg = np.zeros(L, np.float32)
        elif src == 1:
            seg = W.room_pad(L, rng)
        elif src == 2:
            seg = W.noise_segment(pool, L, rng).copy() * np.float32(10 ** (rng.uniform(-14, 4) / 20))
        elif src == 3:
            sp = speech_pool[rng.integers(0, len(speech_pool))]
            seg = np.concatenate([W.room_pad(int(rng.uniform(0, 1.0) * SR), rng), sp])[:L]
        else:
            sp = syn_pool[rng.integers(0, len(syn_pool))]
            seg = np.concatenate([W.room_pad(int(rng.uniform(0, 1.0) * SR), rng), sp])[:L]
        if len(seg) < L:
            seg = np.concatenate([seg, W.room_pad(L - len(seg), rng)])
        if src >= 3 and rng.random() < 0.5:
            seg = augment_stream(seg.astype(np.float32), W.rms(seg) + 1e-4, pool, rng, p_noise=1.0)
        gid += 1
        acc.add(W.startup_windows(af_s, seg), 0, 1.0, "startup_neg", hpct(f"startup{j}") < 15, gid)
        n_su += 1
    stats["startup_streams"] = n_su
    print("startup negatives done", n_su, time.time() - t0)

    # ---------------- generic negative features (blocks) ----------------
    G = np.load(NPY_GENERIC, mmap_mode="r")
    n_blocks = len(G) // GEN_BLOCK
    hold_blocks, gtrain, gval = [], 0, 0
    for b in range(n_blocks):
        blk = np.asarray(G[b * GEN_BLOCK:(b + 1) * GEN_BLOCK], dtype=np.float32)
        if hpct(f"gen{b}") < GEN_HOLDOUT_PCT:
            hold_blocks.append(blk)
            continue
        is_val = hpct(f"genval{b}") < 25
        Wn = W.windows(blk, range(W.N_FRAMES - 1, len(blk), GEN_STRIDE))
        acc.add(Wn, 0, 1.0, "generic_neg", is_val, 100000 + b)
        gtrain += len(Wn)
    hold = np.stack(hold_blocks)  # (n_hold_blocks, GEN_BLOCK, 96)
    np.save(OUT / "generic_holdout_blocks.npy", hold)
    stats["generic_train_windows"] = gtrain
    stats["generic_holdout_blocks"] = int(len(hold_blocks))
    stats["generic_holdout_hours"] = float(len(hold_blocks) * GEN_BLOCK * 0.08 / 3600)
    print("generic done", time.time() - t0)

    X = np.concatenate(acc.X)
    np.savez(OUT / "train_data.npz", X=X, y=np.concatenate(acc.y), w=np.concatenate(acc.w),
             kind=np.concatenate(acc.k), val=np.concatenate(acc.val), group=np.concatenate(acc.g))
    kc, npos, nval = acc.counts()
    stats.update({"X_shape": list(X.shape), "windows_by_kind": kc, "n_positive_windows": npos,
                  "n_inner_val": nval, "extract_time_s": time.time() - t0})
    (OUT / "extract_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
