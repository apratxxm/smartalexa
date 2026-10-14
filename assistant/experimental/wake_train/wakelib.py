"""Shared helpers: embedding extraction, word-end estimation, augmentation.

All embeddings use the exact openWakeWord front-end (melspectrogram.onnx +
embedding_model.onnx via openwakeword.utils.AudioFeatures), fed int16-scaled
audio like the streaming Model does.
"""
from __future__ import annotations

import numpy as np

from audio_io import SR, frame_rms

MODELS_DIR = r"A:\wake-data\models"
HOP = 1280          # samples between embedding frames (80 ms)
RF_END = 12400      # sample index (approx) where embedding frame 0's receptive field ends
N_FRAMES = 16


def get_af():
    from openwakeword.utils import AudioFeatures
    return AudioFeatures(
        melspec_model_path=MODELS_DIR + r"\melspectrogram.onnx",
        embedding_model_path=MODELS_DIR + r"\embedding_model.onnx",
        inference_framework="onnx",
    )


def to_i16(x: np.ndarray) -> np.ndarray:
    return np.clip(np.round(x * 32767.0), -32768, 32767).astype(np.int16)


def embed(af, x: np.ndarray) -> np.ndarray:
    """x float32 [-1,1] -> (T, 96) embeddings, 80 ms stride. Needs >= ~0.8 s."""
    return af._get_embeddings(to_i16(x)).astype(np.float32)


def frame_of_sample(s: int) -> int:
    """Index of the first embedding frame whose receptive field ends at/after sample s."""
    return int(np.ceil((s - RF_END) / HOP))


def windows(E: np.ndarray, ends, ) -> np.ndarray:
    """Stack 16-frame windows whose last frame index is in `ends`."""
    out = [E[t - N_FRAMES + 1: t + 1] for t in ends if t - N_FRAMES + 1 >= 0 and t < len(E)]
    return np.stack(out).astype(np.float32) if out else np.zeros((0, N_FRAMES, 96), np.float32)


def estimate_word(x: np.ndarray, search_sec=3.0, gap_sec=0.15, min_word_sec=0.15):
    """Return (onset_sample, word_end_sample) of the leading word ("Alexa").

    Smoothed 20 ms RMS; noise floor = 10th percentile; speech threshold =
    max(2.5 * floor, 0.02). Onset = first frame above threshold (0 if the
    clip starts in speech). Word end = start of the first gap >= gap_sec of
    sub-threshold frames after onset + min_word_sec. Falls back to
    onset + 0.8 s if there is no gap.
    """
    search = x[: int(search_sec * SR)]
    r = frame_rms(search, frame=320, hop=160)
    if len(r) < 8:
        return 0, min(len(x), int(0.8 * SR))
    floor = float(np.percentile(r, 10))
    th = max(2.5 * floor, 0.02)
    hit = r > th
    idx = np.where(hit)[0]
    if len(idx) == 0:
        return 0, min(len(x), int(0.8 * SR))
    on = int(idx[0])
    gap = int(round(gap_sec * SR / 160))
    start = on + int(round(min_word_sec * SR / 160))
    end = None
    i = start
    while i < len(hit) - gap:
        if not hit[i:i + gap].any():
            end = i
            break
        i += 1
    if end is None:
        end_s = on * 160 + int(0.8 * SR)
    else:
        end_s = end * 160
    return on * 160, int(min(len(x), end_s))


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, np.float64) ** 2) + 1e-12))


def speed_perturb(x: np.ndarray, f: float) -> np.ndarray:
    """Resample so duration scales by 1/f (f>1 faster, higher pitch)."""
    if abs(f - 1.0) < 1e-3:
        return x
    n_out = int(len(x) / f)
    return np.interp(np.linspace(0, len(x) - 1, n_out), np.arange(len(x)), x).astype(np.float32)


def colored_noise(n: int, rng, kind=None) -> np.ndarray:
    kind = kind or rng.choice(["white", "pink", "brown"])
    w = rng.standard_normal(n).astype(np.float32)
    if kind == "white":
        return w
    F = np.fft.rfft(w)
    f = np.maximum(np.arange(len(F)), 1).astype(np.float32)
    F = F / (np.sqrt(f) if kind == "pink" else f)
    out = np.fft.irfft(F, n).astype(np.float32)
    return out / (np.std(out) + 1e-9)


def noise_segment(pool, n: int, rng) -> np.ndarray:
    """Random n-sample slice of a random recording from pool (tiled if short)."""
    src = pool[rng.integers(0, len(pool))]
    if len(src) < n:
        src = np.tile(src, int(np.ceil(n / len(src))))
    s = rng.integers(0, len(src) - n + 1)
    return src[s:s + n]


def mix_snr(sig: np.ndarray, noise: np.ndarray, snr_db: float, sig_ref_rms: float) -> np.ndarray:
    nr = rms(noise)
    if nr < 1e-7:
        return sig
    noise = noise * (sig_ref_rms / (10 ** (snr_db / 20)) / nr)
    out = sig + noise
    pk = np.max(np.abs(out))
    if pk > 0.99:
        out = out * (0.99 / pk)
    return out.astype(np.float32)


ROOM_BANK: list = []


def build_room_bank(arrays, min_sec=0.4, max_floor=0.03):
    """Collect quiet (room-noise) segments from real TRAIN-side clips."""
    bank = []
    for x in arrays:
        r = frame_rms(x, 320, 160)
        if len(r) < 60:
            continue
        floor = float(np.percentile(r, 10))
        if floor > max_floor:
            continue
        q = (r < 1.8 * floor) & (r < max_floor)
        i, n = 0, len(q)
        while i < n:
            if q[i]:
                j = i
                while j < n and q[j]:
                    j += 1
                if (j - i) * 160 >= min_sec * SR:
                    bank.append(x[(i + 3) * 160: (j - 3) * 160 + 320].astype(np.float32).copy())
                i = j
            else:
                i += 1
    return bank


def room_pad(n: int, rng) -> np.ndarray:
    out = np.zeros(0, np.float32)
    xf = 320
    while len(out) < n:
        seg = ROOM_BANK[rng.integers(0, len(ROOM_BANK))]
        if rng.random() < 0.5:
            seg = seg[::-1]
        # mostly typical level, sometimes a much quieter (noise-gated / AEC-cleaned) background
        gdb = rng.uniform(-26, -6) if rng.random() < 0.35 else rng.uniform(-3, 3)
        seg = seg * np.float32(10 ** (gdb / 20))
        if len(out) == 0:
            out = seg.copy()
        else:
            ramp = np.linspace(0, 1, xf, dtype=np.float32)
            head = out[-xf:] * (1 - ramp) + seg[:xf] * ramp
            out = np.concatenate([out[:-xf], head, seg[xf:]])
    return out[:n].astype(np.float32)


def noise_floor_pad(n: int, x: np.ndarray, rng) -> np.ndarray:
    """Background-context padding: real room noise from the bank (falls back
    to Gaussian at the clip's floor only if no bank has been built)."""
    if ROOM_BANK:
        return room_pad(n, rng)
    r = frame_rms(x[: 3 * SR], 320, 160)
    fl = max(float(np.percentile(r, 10)), 3e-4)
    return (rng.standard_normal(n) * fl).astype(np.float32)


def startup_windows(af_stream, x: np.ndarray, n_chunks: int = 22) -> np.ndarray:
    """Run a FRESH openWakeWord streaming front-end over the first n_chunks*80ms
    of x and return the 16x96 classifier input after every chunk (these
    include the artificial init-noise embeddings the Model starts with)."""
    af_stream.reset()
    out = []
    xi = to_i16(x)
    for c in range(n_chunks):
        chunk = xi[c * HOP:(c + 1) * HOP]
        if len(chunk) < HOP:
            break
        af_stream(chunk)
        out.append(af_stream.get_features(16)[0])
    return np.stack(out).astype(np.float32) if out else np.zeros((0, 16, 96), np.float32)
