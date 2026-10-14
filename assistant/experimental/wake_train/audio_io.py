"""Small local audio helpers shared by the wake-word training scripts.

Kept dependency-light on purpose (wave + numpy only) so it works from
either the system Python (for quick checks) or the .venv-wake venv.
"""
from __future__ import annotations

import wave
import numpy as np

SR = 16000


def read_wav_mono16k(path: str) -> np.ndarray:
    """Read a 16-bit PCM wav, return float32 samples in [-1, 1] at 16kHz.

    Raises if the file is not already mono 16kHz 16-bit (all clips in this
    dataset are recorded that way, so a mismatch signals a real problem
    rather than something to silently resample).
    """
    with wave.open(path, "rb") as w:
        assert w.getsampwidth() == 2, f"{path}: expected 16-bit PCM"
        assert w.getframerate() == SR, f"{path}: expected {SR} Hz, got {w.getframerate()}"
        nch = w.getnchannels()
        raw = w.readframes(w.getnframes())
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if nch > 1:
        x = x.reshape(-1, nch).mean(axis=1)
    return x


def write_wav_mono16k(path: str, x: np.ndarray) -> None:
    x = np.clip(x, -1.0, 1.0)
    pcm = (x * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def frame_rms(x: np.ndarray, frame=320, hop=160) -> np.ndarray:
    """20ms frames / 10ms hop RMS energy."""
    n = 1 + max(0, (len(x) - frame) // hop)
    if n <= 0:
        return np.array([np.sqrt(np.mean(x ** 2) + 1e-12)])
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        seg = x[i * hop: i * hop + frame]
        out[i] = np.sqrt(np.mean(seg.astype(np.float64) ** 2) + 1e-12)
    return out


def energy_onset(x: np.ndarray, search_sec: float = 2.5, frame=320, hop=160) -> int:
    """Rough speech-onset sample index using an energy threshold.

    Estimates the noise floor from the first 100ms, then finds the first
    run of >=3 consecutive frames whose RMS exceeds max(4x floor, a small
    absolute floor). Falls back to sample 0 if nothing crosses (e.g. very
    quiet clip -- better to keep the original start than guess wrong).
    """
    search = x[: int(search_sec * SR)]
    rms = frame_rms(search, frame, hop)
    if len(rms) < 4:
        return 0
    noise_floor = float(np.median(rms[: max(1, int(0.1 * SR / hop))]))
    thresh = max(noise_floor * 4.0, 0.01)
    hits = rms > thresh
    for i in range(len(hits) - 2):
        if hits[i] and hits[i + 1] and hits[i + 2]:
            onset_frame = i
            return max(0, onset_frame * hop - int(0.05 * SR))  # 50ms pre-roll
    return 0


def energy_onset_offset(x: np.ndarray, search_sec: float = 2.5, frame=320, hop=160,
                         min_word_sec: float = 0.25):
    """Estimate (onset_sample, offset_sample) of the spoken word.

    Many of these clips start recording right at (or just before) speech
    onset, so there is no reliable leading-silence segment to calibrate a
    noise floor from. Use the 10th percentile RMS over the whole search
    window instead (robust whether or not the clip happens to start with
    silence). offset = first silence gap (>=250ms below threshold) found
    at least `min_word_sec` after onset (rejects spurious micro-gaps
    between syllables, e.g. the stop consonant in "-lex-"), else
    onset + 0.9s as a generic single/two-syllable-word fallback duration.
    """
    search = x[: int(search_sec * SR)]
    rms = frame_rms(search, frame, hop)
    if len(rms) < 4:
        return 0, min(len(x), int(0.9 * SR))
    noise_floor = float(np.percentile(rms, 10))
    thresh = max(noise_floor * 3.5, 0.015)
    hits = rms > thresh
    onset_frame = None
    for i in range(len(hits) - 2):
        if hits[i] and hits[i + 1] and hits[i + 2]:
            onset_frame = i
            break
    if onset_frame is None:
        return 0, min(len(x), int(0.9 * SR))
    onset_sample = max(0, onset_frame * hop - int(0.05 * SR))

    gap_frames = max(1, int(0.25 * SR / hop))
    min_word_frames = max(1, int(min_word_sec * SR / hop))
    offset_frame = None
    i = onset_frame + min_word_frames
    while i < len(hits) - gap_frames:
        if not hits[i:i + gap_frames].any():
            offset_frame = i
            break
        i += 1
    if offset_frame is None:
        offset_sample = onset_sample + int(0.9 * SR)
    else:
        offset_sample = offset_frame * hop
    return onset_sample, min(len(x), offset_sample)


def fixed_length(x: np.ndarray, n_samples: int, start: int = 0) -> np.ndarray:
    """Take/pad a fixed-length window starting at `start`."""
    seg = x[start:start + n_samples]
    if len(seg) < n_samples:
        pad = np.zeros(n_samples - len(seg), dtype=x.dtype)
        # pad with tiny noise floor instead of hard zeros to avoid a click
        pad = pad + np.random.default_rng(0).normal(0, 1e-4, size=pad.shape).astype(x.dtype)
        seg = np.concatenate([seg, pad])
    return seg


def window_ending_at(x: np.ndarray, end_sample: int, n_samples: int, rng_seed: int = 0) -> np.ndarray:
    """Return a fixed-length window whose LAST sample is `end_sample`
    (clamped to len(x)). If there isn't enough real audio before it, the
    deficit is padded with low-level noise at the FRONT so the real audio
    stays anchored at the end -- critical so a downstream "last N frames"
    feature window still lands on the real word, not on whatever audio
    happens to occupy the clip's absolute start.
    """
    end_sample = max(0, min(end_sample, len(x)))
    start = end_sample - n_samples
    if start >= 0:
        return x[start:end_sample].copy()
    pad_len = -start
    real = x[0:end_sample]
    pad = np.random.default_rng(rng_seed).normal(0, 1e-4, size=pad_len).astype(x.dtype)
    return np.concatenate([pad, real])
