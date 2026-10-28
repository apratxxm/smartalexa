"""Piper rendering and seeded augmentation (reverb, room noise, gain). numpy only at import time."""
from __future__ import annotations

import numpy as np

SR = 16000
GAIN_DB = (-4.0, 4.0)
SNR_CHOICES = (5.0, 10.0, 15.0, 20.0)
NOISE_PROB = 0.85
REVERB_PROB = 0.5


def trim_silence(x: np.ndarray, pad: float = 0.08, floor: float = 0.01) -> np.ndarray:
    idx = np.where(np.abs(x) > floor)[0]
    if idx.size == 0:
        return x
    return x[max(0, idx[0] - int(pad * SR)): idx[-1] + int(pad * SR)]


def resample_rate(x: np.ndarray, factor: float) -> np.ndarray:
    """Duration scales by 1/factor (factor > 1 is faster and higher pitched)."""
    if abs(factor - 1.0) < 1e-6:
        return x
    n_out = max(2, int(len(x) / factor))
    return np.interp(np.linspace(0, len(x) - 1, n_out), np.arange(len(x)), x).astype(np.float32)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(x, np.float64) ** 2) + 1e-12))


def synthetic_rir(rng: np.random.Generator, sr: int = SR) -> np.ndarray:
    """Generated exponential-decay impulse response (no external asset)."""
    rt60 = rng.uniform(0.15, 0.5)
    n = int(rt60 * sr)
    t = np.arange(n) / sr
    h = rng.standard_normal(n) * np.exp(-6.9 * t / rt60)
    h[0] = 1.0
    return (h / np.sqrt(np.sum(h ** 2))).astype(np.float32)


def reverb(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h = synthetic_rir(rng)
    y = np.convolve(x, h)[: len(x)]
    gain = rms(x) / max(rms(y), 1e-9)
    return (y * gain).astype(np.float32)


def colored_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """Synthetic fallback noise used only when no archive room bank exists."""
    kind = ("white", "pink", "brown")[int(rng.integers(0, 3))]
    w = rng.standard_normal(n).astype(np.float32)
    if kind == "white":
        return w
    spec = np.fft.rfft(w)
    f = np.maximum(np.arange(len(spec)), 1).astype(np.float32)
    spec = spec / (np.sqrt(f) if kind == "pink" else f)
    out = np.fft.irfft(spec, n).astype(np.float32)
    return out / (np.std(out) + 1e-9)


def noise_segment(bank: list[np.ndarray], n: int, rng: np.random.Generator) -> np.ndarray:
    if not bank:
        return colored_noise(n, rng)
    src = bank[int(rng.integers(0, len(bank)))]
    if len(src) < n:
        src = np.tile(src, int(np.ceil(n / len(src))))
    s = int(rng.integers(0, len(src) - n + 1))
    return src[s:s + n]


def mix_at_snr(x: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    nr = rms(noise)
    if nr < 1e-7:
        return x
    noise = noise * (rms(x) / (10 ** (snr_db / 20)) / nr)
    out = x + noise
    peak = float(np.max(np.abs(out)))
    if peak > 0.99:
        out = out * (0.99 / peak)
    return out.astype(np.float32)


def augment(x: np.ndarray, rng: np.random.Generator, room_bank: list[np.ndarray]) -> tuple[np.ndarray, dict]:
    """Deterministic given rng. Returns audio and the chosen options (for provenance)."""
    info: dict = {}
    gain = float(rng.uniform(*GAIN_DB))
    out = (x * 10 ** (gain / 20)).astype(np.float32)
    info["gain_db"] = round(gain, 3)
    if rng.random() < REVERB_PROB:
        out = reverb(out, rng)
        info["reverb"] = True
    if rng.random() < NOISE_PROB:
        snr = float(rng.choice(SNR_CHOICES))
        out = mix_at_snr(out, noise_segment(room_bank, len(out), rng), snr)
        info["snr_db"] = snr
        info["noise_source"] = "archive_room_bank" if room_bank else "synthetic_colored_fallback"
    return out, info


def to_pcm16(x: np.ndarray) -> np.ndarray:
    return np.clip(np.round(np.asarray(x, np.float64) * 32767.0), -32768, 32767).astype(np.int16)


def render_piper(voice, speaker: int, text: str, *, length_scale: float, noise_scale: float,
                 noise_w: float, rate: float) -> np.ndarray:
    """Synthesize with a loaded PiperVoice; returns float32 mono at 16 kHz."""
    from piper.config import SynthesisConfig

    cfg = SynthesisConfig(speaker_id=speaker, length_scale=length_scale, noise_scale=noise_scale,
                          noise_w_scale=noise_w, volume=1.0)
    chunks = list(voice.synthesize(text, cfg))
    if not chunks:
        raise ValueError(f"Piper produced no audio for {text!r}")
    audio = np.concatenate([c.audio_float_array for c in chunks]).astype(np.float32)
    sr = int(chunks[0].sample_rate)
    if sr != SR:
        n_out = int(len(audio) / sr * SR)
        audio = np.interp(np.linspace(0, len(audio), n_out, endpoint=False),
                          np.arange(len(audio)), audio).astype(np.float32)
    return resample_rate(trim_silence(audio), rate)
