"""Lightweight augmentation: gain jitter + noise/music mixing at a target
SNR, using only local negative audio as the noise source (never anything
downloaded, and never anything from the held-out split).
"""
from __future__ import annotations

import numpy as np


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2) + 1e-12))


def random_gain(x: np.ndarray, rng: np.random.Generator, db_range=(-6, 6)) -> np.ndarray:
    db = rng.uniform(*db_range)
    return (x * (10 ** (db / 20))).astype(np.float32)


def mix_at_snr(signal: np.ndarray, noise: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Mix `noise` under `signal` at the given SNR (signal power fixed)."""
    if len(noise) < len(signal):
        reps = int(np.ceil(len(signal) / max(1, len(noise))))
        noise = np.tile(noise, reps)
    start = 0
    if len(noise) > len(signal):
        start = rng.integers(0, len(noise) - len(signal) + 1)
    noise = noise[start:start + len(signal)]
    sig_rms = rms(signal)
    noise_rms = rms(noise)
    if noise_rms < 1e-8:
        return signal
    target_noise_rms = sig_rms / (10 ** (snr_db / 20))
    noise = noise * (target_noise_rms / noise_rms)
    out = signal + noise
    peak = np.max(np.abs(out))
    if peak > 0.99:
        out = out * (0.99 / peak)
    return out.astype(np.float32)


def augment_clip(signal: np.ndarray, noise_pool: list, rng: np.random.Generator,
                  snr_choices=(0, 3, 6, 10)) -> np.ndarray:
    out = random_gain(signal, rng, db_range=(-4, 4))
    if noise_pool and rng.random() < 0.85:
        noise = noise_pool[rng.integers(0, len(noise_pool))]
        snr = float(rng.choice(snr_choices))
        out = mix_at_snr(out, noise, snr, rng)
    return out
