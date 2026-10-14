from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


def read_wav_int16(path: Path, offset_samples: int = 0, num_samples: int = None) -> np.ndarray:
    with wave.open(str(path), "rb") as f:
        sr = f.getframerate()
        assert sr == 16000, f"expected 16kHz, got {sr} for {path}"
        if offset_samples:
            f.setpos(offset_samples)
        n = num_samples if num_samples is not None else (f.getnframes() - offset_samples)
        raw = f.readframes(n)
    return np.frombuffer(raw, dtype=np.int16)


def to_float32(samples_int16: np.ndarray) -> np.ndarray:
    return samples_int16.astype(np.float32) / 32768.0
