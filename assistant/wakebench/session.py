"""PCM sessions, acoustic anchors, and explicit alignment evidence."""
from __future__ import annotations

import json
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from assistant.wake import FRAME_LENGTH, SAMPLE_RATE

RATE = SAMPLE_RATE
FRAME = FRAME_LENGTH
CONFIDENCE = 0.45


def chirp() -> np.ndarray:
    """Fixed 60 ms, 1.5–4 kHz sweep; PCM16 waveform is saved per session."""
    n = round(.060 * RATE)
    t = np.arange(n) / RATE
    phase = 2 * np.pi * (1500 * t + (4000 - 1500) / (.120) * t * t)
    return np.round(np.sin(phase) * np.hanning(n) * 6000).astype(np.int16)


def write_wav(path: Path, audio: np.ndarray, rate: int = RATE) -> None:
    with wave.open(str(path), "wb") as out:
        out.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        out.writeframes(np.asarray(audio, dtype="<i2").tobytes())


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as src:
        if src.getsampwidth() != 2 or src.getnchannels() != 1:
            raise ValueError(f"Expected mono PCM16 WAV: {path}")
        return np.frombuffer(src.readframes(src.getnframes()), dtype="<i2").copy(), src.getframerate()


def read_audio(path: Path) -> np.ndarray:
    audio, rate = read_wav(path)
    if rate == RATE:
        return audio
    import soxr
    # Same streaming resampler and int16 conversion as Microphone.read().
    resampler = soxr.ResampleStream(rate, RATE, 1, dtype="float32")
    parts = []
    for start in range(0, audio.size, rate):
        part = resampler.resample_chunk(audio[start:start + rate].astype(np.float32),
                                        last=start + rate >= audio.size)
        parts.append(np.clip(part, -32768, 32767).astype(np.int16))
    return np.concatenate(parts) if parts else np.zeros(0, np.int16)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as src:
        rows = [json.loads(line) for line in src if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"JSONL rows must be objects: {path}")
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


def window(cue_end: int) -> tuple[int, int]:
    return cue_end + round(.150 * RATE), cue_end + round(3.5 * RATE)


def locate(audio: np.ndarray, cue: np.ndarray, expected: int,
           radius: int = RATE * 2) -> tuple[int | None, float]:
    """Normalized matched filter, with FFT convolution and rolling energy."""
    lo = max(0, int(expected) - radius)
    hi = min(len(audio), int(expected) + radius + len(cue))
    x = audio[lo:hi].astype(np.float64)
    template = cue.astype(np.float64)
    if x.size < template.size or not np.any(template):
        return None, 0.0
    n = 1 << (len(x) + len(template) - 2).bit_length()
    corr = np.fft.irfft(np.fft.rfft(x, n) * np.fft.rfft(template[::-1], n), n)
    corr = corr[len(template) - 1:len(x)]
    energy = np.concatenate(([0.], np.cumsum(x * x)))
    energy = energy[len(template):] - energy[:-len(template)]
    confidence = np.abs(corr) / np.sqrt(np.maximum(energy * np.dot(template, template), 1.))
    best = int(np.argmax(confidence))
    score = float(np.clip(confidence[best], 0, 1))
    return (lo + best if score >= CONFIDENCE else None), score


@dataclass
class Alignment:
    mic: np.ndarray
    reference: np.ndarray

    def map(self, positions, *, inverse=False):
        x, y = (self.reference, self.mic) if inverse else (self.mic, self.reference)
        p = np.asarray(positions, dtype=np.float64)
        if len(x) == 1:
            return p + y[0] - x[0]
        segment = np.clip(np.searchsorted(x, p, side="right") - 1, 0, len(x) - 2)
        return y[segment] + (p - x[segment]) * (y[segment + 1] - y[segment]) / (x[segment + 1] - x[segment])


def manifest_path(path: Path) -> Path:
    return path / "manifest.jsonl" if path.is_dir() else path


def finalize(path: Path) -> list[dict]:
    path = manifest_path(path)
    rows = read_jsonl(path)
    headers = [r for r in rows if r.get("type") == "session"]
    if len(headers) != 1:
        raise ValueError("Expected exactly one session header")
    header = headers[0]
    if header.get("mode") not in ("positive", "negative") or not header.get("session_id"):
        raise ValueError("Session requires a mode and session_id")
    if any(r.get("session_id") != header["session_id"] for r in rows):
        raise ValueError("Manifest contains mismatched session IDs")
    if header.get("aborted") or header.get("writer_error"):
        raise ValueError("Aborted/incomplete session cannot be finalized")
    cue, cue_rate = read_wav(path.parent / header["cue_wav"])
    if cue_rate != RATE:
        raise ValueError("Cue waveform must be at runtime sample rate")
    mic = read_audio(path.parent / header["mic_wav"])
    ref = read_audio(path.parent / header["loopback_wav"])
    if len(mic) < FRAME or len(ref) < FRAME:
        raise ValueError("Paired WAVs must contain at least one runtime frame")
    cues = [r for r in rows if r.get("type") == "cue"]
    if not cues:
        raise ValueError("Session has no alignment cues")
    for row in cues:
        for name, audio in (("mic", mic), ("loopback", ref)):
            start, confidence = locate(audio, cue, row[f"expected_{name}_sample"])
            row[f"{name}_cue_start"] = start
            row[f"{name}_confidence"] = confidence
        row["invalid_reasons"] = ["user_marked"] if row.get("user_invalid") else []
        if row["mic_cue_start"] is None or row["loopback_cue_start"] is None:
            row["invalid_reasons"].append("missing_cue")
    matched = [r for r in cues if r["mic_cue_start"] is not None and r["loopback_cue_start"] is not None]
    if not matched:
        header["alignment"] = None
        alignment = None
    else:
        mx = np.array([r["mic_cue_start"] for r in matched])
        rx = np.array([r["loopback_cue_start"] for r in matched])
        if np.any(np.diff(mx) <= 0) or np.any(np.diff(rx) <= 0):
            raise ValueError("Cues are duplicated or out of order; alignment refused")
        alignment = Alignment(mx, rx)
        header["alignment"] = {"mic_samples": mx.tolist(), "loopback_samples": rx.tolist(),
                               "method": "piecewise_linear_chirp", "confidence_minimum": CONFIDENCE}
    bad = []
    for stream in ("mic", "loopback"):
        for block in read_jsonl(path.parent / header[f"{stream}_blocks"]):
            if block.get("discontinuity") or block.get("overflow"):
                start = block["start_sample"] * RATE / header[f"{stream}_rate"]
                end = (block["start_sample"] + block["samples"]) * RATE / header[f"{stream}_rate"]
                if stream == "loopback":
                    if alignment is None:
                        bad.append([0, len(mic)])
                        continue
                    start, end = alignment.map([start, end], inverse=True)
                bad.append([max(0, int(np.floor(start))), min(len(mic), int(np.ceil(end)))])
    for row in cues:
        if row["mic_cue_start"] is not None:
            row["cue_onset_sample"] = row["mic_cue_start"]
            row["cue_offset_sample"] = row["mic_cue_start"] + len(cue)
            start, end = window(row["cue_offset_sample"])
        else:
            start, end = window(row["expected_mic_sample"] + len(cue))
        row["window_start_sample"], row["window_end_sample"] = start, end
        if end > len(mic) and not row.get("anchor"):
            row["invalid_reasons"].append("incomplete_window")
        if any(a < end and b > start for a, b in bad):
            row["invalid_reasons"].append("discontinuity")
        row["invalid"] = bool(row["invalid_reasons"])
    header.update(sample_rate=RATE, frame_samples=FRAME, duration_samples=len(mic),
                  loopback_duration_samples=len(ref), discontinuities=bad,
                  finalized=True, cue_samples=len(cue))
    write_jsonl(path.parent / "finalized.jsonl", rows)
    return rows
