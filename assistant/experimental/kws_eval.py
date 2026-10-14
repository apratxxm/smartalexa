#!/usr/bin/env python3
"""
Offline evaluation of sherpa-onnx open-vocabulary Zipformer keyword spotting
("KWS") as a candidate wake-word detector for this user's "Alexa" (Indian
English accent), run entirely locally against real logged audio.

This script is meant to be run with the SEPARATE evaluation venv at
    A:\\pisha\\memoir\\.venv-kws
which has `sherpa-onnx` + `numpy` installed. It does NOT touch, import, or
require anything from A:\\pisha\\memoir\\.venv or assistant/*.py, and it never
reads assistant/.env or assistant/.cache. No audio is ever transmitted
anywhere -- inference is 100% local ONNX Runtime CPU inference.

Usage (PowerShell):
    A:\\pisha\\memoir\\.venv-kws\\Scripts\\python.exe ^
        A:\\pisha\\memoir\\assistant\\experimental\\kws_eval.py

Model used: sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01
  (open-vocabulary streaming Zipformer keyword spotter, ~3.3M params,
   English, Apache-2.0, from https://github.com/k2-fsa/sherpa-onnx
   releases tag `kws-models`), extracted to:
    A:\\kws-models\\sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01
The keyword "ALEXA" needs no training -- its BPE token sequence
("_A LE X A") already ships in that model's own keywords_raw.txt/keywords.txt,
and is otherwise obtained by running:
    sherpa-onnx-cli text2token --tokens tokens.txt --tokens-type bpe \\
        --bpe-model bpe.model keywords_raw.txt keywords.txt

Data sources (read-only):
  - Positive "Alexa" clips: assistant/logs/telemetry.jsonl events with
    trigger_source == "stt" and (trigger_transcript or transcript) starting
    with "alexa" (case-insensitive) -> audio_path. First 2.5s used.
  - Negative audio:
      * assistant/logs/aec/*/mic16k.wav and assistant/logs/aec/live-*/raw.wav
        (mic-captured audio, mostly with music playing), sliced into
        non-overlapping 3s windows.
      * telemetry retry_audio_path / clarification_audio_path /
        follow_up_audio_path clips whose associated transcript (when one
        exists) does not contain "alexa" -- real user speech without the
        wake word.
"""
from __future__ import annotations

import json
import sys
import time
import wave
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import sherpa_onnx

REPO_ROOT = Path(r"A:\pisha\memoir")
TELEMETRY_PATH = REPO_ROOT / "assistant" / "logs" / "telemetry.jsonl"
AEC_DIR = REPO_ROOT / "assistant" / "logs" / "aec"
MODEL_DIR = Path(r"A:\kws-models\sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01")

SAMPLE_RATE = 16000
FRAME_SAMPLES = 512  # matches the 512-sample/16kHz streaming frame size used in production
POS_CLIP_SECONDS = 2.5
NEG_WINDOW_SECONDS = 3.0
TAIL_PAD_SECONDS = 0.66  # per sherpa-onnx example, lets the decoder flush trailing state


def read_wav_mono16k(path: Path) -> Tuple[Optional[np.ndarray], int]:
    """Returns (int16 samples, sample_rate). samples is None if not 16-bit."""
    with wave.open(str(path), "rb") as f:
        if f.getsampwidth() != 2:
            return None, f.getframerate()
        n = f.getnframes()
        raw = f.readframes(n)
        sr = f.getframerate()
        ch = f.getnchannels()
    samples = np.frombuffer(raw, dtype=np.int16)
    if ch > 1:
        samples = samples.reshape(-1, ch)[:, 0].copy()
    if sr != SAMPLE_RATE:
        return None, sr
    return samples, sr


def to_float32(samples_int16: np.ndarray) -> np.ndarray:
    return samples_int16.astype(np.float32) / 32768.0


def load_telemetry() -> List[dict]:
    events = []
    with open(TELEMETRY_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def resolve_path(rel: str) -> Path:
    return REPO_ROOT / rel.replace("\\", "/")


def collect_positive_clips(events: List[dict]) -> List[Path]:
    clips = []
    for e in events:
        if e.get("trigger_source") != "stt":
            continue
        transcript = (e.get("trigger_transcript") or e.get("transcript") or "").strip().lower()
        if not transcript.startswith("alexa"):
            continue
        rel = e.get("audio_path")
        if not rel:
            continue
        path = resolve_path(rel)
        if path.exists():
            clips.append(path)
    return clips


def collect_negative_clips(events: List[dict]) -> List[Tuple[str, np.ndarray]]:
    clips: List[Tuple[str, np.ndarray]] = []
    win = int(NEG_WINDOW_SECONDS * SAMPLE_RATE)

    aec_files = sorted(AEC_DIR.glob("*/mic16k.wav")) + sorted(AEC_DIR.glob("live-*/raw.wav"))
    for wav_path in aec_files:
        samples, sr = read_wav_mono16k(wav_path)
        if samples is None:
            print(f"  skip {wav_path}: unsupported/non-16k format (sr={sr})", file=sys.stderr)
            continue
        for start in range(0, len(samples) - win + 1, win):
            clips.append((f"{wav_path.parent.name}/{wav_path.name}@{start/SAMPLE_RATE:.1f}s",
                          samples[start:start + win]))

    field_pairs = [
        ("retry_audio_path", "retry_transcript"),
        ("clarification_audio_path", "clarification_transcript"),
        ("follow_up_audio_path", None),
    ]
    for e in events:
        for path_field, transcript_field in field_pairs:
            rel = e.get(path_field)
            if not rel:
                continue
            if transcript_field:
                t = (e.get(transcript_field) or "").strip().lower()
                if "alexa" in t:
                    continue
            path = resolve_path(rel)
            if not path.exists():
                continue
            samples, sr = read_wav_mono16k(path)
            if samples is None:
                continue
            samples = samples[:win] if len(samples) > win else samples
            clips.append((f"telemetry/{path.name}", samples))
    return clips


def build_spotter(keywords_file: Path, keywords_score: float, keywords_threshold: float,
                   num_threads: int = 1) -> sherpa_onnx.KeywordSpotter:
    return sherpa_onnx.KeywordSpotter(
        tokens=str(MODEL_DIR / "tokens.txt"),
        encoder=str(MODEL_DIR / "encoder-epoch-12-avg-2-chunk-16-left-64.onnx"),
        decoder=str(MODEL_DIR / "decoder-epoch-12-avg-2-chunk-16-left-64.onnx"),
        joiner=str(MODEL_DIR / "joiner-epoch-12-avg-2-chunk-16-left-64.onnx"),
        keywords_file=str(keywords_file),
        num_threads=num_threads,
        sample_rate=SAMPLE_RATE,
        keywords_score=keywords_score,
        keywords_threshold=keywords_threshold,
        provider="cpu",
    )


def run_stream(spotter: sherpa_onnx.KeywordSpotter, samples_int16: np.ndarray,
               clip_seconds_limit: Optional[float] = None) -> Tuple[int, Optional[float], float, float]:
    """Feed audio through the spotter in 512-sample/16kHz frames, exactly like a
    real streaming mic loop would. Returns (num_detections, first_detect_ms,
    wall_seconds, audio_seconds)."""
    samples_f32 = to_float32(samples_int16)
    if clip_seconds_limit is not None:
        samples_f32 = samples_f32[: int(clip_seconds_limit * SAMPLE_RATE)]

    stream = spotter.create_stream()
    num_detections = 0
    first_detect_ms: Optional[float] = None
    n = len(samples_f32)
    pos = 0

    t0 = time.perf_counter()
    while pos < n:
        chunk = samples_f32[pos: pos + FRAME_SAMPLES]
        stream.accept_waveform(SAMPLE_RATE, chunk)
        pos += FRAME_SAMPLES
        while spotter.is_ready(stream):
            spotter.decode_stream(stream)
            r = spotter.get_result(stream)
            if r:
                num_detections += 1
                if first_detect_ms is None:
                    first_detect_ms = (pos / SAMPLE_RATE) * 1000.0
                spotter.reset_stream(stream)

    tail = np.zeros(int(TAIL_PAD_SECONDS * SAMPLE_RATE), dtype=np.float32)
    stream.accept_waveform(SAMPLE_RATE, tail)
    stream.input_finished()
    while spotter.is_ready(stream):
        spotter.decode_stream(stream)
        r = spotter.get_result(stream)
        if r:
            num_detections += 1
            if first_detect_ms is None:
                first_detect_ms = (n / SAMPLE_RATE) * 1000.0
            spotter.reset_stream(stream)
    wall = time.perf_counter() - t0

    return num_detections, first_detect_ms, wall, n / SAMPLE_RATE


def evaluate(keywords_file: Path, score: float, threshold: float,
             positive_clips: List[Path], negative_clips: List[Tuple[str, np.ndarray]]) -> dict:
    spotter = build_spotter(keywords_file, score, threshold)

    pos_hits = 0
    total_wall = 0.0
    total_audio = 0.0
    for path in positive_clips:
        samples, sr = read_wav_mono16k(path)
        if samples is None:
            continue
        num_det, first_ms, wall, audio_s = run_stream(spotter, samples, clip_seconds_limit=POS_CLIP_SECONDS)
        total_wall += wall
        total_audio += audio_s
        if num_det > 0:
            pos_hits += 1

    neg_false_events = 0
    neg_windows_with_false = 0
    neg_audio_seconds = 0.0
    for _label, samples in negative_clips:
        num_det, first_ms, wall, audio_s = run_stream(spotter, samples)
        total_wall += wall
        total_audio += audio_s
        neg_audio_seconds += audio_s
        neg_false_events += num_det
        if num_det > 0:
            neg_windows_with_false += 1

    pos_total = len(positive_clips)
    neg_minutes = neg_audio_seconds / 60.0
    return {
        "keywords_score": score,
        "keywords_threshold": threshold,
        "pos_hits": pos_hits,
        "pos_total": pos_total,
        "detection_rate": round(pos_hits / pos_total, 4) if pos_total else 0.0,
        "neg_false_events": neg_false_events,
        "neg_windows_with_false": neg_windows_with_false,
        "neg_windows_total": len(negative_clips),
        "neg_audio_minutes": round(neg_minutes, 2),
        "false_events_per_minute": round(neg_false_events / neg_minutes, 4) if neg_minutes else 0.0,
        "rtf_cpu": round(total_wall / total_audio, 4) if total_audio else 0.0,
        "avg_processing_ms_per_audio_sec": round(1000.0 * total_wall / total_audio, 2) if total_audio else 0.0,
    }


def main() -> None:
    print(f"Loading telemetry from {TELEMETRY_PATH}")
    events = load_telemetry()

    positive_clips = collect_positive_clips(events)
    negative_clips = collect_negative_clips(events)
    neg_minutes = sum(len(s) for _, s in negative_clips) / SAMPLE_RATE / 60.0
    print(f"positives: {len(positive_clips)} clips (first {POS_CLIP_SECONDS}s each)")
    print(f"negatives: {len(negative_clips)} windows, {neg_minutes:.2f} min total")

    keywords_file = MODEL_DIR / "keywords_alexa.txt"
    # BPE token sequence for "ALEXA", taken as-is from the model's own
    # keywords_raw.txt / keywords.txt (no training or per-line boost/threshold
    # baked in -- those are swept via the KeywordSpotter constructor below).
    keywords_file.write_text("\u2581A LE X A\n", encoding="utf-8")

    sweeps = [
        (1.0, 0.25),  # sherpa-onnx library defaults
        (2.0, 0.25),  # boosted score, same threshold
        (2.0, 0.35),  # boosted score, stricter threshold
    ]

    results = []
    for score, threshold in sweeps:
        print(f"\n--- keywords_score={score} keywords_threshold={threshold} ---")
        r = evaluate(keywords_file, score, threshold, positive_clips, negative_clips)
        results.append(r)
        print(json.dumps(r, indent=2))

    out_path = Path(__file__).with_name("kws_eval_results.json")
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
