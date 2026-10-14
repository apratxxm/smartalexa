#!/usr/bin/env python3
"""Offline, non-authoritative wake-label proposals for human listening review.

Teacher predictions are proposals, not independent audio reviews. Reusing the
same model identity cannot claim independent evidence. This module never writes
human reviews, trains a model, or promotes labels; ASR absence is never negative.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import wave

try:
    from . import active_learning as review
    from . import review_server
except ImportError:  # direct script execution
    import active_learning as review
    import review_server

MAX_CANDIDATES = 30
MAX_CLIPS = 8
MAX_SECONDS = 60
MAX_CLIP_SECONDS = 24
MAX_WAV_BYTES = 1024 * 1024
MIN_FREE_BYTES = 1536 * 1024 * 1024
MODEL_OVERHEAD_BYTES = 512 * 1024 * 1024
PROPOSAL_SCHEMA = "memoir.wake-machine-proposals.v1"
OUTPUT_ROOT = Path(__file__).parent / "proposals"


def exact_wake_token(text):
    """True only for a whole Alexa token, not substrings such as 'Alexis'."""
    return isinstance(text, str) and re.search(r"(?<!\w)alexa(?!\w)", text, re.IGNORECASE) is not None


def proposal_presence(teacher_text):
    return "positive" if exact_wake_token(teacher_text) else "unsure"


def _finite_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if math.isfinite(value) else None


def _safe_scores(value):
    if not isinstance(value, dict):
        return {}
    return {str(key): score for key, raw in value.items()
            if (score := _finite_number(raw)) is not None}


def _model_identity(row):
    value = row.get("stt_model") or row.get("trigger_stt_model") or ""
    return value if isinstance(value, str) else ""


def _validate_wav(path):
    path = Path(path)
    if path.stat().st_size > MAX_WAV_BYTES:
        raise ValueError("WAV exceeds 1 MiB proposal limit")
    identity = review.wav_identity(path)
    duration = identity["n_frames"] / 16000
    if duration > MAX_CLIP_SECONDS:
        raise ValueError("WAV exceeds 24 second proposal limit")
    return identity, duration


def _asr_evidence(telemetry, event_ids):
    """Read only bounded JSONL rows relevant to already-selected event IDs."""
    wanted, found = set(event_ids), {}
    if not wanted:
        return found
    with Path(telemetry).open(encoding="utf-8") as stream:
        for line in stream:
            if len(line) > 1024 * 1024:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict) and row.get("event_id") in wanted:
                found.setdefault(row["event_id"], {
                    "text": row.get("transcript", "") if isinstance(row.get("transcript", ""), str) else "",
                    "model_id": _model_identity(row),
                    "confidence": _finite_number(row.get("stt_confidence")),
                    "wake_scores": _safe_scores(row.get("wake_scores", {})),
                    "wake_score": _finite_number(row.get("wake_score")),
                })
    return found


def build_proposals(*, telemetry=review.TELEMETRY, audio_root=review.AUDIO_ROOT,
                    project_root=review.PROJECT_ROOT, store=None, candidates=None,
                    teacher=None, decoder=None, max_candidates=MAX_CANDIDATES,
                    exclude_reviewed=False):
    """Build bounded proposal rows; injected decoder enables offline unit tests."""
    if (isinstance(max_candidates, bool) or not isinstance(max_candidates, int)
            or not 1 <= max_candidates <= MAX_CANDIDATES):
        raise ValueError("max_candidates must be between 1 and 30")
    if candidates is None:
        candidates = review_server.recent_candidates(telemetry, audio_root, project_root,
                                                     limit=max_candidates, selection="recent")
    candidates = list(candidates)[:max_candidates]
    metadata, _usable, _missing = review.review_history(
        store if store is not None else {"schema": review.STORE_SCHEMA, "reviews": []},
        audio_root, project_root)
    reviewed_events = {(row["event_id"], row["audio_field"]): row for row in metadata}
    reviewed_hashes = {row["wav_sha256"] for row in metadata if row.get("wav_sha256")}
    asr = _asr_evidence(telemetry, [c.get("event_id") for c in candidates]) if telemetry else {}
    seen_hashes, rows = set(), []
    total_seconds = 0.0
    for candidate in candidates:
        key = (candidate.get("event_id"), candidate.get("audio_field"))
        prior = reviewed_events.get(key)
        if exclude_reviewed and prior:
            continue
        if candidate.get("unavailable") or not candidate.get("path"):
            continue
        path = review.audio_path(candidate["path"], audio_root, project_root)
        identity, duration = _validate_wav(path)
        digest = identity["wav_sha256"]
        if digest in seen_hashes or digest in reviewed_hashes:
            continue
        if prior:
            # Never contradict known human label; known rows are not proposals.
            continue
        if len(rows) >= MAX_CLIPS or total_seconds + duration > MAX_SECONDS:
            continue
        seen_hashes.add(digest)
        total_seconds += duration
        evidence = asr.get(candidate.get("event_id"), {})
        teacher_result = None
        teacher_id = None
        if teacher is not None:
            if decoder is None:
                raise ValueError("teacher mode requires an injected or local decoder")
            teacher_result = decoder(path)
            teacher_id = teacher
        rows.append({
            "event_id": key[0], "audio_field": key[1], "wav_sha256": digest,
            "duration_seconds": duration, "source_model_ids": {
                "asr": evidence.get("model_id", ""), "teacher": teacher_id},
            "asr": {"text": evidence.get("text", ""), "confidence": evidence.get("confidence"),
                    "wake_scores": evidence.get("wake_scores", {}), "wake_score": evidence.get("wake_score")},
            "teacher": {"text": teacher_result, "evidence": "whole-token Alexa" if exact_wake_token(teacher_result) else "no exact token; unsure"} if teacher is not None else None,
            "proposed_presence": proposal_presence(teacher_result) if teacher is not None else "unsure",
            "human_reviewed": False, "training_eligible": False, "addressed_assistant": "unsure",
        })
    return {"schema": PROPOSAL_SCHEMA, "rows": rows, "human_reviewed": False,
            "training_eligible": False, "training_started": False,
            "teacher_is_human_review": False, "clip_count": len(rows), "total_seconds": total_seconds}


def physical_free_bytes():
    """Physical memory gate; unavailable platforms fail closed."""
    if os.name != "nt":
        raise RuntimeError("physical-memory gate unavailable; teacher inference disabled")
    import ctypes
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    status = MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise RuntimeError("could not read physical memory; teacher inference disabled")
    return int(status.ullAvailPhys)


def required_free_bytes(model_path):
    path = Path(model_path).resolve(strict=True)
    if not path.is_dir() or not (path / "model.bin").is_file() or not (path / "config.json").is_file():
        raise ValueError("teacher requires an existing local faster-whisper model directory with model.bin and config.json")
    size = (path / "model.bin").stat().st_size
    return max(MIN_FREE_BYTES, 3 * size + MODEL_OVERHEAD_BYTES)


def make_local_decoder(model_path, *, memory_gate=physical_free_bytes, loader=None):
    """Create serial CPU int8 decoder only after local-model and RAM admission."""
    model_path = Path(model_path).resolve(strict=True)
    required = required_free_bytes(model_path)
    available = memory_gate()
    if available < required:
        raise MemoryError(f"teacher inference requires {required} free physical bytes; found {available}")
    if loader is None:
        from faster_whisper import WhisperModel
        loader = WhisperModel
    model = loader(str(model_path), device="cpu", compute_type="int8", local_files_only=True,
                   cpu_threads=2, num_workers=1)
    def decode(path):
        segments, _info = model.transcribe(str(path), language="en", beam_size=1,
            best_of=1, temperature=0, condition_on_previous_text=False,
            max_new_tokens=32, word_timestamps=False)
        texts = []
        for index, segment in enumerate(segments):
            if index >= 32:
                break
            texts.append(str(segment.text))
        return " ".join(texts)
    return decode


def _safe_output(path):
    raw = Path(path)
    target = raw.resolve()
    allowed = OUTPUT_ROOT.resolve()
    if not target.is_relative_to(allowed) or target.suffix.lower() != ".json":
        raise ValueError("proposal output must be a JSON file under experimental/wake_train/proposals")
    protected_names = review.LEGACY_NAMES | {"human_reviews.json", "wake.onnx", "model.onnx"}
    if target.name.lower() in protected_names or target.exists() or os.path.lexists(raw):
        raise ValueError("proposal output exists, aliases a file, or uses a protected legacy name")
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--telemetry", default=str(review.TELEMETRY))
    parser.add_argument("--audio-root", default=str(review.AUDIO_ROOT))
    parser.add_argument("--project-root", default=str(review.PROJECT_ROOT))
    parser.add_argument("--store")
    parser.add_argument("--exclude-reviewed", action="store_true")
    parser.add_argument("--teacher-model")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    try:
        # Preflight paths and human history before importing/loading any model.
        target = _safe_output(args.output) if args.output else None
        store_path = args.store or str(review_server.DEFAULT_STORE)
        if Path(store_path).exists():
            human_store = review.load_store(store_path)
        else:
            human_store = {"schema": review.STORE_SCHEMA, "reviews": []}
        # Validate full human metadata and archive identity before any model import/load.
        review.review_history(human_store, args.audio_root, args.project_root)
        # Dry-run is metadata-only and never imports faster-whisper.
        decoder = make_local_decoder(args.teacher_model) if args.teacher_model else None
        report = build_proposals(telemetry=args.telemetry, audio_root=args.audio_root,
            project_root=args.project_root,
            store=human_store,
            teacher=str(Path(args.teacher_model).resolve()) if args.teacher_model else None,
            decoder=decoder, exclude_reviewed=args.exclude_reviewed)
        if target is not None:
            # Serialize fully before creating the new artifact.
            payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("x", encoding="utf-8") as stream:
                stream.write(payload)
        else:
            print(json.dumps(report, allow_nan=False))
        return 0
    except (OSError, ValueError, MemoryError, RuntimeError) as exc:
        parser.exit(2, f"wake proposal refused: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
