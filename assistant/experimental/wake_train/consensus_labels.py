#!/usr/bin/env python3
"""Create bounded same-WAV ASR consensus proposals for human listening.

Both transcripts are model evidence, never human review. A positive proposal
requires the exact word Alexa in both independently generated transcripts of
the same verified archived WAV. All other rows remain unsure; absence is never
a negative label. This tool never writes the human review store or trains or
promotes a model.
"""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import wave

try:
    from . import active_learning as review
    from . import machine_proposals as proposal_runtime
except ImportError:  # direct script execution
    import active_learning as review
    import machine_proposals as proposal_runtime

SCHEMA = "memoir.wake-machine-consensus.v1"
OUTPUT_ROOT = Path(__file__).parent / "proposals"
MAX_TELEMETRY_LINES = 30000
MAX_TELEMETRY_LINE_BYTES = 1024 * 1024
MAX_AGE_SECONDS = 48 * 60 * 60
MAX_ACCEPTED_EVENTS = 16
MAX_FRESH_EVENTS = 8
MAX_RECORDS = MAX_ACCEPTED_EVENTS + MAX_FRESH_EVENTS
MAX_TOTAL_SECONDS = 600.0
MAX_CLIP_SECONDS = 24.0
MAX_WAV_BYTES = 1024 * 1024
MAX_TEXT_CHARS = 400
MIN_FREE_BYTES = 1536 * 1024 * 1024
MODEL_OVERHEAD_BYTES = 512 * 1024 * 1024
MAX_STAGE_INPUT_BYTES = 256 * 1024
MAX_HUMAN_STORE_BYTES = review.MAX_PREPARATION_STORE_BYTES
DEFAULT_STORE = review.PROJECT_ROOT / "assistant" / "experimental" / "wake_train" / "human_reviews.json"
LITERAL_ALEXA = re.compile(r"(?<!\w)alexa(?!\w)", re.IGNORECASE)

MOONSHINE_REPO = "moonshine-ai/moonshine"
MOONSHINE_REVISION = "48b4e427b587bcf67797a5be706d6ddc4a298149"
MOONSHINE_DIR = Path("A:/hf-cache/moonshine/models--moonshine-ai--moonshine/snapshots/48b4e427b587bcf67797a5be706d6ddc4a298149/onnx/merged/tiny/quantized")
MOONSHINE_FILES = {
    "encoder_model.onnx": (7937661, "c6fc4b7bc5af75c0591fd157a1f3829b533d18e9769a888fd95a62e470dd4f4a"),
    "decoder_model_merged.onnx": (20243286, "eed87831c3a6103534aae7d47a5d485025c659a1323901513961c39fe8a1a367"),
}
MOONSHINE_TOKENIZER = Path(
    r"A:\pisha\memoir\.venv\Lib\site-packages\moonshine_onnx\assets\tokenizer.json"
)
MOONSHINE_TOKENIZER_SHA256 = "b68b995a58b3373db6c4f46864a9db50cd1333d086150ffa2c1096581ede9c10"

WHISPER_REPO = "Systran/faster-whisper-tiny.en"
WHISPER_REVISION = "0d3d19a32d3338f10357c0889762bd8d64bbdeba"
WHISPER_DIR = Path("A:/hf-cache/ct2/models--Systran--faster-whisper-tiny.en/snapshots/0d3d19a32d3338f10357c0889762bd8d64bbdeba")
WHISPER_FILES = {
    "config.json": (2317, "14b1b421a90349bc551b881461426b561a874049cb9e4c4864f2ca384f6a7cc5"),
    "model.bin": (75537502, "1a5afae06a4db91c975c9a9d78be5cc110ee4ea022ad57d55492e4550e936b2a"),
    "tokenizer.json": (2128466, "929c5252409436dce1b38a75d1abbcb5e132d170d8e324e4e04ed915fa2d22df"),
    "vocabulary.txt": (422309, "ff77588746d3a2595d32ab5b69ffd7b95ce2441ac57533cb66fc3eb575a115cf"),
}

ACCEPTED_SOURCES = {"wake", "stt"}
ACCEPTED_OUTCOMES = {"ok", "accepted"}
NEAR_MISS_SOURCES = {"near_miss", "wake_rejected", "miss"}
NEAR_MISS_OUTCOMES = {"near_miss", "rejected", "miss_reported", "wake_rejected"}
TEXT_FIELDS = {
    "intent_query", "heard_query", "resolved_label", "feedback_text",
    "brain_request_text", "rejection_reason", "error", "matched_rule",
}


def exact_alexa(text):
    return isinstance(text, str) and LITERAL_ALEXA.search(text) is not None


def _sha256_path(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bounded_jsonl(path, *, now=None, max_lines=MAX_TELEMETRY_LINES,
                   max_line_bytes=MAX_TELEMETRY_LINE_BYTES):
    now = time.time() if now is None else float(now)
    cutoff = now - MAX_AGE_SECONDS
    tail = deque(maxlen=max_lines)
    stats = {"lines_read": 0, "oversize_lines": 0, "invalid_json": 0, "outside_window": 0}
    with Path(path).open("rb") as source:
        while True:
            raw = source.readline(max_line_bytes + 1)
            if not raw:
                break
            stats["lines_read"] += 1
            oversized = len(raw) > max_line_bytes
            if oversized and not raw.endswith(b"\n"):
                while True:
                    rest = source.readline(max_line_bytes + 1)
                    if not rest or rest.endswith(b"\n"):
                        break
            if oversized:
                stats["oversize_lines"] += 1
                tail.append(None)
            else:
                tail.append(raw)
    events = []
    for raw in tail:
        if raw is None:
            continue
        try:
            event = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError):
            stats["invalid_json"] += 1
            continue
        if not isinstance(event, dict):
            stats["invalid_json"] += 1
            continue
        stamp = event.get("ts")
        if (isinstance(stamp, bool) or not isinstance(stamp, (int, float))
                or not math.isfinite(stamp) or not cutoff <= stamp <= now):
            stats["outside_window"] += 1
            continue
        events.append(event)
    return events, stats


def _logged_text_fields(event):
    for key, value in event.items():
        if isinstance(value, str) and ("transcript" in key or key in TEXT_FIELDS):
            yield key, value


def _accepted_hint(event):
    source = event.get("trigger_source")
    accepted = event.get("outcome") in ACCEPTED_OUTCOMES or event.get("brain_outcome") == "accepted"
    return source in ACCEPTED_SOURCES and accepted and any(exact_alexa(text) for _, text in _logged_text_fields(event))


def _wake_peak(event):
    values = []
    score = event.get("wake_score")
    if isinstance(score, (int, float)) and not isinstance(score, bool) and math.isfinite(score):
        values.append(float(score))
    scores = event.get("wake_scores")
    if isinstance(scores, dict):
        for value in scores.values():
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                values.append(float(value))
    return max(values, default=0.0)


def _near_miss(event):
    marker = (event.get("trigger_source") in NEAR_MISS_SOURCES
              or event.get("outcome") in NEAR_MISS_OUTCOMES
              or event.get("candidate_kind") in {"near_miss", "rejected", "wake_rejected"})
    return marker and _wake_peak(event) >= 0.5


def _source_value(event, *, audio_root=review.AUDIO_ROOT, project_root=review.PROJECT_ROOT):
    processed = event.get("processed_audio_path")
    for field in ("trigger_audio_path", "audio_path"):
        value = event.get(field)
        if isinstance(value, str) and value.strip():
            try:
                path = review.audio_path(value, audio_root, project_root)
                if (isinstance(processed, str) and processed.strip()
                        and path == review.audio_path(processed, audio_root, project_root)):
                    continue
                if path.is_file() and path.stat().st_size <= MAX_WAV_BYTES:
                    identity = review.wav_identity(path)
                    duration = identity["n_frames"] / 16000.0
                    if (identity["sample_rate"] != 16000 or duration <= 0
                            or duration > MAX_CLIP_SECONDS):
                        continue
                    return field, value
            except (OSError, ValueError, TypeError, KeyError, wave.Error, EOFError):
                continue
    return None, None


def _round_robin(events, limit):
    groups = {}
    order = []
    for event in events:
        group = event.get("capture_session_id")
        if not isinstance(group, str) or not group.strip():
            continue
        if group not in groups:
            groups[group] = deque()
            order.append(group)
        groups[group].append(event)
    result = []
    while len(result) < limit:
        advanced = False
        for group in order:
            if groups[group] and len(result) < limit:
                result.append(groups[group].popleft())
                advanced = True
        if not advanced:
            break
    return result


def load_human_history(store_path, *, audio_root=review.AUDIO_ROOT,
                       project_root=review.PROJECT_ROOT):
    path = Path(store_path)
    if not path.exists():
        store = {"schema": review.STORE_SCHEMA, "reviews": []}
        raw_digest = None
        exists = False
    else:
        with path.open("rb") as source:
            data = source.read(MAX_HUMAN_STORE_BYTES + 1)
        if len(data) > MAX_HUMAN_STORE_BYTES:
            raise ValueError("human review store exceeds bounded size")
        store = json.loads(data.decode("utf-8"))
        raw_digest = hashlib.sha256(data).hexdigest()
        exists = True
    metadata, usable, unavailable = review.review_history(store, audio_root, project_root)
    if exists and _sha256_path(path) != raw_digest:
        raise ValueError("human review store changed during preflight")
    return {"store": store, "exists": exists, "sha256": raw_digest,
            "metadata": metadata, "usable": usable, "unavailable": unavailable,
            "reviewed_hashes": {row["wav_sha256"] for row in metadata}}


def assert_human_history_unchanged(before, store_path, *, audio_root=review.AUDIO_ROOT,
                                  project_root=review.PROJECT_ROOT):
    after = load_human_history(store_path, audio_root=audio_root, project_root=project_root)
    if (after["exists"] != before["exists"] or after["sha256"] != before["sha256"]
            or after["reviewed_hashes"] != before["reviewed_hashes"]):
        raise ValueError("human review store/history changed during producer run")
    return after


def _logged_audio_hash(event, field, value):
    expected = event.get("audio_sha256")
    if field == "audio_path" or value == event.get("audio_path"):
        return expected if isinstance(expected, str) and expected else None
    field_hash = event.get(field.replace("_path", "_sha256"))
    return field_hash if isinstance(field_hash, str) and field_hash else None


def _candidate_pool(events):
    accepted, fresh = [], []
    accepted_ids = set()
    for event in sorted(events, key=lambda row: row["ts"], reverse=True):
        if _accepted_hint(event):
            accepted.append({**event, "selection_reason": "accepted_text_hint"})
            accepted_ids.add(event.get("event_id"))
    for event in sorted(events, key=lambda row: row["ts"], reverse=True):
        if event.get("event_id") in accepted_ids:
            continue
        if _near_miss(event):
            fresh.append({**event, "selection_reason": "near_miss_wake_peak"})
    unknown = sum(1 for row in accepted + fresh
                  if not isinstance(row.get("capture_session_id"), str)
                  or not row.get("capture_session_id", "").strip())
    return (_round_robin(accepted, len(accepted)), _round_robin(fresh, len(fresh)), unknown)


def build_selection(*, telemetry=review.TELEMETRY, store_path=DEFAULT_STORE,
                    audio_root=review.AUDIO_ROOT, project_root=review.PROJECT_ROOT,
                    now=None, events=None, history=None, identity_fn=review.wav_identity):
    if history is None:
        history = load_human_history(store_path, audio_root=audio_root, project_root=project_root)
    if events is None:
        events, line_stats = _bounded_jsonl(telemetry, now=now)
    else:
        events = list(events)
        line_stats = {"lines_read": len(events), "oversize_lines": 0, "invalid_json": 0, "outside_window": 0}
    accepted, fresh, unknown_sessions = _candidate_pool(events)
    counters = {"unknown_session": 0, "truncated": 0, "missing_path": 0,
                "missing_or_invalid_wav": 0, "logged_hash_mismatch": 0,
                "human_reviewed_hash": 0, "duplicate_hash": 0, "duration_cap": 0}
    counters["unknown_session"] = unknown_sessions
    selected, seen = [], set()
    total_seconds = 0.0
    selected_accepted = selected_fresh = 0
    for event in accepted + fresh:
        is_accepted = event["selection_reason"] == "accepted_text_hint"
        if (is_accepted and selected_accepted >= MAX_ACCEPTED_EVENTS
                or not is_accepted and selected_fresh >= MAX_FRESH_EVENTS):
            continue
        group = event.get("capture_session_id")
        event_id = event.get("event_id")
        if not isinstance(group, str) or not group.strip():
            continue
        if event.get("recording_truncated") is True or event.get("capture_truncated") is True:
            counters["truncated"] += 1
            continue
        field, value = _source_value(event, audio_root=audio_root, project_root=project_root)
        if not field:
            counters["missing_path"] += 1
            continue
        try:
            path = review.audio_path(value, audio_root, project_root)
            if path.stat().st_size > MAX_WAV_BYTES:
                raise ValueError("WAV exceeds 1 MiB")
            identity = identity_fn(path)
            duration = identity["n_frames"] / 16000.0
            if duration <= 0 or duration > MAX_CLIP_SECONDS:
                raise ValueError("WAV exceeds 24 second bound")
        except (OSError, ValueError, TypeError, KeyError, wave.Error, EOFError):
            counters["missing_or_invalid_wav"] += 1
            continue
        digest = identity["wav_sha256"]
        expected = _logged_audio_hash(event, field, value)
        if expected and expected.casefold() != digest.casefold():
            counters["logged_hash_mismatch"] += 1
            continue
        if digest in history["reviewed_hashes"]:
            counters["human_reviewed_hash"] += 1
            continue
        if digest in seen:
            counters["duplicate_hash"] += 1
            continue
        if total_seconds + duration > MAX_TOTAL_SECONDS:
            counters["duration_cap"] += 1
            continue
        seen.add(digest)
        total_seconds += duration
        selected.append({
            "event_id": event_id, "path": str(path), "wav_sha256": digest,
            "n_frames": identity["n_frames"], "sample_rate": identity["sample_rate"],
            "duration_seconds": duration, "capture_session_id": group,
            "session_id": group, "group_id": group,
            "split": review.session_split(group), "truncated": False,
            "selection_reason": event["selection_reason"],
            "logged_alexa_hint": _accepted_hint(event),
        })
        if is_accepted:
            selected_accepted += 1
        else:
            selected_fresh += 1
        if len(selected) == MAX_RECORDS:
            break
    return {
        "schema": SCHEMA, "mode": "metadata_only", "human_reviewed": False,
        "production_eligible": False, "records": selected,
        "total_original_audio_seconds": total_seconds,
        "selection_limits": {"accepted_hint_events": MAX_ACCEPTED_EVENTS,
                              "near_miss_events": MAX_FRESH_EVENTS,
                              "total_records": MAX_RECORDS,
                              "max_clip_seconds": MAX_CLIP_SECONDS,
                              "max_total_seconds": MAX_TOTAL_SECONDS},
        "selection_counts": {**line_stats, **counters,
                             "candidate_accepted": len(accepted), "candidate_near_miss": len(fresh),
                             "selected": len(selected)},
        "human_review_store": {"exists": history["exists"], "sha256": history["sha256"],
                               "metadata_rows": len(history["metadata"]),
                               "unavailable_audio_rows": len(history["unavailable"])},
        "teacher_evidence": None,
        "limitations": "Metadata preview only; no teacher inference, human review, negative labels, spans, training, or production eligibility.",
    }


def _hash_matches(path, expected_hash, expected_frames):
    identity = review.wav_identity(path)
    if identity["wav_sha256"] != expected_hash or identity["n_frames"] != expected_frames:
        raise ValueError("source WAV changed since selection; refusing consensus label")
    return identity


def _model_files(stage):
    if stage == "moonshine":
        return MOONSHINE_DIR, MOONSHINE_FILES, {
            "repository": MOONSHINE_REPO, "revision": MOONSHINE_REVISION,
            "tokenizer_path": str(MOONSHINE_TOKENIZER),
            "tokenizer_sha256": MOONSHINE_TOKENIZER_SHA256,
        }
    if stage == "whisper":
        return WHISPER_DIR, WHISPER_FILES, {"repository": WHISPER_REPO, "revision": WHISPER_REVISION}
    raise ValueError("unknown teacher stage")


def _validate_model_files(stage):
    directory, files, provenance = _model_files(stage)
    directory = Path(directory).resolve(strict=True)
    hashes = {}
    weight_bytes = 0
    for name, (expected_size, expected_sha) in files.items():
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(f"pinned {stage} file missing: {name}")
        size = path.stat().st_size
        digest = _sha256_path(path)
        if size != expected_size or digest != expected_sha:
            raise ValueError(f"pinned {stage} file mismatch: {name}")
        hashes[name] = {"bytes": size, "sha256": digest}
        if name.endswith((".onnx", ".bin")):
            weight_bytes += size
    if stage == "moonshine":
        tokenizer_path = Path(provenance["tokenizer_path"]).resolve(strict=True)
        if _sha256_path(tokenizer_path) != provenance["tokenizer_sha256"]:
            raise ValueError("packaged Moonshine tokenizer hash mismatch")
        provenance["tokenizer_path"] = str(tokenizer_path)
    provenance.update({"directory": str(directory), "files": hashes,
                       "weight_bytes": weight_bytes})
    return directory, provenance


def _validate_model_unchanged(stage, provenance):
    directory, files, _metadata = _model_files(stage)
    for name, (size, digest) in files.items():
        path = Path(directory) / name
        if path.stat().st_size != size or _sha256_path(path) != digest:
            raise ValueError(f"pinned {stage} file changed during inference: {name}")
    if stage == "moonshine" and _sha256_path(MOONSHINE_TOKENIZER) != MOONSHINE_TOKENIZER_SHA256:
        raise ValueError("packaged Moonshine tokenizer changed during inference")


def _required_free_bytes(weight_bytes):
    return max(MIN_FREE_BYTES, 3 * int(weight_bytes) + MODEL_OVERHEAD_BYTES)


def _read_pcm(path, expected_hash, expected_frames):
    path = Path(path)
    if path.stat().st_size > MAX_WAV_BYTES:
        raise ValueError("source WAV exceeds 1 MiB stage bound")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_hash:
        raise ValueError("source WAV hash changed before inference")
    with wave.open(__import__("io").BytesIO(raw), "rb") as wav:
        if (wav.getnchannels(), wav.getframerate(), wav.getsampwidth(), wav.getcomptype()) != (1, 16000, 2, "NONE"):
            raise ValueError("source WAV format changed before inference")
        frames = wav.getnframes()
        if frames != expected_frames or frames / 16000.0 > MAX_CLIP_SECONDS:
            raise ValueError("source WAV frame count changed before inference")
        pcm_bytes = wav.readframes(frames)
    if len(pcm_bytes) != frames * 2:
        raise ValueError("source WAV is truncated")
    return raw, pcm_bytes, frames


def _clip_text(text):
    value = str(text or "").strip()
    return value[:MAX_TEXT_CHARS], len(value) > MAX_TEXT_CHARS


def _load_moonshine(model_dir):
    import onnxruntime as ort
    from moonshine_onnx.model import MoonshineOnnxModel
    from moonshine_onnx.transcribe import load_tokenizer
    original = ort.InferenceSession
    def cpu_session(path, *args, **kwargs):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        kwargs.pop("sess_options", None)
        kwargs["sess_options"] = options
        kwargs["providers"] = ["CPUExecutionProvider"]
        return original(path, *args, **kwargs)
    ort.InferenceSession = cpu_session
    try:
        model = MoonshineOnnxModel(models_dir=str(model_dir), model_name="tiny")
    finally:
        ort.InferenceSession = original
    return model, load_tokenizer(), {
        "encoder": model.encoder.get_providers(), "decoder": model.decoder.get_providers()}


def _load_whisper(model_dir):
    from faster_whisper import WhisperModel
    model = WhisperModel(str(model_dir), device="cpu", compute_type="int8",
                         cpu_threads=2, num_workers=1, local_files_only=True)
    return model, None, {"whisper": "cpu"}


def _check_memory(stage, model_dir, memory_gate):
    _directory, files, _provenance = _model_files(stage)
    weight_bytes = sum((Path(model_dir) / name).stat().st_size
                       for name in files if name.endswith((".onnx", ".bin")))
    required = _required_free_bytes(weight_bytes)
    available = memory_gate()
    if available < required:
        raise MemoryError(f"{stage} stage requires {required} free physical bytes; found {available}")
    return available, required


def run_teacher_stage(stage, payload, *, memory_gate=proposal_runtime.physical_free_bytes,
                      loader=None, identity_fn=review.wav_identity):
    """Run one isolated teacher stage; ML imports occur only after RAM admission."""
    if stage not in {"moonshine", "whisper"}:
        raise ValueError("unknown teacher stage")
    _validate_stage_payload(payload)
    model_dir, provenance = _validate_model_files(stage)
    free_before, required = _check_memory(stage, model_dir, memory_gate)
    started = time.perf_counter()
    if loader is None:
        model, tokenizer, providers = (_load_moonshine(model_dir) if stage == "moonshine"
                                       else _load_whisper(model_dir))
    else:
        model, tokenizer, providers = loader(stage, model_dir)
    load_elapsed = time.perf_counter() - started
    records = []
    for source in payload["records"]:
        event_id = source["event_id"]
        path = review.audio_path(source["path"], payload["audio_root"], payload["project_root"])
        _hash_matches(path, source["wav_sha256"], source["n_frames"])
        raw, pcm_bytes, frames = _read_pcm(path, source["wav_sha256"], source["n_frames"])
        import numpy as np
        pcm = np.frombuffer(pcm_bytes, dtype=np.int16)
        audio = pcm.astype(np.float32) / np.float32(32768.0)
        clip_started = time.perf_counter()
        if stage == "moonshine":
            tokens = model.generate(audio.reshape(1, -1), max_len=32)
            decoded = tokenizer.decode_batch(tokens)
            text = decoded[0] if decoded else ""
        else:
            segments, _info = model.transcribe(
                audio, language="en", beam_size=1, best_of=1, temperature=0,
                condition_on_previous_text=False, max_new_tokens=32,
                vad_filter=False, word_timestamps=False,
            )
            texts = []
            char_count = 0
            truncated = False
            for segment in segments:
                piece = str(segment.text)
                if char_count >= MAX_TEXT_CHARS:
                    truncated = True
                    break
                texts.append(piece[:MAX_TEXT_CHARS - char_count])
                char_count += len(texts[-1])
                if len(piece) > len(texts[-1]):
                    truncated = True
            text = " ".join(texts).strip()
        text, was_truncated = _clip_text(text)
        was_truncated = was_truncated or (stage == "whisper" and truncated)
        elapsed = time.perf_counter() - clip_started
        identity = identity_fn(path)
        if (identity["wav_sha256"] != source["wav_sha256"]
                or identity["n_frames"] != frames):
            raise ValueError("source WAV changed during inference; stage discarded")
        records.append({"event_id": event_id, "wav_sha256": identity["wav_sha256"],
                        "n_frames": frames, "sample_rate": 16000, "text": text,
                        "exact_alexa": exact_alexa(text), "text_truncated": was_truncated,
                        "source": "offline_same_wav_inference", "elapsed_seconds": elapsed})
        del raw, pcm_bytes, pcm, audio
    _validate_model_unchanged(stage, provenance)
    return {"stage": stage, "records": records, "model": provenance,
            "providers": providers, "memory": {"free_before": free_before,
            "required_free": required}, "model_load_seconds": load_elapsed,
            "inference_seconds": sum(row["elapsed_seconds"] for row in records)}


def _write_private_json(path, value):
    payload = json.dumps(value, ensure_ascii=False, allow_nan=False)
    if len(payload.encode("utf-8")) > MAX_STAGE_INPUT_BYTES:
        raise ValueError("private stage JSON exceeds bounded size")
    with Path(path).open("x", encoding="utf-8") as target:
        target.write(payload)


def _validate_stage_payload(payload):
    records = payload.get("records") if isinstance(payload, dict) else None
    if not isinstance(records, list) or len(records) > MAX_RECORDS:
        raise ValueError("stage payload record count exceeds bound")
    event_ids = set()
    total_frames = 0
    for row in records:
        if not isinstance(row, dict):
            raise ValueError("invalid stage record")
        event_id = row.get("event_id")
        if not isinstance(event_id, str) or not event_id or event_id in event_ids:
            raise ValueError("missing or duplicate stage event id")
        event_ids.add(event_id)
        frames = row.get("n_frames")
        rate = row.get("sample_rate")
        if (isinstance(frames, bool) or not isinstance(frames, int) or frames <= 0
                or rate != 16000 or frames / 16000.0 > MAX_CLIP_SECONDS):
            raise ValueError("stage clip duration exceeds bound")
        declared = row.get("duration_seconds")
        if declared is not None:
            if (isinstance(declared, bool) or not isinstance(declared, (int, float))
                    or not math.isfinite(declared)
                    or abs(float(declared) - frames / 16000.0) > 1 / 16000.0):
                raise ValueError("stage duration metadata disagrees with WAV frames")
        total_frames += frames
    if total_frames / 16000.0 > MAX_TOTAL_SECONDS:
        raise ValueError("stage total duration exceeds bound")


def _run_subprocess_stage(stage, records, temp_dir, project_root, audio_root):
    stage_input = Path(temp_dir) / f"{stage}-input.json"
    stage_output = Path(temp_dir) / f"{stage}-output.json"
    _write_private_json(stage_input, {"records": records, "project_root": str(project_root),
                                      "audio_root": str(audio_root)})
    environment = os.environ.copy()
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    command = [sys.executable, "-m", "assistant.experimental.wake_train.consensus_labels",
               "--_teacher-stage", stage, "--_stage-input", str(stage_input),
               "--_stage-output", str(stage_output)]
    result = subprocess.run(command, capture_output=True, text=True, env=environment, check=False)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "no child diagnostic")[-2000:]
        raise RuntimeError(f"{stage} teacher stage failed with exit {result.returncode}: {detail}")
    if not stage_output.is_file():
        raise RuntimeError(f"{stage} teacher stage exited successfully without a result")
    if stage_output.stat().st_size > MAX_STAGE_INPUT_BYTES:
        raise ValueError("teacher stage output exceeds bounded size")
    data = json.loads(stage_output.read_text(encoding="utf-8"))
    if data.get("stage") != stage:
        raise ValueError(f"{stage} teacher returned a mismatched stage result")
    return data


def run_teacher_stages(records, *, temp_parent, project_root, audio_root, stage_runner=None):
    if stage_runner is not None:
        moonshine = stage_runner("moonshine", records)
        whisper = stage_runner("whisper", records)
        return moonshine, whisper
    with tempfile.TemporaryDirectory(prefix=".wake-consensus-stage-", dir=str(temp_parent)) as directory:
        moonshine = _run_subprocess_stage("moonshine", records, directory, project_root, audio_root)
        whisper = _run_subprocess_stage("whisper", records, directory, project_root, audio_root)
        return moonshine, whisper


def merge_consensus(selection, moonshine, whisper):
    by_stage = []
    for expected_stage, stage in (("moonshine", moonshine), ("whisper", whisper)):
        if stage.get("stage") != expected_stage:
            raise ValueError("teacher stage identity mismatch")
        records = stage.get("records")
        if not isinstance(records, list):
            raise ValueError("teacher result has no records")
        mapping = {}
        for row in records:
            if row["event_id"] in mapping:
                raise ValueError("teacher returned duplicate event ids")
            mapping[row["event_id"]] = row
        if len(mapping) != len(records):
            raise ValueError("teacher returned invalid event coverage")
        by_stage.append(mapping)
    output_records = []
    for source in selection["records"]:
        event_id = source["event_id"]
        evidence = [mapping.get(event_id) for mapping in by_stage]
        if any(row is None for row in evidence):
            raise ValueError("teacher results do not cover selected event")
        if any(row["wav_sha256"] != source["wav_sha256"]
               or row["n_frames"] != source["n_frames"]
               or row.get("source") != "offline_same_wav_inference" for row in evidence):
            raise ValueError("teachers did not infer on the same selected WAV bytes")
        for row in evidence:
            text = row.get("text")
            if not isinstance(text, str) or len(text) > MAX_TEXT_CHARS:
                raise ValueError("teacher transcript exceeds bounded evidence")
            if row.get("exact_alexa") is not exact_alexa(text):
                raise ValueError("teacher exact-token evidence does not match transcript")
        label = "positive" if all(row.get("exact_alexa") is True
            and not row.get("text_truncated") and exact_alexa(row.get("text"))
            for row in evidence) else "unsure"
        output_records.append({**source, "label": label, "label_source": "machine_consensus",
            "human_reviewed": False, "production_eligible": False,
            "evidence": {stage_name: {"family": stage_name, "text": row["text"], "exact_alexa": row["exact_alexa"],
                "text_truncated": row["text_truncated"], "audio_sha256": row["wav_sha256"],
                "source": row["source"], "elapsed_seconds": row["elapsed_seconds"],
                "model": stage_obj["model"]}
                for stage_name, row, stage_obj in (("moonshine", evidence[0], moonshine),
                                                    ("whisper", evidence[1], whisper))}})
    return {"schema": SCHEMA, "mode": "same_wav_consensus", "human_reviewed": False,
            "production_eligible": False, "records": output_records,
            "total_original_audio_seconds": selection["total_original_audio_seconds"],
            "selection_counts": selection["selection_counts"],
            "human_review_store": selection["human_review_store"],
            "teacher_stages": {"moonshine": {k: v for k, v in moonshine.items() if k != "records"},
                               "whisper": {k: v for k, v in whisper.items() if k != "records"}},
            "label_counts": {"positive": sum(r["label"] == "positive" for r in output_records),
                             "unsure": sum(r["label"] == "unsure" for r in output_records)},
            "limitations": "Machine transcript agreement proposes positives only; it does not establish human listening review, negative examples, word boundaries, address intent, training eligibility, or production readiness."}


def _safe_output(path):
    if not path:
        raise ValueError("explicit --output under experimental/wake_train/proposals is required")
    raw = Path(path)
    target = raw.resolve()
    root = OUTPUT_ROOT.resolve()
    if not target.is_relative_to(root) or target.suffix.lower() != ".json":
        raise ValueError("output must be a JSON file under experimental/wake_train/proposals")
    if raw.is_symlink() or os.path.lexists(raw) or target.name.lower() in review.LEGACY_NAMES:
        raise ValueError("output exists, aliases a file, or uses a protected legacy name")
    return target


def _publish(path, data):
    path = _safe_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    review._atomic_json(path, data)
    return path


def produce(*, telemetry=review.TELEMETRY, store_path=DEFAULT_STORE,
            audio_root=review.AUDIO_ROOT, project_root=review.PROJECT_ROOT,
            now=None, events=None, run=False, output=None, stage_runner=None,
            history_loader=load_human_history, identity_fn=review.wav_identity):
    target = _safe_output(output) if output else None
    if run and target is None:
        raise ValueError("--run requires an explicit --output")
    history = history_loader(store_path, audio_root=audio_root, project_root=project_root)
    selection = build_selection(telemetry=telemetry, store_path=store_path,
        audio_root=audio_root, project_root=project_root, now=now, events=events,
        history=history, identity_fn=identity_fn)
    if not run:
        if target is not None:
            _publish(target, selection)
        return selection
    _validate_model_files("moonshine")
    _validate_model_files("whisper")
    target.parent.mkdir(parents=True, exist_ok=True)
    selected = selection["records"]
    moonshine, whisper = run_teacher_stages(selected, temp_parent=target.parent,
        project_root=project_root, audio_root=audio_root, stage_runner=stage_runner)
    _validate_model_files("moonshine")
    _validate_model_files("whisper")
    for row in selected:
        _hash_matches(row["path"], row["wav_sha256"], row["n_frames"])
    assert_human_history_unchanged(history, store_path, audio_root=audio_root, project_root=project_root)
    report = merge_consensus(selection, moonshine, whisper)
    for row in selected:
        _hash_matches(row["path"], row["wav_sha256"], row["n_frames"])
    assert_human_history_unchanged(history, store_path, audio_root=audio_root, project_root=project_root)
    _publish(target, report)
    return report


def _stage_main(stage, input_path, output_path):
    source = Path(input_path)
    if source.stat().st_size > MAX_STAGE_INPUT_BYTES:
        raise ValueError("stage input exceeds bounded size")
    payload = json.loads(source.read_text(encoding="utf-8"))
    result = run_teacher_stage(stage, payload)
    _write_private_json(output_path, result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--telemetry", default=str(review.TELEMETRY))
    parser.add_argument("--audio-root", default=str(review.AUDIO_ROOT))
    parser.add_argument("--project-root", default=str(review.PROJECT_ROOT))
    parser.add_argument("--store", default=str(DEFAULT_STORE))
    parser.add_argument("--output")
    parser.add_argument("--run", action="store_true", help="run the two pinned local teachers serially")
    parser.add_argument("--_teacher-stage", choices=("moonshine", "whisper"), help=argparse.SUPPRESS)
    parser.add_argument("--_stage-input", help=argparse.SUPPRESS)
    parser.add_argument("--_stage-output", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if args._teacher_stage:
            if not args._stage_input or not args._stage_output:
                raise ValueError("internal teacher stage requires private input and output paths")
            _stage_main(args._teacher_stage, args._stage_input, args._stage_output)
            return 0
        report = produce(telemetry=args.telemetry, store_path=args.store,
            audio_root=args.audio_root, project_root=args.project_root,
            run=args.run, output=args.output)
        if not args.output:
            print(json.dumps(report, ensure_ascii=False, allow_nan=False))
        return 0
    except (OSError, ValueError, TypeError, KeyError, MemoryError, RuntimeError, wave.Error) as exc:
        parser.exit(2, f"wake consensus refused: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
