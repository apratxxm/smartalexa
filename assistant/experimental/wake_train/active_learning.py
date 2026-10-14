#!/usr/bin/env python3
"""Human review preparation for archived wake audio, without inference/training.

Default/list prints uncertain JSONL candidates without writing. `label` stores
an explicit listening review, never ASR/model/outcome-derived ground truth.
Positive spans are seconds in original PCM WAV, sample aligned. Negatives
require listening to the entire clip. Session IDs identify independent capture
sessions, never individual events/augmentations. Wake presence describes sound;
it does not establish that the user intended to address the assistant.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import wave

PROJECT_ROOT = Path(__file__).resolve().parents[3]
AUDIO_ROOT = PROJECT_ROOT / "assistant/logs/audio"
TELEMETRY = PROJECT_ROOT / "assistant/logs/telemetry.jsonl"
STORE_SCHEMA = "memoir.wake-human-review.v1"
MANIFEST_SCHEMA = "memoir.wake-reviewed-manifest.v1"
SECTIONS = ("positives", "negatives_short", "negatives_long")
AUDIO_FIELDS = ("audio_path", "trigger_audio_path", "retry_audio_path",
                "follow_up_audio_path", "clarification_audio_path")
LEGACY_NAMES = {"manifest.json", "split_manifest.json", "split_manifest_v1.json", "enroll_split.json"}
PREPARATION_SCHEMA = "memoir.wake-training-preparation.v1"
MAX_PREPARATION_STORE_BYTES = 2 * 1024 * 1024
MAX_PREPARATION_REVIEWS = 100
MAX_PREPARATION_CLIP_SECONDS = 24
MAX_PREPARATION_TOTAL_SECONDS = 600
MAX_PREPARATION_WINDOWS = 8192
MAX_PREPARATION_WAV_BYTES = 1024 * 1024
MAX_RECOVERY_ROWS = 100
MAX_RECOVERY_NOTE_CHARS = 1000
MAX_RECOVERY_REVIEWER_CHARS = 200
MAX_RECOVERY_PROVENANCE_CHARS = 1000
REVIEW_CATEGORIES = ("song_music", "room_sound", "coughing", "people_talking", "other")
MAX_REVIEW_NOTE_CHARS = MAX_RECOVERY_NOTE_CHARS
LEGACY_NEGATIVE_SESSION_PROVENANCE = "unknown_legacy_capture"
LEGACY_NEGATIVE_POOL_ID = "unknown-legacy-capture-negative-pool-v1"
PROPOSAL_LIMITATION = (
    "A classifier/verifier on already proposed windows cannot recover audio excluded "
    "by upstream VAD or wake proposals. This is not base wake-detector retraining or promotion."
)


def audio_path(value, audio_root=AUDIO_ROOT, project_root=PROJECT_ROOT):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("audio path is required")
    root = Path(audio_root).resolve(strict=True)
    candidate = Path(value)
    path = (candidate if candidate.is_absolute() else Path(project_root) / candidate).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file() or path.suffix.lower() != ".wav":
        raise ValueError("WAV must be an existing file within the archival audio root")
    return path


def canonical_audio_path(value, audio_root=AUDIO_ROOT, project_root=PROJECT_ROOT):
    """Resolve an archive path safely without requiring the WAV to exist."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("audio path is required")
    root = Path(audio_root).resolve(strict=True)
    candidate = Path(value)
    source = candidate if candidate.is_absolute() else Path(project_root) / candidate
    path = source.resolve(strict=False)
    if not path.is_relative_to(root) or path.suffix.lower() != ".wav":
        raise ValueError("WAV must be within the archival audio root")
    return path


def wav_identity(path):
    """Bounded-memory hash and actual PCM length validation in one file handle."""
    try:
        with Path(path).open("rb") as stream:
            hasher = hashlib.sha256()
            while block := stream.read(1024 * 1024):
                hasher.update(block)
            digest = hasher.hexdigest()
            stream.seek(0)
            with wave.open(stream, "rb") as wav:
                if (wav.getnchannels(), wav.getframerate(), wav.getsampwidth(), wav.getcomptype()) != (1, 16000, 2, "NONE"):
                    raise ValueError("audio must be mono 16kHz PCM16 WAV")
                frames = wav.getnframes()
                actual = 0
                while chunk := wav.readframes(65536):
                    actual += len(chunk)
                if frames <= 0 or actual != frames * 2:
                    raise ValueError("empty or truncated WAV")
    except (wave.Error, EOFError) as exc:
        raise ValueError("invalid PCM WAV") from exc
    return {"wav_sha256": digest, "sample_rate": 16000, "n_frames": frames}


def iter_candidates(telemetry=TELEMETRY, audio_root=AUDIO_ROOT, project_root=PROJECT_ROOT, *,
                    event_id=None, min_ts=None, max_ts=None, resolve_audio_paths=True):
    if event_id is not None and (not isinstance(event_id, str) or not event_id):
        raise ValueError("event_id filter must be a non-empty string")
    for name, bound in (("min_ts", min_ts), ("max_ts", max_ts)):
        if bound is not None and (isinstance(bound, bool) or not isinstance(bound, (int, float))
                                  or not math.isfinite(bound)):
            raise ValueError(f"{name} filter must be a finite number")
    if min_ts is not None and max_ts is not None and min_ts > max_ts:
        raise ValueError("min_ts must not exceed max_ts")
    if not isinstance(resolve_audio_paths, bool):
        raise ValueError("resolve_audio_paths must be boolean")
    with Path(telemetry).open(encoding="utf-8") as stream:
        for line in stream:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            if event_id is not None and event.get("event_id") != event_id:
                continue
            stamp = event.get("ts")
            if (not isinstance(stamp, (int, float)) or isinstance(stamp, bool)
                    or not math.isfinite(stamp)):
                stamp = None
            if min_ts is not None or max_ts is not None:
                if stamp is None or (min_ts is not None and stamp < min_ts) or (
                        max_ts is not None and stamp > max_ts):
                    continue
            scores = event.get("wake_scores")
            scores = scores if isinstance(scores, dict) else {}
            finite_scores = {str(k): v for k, v in scores.items()
                             if isinstance(v, (float, int)) and not isinstance(v, bool) and math.isfinite(v)}
            score = event.get("wake_score")
            if isinstance(score, (float, int)) and not isinstance(score, bool) and math.isfinite(score):
                finite_scores["wake_score"] = score
            candidate_metadata = {"capture_session_id": event.get("capture_session_id", ""),
                                  "segment_id": event.get("segment_id", ""),
                                  "overlap_of": event.get("overlap_of", ""),
                                  "candidate_kind": event.get("candidate_kind", ""),
                                  "recording_truncated": event.get("recording_truncated", False),
                                  "end_reason": event.get("end_reason", "")}
            if (not any(event.get(field) for field in AUDIO_FIELDS)
                    and (event.get("candidate_kind") or event.get("trigger_source") in {"near_miss", "wake_rejected", "miss"})):
                yield {"event_id": event.get("event_id", ""), "audio_field": "audio_path",
                       "path": "", "reason": event.get("rejection_reason") or event.get("outcome") or "candidate decision",
                       "scores": finite_scores, "proposed_label": "uncertain",
                       "unavailable": event.get("audio_reason") or "audio_not_archived",
                       "event_ts": stamp,
                       "trigger_source": event.get("trigger_source", ""), **candidate_metadata}
            for field in AUDIO_FIELDS:
                value = event.get(field)
                if not value:
                    continue
                if resolve_audio_paths:
                    try:
                        path = str(audio_path(value, audio_root, project_root))
                        unavailable = None
                    except (OSError, ValueError) as exc:
                        path, unavailable = str(value), str(exc)
                else:
                    path, unavailable = str(value), None
                yield {"event_id": event.get("event_id", ""), "audio_field": field,
                       "path": path, "reason": event.get("rejection_reason") or event.get("outcome") or event.get("trigger_source") or "archived audio",
                       "scores": finite_scores, "proposed_label": "uncertain",
                       "unavailable": unavailable, "event_ts": stamp,
                       "trigger_source": event.get("trigger_source", ""), **candidate_metadata}


def print_candidates(*args):
    for candidate in iter_candidates(*args):
        print(json.dumps(candidate, allow_nan=False))


def load_store(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema") != STORE_SCHEMA or not isinstance(data.get("reviews"), list):
        raise ValueError("invalid human review store schema")
    if "retracted_reviews" in data and not isinstance(data["retracted_reviews"], list):
        raise ValueError("invalid retracted review history")
    return data


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} is required")
    return value.strip()


def _span(start, end, frames):
    for value in (start, end):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("wake span bounds must be finite seconds")
        if not math.isclose(value * 16000, round(value * 16000), rel_tol=0, abs_tol=1e-6):
            raise ValueError("wake span must align to PCM samples")
    first, last = round(start * 16000), round(end * 16000)
    if not 0 <= first < last <= frames:
        raise ValueError("wake span must be non-empty and within the WAV")
    return first, last


def validate_review_metadata(row, audio_root=AUDIO_ROOT, project_root=PROJECT_ROOT):
    """Validate human-label metadata and archive identity without reading audio."""
    if not isinstance(row, dict) or row.get("human_reviewed") is not True:
        raise ValueError("explicit human listening review is required")
    result = dict(row)
    for name in ("event_id", "audio_field", "source_provenance", "reviewer", "reviewed_at"):
        result[name] = _text(row.get(name), name)
    session_provenance = row.get("session_provenance")
    legacy_negative = session_provenance == LEGACY_NEGATIVE_SESSION_PROVENANCE
    if legacy_negative:
        if row.get("session_id") is not None:
            raise ValueError("unknown legacy capture mode cannot assert a session_id")
        for name in ("capture_session_id", "segment_id", "overlap_of"):
            if row.get(name, "") not in ("", None):
                raise ValueError("unknown legacy capture mode requires missing capture metadata")
            result[name] = ""
        note = _text(row.get("review_note"), "review_note")
        if len(note) > MAX_RECOVERY_NOTE_CHARS:
            raise ValueError("review_note exceeds recovery note limit")
        result["review_note"] = note
    else:
        if session_provenance is not None:
            raise ValueError("unsupported session provenance")
        result["session_id"] = _text(row.get("session_id"), "session_id")
    if result["audio_field"] not in AUDIO_FIELDS:
        raise ValueError("unknown archival audio field")
    presence = row.get("presence")
    if presence not in ("positive", "negative", "unsure"):
        raise ValueError("presence must be positive, negative or unsure")
    wake_presence = row.get("wake_presence")
    if wake_presence is not None:
        if wake_presence not in ("present", "absent"):
            raise ValueError("wake_presence must be present or absent")
        if wake_presence == "present" and presence != "unsure":
            raise ValueError("presence-only wake annotations must remain unsure")
        if wake_presence == "absent" and presence not in ("negative", "unsure"):
            raise ValueError("absent wake annotations cannot be positive")
    category = row.get("review_category")
    if category is not None and category not in REVIEW_CATEGORIES:
        raise ValueError("review_category is invalid")
    note = row.get("review_note", "")
    if not isinstance(note, str) or len(note) > MAX_REVIEW_NOTE_CHARS:
        raise ValueError("review_note exceeds its text limit")
    if legacy_negative and (presence != "negative" or row.get("entire_clip_negative") is not True):
        raise ValueError("unknown legacy capture mode is limited to confirmed entire-clip negatives")
    if row.get("addressed_assistant") not in ("yes", "no", "unsure"):
        raise ValueError("addressed_assistant must be yes, no or unsure; wake presence alone is not intent")
    path = canonical_audio_path(row.get("path"), audio_root, project_root)
    digest = row.get("wav_sha256")
    if (not isinstance(digest, str) or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)):
        raise ValueError("review WAV hash must be 64 lowercase hexadecimal characters")
    frames = row.get("n_frames")
    if isinstance(frames, bool) or not isinstance(frames, int) or frames <= 0:
        raise ValueError("review WAV frame count must be a positive integer")
    if type(row.get("sample_rate")) is not int or row["sample_rate"] != 16000:
        raise ValueError("WAV frame metadata mismatch")
    if presence == "positive":
        first, last = _span(row.get("wake_start_s"), row.get("wake_end_s"), frames)
        if row.get("entire_clip_negative"):
            raise ValueError("positive review cannot assert entire-clip negative")
        result.update(wake_start_sample=first, wake_end_sample=last)
    else:
        if row.get("wake_start_s") is not None or row.get("wake_end_s") is not None:
            raise ValueError("wake spans are only valid for positive reviews")
        if presence == "negative" and row.get("entire_clip_negative") is not True:
            raise ValueError("negative review requires entire-clip listening confirmation")
    result.update(wav_sha256=digest, n_frames=frames, sample_rate=16000,
                  path=str(path))
    if legacy_negative:
        result.update(session_id=None, split="train", group_id=LEGACY_NEGATIVE_POOL_ID)
    else:
        result.update(split=session_split(result["session_id"]), group_id=result["session_id"])
    return result


def validate_review(row, audio_root=AUDIO_ROOT):
    """Strictly validate metadata and the current WAV identity."""
    result = validate_review_metadata(row, audio_root)
    path = Path(result["path"])
    identity = wav_identity(path)
    if row.get("wav_sha256") != identity["wav_sha256"]:
        raise ValueError("WAV hash mismatch: review is stale")
    if row.get("n_frames") != identity["n_frames"] or row.get("sample_rate") != 16000:
        raise ValueError("WAV frame metadata mismatch")
    result.update(identity, path=str(path))
    return result


def session_split(session_id):
    value = _text(session_id, "session_id")
    bucket = int(hashlib.sha256(("memoir-wake-session-v1:" + value).encode()).hexdigest(), 16) % 100
    return "holdout" if bucket < 30 else "train"


def _deduplicate_reviews(rows):
    result, by_hash, event_groups = [], {}, {}
    for row in rows:
        group_id = row["group_id"]
        previous_group = event_groups.setdefault(row["event_id"], group_id)
        if previous_group != group_id:
            raise ValueError("same event assigned to conflicting sessions")
        siblings = by_hash.setdefault(row["wav_sha256"], [])
        duplicate = False
        for previous in siblings:
            if previous["group_id"] != group_id:
                raise ValueError("duplicated WAV hash assigned to conflicting sessions/groups")
            if row["presence"] == "unsure" or previous["presence"] == "unsure":
                duplicate = duplicate or row["presence"] == previous["presence"]
                continue
            if row["presence"] != previous["presence"]:
                raise ValueError("conflicting positive/negative reviews of duplicated WAV")
            if row["presence"] == "negative":
                duplicate = True
            else:
                bounds = (row["wake_start_sample"], row["wake_end_sample"])
                old_bounds = (previous["wake_start_sample"], previous["wake_end_sample"])
                if bounds == old_bounds:
                    duplicate = True
                elif max(bounds[0], old_bounds[0]) < min(bounds[1], old_bounds[1]):
                    raise ValueError("overlapping reviewed wake spans require reconciliation")
        if not duplicate:
            siblings.append(row)
            result.append(row)
    return result


def metadata_reviews(store, audio_root=AUDIO_ROOT, project_root=PROJECT_ROOT):
    """Deduplicated metadata-valid history; missing WAVs are allowed here only."""
    if not isinstance(store, dict) or store.get("schema") != STORE_SCHEMA or not isinstance(store.get("reviews"), list):
        raise ValueError("invalid human review store schema")
    return _deduplicate_reviews([validate_review_metadata(row, audio_root, project_root)
                                 for row in store["reviews"]])


def review_history(store, audio_root=AUDIO_ROOT, project_root=PROJECT_ROOT):
    """Return metadata history, currently usable rows, and missing-audio rows.

    Only FileNotFoundError is tolerated. Existing but changed or invalid WAVs
    remain a store validation failure.
    """
    metadata = metadata_reviews(store, audio_root, project_root)
    verified = []
    for raw in store["reviews"]:
        try:
            verified.append(validate_review(raw, audio_root))
        except FileNotFoundError:
            continue
    usable = _deduplicate_reviews(verified)
    usable_hashes = {row["wav_sha256"] for row in usable}
    unavailable = [row for row in metadata if row["wav_sha256"] not in usable_hashes]
    return metadata, usable, unavailable


def validated_reviews(store, audio_root=AUDIO_ROOT):
    """Training-facing view: every reviewed WAV must currently verify."""
    if not isinstance(store, dict) or store.get("schema") != STORE_SCHEMA or not isinstance(store.get("reviews"), list):
        raise ValueError("invalid human review store schema")
    return _deduplicate_reviews([validate_review(raw, audio_root) for raw in store["reviews"]])


def readiness(rows):
    groups = {side: {presence: set() for presence in ("positive", "negative")} for side in ("train", "holdout")}
    unknowns = 0
    legacy_pools = Counter()
    for row in rows:
        if row["presence"] == "unsure":
            unknowns += 1
        elif row.get("session_provenance") == LEGACY_NEGATIVE_SESSION_PROVENANCE:
            legacy_pools[row["group_id"]] += 1
        else:
            groups[row["split"]][row["presence"]].add(row["session_id"])
    missing = [f"missing {side} {presence} session groups" for side in groups for presence in groups[side]
               if not groups[side][presence]
               and not (side == "train" and presence == "negative" and legacy_pools)]
    return {"data_ready": not missing, "training_ready": False,
            "blocked_reasons": missing + ["legacy feature extractor must honor reviewed spans and session splits before training"],
            "session_groups": {s: {p: sorted(ids) for p, ids in values.items()} for s, values in groups.items()},
            "train_only_legacy_pools": sorted(legacy_pools),
            "train_only_legacy_pool_counts": dict(sorted(legacy_pools.items())),
            "excluded_unsure": unknowns, "model_trained": False,
            "limits": "Unknown-capture legacy negatives share one fixed TRAIN-only leakage pool; it is not a genuine capture session or holdout group. Session independence is human attested; counts do not establish accuracy or room generalization. " + PROPOSAL_LIMITATION}


def load_preparation_store(path):
    """Bound input before JSON decoding; no ML imports, outputs or lock files."""
    if not Path(path).exists():
        return {"schema": STORE_SCHEMA, "reviews": []}, None
    with Path(path).open("rb") as source:
        payload = source.read(MAX_PREPARATION_STORE_BYTES + 1)
    if len(payload) > MAX_PREPARATION_STORE_BYTES:
        raise ValueError("review store exceeds preparation byte limit")
    store = json.loads(payload)
    return store, hashlib.sha256(payload).hexdigest()


def validate_session_provenance(rows):
    """Reject inconsistent capture and overlapping-segment group assignments."""
    capture_sessions, segments = {}, {}
    for row in rows:
        for key in ("capture_session_id", "segment_id", "overlap_of"):
            value = row.get(key, "")
            if not isinstance(value, str) or len(value) > 128:
                raise ValueError("invalid capture-session/segment provenance")
        capture = row.get("capture_session_id", "")
        if capture and capture_sessions.setdefault(capture, row["group_id"]) != row["group_id"]:
            raise ValueError("same capture session assigned to conflicting human session groups")
        # Associate even an absent/pruned overlap parent. Two children of a
        # missing parent still cannot cross partitions or count independently.
        for key in ("segment_id", "overlap_of"):
            segment = row.get(key, "")
            if segment and segments.setdefault(segment, row["group_id"]) != row["group_id"]:
                raise ValueError("same/overlapping segments assigned to conflicting human session groups")


def prepare_training(store, audio_root=AUDIO_ROOT, *, max_reviews=MAX_PREPARATION_REVIEWS,
                     max_total_seconds=MAX_PREPARATION_TOTAL_SECONDS,
                     max_windows=MAX_PREPARATION_WINDOWS):
    """Read-only, bounded eligibility gate; does not extract features or fit.

    Positive source bounds are the exact human half-open sample interval.
    Negative bounds cover the entire human-reviewed clip. The conservative
    window bound budgets one 80ms hop plus at most six seconds of context per
    reviewed source; it is NOT a temporal embedding alignment or a new label.
    The legacy extractor and trainer remain unsupported and fail closed.
    """
    for value, maximum, name in ((max_reviews, MAX_PREPARATION_REVIEWS, "reviews"),
                                 (max_total_seconds, MAX_PREPARATION_TOTAL_SECONDS, "duration"),
                                 (max_windows, MAX_PREPARATION_WINDOWS, "windows")):
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not 0 < value <= maximum
                or (name != "duration" and not isinstance(value, int))):
            raise ValueError(f"invalid bounded preparation {name} limit")
    if (not isinstance(store, dict) or store.get("schema") != STORE_SCHEMA
            or not isinstance(store.get("reviews"), list)):
        raise ValueError("only an explicit human review store can be prepared")
    raw = store["reviews"]
    if len(raw) > max_reviews:
        raise ValueError("review count exceeds preparation limit; select a smaller explicitly reviewed store")
    # These cheap checks happen before hashing any WAV. Reject oversized raw
    # metadata/files, not just the later deduplicated or known-label subset.
    for row in raw:
        if not isinstance(row, dict):
            raise ValueError("invalid review row")
        frames = row.get("n_frames")
        if (isinstance(frames, bool) or not isinstance(frames, int)
                or not 0 < frames <= MAX_PREPARATION_CLIP_SECONDS * 16000):
            raise ValueError("reviewed clip exceeds preparation duration limit")
        path = audio_path(row.get("path"), audio_root)
        if path.stat().st_size > MAX_PREPARATION_WAV_BYTES:
            raise ValueError("reviewed WAV exceeds preparation byte limit")
    rows = validated_reviews(store, audio_root)
    validate_session_provenance(rows)
    known = [row for row in rows if row["presence"] != "unsure"]
    unique_sources = {row["wav_sha256"]: row["n_frames"] for row in rows}
    total_seconds = sum(unique_sources.values()) / 16000
    planned = []
    for row in known:
        start, end = ((row["wake_start_sample"], row["wake_end_sample"])
                      if row["presence"] == "positive" else (0, row["n_frames"]))
        planned.append({**row, "source_start_sample": start, "source_end_sample": end,
                        "window_upper_bound": (end - start + 6 * 16000 + 1279) // 1280})
    windows = sum(row["window_upper_bound"] for row in planned)
    report = readiness(rows)
    reasons = [reason for reason in report["blocked_reasons"] if reason.startswith("missing ")]
    if total_seconds > max_total_seconds:
        reasons.append("unique reviewed audio exceeds preparation total-duration limit")
    if windows > max_windows:
        reasons.append("conservative feature-window budget exceeds preparation limit")
    return {"schema": PREPARATION_SCHEMA, "eligible_for_feature_preparation": not reasons,
            "training_ready": False, "feature_extraction_started": False,
            "feature_extractor_implemented": False, "training_started": False, "model_trained": False,
            "blocked_reasons": reasons + ["reviewed-span feature extraction and candidate training are not implemented"],
            "readiness": report, "rows": planned, "excluded_unsure": report["excluded_unsure"],
            "unique_audio_seconds": total_seconds, "window_upper_bound": windows,
            "resource_limits": {"max_reviews": max_reviews,
                                "max_clip_seconds": MAX_PREPARATION_CLIP_SECONDS,
                                "max_total_seconds": max_total_seconds, "max_windows": max_windows,
                                "max_feature_bytes_float32": max_windows * 16 * 96 * 4},
            "review_store_sha256": hashlib.sha256(json.dumps(store, sort_keys=True, allow_nan=False).encode()).hexdigest(),
            "limitation": PROPOSAL_LIMITATION}


def build_reviewed_manifest(store, audio_root=AUDIO_ROOT):
    rows = validated_reviews(store, audio_root)
    manifest = {"schema": MANIFEST_SCHEMA, "review_only": True,
                "training_ready": False, **{section: [] for section in SECTIONS}}
    for row in rows:
        if row["presence"] == "unsure":
            continue
        section = "positives" if row["presence"] == "positive" else "negatives_short"
        manifest[section].append({**row, "source": "human_review", "transcript": "alexa" if section == "positives" else ""})
    manifest["readiness"] = readiness(rows)
    manifest["review_store_sha256"] = hashlib.sha256(
        json.dumps(store, sort_keys=True, allow_nan=False).encode("utf-8")).hexdigest()
    return manifest


def _protected_output(path):
    target = Path(path).resolve()
    if target.suffix.lower() != ".json":
        raise ValueError("review stores/manifests must use a .json output")
    if target.name.lower() in LEGACY_NAMES:
        raise ValueError("legacy/default manifests and fixed splits cannot be overwritten")
    return target


def _atomic_json(path, value, *, replace=False):
    target = _protected_output(path)
    if target.exists() and not replace:
        raise ValueError("output exists; choose a new explicit output")
    payload = json.dumps(value, indent=2, allow_nan=False) + "\n"
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=target.parent,
                                         prefix=target.name + ".", suffix=".tmp", delete=False) as stream:
            temp_name = stream.name
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(temp_name, target)
        else:
            os.link(temp_name, target)
    finally:
        if temp_name is not None:
            Path(temp_name).unlink(missing_ok=True)


def write_new_manifest(path, manifest):
    _atomic_json(path, manifest)


def candidate_output(path, production_root=PROJECT_ROOT / "assistant/models/wake"):
    """Require a separate new experiment file; resolve aliases before checking."""
    if not path:
        raise ValueError("explicit candidate --out is required; production export is disabled")
    raw = Path(path)
    target = raw.resolve()
    if target.suffix.lower() != ".onnx":
        raise ValueError("candidate output must be an .onnx file")
    if target.is_relative_to(Path(production_root).resolve()):
        raise ValueError("candidate output cannot be inside the production model directory")
    if raw.is_symlink() or target.exists() or os.path.lexists(raw):
        raise ValueError("candidate output already exists or is an alias; overwrites are disabled")
    if os.path.lexists(target.with_suffix(".meta.json")):
        raise ValueError("candidate metadata already exists; overwrites are disabled")
    return target


def require_supported_training(stage):
    raise ValueError(
        f"{stage} blocked: legacy heuristic labels/features are unaudited, and reviewed "
        "manifest spans/session groups are not wired into this extractor/trainer. "
        "Review data only; no model was trained or promoted.")


def label_candidate(*, telemetry, store_path, event_id, audio_field, presence,
                    session_id, source_provenance, reviewer, confirm_reviewed=False,
                    wake_start=None, wake_end=None, entire_clip_negative=False,
                    addressed_assistant="unsure", wake_presence=None, review_category=None,
                    review_note="", audio_root=AUDIO_ROOT, project_root=PROJECT_ROOT):
    if not confirm_reviewed:
        raise ValueError("--confirm-reviewed requires actual human listening review")
    selected = [c for c in iter_candidates(telemetry, audio_root, project_root,
                                          event_id=event_id)
                if c["event_id"] == event_id and c["audio_field"] == audio_field]
    paths = {c["path"] for c in selected}
    if len(paths) != 1 or any(c["unavailable"] for c in selected):
        raise ValueError("candidate missing, pruned, outside archive, or ambiguous")
    candidate = selected[0]
    identity = wav_identity(candidate["path"])
    row = {**candidate, **identity, "presence": presence, "session_id": session_id,
           "source_provenance": source_provenance, "reviewer": reviewer, "human_reviewed": True,
           "reviewed_at": datetime.now(timezone.utc).isoformat(), "wake_start_s": wake_start,
           "wake_end_s": wake_end, "entire_clip_negative": entire_clip_negative,
           "addressed_assistant": addressed_assistant,
           "wake_presence": wake_presence, "review_category": review_category,
           "review_note": review_note,
           "telemetry_source": str(Path(telemetry).resolve())}
    target = _protected_output(store_path)
    lock = target.with_name(target.name + ".lock")
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise ValueError("review store is locked by another writer") from exc
    try:
        os.close(descriptor)
        store = load_store(target) if target.exists() else {"schema": STORE_SCHEMA, "reviews": []}
        proposed = {**store, "schema": STORE_SCHEMA, "reviews": store["reviews"] + [row]}
        # The newly reviewed candidate was just read and hashed above. Keep
        # unavailable historical rows, but validate all metadata/conflicts.
        validate_review(row, audio_root)
        review_history(proposed, audio_root, project_root)
        _atomic_json(target, proposed, replace=True)
    finally:
        lock.unlink(missing_ok=True)
    return validate_review(row, audio_root)


def retract_latest_review(*, store_path, expected_identity, reason,
                          audio_root=AUDIO_ROOT, project_root=PROJECT_ROOT):
    """Atomically retract only the exact latest reviewed row, preserving its audit copy."""
    if (not isinstance(expected_identity, dict)
            or set(expected_identity) != {"event_id", "audio_field", "reviewed_at", "wav_sha256"}):
        raise ValueError("exact review identity is required")
    for name, value in expected_identity.items():
        if not isinstance(value, str) or not value:
            raise ValueError("exact review identity is invalid")
    if (len(expected_identity["wav_sha256"]) != 64
            or any(char not in "0123456789abcdef" for char in expected_identity["wav_sha256"])):
        raise ValueError("exact review identity is invalid")
    reason = _text(reason, "retraction reason")
    if len(reason) > 200:
        raise ValueError("retraction reason exceeds its limit")
    try:
        expected_time = datetime.fromisoformat(expected_identity["reviewed_at"].replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("exact review identity has an invalid timestamp") from exc
    if expected_time.tzinfo is None:
        raise ValueError("exact review identity timestamp must include a timezone")

    target = _protected_output(store_path)
    lock = target.with_name(target.name + ".lock")
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise ValueError("review store is locked by another writer") from exc
    try:
        os.close(descriptor)
        if not target.exists():
            raise ValueError("review store does not exist")
        store = load_store(target)
        metadata_reviews(store, audio_root, project_root)
        reviews = store["reviews"]
        if not reviews:
            raise ValueError("review store has no active reviews")
        try:
            indexed = [(datetime.fromisoformat(row["reviewed_at"].replace("Z", "+00:00")), i, row)
                       for i, row in enumerate(reviews)]
        except (TypeError, KeyError, ValueError, AttributeError) as exc:
            raise ValueError("review store has an invalid review timestamp") from exc
        if any(stamp.tzinfo is None for stamp, _i, _row in indexed):
            raise ValueError("review store timestamps must include a timezone")
        _stamp, latest_index, latest = max(indexed, key=lambda item: (item[0], item[1]))
        actual_identity = {name: latest.get(name) for name in expected_identity}
        if actual_identity != expected_identity:
            raise ValueError("latest review identity changed; nothing was retracted")
        if latest.get("reviewer") != "local-wake-reviewer":
            raise ValueError("latest review was not created by the local review UI")
        retracted = store.get("retracted_reviews", [])
        if not isinstance(retracted, list):
            raise ValueError("invalid retracted review history")
        audit = {"review": dict(latest),
                 "retracted_at": datetime.now(timezone.utc).isoformat(),
                 "retraction_reason": reason}
        proposed = {**store, "reviews": reviews[:latest_index] + reviews[latest_index + 1:],
                    "retracted_reviews": retracted + [audit]}
        metadata_reviews(proposed, audio_root, project_root)
        _atomic_json(target, proposed, replace=True)
        return {"event_id": latest["event_id"], "audio_field": latest["audio_field"],
                "reviewed_at": latest["reviewed_at"], "presence": latest["presence"],
                "retracted_at": audit["retracted_at"]}
    finally:
        lock.unlink(missing_ok=True)


def recover_confirmed_negatives(feedback, *, telemetry, store_path, reviewer, source_provenance,
                                confirm_reviewed=False, entire_clip_negative=False,
                                addressed_assistant="unsure",
                                audio_root=AUDIO_ROOT, project_root=PROJECT_ROOT):
    """Atomically merge explicitly confirmed legacy negatives into the review store.

    Missing historical capture IDs are represented by one fixed TRAIN-only
    leakage pool. This pool is not a real session or independent holdout.
    Notes identify the human's observation; they never decide the label.
    """
    if confirm_reviewed is not True or entire_clip_negative is not True:
        raise ValueError("recovery requires explicit human review and whole-clip negative confirmation")
    if addressed_assistant not in ("yes", "no", "unsure"):
        raise ValueError("addressed_assistant must be yes, no or unsure")
    if not isinstance(feedback, list) or not 0 < len(feedback) <= MAX_RECOVERY_ROWS:
        raise ValueError("recovery feedback must contain 1 to 100 rows")
    reviewer = _text(reviewer, "reviewer")
    source_provenance = _text(source_provenance, "source_provenance")
    if len(reviewer) > MAX_RECOVERY_REVIEWER_CHARS:
        raise ValueError("reviewer exceeds recovery field limit")
    if len(source_provenance) > MAX_RECOVERY_PROVENANCE_CHARS:
        raise ValueError("source_provenance exceeds recovery field limit")

    prepared, requested, requested_rows = [], set(), {}
    for item in feedback:
        if not isinstance(item, dict) or set(item) != {"event_id", "audio_field", "note"}:
            raise ValueError("recovery rows require exactly event_id, audio_field and note")
        event_id = _text(item.get("event_id"), "event_id")
        audio_field = _text(item.get("audio_field"), "audio_field")
        note = _text(item.get("note"), "review_note")
        if len(note) > MAX_RECOVERY_NOTE_CHARS:
            raise ValueError("review_note exceeds recovery note limit")
        if audio_field not in AUDIO_FIELDS:
            raise ValueError("unknown archival audio field")
        key = (event_id, audio_field)
        if key in requested:
            raise ValueError("duplicate recovery event/audio field")
        requested.add(key)
        requested_rows[key] = note
    candidate_matches = {key: [] for key in requested}
    for candidate in iter_candidates(telemetry, audio_root, project_root):
        key = (candidate["event_id"], candidate["audio_field"])
        if key in candidate_matches and len(candidate_matches[key]) < 2:
            candidate_matches[key].append(candidate)

    for event_id, audio_field in requested:
        key = (event_id, audio_field)
        matches = candidate_matches[key]
        note = requested_rows[key]
        paths = {candidate["path"] for candidate in matches}
        if len(matches) != 1 or len(paths) != 1 or any(candidate.get("unavailable") for candidate in matches):
            raise ValueError("recovery candidate is missing, pruned, outside archive, or ambiguous")
        candidate = matches[0]
        for name in ("capture_session_id", "segment_id", "overlap_of"):
            if candidate.get(name, "") not in ("", None):
                raise ValueError("recovery is limited to candidates missing capture metadata")
        path = audio_path(candidate.get("path"), audio_root, project_root)
        if path.stat().st_size > MAX_PREPARATION_WAV_BYTES:
            raise ValueError("recovery WAV exceeds bounded byte limit")
        identity = wav_identity(path)
        if identity["n_frames"] > MAX_PREPARATION_CLIP_SECONDS * 16000:
            raise ValueError("recovery WAV exceeds bounded duration limit")
        row = {**candidate, **identity, "event_id": event_id, "audio_field": audio_field,
               "path": str(path), "presence": "negative", "session_id": None,
               "session_provenance": LEGACY_NEGATIVE_SESSION_PROVENANCE,
               "review_note": note, "source_provenance": source_provenance,
               "reviewer": reviewer, "human_reviewed": True,
               "reviewed_at": datetime.now(timezone.utc).isoformat(),
               "wake_start_s": None, "wake_end_s": None,
               "entire_clip_negative": True, "addressed_assistant": addressed_assistant,
               "capture_session_id": "", "segment_id": "", "overlap_of": "",
               "telemetry_source": str(Path(telemetry).resolve())}
        prepared.append(validate_review(row, audio_root))

    target = _protected_output(store_path)
    lock = target.with_name(target.name + ".lock")
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise ValueError("review store is locked by another writer") from exc
    try:
        os.close(descriptor)
        store, _ = load_preparation_store(target)
        if not isinstance(store, dict) or store.get("schema") != STORE_SCHEMA or not isinstance(store.get("reviews"), list):
            raise ValueError("invalid human review store schema")
        if "retracted_reviews" in store and not isinstance(store["retracted_reviews"], list):
            raise ValueError("invalid retracted review history")
        if len(store["reviews"]) > MAX_RECOVERY_ROWS:
            raise ValueError("existing review store exceeds recovery row limit")
        existing_raw = metadata_reviews(store, audio_root, project_root)
        to_add = []
        for row in prepared:
            same_event = [old for old in existing_raw
                          if old["event_id"] == row["event_id"] and old["audio_field"] == row["audio_field"]]
            if same_event:
                if all(old["wav_sha256"] == row["wav_sha256"] and old["presence"] == "negative"
                       and old["group_id"] == row["group_id"]
                       for old in same_event):
                    continue
                raise ValueError("existing review conflicts with recovered negative")
            to_add.append(row)
        if len(store["reviews"]) + len(to_add) > MAX_RECOVERY_ROWS:
            raise ValueError("merged review store exceeds recovery row limit")
        merged = {**store, "schema": STORE_SCHEMA, "reviews": store["reviews"] + to_add}
        _metadata_rows, unique_rows, _unavailable_rows = review_history(
            merged, audio_root, project_root)
        if to_add:
            _atomic_json(target, merged, replace=True)
        pool_negatives = [row for row in unique_rows
                          if row["group_id"] == LEGACY_NEGATIVE_POOL_ID and row["presence"] == "negative"]
        report = readiness(unique_rows)
        return {"added": len(to_add), "stored": len(merged["reviews"]),
                "unique_negatives": sum(row["presence"] == "negative" for row in unique_rows),
                "train_only_legacy_pool": LEGACY_NEGATIVE_POOL_ID,
                "train_only_legacy_pool_unique_negatives": len(pool_negatives),
                "readiness": report, "training_started": False, "model_trained": False}
    finally:
        lock.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--telemetry", default=str(TELEMETRY))
    parser.add_argument("--audio-root", default=str(AUDIO_ROOT))
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("list")
    label = commands.add_parser("label")
    label.add_argument("--store", required=True)
    label.add_argument("--event-id", required=True)
    label.add_argument("--audio-field", choices=AUDIO_FIELDS, default="audio_path")
    label.add_argument("--presence", choices=("positive", "negative", "unsure"), required=True)
    label.add_argument("--session-id", required=True)
    label.add_argument("--source-provenance", required=True)
    label.add_argument("--reviewer", required=True)
    label.add_argument("--confirm-reviewed", action="store_true")
    label.add_argument("--wake-start", type=float)
    label.add_argument("--wake-end", type=float)
    label.add_argument("--entire-clip-negative", action="store_true")
    label.add_argument("--addressed-assistant", choices=("yes", "no", "unsure"), default="unsure")
    report = commands.add_parser("report")
    report.add_argument("--store", required=True)
    prepare = commands.add_parser("prepare-training", help="read-only reviewed-data eligibility; no features or fit")
    prepare.add_argument("--store", required=True)
    prepare.add_argument("--max-reviews", type=int, default=MAX_PREPARATION_REVIEWS)
    prepare.add_argument("--max-total-seconds", type=float, default=MAX_PREPARATION_TOTAL_SECONDS)
    prepare.add_argument("--max-windows", type=int, default=MAX_PREPARATION_WINDOWS)
    args = parser.parse_args(argv)
    try:
        if args.command in (None, "list"):
            print_candidates(args.telemetry, args.audio_root, args.project_root)
        elif args.command == "prepare-training":
            store, input_digest = load_preparation_store(args.store)
            prepared = prepare_training(store, args.audio_root, max_reviews=args.max_reviews,
                                        max_total_seconds=args.max_total_seconds, max_windows=args.max_windows)
            prepared["input_file_sha256"] = input_digest
            prepared["human_review_store_exists"] = input_digest is not None
            print(json.dumps(prepared, allow_nan=False))
        elif args.command == "report":
            print(json.dumps(readiness(validated_reviews(load_store(args.store), args.audio_root)), allow_nan=False))
        else:
            row = label_candidate(telemetry=args.telemetry, store_path=args.store,
                                  event_id=args.event_id, audio_field=args.audio_field,
                                  presence=args.presence, session_id=args.session_id,
                                  source_provenance=args.source_provenance, reviewer=args.reviewer,
                                  confirm_reviewed=args.confirm_reviewed, wake_start=args.wake_start,
                                  wake_end=args.wake_end, entire_clip_negative=args.entire_clip_negative,
                                  addressed_assistant=args.addressed_assistant,
                                  audio_root=args.audio_root, project_root=args.project_root)
            print(json.dumps(row, allow_nan=False))
        return 0
    except (OSError, ValueError) as exc:
        parser.exit(2, f"review preparation refused: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
