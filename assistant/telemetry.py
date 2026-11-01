"""Structured event logging.

This is the tuning loop, not an afterthought. Every trigger produces one JSONL
object and (optionally) one archived WAV. After a couple of weeks that corpus is
what decides the open questions in the plan: which STT model, whether the intent
LLM earns its VRAM, what the real false-accept rate is.

The schema is deliberately model-agnostic so the corpus survives swapping the
wake word, STT or intent engine underneath it.
"""
from __future__ import annotations

import json
import hashlib
import logging
import os
import shutil
import re
import time
import uuid
import threading
import wave
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from logging.handlers import RotatingFileHandler
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .config import Config
from .diag import EventFormatter, EventIdFilter, LOG_FORMAT, stage_deltas

log = logging.getLogger("memoir")


_DIAGNOSTIC_STAGES = frozenset(("unrun", "completed", "failed"))


def _diagnostic_bundle(value):
    """Structurally validate and copy selected fields; this proves neither capture
    association nor wake evidence nor ground truth.
    """
    if value is None:
        return None, "unavailable"
    if not isinstance(value, Mapping):
        return None, "invalid"

    def bounded_map(container, allowed, name):
        if not isinstance(container, Mapping):
            raise ValueError(name)
        output = {}
        for key in allowed:
            if key not in container:
                continue
            item = container[key]
            if key in ("stock_score", "personal_score", "stock_threshold",
                       "personal_threshold", "vad_threshold"):
                if (isinstance(item, bool) or not isinstance(item, Real)
                        or not np.isfinite(float(item)) or not 0.0 <= float(item) <= 1.0):
                    raise ValueError(name + "." + key)
                output[key] = float(item)
            elif key in ("personal_hit", "fallback_target_registered",
                         "stock_qualified", "personal_qualified"):
                if not isinstance(item, bool):
                    raise ValueError(name + "." + key)
                output[key] = item
            elif key in ("stock_model", "target_phrase"):
                if not isinstance(item, str) or len(item) > 128:
                    raise ValueError(name + "." + key)
                output[key] = item
            elif key in ("epoch", "start_sample", "end_sample"):
                if isinstance(item, bool) or not isinstance(item, Integral) or item < 0:
                    raise ValueError(name + "." + key)
                output[key] = int(item)
            elif key == "capture_id":
                if not isinstance(item, str) or len(item) > 128:
                    raise ValueError(name + ".capture_id")
                output[key] = item
        return output

    try:
        clean = {}
        nested = {
            "acoustic_proposal": ("stock_model", "stock_score", "personal_hit", "personal_score"),
            "proposal_qualifiers": ("stock_model", "personal_hit", "target_phrase",
                                    "fallback_target_registered", "stock_qualified",
                                    "personal_qualified", "stock_threshold",
                                    "personal_threshold", "vad_threshold"),
            "confirmation_interval": ("epoch", "capture_id", "start_sample", "end_sample"),
        }
        for name, allowed in nested.items():
            if name in value:
                candidate = bounded_map(value[name], allowed, name)
                if name == "confirmation_interval":
                    if (not all(key in candidate for key in allowed)
                            or not candidate["capture_id"]
                            or candidate["end_sample"] <= candidate["start_sample"]):
                        raise ValueError(name)
                clean[name] = candidate
        for name in ("trigger_stage_state", "second_stt_stage_state"):
            if name in value:
                item = value[name]
                if item not in _DIAGNOSTIC_STAGES:
                    raise ValueError(name)
                clean[name] = item
        for name in ("trigger_transcript_truncated", "second_stt_transcript_truncated"):
            if name in value:
                item = value[name]
                if not isinstance(item, bool):
                    raise ValueError(name)
                clean[name] = item
        return clean, "structurally_validated"
    except (TypeError, ValueError, OverflowError):
        return None, "invalid"


_REPORT_BREAK_REASONS = frozenset((
    "read_id_gap", "flush", "callback_overflow", "callback_status",
    "intentional_discard", "echo_generation", "inactive_or_invalid_clean",
    "nonmonotonic_source_time", "flush_after_tail", "pair_append_failure",
))


def _report_coverage(value, raw_size: int):
    """Validate and copy only the paired-report coverage v1 contract."""
    required = {
        "version", "sample_rate", "clock", "window_start", "window_end",
        "diagnostic_generation", "read_start", "read_end_exclusive",
        "retained_samples", "pairing", "sample_alignment",
        "capture_continuity", "padding_possible", "status", "break_reasons",
    }
    optional = {"estimated_source_start", "estimated_source_end",
                "read_completed_start", "read_completed_end"}
    if not isinstance(value, Mapping) or required - set(value) or set(value) - required - optional:
        return None, "invalid_coverage"

    def integer(name):
        item = value[name]
        return isinstance(item, Integral) and not isinstance(item, bool) and item >= 0

    def finite(item):
        return (item is None or (isinstance(item, Real) and not isinstance(item, bool)
                                 and np.isfinite(float(item))))

    try:
        valid = (integer("version") and value["version"] == 1
                 and integer("sample_rate") and value["sample_rate"] == 16000
                 and value["clock"] == "perf_counter"
                 and finite(value["window_start"]) and value["window_start"] is not None
                 and finite(value["window_end"]) and value["window_end"] is not None
                 and value["window_start"] <= value["window_end"]
                 and integer("diagnostic_generation")
                 and integer("retained_samples") and value["retained_samples"] == raw_size
                 and value["pairing"] == "read_invocation"
                 and value["sample_alignment"] == "unverified"
                 and value["capture_continuity"] == "unverified"
                 and value["padding_possible"] is True
                 and value["status"] in {"available", "partial", "missing", "failed"})
        if not valid:
            return None, "invalid_coverage"
        if any(not finite(value[name]) for name in optional if name in value):
            return None, "invalid_coverage"
        for name in ("read_start", "read_end_exclusive"):
            item = value[name]
            if item is not None and not integer(name):
                return None, "invalid_coverage"
        if ((value["read_start"] is None) != (value["read_end_exclusive"] is None)
                or (value["read_start"] is not None
                    and value["read_end_exclusive"] <= value["read_start"])):
            return None, "invalid_coverage"
        if raw_size and value["read_start"] is None:
            return None, "invalid_coverage"
        for first, last in (("estimated_source_start", "estimated_source_end"),
                            ("read_completed_start", "read_completed_end")):
            if (first in value) != (last in value):
                return None, "invalid_coverage"
            if (first in value and ((value[first] is None) != (value[last] is None)
                    or (value[first] is not None and value[last] < value[first]))):
                return None, "invalid_coverage"
        reasons = value["break_reasons"]
        if (not isinstance(reasons, list) or len(reasons) > 12
                or any(not isinstance(code, str) or code not in _REPORT_BREAK_REASONS
                       for code in reasons)):
            return None, "invalid_coverage"
        clean = dict(value)
        clean["break_reasons"] = list(reasons)
        for name in ("version", "sample_rate", "diagnostic_generation",
                     "read_start", "read_end_exclusive", "retained_samples"):
            if clean[name] is not None:
                clean[name] = int(clean[name])
        for name in optional | {"window_start", "window_end"}:
            if clean.get(name) is not None:
                clean[name] = float(clean[name])
        return clean, "structurally_validated"
    except (TypeError, ValueError, OverflowError):
        return None, "invalid_coverage"


@dataclass
class Event:
    """One trigger, from wake word through to Spotify outcome."""

    ts: float = field(default_factory=time.time)
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    trigger_source: str = ""          # wake | wake_rejected | hotkey | miss | conversation_reply
    input_device: str = ""            # laptop | sounddrum
    wake_tier: str = ""               # loose | both
    wake_score: Optional[float] = None
    wake_model: str = ""
    rejection_reason: str = ""
    rejection_count: int = 0

    duck_from: Optional[int] = None
    duck_to: Optional[int] = None
    music_was_playing: Optional[bool] = None
    music_volume_at_trigger: Optional[int] = None

    audio_path: str = ""
    processed_audio_path: str = ""
    capture_session_id: str = ""
    segment_id: str = ""
    overlap_of: str = ""
    candidate_kind: str = ""
    review_state: str = "unknown"
    decision_source: str = ""
    end_reason: str = ""
    capture_end_reason: str = ""
    capture_wait_ms: Optional[int] = None
    capture_truncated: bool = False
    recording_truncated: bool = False
    relative_start_sample: Optional[int] = None
    relative_end_sample: Optional[int] = None
    audio_original_n_frames: Optional[int] = None
    audio_n_frames: Optional[int] = None
    audio_status: str = "not_attempted"
    audio_reason: str = ""
    jsonl_status: str = "pending"
    audio_sha256: str = ""
    processed_audio_status: str = "unavailable"
    processed_audio_reason: str = ""
    processed_audio_sha256: str = ""
    processed_alignment: str = "unavailable"
    vad_peak: Optional[float] = None
    vad_threshold: Optional[float] = None
    voiced_frames: Optional[int] = None
    source_epoch: Optional[int] = None
    capture_id: str = ""
    proposal_id: str = ""
    diagnostic_stage: str = ""
    source_start_sample: Optional[int] = None
    source_end_sample: Optional[int] = None
    source_started_at: Optional[float] = None
    source_completed_at: Optional[float] = None
    diagnostic_only: bool = False
    diagnostic_metadata: Optional[dict] = None
    diagnostic_metadata_status: str = "unavailable"
    pruned_audio_path: str = ""
    trigger_audio_path: str = ""
    follow_up_audio_path: str = ""
    retry_audio_path: str = ""
    retry_transcript: str = ""
    retry_stt_latency_ms: Optional[int] = None
    clarification_audio_path: str = ""
    clarification_transcript: str = ""
    clarification_stt_latency_ms: Optional[int] = None
    recovery_transcript: str = ""
    recovery_hotwords: str = ""
    recovery_stt_latency_ms: Optional[int] = None
    audio_duration_ms: Optional[int] = None
    vad_speech_start_ms: Optional[int] = None
    vad_speech_end_ms: Optional[int] = None

    # ``stt_model`` remains the command recognizer for compatibility. The
    # trigger and command passes are tracked separately because STT listener
    # mode uses small.en to detect the phrase and turbo to understand it.
    stt_model: str = ""
    trigger_stt_model: str = ""
    trigger_stt_device: str = ""
    command_stt_model: str = ""
    stt_latency_ms: Optional[int] = None
    trigger_stt_latency_ms: Optional[int] = None
    trigger_confirmation_mode: str = ""
    trigger_confirmation_latency_ms: Optional[int] = None
    wake_to_ack_ms: Optional[int] = None
    handler_latency_ms: Optional[int] = None
    trigger_transcript: str = ""
    transcript: str = ""
    stt_confidence: Optional[float] = None
    audio_rms_dbfs: Optional[float] = None
    audio_clipped_pct: Optional[float] = None

    intent_action: str = ""
    intent_query: str = ""
    intent_value: Optional[int] = None
    intent_volume_mode: str = "absolute"
    intent_volume_after_play: Optional[int] = None
    intent_latency_ms: Optional[int] = None
    intent_library_only: bool = False
    intent_source: str = ""           # rule | llm | none
    matched_rule: str = ""

    global_action: str = ""
    global_target: str = ""
    global_prompt_chars: Optional[int] = None
    global_model: str = ""
    global_status: str = ""
    global_turn_id: str = ""

    spotify_action: str = ""
    spotify_latency_ms: Optional[int] = None
    spotify_device: str = ""
    token_age_days: Optional[float] = None

    feedback_kind: str = ""           # beep_wake | beep_ok | beep_err | speech
    feedback_text: str = ""
    feedback_latency_ms: Optional[int] = None

    request_phase: str = ""           # trigger | capture | transcribe | intent | execute | complete
    outcome: str = ""                 # ok | false_accept | unknown_intent | error
    error: str = ""

    # Diagnostics. New fields only: older JSONL lines simply lack them.
    # wake_path: acoustic | peek | final_text | second_stt | alias_acoustic |
    # hotkey | paused_mic -- which detector first got the command started.
    wake_path: str = ""
    wake_scores: dict = field(default_factory=dict)   # peak oWW score per model
    woke_early: Optional[bool] = None                 # near-miss: a peek had matched
    second_stt_transcript: str = ""                   # what the command model heard
    # stage -> ms from the first evidence (see diag.STAGES); wake_evidence is 0
    # unless the user was already speaking before the wake word was recognised.
    timeline: dict = field(default_factory=dict)

    heard_query: str = ""             # query before an ASR recovery rewrote intent_query
    resolution_path: str = ""         # track_cache | catalogue | search_fielded | ... (spotify.py)
    resolution_score: Optional[float] = None
    resolution_runner_up: str = ""
    resolution_margin: Optional[float] = None
    resolved_label: str = ""          # "Title by Artist" that was actually played/queued
    playback_method: str = ""         # local_media | web_api | desktop_uri | web_api_verified | queue_api
    radio_count: Optional[int] = None
    queued: Optional[int] = None      # songs added to the queue (None: not a queue command)

    brain_tool: str = ""
    brain_args: str = ""
    brain_latency_ms: Optional[int] = None
    brain_request_status: str = ""
    brain_request_start: Optional[int] = None
    brain_request_end: Optional[int] = None
    brain_request_text: str = ""
    # accepted | rules_fast | validated_away:<why> | unavailable:<error> | ...
    brain_outcome: str = ""

    # Whole-turn conversation routing. Keep summaries to branch/status/count;
    # never record model answers, prompts, or tool arguments here.
    conversation_kind: str = ""
    conversation_status: str = ""
    conversation_latency_ms: Optional[int] = None
    conversation_step_count: Optional[int] = None
    conversation_results_summary: str = ""
    conversation_parent_event_id: str = ""

    def __post_init__(self) -> None:
        # A plain attribute, not a field, so it stays out of the JSONL.
        self._origin = time.perf_counter()

    def set_origin(self, *perf_times: float) -> None:
        """Anchor the timeline at the earliest of these perf_counter() times."""
        times = [t for t in perf_times if isinstance(t, (int, float)) and t > 0]
        if times:
            self._origin = min([self._origin] + times)

    def mark(self, stage: str, at: Optional[float] = None, overwrite: bool = False) -> None:
        """Record when ``stage`` happened, in ms from the origin. First mark wins
        unless ``overwrite`` (for stages that can repeat, e.g. a second STT)."""
        if not overwrite and stage in self.timeline:
            return
        moment = time.perf_counter() if at is None else at
        self.timeline[stage] = int(round((moment - self._origin) * 1000))


NEAR_MISS_DIR = "near_miss"
_GENERATED_WAV = re.compile(r"^\d{8}-\d{6}_[0-9a-f]+(?:-[a-z0-9]+)*\.wav$", re.IGNORECASE)
_REVIEW_STORE_PATH = (Path(__file__).resolve().parent / "experimental" / "wake_train"
                      / "human_reviews.json")
_REVIEW_STORE_MAX_BYTES = 2 * 1024 * 1024
_REVIEW_STORE_MAX_ROWS = 512
_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)
# Minimum seconds between archived clips of each kind.
NEAR_MISS_MIN_GAP_S = {"near_miss": 5.0, "second_stt": 1.0, "acoustic_unconfirmed": 3.0,
                       "below_vad": 30.0, "no_trigger": 30.0, "short_segment": 5.0}


def _fmt(value, spec: str = "") -> str:
    return format(value, spec) if isinstance(value, (int, float)) else "-"


def format_summary(event: Event) -> str:
    """One INFO line per command: ``grep <event_id> memoir.log`` tells the story.

    Other lines logged while the command ran carry the same id via EventIdFilter.
    """
    parts = ["cmd id={} src={}".format(event.event_id, event.trigger_source or "-")]
    if event.trigger_source == "near_miss":
        parts.append("reason={} heard={!r}".format(event.rejection_reason, event.transcript))
        if event.second_stt_transcript:
            parts.append("turbo={!r}".format(event.second_stt_transcript))
        parts.append("woke_early={}".format(event.woke_early))
        parts.append("audio_status={} audio_reason={} index={}".format(
            event.audio_status, event.audio_reason or "-", event.jsonl_status))
        if event.wake_scores:
            parts.append("oww=" + ",".join("{}:{:.2f}".format(m, s)
                                           for m, s in sorted(event.wake_scores.items())))
        if event.audio_path:
            parts.append("audio={}".format(event.audio_path))
        return " ".join(parts)
    if event.wake_path:
        parts.append("wake={}".format(event.wake_path))
    parts.append("mic={}".format(event.input_device or "-"))
    if event.wake_scores:
        parts.append("oww=" + ",".join("{}:{:.2f}".format(m, s)
                                       for m, s in sorted(event.wake_scores.items())))
    parts.append("heard={!r}".format(event.transcript))
    intent = "intent={}({!r})".format(event.intent_action or "-", event.intent_query or "")
    if event.brain_tool:
        intent += "[brain:{}{}]".format(event.brain_tool,
                                        "" if event.brain_outcome in ("", "accepted")
                                        else "/" + event.brain_outcome)
    elif event.intent_source:
        intent += "[{}]".format(event.intent_source)
    parts.append(intent)
    if event.resolution_path or event.resolved_label:
        res = "res={}".format(event.resolution_path or "-")
        if event.resolution_score is not None:
            res += ":{}".format(_fmt(event.resolution_score, ".2f"))
        if event.resolution_runner_up:
            res += " (runner-up {!r} margin {})".format(
                event.resolution_runner_up, _fmt(event.resolution_margin, ".2f"))
        if event.resolved_label:
            res += " -> {!r}".format(event.resolved_label)
        parts.append(res)
    if event.playback_method:
        play = "play={}".format(event.playback_method)
        if event.radio_count:
            play += "(+{} radio)".format(event.radio_count)
        parts.append(play)
    parts.append("outcome={}".format(event.outcome or "-"))
    if event.error:
        parts.append("error={!r}".format(event.error))
    deltas = stage_deltas(event.timeline)
    if deltas:
        parts.append("t[ms]: " + " ".join(
            ("{}={}" if i == 0 else "{}+{}").format(name, delta)
            for i, (name, delta) in enumerate(deltas)))
    return " ".join(parts)


class Telemetry:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.jsonl_path = cfg.path("telemetry.jsonl_path")
        self.audio_dir = cfg.path("telemetry.audio_dir")
        self.archive_audio = cfg.get("telemetry.archive_audio", True)
        self.max_audio_mb = cfg.get("telemetry.max_audio_mb", 2048)
        # Near-miss clips share the cap above; this keeps them from crowding
        # out the command archive.
        self.max_near_miss_mb = cfg.get("telemetry.max_near_miss_mb", 256)
        self._near_miss_at: dict = {}
        self.near_miss_suppressed = 0
        self.sample_rate = cfg.get("audio.sample_rate", 16000)
        self.capture_session_id = uuid.uuid4().hex
        self._archive_lock = threading.RLock()
        self._archive_protected_paths: set[Path] = set()
        self._stats_lock = threading.Lock()
        self._diagnostic_outcomes = {
            "jsonl_write_success": 0,
            "jsonl_write_failure": 0,
            "near_miss_audio_saved": 0,
            "near_miss_audio_rate_limited": 0,
            "near_miss_audio_cap_suppressed": 0,
            "near_miss_audio_other_suppressed": 0,
            "near_miss_audio_unavailable": 0,
            "near_miss_audio_failed": 0,
            "near_miss_audio_disabled": 0,
            "near_miss_audio_other": 0,
            "near_miss_saved_unindexed": 0,
        }
        self.candidate_max_frames = max(1, int(6.0 * self.sample_rate))

        # Telemetry is strictly best effort: a full or read-only disk must not
        # prevent the assistant from starting or handling a command.
        try:
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            self.audio_dir.mkdir(parents=True, exist_ok=True)
            self._setup_text_log(cfg)
        except OSError as exc:
            log.warning("telemetry setup unavailable: %s", exc)

    def _setup_text_log(self, cfg: Config) -> None:
        """Human-readable log alongside the JSONL, for actually reading."""
        path = cfg.path("telemetry.text_log_path")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            log.warning("text telemetry unavailable: %s", exc)
            return
        if any(isinstance(h, RotatingFileHandler) for h in log.handlers):
            return
        handler = RotatingFileHandler(path, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
        # Lines logged while a command is being handled carry its event id.
        handler.setFormatter(EventFormatter(LOG_FORMAT))
        handler.addFilter(EventIdFilter())
        log.addHandler(handler)
        console = logging.StreamHandler()
        console.setFormatter(EventFormatter("%(event_tag)s%(message)s"))
        console.addFilter(EventIdFilter())
        log.addHandler(console)
        log.setLevel(logging.INFO)
        # Our own handlers cover both destinations; propagating to root would
        # duplicate every line via basicConfig's handler.
        log.propagate = False

    def save_audio(self, event: Event, pcm: np.ndarray, subdir: str = "") -> str:
        with self._archive_lock:
            return self._save_audio(event, pcm, subdir)

    def _save_audio(self, event: Event, pcm: np.ndarray, subdir: str = "") -> str:
        """Archive a trigger's audio, including pre-roll.

        Pre-roll matters: a clip that starts at the moment of detection contains
        the command but NOT the wake word that triggered it, which makes it
        useless for wake-word training -- the archive's main purpose.
        """
        event.audio_status, event.audio_reason = "not_attempted", ""
        event.audio_sha256, event.audio_n_frames = "", None
        if not event.capture_session_id:
            event.capture_session_id = self.capture_session_id
        if not self.archive_audio:
            event.audio_status, event.audio_reason = "disabled", "archive_disabled"
            return ""
        if not isinstance(pcm, np.ndarray) or pcm.ndim != 1 or pcm.size == 0:
            event.audio_status, event.audio_reason = "unavailable", "empty_or_invalid_audio"
            return ""
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(event.ts))
        target = self.audio_dir / subdir if subdir else self.audio_dir
        path = target / f"{stamp}_{event.event_id}.wav"
        payload = pcm.astype(np.int16).tobytes()
        created = False
        try:
            if subdir:
                target.mkdir(parents=True, exist_ok=True)
            # WAV has a small header in addition to PCM frames. Reserve archive
            # capacity before opening the file, so a full archive can make room.
            required = len(payload) + 44
            if not self._prune_audio(required, subdir):
                event.audio_status, event.audio_reason = "suppressed", "archive_cap"
                return ""
            # Leave a small margin for filesystem metadata and concurrent logs.
            if shutil.disk_usage(self.audio_dir).free < required + 4096:
                event.audio_reason = "insufficient_disk_space"
                raise OSError("insufficient disk space for telemetry audio")
            # Exclusive creation prevents a repeated event id from overwriting
            # an earlier archive; cleanup below only applies after this succeeds.
            with path.open("xb") as raw:
                created = True
                with wave.open(raw, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(self.sample_rate)
                    wf.writeframes(payload)
            hasher = hashlib.sha256()
            with path.open("rb") as saved:
                while block := saved.read(1024 * 1024):
                    hasher.update(block)
            event.audio_sha256 = hasher.hexdigest()
            event.audio_n_frames = int(pcm.size)
            event.audio_status = "saved"
            return str(path.relative_to(self.jsonl_path.parent.parent.parent))
        except (OSError, ValueError, wave.Error) as exc:
            # Only remove the file allocated for this invocation.
            try:
                if created and path.exists():
                    path.unlink()
            except OSError:
                pass
            log.warning("audio telemetry unavailable: %s", exc)
            event.audio_status = "failed"
            event.audio_reason = event.audio_reason or ("filename_collision" if isinstance(exc, FileExistsError)
                                                       else "disk_full" if getattr(exc, "errno", None) == 28
                                                       else "audio_write_error")
            return ""

    @staticmethod
    def _generated_wavs(directory: Path) -> list:
        """[(mtime, size, path)] of the clips this archive wrote, oldest first.

        One directory scan, so the stat results come from the listing instead
        of a fresh call per file (the archive holds thousands of clips)."""
        found = []
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    if _GENERATED_WAV.match(entry.name) and entry.is_file():
                        info = entry.stat()
                        found.append((info.st_mtime, info.st_size, Path(entry.path)))
        except OSError:
            return []
        found.sort(key=lambda item: item[0])
        return found

    def _reviewed_wav_paths(self) -> tuple:
        """Return reviewed WAV paths and whether the fixed store was readable.

        The store is read only during archive pruning, never in the capture
        frame loop. Bounds keep malformed or unexpectedly large review data
        from turning retention into unbounded work.
        """
        try:
            with _REVIEW_STORE_PATH.open("rb") as source:
                payload = source.read(_REVIEW_STORE_MAX_BYTES + 1)
            if len(payload) > _REVIEW_STORE_MAX_BYTES:
                raise ValueError("review store exceeds byte limit")
            document = json.loads(payload.decode("utf-8"))
            if not isinstance(document, dict) or document.get("schema") != "memoir.wake-human-review.v1":
                raise ValueError("unsupported review store schema")
            rows = document.get("reviews")
            if not isinstance(rows, list) or len(rows) > _REVIEW_STORE_MAX_ROWS:
                raise ValueError("invalid review rows")
            if any(not isinstance(row, dict) for row in rows):
                raise ValueError("malformed review row")
        except (OSError, UnicodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
            log.warning("reviewed audio retention unavailable; pruning disabled: %s", exc)
            return set(), False

        root = self.audio_dir.resolve(strict=False)
        pinned = set()
        for row in rows:
            if (row.get("human_reviewed") is not True
                    or row.get("presence") not in ("positive", "negative")
                    or not isinstance(row.get("wav_sha256"), str)
                    or not _SHA256_HEX.fullmatch(row["wav_sha256"])):
                continue
            raw_path = row.get("path")
            if not isinstance(raw_path, str) or not raw_path:
                continue
            try:
                candidate = Path(raw_path)
                if (not candidate.is_absolute() or candidate.suffix.lower() != ".wav"
                        or not candidate.is_file()):
                    continue
                canonical = candidate.resolve(strict=False)
                # Reject path aliases (including symlinks and dot segments),
                # then require the canonical target to remain under audio_dir.
                if os.path.normcase(os.path.abspath(str(candidate))) != os.path.normcase(str(canonical)):
                    continue
                canonical.relative_to(root)
            except (OSError, RuntimeError, ValueError):
                continue
            pinned.add(canonical)
        return pinned, True

    def _prune_audio(self, required_bytes: int = 0, subdir: str = "",
                     protected_paths=None) -> bool:
        """Rolling, size-capped archive. Oldest out first.

        Sub-directory clips (near misses) count toward the same cap and also
        have a smaller cap of their own."""
        cap = self.max_audio_mb * 1024 * 1024
        if required_bytes > cap:
            return False
        pinned, review_store_valid = self._reviewed_wav_paths()
        protected = set(self._archive_protected_paths)
        if protected_paths:
            protected.update(Path(path).resolve(strict=False) for path in protected_paths)

        def is_pinned(entry) -> bool:
            try:
                path = entry[2].resolve(strict=False)
                return path in pinned or path in protected
            except (OSError, RuntimeError):
                # An ambiguous candidate path is protected from deletion.
                return True

        if subdir:
            own = self._generated_wavs(self.audio_dir / subdir)
            own_total = sum(size for _, size, _ in own)
            own_cap = self.max_near_miss_mb * 1024 * 1024
            if required_bytes > own_cap:
                return False
            while own_total + required_bytes > own_cap:
                victim_entry = next((entry for entry in own if not is_pinned(entry)), None)
                if victim_entry is None or not review_store_valid:
                    return False
                own.remove(victim_entry)
                _, size, victim = victim_entry
                own_total -= size
                victim.unlink(missing_ok=True)
                self._record_pruned(victim, "candidate_archive_cap")
        files = self._generated_wavs(self.audio_dir)
        try:
            with os.scandir(self.audio_dir) as entries:
                children = [Path(e.path) for e in entries if e.is_dir()]
        except OSError:
            children = []
        for child in children:
            files += self._generated_wavs(child)
        files.sort(key=lambda item: item[0])
        total = sum(size for _, size, _ in files)
        while total + required_bytes > cap:
            victim_entry = next((entry for entry in files if not is_pinned(entry)), None)
            if victim_entry is None or not review_store_valid:
                return False
            files.remove(victim_entry)
            _, size, victim = victim_entry
            total -= size
            victim.unlink(missing_ok=True)
            self._record_pruned(victim, "total_archive_cap")
        return total + required_bytes <= cap

    def _record_pruned(self, path: Path, reason: str) -> None:
        try:
            relative = str(path.relative_to(self.jsonl_path.parent.parent.parent))
        except ValueError:
            relative = str(path)
        self.write(Event(trigger_source="archive_pruned", outcome="pruned",
                         capture_session_id=self.capture_session_id,
                         pruned_audio_path=relative, audio_status="pruned", audio_reason=reason))

    def diagnostic_outcome_snapshot(self) -> dict[str, int]:
        """Return fixed-category diagnostic outcome totals without event identifiers."""
        with self._stats_lock:
            return dict(self._diagnostic_outcomes)

    def _count_diagnostic_outcome(self, key: str) -> None:
        with self._stats_lock:
            self._diagnostic_outcomes[key] += 1

    def record_report_diagnostic(self, parent: Event, raw: np.ndarray,
                                 processed: Optional[np.ndarray], *,
                                 coverage: Mapping, parent_indexed: bool) -> Event:
        """Archive one paired diagnostic child without changing its report parent."""
        event = Event(
            trigger_source="miss_diagnostic", outcome="unknown", request_phase="diagnostic",
            capture_session_id=self.capture_session_id, overlap_of=getattr(parent, "event_id", ""),
            review_state="unknown", diagnostic_only=True,
        )
        raw_size = int(raw.size) if isinstance(raw, np.ndarray) else 0
        cleaned, coverage_status = _report_coverage(coverage, raw_size)
        valid_pair = (isinstance(raw, np.ndarray) and raw.ndim == 1 and raw.size > 0
                      and raw.dtype == np.dtype(np.int16)
                      and isinstance(processed, np.ndarray) and processed.ndim == 1
                      and processed.dtype == np.dtype(np.int16)
                      and processed.shape == raw.shape)
        archive_status = "invalid_coverage" if cleaned is None else "invalid_pair" if not valid_pair else "suppressed"
        event.audio_status = "unavailable" if cleaned is None or not valid_pair else "suppressed"
        event.audio_reason = coverage_status if cleaned is None else "invalid_pcm_pair" if not valid_pair else ""
        if valid_pair and raw.size > 480000:
            event.audio_status, event.audio_reason = "unavailable", "pair_too_large"
            archive_status = "oversize_pair"

        parent_path = None
        if valid_pair and cleaned is not None and raw.size <= 480000:
            try:
                root = self.jsonl_path.parent.parent.parent.resolve(strict=False)
                audio_root = self.audio_dir.resolve(strict=False)
                raw_parent_path = Path(parent.audio_path)
                candidate = raw_parent_path if raw_parent_path.is_absolute() else root / raw_parent_path
                resolved = candidate.resolve(strict=True)
                if (parent.audio_status == "saved" and candidate.is_file()
                        and os.path.normcase(os.path.abspath(str(candidate))) == os.path.normcase(str(resolved))
                        and resolved.suffix.lower() == ".wav"
                        and _GENERATED_WAV.match(resolved.name)):
                    resolved.relative_to(audio_root)
                    parent_path = resolved
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                parent_path = None

        if valid_pair and cleaned is not None and raw.size <= 480000:
            if parent_path is None:
                event.audio_status, event.audio_reason = "unavailable", "parent_archive_untrusted"
                archive_status = "parent_archive_unavailable"
            elif not isinstance(parent_indexed, bool) or not parent_indexed:
                event.audio_status, event.audio_reason = "unavailable", "parent_not_indexed"
                archive_status = "parent_not_indexed"
            elif not self.archive_audio:
                event.audio_status, event.audio_reason = "disabled", "archive_disabled"
                archive_status = "disabled"
            else:
                raw_event = Event(ts=event.ts, event_id=event.event_id, capture_session_id=event.capture_session_id)
                clean_event = Event(ts=event.ts, event_id=event.event_id + "-processed",
                                    capture_session_id=event.capture_session_id)
                raw_bytes = int(raw.nbytes) + 44
                clean_bytes = int(processed.nbytes) + 44
                with self._archive_lock:
                    previous_protected = set(self._archive_protected_paths)
                    self._archive_protected_paths.add(parent_path)
                    try:
                        if not self._prune_audio(raw_bytes + clean_bytes):
                            event.audio_status, event.audio_reason = "suppressed", "archive_cap"
                            archive_status = "archive_cap"
                        else:
                            event.audio_path = self.save_audio(raw_event, raw)
                            event.audio_status = raw_event.audio_status
                            event.audio_reason = raw_event.audio_reason
                            event.audio_sha256 = raw_event.audio_sha256
                            event.audio_n_frames = raw_event.audio_n_frames
                            if event.audio_path:
                                raw_path = (root / event.audio_path).resolve(strict=False)
                                self._archive_protected_paths.add(raw_path)
                                event.processed_audio_path = self.save_audio(clean_event, processed)
                                event.processed_audio_status = clean_event.audio_status
                                event.processed_audio_reason = clean_event.audio_reason
                                event.processed_audio_sha256 = clean_event.audio_sha256
                                event.processed_alignment = "unverified"
                                archive_status = ("saved" if event.processed_audio_path else "raw_saved_clean_failed")
                            else:
                                event.processed_audio_status = "unavailable"
                                event.processed_audio_reason = "raw_archive_unavailable"
                                archive_status = "raw_" + event.audio_status
                    except Exception:
                        if event.audio_status != "saved":
                            event.audio_status = "failed"
                            event.audio_reason = "archive_transaction_error"
                        if event.processed_audio_status == "unavailable":
                            event.processed_audio_status = clean_event.audio_status
                            event.processed_audio_reason = clean_event.audio_reason
                        archive_status = "archive_transaction_error"
                        log.warning("paired report diagnostic archive failed", exc_info=True)
                    finally:
                        self._archive_protected_paths.clear()
                        self._archive_protected_paths.update(previous_protected)

        saved_samples = (int(event.audio_n_frames or 0) if event.audio_status == "saved" else 0)
        diagnostic = {
            "coverage": cleaned,
            "archive_status": archive_status,
            "saved_samples": saved_samples,
            "crop_start_sample": 0,
            "crop_end_sample_exclusive": saved_samples,
        }
        event.diagnostic_metadata = diagnostic
        event.diagnostic_metadata_status = coverage_status
        self.write(event)
        return event

    def write(self, event: Event) -> bool:
        """Append one JSONL record and return whether its index was written."""
        if not event.capture_session_id:
            event.capture_session_id = self.capture_session_id
        # The append outcome cannot be truthfully included in the record being
        # appended. Keep it on the returned event and human summary instead.
        event.jsonl_status = "unconfirmed"
        record: dict[str, Any] = asdict(event)
        record.pop("jsonl_status", None)
        indexed = True
        try:
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.jsonl_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:
            indexed = False
            event.jsonl_status = "failed"
            log.warning("json telemetry unavailable: %s", exc)
        else:
            event.jsonl_status = "indexed"
        self._count_diagnostic_outcome("jsonl_write_success" if indexed else "jsonl_write_failure")
        try:
            log.info("%s", format_summary(event))
        except Exception:  # a formatting slip must not cost the JSONL record
            log.debug("could not format command summary", exc_info=True)
        return indexed

    def record_near_miss(self, pcm: np.ndarray, transcript: str, reason: str, *,
                         woke_early: bool = False, scores: Optional[dict] = None,
                         second_stt_transcript: str = "", input_device: str = "laptop",
                         kind: str = "near_miss", now: Optional[float] = None,
                         trigger_stt_model: str = "", trigger_stt_device: str = "",
                         trigger_stt_latency_ms: Optional[int] = None,
                         second_stt_latency_ms: Optional[int] = None,
                         processed_audio: Optional[np.ndarray] = None,
                         segment_id: str = "", overlap_of: str = "", end_reason: str = "",
                         truncated: bool = False, relative_start_sample: int = 0,
                         relative_end_sample: Optional[int] = None,
                         processed_alignment: str = "unavailable", review_state: str = "unknown",
                         sample_rate: int = 16000, vad_peak: Optional[float] = None,
                         vad_threshold: Optional[float] = None, voiced_frames: Optional[int] = None,
                         decision_source: str = "listener", diagnostic_only: bool = False,
                         source_epoch: Optional[int] = None, capture_id: str = "",
                         proposal_id: str = "", diagnostic_stage: str = "",
                         source_start_sample: Optional[int] = None,
                         source_end_sample: Optional[int] = None,
                         source_started_at: Optional[float] = None,
                         source_completed_at: Optional[float] = None,
                         diagnostic_metadata: Optional[Mapping] = None
                         ) -> Event:
        """Keep the audio and a JSONL record of a wake-like utterance we refused.

        These clips are the raw material for wake-word retraining. Rate-limited
        per kind (near misses once per 5 s, turbo rejections once per second --
        each of those already cost a full decode), and written under
        audio/near_miss/, which has its own size cap inside max_audio_mb.
        Every decision is recorded; rate limiting suppresses WAV writes only.
        """
        moment = time.monotonic() if now is None else now
        cleaned_diagnostic, diagnostic_status = _diagnostic_bundle(diagnostic_metadata)
        gap = NEAR_MISS_MIN_GAP_S.get(kind, 5.0)
        key = (input_device, kind)
        last = self._near_miss_at.get(key)
        suppressed = last is not None and moment - last < gap
        if suppressed:
            self.near_miss_suppressed += 1
        else:
            self._near_miss_at[key] = moment
        event = Event(
            trigger_source="near_miss", input_device=input_device,
            outcome="near_miss", request_phase="trigger",
            transcript=transcript, trigger_transcript=transcript,
            second_stt_transcript=second_stt_transcript,
            trigger_confirmation_mode="second_stt" if "second_stt" in reason else "",
            trigger_confirmation_latency_ms=second_stt_latency_ms,
            rejection_reason=reason, rejection_count=1, woke_early=bool(woke_early),
            wake_scores=dict(scores or {}),
            trigger_stt_model=trigger_stt_model, trigger_stt_device=trigger_stt_device,
            trigger_stt_latency_ms=trigger_stt_latency_ms,
            audio_duration_ms=int(pcm.size * 1000 / self.sample_rate) if isinstance(pcm, np.ndarray) else None,
            capture_session_id=self.capture_session_id, segment_id=segment_id,
            overlap_of=overlap_of, candidate_kind=kind, review_state="unknown",
            decision_source=decision_source, end_reason=end_reason,
            recording_truncated=bool(truncated), vad_peak=vad_peak,
            vad_threshold=vad_threshold, voiced_frames=voiced_frames,
            diagnostic_only=bool(diagnostic_only), source_epoch=source_epoch,
            capture_id=capture_id, proposal_id=proposal_id,
            diagnostic_stage=diagnostic_stage,
            source_start_sample=source_start_sample, source_end_sample=source_end_sample,
            source_started_at=source_started_at, source_completed_at=source_completed_at,
            diagnostic_metadata=cleaned_diagnostic,
            diagnostic_metadata_status=diagnostic_status,
        )
        valid = isinstance(pcm, np.ndarray) and pcm.ndim == 1 and pcm.size > 0 and sample_rate == self.sample_rate
        source_supplied = any(value is not None for value in
                              (source_epoch, source_start_sample, source_end_sample,
                               source_started_at, source_completed_at))
        if source_supplied:
            source_valid = (all(isinstance(value, int) and not isinstance(value, bool) and value >= 0
                                for value in (source_epoch, source_start_sample, source_end_sample))
                            and isinstance(source_started_at, (int, float)) and not isinstance(source_started_at, bool)
                            and isinstance(source_completed_at, (int, float)) and not isinstance(source_completed_at, bool))
            if source_valid and valid:
                source_valid = (np.isfinite(source_started_at) and np.isfinite(source_completed_at)
                                and source_completed_at >= source_started_at
                                and source_end_sample - source_start_sample == pcm.size)
            elif source_valid:
                source_valid = False
            if not source_valid:
                valid = False
                event.audio_status, event.audio_reason = "unavailable", "invalid_source_interval_metadata"
        bounds_valid = (isinstance(relative_start_sample, int) and not isinstance(relative_start_sample, bool)
                        and relative_start_sample >= 0
                        and (relative_end_sample is None or (isinstance(relative_end_sample, int)
                             and not isinstance(relative_end_sample, bool))))
        if not bounds_valid:
            valid = False
            event.audio_status, event.audio_reason = "unavailable", "invalid_relative_bounds"
        if valid:
            original_frames = int(pcm.size)
            event.audio_original_n_frames = original_frames
            event.relative_start_sample = relative_start_sample
            event.relative_end_sample = (relative_end_sample if diagnostic_stage else
                                          relative_start_sample + min(original_frames, self.candidate_max_frames))
            if relative_end_sample is not None and relative_end_sample != relative_start_sample + original_frames:
                valid = False
                event.audio_status, event.audio_reason = "unavailable", "invalid_relative_bounds"
        if valid:
            # Deterministic prefix crop, retaining original relative alignment metadata.
            pcm = pcm[:self.candidate_max_frames]
            event.audio_duration_ms = int(pcm.size * 1000 / self.sample_rate)
            if original_frames > pcm.size:
                event.recording_truncated = True
                event.end_reason = (event.end_reason + ":" if event.end_reason else "") + "archive_length_limit"
            aligned = (isinstance(processed_audio, np.ndarray) and processed_audio.ndim == 1
                       and processed_audio.size == original_frames)
            if aligned:
                event.processed_alignment = (processed_alignment
                    if processed_alignment == "sample_aligned" or processed_alignment == "frame_paired"
                    else "frame_paired")
            else:
                event.processed_alignment = "unavailable"
            if processed_audio is not None and not aligned:
                event.processed_audio_reason = "length_or_shape_mismatch"
            values = pcm.astype(np.float64)
            rms = float(np.sqrt(np.mean(values * values)))
            event.audio_rms_dbfs = round(float(20 * np.log10(max(rms, 1.0) / 32768.0)), 2)
            event.audio_clipped_pct = round(float(np.mean(np.abs(values) >= 32767) * 100), 3)
            if suppressed:
                event.audio_status, event.audio_reason = "suppressed", "rate_limited"
                if aligned:
                    event.processed_audio_status, event.processed_audio_reason = "suppressed", "rate_limited"
            else:
                with self._archive_lock:
                    # Reserve BOTH WAVs before writing, so processed allocation cannot prune raw.
                    required = (pcm.size * 2 + 44) * (2 if aligned else 1)
                    if self.archive_audio and not self._prune_audio(required, NEAR_MISS_DIR):
                        event.audio_status, event.audio_reason = "suppressed", "archive_cap"
                        if aligned:
                            event.processed_audio_status, event.processed_audio_reason = "suppressed", "archive_cap"
                    else:
                        event.audio_path = self.save_audio(event, pcm, subdir=NEAR_MISS_DIR)
                        if aligned and event.audio_path:
                            processed_event = Event(ts=event.ts, event_id=event.event_id + "-processed")
                            event.processed_audio_path = self.save_audio(processed_event, processed_audio[:pcm.size], subdir=NEAR_MISS_DIR)
                            event.processed_audio_status = processed_event.audio_status
                            event.processed_audio_reason = processed_event.audio_reason
                            event.processed_audio_sha256 = processed_event.audio_sha256
                        elif aligned:
                            event.processed_audio_status = event.audio_status
                            event.processed_audio_reason = event.audio_reason or "raw_archive_unavailable"
        elif event.audio_status == "not_attempted":
            event.audio_status, event.audio_reason = "unavailable", "empty_or_invalid_audio"
        indexed = self.write(event)
        outcome_key = {
            "saved": "near_miss_audio_saved",
            "unavailable": "near_miss_audio_unavailable",
            "failed": "near_miss_audio_failed",
            "disabled": "near_miss_audio_disabled",
        }.get(event.audio_status)
        if event.audio_status == "suppressed":
            outcome_key = {
                "rate_limited": "near_miss_audio_rate_limited",
                "archive_cap": "near_miss_audio_cap_suppressed",
            }.get(event.audio_reason, "near_miss_audio_other_suppressed")
        self._count_diagnostic_outcome(outcome_key or "near_miss_audio_other")
        if not indexed and event.audio_status == "saved":
            # The WAV is intentionally retained; the returned event identifies
            # its path while this aggregate records that it lacks a JSONL index.
            self._count_diagnostic_outcome("near_miss_saved_unindexed")
        return event
