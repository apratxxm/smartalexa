"""Experimental event-driven STT frontend over a caller-owned ASR lane.

This adapter deliberately has no recognizer lifecycle, device reader, callback,
or application wiring. The caller owns the lane worker and polls immutable
events on the same thread that feeds audio. An acoustic proposal reserves a
bounded retained interval for speech-only confirmation; the proposal itself
never authorizes a wake or command.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
import re
import threading
import time
import uuid
from numbers import Real
from typing import Any, Callable, Mapping, Optional

import numpy as np

from .audio_jobs import AudioSnapshot, DecodeJob, DecodeResult
from .listener import SttListener, Utterance


_RATE = 16000
_FRAME = 512
_FRAME_SECONDS = _FRAME / _RATE
_MAX_EVENTS = 16
_MAX_SEALED = 2
_MAX_AUTH_ENTRIES = 64
_MAX_AUTH_KEY = 128
_JOB_DEADLINE_SECONDS = 3.0
_MAX_HISTORY_FRAMES = (30 * _RATE) // _FRAME
_MAX_DIAGNOSTIC_EVENTS = 8


@dataclass(frozen=True)
class AcousticProposal:
    """Wake-model scores are a bounded request for speech confirmation only."""

    stock_model: str
    stock_score: float
    personal_hit: bool
    personal_score: float

    def __post_init__(self) -> None:
        if not isinstance(self.stock_model, str) or not self.stock_model \
                or len(self.stock_model) > 128:
            raise ValueError("stock_model must be a bounded non-empty string")
        for name in ("stock_score", "personal_score"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not math.isfinite(value) or not 0.0 <= float(value) <= 1.0):
                raise ValueError("{} must be a finite score from zero to one".format(name))
            object.__setattr__(self, name, float(value))
        if not isinstance(self.personal_hit, bool):
            raise ValueError("personal_hit must be a bool")


@dataclass(frozen=True)
class ListenerEvent:
    """Immutable event envelope carrying an independently owned Utterance.

    The legacy-compatible Utterance object is mutable, but its PCM arrays are
    copied and it has no references back into listener state or lane snapshots.
    """

    kind: str
    reason: str = ""
    utterance: Optional[Utterance] = None
    segment_id: str = ""
    capture_id: str = ""
    phrase: str = ""
    wake_path: str = ""
    evidence_at: float = 0.0
    diagnostic_snapshot: Optional[AudioSnapshot] = None
    diagnostic_stage: str = ""
    proposal_id: str = ""
    trigger_transcript: Optional[str] = None
    second_stt_transcript: Optional[str] = None
    trigger_stt_latency_ms: Optional[int] = None
    second_stt_latency_ms: Optional[int] = None


@dataclass(frozen=True)
class _FrameRecord:
    raw: np.ndarray
    processed: Optional[np.ndarray]
    start_sample: int
    captured_at: float
    probability: float
    authorization: Mapping[str, bool]

    @property
    def end_sample(self) -> int:
        return self.start_sample + _FRAME


@dataclass
class _Segment:
    segment_id: str
    records: list[_FrameRecord]
    speech_started_at: float
    voiced_frames: int = 1
    vad_peak: float = 0.0
    silence_run: int = 0
    next_peek: int = 0
    revision: int = 0
    overlap_of: str = ""
    rejections: list[str] = field(default_factory=list)
    wake_phrase: str = ""
    wake_path: str = ""
    wake_evidence_at: float = 0.0
    wake_emitted: bool = False
    wake_mid: bool = False
    confirmed_audio_handoff: bool = False
    reserved_events: int = 2
    extending: bool = False
    extension_frames: int = 0
    last_peek_result_revision: int = 0
    provisional: bool = False
    live_frames: int = 0
    live_voiced_frames: int = 0
    acoustic_receipt_epoch: Optional[int] = None
    acoustic_receipt_start: Optional[int] = None
    acoustic_receipt_end: Optional[int] = None
    acoustic_receipt_phrase: str = ""


@dataclass
class _Boundary:
    segment: _Segment
    job_id: str
    deadline: float
    continuation: list[_FrameRecord] = field(default_factory=list)
    silence_run: int = 0
    silence_end: Optional[int] = None
    snapshot_end: int = 0
    deferred_pending: Optional["_Pending"] = None
    deferred_result: Optional[DecodeResult] = None


@dataclass
class _Pending:
    job: DecodeJob
    segment: _Segment
    purpose: str
    end_reason: str = ""
    truncated: bool = False
    original_transcript: str = ""
    original_latency_ms: int = 0
    proposed_phrase: str = ""
    proposed_mid: bool = False
    boundary: Optional[_Boundary] = None


@dataclass
class _AcousticReservation:
    proposal_id: str
    epoch: int
    admitted_at: float
    interval_start: int
    interval_end: int
    target: _Segment
    snapshot: AudioSnapshot
    proposal: AcousticProposal
    target_phrase: str
    deadline: float
    job: Optional[DecodeJob] = None
    transcript: str = ""
    latency_ms: int = 0
    verify_job: Optional[DecodeJob] = None
    resolved: bool = False
    target_boundary: Optional[_Boundary] = None
    trigger_transcript: Optional[str] = None
    trigger_latency_ms: Optional[int] = None
    trigger_stage_state: str = "unrun"
    second_transcript: Optional[str] = None
    second_latency_ms: Optional[int] = None
    second_stage_state: str = "unrun"
    trigger_transcript_truncated: bool = False
    second_transcript_truncated: bool = False
    stock_threshold: float = 0.75
    personal_threshold: float = 0.5
    vad_threshold: float = 0.35
    fallback_target_registered: bool = False


class AsyncSttListener:
    """A single-owner segmentation frontend backed by ``SerializedAsrLane``.

    ``lane.start()`` belongs to the caller. ``feed_async``, result
    application, polling, and lifecycle changes must all occur on one owner
    thread. VAD runs exactly once per externally supplied frame; replayed
    continuation records reuse their cached probabilities.
    """

    def __init__(self, cfg, vad, lane, *, clock=time.monotonic,
                 listener_recognizer="listener", verify_recognizer="command",
                 verify_text=None):
        if not callable(clock):
            raise ValueError("clock must be callable")
        if not callable(vad):
            raise ValueError("vad must be callable")
        if lane is None or not callable(getattr(lane, "submit", None)):
            raise ValueError("lane must provide submit()")
        for value, label in ((listener_recognizer, "listener_recognizer"),
                             (verify_recognizer, "verify_recognizer")):
            if not isinstance(value, str) or not value:
                raise ValueError("{} must be a non-empty string".format(label))
        if verify_text is not None and not callable(verify_text):
            raise ValueError("verify_text must be callable or None")
        self.cfg = cfg
        self.vad = vad
        self.lane = lane
        self.clock = clock
        self.listener_recognizer = listener_recognizer
        self.verify_recognizer = verify_recognizer
        self.verify_text = verify_text
        get = getattr(cfg, "get", lambda _key, default=None: default)

        frame_ms = _FRAME_SECONDS * 1000.0
        def duration_frames(key, default, *, minimum=0, seconds=False):
            raw = get(key, default)
            if (isinstance(raw, bool) or not isinstance(raw, Real)
                    or not math.isfinite(raw) or raw < 0):
                raise ValueError("{} must be a finite nonnegative duration".format(key))
            milliseconds = float(raw) * (1000.0 if seconds else 1.0)
            return max(minimum, int(milliseconds / frame_ms))

        self.vad_threshold = float(get("listener.vad_threshold",
                                       get("capture.vad_threshold", 0.35)))
        self.min_frames = duration_frames("listener.min_speech_ms", 300)
        self.silence_frames = duration_frames(
            "listener.silence_ms", get("capture.silence_ms", 800), minimum=1)
        self.max_frames = duration_frames(
            "listener.max_utterance_s", 6.0, minimum=1, seconds=True)
        self.pad_frames = duration_frames("listener.pad_ms", 300)
        self.overlap_frames = duration_frames("listener.overlap_ms", 1500)
        self.extend_frames = duration_frames("listener.extend_ms", 4000)
        self.peek_first = duration_frames("listener.peek_first_ms", 900)
        self.peek_every = duration_frames("listener.peek_every_ms", 1000, minimum=1)
        self.peek_until = duration_frames("listener.peek_until_ms", 10000)
        self.peek_window = duration_frames("listener.peek_window_ms", 1600, minimum=1)
        confirmation_seconds = get("wake.confirm_seconds", 1.6)
        if (isinstance(confirmation_seconds, bool)
                or not isinstance(confirmation_seconds, Real)
                or not math.isfinite(confirmation_seconds)
                or confirmation_seconds <= 0 or confirmation_seconds > 30):
            raise ValueError("wake.confirm_seconds must be finite and from zero to 30 seconds")
        self.confirmation_frames = max(
            1, int(math.ceil(float(confirmation_seconds) / _FRAME_SECONDS)))
        self.acoustic_threshold = float(get("wake.acoustic_activation_threshold", 0.75))
        self.personal_threshold = float(get("wake.personal_threshold", 0.5))
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0
               for value in (self.acoustic_threshold, self.personal_threshold)):
            raise ValueError("acoustic wake thresholds must be finite scores")
        self.weak_enabled = bool(get("telemetry.wake_candidates_enabled", False))
        self.weak_floor = float(get("telemetry.wake_candidate_vad_threshold", 0.08))
        if not math.isfinite(self.weak_floor) or not 0.0 <= self.weak_floor <= 1.0:
            raise ValueError("telemetry.wake_candidate_vad_threshold must be a score")
        self.weak_min_frames = 3
        self.weak_silence_frames = 11
        self.weak_max_frames = 187
        self.weak_pre_frames = 9
        max_snapshot_frames = (30 * _RATE) // _FRAME
        if (self.max_frames + self.extend_frames + self.pad_frames > max_snapshot_frames
                or self.peek_window > max_snapshot_frames
                or self.overlap_frames > max_snapshot_frames):
            raise ValueError("listener duration settings exceed the bounded audio snapshot")

        # Copy only the pure matching policy from the existing listener.
        self.phrases = [str(p).lower() for p in get(
            "listener.phrases", ["memoir", "jarvis", "alexa"])]
        self.aliases = {str(a).lower(): str(t).lower() for a, t in
                        (get("listener.aliases", {}) or {}).items()}
        self.fuzzy_cutoff = float(get("listener.fuzzy_cutoff", 0.72))
        self.allow_fuzzy = bool(get("listener.allow_fuzzy", False))
        self.mid_phrases = [str(p).lower() for p in get(
            "listener.mid_phrases", ["alexa", "jarvis", "memoir"])]
        self.mid_fuzzy_cutoff = float(get("listener.mid_fuzzy_cutoff", 0.9))

        self._owner_thread_id: Optional[int] = None
        self._stream_epoch = 0
        self._sample_counter = 0
        self._last_captured_at = float("-inf")
        self._pre: deque[_FrameRecord] = deque(maxlen=self.pad_frames or None)
        self._history: deque[_FrameRecord] = deque(maxlen=self.confirmation_frames)
        self._active: Optional[_Segment] = None
        self._boundary: Optional[_Boundary] = None
        self._pending: dict[str, _Pending] = {}
        self._acoustic_pending: dict[str, tuple[DecodeJob, _AcousticReservation, str]] = {}
        self._reservation: Optional[_AcousticReservation] = None
        self._held_terminal: dict[str, tuple[_Segment, str, Optional[_Pending], Optional[DecodeResult]]] = {}
        self._events: deque[ListenerEvent] = deque()
        self._reserved_events = 0
        self._diagnostic_events = 0
        self._diagnostic_drops = 0
        self._weak_records: list[_FrameRecord] = []
        self._weak_pre: deque[_FrameRecord] = deque(maxlen=self.weak_pre_frames)
        self._weak_voiced = 0
        self._weak_silence = 0
        self._weak_peak = 0.0
        self._replay: deque[_FrameRecord] = deque()
        self._job_counter = 0
        self._closed = False
        self._suspended = False

    @property
    def stream_epoch(self) -> int:
        return self._stream_epoch

    def _claim_owner(self) -> None:
        ident = threading.get_ident()
        if self._owner_thread_id is None:
            self._owner_thread_id = ident
        elif self._owner_thread_id != ident:
            raise RuntimeError("AsyncSttListener is owned by another thread")

    def _now(self) -> float:
        value = self.clock()
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError("clock must return a finite monotonic value")
        return float(value)

    @staticmethod
    def _validate_authorization(value) -> dict[str, bool]:
        if value is None:
            return {}
        if not isinstance(value, Mapping) or len(value) > _MAX_AUTH_ENTRIES:
            raise ValueError("authorization must be a bounded string-key mapping")
        result = {}
        for key, allowed in value.items():
            if (not isinstance(key, str) or not key or len(key) > _MAX_AUTH_KEY
                    or not isinstance(allowed, bool)):
                raise ValueError("authorization entries must be bounded string/bool pairs")
            result[key] = allowed
        return result

    def _matcher(self, authorization: Mapping[str, bool], duration_ms: int):
        matcher = SttListener.__new__(SttListener)
        matcher.phrases = list(self.phrases)
        matcher.aliases = dict(self.aliases)
        matcher.fuzzy_cutoff = self.fuzzy_cutoff
        matcher.mid_fuzzy_cutoff = self.mid_fuzzy_cutoff
        matcher.allow_fuzzy = self.allow_fuzzy
        matcher.mid_phrases = list(self.mid_phrases)
        matcher._authorize_trigger = lambda source, phrase: bool(
            authorization.get("{}:{}".format(source, phrase), False))
        matcher._on_reject = None
        matcher._pending_rejections = []
        matcher._current_duration_ms = duration_ms
        return matcher

    def _has_capacity(self, needed: int = 1) -> bool:
        return len(self._events) + self._reserved_events + needed <= _MAX_EVENTS

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("AsyncSttListener is closed")

    def feed_async(self, frame, processed_frame=None, *, captured_at=None,
                   authorization=None, acoustic_proposal=None) -> None:
        """Consume one raw 512-sample frame without decoding or callbacks."""
        self._claim_owner()
        self._ensure_open()
        if self._suspended:
            raise RuntimeError("AsyncSttListener is suspended")
        self._expire_acoustic_reservation()
        self._drain_replay()
        if self._replay:
            raise BufferError("continuation replay is backpressured; poll events before feeding")
        self._validate_frame(frame, "frame")
        if processed_frame is not None:
            self._validate_frame(processed_frame, "processed_frame")
        if acoustic_proposal is not None and not isinstance(acoustic_proposal, AcousticProposal):
            raise ValueError("acoustic_proposal must be an AcousticProposal or None")
        auth = self._validate_authorization(authorization)
        when = self._now() if captured_at is None else self._finite_time(captured_at)
        if when < self._last_captured_at:
            raise ValueError("captured_at must be monotonic nondecreasing")
        if self._active is None and self._boundary is None and not self._has_capacity(2):
            raise BufferError("event queue is full; poll events before feeding")
        vad_input = processed_frame if processed_frame is not None else frame
        probability = self.vad(vad_input)  # exactly one VAD call for this external frame
        if (isinstance(probability, bool) or not isinstance(probability, Real)
                or not math.isfinite(probability) or not 0.0 <= float(probability) <= 1.0):
            raise ValueError("VAD must return a finite probability from zero to one")
        probability = float(probability)
        record = _FrameRecord(np.array(frame, dtype=np.int16, copy=True),
                              None if processed_frame is None else np.array(
                                  processed_frame, dtype=np.int16, copy=True),
                              self._sample_counter, when, probability, auth)
        self._sample_counter += _FRAME
        self._last_captured_at = when
        self._history.append(record)
        provisional = False
        if acoustic_proposal is not None:
            provisional = self._admit_acoustic_proposal(acoustic_proposal, record)
        primary_speech = (record.probability >= self.vad_threshold
                          or self._active is not None or self._boundary is not None)
        self._feed_weak_candidate(record, primary_speech)
        if provisional:
            pass
        elif self._boundary is not None:
            self._feed_boundary(record)
        else:
            self._consume_record(record, replay=False)

    def _admit_acoustic_proposal(self, proposal: AcousticProposal,
                                 record: _FrameRecord) -> bool:
        """Reserve the current capture interval before segmentation can seal it."""
        now = self._now()
        if self._boundary is not None and now >= self._boundary.deadline:
            self._abort_boundary("boundary_deadline_exceeded")
            return False
        current = self._reservation
        if current is not None:
            target = (self._active or
                      (self._boundary.segment if self._boundary is not None else None))
            if target is current.target:
                # Coalescing is diagnostic only: it cannot extend the deadline.
                self._diagnostic_event("acoustic_proposal_coalesced", target,
                                       proposal_id=current.proposal_id, stage="admission")
            else:
                self._diagnostic_event("acoustic_proposal_busy", target,
                                       proposal_id=uuid.uuid4().hex, stage="admission")
            return False
        qualified = ((proposal.personal_hit
                      and proposal.personal_score >= self.personal_threshold)
                     or proposal.stock_score >= self.acoustic_threshold)
        frames = list(self._history)
        if not qualified:
            self._diagnostic_for_records("acoustic_weak_proposal", frames, proposal,
                                         stage="admission")
            return False
        target_phrase = ("alexa" if proposal.personal_hit else
                         proposal.stock_model.replace("_", " ").lower())
        target = self._active or (self._boundary.segment if self._boundary else None)
        if target is not None and target.wake_emitted:
            self._diagnostic_event("acoustic_superseded_by_text", target,
                                   proposal_id=uuid.uuid4().hex, stage="admission")
            return False
        provisional = False
        if target is None:
            # The current input record is already in history. Keep its prefix once
            # and start endpoint silence accounting at this proposal frame.
            target = _Segment(uuid.uuid4().hex[:16], list(frames),
                              speech_started_at=record.captured_at,
                              voiced_frames=sum(item.probability >= self.vad_threshold
                                                for item in frames),
                              vad_peak=max((item.probability for item in frames), default=0.0),
                              silence_run=0 if record.probability >= self.vad_threshold else 1,
                              next_peek=self.peek_first, provisional=True,
                              live_frames=1,
                              live_voiced_frames=int(record.probability >= self.vad_threshold))
            if not self._reserve_for_segment(target):
                self._diagnostic_for_records("acoustic_event_backpressure", frames, proposal,
                                             stage="admission")
                return False
            self._active = target
            provisional = True
        # A proposal arriving after primary speech began may need retained
        # preroll that predates the active list. Prepend it once and keep the
        # actual confirmation window exactly bounded by the ring.
        if not provisional and target.records and frames:
            first_active = target.records[0].start_sample
            missing = [item for item in frames if item.end_sample <= first_active]
            if missing:
                known = {item.start_sample for item in target.records}
                missing = [item for item in missing if item.start_sample not in known]
                if missing:
                    target.records[0:0] = missing
                    target.voiced_frames += sum(
                        item.probability >= self.vad_threshold for item in missing)
                    target.vad_peak = max(target.vad_peak,
                                          max(item.probability for item in missing))
        records = list(frames)
        if not records:
            return provisional
        try:
            snapshot = self._snapshot(target, records, end_reason="acoustic_confirmation",
                                      truncated=False, authorization=record.authorization)
        except (ValueError, TypeError):
            self._diagnostic_for_records("acoustic_snapshot_failed", frames, proposal,
                                         stage="admission")
            return provisional
        proposal_id = uuid.uuid4().hex
        reservation = _AcousticReservation(
            proposal_id, self._stream_epoch, now, snapshot.start_sample,
            snapshot.end_sample, target, snapshot, proposal, target_phrase,
            now + _JOB_DEADLINE_SECONDS,
            target_boundary=(self._boundary if self._boundary is not None
                             and self._boundary.segment is target else None),
            stock_threshold=self.acoustic_threshold,
            personal_threshold=self.personal_threshold,
            vad_threshold=self.vad_threshold,
            fallback_target_registered=target_phrase in self.phrases)
        self._reservation = reservation
        job = DecodeJob(self._next_job_id(target, "acoustic_listener"),
                        self._stream_epoch, "acoustic_listener", snapshot,
                        reservation.deadline, self.listener_recognizer)
        reservation.job = job
        if not self.lane.submit(job):
            self._fail_acoustic_reservation(reservation, "acoustic_queue_busy",
                                            stage="admission")
        else:
            self._acoustic_pending[job.job_id] = (job, reservation, "listener")
        if provisional:
            self._finish_provisional_frame(target)
        return provisional

    def _finish_provisional_frame(self, segment):
        if self._reservation is None or self._reservation.target is not segment:
            self._maybe_submit_peek(segment)
        if segment.silence_run >= self.silence_frames:
            self._seal_final(segment, "silence", truncated=False)
        elif segment.live_frames >= self.max_frames:
            self._seal_boundary(segment)

    def _diagnostic_for_records(self, reason, records, proposal, *, stage):
        if not records:
            self._diagnostic_event(reason, None, proposal_id=uuid.uuid4().hex,
                                   stage=stage)
            return
        segment = _Segment(uuid.uuid4().hex[:16], list(records),
                           speech_started_at=records[0].captured_at,
                           voiced_frames=sum(item.probability >= self.vad_threshold
                                             for item in records),
                           vad_peak=max(item.probability for item in records))
        snapshot = self._snapshot(segment, list(records), end_reason=reason,
                                  truncated=False,
                                  authorization=records[-1].authorization)
        metadata = dict(snapshot.metadata)
        metadata.update(self._frozen_proposal_metadata(proposal))
        snapshot = AudioSnapshot(
            snapshot.stream_epoch, snapshot.segment_id, snapshot.revision,
            snapshot.start_sample, snapshot.end_sample, snapshot.captured_at,
            snapshot.completed_at, snapshot.audio, snapshot.processed_audio,
            snapshot.end_reason, snapshot.truncated, snapshot.overlap_of, metadata)
        self._diagnostic_event(reason, segment, proposal_id=uuid.uuid4().hex,
                               stage=stage, snapshot=snapshot)

    @staticmethod
    def _frozen_proposal_metadata(proposal):
        return {"diagnostic_provenance": "unknown", "acoustic_proposal": {
            "stock_model": proposal.stock_model,
            "stock_score": proposal.stock_score,
            "personal_hit": proposal.personal_hit,
            "personal_score": proposal.personal_score,
        }}

    @staticmethod
    def _bounded_transcript(value):
        return value[:4096]

    def _diagnostic_event(self, reason, segment, *, proposal_id="", stage="",
                          snapshot=None, reservation=None):
        if self._diagnostic_events >= _MAX_DIAGNOSTIC_EVENTS or not self._has_capacity():
            self._diagnostic_drops += 1
            return
        self._diagnostic_events += 1
        self._events.append(ListenerEvent(
            "diagnostic", reason=reason,
            segment_id="" if segment is None else segment.segment_id,
            capture_id="" if segment is None else self._capture_id(segment),
            diagnostic_snapshot=snapshot, diagnostic_stage=stage,
            proposal_id=proposal_id,
            trigger_transcript=(None if reservation is None
                                else reservation.trigger_transcript),
            second_stt_transcript=(None if reservation is None
                                   else reservation.second_transcript),
            trigger_stt_latency_ms=(None if reservation is None
                                    else reservation.trigger_latency_ms),
            second_stt_latency_ms=(None if reservation is None
                                   else reservation.second_latency_ms)))

    def _terminal_snapshot(self, segment, reason):
        records = list(segment.records)
        if not records:
            return None
        try:
            snapshot = self._snapshot(segment, records, end_reason=reason,
                                      truncated=segment.extending,
                                      authorization=records[-1].authorization)
        except (ValueError, TypeError):
            return None
        metadata = dict(snapshot.metadata)
        metadata.update({"diagnostic_provenance": "unknown",
                         "sample_interval": [snapshot.start_sample, snapshot.end_sample]})
        return AudioSnapshot(snapshot.stream_epoch, snapshot.segment_id, snapshot.revision,
                             snapshot.start_sample, snapshot.end_sample,
                             snapshot.captured_at, snapshot.completed_at,
                             snapshot.audio, snapshot.processed_audio,
                             snapshot.end_reason, snapshot.truncated,
                             snapshot.overlap_of, metadata)

    def _feed_weak_candidate(self, record, primary_speech):
        if not self.weak_enabled:
            return
        if primary_speech:
            self._weak_records.clear()
            self._weak_pre.clear()
            self._weak_voiced = self._weak_silence = 0
            self._weak_peak = 0.0
            return
        weak = record.probability >= self.weak_floor
        if not self._weak_records:
            if not weak:
                self._weak_pre.append(record)
                return
            self._weak_records = list(self._weak_pre) + [record]
            self._weak_pre.clear()
            self._weak_voiced = 1
            self._weak_silence = 0
            self._weak_peak = record.probability
            return
        self._weak_records.append(record)
        self._weak_voiced += int(weak)
        self._weak_silence = 0 if weak else self._weak_silence + 1
        self._weak_peak = max(self._weak_peak, record.probability)
        forced = len(self._weak_records) >= self.weak_max_frames
        if not forced and self._weak_silence < self.weak_silence_frames:
            return
        if self._weak_voiced >= self.weak_min_frames:
            segment = _Segment(uuid.uuid4().hex[:16], list(self._weak_records),
                               speech_started_at=self._weak_records[0].captured_at,
                               voiced_frames=self._weak_voiced,
                               vad_peak=self._weak_peak)
            snapshot = self._snapshot(segment, list(self._weak_records),
                                      end_reason="below_primary_vad",
                                      truncated=forced,
                                      authorization=self._weak_records[-1].authorization)
            self._diagnostic_event("below_primary_vad", segment, stage="weak_candidate",
                                   snapshot=snapshot)
        self._weak_records.clear()
        self._weak_voiced = self._weak_silence = 0
        self._weak_peak = 0.0

    def _resolve_held_terminal(self, segment, fallback):
        held = self._held_terminal.pop(segment.segment_id, None)
        if held is not None and segment.reserved_events > 0:
            _held_segment, reason, pending, result = held
            if (segment.confirmed_audio_handoff and pending is not None
                    and result is not None):
                self._finish_utterance(pending, result, segment.wake_phrase, "",
                                       segment.wake_mid, "acoustic",
                                       result.transcript, result.latency_ms)
            else:
                # The reservation has now resolved; let the terminal outcome out.
                self._reservation = None
                self._emit_terminal_rejection(segment, reason or fallback)

    @staticmethod
    def _validate_frame(frame, name: str) -> None:
        if (not isinstance(frame, np.ndarray) or frame.ndim != 1
                or frame.dtype != np.int16 or frame.size != _FRAME):
            raise ValueError("{} must be exactly 512 int16 mono samples".format(name))

    @staticmethod
    def _finite_time(value) -> float:
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError("captured_at must be finite")
        return float(value)

    def _reserve_for_segment(self, segment: _Segment) -> bool:
        if not self._has_capacity(2):
            return False
        segment.reserved_events = 2
        self._reserved_events += 2
        return True

    def _emit_reserved(self, segment: _Segment, event: ListenerEvent) -> None:
        if segment.reserved_events <= 0:
            self._diagnostic("event_reservation_missing", segment.segment_id)
            return
        segment.reserved_events -= 1
        self._reserved_events -= 1
        self._events.append(event)

    def _release_reservation(self, segment: _Segment) -> None:
        self._reserved_events -= segment.reserved_events
        segment.reserved_events = 0

    def _diagnostic(self, reason: str, segment_id: str = "") -> None:
        if self._has_capacity():
            self._events.append(ListenerEvent("diagnostic", reason=reason,
                                              segment_id=segment_id))

    @staticmethod
    def _reason_strings(values) -> list[str]:
        reasons = []
        for value in values:
            reason = value[0] if isinstance(value, tuple) and value else value
            if isinstance(reason, str) and reason:
                reasons.append(reason[:80])
        return list(dict.fromkeys(reasons))[:12]

    def _consume_record(self, record: _FrameRecord, *, replay: bool) -> None:
        is_speech = record.probability >= self.vad_threshold
        if self._active is None:
            if self.pad_frames:
                self._pre.append(record)
            if not is_speech:
                return
            records = list(self._pre) if self.pad_frames else []
            if not records or records[-1].start_sample != record.start_sample:
                records.append(record)
            self._pre.clear()
            segment = _Segment(uuid.uuid4().hex[:16], records,
                               speech_started_at=record.captured_at,
                               voiced_frames=1, vad_peak=record.probability,
                               next_peek=self.peek_first, live_frames=1,
                               live_voiced_frames=1)
            if not self._reserve_for_segment(segment):
                # Replayed frames remain recoverable; live input is rejected
                # before VAD whenever a new reservation cannot be made.
                self._replay.appendleft(record)
                return
            self._active = segment
            self._maybe_submit_peek(segment)
            return

        segment = self._active
        segment.records.append(record)
        segment.live_frames += 1
        segment.vad_peak = max(segment.vad_peak, record.probability)
        if is_speech:
            segment.voiced_frames += 1
            segment.live_voiced_frames += 1
        segment.silence_run = 0 if is_speech else segment.silence_run + 1
        if segment.extending:
            segment.extension_frames += 1
        self._maybe_submit_peek(segment)
        if segment.silence_run >= self.silence_frames:
            self._seal_final(segment, "silence", truncated=False)
        elif segment.extending and segment.extension_frames >= self.extend_frames:
            self._seal_final(segment, "extension_limit", truncated=True)
        elif not segment.extending and segment.live_frames >= self.max_frames:
            self._seal_boundary(segment)

    def _snapshot(self, segment: _Segment, records: list[_FrameRecord], *,
                  end_reason: str, truncated: bool, authorization=None) -> AudioSnapshot:
        if not records:
            raise ValueError("cannot snapshot an empty segment")
        if not self._records_contiguous(records):
            raise ValueError("snapshot records must be contiguous")
        raw = np.concatenate([item.raw for item in records])
        processed = (np.concatenate([item.processed for item in records])
                     if all(item.processed is not None for item in records) else None)
        segment.revision += 1
        wake = {"phrase": segment.wake_phrase, "path": segment.wake_path,
                "evidence_at": segment.wake_evidence_at, "mid": segment.wake_mid}
        rejections = tuple(dict.fromkeys(segment.rejections))[:12]
        metadata = {
            "duration_ms": int(len(records) * _FRAME_SECONDS * 1000),
            "segment_frame_count": len(segment.records),
            "speech_started_at": segment.speech_started_at,
            "wake_evidence": wake,
            "rejections": rejections,
            "voiced_frames": segment.voiced_frames,
            "vad_peak": segment.vad_peak,
            "vad_threshold": self.vad_threshold,
            "confirmation_handoff": segment.confirmed_audio_handoff,
            "sample_interval": [records[0].start_sample, records[-1].end_sample],
            "vad_summary": {
                "threshold": self.vad_threshold,
                "peak": max(item.probability for item in records),
                "primary_voiced_frames": sum(
                    item.probability >= self.vad_threshold for item in records),
                "frame_count": len(records),
            },
            "processed_alignment": ("frame_paired" if all(
                item.processed is not None for item in records) else "unavailable"),
            "diagnostic_provenance": "unknown",
            # Freeze authorization from this exact snapshot's final frame.
            # Delayed result application never consults live state or stale
            # positive grants from earlier frames.
            "authorization": dict(records[-1].authorization if authorization is None
                                   else authorization),
        }
        if segment.acoustic_receipt_epoch is not None:
            metadata["acoustic_receipt"] = {
                "epoch": segment.acoustic_receipt_epoch,
                "capture_id": segment.segment_id,
                "start_sample": segment.acoustic_receipt_start,
                "end_sample": segment.acoustic_receipt_end,
                "phrase": segment.acoustic_receipt_phrase,
            }
        first, last = records[0], records[-1]
        completed = last.captured_at + _FRAME_SECONDS
        return AudioSnapshot(
            self._stream_epoch, segment.segment_id, segment.revision,
            first.start_sample, last.end_sample, first.captured_at, completed,
            raw, processed, end_reason=end_reason, truncated=truncated,
            overlap_of=segment.overlap_of, metadata=metadata)

    def _next_job_id(self, segment: _Segment, purpose: str) -> str:
        self._job_counter += 1
        return "{}:{}:{}:{}:{}".format(
            self._stream_epoch, segment.segment_id, segment.revision,
            purpose, self._job_counter)

    def _submit(self, segment: _Segment, purpose: str, recognizer: str, *,
                records: Optional[list[_FrameRecord]] = None, end_reason="",
                truncated=False, original_transcript="", original_latency_ms=0,
                proposed_phrase="",
                proposed_mid=False, boundary=None, authorization=None) -> Optional[_Pending]:
        sealed = sum(1 for item in self._pending.values() if item.purpose != "peek")
        if purpose != "peek" and sealed >= _MAX_SEALED:
            return None
        chosen = segment.records if records is None else records
        snapshot = self._snapshot(segment, chosen, end_reason=end_reason,
                                  truncated=truncated, authorization=authorization)
        job = DecodeJob(self._next_job_id(segment, purpose), self._stream_epoch,
                        purpose, snapshot, self._now() + _JOB_DEADLINE_SECONDS,
                        recognizer)
        pending = _Pending(job, segment, purpose, end_reason, truncated,
                           original_transcript, original_latency_ms,
                           proposed_phrase, proposed_mid,
                           boundary)
        if not self.lane.submit(job):
            return None
        self._pending[job.job_id] = pending
        return pending

    def _maybe_submit_peek(self, segment: _Segment) -> None:
        if self._reservation is not None and self._reservation.target is segment:
            return
        count = len(segment.records)
        if (self.peek_first <= 0 or count < segment.next_peek
                or count > self.peek_until):
            return
        segment.next_peek = count + self.peek_every
        records = segment.records[-self.peek_window:]
        if self._submit(segment, "peek", self.listener_recognizer,
                        records=records, end_reason="peek") is None:
            segment.rejections.append("peek_backpressure")

    def _seal_final(self, segment: _Segment, reason: str, *, truncated: bool,
                    records: Optional[list[_FrameRecord]] = None) -> None:
        self._active = None
        self._pre.clear()
        too_short = ((segment.live_voiced_frames < self.min_frames
                      and segment.live_frames < self.min_frames)
                     if segment.provisional else
                     (segment.voiced_frames < self.min_frames
                      and len(segment.records if records is None else records) < self.min_frames))
        if too_short and not segment.extending:
            self._reject_segment(segment, "speech_too_short")
            return
        pending = self._submit(segment, "final", self.listener_recognizer,
                               records=records, end_reason=reason, truncated=truncated)
        if pending is None:
            self._reject_segment(segment, "final_backpressure")

    def _seal_boundary(self, segment: _Segment) -> None:
        self._active = None
        self._pre.clear()
        boundary = _Boundary(segment, "", self._now() + _JOB_DEADLINE_SECONDS,
                             silence_run=segment.silence_run)
        pending = self._submit(segment, "boundary", self.listener_recognizer,
                               end_reason="max_utterance", truncated=True,
                               boundary=boundary)
        if pending is None:
            self._reject_segment(segment, "boundary_backpressure")
            return
        boundary.job_id = pending.job.job_id
        boundary.snapshot_end = pending.job.snapshot.end_sample
        self._boundary = boundary

    def _feed_boundary(self, record: _FrameRecord) -> None:
        boundary = self._boundary
        if boundary is None:
            return
        if self._now() >= boundary.deadline:
            self._abort_boundary("boundary_deadline_exceeded")
            self._replay.append(record)
            return
        boundary.continuation.append(record)
        if record.probability >= self.vad_threshold:
            boundary.silence_run = 0
        else:
            boundary.silence_run += 1
            if boundary.silence_run >= self.silence_frames and boundary.silence_end is None:
                boundary.silence_end = len(boundary.continuation)
        cap = self.extend_frames + self.pad_frames
        if len(boundary.continuation) > cap:
            self._abort_boundary("boundary_buffer_overflow")

    def _abort_boundary(self, reason: str) -> None:
        boundary = self._boundary
        if boundary is None:
            return
        self._boundary = None
        if self._reservation is not None:
            self._fail_acoustic_reservation(self._reservation,
                                            "stream_reset_" + reason,
                                            stage="lifecycle")
        # Epoch invalidation discards every older lane result, so give each
        # accepted in-flight segment an explicit terminal event first.
        affected = {id(pending.segment): pending.segment
                    for pending in self._pending.values()}
        affected[id(boundary.segment)] = boundary.segment
        for segment in affected.values():
            if segment.reserved_events <= 0:
                continue
            terminal_reason = (reason if segment is boundary.segment
                               else "stream_reset_" + reason)
            self._emit_terminal_rejection(segment, terminal_reason)
        # No transcript from an over-budget/truncated boundary can authorize
        # a command. Invalidate late lane completions without changing the
        # caller-owned worker lifecycle.
        self._stream_epoch += 1
        self.lane.invalidate(self._stream_epoch)
        self._pending.clear()
        self._acoustic_pending.clear()
        self._reservation = None
        self._held_terminal.clear()
        self._active = None
        self._pre.clear()
        self._replay.clear()
        self._history.clear()
        self._weak_records.clear()
        self._weak_pre.clear()
        reset = getattr(self.vad, "reset", None)
        if callable(reset):
            reset()

    def _emit_terminal_rejection(self, segment: _Segment, reason: str) -> None:
        reservation = self._reservation
        if (reservation is not None and reservation.target is segment
                and not reservation.resolved):
            self._held_terminal[segment.segment_id] = (segment, reason, None, None)
            return
        snapshot = self._terminal_snapshot(segment, reason)
        if snapshot is not None and self._reservation is not None \
                and self._reservation.target is segment:
            metadata = dict(snapshot.metadata)
            metadata.update(self._frozen_proposal_metadata(self._reservation.proposal))
            snapshot = AudioSnapshot(
                snapshot.stream_epoch, snapshot.segment_id, snapshot.revision,
                snapshot.start_sample, snapshot.end_sample, snapshot.captured_at,
                snapshot.completed_at, snapshot.audio, snapshot.processed_audio,
                snapshot.end_reason, snapshot.truncated, snapshot.overlap_of, metadata)
        proposal_id = (self._reservation.proposal_id
                       if self._reservation is not None
                       and self._reservation.target is segment else "")
        self._emit_reserved(segment, ListenerEvent(
            "rejected", reason=reason, segment_id=segment.segment_id,
            capture_id=self._capture_id(segment), diagnostic_snapshot=snapshot,
            diagnostic_stage="terminal" if snapshot is not None else "",
            proposal_id=proposal_id))
        self._release_reservation(segment)

    def _finish_acoustic_reservation(self, reservation, phrase, result, mode, mid=False):
        if self._reservation is not reservation or reservation.epoch != self._stream_epoch:
            return
        if phrase not in self.phrases:
            self._fail_acoustic_reservation(reservation, "acoustic_target_mismatch",
                                            result, "confirmation")
            return
        boundary = reservation.target_boundary
        if boundary is not None and self._boundary is boundary \
                and self._now() >= boundary.deadline:
            self._abort_boundary("boundary_deadline_exceeded")
            return
        reservation.resolved = True
        self._reservation = None
        segment = reservation.target
        segment.confirmed_audio_handoff = True
        segment.acoustic_receipt_epoch = reservation.epoch
        segment.acoustic_receipt_start = reservation.interval_start
        segment.acoustic_receipt_end = reservation.interval_end
        segment.acoustic_receipt_phrase = phrase
        segment.wake_phrase = phrase
        segment.wake_mid = bool(mid)
        segment.wake_path = "alias_acoustic" if mode == "acoustic" else "acoustic_confirmed"
        segment.wake_evidence_at = result.finished_at
        segment.wake_emitted = True
        self._emit_reserved(segment, ListenerEvent(
            "wake", segment_id=segment.segment_id,
            capture_id=self._capture_id(segment), phrase=phrase,
            wake_path=segment.wake_path, evidence_at=result.finished_at,
            proposal_id=reservation.proposal_id))
        if boundary is not None and self._boundary is boundary:
            if reservation.interval_end > boundary.snapshot_end:
                if not self._continuation_covers_receipt(boundary, reservation):
                    self._abort_boundary("acoustic_extension_cutoff")
                    return
                if boundary.deferred_pending is not None:
                    self._continue_acoustic_boundary(boundary)
            elif boundary.deferred_pending is not None:
                self._resume_deferred_boundary(boundary, confirmed=True)
            return
        held = self._held_terminal.pop(segment.segment_id, None)
        if held is not None and segment.reserved_events > 0:
            _segment, _reason, pending, full_result = held
            if pending is not None and full_result is not None:
                self._finish_utterance(pending, full_result, phrase, "", segment.wake_mid,
                                       "acoustic", full_result.transcript,
                                       full_result.latency_ms)
            else:
                self._emit_terminal_rejection(segment, "acoustic_confirmed_without_full_capture")

    def _fail_acoustic_reservation(self, reservation, reason, result=None, stage="confirmation"):
        if self._reservation is not reservation:
            return
        reservation.resolved = True
        self._reservation = None
        for job_id, (_job, pending_reservation, _stage) in tuple(self._acoustic_pending.items()):
            if pending_reservation is reservation:
                self._acoustic_pending.pop(job_id, None)
        self._diagnostic_event(reason, reservation.target,
                               proposal_id=reservation.proposal_id, stage=stage,
                               snapshot=self._decorate_reservation_snapshot(reservation),
                               reservation=reservation)
        held = self._held_terminal.pop(reservation.target.segment_id, None)
        if held is not None and reservation.target.reserved_events > 0:
            _segment, held_reason, _pending, _full_result = held
            self._emit_terminal_rejection(reservation.target, held_reason or reason)
        boundary = reservation.target_boundary
        if boundary is not None and boundary is self._boundary \
                and boundary.deferred_pending is not None:
            if self._now() >= boundary.deadline:
                self._abort_boundary("boundary_deadline_exceeded")
            else:
                self._resume_deferred_boundary(boundary, confirmed=False)

    @staticmethod
    def _decorate_reservation_snapshot(reservation):
        snapshot = reservation.snapshot
        metadata = dict(snapshot.metadata)
        metadata.update(AsyncSttListener._frozen_proposal_metadata(reservation.proposal))
        scores = {reservation.proposal.stock_model: reservation.proposal.stock_score}
        if reservation.proposal.personal_hit:
            scores["alexa_user"] = reservation.proposal.personal_score
        metadata.update({
            "scores": scores,
            "confirmation_interval": {
                "epoch": reservation.epoch,
                "capture_id": reservation.target.segment_id,
                "start_sample": reservation.interval_start,
                "end_sample": reservation.interval_end,
            },
            "proposal_qualifiers": {
                "stock_model": reservation.proposal.stock_model,
                "personal_hit": reservation.proposal.personal_hit,
                "target_phrase": reservation.target_phrase,
                "fallback_target_registered": reservation.fallback_target_registered,
                "stock_qualified": (reservation.proposal.stock_score
                                    >= reservation.stock_threshold),
                "personal_qualified": (reservation.proposal.personal_hit
                                       and reservation.proposal.personal_score
                                       >= reservation.personal_threshold),
                "stock_threshold": reservation.stock_threshold,
                "personal_threshold": reservation.personal_threshold,
                "vad_threshold": reservation.vad_threshold,
            },
            "trigger_transcript": (reservation.trigger_transcript or ""),
            "second_stt_transcript": (reservation.second_transcript or ""),
            "trigger_stage_state": reservation.trigger_stage_state,
            "second_stt_stage_state": reservation.second_stage_state,
            "trigger_transcript_truncated": reservation.trigger_transcript_truncated,
            "second_stt_transcript_truncated": reservation.second_transcript_truncated,
        })
        if reservation.trigger_latency_ms is not None \
                and reservation.trigger_stage_state == "completed":
            metadata["trigger_stt_latency_ms"] = reservation.trigger_latency_ms
        if reservation.second_latency_ms is not None \
                and reservation.second_stage_state == "completed":
            metadata["second_stt_latency_ms"] = reservation.second_latency_ms
        return AudioSnapshot(snapshot.stream_epoch, snapshot.segment_id, snapshot.revision,
                             snapshot.start_sample, snapshot.end_sample,
                             snapshot.captured_at, snapshot.completed_at,
                             snapshot.audio, snapshot.processed_audio,
                             snapshot.end_reason, snapshot.truncated,
                             snapshot.overlap_of, metadata)

    def _apply_acoustic_result(self, reservation, stage, result):
        if stage == "listener":
            safe = not (result.error_category or result.expired or result.cancelled)
            reservation.trigger_transcript = (self._bounded_transcript(result.transcript)
                                              if safe else None)
            reservation.trigger_transcript_truncated = safe and len(result.transcript) > 4096
            reservation.trigger_latency_ms = result.latency_ms if safe else None
            reservation.trigger_stage_state = (
                "completed" if safe else "failed")
        else:
            safe = not (result.error_category or result.expired or result.cancelled)
            reservation.second_transcript = (self._bounded_transcript(result.transcript)
                                             if safe else None)
            reservation.second_transcript_truncated = safe and len(result.transcript) > 4096
            reservation.second_latency_ms = result.latency_ms if safe else None
            reservation.second_stage_state = (
                "completed" if safe else "failed")
        boundary = reservation.target_boundary
        if (boundary is not None and self._boundary is boundary
                and self._now() >= boundary.deadline):
            self._abort_boundary("boundary_deadline_exceeded")
            return
        if (self._reservation is not reservation or reservation.epoch != self._stream_epoch
                or result.finished_at >= reservation.deadline):
            self._fail_acoustic_reservation(reservation, "acoustic_deadline_exceeded",
                                            result, stage)
            return
        if result.error_category or result.expired or result.cancelled:
            self._fail_acoustic_reservation(
                reservation, result.error_category or "acoustic_decode_failed", result, stage)
            return
        if stage == "listener":
            matcher = self._matcher(reservation.snapshot.metadata.get("authorization", {}),
                                    int(reservation.snapshot.metadata.get("duration_ms", 0)))
            phrase, _command, _mid = matcher.locate_trigger(result.transcript)
            mode = getattr(matcher, "_match_source", "text_exact") if phrase else ""
            if phrase and phrase in self.phrases:
                self._finish_acoustic_reservation(reservation, phrase, result, mode, _mid)
                return
            qualified = ((reservation.proposal.personal_hit
                          and reservation.proposal.personal_score >= self.personal_threshold)
                         or reservation.proposal.stock_score >= self.acoustic_threshold)
            if not reservation.fallback_target_registered:
                self._fail_acoustic_reservation(
                    reservation, "acoustic_fallback_target_unregistered",
                    result, "confirmation")
                return
            if not qualified or self.verify_text is None or self._now() >= reservation.deadline:
                self._fail_acoustic_reservation(reservation, "acoustic_unconfirmed",
                                                result, "confirmation")
                return
            job = DecodeJob(
                self._next_job_id(reservation.target, "acoustic_verify"),
                reservation.epoch, "acoustic_verify", reservation.snapshot,
                reservation.deadline, self.verify_recognizer)
            if not self.lane.submit(job):
                self._fail_acoustic_reservation(reservation, "acoustic_verify_backpressure",
                                                result, "verification")
                return
            reservation.transcript = result.transcript
            reservation.latency_ms = result.latency_ms
            reservation.verify_job = job
            self._acoustic_pending[job.job_id] = (job, reservation, "verify")
            return
        try:
            matched = self.verify_text(result.transcript, reservation.target_phrase, False)
        except Exception:
            matched = None
        if (not isinstance(matched, tuple) or len(matched) != 3
                or not isinstance(matched[0], str)
                or not isinstance(matched[1], str)
                or not isinstance(matched[2], bool)
                or matched[0].strip().lower() != reservation.target_phrase):
            self._fail_acoustic_reservation(reservation, "acoustic_verification_refused",
                                            result, "verification")
            return
        self._finish_acoustic_reservation(reservation, reservation.target_phrase,
                                          result, "second_stt", matched[2])

    @staticmethod
    def _capture_id(segment: _Segment) -> str:
        # Capture identity names the capture, not a mutable endpoint revision.
        return segment.segment_id

    def _emit_wake(self, segment: _Segment, phrase: str, path: str, at: float,
                   mid: bool) -> None:
        self._cancel_acoustic_for_segment(segment)
        if segment.wake_emitted:
            return
        segment.wake_phrase = phrase
        segment.wake_path = path
        segment.wake_evidence_at = at
        segment.wake_mid = mid
        segment.wake_emitted = True
        self._emit_reserved(segment, ListenerEvent(
            "wake", segment_id=segment.segment_id,
            capture_id=self._capture_id(segment), phrase=phrase,
            wake_path=path, evidence_at=at))

    def _cancel_acoustic_for_segment(self, segment):
        reservation = self._reservation
        if reservation is None or reservation.target is not segment:
            return
        reservation.resolved = True
        self._reservation = None
        for job_id, (_job, pending_reservation, _stage) in tuple(self._acoustic_pending.items()):
            if pending_reservation is reservation:
                self._acoustic_pending.pop(job_id, None)
        self._diagnostic_event("acoustic_superseded_by_text", segment,
                               proposal_id=reservation.proposal_id,
                               stage="resolution",
                               snapshot=self._decorate_reservation_snapshot(reservation),
                               reservation=reservation)

    def _reject_segment(self, segment: _Segment, reason: str) -> None:
        self._emit_terminal_rejection(segment, reason)

    def _apply_peek(self, pending: _Pending, result: DecodeResult) -> None:
        segment = pending.segment
        active = self._active
        snapshot = result.job.snapshot
        if (active is None or active.segment_id != segment.segment_id
                or snapshot.start_sample < active.records[0].start_sample
                or snapshot.end_sample > active.records[-1].end_sample
                or snapshot.revision <= active.last_peek_result_revision):
            return
        active.last_peek_result_revision = snapshot.revision
        if result.error_category or result.expired or result.cancelled:
            active.rejections.append(result.error_category or "peek_failed")
            return
        auth = snapshot.metadata.get("authorization", {})
        matcher = self._matcher(auth, int(snapshot.metadata.get("duration_ms", 0)))
        phrase, _command, mid = matcher.locate_trigger(result.transcript)
        mid = mid or int(snapshot.metadata.get("segment_frame_count", 0)) > self.peek_window
        if phrase:
            path = "alias_acoustic" if getattr(matcher, "_match_source", "") == "acoustic" else "peek"
            self._emit_wake(active, phrase, path, result.finished_at, mid)
        else:
            candidate, _command, _mid = matcher._unconfirmed_candidate(result.transcript)
            if candidate:
                # Proposal only; final audio may request command-ASR verification.
                active.wake_phrase = active.wake_phrase or candidate
                active.wake_mid = (active.wake_mid or _mid
                                   or int(snapshot.metadata.get("segment_frame_count", 0))
                                   > self.peek_window)
                active.rejections.extend(self._reason_strings(matcher._pending_rejections))

    def _apply_final(self, pending: _Pending, result: DecodeResult) -> None:
        segment = pending.segment
        if result.error_category or result.expired or result.cancelled:
            if (self._reservation is not None and self._reservation.target is segment
                    and not self._reservation.resolved):
                self._held_terminal[segment.segment_id] = (
                    segment, result.error_category or "decode_failed", pending, result)
                return
            self._emit_terminal_rejection(segment, result.error_category or "decode_failed")
            return
        snapshot = result.job.snapshot
        if (segment.acoustic_receipt_epoch is not None
                and not self._snapshot_covers_receipt(snapshot, segment)):
            self._emit_terminal_rejection(segment, "acoustic_evidence_not_in_capture")
            return
        auth = snapshot.metadata.get("authorization", {})
        duration = int(snapshot.metadata.get("duration_ms", 0))
        matcher = self._matcher(auth, duration)
        phrase, command, mid = matcher.locate_trigger(result.transcript)
        mode = getattr(matcher, "_match_source", "text_exact") if phrase else ""
        if not phrase:
            if segment.confirmed_audio_handoff and segment.wake_phrase:
                self._finish_utterance(pending, result, segment.wake_phrase, "",
                                       segment.wake_mid, "acoustic",
                                       result.transcript, result.latency_ms)
                return
            candidate, _candidate_command, candidate_mid = matcher._unconfirmed_candidate(
                result.transcript)
            candidate = candidate or segment.wake_phrase
            candidate_mid = candidate_mid or segment.wake_mid
            if candidate:
                verified = self._submit_verify(pending, result.transcript,
                                                candidate, candidate_mid,
                                                result.latency_ms)
                if verified:
                    return
                self._emit_terminal_rejection(segment, "verification_unavailable")
                return
            reasons = self._reason_strings(matcher._pending_rejections) or segment.rejections
            if (self._reservation is not None and self._reservation.target is segment
                    and not self._reservation.resolved):
                self._held_terminal[segment.segment_id] = (
                    segment, ",".join(dict.fromkeys(reasons))[:240]
                    or "speech_no_trigger", pending, result)
                return
            self._emit_terminal_rejection(segment,
                ",".join(dict.fromkeys(reasons))[:240] or "speech_no_trigger")
            return
        self._finish_utterance(pending, result, phrase, command, mid, mode,
                               result.transcript, result.latency_ms)

    def _submit_verify(self, source: _Pending, transcript: str, phrase: str,
                       mid: bool, latency_ms: int) -> bool:
        if self.verify_text is None:
            return False
        pending = self._submit_snapshot_job(
            source.segment, "verify", self.verify_recognizer,
            source.job.snapshot, end_reason=source.end_reason,
            truncated=source.truncated, original_transcript=transcript,
            original_latency_ms=latency_ms, proposed_phrase=phrase,
            proposed_mid=mid, boundary=source.boundary)
        if pending is None:
            return False
        return True

    def _finish_utterance(self, pending: _Pending, result: DecodeResult,
                          phrase: str, command: str, mid: bool, mode: str,
                          transcript: str, latency_ms: int, *,
                          confirmation_transcript: str = "",
                          confirmation_latency_ms=None) -> None:
        segment = pending.segment
        snapshot = result.job.snapshot
        if result.error_category or result.expired or result.cancelled:
            reason = result.error_category
            if not reason:
                reason = ("decode_expired" if result.expired else
                          "decode_cancelled" if result.cancelled else
                          "decode_failed")
            self._emit_terminal_rejection(segment, reason)
            return
        if (segment.acoustic_receipt_epoch is not None
                and not self._snapshot_covers_receipt(snapshot, segment)):
            self._emit_terminal_rejection(segment, "acoustic_evidence_not_in_capture")
            return
        if pending.truncated:
            command = ""
        matcher = self._matcher(snapshot.metadata.get("authorization", {}),
                                int(snapshot.metadata.get("duration_ms", 0)))
        repeated, remainder = matcher.match_trigger(command)
        if repeated and not remainder:
            command = ""
        path = (segment.wake_path if segment.confirmed_audio_handoff else
                "second_stt" if mode == "second_stt" else
                "alias_acoustic" if mode == "acoustic" else
                "peek" if segment.wake_emitted else "final_text")
        evidence_at = segment.wake_evidence_at or result.finished_at
        self._emit_wake(segment, phrase, path, evidence_at, mid)
        utterance = Utterance(np.array(snapshot.audio, copy=True), transcript, latency_ms,
                              completed_at=snapshot.completed_at)
        utterance.processed_audio = (None if snapshot.processed_audio is None else
                                     np.array(snapshot.processed_audio, copy=True))
        utterance.segment_id = segment.segment_id
        utterance.end_reason = pending.end_reason
        utterance.truncated = pending.truncated
        utterance.confirmed_audio_handoff = segment.confirmed_audio_handoff
        utterance.confirmation_mode = mode
        utterance.confirmation_latency_ms = confirmation_latency_ms
        utterance.confirmation_transcript = confirmation_transcript
        utterance.speech_started_at = float(snapshot.metadata.get("speech_started_at", 0.0))
        utterance.trigger_done_at = result.finished_at
        utterance.wake_path = path
        utterance.wake_evidence_at = evidence_at
        utterance.triggered = True
        utterance.matched_phrase = phrase
        utterance.command = command
        utterance.mid_utterance = segment.wake_mid or mid
        self._emit_reserved(segment, ListenerEvent(
            "utterance", utterance=utterance, segment_id=segment.segment_id,
            capture_id=self._capture_id(segment)))
        self._release_reservation(segment)

    def _apply_boundary(self, pending: _Pending, result: DecodeResult) -> None:
        boundary = pending.boundary
        if boundary is None or self._boundary is not boundary:
            return
        if self._now() >= boundary.deadline:
            self._abort_boundary("boundary_deadline_exceeded")
            return
        segment = boundary.segment
        if result.error_category or result.expired or result.cancelled:
            self._abort_boundary(result.error_category or "boundary_decode_failed")
            return
        matcher = self._matcher(result.job.snapshot.metadata.get("authorization", {}),
                                int(result.job.snapshot.metadata.get("duration_ms", 0)))
        phrase, command, mid = matcher.locate_trigger(result.transcript)
        mode = getattr(matcher, "_match_source", "text_exact") if phrase else ""
        reservation = self._reservation
        if (not phrase and reservation is not None and reservation.target is segment
                and not reservation.resolved):
            self._defer_boundary_result(boundary, pending, result)
            return
        if (segment.acoustic_receipt_epoch is not None
                and not self._snapshot_covers_receipt(result.job.snapshot, segment)):
            self._defer_boundary_result(boundary, pending, result)
            self._continue_acoustic_boundary(boundary)
            return
        if not phrase:
            if segment.confirmed_audio_handoff and segment.wake_phrase:
                self._emit_boundary_utterance(pending, result, segment.wake_phrase,
                                              segment.wake_mid, "acoustic")
                self._boundary = None
                self._start_overlap_replay(boundary, include_overlap=False)
                return
            candidate, _, candidate_mid = matcher._unconfirmed_candidate(result.transcript)
            if not candidate and segment.wake_phrase:
                candidate, candidate_mid = segment.wake_phrase, segment.wake_mid
            if candidate and self.verify_text is not None:
                # Verification must use a command-ASR job, never a legacy callback.
                verify = self._submit_snapshot_job(
                    segment, "verify", self.verify_recognizer, result.job.snapshot,
                    end_reason="max_utterance", truncated=True,
                    original_transcript=result.transcript,
                    original_latency_ms=result.latency_ms,
                    proposed_phrase=candidate, proposed_mid=candidate_mid,
                    boundary=boundary)
                if verify is not None:
                    return
            self._emit_terminal_rejection(segment,
                ",".join(self._reason_strings(matcher._pending_rejections))[:240]
                or "speech_no_trigger")
            self._boundary = None
            self._start_overlap_replay(boundary, include_overlap=True)
            return
        if not command:
            self._emit_wake(segment, phrase,
                            "alias_acoustic" if mode == "acoustic" else
                            "peek" if segment.wake_emitted else "final_text",
                            segment.wake_evidence_at or result.finished_at, mid)
            self._emit_boundary_utterance(pending, result, phrase, mid, mode)
            self._boundary = None
            self._start_overlap_replay(boundary, include_overlap=False)
            return
        segment.wake_phrase = phrase
        segment.wake_mid = mid
        segment.wake_path = "alias_acoustic" if mode == "acoustic" else "final_text"
        segment.wake_evidence_at = result.finished_at
        self._emit_wake(segment, phrase, segment.wake_path,
                        segment.wake_evidence_at, mid)
        self._continue_after_boundary(boundary)

    @staticmethod
    def _snapshot_covers_receipt(snapshot, segment):
        if not (segment.acoustic_receipt_epoch == snapshot.stream_epoch
                and snapshot.segment_id == segment.segment_id
                and segment.acoustic_receipt_start is not None
                and segment.acoustic_receipt_end is not None
                and snapshot.start_sample <= segment.acoustic_receipt_start
                and snapshot.end_sample >= segment.acoustic_receipt_end):
            return False
        receipt = snapshot.metadata.get("acoustic_receipt")
        if receipt is not None:
            return (receipt.get("epoch") == segment.acoustic_receipt_epoch
                    and receipt.get("capture_id") == segment.segment_id
                    and receipt.get("start_sample") == segment.acoustic_receipt_start
                    and receipt.get("end_sample") == segment.acoustic_receipt_end
                    and receipt.get("phrase") == segment.acoustic_receipt_phrase)
        return True

    @staticmethod
    def _records_contiguous(records):
        return bool(records) and all(left.end_sample == right.start_sample
                                     for left, right in zip(records, records[1:]))

    def _continuation_covers_receipt(self, boundary, reservation):
        records = list(boundary.segment.records) + list(boundary.continuation)
        if (reservation.epoch != self._stream_epoch
                or not self._records_contiguous(records)
                or not records
                or records[0].start_sample > reservation.interval_start
                or records[-1].end_sample < reservation.interval_end):
            return False
        return True

    def _defer_boundary_result(self, boundary, pending, result):
        if boundary.deferred_pending is not None:
            self._diagnostic_event("boundary_deferred_result_collision", boundary.segment,
                                   stage="boundary")
            return False
        boundary.deferred_pending = pending
        boundary.deferred_result = result
        return True

    def _resume_deferred_boundary(self, boundary, *, confirmed):
        if self._boundary is not boundary or boundary.deferred_pending is None:
            return
        pending, result = boundary.deferred_pending, boundary.deferred_result
        boundary.deferred_pending = None
        boundary.deferred_result = None
        if result is None:
            self._abort_boundary("boundary_deferred_result_missing")
            return
        if confirmed and not self._snapshot_covers_receipt(result.job.snapshot,
                                                           boundary.segment):
            boundary.deferred_pending = pending
            boundary.deferred_result = result
            self._continue_acoustic_boundary(boundary)
            return
        if pending.purpose == "boundary":
            self._apply_boundary(pending, result)
        elif pending.purpose == "verify":
            self._apply_verify(pending, result)
        else:
            self._abort_boundary("boundary_deferred_purpose_invalid")

    def _continue_acoustic_boundary(self, boundary):
        segment = boundary.segment
        if (self._boundary is not boundary or not segment.confirmed_audio_handoff
                or segment.acoustic_receipt_epoch != self._stream_epoch
                or segment.acoustic_receipt_end is None):
            return
        if self._now() >= boundary.deadline:
            self._abort_boundary("boundary_deadline_exceeded")
            return
        reservation = self._reservation
        if reservation is not None and reservation.target is segment:
            return
        if not self._records_contiguous(list(segment.records) + list(boundary.continuation)):
            self._abort_boundary("acoustic_extension_gap")
            return
        required = max(0, (segment.acoustic_receipt_end - boundary.snapshot_end
                           + _FRAME - 1) // _FRAME)
        if (required > self.extend_frames or required > len(boundary.continuation)
                or len(boundary.continuation) > self.extend_frames + self.pad_frames):
            self._abort_boundary("acoustic_extension_cutoff")
            return
        boundary.deferred_pending = None
        boundary.deferred_result = None
        self._continue_after_boundary(boundary, min_extension_frames=required)

    def _release_deferred_boundary(self, boundary, reason):
        if self._boundary is not boundary:
            return
        boundary.deferred_pending = None
        boundary.deferred_result = None
        self._boundary = None
        self._emit_terminal_rejection(boundary.segment, reason)
        self._start_overlap_replay(boundary, include_overlap=True)

    def _submit_snapshot_job(self, segment: _Segment, purpose: str, recognizer: str,
                             snapshot: AudioSnapshot, *, end_reason: str,
                             truncated: bool, original_transcript="",
                             original_latency_ms=0,
                             proposed_phrase="", proposed_mid=False,
                             boundary=None) -> Optional[_Pending]:
        sealed = sum(1 for item in self._pending.values() if item.purpose != "peek")
        if sealed >= _MAX_SEALED:
            return None
        self._job_counter += 1
        job_id = "{}:{}:{}:{}:{}".format(self._stream_epoch, segment.segment_id,
                                          snapshot.revision + self._job_counter,
                                          purpose, self._job_counter)
        job = DecodeJob(job_id, self._stream_epoch, purpose, snapshot,
                        self._now() + _JOB_DEADLINE_SECONDS, recognizer)
        pending = _Pending(job, segment, purpose, end_reason, truncated,
                           original_transcript, original_latency_ms,
                           proposed_phrase, proposed_mid, boundary)
        if not self.lane.submit(job):
            return None
        self._pending[job_id] = pending
        return pending

    def _apply_verify(self, pending: _Pending, result: DecodeResult) -> None:
        segment = pending.segment
        boundary = pending.boundary
        if boundary is not None and self._boundary is boundary:
            if self._now() >= boundary.deadline:
                self._abort_boundary("boundary_deadline_exceeded")
                return
            reservation = self._reservation
            if (reservation is not None and reservation.target is segment
                    and not reservation.resolved):
                self._defer_boundary_result(boundary, pending, result)
                return
            if (segment.acoustic_receipt_epoch is not None
                    and not self._snapshot_covers_receipt(result.job.snapshot, segment)):
                self._defer_boundary_result(boundary, pending, result)
                self._continue_acoustic_boundary(boundary)
                return
        if result.error_category or result.expired or result.cancelled:
            if pending.boundary is not None:
                boundary = pending.boundary
                self._emit_terminal_rejection(segment,
                                              result.error_category or "verify_failed")
                self._boundary = None
                self._start_overlap_replay(boundary, include_overlap=True)
            else:
                self._emit_terminal_rejection(segment,
                                              result.error_category or "verify_failed")
            return
        try:
            matched = self.verify_text(result.transcript, pending.proposed_phrase,
                                       pending.proposed_mid)
        except Exception:
            matched = None
        if (not isinstance(matched, tuple) or len(matched) != 3
                or not isinstance(matched[0], str) or not isinstance(matched[1], str)
                or not isinstance(matched[2], bool)
                or matched[0].strip().lower() != pending.proposed_phrase):
            if pending.boundary is not None:
                boundary = pending.boundary
                self._emit_terminal_rejection(segment, "verification_refused")
                self._boundary = None
                self._start_overlap_replay(boundary, include_overlap=True)
            else:
                self._emit_terminal_rejection(segment, "verification_refused")
            return
        phrase, command, mid = matched
        if pending.boundary is not None:
            boundary = pending.boundary
            if not command:
                self._emit_wake(segment, phrase, "second_stt", result.finished_at, mid)
                synthetic = _Pending(pending.job, segment, "boundary",
                                     "max_utterance", True, boundary=boundary)
                synthetic.original_transcript = pending.original_transcript
                synthetic.original_latency_ms = pending.original_latency_ms
                self._emit_boundary_utterance(synthetic, result, phrase, mid,
                                              "second_stt")
                self._boundary = None
                self._start_overlap_replay(boundary, include_overlap=False)
            else:
                segment.wake_phrase = phrase
                segment.wake_mid = mid
                segment.wake_path = "second_stt"
                segment.wake_evidence_at = result.finished_at
                self._emit_wake(segment, phrase, "second_stt", result.finished_at, mid)
                self._continue_after_boundary(boundary)
            return
        self._finish_utterance(pending, result, phrase, command, mid, "second_stt",
                               pending.original_transcript, pending.original_latency_ms,
                               confirmation_transcript=result.transcript,
                               confirmation_latency_ms=result.latency_ms)

    def _emit_boundary_utterance(self, pending: _Pending, result: DecodeResult,
                                 phrase: str, mid: bool, mode: str) -> None:
        segment = pending.segment
        snapshot = result.job.snapshot
        if (segment.acoustic_receipt_epoch is not None
                and not self._snapshot_covers_receipt(snapshot, segment)):
            self._emit_terminal_rejection(segment, "acoustic_evidence_not_in_capture")
            return
        utterance = Utterance(np.array(snapshot.audio, copy=True),
                              pending.original_transcript or result.transcript,
                              pending.original_latency_ms or result.latency_ms,
                              completed_at=snapshot.completed_at)
        utterance.processed_audio = (None if snapshot.processed_audio is None else
                                     np.array(snapshot.processed_audio, copy=True))
        utterance.segment_id = segment.segment_id
        utterance.end_reason = "max_utterance"
        utterance.truncated = True
        utterance.confirmed_audio_handoff = segment.confirmed_audio_handoff
        utterance.confirmation_mode = mode
        utterance.confirmation_latency_ms = (
            result.latency_ms if mode == "second_stt" else None)
        utterance.confirmation_transcript = (
            result.transcript if mode == "second_stt" else "")
        utterance.speech_started_at = float(snapshot.metadata.get("speech_started_at", 0.0))
        utterance.trigger_done_at = result.finished_at
        utterance.wake_path = segment.wake_path or "final_text"
        utterance.wake_evidence_at = segment.wake_evidence_at or result.finished_at
        utterance.triggered = True
        utterance.matched_phrase = phrase
        utterance.command = ""
        utterance.mid_utterance = segment.wake_mid or mid
        self._emit_reserved(segment, ListenerEvent(
            "utterance", utterance=utterance, segment_id=segment.segment_id,
            capture_id=self._capture_id(segment)))
        self._release_reservation(segment)

    def _continue_after_boundary(self, boundary: _Boundary, *, min_extension_frames=0) -> None:
        segment = boundary.segment
        continuation = list(boundary.continuation)
        split = boundary.silence_end
        cap = self.extend_frames
        if min_extension_frames:
            if (min_extension_frames > cap or min_extension_frames > len(continuation)
                    or len(continuation) > self.extend_frames + self.pad_frames):
                self._abort_boundary("acoustic_extension_cutoff")
                return
            if split is not None and min_extension_frames <= split <= cap:
                extension, tail = continuation[:split], continuation[split:]
                reason, truncated = "silence", False
            elif len(continuation) >= cap:
                extension, tail = continuation[:cap], continuation[cap:]
                reason, truncated = "extension_limit", True
            elif boundary.silence_run >= self.silence_frames:
                extension, tail = continuation, []
                reason, truncated = "silence", False
            else:
                extension, tail = continuation, []
                reason, truncated = "", False
        elif split is not None and split <= cap:
            extension, tail = continuation[:split], continuation[split:]
            reason, truncated = "silence", False
        elif len(continuation) >= cap:
            extension, tail = continuation[:cap], continuation[cap:]
            reason, truncated = "extension_limit", True
        elif split is not None:
            extension, tail = continuation[:split], continuation[split:]
            reason, truncated = "silence", False
        else:
            extension, tail = continuation, []
            reason = ""
            truncated = False

        segment.extending = True
        segment.extension_frames = len(extension)
        segment.live_frames += len(extension)
        for item in extension:
            segment.records.append(item)
            segment.vad_peak = max(segment.vad_peak, item.probability)
            if item.probability >= self.vad_threshold:
                segment.voiced_frames += 1
                segment.live_voiced_frames += 1
        self._boundary = None
        if reason:
            self._seal_final(segment, reason, truncated=truncated)
            self._queue_replay(tail)
        else:
            segment.silence_run = boundary.silence_run
            self._active = segment

    def _start_overlap_replay(self, boundary: _Boundary, *, include_overlap: bool) -> None:
        segment = boundary.segment
        records = list(boundary.continuation)
        if include_overlap and self.overlap_frames:
            overlap = segment.records[-self.overlap_frames:]
            if overlap:
                seed = _Segment(uuid.uuid4().hex[:16], list(overlap),
                                speech_started_at=overlap[0].captured_at,
                                voiced_frames=sum(item.probability >= self.vad_threshold
                                                  for item in overlap),
                                vad_peak=max(item.probability for item in overlap),
                                overlap_of=segment.segment_id,
                                live_frames=len(overlap),
                                live_voiced_frames=sum(
                                    item.probability >= self.vad_threshold for item in overlap))
                seed.next_peek = self.peek_first
                if self._reserve_for_segment(seed):
                    self._active = seed
                else:
                    records = list(overlap) + records
        self._queue_replay(records)

    def _queue_replay(self, records: list[_FrameRecord]) -> None:
        self._replay.extend(records)
        self._drain_replay()

    def _drain_replay(self) -> None:
        while self._replay:
            record = self._replay[0]
            if self._active is None and record.probability >= self.vad_threshold \
                    and not self._has_capacity(2):
                return
            self._replay.popleft()
            if self._boundary is not None:
                self._feed_boundary(record)
            else:
                self._consume_record(record, replay=True)

    def apply_result(self, result: DecodeResult) -> None:
        """Apply one lane result; stale, malformed, or superseded results are inert."""
        self._claim_owner()
        self._ensure_open()
        if not isinstance(result, DecodeResult) or not isinstance(result.job, DecodeJob):
            return
        acoustic = self._acoustic_pending.get(result.job.job_id)
        if acoustic is not None:
            expected, reservation, stage = acoustic
            if result.job is not expected:
                return
            del self._acoustic_pending[result.job.job_id]
            self._apply_acoustic_result(reservation, stage, result)
            return
        pending = self._pending.get(result.job.job_id)
        if pending is None or result.job is not pending.job:
            return
        del self._pending[result.job.job_id]
        if result.job.stream_epoch != self._stream_epoch:
            return
        if pending.purpose == "peek":
            self._apply_peek(pending, result)
        elif pending.purpose == "boundary":
            self._apply_boundary(pending, result)
        elif pending.purpose == "verify":
            self._apply_verify(pending, result)
        else:
            self._apply_final(pending, result)
        self._drain_replay()

    def poll_events(self) -> tuple[ListenerEvent, ...]:
        self._claim_owner()
        self._ensure_open()
        if self._boundary is not None and self._now() >= self._boundary.deadline:
            self._abort_boundary("boundary_deadline_exceeded")
        poll = getattr(self.lane, "poll", None)
        if callable(poll):
            for result in poll():
                self.apply_result(result)
        self._expire_acoustic_reservation()
        self._drain_replay()
        events = tuple(self._events)
        self._events.clear()
        self._diagnostic_events = 0
        return events

    @property
    def diagnostic_drops(self) -> int:
        return self._diagnostic_drops

    def _expire_acoustic_reservation(self):
        reservation = self._reservation
        if reservation is not None and self._now() >= reservation.deadline:
            self._fail_acoustic_reservation(reservation, "acoustic_deadline_exceeded",
                                            stage="expiry")

    def _clear_frontend(self) -> None:
        self._active = None
        self._boundary = None
        self._pre.clear()
        self._pending.clear()
        self._acoustic_pending.clear()
        self._reservation = None
        self._held_terminal.clear()
        self._events.clear()
        self._reserved_events = 0
        self._replay.clear()
        self._history.clear()
        self._weak_records.clear()
        self._weak_pre.clear()
        self._weak_voiced = self._weak_silence = 0
        self._weak_peak = 0.0
        self._diagnostic_events = 0
        reset = getattr(self.vad, "reset", None)
        if callable(reset):
            reset()

    def flush(self, reason="manual") -> None:
        self._claim_owner()
        self._ensure_open()
        if not isinstance(reason, str):
            raise ValueError("reason must be a string")
        self._stream_epoch += 1
        self.lane.invalidate(self._stream_epoch)
        self._clear_frontend()

    def suspend(self, timeout) -> bool:
        self._claim_owner()
        self._ensure_open()
        suspend = getattr(self.lane, "suspend", None)
        if not callable(suspend):
            raise RuntimeError("lane does not support suspend()")
        result = bool(suspend(timeout))
        self._stream_epoch += 1
        self.lane.invalidate(self._stream_epoch)
        self._clear_frontend()
        self._suspended = True
        return result

    def resume(self) -> bool:
        self._claim_owner()
        self._ensure_open()
        epoch = self._stream_epoch + 1
        resume = getattr(self.lane, "resume", None)
        if not callable(resume) or not resume(epoch):
            return False
        self._stream_epoch = epoch
        self._clear_frontend()
        self._suspended = False
        return True

    def close(self, timeout) -> bool:
        self._claim_owner()
        close = getattr(self.lane, "close", None)
        if not callable(close):
            raise RuntimeError("lane does not support close()")
        if self._closed:
            return bool(close(timeout))
        self._closed = True
        self._stream_epoch += 1
        self.lane.invalidate(self._stream_epoch)
        self._clear_frontend()
        return bool(close(timeout))
