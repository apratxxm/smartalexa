"""Small lifecycle facade for the opt-in asynchronous capture path."""
from __future__ import annotations

import logging
from numbers import Real
import math

from .async_listener import AsyncSttListener
from .audio_jobs import SerializedAsrLane
from .diagnostic_archive import QueuedDiagnosticArchive

log = logging.getLogger("memoir.async_runtime")


def validate_async_config(cfg) -> bool:
    """Validate unsupported async combinations before model or worker startup."""
    enabled = cfg.get("listener.async_enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("listener.async_enabled must be a boolean")
    if not enabled:
        return False
    mode = cfg.get("listener.mode", "stt")
    if mode != "stt":
        raise ValueError("listener.async_enabled requires listener.mode='stt'")
    if cfg.get("audio.paused_input_device", ""):
        raise ValueError("listener.async_enabled does not support audio.paused_input_device")
    if cfg.get("ducking.duck_trigger", "wakeword") == "speech":
        raise ValueError("listener.async_enabled does not support speech-triggered ducking")
    if cfg.get("ducking.pause_on_high_confidence", False):
        raise ValueError("listener.async_enabled does not support pause_on_high_confidence")
    return True


class _ObservedVad:
    """Record the one frontend VAD result without introducing a second pass."""

    def __init__(self, vad):
        self.vad = vad
        self.last_probability = None

    def __call__(self, frame):
        probability = self.vad(frame)
        if (isinstance(probability, bool) or not isinstance(probability, Real)
                or not math.isfinite(float(probability))):
            self.last_probability = None
        else:
            self.last_probability = float(probability)
        return probability

    def reset(self):
        self.last_probability = None
        reset = getattr(self.vad, "reset", None)
        if callable(reset):
            reset()


class AsyncCaptureRuntime:
    """Own one async frontend/lane/archive set and expose bounded transitions."""

    HANDOFF_TIMEOUT_SECONDS = 0.25
    CLOSE_TIMEOUT_SECONDS = 0.75

    def __init__(self, cfg, vad, recognizers, verify_text, telemetry, *,
                 lane_factory=SerializedAsrLane,
                 frontend_factory=AsyncSttListener,
                 archive_factory=QueuedDiagnosticArchive):
        self.observed_vad = _ObservedVad(vad)
        self.lane = lane_factory(recognizers)
        self.frontend = frontend_factory(
            cfg, self.observed_vad, self.lane, verify_text=verify_text)
        self.archive = archive_factory(telemetry)
        self.accepting = False
        # ``accepting`` is a feed policy. Foreground code may borrow the VAD
        # only after the async frontend confirms it has stopped using it.
        self.foreground_authorized = False
        self.started = False
        self.closed = False
        self.counters = {
            "feed_backpressure": 0,
            "feed_while_suspended": 0,
            "fenced_frames": 0,
            "paused_frames": 0,
            "handoff_attempts": 0,
            "completed_captures_discarded": 0,
            "archive_dropped": 0,
            "frontend_diagnostic_drops": 0,
            "suspend_unacknowledged": 0,
            "resume_unavailable": 0,
        }
        self._reported_frontend_drops = 0
        self.close_status = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close_status = self.close()
        if not all(self.close_status.values()):
            log.error("Async shutdown incomplete: lane_closed=%s archive_closed=%s",
                      self.close_status["lane_closed"],
                      self.close_status["archive_closed"])
        return False

    @property
    def last_vad_probability(self):
        return self.observed_vad.last_probability

    def start(self):
        self.foreground_authorized = False
        if self.closed:
            raise RuntimeError("async capture runtime is closed")
        if self.started:
            return True
        try:
            lane_started = bool(self.lane.start())
        except Exception:
            self._cleanup_failed_start()
            raise
        if not lane_started:
            self._cleanup_failed_start()
            raise RuntimeError("async ASR lane did not start")
        try:
            self.archive.start()
        except Exception:
            self._cleanup_failed_start()
            raise
        self.started = True
        self.accepting = True
        return True

    def _cleanup_failed_start(self):
        self.accepting = False
        self.closed = True
        try:
            lane_closed = bool(self.lane.close(self.CLOSE_TIMEOUT_SECONDS))
        except Exception:
            lane_closed = False
        try:
            archive_closed = bool(self.archive.close(self.CLOSE_TIMEOUT_SECONDS))
        except Exception:
            archive_closed = False
        self.close_status = {"lane_closed": lane_closed,
                             "archive_closed": archive_closed}
        if not all(self.close_status.values()):
            log.error("Async startup cleanup incomplete: lane_closed=%s archive_closed=%s",
                      lane_closed, archive_closed)

    def feed(self, frame, processed_frame, *, captured_at, authorization,
             acoustic_proposal=None):
        self.foreground_authorized = False
        if not self.accepting or self.closed:
            self.counters["feed_while_suspended"] += 1
            return False
        try:
            self.frontend.feed_async(
                frame, processed_frame, captured_at=captured_at,
                authorization=authorization, acoustic_proposal=acoustic_proposal)
        except BufferError:
            self.counters["feed_backpressure"] += 1
            raise
        return True

    def poll_events(self):
        if not self.started or self.closed:
            return ()
        # The frontend is the sole poller and result drainer for the lane.
        events = self.frontend.poll_events()
        dropped = int(getattr(self.frontend, "diagnostic_drops", 0) or 0)
        self.counters["frontend_diagnostic_drops"] = dropped
        if dropped > self._reported_frontend_drops:
            log.warning("Async frontend dropped %d diagnostic event(s)",
                        dropped - self._reported_frontend_drops)
            self._reported_frontend_drops = dropped
        return events

    def suspend(self, timeout=None):
        self.foreground_authorized = False
        if not self.started or self.closed:
            return False
        self.accepting = False
        timeout = (self.HANDOFF_TIMEOUT_SECONDS if timeout is None else timeout)
        acknowledged = self.frontend.suspend(timeout)
        self.foreground_authorized = bool(acknowledged and self.started and not self.closed)
        if not acknowledged:
            self.counters["suspend_unacknowledged"] += 1
        return acknowledged

    def resume_if_quiescent(self):
        self.foreground_authorized = False
        if not self.started or self.closed or self.accepting:
            return self.accepting
        if self.frontend.resume():
            self.accepting = True
            return True
        self.counters["resume_unavailable"] += 1
        return False

    def flush(self, reason):
        if self.started and not self.closed and self.accepting:
            self.frontend.flush(reason)

    def queue_diagnostic(self, event, *, input_device="laptop"):
        snapshot = getattr(event, "diagnostic_snapshot", None)
        if snapshot is None:
            return False
        accepted = self.archive.submit(
            snapshot, event.reason or "async_diagnostic",
            stage=event.diagnostic_stage or "listener",
            capture_id=event.capture_id or "",
            proposal_id=event.proposal_id or "",
            transcript=event.trigger_transcript or "",
            second_stt_transcript=event.second_stt_transcript or "",
            latency_ms=event.trigger_stt_latency_ms,
            second_latency_ms=event.second_stt_latency_ms,
            input_device=input_device)
        if not accepted:
            self.counters["archive_dropped"] += 1
            log.warning("Async diagnostic archive queue dropped capture %s at %s",
                        event.capture_id or event.segment_id, event.diagnostic_stage)
        return accepted

    def close(self, timeout=None):
        self.foreground_authorized = False
        self.accepting = False
        self.closed = True
        timeout = self.CLOSE_TIMEOUT_SECONDS if timeout is None else timeout
        try:
            lane_closed = self.frontend.close(timeout)
        except Exception:
            lane_closed = False
        try:
            archive_closed = self.archive.close(timeout)
        except Exception:
            archive_closed = False
        self.close_status = {"lane_closed": bool(lane_closed),
                             "archive_closed": bool(archive_closed)}
        return dict(self.close_status)
