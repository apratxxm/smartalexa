"""Single microphone owner with a bounded, raw, contiguous turn buffer."""
from __future__ import annotations

import collections
import logging
import math
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from .transcription import TimestampedTranscriber, NoWakeBoundary, trim_to_wake


class DeviceFailure(RuntimeError):
    """A microphone read failed; the supervisor must restart the process."""


class EndReason(str, Enum):
    ENDPOINT = "endpoint"
    HARD_LIMIT = "hard_limit"
    OVERFLOW = "overflow"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class CaptureResult:
    turn_id: str
    pcm: np.ndarray
    preroll_ms: int
    end_reason: EndReason
    continuous: bool | None
    truncated: bool
    wake_score: float = 0.0
    speech_end: float = 0.0
    wake_at: float = 0.0
    timings: dict = field(default_factory=dict)
    audio_started_at: float | None = None

    @property
    def complete(self):
        return (self.end_reason == EndReason.ENDPOINT
                and self.continuous is True and not self.truncated)


def admissible(result, current_turn_id):
    return bool(result and result.turn_id == current_turn_id and result.complete)


def _health(mic):
    values = tuple(getattr(mic, name, None) for name in
                   ("queue_overflow_count", "_callback_status_count", "intentional_discard_count"))
    return values if all(type(value) is int for value in values) else None


class _Stream:
    """Endpointer facade: preserve EVERY read, including frames it trims away."""
    def __init__(self, mic, frame_length, sample_rate, clock):
        self.mic, self.frame_length, self.sample_rate, self.clock = mic, frame_length, sample_rate, clock
        self.history = collections.deque(maxlen=math.ceil(sample_rate / frame_length) + 1)
        self.recording = None
        self.judged = None
        self.last_read_at = 0.0

    def read(self, timeout_s=3.0):
        before = _health(self.mic)
        try:
            raw = self.mic.read(timeout_s=timeout_s)
        except Exception as exc:
            raise DeviceFailure(type(exc).__name__) from exc
        after = _health(self.mic)
        if not isinstance(raw, np.ndarray) or raw.dtype != np.int16 or raw.shape != (self.frame_length,):
            raise DeviceFailure("invalid microphone frame")
        clean = getattr(self.mic, "last_clean", None)
        self.judged = (clean if getattr(self.mic, "echo", None) is not None
                       and isinstance(clean, np.ndarray) and clean.shape == raw.shape else raw)
        self.last_read_at = self.clock()
        item = (raw.copy(), before, after)
        self.history.append(item)
        if self.recording is not None:
            self.recording.append(item)
        return raw

    def vad_frame(self, raw):
        return self.judged if self.judged is not None else raw


class _TimedVAD:
    def __init__(self, vad, stream, threshold=0.5):
        self.vad, self.stream, self.threshold = vad, stream, threshold
        self.last_voice_at = None

    def reset(self):
        self.last_voice_at = None
        self.vad.reset()

    def __call__(self, frame):
        score = self.vad(frame)
        if score >= self.threshold:
            self.last_voice_at = self.stream.last_read_at
        return score


class CaptureRuntime:
    def __init__(self, cfg, *, mic_factory=None, wake_factory=None, vad_factory=None,
                 endpointer_factory=None, stt_factory=None, echo_factory=None,
                 clock=time.monotonic):
        self.cfg, self.clock = cfg, clock
        self._factories = mic_factory, wake_factory, vad_factory, endpointer_factory, stt_factory, echo_factory
        self.mic = self.wake = self.vad = self.stt = self.echo_ref = self.stream = None
        self.turn_id, self.wake_score, self.wake_at = "", 0.0, 0.0
        self.trimmed_prefix_length = None

    def __enter__(self):
        from ..audio import Microphone, SileroVAD, Endpointer
        from ..wake import WakeDetector
        from ..echo import SpeakerReference, EchoCanceller
        mf, wf, vf, ef, sf, xf = self._factories
        rate, frame = self.cfg.get("audio.sample_rate", 16000), self.cfg.get("audio.frame_length", 512)
        try:
            self.wake = (wf or WakeDetector)(self.cfg)
            self.stt = (sf or TimestampedTranscriber)(self.cfg.get("stt.model"), rate)
            self.mic = (mf or Microphone)(
                sample_rate=rate, frame_length=frame,
                ring_seconds=self.cfg.get("audio.ring_buffer_seconds", 30),
                device=self.cfg.get("audio.input_device"),
                capture_rate=self.cfg.get("audio.capture_rate"),
                exclusive=self.cfg.get("audio.exclusive_mode", False))
            self.stream = _Stream(self.mic, frame, rate, self.clock)
            self.vad = _TimedVAD((vf or SileroVAD)(rate), self.stream)
            self.endpointer = (ef or Endpointer)(
                self.vad, sample_rate=rate, frame_length=frame,
                max_seconds=float(self.cfg.get("core.capture.max_seconds", 8)))
            if self.cfg.get("audio.echo_cancellation", True):
                ref_cls, echo_cls = xf or (SpeakerReference, EchoCanceller)
                self.echo_ref = ref_cls(self.cfg.get("audio.echo_reference", ""))
                self.echo_ref.start()
                self.mic.echo = echo_cls(self.echo_ref)
            self.mic.__enter__()
            return self
        except Exception:
            self.close()
            raise

    def close(self):
        operations = [(self.mic, lambda obj: obj.__exit__(None, None, None)),
                      (self.echo_ref, lambda obj: obj.stop()), (self.wake, lambda obj: obj.close())]
        for obj, close in operations:
            if obj is None:
                continue
            def finish(target=obj, action=close):
                try:
                    action(target)
                except Exception:
                    logging.exception("Core resource shutdown failed")
            thread = threading.Thread(target=finish, daemon=True)
            thread.start()
            thread.join(timeout=1.0)
            if thread.is_alive():
                logging.error("Core resource shutdown exceeded one second: %s", type(obj).__name__)
        self.mic = self.echo_ref = self.wake = self.stt = self.stream = self.vad = self.endpointer = None

    def __exit__(self, *_):
        self.close()

    def reset(self):
        try:
            self.mic.flush()
        except Exception as exc:
            raise DeviceFailure(type(exc).__name__) from exc
        self.stream.history.clear()
        self.stream.recording = None
        self.wake.reset()
        self.vad.reset()
        self.turn_id = ""

    def next_capture(self, acknowledge=None):
        self.turn_id = ""
        self.stream.history.clear()
        self.stream.recording = None
        while True:
            raw = self.stream.read()
            if self.wake.process(self.stream.vad_frame(raw)):
                break
        self.turn_id, self.wake_at = uuid.uuid4().hex, self.clock()
        self.wake_score = float(getattr(self.wake, "last_score", 0.0))
        pre = list(self.stream.history)
        self.wake.reset()
        if acknowledge is not None:
            acknowledge()
        ack_ms = (self.clock() - self.wake_at) * 1000 if acknowledge is not None else None
        self.stream.recording = []
        try:
            endpoint = self.endpointer.record(self.stream, after_prompt=False)
            frames = pre + self.stream.recording
        finally:
            self.stream.recording = None
        audio = np.concatenate([item[0] for item in frames])
        states = [state for item in frames for state in item[1:]] + [_health(self.mic)]
        preroll_ms = pre and sum(item[0].size for item in pre) * 1000 / self.stream.sample_rate
        known = bool(states) and all(state is not None for state in states)
        continuous = True if known and all(state == states[0] for state in states) else None
        overflow = known and any(state[0] != states[0][0] for state in states)
        # No assertion of continuity if the full second BEFORE the hit is absent.
        if len(pre) < self.stream.history.maxlen:
            continuous = None
        if overflow:
            reason, continuous = EndReason.OVERFLOW, False
        elif endpoint.get("truncated") or endpoint.get("end_reason") in {"capture_limit", "utterance_limit"}:
            reason = EndReason.HARD_LIMIT
        elif endpoint.get("end_reason") == "silence" and endpoint.get("speech_started"):
            reason = EndReason.ENDPOINT
        else:
            reason = EndReason.CANCELLED
        speech_end = self.vad.last_voice_at
        return CaptureResult(
            self.turn_id, audio, int(preroll_ms), reason, continuous,
            reason in {EndReason.HARD_LIMIT, EndReason.OVERFLOW}, self.wake_score,
            speech_end or self.clock(), self.wake_at,
            {"wake_to_ack_ms": ack_ms,
             "speech_end": None if speech_end is None else (speech_end - self.wake_at) * 1000},
            # Detection consumed the last pre-roll frame. Anchor the contiguous
            # PCM sample timeline to that detector instant on the same clock.
            audio_started_at=self.wake_at - preroll_ms / 1000)

    def transcribe(self, result):
        if not admissible(result, self.turn_id):
            raise ValueError("incomplete or stale capture")
        self.trimmed_prefix_length = None
        timed = self.stt.transcribe(result.pcm)
        try:
            text, self.trimmed_prefix_length = trim_to_wake(
                timed, self.cfg.get("listener.aliases", {}), wake_at=result.wake_at,
                audio_started_at=result.audio_started_at)
        except NoWakeBoundary as exc:
            self.trimmed_prefix_length = exc.prefix_length
            raise
        return text, timed.latency_ms
