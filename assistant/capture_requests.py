"""Pure, single-owner capture-request state machine.

Callers supply already captured 512-sample frames and cached VAD scores. Source
``captured_at`` values must share the manager clock's monotonic domain and be
no later than that clock. Outcome end times use the owner clock; they are not
evidence of callback timing. The owner must call ``expire`` during frame stalls;
``poll`` only drains results and does not advance timeouts. This module has no
microphone, VAD, ASR, callback, or background-thread integration.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from numbers import Integral, Real
from threading import get_ident
import time
from typing import Optional

import numpy as np


_RATE = 16000
_FRAME = 512
_FRAME_SECONDS = _FRAME / _RATE
_MAX_DURATION = 30.0
_MAX_PADDING_FRAMES = 10
_MAX_SAMPLES = int(_MAX_DURATION * _RATE) + 2 * _MAX_PADDING_FRAMES * _FRAME
_MODES = frozenset(("command", "reply", "confirmation", "dictation"))
_STARTS = frozenset(("immediate", "after_prompt"))
_STATUSES = frozenset(("complete", "no_speech", "start_timeout",
                       "max_duration", "cancelled", "superseded",
                       "device_lost", "gap", "deadline"))


def _number(value, name, *, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("{} must be a finite number".format(name))
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("{} must be a finite number".format(name))
    if minimum is not None and result < minimum:
        raise ValueError("{} is below its minimum".format(name))
    if maximum is not None and result > maximum:
        raise ValueError("{} exceeds its maximum".format(name))
    return result


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError("{} must be an integer >= {}".format(name, minimum))
    return int(value)


def _label(value, name):
    if not isinstance(value, str) or not value or len(value) > 128:
        raise ValueError("{} must be a non-empty string of at most 128 characters".format(name))
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError("{} must not contain control characters".format(name))
    return value


def _owned_pcm(value, name):
    if not isinstance(value, np.ndarray) or value.dtype != np.int16 or value.ndim != 1:
        raise ValueError("{} must be one-dimensional int16 PCM".format(name))
    if value.size < _FRAME or value.size > _MAX_SAMPLES or value.size % _FRAME:
        raise ValueError("{} has an invalid bounded frame count".format(name))
    return np.frombuffer(np.array(value, dtype=np.int16, order="C", copy=True).tobytes(),
                         dtype=np.int16)


@dataclass(frozen=True)
class CaptureRequest:
    request_id: str
    turn_id: str
    generation: int
    mode: str
    start_condition: str
    start_timeout_seconds: float
    max_duration_seconds: float
    deadline: float

    def __post_init__(self):
        _label(self.request_id, "request_id")
        _label(self.turn_id, "turn_id")
        object.__setattr__(self, "generation", _integer(self.generation, "generation"))
        if not isinstance(self.mode, str) or self.mode not in _MODES:
            raise ValueError("unsupported capture mode")
        if not isinstance(self.start_condition, str) or self.start_condition not in _STARTS:
            raise ValueError("unsupported start_condition")
        object.__setattr__(self, "start_timeout_seconds",
                           _number(self.start_timeout_seconds, "start_timeout_seconds",
                                   minimum=0.0, maximum=30.0))
        duration = _number(self.max_duration_seconds, "max_duration_seconds",
                           minimum=_FRAME_SECONDS, maximum=_MAX_DURATION)
        object.__setattr__(self, "max_duration_seconds", duration)
        object.__setattr__(self, "deadline", _number(self.deadline, "deadline"))


@dataclass(frozen=True)
class CaptureAudio:
    source_epoch: int
    start_sample: int
    end_sample: int
    raw: np.ndarray
    processed: Optional[np.ndarray]
    captured_at: float
    completed_at: float
    truncated: bool = False
    gap: bool = False

    def __post_init__(self):
        object.__setattr__(self, "source_epoch", _integer(self.source_epoch, "source_epoch"))
        start = _integer(self.start_sample, "start_sample")
        end = _integer(self.end_sample, "end_sample")
        raw = _owned_pcm(self.raw, "raw")
        if end - start != raw.size:
            raise ValueError("sample interval must match raw PCM length")
        processed = None if self.processed is None else _owned_pcm(self.processed, "processed")
        if processed is not None and processed.size != raw.size:
            raise ValueError("processed PCM must pair with raw PCM")
        if not isinstance(self.truncated, bool) or not isinstance(self.gap, bool):
            raise ValueError("truncated and gap must be bools")
        captured = _number(self.captured_at, "captured_at")
        completed = _number(self.completed_at, "completed_at")
        if completed < captured:
            raise ValueError("completed_at must be >= captured_at")
        object.__setattr__(self, "start_sample", start)
        object.__setattr__(self, "end_sample", end)
        object.__setattr__(self, "raw", raw)
        object.__setattr__(self, "processed", processed)
        object.__setattr__(self, "captured_at", captured)
        object.__setattr__(self, "completed_at", completed)


@dataclass(frozen=True)
class CaptureOutcome:
    request_id: str
    turn_id: str
    generation: int
    mode: str
    status: str
    started_at: Optional[float]
    ended_at: float
    audio: Optional[CaptureAudio] = None

    def __post_init__(self):
        _label(self.request_id, "request_id")
        _label(self.turn_id, "turn_id")
        object.__setattr__(self, "generation", _integer(self.generation, "generation"))
        if (not isinstance(self.mode, str) or self.mode not in _MODES
                or not isinstance(self.status, str) or self.status not in _STATUSES):
            raise ValueError("invalid outcome mode or status")
        ended = _number(self.ended_at, "ended_at")
        started = None if self.started_at is None else _number(self.started_at, "started_at")
        if started is not None and ended < started:
            raise ValueError("ended_at must be >= started_at")
        if self.audio is not None and not isinstance(self.audio, CaptureAudio):
            raise ValueError("audio must be CaptureAudio or None")
        if self.audio is not None and ended < self.audio.completed_at:
            raise ValueError("ended_at must be >= audio completed_at")
        if self.status == "complete":
            if (started is None or self.audio is None or self.audio.raw.size == 0
                    or self.audio.truncated or self.audio.gap):
                raise ValueError("complete outcomes require started, nontruncated contiguous audio")
        if self.audio is not None:
            if self.audio.gap and self.status != "gap":
                raise ValueError("gap audio requires gap status")
            if self.status == "gap" and not self.audio.gap:
                raise ValueError("gap outcomes with audio require gap metadata")
            if self.audio.truncated and self.status not in ("max_duration", "deadline"):
                raise ValueError("truncated audio requires a limit status")
            if self.status in ("max_duration", "deadline") and not self.audio.truncated:
                raise ValueError("limit outcomes with audio require truncation metadata")
        object.__setattr__(self, "started_at", started)
        object.__setattr__(self, "ended_at", ended)


@dataclass(frozen=True)
class Admission:
    status: str
    request_id: str

    @property
    def accepted(self):
        return self.status == "accepted"


class CaptureRequestManager:
    """One foreground request, one owner thread, and reserved terminal slots."""

    def __init__(self, *, clock=time.monotonic, vad_threshold=0.35,
                 min_speech_seconds=0.3, silence_seconds=0.35,
                 padding_seconds=0.25, max_results=8, pre_roll_frames=10):
        if not callable(clock):
            raise ValueError("clock must be callable")
        self.clock = clock
        self.vad_threshold = _number(vad_threshold, "vad_threshold", minimum=0, maximum=1)
        self.min_speech_seconds = _number(min_speech_seconds, "min_speech_seconds",
                                          minimum=0, maximum=_MAX_DURATION)
        self.silence_seconds = _number(silence_seconds, "silence_seconds",
                                       minimum=_FRAME_SECONDS, maximum=_MAX_DURATION)
        pad = _number(padding_seconds, "padding_seconds", minimum=0,
                      maximum=_MAX_PADDING_FRAMES * _FRAME_SECONDS)
        self.pad_frames = int(math.ceil(pad / _FRAME_SECONDS))
        self.max_results = _integer(max_results, "max_results", 1)
        if self.max_results > 32:
            raise ValueError("max_results must be <= 32")
        pre_roll_frames = _integer(pre_roll_frames, "pre_roll_frames")
        if pre_roll_frames > 10:
            raise ValueError("pre_roll_frames must be <= 10")
        self.pre_roll_frames = min(pre_roll_frames, self.pad_frames)
        self.min_frames = int(math.ceil(self.min_speech_seconds / _FRAME_SECONDS))
        self.silence_frames = int(math.ceil(self.silence_seconds / _FRAME_SECONDS))
        self._owner = None
        self._active = None
        self._results = deque()
        self._preroll = deque(maxlen=self.pre_roll_frames)
        self._last_capture_time = float("-inf")

    def _claim(self):
        ident = get_ident()
        if self._owner is None:
            self._owner = ident
        elif self._owner != ident:
            raise RuntimeError("CaptureRequestManager may only be used by its owner thread")

    def _reset_frames(self):
        self._preroll.clear()
        self._last_capture_time = float("-inf")

    def request(self, request):
        self._claim()
        if not isinstance(request, CaptureRequest):
            raise ValueError("request must be CaptureRequest")
        if self._active is not None:
            return Admission("busy", request.request_id)
        if len(self._results) >= self.max_results:
            return Admission("result_backpressure", request.request_id)
        now = _number(self.clock(), "clock")
        self._reset_frames()
        started = now if request.start_condition == "immediate" else None
        self._active = {"request": request, "eligible_at": started,
                        "start_deadline": (None if started is None else
                                           started + request.start_timeout_seconds),
                        "started_at": None, "records": [], "voiced": 0,
                        "silence": 0, "epoch": None, "expected": None,
                        "owned_frames": 0}
        return Admission("accepted", request.request_id)

    def prompt_finished(self, request_id, generation):
        self._claim()
        _label(request_id, "request_id")
        _integer(generation, "generation")
        now = _number(self.clock(), "clock")
        if self._expire_at(now):
            return False
        active = self._matching(request_id, generation)
        if active is None:
            return False
        request = active["request"]
        if request.start_condition != "after_prompt" or active["eligible_at"] is not None:
            return False
        active["eligible_at"] = now
        active["start_deadline"] = now + request.start_timeout_seconds
        return True

    def cancel(self, request_id, generation, *, superseded=False):
        self._claim()
        if not isinstance(superseded, bool):
            raise ValueError("superseded must be bool")
        active = self._matching(request_id, generation)
        if active is None:
            return False
        self._finish("superseded" if superseded else "cancelled")
        return True

    def device_lost(self):
        self._claim()
        if self._active is None:
            self._reset_frames()
            return False
        self._finish("device_lost")
        self._reset_frames()
        return True

    def gap(self, source_epoch=None, start_sample=None):
        self._claim()
        active = self._active
        if active is None:
            self._reset_frames()
            return False
        self._finish("gap", gap=True)
        self._reset_frames()
        return True

    def expire(self):
        self._claim()
        return self._expire_at(_number(self.clock(), "clock"))

    def _expire_at(self, now):
        if self._active is None:
            return False
        request = self._active["request"]
        if now >= request.deadline:
            self._finish("deadline", now=now, truncated=self._active["started_at"] is not None)
            return True
        if (self._active["started_at"] is not None
                and now - self._active["started_at"] >= request.max_duration_seconds):
            self._finish("max_duration", now=now, truncated=True)
            return True
        if (self._active["eligible_at"] is not None and self._active["started_at"] is None
                and now >= self._active["start_deadline"]):
            self._finish("start_timeout", now=now)
            return True
        return False

    def feed(self, source_epoch, start_sample, captured_at, raw_frame,
             processed_frame=None, probability=0.0, *, eligible=True):
        self._claim()
        if not isinstance(eligible, bool):
            raise ValueError("eligible must be bool")
        captured = _number(captured_at, "captured_at")
        owner_now = _number(self.clock(), "clock")
        if captured > owner_now:
            raise ValueError("captured_at must not be later than owner clock")
        if captured < self._last_capture_time:
            raise ValueError("captured_at must be monotonic")
        if not eligible:
            self._last_capture_time = captured
            if self._expire_at(owner_now):
                return False
            active = self._active
            if active is not None and active["started_at"] is not None:
                self._finish("gap", now=owner_now, gap=True)
            else:
                self._preroll.clear()
            return False
        epoch = _integer(source_epoch, "source_epoch")
        start = _integer(start_sample, "start_sample")
        score = _number(probability, "probability", minimum=0, maximum=1)
        raw = self._frame(raw_frame, "raw_frame")
        processed = None if processed_frame is None else self._frame(processed_frame,
                                                                       "processed_frame")
        if processed is not None and processed.size != raw.size:
            raise ValueError("processed_frame must pair with raw_frame")
        self._last_capture_time = captured
        if self._expire_at(owner_now):
            return False
        active = self._active
        if active is None or active["eligible_at"] is None:
            self._preroll.clear()
            return False
        if active["started_at"] is None:
            if score < self.vad_threshold:
                if self._preroll:
                    last = self._preroll[-1]
                    if epoch != last[0] or start != last[1] + _FRAME or captured < last[2]:
                        self._preroll.clear()
                self._preroll.append((epoch, start, captured, raw, processed, score))
                return False
            self._start_capture(active, epoch, start, captured)
            valid_preroll = []
            for prior in reversed(self._preroll):
                if prior[0] != epoch or prior[1] + _FRAME != (valid_preroll[-1][1] if valid_preroll else start):
                    break
                valid_preroll.append(prior)
            valid_preroll.reverse()
            if valid_preroll:
                active["expected"] = valid_preroll[0][1]
            for prior in valid_preroll:
                self._append(active, *prior[:5], prior[5])
            self._preroll.clear()
            self._append(active, epoch, start, captured, raw, processed, score)
            active["owned_frames"] = 1
            return self._after_frame(active, score)

        if epoch != active["epoch"] or start != active["expected"]:
            self._finish("gap", now=owner_now, gap=True)
            self._reset_frames()
            return False
        if captured < active["records"][-1][2]:
            raise ValueError("captured_at must be monotonic")
        self._append(active, epoch, start, captured, raw, processed, score)
        active["owned_frames"] += 1
        return self._after_frame(active, score)

    @staticmethod
    def _frame(value, name):
        if not isinstance(value, np.ndarray) or value.dtype != np.int16 or value.ndim != 1 or value.size != _FRAME:
            raise ValueError("{} must be 512 int16 samples".format(name))
        return np.array(value, dtype=np.int16, order="C", copy=True)

    def _start_capture(self, active, epoch, start, captured):
        active["started_at"] = captured
        active["epoch"] = epoch
        active["expected"] = start
        active["records"] = []
        active["voiced"] = 0
        active["silence"] = 0

    @staticmethod
    def _append(active, epoch, start, captured, raw, processed, score):
        if active["epoch"] is None:
            active["epoch"] = epoch
            active["expected"] = start
        if epoch != active["epoch"] or start != active["expected"]:
            raise RuntimeError("attempted to append noncontiguous capture PCM")
        active["records"].append((epoch, start, captured, raw, processed, score))
        active["expected"] = start + _FRAME

    def _after_frame(self, active, score):
        # _append records a raw frame; voiced count is tracked from cached scores.
        active["voiced"] = sum(item[5] >= self.vad_threshold
                                for item in active["records"])
        active["silence"] = active["silence"] + 1 if score < self.vad_threshold else 0
        request = active["request"]
        frames = len(active["records"])
        elapsed = active["owned_frames"] * _FRAME_SECONDS
        now = self._number_clock()
        if now >= request.deadline:
            self._finish("deadline", now=now, truncated=True)
        elif (elapsed >= request.max_duration_seconds
              or now - active["started_at"] >= request.max_duration_seconds):
            self._finish("max_duration", now=now, truncated=True)
        elif active["silence"] >= self.silence_frames:
            enough = (active["voiced"] >= self.min_frames or frames >= self.min_frames)
            self._finish("complete" if enough else "no_speech", now=now)
        return True

    def _number_clock(self):
        return _number(self.clock(), "clock")

    def _matching(self, request_id, generation):
        _label(request_id, "request_id")
        _integer(generation, "generation")
        active = self._active
        if active is None:
            return None
        request = active["request"]
        if request.request_id != request_id or request.generation != generation:
            return None
        return active

    def _finish(self, status, *, now=None, truncated=False, gap=False):
        active = self._active
        if active is None:
            return
        request = active["request"]
        now = self._number_clock() if now is None else _number(now, "ended_at")
        audio = None
        records = active["records"]
        if records:
            raw = np.concatenate([item[3] for item in records])
            if raw.size > _MAX_SAMPLES:
                raw = raw[-_MAX_SAMPLES:]
                records = records[-(_MAX_SAMPLES // _FRAME):]
            processed = None if any(item[4] is None for item in records) else np.concatenate(
                [item[4] for item in records])
            audio = CaptureAudio(records[0][0], records[0][1],
                                 records[0][1] + raw.size, raw, processed,
                                 records[0][2], now, truncated=truncated, gap=gap)
        self._results.append(CaptureOutcome(
            request.request_id, request.turn_id, request.generation, request.mode,
            status, active["started_at"], now, audio))
        self._active = None
        self._reset_frames()

    def poll(self):
        self._claim()
        results = tuple(self._results)
        self._results.clear()
        return results
