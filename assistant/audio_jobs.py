"""Bounded, serialized ASR jobs detached from audio capture and policy.

This module owns no recognizers, devices, callbacks, or model lifecycle. The
caller supplies already-created recognizer callables and explicitly starts one
worker when it is ready to schedule decoding.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
from numbers import Integral, Real
from threading import Condition, Thread
import time
from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional

import numpy as np


_PURPOSES = frozenset(("peek", "boundary", "final", "verify",
                       "acoustic_listener", "acoustic_verify"))
_URGENT_PURPOSES = frozenset(("boundary", "final", "verify",
                              "acoustic_listener", "acoustic_verify"))
_MAX_PCM_SAMPLES = 30 * 16000
_MAX_METADATA_DEPTH = 8
_MAX_METADATA_NODES = 256
_MAX_METADATA_KEY_LENGTH = 128
_MAX_METADATA_STRING_LENGTH = 16384
_MAX_PRIORITY_PENDING = 8
_MAX_RESULTS = 32


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError("{} must be an integer >= {}".format(name, minimum))
    return int(value)


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("{} must be a finite number".format(name))
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("{} must be a finite number".format(name))
    return result


def _validate_pcm(value: Any, name: str) -> None:
    if not isinstance(value, np.ndarray) or value.ndim != 1 or value.dtype != np.int16:
        raise ValueError("{} must be one-dimensional int16 mono PCM".format(name))
    if value.size == 0 or value.size > _MAX_PCM_SAMPLES:
        raise ValueError("{} must contain 1 to {} samples".format(name, _MAX_PCM_SAMPLES))


def _owned_pcm(value: np.ndarray) -> np.ndarray:
    # An immutable bytes base prevents callers from re-enabling WRITEABLE on
    # the returned array, unlike a read-only view over caller-owned memory.
    contiguous = np.array(value, dtype=np.int16, order="C", copy=True)
    return np.frombuffer(contiguous.tobytes(order="C"), dtype=np.int16)


def _freeze_metadata(value: Any, path: str = "metadata", depth: int = 0,
                     nodes: Optional[list[int]] = None) -> Any:
    if nodes is None:
        nodes = [0]
    if depth > _MAX_METADATA_DEPTH:
        raise ValueError("{} exceeds maximum metadata depth".format(path))
    nodes[0] += 1
    if nodes[0] > _MAX_METADATA_NODES:
        raise ValueError("metadata exceeds maximum node count")
    if isinstance(value, bool):
        return value
    if isinstance(value, Integral):
        result = int(value)
        if result.bit_length() > 64:
            raise ValueError("{} integer is outside the safe metadata range".format(path))
        return result
    if isinstance(value, Real):
        return _finite_number(value, path)
    if isinstance(value, str):
        if len(value) > _MAX_METADATA_STRING_LENGTH:
            raise ValueError("{} exceeds maximum metadata string length".format(path))
        return value
    if isinstance(value, Mapping):
        frozen = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("{} keys must be strings".format(path))
            if len(key) > _MAX_METADATA_KEY_LENGTH:
                raise ValueError("{} key exceeds maximum metadata key length".format(path))
            safe_key = _freeze_metadata(key, "{}.key".format(path), depth + 1, nodes)
            frozen[safe_key] = _freeze_metadata(item, "{}.{}".format(path, key),
                                                depth + 1, nodes)
        return MappingProxyType(frozen)
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_metadata(item, "{}[]".format(path), depth + 1, nodes)
                     for item in value)
    raise ValueError("{} contains an unsupported value".format(path))


@dataclass(frozen=True)
class AudioSnapshot:
    """Owned, immutable PCM and capture provenance for one segment revision."""

    stream_epoch: int
    segment_id: str
    revision: int
    start_sample: int
    end_sample: int
    captured_at: float
    completed_at: float
    audio: np.ndarray
    processed_audio: Optional[np.ndarray] = None
    end_reason: str = ""
    truncated: bool = False
    overlap_of: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "stream_epoch", _integer(self.stream_epoch, "stream_epoch"))
        object.__setattr__(self, "revision", _integer(self.revision, "revision"))
        start = _integer(self.start_sample, "start_sample")
        end = _integer(self.end_sample, "end_sample")
        if end < start:
            raise ValueError("end_sample must be >= start_sample")
        if not isinstance(self.segment_id, str) or not self.segment_id:
            raise ValueError("segment_id must be a non-empty string")
        if not isinstance(self.end_reason, str) or not isinstance(self.overlap_of, str):
            raise ValueError("end_reason and overlap_of must be strings")
        if not isinstance(self.truncated, bool):
            raise ValueError("truncated must be a bool")
        captured = _finite_number(self.captured_at, "captured_at")
        completed = _finite_number(self.completed_at, "completed_at")
        if completed < captured:
            raise ValueError("completed_at must be >= captured_at")
        _validate_pcm(self.audio, "audio")
        if end - start != self.audio.size:
            raise ValueError("sample span must equal audio length")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be a mapping")
        metadata = _freeze_metadata(self.metadata)
        if self.processed_audio is not None:
            _validate_pcm(self.processed_audio, "processed_audio")
            if self.processed_audio.size != self.audio.size:
                raise ValueError("processed_audio must align with audio")
            processed = _owned_pcm(self.processed_audio)
        else:
            processed = None
        audio = _owned_pcm(self.audio)
        object.__setattr__(self, "start_sample", start)
        object.__setattr__(self, "end_sample", end)
        object.__setattr__(self, "captured_at", captured)
        object.__setattr__(self, "completed_at", completed)
        object.__setattr__(self, "audio", audio)
        object.__setattr__(self, "processed_audio", processed)
        object.__setattr__(self, "metadata", metadata)


@dataclass(frozen=True)
class DecodeJob:
    """A bounded request naming a recognizer by key, never by model object."""

    job_id: str
    stream_epoch: int
    purpose: str
    snapshot: AudioSnapshot
    deadline: float
    recognizer: str

    def __post_init__(self) -> None:
        if not isinstance(self.job_id, str) or not self.job_id:
            raise ValueError("job_id must be a non-empty string")
        object.__setattr__(self, "stream_epoch", _integer(self.stream_epoch, "stream_epoch"))
        if not isinstance(self.purpose, str) or self.purpose not in _PURPOSES:
            raise ValueError("purpose must be one of {}".format(", ".join(sorted(_PURPOSES))))
        if not isinstance(self.snapshot, AudioSnapshot):
            raise ValueError("snapshot must be an AudioSnapshot")
        if self.snapshot.stream_epoch != self.stream_epoch:
            raise ValueError("job and snapshot stream_epoch must match")
        object.__setattr__(self, "deadline", _finite_number(self.deadline, "deadline"))
        if not isinstance(self.recognizer, str) or not self.recognizer:
            raise ValueError("recognizer must be a non-empty key string")


@dataclass(frozen=True)
class DecodeResult:
    """A terminal outcome for a job; stale-epoch results are intentionally suppressed."""

    job: DecodeJob
    transcript: str = ""
    latency_ms: int = 0
    error_category: str = ""
    finished_at: float = 0.0
    expired: bool = False
    cancelled: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.job, DecodeJob):
            raise ValueError("job must be a DecodeJob")
        if not isinstance(self.transcript, str):
            raise ValueError("transcript must be a string")
        object.__setattr__(self, "latency_ms", _integer(self.latency_ms, "latency_ms"))
        if not isinstance(self.error_category, str):
            raise ValueError("error_category must be a string")
        object.__setattr__(self, "finished_at", _finite_number(self.finished_at, "finished_at"))
        if not isinstance(self.expired, bool) or not isinstance(self.cancelled, bool):
            raise ValueError("expired and cancelled must be bools")


class SerializedAsrLane:
    """One daemon ASR worker with a bounded urgent FIFO, replaceable peek, and results.

    Results reserve capacity at admission. `invalidate` suppresses all older
    stream epochs; callers intentionally discard those outcomes when changing
    the stream generation. `suspend` instead publishes cancellation results
    for queued work and waits, with a real bounded timeout, for in-flight ASR.
    """

    def __init__(self, recognizers: Mapping[str, Callable[[np.ndarray], tuple]], *,
                 max_priority_pending: int = 2, max_results: int = 8,
                 clock: Callable[[], float] = time.monotonic) -> None:
        if not isinstance(recognizers, Mapping) or not recognizers:
            raise ValueError("recognizers must be a non-empty mapping")
        copied = dict(recognizers)
        if any(not isinstance(key, str) or not key or not callable(fn)
               for key, fn in copied.items()):
            raise ValueError("recognizers must map non-empty string keys to callables")
        if not callable(clock):
            raise ValueError("clock must be callable")
        self.max_priority_pending = _integer(max_priority_pending, "max_priority_pending", minimum=1)
        self.max_results = _integer(max_results, "max_results", minimum=1)
        if self.max_priority_pending > _MAX_PRIORITY_PENDING:
            raise ValueError("max_priority_pending must be <= {}".format(_MAX_PRIORITY_PENDING))
        if self.max_results > _MAX_RESULTS:
            raise ValueError("max_results must be <= {}".format(_MAX_RESULTS))
        self._recognizers = copied
        self._clock = clock
        self._condition = Condition()
        self._urgent: deque[DecodeJob] = deque()
        self._peek: Optional[DecodeJob] = None
        self._results: deque[DecodeResult] = deque()
        self._active_ids: set[str] = set()
        self._stream_epoch: Optional[int] = None
        self._inflight: Optional[DecodeJob] = None
        self._thread: Optional[Thread] = None
        self._closed = False
        self._suspended = False

    def start(self) -> bool:
        """Start the sole worker; construction itself never creates a thread."""
        with self._condition:
            if self._closed:
                return False
            if self._thread is not None:
                return self._thread.is_alive()
            self._thread = Thread(target=self._worker, name="memoir-asr-lane", daemon=True)
            self._thread.start()
            return True

    def submit(self, job: DecodeJob) -> bool:
        """Admit a job without waiting; False means no capacity/authority."""
        if not isinstance(job, DecodeJob):
            raise ValueError("job must be a DecodeJob")
        if job.recognizer not in self._recognizers:
            raise ValueError("unknown recognizer key: {}".format(job.recognizer))
        with self._condition:
            if self._closed or self._suspended or self._thread is None:
                return False
            if self._stream_epoch is None:
                self._stream_epoch = job.stream_epoch
            if job.stream_epoch != self._stream_epoch:
                return False
            if job.job_id in self._active_ids:
                raise ValueError("job_id is already active: {}".format(job.job_id))
            if len(self._active_ids) >= self.max_results:
                return False
            if job.purpose in _URGENT_PURPOSES:
                if len(self._urgent) >= self.max_priority_pending:
                    return False
                self._urgent.append(job)
            elif self._peek is None:
                self._peek = job
            else:
                # The old peek keeps its reserved result slot as an explicit
                # superseded outcome; replacement itself needs another slot.
                old = self._peek
                superseded = DecodeResult(
                    old, error_category="superseded", finished_at=self._now(),
                    cancelled=True)
                self._peek = job
                self._active_ids.add(job.job_id)
                self._results.append(superseded)
                self._condition.notify()
                return True
            self._active_ids.add(job.job_id)
            self._condition.notify()
            return True

    def poll(self) -> tuple[DecodeResult, ...]:
        """Drain currently available terminal results without blocking."""
        with self._condition:
            results = tuple(self._results)
            self._results.clear()
            for result in results:
                self._active_ids.discard(result.job.job_id)
            return results

    def invalidate(self, stream_epoch: int) -> None:
        """Advance the current stream epoch and discard queued/results from older epochs."""
        epoch = _integer(stream_epoch, "stream_epoch")
        with self._condition:
            if self._stream_epoch is not None and epoch < self._stream_epoch:
                raise ValueError("stream_epoch cannot move backwards")
            self._stream_epoch = epoch
            kept = deque()
            for job in self._urgent:
                if job.stream_epoch < epoch:
                    self._active_ids.discard(job.job_id)
                else:
                    kept.append(job)
            self._urgent = kept
            if self._peek is not None and self._peek.stream_epoch < epoch:
                self._active_ids.discard(self._peek.job_id)
                self._peek = None
            kept_results = deque()
            for result in self._results:
                if result.job.stream_epoch < epoch:
                    self._active_ids.discard(result.job.job_id)
                else:
                    kept_results.append(result)
            self._results = kept_results
            self._condition.notify_all()

    def suspend(self, timeout: float) -> bool:
        """Reject submissions, cancel queued work, and wait boundedly for ASR."""
        timeout = _finite_number(timeout, "timeout")
        if timeout < 0:
            raise ValueError("timeout must be >= 0")
        end = time.monotonic() + timeout
        with self._condition:
            if self._closed:
                return self._inflight is None
            self._suspended = True
            self._cancel_queued_locked("suspended")
            while self._inflight is not None:
                remaining = end - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True

    def resume(self, epoch: int) -> bool:
        """Resume only after in-flight work is gone, discarding prior epoch state."""
        epoch = _integer(epoch, "epoch")
        with self._condition:
            if self._closed or self._inflight is not None:
                return False
            if self._stream_epoch is not None and epoch < self._stream_epoch:
                raise ValueError("epoch cannot move backwards")
            self._cancel_queued_locked("resumed")
            self._results.clear()
            self._active_ids.clear()
            self._stream_epoch = epoch
            self._suspended = False
            self._condition.notify_all()
            return True

    def close(self, timeout: float) -> bool:
        """Close and boundedly join; False truthfully means the worker is still busy."""
        timeout = _finite_number(timeout, "timeout")
        if timeout < 0:
            raise ValueError("timeout must be >= 0")
        with self._condition:
            self._closed = True
            self._suspended = True
            self._cancel_queued_locked("closed")
            thread = self._thread
            self._condition.notify_all()
        if thread is None:
            return True
        thread.join(timeout)
        return not thread.is_alive()

    def _cancel_queued_locked(self, category: str) -> None:
        now = self._now()
        jobs = list(self._urgent)
        self._urgent.clear()
        if self._peek is not None:
            jobs.append(self._peek)
            self._peek = None
        for job in jobs:
            # Every accepted job holds an active-ID/result reservation, so
            # these terminal outcomes fit without evicting another result.
            self._results.append(DecodeResult(
                job, error_category=category, finished_at=now, cancelled=True))

    def _now(self) -> float:
        return _finite_number(self._clock(), "clock result")

    def _safe_now(self) -> float:
        try:
            return self._now()
        except Exception:
            return time.monotonic()

    def _worker(self) -> None:
        while True:
            with self._condition:
                while not self._closed and (self._suspended or
                                            (not self._urgent and self._peek is None)):
                    self._condition.wait()
                if self._closed and not self._urgent and self._peek is None:
                    return
                if self._suspended:
                    continue
                job = self._urgent.popleft() if self._urgent else self._peek
                if job is None:
                    continue
                if job.purpose == "peek":
                    self._peek = None
                self._inflight = job

            result = self._execute(job)
            with self._condition:
                self._inflight = None
                if (self._stream_epoch is not None
                        and job.stream_epoch < self._stream_epoch):
                    # Invalidation supersedes any late completion; its reserved
                    # slot is released without publishing stale authority.
                    self._active_ids.discard(job.job_id)
                else:
                    self._results.append(result)
                self._condition.notify_all()

    def _execute(self, job: DecodeJob) -> DecodeResult:
        try:
            started = self._now()
        except Exception:
            return DecodeResult(job, error_category="clock_error",
                                finished_at=time.monotonic())
        if started >= job.deadline:
            return DecodeResult(job, error_category="deadline_exceeded",
                                finished_at=started, expired=True)
        try:
            output = self._recognizers[job.recognizer](job.snapshot.audio)
            failed = False
        except Exception:
            output = None
            failed = True
        try:
            finished = self._now()
        except Exception:
            return DecodeResult(job, error_category="clock_error",
                                finished_at=time.monotonic())
        elapsed_ms = max(0, int((finished - started) * 1000))
        if finished >= job.deadline:
            latency = (int(output[1]) if not failed and isinstance(output, tuple)
                       and len(output) == 2 and isinstance(output[1], Integral)
                       and not isinstance(output[1], bool) and output[1] >= 0
                       else elapsed_ms)
            return DecodeResult(job, latency_ms=latency,
                                error_category="deadline_exceeded", finished_at=finished,
                                expired=True)
        if failed:
            return DecodeResult(job, latency_ms=elapsed_ms,
                                error_category="decode_error", finished_at=finished)
        if (not isinstance(output, tuple) or len(output) != 2
                    or not isinstance(output[0], str)
                    or isinstance(output[1], bool)
                    or not isinstance(output[1], Integral)
                    or output[1] < 0):
            return DecodeResult(job, latency_ms=elapsed_ms,
                                error_category="invalid_result", finished_at=finished)
        latency = int(output[1])
        return DecodeResult(job, transcript=output[0], latency_ms=latency,
                            finished_at=finished)
