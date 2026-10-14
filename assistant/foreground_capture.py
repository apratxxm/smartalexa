"""Blocking, app-owned adapter from the microphone to CaptureRequestManager.

The manager owns capture semantics; this driver only serializes one foreground
request, acquires frames, and returns legacy Endpointer-shaped results. Frame
timestamps are the owner clock at receipt, not microphone callback timestamps.
The manager does not retain exact VAD onset offsets, so complete results report
speech_start_ms=0 as a conservative PCM-relative boundary. Sample indices and
source epochs are driver-relative transitional metadata, not device coordinates.
"""
from __future__ import annotations

import math
import threading
import time
import uuid

import numpy as np

from .capture_requests import CaptureRequest, CaptureRequestManager


_FRAME_SAMPLES = 512
_FRAME_SECONDS = _FRAME_SAMPLES / 16000.0
_MAX_DURATION_SECONDS = 30.0
_READ_POLL_SECONDS = 0.05


class ForegroundCaptureDriver:
    """Single-thread foreground microphone capture; no retries or model work."""

    def __init__(self, vad, *, clock=time.monotonic):
        if not callable(clock):
            raise ValueError("clock must be callable")
        self.vad = vad
        self.clock = clock
        self.manager = CaptureRequestManager(
            clock=clock, min_speech_seconds=0.032, padding_seconds=0.25,
        )
        self._source_epoch = 0
        self._running = False
        self._run_lock = threading.Lock()

    def record(self, mic, *, mode, turn_id, generation, start_timeout_seconds,
               max_duration_seconds, deadline, threshold, silence_seconds,
               cancelled=None) -> dict:
        """Record one already-admitted foreground request.

        ``deadline`` is an absolute value in the driver's monotonic clock
        domain. A non-complete terminal outcome always returns empty PCM.
        """
        if not self._run_lock.acquire(blocking=False):
            raise RuntimeError("foreground capture driver is already recording")
        self._running = True
        try:
            return self._record(mic, mode=mode, turn_id=turn_id,
                                generation=generation,
                                start_timeout_seconds=start_timeout_seconds,
                                max_duration_seconds=max_duration_seconds,
                                deadline=deadline, threshold=threshold,
                                silence_seconds=silence_seconds,
                                cancelled=cancelled)
        finally:
            self._running = False
            self._run_lock.release()

    def _record(self, mic, *, mode, turn_id, generation,
                start_timeout_seconds, max_duration_seconds, deadline,
                threshold, silence_seconds, cancelled):
        if (isinstance(max_duration_seconds, bool)
                or not isinstance(max_duration_seconds, (int, float))
                or not math.isfinite(float(max_duration_seconds))
                or not _FRAME_SECONDS <= float(max_duration_seconds) <= _MAX_DURATION_SECONDS):
            raise ValueError("max_duration_seconds must be between one frame and 30 seconds")
        if (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
                or not math.isfinite(float(threshold)) or not 0 <= threshold <= 1):
            raise ValueError("threshold must be a finite number from 0 to 1")
        if (isinstance(silence_seconds, bool)
                or not isinstance(silence_seconds, (int, float))
                or not math.isfinite(float(silence_seconds))
                or not _FRAME_SECONDS <= float(silence_seconds) <= _MAX_DURATION_SECONDS):
            raise ValueError("silence_seconds must be between one frame and 30 seconds")

        now = float(self.clock())
        request_id = uuid.uuid4().hex
        self._source_epoch += 1
        epoch = self._source_epoch
        request = CaptureRequest(
            request_id, turn_id, generation, mode, "immediate",
            start_timeout_seconds, max_duration_seconds, deadline,
        )
        # Threshold/silence are request-specific while manager ownership stays
        # on this one thread. Restore defaults after polling the terminal result.
        previous_threshold = self.manager.vad_threshold
        previous_silence_frames = self.manager.silence_frames
        self.manager.vad_threshold = float(threshold)
        self.manager.silence_frames = math.ceil(float(silence_seconds) / _FRAME_SECONDS)
        try:
            admission = self.manager.request(request)
        except Exception:
            self.manager.vad_threshold = previous_threshold
            self.manager.silence_frames = previous_silence_frames
            raise
        if not admission.accepted:
            self.manager.vad_threshold = previous_threshold
            self.manager.silence_frames = previous_silence_frames
            raise RuntimeError("foreground capture request rejected: {}".format(admission.status))

        wait_started = now
        sample = 0
        last_overflow = getattr(mic, "queue_overflow_count", 0)
        max_frames = (math.ceil((float(start_timeout_seconds)
                                 + float(max_duration_seconds)) / _FRAME_SECONDS) + 2)
        stall_limit = math.ceil((float(start_timeout_seconds)
                                 + float(max_duration_seconds)) / _READ_POLL_SECONDS) + 2
        acquired_frames = 0
        consecutive_stalls = 0
        try:
            # A single reset is safe only after the caller has suspended and
            # acknowledged the asynchronous VAD lane.
            try:
                self.vad.reset()
            except Exception as exc:
                raise RuntimeError("foreground VAD reset failed") from exc
            while acquired_frames < max_frames:
                status = self._cancel_status(cancelled)
                if status is not None:
                    self.manager.cancel(request_id, generation, superseded=True)
                    break
                if self.manager.expire():
                    break
                before_read = getattr(mic, "queue_overflow_count", last_overflow)
                if before_read != last_overflow:
                    self.manager.gap(epoch, sample)
                    break
                remaining = max(0.0, float(deadline) - float(self.clock()))
                read_timeout = min(_READ_POLL_SECONDS, remaining)
                if read_timeout <= 0:
                    self.manager.expire()
                    break
                try:
                    frame = mic.read(timeout_s=read_timeout)
                except TimeoutError:
                    consecutive_stalls += 1
                    after_timeout = getattr(mic, "queue_overflow_count", last_overflow)
                    if after_timeout != last_overflow:
                        self.manager.gap(epoch, sample)
                        break
                    last_overflow = after_timeout
                    status = self._cancel_status(cancelled)
                    if status is not None:
                        self.manager.cancel(request_id, generation, superseded=True)
                        break
                    if self.manager.expire():
                        break
                    if consecutive_stalls >= stall_limit:
                        raise RuntimeError("foreground microphone made no progress")
                    continue
                except Exception as exc:
                    self.manager.device_lost()
                    self.manager.poll()
                    raise RuntimeError("foreground microphone read failed") from exc

                captured_at = float(self.clock())
                consecutive_stalls = 0
                after_read = getattr(mic, "queue_overflow_count", last_overflow)
                if after_read != last_overflow:
                    self.manager.gap(epoch, sample)
                    break
                last_overflow = after_read
                status = self._cancel_status(cancelled)
                if status is not None:
                    self.manager.cancel(request_id, generation, superseded=True)
                    break
                if self.manager.expire():
                    break
                try:
                    self._validate_frame(frame)
                except RuntimeError:
                    self.manager.device_lost()
                    self.manager.poll()
                    raise
                raw_frame = np.array(frame, dtype=np.int16, order="C", copy=True)
                try:
                    judged = mic.vad_frame(frame) if hasattr(mic, "vad_frame") else frame
                    before_vad = getattr(mic, "queue_overflow_count", last_overflow)
                    if before_vad != last_overflow:
                        self.manager.gap(epoch, sample)
                        break
                    probability = self.vad(judged)
                except Exception as exc:
                    self.manager.device_lost()
                    self.manager.poll()
                    raise RuntimeError("foreground VAD processing failed") from exc
                after_vad = getattr(mic, "queue_overflow_count", last_overflow)
                if after_vad != last_overflow:
                    self.manager.gap(epoch, sample)
                    break
                last_overflow = after_vad
                status = self._cancel_status(cancelled)
                if status is not None:
                    self.manager.cancel(request_id, generation, superseded=True)
                    break
                if self.manager.expire():
                    break
                try:
                    self.manager.feed(epoch, sample, captured_at, raw_frame,
                                      probability=probability)
                except (TypeError, ValueError) as exc:
                    self.manager.device_lost()
                    self.manager.poll()
                    raise RuntimeError("foreground capture received an invalid frame or VAD score") from exc
                sample += _FRAME_SAMPLES
                acquired_frames += 1
                outcomes = self.manager.poll()
                if outcomes:
                    return self._format(outcomes[0], wait_started, request_id, epoch)
            outcomes = self.manager.poll()
            if outcomes:
                return self._format(outcomes[0], wait_started, request_id, epoch)
            raise RuntimeError("foreground capture exceeded its frame safety bound")
        except BaseException:
            # Any unexpected exit after admission must release this request and
            # its reserved terminal result, including failures during reset.
            self.manager.cancel(request_id, generation, superseded=True)
            self.manager.poll()
            raise
        finally:
            self.manager.vad_threshold = previous_threshold
            self.manager.silence_frames = previous_silence_frames

    @staticmethod
    def _validate_frame(frame):
        if (not isinstance(frame, np.ndarray) or frame.dtype != np.int16
                or frame.ndim != 1 or frame.size != _FRAME_SAMPLES):
            raise RuntimeError("foreground microphone returned a malformed frame")

    @staticmethod
    def _cancel_status(cancelled):
        if cancelled is None:
            return None
        try:
            if not callable(cancelled):
                raise TypeError("cancelled must be callable")
            return "superseded" if cancelled() else None
        except Exception:
            return "superseded"

    @staticmethod
    def _format(outcome, wait_started, request_id, source_epoch):
        complete = outcome.status == "complete"
        pcm = outcome.audio.raw if complete and outcome.audio is not None else np.array([], dtype=np.int16)
        duration_ms = int(pcm.size * 1000 / 16000)
        had_speech = outcome.started_at is not None
        end_reason = {
            "complete": "silence",
            "no_speech": "speech_timeout",
            "start_timeout": "speech_timeout",
            "max_duration": "capture_limit",
            "deadline": "capture_limit",
        }.get(outcome.status, outcome.status)
        return {
            "audio": pcm,
            "speech_started": complete and had_speech,
            "speech_start_ms": 0 if complete and had_speech else None,
            "speech_end_ms": duration_ms,
            "duration_ms": duration_ms,
            "wait_duration_ms": max(0, int((outcome.ended_at - wait_started) * 1000)),
            "end_reason": end_reason,
            "truncated": (outcome.status in {"max_duration", "deadline", "gap", "device_lost"}
                          and had_speech),
            "capture_status": outcome.status,
            "request_id": request_id,
            "source_epoch": source_epoch,
        }
