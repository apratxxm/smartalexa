"""Best-effort archival for immutable capture diagnostics.

Submission only validates and enqueues owned snapshots. Filesystem pruning and
WAV/JSONL writes happen on one daemon worker, away from microphone capture.
"""
from __future__ import annotations

from collections import OrderedDict, deque
from collections.abc import Mapping
from numbers import Integral, Real
import math
from queue import Empty, Full, Queue
from threading import Lock, Thread
import time
from typing import Any

from .audio_jobs import AudioSnapshot


def _text(value: Any, name: str, limit: int) -> str:
    if not isinstance(value, str) or len(value) > limit:
        raise ValueError("{} must be a string of at most {} characters".format(name, limit))
    return value


def _latency(value: Any, name: str):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError("{} must be a nonnegative integer".format(name))
    return int(value)


class QueuedDiagnosticArchive:
    """Bounded, nonblocking front end to ``Telemetry.record_near_miss``."""

    def __init__(self, telemetry, *, max_pending: int = 8, max_failures: int = 16,
                 clock=time.monotonic):
        if isinstance(max_pending, bool) or not isinstance(max_pending, Integral) or not 1 <= max_pending <= 8:
            raise ValueError("max_pending must be between 1 and 8")
        if isinstance(max_failures, bool) or not isinstance(max_failures, Integral) or not 1 <= max_failures <= 16:
            raise ValueError("max_failures must be between 1 and 16")
        self._telemetry = telemetry
        self._queue = Queue(maxsize=int(max_pending))
        self._max_failures = int(max_failures)
        self._clock = clock
        self._lock = Lock()
        self._failures = deque(maxlen=self._max_failures)
        self._dedupe = OrderedDict()
        # completed counts returned telemetry calls; archived counts saved raw WAVs.
        self._counts = dict(accepted=0, completed=0, archived=0, failed=0,
                            suppressed=0, disabled=0, partial=0, dropped=0,
                            duplicates=0, lost_failures=0)
        self._thread = None
        self._closing = False

    def start(self) -> None:
        with self._lock:
            if self._thread is not None:
                return
            if self._closing:
                raise RuntimeError("archive is closed")
            self._thread = Thread(target=self._run, name="diagnostic-archive", daemon=True)
            self._thread.start()

    def submit(self, snapshot, reason, *, stage, capture_id="", proposal_id="",
               transcript="", second_stt_transcript="", latency_ms=None,
               second_latency_ms=None, input_device="laptop") -> bool:
        if not isinstance(snapshot, AudioSnapshot):
            raise ValueError("snapshot must be an AudioSnapshot")
        reason = _text(reason, "reason", 240)
        stage = _text(stage, "stage", 128)
        if not reason.strip() or not stage.strip():
            raise ValueError("reason and stage must not be blank")
        capture_id = _text(capture_id, "capture_id", 128)
        proposal_id = _text(proposal_id, "proposal_id", 128)
        transcript = _text(transcript, "transcript", 16000)
        second_stt_transcript = _text(second_stt_transcript, "second_stt_transcript", 16000)
        input_device = _text(input_device, "input_device", 128)
        latency_ms = _latency(latency_ms, "latency_ms")
        second_latency_ms = _latency(second_latency_ms, "second_latency_ms")
        if not capture_id:
            metadata_capture_id = snapshot.metadata.get("capture_id", "")
            capture_id = (metadata_capture_id if isinstance(metadata_capture_id, str)
                          and 0 < len(metadata_capture_id) <= 128 else snapshot.segment_id)
        # Copy every scalar argument before publishing it to the worker.
        item = (snapshot, str(reason), str(stage), str(capture_id), str(proposal_id),
                str(transcript), str(second_stt_transcript), latency_ms,
                second_latency_ms, str(input_device))
        key = (capture_id or snapshot.segment_id, stage)
        with self._lock:
            if self._thread is None or self._closing:
                self._counts["dropped"] += 1
                return False
            if key in self._dedupe:
                self._dedupe.move_to_end(key)
                self._counts["duplicates"] += 1
                return True
            try:
                self._queue.put_nowait(item)
            except Full:
                self._counts["dropped"] += 1
                return False
            self._dedupe[key] = None
            if len(self._dedupe) > 64:
                self._dedupe.popitem(last=False)
            self._counts["accepted"] += 1
            return True

    def counters(self) -> dict[str, int]:
        """Return bounded outcome counts.

        ``completed`` means telemetry returned, ``archived`` means its raw WAV
        was saved, and ``partial`` means raw saved but supplied processed audio
        did not. Suppressed/disabled are non-I/O outcomes; failed covers raised
        exceptions and failed/unavailable audio statuses.
        """
        with self._lock:
            return dict(self._counts)

    def poll_failures(self) -> list[str]:
        with self._lock:
            rows = list(self._failures)
            self._failures.clear()
            return rows

    def close(self, timeout) -> bool:
        if isinstance(timeout, bool) or not isinstance(timeout, Real) or not math.isfinite(float(timeout)) or timeout < 0:
            raise ValueError("timeout must be a finite nonnegative number")
        with self._lock:
            self._closing = True
            thread = self._thread
        if thread is None:
            return True
        thread.join(float(timeout))
        return not thread.is_alive()

    def _run(self):
        while True:
            try:
                item = self._queue.get(timeout=0.05)
            except Empty:
                with self._lock:
                    if self._closing:
                        return
                continue
            try:
                (snapshot, reason, stage, capture_id, proposal_id, transcript,
                 second_transcript, latency, second_latency, input_device) = item
                metadata = snapshot.metadata
                scores = metadata.get("scores")
                if isinstance(scores, Mapping):
                    scores = {k: float(v) for k, v in list(scores.items())[:32]
                              if isinstance(k, str) and len(k) <= 128
                              and isinstance(v, Real) and not isinstance(v, bool)
                              and math.isfinite(float(v)) and 0.0 <= float(v) <= 1.0}
                else:
                    scores = None
                diagnostic_keys = (
                    "acoustic_proposal", "proposal_qualifiers", "confirmation_interval",
                    "trigger_stage_state", "second_stt_stage_state",
                    "trigger_transcript_truncated", "second_stt_transcript_truncated",
                )
                diagnostic_metadata = {key: metadata[key] for key in diagnostic_keys
                                       if key in metadata}
                kwargs = {
                    "kind": "diagnostic", "segment_id": snapshot.segment_id,
                    "overlap_of": snapshot.overlap_of, "end_reason": snapshot.end_reason,
                    "truncated": snapshot.truncated, "relative_start_sample": 0,
                    "relative_end_sample": int(snapshot.audio.size),
                    "processed_audio": snapshot.processed_audio,
                    "processed_alignment": "frame_paired" if snapshot.processed_audio is not None else "unavailable",
                    "review_state": "unknown", "diagnostic_only": True,
                    "source_epoch": snapshot.stream_epoch,
                    "source_start_sample": snapshot.start_sample,
                    "source_end_sample": snapshot.end_sample,
                    "source_started_at": snapshot.captured_at,
                    "source_completed_at": snapshot.completed_at,
                    "capture_id": capture_id, "proposal_id": proposal_id,
                    "diagnostic_stage": stage, "input_device": input_device,
                    "vad_peak": metadata.get("vad_peak") if isinstance(metadata.get("vad_peak"), Real) else None,
                    "vad_threshold": metadata.get("vad_threshold") if isinstance(metadata.get("vad_threshold"), Real) else None,
                    "voiced_frames": metadata.get("voiced_frames") if isinstance(metadata.get("voiced_frames"), Integral) else None,
                    "woke_early": bool(metadata.get("woke_early", False)),
                    "scores": scores,
                    "diagnostic_metadata": diagnostic_metadata or None,
                    "trigger_stt_latency_ms": latency,
                    "second_stt_latency_ms": second_latency,
                }
                result = self._telemetry.record_near_miss(
                    snapshot.audio, transcript, reason,
                    second_stt_transcript=second_transcript,
                    now=self._clock(), **kwargs)
                status = getattr(result, "audio_status", None)
                with self._lock:
                    self._counts["completed"] += 1
                    if status == "saved":
                        self._counts["archived"] += 1
                    elif status == "suppressed":
                        self._counts["suppressed"] += 1
                    elif status == "disabled":
                        self._counts["disabled"] += 1
                    elif status in ("failed", "unavailable"):
                        self._counts["failed"] += 1
                        self._record_failure("telemetry audio {}: {}".format(
                            status, getattr(result, "audio_reason", "")))
                    if (status == "saved" and snapshot.processed_audio is not None and
                            getattr(result, "processed_audio_status", "") in
                            ("failed", "unavailable", "suppressed", "disabled")):
                        self._counts["partial"] += 1
            except Exception as exc:
                message = "{}: {}".format(type(exc).__name__, str(exc))[:512]
                with self._lock:
                    self._counts["failed"] += 1
                    self._record_failure(message)
            finally:
                self._queue.task_done()

    def _record_failure(self, message):
        if len(self._failures) == self._max_failures:
            self._counts["lost_failures"] += 1
        self._failures.append(message[:512])
