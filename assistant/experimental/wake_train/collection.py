"""Explicit, bounded collection of unlabelled wake-word audio samples.

This module has no microphone or model integration. Callers supply bytes only
after an explicit collection POST has been made.
"""
from __future__ import annotations

from dataclasses import dataclass
import io
import math
import secrets
import threading
import time
import wave

import numpy as np

from ...telemetry import Event
from ... import runtime
from .active_learning import session_split


_CONDITIONS = frozenset(("quiet", "music", "noise"))
_HEX32 = frozenset("0123456789abcdef")
_MAX_SESSIONS = 8
_SESSION_TTL = 3600
_MAX_CLIPS = 12
_SAMPLE_RATE = 16000
_MAX_FRAMES = _SAMPLE_RATE * 4
_MIN_FRAMES = _SAMPLE_RATE // 4
_MAX_WAV_BYTES = 132096


@dataclass
class _Session:
    session_id: str
    condition: str
    split: str
    expires_at: float
    saved: int = 0


@dataclass
class _Capture:
    capture_id: str
    session_id: str
    expires_at: float


class CollectionManager:
    """Own at most eight short-lived sessions and one global capture lease."""

    def __init__(self, telemetry, lease_dir, *, clock=time.time):
        self.telemetry = telemetry
        self.lease_dir = lease_dir
        self.clock = clock
        self._lock = threading.RLock()
        self._sessions: dict[str, _Session] = {}
        self._capture: _Capture | None = None

    @staticmethod
    def _valid_id(value):
        return (isinstance(value, str) and len(value) == 32
                and all(char in _HEX32 for char in value))

    def _now(self):
        value = self.clock()
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise RuntimeError("collection clock is invalid")
        return float(value)

    def _release_capture(self):
        capture, self._capture = self._capture, None
        if capture is not None:
            runtime.release_collection_pause(self.lease_dir, capture.capture_id)

    def _expire(self, now):
        if self._capture is not None and now >= self._capture.expires_at:
            self._release_capture()
        for session_id, session in tuple(self._sessions.items()):
            if now >= session.expires_at:
                if self._capture is not None and self._capture.session_id == session_id:
                    self._release_capture()
                del self._sessions[session_id]

    def create_session(self, condition):
        if not isinstance(condition, str) or condition not in _CONDITIONS:
            raise ValueError("condition must be quiet, music, or noise")
        with self._lock:
            now = self._now()
            self._expire(now)
            if len(self._sessions) >= _MAX_SESSIONS:
                raise RuntimeError("maximum active collection sessions reached")
            session_id = secrets.token_hex(16)
            split = session_split(session_id)
            session = _Session(session_id, condition, split, now + _SESSION_TTL)
            self._sessions[session_id] = session
            return {"session_id": session_id, "split": split, "condition": condition,
                    "max_clips": _MAX_CLIPS, "clip_seconds": 4}

    def arm(self, session_id):
        with self._lock:
            now = self._now()
            self._expire(now)
            if not self._valid_id(session_id):
                raise ValueError("session_id must be a 32-character lowercase hexadecimal ID")
            session = self._sessions.get(session_id)
            if session is None:
                raise ValueError("unknown or expired collection session")
            if session.saved >= _MAX_CLIPS:
                raise RuntimeError("collection session clip limit reached")
            if self._capture is not None:
                raise RuntimeError("another collection capture is active")
            capture_id = secrets.token_hex(16)
            expires_at = now + 30
            runtime.create_collection_pause(self.lease_dir, capture_id, now=now, seconds=30)
            self._capture = _Capture(capture_id, session_id, expires_at)
            return {"capture_id": capture_id, "expires_at": expires_at,
                    "sample_rate": _SAMPLE_RATE, "max_seconds": 4}

    def ready(self, capture_id):
        with self._lock:
            now = self._now()
            self._expire(now)
            capture = self._capture
            if not self._valid_id(capture_id) or capture is None or capture.capture_id != capture_id:
                raise ValueError("unknown, expired, or replayed collection capture")
            acknowledged = runtime.collection_pause_acknowledged(
                self.lease_dir, capture_id, now=now)
            if now >= capture.expires_at:
                self._release_capture()
                return {"ready": False, "expired": True}
            if not acknowledged:
                return {"ready": False, "expired": False}
            return {"ready": True, "expired": False}

    @staticmethod
    def _decode_wav(wav_bytes):
        if not isinstance(wav_bytes, (bytes, bytearray, memoryview)):
            raise ValueError("WAV body must be bytes")
        if len(wav_bytes) > _MAX_WAV_BYTES:
            raise ValueError("WAV body exceeds the 4 second size limit")
        try:
            with wave.open(io.BytesIO(bytes(wav_bytes)), "rb") as source:
                if (source.getnchannels() != 1 or source.getsampwidth() != 2
                        or source.getframerate() != _SAMPLE_RATE
                        or source.getcomptype() != "NONE"):
                    raise ValueError("WAV must be mono PCM16LE at 16000 Hz")
                frames = source.getnframes()
                if not _MIN_FRAMES <= frames <= _MAX_FRAMES:
                    raise ValueError("WAV duration must be between 0.25 and 4 seconds")
                payload = source.readframes(frames)
                if len(payload) != frames * 2 or source.readframes(1):
                    raise ValueError("WAV frame count does not match its PCM payload")
        except (wave.Error, EOFError, OSError) as exc:
            raise ValueError("invalid WAV data") from exc
        samples = np.frombuffer(payload, dtype="<i2").copy()
        return samples, frames

    def submit(self, capture_id, wav_bytes):
        with self._lock:
            now = self._now()
            self._expire(now)
            capture = self._capture
            if not self._valid_id(capture_id) or capture is None or capture.capture_id != capture_id:
                raise ValueError("unknown, expired, or replayed collection capture")
            self._capture = None  # Every matching submission consumes this token once.
            try:
                if now >= capture.expires_at:
                    raise ValueError("collection capture lease expired")
                if not runtime.collection_pause_acknowledged(
                        self.lease_dir, capture_id, now=now):
                    raise RuntimeError("collection pause has not been acknowledged")
                session = self._sessions.get(capture.session_id)
                if session is None or now >= session.expires_at:
                    raise ValueError("collection session expired")
                if session.saved >= _MAX_CLIPS:
                    raise RuntimeError("collection session clip limit reached")
                pcm, frames = self._decode_wav(wav_bytes)
                event = Event(ts=now, trigger_source="wake_collection",
                              candidate_kind="intentional_wake", outcome="pending_review",
                              review_state="unknown", transcript="", capture_session_id=session.session_id,
                              segment_id=capture.capture_id, input_device="browser_microphone",
                              end_reason="explicit_user_collection",
                              decision_source="explicit_collection_" + session.condition)
                path = self.telemetry.save_audio(event, pcm, subdir="collection")
                if not path or event.audio_status != "saved":
                    raise RuntimeError("collection WAV save failed; no retry was attempted")
                session.saved += 1
                event.audio_path = path
                if (event.audio_n_frames != frames or len(event.audio_sha256) != 64
                        or any(char not in "0123456789abcdef" for char in event.audio_sha256)):
                    raise RuntimeError("telemetry did not report saved WAV identity")
                indexed = bool(self.telemetry.write(event))
                if not indexed:
                    raise RuntimeError("collection WAV saved but telemetry indexing failed; orphan retained")
                return {"event_id": event.event_id, "indexed": indexed, "saved": True,
                        "session_id": session.session_id, "split": session.split,
                        "review_state": "unknown"}
            finally:
                runtime.release_collection_pause(self.lease_dir, capture_id)

    def cancel(self, capture_id):
        with self._lock:
            now = self._now()
            self._expire(now)
            capture = self._capture
            if not self._valid_id(capture_id) or capture is None or capture.capture_id != capture_id:
                raise ValueError("unknown, expired, or replayed collection capture")
            self._capture = None
            try:
                return {"cancelled": True}
            finally:
                runtime.release_collection_pause(self.lease_dir, capture_id)
