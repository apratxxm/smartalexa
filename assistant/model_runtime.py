"""Shared, bounded request ownership for one local Ollama model tag."""
from __future__ import annotations

import copy
import math
import threading
import time
from typing import Callable


class RuntimeAdmissionError(TimeoutError):
    """A request was not admitted to the local model transport."""


class LocalModelRuntime:
    """Own residency policy, readiness and serialized requests for one model."""

    def __init__(self, model: str, *, num_ctx: int = 4096, keep_alive: int = -1,
                 cooldown_seconds: float = 30, clock: Callable[[], float] = time.monotonic) -> None:
        if not isinstance(model, str) or not model.strip() or "cloud" in model.casefold():
            raise ValueError("model must be a nonempty local model tag")
        if (not isinstance(num_ctx, int) or isinstance(num_ctx, bool)
                or not 256 <= num_ctx <= 32_768):
            raise ValueError("num_ctx must be an integer from 256 to 32768")
        if not isinstance(keep_alive, int) or isinstance(keep_alive, bool):
            raise ValueError("keep_alive must be an integer")
        if (not isinstance(cooldown_seconds, (int, float)) or isinstance(cooldown_seconds, bool)
                or not math.isfinite(cooldown_seconds) or cooldown_seconds <= 0):
            raise ValueError("cooldown_seconds must be finite and positive")
        if not callable(clock):
            raise ValueError("clock must be callable")
        self.model = model.strip()
        self.num_ctx = num_ctx
        self.keep_alive = keep_alive
        self._cooldown_seconds = float(cooldown_seconds)
        self._clock = clock
        self._request_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._state = "cold"
        self._retry_at = 0.0

    @property
    def state(self) -> str:
        with self._state_lock:
            if self._state == "degraded" and self._clock() >= self._retry_at:
                # Admission reopens after cooldown, but readiness remains degraded
                # until an actual request succeeds.
                return "degraded"
            return self._state

    def mark_warming(self) -> None:
        with self._state_lock:
            self._state = "warming"

    def mark_failed(self) -> None:
        with self._state_lock:
            self._state = "degraded"
            self._retry_at = self._clock() + self._cooldown_seconds

    def can_attempt(self) -> bool:
        with self._state_lock:
            return self._can_attempt_locked(False)

    def _can_attempt_locked(self, warmup: bool) -> bool:
        if warmup:
            return True
        if self._state in {"cold", "ready"}:
            return True
        if self._state == "degraded" and self._clock() >= self._retry_at:
            return True
        return False

    def run(self, transport, body: dict, timeout: float, *, warmup: bool = False) -> dict:
        if (not isinstance(timeout, (int, float)) or isinstance(timeout, bool)
                or not math.isfinite(timeout) or timeout <= 0):
            raise ValueError("timeout must be finite and positive")
        if not isinstance(body, dict):
            raise ValueError("request body must be an object")
        request = copy.deepcopy(body)
        if request.get("model") != self.model:
            raise ValueError("request model does not match runtime owner")
        options = request.get("options")
        if options is None:
            options = {}
            request["options"] = options
        if not isinstance(options, dict):
            raise ValueError("request options must be an object")
        options["num_ctx"] = self.num_ctx
        request["keep_alive"] = self.keep_alive

        started = self._clock()
        remaining = float(timeout) - (self._clock() - started)
        if remaining <= 0 or not self._request_lock.acquire(timeout=max(0.0, remaining)):
            raise RuntimeAdmissionError("local model request deadline exceeded before dispatch")
        try:
            remaining = float(timeout) - (self._clock() - started)
            if remaining <= 0:
                raise RuntimeAdmissionError("local model request deadline exceeded before dispatch")
            with self._state_lock:
                if not self._can_attempt_locked(warmup):
                    raise RuntimeAdmissionError("local model runtime is not accepting requests")
            try:
                result = transport(request, remaining)
                if float(timeout) - (self._clock() - started) <= 0:
                    raise RuntimeAdmissionError("local model request deadline exceeded; model work may continue")
            except Exception:
                self.mark_failed()
                raise
            with self._state_lock:
                self._state = "ready"
                self._retry_at = 0.0
            return result
        finally:
            self._request_lock.release()
