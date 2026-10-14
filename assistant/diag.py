"""Diagnostics plumbing shared by the assistant and the report (stdlib only).

Kept free of numpy/Config so ``python -m assistant.report`` stays instant and
low-level modules (brain, spotify) can annotate the command in flight without
importing the telemetry stack.

* the current command's Event is held in a contextvar, so every log line the
  handler thread writes can carry its id (``grep <event_id> memoir.log``);
* ``WakeTrace`` remembers the openWakeWord scores seen around an utterance;
* ``stage_deltas`` turns an Event.timeline into per-stage durations.
"""
from __future__ import annotations

import contextvars
import logging
import time
from typing import Any, Optional

# Canonical order of timeline stages. Marks outside this list are still kept;
# they sort by their offset.
STAGES = ("speech_start", "wake_evidence", "duck_start", "speech_end",
          "trigger_stt_done", "stt_done", "intent_done", "resolved",
          "playback_confirmed", "feedback_done", "handler_done")

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(event_tag)s%(message)s"


def stage_deltas(timeline: Optional[dict]) -> list:
    """[(stage, ms since the previous stage)] in time order.

    The first stage's delta is its offset from the timeline origin.
    """
    if not isinstance(timeline, dict):
        return []
    items = [(k, v) for k, v in timeline.items() if isinstance(v, (int, float))]
    order = {name: i for i, name in enumerate(STAGES)}
    items.sort(key=lambda kv: (kv[1], order.get(kv[0], len(order))))
    out, previous = [], 0
    for name, offset in items:
        out.append((name, int(offset - previous)))
        previous = offset
    return out


# ---------- current command ----------

_current_event: contextvars.ContextVar = contextvars.ContextVar(
    "memoir_current_event", default=None)


def bind_event(event: Any) -> contextvars.Token:
    """Make ``event`` the current command for this thread's logs/annotations."""
    return _current_event.set(event)


def unbind_event(token: contextvars.Token) -> None:
    try:
        _current_event.reset(token)
    except ValueError:  # token from another context; fall back to clearing
        _current_event.set(None)


def current_event() -> Any:
    return _current_event.get()


def annotate_event(**fields: Any) -> None:
    """Set fields on the command in flight, if any. Never raises."""
    event = _current_event.get()
    if event is None:
        return
    for name, value in fields.items():
        try:
            if hasattr(event, name):
                setattr(event, name, value)
        except Exception:  # diagnostics must not break a command
            pass


class EventIdFilter(logging.Filter):
    """Add ``event_tag`` ("[<id>] " or "") to records made during a command.

    Attach to handlers, not loggers: a logger's filters skip records that come
    from its children (memoir.spotify, memoir.brain ...).
    """

    def filter(self, record: logging.LogRecord) -> bool:
        event = _current_event.get()
        event_id = getattr(event, "event_id", "") if event is not None else ""
        record.event_id = event_id
        tag = "[{}] ".format(event_id) if event_id else ""
        if tag:
            # The per-command summary line already names the id.
            try:
                if event_id in record.getMessage():
                    tag = ""
            except Exception:
                pass
        record.event_tag = tag
        return True


class EventFormatter(logging.Formatter):
    """Formatter that tolerates records which never passed EventIdFilter."""

    def format(self, record: logging.LogRecord) -> str:
        if not hasattr(record, "event_tag"):
            record.event_tag = ""
        return super().format(record)


# ---------- wake scores ----------

class WakeTrace:
    """openWakeWord evidence around the current utterance.

    The detector exposes only its best model per chunk, so this keeps the peak
    score of every model that led at some point (a model that never led scored
    lower than one that did). ``note`` runs per audio frame: below the floor it
    is one comparison.
    """

    SCORE_FLOOR = 0.05   # peaks below this are noise, not evidence
    WINDOW_S = 12.0      # older peaks belong to some earlier utterance
    GAP_S = 3.0          # hits further apart than this are separate attempts

    def __init__(self, hit_floor: float = 0.15) -> None:
        self.hit_floor = hit_floor
        self._peaks: dict = {}
        self.first_at = 0.0       # perf_counter of the first score >= hit_floor
        self._last_hit_at = 0.0

    def note(self, score: float, model: str) -> None:
        if score < self.SCORE_FLOOR:
            return
        now = time.perf_counter()
        peak = self._peaks.get(model)
        if peak is None or score >= peak[0] or now - peak[1] > self.WINDOW_S:
            self._peaks[model] = (score, now)
        if score >= self.hit_floor:
            if not self.first_at or now - self._last_hit_at > self.GAP_S:
                self.first_at = now
            self._last_hit_at = now

    def snapshot(self) -> dict:
        now = time.perf_counter()
        return {model: round(score, 3) for model, (score, at) in self._peaks.items()
                if now - at <= self.WINDOW_S}

    def take(self) -> tuple:
        """(perf_counter of first evidence or 0.0, {model: peak}); then forget both."""
        now = time.perf_counter()
        first = self.first_at if self.first_at and now - self._last_hit_at <= self.WINDOW_S else 0.0
        scores = self.snapshot()
        self.reset()
        return first, scores

    def reset(self) -> None:
        self._peaks = {}
        self.first_at = 0.0
        self._last_hit_at = 0.0
