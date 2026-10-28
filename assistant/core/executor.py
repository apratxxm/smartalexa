"""Phase-one adapters and the single coordinator receipt boundary."""
from __future__ import annotations

from dataclasses import dataclass
from ..conversation_contracts import ToolResult, step_to_intent
from . import volume
from .playback import guarded_playback, remaining, TurnExpired


@dataclass(frozen=True)
class Receipt:
    state: str
    detail: str = ""
    content: str = ""
    error: str = ""

    def __post_init__(self):
        if self.state not in {"requested", "observed", "failed", "unknown"}:
            raise ValueError("invalid receipt")


def coordinator_result(operation_id, receipt):
    status = {"observed": "completed", "requested": "accepted",
              "failed": "failed", "unknown": "unknown_delivery"}[receipt.state]
    return ToolResult(operation_id, status,
                      {"receipt": receipt.state, "detail": receipt.detail,
                       "content": receipt.content}, receipt.error)


class Executor:
    ALLOWED = frozenset({"play_song", "pause", "resume", "next_track",
                         "set_volume", "change_volume", "answer"})

    def __init__(self, spotify=None, volume=None):
        self.spotify, self.volume = spotify, volume
        self._exact_candidate = None
        self.handlers = {}  # Routed adapter handlers, set by the engine from the registry.

    def _adapter(self, step, handler, guard):
        try:
            remaining(guard)
        except TurnExpired:
            return Receipt("failed", "That turn expired.")
        args = dict(step.arguments)
        handler.executor = self
        try:
            receipt = handler.execute(args, guard)
            if not isinstance(receipt, Receipt):
                raise TypeError("invalid receipt")
        except Exception as exc:
            return Receipt("unknown", "I can't confirm the result.", error=type(exc).__name__)
        try:
            text = handler.reply(receipt, args)
        except Exception as exc:
            text = "I can't confirm the result."
            return Receipt(receipt.state, text, receipt.content, type(exc).__name__)
        return Receipt(receipt.state, str(text), receipt.content, receipt.error)

    def exact_match(self, args, guard):
        import re
        try:
            remaining(guard)
            if self.spotify is None:
                from ..spotify import Spotify
                self.spotify = Spotify()
            with guarded_playback(self.spotify, guard):
                candidate = self.spotify.resolve_track(args["title"])
            remaining(guard)
            normalize = lambda value: tuple(re.findall(r"[^\W_]+", value.casefold()))
            matched = normalize(candidate.name) == normalize(args["title"])
            self._exact_candidate = (args["title"], candidate) if matched else None
            return matched
        except Exception:
            self._exact_candidate = None
            return False

    def execute(self, step, operation_id="", *, guard=lambda: float("inf")):
        cap = step.capability
        handler = self.handlers.get(cap)
        if handler is not None and not callable(getattr(handler, "execute", None)):
            handler = None
        if handler is not None:
            return self._adapter(step, handler, guard)
        if cap not in self.ALLOWED:
            return Receipt("failed", "I can't do that yet.")
        if cap == "answer":
            return Receipt("observed", str(step.arguments.get("text", "")))
        try:
            remaining(guard)
        except TurnExpired:
            return Receipt("failed", "That turn expired.")
        if cap in {"set_volume", "change_volume"}:
            vol = self.volume or volume
            try:
                if cap == "set_volume":
                    level = step.arguments["level"]
                else:
                    args = step.arguments
                    before = vol.get_system_volume() * 100
                    amount = args["amount"] or 10
                    delta = amount if args["unit"] == "points" or not args["amount"] else before * amount / 100
                    level = max(0, min(100, before + delta * (1 if args["direction"] == "up" else -1)))
                if isinstance(level, bool) or not isinstance(level, (int, float)) or not 0 <= level <= 100:
                    return Receipt("failed", "I couldn't set that volume.")
            except Exception as exc:  # No write has been attempted.
                return Receipt("failed", "I couldn't read the Windows volume.", error=type(exc).__name__)
            try:
                remaining(guard)
            except TurnExpired:
                return Receipt("failed", "That turn expired.")
            try:
                vol.set_system_volume(level / 100, guard=guard)
                return Receipt("observed", "Volume %s." % round(level))
            except TurnExpired:
                return Receipt("failed", "That turn expired.")
            except Exception as exc:
                return Receipt("unknown", "I asked Windows but can't confirm the volume changed.",
                               error=type(exc).__name__)
        try:
            if self.spotify is None:
                from ..spotify import Spotify
                self.spotify = Spotify()
        except Exception as exc:  # Adapter construction precedes playback.
            return Receipt("failed", "Spotify is unavailable.", error=type(exc).__name__)
        try:
            if cap == "play_song":
                with guarded_playback(self.spotify, guard):
                    query = step_to_intent(step).query
                    saved, self._exact_candidate = self._exact_candidate, None
                    candidate = saved[1] if saved and saved[0] == query else self.spotify.resolve_track(query)
            remaining(guard)
        except TurnExpired:
            return Receipt("failed", "That turn expired.")
        except Exception as exc:
            return Receipt("failed", "I couldn't resolve that song.", error=type(exc).__name__)
        try:
            with guarded_playback(self.spotify, guard):
                remaining(guard)
                if cap == "play_song":
                    text = self.spotify.play_candidate(candidate)
                    return Receipt("observed", text or "Playing.")
                if cap == "pause":
                    self.spotify.pause()
                    text = "Paused."
                elif cap == "resume":
                    self.spotify.play_resume()
                    text = "Playing."
                else:
                    self.spotify.next()
                    text = "Next."
                return Receipt("observed", text)
        except Exception as exc:
            # Playback wrappers can raise after sending. Conservatively retain
            # uncertainty, even if a rate-limit/transport error looks retryable.
            return Receipt("unknown", "I asked Spotify but can't confirm it played.",
                           error=type(exc).__name__)

    to_tool_result = staticmethod(coordinator_result)
