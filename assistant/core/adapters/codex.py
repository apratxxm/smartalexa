"""Phase 3 Codex admission and single-shot delegation."""
from __future__ import annotations

import re
import threading
try:
    import tomllib  # py3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # py3.10
from pathlib import Path

from ... import config
from ...codex_bridge import CodexBridge, DeliveryUnknown, DeadlineExpired
from .. import observe
from .base import Adapter, Clause, Guard, Receipt, TurnExpired


def _raise_codex_window(guard):
    import ctypes
    from ctypes import wintypes

    windows = [w for w in observe.enumerate_windows() if w.image.casefold() == "codex.exe"]
    if not windows:
        return False
    window = windows[0]
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.IsIconic.argtypes = [wintypes.HWND]
    user32.IsIconic.restype = wintypes.BOOL
    user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
    user32.SetForegroundWindow.argtypes = [wintypes.HWND]
    if user32.IsIconic(window.hwnd):
        if guard() <= 0:
            raise TurnExpired()
        user32.ShowWindow(window.hwnd, 9)
    if guard() <= 0:
        raise TurnExpired()
    user32.SetForegroundWindow(window.hwnd)
    foreground = observe.foreground_window()
    if not (foreground and foreground.hwnd == window.hwnd and foreground.image.casefold() == "codex.exe"):
        if guard() <= 0:
            raise TurnExpired()
        user32.keybd_event(0x12, 0, 0, 0)  # VK_MENU down
        try:
            if guard() <= 0:
                raise TurnExpired()
        finally:
            # Always release the key if the deadline expires after key-down.
            user32.keybd_event(0x12, 0, 2, 0)  # KEYEVENTF_KEYUP
        if guard() <= 0:
            raise TurnExpired()
        user32.SetForegroundWindow(window.hwnd)
        foreground = observe.foreground_window()
    return bool(foreground and foreground.hwnd == window.hwnd and foreground.image.casefold() == "codex.exe")


# Second safety net behind the grammar: the request is a speech transcript, so Codex is told to
# stop on any sign the speaker withdrew it or didn't mean it.
_VOICE_PREAMBLE = ("[Voice request, transcribed from speech. If any part of it withdraws, jokes about, "
                   "or only imagines the request, make no changes and reply 'Skipped: request withdrawn.']\n\n")


class CodexRequest:
    mutating = True
    verbs = ("ask codex to", "tell codex to", "get codex to", "have codex",
             "send codex", "give codex")
    object_kind = "text"
    effect_words = ("sent", "asked", "delegated", "started", "submitted", "forwarded")

    _unsafe = re.compile(
        r"[\"`\u201c\u201d\n\r?;:]|(?<!\w)['\u2018\u2019]|['\u2018\u2019](?!\w)|"
        r"\b(?:don['\u2019]?t|do\s+not|not\s+yet|later|wait|hold\s+on|never\s+mind|"
        r"or\s+not|maybe|should\s+i|what\s+does|what\s+is|can\s+you\s+explain|"
        r"just\s+kidding|forget\s+(?:about\s+)?(?:it|that|this)|drop\s+(?:it|that)|on\s+second\s+thought|second\s+thoughts|jk|lol|lmao|lmfao|rofl|haha+|ha\s+ha|hehe|psych|whatever|or\s+something|its\s+fine|it\s+is\s+fine|nah|nope|"
        r"scratch\s+that|ignore\s+that|disregard\s+that|"
        r"(?:just\s+|only\s+)?joking|kidding|no\s+wait|nevermind|for\s+(?:example|instance)|"
        r"hypothetically|imagine|pretend|suppose|lets\s+say|let\s+us\s+say|someday|eventually|"
        r"sometime|tomorrow|tonight|next\s+week|cancel|but|however|if|unless|said|says|told|quote|reported)\b",
        re.IGNORECASE,
    )

    # Deliberately reject first-person subjects, including transcript contractions:
    # the speaker must rephrase even an otherwise benign relative clause.
    _withdrawal = re.compile(
        r"\b(?:i(?:['\u2019](?:m|ve|d|ll))?|im|ive|id|ill|"
        r"joke|joking|jest|example|sample|take\s+(?:(?:that|it)\s+)?back|"
        r"retract|not\s+serious|seriously|kidding|that\s+was|it\s+was|was\s+just)\b",
        re.IGNORECASE,
    )

    def __init__(self, *, project_dir=None, bridge_factory=CodexBridge, raise_window=_raise_codex_window, config_path=None):
        self._project_dir = project_dir
        self._config_path = Path(config_path) if config_path is not None else Path.home() / ".codex" / "config.toml"
        self._bridge_factory = bridge_factory
        self._raise_window = raise_window
        self._bridge = None
        self._root = None
        self._lock = threading.Lock()

    def license(self, clause: Clause) -> dict | None:
        if not isinstance(clause, Clause) or clause.verb not in self.verbs:
            return None
        obj = clause.object
        if (not isinstance(obj, str) or not obj or obj != obj.strip()
                or len(obj) > 12000 or not isinstance(clause.text, str)
                or self._unsafe.search(clause.text) or self._unsafe.search(obj)
                or self._withdrawal.search(clause.text) or self._withdrawal.search(obj)
                or len(re.findall(r"[^\W_]+", obj)) < 3):
            return None
        return {"request": obj}

    def execute(self, args: dict, guard: Guard) -> Receipt:
        request = args.get("request") if isinstance(args, dict) else None
        if not isinstance(request, str) or not request or len(request) > 12000 or request != request.strip():
            return Receipt("failed", "invalid_request")
        if not self._lock.acquire(blocking=False):
            return Receipt("failed", "busy")
        try:
            return self._execute(request, guard)
        finally:
            self._lock.release()

    def _execute(self, request, guard):
        try:
            with self._config_path.open("rb") as f:
                sandbox = tomllib.load(f).get("sandbox_mode")
            if sandbox not in ("workspace-write", "danger-full-access"):
                return Receipt("failed", "sandbox_restricted")
            configured = self._project_dir if self._project_dir is not None else config.load().get("core.codex.project_dir")
            if not isinstance(configured, (str, Path)) or not str(configured).strip():
                return Receipt("failed", "project_missing")
            root = Path(configured)
            if not root.is_absolute() or not root.is_dir():
                return Receipt("failed", "project_invalid")
            root = root.resolve(strict=True)
            if self._bridge is not None and root != self._root:
                return Receipt("failed", "project_changed")
            if self._bridge is None:
                self._bridge = self._bridge_factory(root)
                self._root = root
            bridge = self._bridge
            if bridge.active_turn is not None or bridge.mutation_uncertain:
                return Receipt("failed", "busy")
            if guard() <= 0:
                raise TurnExpired()
            if bridge.mutation_uncertain:
                return Receipt("unknown", "bridge_uncertain")
        except Exception as exc:
            return Receipt("failed", "preparation_failed", error=type(exc).__name__)
        try:
            created = bridge.create_thread(title=" ".join(request.split()[:6]), sandbox="workspace-write", guard=guard)
            thread_id = created.get("id")
            if not isinstance(thread_id, str) or not thread_id or created.get("can_submit") is False or bridge.mutation_uncertain:
                return Receipt("unknown", "creation_uncertain")
        except DeadlineExpired as exc:
            return Receipt("failed", "deadline_expired", error=type(exc).__name__)
        except Exception as exc:
            return Receipt("unknown", "creation_uncertain", error=type(exc).__name__)
        try:
            if guard() <= 0:
                raise TurnExpired()
        except Exception as exc:
            return Receipt("unknown", "expired_after_creation", error=type(exc).__name__)
        try:
            turn = bridge.start_turn(thread_id, _VOICE_PREAMBLE + request, guard=guard)
            if not isinstance(turn, dict) or not isinstance(turn.get("id"), str) or not turn["id"]:
                return Receipt("unknown", "submission_unconfirmed")
        except DeadlineExpired as exc:
            return Receipt("unknown", "expired_after_creation", error=type(exc).__name__)
        except DeliveryUnknown as exc:
            return Receipt("unknown", "submission_uncertain", error=type(exc).__name__)
        except Exception as exc:
            return Receipt("unknown", "submission_uncertain", error=type(exc).__name__)
        try:
            foreground = self._raise_window(guard)
        except Exception as exc:
            return Receipt("unknown", "foreground_error", error=type(exc).__name__)
        return Receipt("observed" if foreground else "requested", "foreground" if foreground else "unobserved")

    def reply(self, receipt: Receipt, args: dict) -> str:
        if receipt.state == "failed" and receipt.detail == "busy":
            return "Codex is still busy with the last request."
        if receipt.state == "observed":
            return "Sent to Codex."
        if receipt.state in {"requested", "unknown"}:
            return "I asked Codex but can't confirm it got the request."
        return "I couldn't reach Codex."


ADAPTER = Adapter("codex", "3", {"codex_request": CodexRequest()})
