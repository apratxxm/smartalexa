"""Small, deterministic global-command and Codex dictation state helpers.

This module deliberately has no connection to a browser, Codex, a model, or
the music intent parser.  It only recognizes the narrow commands that the
host application may choose to route elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import secrets
import time
import uuid
from typing import Callable, Optional


@dataclass(frozen=True)
class GlobalCommand:
    action: str
    text: str = ""
    direction: str = ""
    title: str = ""
    inferred: bool = False
    target_kind: str = "unspecified"
    target_name: str = ""
    model: str = ""
    show_chat: bool = False


def creation_model(text: str) -> str:
    """Read an explicit model instruction, not a title or inferred substitution."""
    matches = list(re.finditer(r"\b(?:use|using|with)\s+(?:the\s+)?(?:model\s+)?"
                              r"(gpt[- ]?\d+(?:\.\d+)?(?:[- ]+(?:luna|sol|astra|terra|codex))?)\b", text, re.I))
    for match in matches:
        prefix = text[:match.start()]
        if re.search(r"\b(?:do\s+not|don't|dont|never|not)\s*$", prefix, re.I):
            raise ValueError("Negated model constraints require clarification")
    models = {re.sub(r"[- ]+", "-", match[1].casefold()) for match in matches}
    # GPT6 is normal model spelling, not an alias for another version.
    models = {re.sub(r"^gpt(?=\d)", "gpt-", value) for value in models}
    if len(models) > 1:
        raise ValueError("Conflicting requested models")
    return next(iter(models), "")


def _literal_label(value: str) -> str | None:
    for opening, closing in (("\"", "\""), ("'", "'"), ("“", "”"), ("‘", "’")):
        if value.startswith(opening) and value.endswith(closing) and len(value) > 2:
            inside = value[1:-1]
            if opening not in inside and closing not in inside:
                return inside
    return None


def _has_followon(value: str) -> bool:
    # Unquoted connectors are an ambiguous title/task boundary. Let the
    # semantic classifier decide while retaining the entire source, rather
    # than maintaining a whitelist of verbs that can follow a chat title.
    return bool(re.search(r"\s+(?:and|then|to)\s+\S", value, re.I))


def parse_global(text: str) -> Optional[GlobalCommand]:
    """Parse one complete, narrowly-scoped global command.

    Matching is anchored to the complete input.  Apart from removing the
    command's boundary whitespace, captured text is returned unchanged so a
    prompt's punctuation and case survive routing.
    """

    if not isinstance(text, str):
        return None
    value = text.strip()
    if not value:
        return None
    if re.fullmatch(r"use\s+luna\s+for\s+(?:this|the\s+current)\s+codex\s+chat[.!?]*", value, flags=re.IGNORECASE):
        return GlobalCommand("use_luna")
    # ASR commonly appends sentence punctuation to a short command.  This is
    # only for non-payload command matching; prompt/title/name text below is
    # always captured from ``value`` unchanged.
    syntax_value = re.sub(r"[.?!]+$", "", value).rstrip()

    if re.fullmatch(r"(?:please\s+)?(?:open|show)\s+(?:the\s+)?(?:codex\s+)?chat\s+"
                    r"(?:you\s+(?:just\s+)?(?:made|created)|just\s+created)", syntax_value, re.I):
        return GlobalCommand("open", "last_created")
    if re.fullmatch(r"(?:please\s+)?(?:open|show)\s+(?:the\s+)?(?:last(?:\s+created)?|latest)\s+"
                    r"(?:codex\s+)?chat", syntax_value, re.I):
        return GlobalCommand("open", "last_created")

    # Chat creation.  ``called`` is optional; an omitted title is represented
    # by an empty text value for the host to name as it wishes.
    match = re.fullmatch(
        r"(?:create|new|start)\s+(?:a\s+)?(?:new\s+)?codex\s+chat"
        r"(?:\s+called\s+(.+))?",
        syntax_value,
        flags=re.IGNORECASE,
    )
    if match:
        title = match.group(1) or ""
        literal = _literal_label(title)
        if literal is None and _has_followon(title):
            return None
        return GlobalCommand("create", literal if literal is not None else title)

    if re.fullmatch(r"list\s+codex\s+chats?", syntax_value, flags=re.IGNORECASE):
        return GlobalCommand("list")

    match = re.fullmatch(
        r"(switch|open|select)(?:\s+to)?\s+(current|this|last|latest)\s+codex\s+chat",
        syntax_value, flags=re.IGNORECASE,
    )
    if match:
        return GlobalCommand("open" if match.group(1).casefold() == "open" else "select", match.group(2).casefold())

    # Keep NAME verbatim.  ``last`` and ``current`` are intentionally names,
    # leaving resolution of those aliases to the caller.
    match = re.fullmatch(
        r"(?:switch|open|select)(?:\s+to)?\s+codex\s+chat\s+(.+)",
        value,
        flags=re.IGNORECASE,
    )
    if match:
        # Preserve the requested UI/data operation for the caller: selecting
        # a chat is distinct from opening its desktop view.
        literal = _literal_label(match.group(1))
        if literal is None and _has_followon(match.group(1)):
            return None
        verb = value.casefold().split()[0]
        target = literal if literal is not None else match.group(1)
        alias = re.sub(r"[.?!]+$", "", target).strip().casefold()
        if alias in {"current", "this", "last", "latest"}:
            target = alias
        return GlobalCommand("open" if verb == "open" else "select", target)

    # These are the only direct prompt-sending forms.  Anchoring prevents
    # incidental prose such as "he said ask Codex ..." from being routed.
    match = re.fullmatch(
        r"(?:ask|tell|send)\s+codex\s+(.+)", value, flags=re.IGNORECASE
    )
    if match:
        return GlobalCommand("ask", match.group(1))

    # Dictation has an optional ``a prompt`` lead-in and an optional Codex
    # destination suffix.  A bare ``dictate`` is useful to arm a session.
    if re.fullmatch(r"dictate", syntax_value, flags=re.IGNORECASE):
        return GlobalCommand("dictate")
    if re.fullmatch(
        r"dictate\s+a\s+prompt\s+(?:for|to)\s+codex",
        syntax_value,
        flags=re.IGNORECASE,
    ):
        return GlobalCommand("dictate")
    if re.fullmatch(r"dictate\s+(?:for|to)\s+codex", syntax_value, flags=re.IGNORECASE):
        return GlobalCommand("dictate")
    match = re.fullmatch(
        r"dictate\s+(?:a\s+prompt\s+)?(?:for|to)\s+codex\s+(.+)",
        value, flags=re.IGNORECASE,
    )
    if match:
        return GlobalCommand("dictate", match.group(1))
    match = re.fullmatch(
        r"dictate\s+(?:a\s+prompt\s+)?(.+?)\s+(?:for|to)\s+codex\s*",
        syntax_value,
        flags=re.IGNORECASE,
    )
    if match:
        return GlobalCommand("dictate", match.group(1))
    match = re.fullmatch(
        r"dictate\s+(?:a\s+prompt\s+)?(.+)", value, flags=re.IGNORECASE
    )
    if match:
        return GlobalCommand("dictate", match.group(1))

    if re.fullmatch(r"review\s+codex\s+approval", syntax_value, flags=re.IGNORECASE):
        return GlobalCommand("review_approval")
    if re.fullmatch(r"approve\s+codex\s+action", syntax_value, flags=re.IGNORECASE):
        return GlobalCommand("approve")
    if re.fullmatch(r"deny\s+codex\s+action", syntax_value, flags=re.IGNORECASE):
        return GlobalCommand("deny")

    if re.fullmatch(
        r"(?:codex\s+status|what\s+is\s+codex\s+doing)",
        syntax_value,
        flags=re.IGNORECASE,
    ):
        return GlobalCommand("status")

    match = re.fullmatch(
        r"(?:cancel|stop)\s+codex(?:\s+(job|task))?",
        syntax_value,
        flags=re.IGNORECASE,
    )
    if match:
        return GlobalCommand("cancel", match.group(1) or "")

    match = re.fullmatch(
        r"(?:next|previous)(?:\s+browser)?\s+tab",
        syntax_value,
        flags=re.IGNORECASE,
    )
    if match:
        direction = value.casefold().split()[0]
        return GlobalCommand("switch_tab", direction=direction)

    match = re.fullmatch(
        r"switch\s+to\s+(next|previous)(?:\s+browser)?\s+tab",
        syntax_value,
        flags=re.IGNORECASE,
    )
    if match:
        return GlobalCommand("switch_tab", direction=match.group(1).casefold())

    return None


@dataclass(frozen=True)
class SessionReply:
    action: str
    message: str = ""
    prompt: str = ""
    target_id: str = ""
    confirmation_id: str = ""
    plan_id: str = ""


_TARGET_ID = re.compile(r"thr_[A-Za-z0-9_-]+\Z")
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")
_MAX_DISPLAY_NAME = 120
MAX_PROMPT_CHARS = 12000
MAX_PROMPT_SEGMENTS = 30


def _control_text(value: str) -> str:
    """Normalize only terminal ASR sentence punctuation for controls.

    Quoted speech is intentionally not a control: saying ``"send it"`` is
    either dictation content or rejected review speech, never confirmation.
    """

    stripped = value.strip()
    if not stripped or stripped[:1] in {'"', "'", "“", "‘"} or stripped[-1:] in {'"', "'", "”", "’"}:
        return ""
    return re.sub(r"[.?!]+$", "", stripped).rstrip().casefold()


def _valid_target_id(value: str) -> bool:
    if not isinstance(value, str) or not value:
        return False
    if _TARGET_ID.fullmatch(value):
        return True
    try:
        uuid.UUID(value)
    except (ValueError, AttributeError, TypeError):
        return False
    return True


def _display_name(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("target name must be a string")
    # Names are spoken/displayed only.  Remove control characters and bound
    # their size so a persisted or remote name cannot become a giant prompt.
    cleaned = _CONTROL_CHARS.sub(" ", value).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    if not cleaned:
        raise ValueError("target name must not be empty")
    return cleaned[:_MAX_DISPLAY_NAME]


class VoiceSession:
    """Bounded two-step dictation state machine for a selected Codex chat."""

    def __init__(
        self,
        clock: Callable[[], float] = time.monotonic,
        timeout_seconds: float = 90,
        max_chars: int = 12000,
        max_segments: int = 30,
    ) -> None:
        if timeout_seconds <= 0 or max_chars <= 0 or max_segments <= 0:
            raise ValueError("session limits must be positive")
        if max_chars > MAX_PROMPT_CHARS:
            raise ValueError(f"max_chars must be <= {MAX_PROMPT_CHARS}")
        if max_segments > MAX_PROMPT_SEGMENTS:
            raise ValueError(f"max_segments must be <= {MAX_PROMPT_SEGMENTS}")
        self.clock = clock
        self.timeout_seconds = float(timeout_seconds)
        self.max_chars = int(max_chars)
        self.max_segments = int(max_segments)
        self.phase = "idle"
        self.draft = ""
        self.target_id = ""
        self.target_name = ""
        self.target_revision = 0
        self._deadline: Optional[float] = None
        self._segments = 0
        self._confirmation: Optional[dict[str, object]] = None
        self.on_invalidate: Optional[Callable[[], None]] = None
        self._plan_id = ""
        self._hard_deadline: Optional[float] = None
        self._feedback = None

    @property
    def active(self) -> bool:
        if self.phase not in {"dictating", "review"}:
            return False
        now = self.clock()
        if ((self._hard_deadline is not None and now >= self._hard_deadline)
                or (self.deadline is not None and now >= self.deadline)):
            self._expire()
            return False
        return True

    def _expire(self) -> None:
        self.phase = "idle"
        self.draft = ""
        self._segments = 0
        self._deadline = None
        self._confirmation = None
        self._plan_id = ""
        self._hard_deadline = None
        self._feedback = None
        if self.on_invalidate is not None:
            self.on_invalidate()

    def _clear_after_submit(self) -> None:
        """Close dictation while retaining the newly-issued confirmation."""
        self.phase = "idle"
        self.draft = ""
        self._segments = 0
        self._deadline = None

    def _ready_message(self) -> str:
        return f"Ready for {self.target_name}. Say send it or cancel."

    def _limit(self) -> SessionReply:
        self._expire()
        return SessionReply("limit", "Dictation limit reached; nothing was submitted.")

    def select(self, thread_id: str, name: str) -> None:
        if not _valid_target_id(thread_id):
            raise ValueError("invalid Codex target id")
        safe_name = _display_name(name)
        changed = thread_id != self.target_id or safe_name != self.target_name
        self.target_id = thread_id
        self.target_name = safe_name
        if changed:
            self.target_revision += 1
            self._expire()

    def begin(self, initial: str = "") -> SessionReply:
        self._expire()
        if not self.target_id:
            return SessionReply("error", "Select a Codex chat first.")
        # A new dictation cannot inherit an old confirmation.
        self._confirmation = None
        if not isinstance(initial, str):
            initial = str(initial)
        self._deadline = self.clock() + self.timeout_seconds
        self._hard_deadline = self.clock() + 270
        self._deadline = min(self._deadline, self._hard_deadline)
        self._segments = 0
        self.draft = ""
        if initial:
            if len(initial) > self.max_chars:
                return self._limit()
            self.draft = initial
            self._segments = 1
            self.phase = "review"
            return SessionReply("review", self._ready_message())
        self.phase = "dictating"
        return SessionReply("dictating", f"Dictating for {self.target_name}.")

    def reset(self) -> None:
        self._expire()

    def bind_review(self, plan_id: str) -> None:
        if self.phase != "review" or not self.active:
            raise ValueError("no review to bind")
        self._plan_id = plan_id

    @property
    def deadline(self):
        if self._deadline is None:
            return None
        elapsed = max(0, self.clock() - self._feedback[1]) if self._feedback else 0
        return min(self._deadline + elapsed, self._hard_deadline)

    @property
    def hard_deadline(self):
        return self._hard_deadline

    def constrain_deadline(self, hard_deadline: float) -> None:
        """A restored local draft cannot extend its original absolute cap."""
        if self._hard_deadline is not None and self._deadline is not None:
            self._hard_deadline = min(self._hard_deadline, hard_deadline)
            self._deadline = min(self._deadline, self._hard_deadline)

    @property
    def feedback_elapsed(self):
        return max(0, self.clock() - self._feedback[1]) if self._feedback else 0

    def begin_feedback(self):
        if not self.active or self._feedback is not None:
            return None
        token = secrets.token_hex(16)
        self._feedback = (token, self.clock(), self.target_id, self.target_revision, self._plan_id)
        return token

    def end_feedback(self, token, success: bool) -> bool:
        feedback = self._feedback
        if not feedback or token != feedback[0]:
            return False
        if (type(success) is not bool or not success or not self.active
                or feedback[2:] != (self.target_id, self.target_revision, self._plan_id)):
            self._expire()
            return False
        deadline = self.deadline
        self._feedback = None
        self._deadline = deadline
        return True

    def clear_selection(self) -> None:
        """Clear a temporary selection without inventing a real target."""
        self.target_id = self.target_name = ""
        self.target_revision += 1
        self._expire()

    def shutdown(self) -> None:
        """Invalidate active dictation and any outstanding confirmation."""
        self._expire()

    def consume(self, text: str) -> SessionReply:
        if not self.active:
            return SessionReply("expired" if self._deadline is None else "error", "No active dictation.")
        if self._feedback is not None:
            return SessionReply("error", "Wait until the readback finishes. Nothing sent.")

        raw = text if isinstance(text, str) else str(text)
        control = _control_text(raw)

        if control in {
            "cancel",
            "cancel that",
            "cancelthat",
            "cancel dictation",
            "canceldictation",
            "stop",
            "stop that",
            "stopthat",
            "stop dictation",
            "stopdictation",
        }:
            self._expire()
            return SessionReply("cancel", "Dictation cancelled.")

        if control in {"finish", "done", "finish dictation", "finishdictation"}:
            if not self.draft:
                return SessionReply("error", "There is no dictation to review.")
            self.phase = "review"
            return SessionReply("review", self._ready_message())

        if control in {"read it back", "read back", "readitback"}:
            if not self.draft:
                return SessionReply("error", "There is no dictation to read back.")
            return SessionReply("readback", prompt=self.draft[: self.max_chars])

        if control in {"edit prompt", "editprompt", "start again", "startagain"}:
            if self._plan_id:
                self._expire()
                return SessionReply("cancel", "Request cancelled. Say Alexa and give the revised request.")
            self.phase = "dictating"
            self.draft = ""
            self._segments = 0
            return SessionReply("dictating", f"Starting again for {self.target_name}.")

        if control in {"send it", "send the prompt", "submit prompt", "sendit", "sendtheprompt", "submitprompt"}:
            if self.phase != "review":
                return SessionReply("error", "Say finish before sending the prompt.")
            if not self.draft:
                return SessionReply("error", "There is no prompt to send.")
            prompt, target = self.draft, self.target_id
            confirmation_id = secrets.token_hex(12)
            expires_at = self._deadline if self._deadline is not None else self.clock()
            self._confirmation = {
                "id": confirmation_id,
                "prompt": prompt,
                "target_id": target,
                "target_revision": self.target_revision,
                "plan_id": self._plan_id,
                "expires_at": expires_at,
            }
            self._clear_after_submit()
            return SessionReply(
                "submit",
                prompt=prompt,
                target_id=target,
                confirmation_id=confirmation_id,
                plan_id=self._plan_id,
            )

        # In review, unrecognized speech must never become a new prompt or a
        # second command.  While dictating, every other segment is payload,
        # including phrases containing words such as "send it".
        if self.phase == "review":
            return SessionReply("error", "Say send it, start again, or cancel.")

        if not raw:
            return SessionReply("error", "Dictation segment is empty.")
        candidate = raw if not self.draft else f"{self.draft} {raw}"
        if self._segments + 1 > self.max_segments or len(candidate) > self.max_chars:
            return self._limit()
        self.draft = candidate
        self._segments += 1
        return SessionReply("dictating")

    def take_confirmation(self, reply: SessionReply) -> bool:
        """Atomically consume a valid, unexpired submit confirmation once."""

        pending = self._confirmation
        if pending is None or not isinstance(reply, SessionReply):
            return False
        if reply.action != "submit":
            return False
        if (self.clock() >= float(pending["expires_at"])
                or (self._hard_deadline is not None and self.clock() >= self._hard_deadline)):
            self._confirmation = None
            return False
        if (
            reply.confirmation_id != pending["id"]
            or reply.prompt != pending["prompt"]
            or reply.target_id != pending["target_id"]
            or self.target_id != pending["target_id"]
            or self.target_revision != pending["target_revision"]
            or reply.plan_id != pending["plan_id"]
        ):
            return False
        self._confirmation = None
        return True

    @property
    def confirmation_expires_at(self):
        """Read-only deadline of the issued token, without consuming it."""
        return float(self._confirmation["expires_at"]) if self._confirmation else None


__all__ = [
    "GlobalCommand",
    "MAX_PROMPT_CHARS",
    "MAX_PROMPT_SEGMENTS",
    "SessionReply",
    "VoiceSession",
    "parse_global",
]
