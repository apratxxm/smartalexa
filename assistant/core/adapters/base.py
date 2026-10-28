"""Shared adapter contract (docs/CAPABILITY_CONTRACT.md). Lead-owned; adapters import, never edit."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Protocol

from ...conversation_contracts import CAPABILITIES
from ..executor import Receipt
from ..playback import TurnExpired

Guard = Callable[[], float]  # remaining seconds; raises TurnExpired once the turn is stale


@dataclass(frozen=True)
class Clause:
    """The single whole-utterance command core's grammar parsed. Adapters never re-parse the transcript."""
    text: str
    verb: str
    object: str = ""
    polite: bool = False
    suspicious: bool = False        # C1: core found a verb/negation/correction token inside the object


class Handler(Protocol):
    mutating: bool
    verbs: tuple[str, ...]          # verb phrases this capability claims, lowercase, e.g. ("open", "launch")
    object_kind: str                # "none" | "text" | "number"
    effect_words: tuple[str, ...]   # words a reply may only use with an observed receipt
    # Optional:
    #   c1_exempt: bool = False     -- non-destructive handlers only (search_browser); lead-approved
    #   def exact_match(self, args: dict, guard: Guard) -> bool
    #       C1: for a suspicious clause, core calls this before execute(); a mutating handler
    #       without it (and not c1_exempt) is refused. Must not mutate anything.

    def license(self, clause: Clause) -> dict | None: ...
    def execute(self, args: dict, guard: Guard) -> Receipt: ...
    def reply(self, receipt: Receipt, args: dict) -> str: ...


_KINDS = {"none", "text", "number"}


@dataclass(frozen=True)
class Adapter:
    area: str
    phase: str                      # "1", "1b", "3" or "4"
    handlers: Mapping[str, Handler] = field(default_factory=dict)

    def __post_init__(self):
        if self.phase not in {"1", "1b", "3", "4"}:
            raise ValueError("unknown phase %r" % self.phase)
        for name, handler in self.handlers.items():
            if name not in CAPABILITIES and name != "answer":
                raise ValueError("capability %r is not in conversation_contracts.CAPABILITIES" % name)
            if getattr(handler, "object_kind", None) not in _KINDS:
                raise ValueError("%s: object_kind must be one of %s" % (name, sorted(_KINDS)))
            verbs = getattr(handler, "verbs", ())
            if not verbs or any(not v or v != v.strip().lower() for v in verbs):
                raise ValueError("%s: verbs must be non-empty lowercase phrases" % name)


__all__ = ["Adapter", "Clause", "Guard", "Handler", "Receipt", "TurnExpired"]
