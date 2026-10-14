"""Small turn coordinator for the opt-in conversation interpreter."""
from __future__ import annotations

import time
import threading
import uuid
from dataclasses import dataclass
from typing import Any, Callable

from .conversation_contracts import (
    ContractError, Decision, Step, ToolResult, Transcript,
    validate_decision,
)


@dataclass(frozen=True)
class ConversationResult:
    decision_kind: str
    text: str = ""
    outcome: str = "handled"
    results: tuple[ToolResult, ...] = ()
    generation: int = 0
    latency_ms: int = 0


@dataclass(frozen=True)
class PendingClarification:
    sources: tuple[str, ...]
    question: str
    expires_at: float
    attempts: int = 0


class ConversationCoordinator:
    """Interpret a whole turn once, then run only sequentially verified steps.

    ``execute`` and ``delegate`` are application adapters. This module owns no
    devices, I/O, speech, or model lifecycle.
    """

    def __init__(self, interpreter: Any,
                 execute: Callable[[Step, str], ToolResult],
                 delegate: Callable[[Step, str], ToolResult],
                 clock: Callable[[], float] | None = None,
                 clarification_ttl_seconds: float = 30.0,
                 max_clarification_turns: int = 2) -> None:
        self.interpreter = interpreter
        self.execute = execute
        self.delegate = delegate
        self.clock = clock or time.perf_counter
        self._generation = 0
        self._state_lock = threading.RLock()
        self.clarification_ttl_seconds = clarification_ttl_seconds
        self.max_clarification_turns = max_clarification_turns
        self.pending_clarification: PendingClarification | None = None

    @property
    def generation(self) -> int:
        with self._state_lock:
            return self._generation

    def _begin_turn(self, expected_generation=None):
        with self._state_lock:
            if expected_generation is not None and expected_generation != self._generation:
                return None
            self._generation += 1
            return self._generation, self.pending_clarification

    def _clear_pending_if_current(self, generation: int) -> bool:
        with self._state_lock:
            if generation != self._generation:
                return False
            self.pending_clarification = None
            return True

    def _set_pending_if_current(self, generation: int,
                                pending: PendingClarification | None) -> bool:
        with self._state_lock:
            if generation != self._generation:
                return False
            self.pending_clarification = pending
            return True

    def invalidate(self) -> int:
        with self._state_lock:
            self.pending_clarification = None
            self._generation += 1
            return self._generation

    def invalidate_if_current(self, expected_generation: int) -> bool:
        """Cancel this turn without clearing a newer turn's question."""
        with self._state_lock:
            if self._generation != expected_generation:
                return False
            self.pending_clarification = None
            self._generation += 1
            return True

    def _result(self, kind, text, outcome, generation, started, results=()):
        return ConversationResult(kind, text, outcome, tuple(results), generation,
                                  max(0, int((self.clock() - started) * 1000)))

    @staticmethod
    def _failure_text(status: str) -> str:
        return {
            "accepted": "The request was accepted and is still pending.",
            "deferred": "That request is delayed. Please try again shortly.",
            "unavailable": "That capability is unavailable right now.",
            "needs_clarification": "I need one more detail before I can continue.",
            "not_found": "I couldn't find a matching result.",
            "cancelled": "That request was cancelled.",
            "failed": "Sorry, that action failed.",
            "unknown_delivery": "I can't confirm whether that action happened.",
        }.get(status, "I can't confirm whether that action happened.")

    @staticmethod
    def _revalidate(decision: Decision, transcript: Transcript) -> Decision:
        """Do not trust even a typed decision injected around the brain schema."""
        if decision.kind == "answer":
            payload = {"kind": "answer", "answer": decision.text,
                       "basis": "general_knowledge",
                       "evidence_quotes": list(decision.evidence_quotes)}
        elif decision.kind == "clarify":
            payload = {"kind": "clarify", "question": decision.text,
                       "evidence_quotes": list(decision.evidence_quotes)}
        elif decision.kind == "plan":
            if not isinstance(decision.steps, tuple) or not 1 <= len(decision.steps) <= 3:
                raise ContractError("plan must contain one to three steps")
            payload = {"kind": "plan", "steps": [
                {"id": step.id, "capability": step.capability,
                 "arguments": step.arguments, "after": list(step.after)}
                for step in decision.steps],
                "evidence_quotes": list(decision.evidence_quotes)}
            if decision.answer or decision.answer_basis:
                payload["answer"] = decision.answer
                payload["answer_basis"] = decision.answer_basis
        elif decision.kind in {"no_request", "unsupported"}:
            payload = {"kind": decision.kind, "reason": decision.reason,
                       "evidence_quotes": list(decision.evidence_quotes)}
        else:
            raise ContractError("unknown decision kind")
        return validate_decision(payload, transcript)

    def handle(self, transcript, context=None, *, expected_generation=None) -> ConversationResult:
        from .conversation_contracts import Transcript

        started = self.clock()
        admitted = self._begin_turn(expected_generation)
        if admitted is None:
            return self._result("unsupported", "", "invalidated", expected_generation, started)
        generation, pending = admitted
        if not isinstance(transcript, Transcript):
            if not self._clear_pending_if_current(generation):
                return self._result("unsupported", "", "invalidated", generation, started)
            return self._result("unsupported", "I couldn't process that request.",
                                "invalid_transcript", generation, started)
        if transcript.truncated:
            if not self._clear_pending_if_current(generation):
                return self._result("clarify", "", "invalidated", generation, started)
            return self._result("clarify", "That was cut short. Please repeat the full request.",
                                "truncated", generation, started)
        # Caller-provided grounding is never authority. Only a live pending
        # clarification owned by this coordinator can add prior source text.
        transcript = Transcript(transcript.text, transcript.turn_id, transcript.truncated)
        if pending is not None and self.clock() >= pending.expires_at:
            if not self._clear_pending_if_current(generation):
                return self._result("clarify", "", "invalidated", generation, started)
            return self._result("clarify", "That question expired. Please restate the full request.",
                                "clarification_expired", generation, started)
        if pending is not None and transcript.text.strip().casefold().rstrip(".!?") in {
                "cancel", "never mind", "nevermind"}:
            if not self._clear_pending_if_current(generation):
                return self._result("cancelled", "", "invalidated", generation, started)
            return self._result("cancelled", "Clarification cancelled.",
                                "cancelled", generation, started)
        if pending is not None:
            if len(transcript.text) > 1000 or pending.attempts >= self.max_clarification_turns:
                if not self._clear_pending_if_current(generation):
                    return self._result("clarify", "", "invalidated", generation, started)
                return self._result("clarify", "Please restate the full request.",
                                    "clarification_restate", generation, started)
            # The current reply remains transcript.text. Grounding sources are
            # the original request plus prior replies, never a rewritten prompt.
            transcript = Transcript(transcript.text, transcript.turn_id, False,
                                    pending.sources)
            effective_context = dict(context) if isinstance(context, dict) else {}
            effective_context["pending_request"] = {
                "sources": list(pending.sources), "question": pending.question}
            context = effective_context
        try:
            decision: Decision = self.interpreter.decide(transcript, context=context)
        except Exception as exc:
            if generation != self.generation:
                return self._result("unsupported", "", "invalidated", generation, started)
            if not self._clear_pending_if_current(generation):
                return self._result("unsupported", "", "invalidated", generation, started)
            # Interpreter errors are terminal for this turn. The caller must
            # never reinterpret the same source with the legacy parser.
            kind = ("unavailable" if type(exc).__name__ == "InterpreterUnavailable"
                    else "failed")
            return self._result("unsupported", "I couldn't understand that request. Please try again.",
                                "interpreter_" + kind, generation, started)
        if generation != self.generation:
            return self._result(getattr(decision, "kind", "unsupported"), "",
                                "invalidated", generation, started)
        if not isinstance(decision, Decision):
            if not self._clear_pending_if_current(generation):
                return self._result("unsupported", "", "invalidated", generation, started)
            return self._result("unsupported", "I couldn't process that request.",
                                "invalid_decision", generation, started)
        try:
            decision = self._revalidate(decision, transcript)
        except Exception:
            if generation != self.generation:
                return self._result("unsupported", "", "invalidated", generation, started)
            if not self._clear_pending_if_current(generation):
                return self._result("unsupported", "", "invalidated", generation, started)
            return self._result("unsupported", "I couldn't process that request.",
                                "invalid_decision", generation, started)
        if generation != self.generation:
            return self._result("unsupported", "", "invalidated", generation, started)
        if decision.kind == "answer":
            if not self._clear_pending_if_current(generation):
                return self._result("answer", "", "invalidated", generation, started)
            return self._result("answer", decision.text, "answered", generation, started)
        if decision.kind == "clarify":
            if pending is None:
                if len(transcript.text) <= 1000:
                    stored = self._set_pending_if_current(generation, PendingClarification(
                        (transcript.text,), decision.text,
                        self.clock() + self.clarification_ttl_seconds, 0))
                else:
                    if not self._clear_pending_if_current(generation):
                        return self._result("clarify", "", "invalidated", generation, started)
                    return self._result("clarify", "Please restate the full request.",
                                        "clarification_restate", generation, started)
            else:
                attempts = pending.attempts + 1
                if attempts >= self.max_clarification_turns:
                    if not self._clear_pending_if_current(generation):
                        return self._result("clarify", "", "invalidated", generation, started)
                    return self._result("clarify", "Please restate the full request.",
                                        "clarification_restate", generation, started)
                original = pending.sources[0] if pending.sources else ""
                sources = tuple(item for item in (original, transcript.text) if item)[-2:]
                stored = self._set_pending_if_current(generation, PendingClarification(
                    sources, decision.text, pending.expires_at, attempts))
            if not stored:
                return self._result("clarify", "", "invalidated", generation, started)
            return self._result("clarify", decision.text, "clarification_needed", generation, started)
        if decision.kind == "no_request":
            if generation != self.generation:
                return self._result("no_request", "", "invalidated", generation, started)
            if pending is not None and self.clock() >= pending.expires_at:
                if not self._clear_pending_if_current(generation):
                    return self._result("no_request", "", "invalidated", generation, started)
                return self._result("clarify", "That question expired. Please restate the full request.",
                                    "clarification_expired", generation, started)
            return self._result("no_request", "", "no_request", generation, started)
        if decision.kind == "unsupported":
            if not self._clear_pending_if_current(generation):
                return self._result("unsupported", "", "invalidated", generation, started)
            return self._result("unsupported", decision.reason, "unsupported", generation, started)
        if decision.kind != "plan":
            if not self._clear_pending_if_current(generation):
                return self._result("unsupported", "", "invalidated", generation, started)
            return self._result("unsupported", "I couldn't process that request.",
                                "invalid_decision", generation, started)

        results: list[ToolResult] = []
        completed: set[str] = set()
        for step in decision.steps:
            if generation != self.generation:
                return self._result("plan", "", "invalidated", generation, started, results)
            if any(parent not in completed for parent in step.after):
                if not self._clear_pending_if_current(generation):
                    return self._result("plan", "", "invalidated", generation, started, results)
                return self._result("plan", "", "dependency_not_completed", generation,
                                    started, results)
            # IDs belong to the application turn, never the model. Include the
            # coordinator generation so retries sharing an event ID stay distinct.
            operation_id = f"{transcript.turn_id or 'turn'}:{generation}:{uuid.uuid4().hex}:{step.id}"
            try:
                result = (self.delegate if step.capability == "codex_request" else self.execute)(
                    step, operation_id)
            except Exception:
                # A callback can raise after an external side effect. Without a
                # receipt, report uncertainty and stop; never retry the step.
                result = ToolResult(operation_id, "unknown_delivery", {"feedback_spoken": False})
            known_statuses = {"completed", "accepted", "needs_clarification", "not_found",
                              "unavailable", "deferred", "failed", "unknown_delivery", "cancelled"}
            if (not isinstance(result, ToolResult) or result.operation_id != operation_id
                    or not isinstance(result.status, str) or result.status not in known_statuses
                    or not isinstance(result.data, dict)):
                result = ToolResult(operation_id, "unknown_delivery", {"feedback_spoken": False})
            # The adapter's effect may already have happened. Record its valid
            # receipt before cancellation suppresses all later work and speech.
            results.append(result)
            if generation != self.generation:
                return self._result("plan", "", "invalidated", generation, started, results)
            if result.status != "completed":
                status_text = ("" if result.data.get("feedback_spoken") is True
                               else self._failure_text(result.status))
                text = " ".join(part for part in (status_text, decision.answer) if part)[:1700]
                if not self._clear_pending_if_current(generation):
                    return self._result("plan", "", "invalidated", generation, started, results)
                return self._result("plan", text, "partial_" + result.status,
                                    generation, started, results)
            completed.add(step.id)
        if not self._clear_pending_if_current(generation):
            return self._result("plan", "", "invalidated", generation, started, results)
        return self._result("plan", decision.answer, "completed", generation, started, results)
