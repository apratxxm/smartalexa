"""Turn orchestration: planning fallback is confined to pre-execution admission."""
from __future__ import annotations

import logging
import re
import time
import uuid
from dataclasses import asdict, replace
from types import SimpleNamespace

from ..conversation import ConversationCoordinator
from ..conversation_contracts import Decision, Transcript
from .adapters.knowledge import KNOWLEDGE, public_topic
from .capture import DeviceFailure, admissible
from .executor import Executor, Receipt, coordinator_result
from .fallback import admit
from .provider import ProviderError
from .authority import REFUSAL, NOT_DONE, _registry_default, parsed_licenses, admit_step, claims_effect, licensed_codex_args
from .transcription import NoWakeBoundary

_TIMINGS = ("wake_to_ack_ms", "speech_end", "stt_ms", "plan_ms",
            "execute_to_observed_ms", "reply_ms")


def one_sentence(text):
    text = " ".join(str(text).split())[:400]
    return re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0]


def receipt_reply(receipts):
    parts = []
    for receipt in receipts:
        state, cap = receipt["state"], receipt["capability"]
        if state in {"requested", "unknown"} and cap in Executor.ALLOWED:
            parts.append("I asked Windows but can't confirm the volume changed." if
                         cap in {"set_volume", "change_volume"} else
                         "I asked Spotify but can't confirm it played.")
        else:
            # Adapter receipts carry their fixed template in detail.
            parts.append(receipt["detail"] or "I couldn't complete that request.")
    return "; ".join(part.rstrip(".!?") for part in parts) + "." if parts else ""


class CoreRuntime:
    def __init__(self, provider, executor, feedback, *, toast=lambda text: None,
                 log=lambda row: None, clock=time.monotonic, reply_builder=receipt_reply, registry=None,
                 knowledge=False, knowledge_source=None):
        self.provider, self.executor, self.feedback = provider, executor, feedback
        self.registry = registry if registry is not None else _registry_default()
        for handler in self.registry.handlers.values():
            if hasattr(handler, "executor"):
                handler.executor = executor
        self.routed = dict(self.registry.handlers)
        if isinstance(executor, Executor):
            executor.handlers = self.routed
        self.knowledge, self.knowledge_source = bool(knowledge), knowledge_source or KNOWLEDGE
        self.toast, self.log, self.clock, self.reply_builder = toast, log, clock, reply_builder
        self.current_turn = None
        self.row = None
        self.deadline = 0.0
        self.stage = "capture"
        self.execution_started = False
        self.coordinator = ConversationCoordinator(
            SimpleNamespace(decide=self._plan), self._execute, self._execute, clock=clock)

    def _new_row(self, turn_id=None):
        return {"turn_id": turn_id or uuid.uuid4().hex, "wake_score": None,
                "capture": {"end_reason": None, "continuous": None, "truncated": None},
                "transcript": "", "path": "refused", "decision_summary": None,
                "receipts": [], "timings": {name: None for name in _TIMINGS},
                "provider_error": None, "errors": [], "outcome": "pending",
                "trimmed_prefix_length": None, "authority_refused": False, "action_not_executed": False}

    def _error(self, exc, stage=None):
        item = {"stage": stage or self.stage, "type": type(exc).__name__}
        if isinstance(exc, ProviderError):
            item.update(category=exc.category, reason=exc.reason)
        self.row["errors"].append(item)

    def _current(self, transcript):
        return self.current_turn == transcript.turn_id and self.clock() < self.deadline

    def _plan(self, transcript, context=None):
        self.stage = "plan"
        started = self.clock()
        self.row["path"] = "brain"
        try:
            decision = self.provider.decide(
                transcript, context, deadline=self.deadline,
                current_turn=lambda: self.current_turn)
        except ProviderError as exc:
            self.row["provider_error"] = exc.category
            self._error(exc)
            decision = None
            if (exc.category in {"unavailable", "rate_limited"} and not self.execution_started
                    and self._current(transcript)):
                decision = admit(transcript.text, self.registry, lambda: max(0.0, self.deadline - self.clock()) if self.current_turn == transcript.turn_id else 0.0)
            if decision is None:
                self.row["path"] = "refused"
                self.row["authority_refused"] = True
                decision = Decision("unsupported", reason="Planning unavailable.")
            else:
                self.row["path"] = "fallback"
        except Exception as exc:
            self._error(exc)
            raise
        finally:
            self.row["timings"]["plan_ms"] = (self.clock() - started) * 1000
        if not self._current(transcript):
            self.coordinator.invalidate()
            self.row["path"], self.row["outcome"] = "refused", "expired_or_stale"
        if (isinstance(decision, Decision) and decision.kind == "plan"
                and any(step.capability not in self.executor.ALLOWED and step.capability not in self.routed
                        for step in decision.steps)):
            decision = Decision("unsupported", reason="Capability unavailable in phase one.")
            self.row["path"] = "refused"
        elif isinstance(decision, Decision) and decision.kind == "plan":
            if len(decision.steps) != 1 or not all(admit_step(step, transcript.text, self.registry, lambda: max(0.0, self.deadline - self.clock()) if self.current_turn == transcript.turn_id else 0.0) for step in decision.steps):
                self.row["authority_refused"], self.row["path"] = True, "refused"
                decision = Decision("unsupported", reason=REFUSAL)
            else:
                # Legacy coordinator validation requires the exact transcript. Authority
                # has checked either representation; execution uses the licensed object.
                decision = replace(decision, steps=tuple(
                    replace(step, arguments={"request": transcript.text})
                    if step.capability == "codex_request" else step
                    for step in decision.steps))
        elif isinstance(decision, Decision) and decision.kind in {"answer", "clarify"}:
            if not self.registry.answer_enabled:
                self.row["path"] = "refused"
                decision = Decision("unsupported", reason="Answers are disabled.")
            elif bool(parsed_licenses(transcript.text, self.registry)[1]) or claims_effect(decision.text, self.registry.handlers.values()):
                self.row["action_not_executed"], self.row["path"] = True, "refused"
                decision = Decision("unsupported", reason=NOT_DONE)
            elif self.knowledge and "4" in self.registry.enabled_phases and decision.kind == "answer" and self._current(transcript):
                topic = public_topic(transcript.text)
                if topic:
                    decision = self._knowledge_answer(transcript, decision, topic)
        self.row["decision_summary"] = (
            {"kind": decision.kind, "capabilities": [s.capability for s in decision.steps]}
            if isinstance(decision, Decision) else {"kind": "invalid"})
        return decision

    def _knowledge_answer(self, transcript, decision, topic):
        """Replace a model answer with one bounded synthesis over retrieved passages.

        Any failure keeps the model's original answer, which already passed the effect guard.
        """
        def guard():
            return max(0.0, self.deadline - self.clock()) if self.current_turn == transcript.turn_id else 0.0
        try:
            passages = self.knowledge_source.retrieve(topic, guard)
        except Exception as exc:
            self._error(exc, "knowledge")
            return decision
        if not passages:
            return decision
        try:
            text = self.provider.synthesize(transcript.text, passages, turn_id=transcript.turn_id,
                                            deadline=self.deadline,
                                            current_turn=lambda: self.current_turn)
            text = one_sentence(text) if isinstance(text, str) else ""
            if not text or claims_effect(text, self.registry.handlers.values()):
                raise ValueError("synthesis rejected")
        except Exception as exc:
            self._error(exc, "synthesis")
            return decision
        self.row["knowledge_source"] = passages[0].title
        return replace(decision, text=text)

    def _execute(self, step, operation_id):
        self.stage = "execute"
        self.execution_started = True
        started = self.clock()
        if not admit_step(step, self.row["transcript"], self.registry, lambda: max(0.0, self.deadline - self.clock()) if self.current_turn == self.row["turn_id"] else 0.0):
            receipt = Receipt("failed", REFUSAL)
        elif self.current_turn != self.row["turn_id"] or started >= self.deadline:
            receipt = Receipt("failed", "That turn expired.")
        else:
            try:
                if step.capability == "codex_request":
                    step = replace(step, arguments=licensed_codex_args(
                        self.row["transcript"], self.registry))
                receipt = self.executor.execute(step, operation_id, guard=lambda:
                    max(0.0, self.deadline - self.clock()) if self.current_turn == self.row["turn_id"] else 0.0)
                if not isinstance(receipt, Receipt):
                    raise TypeError("invalid receipt")
            except Exception as exc:
                self._error(exc)
                receipt = Receipt("unknown", "I can't confirm the result.", error=type(exc).__name__)
        self.row["receipts"].append(dict(asdict(receipt), operation_id=operation_id,
                                         capability=step.capability))
        elapsed = (self.clock() - started) * 1000
        self.row["timings"]["execute_ms"] = self.row["timings"].get("execute_ms", 0) + elapsed
        if receipt.state == "observed":
            self.row["timings"]["execute_to_observed_ms"] = (
                self.row["timings"]["execute_to_observed_ms"] or 0) + elapsed
        if receipt.error:
            self.row["errors"].append({"stage": "execute", "type": receipt.error})
        return coordinator_result(operation_id, receipt)

    def _reply(self, result):
        if self.row["authority_refused"]:
            return REFUSAL
        if self.row["action_not_executed"]:
            return NOT_DONE
        if self.row["receipts"]:
            try:
                return self.reply_builder(self.row["receipts"])
            except Exception as exc:
                # Synthesis/rendering is terminal after an effect. Report the
                # existing receipts locally, with no reinterpretation or repeat.
                self._error(exc, "reply")
                return receipt_reply(self.row["receipts"])
        if result.outcome == "invalidated":
            return ""
        kind = result.decision_kind
        if kind == "no_request":
            return ""
        if kind == "answer":
            return result.text
        if kind == "clarify":
            return result.text  # Speak one question, then drop the pending turn.
        if kind == "unsupported":
            if self.row["provider_error"] in {"unavailable", "rate_limited"}:
                return "I can't reach my brain right now."
            return "I can't do that yet."
        return "I couldn't process that request."

    def _emit(self, reply, *, clarification=False):
        self.stage = "reply"
        started = self.clock()
        try:
            reply = " ".join(str(reply).split())[:200] if clarification else one_sentence(reply)
            if not reply:
                return
            did = ", ".join(item["capability"] for item in self.row["receipts"])
            did = did or (self.row["decision_summary"] or {}).get("kind", "none")
            try:
                line = "Heard: %s · Did: %s · Result: %s" % (self.row["transcript"], did, reply)
                if self.row.get("knowledge_source"):
                    line += " · Source: %s" % self.row["knowledge_source"]
                self.toast(line)
            except Exception as exc:
                self._error(exc, "toast")
            self.feedback.say(reply, force=True, blocking=True)
            self.row["reply"] = reply
        except Exception as exc:
            self._error(exc)
        finally:
            self.row["timings"]["reply_ms"] = (self.clock() - started) * 1000

    def _finish(self):
        self.coordinator.invalidate()  # Clarifications require a fresh full wake turn.
        self.current_turn = None
        try:
            self.log(self.row)
        except Exception:
            logging.exception("Core turn log failed")

    def process(self, transcript, *, row=None, deadline=None, write_log=True):
        self.row = row if row is not None else self._new_row(transcript.turn_id)
        self.row["transcript"] = transcript.text
        self.current_turn, self.execution_started = transcript.turn_id, False
        self.deadline = deadline if deadline is not None else self.clock() + self.provider.timeout
        self.stage = "plan"
        try:
            result = self.coordinator.handle(transcript)
            if self.row["outcome"] != "expired_or_stale":
                self.row["outcome"] = result.outcome
            self._emit(self._reply(result), clarification=result.decision_kind == "clarify")
        except Exception as exc:
            self._error(exc)
            self.row["outcome"] = "turn_error"
            self._emit(receipt_reply(self.row["receipts"]) if self.row["receipts"]
                       else "I couldn't process that request.")
        finally:
            if write_log:
                self._finish()
        return self.row

    def step(self, capture):
        self.row = self._new_row()
        self.stage = "capture"
        device_failure = None
        try:
            result = capture.next_capture(acknowledge=self.feedback.tick)
            self.row.update(turn_id=result.turn_id, wake_score=result.wake_score)
            self.row["capture"] = {"end_reason": result.end_reason.value,
                                   "continuous": result.continuous, "truncated": result.truncated}
            self.row["timings"].update(result.timings)
            if not admissible(result, capture.turn_id):
                self.row["outcome"] = "capture_rejected"
            else:
                self.stage = "stt"
                started = self.clock()
                try:
                    text, _ = capture.transcribe(result)
                    self.row["trimmed_prefix_length"] = getattr(capture, "trimmed_prefix_length", None)
                finally:
                    self.row["timings"]["stt_ms"] = (self.clock() - started) * 1000
                self.process(Transcript(text, result.turn_id, not result.complete), row=self.row,
                             deadline=result.wake_at + self.provider.timeout, write_log=False)
        except Exception as exc:
            if getattr(capture, "turn_id", ""):
                self.row.update(turn_id=capture.turn_id, wake_score=capture.wake_score)
            self._error(exc)
            if isinstance(exc, NoWakeBoundary):
                self.row["wake_boundary"] = {"reason": exc.reason, "prefix_length": exc.prefix_length}
                self.row["trimmed_prefix_length"] = exc.prefix_length
            self.row["outcome"] = ("device_error" if isinstance(exc, DeviceFailure) else
                                   "capture_rejected: no wake boundary" if isinstance(exc, NoWakeBoundary) else "turn_error")
            if isinstance(exc, DeviceFailure):
                device_failure = exc
            else:
                self._emit(REFUSAL if isinstance(exc, NoWakeBoundary) else "I couldn't process that request.")
        finally:
            try:
                capture.reset()
            except Exception as exc:
                self._error(exc, "capture_reset")
                self.row["outcome"] = "device_error"
                device_failure = exc if isinstance(exc, DeviceFailure) else DeviceFailure(type(exc).__name__)
            self._finish()
        if device_failure is not None:
            raise device_failure
        return self.row
