"""Global capabilities above the music brain, with an isolated Codex worker.

Only the audio owner calls handle/poll/session methods. The worker returns data;
it never speaks, touches a microphone, changes music, or mutates voice state.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
import secrets
import re
from dataclasses import dataclass, replace
from collections import deque
from pathlib import Path

from .codex_bridge import CodexBridge, CodexError, DeliveryUnknown, PERMITTED_MODELS
from .desktop_controls import open_codex_thread, switch_browser_tab
from .global_commands import GlobalCommand, VoiceSession, SessionReply, parse_global, _control_text, _display_name, creation_model
from .global_router import routing_candidate, validate_classification

log = logging.getLogger("memoir.global")


@dataclass(frozen=True)
class GlobalReply:
    message: str
    session: bool = False
    await_token: str = ""
    review: bool = False
    status: str = ""


@dataclass(frozen=True)
class _PendingPlan:
    id: str
    kind: str
    source: str
    final_prompt: str
    target_id: str
    revision: int
    expires_at: float
    prior_id: str
    prior_name: str
    title: str = "Memoir voice task"
    model: str = ""
    show_chat: bool = False


@dataclass(frozen=True)
class _PendingLookup:
    token: str
    command: object
    selection: tuple
    expires_at: float


@dataclass(frozen=True)
class _PendingChoice:
    command: object
    rows: tuple
    selection: tuple
    expires_at: float


@dataclass(frozen=True)
class _UnsentRequest:
    source: str
    prompt: str
    selection: tuple
    generation: int
    expires_at: float
    hard_deadline: float
    model: str = ""
    show_chat: bool = False


_PLAN_TARGET_PREFIX = "thr_voice_plan_"
SEND_ACCEPTED = "Sending the confirmed prompt."
CREATE_ACCEPTED = "Creating the confirmed chat."
_NEW_CHAT_CONTEXT = (
    "The outer request to create this Codex chat has already been fulfilled. "
    "Perform the task requested below in this chat; do not repeat that outer creation request."
    "\n\nOriginal voice request:\n"
)
_EXISTING_CHAT_CONTEXT = (
    "The outer request to send a task to this Codex chat has already been routed here. "
    "Perform the requested task in this chat; do not repeat that outer dispatch request."
    "\n\nOriginal voice request:\n"
)


class GlobalAssistant:
    def __init__(self, cfg, root: Path, *, bridge=None, navigator=None, tabs=None, router=None):
        self.root = root.resolve()
        self.bridge = bridge or CodexBridge(self.root, model=cfg.get("codex.model", "gpt-5.6-luna"),
                                          executable=cfg.get("codex.executable") or None)
        self.session = VoiceSession(timeout_seconds=cfg.get("codex.dictation_timeout_seconds", 90),
                                    max_chars=cfg.get("codex.max_prompt_chars", 12000))
        self._navigator = navigator or open_codex_thread
        self._tabs = tabs or switch_browser_tab
        self._results = queue.Queue(maxsize=8)
        self._busy = False
        self._pending_kind = ""
        self._closed = False
        self._cancel_pending = False
        self._threads = []
        self._notice = "No Codex job has been started."
        self._job_started = None
        self._job_limit = float(cfg.get("codex.job_timeout_seconds", 900))
        self._lock = threading.Lock()
        self._approval_review = None
        self._luna_permission = set()
        self._router = router
        self._pending_draft = None
        self._review_notice = False
        self._continuation = None
        self._pending_plan = None
        self._created_thread_id = ""
        self._default_model = cfg.get("codex.model", "gpt-5.6-luna")
        reader = getattr(self.bridge, "last_created", None)
        self._created_receipt = reader() if callable(reader) else None
        self._receipts = deque(maxlen=8)
        self._create_cancel = None
        self._pending_lookup = None
        self._pending_choice = None
        self._lookup_continuation = None
        self._unresolved_request = None
        self._feedback_context = None
        self._notices = deque(maxlen=8)
        self._owned_jobs = set()
        self._current_job = None
        self._review_armed = None
        self._session_restore = None
        self._request_generation = 0
        self._navigation_epoch = 0
        self._unsent_request = None
        self.session.on_invalidate = self._discard_plan

    def _selection(self):
        return (self.session.target_id, self.session.target_name, self.session.target_revision)

    @property
    def operation_kind(self):
        """Read-only name of the current bounded worker operation, if any."""
        with self._lock:
            return self._pending_kind if self._busy else ""

    def _discard_plan(self):
        """Owner-only cleanup; never overwrite a newer explicit selection."""
        plan, self._pending_plan = self._pending_plan, None
        self._pending_lookup = None
        self._pending_choice = None
        self._lookup_continuation = None
        self._feedback_context = None
        self._review_armed = None
        restore, self._session_restore = self._session_restore, None
        if restore and self._selection() == restore[1]:
            self._restore_selection(restore[0])
        if (plan and (plan.kind != "existing-submit" or plan.prior_id != plan.target_id)
                and self.session.target_id == plan.target_id
                and self.session.target_revision == plan.revision):
            if plan.prior_id:
                self.session.select(plan.prior_id, plan.prior_name)
            else:
                self.session.clear_selection()

    def _restore_selection(self, selection):
        if selection[0]:
            self.session.select(selection[0], selection[1])
        else:
            self.session.clear_selection()

    def end_session(self):
        self.session.reset()
        self._discard_plan()

    def _new_intent(self):
        self._request_generation += 1
        self._unsent_request = None
        self.retire_navigation()

    def retire_navigation(self):
        """Owner-only: retire pending views, not jobs, drafts or receipts.

        An explicitly executed desktop action may call this immediately before
        its external opener. No bridge operation or request permission changes.
        """
        self._navigation_epoch += 1
        return self._navigation_epoch

    def _valid_unsent(self, request):
        return bool(request and not self._closed and request.generation == self._request_generation
                    and request.selection == self._selection()
                    and self.session.clock() < min(request.expires_at, request.hard_deadline))

    def _restage_unsent(self, request):
        self.end_session()
        reply = self.session.begin(request.prompt)
        self.session.constrain_deadline(min(request.expires_at, request.hard_deadline))
        if not self.session.active:
            return GlobalReply("That held request expired. Nothing sent.")
        plan = _PendingPlan(secrets.token_hex(16), "existing-submit", request.source, request.prompt,
                            self.session.target_id, self.session.target_revision, self.session.deadline,
                            self.session.target_id, self.session.target_name,
                            model=request.model or self._default_model, show_chat=request.show_chat)
        self._pending_plan = plan
        self.session.bind_review(plan.id)
        self._review_notice = True
        return GlobalReply(self.review_message(), True, review=True)

    def _review_identity(self):
        plan = self._pending_plan
        if self.session.phase != "review" or not self.session.draft:
            return None
        return (plan.id if plan else "", self.session.target_id,
                self.session.target_revision, self.session.draft)

    def begin_feedback(self, review=False):
        token = self.session.begin_feedback() if not self._closed else None
        if token:
            plan = self._pending_plan
            self._feedback_context = (token, plan.id if plan else "", self._review_identity(),
                                      review is True, self._review_armed)
        return token

    def end_feedback(self, token, success: bool) -> bool:
        if token is None:
            if type(success) is not bool or not success:
                self.end_session()
                return False
            return type(success) is bool and success and not self._closed and self._feedback_context is None
        context = self._feedback_context
        if not context or context[0] != token:
            return False
        plan = self._pending_plan
        if context[1] != (plan.id if plan else "") or not self.session.end_feedback(token, success):
            self.end_session()
            return False
        self._feedback_context = None
        if plan:
            self._pending_plan = replace(plan, expires_at=self.session.deadline)
        identity = self._review_identity()
        if identity and context[2] == identity and (context[3] or context[4] == identity):
            self._review_armed = identity
        return True

    def keep_review_waiting(self) -> bool:
        self.poll()
        if self._closed or not self.session.active or not self.session.draft:
            return False
        self.session.phase = "review"
        return True

    def consume(self, text):
        if self._closed:
            return SessionReply("expired", "Codex voice tools are closed. Nothing sent.")
        if _control_text(text) in {"send it", "send the prompt", "submit prompt", "sendit", "sendtheprompt", "submitprompt"}:
            if self._busy or self.bridge.active_turn is not None:
                return SessionReply("error", "Codex is already running. Nothing sent.")
            if self.session.phase == "review" and self._review_armed != self._review_identity():
                return SessionReply("review", self.review_message())
        reply = self.session.consume(text)
        if reply.action == "dictating":
            self._review_armed = None
        if reply.action == "cancel" or _control_text(text) in {"start again", "startagain", "edit prompt", "editprompt"}:
            self._new_intent()
        return reply

    def take_notice(self):
        self.poll()
        return self._notices.popleft() if self._notices else None

    def take_receipt(self):
        """Owner telemetry handoff: never contains a prompt or transport payload."""
        self.poll()
        return self._receipts.popleft() if self._receipts else None

    def _remember_created(self, created, result=None):
        result = result or {}
        row = {"id": created["id"], "name": _display_name(created.get("name") or "Untitled Codex chat"),
               "model": created.get("model") or self._default_model,
               "submitted": result.get("submitted") is True, "turn_id": result.get("turn_id"),
               "status": "submission_unknown" if result.get("submission_uncertain") else
                         "submitted" if result.get("submitted") else "created"}
        self._created_thread_id = row["id"]
        self._created_receipt = row
        self._receipts.append(dict(row))
        log.info("Codex native receipt thread_id=%s model=%s submitted=%s turn_id=%s status=%s",
                 row["id"], row["model"], row["submitted"], row["turn_id"] or "-", row["status"])

    def _request_open(self, thread_id):
        if not thread_id or thread_id.startswith(_PLAN_TARGET_PREFIX):
            return "No actual Codex chat is available to open."
        try:
            result = self._navigator(thread_id)
            if isinstance(result, str) and result.strip():
                return result
            return "Desktop opening returned no receipt. Check Codex; no task was resent."
        except Exception:
            return "The chat exists, but desktop opening could not be requested. No task was resent."

    def _read_created(self):
        receipt = self._created_receipt
        if not receipt:
            raise CodexError("No saved created-chat identity", code="receipt")
        # A persisted receipt does not restore bridge ownership. Verify exact
        # identity and project scope before requesting its desktop view.
        thread = self.bridge.read_thread(receipt["id"])
        if (thread.get("id") or thread.get("threadId")) != receipt["id"]:
            raise CodexError("Created-chat identity changed", code="receipt")
        cwd = thread.get("cwd")
        if not isinstance(cwd, str) or Path(cwd).resolve() != self.root:
            raise CodexError("Saved created chat is outside the project", code="scope")
        return {"id": receipt["id"], "name": thread.get("name") or receipt["name"]}

    def _publish_notice(self):
        self._notices.append(self._notice)

    def _resolve_named(self, command):
        self.end_session()
        token = secrets.token_hex(16)
        self._pending_lookup = _PendingLookup(token, command, self._selection(), self.session.clock() + 90)
        self._lookup_continuation = (token, None, None)
        def catalogue():
            return [row for row in self.bridge.list_threads()
                    if row.get("cwd") and Path(row["cwd"]).resolve() == self.root]
        if not self._launch("resolve_named", catalogue):
            self._pending_lookup = None
            self._lookup_continuation = None
            return GlobalReply("Codex is busy. Nothing sent.")
        return GlobalReply("Looking for " + command.target_name + ". One moment.", await_token=token)

    def _stage_dictation(self, prior_selection=None):
        self.end_session()
        if not self.session.target_id or self.session.target_id.startswith(_PLAN_TARGET_PREFIX):
            return GlobalReply("Create or select a Codex chat first, then ask to dictate a prompt. Nothing sent.")
        reply = self.session.begin()
        if prior_selection and prior_selection != self._selection():
            self._session_restore = (prior_selection, self._selection())
        return GlobalReply(reply.message + " Speak in segments, then say finish dictation. Nothing is sent until you confirm.", self.session.active)

    def _stage_resolved(self, command, prior):
        selected = self._selection()
        reply = (self._stage_dictation(prior) if command.action == "dictate"
                 else self._stage_inferred(command, prior))
        if not reply.session and self._selection() == selected:
            self._restore_selection(prior)
        return reply

    def _choice_message(self, choice):
        names = [_display_name(row.get("name") or "Untitled") for row in choice.rows]
        return ("I couldn't identify one exact chat. Choose "
                + "; ".join(f"{i + 1}: {name}" for i, name in enumerate(names))
                + ". Say first, second, third, fourth, the exact name, or cancel. Your request is held; nothing sent.")

    @staticmethod
    def _thread_name_key(name):
        return re.sub(r"\s+", "", str(name)).casefold()

    def _handle_choice(self, text):
        choice = self._pending_choice
        if not choice:
            return None
        if self.session.clock() >= choice.expires_at or self._selection() != choice.selection:
            self._pending_choice = None
            return GlobalReply("The chat choice expired or changed. Nothing sent.")
        control = _control_text(text)
        if control in {"cancel", "cancel that", "cancel Codex".casefold(), "start again", "startagain"}:
            self._pending_choice = None
            self._new_intent()
            return GlobalReply("Request cancelled. Nothing sent.")
        if control in {"status", "codex status", "list", "list codex chats"}:
            return GlobalReply(self._choice_message(choice))
        indexes = {"first": 0, "second": 1, "third": 2, "fourth": 3, "1": 0, "2": 1, "3": 2, "4": 3}
        matches = [row for row in choice.rows
                   if self._thread_name_key(row.get("name", "")) == self._thread_name_key(text)]
        if control in indexes:
            matches = list(choice.rows[indexes[control]:indexes[control] + 1])
        if len(matches) != 1:
            return GlobalReply(self._choice_message(choice))
        row = matches[0]
        self._pending_choice = None
        self.session.select(row["id"], row.get("name") or "selected chat")
        command = replace(choice.command, target_kind="selected", target_name="")
        return self._stage_resolved(command, choice.selection)

    def _stage_inferred(self, command, prior_selection=None):
        self.end_session()
        source = command.text
        if (not isinstance(source, str) or not source.strip() or len(source) > self.session.max_chars
                or any(ord(char) < 32 or ord(char) == 127 for char in source)):
            return GlobalReply("That transcript cannot be safely reviewed. Nothing sent.")
        prior_id, prior_name, _ = self._selection()
        if prior_id.startswith(_PLAN_TARGET_PREFIX):
            return GlobalReply("That temporary target is no longer valid. Nothing sent.")
        if command.target_kind == "selected" and not prior_id:
            self._unresolved_request = command
            return GlobalReply("No Codex chat is selected. Select or list the project chats, then repeat this request. Nothing sent.")
        kind = ("create-only" if command.action == "create" else
                "create+submit" if command.action == "create_ask" or not prior_id else "existing-submit")
        final_prompt = (_NEW_CHAT_CONTEXT + source if command.inferred and kind == "create+submit" else
                        _EXISTING_CHAT_CONTEXT + source if command.inferred and kind == "existing-submit" else source)
        title = command.title or "Memoir voice task"
        bound_reader = getattr(self.bridge, "bound_model", None)
        bound_model = bound_reader(prior_id) if kind == "existing-submit" and callable(bound_reader) else ""
        model = command.model or bound_model or self._default_model
        if model not in PERMITTED_MODELS:
            return GlobalReply("That requested model is not a permitted Luna model. No chat created or prompt sent.")
        if len(title) > 120 or any(ord(char) < 32 or ord(char) == 127 for char in title):
            return GlobalReply("That chat title cannot be safely reviewed. Nothing sent.")
        if len(final_prompt) > self.session.max_chars:
            return GlobalReply("The complete reviewed prompt is too long. Nothing sent. Please shorten the request.")
        if kind != "existing-submit":
            self.session.select(_PLAN_TARGET_PREFIX + secrets.token_hex(16), "a new Codex chat")
        reply = self.session.begin(final_prompt)
        if not self.session.active:
            if kind != "existing-submit":
                if prior_id:
                    self.session.select(prior_id, prior_name)
                else:
                    self.session.clear_selection()
            return GlobalReply(reply.message)
        plan = _PendingPlan(secrets.token_hex(16), kind, source, final_prompt, self.session.target_id,
                            self.session.target_revision, self.session.clock() + self.session.timeout_seconds,
                            prior_selection[0] if prior_selection else prior_id,
                            prior_selection[1] if prior_selection else prior_name, title,
                            model, command.show_chat or kind != "existing-submit")
        self._pending_plan = plan
        self.session.bind_review(plan.id)
        self._review_notice = True
        return GlobalReply(self._review_message(), True, review=True)

    @staticmethod
    def _failure_message(exc):
        if isinstance(exc, DeliveryUnknown):
            return "Delivery is unknown. Check the chat; I won't resend automatically."
        messages = {
            "quota": "Codex reached its usage limit. I won't retry or upgrade models.",
            "auth": "Codex needs your ChatGPT sign-in. No API key will be used.",
            "model": "The permitted Luna model isn't available. I won't choose an expensive model.",
            "scope": "That chat or action is outside the permitted project.",
            "model_permission": "That chat's model can't be safely preserved. Say use Luna for this Codex chat, then dictate again.",
            "transport": "Codex connection is unavailable. Nothing sent.",
        }
        return messages.get(getattr(exc, "code", ""),
                            "Codex couldn't complete that request. Check its sign-in or model availability.")

    @staticmethod
    def _safe_error_code(exc):
        code = getattr(exc, "code", "")
        allowed = {"auth", "quota", "model", "scope", "model_permission", "transport", "protocol", "receipt"}
        return code if isinstance(code, str) and code in allowed else "other"

    def review_message(self, full=False):
        if not self.session.active:
            return "No prompt awaiting confirmation. Nothing sent."
        if full:
            return self.session.draft
        plan = self._pending_plan
        if plan and plan.kind != "existing-submit":
            purpose = ("Create an empty Codex chat" if plan.kind == "create-only" else
                       "Create a new Codex chat and send")
            shown = " and request its Codex view" if plan.show_chat else ""
            return purpose + " using " + plan.model + shown + ": " + plan.source + ". Say send it, start again, or cancel."
        return ("Send to " + self.session.target_name + " using " + (plan.model or self._default_model if plan else self._default_model)
                + (" and request its Codex view" if plan and plan.show_chat else "") + ": "
                + (plan.source if plan else self.session.draft)
                + ". Say send it, start again, or cancel.")

    def _review_message(self):
        return self.review_message()

    def _create_with_draft(self, prompt, title, model=""):
        return self._stage_inferred(GlobalCommand("create_ask", prompt, title=title, target_kind="new", model=model))

    def continue_creation(self, token):
        """One-use local handoff, consumed only by the active audio owner."""
        self.poll()
        lookup = self._lookup_continuation
        if lookup and lookup[0] == token:
            if self._closed:
                return GlobalReply("That request ended. Nothing sent.")
            if self._busy and lookup[1] is None:
                return None
            self._lookup_continuation = None
            if (lookup[1] and lookup[1].session
                    and (not self.session.active or self._selection() != lookup[2])):
                return GlobalReply("That request ended. Nothing sent.")
            return lookup[1] or GlobalReply("That request ended. Nothing sent.")
        continuation = self._continuation
        if self._closed or not continuation or continuation[0] != token:
            return GlobalReply("That request ended. Nothing sent.")
        if self._busy:
            return None
        self._continuation = None
        target = continuation[1]
        if (target and self.session.target_id == target and self.session.active
                and self.session.phase == "review"
                and self.session.target_revision == continuation[2]
                and self.session.draft == continuation[3]):
            return GlobalReply(self._review_message(), True, review=True)
        return GlobalReply(self._notice if not self.session.active else "That request ended. Nothing sent.")

    def _launch(self, kind, operation):
        with self._lock:
            if self._closed or self._busy:
                return False
            self._busy = True
            self._pending_kind = kind

        def run():
            started = time.perf_counter()
            def elapsed_ms():
                return max(0, int((time.perf_counter() - started) * 1000))
            try:
                value = operation()
                log.info("Codex operation completed: kind=%s elapsed_ms=%d", kind, elapsed_ms())
                result = (kind, value, None)
            except DeliveryUnknown as exc:
                log.warning("Codex operation failed: kind=%s elapsed_ms=%d code=transport",
                            kind, elapsed_ms())
                result = (kind, None, self._failure_message(exc))
            except CodexError as exc:
                log.warning("Codex operation failed: kind=%s elapsed_ms=%d code=%s",
                            kind, elapsed_ms(), self._safe_error_code(exc))
                result = (kind, None, self._failure_message(exc))
            except Exception:
                # Do not echo raw RPC/auth/error payloads into speech or logs.
                log.warning("Codex operation failed: kind=%s elapsed_ms=%d code=other",
                            kind, elapsed_ms())
                result = (kind, None, "Codex couldn't complete that request. Check its sign-in or model availability.")
            if not self._closed:
                self._results.put_nowait(result)

        threading.Thread(target=run, name="memoir-codex-job", daemon=True).start()
        return True

    def poll(self):
        """Bounded local queue drains only; never waits on Codex or plays feedback."""
        if self._closed:
            return
        if self._unsent_request and not self._valid_unsent(self._unsent_request):
            self._unsent_request = None
        plan = self._pending_plan
        if plan and (self.session.clock() >= min(plan.expires_at + self.session.feedback_elapsed,
                                                 self.session.hard_deadline or plan.expires_at)
                     or self.session.target_id != plan.target_id
                     or self.session.target_revision != plan.revision):
            self.end_session()
        for _ in range(8):
            try:
                kind, result, error = self._results.get_nowait()
            except queue.Empty:
                break
            self._busy = False
            self._pending_kind = ""
            self._review_notice = False
            if error:
                self._pending_draft = None
                self._notice = error
                self._cancel_pending = False
                if kind == "resolve_named" and self._pending_lookup:
                    pending, self._pending_lookup = self._pending_lookup, None
                    self._lookup_continuation = (pending.token, GlobalReply(error, status="failed"), None)
                self._publish_notice()
                log.info("Codex operation ended without success: kind=%s", kind)
                continue
            log.info("Codex operation completed: kind=%s", kind)
            if kind == "resolve_named":
                pending, self._pending_lookup = self._pending_lookup, None
                if (not pending or self.session.clock() >= pending.expires_at
                        or self._selection() != pending.selection):
                    self._notice = "That chat lookup expired or changed. Nothing sent."
                    self._publish_notice()
                    continue
                requested_name = self._thread_name_key(pending.command.target_name)
                matches = [row for row in result
                           if self._thread_name_key(row.get("name", "")) == requested_name]
                if len(matches) == 1:
                    row = matches[0]
                    self.session.select(row["id"], row.get("name") or "selected chat")
                    command = replace(pending.command, target_kind="selected", target_name="")
                    reply = self._stage_resolved(command, pending.selection)
                    self._lookup_continuation = (pending.token, reply, self._selection())
                    self._notice = "Chat found. Request awaiting confirmation." if reply.session else reply.message
                else:
                    self._unresolved_request = pending.command
                    if result:
                        choice_rows = matches[:4] if matches else result[:4]
                        choice = _PendingChoice(pending.command, tuple(choice_rows), self._selection(), pending.expires_at)
                        self._pending_choice = choice
                        message = self._choice_message(choice)
                    else:
                        message = "No project chats found. Select or create a chat, then repeat your request. Nothing sent."
                    self._lookup_continuation = (pending.token, GlobalReply(message), None)
                    self._notice = message
                continue
            elif kind == "confirmed_create":
                self._create_cancel = None
                created = result.get("created")
                selection_unchanged = self._selection() == result["selection"]
                if created:
                    self._remember_created(created, result)
                    if result.get("submitted") or result.get("submission_uncertain"):
                        self._owned_jobs.add(created["id"])
                    if result.get("submitted"):
                        self._current_job = (created["id"], result.get("turn_id"))
                    if selection_unchanged:
                        self.session.select(created["id"], created.get("name") or "Memoir voice task")
                if result.get("error"):
                    prefix = "Chat created. " if created else ""
                    self._notice = prefix + result["error"]
                elif result.get("cancelled"):
                    self._notice = "Chat created; prompt cancelled before sending."
                elif result.get("submitted"):
                    self._notice = "Chat created. Codex accepted the confirmed prompt."
                    self._job_started = time.monotonic()
                else:
                    self._notice = "Created the confirmed chat. No prompt sent."
                if created and created.get("warning") and not result.get("error"):
                    self._notice += " " + created["warning"]
                if created:
                    self._notice += " Model: " + (created.get("model") or self._default_model) + "."
                    if (result.get("show_chat") and selection_unchanged and not self._cancel_pending
                            and result.get("_navigation_epoch") == self._navigation_epoch
                            and not result.get("cancelled")):
                        self._notice += " " + self._request_open(created["id"])
            elif kind == "create":
                self._remember_created(result)
                token = result.get("_draft_token")
                if token:
                    pending, self._pending_draft = self._pending_draft, None
                    if (not pending or pending[0] != token or time.monotonic() >= pending[2]
                            or self.session.target_revision != pending[3]):
                        self._notice = "Chat creation finished, but the held prompt expired or was cancelled. Nothing sent."
                        continue
                    self.session.select(result["id"], result.get("name") or "Memoir voice task")
                    self.session.begin(pending[1])
                    if self._continuation and self._continuation[0] == token:
                        self._continuation = (token, self.session.target_id,
                                              self.session.target_revision, self.session.draft)
                    self._notice = self._review_message() if self.session.active else "Prompt could not be staged. Nothing sent."
                    self._review_notice = self.session.active
                    continue
                self.session.select(result["id"], result.get("name") or "Memoir voice task")
                self._notice = "Created and selected " + self.session.target_name + ". Say dictate a prompt."
                if result.get("warning"):
                    self._notice += " " + result["warning"]
            elif kind == "list":
                self._threads = result
                names = [row.get("name") or row.get("preview", "Untitled")[:60] for row in result[:6]]
                self._notice = "Codex chats: " + "; ".join(names) if names else "No chats found for this project."
            elif kind == "select":
                self.session.select(result["id"], result.get("name") or "selected chat")
                self._notice = "Selected " + self.session.target_name + "."
            elif kind == "open":
                if (result.get("_navigation_generation") != self._request_generation
                        or result.get("_navigation_epoch") != self._navigation_epoch
                        or result.get("_navigation_selection") != self._selection()
                        or self.session.clock() >= result.get("_navigation_expires", 0)):
                    self._notice = "That opening request expired or changed. Nothing opened."
                    self._publish_notice()
                    continue
                self.session.select(result["id"], result.get("name") or "selected chat")
                self._notice = self._request_open(result["id"])
            elif kind == "submit":
                if result.get("status") == "unsent_model_permission":
                    self._cancel_pending = False
                    request = result.get("request")
                    if self._valid_unsent(request):
                        self._unsent_request = request
                        self._notice = "That chat's model needs your permission. Your exact task is held. Say use Luna for this Codex chat to review it again. Nothing sent."
                    else:
                        self._notice = "That chat's model needs permission, but the held request expired or changed. Nothing sent."
                    self._publish_notice()
                    continue
                self._notice = "Prompt queued for the desktop chat." if result.get("status") == "queued" else "Codex accepted the prompt."
                if result.get("status") != "queued":
                    self._job_started = time.monotonic()
                    if result.get("_target_id"):
                        self._owned_jobs.add(result["_target_id"])
                        self._current_job = (result["_target_id"], result.get("id"))
                if (result.get("_show_chat") and result.get("_target_id")
                        and result.get("_navigation_epoch") == self._navigation_epoch
                        and self._selection() == result.get("_show_selection") and not self._cancel_pending):
                    self._notice += " " + self._request_open(result["_target_id"])
            elif kind == "cancel":
                self._notice = "Cancellation requested." if result else "No locally owned Codex job is running."
                self._job_started = None
            elif kind == "approve":
                self._notice = "Approval answered." if result else "That approval expired. Nothing approved."
            if self._cancel_pending and kind in {"submit", "confirmed_create"}:
                self._cancel_pending = False
                if (kind == "confirmed_create" and not result.get("submitted")
                        and not result.get("submission_uncertain") and self.bridge.active_turn is None):
                    pass
                elif result.get("status") == "queued":
                    self._notice = "The prompt was already queued to the desktop chat. I can't retract it."
                else:
                    self._launch("cancel", self.bridge.cancel_turn)
            self._publish_notice()
        for event in self.bridge.drain_events():
            kind = event.get("kind") or event.get("type")
            target = event.get("thread_id") or event.get("threadId")
            turn_id = event.get("turn_id") or event.get("turnId")
            if (kind == "error" and not target and self._current_job
                    and isinstance(getattr(self.bridge, "_fatal", None), CodexError)):
                self._notice = "Codex connection failed. Job status is unknown. Check the chat; nothing will be resent."
                self._publish_notice()
                self._job_started = None
                continue
            if (kind in {"completed", "error", "approval"} and self._current_job
                    and target == self._current_job[0] and self._current_job[1]
                    and turn_id == self._current_job[1]):
                self._review_notice = False
                self._notice = {
                    "completed": "Codex finished.", "error": "Codex reported an error.",
                    "approval": "Codex needs approval. Open its chat to review it.",
                }[kind]
                if kind == "error" and event.get("code") == "quota":
                    self._notice = "Codex reached its usage limit. I won't retry or upgrade models."
                self._publish_notice()
                if kind in {"completed", "error"}:
                    self._job_started = None
                    self._current_job = None
        if self._job_started is not None and time.monotonic() - self._job_started >= self._job_limit:
            if self._launch("cancel", self.bridge.cancel_turn):
                self._job_started = None
                self._notice = "Codex reached its time budget. Cancellation requested."
                self._publish_notice()

    def _find_thread(self, name):
        rows = self.bridge.list_threads()
        rows = [row for row in rows if row.get("cwd") and Path(row["cwd"]).resolve() == self.root]
        if name.lower() in {"last", "latest"}:
            matches = rows[:1]
        else:
            matches = [row for row in rows
                       if self._thread_name_key(row.get("name", "")) == self._thread_name_key(name)]
        if len(matches) != 1:
            raise ValueError("Chat name must identify exactly one chat.")
        return matches[0]

    def handle(self, text):
        requested_target = self.session.target_id
        if self._closed:
            return GlobalReply("Codex voice tools are closed. Nothing sent.") if routing_candidate(text) else None
        if isinstance(text, str) and not text.strip() and self.session.active:
            return GlobalReply(self.review_message(), True, review=True)
        if self._pending_choice and isinstance(text, str):
            return self._handle_choice(text)
        control = _control_text(text) if isinstance(text, str) else ""
        if (self._valid_unsent(self._unsent_request)
                and control in {"cancel", "cancel that", "start again", "startagain", "cancel codex", "cancel codex task", "cancel codex job"}):
            self._new_intent()
            return GlobalReply("Held request cancelled. Nothing sent.")
        controls = {"send it", "send the prompt", "submit prompt", "sendit", "sendtheprompt", "submitprompt",
                    "read it back", "read back", "readitback", "cancel", "cancel that", "stop", "stop that",
                    "start again", "startagain", "edit prompt", "editprompt", "status", "codex status"}
        if self.session.active and control in controls:
            if control in {"status", "codex status"}:
                return GlobalReply(self.review_message(), True, review=True)
            reply = self.consume(text)
            if reply.action == "submit":
                return GlobalReply(self.submit_confirmed(reply))
            if reply.action == "readback":
                return GlobalReply(self.review_message(full=True), self.session.active, review=True)
            if reply.action == "review":
                return GlobalReply(self.review_message(), self.session.active, review=True)
            return GlobalReply(reply.message, self.session.active)
        command = parse_global(text)
        if command is None:
            if not routing_candidate(text):
                return None
            if self._router is None:
                return GlobalReply("I couldn't safely understand that Codex request. Nothing sent. Try create a Codex chat, then dictate a prompt.")
            try:
                command = self._router.route(text)
            except Exception:
                return GlobalReply("I couldn't safely understand that Codex request. Nothing sent.")
            if command is None:
                return None
            if command.action == "refused":
                return GlobalReply("I couldn't safely understand that Codex request. Nothing sent. Try create a Codex chat, then dictate a prompt.")
            if command.inferred and (command.action not in {"create", "create_ask", "ask", "dictate", "status", "list", "open"}
                                     or command.text != text or command.title or command.direction):
                return GlobalReply("I couldn't safely understand that Codex request. Nothing sent.")
            if command.inferred:
                try:
                    validate_classification(text, {"action": command.action, "direct_request": True,
                                                   "target_kind": command.target_kind, "target_name": command.target_name})
                except ValueError:
                    return GlobalReply("I couldn't safely understand that Codex request. Nothing sent.")
        if command.action in {"create", "create_ask", "ask"}:
            try:
                requested_model = creation_model(text)
            except ValueError:
                return GlobalReply("The model constraint needs clarification. No chat created or prompt sent.")
            if requested_model and requested_model not in PERMITTED_MODELS:
                return GlobalReply("That requested model is not a permitted Luna model. No chat created or prompt sent.")
            if requested_model and command.action == "ask":
                return GlobalReply("Select the chat and explicitly permit its model first. Your model request was not silently changed. Nothing sent.")
            command = replace(command, model=requested_model,
                              show_chat=command.action in {"create", "create_ask"}
                              or bool(re.match(r"^(?:(?:please|can you|could you|would you)\s+)?(?:open|show|display|bring up)\b", text.strip(), re.I)))
            if command.action == "create" and not command.inferred:
                command = replace(command, title=command.title or command.text,
                                  text=text, inferred=True, target_kind="new")
        if command.action == "open" and command.inferred:
            command = replace(command, inferred=False,
                              text=command.target_name if command.target_kind == "named" else
                                   "last_created" if command.target_kind == "last_created" else "current")
        self.poll()
        action = command.action
        if action not in {"status", "use_luna"}:
            self._new_intent()
        if self._pending_lookup and action not in {"status", "list"}:
            self.end_session()
        if action == "status":
            if not self._busy and self.session.active and self.session.phase == "review":
                return GlobalReply(self._review_message(), True, review=True)
            if self._review_notice and not self.session.active and not self._busy:
                self._notice = "No prompt awaiting confirmation. Nothing sent."
                self._review_notice = False
            return GlobalReply("Checking Codex." if self._busy else self._notice)
        if action == "use_luna":
            held = self._unsent_request if self._valid_unsent(self._unsent_request) else None
            temporary_plan = self._pending_plan and self._pending_plan.kind != "existing-submit"
            self.end_session()
            if (temporary_plan or requested_target.startswith(_PLAN_TARGET_PREFIX)
                    or self.session.target_id != requested_target):
                return GlobalReply("Select the actual Codex chat, then repeat the Luna permission command. No permission changed.")
            if not self.session.target_id or self.session.target_id.startswith(_PLAN_TARGET_PREFIX):
                return GlobalReply("Select a Codex chat first.")
            self.session.reset()
            self._pending_draft = None
            self._continuation = None
            self._luna_permission.add(self.session.target_id)
            self._new_intent()
            if held:
                return self._restage_unsent(held)
            return GlobalReply("Luna is permitted for future prompts in " + self.session.target_name + ". No prompt sent.")
        if action == "cancel":
            had_plan = self._pending_plan is not None
            self._pending_draft = None
            self._continuation = None
            self._review_notice = False
            self.end_session()
            self._approval_review = None
            if self._busy:
                # Only submissions may create a turn requiring subsequent cancel.
                if self._pending_kind in {"submit", "confirmed_create"}:
                    if self._pending_kind == "confirmed_create" and self._create_cancel is not None:
                        self._create_cancel.set()
                        self._cancel_pending = True
                        return GlobalReply("Cancellation requested. If the prompt is already sending, cancellation will follow it.")
                    self._cancel_pending = True
                    return GlobalReply("Cancellation will follow the pending submission; I won't submit another prompt.")
                return GlobalReply("That request doesn't start a coding job. No prompt will be sent.")
            if had_plan and self.bridge.active_turn is None:
                return GlobalReply("Request cancelled. Nothing sent.")
            return GlobalReply("Requesting cancellation." if self._launch("cancel", self.bridge.cancel_turn) else "Codex is busy.")
        if action == "review_approval":
            pending = self.bridge.pending_approval()
            self._approval_review = (pending, time.monotonic() + 30) if pending else None
            return GlobalReply(("Codex requests: " + pending["preview"] + ". Say approve Codex action or deny Codex action.")
                               if pending else "No supported Codex approval is pending.")
        if action in {"approve", "deny"}:
            reviewed, self._approval_review = self._approval_review, None
            if not reviewed or time.monotonic() >= reviewed[1]:
                return GlobalReply("Review the Codex approval first. Nothing approved.")
            pending = reviewed[0]
            operation = lambda: self.bridge.answer_approval(pending["request_id"], pending["thread_id"], pending["turn_id"], action == "approve")
            return GlobalReply("Answering the reviewed approval." if self._launch("approve", operation) else "Codex is busy. Nothing approved.")
        if action in {"select", "open"}:
            self.end_session()
            self._pending_draft = None
            self._continuation = None
            if (command.text.lower() in {"current", "this"}
                    and requested_target.startswith(_PLAN_TARGET_PREFIX)):
                return GlobalReply("That new chat has not been created yet. Explicitly choose an existing Codex chat before opening or selecting it.")
        if self._busy or self.bridge.active_turn is not None:
            if action not in {"open", "switch_tab"}:
                return GlobalReply("Codex is busy. Ask for status or cancel the job.")
        if not command.inferred and action in {"create", "create_ask", "ask", "dictate"}:
            self.end_session()
        if command.inferred and action == "dictate":
            return self._resolve_named(command) if command.target_kind == "named" else self._stage_dictation()
        if command.inferred and action in {"create", "create_ask", "ask"}:
            return self._resolve_named(command) if command.target_kind == "named" else self._stage_inferred(command)
        if action == "create":
            self._continuation = None
            title = command.title or command.text or "Memoir voice task"
            return GlobalReply("Creating a Luna chat. Ask for Codex status in a moment." if self._launch("create", lambda: self.bridge.create_thread(title)) else "Codex is busy.")
        if action == "create_ask":
            return self._create_with_draft(command.text, command.title, command.model)
        if action == "list":
            return GlobalReply("Checking the project chats. Ask for Codex status in a moment." if self._launch("list", lambda: [row for row in self.bridge.list_threads() if row.get("cwd") and Path(row["cwd"]).resolve() == self.root]) else "Codex is busy.")
        if action in {"select", "open"}:
            if self.session.target_id.startswith(_PLAN_TARGET_PREFIX):
                return GlobalReply("That temporary target is no longer valid. Nothing opened.")
            if command.text.lower() in {"current", "this"} and self.session.target_id:
                if action == "open":
                    return GlobalReply(self._request_open(self.session.target_id))
                return GlobalReply("Selected " + self.session.target_name + ".")
            # Selecting a chat is a data operation; opening it remains an explicit
            # subsequent command, never an unsolicited worker UI mutation.
            if command.text.lower() == "last_created":
                if not self._created_receipt:
                    return GlobalReply("No saved created-chat identity is available. Nothing opened.")
                operation = self._read_created
            else:
                operation = lambda: self._find_thread(command.text)
            if action == "open":
                lookup = operation
                generation, selection = self._request_generation, self._selection()
                navigation_epoch = self._navigation_epoch
                expires = self.session.clock() + 90
                operation = lambda: dict(lookup(), _navigation_generation=generation,
                                         _navigation_epoch=navigation_epoch,
                                         _navigation_selection=selection, _navigation_expires=expires)
            return GlobalReply("Looking for that chat." if self._launch(action, operation) else "Codex is busy.")
        if action == "switch_tab":
            return GlobalReply(self._tabs(command.direction))
        if action in {"dictate", "ask"}:
            if self.session.target_id.startswith(_PLAN_TARGET_PREFIX):
                return GlobalReply("That temporary target is no longer valid. Nothing sent.")
            if action == "ask":
                return self._stage_inferred(command)
            self._continuation = None
            reply = self.session.begin(command.text)
            message = self._review_message() if command.text and self.session.active else reply.message
            if not command.text and self.session.active:
                message += " Speak in segments, then say finish dictation. Nothing is sent until you confirm."
            return GlobalReply(message, self.session.active, review=bool(command.text and self.session.active))
        return GlobalReply("That global action isn't available yet.")

    def submit_confirmed(self, reply):
        from .global_commands import SessionReply
        if (not isinstance(reply, SessionReply)
                or any(not isinstance(value, str) for value in
                       (reply.target_id, reply.prompt, reply.confirmation_id, reply.plan_id))):
            return "The prompt confirmation expired or changed. Nothing sent."
        if self._closed:
            self.end_session()
            return "Codex voice tools are closed. Nothing sent."
        if self._busy or self.bridge.active_turn is not None:
            return "Codex is already running. Nothing sent."
        plan = self._pending_plan
        confirmation_expiry = self.session.confirmation_expires_at
        if plan and (reply.plan_id != plan.id or reply.prompt != plan.final_prompt
                     or reply.target_id != plan.target_id or self.session.target_id != plan.target_id
                     or self.session.target_revision != plan.revision
                     or self.session.clock() >= plan.expires_at):
            self.end_session()
            return "The prompt confirmation expired or changed. Nothing sent."
        if ((reply.plan_id and not plan)
                or (reply.target_id.startswith(_PLAN_TARGET_PREFIX) and not plan)
                or reply.action != "submit" or not reply.prompt
                or reply.target_id != self.session.target_id or not self.session.take_confirmation(reply)):
            if plan:
                self.end_session()
            return "The prompt confirmation expired or changed. Nothing sent."
        confirmed_identity = (reply.plan_id, reply.target_id, self.session.target_revision, reply.prompt)
        if self._review_armed != confirmed_identity:
            self.end_session()
            return "The task has not been read back successfully. Nothing sent. Ask again to review it."
        thread_id, prompt = reply.target_id, reply.prompt
        recovery = _UnsentRequest(plan.source if plan else prompt, prompt, self._selection(),
                                  self._request_generation, confirmation_expiry,
                                  self.session.hard_deadline or self.session.clock(),
                                  plan.model if plan else self._default_model, bool(plan and plan.show_chat))
        self._review_notice = False
        navigation_epoch = self._navigation_epoch
        if plan and plan.kind != "existing-submit":
            self.end_session()
            selection = self._selection()
            cancel_event = threading.Event()
            self._create_cancel = cancel_event
            def create_confirmed():
                try:
                    if plan.model and plan.model != self._default_model:
                        created = dict(self.bridge.create_thread(plan.title, model=plan.model))
                    else:
                        created = dict(self.bridge.create_thread(plan.title))
                except CodexError as exc:
                    if not isinstance(exc, DeliveryUnknown):
                        return {"selection": selection, "created": None,
                                "error": self._failure_message(exc) + " No chat created."}
                    return {"selection": selection, "created": None,
                            "error": "Chat creation could not be confirmed. Check Codex before requesting another chat. Nothing was retried."}
                except Exception:
                    return {"selection": selection, "created": None,
                            "error": "Chat creation could not be confirmed. Check Codex before requesting another chat. Nothing was retried."}
                result = {"selection": selection, "created": created, "submitted": False,
                          "_navigation_epoch": navigation_epoch,
                          "show_chat": plan.show_chat}
                if plan.kind == "create+submit":
                    if created.get("can_submit") is False:
                        result["error"] = "The requested name could not be confirmed. No prompt sent. Check the created chat before retrying."
                        return result
                    if cancel_event.is_set():
                        result["cancelled"] = True
                        return result
                    try:
                        turn = self.bridge.start_turn(created["id"], plan.final_prompt)
                        result["turn_id"] = turn.get("id")
                        result["submitted"] = True
                    except Exception as exc:
                        result["error"] = self._failure_message(exc)
                        result["submission_uncertain"] = isinstance(exc, DeliveryUnknown)
                return result
            accepted = self._launch("confirmed_create", create_confirmed)
            if not accepted:
                self._create_cancel = None
            return (CREATE_ACCEPTED if plan.kind == "create-only" else SEND_ACCEPTED) if accepted else "Codex is busy. Nothing sent."
        if plan:
            self._pending_plan = None
        self._review_armed = None
        self._session_restore = None
        def submit():
            model_options = {"model": plan.model} if plan and plan.model and plan.model != self._default_model else {}
            if thread_id in self.bridge.owned_threads:
                result = self.bridge.start_turn(thread_id, prompt)
            elif thread_id in self._luna_permission:
                try:
                    result = self.bridge.queue_existing(thread_id, prompt, allow_model_change=True, **model_options)
                except CodexError as exc:
                    if exc.code == "model_permission":
                        return {"status": "unsent_model_permission", "request": recovery}
                    raise
            else:
                try:
                    result = self.bridge.queue_existing(thread_id, prompt, **model_options)
                except CodexError as exc:
                    if exc.code == "model_permission":
                        return {"status": "unsent_model_permission", "request": recovery}
                    raise
            return dict(result, _target_id=thread_id, _show_chat=bool(plan and plan.show_chat),
                        _navigation_epoch=navigation_epoch,
                        _show_selection=recovery.selection)
        return SEND_ACCEPTED if self._launch("submit", submit) else "Codex is busy. Nothing sent."

    def close(self):
        self._closed = True
        self._new_intent()
        if self._create_cancel is not None:
            self._create_cancel.set()
        self._pending_draft = None
        self._continuation = None
        self.end_session()
        self.bridge.close()
