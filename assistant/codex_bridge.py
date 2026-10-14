"""A deliberately small, subscription-only bridge to Codex app-server.

The bridge owns only processes that it starts itself.  It never resumes or
interrupts a desktop turn and it does not retry a prompt after an uncertain
delivery result.
"""

from __future__ import annotations

import json
import itertools
import logging
import math
import os
import queue
import re
import shutil
import subprocess
import threading
import tempfile
import time
from collections import deque
from pathlib import Path
from typing import Any


class CodexError(RuntimeError):
    """The app-server or local transport could not complete an operation."""

    def __init__(self, message: str, *, code: str = "protocol", retry_at: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.retry_at = retry_at


class DeliveryUnknown(CodexError):
    """A CLI queue operation timed out after its delivery was indeterminate."""

    def __init__(self, message: str = "Delivery is unknown") -> None:
        super().__init__(message, code="transport")


class _TransportUnknown(CodexError):
    """Transport failure after a request may have reached the server."""


_UUID = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
_THR = re.compile(r"^thr_[A-Za-z0-9]+$")
_DEFAULT_MODEL = "gpt-5.6-luna"
PERMITTED_MODELS = frozenset({"gpt-5.6-luna", "gpt-6-luna"})
log = logging.getLogger("memoir.codex")
_MAX_EVENTS = 256
_READINESS_RETRY_COOLDOWN = 5.0
_QUOTA_RECHECK_COOLDOWN = 30.0
_VERSION_BUDGET = 4.5
_CODEX_VERSION = re.compile(
    r"codex-cli (0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)"
    r"(?:-([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?"
    r"(?:\+([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?(?:\r?\n)?"
)


def _parse_codex_version(output: str) -> tuple | None:
    """Parse one bounded SemVer version line and return its precedence key."""
    if not isinstance(output, str) or len(output) > 256:
        return None
    match = _CODEX_VERSION.fullmatch(output)
    if match is None:
        return None
    major, minor, patch = (int(value) for value in match.group(1, 2, 3))
    prerelease = match.group(4)
    prerelease_key = []
    if prerelease is not None:
        for identifier in prerelease.split("."):
            if identifier.isdigit():
                if len(identifier) > 1 and identifier.startswith("0"):
                    return None
                prerelease_key.append((0, int(identifier)))
            else:
                prerelease_key.append((1, identifier))
    # Build metadata is validated by the regex but intentionally excluded from
    # SemVer precedence. Stable releases sort above prereleases of the same core.
    return major, minor, patch, int(prerelease is None), tuple(prerelease_key)


def _resolve_codex_executable(*, platform=None, local_app_data=None, which=None, run=None, clock=None) -> str | None:
    """Lazy, read-only selection among bounded known installed binaries.

    PATH can point at the older standalone CLI. It cannot introduce a random
    cwd executable into this Windows candidate pool. Nothing is installed,
    downloaded, persisted, or launched as an app-server during selection.
    """
    which, run, clock = which or shutil.which, run or subprocess.run, clock or time.monotonic
    path_candidate = which("codex")
    if (platform or os.name) != "nt":
        return path_candidate
    local = local_app_data if local_app_data is not None else os.environ.get("LOCALAPPDATA")
    if not local:
        return None
    started = clock()
    try:
        # LOCALAPPDATA is the trust anchor. Known suffixes must remain under
        # this canonical anchor, not redirect through an ancestor junction.
        local = Path(local).resolve(strict=True)
    except (OSError, ValueError):
        return None
    roots = ((local / "OpenAI" / "Codex" / "bin", 1),
             (local / "Programs" / "OpenAI" / "Codex" / "bin", 0))
    candidates = {}
    for root, preference in roots:
        try:
            canonical_root = root.resolve(strict=True)
        except (OSError, ValueError):
            continue
        if canonical_root != root or local not in canonical_root.parents:
            continue
        paths = [root / "codex.exe"]
        if preference:
            try:
                children = [child for child in itertools.islice(root.iterdir(), 64)
                            if re.fullmatch(r"[0-9a-fA-F]{16,64}", child.name)
                            and child.is_dir() and not child.is_symlink()]
                children.sort(key=lambda child: child.stat().st_mtime, reverse=True)
                paths += [child / "codex.exe" for child in children[:6]]
            except OSError:
                pass
        if path_candidate:
            paths.append(Path(path_candidate))
        for path in paths:
            try:
                resolved = path.resolve(strict=True)
                if (path.is_symlink() or not path.is_file() or canonical_root not in resolved.parents
                        or resolved != path.absolute()):
                    continue
                candidates[str(resolved)] = preference
            except OSError:
                continue
    verified = []
    # Prefer probing current desktop installs within the total bounded budget.
    ordered = sorted(candidates.items(), key=lambda row: row[1], reverse=True)[:8]
    for executable, preference in ordered:
        remaining = _VERSION_BUDGET - (clock() - started)
        if remaining <= 0:
            break
        try:
            result = run([executable, "--version"], stdin=subprocess.DEVNULL,
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                         timeout=min(0.6, remaining), check=False,
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            version = _parse_codex_version(result.stdout)
            if result.returncode == 0 and version is not None:
                verified.append((version, preference, executable))
        except (OSError, ValueError, subprocess.TimeoutExpired):
            continue
    return max(verified)[2] if verified else None


def _safe_id(value: Any) -> bool:
    return isinstance(value, str) and bool(_UUID.fullmatch(value) or _THR.fullmatch(value))


def _text_from_item(item: Any) -> str | None:
    """Extract agent text without assuming one server item representation."""
    if not isinstance(item, dict):
        return None
    typ = item.get("type") or item.get("itemType")
    if typ not in (None, "agentMessage", "agent_message", "message"):
        return None
    for key in ("text", "message", "content"):
        value = item.get(key)
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            parts: list[str] = []
            for part in value:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict) and isinstance(part.get("text"), str):
                    parts.append(part["text"])
            if parts:
                return "".join(parts)
    return None


class _Pending:
    def __init__(self) -> None:
        self.event = threading.Event()
        self.response: dict[str, Any] | None = None
        self.error: BaseException | None = None


class CodexBridge:
    """Bounded client for the signed-in local Codex app-server."""

    def __init__(
        self,
        project_root: Path,
        model: str = _DEFAULT_MODEL,
        timeout: float = 8.0,
        executable: str | None = None,
    ) -> None:
        root = Path(project_root).resolve(strict=True)
        if not root.is_dir():
            raise CodexError(f"project_root is not a directory: {root}")
        if model not in PERMITTED_MODELS:
            raise CodexError("CodexBridge only permits supported Luna models", code="model")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.project_root = root
        self.requested_model = model
        self.timeout = float(timeout)
        # Discovery and version processes belong to first-command readiness,
        # not application construction or the audio owner's startup path.
        self.executable = executable or None
        self.owned_threads: set[str] = set()
        self._thread_models: dict[str, str] = {}
        self._available_models: dict[str, dict] = {}
        self._receipt_lock = threading.RLock()
        self._receipt_path = self.project_root / "assistant" / "logs" / ".codex-last-created.json"
        self._last_created = self._load_receipt()

        self._proc: subprocess.Popen[str] | None = None
        self._reader: threading.Thread | None = None
        self._write_lock = threading.Lock()
        self._startup_lock = threading.Lock()
        self._state_lock = threading.RLock()
        self._pending_lock = threading.Lock()
        self._pending: dict[int, _Pending] = {}
        self._next_id = 0
        self._ready = False
        self._closed = False
        self._fatal: CodexError | None = None
        self._readiness_error: CodexError | None = None
        self._readiness_retry_at = 0.0
        self._selected_model: str | None = None
        self._model_effort: str | None = None
        self._events: deque[dict[str, Any]] = deque(maxlen=_MAX_EVENTS)
        self._known_threads: dict[str, dict[str, Any]] = {}
        self._active: tuple[str, str] | None = None
        self._turn_start_pending = False
        self._mutation_pending = 0
        self._cancel_pending = False
        self._completed_turns: deque[str] = deque(maxlen=_MAX_EVENTS)
        self._completed_turn_set: set[str] = set()
        self._turn_text: dict[str, str] = {}
        self._turn_text_order: deque[str] = deque(maxlen=_MAX_EVENTS)
        self._mutation_uncertain = False
        self._quota_blocked = False
        self._quota_recheck_at = 0.0
        self._quota_check_lock = threading.Lock()
        self._pending_approval: dict[str, Any] | None = None

    @property
    def mutation_uncertain(self) -> bool:
        with self._state_lock:
            return self._mutation_uncertain

    @property
    def active_turn(self) -> tuple[str, str] | None:
        with self._state_lock:
            return self._active

    def _fail(self, message: str) -> CodexError:
        return CodexError(message)

    def _spawn(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        if not self.executable:
            self.executable = _resolve_codex_executable()
        if not self.executable:
            raise CodexError("A verified installed Codex executable was not found", code="transport")
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            proc = subprocess.Popen(
                [self.executable, "app-server", "--stdio"],
                cwd=str(self.project_root),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=flags,
                bufsize=1,
            )
        except OSError as exc:
            raise CodexError(f"Could not start Codex app-server: {exc}") from exc
        self._proc = proc
        self._reader = threading.Thread(target=self._read_loop, args=(proc,), daemon=True)
        self._reader.start()

    def _read_loop(self, proc: subprocess.Popen[str]) -> None:
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                # Retiring a transport must also retire its responses and EOF.
                # Otherwise a late old reader can poison a fresh handshake.
                with self._state_lock:
                    if proc is not self._proc:
                        return
                try:
                    message = json.loads(line)
                except (json.JSONDecodeError, TypeError) as exc:
                    self._connection_failed(CodexError(f"Malformed app-server response: {exc}"), proc=proc)
                    return
                if not isinstance(message, dict):
                    self._connection_failed(CodexError("Malformed app-server response"), proc=proc)
                    return
                if "id" in message and message.get("id") is not None and "method" in message:
                    self._handle_server_request(message, proc=proc)
                elif "id" in message and message.get("id") is not None:
                    self._dispatch_response(message, proc=proc)
                else:
                    self._dispatch_notification(message, proc=proc)
        except (OSError, ValueError) as exc:
            self._connection_failed(CodexError(f"Codex app-server read failed: {exc}", code="transport"), proc=proc)
        finally:
            self._connection_failed(CodexError("Codex app-server closed its output", code="transport"), proc=proc)

    def _connection_failed(self, error: CodexError, *, proc=None) -> None:
        with self._state_lock:
            if proc is not None and proc is not self._proc:
                return
            if self._fatal is None:
                self._fatal = error
                self._readiness_retry_at = time.monotonic() + _READINESS_RETRY_COOLDOWN
            self._ready = False
            if self._active is not None:
                self._mutation_uncertain = True
                # Keep the turn visible: its result is unknown and must not be
                # replayed by a future call.
                self._enqueue({"kind": "error", "message": str(error)})
            if self._turn_start_pending or self._mutation_pending:
                self._mutation_uncertain = True
            # Keep origin validation and fan-out atomic with retirement. An
            # old reader must never fail a replacement handshake's waiters.
            with self._pending_lock:
                for pending in self._pending.values():
                    pending.error = error
                    pending.event.set()

    def _dispatch_response(self, message: dict[str, Any], *, proc=None) -> None:
        with self._state_lock:
            if proc is not None and proc is not self._proc:
                return
            ident = message.get("id")
            with self._pending_lock:
                pending = self._pending.get(ident)
                if pending is None:
                    return
                pending.response = message
                pending.event.set()

    def _enqueue(self, event: dict[str, Any]) -> None:
        with self._state_lock:
            self._events.append(event)

    def _remember_completed(self, turn_id: str) -> None:
        if turn_id in self._completed_turn_set:
            return
        if len(self._completed_turns) >= _MAX_EVENTS:
            old = self._completed_turns.popleft()
            self._completed_turn_set.discard(old)
        self._completed_turns.append(turn_id)
        self._completed_turn_set.add(turn_id)

    def _dispatch_notification(self, message: dict[str, Any], *, proc=None) -> None:
        # No request wait or handshake occurs here. State changes cannot cross
        # transport retirement after the initial decoded-message check.
        with self._state_lock:
            if proc is not None and proc is not self._proc:
                return
            self._dispatch_current_notification(message)

    def _dispatch_current_notification(self, message: dict[str, Any]) -> None:
        method = message.get("method")
        params = message.get("params")
        if not isinstance(params, dict):
            params = {}
        if method in ("turn/started", "turn/created"):
            thread_id = params.get("threadId") or params.get("thread_id")
            turn = params.get("turn") if isinstance(params.get("turn"), dict) else params
            turn_id = (turn.get("id") or params.get("turnId") or params.get("turn_id")) if isinstance(turn, dict) else params.get("turnId")
            if _safe_id(thread_id) and _safe_id(turn_id):
                with self._state_lock:
                    if thread_id in self.owned_threads and turn_id not in self._completed_turn_set:
                        self._active = (thread_id, turn_id)
                        self._cancel_pending = False
            return
        if method in ("turn/completed", "turn/failed", "turn/cancelled", "turn/interrupted"):
            thread_id = params.get("threadId") or params.get("thread_id")
            turn = params.get("turn") if isinstance(params.get("turn"), dict) else params
            turn_id = (turn.get("id") or params.get("turnId") or params.get("turn_id")) if isinstance(turn, dict) else params.get("turnId")
            with self._state_lock:
                active = self._active
                owned = thread_id in self.owned_threads
                already_completed = _safe_id(turn_id) and turn_id in self._completed_turn_set
                owned_active = bool(
                    active and thread_id == active[0] and turn_id == active[1]
                )
                if owned and _safe_id(turn_id):
                    self._remember_completed(turn_id)
                if owned_active:
                    self._active = None
                    self._cancel_pending = False
                    self._decline_pending_approval()
            if owned and not already_completed and (active is None or owned_active):
                status = turn.get("status") if isinstance(turn, dict) else None
                code, retry_at = _classify_error(turn.get("error") if isinstance(turn, dict) else None)
                if code == "quota":
                    with self._state_lock:
                        self._quota_blocked = True
                successful = method == "turn/completed" and status not in {
                    "failed", "error", "cancelled", "interrupted"
                }
                self._enqueue({
                    "kind": "completed" if successful else "error",
                    "thread_id": thread_id,
                    "turn_id": turn_id,
                    "status": status,
                    "code": code if not successful else None,
                    "retry_at": retry_at,
                    "text": self._turn_text.pop(turn_id, None) if successful else None,
                    "message": None if successful else ("Codex reached its usage limit. I won't retry or upgrade models." if code == "quota" else "Codex turn failed"),
                })
            return
        if method == "item/completed":
            thread_id = params.get("threadId") or params.get("thread_id")
            item = params.get("item")
            text = _text_from_item(item)
            if thread_id in self.owned_threads and text is not None:
                turn_id = params.get("turnId") or params.get("turn_id")
                if _safe_id(turn_id):
                    if len(self._turn_text_order) >= _MAX_EVENTS:
                        old_text_turn = self._turn_text_order.popleft()
                        self._turn_text.pop(old_text_turn, None)
                    self._turn_text[turn_id] = text
                    self._turn_text_order.append(turn_id)
                self._enqueue({
                    "kind": "message",
                    "thread_id": thread_id,
                    "turn_id": turn_id,
                    "text": text,
                })
            return
        if method in ("error", "turn/error"):
            thread_id = params.get("threadId") or params.get("thread_id")
            turn_id = params.get("turnId") or params.get("turn_id")
            if thread_id in self.owned_threads or thread_id is None:
                code, retry_at = _classify_error(params.get("error") or params)
                with self._state_lock:
                    if code == "quota":
                        self._quota_blocked = True
                    if (self._active == (thread_id, turn_id) and params.get("willRetry") is False):
                        self._remember_completed(turn_id)
                        self._active = None
                        self._cancel_pending = False
                        self._decline_pending_approval()
                text = "Codex reached its usage limit. I won't retry or upgrade models." if code == "quota" else "Codex reported an error."
                self._enqueue({"kind": "error", "thread_id": thread_id, "turn_id": turn_id,
                               "code": code, "retry_at": retry_at, "message": text})
            return
        # Other notifications are intentionally ignored.  In particular, do
        # not expose raw server payloads, which may contain private metadata.

    def _handle_server_request(self, message: dict[str, Any], *, proc=None) -> None:
        with self._state_lock:
            if proc is not None and proc is not self._proc:
                return
        method = message.get("method")
        ident = message.get("id")
        params = message.get("params") if isinstance(message.get("params"), dict) else {}
        if method == "item/commandExecution/requestApproval":
            self._handle_command_approval(ident, params, proc=proc)
        elif method == "item/fileChange/requestApproval":
            self._decline_request(ident, params, "File changes require the Codex desktop.", proc=proc)
        else:
            self._send({
                "jsonrpc": "2.0",
                "id": ident,
                "error": {"code": -32601, "message": "Unsupported server request"},
            }, proc=proc)

    def _decline_request(self, ident: Any, params: dict[str, Any], message: str, *, proc=None) -> None:
        thread_id = params.get("threadId") or params.get("thread_id")
        with self._state_lock:
            if proc is not None and proc is not self._proc:
                return
            self._enqueue({
                "kind": "approval",
                "thread_id": thread_id if _safe_id(thread_id) else None,
                "message": message,
            })
        self._send({"jsonrpc": "2.0", "id": ident, "result": {"decision": "decline"}}, proc=proc)

    def _approval_preview(self, params: dict[str, Any]) -> str | None:
        value = params.get("command") or params.get("commandLine") or params.get("cmd")
        if isinstance(value, list):
            if not all(isinstance(part, str) for part in value):
                return None
            value = " ".join(value)
        if not isinstance(value, str) or not value or len(value) > 500:
            return None
        if any(ord(ch) < 32 and ch not in "\t" for ch in value):
            return None
        return value

    def _handle_command_approval(self, ident: Any, params: dict[str, Any], *, proc=None) -> None:
        thread_id = params.get("threadId") or params.get("thread_id")
        turn_id = params.get("turnId") or params.get("turn_id")
        preview = self._approval_preview(params)
        # Command approvals normally carry cwd, not the file-change-only
        # grantRoot field. Missing scope still fails closed.
        grant_root = params.get("grantRoot") or params.get("grant_root") or params.get("cwd")
        root_ok = grant_root is not None
        if grant_root is not None:
            try:
                grant_path = Path(grant_root).resolve(strict=False)
                root_ok = grant_path == self.project_root or self.project_root in grant_path.parents
            except (OSError, TypeError, ValueError):
                root_ok = False
        with self._state_lock:
            if proc is not None and proc is not self._proc:
                return
            active_ok = (
                _safe_id(thread_id)
                and _safe_id(turn_id)
                and self._active == (thread_id, turn_id)
                and self._pending_approval is None
            )
            if active_ok and preview is not None and root_ok:
                self._pending_approval = {
                    "request_id": ident,
                    "thread_id": thread_id,
                    "turn_id": turn_id,
                    "command": preview,
                    "grant_root": str(grant_root) if grant_root is not None else None,
                    "expires_at": time.monotonic() + 30.0,
                    "preview": preview,
                }
                pending = dict(self._pending_approval)
            else:
                pending = None
        if pending is None:
            self._decline_request(ident, params, "Approval details were missing or unsafe; use the Codex desktop.", proc=proc)
            return
        self._enqueue({
            "kind": "approval",
            "thread_id": thread_id,
            "turn_id": turn_id,
            "request_id": ident,
            "command": preview,
            "message": "Codex action awaits explicit approval.",
        })
        # Expire even if the user never asks for status/review. The reader
        # must remain free to receive turn completion and cancellation.
        timer = threading.Timer(30.0, self._expire_approval, args=(ident, thread_id, turn_id))
        timer.daemon = True
        timer.start()

    def _expire_approval(self, ident: Any, thread_id: str, turn_id: str) -> None:
        with self._state_lock:
            pending = self._pending_approval
            if (not pending or pending["request_id"] != ident or pending["thread_id"] != thread_id
                    or pending["turn_id"] != turn_id or pending["expires_at"] > time.monotonic()):
                return
            self._decline_pending_approval()

    def _decline_pending_approval(self) -> None:
        with self._state_lock:
            pending = self._pending_approval
            self._pending_approval = None
        if pending is not None:
            try:
                self._send({"jsonrpc": "2.0", "id": pending["request_id"], "result": {"decision": "decline"}})
            except CodexError:
                pass

    def pending_approval(self) -> dict[str, Any] | None:
        with self._state_lock:
            pending = dict(self._pending_approval) if self._pending_approval else None
        if pending and pending["expires_at"] <= time.monotonic():
            self._decline_pending_approval()
            return None
        if pending:
            pending["expires_in"] = max(0.0, pending["expires_at"] - time.monotonic())
        return pending

    def answer_approval(self, request_id: Any, thread_id: str, turn_id: str, approved: bool) -> bool:
        if not isinstance(approved, bool) or not _safe_id(thread_id) or not _safe_id(turn_id):
            return False
        with self._state_lock:
            pending = self._pending_approval
            if not pending:
                return False
            if pending["expires_at"] <= time.monotonic():
                expired = pending is not None
                self._pending_approval = None
            else:
                expired = False
                if (
                    pending["request_id"] != request_id
                    or pending["thread_id"] != thread_id
                    or pending["turn_id"] != turn_id
                    or self._active != (thread_id, turn_id)
                ):
                    return False
                self._pending_approval = None
        if expired:
            if pending is not None:
                try:
                    self._send({"jsonrpc": "2.0", "id": pending["request_id"], "result": {"decision": "decline"}})
                except CodexError:
                    pass
            return False
        try:
            self._send({
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"decision": "accept" if approved else "decline"},
            })
        except CodexError:
            return False
        return True

    def _send(self, message: dict[str, Any], *, proc=None) -> None:
        # Server-request replies stay bound to the process that asked, even if
        # retirement races the write. They can never answer a newer server ID.
        proc = self._proc if proc is None else proc
        if proc is None or proc.stdin is None:
            raise CodexError("Codex app-server connection is closed")
        try:
            raw = json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n"
            with self._write_lock:
                proc.stdin.write(raw)
                proc.stdin.flush()
        except (OSError, BrokenPipeError, ValueError) as exc:
            self._connection_failed(CodexError(f"Codex app-server write failed: {exc}", code="transport"), proc=proc)
            raise CodexError("Codex app-server write failed") from exc

    def _request(
        self,
        method: str,
        params: dict[str, Any],
        *,
        timeout: float | None = None,
        mutation: bool = False,
    ) -> dict[str, Any]:
        if mutation:
            with self._state_lock:
                self._mutation_pending += 1
        try:
            return self._request_once(method, params, timeout=timeout, mutation=mutation)
        finally:
            if mutation:
                with self._state_lock:
                    self._mutation_pending -= 1

    def _request_once(
        self,
        method: str,
        params: dict[str, Any],
        *,
        timeout: float | None = None,
        mutation: bool = False,
    ) -> dict[str, Any]:
        with self._state_lock:
            if self._closed:
                raise CodexError("CodexBridge is closed")
            if self._fatal is not None:
                raise self._fatal
        with self._pending_lock:
            self._next_id += 1
            ident = self._next_id
            pending = _Pending()
            self._pending[ident] = pending
        try:
            self._send({"jsonrpc": "2.0", "id": ident, "method": method, "params": params})
        except Exception as exc:
            with self._pending_lock:
                self._pending.pop(ident, None)
            if mutation:
                self._mark_mutation_uncertain()
                raise DeliveryUnknown(f"Delivery of {method} is unknown") from exc
            raise
        wait_for = self.timeout if timeout is None else timeout
        if not pending.event.wait(wait_for):
            with self._pending_lock:
                self._pending.pop(ident, None)
            if mutation:
                self._mark_mutation_uncertain()
                raise DeliveryUnknown(f"Delivery of {method} is unknown")
            raise CodexError(f"Timed out waiting for app-server method {method}")
        with self._pending_lock:
            self._pending.pop(ident, None)
        if pending.error is not None:
            if mutation:
                self._mark_mutation_uncertain()
                raise DeliveryUnknown(f"Delivery of {method} is unknown") from pending.error
            raise CodexError(str(pending.error)) from pending.error
        response = pending.response
        if not isinstance(response, dict):
            raise CodexError("Missing app-server response")
        if "error" in response:
            error = response.get("error")
            code, retry_at = _classify_error(error)
            if code == "quota":
                with self._state_lock:
                    self._quota_blocked = True
            raise CodexError(f"Codex app-server method {method} failed", code=code, retry_at=retry_at)
        return response.get("result") if isinstance(response.get("result"), dict) else {}

    def _mark_mutation_uncertain(self) -> None:
        with self._state_lock:
            self._mutation_uncertain = True
            self._ready = False

    def _guard_mutation(self) -> None:
        with self._state_lock:
            if self._mutation_uncertain:
                raise DeliveryUnknown("A previous mutation has unknown delivery; create a new bridge")
            blocked = self._quota_blocked
        if blocked:
            self._recheck_quota()
        with self._state_lock:
            if self._mutation_uncertain:
                raise DeliveryUnknown("A previous mutation has unknown delivery; create a new bridge")
            if self._quota_blocked:
                raise CodexError("Codex quota reset has not been verified; this bridge remains blocked", code="quota")

    def _recheck_quota(self) -> None:
        """A later explicit mutation may verify allowance, never retry a prompt."""
        with self._quota_check_lock:
            with self._state_lock:
                if (not self._quota_blocked or self._closed or self._mutation_uncertain
                        or time.monotonic() < self._quota_recheck_at):
                    return
                self._quota_recheck_at = time.monotonic() + _QUOTA_RECHECK_COOLDOWN
            try:
                self._check_account()
                result = self._request("account/rateLimits/read", {})
            except CodexError:
                # Unsupported endpoints and unknown/auth failures are not zero
                # usage. No credit reset, purchase, or mutation is attempted.
                return
            if _quota_snapshot_available(result):
                with self._state_lock:
                    if not self._mutation_uncertain and not self._closed:
                        self._quota_blocked = False

    def _check_account(self) -> None:
        result = self._request("account/read", {"refreshToken": False})
        account = result.get("account") if isinstance(result.get("account"), dict) else result
        if account.get("type") != "chatgpt":
            raise CodexError("Codex account is not a signed-in ChatGPT account", code="auth")

    def _model_entries(self, result: dict[str, Any]) -> list[dict[str, Any]]:
        values = result.get("data") or result.get("models") or []
        return [value for value in values if isinstance(value, dict)]

    def _supports_low(self, entry: dict[str, Any]) -> bool:
        values = (
            entry.get("reasoningEfforts")
            or entry.get("reasoning_efforts")
            or entry.get("supportedReasoningEfforts")
            or entry.get("supported_reasoning_efforts")
            or entry.get("supportedEfforts")
            or entry.get("supported_efforts")
        )
        if values is None and isinstance(entry.get("capabilities"), dict):
            values = entry["capabilities"].get("reasoningEfforts")
        if isinstance(values, (list, tuple, set)):
            return "low" in values or any(
                isinstance(item, dict) and item.get("reasoningEffort") == "low" for item in values
            )
        return False

    def _choose_model(self) -> str:
        entries: list[dict[str, Any]] = []
        cursor: Any = None
        while True:
            params: dict[str, Any] = {"limit": 100}
            if cursor:
                params["cursor"] = cursor
            result = self._request("model/list", params)
            entries.extend(self._model_entries(result))
            cursor = result.get("nextCursor") or result.get("next_cursor")
            if not cursor:
                break
        by_id = {entry.get("id") or entry.get("model"): entry for entry in entries}
        self._available_models = by_id
        selected = self.requested_model
        entry = by_id.get(selected)
        if not isinstance(entry, dict):
            raise CodexError(f"Configured Luna model is unavailable: {self.requested_model}", code="model")
        if not _safe_model_name(selected) or not self._supports_low(entry):
            raise CodexError(f"Model does not support low reasoning effort: {selected}", code="model")
        self._selected_model = selected
        self._model_effort = "low"
        return selected

    def _validated_model(self, model: str | None = None) -> str:
        chosen = model or self.requested_model
        entry = self._available_models.get(chosen)
        if chosen not in PERMITTED_MODELS or not isinstance(entry, dict) or not self._supports_low(entry):
            raise CodexError("The requested Luna model is unavailable; no fallback was selected", code="model")
        return chosen

    def _load_receipt(self) -> dict | None:
        try:
            path = self._receipt_path
            if path.is_symlink() or not path.is_file() or path.stat().st_size > 4096:
                return None
            if self.project_root not in path.resolve().parents:
                return None
            row = json.loads(path.read_text(encoding="utf-8"))
            if (not isinstance(row, dict) or row.get("schema") != 1
                    or row.get("project_root") != str(self.project_root)
                    or not _safe_id(row.get("id")) or row.get("model") not in PERMITTED_MODELS
                    or not isinstance(row.get("name"), str) or len(row["name"]) > 120
                    or type(row.get("submitted")) is not bool
                    or row.get("status") not in {"created", "submitted", "submission_unknown"}
                    or not isinstance(row.get("created_at"), (int, float))
                    or isinstance(row["created_at"], bool) or not math.isfinite(row["created_at"])
                    or (row.get("turn_id") is not None and not _safe_id(row["turn_id"]))):
                return None
            return row
        except (OSError, ValueError, TypeError):
            return None

    def last_created(self) -> dict | None:
        """Navigation receipt only: reading it never grants ownership or resends work."""
        with self._receipt_lock:
            return dict(self._last_created) if self._last_created else None

    def bound_model(self, thread_id: str) -> str:
        """A model hint for review; an unowned chat is rechecked before queueing."""
        with self._state_lock:
            model = self._thread_models.get(thread_id)
            if not model:
                row = self._known_threads.get(thread_id) or {}
                model = row.get("model") or row.get("modelId") or row.get("model_id")
            return model if isinstance(model, str) and model in PERMITTED_MODELS else ""

    def _record_receipt(self, row: dict) -> bool:
        with self._receipt_lock:
            self._last_created = dict(row)
            temporary = None
            try:
                path = self._receipt_path
                if self.project_root not in path.parent.resolve().parents:
                    raise OSError("Receipt directory is outside the project")
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.is_symlink() or self.project_root not in path.parent.resolve().parents:
                    raise OSError("Receipt path is outside the project")
                with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                                 prefix=".codex-receipt-", suffix=".tmp", delete=False) as stream:
                    temporary = Path(stream.name)
                    json.dump(row, stream, ensure_ascii=False, separators=(",", ":"))
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, path)
                persisted = True
            except (OSError, TypeError, ValueError):
                persisted = False
            finally:
                if temporary is not None:
                    try:
                        temporary.unlink(missing_ok=True)
                    except OSError:
                        pass
            log.info("Codex receipt thread_id=%s model=%s submitted=%s turn_id=%s status=%s persisted=%s",
                     row["id"], row["model"], row["submitted"], row.get("turn_id") or "-",
                     row["status"], persisted)
            return persisted

    def _update_receipt(self, thread_id: str, **updates) -> None:
        with self._receipt_lock:
            if self._last_created and self._last_created["id"] == thread_id:
                self._record_receipt({**self._last_created, **updates})

    def ensure_ready(self) -> str:
        with self._startup_lock:
            with self._state_lock:
                if self._closed:
                    raise CodexError("CodexBridge is closed")
                if self._mutation_uncertain:
                    raise DeliveryUnknown("A previous mutation has unknown delivery; check Codex before reconnecting")
                if self._ready and self._fatal is None:
                    assert self._selected_model is not None
                    return self._selected_model
                if self._fatal is not None or self._readiness_error is not None:
                    error = self._readiness_error or self._fatal
                    if self.owned_threads or self._active is not None or self._turn_start_pending or self._mutation_pending:
                        raise CodexError("Codex connection needs an explicit restart; owned chat state cannot be reloaded automatically", code="transport")
                    if time.monotonic() < self._readiness_retry_at:
                        assert error is not None
                        raise error
            # No owned chats or uncertain/pending mutation can cross this path.
            # This is a new explicit readiness attempt, never a prompt replay.
            if self._fatal is not None or self._readiness_error is not None:
                self._terminate_process()
                with self._state_lock:
                    self._fatal = None
                    self._readiness_error = None
            try:
                self._spawn()
                self._request(
                    "initialize",
                    {
                        "clientInfo": {"name": "memoir", "version": "0.1.0"},
                        "capabilities": {},
                    },
                )
                self._send({"jsonrpc": "2.0", "method": "initialized", "params": {}})
                self._check_account()
                selected = self._choose_model()
                with self._state_lock:
                    if self._fatal is not None:
                        raise self._fatal
                    if self._closed or self._proc is None or self._proc.poll() is not None:
                        raise CodexError("Codex connection closed during readiness", code="transport")
                    self._ready = True
                return selected
            except Exception as exc:
                with self._state_lock:
                    self._ready = False
                    self._readiness_error = exc if isinstance(exc, CodexError) else CodexError("Codex readiness failed")
                    self._readiness_retry_at = time.monotonic() + _READINESS_RETRY_COOLDOWN
                self._terminate_process()
                raise

    def list_threads(self) -> list[dict[str, Any]]:
        self.ensure_ready()
        values: list[dict[str, Any]] = []
        cursor: Any = None
        for _page in range(10):
            params: dict[str, Any] = {
                "limit": 100,
                "cwd": str(self.project_root),
                "sourceKinds": ["appServer", "exec", "vscode", "cli"],
            }
            if cursor:
                params["cursor"] = cursor
            result = self._request("thread/list", params)
            chunk = result.get("data") or result.get("threads") or []
            if isinstance(chunk, list):
                for thread in chunk:
                    if not isinstance(thread, dict):
                        continue
                    ident = thread.get("id") or thread.get("threadId")
                    if _safe_id(ident):
                        copy = dict(thread)
                        copy.setdefault("id", ident)
                        values.append(copy)
                        self._known_threads[ident] = copy
            cursor = result.get("nextCursor") or result.get("next_cursor")
            if not cursor:
                break
        return values

    def create_thread(self, title: str = "Memoir voice task", *, model: str | None = None) -> dict[str, Any]:
        if not isinstance(title, str) or not title:
            raise ValueError("title must be a non-empty string")
        self.ensure_ready()
        chosen_model = self._validated_model(model)
        self._guard_mutation()
        self._check_account()
        result = self._request(
            "thread/start",
            {
                "cwd": str(self.project_root),
                "approvalPolicy": "on-request",
                "sandbox": "workspace-write",
                "model": chosen_model,
                "modelProvider": "openai",
                "reasoningEffort": self._model_effort,
            },
            mutation=True,
        )
        thread = result.get("thread") if isinstance(result.get("thread"), dict) else result
        ident = thread.get("id") if isinstance(thread, dict) else None
        if not _safe_id(ident):
            self._mark_mutation_uncertain()
            raise DeliveryUnknown("thread/start did not confirm the created thread identity")
        name = thread.get("name") if isinstance(thread, dict) else None
        output = {"id": ident, "name": name if isinstance(name, str) and name else "Untitled Codex chat",
                  "model": chosen_model}
        # Creation is already confirmed. A subsequent naming failure must not
        # hide this identity or invite another automatic thread/start.
        with self._state_lock:
            self.owned_threads.add(ident)
            self._thread_models[ident] = chosen_model
        self._known_threads[ident] = output
        persisted = self._record_receipt({"schema": 1, "project_root": str(self.project_root),
                                         **output, "name": output["name"][:120], "created_at": time.time(),
                                         "submitted": False, "turn_id": None, "status": "created"})
        try:
            self._request(
                "thread/name/set",
                {"threadId": ident, "name": title},
                mutation=True,
            )
        except Exception as exc:
            if not isinstance(exc, CodexError):
                self._mark_mutation_uncertain()
            output["warning"] = "The requested chat name could not be confirmed."
            with self._state_lock:
                output["can_submit"] = not self._mutation_uncertain and not self._quota_blocked and self._fatal is None
        else:
            output["name"] = title
            self._update_receipt(ident, name=title[:120])
        if not persisted:
            output["warning"] = (output.get("warning", "") + " The created identity could not be saved for restart recovery.").strip()
        return output

    def start_turn(self, thread_id: str, prompt: str) -> dict[str, Any]:
        self._validate_id(thread_id)
        if not isinstance(prompt, str) or not 1 <= len(prompt) <= 12000:
            raise ValueError("prompt must contain 1..12000 characters")
        self.ensure_ready()
        self._guard_mutation()
        self._check_account()
        with self._state_lock:
            if thread_id not in self.owned_threads:
                raise CodexError("start_turn is restricted to bridge-owned threads")
            if self._active is not None or self._turn_start_pending:
                raise CodexError("only one bridge turn may be active")
            self._turn_start_pending = True
            chosen_model = self._thread_models.get(thread_id, self._selected_model)
        # A process exit during delivery must not leave a durable "unsent"
        # receipt. This conservative marker grants no permission to retry.
        self._update_receipt(thread_id, status="submission_unknown")
        try:
            result = self._request(
                "turn/start",
                {
                    "threadId": thread_id,
                    "cwd": str(self.project_root),
                    "input": [{"type": "text", "text": prompt}],
                    "model": chosen_model,
                    "effort": self._model_effort,
                },
                mutation=True,
            )
        except DeliveryUnknown:
            self._update_receipt(thread_id, status="submission_unknown")
            raise
        except CodexError:
            self._update_receipt(thread_id, status="created")
            raise
        finally:
            with self._state_lock:
                self._turn_start_pending = False
        turn = result.get("turn") if isinstance(result.get("turn"), dict) else result
        turn_id = turn.get("id") if isinstance(turn, dict) else None
        if not _safe_id(turn_id):
            self._mark_mutation_uncertain()
            self._update_receipt(thread_id, status="submission_unknown")
            raise DeliveryUnknown("turn/start did not confirm the submitted turn identity")
        with self._state_lock:
            already_completed = turn_id in self._completed_turn_set
            if not already_completed and self._active is None:
                self._active = (thread_id, turn_id)
        output = {"id": turn_id}
        self._update_receipt(thread_id, submitted=True, turn_id=turn_id, status="submitted")
        if isinstance(turn, dict) and isinstance(turn.get("status"), str):
            output["status"] = turn["status"]
        return output

    def cancel_turn(self) -> bool:
        with self._state_lock:
            active = self._active
            if active is None:
                return False
            if self._cancel_pending:
                return True
            self._cancel_pending = True
        try:
            self._guard_mutation()
            self._check_account()
            self._request(
                "turn/interrupt",
                {"threadId": active[0], "turnId": active[1]},
                mutation=True,
            )
        except Exception:
            with self._state_lock:
                self._cancel_pending = False
            raise
        return True

    def read_thread(self, thread_id: str) -> dict[str, Any]:
        self._validate_id(thread_id)
        self.ensure_ready()
        result = self._request("thread/read", {"threadId": thread_id})
        thread = result.get("thread") if isinstance(result.get("thread"), dict) else result
        if not isinstance(thread, dict):
            raise CodexError("thread/read returned malformed data")
        returned_id = thread.get("id") or thread.get("threadId")
        if returned_id != thread_id:
            raise CodexError("thread/read returned a different thread id")
        self._known_threads[thread_id] = dict(thread)
        return dict(thread)

    def queue_existing(self, thread_id: str, prompt: str, *, allow_model_change: bool = False,
                       model: str | None = None) -> dict[str, Any]:
        self._validate_id(thread_id)
        if not isinstance(prompt, str) or not 1 <= len(prompt) <= 12000:
            raise ValueError("prompt must contain 1..12000 characters")
        self.ensure_ready()
        chosen_model = self._validated_model(model)
        self._guard_mutation()
        with self._state_lock:
            if thread_id in self.owned_threads:
                raise CodexError("queue_existing cannot target a bridge-owned thread")
            if self._active is not None:
                raise CodexError("cannot queue an existing thread while a bridge turn is active")
        # Re-read every time: a cached model can be stale after a desktop turn.
        thread = self.read_thread(thread_id)
        try:
            cwd = thread.get("cwd")
            scoped = isinstance(cwd, str) and Path(cwd).resolve(strict=True) == self.project_root
        except (OSError, ValueError):
            scoped = False
        if not scoped:
            raise CodexError("Existing chat is outside the permitted project", code="scope")
        if thread.get("modelProvider") not in (None, "openai"):
            raise CodexError("Existing chat uses a different provider", code="auth")
        stored_model = thread.get("model") or thread.get("modelId") or thread.get("model_id")
        if allow_model_change is not True and stored_model != chosen_model:
            raise CodexError("Explicit Luna permission is required for this existing chat", code="model_permission")
        self._check_account()
        if not self.executable:
            raise CodexError("Codex executable was not found")
        argv = [
            self.executable,
            "queue",
            "-c",
            "model_provider=openai",
            "--thread",
            thread_id,
            "--message",
            prompt,
            "--model",
            chosen_model,
        ]
        try:
            completed = subprocess.run(
                argv,
                cwd=str(self.project_root),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=self.timeout,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except subprocess.TimeoutExpired as exc:
            self._mark_mutation_uncertain()
            raise DeliveryUnknown("Codex queue timed out; delivery is unknown") from exc
        except OSError as exc:
            raise CodexError(f"Codex queue failed to start: {exc}") from exc
        if completed.returncode != 0:
            raise CodexError(f"Codex queue exited with status {completed.returncode}")
        return {"status": "queued"}

    def drain_events(self) -> list[dict[str, Any]]:
        with self._state_lock:
            events = list(self._events)
            self._events.clear()
            return events

    def _validate_id(self, value: Any) -> None:
        if not _safe_id(value):
            raise ValueError("thread id must be a UUID or thr_ identifier")

    def _terminate_process(self) -> None:
        with self._state_lock:
            proc, self._proc = self._proc, None
            self._reader = None
        if proc is None:
            return
        try:
            if proc.stdin:
                proc.stdin.close()
        except (OSError, ValueError):
            pass
        try:
            proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=1)

    def close(self) -> None:
        with self._state_lock:
            if self._closed:
                return
        self._decline_pending_approval()
        with self._state_lock:
            self._closed = True
        self._connection_failed(CodexError("CodexBridge is closed"))
        self._terminate_process()


def _safe_model_name(value: Any) -> bool:
    return isinstance(value, str) and value.endswith("-luna") and not any(ch.isspace() or ord(ch) < 32 for ch in value)


def _quota_snapshot_available(result: Any) -> bool:
    """Require fresh, complete ordinary allowance; absent values fail closed."""
    if not isinstance(result, dict):
        return False
    mapped = result.get("rateLimitsByLimitId")
    if mapped is not None and not isinstance(mapped, dict):
        return False
    buckets = list(mapped.values()) if mapped else [result.get("rateLimits")]
    if not buckets:
        return False
    for bucket in buckets:
        if (not isinstance(bucket, dict) or bucket.get("limitId") != "codex"
                or "rateLimitReachedType" not in bucket or bucket["rateLimitReachedType"] is not None
                or bucket.get("spendControlReached") is not False
                or bucket.get("individualLimit") is not None):
            return False
        for name in ("primary", "secondary"):
            window = bucket.get(name)
            if not isinstance(window, dict):
                return False
            used = window.get("usedPercent")
            if type(used) not in (int, float) or not math.isfinite(used) or not 0 <= used < 100:
                return False
    # A zero credit balance does not block ordinary subscription windows.
    # Explicit reached/spend limits above still prevent recovery.
    return True


def _classify_error(error: Any) -> tuple[str, Any]:
    """Classify only stable error metadata; never retain raw server text."""
    if not isinstance(error, dict):
        return "protocol", None
    markers = []
    pending = [(error, 0)]
    for _ in range(32):
        if not pending:
            break
        item, depth = pending.pop()
        if isinstance(item, str):
            markers.append(item[:120].lower())
        elif isinstance(item, dict):
            for key in ("code", "type", "status"):
                scalar = item.get(key)
                if isinstance(scalar, (str, int)):
                    markers.append(str(scalar)[:120].lower())
            # Tagged enums represent errors as keys, not a 'type' value.
            markers.extend(str(key).lower() for key in item if key in {
                "usageLimitReached", "rateLimitExceeded", "unauthorized", "authenticationError"})
            if depth < 4:
                for key in ("data", "error", "codexErrorInfo", "nativeCodexErrorInfo"):
                    if key in item:
                        pending.append((item[key], depth + 1))
    markers = " ".join(markers)
    if any(token in markers for token in ("quota", "rate_limit", "ratelimit", "usage")) or error.get("code") == 429:
        code = "quota"
    elif any(token in markers for token in ("auth", "unauthor", "forbidden", "login")):
        code = "auth"
    elif any(token in markers for token in ("model", "unsupported_model")):
        code = "model"
    else:
        code = "protocol"
    retry_at = error.get("retryAt") or error.get("retry_at")
    if not isinstance(retry_at, (str, int, float)):
        retry_at = None
    return code, retry_at


__all__ = ["CodexBridge", "CodexError", "DeliveryUnknown"]
