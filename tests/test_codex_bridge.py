from __future__ import annotations

import json
import queue
import subprocess
import threading
from pathlib import Path

import pytest

from assistant.codex_bridge import CodexBridge, CodexError, DeliveryUnknown


THREAD_ID = "thr_memoir123"
TURN_ID = "thr_turn123"


class _Input:
    def __init__(self, owner):
        self.owner = owner
        self.lines = []
        self.closed = False

    def write(self, raw):
        self.lines.append(json.loads(raw))
        self.owner.handle(self.lines[-1])

    def flush(self):
        return None

    def close(self):
        self.closed = True


class _Output:
    def __init__(self):
        self.items = queue.Queue()

    def put(self, value):
        self.items.put(value)

    def __iter__(self):
        while True:
            item = self.items.get()
            if item is None:
                return
            yield item


class _FakeProc:
    def __init__(self):
        self.stdout = _Output()
        self.stdin = _Input(self)
        self.returncode = None
        self.seen = []
        self.complete_before_turn_response = False
        self.account_type = "chatgpt"
        self.errors = {}
        self.results = {}
        self.rate_limits = {"rateLimitsByLimitId": {"codex": {
            "limitId": "codex", "primary": {"usedPercent": 100},
            "secondary": {"usedPercent": 20}, "rateLimitReachedType": "primary",
            "spendControlReached": False, "individualLimit": None}}}
        self.model_catalog = [{"id": "gpt-5.6-luna", "supportedReasoningEfforts": [{"reasoningEffort": "low"}]}]

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = 0
        return 0

    def terminate(self):
        self.returncode = 0
        self.stdout.put(None)

    def kill(self):
        self.terminate()

    def handle(self, request):
        self.seen.append(request)
        if "method" not in request:
            return
        if "id" not in request:
            return
        method = request["method"]
        if method in self.errors:
            self.emit({"jsonrpc": "2.0", "id": request["id"], "error": self.errors[method]})
            return
        if method == "initialize":
            result = {}
        elif method == "account/read":
            result = {"type": self.account_type}
        elif method == "model/list":
            result = {"data": self.model_catalog}
        elif method == "account/rateLimits/read":
            result = self.rate_limits
        elif method == "thread/list":
            result = {"data": []}
        elif method == "thread/start":
            result = {"thread": {"id": THREAD_ID}}
        elif method == "thread/read":
            result = {"thread": {"id": request["params"]["threadId"], "model": "gpt-5.6-luna", "cwd": self.cwd}}
        elif method == "turn/start":
            if self.complete_before_turn_response:
                self.emit({"jsonrpc": "2.0", "method": "turn/started", "params": {
                    "threadId": THREAD_ID, "turnId": TURN_ID,
                }})
                self.emit({"jsonrpc": "2.0", "method": "item/completed", "params": {
                    "threadId": THREAD_ID, "turnId": TURN_ID,
                    "item": {"type": "agentMessage", "text": "done"},
                }})
                self.emit({"jsonrpc": "2.0", "method": "turn/completed", "params": {
                    "threadId": THREAD_ID, "turnId": TURN_ID,
                    "turn": {"id": TURN_ID, "status": "completed"},
                }})
            result = {"turn": {"id": TURN_ID}}
        elif method == "turn/interrupt":
            result = {}
        else:
            result = {}
        self.emit({"jsonrpc": "2.0", "id": request["id"], "result": self.results.get(method, result)})

    def emit(self, value):
        self.stdout.put(json.dumps(value) + "\n")


@pytest.fixture
def fake_server(monkeypatch):
    proc = _FakeProc()
    def spawn(*args, **kwargs):
        proc.cwd = kwargs.get("cwd")
        return proc
    monkeypatch.setattr(subprocess, "Popen", spawn)
    return proc


def _bridge(tmp_path, fake_server, **kwargs):
    return CodexBridge(tmp_path, executable="codex-test", timeout=0.3, **kwargs)


def test_handshake_model_create_and_exact_prompt(tmp_path, fake_server):
    bridge = _bridge(tmp_path, fake_server)
    try:
        assert bridge.ensure_ready() == "gpt-5.6-luna"
        created = bridge.create_thread("Memoir voice task")
        start = next(x for x in fake_server.seen if x.get("method") == "thread/start")
        assert start["params"]["approvalPolicy"] == "on-request"
        assert start["params"]["sandbox"] == "workspace-write"
        assert created["id"] == THREAD_ID
        fake_server.complete_before_turn_response = True
        turn = bridge.start_turn(THREAD_ID, "line 1\nline 2")
        assert turn["id"] == TURN_ID
        request = next(x for x in fake_server.seen if x.get("method") == "turn/start")
        assert request["params"]["input"][0]["text"] == "line 1\nline 2"
        assert bridge.active_turn is None
        assert any(event.get("text") == "done" for event in bridge.drain_events())
    finally:
        bridge.close()


def test_out_of_order_response_and_id_validation(tmp_path, fake_server):
    bridge = _bridge(tmp_path, fake_server)
    try:
        bridge.ensure_ready()
        with pytest.raises(ValueError):
            bridge.read_thread("bad\nthread")
        assert bridge.list_threads() == []
    finally:
        bridge.close()


def test_auth_and_model_catalog_fail_closed(tmp_path, fake_server, monkeypatch):
    fake_server.account_type = "apiKey"
    bridge = _bridge(tmp_path, fake_server)
    try:
        with pytest.raises(CodexError) as error:
            bridge.ensure_ready()
        assert error.value.code == "auth"
    finally:
        bridge.close()

    unavailable = _FakeProc()
    unavailable.model_catalog = [{"id": "gpt-5.6-codex", "supportedReasoningEfforts": [{"reasoningEffort": "low"}]}]
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: unavailable)
    bridge = _bridge(tmp_path, unavailable)
    try:
        with pytest.raises(CodexError) as error:
            bridge.ensure_ready()
        assert error.value.code == "model"
    finally:
        bridge.close()


def test_approval_request_is_declined_without_raw_payload(tmp_path, fake_server):
    bridge = _bridge(tmp_path, fake_server)
    try:
        bridge.ensure_ready()
        fake_server.emit({
            "jsonrpc": "2.0", "id": 991,
            "method": "item/commandExecution/requestApproval",
            "params": {"threadId": THREAD_ID, "reason": "private"},
        })
        # Give the daemon reader a scheduling opportunity.
        assert _wait_for(lambda: any(x.get("id") == 991 for x in fake_server.seen))
        response = next(x for x in fake_server.seen if x.get("id") == 991)
        assert response["result"] == {"decision": "decline"}
        assert bridge.drain_events()[0]["kind"] == "approval"
    finally:
        bridge.close()


def test_command_approval_is_single_use_and_turn_bound(tmp_path, fake_server):
    bridge = _bridge(tmp_path, fake_server)
    try:
        bridge.ensure_ready()
        bridge.create_thread()
        bridge.start_turn(THREAD_ID, "inspect")
        fake_server.emit({
            "jsonrpc": "2.0", "id": 992,
            "method": "item/commandExecution/requestApproval",
            "params": {"threadId": THREAD_ID, "turnId": TURN_ID,
                        "command": ["python", "-c", "print(1)"], "grantRoot": str(tmp_path)},
        })
        assert _wait_for(lambda: bridge.pending_approval() is not None)
        pending = bridge.pending_approval()
        assert pending["command"] == "python -c print(1)"
        assert bridge.answer_approval(992, THREAD_ID, TURN_ID, True) is True
        assert bridge.pending_approval() is None
        assert bridge.answer_approval(992, THREAD_ID, TURN_ID, True) is False
        response = next(x for x in fake_server.seen if x.get("id") == 992)
        assert response["result"] == {"decision": "accept"}
    finally:
        bridge.close()


def test_queue_existing_uses_exact_argv_and_timeout_is_unknown(tmp_path, fake_server, monkeypatch):
    bridge = _bridge(tmp_path, fake_server)
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda argv, **kwargs: calls.append((argv, kwargs)) or type("R", (), {"returncode": 0})())
    try:
        bridge.ensure_ready()
        result = bridge.queue_existing(THREAD_ID, "do not alter this prompt")
        assert result == {"status": "queued"}
        assert calls[0][0] == ["codex-test", "queue", "-c", "model_provider=openai", "--thread", THREAD_ID, "--message", "do not alter this prompt", "--model", "gpt-5.6-luna"]
    finally:
        bridge.close()


def test_queue_timeout_is_delivery_unknown_and_poisoned(tmp_path, fake_server, monkeypatch):
    bridge = _bridge(tmp_path, fake_server)

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs.get("timeout"))

    monkeypatch.setattr(subprocess, "run", timeout)
    try:
        bridge.ensure_ready()
        with pytest.raises(DeliveryUnknown):
            bridge.queue_existing(THREAD_ID, "uncertain")
        assert bridge.mutation_uncertain is True
        with pytest.raises(DeliveryUnknown):
            bridge.queue_existing(THREAD_ID, "must not replay")
    finally:
        bridge.close()


def _wait_for(predicate):
    deadline = threading.Event()
    for _ in range(100):
        if predicate():
            return True
        deadline.wait(0.005)
    return False


def test_native_nested_quota_terminal_error_clears_only_matching_turn(tmp_path, fake_server):
    bridge = _bridge(tmp_path, fake_server)
    try:
        bridge.create_thread()
        bridge.start_turn(THREAD_ID, "inspect")
        bridge._dispatch_notification({"method": "turn/error", "params": {
            "threadId": THREAD_ID, "turnId": TURN_ID, "willRetry": False,
            "error": {"codexErrorInfo": {"usageLimitReached": {}}}}})
        assert bridge.active_turn is None
        events = bridge.drain_events()
        assert events[-1]["code"] == "quota"
        assert "won't retry" in events[-1]["message"]
        with pytest.raises(CodexError) as caught:
            bridge.start_turn(THREAD_ID, "must not retry")
        assert caught.value.code == "quota"
    finally:
        bridge.close()


def test_retriable_or_stale_error_does_not_clear_current_turn(tmp_path, fake_server):
    bridge = _bridge(tmp_path, fake_server)
    try:
        bridge.create_thread()
        bridge.start_turn(THREAD_ID, "inspect")
        for turn_id, retry in [(TURN_ID, True), ("thr_oldturn", False)]:
            bridge._dispatch_notification({"method": "turn/error", "params": {
                "threadId": THREAD_ID, "turnId": turn_id, "willRetry": retry,
                "error": {"codexErrorInfo": "serverOverloaded"}}})
            assert bridge.active_turn == (THREAD_ID, TURN_ID)
    finally:
        bridge.close()


def test_existing_scope_is_revalidated_even_with_explicit_luna_permission(tmp_path, fake_server, monkeypatch):
    bridge = _bridge(tmp_path, fake_server)
    run = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: run.append(a))
    try:
        bridge.ensure_ready()
        bridge.read_thread = lambda _: {"id": THREAD_ID, "cwd": str(tmp_path.parent), "modelProvider": "openai"}
        with pytest.raises(CodexError) as caught:
            bridge.queue_existing(THREAD_ID, "no external scope", allow_model_change=True)
        assert caught.value.code == "scope"
        assert run == []
    finally:
        bridge.close()


def test_missing_existing_model_requires_explicit_luna_permission(tmp_path, fake_server, monkeypatch):
    bridge = _bridge(tmp_path, fake_server)
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: calls.append(a) or type("R", (), {"returncode": 0})())
    try:
        bridge.ensure_ready()
        bridge.read_thread = lambda _: {"id": THREAD_ID, "cwd": str(tmp_path), "modelProvider": "openai"}
        with pytest.raises(CodexError) as caught:
            bridge.queue_existing(THREAD_ID, "preserve choice")
        assert caught.value.code == "model_permission"
        assert calls == []
        assert bridge.queue_existing(THREAD_ID, "explicitly choose Luna", allow_model_change=True)["status"] == "queued"
        assert len(calls) == 1
    finally:
        bridge.close()


def test_known_startup_auth_failure_recovers_only_on_later_explicit_attempt(tmp_path, monkeypatch):
    from assistant import codex_bridge as module
    now = [10.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    failed, healthy = _FakeProc(), _FakeProc()
    failed.account_type = "apiKey"
    spawned = []
    def spawn(*args, **kwargs):
        proc = (failed, healthy)[len(spawned)]
        proc.cwd = kwargs.get("cwd")
        spawned.append(proc)
        return proc
    monkeypatch.setattr(subprocess, "Popen", spawn)
    bridge = _bridge(tmp_path, failed)
    try:
        with pytest.raises(CodexError) as caught:
            bridge.ensure_ready()
        assert caught.value.code == "auth"
        with pytest.raises(CodexError) as caught:
            bridge.ensure_ready()
        assert caught.value.code == "auth" and len(spawned) == 1
        now[0] += 6
        assert bridge.ensure_ready() == "gpt-5.6-luna"
        # Simulate EOF delivered by the retired reader after the new handshake.
        bridge._connection_failed(CodexError("old EOF"), proc=failed)
        assert bridge._fatal is None and bridge.list_threads() == []
        assert len(spawned) == 2
        assert not any(request.get("method") in {"thread/start", "turn/start"}
                       for proc in spawned for request in proc.seen)
    finally:
        bridge.close()


def test_readonly_connection_failure_reconnects_without_owned_chat_replay(tmp_path, monkeypatch):
    from assistant import codex_bridge as module
    now = [10.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    procs = [_FakeProc(), _FakeProc()]
    spawned = []
    def spawn(*args, **kwargs):
        proc = procs[len(spawned)]
        proc.cwd = kwargs.get("cwd")
        spawned.append(proc)
        return proc
    monkeypatch.setattr(subprocess, "Popen", spawn)
    bridge = _bridge(tmp_path, procs[0])
    try:
        bridge.ensure_ready()
        bridge._connection_failed(CodexError("connection lost", code="transport"), proc=procs[0])
        with pytest.raises(CodexError):
            bridge.ensure_ready()
        assert len(spawned) == 1
        now[0] += 6
        assert bridge.list_threads() == [] and len(spawned) == 2
    finally:
        bridge.close()


@pytest.mark.parametrize("owned,uncertain,pending", [(True, False, False), (False, True, False), (False, False, True)])
def test_reconnect_never_crosses_owned_or_uncertain_or_pending_mutation(tmp_path, fake_server, monkeypatch, owned, uncertain, pending):
    bridge = _bridge(tmp_path, fake_server)
    try:
        bridge.ensure_ready()
        if owned:
            bridge.owned_threads.add(THREAD_ID)
        bridge._mutation_uncertain = uncertain
        bridge._turn_start_pending = pending
        bridge._connection_failed(CodexError("connection lost"), proc=fake_server)
        bridge._readiness_retry_at = 0
        spawn = pytest.MonkeyPatch()
        spawn.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("unsafe reconnect"))
        try:
            with pytest.raises(CodexError):
                bridge.ensure_ready()
        finally:
            spawn.undo()
    finally:
        bridge.close()


@pytest.mark.parametrize("method", ["thread/start", "turn/start"])
def test_malformed_mutation_acknowledgement_poisoned_without_replay(tmp_path, fake_server, method):
    bridge = _bridge(tmp_path, fake_server)
    try:
        if method == "turn/start":
            bridge.create_thread()
        fake_server.results[method] = {}
        with pytest.raises(DeliveryUnknown):
            bridge.create_thread() if method == "thread/start" else bridge.start_turn(THREAD_ID, "first")
        before = len(fake_server.seen)
        assert bridge.mutation_uncertain
        with pytest.raises(DeliveryUnknown):
            bridge.create_thread()
        assert len(fake_server.seen) == before
    finally:
        bridge.close()


def test_known_naming_failure_retains_created_id_and_allows_confirmed_task(tmp_path, fake_server):
    fake_server.errors["thread/name/set"] = {"code": -32601}
    bridge = _bridge(tmp_path, fake_server)
    try:
        created = bridge.create_thread("Requested name")
        assert created["id"] == THREAD_ID and created["name"] == "Untitled Codex chat"
        assert created["can_submit"] is True and "could not be confirmed" in created["warning"]
        assert THREAD_ID in bridge.owned_threads
        assert bridge._known_threads[THREAD_ID] is created
        assert bridge.start_turn(THREAD_ID, "confirmed task")["id"] == TURN_ID
        assert len([row for row in fake_server.seen if row.get("method") == "thread/start"]) == 1
    finally:
        bridge.close()


def test_uncertain_naming_retains_created_id_without_prompt_send(tmp_path, fake_server):
    bridge = _bridge(tmp_path, fake_server)
    request = bridge._request
    def uncertain(method, *args, **kwargs):
        if method == "thread/name/set":
            bridge._mark_mutation_uncertain()
            raise DeliveryUnknown()
        return request(method, *args, **kwargs)
    bridge._request = uncertain
    try:
        created = bridge.create_thread()
        assert created["id"] == THREAD_ID and created["can_submit"] is False
        assert THREAD_ID in bridge.owned_threads and bridge.mutation_uncertain
        with pytest.raises(DeliveryUnknown):
            bridge.start_turn(THREAD_ID, "must not send")
        assert not any(row.get("method") == "turn/start" for row in fake_server.seen)
    finally:
        bridge.close()


def test_quota_latch_does_not_claim_an_unverified_reset(tmp_path, fake_server):
    bridge = _bridge(tmp_path, fake_server)
    try:
        bridge.ensure_ready()
        bridge._quota_blocked = True
        with pytest.raises(CodexError, match="reset has not been verified") as caught:
            bridge.create_thread()
        assert caught.value.code == "quota"
        assert bridge.list_threads() == []
        assert not any(row.get("method") == "thread/start" for row in fake_server.seen)
    finally:
        bridge.close()


def _allowance(primary=10, secondary=20, **overrides):
    return {"rateLimitsByLimitId": {"codex": dict({
        "limitId": "codex", "primary": {"usedPercent": primary},
        "secondary": {"usedPercent": secondary}, "rateLimitReachedType": None,
        "spendControlReached": False, "individualLimit": None,
        "credits": {"hasCredits": False, "unlimited": False, "balance": "0"},
    }, **overrides)}}


def test_explicit_quota_attempt_recovers_only_after_fresh_complete_allowance(tmp_path, fake_server):
    bridge = _bridge(tmp_path, fake_server)
    try:
        bridge.ensure_ready()
        bridge._quota_blocked = True
        fake_server.rate_limits = _allowance()
        created = bridge.create_thread("Fresh explicit request")
        assert created["id"] == THREAD_ID and not bridge._quota_blocked
        methods = [row.get("method") for row in fake_server.seen]
        assert methods.count("account/rateLimits/read") == 1
        assert methods.index("account/rateLimits/read") < methods.index("thread/start")
        assert not any("credit" in (method or "").casefold() for method in methods)
    finally:
        bridge.close()


@pytest.mark.parametrize("limits", [
    {}, {"rateLimitsByLimitId": "unknown"},
    _allowance(primary=100), _allowance(secondary=100), _allowance(primary=None),
    _allowance(primary=True), _allowance(primary=float("nan")),
    _allowance(rateLimitReachedType="credits"), _allowance(spendControlReached=True),
    _allowance(primary={}), _allowance(secondary=None), _allowance(limitId="unknown"),
    _allowance(individualLimit={"usedPercent": 100}),
])
def test_unknown_or_exhausted_allowance_never_clears_quota(tmp_path, fake_server, limits):
    bridge = _bridge(tmp_path, fake_server)
    try:
        bridge.ensure_ready()
        bridge._quota_blocked = True
        fake_server.rate_limits = limits
        with pytest.raises(CodexError) as caught:
            bridge.create_thread()
        assert caught.value.code == "quota" and bridge._quota_blocked
        assert not any(row.get("method") == "thread/start" for row in fake_server.seen)
    finally:
        bridge.close()


def test_quota_recheck_is_explicit_and_cooldown_bounded(tmp_path, fake_server, monkeypatch):
    from assistant import codex_bridge as module
    now = [10.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    bridge = _bridge(tmp_path, fake_server)
    try:
        bridge.ensure_ready()
        bridge._quota_blocked = True
        for _ in range(2):
            with pytest.raises(CodexError):
                bridge.create_thread()
        assert len([row for row in fake_server.seen if row.get("method") == "account/rateLimits/read"]) == 1
        fake_server.rate_limits = _allowance()
        bridge.list_threads()
        assert bridge._quota_blocked  # Read-only listing never retries a held mutation.
        now[0] += 31
        assert bridge.create_thread()["id"] == THREAD_ID
        assert len([row for row in fake_server.seen if row.get("method") == "account/rateLimits/read"]) == 2
    finally:
        bridge.close()


def test_unsupported_quota_endpoint_stays_blocked_without_retrying_mutation(tmp_path, fake_server):
    fake_server.errors["account/rateLimits/read"] = {"code": -32601}
    bridge = _bridge(tmp_path, fake_server)
    try:
        bridge.ensure_ready()
        bridge._quota_blocked = True
        with pytest.raises(CodexError, match="reset has not been verified"):
            bridge.create_thread()
        assert bridge._quota_blocked
        assert not any(row.get("method") == "thread/start" for row in fake_server.seen)
    finally:
        bridge.close()


def test_verified_allowance_does_not_unlock_uncertain_delivery(tmp_path, fake_server):
    bridge = _bridge(tmp_path, fake_server)
    try:
        bridge.ensure_ready()
        bridge._quota_blocked = True
        bridge._mark_mutation_uncertain()
        fake_server.rate_limits = _allowance()
        before = len(fake_server.seen)
        with pytest.raises(DeliveryUnknown):
            bridge.create_thread()
        assert len(fake_server.seen) == before and bridge._quota_blocked
    finally:
        bridge.close()


def test_connection_failure_during_handshake_cannot_report_ready(tmp_path, fake_server):
    bridge = _bridge(tmp_path, fake_server)
    choose = bridge._choose_model
    def fail_after_model():
        selected = choose()
        bridge._connection_failed(CodexError("connection closed", code="transport"), proc=fake_server)
        return selected
    bridge._choose_model = fail_after_model
    try:
        with pytest.raises(CodexError):
            bridge.ensure_ready()
        assert not bridge._ready
    finally:
        bridge.close()


def test_quota_parser_uses_all_mapped_buckets_and_known_legacy_shape():
    from assistant.codex_bridge import _quota_snapshot_available
    good = _allowance()
    bucket = good["rateLimitsByLimitId"]["codex"]
    assert _quota_snapshot_available({"rateLimits": bucket})
    assert not _quota_snapshot_available({"rateLimits": bucket,
        "rateLimitsByLimitId": {"codex": bucket, "unknown": {}}})
    missing_reached = dict(bucket)
    del missing_reached["rateLimitReachedType"]
    assert not _quota_snapshot_available({"rateLimits": missing_reached})


@pytest.mark.parametrize("method", ["thread/start", "thread/name/set", "turn/interrupt"])
def test_connection_failure_poisoned_for_every_inflight_mutation(tmp_path, fake_server, method):
    bridge = _bridge(tmp_path, fake_server)
    try:
        bridge.ensure_ready()
        def fail_during_request(request_method, params, **kwargs):
            assert request_method == method and bridge._mutation_pending == 1
            bridge._connection_failed(CodexError("lost during mutation"), proc=fake_server)
            raise DeliveryUnknown()
        bridge._request_once = fail_during_request
        with pytest.raises(DeliveryUnknown):
            bridge._request(method, {}, mutation=True)
        assert bridge._mutation_pending == 0 and bridge.mutation_uncertain
        with pytest.raises(DeliveryUnknown):
            bridge.ensure_ready()
    finally:
        bridge.close()


def test_retired_decoded_notifications_and_requests_cannot_touch_new_transport(tmp_path, fake_server):
    bridge = _bridge(tmp_path, fake_server)
    try:
        bridge.ensure_ready()
        replacement = _FakeProc()
        bridge._proc = replacement
        bridge.owned_threads.add(THREAD_ID)
        bridge._active = (THREAD_ID, TURN_ID)
        bridge._dispatch_notification({"method": "turn/completed", "params": {
            "threadId": THREAD_ID, "turnId": TURN_ID,
            "turn": {"id": TURN_ID, "status": "completed"}}}, proc=fake_server)
        bridge._handle_server_request({"id": 7, "method": "unsupported", "params": {}}, proc=fake_server)
        assert bridge.active_turn == (THREAD_ID, TURN_ID)
        assert bridge.drain_events() == [] and replacement.seen == []
    finally:
        bridge.close()


def test_server_request_reply_remains_bound_when_retirement_races_write(tmp_path, fake_server):
    bridge = _bridge(tmp_path, fake_server)
    try:
        bridge.ensure_ready()
        replacement = _FakeProc()
        send = bridge._send
        def retire_before_write(message, *, proc=None):
            assert proc is fake_server
            bridge._proc = replacement
            send(message, proc=proc)
        bridge._send = retire_before_write
        bridge._handle_server_request({"id": 7, "method": "unsupported", "params": {}}, proc=fake_server)
        assert replacement.seen == []
        assert fake_server.seen[-1]["id"] == 7 and "error" in fake_server.seen[-1]
    finally:
        bridge.close()


def test_connection_failure_fanout_is_atomic_with_transport_retirement(tmp_path, fake_server):
    from types import SimpleNamespace

    bridge = _bridge(tmp_path, fake_server)
    original_lock = bridge._pending_lock

    class ObservedPendingLock:
        def __enter__(self):
            assert bridge._state_lock._is_owned()
            original_lock.acquire()

        def __exit__(self, *args):
            original_lock.release()

    try:
        bridge.ensure_ready()
        pending = SimpleNamespace(error=None, event=threading.Event())
        bridge._pending[999] = pending
        bridge._pending_lock = ObservedPendingLock()
        error = CodexError("old connection failed", code="transport")
        bridge._connection_failed(error, proc=fake_server)
        assert pending.error is error and pending.event.is_set()

        # A retired reader must not poison a replacement handshake's waiter.
        bridge._proc = _FakeProc()
        replacement_pending = SimpleNamespace(error=None, event=threading.Event())
        bridge._pending[999] = replacement_pending
        bridge._connection_failed(error, proc=fake_server)
        assert replacement_pending.error is None
        assert not replacement_pending.event.is_set()
    finally:
        bridge._pending_lock = original_lock
        bridge._pending.pop(999, None)
        bridge.close()


def test_requested_luna_is_catalogue_checked_and_bound_to_created_turn(tmp_path, fake_server):
    fake_server.model_catalog.append({"id": "gpt-6-luna", "supportedReasoningEfforts": [{"reasoningEffort": "low"}]})
    bridge = _bridge(tmp_path, fake_server)
    try:
        created = bridge.create_thread("Six", model="gpt-6-luna")
        assert created["model"] == "gpt-6-luna"
        bridge.start_turn(created["id"], "Exact task")
        mutations = [row for row in fake_server.seen if row.get("method") in {"thread/start", "turn/start"}]
        assert [row["params"]["model"] for row in mutations] == ["gpt-6-luna", "gpt-6-luna"]
        assert bridge._selected_model == "gpt-5.6-luna"
    finally:
        bridge.close()


@pytest.mark.parametrize("model", ["gpt-6-luna", "gpt-6-astra", "gpt-9-luna"])
def test_unavailable_or_disallowed_requested_model_never_creates(tmp_path, fake_server, model):
    bridge = _bridge(tmp_path, fake_server)
    try:
        with pytest.raises(CodexError) as failure:
            bridge.create_thread("No substitution", model=model)
        assert failure.value.code == "model"
        assert not any(row.get("method") == "thread/start" for row in fake_server.seen)
    finally:
        bridge.close()


def test_native_receipt_survives_restart_without_restoring_ownership(tmp_path, fake_server):
    bridge = _bridge(tmp_path, fake_server)
    try:
        created = bridge.create_thread("Receipt")
        bridge.start_turn(created["id"], "Private prompt must not enter the receipt")
        receipt_path = tmp_path / "assistant" / "logs" / ".codex-last-created.json"
        payload = receipt_path.read_text(encoding="utf-8")
        assert "Private prompt" not in payload
        row = json.loads(payload)
        assert row["id"] == THREAD_ID and row["turn_id"] == TURN_ID and row["submitted"] is True
        assert row["model"] == "gpt-5.6-luna"
    finally:
        bridge.close()
    restored = CodexBridge(tmp_path, executable="codex-test")
    assert restored.last_created()["id"] == THREAD_ID
    assert not restored.owned_threads and restored.active_turn is None and restored._proc is None


def test_receipt_identity_is_saved_before_best_effort_naming(tmp_path, fake_server):
    fake_server.errors["thread/name/set"] = {"message": "naming unavailable"}
    bridge = _bridge(tmp_path, fake_server)
    try:
        created = bridge.create_thread("Wanted")
        assert created["id"] == THREAD_ID and created["warning"]
        assert bridge.last_created()["id"] == THREAD_ID
        assert json.loads(bridge._receipt_path.read_text())["id"] == THREAD_ID
        assert len([row for row in fake_server.seen if row.get("method") == "thread/start"]) == 1
    finally:
        bridge.close()


def test_untrusted_recovery_receipt_cannot_restore_a_temporary_plan(tmp_path, fake_server):
    bridge = _bridge(tmp_path, fake_server)
    path = bridge._receipt_path
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"schema": 1, "project_root": str(tmp_path), "id": "thr_voice_plan_fake",
                               "name": "Fake", "model": "gpt-5.6-luna", "created_at": 1,
                               "submitted": False, "turn_id": None, "status": "created"}))
    assert bridge._load_receipt() is None


def test_receipt_write_failure_never_loses_confirmed_identity_or_recreates(tmp_path, fake_server, monkeypatch):
    import assistant.codex_bridge as module
    monkeypatch.setattr(module.os, "replace", lambda *args: (_ for _ in ()).throw(OSError("disk failure")))
    bridge = _bridge(tmp_path, fake_server)
    try:
        created = bridge.create_thread("Retained")
        assert created["id"] == THREAD_ID and "saved" in created["warning"]
        assert bridge.last_created()["id"] == THREAD_ID
        assert len([row for row in fake_server.seen if row.get("method") == "thread/start"]) == 1
    finally:
        bridge.close()


def test_existing_reviewed_six_model_is_preserved_without_changing_default(tmp_path, fake_server, monkeypatch):
    fake_server.model_catalog.append({"id": "gpt-6-luna", "supportedReasoningEfforts": [{"reasoningEffort": "low"}]})
    bridge = _bridge(tmp_path, fake_server)
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda argv, **kwargs: calls.append(argv) or subprocess.CompletedProcess(argv, 0))
    try:
        bridge.ensure_ready()
        bridge.read_thread = lambda ident: {"id": ident, "cwd": str(tmp_path), "model": "gpt-6-luna", "modelProvider": "openai"}
        assert bridge.queue_existing(THREAD_ID, "Exact task", model="gpt-6-luna")["status"] == "queued"
        assert calls[0][-2:] == ["--model", "gpt-6-luna"]
        assert bridge._selected_model == "gpt-5.6-luna"
    finally:
        bridge.close()


def test_submission_is_durably_unknown_before_request_enters_transport(tmp_path, fake_server):
    bridge = _bridge(tmp_path, fake_server)
    try:
        created = bridge.create_thread("Crash-safe")
        request = bridge._request
        def observing(method, params, **options):
            if method == "turn/start":
                saved = json.loads(bridge._receipt_path.read_text())
                assert saved["status"] == "submission_unknown" and not saved["submitted"]
            return request(method, params, **options)
        bridge._request = observing
        bridge.start_turn(created["id"], "Private task")
        assert bridge.last_created()["status"] == "submitted"
    finally:
        bridge.close()


def _installed_binary(tmp_path, relative):
    path = tmp_path / relative / "codex.exe"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return str(path.resolve())


def test_lazy_resolver_chooses_highest_verified_version_not_first_path(tmp_path):
    from assistant.codex_bridge import _resolve_codex_executable
    old = _installed_binary(tmp_path, "Programs/OpenAI/Codex/bin")
    current = _installed_binary(tmp_path, "OpenAI/Codex/bin/5ea220ae823df3d7")
    versions = {old: "codex-cli 0.152.1\n", current: "codex-cli 0.160.1\n"}
    calls = []
    def version(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, stdout=versions[argv[0]])
    selected = _resolve_codex_executable(platform="nt", local_app_data=tmp_path,
                                        which=lambda _: old, run=version)
    assert selected == current
    assert len(calls) == 2 and all(argv[1:] == ["--version"] for argv, _ in calls)
    assert all(0 < options["timeout"] <= .6 for _, options in calls)


def test_lazy_resolver_accepts_installed_prerelease_codex_version(tmp_path):
    from assistant.codex_bridge import _resolve_codex_executable
    old = _installed_binary(tmp_path, "Programs/OpenAI/Codex/bin")
    current = _installed_binary(tmp_path, "OpenAI/Codex/bin/5ea220ae823df3d7")
    versions = {
        old: "codex-cli 0.160.1\n",
        current: "codex-cli 0.162.0-alpha.2\n",
    }

    def version(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 0, stdout=versions[argv[0]])

    selected = _resolve_codex_executable(
        platform="nt", local_app_data=tmp_path, which=lambda _: None, run=version)
    assert selected == current


@pytest.mark.parametrize(
    ("left_version", "right_version", "expected"),
    [
        ("0.162.0-alpha.2", "0.162.0-alpha.10", "right"),
        ("0.162.0-alpha.10", "0.162.0", "right"),
        ("0.162.99", "0.163.0-alpha.1", "right"),
        # Build metadata must not affect precedence; the path breaks an exact tie.
        ("0.162.0+build.99", "0.162.0+build.1", "right"),
    ],
)
def test_resolver_ranks_codex_semver_prerelease_release_and_build(
        tmp_path, left_version, right_version, expected):
    from assistant.codex_bridge import _resolve_codex_executable
    left = _installed_binary(tmp_path, "OpenAI/Codex/bin/aaaaaaaaaaaaaaaa")
    right = _installed_binary(tmp_path, "OpenAI/Codex/bin/bbbbbbbbbbbbbbbb")
    versions = {
        left: f"codex-cli {left_version}\n",
        right: f"codex-cli {right_version}\n",
    }

    def version(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 0, stdout=versions[argv[0]])

    selected = _resolve_codex_executable(
        platform="nt", local_app_data=tmp_path, which=lambda _: None, run=version)
    assert selected == (left if expected == "left" else right)


@pytest.mark.parametrize(
    "output",
    [
        "codex-cli 01.2.3\n",
        "codex-cli 1.2.3-alpha.02\n",
        "codex-cli 1.2.3-alpha..2\n",
        "codex-cli 1.2.3+\n",
        "codex-cli 1.2.3+build..2\n",
        "codex-cli 1.2.3+build/shell\n",
        "codex-cli 1.2.3\nextra\n",
        "codex-cli 1.2.3\ncodex-cli 1.2.4\n",
        "codex-cli 1.2.3 ; shell text\n",
    ],
)
def test_resolver_rejects_malformed_semver_suffixes_and_extra_lines(tmp_path, output):
    from assistant.codex_bridge import _resolve_codex_executable
    candidate = _installed_binary(tmp_path, "OpenAI/Codex/bin/5ea220ae823df3d7")

    def version(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 0, stdout=output)

    assert _resolve_codex_executable(
        platform="nt", local_app_data=tmp_path, which=lambda _: None, run=version) is None


def test_resolver_ignores_invalid_or_failed_versions_and_untrusted_cwd(tmp_path):
    from assistant.codex_bridge import _resolve_codex_executable
    old = _installed_binary(tmp_path, "Programs/OpenAI/Codex/bin")
    newer = _installed_binary(tmp_path, "OpenAI/Codex/bin/5ea220ae823df3d7")
    unknown = _installed_binary(tmp_path, "untrusted-workspace")
    calls = []
    def version(argv, **kwargs):
        calls.append(argv[0])
        if argv[0] == newer:
            raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
        return subprocess.CompletedProcess(argv, 0, stdout="codex-cli 0.152.1\n")
    assert _resolve_codex_executable(platform="nt", local_app_data=tmp_path,
                                    which=lambda _: unknown, run=version) == old
    assert unknown not in calls
    assert _resolve_codex_executable(platform="nt", local_app_data=tmp_path,
                                    which=lambda _: unknown,
                                    run=lambda argv, **kwargs: subprocess.CompletedProcess(argv, 0, stdout="0.999.9 unverified")) is None


def test_resolver_version_budget_is_absolute_and_bounded(tmp_path):
    from assistant.codex_bridge import _resolve_codex_executable
    _installed_binary(tmp_path, "Programs/OpenAI/Codex/bin")
    _installed_binary(tmp_path, "OpenAI/Codex/bin/5ea220ae823df3d7")
    now, calls = [0.0], []
    def version(argv, **kwargs):
        calls.append(kwargs["timeout"])
        now[0] = 4.5
        return subprocess.CompletedProcess(argv, 0, stdout="codex-cli 0.160.1\n")
    assert _resolve_codex_executable(platform="nt", local_app_data=tmp_path,
                                    which=lambda _: None, run=version, clock=lambda: now[0])
    assert len(calls) == 1


@pytest.mark.parametrize("redirect_inside_anchor", [False, True])
def test_resolver_rejects_redirected_known_root_ancestor_before_version_process(tmp_path, monkeypatch, redirect_inside_anchor):
    from assistant.codex_bridge import _resolve_codex_executable
    from pathlib import Path
    installed = _installed_binary(tmp_path, "OpenAI/Codex/bin/5ea220ae823df3d7")
    anchor = tmp_path.resolve()
    root = anchor / "OpenAI" / "Codex" / "bin"
    redirect = (anchor / "redirected" if redirect_inside_anchor else anchor.parent / "outside")
    original_resolve = Path.resolve
    def resolved(path, *args, **kwargs):
        if path == root:
            return redirect / "Codex" / "bin"
        return original_resolve(path, *args, **kwargs)
    monkeypatch.setattr(Path, "resolve", resolved)
    # The leaf need not be a symlink: an ancestor junction is sufficient.
    monkeypatch.setattr(Path, "is_symlink", lambda path: False)
    calls = []
    assert _resolve_codex_executable(platform="nt", local_app_data=anchor,
        which=lambda _: installed, run=lambda *args, **kwargs: calls.append(args)) is None
    assert not calls


def test_executable_discovery_is_lazy_and_explicit_override_never_probed(tmp_path, fake_server, monkeypatch):
    import assistant.codex_bridge as module
    calls = []
    monkeypatch.setattr(module, "_resolve_codex_executable", lambda: calls.append("resolve") or "resolved-codex")
    lazy = CodexBridge(tmp_path, timeout=.3)
    assert lazy.executable is None and not calls
    lazy.ensure_ready()
    assert calls == ["resolve"] and lazy.executable == "resolved-codex"
    lazy.close()
    explicit = CodexBridge(tmp_path, executable="explicit-codex", timeout=.3)
    assert explicit.executable == "explicit-codex" and calls == ["resolve"]
    fresh = _FakeProc()
    fresh.cwd = str(tmp_path)
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: fresh)
    explicit.ensure_ready()
    assert calls == ["resolve"]
    explicit.close()


def test_non_windows_resolver_preserves_path_without_version_processes():
    from assistant.codex_bridge import _resolve_codex_executable
    def unexpected(*args, **kwargs):
        pytest.fail("Non-Windows path lookup must not run version processes")
    assert _resolve_codex_executable(platform="posix", which=lambda _: "/opt/codex", run=unexpected) == "/opt/codex"
