from __future__ import annotations

import json
import threading
import time
from http.client import HTTPConnection
from pathlib import Path
from http.server import ThreadingHTTPServer

import pytest

from token_tracker import limit_reset, server
from token_tracker.codex_client import CodexUnavailable
from token_tracker.limit_reset import CodexQuotaCache, LimitResetService, LimitSample


def _rate_result(five=12, weekly=None, five_reset=None, weekly_reset=None):
    primary = {}
    secondary = {}
    if five is not None:
        primary["usedPercent"] = five
    if five_reset is not None:
        primary["resetsAt"] = five_reset
    if weekly is not None:
        secondary["usedPercent"] = weekly
    if weekly_reset is not None:
        secondary["resetsAt"] = weekly_reset
    return {"rateLimits": {"primary": primary, "secondary": secondary}}


class FakeCodexClient:
    def __init__(self, *results, block=False):
        self.results = list(results)
        self.calls = 0
        self.lock = threading.Lock()
        self.started = threading.Event()
        self.release = threading.Event()
        self.block = block

    def rate_limits(self):
        with self.lock:
            index = self.calls
            self.calls += 1
        self.started.set()
        if self.block:
            assert self.release.wait(3)
        result = self.results[min(index, len(self.results) - 1)]
        if isinstance(result, Exception):
            raise result
        return result


def _wait_idle(cache: CodexQuotaCache):
    with cache.condition:
        assert cache.condition.wait_for(lambda: not cache.refreshing, timeout=2)


def test_quota_cache_tests_import_this_checkout():
    expected = Path(__file__).resolve().parents[1] / "token_tracker" / "limit_reset.py"
    assert Path(limit_reset.__file__).resolve() == expected
    assert limit_reset._CODEX_QUOTA_CACHE.client is limit_reset._CODEX_LIMIT_CLIENT


def test_cold_http_read_is_nonblocking_and_action_joins_the_same_inflight_read():
    client = FakeCodexClient(_rate_result(five=12, five_reset=1_800_000_100), block=True)
    cache = CodexQuotaCache(client=client)
    action_result, action_errors = [], []
    action_waiting = threading.Event()
    original_wait = cache.condition.wait

    def observed_wait(timeout=None):
        action_waiting.set()
        return original_wait(timeout)

    cache.condition.wait = observed_wait
    try:
        first = cache.http_view()
        assert client.started.wait(1)
        second = cache.http_view()
        assert first["quotaCache"] == {
            "ready": False, "fresh": False, "ageSeconds": None,
            "refreshing": True, "errorCategory": None,
        }
        assert "lastStatus" not in first["providers"]["Codex"]
        assert second["quotaCache"]["refreshing"] is True
        assert client.calls == 1

        def action_read():
            try:
                action_result.append(cache.read_for_action())
            except Exception as exc:  # surface thread errors to the test
                action_errors.append(exc)

        action = threading.Thread(target=action_read)
        action.start()
        assert action_waiting.wait(1)
    finally:
        client.release.set()
    action.join(2)

    assert not action.is_alive() and action_errors == []
    assert client.calls == 1
    sample = action_result[0]
    view = cache.http_view()
    status = view["providers"]["Codex"]["lastStatus"]
    assert view["quotaCache"]["ready"] is True
    assert view["quotaCache"]["fresh"] is True
    assert status["five_hour_percent"] == 12
    assert status["reset_at"] == 1_800_000_100
    assert "weekly_percent" not in status and "weekly_reset_at" not in status
    assert status["observed_at"] == sample.observed_at
    assert status["checkedAt"] == limit_reset._iso_from_epoch(sample.observed_at)


def test_action_reads_a_new_generation_even_when_http_sample_is_fresh():
    client = FakeCodexClient(_rate_result(five=10), _rate_result(five=20))
    cache = CodexQuotaCache(client=client)

    first = cache.read_for_action()
    fresh_http = cache.http_view()
    second = cache.read_for_action()

    assert fresh_http["quotaCache"]["fresh"] is True
    assert client.calls == 2
    assert cache.generation == 2
    assert first.five_hour_percent == 10 and second.five_hour_percent == 20


def test_background_failure_keeps_original_sample_time_age_and_cooldown_recovery():
    clock = [100.0]
    client = FakeCodexClient(
        _rate_result(five=11, five_reset=1_800_000_100),
        RuntimeError("private provider path detail"),
        _rate_result(five=22, five_reset=1_800_000_200),
    )
    cache = CodexQuotaCache(client=client, clock=lambda: clock[0])
    initial = cache.read_for_action()
    checked_at = limit_reset._iso_from_epoch(initial.observed_at)
    clock[0] += 3

    stale = cache.http_view()
    _wait_idle(cache)
    stale = cache.http_view()
    last_status = stale["providers"]["Codex"]["lastStatus"]
    assert stale["quotaCache"] == {
        "ready": True, "fresh": False, "ageSeconds": 3.0,
        "refreshing": False, "errorCategory": "read_failed",
    }
    assert last_status["checkedAt"] == checked_at
    assert last_status["five_hour_percent"] == 11
    assert "private provider path detail" not in json.dumps(stale)
    cache.http_view()
    assert client.calls == 2

    clock[0] += limit_reset.CODEX_QUOTA_RETRY_SECONDS + 0.01
    recovering = cache.http_view()
    assert recovering["quotaCache"]["refreshing"] is True
    _wait_idle(cache)
    recovered = cache.http_view()
    assert recovered["providers"]["Codex"]["lastStatus"]["five_hour_percent"] == 22
    assert recovered["quotaCache"]["ready"] is True
    assert recovered["quotaCache"]["fresh"] is True
    assert recovered["quotaCache"]["errorCategory"] is None
    assert client.calls == 3


def test_worker_start_failure_is_bounded_and_action_fails_closed():
    clock = [100.0]
    client = FakeCodexClient(_rate_result(five=10))

    class FailedThread:
        def __init__(self, **_kwargs):
            pass

        def start(self):
            raise RuntimeError("private thread start detail")

    cache = CodexQuotaCache(client=client, clock=lambda: clock[0], thread_factory=FailedThread)
    first = cache.http_view()
    assert first["quotaCache"]["ready"] is False
    assert first["quotaCache"]["refreshing"] is False
    assert first["quotaCache"]["errorCategory"] == "worker_start_failed"
    assert "private thread start detail" not in json.dumps(first)
    cache.http_view()
    assert client.calls == 0
    with pytest.raises(RuntimeError, match="private thread start detail"):
        cache.read_for_action()

    clock[0] += limit_reset.CODEX_QUOTA_RETRY_SECONDS + 0.01
    cache.thread_factory = threading.Thread
    cache.http_view()
    _wait_idle(cache)
    assert cache.http_view()["quotaCache"]["ready"] is True
    assert client.calls == 1


def test_watcher_checked_at_preserves_source_observation_time(tmp_path):
    observed = time.time() - 60
    service = LimitResetService(
        sources={"Codex": lambda: LimitSample(50, 20, observed)},
        senders={}, state_path=tmp_path / "state.json", log_path=tmp_path / "events.jsonl",
        enabled=True, clock=lambda: observed + 120,
    )
    service.check_provider("Codex", LimitSample(50, 20, observed))

    checked_at = service._provider_state("Codex")["lastStatus"]["checkedAt"]
    assert checked_at == limit_reset._iso_from_epoch(observed)
    assert checked_at != limit_reset._iso_from_epoch(observed + 120)


def test_action_does_not_use_stale_cache_sample_to_advance_verification(tmp_path, monkeypatch):
    now = time.time()
    stale = LimitSample(80, 20, now - 30, now + 300, now + 1000)
    client = FakeCodexClient(_rate_result(five=5))
    cache = CodexQuotaCache(client=client, clock=lambda: 50.0)
    cache.sample = stale
    cache.sampled_at = 20.0
    cache.generation = 4
    cache.retry_after = 55.0
    cache.error_category = "read_failed"
    cache._last_error = RuntimeError("transient quota read failure")
    monkeypatch.setattr(limit_reset, "_CODEX_QUOTA_CACHE", cache)

    service = LimitResetService(
        sources={"Codex": limit_reset.codex_limit_sample},
        senders={"Codex": lambda _cwd: pytest.fail("stale quota must not start an action")},
        state_path=tmp_path / "state.json", log_path=tmp_path / "events.jsonl",
        enabled=True, clock=lambda: now,
    )
    state = service._provider_state("Codex")
    state.update({
        "awaitingVerification": True, "verificationDeadline": now + 60,
        "preSendPercent": 10, "preSendResetAt": now + 100,
        "lastAttemptAt": limit_reset._iso_from_epoch(now - 1),
    })
    original_status = state.get("lastStatus")
    monkeypatch.setattr(service, "_save_state", lambda: True)
    service.check_once()

    assert state["awaitingVerification"] is True
    assert state.get("lastVerifiedAt") is None
    assert state.get("lastStatus") == original_status
    assert client.calls == 0


@pytest.mark.parametrize("enabled", [False, True])
def test_http_quota_route_is_origin_guarded_and_never_invokes_watcher_actions(tmp_path, monkeypatch, enabled):
    client = FakeCodexClient(_rate_result(five=12), block=True)
    cache = CodexQuotaCache(client=client)
    monkeypatch.setattr(limit_reset, "_CODEX_QUOTA_CACHE", cache)
    calls = []

    class EnabledWatcher:
        def __init__(self, enabled):
            self.enabled = enabled

        def status(self):
            calls.append("status")
            return {"enabled": True, "providers": {}}

        def check_once(self):
            calls.append("check")

        def check_provider(self, *_args):
            calls.append("arm")

        def _save_state(self):
            calls.append("persist")

    monkeypatch.setattr(server.Handler, "reset_service", EnabledWatcher(enabled))
    httpd = server.ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    worker = threading.Thread(target=httpd.serve_forever, daemon=True)
    worker.start()
    connection = HTTPConnection("127.0.0.1", httpd.server_port, timeout=2)
    origin = f"http://127.0.0.1:{httpd.server_port}"
    try:
        connection.request("GET", "/api/quota", headers={"Origin": origin, "Sec-Fetch-Site": "same-origin"})
        response = connection.getresponse()
        payload = json.loads(response.read())
        assert response.status == 200
        assert payload["quotaCache"]["ready"] is False
        assert "lastStatus" not in payload["providers"]["Codex"]
        assert client.started.wait(1)
        client.release.set()
        _wait_idle(cache)

        connection.request("GET", "/api/quota", headers={"Origin": origin, "Sec-Fetch-Site": "same-origin"})
        response = connection.getresponse()
        payload = json.loads(response.read())
        assert response.status == 200
        assert payload["providers"]["Codex"]["lastStatus"]["five_hour_percent"] == 12
        assert payload["quotaCache"]["ready"] is True

        connection.request("GET", "/api/quota", headers={"Origin": "http://evil.example"})
        assert connection.getresponse().status == 403
        connection.request("POST", "/api/quota", body="{}", headers={
            "Origin": origin, "Content-Type": "application/json", "Content-Length": "2",
        })
        assert connection.getresponse().status == 404
    finally:
        client.release.set()
        connection.close()
        httpd.shutdown()
        httpd.server_close()
        worker.join(timeout=2)

    assert calls == []
    assert client.calls == 1
