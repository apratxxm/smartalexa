import threading

import pytest

from assistant.model_runtime import LocalModelRuntime, RuntimeAdmissionError


def test_runtime_enforces_policy_on_copy_and_marks_ready():
    runtime = LocalModelRuntime("local-model", num_ctx=8192, keep_alive=-1)
    body = {"model": "local-model", "keep_alive": 4,
            "options": {"num_ctx": 1024, "temperature": 0}, "messages": [{"content": "hi"}]}
    seen = []

    def transport(request, timeout):
        seen.append((request, timeout))
        return {"ok": True}

    assert runtime.run(transport, body, 2.0) == {"ok": True}
    assert runtime.state == "ready" and runtime.can_attempt()
    assert seen[0][0]["keep_alive"] == -1
    assert seen[0][0]["options"] == {"num_ctx": 8192, "temperature": 0}
    assert body["keep_alive"] == 4 and body["options"]["num_ctx"] == 1024


@pytest.mark.parametrize("num_ctx", [255, 32769, True, 4096.0])
def test_context_size_requires_bounded_nonbool_integer(num_ctx):
    with pytest.raises(ValueError):
        LocalModelRuntime("local-model", num_ctx=num_ctx)


@pytest.mark.parametrize("model", ["", "  ", "cloud-model"])
def test_runtime_rejects_invalid_or_remote_model(model):
    with pytest.raises(ValueError):
        LocalModelRuntime(model)


def test_model_mismatch_fails_before_transport():
    runtime = LocalModelRuntime("local-model")
    calls = []
    with pytest.raises(ValueError, match="does not match"):
        runtime.run(lambda *args: calls.append(args), {"model": "other"}, 1.0)
    assert calls == [] and runtime.state == "cold"


def test_failed_warmup_stays_degraded_after_cooldown_until_success():
    now = [0.0]
    runtime = LocalModelRuntime("local-model", cooldown_seconds=5, clock=lambda: now[0])
    runtime.mark_warming()
    with pytest.raises(RuntimeError):
        runtime.run(lambda *_: (_ for _ in ()).throw(RuntimeError("offline")),
                    {"model": "local-model"}, 1.0, warmup=True)
    assert runtime.state == "degraded" and not runtime.can_attempt()
    now[0] = 5.0
    assert runtime.state == "degraded" and runtime.can_attempt()
    assert runtime.run(lambda *_: {}, {"model": "local-model"}, 1.0) == {}
    assert runtime.state == "ready"


def test_warming_and_cooldown_admission_refusal_does_not_call_transport():
    runtime = LocalModelRuntime("local-model")
    runtime.mark_warming()
    calls = []
    with pytest.raises(RuntimeAdmissionError):
        runtime.run(lambda *args: calls.append(args), {"model": "local-model"}, 1.0)
    assert calls == [] and runtime.state == "warming"
    runtime.mark_failed()
    with pytest.raises(RuntimeAdmissionError):
        runtime.run(lambda *args: calls.append(args), {"model": "local-model"}, 1.0)
    assert calls == [] and runtime.state == "degraded"


def test_serializes_transports_and_queue_deadline_does_not_degrade_owner():
    runtime = LocalModelRuntime("local-model")
    entered = threading.Event()
    release = threading.Event()
    active = [0]
    max_active = [0]
    first_result = []

    def blocking_transport(_body, _timeout):
        active[0] += 1
        max_active[0] = max(max_active[0], active[0])
        entered.set()
        assert release.wait(2)
        active[0] -= 1
        return {}

    first = threading.Thread(target=lambda: first_result.append(
        runtime.run(blocking_transport, {"model": "local-model"}, 2.0)))
    first.start()
    assert entered.wait(1)
    second_calls = []
    with pytest.raises(RuntimeAdmissionError):
        runtime.run(lambda *args: second_calls.append(args), {"model": "local-model"}, 0.02)
    assert second_calls == [] and runtime.state == "cold"
    release.set()
    first.join(2)
    assert not first.is_alive() and first_result == [{}]
    assert max_active == [1] and runtime.state == "ready"


def test_transport_exception_and_late_return_degrade_then_release_lock():
    now = [0.0]
    runtime = LocalModelRuntime("local-model", cooldown_seconds=2, clock=lambda: now[0])
    with pytest.raises(RuntimeError, match="offline"):
        runtime.run(lambda *_: (_ for _ in ()).throw(RuntimeError("offline")),
                    {"model": "local-model"}, 1.0)
    assert runtime.state == "degraded"
    now[0] = 2.0

    def late(*_):
        now[0] += 2.0
        return {}

    with pytest.raises(RuntimeAdmissionError, match="deadline exceeded"):
        runtime.run(late, {"model": "local-model"}, 1.0)
    assert runtime.state == "degraded"
    now[0] += 2.0
    assert runtime.run(lambda *_: {}, {"model": "local-model"}, 1.0) == {}
    assert runtime.state == "ready"

