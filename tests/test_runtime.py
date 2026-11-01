"""Exercise the actual Windows kernel guard, without starting the assistant."""
import sys
import json
from unittest.mock import Mock

import pytest

from assistant.runtime import (AlreadyRunning, active_collection_pauses,
                               acknowledge_collection_pauses,
                               collection_pause_acknowledged,
                               create_collection_pause, pause_requested,
                               release_collection_pause, single_instance)
from assistant import supervise


def test_pause_missing_and_manual_flags(tmp_path):
    flag = tmp_path / "pause"
    assert not pause_requested(flag, now=100)
    for text in ("", "manual", "{}", "[]", "x" * 4097):
        flag.write_text(text)
        assert pause_requested(flag, now=100)


def test_probe_pause_expires_without_deleting_flag(tmp_path):
    flag = tmp_path / "pause"
    raw = json.dumps(dict(schema="memoir.probe-pause.v1", owner="probe-123",
                          issued_at=100, expires_at=400))
    flag.write_text(raw)
    assert pause_requested(flag, now=399)
    assert not pause_requested(flag, now=400)
    assert flag.read_text() == raw
    flag.write_text("manual replacement")
    assert pause_requested(flag, now=500)


@pytest.mark.parametrize("patch", [
    {"owner": ""}, {"schema": "unknown"}, {"issued_at": 101},
    {"issued_at": False}, {"expires_at": float("nan")},
    {"expires_at": 1000.1}, {"expires_at": 99}, {"expires_at": "101"},
])
def test_invalid_probe_pause_stays_paused(tmp_path, patch):
    flag = tmp_path / "pause"
    lease = dict(schema="memoir.probe-pause.v1", owner="probe-123",
                 issued_at=100, expires_at=400)
    lease.update(patch)
    flag.write_text(json.dumps(lease))
    assert pause_requested(flag, now=100)


def test_unreadable_pause_stays_paused(tmp_path):
    flag = tmp_path / "pause"
    flag.write_bytes(b"\xff")
    assert pause_requested(flag, now=100)


def test_collection_pause_lifecycle_is_private_bounded_and_exclusive(tmp_path):
    directory = tmp_path / "collection_pauses"
    capture_id = "a" * 32
    create_collection_pause(directory, capture_id, now=100)
    assert pause_requested(tmp_path / "missing", now=100, lease_dir=directory)
    assert len(active_collection_pauses(directory, now=100)) == 1
    assert not collection_pause_acknowledged(directory, capture_id, now=100)
    acknowledge_collection_pauses(directory, now=100)
    assert collection_pause_acknowledged(directory, capture_id, now=100)
    with pytest.raises(FileExistsError):
        create_collection_pause(directory, capture_id, now=101)
    assert release_collection_pause(directory, capture_id)
    assert not pause_requested(tmp_path / "missing", now=101, lease_dir=directory)


@pytest.mark.parametrize("patch", [
    {"issued_at": 101}, {"expires_at": 131}, {"expires_at": 99},
    {"issued_at": True}, {"expires_at": float("inf")},
])
def test_collection_invalid_and_expired_leases_are_ignored(tmp_path, patch):
    directory = tmp_path / "collection_pauses"
    directory.mkdir()
    capture_id = "b" * 32
    value = {"schema": "memoir.collection-pause.v1", "capture_id": capture_id,
             "issued_at": 100, "expires_at": 130}
    value.update(patch)
    (directory / f"{capture_id}.json").write_text(json.dumps(value))
    (directory / f"{capture_id}.ack").write_text(capture_id)
    assert not collection_pause_acknowledged(directory, capture_id, now=100)
    assert not active_collection_pauses(directory, now=100)


def test_collection_pause_rejects_bad_ids_and_symlink_leases(tmp_path):
    directory = tmp_path / "collection_pauses"
    directory.mkdir()
    with pytest.raises(ValueError):
        create_collection_pause(directory, "../" + "c" * 30, now=100)
    capture_id = "c" * 32
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps({"schema": "memoir.collection-pause.v1",
                                   "capture_id": capture_id, "issued_at": 100,
                                   "expires_at": 120}))
    try:
        (directory / f"{capture_id}.json").symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    assert not active_collection_pauses(directory, now=101)
    assert not collection_pause_acknowledged(directory, capture_id, now=101)


def test_collection_release_never_removes_manual_pause(tmp_path):
    directory = tmp_path / "collection_pauses"
    manual = tmp_path / ".listening_paused"
    manual.write_text("manual")
    capture_id = "d" * 32
    create_collection_pause(directory, capture_id, now=100)
    acknowledge_collection_pauses(directory, now=100)
    assert release_collection_pause(directory, capture_id)
    assert manual.read_text() == "manual"
    assert pause_requested(manual, now=101)


def test_collection_ack_is_bound_to_current_lease_expiry(tmp_path):
    directory = tmp_path / "collection_pauses"
    capture_id = "f" * 32
    create_collection_pause(directory, capture_id, now=100)
    ack = {"schema": "memoir.collection-pause.v1", "capture_id": capture_id,
           "issued_at": 100, "expires_at": 129}
    (directory / f"{capture_id}.ack").write_text(json.dumps(ack))
    assert not collection_pause_acknowledged(directory, capture_id, now=101)
    assert not release_collection_pause(directory, capture_id)


def test_collection_cancel_before_ack_releases_owned_lease(tmp_path):
    directory = tmp_path / "collection_pauses"
    capture_id = "1" * 32
    create_collection_pause(directory, capture_id)
    assert not collection_pause_acknowledged(directory, capture_id)
    assert release_collection_pause(directory, capture_id)
    assert not (directory / f"{capture_id}.json").exists()
    assert not (directory / f"{capture_id}.ack").exists()


def test_collection_namespace_overflow_fails_closed_without_ack(tmp_path):
    directory = tmp_path / "collection_pauses"
    for index in range(33):
        create_collection_pause(directory, f"{index:032x}", now=100)
    with pytest.raises(ValueError, match="too many"):
        active_collection_pauses(directory, now=100)
    assert pause_requested(tmp_path / "missing", now=100, lease_dir=directory)
    with pytest.raises(ValueError, match="too many"):
        acknowledge_collection_pauses(directory, now=100)
    assert not list(directory.glob("*.ack"))


@pytest.fixture(autouse=True)
def reset_supervisor_server_handle(monkeypatch):
    monkeypatch.setattr(supervise, "_ollama_process", None)
    monkeypatch.setattr(supervise, "_review_process", None)


def test_review_health_probe_is_bounded_and_bypasses_proxy(monkeypatch):
    response = Mock(status=200)
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    response.read.return_value = json.dumps(supervise.REVIEW_SERVICE_ID).encode()
    opener = Mock()
    opener.open.return_value = response
    build_opener = Mock(return_value=opener)
    monkeypatch.setattr(supervise.urllib.request, "build_opener", build_opener)

    assert supervise._review_health()

    handler = build_opener.call_args.args[0]
    assert isinstance(handler, supervise.urllib.request.ProxyHandler)
    assert handler.proxies == {}
    opener.open.assert_called_once_with(supervise.REVIEW_HEALTH, timeout=0.25)
    response.read.assert_called_once_with(1025)


def test_review_server_reuses_expected_listener(monkeypatch):
    launch = Mock()
    monkeypatch.setattr(supervise, "_review_health", lambda: True)
    monkeypatch.setattr(supervise, "_review_port_open", Mock())
    monkeypatch.setattr(supervise.subprocess, "Popen", launch)

    supervise.ensure_review_server()

    launch.assert_not_called()
    supervise._review_port_open.assert_not_called()


def test_review_server_reports_unrelated_listener_without_launch(monkeypatch, caplog):
    launch = Mock()
    monkeypatch.setattr(supervise, "_review_health", lambda: False)
    monkeypatch.setattr(supervise, "_review_port_open", lambda: True)
    monkeypatch.setattr(supervise.subprocess, "Popen", launch)

    supervise.ensure_review_server()

    launch.assert_not_called()
    assert "Port 8767 is occupied by an unexpected service" in caplog.text


def test_review_server_keeps_slow_starting_process_without_duplicates(monkeypatch):
    server = Mock()
    server.poll.return_value = None
    launch = Mock(return_value=server)
    clock = [0.0]
    monkeypatch.setattr(supervise, "_review_health", lambda: False)
    monkeypatch.setattr(supervise, "_review_port_open", lambda: False)
    monkeypatch.setattr(supervise.subprocess, "Popen", launch)
    monkeypatch.setattr(supervise.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(supervise.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))

    supervise.ensure_review_server()
    supervise.ensure_review_server()

    launch.assert_called_once()
    args, kwargs = launch.call_args
    assert args[0] == [sys.executable, "-m", "assistant.experimental.wake_train.review_server",
                       "--port", "8767", "--strict-port"]
    assert kwargs["stdin"] is supervise.subprocess.DEVNULL
    assert kwargs["stdout"] is supervise.subprocess.DEVNULL
    assert kwargs["stderr"] is supervise.subprocess.DEVNULL


def test_review_server_restarts_owned_process_after_exit(monkeypatch):
    old_server = Mock()
    old_server.poll.return_value = 1
    launch = Mock()
    monkeypatch.setattr(supervise, "_review_process", old_server)
    monkeypatch.setattr(supervise, "_review_health", Mock(side_effect=[False, True]))
    monkeypatch.setattr(supervise, "_review_port_open", lambda: False)
    monkeypatch.setattr(supervise.subprocess, "Popen", launch)

    supervise.ensure_review_server()

    launch.assert_called_once()
    assert supervise._review_process is launch.return_value


@pytest.mark.skipif(sys.platform != "win32", reason="Windows kernel mutex")
def test_duplicate_instance_rejected_and_exit_releases(tmp_path):
    with single_instance(tmp_path):
        with pytest.raises(AlreadyRunning):
            with single_instance(tmp_path):
                pytest.fail("duplicate acquired")
    with single_instance(tmp_path):
        pass


@pytest.mark.skipif(sys.platform != "win32", reason="Windows kernel mutex")
def test_exception_releases_instance(tmp_path):
    with pytest.raises(ValueError):
        with single_instance(tmp_path):
            raise ValueError("failed startup")
    with single_instance(tmp_path):
        pass


@pytest.mark.parametrize("configured", [False, True])
def test_supervisor_starts_local_ollama_only_when_needed(monkeypatch, configured):
    process = Mock()
    for name, value in (("OLLAMA_NUM_PARALLEL", "2"), ("OLLAMA_MAX_LOADED_MODELS", "3")):
        if configured:
            monkeypatch.setenv(name, value)
        else:
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OLLAMA_NO_CLOUD", "0")
    monkeypatch.setattr(supervise, "load", lambda: {"intent.backend": "brain"})
    monkeypatch.setattr(supervise, "_ollama_ready", Mock(side_effect=[False, True]))
    monkeypatch.setattr(supervise, "_ollama_executable", lambda: r"C:\Ollama\ollama.exe")
    monkeypatch.setattr(supervise.subprocess, "Popen", process)

    supervise.ensure_local_ollama()

    args, kwargs = process.call_args
    assert args[0] == [r"C:\Ollama\ollama.exe", "serve"]
    assert kwargs["env"]["OLLAMA_HOST"] == "127.0.0.1:11434"
    # The supervisor deliberately overrides inherited parallelism to prevent
    # duplicate model/cache allocations on this laptop.
    assert kwargs["env"]["OLLAMA_NUM_PARALLEL"] == "1"
    assert kwargs["env"]["OLLAMA_MAX_LOADED_MODELS"] == "1"
    assert kwargs["env"]["OLLAMA_NO_CLOUD"] == "1"
    assert kwargs["stdout"] is supervise.subprocess.DEVNULL


def test_supervisor_does_not_launch_duplicate_or_unneeded_server(monkeypatch):
    process = Mock()
    monkeypatch.setattr(supervise.subprocess, "Popen", process)
    monkeypatch.setattr(supervise, "_ollama_ready", lambda: True)
    monkeypatch.setattr(supervise, "load", lambda: {"intent.backend": "brain"})
    supervise.ensure_local_ollama()
    monkeypatch.setattr(supervise, "load", lambda: Mock(get=lambda key: "rules"))
    supervise.ensure_local_ollama()
    process.assert_not_called()


def test_supervisor_gives_up_waiting_and_starts_assistant(monkeypatch):
    clock = [0.0]
    process = Mock()
    monkeypatch.setattr(supervise, "load", lambda: {"intent.backend": "brain"})
    monkeypatch.setattr(supervise, "_ollama_ready", lambda: False)
    monkeypatch.setattr(supervise, "_ollama_executable", lambda: "ollama.exe")
    monkeypatch.setattr(supervise.subprocess, "Popen", process)
    monkeypatch.setattr(supervise.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(supervise.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))

    supervise.ensure_local_ollama()

    process.assert_called_once()
    assert clock[0] == supervise.OLLAMA_READY_SECONDS


def test_supervisor_keeps_slow_starting_server_without_duplicates(monkeypatch):
    server = Mock()
    server.poll.return_value = None
    launch = Mock(return_value=server)
    clock = [0.0]
    monkeypatch.setattr(supervise, "load", lambda: {"intent.backend": "local_llm"})
    monkeypatch.setattr(supervise, "_ollama_ready", lambda: False)
    monkeypatch.setattr(supervise, "_ollama_executable", lambda: "ollama.exe")
    monkeypatch.setattr(supervise.subprocess, "Popen", launch)
    monkeypatch.setattr(supervise.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(supervise.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    supervise.ensure_local_ollama()
    supervise.ensure_local_ollama()
    launch.assert_called_once()
    assert clock[0] == supervise.OLLAMA_READY_SECONDS


def test_supervisor_replaces_exited_server(monkeypatch):
    old_server = Mock()
    old_server.poll.return_value = 1
    launch = Mock()
    monkeypatch.setattr(supervise, "_ollama_process", old_server)
    monkeypatch.setattr(supervise, "load", lambda: {"intent.backend": "brain"})
    monkeypatch.setattr(supervise, "_ollama_ready", Mock(side_effect=[False, True]))
    monkeypatch.setattr(supervise, "_ollama_executable", lambda: "ollama.exe")
    monkeypatch.setattr(supervise.subprocess, "Popen", launch)
    supervise.ensure_local_ollama()
    launch.assert_called_once()
    assert supervise._ollama_process is launch.return_value


def test_supervisor_checks_dependencies_while_same_assistant_runs(monkeypatch):
    monkeypatch.setattr(supervise, "install_process_lifetime", Mock())
    child = Mock()
    child.wait.side_effect = [supervise.subprocess.TimeoutExpired("assistant", 30), 0]
    launch = Mock(return_value=child)
    ensure = Mock()
    monkeypatch.setattr(supervise, "ensure_review_server", Mock())
    monkeypatch.setattr(supervise, "ensure_local_ollama", ensure)
    monkeypatch.setattr(supervise.subprocess, "Popen", launch)
    monkeypatch.setattr(supervise.logging, "basicConfig", Mock())
    supervise.main()
    launch.assert_called_once()
    assert ensure.call_count == 2
    assert child.wait.call_count == 2
    child.wait.assert_called_with(timeout=supervise.DEPENDENCY_CHECK_SECONDS)


def test_supervisor_preserves_failure_backoff(monkeypatch):
    monkeypatch.setattr(supervise, "install_process_lifetime", Mock())
    failed, stopped = Mock(), Mock()
    failed.wait.return_value = 1
    stopped.wait.return_value = 0
    launch = Mock(side_effect=[failed, stopped])
    sleep = Mock()
    monkeypatch.setattr(supervise, "ensure_review_server", Mock())
    monkeypatch.setattr(supervise, "ensure_local_ollama", Mock())
    monkeypatch.setattr(supervise.subprocess, "Popen", launch)
    monkeypatch.setattr(supervise.logging, "basicConfig", Mock())
    monkeypatch.setattr(supervise.time, "monotonic", Mock(side_effect=[0, 1, 11, 12]))
    monkeypatch.setattr(supervise.time, "sleep", sleep)
    supervise.main()
    assert launch.call_count == 2
    sleep.assert_called_once_with(10.0)
