"""Exercise the actual Windows kernel guard, without starting the assistant."""
import sys
from unittest.mock import Mock

import pytest

from assistant.runtime import AlreadyRunning, single_instance
from assistant import supervise


@pytest.fixture(autouse=True)
def reset_supervisor_server_handle(monkeypatch):
    monkeypatch.setattr(supervise, "_ollama_process", None)


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
    monkeypatch.setattr(supervise, "load", lambda: Mock(get=lambda key: "brain"))
    monkeypatch.setattr(supervise, "_ollama_ready", Mock(side_effect=[False, True]))
    monkeypatch.setattr(supervise, "_ollama_executable", lambda: r"C:\Ollama\ollama.exe")
    monkeypatch.setattr(supervise.subprocess, "Popen", process)

    supervise.ensure_local_ollama()

    args, kwargs = process.call_args
    assert args[0] == [r"C:\Ollama\ollama.exe", "serve"]
    assert kwargs["env"]["OLLAMA_HOST"] == "127.0.0.1:11434"
    assert kwargs["env"]["OLLAMA_NUM_PARALLEL"] == ("2" if configured else "1")
    assert kwargs["env"]["OLLAMA_MAX_LOADED_MODELS"] == ("3" if configured else "1")
    assert kwargs["env"]["OLLAMA_NO_CLOUD"] == "1"
    assert kwargs["stdout"] is supervise.subprocess.DEVNULL


def test_supervisor_does_not_launch_duplicate_or_unneeded_server(monkeypatch):
    process = Mock()
    monkeypatch.setattr(supervise.subprocess, "Popen", process)
    monkeypatch.setattr(supervise, "_ollama_ready", lambda: True)
    monkeypatch.setattr(supervise, "load", lambda: Mock(get=lambda key: "brain"))
    supervise.ensure_local_ollama()
    monkeypatch.setattr(supervise, "load", lambda: Mock(get=lambda key: "rules"))
    supervise.ensure_local_ollama()
    process.assert_not_called()


def test_supervisor_gives_up_waiting_and_starts_assistant(monkeypatch):
    clock = [0.0]
    process = Mock()
    monkeypatch.setattr(supervise, "load", lambda: Mock(get=lambda key: "brain"))
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
    monkeypatch.setattr(supervise, "load", lambda: Mock(get=lambda key: "local_llm"))
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
    monkeypatch.setattr(supervise, "load", lambda: Mock(get=lambda key: "brain"))
    monkeypatch.setattr(supervise, "_ollama_ready", Mock(side_effect=[False, True]))
    monkeypatch.setattr(supervise, "_ollama_executable", lambda: "ollama.exe")
    monkeypatch.setattr(supervise.subprocess, "Popen", launch)
    supervise.ensure_local_ollama()
    launch.assert_called_once()
    assert supervise._ollama_process is launch.return_value


def test_supervisor_checks_dependencies_while_same_assistant_runs(monkeypatch):
    child = Mock()
    child.wait.side_effect = [supervise.subprocess.TimeoutExpired("assistant", 30), 0]
    launch = Mock(return_value=child)
    ensure = Mock()
    monkeypatch.setattr(supervise, "ensure_local_ollama", ensure)
    monkeypatch.setattr(supervise.subprocess, "Popen", launch)
    monkeypatch.setattr(supervise.logging, "basicConfig", Mock())
    supervise.main()
    launch.assert_called_once()
    assert ensure.call_count == 2
    assert child.wait.call_count == 2
    child.wait.assert_called_with(timeout=supervise.DEPENDENCY_CHECK_SECONDS)


def test_supervisor_preserves_failure_backoff(monkeypatch):
    failed, stopped = Mock(), Mock()
    failed.wait.return_value = 1
    stopped.wait.return_value = 0
    launch = Mock(side_effect=[failed, stopped])
    sleep = Mock()
    monkeypatch.setattr(supervise, "ensure_local_ollama", Mock())
    monkeypatch.setattr(supervise.subprocess, "Popen", launch)
    monkeypatch.setattr(supervise.logging, "basicConfig", Mock())
    monkeypatch.setattr(supervise.time, "monotonic", Mock(side_effect=[0, 1, 11, 12]))
    monkeypatch.setattr(supervise.time, "sleep", sleep)
    supervise.main()
    assert launch.call_count == 2
    sleep.assert_called_once_with(10.0)
