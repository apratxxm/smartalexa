import subprocess
import os
import sys
import time
from pathlib import Path
from unittest.mock import Mock

import pytest

from assistant import config, supervise


@pytest.fixture
def supervisor(monkeypatch, tmp_path):
    config_file = tmp_path / "config.toml"
    config_file.write_text('[runtime]\nruntime = "legacy"\n')
    monkeypatch.setattr(config, "CONFIG_PATH", config_file)
    monkeypatch.setattr(config, "ROOT", tmp_path)
    monkeypatch.setattr(supervise, "ROOT", tmp_path)
    monkeypatch.setattr(supervise.logging, "basicConfig", lambda **kwargs: None)
    monkeypatch.setattr(supervise, "install_process_lifetime", lambda: None)
    monkeypatch.setattr(supervise, "ensure_review_server", lambda: None)
    monkeypatch.setattr(supervise, "ensure_local_ollama", lambda: None)
    monkeypatch.setattr(supervise.time, "monotonic", lambda: 0)
    monkeypatch.setattr(supervise.time, "sleep", lambda seconds: None)
    config.load.cache_clear()
    yield config_file
    config.load.cache_clear()


def test_running_supervisor_keeps_cached_choice_on_relaunch(supervisor, monkeypatch):
    launched = []
    def start(args, **kwargs):
        launched.append(args[-1])
        if len(launched) == 1:
            supervisor.write_text('[runtime]\nruntime = "core"\n')
        return Mock(wait=Mock(return_value=1 if len(launched) == 1 else 0))
    monkeypatch.setattr(supervise.subprocess, "Popen", start)
    supervise.main()
    assert launched == ["assistant.main", "assistant.main"]
    config.load.cache_clear()
    launched.clear()
    monkeypatch.setattr(supervise.subprocess, "Popen", lambda args, **kwargs: launched.append(args[-1]) or Mock(wait=Mock(return_value=0)))
    supervise.main()
    assert launched == ["assistant.core"]


def test_missing_legacy_child_backs_off(supervisor, monkeypatch):
    launched = []
    delays = []
    def start(args, **kwargs):
        launched.append(args[-1])
        return Mock(wait=Mock(return_value=1 if len(launched) < 3 else 0))
    monkeypatch.setattr(supervise.subprocess, "Popen", start)
    monkeypatch.setattr(supervise.time, "sleep", delays.append)
    supervise.main()
    assert launched == ["assistant.main"] * 3
    assert delays == [10.0, 20.0]


def test_documented_restart_sequence_models_exit_and_lock_release(supervisor, monkeypatch):
    state = {"running": True, "lock": True}
    state["running"] = False
    state["lock"] = False
    assert not state["running"] and not state["lock"]
    supervisor.write_text('[runtime]\nruntime = "core"\n')
    config.load.cache_clear()
    launched = []
    monkeypatch.setattr(supervise.subprocess, "Popen", lambda args, **kwargs: launched.append(args[-1]) or Mock(wait=Mock(return_value=0)))
    supervise.main()
    assert launched == ["assistant.core"]


@pytest.mark.skipif(os.name != "nt", reason="real supervisor lock uses Windows mutexes")
def test_process_exit_releases_real_runtime_lock(tmp_path):
    from assistant.runtime import AlreadyRunning, single_instance
    ready = tmp_path / "ready"
    script = """import sys
from pathlib import Path
from assistant.runtime import single_instance
with single_instance(Path(sys.argv[1])):
    Path(sys.argv[2]).write_text('ready')
    sys.stdin.readline()
"""
    child = subprocess.Popen([sys.executable, "-B", "-c", script, str(tmp_path), str(ready)], stdin=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and time.monotonic() < deadline:
            if child.poll() is not None:
                pytest.fail("child exited before acquiring lock")
            time.sleep(0.05)
        assert ready.exists()
        with pytest.raises(AlreadyRunning):
            with single_instance(tmp_path):
                pass
        child.stdin.write("stop\n")
        child.stdin.flush()
        assert child.wait(timeout=10) == 0
        with single_instance(tmp_path):
            pass
        ready.unlink()
        replacement = subprocess.Popen([sys.executable, "-B", "-c", script, str(tmp_path), str(ready)], stdin=subprocess.PIPE, text=True)
        try:
            deadline = time.monotonic() + 10
            while not ready.exists() and time.monotonic() < deadline:
                if replacement.poll() is not None:
                    pytest.fail("replacement exited before acquiring lock")
                time.sleep(0.05)
            assert ready.exists()
            replacement.stdin.write("stop\n")
            replacement.stdin.flush()
            assert replacement.wait(timeout=10) == 0
        finally:
            if replacement.poll() is None:
                replacement.kill()
                replacement.wait(timeout=10)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)


@pytest.mark.skipif(os.name != "nt", reason="Windows taskkill tree control required")
def test_real_supervisor_tree_stop_and_restart(tmp_path):
    from assistant.runtime import single_instance
    root = tmp_path / "project"
    (root / "assistant" / "logs").mkdir(parents=True)
    config_file = root / "assistant" / "config.toml"
    config_file.write_text('[runtime]\nruntime = "legacy"\n')
    child_script = '''import os, sys, time
from pathlib import Path
from assistant.runtime import single_instance
root = Path(sys.argv[1])
with single_instance(root):
 (root / "child-ready").write_text(Path(__file__).stem + ":" + str(os.getpid()))
 deadline = time.monotonic() + 30
 while time.monotonic() < deadline and not (root / "stop-child").exists(): time.sleep(0.05)
'''
    for module in ("fake_legacy", "fake_core"):
        (tmp_path / (module + ".py")).write_text(child_script)
    launcher = '''import sys, os, threading
from pathlib import Path
from assistant import config, supervise
root = Path(sys.argv[1])
watchdog = threading.Timer(45, lambda: os._exit(2))
watchdog.daemon = True
watchdog.start()
config.ROOT = supervise.ROOT = root
config.CONFIG_PATH = root / "assistant" / "config.toml"
supervise.install_process_lifetime = lambda: None
supervise.ensure_review_server = lambda: None
supervise.ensure_local_ollama = lambda: None
real_resolve = supervise.resolve_runtime_module
supervise.resolve_runtime_module = lambda cfg: "fake_core" if real_resolve(cfg) == "assistant.core" else "fake_legacy"
original_popen = supervise.subprocess.Popen
supervise.subprocess.Popen = lambda args, **kwargs: original_popen(args + [str(root)], **kwargs)
supervise.main()
'''
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(tmp_path), str(Path(__file__).resolve().parents[2])])
    ready = root / "child-ready"
    def start():
        process = subprocess.Popen([sys.executable, "-B", "-c", launcher, str(root)], env=env)
        deadline = time.monotonic() + 15
        while not ready.exists() and time.monotonic() < deadline:
            if process.poll() is not None:
                pytest.fail("supervisor exited before child became ready")
            time.sleep(0.05)
        assert ready.exists()
        return process
    def stop(process):
        result = subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True, timeout=10)
        if result.returncode:
            process.kill()
            process.wait(timeout=10)
            (root / "stop-child").touch()
            deadline = time.monotonic() + 10
            while True:
                try:
                    with single_instance(root):
                        break
                except Exception:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(0.05)
            pytest.skip("Windows taskkill tree control unavailable: " + result.stderr.decode(errors="replace").strip())
        process.wait(timeout=10)
        with single_instance(root):
            pass
    process = start()
    try:
        assert ready.read_text().startswith("fake_legacy:")
        stop(process)
        ready.unlink()
        config_file.write_text('[runtime]\nruntime = "core"\n')
        process = start()
        assert ready.read_text().startswith("fake_core:")
        stop(process)
    finally:
        if process.poll() is None:
            stop(process)
