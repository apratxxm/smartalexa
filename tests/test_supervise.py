from unittest.mock import Mock

import pytest

from assistant import supervise


class Config:
    def __init__(self, **values):
        self.values = values

    def get(self, key):
        return self.values.get(key)


def prepare_start(monkeypatch, config, *, ready=None):
    monkeypatch.setattr(supervise, "_ollama_process", None)
    monkeypatch.setattr(supervise, "load", Mock(return_value=config))
    monkeypatch.setattr(supervise, "_ollama_ready", Mock(side_effect=ready or [False, True]))
    monkeypatch.setattr(supervise.time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(supervise.time, "sleep", lambda _seconds: None)


def test_only_configured_backends_or_enabled_conversation_require_ollama(monkeypatch):
    for config, expected in [
        (Config(**{"intent.backend": "rules", "conversation.enabled": False}), False),
        (Config(**{"intent.backend": "rules", "conversation.enabled": True}), True),
        (Config(**{"conversation.enabled": True}), True),
        (Config(**{"intent.backend": "local_llm", "conversation.enabled": False}), True),
        (Config(**{"intent.backend": "brain", "conversation.enabled": False}), True),
        (Config(**{"intent.backend": "rules", "conversation.enabled": "false"}), False),
    ]:
        prepare_start(monkeypatch, config)
        launch = Mock()
        monkeypatch.setattr(supervise, "_ollama_executable", lambda: "ollama.exe")
        monkeypatch.setattr(supervise.subprocess, "Popen", launch)
        supervise.ensure_local_ollama()
        assert bool(launch.called) is expected


def test_existing_server_is_reused_without_spawning(monkeypatch):
    prepare_start(monkeypatch, Config(**{"intent.backend": "brain"}), ready=[True])
    launch = Mock()
    monkeypatch.setattr(supervise.subprocess, "Popen", launch)
    supervise.ensure_local_ollama()
    launch.assert_not_called()


def test_tracked_live_server_is_not_duplicated(monkeypatch):
    server = Mock()
    server.poll.return_value = None
    prepare_start(monkeypatch, Config(**{"intent.backend": "brain"}), ready=[False])
    monkeypatch.setattr(supervise, "_ollama_process", server)
    launch = Mock()
    monkeypatch.setattr(supervise.subprocess, "Popen", launch)
    supervise.ensure_local_ollama()
    launch.assert_not_called()


@pytest.mark.parametrize("invalid_path", ["relative\\ollama.exe", "", 42])
def test_invalid_explicit_executable_fails_closed_without_path_fallback(monkeypatch, invalid_path):
    prepare_start(monkeypatch, Config(**{"intent.backend": "brain",
                                         "intent.ollama_executable": "relative\\ollama.exe"}),
                  ready=[False])
    which = Mock(return_value="fallback-ollama.exe")
    launch = Mock()
    monkeypatch.setattr(supervise.shutil, "which", which)
    monkeypatch.setattr(supervise.subprocess, "Popen", launch)
    supervise.ensure_local_ollama()
    which.assert_not_called()
    launch.assert_not_called()


def test_missing_absolute_executable_fails_closed_without_path_fallback(monkeypatch, tmp_path):
    path = str(tmp_path / "missing.exe")
    prepare_start(monkeypatch, Config(**{"intent.backend": "brain",
                                         "intent.ollama_executable": path}), ready=[False])
    which = Mock(return_value="fallback-ollama.exe")
    launch = Mock()
    monkeypatch.setattr(supervise.shutil, "which", which)
    monkeypatch.setattr(supervise.subprocess, "Popen", launch)
    supervise.ensure_local_ollama()
    which.assert_not_called()
    launch.assert_not_called()


def test_explicit_executable_is_selected_and_parallelism_is_forced(tmp_path, monkeypatch):
    executable = tmp_path / "portable-ollama.exe"
    executable.touch()
    prepare_start(monkeypatch, Config(**{"intent.backend": "brain",
                                         "intent.ollama_executable": str(executable)}),
                  ready=[False, True])
    monkeypatch.setenv("OLLAMA_NUM_PARALLEL", "8")
    monkeypatch.setenv("OLLAMA_MAX_LOADED_MODELS", "4")
    monkeypatch.setenv("LLAMA_ARG_CACHE_RAM", "8192")
    monkeypatch.setenv("LLAMA_ARG_CTX_CHECKPOINTS", "32")
    launch = Mock()
    monkeypatch.setattr(supervise.subprocess, "Popen", launch)
    supervise.ensure_local_ollama()
    args, kwargs = launch.call_args
    assert args[0] == [str(executable), "serve"]
    assert kwargs["env"]["OLLAMA_HOST"] == "127.0.0.1:11434"
    assert kwargs["env"]["OLLAMA_NUM_PARALLEL"] == "1"
    assert kwargs["env"]["OLLAMA_MAX_LOADED_MODELS"] == "1"
    assert kwargs["env"]["OLLAMA_NO_CLOUD"] == "1"
    assert kwargs["env"]["LLAMA_ARG_CACHE_RAM"] == "256"
    assert kwargs["env"]["LLAMA_ARG_CTX_CHECKPOINTS"] == "0"
    assert supervise.os.environ["LLAMA_ARG_CACHE_RAM"] == "8192"
    assert supervise.os.environ["LLAMA_ARG_CTX_CHECKPOINTS"] == "32"

