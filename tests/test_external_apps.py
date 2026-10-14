"""External app launch policy tests; no native apps are started."""
from pathlib import PureWindowsPath
import subprocess
from unittest.mock import Mock

import pytest

from assistant import external_apps


def test_external_creation_flags_preserve_unmanaged_flags_and_breakaway_only_when_marked(monkeypatch):
    base = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    monkeypatch.setattr(external_apps.os, "name", "nt")
    monkeypatch.delenv("MEMOIR_MANAGED_PROCESS_TREE", raising=False)
    assert external_apps.external_creation_flags() == base
    monkeypatch.setenv("MEMOIR_MANAGED_PROCESS_TREE", "1")
    assert external_apps.external_creation_flags() == base | 0x01000000
    monkeypatch.setenv("MEMOIR_MANAGED_PROCESS_TREE", "true")
    assert external_apps.external_creation_flags() == base


def test_unmanaged_target_uses_exact_legacy_startfile_behavior(monkeypatch):
    opened = Mock()
    monkeypatch.setattr(external_apps.os, "name", "posix")
    monkeypatch.delenv("MEMOIR_MANAGED_PROCESS_TREE", raising=False)
    monkeypatch.setattr(external_apps.os, "startfile", opened, raising=False)
    launch = Mock()
    monkeypatch.setattr(external_apps.subprocess, "Popen", launch)

    external_apps.open_external_target("spotify:track:abc")

    opened.assert_called_once_with("spotify:track:abc")
    launch.assert_not_called()


def test_managed_target_uses_literal_argv_and_breakaway_flags(monkeypatch):
    explorer = PureWindowsPath("C:/Windows/explorer.exe")
    windows = PureWindowsPath("C:/Windows")
    monkeypatch.setattr(external_apps.os, "name", "nt")
    monkeypatch.setenv("MEMOIR_MANAGED_PROCESS_TREE", "1")
    monkeypatch.setattr(external_apps, "_trusted_explorer", lambda: (explorer, windows))
    launch = Mock()
    monkeypatch.setattr(external_apps.subprocess, "Popen", launch)

    target = "spotify:track:abc?x&y"
    external_apps.open_external_target(target)

    assert launch.call_args.args[0] == [str(explorer), target]
    assert launch.call_args.kwargs["creationflags"] == getattr(subprocess, "CREATE_NO_WINDOW", 0) | 0x01000000
    assert launch.call_args.kwargs["cwd"] == str(windows)
    assert launch.call_args.kwargs["stdin"] is subprocess.DEVNULL


def test_managed_launch_failure_is_not_retried_through_startfile(monkeypatch):
    explorer = PureWindowsPath("C:/Windows/explorer.exe")
    windows = PureWindowsPath("C:/Windows")
    monkeypatch.setattr(external_apps.os, "name", "nt")
    monkeypatch.setenv("MEMOIR_MANAGED_PROCESS_TREE", "1")
    monkeypatch.setattr(external_apps, "_trusted_explorer", lambda: (explorer, windows))
    launch = Mock(side_effect=OSError("breakaway denied"))
    opened = Mock()
    monkeypatch.setattr(external_apps.subprocess, "Popen", launch)
    monkeypatch.setattr(external_apps.os, "startfile", opened, raising=False)

    with pytest.raises(OSError, match="breakaway denied"):
        external_apps.open_external_target("spotify:")

    launch.assert_called_once()
    opened.assert_not_called()


def test_relative_windows_root_is_rejected_before_any_process_launch(monkeypatch):
    monkeypatch.setattr(external_apps.os, "name", "nt")
    monkeypatch.setenv("MEMOIR_MANAGED_PROCESS_TREE", "1")
    monkeypatch.setenv("WINDIR", "Windows")
    monkeypatch.delenv("SystemRoot", raising=False)
    class RelativeRoot:
        def is_absolute(self):
            return False
        def resolve(self, **_kwargs):
            raise AssertionError("Relative Windows roots must be rejected before resolution")
    monkeypatch.setattr(external_apps, "Path", lambda _value: RelativeRoot())
    launch = Mock()
    monkeypatch.setattr(external_apps.subprocess, "Popen", launch)

    with pytest.raises(OSError, match="could not be verified"):
        external_apps.open_external_target("spotify:")

    launch.assert_not_called()


@pytest.mark.parametrize("target", ["", "spotify:\ncommand", "x" * 2049])
def test_external_target_rejects_unbounded_or_controlled_values(target):
    with pytest.raises(ValueError):
        external_apps.open_external_target(target)
