"""No native launches: app identities, arguments and browser views are mocked."""
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from assistant import computer_tools as tools


@pytest.fixture(autouse=True)
def clean_catalog(monkeypatch):
    monkeypatch.setattr(tools, "_catalog", ())
    monkeypatch.setattr(tools, "_loaded_at", 0.0)
    monkeypatch.setattr(tools, "_attempted_at", 0.0)
    monkeypatch.setattr(tools, "_warming", False)


def fresh_catalog(monkeypatch, rows):
    monkeypatch.setattr(tools, "_catalog", tuple(rows))
    monkeypatch.setattr(tools, "_loaded_at", 99.0)
    monkeypatch.setattr(tools.time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(tools, "_windows_binary", lambda suffix: Path("C:/Windows") / suffix)


@pytest.mark.parametrize("heard", ["Open Notepad", "Could you pull up Opera", "Look up logs in Opera",
    "I'd like to see the album Blue Lights", "Restart the current album"])
def test_desktop_admission_does_not_authorize_execution(heard):
    assert tools.desktop_domain(heard)
    assert not tools.desktop_domain("Open the Codex chat to check logs")
    assert not tools.desktop_domain("He said open Notepad")


def test_open_app_uses_unique_os_catalogue_id_not_heard_shell_or_path(monkeypatch):
    fresh_catalog(monkeypatch, [("Opera GX Browser", "OperaSoftware.OperaGXWebBrowser.1720893516"),
                                ("Notepad", "Microsoft.WindowsNotepad_8wekyb3d8bbwe!App")])
    launch = Mock()
    monkeypatch.setattr(tools.subprocess, "Popen", launch)
    assert tools.open_app("Opera") == "Requested Opera GX Browser."
    argv = launch.call_args.args[0]
    assert argv == [str(Path("C:/Windows/explorer.exe")),
                    "shell:AppsFolder\\OperaSoftware.OperaGXWebBrowser.1720893516"]
    assert "shell" not in launch.call_args.kwargs
    for query in ["cmd.exe /c delete", "C:/Windows/notepad.exe", "Unknown App"]:
        with pytest.raises(tools.ComputerControlError):
            tools.open_app(query)
    assert launch.call_count == 1


def test_ambiguous_installed_app_never_guesses(monkeypatch):
    fresh_catalog(monkeypatch, [("Opera Browser", "OperaStable"), ("Opera GX Browser", "OperaGX")])
    launch = Mock()
    monkeypatch.setattr(tools.subprocess, "Popen", launch)
    with pytest.raises(tools.ComputerControlError, match="ambiguous"):
        tools.open_app("Opera")
    launch.assert_not_called()


def test_catalogue_display_names_admit_novel_app_request_without_launching(monkeypatch):
    fresh_catalog(monkeypatch, [("Notepad", "NotepadApp")])
    assert tools.desktop_domain("Can you get Notepad running")
    assert tools.desktop_domain("Start the Notepad app")
    assert not tools.desktop_domain("He said get Notepad running")
    assert not tools.desktop_domain("Get some random phrase running")


def test_cold_or_stale_catalogue_never_blocks_voice_on_powershell(monkeypatch):
    warm = Mock()
    monkeypatch.setattr(tools, "warm_catalog", warm)
    monkeypatch.setattr(tools.subprocess, "run", Mock(side_effect=AssertionError("Foreground lookup")))
    with pytest.raises(tools.ComputerControlError, match="warming"):
        tools.open_app("Notepad")
    warm.assert_called_once()
    tools.subprocess.run.assert_not_called()


def test_expired_catalogue_reuses_validated_identity_while_refreshing(monkeypatch):
    fresh_catalog(monkeypatch, [("Notepad", "NotepadApp")])
    monkeypatch.setattr(tools.time, "monotonic", lambda: 500.0)
    warm, launch = Mock(), Mock()
    monkeypatch.setattr(tools, "warm_catalog", warm)
    monkeypatch.setattr(tools.subprocess, "Popen", launch)
    monkeypatch.setattr(tools.subprocess, "run", Mock(side_effect=AssertionError("No foreground PowerShell")))
    assert tools.open_app("Notepad") == "Requested Notepad."
    warm.assert_called_once()
    launch.assert_called_once()
    assert launch.call_args.args[0][1] == "shell:AppsFolder\\NotepadApp"
    tools.subprocess.run.assert_not_called()


def test_catalogue_is_fixed_bounded_and_discards_path_or_command_ids(monkeypatch):
    monkeypatch.setattr(tools, "_windows_binary", lambda _: Path("powershell.exe"))
    rows = [{"Name": "Notepad", "AppID": "Microsoft.WindowsNotepad!App"},
            {"Name": "Evil", "AppID": "C:\\evil.exe"},
            {"Name": "Shell", "AppID": "cmd.exe & delete"},
            {"Name": "Bad\nLabel", "AppID": "bad"},
            {"Name": "Notepad", "AppID": "Microsoft.WindowsNotepad!App"}]
    run = Mock(return_value=SimpleNamespace(returncode=0, stdout=json.dumps(rows)))
    monkeypatch.setattr(tools.subprocess, "run", run)
    assert tools._load_catalog() == (("Notepad", "Microsoft.WindowsNotepad!App"),)
    assert run.call_args.args[0][-1] == tools._CATALOG_SCRIPT
    assert "-First 256" in tools._CATALOG_SCRIPT and run.call_args.kwargs["timeout"] == 2.5
    assert "shell" not in run.call_args.kwargs


def test_catalogue_warm_starts_one_daemon_and_retries_only_after_cooldown(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(tools.time, "monotonic", lambda: now[0])
    threads = []
    class Thread:
        def __init__(self, **kwargs):
            self.options = kwargs
            threads.append(self)
        def start(self):
            pass
    monkeypatch.setattr(tools.threading, "Thread", Thread)
    monkeypatch.setattr(tools, "_load_catalog", Mock(side_effect=OSError("unavailable")))
    tools.warm_catalog()
    tools.warm_catalog()
    assert len(threads) == 1 and threads[0].options["daemon"]
    threads[0].options["target"]()
    tools.warm_catalog()
    assert len(threads) == 1
    now[0] += 16
    tools.warm_catalog()
    assert len(threads) == 2


def test_opera_search_is_one_encoded_https_new_tab_not_arbitrary_shell(monkeypatch):
    monkeypatch.setattr(tools, "_opera_binary", lambda: Path("trusted/opera.exe"))
    launch = Mock()
    monkeypatch.setattr(tools.subprocess, "Popen", launch)
    receipt = tools.search_browser('C++ & file:///private "quote"', "Opera")
    argv = launch.call_args.args[0]
    assert argv[:2] == [str(Path("trusted/opera.exe")), "--new-tab"]
    assert argv[2] == "https://www.google.com/search?q=C%2B%2B%20%26%20file%3A%2F%2F%2Fprivate%20%22quote%22"
    assert "Requested" in receipt and "shell" not in launch.call_args.kwargs
    with pytest.raises(tools.ComputerControlError):
        tools.search_browser("logs", "evil.exe")
    assert launch.call_count == 1


def test_opera_discovery_uses_known_roots_and_rejects_redirected_ancestor(tmp_path, monkeypatch):
    path = tmp_path / "Programs" / "Opera GX" / "opera.exe"
    path.parent.mkdir(parents=True)
    path.touch()
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.delenv("PROGRAMFILES", raising=False)
    assert tools._opera_binary() == path.resolve()
    original = Path.resolve
    def resolve(value, *args, **kwargs):
        if value == path:
            return tmp_path.parent / "redirected" / "opera.exe"
        return original(value, *args, **kwargs)
    monkeypatch.setattr(Path, "resolve", resolve)
    assert tools._opera_binary() is None


@pytest.mark.parametrize("kind", ["track", "album"])
def test_show_spotify_uses_bounded_native_search_helper(monkeypatch, kind):
    run = Mock(return_value=SimpleNamespace(
        returncode=0,
        stdout=json.dumps({"ok": True, "message": "Spotify desktop search results are visible for that query."}),
    ))
    opened = Mock(side_effect=AssertionError("No browser fallback"))
    monkeypatch.setattr(tools.subprocess, "run", run)
    monkeypatch.setattr(tools.webbrowser, "open", opened)
    receipt = tools.show_spotify("Red by Seedhe Maut", kind)
    assert receipt == "Spotify desktop search results are visible for that query."
    argv = run.call_args.args[0]
    assert argv == [tools.sys.executable, "-m", "assistant.spotify_jam", "--search",
                    "--kind", kind, "--query=Red by Seedhe Maut"]
    assert run.call_args.kwargs["timeout"] == 22.0
    assert run.call_args.kwargs["stdin"] == tools.subprocess.DEVNULL
    assert "shell" not in run.call_args.kwargs
    opened.assert_not_called()


def test_native_or_browser_failure_is_truthful_refusal(monkeypatch):
    fresh_catalog(monkeypatch, [("Notepad", "NotepadApp")])
    monkeypatch.setattr(tools.subprocess, "Popen", Mock(side_effect=OSError("failed")))
    with pytest.raises(tools.ComputerControlError, match="failed"):
        tools.open_app("Notepad")


@pytest.mark.parametrize("query", ["", None, "x\ncommand", "x" * 401])
def test_bad_queries_cannot_reach_launch_or_browser(monkeypatch, query):
    launch, opened = Mock(), Mock()
    monkeypatch.setattr(tools.subprocess, "Popen", launch)
    run = Mock()
    monkeypatch.setattr(tools.subprocess, "run", run)
    monkeypatch.setattr(tools.webbrowser, "open", opened)
    with pytest.raises(tools.ComputerControlError):
        tools.search_browser(query)
    with pytest.raises(tools.ComputerControlError):
        tools.show_spotify(query, "track")
    launch.assert_not_called()
    run.assert_not_called()
    opened.assert_not_called()


def test_show_spotify_passes_shell_syntax_as_one_search_argument(monkeypatch):
    run = Mock(return_value=SimpleNamespace(
        returncode=0, stdout=json.dumps({"ok": True, "message": "Spotify results are visible."}),
    ))
    monkeypatch.setattr(tools.subprocess, "run", run)
    tools.show_spotify("Red; Remove-Item important", "track")
    assert run.call_args.args[0][-1] == "--query=Red; Remove-Item important"
    assert "shell" not in run.call_args.kwargs


@pytest.mark.parametrize("kind", ["playlist", None, []])
def test_show_spotify_rejects_invalid_kind_before_starting_helper(monkeypatch, kind):
    run = Mock()
    monkeypatch.setattr(tools.subprocess, "run", run)
    with pytest.raises(tools.ComputerControlError, match="song or album"):
        tools.show_spotify("Red", kind)
    run.assert_not_called()


def test_show_spotify_timeout_is_not_reported_as_success(monkeypatch):
    run = Mock(side_effect=subprocess.TimeoutExpired("spotify-search-helper", 22.0))
    monkeypatch.setattr(tools.subprocess, "run", run)
    with pytest.raises(tools.ComputerDeliveryUnknown, match="search may have been submitted") as caught:
        tools.show_spotify("Red Seedhe Maut", "track")
    assert isinstance(caught.value, tools.ComputerControlError)
    run.assert_called_once()


def test_start_spotify_jam_uses_bounded_fixed_helper_and_returns_verified_result(monkeypatch):
    run = Mock(return_value=SimpleNamespace(
        returncode=0,
        stdout=json.dumps({"ok": True, "message": "Spotify shows the Jam invitation instructions and Copy link button."}),
    ))
    monkeypatch.setattr(tools.subprocess, "run", run)
    message = tools.start_spotify_jam()
    assert message.startswith("Spotify shows the Jam invitation")
    assert run.call_args.args[0] == [tools.sys.executable, "-m", "assistant.spotify_jam", "--run"]
    assert run.call_args.kwargs["timeout"] == 15.0
    assert "shell" not in run.call_args.kwargs
    assert run.call_args.kwargs["stdin"] == tools.subprocess.DEVNULL


def test_start_spotify_jam_surfaces_child_refusal_without_claiming_success(monkeypatch):
    monkeypatch.setattr(tools.subprocess, "run", Mock(return_value=SimpleNamespace(
        returncode=0, stdout=json.dumps({"ok": False, "message": "Spotify's Jam Invite button is disabled."}),
    )))
    with pytest.raises(tools.ComputerControlError, match="Invite button is disabled"):
        tools.start_spotify_jam()


@pytest.mark.parametrize("result", [
    SimpleNamespace(returncode=1, stdout='{"ok":true,"message":"done"}'),
    SimpleNamespace(returncode=0, stdout="not json"),
    SimpleNamespace(returncode=0, stdout=json.dumps({"ok": True, "message": "done", "extra": "bad"})),
    SimpleNamespace(returncode=0, stdout=json.dumps({"ok": True, "message": "done\n"})),
    SimpleNamespace(returncode=0, stdout="x" * 4097),
])
def test_start_spotify_jam_rejects_invalid_helper_output(monkeypatch, result):
    monkeypatch.setattr(tools.subprocess, "run", Mock(return_value=result))
    with pytest.raises(tools.ComputerControlError, match="invalid result"):
        tools.start_spotify_jam()


def test_start_spotify_jam_timeout_warns_that_session_may_already_exist(monkeypatch):
    run = Mock(side_effect=subprocess.TimeoutExpired("spotify-jam-helper", 15.0))
    monkeypatch.setattr(tools.subprocess, "run", run)
    with pytest.raises(tools.ComputerDeliveryUnknown, match="Check Spotify before trying again"):
        tools.start_spotify_jam()
    assert run.call_args.kwargs["timeout"] == 15.0
    run.assert_called_once()


@pytest.mark.parametrize("result", [
    SimpleNamespace(returncode=1, stdout='{"ok":true,"message":"done"}'),
    SimpleNamespace(returncode=0, stdout="not json"),
    SimpleNamespace(returncode=0, stdout=json.dumps({"ok": True, "message": "done", "extra": "bad"})),
    SimpleNamespace(returncode=0, stdout=json.dumps({"ok": True, "message": "bad\x01message"})),
    SimpleNamespace(returncode=0, stdout="x" * 4097),
    SimpleNamespace(returncode=0, stdout=b"not decoded text"),
])
def test_spotify_helper_malformed_post_launch_result_is_delivery_unknown(monkeypatch, result):
    run = Mock(return_value=result)
    monkeypatch.setattr(tools.subprocess, "run", run)
    with pytest.raises(tools.ComputerDeliveryUnknown, match="Check Spotify before trying again") as caught:
        tools._run_spotify_helper(["--run"], timeout_message="Check Spotify before trying again.")
    assert isinstance(caught.value, tools.ComputerControlError)
    run.assert_called_once()


def test_spotify_helper_launch_failure_is_known_pre_dispatch_refusal(monkeypatch):
    run = Mock(side_effect=OSError("missing interpreter"))
    monkeypatch.setattr(tools.subprocess, "run", run)
    with pytest.raises(tools.ComputerControlError, match="could not be started") as caught:
        tools._run_spotify_helper(["--run"], timeout_message="Check Spotify before trying again.")
    assert not isinstance(caught.value, tools.ComputerDeliveryUnknown)
    run.assert_called_once()


@pytest.mark.parametrize("message", [
    "Spotify search was submitted, but the results could not be verified.",
    "The Jam may have started, but its invite was not verified.",
    "Jam Invite button is disabled.",
])
def test_spotify_child_refusal_is_delivery_unknown_without_retry(monkeypatch, message):
    run = Mock(return_value=SimpleNamespace(returncode=0, stdout=json.dumps({"ok": False, "message": message})))
    monkeypatch.setattr(tools.subprocess, "run", run)
    with pytest.raises(tools.ComputerDeliveryUnknown, match=message) as caught:
        tools._run_spotify_helper(["--run"], timeout_message="Check Spotify before trying again.")
    assert isinstance(caught.value, tools.ComputerControlError)
    run.assert_called_once()
