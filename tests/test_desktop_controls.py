from unittest.mock import Mock

import pytest

from assistant.desktop_controls import NavigationError, open_codex_thread, switch_browser_tab


def test_chat_link_only_accepts_identifiers():
    open_it = Mock()
    open_codex_thread("01a062e2-11a3-7223-8315-df7227b40aa9", open_it)
    open_it.assert_called_once_with("codex://threads/01a062e2-11a3-7223-8315-df7227b40aa9")
    for invalid in ["https://evil", "abc?prompt=send", "../x", "", None, "thr_voice_plan_fake"]:
        with pytest.raises(NavigationError):
            open_codex_thread(invalid, open_it)
    assert open_it.call_count == 1


def test_default_chat_launcher_maps_breakaway_oserror_to_navigation_error(monkeypatch):
    import assistant.desktop_controls as controls
    monkeypatch.setattr(controls.os, "name", "nt")
    monkeypatch.setattr(controls, "open_external_target", Mock(side_effect=OSError("breakaway denied")))
    with pytest.raises(NavigationError, match="could not be opened"):
        controls.open_codex_thread("01a062e2-11a3-7223-8315-df7227b40aa9")


@pytest.mark.parametrize("app", ["Codex.exe", "powershell.exe", "notepad.exe", "LockApp.exe"])
def test_tab_switch_refuses_other_apps(app):
    backend = Mock()
    backend.foreground.return_value = (7, 8, app)
    with pytest.raises(NavigationError):
        switch_browser_tab("next", backend)
    backend.press_tab.assert_not_called()


def test_tab_switch_uses_observed_browser_and_fixed_direction():
    backend = Mock()
    backend.foreground.return_value = (7, 8, "msedge.exe")
    assert "shortcut" in switch_browser_tab("previous", backend)
    backend.press_tab.assert_called_once_with("previous", (7, 8, "msedge.exe"))
    with pytest.raises(NavigationError):
        switch_browser_tab("delete", backend)
    assert backend.press_tab.call_count == 1
