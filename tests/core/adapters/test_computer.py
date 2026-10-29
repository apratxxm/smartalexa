"""Computer adapter contract tests. Every launch, window, filesystem and helper call is a fake."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import quote

import pytest

from assistant import computer_tools as tools
from assistant.core import observe
from assistant.core.adapters import computer as adapter_mod
from assistant.core.adapters.base import Adapter, Clause
from assistant.core.executor import Receipt
from assistant.core.playback import TurnExpired
import sys

AppEntry = adapter_mod.AppEntry
OPERA = Path("C:/Users/test/AppData/Local/Programs/Opera GX/opera.exe")
EXPLORER = Path("C:/Windows/explorer.exe")
SPOTIFY_ID = "SpotifyAB.SpotifyMusic_zpdnekdrzrea0!Spotify"
PACKAGED = (
    AppEntry("Spotify", SPOTIFY_ID, "packaged", "C:\\Packages\\Spotify"),
    AppEntry("Don't Stop", "Vendor.Player!DontStop", "packaged"),
)
EDITOR = "C:\\Apps\\editor.exe"
CLASSIC = (AppEntry("Editor", EDITOR, "exe"),)
EXISTING = {EDITOR, "C:\\Apps\\tool.exe"}


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class Guard:
    def __init__(self, budget=30.0, expire_after=None, log=None):
        self.budget, self.expire_after, self.calls, self.log = budget, expire_after, 0, log

    def __call__(self):
        self.calls += 1
        if self.log is not None:
            self.log.append("guard")
        if self.expire_after is not None and self.calls > self.expire_after:
            raise TurnExpired("stale")
        return self.budget


def win(hwnd, image, title):
    path = "C:\\Packages\\Spotify\\spotify.exe" if image.casefold() == "spotify.exe" else ""
    return observe.Window(hwnd, 100 + hwnd, image, title, path)


class Desktop:
    """Fake top-level windows plus a spawn recorder. `on_spawn` mutates the desktop."""

    def __init__(self, windows=(), foreground=None, on_spawn=None, error=None, log=None):
        self.windows, self.foreground = list(windows), foreground
        self.on_spawn, self.error, self.log = on_spawn, error, log
        self.spawned = []

    def enumerate(self):
        return list(self.windows)

    def foreground_window(self):
        return self.foreground

    def popen(self, argv, **kwargs):
        self.spawned.append((list(argv), kwargs))
        if self.log is not None:
            self.log.append("popen")
        if self.error is not None:
            raise self.error
        if self.on_spawn is not None:
            self.on_spawn(self)
        return object()


def observer_for(desk, clock=None, max_wait=2.0):
    clock = clock or Clock()
    return observe.Observer(enumerate_windows=desk.enumerate, foreground_window=desk.foreground_window,
                            clock=clock, sleep=clock.sleep, poll=0.25, max_wait=max_wait)


def open_handler(desk, catalog=PACKAGED + CLASSIC, clock=None):
    return adapter_mod.OpenApp(catalog=lambda: catalog, explorer=lambda: EXPLORER,
                               isfile=lambda p: p in EXISTING,
                               observer=observer_for(desk, clock), popen=desk.popen)


def search_handler(desk, clock=None):
    return adapter_mod.SearchBrowser(opera=lambda: OPERA, observer=observer_for(desk, clock),
                                     popen=desk.popen)


def spotify_helper(desk=None, result="ok", error=None):
    def helper(*args, **kwargs):
        if error is not None:
            raise error
        return result
    return helper


def show_handler(cls, helper=None, exact=None):
    return cls(helper=helper or spotify_helper(), exact=exact)


# ---- shape -----------------------------------------------------------------

def test_adapter_exports_four_mutating_handlers_in_phase_three():
    assert isinstance(adapter_mod.ADAPTER, Adapter)
    assert adapter_mod.ADAPTER.area == "computer"
    assert adapter_mod.ADAPTER.phase == "3"
    assert set(adapter_mod.ADAPTER.handlers) == {"open_app", "search_browser", "show_track", "show_album"}
    for handler in adapter_mod.ADAPTER.handlers.values():
        assert handler.mutating is True
        assert handler.object_kind == "text"
        assert set(handler.effect_words) >= {"opened", "launched", "opening", "searched", "searching", "showing", "showed"}


def test_verbs_are_lowercase_and_do_not_collide_across_handlers():
    seen = {}
    for name, handler in adapter_mod.ADAPTER.handlers.items():
        for verb in handler.verbs:
            assert verb == verb.lower()
            assert verb not in seen, (verb, seen.get(verb), name)
            seen[verb] = name


# ---- license ----------------------------------------------------------------

def test_license_open_returns_object_only():
    h = adapter_mod.OpenApp()
    assert h.license(Clause("open spotify", "open", "Spotify")) == {"name": "Spotify"}
    assert h.license(Clause("start up spotify", "start up", "Spotify")) == {"name": "Spotify"}


@pytest.mark.parametrize("obj", ["", "   ", "the app", "a browser", "please"])
def test_license_stopword_or_empty_object_is_none(obj):
    assert adapter_mod.OpenApp().license(Clause("open x", "open", obj)) is None
    assert adapter_mod.SearchBrowser().license(Clause("google x", "google", obj)) is None


def test_license_rejects_foreign_verb():
    assert adapter_mod.OpenApp().license(Clause("search x", "search", "x")) is None


def test_license_search_fixes_browser():
    assert adapter_mod.SearchBrowser().license(Clause("search for cats", "search for", "cats")) == {
        "query": "cats", "browser": "Opera"}


def test_license_show_keeps_album_keyword_out_of_show_track():
    assert adapter_mod.ShowTrack().license(Clause("show halo", "show", "Halo")) == {"query": "Halo"}
    assert adapter_mod.ShowTrack().license(Clause("show album halo", "show", "album Halo")) is None
    assert adapter_mod.ShowTrack().license(Clause("show me the album halo", "show me", "the album halo")) is None
    assert adapter_mod.ShowAlbum().license(Clause("show the album halo", "show the album", "Halo")) == {"query": "Halo"}


# ---- C1 ---------------------------------------------------------------------

def test_c1_object_with_negation_needs_exact_catalogue_match_before_dispatch():
    desk = Desktop()
    result = open_handler(desk).execute({"name": "Don't Stop Believin"}, Guard())
    assert result.state == "failed"
    assert desk.spawned == []


def test_c1_exact_catalogue_match_may_launch():
    def foreground_app(desk):
        window = win(50, "editor.exe", "Don't Stop - Vendor")
        desk.windows.append(window)
        desk.foreground = window

    desk = Desktop(on_spawn=foreground_app)
    catalog = (AppEntry("Don't Stop", EDITOR, "exe"),)
    result = open_handler(desk, catalog=catalog).execute({"name": "Don't Stop"}, Guard())
    assert result.state == "observed"
    assert len(desk.spawned) == 1


def test_c1_show_fails_closed_without_dispatch():
    desk = Desktop()
    assert show_handler(adapter_mod.ShowTrack).execute({"query": "halo wait"}, Guard()).state == "failed"
    assert desk.spawned == []


def test_c1_exact_spotify_resolver_permits_show():
    result = show_handler(adapter_mod.ShowTrack, exact=lambda kind, q: True).execute(
        {"query": "Don't Stop"}, Guard())
    assert result.state == "observed"


@pytest.mark.parametrize("query", ["how to not burn rice", "cats don't bite", "actually good ideas",
                                   "nextjs tutorial", "playlist ideas", "how to stop smoking",
                                   "search how to use codex", "how to show hidden files",
                                   "show me halo"])
def test_search_is_exempt_from_negation_and_correction_words(query):
    desk = Desktop(on_spawn=lambda d: d.windows.append(win(70, "opera.exe", f"{query} - Google Search")))
    result = search_handler(desk).execute({"query": query, "browser": "Opera"}, Guard())
    assert result.state != "failed"
    assert len(desk.spawned) == 1


@pytest.mark.parametrize("query", [
    "cats and play music", "cats play music", "rice then pause", "music next",
    "volume up", "open the file", "start a jam", "cats and ask codex to fix it",
    "tell codex to x", "get codex to run", "have codex fix it", "send codex the log",
    "give codex the log", "show me the song halo", "show the album halo", "show me the album x",
    "cats, resume", "turn up the cats", "cats connector and skip",
])
def test_search_refuses_other_capability_verbs_with_or_without_connector(query):
    desk = Desktop()
    result = search_handler(desk).execute({"query": query, "browser": "Opera"}, Guard())
    assert result.state == "failed"
    assert desk.spawned == []


# ---- guard ------------------------------------------------------------------

def test_guard_called_immediately_before_dispatch():
    log = []
    desk = Desktop(log=log)
    open_handler(desk).execute({"name": "Spotify"}, Guard(log=log))
    assert log[:2] == ["guard", "popen"]


def test_expired_guard_means_no_dispatch():
    desk = Desktop()
    assert open_handler(desk).execute({"name": "Spotify"}, Guard(expire_after=0)).state == "failed"
    assert search_handler(desk).execute({"query": "cats", "browser": "Opera"}, Guard(expire_after=0)).state == "failed"
    assert show_handler(adapter_mod.ShowTrack).execute({"query": "Halo"}, Guard(expire_after=0)).state == "failed"
    assert desk.spawned == []


def test_zero_budget_means_no_dispatch():
    desk = Desktop()
    assert open_handler(desk).execute({"name": "Spotify"}, Guard(budget=0.0)).state == "failed"
    assert desk.spawned == []


# ---- C3 receipts (open_app and search_browser) ------------------------------

def test_pre_existing_window_without_transition_is_requested():
    desk = Desktop(windows=[win(10, "spotify.exe", "Spotify Premium")],
                   foreground=win(20, "explorer.exe", "Desktop"))
    result = open_handler(desk).execute({"name": "Spotify"}, Guard())
    assert result.state == "requested"
    assert result.content == "Spotify"


def test_new_matching_window_is_observed():
    desk = Desktop(on_spawn=lambda d: (d.windows.append(win(11, "spotify.exe", "Spotify Premium")), setattr(d, "foreground", d.windows[-1])))
    assert open_handler(desk).execute({"name": "Spotify"}, Guard()).state == "observed"


def test_post_dispatch_foreground_switch_to_existing_window_is_observed():
    desk = Desktop(windows=[win(10, "spotify.exe", "Spotify Premium")],
                   foreground=win(20, "explorer.exe", "Desktop"),
                   on_spawn=lambda d: setattr(d, "foreground", win(10, "spotify.exe", "Spotify Premium")))
    assert open_handler(desk).execute({"name": "Spotify"}, Guard()).state == "observed"


def test_opera_title_without_query_words_is_requested():
    desk = Desktop(on_spawn=lambda d: d.windows.append(win(30, "opera.exe", "Google - Opera")))
    assert search_handler(desk).execute({"query": "python tutorials", "browser": "Opera"}, Guard()).state == "requested"


def test_opera_new_window_with_first_two_query_words_is_observed():
    desk = Desktop(on_spawn=lambda d: (d.windows.append(win(31, "opera.exe", "python tutorials - Google Search")), setattr(d, "foreground", d.windows[-1])))
    assert search_handler(desk).execute({"query": "python tutorials", "browser": "Opera"}, Guard()).state == "observed"


def test_opera_pre_existing_window_is_requested():
    desk = Desktop(windows=[win(32, "opera.exe", "python tutorials - Google Search")],
                   foreground=win(20, "explorer.exe", "Desktop"))
    assert search_handler(desk).execute({"query": "python tutorials", "browser": "Opera"}, Guard()).state == "requested"


# ---- show_*: helper verification is the receipt -----------------------------

def test_show_helper_verified_is_observed_regardless_of_foreground():
    result = show_handler(adapter_mod.ShowTrack, helper=spotify_helper(result="ok")).execute(
        {"query": "Halo"}, Guard())
    assert result.state == "observed"


def test_show_helper_receives_fixed_argv_and_bounded_timeout():
    seen = {}

    def helper(arguments, **kwargs):
        seen["arguments"], seen["kwargs"] = arguments, kwargs
        return "ok"

    show_handler(adapter_mod.ShowAlbum, helper=helper).execute({"query": "Halo"}, Guard(budget=5.0))
    assert seen["arguments"] == ["--search", "--kind", "album", "--query=Halo"]
    assert seen["kwargs"]["timeout"] <= 5.0


def test_show_helper_unverified_is_unknown():
    result = show_handler(adapter_mod.ShowTrack,
                          helper=spotify_helper(error=tools.ComputerDeliveryUnknown("maybe"))).execute(
        {"query": "Halo"}, Guard())
    assert result.state == "unknown"


def test_show_helper_not_started_is_failed():
    result = show_handler(adapter_mod.ShowTrack,
                          helper=spotify_helper(error=tools.ComputerControlError("no"))).execute(
        {"query": "Halo"}, Guard())
    assert result.state == "failed"


def test_show_helper_other_exception_is_unknown():
    result = show_handler(adapter_mod.ShowTrack, helper=spotify_helper(error=RuntimeError("x"))).execute(
        {"query": "Halo"}, Guard())
    assert result.state == "unknown"


# ---- C4 -------------------------------------------------------------------

@pytest.mark.parametrize("entry", [
    AppEntry("Batch Tool", "Vendor.Tool!run.bat", "packaged"),
    AppEntry("Script", "Vendor.Js!x.js", "packaged"),
    AppEntry("Link", "Vendor.Link!site.url", "packaged"),
    AppEntry("Classic", "Microsoft.Classic", "packaged"),
    AppEntry("Batch Exe", "C:\\Apps\\run.bat", "exe"),
    AppEntry("Ps", "C:\\Apps\\run.ps1", "exe"),
    AppEntry("Uri", "ms-settings:display", "exe"),
    AppEntry("Remote", "https://example.com/x.exe", "exe"),
    AppEntry("Relative", "editor.exe", "exe"),
])
def test_c4_script_url_uri_and_non_exe_entries_never_launch(entry):
    desk = Desktop()
    result = open_handler(desk, catalog=(entry,)).execute({"name": entry.label}, Guard())
    assert result.state == "failed"
    assert desk.spawned == []


def test_c4_exe_target_removed_before_spawn_is_failed_without_dispatch():
    desk = Desktop()
    handler = adapter_mod.OpenApp(catalog=lambda: CLASSIC, explorer=lambda: EXPLORER,
                                  isfile=lambda p: False, observer=observer_for(desk), popen=desk.popen)
    assert handler.execute({"name": "Editor"}, Guard()).state == "failed"
    assert desk.spawned == []


def test_classic_exe_launches_directly_with_no_extra_arguments():
    def foreground_app(desk):
        window = win(80, "editor.exe", "Editor - doc")
        desk.windows.append(window)
        desk.foreground = window

    desk = Desktop(on_spawn=foreground_app)
    result = open_handler(desk, catalog=CLASSIC).execute({"name": "the editor"}, Guard())
    assert result.state == "observed"
    argv, kwargs = desk.spawned[0]
    assert argv == [EDITOR]
    assert not kwargs.get("shell", False)


def test_packaged_aumid_still_launches_through_explorer():
    desk = Desktop()
    open_handler(desk).execute({"name": "Spotify"}, Guard())
    argv, _ = desk.spawned[0]
    assert argv == [str(EXPLORER), "shell:AppsFolder\\" + SPOTIFY_ID]


def test_same_label_in_packaged_and_classic_is_ambiguous_and_fails():
    desk = Desktop()
    catalog = (AppEntry("Editor", "Vendor.Ed!Editor", "packaged"),) + CLASSIC
    assert open_handler(desk, catalog=catalog).execute({"name": "Editor"}, Guard()).state == "failed"
    assert desk.spawned == []


def test_prefix_match_still_requires_single_candidate():
    desk = Desktop()
    catalog = (AppEntry("Tool One", "C:\\Apps\\tool.exe", "exe"), AppEntry("Tool Two", "C:\\Apps\\tool.exe", "exe"))
    assert open_handler(desk, catalog=catalog).execute({"name": "Tool"}, Guard()).state == "failed"
    assert desk.spawned == []


def test_build_classic_keeps_only_existing_exe_targets_with_faked_filesystem():
    root = adapter_mod.shortcut_roots({"ProgramData": "C:\\ProgramData", "APPDATA": "C:\\Users\\t\\AppData\\Roaming"})[0]
    fs = {root: [root + "\\Editor.lnk", root + "\\Script.lnk", root + "\\Site.lnk", root + "\\Gone.lnk",
                 root + "\\Notes.txt", root + "\\Dup\\Editor.lnk", root + "\\Tool.lnk", root + "\\Url.lnk"]}
    targets = {
        root + "\\Editor.lnk": EDITOR,
        root + "\\Script.lnk": "C:\\Apps\\run.bat",
        root + "\\Site.lnk": "https://example.com",
        root + "\\Gone.lnk": "C:\\Apps\\gone.exe",
        root + "\\Dup\\Editor.lnk": EDITOR,
        root + "\\Tool.lnk": "C:\\Apps\\tool.exe",
        root + "\\Url.lnk": "C:\\Apps\\page.url",
    }
    entries = adapter_mod.build_classic(
        [root], walk=lambda r: fs.get(r, []),
        resolve=lambda paths: {p: targets[p] for p in paths if p in targets},
        isfile=lambda p: p in EXISTING)
    assert {(e.label, e.target, e.kind) for e in entries} == {
        ("Editor", EDITOR, "exe"), ("Tool", "C:\\Apps\\tool.exe", "exe")}


def test_resolver_passes_paths_on_stdin_never_in_argv():
    calls = []
    paths = ["C:\\ProgramData\\a'; calc; '.lnk", "C:\\Users\\x\\Start Menu\\b.lnk"]

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=0, stdout=json.dumps(
            [{"path": p, "target": "C:\\Apps\\x.exe"} for p in paths]))

    result = adapter_mod.resolve_targets(paths, powershell=lambda: Path("C:/ps.exe"), run=run)
    assert set(result) == set(paths)
    argv, kwargs = calls[0]
    assert all(p not in " ".join(argv) for p in paths)
    assert json.loads(kwargs["input"]) == paths
    assert not kwargs.get("shell", False)


def test_resolver_failure_yields_no_entries():
    def run(argv, **kwargs):
        return SimpleNamespace(returncode=1, stdout="")
    assert adapter_mod.resolve_targets(["C:\\a.lnk"], powershell=lambda: Path("C:/ps.exe"), run=run) == {}


def test_shortcut_roots_skip_missing_environment():
    assert adapter_mod.shortcut_roots({"APPDATA": "C:\\U\\R"}) == [
        "C:\\U\\R\\Microsoft\\Windows\\Start Menu\\Programs"]


# ---- replies ----------------------------------------------------------------

def test_reply_templates():
    h = adapter_mod.OpenApp()
    assert h.reply(Receipt("observed", "window", content="Spotify"), {"name": "spotify"}) == "Opening Spotify."
    assert h.reply(Receipt("requested", "unobserved", content="Spotify"), {"name": "spotify"}) == \
        "I asked Windows to open Spotify but can't see it yet."
    assert h.reply(Receipt("unknown", "dispatch_uncertain", content="Spotify"), {"name": "spotify"}) == \
        "I asked Windows to open Spotify but can't confirm it."
    assert h.reply(Receipt("failed", "not_found"), {"name": "Zed"}) == "I couldn't find an app called Zed."
    assert adapter_mod.SearchBrowser().reply(Receipt("observed", "window"), {"query": "cats", "browser": "Opera"}) == \
        "Searching Opera for cats."
    assert adapter_mod.SearchBrowser().reply(Receipt("requested", "unobserved"), {"query": "cats"}) == \
        "I asked Opera to search for cats but can't see it yet."
    assert adapter_mod.SearchBrowser().reply(Receipt("unknown", "x"), {"query": "cats"}) == \
        "I asked Opera to search for cats but can't see it yet."
    assert adapter_mod.ShowTrack().reply(Receipt("observed", "helper_verified"), {"query": "Halo"}) == \
        "Showing Halo in Spotify."
    assert adapter_mod.ShowAlbum().reply(Receipt("unknown", "x"), {"query": "Halo"}) == \
        "I asked Spotify to show Halo but can't see it yet."
    assert adapter_mod.ShowAlbum().reply(Receipt("failed", "x"), {"query": "Halo"}) == "I can't do that right now."
    assert adapter_mod.SearchBrowser().reply(Receipt("failed", "other_verb"), {"query": "cats"}) == \
        "I can't do that right now."


def test_catalogue_warming_template():
    desk = Desktop()
    handler = adapter_mod.OpenApp(catalog=lambda: (), explorer=lambda: EXPLORER,
                                  isfile=lambda p: True, observer=observer_for(desk), popen=desk.popen)
    receipt = handler.execute({"name": "Spotify"}, Guard())
    assert receipt.state == "failed" and receipt.detail == "catalogue_warming"
    assert handler.reply(receipt, {"name": "Spotify"}) == "I'm still loading your app list. Try again in a moment."
    assert desk.spawned == []


def test_reply_never_includes_helper_messages_or_effect_words_unless_observed():
    leak = "SECRET helper text"
    handlers = [adapter_mod.OpenApp(), adapter_mod.SearchBrowser(), adapter_mod.ShowTrack(), adapter_mod.ShowAlbum()]
    args = {"name": "Spotify", "query": "cats", "browser": "Opera"}
    for handler in handlers:
        for state in ("requested", "unknown", "failed"):
            text = handler.reply(Receipt(state, leak, content="Spotify", error=leak), args)
            assert leak not in text
            assert not any(word in text.casefold() for word in handler.effect_words), (state, text)


# ---- observe ----------------------------------------------------------------

def test_observer_wait_is_bounded_and_returns_none():
    clock = Clock()
    desk = Desktop()
    obs = observer_for(desk, clock, max_wait=1.0)
    before = obs.snapshot()
    assert obs.wait_for(before, lambda w: False, Guard(budget=10.0)) is None
    assert clock.now <= 1.0 + 1e-9


def test_observer_stale_guard_mid_wait_returns_none():
    desk = Desktop()
    obs = observer_for(desk)
    before = obs.snapshot()
    assert obs.wait_for(before, lambda w: True, Guard(expire_after=1)) is None


def test_observe_import_does_not_touch_desktop(monkeypatch):
    import ctypes

    def forbidden(*a, **k):
        raise AssertionError("desktop touched at import")

    monkeypatch.setattr(ctypes, "WinDLL", forbidden)
    name = "assistant.core.observe_fresh_copy"
    spec = importlib.util.spec_from_file_location(name, observe.__file__)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        assert hasattr(module, "Observer")
    finally:
        sys.modules.pop(name, None)


def test_spotify_title_on_notepad_foreground_is_requested():
    impostor = win(41, "notepad.exe", "Spotify notes")
    desk = Desktop(windows=[impostor], foreground=win(20, "explorer.exe", "Desktop"),
                   on_spawn=lambda d: setattr(d, "foreground", impostor))
    assert open_handler(desk).execute({"name": "Spotify"}, Guard()).state == "requested"


def test_matching_new_opera_background_is_requested():
    desk = Desktop(foreground=win(20, "explorer.exe", "Desktop"),
                   on_spawn=lambda d: d.windows.append(win(31, "opera.exe", "python tutorials - Google Search")))
    assert search_handler(desk).execute({"query": "python tutorials", "browser": "Opera"}, Guard()).state == "requested"


def test_catalogue_warming_then_ready_launches():
    clock = Clock()
    desk = Desktop()
    def catalog():
        return CLASSIC if clock.now >= 0.1 else ()
    handler = adapter_mod.OpenApp(catalog=catalog, explorer=lambda: EXPLORER,
                                 isfile=lambda p: p in EXISTING, observer=observer_for(desk),
                                 popen=desk.popen, clock=clock, sleep=clock.sleep)
    assert handler.execute({"name": "Editor"}, Guard()).state == "requested"
    assert len(desk.spawned) == 1


def test_catalogue_still_warming_stops_at_budget():
    clock = Clock()
    desk = Desktop()
    handler = adapter_mod.OpenApp(catalog=lambda: (), observer=observer_for(desk),
                                 popen=desk.popen, clock=clock, sleep=clock.sleep)
    receipt = handler.execute({"name": "Editor"}, Guard(budget=0.2))
    assert receipt.detail == "catalogue_warming"
    assert clock.now == pytest.approx(0.2)
    assert not desk.spawned


def test_packaged_calculator_uses_resolved_executable():
    entry = AppEntry("Calculator", "Microsoft.WindowsCalculator_8wekyb3d8bbwe!App", "packaged", "C:\\Packages\\Calculator")
    window = observe.Window(91, 191, "CALCULATORAPP.EXE", "Calculator", "C:\\Packages\\Calculator\\CalculatorApp.exe")
    def launch(desk):
        desk.windows.append(window)
        desk.foreground = window
    desk = Desktop(on_spawn=launch)
    assert open_handler(desk, catalog=(entry,)).execute({"name": "Calculator"}, Guard()).state == "observed"


def test_packaged_unknown_identity_is_requested():
    entry = AppEntry("Calculator", "Microsoft.WindowsCalculator_8wekyb3d8bbwe!App", "packaged")
    window = win(91, "calculatorapp.exe", "Calculator")
    def launch(desk):
        desk.windows.append(window)
        desk.foreground = window
    desk = Desktop(on_spawn=launch)
    assert open_handler(desk, catalog=(entry,)).execute({"name": "Calculator"}, Guard()).state == "requested"


def test_delayed_opera_foreground_is_observed_within_budget():
    clock = Clock()
    desk = Desktop(foreground=win(20, "explorer.exe", "Desktop"))
    before = observer_for(desk, clock).snapshot()
    window = win(31, "opera.exe", "cats - Google Search")
    def sleep(seconds):
        clock.sleep(seconds)
        if clock.now >= 4:
            desk.windows = [window]
            desk.foreground = window
    observer = observe.Observer(enumerate_windows=desk.enumerate, foreground_window=desk.foreground_window,
                               clock=clock, sleep=sleep, poll=0.25)
    assert observer.wait_for(before, lambda item: item.image == "opera.exe" and "cats" in item.title, Guard(5)) == window
    assert clock.now == 4


def test_observation_limit_is_six_seconds_or_remaining_budget():
    for budget, maximum in [(20, 6), (1.5, 1.5)]:
        clock = Clock()
        desk = Desktop()
        observer = observe.Observer(enumerate_windows=desk.enumerate, foreground_window=desk.foreground_window,
                                   clock=clock, sleep=clock.sleep, max_wait=50)
        assert observer.wait_for(observer.snapshot(), lambda item: True, Guard(budget)) is None
        assert clock.now == pytest.approx(maximum)


@pytest.mark.parametrize("path", [r"C:\Packages\CalculatorEvil\CalculatorApp.exe", r"C:\Packages\Calculator\..\Other\CalculatorApp.exe", ""])
def test_package_path_outside_install_location_is_requested(path):
    entry = AppEntry("Calculator", "Family!App", "packaged", r"C:\Packages\Calculator")
    window = observe.Window(90, 190, "calculatorapp.exe", "Calculator", path)
    def launch(desk):
        desk.windows.append(window)
        desk.foreground = window
    desk = Desktop(on_spawn=launch)
    assert open_handler(desk, catalog=(entry,)).execute({"name": "Calculator"}, Guard()).state == "requested"


def test_spotify_package_path_accepts_runtime_executable_not_manifest_stub():
    entry = AppEntry("Spotify", SPOTIFY_ID, "packaged", r"C:\Packages\Spotify")
    window = observe.Window(91, 191, "spotify.exe", "Spotify Premium", r"c:\packages\SPOTIFY\spotify.exe")
    def launch(desk):
        desk.windows.append(window)
        desk.foreground = window
    desk = Desktop(on_spawn=launch)
    assert open_handler(desk, catalog=(entry,)).execute({"name": "Spotify"}, Guard()).state == "observed"
