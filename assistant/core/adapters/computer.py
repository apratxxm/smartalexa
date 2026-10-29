"""Computer adapter (phase 3): open_app, search_browser, show_track, show_album.

Launch and UI helpers come from assistant.computer_tools. Catalogue filtering (C4),
the pre-dispatch snapshot and window observation (C3, observe.py), and the guard
before each mutation are applied here.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import ntpath
import os
import re
import subprocess
import threading
import time
from urllib.parse import quote

from ... import computer_tools
from ...external_apps import external_creation_flags
from ..playback import TurnExpired, remaining
from .. import observe
from .base import Adapter, Clause, Guard, Receipt

_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_STOP = frozenset("a an the app application browser song track album please thanks to my me it some up of on in for and".split())
# C1: these words in a free-text object require an exact catalogue or Spotify match before dispatch.
# search_browser is exempt from the negation/correction half (see _OTHER_VERBS).
_C1 = re.compile(
    r"\b(?:don['’]?t|do\s+not|not|never|no|actually|cancel|wait|stop|instead|rather|but|sorry|"
    r"nevermind|correction|i\s+mean|open|launch|start|search|google|look|show|play|pause|resume|"
    r"next|skip|send|ask|tell|codex)\b", re.I)
# search_browser: a verb phrase of any other mutating capability, with or without a connector, refuses.
_OTHER_VERBS = re.compile(
    r"\b(?:play|put\s+on|pause|resume|continue|unpause|skip|next|previous|restart|queue|"
    r"volume|louder|quieter|mute|unmute|turn\s+up|turn\s+down|open|launch|start)\b"
    r"|\b(?:ask|tell|get)\s+codex\s+to\b|\bhave\s+codex\b|\b(?:send|give)\s+codex\b"
    r"|\bshow\s+me\s+the\s+(?:song|album)\b|\bshow\s+the\s+(?:song|album)\b", re.I)
_BANNED_SUFFIXES = (".bat", ".cmd", ".ps1", ".vbs", ".js", ".url")
_EFFECT_WORDS = ("opened", "launched", "opening", "searched", "searching", "showing", "showed")
_NOT_FOUND = "not_found"
_TTL = 300.0
_RETRY = 15.0
_CLASSIC_CAP = 512

_CANT = "I can't do that right now."
_WARMING = "I'm still loading your app list. Try again in a moment."
_OPERA_UNSEEN = "I asked Opera to search for {q} but can't see it yet."
_SPOTIFY_UNSEEN = "I asked Spotify to show {q} but can't see it yet."
_WINDOWS_UNSEEN = "I asked Windows to open {label} but can't see it yet."
_WINDOWS_UNCONFIRMED = "I asked Windows to open {label} but can't confirm it."

_RESOLVE_SCRIPT = (
    "$ErrorActionPreference='Stop';"
    "$paths = [Console]::In.ReadToEnd() | ConvertFrom-Json;"
    "$shell = New-Object -ComObject WScript.Shell;"
    "$out = @();"
    "foreach ($p in @($paths)) { try { $out += [pscustomobject]@{path=[string]$p; target=[string]$shell.CreateShortcut([string]$p).TargetPath} } "
    "catch { $out += [pscustomobject]@{path=[string]$p; target=''} } };"
    "ConvertTo-Json -InputObject @($out) -Compress"
)


@dataclass(frozen=True)
class AppEntry:
    label: str
    target: str   # packaged: AUMID; exe: absolute path of a resolved .exe
    kind: str     # "packaged" | "exe"
    image: str = ""


def _key(value: str) -> str:
    return " ".join(re.findall(r"\w+", value.casefold()))


def _clean(value) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value or len(value) > 400 or _CONTROL.search(value):
        return None
    return value


def _valid_label(label: str) -> bool:
    return 0 < len(label) <= 120 and not _CONTROL.search(label)


def _object(clause: Clause) -> str | None:
    """The clause object, or None when empty or stopword-only."""
    if not isinstance(clause, Clause) or not isinstance(clause.object, str):
        return None
    text = clause.object.strip()
    tokens = _key(text).split()
    if not tokens or all(token in _STOP for token in tokens):
        return None
    return text


def _content_words(text: str) -> list[str]:
    return [word for word in _key(text).split() if word not in _STOP]


def _has_all_words(title: str, words: list[str]) -> bool:
    if not words:
        return False
    title_words = set(_key(title).split())
    return all(word in title_words for word in words)


def _has_phrase(title: str, phrase: str) -> bool:
    if not phrase:
        return False
    return (" " + phrase + " ") in (" " + _key(title) + " ")


def _launchable(entry: AppEntry) -> bool:
    """C4: packaged AUMIDs, or a resolved absolute .exe. Scripts, URLs and URIs never."""
    target = entry.target
    if not isinstance(target, str) or not target or "://" in target:
        return False
    if target.casefold().endswith(_BANNED_SUFFIXES):
        return False
    if entry.kind == "packaged":
        return "!" in target and "\\" not in target and "/" not in target
    if entry.kind == "exe":
        return ntpath.isabs(target) and target.casefold().endswith(".exe")
    return False


def _app_candidates(name: str, entries) -> list[AppEntry]:
    # Name rule mirrors computer_tools.open_app: exact, then "the " strip, then prefix.
    usable = [entry for entry in entries if _launchable(entry)]
    wanted = _key(name)
    exact = [entry for entry in usable if _key(entry.label) == wanted]
    if not exact and wanted.startswith("the "):
        wanted = wanted[4:]
        exact = [entry for entry in usable if _key(entry.label) == wanted]
    if exact:
        return exact
    return [entry for entry in usable
            if _key(entry.label).startswith(wanted + " ")
            and set(_key(entry.label)[len(wanted):].split()) <= {"app", "application", "browser", "gx"}]


# ---- classic Start-menu shortcuts (C4) --------------------------------------

def shortcut_roots(env) -> list[str]:
    roots = []
    for var in ("ProgramData", "APPDATA"):
        base = env.get(var)
        if base:
            roots.append(ntpath.join(base, "Microsoft", "Windows", "Start Menu", "Programs"))
    return roots


def walk_files(root: str):
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            yield ntpath.join(dirpath, name)


def resolve_targets(paths, *, powershell=None, run=subprocess.run) -> dict:
    """Resolve .lnk targets with one bounded WScript.Shell call.

    Paths travel as JSON on stdin, never inside the command string or argv.
    """
    if not paths:
        return {}
    binary = powershell() if powershell else computer_tools._windows_binary(
        "System32/WindowsPowerShell/v1.0/powershell.exe")
    try:
        result = run([str(binary), "-NoProfile", "-NonInteractive", "-Command", _RESOLVE_SCRIPT],
                     input=json.dumps(list(paths)), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                     timeout=8.0, check=False, text=True,
                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.TimeoutExpired):
        return {}
    if result.returncode != 0 or not isinstance(result.stdout, str) or len(result.stdout) > 262144:
        return {}
    try:
        rows = json.loads(result.stdout)
    except ValueError:
        return {}
    if isinstance(rows, dict):
        rows = [rows]
    if not isinstance(rows, list):
        return {}
    out = {}
    for row in rows:
        if isinstance(row, dict) and isinstance(row.get("path"), str) and isinstance(row.get("target"), str):
            out[row["path"]] = row["target"]
    return out


def build_classic(roots, *, walk, resolve, isfile) -> tuple[AppEntry, ...]:
    """Pure over its injected filesystem and resolver. Keeps only existing .exe targets."""
    shortcuts: list[str] = []
    for root in roots:
        for path in walk(root):
            if path.casefold().endswith(".lnk") and path not in shortcuts:
                shortcuts.append(path)
                if len(shortcuts) >= _CLASSIC_CAP:
                    break
        if len(shortcuts) >= _CLASSIC_CAP:
            break
    if not shortcuts:
        return ()
    targets = resolve(shortcuts)
    found: dict[tuple[str, str], AppEntry] = {}
    for path in shortcuts:
        label = ntpath.splitext(ntpath.basename(path))[0]
        target = targets.get(path)
        if not _valid_label(label) or not isinstance(target, str):
            continue
        candidate = AppEntry(label, target, "exe")
        if not _launchable(candidate) or not isfile(target):
            continue
        found[(label.casefold(), target.casefold())] = candidate
    return tuple(found.values())


class _ClassicSource:
    """Background warm-up; a voice command never waits on PowerShell."""

    def __init__(self, build):
        self._build = build
        self._lock = threading.Lock()
        self._entries: tuple[AppEntry, ...] = ()
        self._loaded = 0.0
        self._attempted = 0.0
        self._warming = False

    def get(self) -> tuple[AppEntry, ...]:
        now = time.monotonic()
        with self._lock:
            entries = self._entries
            stale = not self._loaded or now - self._loaded >= _TTL
            start = stale and not self._warming and (not self._attempted or now - self._attempted >= _RETRY)
            if start:
                self._warming, self._attempted = True, now
        if start:
            self._start()
        return entries

    def _start(self):
        def load():
            try:
                entries = self._build()
                with self._lock:
                    self._entries, self._loaded = tuple(entries), time.monotonic()
            except Exception:
                pass
            finally:
                with self._lock:
                    self._warming = False
        try:
            threading.Thread(target=load, name="memoir-classic-apps", daemon=True).start()
        except RuntimeError:
            with self._lock:
                self._warming = False


_CLASSIC = _ClassicSource(lambda: build_classic(
    shortcut_roots(os.environ), walk=walk_files, resolve=resolve_targets, isfile=os.path.isfile))


def _default_catalog():
    computer_tools.warm_catalog()
    with computer_tools._lock:
        packaged = tuple(AppEntry(label, ident, "packaged", computer_tools._package_images.get(ident.casefold(), ""))
                         for label, ident in computer_tools._catalog)
    return packaged + _CLASSIC.get()


# ---- handlers ---------------------------------------------------------------

class _Base:
    mutating = True
    object_kind = "text"
    effect_words = _EFFECT_WORDS

    def __init__(self, *, observer=None, popen=None):
        self._observer = observer or observe.Observer()
        self._popen = popen

    def _spawn(self, argv):
        spawn = self._popen or subprocess.Popen
        return spawn(argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL, creationflags=external_creation_flags())


class OpenApp(_Base):
    verbs = ("open", "launch", "start up")

    def __init__(self, *, catalog=None, explorer=None, isfile=None, clock=time.monotonic, sleep=time.sleep, **kwargs):
        super().__init__(**kwargs)
        self._clock, self._sleep = clock, sleep
        self._catalog = catalog or _default_catalog
        if catalog is None:
            computer_tools.warm_catalog()
            _CLASSIC.get()
        self._explorer = explorer or (lambda: computer_tools._windows_binary("explorer.exe"))
        self._isfile = isfile or os.path.isfile

    def license(self, clause: Clause) -> dict | None:
        if clause.verb not in self.verbs:
            return None
        text = _object(clause)
        return {"name": text} if text is not None else None

    def _argv(self, entry: AppEntry) -> list[str]:
        if entry.kind == "packaged":
            return [str(self._explorer()), "shell:AppsFolder\\" + entry.target]
        if not self._isfile(entry.target):
            raise OSError("target no longer exists")
        return [entry.target]

    def execute(self, args: dict, guard: Guard) -> Receipt:
        name = _clean(args.get("name"))
        if name is None:
            return Receipt("failed", "invalid_name")
        try:
            entries = self._catalog()
            candidates = _app_candidates(name, entries)
            if not candidates and (not entries or (self._catalog is _default_catalog and
                                                   (computer_tools._warming or _CLASSIC._warming))):
                try:
                    wait = min(2.0, remaining(guard))
                except TurnExpired:
                    return Receipt("failed", "catalogue_warming")
                end = self._clock() + wait
                while self._clock() < end:
                    self._sleep(min(0.05, end - self._clock()))
                    try:
                        remaining(guard)
                    except TurnExpired:
                        break
                    entries = self._catalog()
                    candidates = _app_candidates(name, entries)
                    if candidates:
                        break
            if not entries or (not candidates and self._catalog is _default_catalog and
                               (computer_tools._warming or _CLASSIC._warming)):
                return Receipt("failed", "catalogue_warming")
        except Exception as exc:  # Nothing dispatched yet.
            return Receipt("failed", "catalogue", error=type(exc).__name__)
        if len(candidates) != 1:
            return Receipt("failed", _NOT_FOUND)
        entry = candidates[0]
        if _C1.search(name) and _key(entry.label) != _key(name):
            return Receipt("failed", "c1_no_exact_match")
        try:
            argv = self._argv(entry)
        except Exception as exc:
            return Receipt("failed", "target", error=type(exc).__name__)
        try:
            before = self._observer.snapshot()
        except Exception as exc:
            return Receipt("failed", "snapshot", error=type(exc).__name__)
        try:
            remaining(guard)
        except TurnExpired:
            return Receipt("failed", "expired")
        try:
            self._spawn(argv)
        except OSError as exc:
            return Receipt("failed", "spawn", error=type(exc).__name__)
        except Exception as exc:
            return Receipt("unknown", "dispatch_uncertain", content=entry.label, error=type(exc).__name__)
        label = entry.label
        def match(window):
            if not _has_phrase(window.title, _key(label)):
                return False
            if entry.kind == "exe":
                return window.image.casefold() == ntpath.basename(entry.target).casefold()
            location = entry.image
            if self._catalog is _default_catalog:
                with computer_tools._lock:
                    location = computer_tools._package_images.get(entry.target.casefold(), location)
            if not location or not window.image_path:
                return False
            root = ntpath.normcase(ntpath.normpath(location))
            path = ntpath.normcase(ntpath.normpath(window.image_path))
            if not ntpath.isabs(root) or not ntpath.isabs(path):
                return False
            return path.startswith(root.rstrip("\\") + "\\")

        try:
            hit = self._observer.wait_for(before, match, guard)
        except Exception as exc:
            return Receipt("unknown", "observe_failed", content=label, error=type(exc).__name__)
        if hit is not None:
            return Receipt("observed", "window", content=label)
        return Receipt("requested", "unobserved", content=label)

    def reply(self, receipt: Receipt, args: dict) -> str:
        label = receipt.content or str(args.get("name", ""))
        if receipt.state == "observed":
            return f"Opening {label}."
        if receipt.state == "requested":
            return _WINDOWS_UNSEEN.format(label=label)
        if receipt.state == "unknown":
            return _WINDOWS_UNCONFIRMED.format(label=label)
        if receipt.state == "failed" and receipt.detail == _NOT_FOUND:
            return f"I couldn't find an app called {args.get('name', '')}."
        if receipt.state == "failed" and receipt.detail == "catalogue_warming":
            return _WARMING
        return _CANT


class SearchBrowser(_Base):
    verbs = ("search for", "search", "google", "look up")
    c1_exempt = True  # contract C1 exemption: non-destructive; other verbs refused in license()
    _browser = "Opera"

    def __init__(self, *, opera=None, **kwargs):
        super().__init__(**kwargs)
        self._opera = opera or computer_tools._opera_binary

    def license(self, clause: Clause) -> dict | None:
        if clause.verb not in self.verbs:
            return None
        text = _object(clause)
        return {"query": text, "browser": self._browser} if text is not None else None

    def execute(self, args: dict, guard: Guard) -> Receipt:
        query = _clean(args.get("query"))
        if query is None or args.get("browser") != self._browser:
            return Receipt("failed", "invalid_query")
        if _OTHER_VERBS.search(query):
            # Another capability's verb phrase inside a search query: refuse, no dispatch.
            return Receipt("failed", "other_verb")
        try:
            binary = self._opera()
        except Exception as exc:
            return Receipt("failed", "opera", error=type(exc).__name__)
        if binary is None:
            return Receipt("failed", "opera_missing")
        url = "https://www.google.com/search?q=" + quote(query, safe="")
        try:
            before = self._observer.snapshot()
        except Exception as exc:
            return Receipt("failed", "snapshot", error=type(exc).__name__)
        try:
            remaining(guard)
        except TurnExpired:
            return Receipt("failed", "expired")
        try:
            self._spawn([str(binary), "--new-tab", url])
        except OSError as exc:
            return Receipt("failed", "spawn", error=type(exc).__name__)
        except Exception as exc:
            return Receipt("unknown", "dispatch_uncertain", error=type(exc).__name__)
        words = _content_words(query)[:2]
        expected = "opera.exe"

        def match(window):
            return window.image.casefold() == expected and _has_all_words(window.title, words)

        try:
            hit = self._observer.wait_for(before, match, guard)
        except Exception as exc:
            return Receipt("unknown", "observe_failed", error=type(exc).__name__)
        if hit is not None:
            return Receipt("observed", "window")
        return Receipt("requested", "unobserved")

    def reply(self, receipt: Receipt, args: dict) -> str:
        query = args.get("query", "")
        if receipt.state == "observed":
            return f"Searching {self._browser} for {query}."
        if receipt.state in {"requested", "unknown"}:
            return _OPERA_UNSEEN.format(q=query)
        return _CANT


class _ShowSpotify(_Base):
    kind = ""

    def __init__(self, *, helper=None, exact=None, **kwargs):
        super().__init__(**kwargs)
        self._helper = helper or computer_tools._run_spotify_helper
        # exact(kind, query) -> bool. Default fails closed: no Spotify exact resolver is wired.
        self._exact = exact or (lambda kind, query: False)

    def license(self, clause: Clause) -> dict | None:
        if clause.verb not in self.verbs:
            return None
        text = _object(clause)
        if text is None or self._extra_refusal(text):
            return None
        return {"query": text}

    def _extra_refusal(self, text: str) -> bool:
        return False

    def execute(self, args: dict, guard: Guard) -> Receipt:
        query = _clean(args.get("query"))
        if query is None:
            return Receipt("failed", "invalid_query")
        if _C1.search(query) and not self._exact(self.kind, query):
            return Receipt("failed", "c1_no_exact_match")
        try:
            budget = remaining(guard)
        except TurnExpired:
            return Receipt("failed", "expired")
        try:
            # The helper's verified ok is the receipt; foreground does not change it.
            self._helper(["--search", "--kind", self.kind, "--query=" + query],
                         timeout=min(22.0, budget), deadline=time.monotonic() + budget,
                         timeout_message="Spotify did not verify desktop search results before the UI timed out. "
                                         "Check Spotify before trying again; the search may have been submitted.")
        except computer_tools.ComputerDeliveryUnknown as exc:
            return Receipt("unknown", "helper_unknown", error=type(exc).__name__)
        except computer_tools.ComputerControlError as exc:
            # Raised only before the helper was spawned.
            return Receipt("failed", "helper_not_started", error=type(exc).__name__)
        except Exception as exc:
            return Receipt("unknown", "helper_failed", error=type(exc).__name__)
        return Receipt("observed", "helper_verified")

    def reply(self, receipt: Receipt, args: dict) -> str:
        query = args.get("query", "")
        if receipt.state == "observed":
            return f"Showing {query} in Spotify."
        if receipt.state in {"requested", "unknown"}:
            return _SPOTIFY_UNSEEN.format(q=query)
        return _CANT


class ShowTrack(_ShowSpotify):
    kind = "track"
    verbs = ("show me the song", "show the song", "show me", "show")

    def _extra_refusal(self, text: str) -> bool:
        # "album" belongs to show_album, so show_track never takes it (C2).
        return "album" in _key(text).split()


class ShowAlbum(_ShowSpotify):
    kind = "album"
    verbs = ("show me the album", "show the album")


ADAPTER = Adapter("computer", "3", {
    "open_app": OpenApp(),
    "search_browser": SearchBrowser(),
    "show_track": ShowTrack(),
    "show_album": ShowAlbum(),
})

__all__ = ["ADAPTER", "AppEntry", "OpenApp", "SearchBrowser", "ShowAlbum", "ShowTrack",
           "build_classic", "resolve_targets", "shortcut_roots"]
