"""Bounded installed-app and browser tools; never model-authored shell input."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
from urllib.parse import quote
import webbrowser

from .intent import capability_domain


class ComputerControlError(ValueError):
    """An actionable safe refusal suitable for voice feedback."""


class ComputerDeliveryUnknown(ComputerControlError):
    """A desktop helper may have applied an action without confirming it."""


_TTL = 300.0
_RETRY = 15.0
_lock = threading.Lock()
_catalog = ()
_loaded_at = 0.0
_attempted_at = 0.0
_warming = False
_APP_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._! -]{0,255}")
_CATALOG_SCRIPT = ("Get-StartApps | Select-Object -First 256 Name,AppID | "
                   "ConvertTo-Json -Compress")


def desktop_domain(text: str) -> bool:
    if capability_domain(text):
        return True
    if not isinstance(text, str) or re.search(r"\b(?:codex|chat|conversation|thread)\b", text, re.I):
        return False
    # Admission can use actual installed display names, but grants no launch
    # authority. The Brain must still ground one complete present request.
    if not re.match(r"^(?:(?:please|can you|could you|would you|will you)\s+)*(?:please\s+)?"
                    r"(?:(?:i(?:'d| would)?\s+(?:like|want)\s+to)\s+)?(?:get|bring|start)\b", text.strip(), re.I):
        return False
    heard = " " + _key(text) + " "
    with _lock:
        entries = _catalog
    return bool(re.search(r"\b(?:app|application|browser)\b", text, re.I)
                or any(" " + _key(label) + " " in heard for label, _ in entries))


def _trusted_path(anchor: Path, suffix: str) -> Path | None:
    try:
        canonical = anchor.resolve(strict=True)
        path = canonical / suffix
        resolved = path.resolve(strict=True)
        if resolved != path or canonical not in resolved.parents or not resolved.is_file():
            return None
        return resolved
    except (OSError, ValueError):
        return None


def _windows_binary(suffix: str) -> Path:
    root = os.environ.get("SystemRoot")
    binary = _trusted_path(Path(root), suffix) if root and os.name == "nt" else None
    if binary is None:
        raise ComputerControlError("Windows application controls are unavailable.")
    return binary


def _load_catalog():
    powershell = _windows_binary("System32/WindowsPowerShell/v1.0/powershell.exe")
    result = subprocess.run([str(powershell), "-NoProfile", "-NonInteractive", "-Command", _CATALOG_SCRIPT],
                            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            timeout=2.5, check=False, text=True,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if result.returncode != 0 or not isinstance(result.stdout, str) or len(result.stdout) > 131072:
        raise ComputerControlError("The installed-app catalogue could not be read.")
    rows = json.loads(result.stdout)
    rows = [rows] if isinstance(rows, dict) else rows
    if not isinstance(rows, list):
        raise ComputerControlError("The installed-app catalogue could not be read.")
    entries = {}
    for row in rows[:256]:
        if not isinstance(row, dict):
            continue
        name, ident = row.get("Name"), row.get("AppID")
        if (isinstance(name, str) and 0 < len(name) <= 120 and not re.search(r"[\x00-\x1f\x7f]", name)
                and isinstance(ident, str) and _APP_ID.fullmatch(ident)):
            entries[(name, ident)] = (name, ident)
    return tuple(entries.values())


def warm_catalog() -> None:
    """Optional daemon warm; never make a voice command wait on PowerShell."""
    global _warming, _attempted_at
    now = time.monotonic()
    with _lock:
        if _warming or (_loaded_at and now - _loaded_at < _TTL) or (_attempted_at and now - _attempted_at < _RETRY):
            return
        _warming, _attempted_at = True, now
    def load():
        global _catalog, _loaded_at, _warming
        try:
            entries = _load_catalog()
            with _lock:
                _catalog, _loaded_at = entries, time.monotonic()
        except (OSError, ValueError, subprocess.TimeoutExpired):
            pass
        finally:
            with _lock:
                _warming = False
    try:
        threading.Thread(target=load, name="memoir-app-catalog", daemon=True).start()
    except RuntimeError:
        with _lock:
            _warming = False


def _text(value: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 400 or re.search(r"[\x00-\x1f\x7f]", value):
        raise ComputerControlError("Repeat the name or query without extra commands.")
    return value.strip()


def _run_spotify_helper(arguments: list[str], *, timeout_message: str, timeout: float = 15.0) -> str:
    command = [sys.executable, "-m", "assistant.spotify_jam", *arguments]
    try:
        result = subprocess.run(
            command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, timeout=timeout, check=False, text=True,
            cwd=str(Path(__file__).resolve().parent.parent),
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except subprocess.TimeoutExpired as exc:
        raise ComputerDeliveryUnknown(timeout_message) from exc
    except OSError as exc:
        raise ComputerControlError("The bounded Spotify desktop helper could not be started.") from exc
    except UnicodeError as exc:
        raise ComputerDeliveryUnknown("Spotify's desktop helper returned an invalid result. Check Spotify before trying again.") from exc
    output = result.stdout
    if (result.returncode != 0 or not isinstance(output, str) or len(output) > 4096
            or re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", output)):
        raise ComputerDeliveryUnknown("Spotify's desktop helper returned an invalid result. Check Spotify before trying again.")
    try:
        payload = json.loads(output)
    except (TypeError, ValueError) as exc:
        raise ComputerDeliveryUnknown("Spotify's desktop helper returned an invalid result. Check Spotify before trying again.") from exc
    if (not isinstance(payload, dict) or set(payload) != {"ok", "message"}
            or not isinstance(payload.get("ok"), bool)
            or not isinstance(payload.get("message"), str)
            or not payload["message"].strip() or len(payload["message"]) > 400
            or re.search(r"[\x00-\x1f\x7f]", payload["message"])):
        raise ComputerDeliveryUnknown("Spotify's desktop helper returned an invalid result. Check Spotify before trying again.")
    if not payload["ok"]:
        raise ComputerDeliveryUnknown(payload["message"])
    return payload["message"]


def _key(value: str) -> str:
    return " ".join(re.findall(r"\w+", value.casefold()))


def open_app(name: str) -> str:
    query = _text(name)
    with _lock:
        fresh = bool(_loaded_at and time.monotonic() - _loaded_at < _TTL)
        entries = _catalog
    if not fresh:
        warm_catalog()
    if not entries:
        raise ComputerControlError("The installed-app catalogue is warming. Repeat the app request shortly.")
    wanted = _key(query)
    exact = [(label, ident) for label, ident in entries if _key(label) == wanted]
    if not exact and wanted.startswith("the "):
        wanted = wanted[4:]
        exact = [(label, ident) for label, ident in entries if _key(label) == wanted]
    candidates = exact or [(label, ident) for label, ident in entries
                           if _key(label).startswith(wanted + " ")
                           and set(_key(label)[len(wanted):].split()) <= {"app", "application", "browser", "gx"}]
    if len(candidates) != 1:
        raise ComputerControlError("That app name is missing or ambiguous. " +
            ("The catalogue is refreshing; repeat its installed name shortly." if not fresh else "Say its installed name."))
    label, ident = candidates[0]
    # The target comes only from the validated OS catalogue, not the model.
    explorer = _windows_binary("explorer.exe")
    try:
        subprocess.Popen([str(explorer), "shell:AppsFolder\\" + ident], stdin=subprocess.DEVNULL,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except OSError as exc:
        raise ComputerControlError("The app opening request failed.") from exc
    return "Requested " + label + "."


def _opera_binary() -> Path | None:
    roots = [(os.environ.get("LOCALAPPDATA"), "Programs/Opera GX/opera.exe"),
             (os.environ.get("LOCALAPPDATA"), "Programs/Opera/opera.exe"),
             (os.environ.get("PROGRAMFILES"), "Opera GX/opera.exe"),
             (os.environ.get("PROGRAMFILES"), "Opera/opera.exe")]
    for anchor, suffix in roots:
        binary = _trusted_path(Path(anchor), suffix) if anchor else None
        if binary:
            return binary
    return None


def search_browser(query: str, browser: str = "Opera") -> str:
    query = _text(query)
    if not isinstance(browser, str) or browser.casefold() not in {"opera", "opera gx"}:
        raise ComputerControlError("Only an installed Opera browser is supported for this request.")
    binary = _opera_binary()
    if binary is None:
        raise ComputerControlError("An installed Opera browser could not be verified.")
    url = "https://www.google.com/search?q=" + quote(query, safe="")
    try:
        subprocess.Popen([str(binary), "--new-tab", url], stdin=subprocess.DEVNULL,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except OSError as exc:
        raise ComputerControlError("The Opera tab request failed.") from exc
    return "Requested a new Opera search tab."


def show_spotify(query: str, kind: str) -> str:
    query = _text(query)
    if not isinstance(kind, str) or kind not in {"track", "album"}:
        raise ComputerControlError("Choose a song or album to show.")
    return _run_spotify_helper(
        ["--search", "--kind", kind, "--query=" + query],
        timeout=22.0,
        timeout_message="Spotify did not verify desktop search results before the UI timed out. Check Spotify before trying again; the search may have been submitted.")


def start_spotify_jam() -> str:
    """Show Spotify's Jam invitation UI and verify it in an isolated helper.

    The child process is deliberately time-limited because Windows UI Automation
    can block inside a third-party application's accessibility provider.
    """
    return _run_spotify_helper(
        ["--run"],
        timeout_message="Spotify's Jam invite was not verified before the UI timed out. Check Spotify before trying again; a Jam may already have started.")
