"""Control the local Spotify app through Windows media controls (SMTC).

Pause, play, next, previous and restart need no Web API call: Windows exposes
the desktop app's transport the same way the keyboard media keys do. Transport
commands are verified against a bounded SMTC snapshot; accepted commands alone
are not treated as completed playback.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import logging
import math
import time
from typing import Optional

log = logging.getLogger("memoir.media")

PLAYING, PAUSED = 4, 5          # GlobalSystemMediaTransportControlsSessionPlaybackStatus
SPOTIFY_APP = "spotify"


@dataclass(frozen=True)
class MediaSnapshot:
    title: str
    artist: str
    playing: bool
    album_title: str = ""
    position_ms: Optional[int] = None


def _run(coro, timeout: float = 3.0):
    return asyncio.run(asyncio.wait_for(coro, timeout))


async def _session():
    from winrt.windows.media.control import (
        GlobalSystemMediaTransportControlsSessionManager as Manager)
    manager = await Manager.request_async()
    for session in manager.get_sessions():
        if SPOTIFY_APP in (session.source_app_user_model_id or "").casefold():
            return session
    return None


def available() -> bool:
    try:
        return _run(_session()) is not None
    except Exception as exc:
        log.debug("Media controls unavailable: %s", exc)
        return False


def status() -> Optional[int]:
    """Spotify's playback status (PLAYING/PAUSED/...), or None without a session."""
    async def read():
        session = await _session()
        return None if session is None else int(session.get_playback_info().playback_status)
    try:
        return _run(read())
    except Exception as exc:
        log.debug("Media status unavailable: %s", exc)
        return None


def snapshot() -> Optional[MediaSnapshot]:
    """Read Spotify's title, credited artist, and playing state together."""
    async def read():
        session = await _session()
        if session is None:
            return None
        info = session.get_playback_info()
        if info is None or info.playback_status is None:
            return None
        props = await session.try_get_media_properties_async()
        if props is None:
            return None
        return MediaSnapshot(
            title=str(props.title or ""),
            artist=str(props.artist or ""),
            playing=int(info.playback_status) == PLAYING,
            album_title=str(getattr(props, "album_title", "") or ""),
        )
    try:
        return _run(read())
    except Exception as exc:
        log.debug("Media snapshot unavailable: %s", exc)
        return None


def is_playing() -> Optional[bool]:
    s = status()
    return None if s is None else s == PLAYING


def title() -> str:
    async def read():
        session = await _session()
        if session is None:
            return ""
        props = await session.try_get_media_properties_async()
        return "{} - {}".format(props.title or "", props.artist or "").strip(" -")
    try:
        return _run(read())
    except Exception:
        return ""


def _command(name: str) -> bool:
    async def act():
        session = await _session()
        if session is None:
            return False
        if name == "restart":
            return bool(await session.try_change_playback_position_async(0))
        return bool(await getattr(session, name)())
    try:
        return _run(act())
    except Exception as exc:
        log.warning("Media command %s failed: %s", name, exc)
        return False


async def _command_on_session(session, name: str) -> bool:
    if name == "restart":
        return bool(await session.try_change_playback_position_async(0))
    return bool(await getattr(session, name)())


def _wait_for(predicate, timeout: float = 1.5) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.1)
    return False


async def _read_transport_snapshot(session) -> Optional[MediaSnapshot]:
    """Read track identity and optional position from this exact session."""
    if session is None:
        return None
    info = session.get_playback_info()
    if info is None or info.playback_status is None:
        return None
    props = await session.try_get_media_properties_async()
    if props is None:
        return None
    position_ms = None
    try:
        timeline = session.get_timeline_properties()
        position = timeline.position if timeline is not None else None
        if position is not None:
            millis = position.total_seconds() * 1000
            if math.isfinite(millis) and millis >= 0:
                position_ms = int(millis)
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError, OverflowError):
        # Track changes remain verifiable without timeline support; same-
        # track previous/restart require it.
        pass
    return MediaSnapshot(str(props.title or ""), str(props.artist or ""),
                         int(info.playback_status) == PLAYING,
                         str(getattr(props, "album_title", "") or ""),
                         position_ms)


def _transport_snapshot(timeout: float = 1.5) -> Optional[MediaSnapshot]:
    """Read one bounded transport snapshot (ordinary callers need no timeline)."""
    async def read():
        return await _read_transport_snapshot(await _session())
    try:
        return _run(read(), timeout=timeout)
    except Exception as exc:
        log.debug("Media transport snapshot unavailable: %s", exc)
        return None


def _identity(snapshot: Optional[MediaSnapshot]) -> tuple[str, str]:
    if snapshot is None or not snapshot.title.strip():
        return "", ""
    return snapshot.title.strip().casefold(), snapshot.artist.strip().casefold()


async def _verified_transport(name: str, predicate_builder) -> bool:
    """Select one session and verify the single mutation against that session."""
    session = await _session()
    if session is None:
        return False
    before = await _read_transport_snapshot(session)
    predicate = predicate_builder(before)
    if predicate is None:
        return False
    if not await _command_on_session(session, name):
        return False
    loop = asyncio.get_running_loop()
    deadline = loop.time() + 1.5
    while loop.time() < deadline:
        after = await _read_transport_snapshot(session)
        if after is not None and predicate(after):
            return True
        remaining = deadline - loop.time()
        if remaining > 0:
            await asyncio.sleep(min(0.1, remaining))
    return False


def pause() -> bool:
    if status() == PAUSED:
        return True
    return _command("try_pause_async") and _wait_for(lambda: status() == PAUSED)


def play() -> bool:
    if status() == PLAYING:
        return True
    return _command("try_play_async") and _wait_for(lambda: status() == PLAYING)


def next_track() -> bool:
    before = title()
    return _command("try_skip_next_async") and _wait_for(lambda: title() != before)


def previous_track() -> bool:
    def build(before):
        identity = _identity(before)
        if not identity[0]:
            return None
        def completed(after):
            current = _identity(after)
            if not current[0]:
                return False
            if current != identity:
                return True
            return (before.position_ms is not None and after.position_ms is not None
                    and after.position_ms <= 1500
                    and before.position_ms - after.position_ms >= 1000)
        return completed
    try:
        return _run(_verified_transport("try_skip_previous_async", build), timeout=3.0)
    except Exception as exc:
        log.debug("Media previous verification unavailable: %s", exc)
        return False


def restart() -> bool:
    def build(before):
        identity = _identity(before)
        if not identity[0]:
            return None
        return lambda after: (_identity(after) == identity
                              and after.position_ms is not None
                              and after.position_ms <= 1500)
    try:
        return _run(_verified_transport("restart", build), timeout=3.0)
    except Exception as exc:
        log.debug("Media restart verification unavailable: %s", exc)
        return False


# ---------- keeping the Spotify window out of sight ----------

def _spotify_pids() -> set:
    import psutil
    return {p.pid for p in psutil.process_iter(["name"])
            if (p.info["name"] or "").casefold() == "spotify.exe"}


def _spotify_windows(pids: Optional[set] = None) -> list:
    """Visible top-level windows owned by Spotify.exe."""
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    pids = _spotify_pids() if pids is None else pids
    found = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def visit(hwnd, _):
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if pid.value in pids and user32.IsWindowVisible(hwnd) and user32.GetWindowTextLengthW(hwnd):
            found.append(hwnd)
        return True

    user32.EnumWindows(visit, 0)
    return found


def spotify_in_front() -> bool:
    import ctypes
    try:
        return ctypes.windll.user32.GetForegroundWindow() in _spotify_windows()
    except Exception:
        return False


def _give_focus_back(hwnd) -> None:
    """SetForegroundWindow from a background process only works while
    attached to the current foreground thread's input queue."""
    import ctypes
    user32, kernel32 = ctypes.windll.user32, ctypes.windll.kernel32
    front = user32.GetForegroundWindow()
    front_thread = user32.GetWindowThreadProcessId(front, None)
    ours = kernel32.GetCurrentThreadId()
    attached = bool(front_thread) and front_thread != ours and user32.AttachThreadInput(ours, front_thread, True)
    try:
        ok = user32.SetForegroundWindow(hwnd)
    finally:
        if attached:
            user32.AttachThreadInput(ours, front_thread, False)
    if not ok or user32.GetForegroundWindow() != hwnd:
        # Stronger switch that Windows allows across processes.
        user32.SwitchToThisWindow(hwnd, True)
    log.debug("Focus handed back (set=%s)", ok)


class InvisibleSpotify:
    """While a spotify: link is opened, keep the app's window see-through and
    hand focus straight back; afterwards restore its look and state.

    Minimising early is not an option: Spotify then never starts the track.
    """

    GWL_EXSTYLE, WS_EX_LAYERED, LWA_ALPHA = -20, 0x80000, 0x2

    def __init__(self) -> None:
        import ctypes
        self.user32 = ctypes.windll.user32
        self.previous = self.user32.GetForegroundWindow()
        self.pids = _spotify_pids()
        windows = _spotify_windows(self.pids)
        self.window = windows[0] if windows else None
        self.was_minimised = bool(self.window) and bool(self.user32.IsIconic(self.window))
        # A minimised Spotify can still hold keyboard focus; that is not the
        # user looking at it.
        self.was_front = (bool(self.window) and self.previous == self.window
                          and not self.was_minimised)
        if self.previous == self.window:
            self.previous = None
        self._style = None

    def __enter__(self):
        if self.window and not self.was_front:
            try:
                self._style = self.user32.GetWindowLongW(self.window, self.GWL_EXSTYLE)
                self.user32.SetWindowLongW(self.window, self.GWL_EXSTYLE,
                                           self._style | self.WS_EX_LAYERED)
                self.user32.SetLayeredWindowAttributes(self.window, 0, 1, self.LWA_ALPHA)
            except Exception as exc:
                log.debug("Could not make Spotify see-through: %s", exc)
                self._style = None
        return self

    def return_focus(self, timeout: float = 3.0) -> None:
        """Once Spotify grabs focus, give it back to the user's window
        (or, if there was none, minimise Spotify so Windows picks one)."""
        if self.was_front:
            return
        if not self.previous:
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                front = self.user32.GetForegroundWindow()
                if front in _spotify_windows(self.pids) and not self.user32.IsIconic(front):
                    self.user32.ShowWindowAsync(front, 6)
                time.sleep(0.02)
            return
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            front = self.user32.GetForegroundWindow()
            if front and front != self.previous and front in _spotify_windows(self.pids):
                _give_focus_back(self.previous)
                time.sleep(0.15)   # let the switch land before checking again
                continue
            time.sleep(0.02)

    def __exit__(self, *exc) -> None:
        if not self.window or self.was_front:
            return
        try:
            if self.was_minimised:
                self.user32.ShowWindowAsync(self.window, 7)   # SW_SHOWMINNOACTIVE
                time.sleep(0.15)
            if self._style is not None:
                self.user32.SetLayeredWindowAttributes(self.window, 0, 255, self.LWA_ALPHA)
                self.user32.SetWindowLongW(self.window, self.GWL_EXSTYLE, self._style)
            if self.previous and self.user32.GetForegroundWindow() != self.previous:
                _give_focus_back(self.previous)
            # Spotify sometimes activates itself a moment after the track
            # starts; keep handing focus back for a little while.
            import threading
            threading.Thread(target=self.return_focus, args=(2.0,), daemon=True).start()
        except Exception as exc:
            log.debug("Could not restore the Spotify window: %s", exc)
