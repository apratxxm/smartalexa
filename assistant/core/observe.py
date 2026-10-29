"""Before/after top-level window observation for computer adapters (contract C3).

Importing this module touches no desktop state. ctypes is only bound inside the
default Windows enumerator on first use; tests inject their own enumerator.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable

from .playback import TurnExpired


@dataclass(frozen=True)
class Window:
    hwnd: int
    pid: int
    image: str   # lowercase executable file name, "" when unreadable
    title: str
    image_path: str = ""


@dataclass(frozen=True)
class Snapshot:
    windows: tuple[Window, ...]
    foreground: Window | None

    @property
    def hwnds(self) -> frozenset[int]:
        return frozenset(w.hwnd for w in self.windows)


_win32_cache = None


def _win32():
    """Bind Win32 functions on first use, never at import."""
    global _win32_cache
    if _win32_cache is not None:
        return _win32_cache
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    enum_proc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    user32.EnumWindows.argtypes = [enum_proc, wintypes.LPARAM]
    user32.IsWindowVisible.argtypes = [wintypes.HWND]
    user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.GetForegroundWindow.restype = wintypes.HWND
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.QueryFullProcessImageNameW.argtypes = [
        wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    user32.FindWindowExW.argtypes = [wintypes.HWND, wintypes.HWND, wintypes.LPCWSTR, wintypes.LPCWSTR]
    user32.FindWindowExW.restype = wintypes.HWND
    _win32_cache = (ctypes, wintypes, user32, kernel32, enum_proc)
    return _win32_cache


def _describe(hwnd: int) -> Window:
    ctypes, wintypes, user32, kernel32, _ = _win32()
    pid = wintypes.DWORD(0)
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    # Modern (UWP) apps draw inside ApplicationFrameHost; the app itself owns the child
    # CoreWindow. Report that process so identity checks see e.g. CalculatorApp.exe.
    core = user32.FindWindowExW(hwnd, None, "Windows.UI.Core.CoreWindow", None)
    if core:
        inner = wintypes.DWORD(0)
        user32.GetWindowThreadProcessId(core, ctypes.byref(inner))
        if inner.value and inner.value != pid.value:
            pid = inner
    image = ""
    image_path = ""
    handle = kernel32.OpenProcess(0x1000, False, pid.value)  # PROCESS_QUERY_LIMITED_INFORMATION
    if handle:
        try:
            size = wintypes.DWORD(1024)
            buffer = ctypes.create_unicode_buffer(1024)
            if kernel32.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size)):
                image_path = buffer.value
                image = image_path.rsplit("\\", 1)[-1].casefold()
        finally:
            kernel32.CloseHandle(handle)
    length = user32.GetWindowTextLengthW(hwnd)
    text = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, text, length + 1)
    return Window(int(hwnd or 0), int(pid.value), image, text.value, image_path)


def enumerate_windows() -> list[Window]:
    """Visible top-level windows. Windows-only; called only from the default observer."""
    ctypes, wintypes, user32, _, enum_proc = _win32()
    found: list[Window] = []

    def visit(hwnd, _lparam):
        if user32.IsWindowVisible(hwnd):
            found.append(_describe(hwnd))
        return True

    callback = enum_proc(visit)
    if not user32.EnumWindows(callback, 0):
        raise OSError("EnumWindows failed")
    return found


def foreground_window() -> Window | None:
    _, _, user32, _, _ = _win32()
    hwnd = user32.GetForegroundWindow()
    return _describe(hwnd) if hwnd else None


class Observer:
    """Snapshot before dispatch, then wait for a new or newly foregrounded match.

    `observed` needs a window that was absent from the snapshot, or one that
    matches and became foreground after dispatch. A pre-existing window that
    stays unchanged yields None (the caller reports `requested`).
    """

    def __init__(self, *, enumerate_windows: Callable[[], list[Window]] = enumerate_windows,
                 foreground_window: Callable[[], Window | None] = foreground_window,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep,
                 poll: float = 0.25, max_wait: float = 6.0):
        self._enumerate = enumerate_windows
        self._foreground = foreground_window
        self._clock = clock
        self._sleep = sleep
        self._poll = poll
        self._max_wait = min(max_wait, 6.0)

    def snapshot(self) -> Snapshot:
        windows = tuple(self._enumerate())
        return Snapshot(windows, self._foreground())

    def wait_for(self, before: Snapshot, match: Callable[[Window], bool], guard) -> Window | None:
        """Return the observed window, or None when unobservable within the guard budget.

        Enumeration errors propagate so the caller can report `unknown`.
        A stale guard ends the wait with None.
        """
        try:
            budget = float(guard())
        except TurnExpired:
            return None
        if budget <= 0:
            return None
        deadline = self._clock() + min(budget, self._max_wait)
        while True:
            windows = self._enumerate()
            foreground = self._foreground()
            for window in windows:
                if foreground is None or foreground.hwnd != window.hwnd or not match(foreground):
                    continue
                is_new = window.hwnd not in before.hwnds
                became_foreground = (foreground is not None and foreground.hwnd == window.hwnd
                                     and (before.foreground is None or before.foreground.hwnd != window.hwnd))
                if is_new or became_foreground:
                    return window
            now = self._clock()
            if now >= deadline:
                return None
            try:
                remaining = float(guard())
            except TurnExpired:
                return None
            if remaining <= 0:
                return None
            self._sleep(min(self._poll, deadline - now, remaining))


__all__ = ["Observer", "Snapshot", "Window", "enumerate_windows", "foreground_window"]
