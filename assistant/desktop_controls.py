"""Narrow desktop navigation; no arbitrary keys, text, commands, or vision."""
from __future__ import annotations

import os
import re

from .external_apps import open_external_target


class NavigationError(RuntimeError):
    pass


_THREAD = re.compile(r"(?:[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}|thr_[A-Za-z0-9_-]{1,100})\Z")
_BROWSERS = {"chrome.exe", "msedge.exe", "firefox.exe", "brave.exe", "vivaldi.exe", "opera.exe"}


def open_codex_thread(thread_id: str, opener=None) -> str:
    if (not isinstance(thread_id, str) or not _THREAD.fullmatch(thread_id)
            or thread_id.startswith("thr_voice_plan_")):
        raise NavigationError("Invalid Codex chat identifier.")
    if opener is None:
        if os.name != "nt":
            raise NavigationError("Desktop chat navigation requires Windows.")
        opener = open_external_target
    target = "codex://threads/" + thread_id
    try:
        opener(target)
    except OSError as exc:
        if opener is open_external_target:
            raise NavigationError("The Codex chat could not be opened.") from exc
        raise
    # Opening a deep link is not proof that the requested chat became visible.
    return "Requested the chat in Codex."


def switch_browser_tab(direction: str, backend=None) -> str:
    if direction not in {"next", "previous"}:
        raise NavigationError("Only next or previous browser tab is supported.")
    backend = backend or _WindowsKeys()
    target = backend.foreground()
    if not target or target[2].lower() not in _BROWSERS:
        raise NavigationError("Put the browser in front first. I won't send keys to another app.")
    backend.press_tab(direction, target)
    return "Sent the browser tab shortcut."


class _WindowsKeys:
    def __init__(self):
        if os.name != "nt":
            raise NavigationError("Browser shortcuts require Windows.")
        import ctypes
        from ctypes import wintypes
        self.ctypes, self.w = ctypes, wintypes
        self.user = ctypes.WinDLL("user32", use_last_error=True)
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.user.GetForegroundWindow.restype = wintypes.HWND
        self.user.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
        self.kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self.kernel.OpenProcess.restype = wintypes.HANDLE
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.kernel.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]

    def foreground(self):
        c, w = self.ctypes, self.w
        hwnd = self.user.GetForegroundWindow()
        if not hwnd:
            return None
        pid = w.DWORD()
        self.user.GetWindowThreadProcessId(hwnd, c.byref(pid))
        proc = self.kernel.OpenProcess(0x1000, False, pid.value)
        if not proc:
            return None
        try:
            path, size = c.create_unicode_buffer(32768), w.DWORD(32768)
            if not self.kernel.QueryFullProcessImageNameW(proc, 0, path, c.byref(size)):
                return None
            return (hwnd, pid.value, path.value.rsplit("\\", 1)[-1])
        finally:
            self.kernel.CloseHandle(proc)

    def press_tab(self, direction, target):
        c, w = self.ctypes, self.w
        class Mouse(c.Structure):
            _fields_ = [("dx", w.LONG), ("dy", w.LONG), ("mouseData", w.DWORD),
                        ("dwFlags", w.DWORD), ("time", w.DWORD), ("extra", c.c_size_t)]
        class Key(c.Structure):
            _fields_ = [("vk", w.WORD), ("scan", w.WORD), ("flags", w.DWORD),
                        ("time", w.DWORD), ("extra", c.c_size_t)]
        class Payload(c.Union):
            _fields_ = [("mi", Mouse), ("ki", Key)]
        class Input(c.Structure):
            _fields_ = [("type", w.DWORD), ("payload", Payload)]
        self.user.SendInput.argtypes = [w.UINT, c.POINTER(Input), c.c_int]
        self.user.SendInput.restype = w.UINT
        modifiers = [0x11] + ([0x10] if direction == "previous" else [])
        pairs = [(key, 0) for key in modifiers] + [(0x09, 0), (0x09, 2)] + [(key, 2) for key in reversed(modifiers)]
        events = (Input * len(pairs))(*(Input(1, Payload(ki=Key(key, 0, flags, 0, 0))) for key, flags in pairs))
        if self.foreground() != target:
            raise NavigationError("Browser focus changed. No shortcut sent.")
        sent = self.user.SendInput(len(events), events, c.sizeof(Input))
        if sent != len(events):
            # Release modifiers after partial injection, but never replay the tab action.
            releases = (Input * len(modifiers))(*(Input(1, Payload(ki=Key(key, 0, 2, 0, 0))) for key in modifiers))
            self.user.SendInput(len(releases), releases, c.sizeof(Input))
            raise NavigationError("Shortcut delivery is uncertain. I won't retry it.")
