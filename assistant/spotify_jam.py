"""Bounded, Spotify-only UI Automation for showing a Jam invitation."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re
import sys
import time
from typing import Iterable

from .external_apps import open_external_target


class SpotifyJamError(RuntimeError):
    """A safe, actionable refusal for an unverified Jam invitation state."""


@dataclass(frozen=True)
class Control:
    name: str
    kind: str
    enabled: bool = True
    offscreen: bool = False
    value: str | None = None


@dataclass(frozen=True)
class Snapshot:
    controls: tuple[Control, ...]
    text: tuple[str, ...] = ()


INVITE_PROMPT = "Invite friends to your Jam"
INVITE_INSTRUCTIONS = "Copy and share the link with your friends, or ask them to scan the QR code."
SUCCESS = "Jam QR is on screen."
_JAM_BUTTON_SUFFIXES = ("'s jam", "’s jam")
SEARCH_CONTROL_NAME = "What do you want to play?"
SEARCH_FILTERS = ("All", "Songs", "Playlists", "Albums")
_SEARCH_FILTER_KINDS = ("button", "tabitem", "radiobutton", "toggle")


def _input_types(ctypes_module, pointer_type=None):
    """Build the pointer-width-correct Win32 INPUT layout (testable off Windows)."""
    c = ctypes_module
    pointer_type = pointer_type or c.c_size_t

    class MOUSEINPUT(c.Structure):
        _fields_ = [("dx", c.c_int32), ("dy", c.c_int32), ("mouseData", c.c_uint32),
                    ("dwFlags", c.c_uint32), ("time", c.c_uint32),
                    ("dwExtraInfo", pointer_type)]

    class KEYBDINPUT(c.Structure):
        _fields_ = [("wVk", c.c_uint16), ("wScan", c.c_uint16), ("dwFlags", c.c_uint32),
                    ("time", c.c_uint32), ("dwExtraInfo", pointer_type)]

    class HARDWAREINPUT(c.Structure):
        _fields_ = [("uMsg", c.c_uint32), ("wParamL", c.c_uint16), ("wParamH", c.c_uint16)]

    class INPUTUNION(c.Union):
        _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("hi", HARDWAREINPUT)]

    class INPUT(c.Structure):
        _fields_ = [("type", c.c_uint32), ("payload", INPUTUNION)]

    return KEYBDINPUT, INPUTUNION, INPUT


def _fold(value: str) -> str:
    return " ".join(value.casefold().replace("’", "'").split())


def _visible(snapshot: Snapshot) -> tuple[Control, ...]:
    return tuple(control for control in snapshot.controls if not control.offscreen)


def _named(controls: Iterable[Control], name: str, *, kinds: tuple[str, ...] = ()) -> list[Control]:
    wanted = _fold(name)
    return [control for control in controls
            if _fold(control.name) == wanted and (not kinds or control.kind.casefold() in kinds)]


def _jam_heading(controls: Iterable[Control]) -> bool:
    # The persistent bottom button is not a heading. Only text/heading controls
    # that identify an owner's Jam establish context for a generic Invite button.
    return any(control.kind.casefold() in {"text", "heading"}
               and _fold(control.name).endswith(_JAM_BUTTON_SUFFIXES)
               for control in controls)


def _invite_evidence(snapshot: Snapshot) -> tuple[bool, bool, list[Control]]:
    visible = _visible(snapshot)
    all_text = [_fold(value) for value in snapshot.text]
    all_text.extend(_fold(control.name) for control in visible)
    has_prompt = _fold(INVITE_PROMPT) in all_text
    has_instructions = _fold(INVITE_INSTRUCTIONS) in all_text
    copy_buttons = _named(visible, "Copy link", kinds=("button",))
    return has_prompt, has_instructions, copy_buttons


def _validate_search(query: str, kind: str) -> str:
    if (not isinstance(query, str) or not query.strip() or len(query) > 400
            or re.search(r"[\x00-\x1f\x7f]", query)):
        raise SpotifyJamError("Search text must be 1–400 characters without control characters.")
    if not isinstance(kind, str) or kind not in {"track", "album"}:
        raise SpotifyJamError("Choose a song or album search.")
    return query.strip()


def _search_verified(snapshot: Snapshot, query: str) -> bool:
    controls = _visible(snapshot)
    search_boxes = _named(controls, SEARCH_CONTROL_NAME, kinds=("combobox",))
    if len(search_boxes) != 1 or not search_boxes[0].enabled:
        return False
    if _fold(search_boxes[0].value or "") != _fold(query):
        return False

    for label in SEARCH_FILTERS:
        matches = _named(controls, label, kinds=_SEARCH_FILTER_KINDS)
        if len(matches) != 1 or not matches[0].enabled:
            return False

    ignored = {_fold(SEARCH_CONTROL_NAME), *(_fold(label) for label in SEARCH_FILTERS)}
    result_text = " ".join(
        control.name for control in controls
        if control.kind.casefold() in {"text", "listitem", "dataitem"}
        and _fold(control.name) not in ignored
    ).casefold()
    found_tokens = set(re.findall(r"\w+", result_text))
    wanted_tokens = set(re.findall(r"\w+", query.casefold())) - {"by", "the"}
    if not wanted_tokens:
        wanted_tokens = set(re.findall(r"\w+", query.casefold()))
    return bool(wanted_tokens) and wanted_tokens <= found_tokens


def drive_spotify_search(backend, query: str, kind: str, *, timeout: float = 10.0,
                         clock=time.monotonic, sleep=time.sleep) -> str:
    """Submit once to Spotify's named native search and await matching results."""
    query = _validate_search(query, kind)
    backend.submit_search(query)
    deadline = clock() + timeout
    while clock() < deadline:
        if _search_verified(backend.snapshot(), query):
            return "Spotify desktop search results are visible for that query."
        sleep(min(0.15, max(0.0, deadline - clock())))
    raise SpotifyJamError("Spotify did not expose fresh results matching that query. The search may have been submitted; check Spotify before retrying.")


def drive_spotify_jam(backend, *, timeout: float = 12.0,
                      clock=time.monotonic, sleep=time.sleep) -> str:
    """Drive a small, one-way state machine using snapshot/click backend methods.

    The backend must return visible named controls and must revalidate the target
    Spotify window and foreground before each click. Tests inject a fake backend.
    """
    deadline = clock() + timeout
    acted: set[str] = set()
    while clock() < deadline:
        snapshot = backend.snapshot()
        controls = _visible(snapshot)
        has_prompt, has_instructions, copy_buttons = _invite_evidence(snapshot)
        if has_prompt or has_instructions:
            if not (has_prompt and has_instructions):
                raise SpotifyJamError("Spotify's Jam invitation panel is incomplete. Check the app before trying again.")
            if len(copy_buttons) > 1:
                raise SpotifyJamError("Spotify shows multiple Copy link controls in the Jam invitation panel. I stopped without choosing one.")
            if len(copy_buttons) != 1 or not copy_buttons[0].enabled:
                raise SpotifyJamError("Spotify's Jam invitation text is visible, but its Copy link button is unavailable.")
            return SUCCESS

        headings = [control for control in controls
                    if control.kind.casefold() in {"text", "heading"}
                    and _fold(control.name).endswith(_JAM_BUTTON_SUFFIXES)]
        if len(headings) > 1:
            raise SpotifyJamError("Spotify shows multiple Jam headings; I stopped without changing anything.")
        has_heading = bool(headings)
        invite_matches = _named(controls, "Invite", kinds=("button",))
        if invite_matches and not has_heading:
            raise SpotifyJamError("Spotify shows Invite without an identifiable Jam heading. I stopped without clicking it.")
        if has_heading:
            if invite_matches:
                if not invite_matches[0].enabled:
                    raise SpotifyJamError("The Jam Invite button is disabled. Check Spotify's account and Jam availability.")
                action = "Invite"
            else:
                jam_buttons = [control for control in controls
                               if control.kind.casefold() in {"button", "toggle"}
                               and _fold(control.name).endswith(_JAM_BUTTON_SUFFIXES)]
                if len(jam_buttons) > 1:
                    raise SpotifyJamError("Spotify shows multiple Jam controls; I stopped without choosing one.")
                if not jam_buttons:
                    action = None
                elif not jam_buttons[0].enabled:
                    raise SpotifyJamError("Spotify's Jam panel control is disabled. Check that the app is signed in.")
                else:
                    action = jam_buttons[0].name
        else:
            jam_buttons = [control for control in controls
                           if control.kind.casefold() in {"button", "toggle"}
                           and _fold(control.name).endswith(_JAM_BUTTON_SUFFIXES)]
            if len(jam_buttons) > 1:
                raise SpotifyJamError("Spotify shows multiple Jam controls; I stopped without choosing one.")
            if jam_buttons:
                if not jam_buttons[0].enabled:
                    raise SpotifyJamError("Spotify's existing Jam control is disabled. Check the app.")
                action = jam_buttons[0].name
                if action not in acted:
                    backend.click(action)
                    acted.add(action)
                    continue
                sleep(min(0.15, max(0.0, deadline - clock())))
                continue
            start_matches = _named(controls, "Start a Jam", kinds=("button",))
            if len(start_matches) > 1:
                raise SpotifyJamError("Spotify shows multiple Start a Jam buttons; I stopped without choosing one.")
            if start_matches:
                if not start_matches[0].enabled:
                    raise SpotifyJamError("Start a Jam is disabled. Check that Spotify is signed in and Jam is available for this account.")
                action = "Start a Jam"
            else:
                action = None
                if any(_named(controls, "Invite", kinds=("button",))):
                    raise SpotifyJamError("Spotify's Jam state is unclear. I won't click Invite without a Jam heading.")
                connect = _named(controls, "Connect to a device", kinds=("button", "toggle"))
                if len(connect) > 1:
                    raise SpotifyJamError("Spotify shows multiple Connect controls; I stopped without choosing one.")
                if connect:
                    if not connect[0].enabled:
                        raise SpotifyJamError("Spotify's Connect panel control is disabled. Check that the app is signed in.")
                    action = connect[0].name

        if action is not None and action not in acted:
            backend.click(action)
            acted.add(action)
            continue
        # A click may have changed the layout asynchronously. Keep observing, but
        # never repeat a mutation when the same layout remains visible.
        sleep(min(0.15, max(0.0, deadline - clock())))

    raise SpotifyJamError("Spotify did not show a verified Jam invitation in time. Check the app before trying again; a Jam may already have started.")


def _spotify_pids() -> set[int]:
    import psutil
    return {process.pid for process in psutil.process_iter(["name"])
            if (process.info.get("name") or "").casefold() == "spotify.exe"}


class WindowsSpotifyBackend:
    """UIA backend tied to one visible Spotify.exe top-level window."""

    _BUTTON = 50000
    _TEXT = 50020
    _HYPERLINK = 50005
    _COMBOBOX = 50003
    _LISTITEM = 50007
    _RADIOBUTTON = 50013
    _TABITEM = 50019
    _DATAITEM = 50029
    _INVOKE_PATTERN = 10000
    _TOGGLE_PATTERN = 10015
    _VALUE_PATTERN = 10002

    def __init__(self, *, clock=time.monotonic, sleep=time.sleep):
        if os.name != "nt":
            raise SpotifyJamError("Spotify desktop controls require Windows.")
        import ctypes
        from ctypes import wintypes
        self.ctypes, self.wintypes = ctypes, wintypes
        self.user32 = ctypes.WinDLL("user32", use_last_error=True)
        self.user32.GetForegroundWindow.restype = wintypes.HWND
        self.user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
        self.user32.GetWindowThreadProcessId.restype = wintypes.DWORD
        for name in ("IsIconic", "IsWindowVisible", "SetForegroundWindow"):
            getattr(self.user32, name).argtypes = [wintypes.HWND]
            getattr(self.user32, name).restype = wintypes.BOOL
        self.user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
        self.user32.GetWindowTextLengthW.restype = ctypes.c_int
        self.user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
        self.user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
        self.user32.GetWindowRect.restype = wintypes.BOOL
        self.user32.SetCursorPos.argtypes = [ctypes.c_int, ctypes.c_int]
        self.user32.SetCursorPos.restype = wintypes.BOOL
        self.user32.mouse_event.argtypes = [wintypes.DWORD, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD, ctypes.c_size_t]
        self.clock, self.sleep = clock, sleep
        self.hwnd = None
        self.pid = None
        self._activated = False
        try:
            from comtypes.client import CreateObject, GetModule
            GetModule("UIAutomationCore.dll")
            from comtypes.gen.UIAutomationClient import IUIAutomation
            self.automation = CreateObject("UIAutomationClient.CUIAutomation", interface=IUIAutomation)
        except Exception as exc:
            raise SpotifyJamError("Windows UI Automation is unavailable. Open Spotify and try again.") from exc

    def _windows(self, pids: set[int]) -> list[tuple[int, int]]:
        c, w = self.ctypes, self.wintypes
        rows: list[tuple[int, int]] = []
        callback_type = getattr(c, "WINFUNCTYPE", c.CFUNCTYPE)(w.BOOL, w.HWND, w.LPARAM)
        self.user32.EnumWindows.argtypes = [callback_type, w.LPARAM]
        self.user32.EnumWindows.restype = w.BOOL
        @callback_type
        def visit(hwnd, _):
            pid = w.DWORD()
            self.user32.GetWindowThreadProcessId(hwnd, c.byref(pid))
            if (pid.value in pids and self.user32.IsWindowVisible(hwnd)
                    and self.user32.GetWindowTextLengthW(hwnd)):
                rows.append((int(hwnd), int(pid.value)))
            return True
        self.user32.EnumWindows(visit, 0)
        return rows

    def _ensure_window(self) -> None:
        if self.hwnd is not None:
            if self.hwnd not in [hwnd for hwnd, _ in self._windows({self.pid})]:
                raise SpotifyJamError("Spotify's window changed during the request. Check the app before trying again.")
            return
        deadline = self.clock() + 12.0
        pids = _spotify_pids()
        if not pids:
            try:
                open_external_target("spotify:")
            except OSError as exc:
                raise SpotifyJamError("Spotify could not be opened through its registered app link.") from exc
        while self.clock() < deadline:
            pids = _spotify_pids()
            windows = self._windows(pids) if pids else []
            if len(windows) == 1:
                self.hwnd, self.pid = windows[0]
                return
            if len(windows) > 1:
                raise SpotifyJamError("Spotify has multiple visible windows. Leave one Spotify window open and try again.")
            self.sleep(0.2)
        raise SpotifyJamError("Spotify's visible desktop window was not found. Open the app and sign in, then try again.")

    def _focus(self) -> None:
        self._ensure_window()
        c = self.ctypes
        if not self._activated:
            if self.user32.IsIconic(self.hwnd):
                self.user32.ShowWindow(self.hwnd, 9)  # SW_RESTORE
            self.user32.SetForegroundWindow(self.hwnd)
        if int(self.user32.GetForegroundWindow() or 0) != self.hwnd:
            raise SpotifyJamError("Spotify focus changed or could not be acquired. No desktop control was activated.")
        pid = self.wintypes.DWORD()
        self.user32.GetWindowThreadProcessId(self.user32.GetForegroundWindow(), c.byref(pid))
        if pid.value != self.pid:
            raise SpotifyJamError("Spotify focus changed. No desktop control was activated.")
        self._activated = True

    def snapshot(self) -> Snapshot:
        self._focus()
        try:
            root = self.automation.ElementFromHandle(self.hwnd)
            descendants = root.FindAll(4, self.automation.CreateTrueCondition())  # TreeScope_Descendants
            if descendants.Length > 1200:
                raise SpotifyJamError("Spotify's accessible UI is too large to inspect safely.")
            controls, text = [], []
            for index in range(descendants.Length):
                element = descendants.GetElement(index)
                name = (element.CurrentName or "").strip()
                if not name or element.CurrentIsOffscreen:
                    continue
                kind_id = element.CurrentControlType
                kind = {
                    self._BUTTON: "button", self._TEXT: "text", self._HYPERLINK: "hyperlink",
                    self._COMBOBOX: "combobox",
                    self._RADIOBUTTON: "radiobutton", self._TABITEM: "tabitem",
                    self._LISTITEM: "listitem", self._DATAITEM: "dataitem",
                }.get(kind_id, "other")
                value = None
                if name == SEARCH_CONTROL_NAME and kind == "combobox":
                    try:
                        from comtypes.gen.UIAutomationClient import IUIAutomationValuePattern
                        value_pattern = element.GetCurrentPattern(self._VALUE_PATTERN).QueryInterface(IUIAutomationValuePattern)
                        value = value_pattern.CurrentValue
                    except Exception:
                        value = None
                if kind != "other":
                    controls.append(Control(name, kind, bool(element.CurrentIsEnabled), False, value))
                text.append(name)
            return Snapshot(tuple(controls), tuple(text))
        except SpotifyJamError:
            raise
        except Exception as exc:
            raise SpotifyJamError("Spotify's desktop controls could not be read. Confirm the app is open and try again.") from exc

    def click(self, name: str) -> None:
        self._focus()
        visible = _visible(self.snapshot())
        matches = [control for control in visible
                   if _fold(control.name) == _fold(name)
                   and control.kind.casefold() in {"button", "toggle"}]
        if len(matches) != 1 or not matches[0].enabled:
            raise SpotifyJamError("Spotify's named Jam control changed or became ambiguous. No click was sent.")
        root = self.automation.ElementFromHandle(self.hwnd)
        descendants = root.FindAll(4, self.automation.CreateTrueCondition())
        elements = []
        for index in range(descendants.Length):
            element = descendants.GetElement(index)
            if (_fold((element.CurrentName or "").strip()) == _fold(name)
                    and not element.CurrentIsOffscreen and element.CurrentIsEnabled
                    and element.CurrentControlType == self._BUTTON):
                elements.append(element)
        if len(elements) != 1:
            raise SpotifyJamError("Spotify's named Jam control changed or became ambiguous. No click was sent.")
        self._focus()
        element = elements[0]
        pattern_id = self._TOGGLE_PATTERN if _fold(name) == _fold("Connect to a device") else self._INVOKE_PATTERN
        from comtypes.gen.UIAutomationClient import IUIAutomationInvokePattern, IUIAutomationTogglePattern
        if pattern_id == self._TOGGLE_PATTERN:
            Pattern, method_name = IUIAutomationTogglePattern, "Toggle"
        else:
            Pattern, method_name = IUIAutomationInvokePattern, "Invoke"
        typed_pattern = None
        try:
            typed_pattern = element.GetCurrentPattern(pattern_id).QueryInterface(Pattern)
        except Exception:
            pass
        if typed_pattern is None:
            self._click_fresh_rectangle(name)
        else:
            try:
                getattr(typed_pattern, method_name)()
            except Exception as exc:
                raise SpotifyJamError("Spotify's Jam control response is uncertain. Check the app before trying again; I won't repeat the click.") from exc
        self._focus()

    def _search_elements(self):
        self._focus()
        root = self.automation.ElementFromHandle(self.hwnd)
        root_hwnd = int(root.CurrentNativeWindowHandle or 0)
        if root_hwnd not in {0, self.hwnd}:
            raise SpotifyJamError("Spotify's search tree no longer belongs to its verified window.")
        descendants = root.FindAll(4, self.automation.CreateTrueCondition())
        matches = []
        for index in range(descendants.Length):
            element = descendants.GetElement(index)
            if ((element.CurrentName or "").strip() == SEARCH_CONTROL_NAME
                    and element.CurrentControlType == self._COMBOBOX
                    and not element.CurrentIsOffscreen and element.CurrentIsEnabled):
                native_hwnd = int(element.CurrentNativeWindowHandle or 0)
                if native_hwnd not in {0, self.hwnd}:
                    raise SpotifyJamError("Spotify's search control belongs to a different native window.")
                matches.append(element)
        if len(matches) > 1:
            raise SpotifyJamError("Spotify shows multiple matching search boxes. I stopped without entering text.")
        return root, matches

    def _focused_in_search(self, search_element) -> bool:
        self._focus()
        try:
            if ((search_element.CurrentName or "").strip() != SEARCH_CONTROL_NAME
                    or search_element.CurrentControlType != self._COMBOBOX
                    or search_element.CurrentIsOffscreen or not search_element.CurrentIsEnabled
                    or int(search_element.CurrentNativeWindowHandle or 0) not in {0, self.hwnd}):
                return False
            focused = self.automation.GetFocusedElement()
            native_hwnd = int(focused.CurrentNativeWindowHandle or 0)
            if native_hwnd not in {0, self.hwnd}:
                return False
            walker = self.automation.ControlViewWalker
            current = focused
            for _ in range(16):
                if self.automation.CompareElements(current, search_element):
                    return True
                current = walker.GetParentElement(current)
                if current is None:
                    return False
                native_hwnd = int(current.CurrentNativeWindowHandle or 0)
                if native_hwnd not in {0, self.hwnd}:
                    return False
        except Exception:
            return False
        return False

    def _press_return(self, search_element) -> None:
        if not self._focused_in_search(search_element):
            raise SpotifyJamError("Spotify's search box did not retain verified focus. No key was sent.")
        c, w = self.ctypes, self.wintypes
        KEYBDINPUT, INPUTUNION, INPUT = _input_types(c)

        self.user32.SendInput.argtypes = [w.UINT, c.POINTER(INPUT), c.c_int]
        self.user32.SendInput.restype = w.UINT
        events = (INPUT * 2)(
            INPUT(1, INPUTUNION(ki=KEYBDINPUT(0x0D, 0, 0, 0, 0))),
            INPUT(1, INPUTUNION(ki=KEYBDINPUT(0x0D, 0, 0x0002, 0, 0))),
        )
        # Focus can change after the element check. Check again immediately
        # before injecting the single Return key pair.
        if not self._focused_in_search(search_element):
            raise SpotifyJamError("Spotify's search focus changed before submission. No key was sent.")
        sent = self.user32.SendInput(2, events, c.sizeof(INPUT))
        if sent != 2:
            if sent == 1:
                # Release a possibly pressed Return key; never replay key-down.
                release = (INPUT * 1)(INPUT(1, INPUTUNION(ki=KEYBDINPUT(0x0D, 0, 0x0002, 0, 0))))
                self.user32.SendInput(1, release, c.sizeof(INPUT))
            raise SpotifyJamError("Spotify's search submission is uncertain. Check the app before retrying.")

    def submit_search(self, query: str) -> None:
        query = _validate_search(query, "track")
        try:
            _root, matches = self._search_elements()
        except SpotifyJamError:
            raise
        except Exception as exc:
            raise SpotifyJamError("Spotify's native search box could not be read.") from exc
        if not matches:
            raise SpotifyJamError("Spotify's named search box is unavailable. Open the Spotify desktop app and try again.")
        element = matches[0]
        try:
            from comtypes.gen.UIAutomationClient import IUIAutomationValuePattern
            value_pattern = element.GetCurrentPattern(self._VALUE_PATTERN).QueryInterface(IUIAutomationValuePattern)
            value_pattern.SetValue(query)
        except Exception as exc:
            raise SpotifyJamError("Spotify's search text update is uncertain. Check the app before retrying; Return was not sent.") from exc

        # Setting the value can open a suggestions panel or refresh the tree.
        # Resolve the exact ComboBox again before focusing or sending Return.
        try:
            _root, matches = self._search_elements()
        except SpotifyJamError:
            raise
        except Exception as exc:
            raise SpotifyJamError("Spotify's search box changed after text entry. No key was sent.") from exc
        if len(matches) != 1:
            raise SpotifyJamError("Spotify's search box changed after text entry. No key was sent.")
        element = matches[0]
        try:
            element.SetFocus()
        except Exception as exc:
            raise SpotifyJamError("Spotify's search box could not be focused. No key was sent.") from exc
        self._press_return(element)

    def _click_fresh_rectangle(self, name: str) -> None:
        """Use a fresh UIA rectangle only after rechecking foreground ownership."""
        self._focus()
        root = self.automation.ElementFromHandle(self.hwnd)
        descendants = root.FindAll(4, self.automation.CreateTrueCondition())
        matches = []
        for index in range(descendants.Length):
            element = descendants.GetElement(index)
            if (_fold((element.CurrentName or "").strip()) == _fold(name)
                    and not element.CurrentIsOffscreen and element.CurrentIsEnabled
                    and element.CurrentControlType == self._BUTTON):
                matches.append(element)
        if len(matches) != 1:
            raise SpotifyJamError("Spotify's named Jam control changed or became ambiguous. No click was sent.")
        target = matches[0]
        rect = target.CurrentBoundingRectangle
        left, top, right, bottom = (int(rect.left), int(rect.top), int(rect.right), int(rect.bottom))
        window = self.wintypes.RECT()
        if not self.user32.GetWindowRect(self.hwnd, self.ctypes.byref(window)):
            raise SpotifyJamError("Spotify's window bounds could not be verified.")
        left, top = max(left, window.left), max(top, window.top)
        right, bottom = min(right, window.right), min(bottom, window.bottom)
        if right <= left or bottom <= top:
            raise SpotifyJamError("Spotify's Jam control has no safe visible click area.")
        self._focus()
        if not self.user32.SetCursorPos((left + right) // 2, (top + bottom) // 2):
            raise SpotifyJamError("Spotify's Jam control could not be targeted safely.")
        self._focus()
        root_after = self.automation.ElementFromHandle(self.hwnd)
        if (not self.automation.CompareElements(root, root_after)
                or int(root_after.CurrentNativeWindowHandle or 0) not in {0, self.hwnd}):
            raise SpotifyJamError("Spotify's window changed before the click. No input was sent.")
        after_tree = root_after.FindAll(4, self.automation.CreateTrueCondition())
        current_targets = []
        for index in range(after_tree.Length):
            element = after_tree.GetElement(index)
            if (_fold((element.CurrentName or "").strip()) == _fold(name)
                    and not element.CurrentIsOffscreen and element.CurrentIsEnabled
                    and element.CurrentControlType == self._BUTTON
                    and self.automation.CompareElements(target, element)):
                current_targets.append(element)
        if len(current_targets) != 1:
            raise SpotifyJamError("Spotify's Jam control changed after targeting. No input was sent.")
        latest = current_targets[0].CurrentBoundingRectangle
        latest_window = self.wintypes.RECT()
        if not self.user32.GetWindowRect(self.hwnd, self.ctypes.byref(latest_window)):
            raise SpotifyJamError("Spotify's window bounds could not be reverified.")
        current_left, current_top = max(int(latest.left), latest_window.left), max(int(latest.top), latest_window.top)
        current_right, current_bottom = min(int(latest.right), latest_window.right), min(int(latest.bottom), latest_window.bottom)
        x, y = (left + right) // 2, (top + bottom) // 2
        if (current_right <= current_left or current_bottom <= current_top
                or not (current_left <= x < current_right and current_top <= y < current_bottom)):
            raise SpotifyJamError("Spotify's Jam control moved before the click. No input was sent.")
        self._focus()
        self.user32.mouse_event(0x0002, 0, 0, 0, 0)  # MOUSEEVENTF_LEFTDOWN
        self.user32.mouse_event(0x0004, 0, 0, 0, 0)  # MOUSEEVENTF_LEFTUP


def _run_native() -> dict[str, object]:
    try:
        message = drive_spotify_jam(WindowsSpotifyBackend())
        return {"ok": True, "message": message}
    except SpotifyJamError as exc:
        return {"ok": False, "message": str(exc)}
    except Exception:
        return {"ok": False, "message": "Spotify's Jam invite state could not be verified. Check the app before trying again."}


def _run_native_search(query: str, kind: str) -> dict[str, object]:
    try:
        query = _validate_search(query, kind)
        message = drive_spotify_search(WindowsSpotifyBackend(), query, kind, timeout=8.0)
        return {"ok": True, "message": message}
    except SpotifyJamError as exc:
        return {"ok": False, "message": str(exc)}
    except Exception:
        return {"ok": False, "message": "Spotify desktop search results could not be verified. Check the app before trying again."}


def main() -> int:
    args = sys.argv[1:]
    if args == ["--run"]:
        payload = _run_native()
    elif (len(args) == 4 and args[0] == "--search" and args[1] == "--kind"
          and args[3].startswith("--query=")):
        payload = _run_native_search(args[3][len("--query="):], args[2])
    else:
        return 2
    print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
