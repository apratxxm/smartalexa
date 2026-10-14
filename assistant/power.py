"""Process-owned Windows power requests for plugged-in listening."""
from __future__ import annotations

import ctypes
import logging
import os
import time
from ctypes import wintypes


SYSTEM_REQUIRED = 1
EXECUTION_REQUIRED = 3
POLL_INTERVAL_SECONDS = 5.0
REFRESH_INTERVAL_SECONDS = 30.0
REASON = "Memoir microphone listening while plugged in"
log = logging.getLogger("memoir.power")


class SYSTEM_POWER_STATUS(ctypes.Structure):
    _fields_ = [
        ("ACLineStatus", ctypes.c_uint8),
        ("BatteryFlag", ctypes.c_uint8),
        ("BatteryLifePercent", ctypes.c_uint8),
        ("SystemStatusFlag", ctypes.c_uint8),
        ("BatteryLifeTime", ctypes.c_uint32),
        ("BatteryFullLifeTime", ctypes.c_uint32),
    ]


class _REASON_DETAIL(ctypes.Structure):
    _fields_ = [
        ("LocalizedReasonModule", ctypes.c_void_p),
        ("LocalizedReasonId", ctypes.c_uint32),
        ("ReasonStringCount", ctypes.c_uint32),
        ("ReasonStrings", ctypes.POINTER(wintypes.LPWSTR)),
    ]


class _REASON_UNION(ctypes.Union):
    _fields_ = [("Detailed", _REASON_DETAIL), ("SimpleReasonString", wintypes.LPWSTR)]


class REASON_CONTEXT(ctypes.Structure):
    _anonymous_ = ("Reason",)
    _fields_ = [("Version", ctypes.c_uint32), ("Flags", ctypes.c_uint32), ("Reason", _REASON_UNION)]


def _reason_context() -> tuple[REASON_CONTEXT, ctypes.Array]:
    # POWER_REQUEST_CONTEXT_SIMPLE_STRING = 0x1, POWER_REQUEST_CONTEXT_VERSION = 0.
    reason = ctypes.create_unicode_buffer(REASON)
    context = REASON_CONTEXT()
    context.Version = 0
    context.Flags = 0x1
    context.SimpleReasonString = ctypes.cast(reason, wintypes.LPWSTR)
    return context, reason


class _WindowsPowerApi:
    """Lazy native API wrapper; importing this module never calls Windows APIs."""

    def __init__(self):
        self.kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel32.PowerCreateRequest.argtypes = [ctypes.POINTER(REASON_CONTEXT)]
        self.kernel32.PowerCreateRequest.restype = ctypes.c_void_p
        self.kernel32.PowerSetRequest.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.kernel32.PowerSetRequest.restype = wintypes.BOOL
        self.kernel32.PowerClearRequest.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.kernel32.PowerClearRequest.restype = wintypes.BOOL
        self.kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        self.kernel32.CloseHandle.restype = wintypes.BOOL
        self.kernel32.GetSystemPowerStatus.argtypes = [ctypes.POINTER(SYSTEM_POWER_STATUS)]
        self.kernel32.GetSystemPowerStatus.restype = wintypes.BOOL

    def ac_power(self):
        status = SYSTEM_POWER_STATUS()
        if not self.kernel32.GetSystemPowerStatus(ctypes.byref(status)):
            raise OSError(ctypes.get_last_error(), "GetSystemPowerStatus failed")
        return int(status.ACLineStatus)

    def create(self):
        context, keepalive = _reason_context()
        handle = self.kernel32.PowerCreateRequest(ctypes.byref(context))
        # Keep local strings alive through the native call; Windows copies this context.
        del keepalive
        invalid = ctypes.c_void_p(-1).value
        if handle is None or handle == invalid:
            raise OSError(ctypes.get_last_error(), "PowerCreateRequest failed")
        return handle

    def set(self, handle, request_type):
        if not self.kernel32.PowerSetRequest(handle, request_type):
            raise OSError(ctypes.get_last_error(), "PowerSetRequest failed")

    def clear(self, handle, request_type):
        if not self.kernel32.PowerClearRequest(handle, request_type):
            raise OSError(ctypes.get_last_error(), "PowerClearRequest failed")

    def close(self, handle):
        if not self.kernel32.CloseHandle(handle):
            raise OSError(ctypes.get_last_error(), "CloseHandle failed")


class KeepAwakeGuard:
    """Maintain SystemRequired and ExecutionRequired only while AC is confirmed."""

    def __init__(self, enabled: bool, *, backend=None, clock=time.monotonic,
                 is_windows: bool | None = None):
        self.enabled = bool(enabled)
        self._is_windows = os.name == "nt" if is_windows is None else is_windows
        self._backend = backend
        self._clock = clock
        self._handle = None
        self._last_poll = float("-inf")
        self._last_refresh = float("-inf")
        self._closed = False
        self._disabled = False
        self._failure_logged = False
        self._listening = None

    @property
    def held(self) -> bool:
        """Whether this process currently owns two successful requests."""
        return self._handle is not None

    @property
    def available(self) -> bool:
        """Whether this guard can still attempt requests on this platform."""
        return self.enabled and self._is_windows and not self._closed and not self._disabled

    def _failure(self, operation, exc):
        if not self._failure_logged:
            log.warning("Power request %s failed (%s); listening continues without the guard",
                        operation, type(exc).__name__)
            self._failure_logged = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def _api(self):
        if self._backend is None:
            self._backend = _WindowsPowerApi()
        return self._backend

    def _release(self, *, transition_log=True):
        handle, self._handle = self._handle, None
        if handle is None:
            return True
        api = self._api()
        succeeded = True
        for kind in (EXECUTION_REQUIRED, SYSTEM_REQUIRED):
            try:
                api.clear(handle, kind)
            except Exception as exc:
                succeeded = False
                self._failure("PowerClearRequest", exc)
        try:
            api.close(handle)
        except Exception as exc:
            succeeded = False
            self._disabled = True
            self._failure("CloseHandle", exc)
        if succeeded and transition_log:
            log.info("AC listening power request released")
        return succeeded

    def _acquire(self, *, transition_log=True):
        api = self._api()
        handle = None
        succeeded = []
        try:
            handle = api.create()
            for kind in (SYSTEM_REQUIRED, EXECUTION_REQUIRED):
                api.set(handle, kind)
                succeeded.append(kind)
        except Exception as exc:
            self._failure("PowerCreateRequest/PowerSetRequest", exc)
            if handle is not None:
                for kind in reversed(succeeded):
                    try:
                        api.clear(handle, kind)
                    except Exception as cleanup_exc:
                        self._failure("PowerClearRequest", cleanup_exc)
                try:
                    api.close(handle)
                except Exception as cleanup_exc:
                    self._disabled = True
                    self._failure("CloseHandle", cleanup_exc)
            return False
        self._handle = handle
        self._last_refresh = self._clock()
        self._failure_logged = False
        if transition_log:
            log.info("AC listening power request acquired; this does not guarantee sleep prevention")
        return True

    def update(self, *, listening: bool = True):
        """Poll at most every five seconds; power API failures release and retry later."""
        if not self.available:
            # A refresh may have installed a replacement just before cleanup
            # of its predecessor failed. Make one cleanup attempt, then forget it.
            if self._disabled and self._handle is not None:
                self._release(transition_log=False)
            return
        if not listening:
            self._listening = False
            self._release()
            # Do not make a power-status call on every paused capture frame.
            self._last_poll = self._clock()
            return
        if self._listening is False:
            # Resume is a deliberate transition, so reacquire immediately.
            self._last_poll = float("-inf")
        self._listening = True
        now = self._clock()
        if now - self._last_poll < POLL_INTERVAL_SECONDS:
            return
        self._last_poll = now
        try:
            on_ac = self._api().ac_power() == 1
        except Exception as exc:
            self._failure("GetSystemPowerStatus", exc)
            self._release()
            return
        if not on_ac:
            self._release()
            return
        if self._handle is None:
            self._acquire()
        elif now - self._last_refresh >= REFRESH_INTERVAL_SECONDS:
            old = self._handle
            self._handle = None
            if self._acquire(transition_log=False):
                # Replacement is live before the old one is cleared and closed.
                temporary = self._handle
                self._handle = old
                self._release(transition_log=False)
                self._handle = temporary
                if self._disabled:
                    self._release(transition_log=False)
            else:
                self._handle = old
                self._release()

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._release()
