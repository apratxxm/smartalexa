"""Windows single-instance guard, acquired before audio or model startup."""
from __future__ import annotations

import ctypes
import hashlib
from contextlib import contextmanager
from pathlib import Path
from ctypes import wintypes


class AlreadyRunning(RuntimeError):
    pass


@contextmanager
def single_instance(project: Path):
    # Scope to this checkout and Windows login session. The kernel closes the
    # handle after a crash; no stale lock file can prevent a later restart.
    identity = hashlib.sha256(str(project.resolve()).casefold().encode()).hexdigest()[:20]
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
    kernel.CreateMutexW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    handle = kernel.CreateMutexW(None, False, "Local\\MemoirAssistant-" + identity)
    error = ctypes.get_last_error()
    if not handle:
        raise ctypes.WinError(error)
    try:
        if error == 183:  # ERROR_ALREADY_EXISTS
            raise AlreadyRunning("Memoir is already running for this project")
        yield
    finally:
        kernel.CloseHandle(handle)
