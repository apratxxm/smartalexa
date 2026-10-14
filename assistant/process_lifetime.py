"""Windows process-tree cleanup for the supervisor's owned children."""
from __future__ import annotations

import ctypes
import os
from ctypes import wintypes

_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
_JOB_OBJECT_LIMIT_BREAKAWAY_OK = 0x00000800
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
MANAGED_PROCESS_TREE_ENV = "MEMOIR_MANAGED_PROCESS_TREE"
_job_handle = None


class _IO_COUNTERS(ctypes.Structure):
    _fields_ = [(name, ctypes.c_ulonglong) for name in (
        "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
        "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]


class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_longlong),
        ("PerJobUserTimeLimit", ctypes.c_longlong),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo", _IO_COUNTERS),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


def install_process_lifetime(*, kernel32=None, platform: str | None = None) -> bool:
    """Assign this process to a kill-on-close job before it launches children.

    On non-Windows platforms this is a no-op. Optional seams allow pure fake
    Win32 tests; the native DLL is loaded only for a real Windows call.
    """
    global _job_handle
    if _job_handle is not None:
        os.environ[MANAGED_PROCESS_TREE_ENV] = "1"
        return True
    platform = os.name if platform is None else platform
    if platform != "nt":
        return False
    native_kernel = kernel32 is None
    if native_kernel:
        if os.name != "nt":
            raise RuntimeError("native Win32 job setup requires Windows")
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    kernel32.SetInformationJobObject.restype = wintypes.BOOL
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    handle = kernel32.CreateJobObjectW(None, None)
    if not handle:
        raise _win32_error(kernel32, "CreateJobObjectW", native_kernel)
    info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    info.BasicLimitInformation.LimitFlags = (
        _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE | _JOB_OBJECT_LIMIT_BREAKAWAY_OK)
    try:
        if not kernel32.SetInformationJobObject(
                handle, _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
                ctypes.byref(info), ctypes.sizeof(info)):
            raise _win32_error(kernel32, "SetInformationJobObject", native_kernel)
        if not kernel32.AssignProcessToJobObject(handle, kernel32.GetCurrentProcess()):
            raise _win32_error(kernel32, "AssignProcessToJobObject", native_kernel)
    except Exception:
        kernel32.CloseHandle(handle)
        raise
    # Keep exactly one non-inheritable handle alive for this process lifetime.
    # Closing a successful self-job handle would terminate this process.
    _job_handle = handle
    os.environ[MANAGED_PROCESS_TREE_ENV] = "1"
    return True


def _win32_error(kernel32, operation: str, native: bool) -> OSError:
    if native:
        return ctypes.WinError(ctypes.get_last_error())
    get_error = getattr(kernel32, "get_last_error", None)
    code = get_error() if callable(get_error) else 0
    return OSError(code, f"{operation} failed (Win32 error {code})")
