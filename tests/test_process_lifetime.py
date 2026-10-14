import ctypes
import os

import pytest

from assistant import process_lifetime as lifetime


class FakeFunction:
    def __init__(self, callback):
        self.callback = callback
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        return self.callback(*args)


class FakeKernel:
    def __init__(self, *, fail=None):
        self.fail = fail
        self.error_code = 5
        self.closed = []
        self.assigned = []
        self.info = None
        self.CreateJobObjectW = FakeFunction(lambda *_: 77)

        def set_info(handle, klass, info, size):
            self.info = (handle, klass, ctypes.cast(
                info, ctypes.POINTER(lifetime._JOBOBJECT_EXTENDED_LIMIT_INFORMATION)
            ).contents, size)
            return self.fail != "set"

        self.SetInformationJobObject = FakeFunction(set_info)
        self.GetCurrentProcess = FakeFunction(lambda: 88)
        self.AssignProcessToJobObject = FakeFunction(
            lambda job, proc: self.assigned.append((job, proc)) or self.fail != "assign")
        self.CloseHandle = FakeFunction(lambda h: self.closed.append(h) or True)
        self.get_last_error = lambda: self.error_code


@pytest.fixture(autouse=True)
def isolated_job_cache(monkeypatch):
    monkeypatch.setattr(lifetime, "_job_handle", None)
    monkeypatch.delenv(lifetime.MANAGED_PROCESS_TREE_ENV, raising=False)


def test_non_windows_is_noop():
    assert lifetime.install_process_lifetime(platform="posix") is False


@pytest.mark.parametrize("failure", ["create", "set", "assign"])
def test_failures_raise_and_close_only_unassigned_handle(monkeypatch, failure):
    kernel = FakeKernel(fail=failure)
    if failure == "create":
        kernel.CreateJobObjectW = FakeFunction(lambda *_: 0)
    with pytest.raises(OSError):
        lifetime.install_process_lifetime(kernel32=kernel, platform="nt")
    assert kernel.closed == ([] if failure == "create" else [77])
    assert lifetime._job_handle is None
    assert lifetime.MANAGED_PROCESS_TREE_ENV not in os.environ


def test_failure_reports_win32_code():
    kernel = FakeKernel(fail="set")
    with pytest.raises(OSError, match="Win32 error 5") as error:
        lifetime.install_process_lifetime(kernel32=kernel, platform="nt")
    assert error.value.errno == 5


def test_success_sets_only_kill_on_close_and_caches_without_closing():
    kernel = FakeKernel()
    assert lifetime.install_process_lifetime(kernel32=kernel, platform="nt") is True
    assert kernel.info[1] == lifetime._JOB_OBJECT_EXTENDED_LIMIT_INFORMATION
    assert kernel.info[3] == ctypes.sizeof(lifetime._JOBOBJECT_EXTENDED_LIMIT_INFORMATION)
    assert kernel.info[2].BasicLimitInformation.LimitFlags == 0x2800
    assert kernel.assigned == [(77, 88)]
    assert kernel.closed == []
    assert os.environ[lifetime.MANAGED_PROCESS_TREE_ENV] == "1"
    assert kernel.CreateJobObjectW.argtypes == [ctypes.c_void_p, lifetime.wintypes.LPCWSTR]
    assert kernel.SetInformationJobObject.argtypes[-1] is lifetime.wintypes.DWORD
    assert lifetime.install_process_lifetime(kernel32=kernel, platform="nt") is True
    assert kernel.closed == []

