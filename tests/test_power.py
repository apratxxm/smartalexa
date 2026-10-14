import pytest

from assistant.power import (
    EXECUTION_REQUIRED, POLL_INTERVAL_SECONDS, REFRESH_INTERVAL_SECONDS,
    SYSTEM_REQUIRED, KeepAwakeGuard, REASON_CONTEXT, SYSTEM_POWER_STATUS,
    _REASON_DETAIL, _REASON_UNION,
)


@pytest.fixture(autouse=True)
def _capture_power_logger(caplog, monkeypatch):
    """Capture this child logger directly even when Memoir disables propagation."""
    import logging
    logger = logging.getLogger("memoir.power")
    monkeypatch.setattr(logger, "handlers", [caplog.handler])
    monkeypatch.setattr(logger, "propagate", False)


class FakePower:
    def __init__(self, ac=1):
        self.ac = ac
        self.handles = set()
        self.calls = []
        self.fail = None
        self.fail_close_handles = set()
        self.next_handle = 0

    def _check(self, name):
        if self.fail == name:
            raise OSError(name)

    def ac_power(self):
        self._check("status")
        self.calls.append(("status",))
        return self.ac

    def create(self):
        self._check("create")
        self.next_handle += 1
        handle = self.next_handle
        self.handles.add(handle)
        self.calls.append(("create", handle))
        return handle

    def set(self, handle, kind):
        self._check(f"set-{kind}")
        self.calls.append(("set", handle, kind))

    def clear(self, handle, kind):
        self._check(f"clear-{kind}")
        self.calls.append(("clear", handle, kind))

    def close(self, handle):
        self.calls.append(("close", handle))
        if handle in self.fail_close_handles:
            raise OSError("close")
        self._check("close")
        self.handles.discard(handle)


def test_enabled_ac_sets_only_system_and_execution_requests():
    api = FakePower()
    guard = KeepAwakeGuard(True, backend=api, clock=lambda: 0, is_windows=True)
    guard.update()
    assert [c[2] for c in api.calls if c[0] == "set"] == [SYSTEM_REQUIRED, EXECUTION_REQUIRED]
    assert all(c[-1] not in (2, 4) for c in api.calls if c[0] == "set")
    guard.close()
    assert api.handles == set()


def test_battery_and_unknown_power_release_request():
    api = FakePower()
    now = [0.0]
    guard = KeepAwakeGuard(True, backend=api, clock=lambda: now[0], is_windows=True)
    guard.update()
    for level in (0, 255):
        api.ac = level
        now[0] += POLL_INTERVAL_SECONDS
        guard.update()
        assert not api.handles
    assert len([c for c in api.calls if c[0] == "create"]) == 1


def test_disabled_and_non_windows_do_not_construct_or_call_backend():
    class Exploding:
        def __getattr__(self, name):
            raise AssertionError(name)
    for enabled, windows in ((False, True), (True, False)):
        guard = KeepAwakeGuard(enabled, backend=Exploding(), is_windows=windows)
        guard.update()
        guard.close()


def test_refresh_replaces_then_closes_old_handle_and_balances_counts():
    api = FakePower()
    now = [0.0]
    guard = KeepAwakeGuard(True, backend=api, clock=lambda: now[0], is_windows=True)
    guard.update()
    now[0] = REFRESH_INTERVAL_SECONDS
    guard.update()
    assert len([c for c in api.calls if c[0] == "create"]) == 2
    assert len([c for c in api.calls if c[0] == "close"]) == 1
    assert api.handles == {2}
    guard.close()
    assert not api.handles
    assert len([c for c in api.calls if c[0] == "clear"]) == 4


def test_failures_release_cleanly_and_do_not_escape_update():
    for failure in ("status", "create", f"set-{EXECUTION_REQUIRED}",
                    f"clear-{SYSTEM_REQUIRED}", "close"):
        api = FakePower()
        now = [0.0]
        guard = KeepAwakeGuard(True, backend=api, clock=lambda: now[0], is_windows=True)
        guard.update()
        api.fail = failure
        now[0] += max(POLL_INTERVAL_SECONDS, REFRESH_INTERVAL_SECONDS)
        guard.update()
        guard.close()
        assert guard._handle is None
        # A native CloseHandle failure cannot be repaired by this process; the
        # manager must still drop logical ownership and never retry/flood logs.
        if failure != "close":
            assert api.handles == set()


def test_context_exit_releases_even_when_body_raises():
    api = FakePower()
    try:
        with KeepAwakeGuard(True, backend=api, clock=lambda: 0, is_windows=True) as guard:
            guard.update()
            raise ValueError("capture failed")
    except ValueError:
        pass
    assert guard._closed and guard._handle is None and not api.handles


def test_abi_layout_matches_windows_pointer_width_contract():
    import ctypes
    import struct
    assert SYSTEM_POWER_STATUS.ACLineStatus.offset == 0
    assert struct.calcsize("P") == 4 or struct.calcsize("P") == 8
    assert ctypes.sizeof(SYSTEM_POWER_STATUS) == 12
    pointer_size = struct.calcsize("P")
    assert ctypes.sizeof(ctypes.c_void_p) == pointer_size
    assert ctypes.sizeof(REASON_CONTEXT) == (32 if pointer_size == 8 else 24)
    detailed_base = REASON_CONTEXT.Reason.offset + _REASON_UNION.Detailed.offset
    assert detailed_base + _REASON_DETAIL.LocalizedReasonModule.offset == 8
    assert detailed_base + _REASON_DETAIL.LocalizedReasonId.offset == 8 + pointer_size
    assert detailed_base + _REASON_DETAIL.ReasonStringCount.offset == 12 + pointer_size
    assert detailed_base + _REASON_DETAIL.ReasonStrings.offset == 16 + pointer_size


def test_repeated_updates_do_not_poll_before_interval():
    api = FakePower()
    now = [3.0]
    guard = KeepAwakeGuard(True, backend=api, clock=lambda: now[0], is_windows=True)
    guard.update()
    for _ in range(20):
        guard.update()
    assert len([c for c in api.calls if c[0] == "status"]) == 1
    now[0] += POLL_INTERVAL_SECONDS - .01
    guard.update()
    assert len([c for c in api.calls if c[0] == "status"]) == 1


def test_partial_acquisition_cleans_up_already_set_request():
    api = FakePower()
    api.fail = f"set-{EXECUTION_REQUIRED}"
    guard = KeepAwakeGuard(True, backend=api, clock=lambda: 0, is_windows=True)
    guard.update()
    assert not api.handles and guard._handle is None
    assert ("clear", 1, SYSTEM_REQUIRED) in api.calls


def test_explicit_pause_releases_and_resume_reacquires_only_after_poll_interval():
    api = FakePower()
    now = [0.0]
    guard = KeepAwakeGuard(True, backend=api, clock=lambda: now[0], is_windows=True)
    guard.update()
    guard.update(listening=False)
    assert not api.handles
    created = len([c for c in api.calls if c[0] == "create"])
    guard.update(listening=True)
    assert len([c for c in api.calls if c[0] == "create"]) == created + 1
    assert api.handles == {2}


def test_windows_dll_exports_and_prototypes_are_bound_without_invocation(monkeypatch):
    import ctypes
    from ctypes import wintypes

    class NativeFunction:
        argtypes = None
        restype = None

        def __call__(self, *args):
            raise AssertionError("prototype inspection must not call native APIs")

    exports = ("PowerCreateRequest", "PowerSetRequest", "PowerClearRequest",
               "CloseHandle", "GetSystemPowerStatus")
    kernel32 = type("FakeKernel32", (), {name: NativeFunction() for name in exports})()
    loaded = []

    def fake_dll(name, **kwargs):
        loaded.append(name)
        assert name == "kernel32"
        assert kwargs == {"use_last_error": True}
        return kernel32

    monkeypatch.setattr(ctypes, "WinDLL", fake_dll, raising=False)
    from assistant.power import _WindowsPowerApi
    _WindowsPowerApi()
    assert loaded == ["kernel32"]
    assert kernel32.GetSystemPowerStatus.argtypes == [ctypes.POINTER(SYSTEM_POWER_STATUS)]
    assert kernel32.GetSystemPowerStatus.restype is wintypes.BOOL
    assert kernel32.PowerCreateRequest.argtypes == [ctypes.POINTER(REASON_CONTEXT)]
    assert kernel32.PowerCreateRequest.restype is ctypes.c_void_p


def test_failure_warning_is_once_per_episode_and_successful_acquisition_resets(caplog):
    import logging
    api = FakePower()
    now = [0.0]
    guard = KeepAwakeGuard(True, backend=api, clock=lambda: now[0], is_windows=True)
    api.fail = "create"
    with caplog.at_level(logging.WARNING, logger="memoir.power"):
        guard.update()
        now[0] += POLL_INTERVAL_SECONDS
        guard.update()
    warnings = [r for r in caplog.records if r.name == "memoir.power" and r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "PowerCreateRequest/PowerSetRequest" in warnings[0].message
    assert "OSError" in warnings[0].message

    api.fail = None
    now[0] += POLL_INTERVAL_SECONDS
    guard.update()
    assert guard.held and guard.available
    api.fail = "status"
    now[0] += POLL_INTERVAL_SECONDS
    with caplog.at_level(logging.WARNING, logger="memoir.power"):
        guard.update()
    assert len([r for r in caplog.records if r.name == "memoir.power" and
                r.levelno == logging.WARNING]) == 2
    assert not guard.held


def test_close_failure_disables_future_acquisition_and_is_not_retried(caplog):
    import logging
    api = FakePower()
    guard = KeepAwakeGuard(True, backend=api, clock=lambda: 0, is_windows=True)
    guard.update()
    api.fail = "close"
    with caplog.at_level(logging.WARNING, logger="memoir.power"):
        guard.update(listening=False)
        guard.update(listening=True)
        guard.update(listening=True)
        guard.close()
    assert not guard.available and not guard.held
    assert len([r for r in caplog.records if r.name == "memoir.power" and
                r.levelno == logging.WARNING]) == 1
    assert len([c for c in api.calls if c[0] == "close"]) == 1


def test_old_handle_close_failure_discards_replacement_without_retrying_old():
    api = FakePower()
    now = [0.0]
    guard = KeepAwakeGuard(True, backend=api, clock=lambda: now[0], is_windows=True)
    guard.update()
    api.fail_close_handles.add(1)
    now[0] += REFRESH_INTERVAL_SECONDS
    guard.update()
    assert not guard.available and not guard.held
    assert api.handles == {1}  # The failed native close cannot be repaired here.
    assert [c for c in api.calls if c[0] == "create"] == [("create", 1), ("create", 2)]
    assert [c for c in api.calls if c == ("close", 1)] == [("close", 1)]
    assert [c for c in api.calls if c == ("close", 2)] == [("close", 2)]
    now[0] += REFRESH_INTERVAL_SECONDS
    guard.update()
    guard.close()
    assert not guard.held
    assert len([c for c in api.calls if c[0] == "close"]) == 2
    assert len([c for c in api.calls if c[0] == "create"]) == 2


def test_successful_refresh_does_not_emit_transition_info(caplog):
    import logging
    api = FakePower()
    now = [0.0]
    guard = KeepAwakeGuard(True, backend=api, clock=lambda: now[0], is_windows=True)
    with caplog.at_level(logging.INFO, logger="memoir.power"):
        guard.update()
        now[0] += REFRESH_INTERVAL_SECONDS
        guard.update()
    infos = [r.message for r in caplog.records if r.name == "memoir.power" and
             r.levelno == logging.INFO]
    assert infos == ["AC listening power request acquired; this does not guarantee sleep prevention"]
    guard.close()
