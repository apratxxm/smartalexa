import pytest
from types import SimpleNamespace
from unittest.mock import Mock

from assistant import ducking


@pytest.fixture
def volume(monkeypatch, tmp_path):
    # "value" is Spotify's mixer session; "system" is Windows output.
    state = {"value": 0.85, "system": 0.9}

    def session(identifier, key):
        audio = SimpleNamespace(
            GetMasterVolume=lambda: state[key],
            SetMasterVolume=lambda value, _event: state.__setitem__(key, value))
        return SimpleNamespace(InstanceIdentifier=identifier, SimpleAudioVolume=audio)

    state["sessions"] = [session("spotify-session-1", "value")]
    state["new_session"] = session
    monkeypatch.setattr(ducking, "_spotify_sessions", lambda: state["sessions"])
    monkeypatch.setattr(ducking, "get_system_volume", lambda: state["system"])
    monkeypatch.setattr(ducking, "set_system_volume", lambda v: state.__setitem__("system", v))
    monkeypatch.setattr(ducking, "_STATE_FILE", tmp_path / "duck.json")
    ducking._LEASE_ORIGINAL = None
    ducking._LEASE_ACTIVE = False
    yield state
    ducking._LEASE_ORIGINAL = None
    ducking._LEASE_ACTIVE = False


@pytest.mark.parametrize("requested,actual", [(0.5, 0.5), (0.5, 0.509), (0.5, 0.51)])
def test_system_volume_set_is_verified_on_same_endpoint(monkeypatch, requested, actual):
    endpoint = SimpleNamespace(
        SetMasterVolumeLevelScalar=Mock(),
        GetMasterVolumeLevelScalar=Mock(return_value=actual),
    )
    acquire = Mock(return_value=endpoint)
    spotify = Mock(side_effect=AssertionError("Spotify must not be queried"))
    monkeypatch.setattr(ducking, "_endpoint", acquire)
    monkeypatch.setattr(ducking, "_spotify_sessions", spotify)

    assert ducking.set_system_volume(requested) is None

    acquire.assert_called_once_with()
    endpoint.SetMasterVolumeLevelScalar.assert_called_once_with(requested, None)
    endpoint.GetMasterVolumeLevelScalar.assert_called_once_with()
    spotify.assert_not_called()


def test_system_volume_mismatch_after_ignored_write_is_unknown(monkeypatch):
    endpoint = SimpleNamespace(
        SetMasterVolumeLevelScalar=Mock(),
        GetMasterVolumeLevelScalar=Mock(return_value=0.2),
    )
    monkeypatch.setattr(ducking, "_endpoint", Mock(return_value=endpoint))
    spotify = Mock(side_effect=AssertionError("Spotify must not be queried"))
    monkeypatch.setattr(ducking, "_spotify_sessions", spotify)

    with pytest.raises(ducking.SystemVolumeVerificationError,
                       match="Windows output volume change could not be verified"):
        ducking.set_system_volume(0.8)

    endpoint.SetMasterVolumeLevelScalar.assert_called_once_with(0.8, None)
    endpoint.GetMasterVolumeLevelScalar.assert_called_once_with()
    spotify.assert_not_called()


@pytest.mark.parametrize("level", [True, False, float("nan"), float("inf"), -float("inf"), None, "0.5"])
def test_system_volume_invalid_input_is_rejected_before_endpoint_acquisition(monkeypatch, level):
    acquire = Mock(side_effect=AssertionError("endpoint must not be acquired"))
    monkeypatch.setattr(ducking, "_endpoint", acquire)
    with pytest.raises(ValueError, match="finite real number"):
        ducking.set_system_volume(level)
    acquire.assert_not_called()


@pytest.mark.parametrize("level,target", [(-2.0, 0.0), (2.0, 1.0)])
def test_system_volume_clamps_finite_out_of_range_inputs(monkeypatch, level, target):
    endpoint = SimpleNamespace(
        SetMasterVolumeLevelScalar=Mock(),
        GetMasterVolumeLevelScalar=Mock(return_value=target),
    )
    monkeypatch.setattr(ducking, "_endpoint", Mock(return_value=endpoint))
    ducking.set_system_volume(level)
    endpoint.SetMasterVolumeLevelScalar.assert_called_once_with(target, None)
    endpoint.GetMasterVolumeLevelScalar.assert_called_once_with()


@pytest.mark.parametrize("readback", [float("nan"), float("inf"), -0.01, 1.01, None, "0.5", True, 0.6])
def test_system_volume_invalid_or_different_readback_is_verification_error(monkeypatch, readback):
    endpoint = SimpleNamespace(
        SetMasterVolumeLevelScalar=Mock(),
        GetMasterVolumeLevelScalar=Mock(return_value=readback),
    )
    monkeypatch.setattr(ducking, "_endpoint", Mock(return_value=endpoint))
    with pytest.raises(ducking.SystemVolumeVerificationError,
                       match="Windows output volume change could not be verified"):
        ducking.set_system_volume(0.5)
    endpoint.SetMasterVolumeLevelScalar.assert_called_once_with(0.5, None)
    endpoint.GetMasterVolumeLevelScalar.assert_called_once_with()


def test_system_volume_endpoint_errors_propagate_without_retry(monkeypatch):
    setter_error = OSError("setter failed")
    endpoint = SimpleNamespace(
        SetMasterVolumeLevelScalar=Mock(side_effect=setter_error),
        GetMasterVolumeLevelScalar=Mock(),
    )
    monkeypatch.setattr(ducking, "_endpoint", Mock(return_value=endpoint))
    with pytest.raises(OSError, match="setter failed"):
        ducking.set_system_volume(0.5)
    endpoint.SetMasterVolumeLevelScalar.assert_called_once_with(0.5, None)
    endpoint.GetMasterVolumeLevelScalar.assert_not_called()

    endpoint.SetMasterVolumeLevelScalar.reset_mock()
    endpoint.SetMasterVolumeLevelScalar.side_effect = None
    read_error = OSError("read failed")
    endpoint.GetMasterVolumeLevelScalar.side_effect = read_error
    with pytest.raises(OSError, match="read failed"):
        ducking.set_system_volume(0.5)
    endpoint.SetMasterVolumeLevelScalar.assert_called_once_with(0.5, None)
    endpoint.GetMasterVolumeLevelScalar.assert_called_once_with()


def test_lease_restores_on_release(volume):
    lease = ducking.DuckLease(20)
    assert lease.acquire() is True
    assert volume["value"] == pytest.approx(0.2)
    lease.release()
    assert volume["value"] == pytest.approx(0.85)


def test_nested_lease_cannot_take_ownership(volume):
    first, second = ducking.DuckLease(20), ducking.DuckLease(10)
    assert first.acquire() is True
    assert second.acquire() is False
    second.release()
    assert volume["value"] == pytest.approx(0.2)
    first.release()
    assert volume["value"] == pytest.approx(0.85)


def test_expired_lease_restores(volume, monkeypatch):
    now = {"t": 100.0}
    monkeypatch.setattr(ducking.time, "monotonic", lambda: now["t"])
    lease = ducking.DuckLease(20, timeout_seconds=2)
    lease.acquire()
    now["t"] = 103.0
    assert lease.expired() is True
    lease.release()
    assert volume["value"] == pytest.approx(0.85)


def test_context_exception_restores(volume):
    with pytest.raises(RuntimeError):
        with ducking.DuckLease(20):
            raise RuntimeError("boom")
    assert volume["value"] == pytest.approx(0.85)


def test_duck_never_increases_an_already_quiet_session(volume):
    volume["value"] = 0.05
    with ducking.ducked(20) as (was, now):
        assert (was, now) == (5, 5)
        assert volume["value"] == pytest.approx(0.05)
    assert volume["value"] == pytest.approx(0.05)


def test_mixed_session_levels_restore_each_exact_value(volume):
    volume["quiet"] = 0.08
    volume["sessions"].append(volume["new_session"]("spotify-session-2", "quiet"))

    with ducking.ducked(20) as levels:
        assert levels == (85, 20)
        assert volume["value"] == pytest.approx(0.2)
        assert volume["quiet"] == pytest.approx(0.08)

    assert volume["value"] == pytest.approx(0.85)
    assert volume["quiet"] == pytest.approx(0.08)


def test_lease_restoration_does_not_touch_new_sessions_or_user_changes(volume):
    lease = ducking.DuckLease(20)
    lease.acquire()
    volume["value"] = 0.55  # User moved this mixer slider while the lease was active.
    volume["new"] = 0.95
    volume["sessions"].append(volume["new_session"]("spotify-session-new", "new"))

    lease.release()

    assert volume["value"] == pytest.approx(0.55)
    assert volume["new"] == pytest.approx(0.95)


def test_legacy_master_state_is_discarded_without_volume_write(volume, monkeypatch):
    import json

    writes = []
    monkeypatch.setattr(ducking, "set_system_volume", writes.append)
    ducking._STATE_FILE.write_text(json.dumps({"original": 1.0, "target": "system"}))

    ducking.repair_orphaned_duck()

    assert writes == []
    assert not ducking._STATE_FILE.exists()


def test_nested_capture_keeps_true_recovery_volume(volume):
    import json
    lease = ducking.DuckLease(20)
    lease.acquire()
    with ducking.ducked(20):
        state_file = json.loads(ducking._STATE_FILE.read_text())
        assert state_file["sessions"][0]["original"] == pytest.approx(0.85)
    assert volume['value'] == pytest.approx(0.2)
    lease.release()
    assert volume['value'] == pytest.approx(0.85)


def test_failed_restore_preserves_recovery_record(volume, monkeypatch):
    lease = ducking.DuckLease(20)
    lease.acquire()
    def fail(value, _event):
        raise OSError('audio device unavailable')
    monkeypatch.setattr(ducking._spotify_sessions()[0].SimpleAudioVolume,
                        'SetMasterVolume', fail)
    with pytest.raises(OSError):
        lease.release()
    assert ducking._STATE_FILE.exists()
    assert not ducking._LEASE_ACTIVE
    assert not lease._owned
    ducking.repair_orphaned_duck()
    assert ducking._STATE_FILE.exists()


def test_scoped_restore_failure_keeps_recoverable_state(volume, monkeypatch):
    session = volume["sessions"][0]
    original_write = session.SimpleAudioVolume.SetMasterVolume
    with pytest.raises(OSError, match="audio device unavailable"):
        with ducking.ducked(20):
            monkeypatch.setattr(session.SimpleAudioVolume, "SetMasterVolume",
                                lambda *_: (_ for _ in ()).throw(OSError("audio device unavailable")))
    assert ducking._STATE_FILE.exists()
    assert not ducking._DUCK_RESTORE_STACK
    monkeypatch.setattr(session.SimpleAudioVolume, "SetMasterVolume", original_write)
    ducking.repair_orphaned_duck()
    assert volume["value"] == pytest.approx(0.85)
    assert not ducking._STATE_FILE.exists()


def test_vanished_session_does_not_block_next_lease(volume):
    lease = ducking.DuckLease(20)
    lease.acquire()
    volume["sessions"].clear()
    lease.release()
    assert not ducking._STATE_FILE.exists()

    volume["new"] = 0.9
    volume["sessions"].append(volume["new_session"]("spotify-session-new", "new"))
    next_lease = ducking.DuckLease(20)
    assert next_lease.acquire() is True
    next_lease.release()
    assert volume["new"] == pytest.approx(0.9)


@pytest.mark.parametrize("use_lease", [True, False])
def test_partial_restore_failure_can_be_repaired(volume, monkeypatch, use_lease):
    import json

    session = volume["sessions"][0]
    original_write = session.SimpleAudioVolume.SetMasterVolume
    writes = [0]

    def partial_restore(value, event):
        writes[0] += 1
        if writes[0] > 1:
            raise OSError("restore unavailable")
        original_write(value, event)

    with pytest.raises(OSError, match="restore unavailable"):
        if use_lease:
            lease = ducking.DuckLease(20)
            lease.acquire()
            monkeypatch.setattr(session.SimpleAudioVolume, "SetMasterVolume", partial_restore)
            lease.release()
        else:
            with ducking.ducked(20):
                monkeypatch.setattr(session.SimpleAudioVolume, "SetMasterVolume", partial_restore)
    saved = json.loads(ducking._STATE_FILE.read_text())["sessions"]
    assert saved[0]["ducked"] == pytest.approx(volume["value"])
    assert 0.2 < volume["value"] < 0.85
    monkeypatch.setattr(session.SimpleAudioVolume, "SetMasterVolume", original_write)
    ducking.repair_orphaned_duck()
    assert volume["value"] == pytest.approx(0.85)
    assert not ducking._STATE_FILE.exists()


def test_partial_acquire_failure_retains_actual_written_level(volume, monkeypatch):
    import json

    volume["other"] = 0.75
    volume["sessions"].append(volume["new_session"]("spotify-session-2", "other"))
    originals = [session.SimpleAudioVolume.SetMasterVolume for session in volume["sessions"]]
    writes = [0]

    def writing(index):
        def write(value, event):
            writes[0] += 1
            if writes[0] > 1:
                raise OSError("duck unavailable")
            originals[index](value, event)
        return write

    for index, session in enumerate(volume["sessions"]):
        monkeypatch.setattr(session.SimpleAudioVolume, "SetMasterVolume", writing(index))
    with pytest.raises(OSError, match="duck unavailable"):
        ducking.DuckLease(20).acquire()

    saved = json.loads(ducking._STATE_FILE.read_text())["sessions"]
    assert len(saved) == 1
    changed_key = "value" if saved[0]["id"] == "spotify-session-1" else "other"
    assert saved[0]["ducked"] == pytest.approx(volume[changed_key])
    assert not ducking._LEASE_ACTIVE

    for session, original_write in zip(volume["sessions"], originals):
        monkeypatch.setattr(session.SimpleAudioVolume, "SetMasterVolume", original_write)
    ducking.repair_orphaned_duck()
    assert volume["value"] == pytest.approx(0.85)
    assert volume["other"] == pytest.approx(0.75)
    assert not ducking._STATE_FILE.exists()


def test_command_volume_survives_early_lease_release(volume):
    lease = ducking.DuckLease(20)
    lease.acquire()
    ducking.set_command_volume(0.5)
    lease.release()
    assert volume['system'] == pytest.approx(0.5)
    assert volume['value'] == pytest.approx(0.85)


def test_command_volume_survives_scoped_duck_restore(volume):
    with ducking.ducked(20):
        ducking.set_command_volume(0.7)
    assert volume['system'] == pytest.approx(0.7)
    assert volume['value'] == pytest.approx(0.85)


def test_duck_lowers_only_spotify_never_windows_output(volume):
    with ducking.ducked(20):
        assert volume['value'] == pytest.approx(0.2)
        assert volume['system'] == pytest.approx(0.9)
    lease = ducking.DuckLease(20)
    lease.acquire()
    assert volume['system'] == pytest.approx(0.9)
    lease.release()
    assert volume['value'] == pytest.approx(0.85)


def test_no_spotify_session_means_no_duck(volume, monkeypatch):
    monkeypatch.setattr(ducking, "_spotify_sessions", lambda: [])
    with ducking.ducked(20) as levels:
        assert levels == (None, None)
    assert volume['system'] == pytest.approx(0.9)
    assert not ducking._STATE_FILE.exists()
