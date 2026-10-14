import asyncio
from datetime import timedelta
from types import SimpleNamespace

from assistant import local_media


def snap(title="Song", artist="Artist", position=5000):
    return local_media.MediaSnapshot(title, artist, True, position_ms=position)


class FakeSession:
    source_app_user_model_id = "Spotify.exe"

    def __init__(self, snapshots, command_result=True):
        self.snapshots = iter(snapshots)
        self.current = next(self.snapshots)
        self.command_result = command_result
        self.commands = []

    def get_playback_info(self):
        return SimpleNamespace(playback_status=local_media.PLAYING)

    async def try_get_media_properties_async(self):
        return SimpleNamespace(title=self.current.title, artist=self.current.artist,
                               album_title=self.current.album_title)

    def get_timeline_properties(self):
        position = self.current.position_ms
        return None if position is None else SimpleNamespace(position=timedelta(milliseconds=position))

    async def try_skip_previous_async(self):
        self.commands.append("previous")
        self.current = next(self.snapshots, self.current)
        return self.command_result

    async def try_change_playback_position_async(self, position):
        self.commands.append(("restart", position))
        self.current = next(self.snapshots, self.current)
        return self.command_result


def install_sessions(monkeypatch, session, other=None):
    calls = []

    async def get_session():
        calls.append(True)
        # A deliberately changing manager result catches accidental reselection.
        return session if len(calls) == 1 or other is None else other

    monkeypatch.setattr(local_media, "_session", get_session)
    return calls


def test_previous_verifies_track_change_on_one_session(monkeypatch):
    target = FakeSession([snap(), snap("Previous", position=30000)])
    decoy = FakeSession([snap("Unrelated")])
    session_calls = install_sessions(monkeypatch, target, decoy)
    assert local_media.previous_track() is True
    assert target.commands == ["previous"]
    assert decoy.commands == []
    assert len(session_calls) == 1


def test_previous_accepts_verified_same_track_rewind_only(monkeypatch):
    target = FakeSession([snap(position=5000), snap(position=1000)])
    install_sessions(monkeypatch, target)
    assert local_media.previous_track() is True
    assert target.commands == ["previous"]

    no_op = FakeSession([snap(position=5000)])
    install_sessions(monkeypatch, no_op)
    assert local_media.previous_track() is False
    assert no_op.commands == ["previous"]


def test_previous_needs_identity_and_timeline_for_same_track_rewind(monkeypatch):
    missing = FakeSession([snap("")])
    install_sessions(monkeypatch, missing)
    assert local_media.previous_track() is False
    assert missing.commands == []

    no_timeline = FakeSession([snap(position=None)])
    install_sessions(monkeypatch, no_timeline)
    assert local_media.previous_track() is False
    assert no_timeline.commands == ["previous"]


def test_rejected_previous_command_is_not_success(monkeypatch):
    target = FakeSession([snap()], command_result=False)
    install_sessions(monkeypatch, target)
    assert local_media.previous_track() is False
    assert target.commands == ["previous"]


def test_restart_verifies_same_track_at_start_including_already_at_start(monkeypatch):
    target = FakeSession([snap(position=5000), snap(position=1000)])
    install_sessions(monkeypatch, target)
    assert local_media.restart() is True
    assert target.commands == [("restart", 0)]

    already_started = FakeSession([snap(position=0), snap(position=0)])
    install_sessions(monkeypatch, already_started)
    assert local_media.restart() is True
    assert already_started.commands == [("restart", 0)]


def test_restart_rejects_track_change_missing_position_and_noop(monkeypatch):
    changed = FakeSession([snap(), snap("Other", position=0)])
    install_sessions(monkeypatch, changed)
    assert local_media.restart() is False

    missing_position = FakeSession([snap(position=None)])
    install_sessions(monkeypatch, missing_position)
    assert local_media.restart() is False

    no_op = FakeSession([snap(position=5000)])
    install_sessions(monkeypatch, no_op)
    assert local_media.restart() is False


def test_transport_snapshot_converts_timedelta_and_fails_closed_nonfinite(monkeypatch):
    target = FakeSession([snap(position=12345)])
    install_sessions(monkeypatch, target)
    result = local_media._transport_snapshot()
    assert result.position_ms == 12345

    target.get_timeline_properties = lambda: SimpleNamespace(
        position=SimpleNamespace(total_seconds=lambda: float("inf")))
    result = local_media._transport_snapshot()
    assert result.position_ms is None


def test_transport_snapshot_preserves_identity_when_timeline_unavailable(monkeypatch):
    target = FakeSession([snap(position=None)])
    install_sessions(monkeypatch, target)
    result = local_media._transport_snapshot()
    assert (result.title, result.artist, result.position_ms) == ("Song", "Artist", None)


def test_hung_properties_or_command_timeout_without_duplicate_mutation(monkeypatch):
    target = FakeSession([snap()])
    calls = install_sessions(monkeypatch, target)
    async def hang_properties():
        await asyncio.Event().wait()
    monkeypatch.setattr(target, "try_get_media_properties_async", hang_properties)
    requested_timeouts = []

    def short_run(coro, timeout):
        requested_timeouts.append(timeout)
        try:
            return asyncio.run(asyncio.wait_for(coro, 0.01))
        except asyncio.TimeoutError:
            return False

    monkeypatch.setattr(local_media, "_run", short_run)
    assert local_media.previous_track() is False
    assert requested_timeouts == [3.0]
    assert len(calls) == 1
    assert target.commands == []

    target = FakeSession([snap()])
    calls = install_sessions(monkeypatch, target)
    async def hang_command():
        target.commands.append("previous")
        await asyncio.Event().wait()
    monkeypatch.setattr(target, "try_skip_previous_async", hang_command)
    requested_timeouts.clear()
    assert local_media.previous_track() is False
    assert requested_timeouts == [3.0]
    assert len(calls) == 1
    assert target.commands == ["previous"]
