import pytest

from assistant import computer_tools, spotify_jam
from assistant.core.adapters import jam


def test_budget_passed_to_helper(monkeypatch):
    seen = []
    monkeypatch.setattr(jam.time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(computer_tools, "start_spotify_jam", lambda **kw: seen.append(kw))
    assert jam.START_JAM.execute({}, lambda: 0.01).state == "observed"
    assert seen == [{"deadline": 100.01}]


def test_helper_timeout_and_deadline_are_bounded(monkeypatch):
    seen = []
    monkeypatch.setattr(computer_tools.time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(computer_tools, "_run_spotify_helper", lambda args, **kw: seen.append((args, kw)))
    computer_tools.start_spotify_jam(deadline=100.01)
    assert seen[0][0] == ["--run", "--deadline=100.01"]
    assert seen[0][1]["timeout"] == pytest.approx(0.01)


def test_snapshot_expires_before_click():
    now = [0.0]
    class Backend:
        clicks = []
        def snapshot(self):
            now[0] = 0.02
            return spotify_jam.Snapshot((spotify_jam.Control("Start a Jam", "button", True, False),), ())
        def click(self, name):
            self.clicks.append(name)
    backend = Backend()
    with pytest.raises(spotify_jam.SpotifyJamError):
        spotify_jam.drive_spotify_jam(backend, clock=lambda: now[0], deadline=0.01)
    assert backend.clicks == []


def test_expired_before_dispatch(monkeypatch):
    calls = []
    monkeypatch.setattr(computer_tools, "start_spotify_jam", lambda **kw: calls.append(kw))
    assert jam.START_JAM.execute({}, lambda: 0).state == "failed"
    assert not calls
