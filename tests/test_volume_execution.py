from unittest.mock import Mock

import pytest

from assistant.main import Assistant
from assistant import ducking
from assistant.intent import Action, Intent, parse
from assistant.telemetry import Event


@pytest.mark.parametrize("text,current,target", [
    ("increase volume to 50%", 80, 50),
    ("decrease volume by 50%", 80, 40),
    ("increase volume by 50%", 40, 60),
    ("turn volume down by 10 points", 80, 70),
    ("volume up", 95, 100), ("volume down", 5, 0),
])
def test_volume_target(text, current, target, monkeypatch):
    app = Assistant.__new__(Assistant)
    app.spotify = Mock()
    writes = []
    monkeypatch.setattr(ducking, 'get_system_volume', lambda: current / 100)
    monkeypatch.setattr(ducking, 'set_command_volume', writes.append)
    app._set_volume(parse(text))
    assert writes == [target / 100]
    app.spotify.set_volume.assert_not_called()
    app.spotify.get_volume.assert_not_called()


def test_missing_current_never_guesses_relative_volume(monkeypatch):
    app = Assistant.__new__(Assistant); app.spotify = Mock()
    monkeypatch.setattr(ducking, 'get_system_volume', Mock(side_effect=RuntimeError('no endpoint')))
    system_write = Mock()
    monkeypatch.setattr(ducking, 'set_command_volume', system_write)
    with pytest.raises(RuntimeError): app._set_volume(parse("decrease volume by 50%"))
    app.spotify.set_volume.assert_not_called()
    system_write.assert_not_called()


def test_volume_execution_never_calls_spotify(monkeypatch):
    app = Assistant.__new__(Assistant)
    app.spotify = Mock()
    app.feedback = Mock()
    write = Mock()
    monkeypatch.setattr(ducking, 'set_command_volume', write)
    event = Event()
    app._execute(parse('set volume to 50 percent'), event)
    write.assert_called_once_with(0.5)
    app.spotify.assert_not_called()
    app.spotify.set_volume.assert_not_called()
    app.spotify.get_volume.assert_not_called()
    assert event.spotify_action == ''


@pytest.mark.parametrize('spoken,target,query', [
    ('play red by seedhe maut full volume', 100, 'red by seedhe maut'),
    ('play red by seedhe maut at half volume', 50, 'red by seedhe maut'),
    ('play red by seedhe maut at 20% volume', 20, 'red by seedhe maut'),
    ('play red by seedhe maut volume 75 percent', 75, 'red by seedhe maut'),
])
def test_compound_play_volume_is_not_in_track_query(spoken, target, query):
    intent = parse(spoken)
    assert intent.query == query
    assert intent.volume_after_play == target


@pytest.mark.parametrize('play_fails', [False, True])
def test_artist_compound_changes_system_volume_only_after_playback(monkeypatch, play_fails):
    app = Assistant.__new__(Assistant)
    app.spotify = Mock()
    app.spotify.ensure_client_running.return_value = False
    app._release_paused_mic_for_playback = Mock()
    app.feedback = Mock()
    writes = Mock()
    monkeypatch.setattr(ducking, 'set_command_volume', writes)
    intent = Intent(Action.PLAY_ARTIST, query='Seedhe Maut', volume_after_play=20)
    if play_fails:
        app.spotify.play_candidate.side_effect = RuntimeError('playback failed')
        with pytest.raises(RuntimeError, match='playback failed'):
            app._execute(intent, Event())
        writes.assert_not_called()
    else:
        app._execute(intent, Event())
        writes.assert_called_once_with(0.2)
    app.spotify.set_volume.assert_not_called()
