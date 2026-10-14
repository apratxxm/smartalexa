"""Verify observed playback, not just successful mutation responses."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import spotipy

from assistant.resolution import Candidate
from assistant.spotify import Spotify, PlaybackVerificationError


def make_spotify(monkeypatch, states):
    monkeypatch.setattr('assistant.spotify.time.sleep', lambda _: None)
    obj = Spotify.__new__(Spotify)
    obj.device = lambda: 'laptop'
    obj._call = Mock()
    obj.sp = SimpleNamespace(current_playback=Mock(side_effect=states),
                             start_playback=Mock(), pause_playback=Mock(), volume=Mock())
    return obj


def state(device='laptop', uri='spotify:track:wanted', playing=True, volume=40):
    return {'device': {'id': device, 'volume_percent': volume},
            'item': {'uri': uri}, 'is_playing': playing}


def test_track_waits_for_correct_device_and_item(monkeypatch):
    obj = make_spotify(monkeypatch, [state(device='other'), state(uri='wrong'), state()])
    obj.play_candidate(Candidate('spotify:track:wanted', 'Wanted'))
    assert obj.sp.current_playback.call_count == 3
    obj._call.assert_called_once()


def test_accepted_api_request_can_still_fail_verification(monkeypatch):
    obj = make_spotify(monkeypatch, [state(uri='wrong')] * 4)
    with pytest.raises(PlaybackVerificationError):
        obj.play_candidate(Candidate('spotify:track:wanted', 'Wanted'))
    obj._call.assert_called_once()


def test_volume_on_other_device_is_not_used(monkeypatch):
    obj = make_spotify(monkeypatch, [state(device='other')])
    assert obj.get_volume() is None


def test_pause_already_paused_on_target_is_idempotent(monkeypatch):
    obj = make_spotify(monkeypatch, [state(playing=False)] * 2)
    obj.pause()
    obj._call.assert_not_called()


def test_volume_must_reach_requested_value(monkeypatch):
    obj = make_spotify(monkeypatch, [state(volume=20), state(volume=40)])
    obj.set_volume(40)
    obj._call.assert_called_once_with(obj.sp.volume, 40)


def test_navigation_is_never_repeated_during_verification(monkeypatch):
    obj = make_spotify(monkeypatch, [state(), state(), state(uri='next')])
    obj.sp.next_track = Mock()
    obj.next()
    obj._call.assert_called_once_with(obj.sp.next_track)


def test_previous_can_restart_current_track(monkeypatch):
    before, after = state(), state()
    before['progress_ms'], after['progress_ms'] = 9000, 100
    obj = make_spotify(monkeypatch, [before, after])
    obj.sp.previous_track = Mock()
    obj.previous()
    obj._call.assert_called_once_with(obj.sp.previous_track)


def test_restricted_pause_is_an_explicit_targeted_failure(monkeypatch):
    obj = make_spotify(monkeypatch, [state(playing=True)])
    obj._call.side_effect = spotipy.SpotifyException(
        403, -1, "Player command failed: Restriction violated"
    )
    with pytest.raises(spotipy.SpotifyException):
        obj.pause()


def test_missing_playback_next_is_an_explicit_failure(monkeypatch):
    obj = make_spotify(monkeypatch, [None])
    obj.sp.next_track = Mock()
    with pytest.raises(PlaybackVerificationError):
        obj.next()
