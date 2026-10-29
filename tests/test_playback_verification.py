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


def idle_spotify(monkeypatch, states, *, active=True):
    import socket
    monkeypatch.setattr('assistant.spotify.time.sleep', lambda _: None)
    obj = Spotify.__new__(Spotify)
    obj._cfg = SimpleNamespace(get=lambda key, default=None: False if key == 'spotify.local_transport' else default)
    obj.device = lambda: 'laptop'
    obj._device_name = socket.gethostname()
    obj.sp = SimpleNamespace(
        devices=Mock(return_value={'devices': [{'id': 'laptop', 'name': socket.gethostname(),
                                               'type': 'Computer', 'is_active': active}]}),
        current_playback=Mock(side_effect=states), start_playback=Mock())
    obj._client_running = Mock(return_value=True)
    opened = Mock()
    monkeypatch.setattr('assistant.spotify.open_external_target', opened)
    return obj, opened


def idle_state():
    return {'device': {'id': 'laptop', 'is_active': True}, 'item': None, 'is_playing': False}


def test_idle_track_uses_uri_before_web_api_and_verifies(monkeypatch):
    obj, opened = idle_spotify(monkeypatch, [idle_state(), state()])
    assert obj.play_candidate(Candidate('spotify:track:wanted', 'Wanted')) == 'Wanted'
    opened.assert_called_once_with('spotify:track:wanted')
    obj.sp.start_playback.assert_not_called()
    assert obj.last_playback['method'] == 'uri_handler'


def test_loaded_track_uses_uri_and_verifies(monkeypatch):
    obj, opened = idle_spotify(monkeypatch, [state()])
    obj.play_candidate(Candidate('spotify:track:wanted', 'Wanted'))
    opened.assert_called_once_with('spotify:track:wanted')
    obj.sp.start_playback.assert_not_called()
    assert obj.last_playback['method'] == 'uri_handler'


def test_device_probe_failure_keeps_web_api_path(monkeypatch):
    obj, opened = idle_spotify(monkeypatch, [state()])
    obj.sp.devices.side_effect = RuntimeError('probe')
    obj.play_candidate(Candidate('spotify:track:wanted', 'Wanted'))
    opened.assert_not_called()
    obj.sp.start_playback.assert_called_once()


def test_remote_device_never_uses_idle_uri(monkeypatch):
    obj, opened = idle_spotify(monkeypatch, [state()])
    obj.sp.devices.return_value['devices'][0]['name'] = 'Other computer'
    obj.play_candidate(Candidate('spotify:track:wanted', 'Wanted'))
    opened.assert_not_called()
    obj.sp.start_playback.assert_called_once()


def test_no_playback_state_with_active_remote_device_uses_web_api(monkeypatch):
    obj, opened = idle_spotify(monkeypatch, [None, state()])
    obj.sp.devices.return_value['devices'].append({'id': 'remote', 'name': 'Phone',
                                                   'type': 'Smartphone', 'is_active': True})
    obj.play_candidate(Candidate('spotify:track:wanted', 'Wanted'))
    opened.assert_not_called()
    obj.sp.start_playback.assert_called_once()


@pytest.mark.parametrize("uri", ["https://example.com/track", "spotify:track:", "spotify:track:abc?bad"])
def test_local_target_rejects_invalid_track_uri_before_dispatch(monkeypatch, uri):
    obj, opened = idle_spotify(monkeypatch, [])
    with pytest.raises(ValueError):
        obj.play_candidate(Candidate(uri, 'Wanted'))
    opened.assert_not_called()
    obj.sp.start_playback.assert_not_called()


def test_active_remote_blocks_local_uri_despite_local_idle_snapshot(monkeypatch):
    obj, opened = idle_spotify(monkeypatch, [idle_state(), state()])
    obj.sp.devices.return_value['devices'].append({'id': 'phone', 'name': 'Phone',
                                                   'type': 'Smartphone', 'is_active': True})
    obj.play_candidate(Candidate('spotify:track:wanted', 'Wanted'))
    opened.assert_not_called()
    obj.sp.start_playback.assert_called_once()


def test_idle_uri_verification_failure_never_replays(monkeypatch):
    obj, opened = idle_spotify(monkeypatch, [idle_state()] + [state(uri='wrong')] * 4)
    with pytest.raises(PlaybackVerificationError):
        obj.play_candidate(Candidate('spotify:track:wanted', 'Wanted'))
    opened.assert_called_once()
    obj.sp.start_playback.assert_not_called()


def test_idle_uri_rejects_late_read_without_another_request(monkeypatch):
    obj, opened = idle_spotify(monkeypatch, [idle_state()])
    clock = [0.0]
    monkeypatch.setattr('assistant.spotify.time.monotonic', lambda: clock[0])
    def playback():
        clock[0] = 5.0
        return state()
    obj.sp.current_playback.side_effect = [idle_state()]
    def read():
        if obj.sp.current_playback.call_count == 1:
            return idle_state()
        return playback()
    obj.sp.current_playback.side_effect = read
    with pytest.raises(PlaybackVerificationError):
        obj.play_candidate(Candidate('spotify:track:wanted', 'Wanted'))
    assert obj.sp.current_playback.call_count == 2
    opened.assert_called_once()
    obj.sp.start_playback.assert_not_called()


def test_idle_uri_guard_expiry_prevents_open(monkeypatch):
    from assistant.core.playback import guarded_playback, TurnExpired
    obj, opened = idle_spotify(monkeypatch, [idle_state()])
    expired = [False]
    devices = obj.sp.devices.return_value
    obj.sp.devices.side_effect = lambda: expired.__setitem__(0, True) or devices
    with guarded_playback(obj, lambda: 0 if expired[0] else 5):
        with pytest.raises(TurnExpired):
            obj.play_candidate(Candidate('spotify:track:wanted', 'Wanted'))
    opened.assert_not_called()
    obj.sp.start_playback.assert_not_called()


def test_idle_album_requires_matching_context(monkeypatch):
    uri = 'spotify:album:wanted'
    current = {**state(), 'context': {'uri': uri}}
    obj, opened = idle_spotify(monkeypatch, [current])
    obj.play_album(Candidate(uri, 'Wanted'))
    opened.assert_not_called()
    obj.sp.start_playback.assert_called_once_with(context_uri=uri, offset={'position': 0}, position_ms=0, device_id='laptop')
    assert obj.last_playback['method'] == 'web_api_album_verified'


def test_idle_playlist_requires_matching_context(monkeypatch):
    uri = 'spotify:playlist:wanted'
    current = {**state(), 'context': {'uri': uri}}
    obj, opened = idle_spotify(monkeypatch, [current])
    obj.sp.current_user_playlists = Mock(return_value={'items': [{'uri': uri, 'name': 'Wanted'}]})
    assert obj.play_playlist('Wanted') == 'Wanted'
    opened.assert_not_called()
    obj.sp.start_playback.assert_called_once_with(context_uri=uri, device_id='laptop')


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
