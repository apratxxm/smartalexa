from types import SimpleNamespace
from unittest.mock import Mock

import psutil
import pytest

from assistant import ducking


@pytest.mark.parametrize('error', [psutil.NoSuchProcess(123), psutil.AccessDenied(123)])
def test_stale_or_inaccessible_session_does_not_hide_local_playback(monkeypatch, error):
    from pycaw.pycaw import AudioUtilities
    dead = SimpleNamespace(Process=SimpleNamespace(name=Mock(side_effect=error)))
    volume = Mock()
    live = SimpleNamespace(Process=SimpleNamespace(name=lambda: 'Spotify.exe'), SimpleAudioVolume=volume)
    monkeypatch.setattr(AudioUtilities, 'GetAllSessions', lambda: [dead, live])
    live.State = 1
    assert ducking.spotify_is_playing_local() is True
