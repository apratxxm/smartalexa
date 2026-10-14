"""Prompt echo must not open or close a recording; prompts must not blast."""
from unittest.mock import Mock

import numpy as np
import pytest

from assistant import feedback as fb_mod
from assistant.audio import Endpointer
from assistant.config import Config
from assistant.intent import Action, parse


class _Vad:
    def __init__(self, probs):
        self.probs = list(probs)

    def reset(self):
        pass

    def __call__(self, _frame):
        return self.probs.pop(0) if self.probs else 0.0


class _Mic:
    def read(self):
        return np.zeros(512, dtype=np.int16)


def _record(probs):
    ep = Endpointer(_Vad(probs), silence_ms=800, max_seconds=6.0,
                    speech_timeout_ms=3500, threshold=0.35)
    return ep.record(_Mic())


def test_prompt_tail_then_user_command_keeps_the_command():
    # Observed live: a ~130 ms burst of our own "go ahead" at capture start,
    # then the user starting about a second later.
    result = _record([0.9] * 4 + [0.0] * 30 + [0.9] * 15 + [0.0] * 30)
    assert result["speech_started"]
    assert 0 <= result["speech_start_ms"] <= 300
    assert result["speech_start_ms"] < result["speech_end_ms"] == result["duration_ms"]
    assert result["duration_ms"] == result["audio"].size * 1000 // 16000


def test_prompt_tail_alone_is_not_a_command():
    result = _record([0.9] * 4 + [0.0] * 200)
    assert not result["speech_started"]


@pytest.mark.parametrize("probs", [
    [0.0] * 3 + [0.9] * 9 + [0.0] * 40,     # quick "pause" right away
    [0.0] * 20 + [0.9] * 5 + [0.0] * 40,    # short word after the echo window
])
def test_genuine_short_commands_still_count(probs):
    assert _record(probs)["speech_started"]


def _feedback(level, monkeypatch, cap=0.35):
    played = {}
    fake_sd = Mock()
    fake_sd.play.side_effect = lambda audio, rate, device=None: played.update(audio=audio)
    monkeypatch.setitem(__import__("sys").modules, "sounddevice", fake_sd)
    monkeypatch.setattr("assistant.ducking.get_system_volume", lambda: level)
    f = fb_mod.Feedback(Config({"feedback": {"max_output_level": cap}}))
    return f, played


def test_prompt_is_attenuated_at_full_windows_volume(monkeypatch):
    f, played = _feedback(1.0, monkeypatch)
    f._play(np.full(100, 32767, dtype=np.int16), 16000)
    assert np.max(np.abs(played["audio"])) == pytest.approx(0.35, abs=0.01)


def test_prompt_is_unchanged_when_windows_volume_is_already_low(monkeypatch):
    f, played = _feedback(0.2, monkeypatch)
    f._play(np.full(100, 16384, dtype=np.int16), 16000)
    assert np.max(np.abs(played["audio"])) == pytest.approx(0.5, abs=0.01)


def test_loudness_cap_never_writes_windows_volume(monkeypatch):
    writes = Mock()
    monkeypatch.setattr("assistant.ducking.set_system_volume", writes)
    monkeypatch.setattr("assistant.ducking.set_command_volume", writes)
    f, _ = _feedback(1.0, monkeypatch)
    f._play(np.ones(10, dtype=np.int16), 16000)
    writes.assert_not_called()


def test_pinned_speaker_does_not_trust_other_endpoint_volume(monkeypatch):
    f, played = _feedback(0.02, monkeypatch)
    f.output_device = "Different speaker"
    monkeypatch.setattr(f, "_device_level", Mock(side_effect=RuntimeError("unknown endpoint")))
    f._play(np.full(100, 32767, dtype=np.int16), 16000)
    assert np.max(np.abs(played["audio"])) == pytest.approx(0.35, abs=0.01)


def test_quiet_selected_speaker_keeps_prompt_audible(monkeypatch):
    f, played = _feedback(1.0, monkeypatch)
    f.output_device = "SoundDrum"
    monkeypatch.setattr(f, "_device_level", lambda device: 0.2)
    f._play(np.full(100, 16384, dtype=np.int16), 16000)
    assert np.max(np.abs(played["audio"])) == pytest.approx(0.5, abs=0.01)


def test_selected_endpoint_accepts_native_active_enum(monkeypatch):
    from enum import Enum
    from pycaw.pycaw import AudioUtilities
    class State(Enum):
        Active = 1
    default = Mock(FriendlyName='Laptop', state=State.Active)
    speaker = Mock(FriendlyName='Headphones (SoundDrum P)', state=State.Active)
    speaker.EndpointVolume.GetMasterVolumeLevelScalar.return_value = 0.2
    monkeypatch.setattr(AudioUtilities, 'GetSpeakers', lambda: default)
    monkeypatch.setattr(AudioUtilities, 'GetAllDevices', lambda: [default, speaker])
    assert fb_mod.Feedback._device_level('Headphones (SoundDrum P), MME') == 0.2


@pytest.mark.parametrize("text", ["play the song", "play this song", "play the track",
                                  "play my music"])
def test_generic_song_words_resume_instead_of_searching(text):
    assert parse(text).action is Action.RESUME


def test_real_title_starting_with_the_song_is_still_searched():
    intent = parse("play the song of the sea")
    assert intent.action is Action.PLAY_TRACK
    assert intent.query == "the song of the sea"
