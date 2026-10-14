"""Speech caching bounds retained PCM without shortening spoken content."""
from concurrent.futures import ThreadPoolExecutor
import threading
from unittest.mock import Mock

import numpy as np
import pytest

from assistant import feedback as fb_mod
from assistant.config import Config


def _feedback(samples):
    feedback = fb_mod.Feedback(Config({}))
    voice = Mock()

    def synthesize(text, wav):
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(samples.tobytes())

    voice.synthesize_wav.side_effect = synthesize
    feedback._piper = voice
    return feedback, voice


def test_short_cue_reuses_exact_trimmed_samples_and_owns_buffer():
    samples = np.zeros(600000, dtype=np.int16)
    samples[3200:6400] = 1000
    feedback, voice = _feedback(samples)
    cue = feedback._synth("Sorry.")
    assert feedback._synth("Sorry.")[0] is cue[0]
    voice.synthesize_wav.assert_called_once()
    # Existing trimming keeps one 20 ms leading and four trailing windows.
    np.testing.assert_array_equal(cue[0], samples[2880:7680])
    assert cue[1] == 16000
    assert cue[0].flags.owndata
    assert cue[0].base is None


@pytest.mark.parametrize("text,size", [
    ("prompt " * 2000, 1600),
    ("Large speech", fb_mod._SPEECH_CACHE_ITEM_BYTES // 2 + 1),
])
def test_long_prompt_or_large_pcm_is_complete_and_uncached(text, size):
    samples = np.full(size, 1000, dtype=np.int16)
    feedback, voice = _feedback(samples)
    for _ in range(2):
        pcm, rate = feedback._synth(text)
        np.testing.assert_array_equal(pcm, samples)
        assert rate == 16000
    assert not feedback._speech_cache
    assert voice.synthesize_wav.call_count == 2
    assert voice.synthesize_wav.call_args.args[0] == text


def test_byte_budget_evicts_oldest_before_entry_limit():
    samples = np.full(fb_mod._SPEECH_CACHE_ITEM_BYTES // 2, 1000, dtype=np.int16)
    feedback, _ = _feedback(samples)
    for i in range(5):
        feedback._synth(str(i))
    assert list(feedback._speech_cache) == ["1", "2", "3", "4"]
    assert sum(cue[0].nbytes for cue in feedback._speech_cache.values()) == 4 * 1024 * 1024


def test_long_readback_plays_all_synthesized_samples():
    samples = np.full(16000, 1000, dtype=np.int16)
    feedback, voice = _feedback(samples)
    feedback._play = Mock(return_value=True)
    text = "read this entire prompt " * 1000
    assert feedback.say(text, force=True, blocking=True) == fb_mod.KIND_SPEECH
    assert voice.synthesize_wav.call_args.args[0] == text
    pcm, rate = feedback._play.call_args.args
    np.testing.assert_array_equal(pcm, samples)
    assert rate == 16000
    assert feedback._play.call_args.kwargs == {"blocking": True}
    assert not feedback._speech_cache


def test_entry_limit_bounds_small_cues():
    feedback, _ = _feedback(np.full(320, 1000, dtype=np.int16))
    for i in range(33):
        feedback._synth(str(i))
    assert list(feedback._speech_cache) == [str(i) for i in range(1, 33)]


def test_blocking_speech_reports_failed_playback():
    feedback, _ = _feedback(np.full(320, 1000, dtype=np.int16))
    feedback._play = Mock(return_value=False)
    assert feedback.say("Review this.", force=True, blocking=True) == fb_mod.KIND_ERR


def test_blocking_speech_reports_failed_synthesis():
    feedback, voice = _feedback(np.full(320, 1000, dtype=np.int16))
    voice.synthesize_wav.side_effect = RuntimeError("synthesis unavailable")
    feedback.error = Mock(return_value=fb_mod.KIND_ERR)
    assert feedback.say("Review this.", force=True, blocking=True) == fb_mod.KIND_ERR


def test_failed_synthesis_is_not_cached_and_can_retry():
    feedback, voice = _feedback(np.full(320, 1000, dtype=np.int16))
    synthesize = voice.synthesize_wav.side_effect
    def fail(text, wav):
        synthesize(text, wav)
        raise RuntimeError("transient failure")

    voice.synthesize_wav.side_effect = fail
    with pytest.raises(RuntimeError, match="transient failure"):
        feedback._synth("Sorry.")
    assert not feedback._speech_cache
    voice.synthesize_wav.side_effect = synthesize
    assert feedback._synth("Sorry.") is not None


def test_cache_lock_does_not_serialize_synthesis(monkeypatch):
    feedback, voice = _feedback(np.full(320, 1000, dtype=np.int16))
    synthesize = voice.synthesize_wav.side_effect
    rendezvous = threading.Barrier(2)

    def concurrent_synthesize(text, wav):
        rendezvous.wait(timeout=5)
        synthesize(text, wav)

    voice.synthesize_wav.side_effect = concurrent_synthesize
    monkeypatch.setattr(fb_mod, "_SPEECH_CACHE_BYTES", 640)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(feedback._synth, ["One.", "Two."]))
    assert len(results) == 2
    assert len(feedback._speech_cache) == 1
    assert sum(cue[0].nbytes for cue in feedback._speech_cache.values()) == 640
