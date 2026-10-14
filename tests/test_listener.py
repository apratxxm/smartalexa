import numpy as np
import pytest
from unittest.mock import Mock

from assistant.listener import SttListener


class Cfg:
    def get(self, key, default=None):
        return {
            "listener.phrases": ["memoir", "alexa"],
            "listener.aliases": {"remember": "memoir"},
            "capture.vad_threshold": 0.35,
        }.get(key, default)


def listener():
    return SttListener(Cfg(), object(), lambda audio: ("", 0))


def test_trigger_must_be_first_word():
    assert listener().match_trigger("please call alexa play red") == ("", "")


def test_first_word_trigger_keeps_one_breath_command():
    assert listener().match_trigger("alexa play red") == ("alexa", "play red")


def test_alias_is_allowed_only_at_prefix():
    assert listener().match_trigger("remember pause") == ("", "")
    heard = SttListener(Cfg(), object(), lambda audio: ("", 0),
                        authorize_trigger=lambda source, phrase: source == "alias" and phrase == "memoir")
    assert heard.match_trigger("remember pause") == ("memoir", "pause")
    assert heard.match_trigger("someone said remember pause") == ("", "")


def test_bare_trigger_has_empty_command():
    assert listener().match_trigger("memoir") == ("memoir", "")


def test_embedded_bare_mention_is_rejected_even_with_acoustic_authorization():
    heard = SttListener(Cfg(), object(), lambda audio: ("", 0),
                        authorize_trigger=lambda source, phrase: True)
    assert heard.locate_trigger("we discuss alexa") == ("", "", False)


def test_confirmed_wake_survives_the_first_speech_frame_and_flush_clears_it():
    heard = SttListener(Cfg(), Mock(return_value=1.0), lambda audio: ("", 0))
    heard.note_wake("alexa")
    assert heard.feed(np.zeros(512, dtype=np.int16)) is None
    assert heard._woke and heard._woke_phrase == "alexa"
    heard.flush()
    assert not heard._woke and heard._pending_wake is None


def test_confirm_wake_seeds_exact_one_breath_pcm_before_current_frame():
    class FastCfg(Cfg):
        def get(self, key, default=None):
            return {"listener.min_speech_ms": 32, "listener.silence_ms": 32,
                    "listener.pad_ms": 0}.get(key, super().get(key, default))

    decode = Mock(return_value=("alexa play music", 1))
    heard = SttListener(FastCfg(), Mock(side_effect=[1.0, 0.0]), decode)
    history = np.concatenate([np.full(512, value, dtype=np.int16)
                              for value in (11, 22, 33)])
    clean = history + np.int16(100)
    assert heard.confirm_wake("Alexa", history, clean)
    assert decode.call_count == 0
    assert heard._woke and heard._woke_phrase == "alexa"
    assert heard._woke_source == "acoustic"
    np.testing.assert_array_equal(np.concatenate(heard._speech), history)
    assert all(not np.shares_memory(saved, history) for saved in heard._speech)
    assert np.array_equal(np.concatenate(heard._speech_processed), clean)

    current = np.full(512, 44, dtype=np.int16)
    assert heard.feed(current) is None
    utterance = heard.feed(np.zeros(512, dtype=np.int16))
    assert utterance is not None and utterance.triggered
    assert utterance.confirmed_audio_handoff is True
    assert np.array_equal(utterance.audio, np.concatenate([history, current,
                                                          np.zeros(512, dtype=np.int16)]))


def test_confirm_wake_does_not_duplicate_an_active_segment():
    class FastCfg(Cfg):
        def get(self, key, default=None):
            return {"listener.min_speech_ms": 32, "listener.silence_ms": 32,
                    "listener.pad_ms": 0}.get(key, super().get(key, default))

    probabilities = iter([1.0, 0.0])
    heard = SttListener(FastCfg(), lambda _: next(probabilities),
                        lambda audio: ("alexa play music", 0))
    first = np.full(512, 7, dtype=np.int16)
    assert heard.feed(first) is None
    before = list(heard._speech)
    assert heard.confirm_wake("alexa", np.full(1024, 9, dtype=np.int16))
    assert len(heard._speech) == len(before)
    assert all(left is right for left, right in zip(heard._speech, before))
    assert heard._woke and heard._woke_phrase == "alexa"
    utterance = heard.feed(np.zeros(512, dtype=np.int16))
    assert utterance is not None and utterance.triggered
    assert utterance.confirmed_audio_handoff is True
    assert np.array_equal(utterance.audio, np.concatenate([first, np.zeros(512, dtype=np.int16)]))


def test_confirmed_handoff_provenance_resets_before_next_ordinary_text_trigger():
    class FastCfg(Cfg):
        def get(self, key, default=None):
            return {"listener.min_speech_ms": 32, "listener.silence_ms": 32,
                    "listener.pad_ms": 0}.get(key, super().get(key, default))

    probabilities = iter([1.0, 0.0, 1.0, 0.0])
    heard = SttListener(FastCfg(), lambda _: next(probabilities),
                        lambda audio: ("alexa play music", 0))
    history = np.full(512, 11, dtype=np.int16)
    assert heard.confirm_wake("alexa", history)
    first_frame = np.full(512, 22, dtype=np.int16)
    assert heard.feed(first_frame) is None
    first = heard.feed(np.zeros(512, dtype=np.int16))
    assert first is not None and first.confirmed_audio_handoff is True
    assert np.array_equal(first.audio, np.concatenate([history, first_frame,
                                                       np.zeros(512, dtype=np.int16)]))

    assert heard.feed(np.full(512, 33, dtype=np.int16)) is None
    second = heard.feed(np.zeros(512, dtype=np.int16))
    assert second is not None and second.triggered
    assert second.confirmed_audio_handoff is False


def test_zero_pad_speech_onset_keeps_first_raw_and_processed_frame_without_handoff():
    class ZeroPadCfg(Cfg):
        def get(self, key, default=None):
            return {"listener.min_speech_ms": 32, "listener.silence_ms": 32,
                    "listener.pad_ms": 0}.get(key, super().get(key, default))

    probabilities = iter([1.0, 0.0])
    heard = SttListener(ZeroPadCfg(), lambda _: next(probabilities),
                        lambda audio: ("alexa pause", 1))
    first_raw = np.full(512, 13, dtype=np.int16)
    first_clean = np.full(512, 130, dtype=np.int16)
    silence_raw = np.full(512, 17, dtype=np.int16)
    silence_clean = np.full(512, 170, dtype=np.int16)

    assert heard.feed(first_raw, first_clean) is None
    np.testing.assert_array_equal(np.concatenate(heard._speech), first_raw)
    np.testing.assert_array_equal(np.concatenate(heard._speech_processed), first_clean)
    utterance = heard.feed(silence_raw, silence_clean)

    assert utterance is not None and utterance.triggered
    assert utterance.confirmed_audio_handoff is False
    np.testing.assert_array_equal(utterance.audio, np.concatenate([first_raw, silence_raw]))
    np.testing.assert_array_equal(utterance.processed_audio,
                                  np.concatenate([first_clean, silence_clean]))


def test_confirm_wake_without_clean_history_keeps_unavailable_alignment():
    heard = listener()
    history = np.arange(1024, dtype=np.int16)
    assert heard.confirm_wake("alexa", history)
    assert len(heard._speech) == 2
    assert heard._speech_processed == [None, None]


def test_confirm_wake_with_misaligned_clean_history_drops_clean_alignment():
    heard = listener()
    history = np.arange(1024, dtype=np.int16)
    assert heard.confirm_wake("alexa", history, np.arange(512, dtype=np.int16))
    assert heard._speech_processed == [None, None]


def test_confirm_wake_does_not_peek_or_decode_until_audio_is_fed():
    class FastPeekCfg(Cfg):
        def get(self, key, default=None):
            return {"listener.peek_first_ms": 1}.get(key, super().get(key, default))

    decode, on_wake = Mock(return_value=("alexa", 1)), Mock()
    heard = SttListener(FastPeekCfg(), lambda _: 1.0, decode, on_wake=on_wake)
    assert heard.confirm_wake("alexa", np.zeros(512, dtype=np.int16))
    assert decode.call_count == 0 and on_wake.call_count == 0
    heard.feed(np.zeros(512, dtype=np.int16))
    assert decode.call_count == 0 and on_wake.call_count == 0


@pytest.mark.parametrize("phrase,history", [
    ("not configured", np.zeros(512, dtype=np.int16)),
    ("alexa", np.zeros(0, dtype=np.int16)),
    ("alexa", np.zeros(513, dtype=np.int16)),
    ("alexa", np.zeros((1, 512), dtype=np.int16)),
    ("alexa", np.zeros(512, dtype=np.float32)),
    ("alexa", None),
])
def test_confirm_wake_rejects_invalid_phrase_or_history(phrase, history):
    heard = listener()
    assert not heard.confirm_wake(phrase, history)
    assert not heard._in_speech and not heard._woke


def test_confirm_wake_does_not_authorize_from_invalid_history_during_active_speech():
    heard = SttListener(Cfg(), lambda _: 1.0, lambda audio: ("", 0))
    heard.feed(np.zeros(512, dtype=np.int16))
    assert not heard.confirm_wake("alexa", np.array([], dtype=np.int16))
    assert not heard._woke


def test_flush_clears_confirmed_wake_audio_authorization():
    heard = SttListener(Cfg(), Mock(), lambda audio: ("", 0))
    assert heard.confirm_wake("alexa", np.zeros(512, dtype=np.int16))
    heard.flush()
    assert not heard._in_speech and not heard._woke
    assert not heard._speech and heard._woke_phrase == ""


def test_ordinary_ambient_speech_does_not_gain_wake_authorization():
    heard = SttListener(Cfg(), lambda _: 1.0, lambda audio: ("alexa play music", 1))
    assert heard.feed(np.zeros(512, dtype=np.int16)) is None
    assert heard._in_speech and not heard._woke


def test_similar_words_do_not_trigger_by_default():
    class ComputerCfg(Cfg):
        def get(self, key, default=None):
            if key == "listener.phrases":
                return ["computer", "alexa"]
            return super().get(key, default)

    heard = SttListener(ComputerCfg(), object(), lambda audio: ("", 0))
    for text in ("completely forgot", "compute the result", "remember the song",
                 "ordinary speech continues"):
        assert heard.locate_trigger(text)[0] == ""


def test_canonical_exact_match_beats_alias():
    class ConflictingCfg(Cfg):
        def get(self, key, default=None):
            if key == "listener.aliases":
                return {"alexa": "memoir"}
            return super().get(key, default)

    heard = SttListener(ConflictingCfg(), object(), lambda audio: ("", 0))
    assert heard.match_trigger("alexa pause") == ("alexa", "pause")


def test_alias_for_unconfigured_target_is_ignored():
    class DisabledTargetCfg(Cfg):
        def get(self, key, default=None):
            if key == "listener.phrases":
                return ["alexa"]
            return super().get(key, default)

    heard = SttListener(DisabledTargetCfg(), object(), lambda audio: ("", 0))
    assert heard.match_trigger("remember pause") == ("", "")
    assert heard.locate_trigger("someone said remember pause")[0] == ""


def test_fuzzy_canonical_match_requires_explicit_opt_in():
    class FuzzyCfg(Cfg):
        def get(self, key, default=None):
            if key == "listener.allow_fuzzy":
                return True
            if key == "listener.phrases":
                return ["computer", "alexa"]
            return super().get(key, default)

    heard = SttListener(FuzzyCfg(), object(), lambda audio: ("", 0),
                        authorize_trigger=lambda source, phrase: source == "fuzzy" and phrase == "computer")
    assert heard.match_trigger("compute play music") == ("computer", "play music")
    # Fuzzy matching never applies to aliases.
    assert heard.match_trigger("remembor pause") == ("", "")


def test_repeated_wake_is_treated_as_bare_trigger():
    class FastCfg(Cfg):
        def get(self, key, default=None):
            return {"listener.min_speech_ms": 32,
                    "listener.silence_ms": 32,
                    "listener.pad_ms": 0}.get(key, super().get(key, default))

    voice = iter([1.0, 0.0])
    heard = SttListener(FastCfg(), lambda frame: next(voice),
                        lambda audio: ("Alexa Alexa", 1))
    frame = np.zeros(512, dtype=np.int16)
    assert heard.feed(frame) is None
    utt = heard.feed(frame)
    assert utt.triggered is True
    assert utt.command == ""


# ---------- wake word over continuous (music) speech ----------

def test_trigger_after_lyrics_is_found_with_its_command():
    assert listener().locate_trigger("tonight we ride alexa play red") == ("", "", False)
    heard = SttListener(Cfg(), object(), lambda audio: ("", 0),
                        authorize_trigger=lambda source, phrase: source == "mid" and phrase == "alexa")
    assert heard.locate_trigger("tonight we ride alexa play red") == ("alexa", "play red", True)


def test_prefix_trigger_is_not_marked_mid_utterance():
    assert listener().locate_trigger("alexa pause") == ("alexa", "pause", False)


def test_computer_in_lyrics_does_not_trigger():
    class ComputerCfg(Cfg):
        def get(self, key, default=None):
            if key == "listener.phrases":
                return ["alexa", "computer"]
            return super().get(key, default)
    heard = SttListener(ComputerCfg(), object(), lambda audio: ("", 0))
    assert heard.locate_trigger("i am a computer play red")[0] == ""
    assert heard.locate_trigger("computer play red")[0] == "computer"


def test_loose_spelling_only_counts_at_the_start():
    assert listener().locate_trigger("my friend alex went home")[0] == ""


class MusicCfg(Cfg):
    def get(self, key, default=None):
        return {"listener.min_speech_ms": 32, "listener.silence_ms": 96,
                "listener.pad_ms": 0, "listener.max_utterance_s": 0.32,   # 10 frames
                "listener.overlap_ms": 96, "listener.extend_ms": 320}.get(
                    key, super().get(key, default))


def _run(vad_seq, transcripts):
    vad = iter(vad_seq)
    said = iter(transcripts)
    heard = SttListener(MusicCfg(), lambda frame: next(vad), lambda audio: (next(said), 1),
                        authorize_trigger=lambda source, phrase: source == "mid" and phrase == "alexa")
    out = []
    frame = np.zeros(512, dtype=np.int16)
    for _ in vad_seq:
        utt = heard.feed(frame)
        if utt is not None:
            out.append(utt)
    return out


def test_forced_cut_keeps_listening_instead_of_acting_on_a_cut_command():
    # 10 frames of music, cut: "... alexa play re" -> extend; the user then
    # finishes and pauses, and the full command is transcribed.
    utts = _run([1.0] * 14 + [0.0] * 4,
                ["la la alexa play re", "la la alexa play red by seedhe maut"])
    assert len(utts) == 1
    assert utts[0].triggered and utts[0].command == "play red by seedhe maut"
    assert utts[0].mid_utterance


def test_command_still_running_at_extension_cap_becomes_bare_wake():
    utts = _run([1.0] * 20, ["la alexa play", "la alexa play something long"])
    assert len(utts) == 1
    assert utts[0].triggered and utts[0].command == ""
    assert utts[0].truncated and utts[0].end_reason == "extension_limit"
    assert utts[0].audio.size == 20 * 512


def test_forced_cut_without_trigger_carries_overlap_into_next_chunk():
    utts = _run([1.0] * 20, ["la la la", "la ale"])
    assert [u.triggered for u in utts] == [False, False]
    assert utts[1].audio.size == (3 + 7) * 512   # 3 overlap frames + 7 new


@pytest.mark.parametrize("verified", [False, True])
def test_early_wake_requires_final_verification_when_final_transcript_loses_wake(verified):
    class PeekCfg(MusicCfg):
        def get(self, key, default=None):
            return {"listener.peek_first_ms": 64, "listener.peek_every_ms": 1000,
                    "listener.max_utterance_s": 3.0}.get(key, super().get(key, default))
    said = iter(["Alexa", "Fuzz!"])
    woke = []
    vad = iter([1.0] * 4 + [0.0] * 4)
    heard = SttListener(PeekCfg(), lambda frame: next(vad), lambda audio: (next(said), 1),
                        on_wake=lambda: woke.append(True),
                        verify_trigger=lambda *_: ("alexa", "pause", False, 12)
                        if verified else ("", "", False, 12))
    frame = np.zeros(512, dtype=np.int16)
    utts = [u for u in (heard.feed(frame) for _ in range(8)) if u is not None]
    assert woke == [True]
    assert utts[0].triggered is verified
    assert utts[0].command == ("pause" if verified else "")


def test_final_alias_uses_verifier_only_after_utterance_ends():
    class FastCfg(Cfg):
        def get(self, key, default=None):
            return {"listener.min_speech_ms": 32, "listener.silence_ms": 32,
                    "listener.pad_ms": 0}.get(key, super().get(key, default))

    verifier = Mock(return_value=("memoir", "pause", False, 12))
    vad = iter([1.0, 0.0])
    heard = SttListener(FastCfg(), lambda frame: next(vad),
                        lambda audio: ("remember pause", 1), verify_trigger=verifier)
    frame = np.zeros(512, dtype=np.int16)
    assert heard.feed(frame) is None
    verifier.assert_not_called()
    utt = heard.feed(frame)
    verifier.assert_called_once()
    assert utt.triggered and utt.command == "pause"
    assert utt.confirmation_mode == "second_stt"
    assert utt.confirmation_latency_ms == 12


def test_final_alias_fails_closed_when_verifier_rejects():
    class FastCfg(Cfg):
        def get(self, key, default=None):
            return {"listener.min_speech_ms": 32, "listener.silence_ms": 32,
                    "listener.pad_ms": 0}.get(key, super().get(key, default))

    vad = iter([1.0, 0.0])
    rejected = []
    heard = SttListener(FastCfg(), lambda frame: next(vad),
                        lambda audio: ("remember pause", 1),
                        verify_trigger=lambda *_: ("", "", False, 12),
                        on_reject=lambda reason, duration: rejected.append((reason, duration)))
    frame = np.zeros(512, dtype=np.int16)
    assert heard.feed(frame) is None
    utt = heard.feed(frame)
    assert not utt.triggered
    assert any(reason == "second_stt_unconfirmed" for reason, _ in rejected)


def test_exact_and_acoustically_corroborated_alias_skip_verifier():
    verifier = Mock(side_effect=AssertionError("unnecessary decode"))
    heard = SttListener(Cfg(), object(), lambda audio: ("", 0),
                        authorize_trigger=lambda source, phrase: True,
                        verify_trigger=verifier)
    assert heard.match_trigger("memoir pause") == ("memoir", "pause")
    assert heard.match_trigger("remember pause") == ("memoir", "pause")
    verifier.assert_not_called()


def test_rolling_peek_catches_late_wake_once_with_bounded_audio_and_cadence():
    class RollingCfg(Cfg):
        def get(self, key, default=None):
            return {"listener.max_utterance_s": 6.0,
                    "listener.pad_ms": 300, "listener.peek_first_ms": 900,
                    "listener.peek_every_ms": 1000, "listener.peek_until_ms": 10000,
                    "listener.peek_window_ms": 1600}.get(key, super().get(key, default))
    woke = Mock()
    decodes = []
    step = [0]
    def transcribe(audio):
        decodes.append((step[0], audio.size))
        return ("alexa pause" if step[0] >= 80 else "music lyrics", 1)
    heard = SttListener(RollingCfg(), lambda _: 1.0, transcribe, on_wake=woke)
    frame = np.zeros(512, dtype=np.int16)
    for index in range(125):
        step[0] = index
        assert heard.feed(frame) is None
    assert len(decodes) == 3
    assert all(b[0] - a[0] >= 31 for a, b in zip(decodes, decodes[1:]))
    assert all(size <= 50 * 512 for _, size in decodes)
    assert decodes[-1][0] > 31
    assert heard._woke_mid
    assert heard._woke_source == "text_exact"
    woke.assert_called_once()


def test_short_segment_reports_discard_and_publishes_single_vad_result():
    class ShortCfg(Cfg):
        def get(self, key, default=None):
            return {"listener.min_speech_ms": 300, "listener.silence_ms": 32,
                    "listener.pad_ms": 0}.get(key, super().get(key, default))
    vad = Mock(side_effect=[0.8, 0.1])
    rejected = Mock()
    transcribe = Mock(side_effect=AssertionError("short speech must not be decoded"))
    candidates = []
    heard = SttListener(ShortCfg(), vad, transcribe, on_reject=rejected,
                        on_candidate=candidates.append)
    raw = np.ones(512, dtype=np.int16)
    clean = np.zeros(512, dtype=np.int16)
    assert heard.feed(raw, clean) is None
    assert heard.last_vad_probability == 0.8
    assert heard.feed(raw, clean) is None
    assert heard.last_vad_probability == 0.1
    assert vad.call_count == 2
    assert vad.call_args.args[0] is clean
    rejected.assert_called_once_with("speech_too_short", 64)
    assert heard.segments_completed == heard.discarded_short_segments == 1
    np.testing.assert_array_equal(candidates[0]["audio"], np.concatenate([raw, raw]))
    np.testing.assert_array_equal(candidates[0]["processed_audio"], np.concatenate([clean, clean]))
    transcribe.assert_not_called()
    heard.flush()
    assert heard.discarded_short_segments == 1


@pytest.mark.parametrize(
    "frame_count, voiced_count, silence_ms, transcript, triggered",
    [
        (19, 1, 576, "alexa pause", True),  # 608 ms captured
        (25, 2, 736, "alexa pause", True),  # 800 ms captured
        (19, 1, 576, "turn the lights on", False),
        (19, 1, 576, "remember pause", False),  # alias needs corroboration
    ],
)
def test_sufficient_capture_reaches_decoder_despite_vad_undercount(
        frame_count, voiced_count, silence_ms, transcript, triggered):
    class UndercountCfg(Cfg):
        def get(self, key, default=None):
            return {"listener.min_speech_ms": 160, "listener.silence_ms": silence_ms,
                    "listener.pad_ms": 0}.get(key, super().get(key, default))

    vad = Mock(side_effect=[0.8] * voiced_count + [0.0] * (frame_count - voiced_count))
    decoded = []
    raw = [np.full(512, i + 1, dtype=np.int16) for i in range(frame_count)]
    def transcribe(audio):
        decoded.append(audio.copy())
        return transcript, 7

    woke = Mock()
    heard = SttListener(UndercountCfg(), vad, transcribe, on_wake=woke)
    results = [heard.feed(frame) for frame in raw]
    utterance = next((item for item in results if item is not None), None)

    assert utterance is not None
    assert len(decoded) == 1
    assert vad.call_count == frame_count
    np.testing.assert_array_equal(decoded[0], np.concatenate(raw))
    assert heard.discarded_short_segments == 0
    assert utterance.triggered is triggered
    if triggered:
        assert utterance.matched_phrase == "alexa"
        assert utterance.command == "pause"
        assert utterance.wake_path == "final_text"
        woke.assert_not_called()
    else:
        assert utterance.matched_phrase == ""
        woke.assert_not_called()


def test_no_onset_still_does_not_reach_decoder_with_sufficient_silence_capture():
    class NoOnsetCfg(Cfg):
        def get(self, key, default=None):
            return {"listener.min_speech_ms": 160, "listener.silence_ms": 160,
                    "listener.pad_ms": 0}.get(key, super().get(key, default))

    decode = Mock(side_effect=AssertionError("no onset must not be decoded"))
    heard = SttListener(NoOnsetCfg(), Mock(return_value=0.0), decode)
    for _ in range(6):
        assert heard.feed(np.zeros(512, dtype=np.int16)) is None
    decode.assert_not_called()


def test_production_capture_timing_decodes_608ms_under_voiced_frame_floor():
    class ProductionTimingCfg(Cfg):
        def get(self, key, default=None):
            return {"listener.min_speech_ms": 160, "listener.silence_ms": 350,
                    "listener.pad_ms": 300, "listener.peek_first_ms": 0}.get(
                        key, super().get(key, default))

    vad = Mock(side_effect=[0.0] * 8 + [0.8] + [0.0] * 10)
    decode = Mock(return_value=("alexa pause", 7))
    heard = SttListener(ProductionTimingCfg(), vad, decode)
    raw = [np.full(512, i + 1, dtype=np.int16) for i in range(19)]
    clean = [np.full(512, i + 101, dtype=np.int16) for i in range(19)]
    results = [heard.feed(frame, processed) for frame, processed in zip(raw, clean)]
    utterance = next(item for item in results if item is not None)

    assert len(utterance.audio) == 19 * 512  # 608 ms including eight-frame pre-roll
    assert utterance.triggered and utterance.matched_phrase == "alexa"
    assert vad.call_count == 19
    decode.assert_called_once()
    np.testing.assert_array_equal(decode.call_args.args[0], np.concatenate(raw))
    np.testing.assert_array_equal(utterance.processed_audio, np.concatenate(clean))


def test_production_peek_and_final_decode_reset_without_duplicate_wake():
    class ProductionPeekCfg(Cfg):
        def get(self, key, default=None):
            return {"listener.min_speech_ms": 160, "listener.silence_ms": 350,
                    "listener.pad_ms": 300, "listener.peek_first_ms": 700,
                    "listener.peek_every_ms": 500, "listener.peek_until_ms": 10000,
                    "listener.peek_window_ms": 1600}.get(key, super().get(key, default))

    decode = Mock(side_effect=[("alexa pause", 4), ("alexa pause", 7),
                               ("turn the lights on", 9)])
    woke = Mock()
    vad = Mock(side_effect=[0.0] * 8 + [0.8] + [0.0] * 5 + [0.8] + [0.0] * 10
               + [0.0] * 8 + [0.8] + [0.0] * 10)
    heard = SttListener(ProductionPeekCfg(), vad, decode, on_wake=woke)

    first_frames = [np.full(512, i + 1, dtype=np.int16) for i in range(25)]
    first_results = [heard.feed(frame) for frame in first_frames]
    first = next(item for item in first_results if item is not None)
    assert len(first.audio) == 25 * 512  # 800 ms with two onset frames
    assert first.triggered and first.wake_path == "peek"
    assert decode.call_count == 2  # one scheduled peek and the final decode
    woke.assert_called_once()
    assert not heard._woke and heard._pending_wake is None

    # A fresh nonwake segment after reset must not inherit the prior peek.
    next_frames = [np.full(512, i + 201, dtype=np.int16) for i in range(19)]
    second_results = [heard.feed(frame) for frame in next_frames]
    second = next(item for item in second_results if item is not None)
    assert not second.triggered
    assert decode.call_count == 3
    woke.assert_called_once()
    assert not heard._woke and heard._pending_wake is None


class CandidateCfg(Cfg):
    def get(self, key, default=None):
        return {"listener.min_speech_ms": 32, "listener.silence_ms": 32,
                "listener.pad_ms": 32, "listener.vad_threshold": .2,
                "telemetry.wake_candidates_enabled": True}.get(key, super().get(key, default))


def test_candidate_callback_keeps_exact_rejected_raw_and_processed_without_asr_label():
    probabilities = iter([.8, 0])
    candidates = []
    legacy = Mock()
    heard = SttListener(CandidateCfg(), lambda _: next(probabilities), lambda _: ("you", 12),
                        on_candidate=candidates.append, on_near_miss=legacy)
    raw = [np.full(512, value, np.int16) for value in (100, 200)]
    clean = [np.full(512, value, np.int16) for value in (10, 20)]
    assert heard.feed(raw[0], clean[0]) is None
    utt = heard.feed(raw[1], clean[1])
    assert not utt.triggered and utt.end_reason == "silence" and not utt.truncated
    assert len(candidates) == 1
    info = candidates[0]
    np.testing.assert_array_equal(info["audio"], np.concatenate(raw))
    np.testing.assert_array_equal(info["processed_audio"], np.concatenate(clean))
    assert info["review_state"] == "unknown" and info["reason"] == "speech_no_trigger"
    assert info["transcript"] == "you" and info["stt_latency_ms"] == 12
    assert info["relative_start_sample"] == 0 and info["relative_end_sample"] == 1024
    assert info["segment_id"] == utt.segment_id
    assert "capture_started_at" not in info  # no fabricated absolute capture clock
    legacy.assert_not_called()


def test_candidate_callback_captures_short_segment_without_decoding():
    class Short(CandidateCfg):
        def get(self, key, default=None):
            return 300 if key == "listener.min_speech_ms" else super().get(key, default)
    probabilities = iter([.8, 0])
    candidates = []
    decode = Mock(side_effect=AssertionError("No decode for short segment"))
    heard = SttListener(Short(), lambda _: next(probabilities), decode, on_candidate=candidates.append)
    frame = np.arange(512, dtype=np.int16)
    heard.feed(frame)
    assert heard.feed(frame) is None
    assert candidates[0]["reason"] == "speech_too_short"
    np.testing.assert_array_equal(candidates[0]["audio"], np.concatenate([frame, frame]))
    assert candidates[0]["processed_audio"] is None
    decode.assert_not_called()


def test_candidate_callback_retains_peek_rejection_and_wake_lost_in_final_decode():
    class Peek(CandidateCfg):
        def get(self, key, default=None):
            return {"listener.peek_first_ms": 64, "listener.peek_every_ms": 1000}.get(key, super().get(key, default))
    probabilities = iter([.8, .8, 0])
    transcripts = iter([("remember", 1), ("you", 3)])
    candidates = []
    heard = SttListener(Peek(), lambda _: next(probabilities), lambda _: next(transcripts),
                        on_wake=Mock(), on_candidate=candidates.append)
    for _ in range(3):
        heard.feed(np.zeros(512, np.int16))
    assert "alias_uncorroborated" in candidates[0]["reason"]
    assert candidates[0]["transcript"] == "you"


def test_candidate_forced_chunks_publish_truncation_and_overlap_provenance():
    class ShortWindow(CandidateCfg):
        def get(self, key, default=None):
            return {"listener.max_utterance_s": .128, "listener.overlap_ms": 32}.get(key, super().get(key, default))
    candidates = []
    heard = SttListener(ShortWindow(), lambda _: 1, lambda _: ("you", 1), on_candidate=candidates.append)
    utterances = []
    for _ in range(7):
        utt = heard.feed(np.ones(512, np.int16))
        if utt is not None:
            utterances.append(utt)
    assert len(candidates) == 2
    assert all(info["truncated"] and info["end_reason"] == "max_utterance" for info in candidates)
    assert candidates[1]["overlap_of"] == candidates[0]["segment_id"]
    assert all(utt.truncated for utt in utterances)


def test_below_vad_diagnostic_lane_never_decodes_triggers_or_ducks():
    candidates = []
    probabilities = iter([0] * 9 + [.1] * 3 + [0] * 11)
    decode, wake, speech = Mock(), Mock(), Mock()
    heard = SttListener(CandidateCfg(), lambda _: next(probabilities), decode,
                        on_wake=wake, on_speech_start=speech, on_candidate=candidates.append)
    raw = np.ones(512, np.int16)
    for _ in range(23):
        assert heard.feed(raw) is None
    assert len(candidates) == 1
    assert candidates[0]["kind"] == "below_vad" and candidates[0]["diagnostic_only"]
    assert candidates[0]["audio"].size <= 6 * 16000
    assert candidates[0]["review_state"] == "unknown"
    decode.assert_not_called()
    wake.assert_not_called()
    speech.assert_not_called()


def test_below_vad_diagnostic_lane_is_bounded_and_yields_to_primary():
    candidates = []
    heard = SttListener(CandidateCfg(), lambda _: .1, Mock(), on_candidate=candidates.append)
    for _ in range(190):
        heard.feed(np.zeros(512, np.int16))
    assert len(candidates) == 1 and candidates[0]["truncated"]
    assert candidates[0]["audio"].size == 187 * 512
    assert len(heard._weak_frames) <= 187
    heard.vad = Mock(return_value=.8)
    heard.feed(np.zeros(512, np.int16))
    assert heard._in_speech and not heard._weak_frames
    heard.flush()
    assert not heard._weak_frames


def test_candidate_logging_failure_does_not_change_rejection_result():
    probabilities = iter([.8, 0])
    callback = Mock(side_effect=OSError("disk full"))
    heard = SttListener(CandidateCfg(), lambda _: next(probabilities), lambda _: ("you", 1), on_candidate=callback)
    heard.feed(np.zeros(512, np.int16))
    assert heard.feed(np.zeros(512, np.int16)).triggered is False
