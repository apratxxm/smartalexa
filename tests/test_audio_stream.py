"""Capture framing regressions, without opening a microphone."""
import numpy as np
import pytest
import soxr

import assistant.echo as echo_module
from assistant.audio import Microphone, device_candidates


def test_streaming_output_matches_reference_without_padding_or_loss():
    mic = Microphone(capture_rate=48000)
    source = (12000 * np.sin(np.arange(48000) * 0.137)).astype(np.int16)
    reference = soxr.ResampleStream(48000, 16000, 1, dtype="float32")
    chunks = [source[i:i + 1536] for i in range(0, len(source) - 1535, 1536)]
    expected = np.concatenate([reference.resample_chunk(c.astype(np.float32)) for c in chunks])
    for chunk in chunks:
        # Native callback periods need not coincide with consumer frames.
        mic._queue.extend([(0.0, chunk[:479]), (0.0, chunk[479:])])
    count = len(expected) // 512
    actual = np.concatenate([mic.read() for _ in range(count)])
    np.testing.assert_array_equal(actual, np.clip(expected[:count * 512], -32768, 32767).astype(np.int16))


def test_flush_discards_filter_and_output_history():
    mic = Microphone(capture_rate=48000)
    mic._queue.extend([(0.0, np.full(1536, 12000, dtype=np.int16)) for _ in range(8)])
    mic.read()
    mic.flush()
    assert mic.ring_contents().size == 0
    assert mic.last_clean is None
    raw, clean = mic.preroll_pair(1)
    assert raw.size == 0 and clean is None
    mic._queue.extend([(0.0, np.zeros(1536, dtype=np.int16)) for _ in range(8)])
    assert not np.any(mic.read())
    assert mic.preroll(0).size == 0
    raw, clean = mic.preroll_pair(1)
    assert raw.size == 512 and not np.any(raw) and clean is None


def test_preroll_pair_copies_clean_output_and_keeps_matching_offsets():
    mic = Microphone(frame_length=4, sample_rate=4, ring_seconds=2)
    reusable = np.zeros(4, dtype=np.int16)
    class Echo:
        def __init__(self):
            self.value = 10
            self.active = False
        def process(self, frame, _timestamp):
            reusable[:] = self.value
            self.value += 10
            self.active = True
            return reusable
    mic.echo = Echo()
    for value in (1, 2, 3):
        mic._queue.append((0.0, np.full(4, value, dtype=np.int16)))
        mic.read()

    raw, clean = mic.preroll_pair(2)  # two frames, same newest-frame offset
    np.testing.assert_array_equal(raw, [2] * 4 + [3] * 4)
    np.testing.assert_array_equal(clean, [20] * 4 + [30] * 4)
    # The newest clean frame remains an owned snapshot after the buffer changes.
    reusable[:] = -1
    np.testing.assert_array_equal(mic.preroll_pair(1)[1], [30] * 4)


def test_preroll_pair_fails_closed_for_missing_or_unaligned_clean_history():
    mic = Microphone(frame_length=4, sample_rate=4)
    mic._ring.extend([np.full(4, n, dtype=np.int16) for n in (1, 2)])
    mic._clean_ring.extend([np.full(4, 10, dtype=np.int16), None])
    raw, clean = mic.preroll_pair(2)
    np.testing.assert_array_equal(raw, [1] * 4 + [2] * 4)
    assert clean is None

    # Legacy/uninitialized fixture state cannot provide a trustworthy pair.
    del mic._clean_ring
    assert mic.preroll_pair(2)[1] is None

    mic = Microphone(frame_length=4, sample_rate=4)
    mic._ring.append(np.ones(4, dtype=np.int16))
    mic._clean_ring.append(np.ones(3, dtype=np.int16))
    assert mic.preroll_pair(1)[1] is None


def test_read_without_echo_records_unavailable_processed_frame():
    mic = Microphone()
    mic._queue.append((0.0, np.ones(512, dtype=np.int16)))
    mic.read()
    raw, clean = mic.preroll_pair(1)
    assert raw.size == 512 and clean is None


def test_attached_but_inactive_echo_is_not_archived_as_processed():
    mic = Microphone()
    class InactiveEcho:
        active = False
        def process(self, frame, _timestamp):
            return frame
    mic.echo = InactiveEcho()
    mic._queue.append((0.0, np.ones(512, dtype=np.int16)))
    mic.read()
    raw, clean = mic.preroll_pair(1)
    assert raw.size == 512 and clean is None


def test_preroll_pair_zero_rounding_returns_empty_frames():
    mic = Microphone()
    mic._ring.append(np.ones(512, dtype=np.int16))
    raw, clean = mic.preroll_pair(0.001)
    assert raw.size == 0 and clean.size == 0


def test_echo_read_failure_does_not_archive_previous_clean_frame():
    mic = Microphone()
    mic.last_clean = np.full(512, 123, dtype=np.int16)
    class BrokenEcho:
        def process(self, _frame, _timestamp):
            raise RuntimeError("AEC failed")
    mic.echo = BrokenEcho()
    mic._queue.append((0.0, np.ones(512, dtype=np.int16)))
    mic.read()
    assert mic.last_clean is None
    assert mic.preroll_pair(1)[1] is None


def test_callback_queue_overflow_is_counted_and_warning_is_rate_limited(caplog, monkeypatch):
    from assistant import audio
    # A preceding telemetry fixture may disable parent propagation. Observe
    # this logger directly, without depending on global test execution order.
    monkeypatch.setattr(audio.log, "handlers", [caplog.handler])
    monkeypatch.setattr(audio.log, "propagate", False)
    mic = Microphone()
    for value in range(mic._queue.maxlen):
        mic._callback(np.full((1, 1), value, dtype=np.int16), 1, None, None)
    mic._callback(np.full((1, 1), 700, dtype=np.int16), 1, None, None)
    mic._callback(np.full((1, 1), 701, dtype=np.int16), 1, None, None)

    assert len(mic._queue) == 512
    assert mic.queue_overflow_count == 2
    assert not caplog.records  # callback must never wait for a log write
    assert mic._queue[0][1][0] == 2
    assert mic._queue[-1][1][0] == 701
    mic.read()
    assert len([record for record in caplog.records
                if "callback queue overflowed" in record.message]) == 1


@pytest.mark.parametrize("failure", ["start", "canceller"])
def test_echo_open_failure_stops_reference(monkeypatch, failure):
    reference = type("Reference", (), {"stop": lambda self: setattr(self, "stopped", True)})()
    if failure == "start":
        def start():
            raise RuntimeError("loopback start failed")
        reference.start = start
        monkeypatch.setattr(echo_module, "SpeakerReference", lambda _hint: reference)
    else:
        reference.start = lambda: None
        monkeypatch.setattr(echo_module, "SpeakerReference", lambda _hint: reference)
        def create(_reference):
            raise RuntimeError("AEC creation failed")
        monkeypatch.setattr(echo_module, "EchoCanceller", create)

    assert echo_module.open_echo_canceller() is None
    assert reference.stopped


def test_native_rate_keeps_samples():
    mic = Microphone()
    expected = np.arange(1024, dtype=np.int16)
    mic._queue.extend([(0.0, expected[:100]), (0.0, expected[100:])])
    np.testing.assert_array_equal(np.concatenate([mic.read(), mic.read()]), expected)


def test_missing_callback_times_out(monkeypatch):
    import time
    ticks = iter([0.0, 4.0])
    monkeypatch.setattr(time, "monotonic", lambda: next(ticks))
    with pytest.raises(TimeoutError, match="Microphone"):
        Microphone().read()


def test_prompt_capture_is_intentionally_discarded_not_queue_overflow():
    mic = Microphone()
    data = np.ones((512, 1), dtype=np.int16)
    with mic.discard_capture():
        for _ in range(700):
            mic._callback(data, 512, None, None)
        assert len(mic._queue) == 0
        assert mic.queue_overflow_count == 0
        assert mic.intentional_discard_count == 700
    mic._callback(data, 512, None, None)
    np.testing.assert_array_equal(mic.read(), data[:, 0])


def test_prompt_discard_unwinds_nested_and_exception_paths():
    mic = Microphone()
    with pytest.raises(RuntimeError):
        with mic.discard_capture():
            with mic.discard_capture():
                assert mic._discard_depth == 2
            assert mic._discard_depth == 1
            raise RuntimeError("prompt output failed")
    assert mic._discard_depth == 0
    assert not mic._queue


def test_real_capture_overflow_remains_visible():
    mic = Microphone()
    data = np.ones((512, 1), dtype=np.int16)
    for _ in range(513):
        mic._callback(data, 512, None, None)
    assert mic.queue_overflow_count == 1
    assert mic.intentional_discard_count == 0


def test_settled_short_reply_is_retained():
    from assistant.audio import Endpointer
    from unittest.mock import Mock
    vad = Mock(side_effect=[0.9] * 5 + [0.0] * 30)
    frames = iter(range(100))
    mic = Mock(read=lambda: np.full(512, next(frames), dtype=np.int16))
    result = Endpointer(vad).record(
        mic, after_prompt=False)
    assert result["speech_started"]
    assert result["speech_start_ms"] == 0
    np.testing.assert_array_equal(result["audio"].reshape(-1, 512)[:5, 0], range(5))
    assert result["duration_ms"] == result["speech_end_ms"]
    assert result["duration_ms"] == result["audio"].size * 1000 // 16000
    assert result["wait_duration_ms"] >= 0


def test_two_second_wait_keeps_only_preroll_and_all_speech_and_tail():
    from assistant.audio import Endpointer
    from unittest.mock import Mock
    start_frames = 63  # just over two seconds at 512/16 kHz
    script = iter([0.0] * start_frames + [0.9] * 8 + [0.0] * 25)
    frame_number = iter(range(200))
    mic = Mock(read=lambda: np.full(512, next(frame_number), dtype=np.int16))
    result = Endpointer(Mock(side_effect=lambda _: next(script)),
                        speech_timeout_ms=2500).record(mic, after_prompt=False)

    assert result["speech_started"] and result["end_reason"] == "silence"
    assert result["speech_start_ms"] == 9 * 32
    assert result["speech_end_ms"] == result["duration_ms"]
    assert result["duration_ms"] == result["audio"].size * 1000 // 16000
    np.testing.assert_array_equal(
        result["audio"].reshape(-1, 512)[:, 0], list(range(54, 96)))


@pytest.mark.parametrize("start_frames", [0, 450])
def test_separate_start_and_utterance_budgets(start_frames, monkeypatch):
    from types import SimpleNamespace
    from assistant.audio import Endpointer
    from assistant import audio as audio_module
    from unittest.mock import Mock
    script = iter([0.0] * start_frames + [0.9] * 1000)
    vad = Mock(side_effect=lambda _: next(script))
    expected_wait_ms = (start_frames + 375) * 32
    ticks = iter([100.0, 100.0 + expected_wait_ms / 1000])
    monkeypatch.setattr(audio_module, "time",
                        SimpleNamespace(perf_counter=lambda: next(ticks)))
    result = Endpointer(vad, max_seconds=27, speech_timeout_ms=15000,
                        utterance_seconds=12).record(
        Mock(read=lambda: np.ones(512, dtype=np.int16)), after_prompt=False)
    assert result["speech_started"]
    assert result["speech_start_ms"] == (9 if start_frames else 0) * 32
    assert result["duration_ms"] - result["speech_start_ms"] == 12000
    assert result["wait_duration_ms"] == expected_wait_ms
    assert result["speech_end_ms"] == result["duration_ms"]
    assert result["truncated"] and result["end_reason"] == "utterance_limit"


def test_long_command_is_not_cut_at_the_old_six_second_budget():
    from assistant.audio import Endpointer
    from unittest.mock import Mock
    script = iter([0.0] * 95 + [0.9] * 300 + [0.0] * 30)
    result = Endpointer(Mock(side_effect=lambda _: next(script)), max_seconds=23.5,
                        speech_timeout_ms=3500, utterance_seconds=20).record(
        Mock(read=lambda: np.ones(512, dtype=np.int16)), after_prompt=False)
    assert result["speech_started"]
    assert result["speech_start_ms"] == 9 * 32
    assert result["duration_ms"] == result["audio"].size * 1000 // 16000
    assert result["duration_ms"] == (9 + 300 + 25) * 32
    assert result["end_reason"] == "silence" and not result["truncated"]


def test_capture_limit_reports_partial_speech_not_normal_completion():
    from assistant.audio import Endpointer
    from unittest.mock import Mock
    result = Endpointer(Mock(return_value=0.9), max_seconds=0.16).record(
        Mock(read=lambda: np.ones(512, dtype=np.int16)), after_prompt=False)
    assert result["end_reason"] == "capture_limit" and result["truncated"]


def test_named_microphone_never_falls_back_to_default(monkeypatch):
    import sounddevice
    monkeypatch.setattr(sounddevice, "query_hostapis", lambda: [{"name": "Windows WASAPI"}])
    monkeypatch.setattr(sounddevice, "query_devices", lambda: [
        {"name": "Realtek microphone", "max_input_channels": 1, "hostapi": 0}])
    assert device_candidates("Realtek") == [0]
    with pytest.raises(RuntimeError, match="not found"):
        device_candidates("Missing mic")


def test_loopback_read_never_consumes_future_audio():
    reference = echo_module.SpeakerReference()
    data = np.arange(160, dtype=np.float32)
    reference._chunks.append((1.0, data))
    reference._held = len(data)
    np.testing.assert_array_equal(reference.take_until(1.005), data[:80])
    assert reference._held == 80
    np.testing.assert_array_equal(reference.take_until(1.01), data[80:])
    assert reference._held == 0


def test_loopback_stop_discards_old_device_audio():
    reference = echo_module.SpeakerReference()
    reference._chunks.append((1.0, np.ones(160, dtype=np.float32)))
    reference._held = 160
    reference.last_data_at = 123.0
    reference.stop()
    assert reference._held == 0
    assert not reference._chunks
    assert reference.last_data_at == 0


def test_loopback_detects_reconnected_preferred_endpoint_with_native_enum(monkeypatch):
    from enum import Enum
    from unittest.mock import Mock
    from pycaw.pycaw import AudioUtilities
    class State(Enum):
        Active = 1
    speaker = Mock(FriendlyName='Headphones (SoundDrum P)', state=State.Active)
    monkeypatch.setattr(AudioUtilities, 'GetAllDevices', lambda: [speaker])
    monkeypatch.setattr(AudioUtilities, 'GetSpeakers', lambda: Mock(FriendlyName='Laptop'))
    reference = echo_module.SpeakerReference('Headphones (SoundDrum P)')
    reference.name = 'Laptop [Loopback]'
    assert reference.route_changed()
    reference.name = 'Headphones (SoundDrum P) [Loopback]'
    assert not reference.route_changed()


def test_reopen_preserves_pinned_input_and_recreates_resampler(monkeypatch):
    from unittest.mock import Mock
    import assistant.audio as audio_module
    candidates = Mock(return_value=[17])
    monkeypatch.setattr(audio_module, "device_candidates", candidates)
    mic = Microphone(device="Realtek", capture_rate=48000)
    previous = mic._resampler
    mic._ring.append(np.ones(512, dtype=np.int16))
    enter = Mock()
    monkeypatch.setattr(Microphone, "__enter__", enter)
    mic.reopen()
    assert mic._resampler is not None and mic._resampler is not previous
    candidates.assert_called_with("Realtek", False)
    assert not mic._ring
    enter.assert_called_once()
