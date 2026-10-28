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


def test_diagnostic_history_survives_stalled_consumer_flush_and_discard():
    mic = Microphone()
    data = np.arange(16000, dtype=np.int16).reshape(-1, 1)
    mic._callback(data, len(data), None, None)
    assert len(mic._queue) == 1  # consumer has not called read()
    live_resampler = mic._resampler
    mic.flush()
    assert not mic._queue and not mic._ring
    assert mic._resampler is live_resampler
    np.testing.assert_array_equal(mic.recent_diagnostic_audio(), data[:, 0])

    with mic.discard_capture():
        mic._callback(np.full((8000, 1), 23, dtype=np.int16), 8000, None, None)
    assert not mic._queue
    recent = mic.recent_diagnostic_audio()
    np.testing.assert_array_equal(recent[-8000:], np.full(8000, 23, dtype=np.int16))
    assert mic._diagnostic_samples <= mic._diagnostic_sample_limit


def test_diagnostic_history_is_sample_bounded_and_discards_stale_chunks(monkeypatch):
    import assistant.audio as audio_module
    mic = Microphone()
    perf_now = [100.0]
    monkeypatch.setattr(audio_module.time, "perf_counter", lambda: perf_now[0])
    oversized = np.arange(31 * 16000, dtype=np.int32).astype(np.int16).reshape(-1, 1)
    mic._callback(oversized, len(oversized), None, None)
    assert mic._diagnostic_samples == 30 * 16000
    np.testing.assert_array_equal(mic.recent_diagnostic_audio(), oversized[-30 * 16000:, 0])

    perf_now[0] = 140.0
    mic._callback(np.ones((16000, 1), dtype=np.int16), 16000, None, None)
    assert mic._diagnostic_samples == 16000
    np.testing.assert_array_equal(mic.recent_diagnostic_audio(), np.ones(16000, dtype=np.int16))


def test_diagnostic_snapshot_resamples_native_48khz_without_touching_live_state():
    mic = Microphone(capture_rate=48000)
    native = (12000 * np.sin(np.arange(48000) * 0.137)).astype(np.int16)
    mic._callback(native.reshape(-1, 1), len(native), None, None)
    resampler = mic._resampler
    live_queue = list(mic._queue)
    output_pending = mic._output_pending.copy()
    expected = soxr.resample(native.astype(np.float32), 48000, 16000)
    actual = mic.recent_diagnostic_audio()
    np.testing.assert_array_equal(actual, np.clip(expected, -32768, 32767).astype(np.int16))
    assert mic._resampler is resampler
    assert len(mic._queue) == len(live_queue)
    np.testing.assert_array_equal(mic._output_pending, output_pending)


def test_diagnostic_history_clears_when_device_exits():
    mic = Microphone()
    mic._callback(np.ones((512, 1), dtype=np.int16), 512, None, None)
    assert mic.recent_diagnostic_audio().size == 512
    mic.__exit__()
    assert mic.recent_diagnostic_audio().size == 0


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


def test_diagnostic_pair_retains_read_owned_pairs_and_respects_end_time(monkeypatch):
    import assistant.audio as audio_module
    mic = Microphone()
    class Echo:
        active = True
        def process(self, frame, _timestamp):
            return frame.astype(np.int16) + 100
    mic.echo = Echo()
    done = iter([10.1, 10.2, 10.3])
    monkeypatch.setattr(audio_module.time, "perf_counter", lambda: next(done))
    for i, stamp in enumerate((9.8, 9.9, 10.0)):
        mic._queue.append((stamp, np.full(512, i + 1, dtype=np.int16)))
        mic.read()
    pair = mic.recent_diagnostic_pair(end_time=10.25)
    assert pair.coverage["read_start"] == 0
    assert pair.coverage["read_end_exclusive"] == 2
    assert pair.coverage["pairing"] == "read_invocation"
    assert pair.coverage["capture_continuity"] == "unverified"
    np.testing.assert_array_equal(pair.raw, np.array([1] * 512 + [2] * 512, dtype=np.int16))
    np.testing.assert_array_equal(pair.processed, pair.raw + 100)
    pair.raw[:] = -1
    assert mic.recent_diagnostic_pair(end_time=10.25).raw[0] == 1


def test_diagnostic_pair_flush_marks_break_but_keeps_tail(monkeypatch):
    import assistant.audio as audio_module
    mic = Microphone()
    class Echo:
        active = True
        def process(self, frame, _timestamp): return frame.copy()
    mic.echo = Echo()
    monkeypatch.setattr(audio_module.time, "perf_counter", lambda: 10.0)
    mic._queue.append((9.9, np.ones(512, dtype=np.int16)))
    mic.read()
    mic.flush()
    snap = mic.recent_diagnostic_pair(end_time=10.1)
    assert snap.raw.size == 512 and snap.processed.size == 512
    assert snap.coverage["status"] == "partial"
    assert "flush_after_tail" in snap.coverage["break_reasons"]


def test_diagnostic_pair_latest_inactive_echo_fails_closed():
    mic = Microphone()
    class Echo:
        active = False
        def process(self, frame, _timestamp): return frame.copy()
    mic.echo = Echo()
    mic._queue.append((1.0, np.ones(512, dtype=np.int16)))
    mic.read()
    snap = mic.recent_diagnostic_pair()
    assert snap.coverage["status"] == "missing"
    assert snap.processed is None and snap.raw.size == 0
    assert "inactive_or_invalid_clean" in snap.coverage["break_reasons"]


@pytest.mark.parametrize("field,code", [
    ("queue_overflow_count", "callback_overflow"),
    ("_callback_status_count", "callback_status"),
    ("intentional_discard_count", "intentional_discard"),
])
def test_diagnostic_pair_marks_capture_discontinuity_counters(field, code, monkeypatch):
    import assistant.audio as audio_module
    mic = Microphone()
    class Echo:
        active = True
        def process(self, frame, _timestamp): return frame.copy()
    mic.echo = Echo()
    ticks = iter([20.0, 20.1])
    monkeypatch.setattr(audio_module.time, "perf_counter", lambda: next(ticks))
    mic._queue.append((10.0, np.ones(512, dtype=np.int16)))
    mic.read()
    setattr(mic, field, getattr(mic, field) + 1)
    mic._queue.append((10.1, np.full(512, 2, dtype=np.int16)))
    mic.read()
    snap = mic.recent_diagnostic_pair(end_time=20.2)
    assert code in snap.coverage["break_reasons"]
    assert snap.raw.size == 512


def test_diagnostic_pair_read_history_is_independent_of_later_callback_data():
    mic = Microphone()
    class Echo:
        active = True
        def process(self, frame, _timestamp): return frame.copy()
    mic.echo = Echo()
    mic._queue.append((10.0, np.ones(512, dtype=np.int16)))
    mic.read()
    mic._callback(np.full((512, 1), 9, dtype=np.int16), 512, None, None)
    snap = mic.recent_diagnostic_pair()
    assert snap.raw.size == 512
    np.testing.assert_array_equal(snap.raw, np.ones(512, dtype=np.int16))


def test_diagnostic_pair_uses_whole_records_under_sample_and_count_bounds():
    mic = Microphone()
    frame = np.ones(1000, dtype=np.int16)
    for i in range(481):
        mic._append_diagnostic_pair(frame, frame, 100.0 + i / 1000, 50.0 + i / 1000)
    snap = mic.recent_diagnostic_pair(end_time=101.0)
    assert snap.raw.size <= 480000
    assert snap.raw.size % frame.size == 0
    assert len(mic._pair_history) <= 4096
    assert mic._pair_samples == sum(r["raw"].size for r in mic._pair_history)
    count_bounded = Microphone()
    one = np.ones(1, dtype=np.int16)
    for i in range(4097):
        count_bounded._append_diagnostic_pair(one, one, 100.0 + i / 10000,
                                              50.0 + i / 10000)
    assert len(count_bounded._pair_history) == 4096
    age_bounded = Microphone()
    for completed in (100.0, 110.0, 120.0, 131.0):
        age_bounded._append_diagnostic_pair(one, one, completed, completed)
    assert all(131.0 - r["completed"] <= 30.0 for r in age_bounded._pair_history)


def test_diagnostic_pair_append_and_snapshot_are_thread_safe():
    import threading
    mic = Microphone()
    frame = np.full(512, 7, dtype=np.int16)
    start = threading.Event()
    def append_many():
        start.wait()
        for i in range(300):
            mic._append_diagnostic_pair(frame, frame, 100.0 + i / 1000,
                                        50.0 + i / 1000)
    worker = threading.Thread(target=append_many)
    worker.start()
    start.set()
    for _ in range(300):
        snap = mic.recent_diagnostic_pair(end_time=101.0)
        assert snap.raw.size == snap.coverage["retained_samples"]
        if snap.processed is not None:
            assert snap.processed.shape == snap.raw.shape
    worker.join(timeout=3)
    assert not worker.is_alive()
    assert mic._pair_samples == 300 * 512


def test_diagnostic_pair_reference_generation_reset_and_nonmonotonic_source_break():
    from types import SimpleNamespace
    mic = Microphone()
    class Echo:
        active = True
        def process(self, frame, _timestamp): return frame.copy()
    echo = Echo()
    echo.reference = SimpleNamespace(generation=1)
    mic.echo = echo
    frame = np.ones(512, dtype=np.int16)
    mic._append_diagnostic_pair(frame, frame, 10.0, 20.0)
    echo.reference.generation = 2
    mic._append_diagnostic_pair(frame, frame, 10.1, 21.0)
    snap = mic.recent_diagnostic_pair(end_time=10.2)
    assert "echo_generation" in snap.coverage["break_reasons"]
    mic._pair_history.clear()
    mic._pair_samples = 0
    mic._append_diagnostic_pair(frame, frame, 11.0, 30.0)
    mic._append_diagnostic_pair(frame, frame, 11.1, 29.0)
    snap = mic.recent_diagnostic_pair(end_time=11.2)
    assert "nonmonotonic_source_time" in snap.coverage["break_reasons"]


def test_diagnostic_pair_nonfinite_source_time_is_omitted():
    mic = Microphone()
    frame = np.ones(512, dtype=np.int16)
    mic._append_diagnostic_pair(frame, frame, 10.0, float("nan"))
    snap = mic.recent_diagnostic_pair(end_time=10.1)
    assert snap.coverage["estimated_source_start"] is None
    assert snap.coverage["estimated_source_end"] is None


def test_diagnostic_pair_omits_both_source_bounds_when_first_is_unknown():
    mic = Microphone()
    frame = np.ones(512, dtype=np.int16)
    mic._append_diagnostic_pair(frame, frame, 10.0, float("nan"))
    mic._append_diagnostic_pair(frame, frame, 10.1, 20.1)
    snap = mic.recent_diagnostic_pair(end_time=10.2)
    assert snap.coverage["estimated_source_start"] is None
    assert snap.coverage["estimated_source_end"] is None


def test_diagnostic_pair_failed_read_is_a_break_and_failed_latest_fails_closed():
    mic = Microphone()
    frame = np.ones(512, dtype=np.int16)
    mic._append_diagnostic_pair(frame, frame, 10.0, 20.0)
    class BadArray:
        def __array__(self, dtype=None, copy=None):
            raise RuntimeError("synthetic copy failure")
    mic._append_diagnostic_pair(BadArray(), frame, 10.1, 21.0)
    middle = mic.recent_diagnostic_pair(end_time=10.15)
    assert middle.coverage["status"] == "failed"
    assert "pair_append_failure" in middle.coverage["break_reasons"]
    mic._append_diagnostic_pair(frame, frame, 10.2, 22.0)
    later = mic.recent_diagnostic_pair(end_time=10.25)
    assert later.coverage["status"] == "partial"
    assert later.raw.size == frame.size
    assert "pair_append_failure" in later.coverage["break_reasons"]
    mic._append_diagnostic_pair(frame, frame, 100.0, 110.0)
    fresh = mic.recent_diagnostic_pair(seconds=1.0, end_time=100.1)
    assert fresh.coverage["status"] == "available"  # no sticky lifetime failure


def test_diagnostic_pair_snapshot_freezes_pending_tombstone_during_copy():
    import threading
    mic = Microphone()
    entered = threading.Event()
    release = threading.Event()
    class SlowArray:
        def __array__(self, dtype=None, copy=None):
            entered.set()
            assert release.wait(2)
            return np.ones(512, dtype=np.int16)
    frame = np.ones(512, dtype=np.int16)
    writer = threading.Thread(target=mic._append_diagnostic_pair,
                               args=(SlowArray(), frame, 10.0, 20.0))
    writer.start()
    assert entered.wait(2)
    pending = mic.recent_diagnostic_pair(end_time=10.1)
    assert pending.coverage["status"] == "missing"
    assert pending.raw.size == 0
    release.set()
    writer.join(timeout=2)
    assert not writer.is_alive()
    assert pending.coverage["status"] == "missing"  # copied snapshot did not change
    assert mic.recent_diagnostic_pair(end_time=10.1).coverage["status"] == "available"


def test_diagnostic_pair_stale_pending_commit_cannot_restore_closed_history():
    import threading
    mic = Microphone()
    entered = threading.Event()
    release = threading.Event()
    class SlowArray:
        def __array__(self, dtype=None, copy=None):
            entered.set()
            assert release.wait(2)
            return np.ones(512, dtype=np.int16)
    frame = np.ones(512, dtype=np.int16)
    writer = threading.Thread(target=mic._append_diagnostic_pair,
                               args=(SlowArray(), frame, 10.0, 20.0))
    writer.start()
    assert entered.wait(2)
    mic.__exit__()
    release.set()
    writer.join(timeout=2)
    assert not writer.is_alive()
    assert not mic._pair_history
    mic._append_diagnostic_pair(frame, frame, 10.2, 21.0)
    snap = mic.recent_diagnostic_pair(end_time=10.3)
    assert snap.coverage["status"] == "available"
    assert snap.raw.size == frame.size
    assert snap.coverage["read_start"] == 1


def test_diagnostic_pair_stale_pending_commit_after_flush_stays_a_break():
    import threading
    mic = Microphone()
    entered = threading.Event()
    release = threading.Event()
    class SlowArray:
        def __array__(self, dtype=None, copy=None):
            entered.set()
            assert release.wait(2)
            return np.ones(512, dtype=np.int16)
    frame = np.ones(512, dtype=np.int16)
    writer = threading.Thread(target=mic._append_diagnostic_pair,
                               args=(SlowArray(), frame, 10.0, 20.0))
    writer.start()
    assert entered.wait(2)
    mic.flush()
    release.set()
    writer.join(timeout=2)
    assert not writer.is_alive()
    mic._append_diagnostic_pair(frame, frame, 10.2, 21.0)
    snap = mic.recent_diagnostic_pair(end_time=10.3)
    assert snap.coverage["status"] == "partial"
    assert snap.raw.size == frame.size
    assert "flush" in snap.coverage["break_reasons"]


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


def test_callback_status_is_reported_from_read_not_callback(caplog, monkeypatch):
    from assistant import audio
    monkeypatch.setattr(audio.log, "handlers", [caplog.handler])
    monkeypatch.setattr(audio.log, "propagate", False)
    mic = Microphone()
    data = np.ones((512, 1), dtype=np.int16)

    mic._callback(data, 512, None, "input_overflow")
    assert not caplog.records
    mic.read()
    warnings = [record for record in caplog.records
                if "callback reported status" in record.message]
    assert len(warnings) == 1
    assert "input_overflow" in warnings[0].message


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
    mic._callback(np.ones((512, 1), dtype=np.int16), 512, None, None)
    mic._append_diagnostic_pair(np.ones(512, dtype=np.int16),
                                np.ones(512, dtype=np.int16), 1.0, 1.0)
    generation = mic.recent_diagnostic_pair(end_time=1.1).coverage["diagnostic_generation"]
    enter = Mock()
    monkeypatch.setattr(Microphone, "__enter__", enter)
    mic.reopen()
    assert mic._resampler is not None and mic._resampler is not previous
    candidates.assert_called_with("Realtek", False)
    assert not mic._ring
    assert mic.recent_diagnostic_audio().size == 0
    assert mic.recent_diagnostic_pair(end_time=1.1).raw.size == 0
    assert mic.recent_diagnostic_pair(end_time=1.1).coverage["diagnostic_generation"] > generation
    enter.assert_called_once()
