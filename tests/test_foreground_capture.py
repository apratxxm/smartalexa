"""Deterministic fake-device contracts for the foreground capture adapter."""
import numpy as np
import pytest
from threading import Thread

from assistant.foreground_capture import ForegroundCaptureDriver


class FakeClock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value


class FakeMic:
    def __init__(self, clock, scores=(), *, overflow_at_vad=None, frames=None):
        self.clock = clock
        self.scores = list(scores)
        self.frames = list(frames or [])
        self.queue_overflow_count = 0
        self.vad_frames = 0
        self.reads = 0
        self.overflow_at_vad = overflow_at_vad

    def read(self, timeout_s):
        assert 0 < timeout_s <= 0.05
        self.reads += 1
        self.clock.value += 0.032
        if self.frames:
            return self.frames.pop(0)
        return np.full(512, self.reads, dtype=np.int16)

    def vad_frame(self, frame):
        self.vad_frames += 1
        if self.overflow_at_vad == self.vad_frames:
            self.queue_overflow_count += 1
        return frame


class FakeVAD:
    def __init__(self, mic, *, overflow_on_score=False, mutate_frame=False):
        self.mic = mic
        self.resets = 0
        self.calls = 0
        self.overflow_on_score = overflow_on_score
        self.mutate_frame = mutate_frame
        self.reset_failures = 0

    def reset(self):
        self.resets += 1
        if self.reset_failures:
            self.reset_failures -= 1
            raise RuntimeError("reset probe")

    def __call__(self, _frame):
        self.calls += 1
        if self.mutate_frame:
            _frame[:] = -7
        if self.overflow_on_score:
            self.mic.queue_overflow_count += 1
        return self.mic.scores.pop(0) if self.mic.scores else 0.0


def record(driver, mic, clock, **overrides):
    options = dict(mode="reply", turn_id="turn-a", generation=4,
                   start_timeout_seconds=0.12, max_duration_seconds=2.0,
                   deadline=10.0, threshold=0.5, silence_seconds=0.064)
    options.update(overrides)
    return driver.record(mic, **options)


def setup(scores=(), *, overflow_on_score=False, mutate_frame=False, **mic_options):
    clock = FakeClock()
    mic = FakeMic(clock, scores, **mic_options)
    vad = FakeVAD(mic, overflow_on_score=overflow_on_score,
                  mutate_frame=mutate_frame)
    return clock, mic, vad, ForegroundCaptureDriver(vad, clock=clock)


def test_silence_completion_returns_manager_pcm_and_vad_once_per_frame():
    clock, mic, vad, driver = setup([0.9, 0.0, 0.0])
    result = record(driver, mic, clock)
    assert result["capture_status"] == "complete"
    assert result["speech_started"] is True
    assert result["speech_start_ms"] == 0
    assert result["audio"].dtype == np.int16 and result["audio"].size == 3 * 512
    assert vad.resets == 1
    assert vad.calls == mic.vad_frames == mic.reads == 3
    assert result["source_epoch"] == 1


def test_start_timeout_and_stalled_wall_clock_have_empty_audio():
    clock, mic, vad, driver = setup()
    mic.read = lambda timeout_s: (_ for _ in ()).throw(TimeoutError())
    original_read = mic.read

    def stalled(timeout_s):
        clock.value += timeout_s
        return original_read(timeout_s)

    mic.read = stalled
    result = record(driver, mic, clock, start_timeout_seconds=0.11)
    assert result["capture_status"] == "start_timeout"
    assert result["audio"].size == 0 and not result["speech_started"]
    assert vad.resets == 1 and vad.calls == 0


def test_overflow_during_vad_discards_frame_and_returns_no_partial_audio():
    clock, mic, vad, driver = setup([0.9], overflow_on_score=True)
    result = record(driver, mic, clock)
    assert result["capture_status"] == "gap"
    assert result["audio"].size == 0 and result["truncated"] is False
    assert vad.calls == mic.vad_frames == 1


def test_supersession_after_frame_returns_empty_audio():
    clock, mic, vad, driver = setup([0.9])
    checks = iter([False, False, True])
    result = record(driver, mic, clock, cancelled=lambda: next(checks))
    assert result["capture_status"] == "superseded"
    assert result["audio"].size == 0 and not result["speech_started"]


def test_malformed_frame_fails_explicitly_without_partial_decode():
    clock, mic, vad, driver = setup(frames=[np.zeros(32, dtype=np.int16)])
    with pytest.raises(RuntimeError, match="malformed frame"):
        record(driver, mic, clock)
    assert vad.resets == 1 and vad.calls == 0


def test_max_duration_never_returns_a_partial_prefix():
    clock, mic, vad, driver = setup([0.9, 0.9])
    result = record(driver, mic, clock, max_duration_seconds=0.064)
    assert result["capture_status"] == "max_duration"
    assert result["truncated"] is True
    assert result["audio"].size == 0 and not result["speech_started"]


def test_cancel_probe_exception_fails_closed():
    clock, mic, vad, driver = setup()

    def broken_cancel_probe():
        raise RuntimeError("unavailable")

    result = record(driver, mic, clock, cancelled=broken_cancel_probe)
    assert result["capture_status"] == "superseded"
    assert result["audio"].size == 0
    assert mic.reads == vad.calls == 0


def test_reset_failure_cleans_request_and_next_capture_can_run():
    clock, mic, vad, driver = setup([0.9, 0.0, 0.0])
    vad.reset_failures = 1
    with pytest.raises(RuntimeError, match="reset failed"):
        record(driver, mic, clock)
    result = record(driver, mic, clock)
    assert result["capture_status"] == "complete"
    assert vad.resets == 2


def test_vad_mutation_cannot_change_saved_raw_capture():
    clock, mic, vad, driver = setup([0.9, 0.0, 0.0], mutate_frame=True)
    result = record(driver, mic, clock)
    assert result["capture_status"] == "complete"
    assert result["audio"][0] == 1
    assert result["audio"][512] == 2


def test_driver_manager_remains_owned_by_first_capture_thread():
    clock, mic, vad, driver = setup([0.9, 0.0, 0.0])
    record(driver, mic, clock)
    errors = []

    def call_from_other_thread():
        other_clock, other_mic = clock, FakeMic(clock)
        try:
            record(driver, other_mic, other_clock)
        except RuntimeError as exc:
            errors.append(str(exc))

    worker = Thread(target=call_from_other_thread)
    worker.start()
    worker.join()
    assert errors and "owner thread" in errors[0]
