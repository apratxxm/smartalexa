"""Deterministic contract tests for the pure foreground capture manager."""
from dataclasses import FrozenInstanceError
from threading import Thread

import numpy as np
import pytest

from assistant.capture_requests import (CaptureAudio, CaptureOutcome, CaptureRequest,
                                        CaptureRequestManager)


def request(now, request_id="r1", *, generation=0, start="immediate",
            timeout=1.0, duration=1.0, deadline=None, mode="command"):
    return CaptureRequest(request_id, "turn-1", generation, mode, start,
                          timeout, duration, now + 10 if deadline is None else deadline)


def frame(value=0):
    return np.full(512, value, dtype=np.int16)


def feed(manager, now, sample, *, epoch=0, value=1, score=0.9, eligible=True):
    now[0] += 0.032
    source = frame(value)
    manager.feed(epoch, sample, now[0], source, frame(value + 1), score,
                 eligible=eligible)
    return source


def test_request_validation_rejects_boolean_numbers_bad_strings_and_long_duration():
    with pytest.raises(ValueError):
        CaptureRequest("bad\n", "turn", 0, "command", "immediate", 1, 1, 2)
    with pytest.raises(ValueError):
        request(0, timeout=True)
    with pytest.raises(ValueError):
        request(0, duration=30.01)
    with pytest.raises(ValueError):
        request(0, deadline=float("inf"))


def test_one_active_request_and_reserved_result_backpressure():
    now = [0.0]
    manager = CaptureRequestManager(clock=lambda: now[0], max_results=1)
    first = request(0)
    assert manager.request(first).status == "accepted"
    assert manager.request(request(0, "busy")).status == "busy"
    assert manager.cancel(first.request_id, first.generation)
    assert manager.request(request(0, "blocked")).status == "result_backpressure"
    assert manager.poll()[0].status == "cancelled"
    assert manager.request(request(0, "after-poll")).accepted


def test_after_prompt_waits_for_matching_signal_and_start_timeout_is_explicit():
    now = [0.0]
    manager = CaptureRequestManager(clock=lambda: now[0])
    req = request(0, start="after_prompt", timeout=0.1, deadline=5)
    assert manager.request(req).accepted
    feed(manager, now, 0, score=1.0)
    assert manager.prompt_finished("other", 0) is False
    assert manager.prompt_finished(req.request_id, 9) is False
    assert manager.prompt_finished(req.request_id, req.generation)
    now[0] = 0.2
    assert manager.expire()
    outcome = manager.poll()[0]
    assert outcome.status == "start_timeout" and outcome.audio is None


@pytest.mark.parametrize("start,deadline,timeout,expected", [
    ("immediate", 5.0, 0.1, "start_timeout"),
    ("after_prompt", 5.0, 0.1, "start_timeout"),
    ("after_prompt", 0.1, 1.0, "deadline"),
])
def test_ineligible_playback_frames_still_resolve_timeouts(start, deadline, timeout, expected):
    now = [0.0]
    manager = CaptureRequestManager(clock=lambda: now[0])
    req = request(0, start=start, timeout=timeout, deadline=deadline)
    assert manager.request(req).accepted
    if start == "after_prompt" and expected == "start_timeout":
        assert manager.prompt_finished(req.request_id, req.generation)
    now[0] = 0.2
    manager.feed(0, 0, 0.1, frame(), probability=0.9, eligible=False)
    assert manager.poll()[0].status == expected


def test_playback_breaks_preroll_and_collected_speech_continuity():
    now = [0.0]
    manager = CaptureRequestManager(clock=lambda: now[0], min_speech_seconds=0,
                                    silence_seconds=0.032, padding_seconds=0.1)
    assert manager.request(request(0)).accepted
    feed(manager, now, 0, value=4, score=0.0)
    now[0] += 0.032
    manager.feed(0, 512, now[0], frame(90), probability=0.0, eligible=False)
    feed(manager, now, 1024, value=7, score=0.9)
    now[0] += 0.032
    manager.feed(0, 1536, now[0], frame(), probability=0.9, eligible=False)
    outcome = manager.poll()[0]
    assert outcome.status == "gap"
    assert outcome.audio.start_sample == 1024
    assert outcome.audio.raw[0] == 7


def test_noncontiguous_preroll_keeps_only_valid_contiguous_suffix():
    now = [0.0]
    manager = CaptureRequestManager(clock=lambda: now[0], min_speech_seconds=0,
                                    silence_seconds=0.032, padding_seconds=0.1)
    assert manager.request(request(0)).accepted
    feed(manager, now, 0, value=1, score=0.0)
    feed(manager, now, 1024, value=2, score=0.0)
    feed(manager, now, 1536, value=3, score=0.9)
    feed(manager, now, 2048, value=4, score=0.0)
    outcome = manager.poll()[0]
    assert outcome.status == "complete"
    assert outcome.audio.start_sample == 1024
    assert np.array_equal(outcome.audio.raw,
                          np.concatenate((frame(2), frame(3), frame(4))))


def test_max_duration_uses_wall_clock_as_well_as_frame_count():
    now = [0.0]
    manager = CaptureRequestManager(clock=lambda: now[0], min_speech_seconds=0,
                                    silence_seconds=0.064, padding_seconds=0)
    assert manager.request(request(0, duration=2.0, deadline=20)).accepted
    feed(manager, now, 0, score=0.9)
    now[0] = 3.0
    assert manager.expire()
    outcome = manager.poll()[0]
    assert outcome.status == "max_duration" and outcome.audio.truncated
    assert outcome.audio.raw.size == 512


def test_absolute_deadline_applies_while_waiting_for_prompt():
    now = [0.0]
    manager = CaptureRequestManager(clock=lambda: now[0])
    req = request(0, start="after_prompt", deadline=0.1)
    assert manager.request(req).accepted
    now[0] = 0.1
    assert manager.prompt_finished(req.request_id, req.generation) is False
    assert manager.poll()[0].status == "deadline"


def test_malformed_prompt_generation_does_not_trigger_or_change_terminal_state():
    now = [0.0]
    manager = CaptureRequestManager(clock=lambda: now[0])
    req = request(0, start="after_prompt", deadline=0.1)
    assert manager.request(req).accepted
    now[0] = 0.1
    with pytest.raises(ValueError):
        manager.prompt_finished(req.request_id, True)
    assert manager.poll() == ()
    assert manager.expire()
    assert manager.poll()[0].status == "deadline"


def test_voiced_undercount_can_complete_from_adequate_raw_capture():
    now = [0.0]
    manager = CaptureRequestManager(clock=lambda: now[0], min_speech_seconds=0.096,
                                    silence_seconds=0.064, padding_seconds=0)
    assert manager.request(request(0, duration=1)).accepted
    feed(manager, now, 0, score=0.9)
    feed(manager, now, 512, score=0.0)
    feed(manager, now, 1024, score=0.0)
    outcome = manager.poll()[0]
    assert outcome.status == "complete"
    assert outcome.audio.raw.size == 3 * 512


def test_playback_frames_are_ineligible_and_source_gap_never_joins_pcm():
    now = [0.0]
    manager = CaptureRequestManager(clock=lambda: now[0], padding_seconds=0)
    assert manager.request(request(0)).accepted
    feed(manager, now, 0, score=1.0, eligible=False)
    feed(manager, now, 512, score=0.9)
    feed(manager, now, 1536, score=0.9)
    outcome = manager.poll()[0]
    assert outcome.status == "gap"
    assert outcome.audio.gap and outcome.audio.raw.size == 512
    assert outcome.audio.start_sample == 512 and outcome.audio.end_sample == 1024


def test_max_duration_is_truncated_and_never_reported_as_complete():
    now = [0.0]
    manager = CaptureRequestManager(clock=lambda: now[0], min_speech_seconds=0,
                                    silence_seconds=0.032, padding_seconds=0)
    assert manager.request(request(0, duration=0.064)).accepted
    feed(manager, now, 0, score=0.9)
    feed(manager, now, 512, score=0.0)
    outcome = manager.poll()[0]
    assert outcome.status == "max_duration"
    assert outcome.audio.truncated


def test_frames_are_copied_and_outcome_pcm_is_immutable():
    now = [0.0]
    manager = CaptureRequestManager(clock=lambda: now[0], min_speech_seconds=0,
                                    silence_seconds=0.032, padding_seconds=0)
    assert manager.request(request(0)).accepted
    source = feed(manager, now, 0, value=7, score=0.9)
    source[:] = -1
    feed(manager, now, 512, score=0.0)
    outcome = manager.poll()[0]
    assert outcome.audio.raw[0] == 7
    with pytest.raises(ValueError):
        outcome.audio.raw.setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        outcome.status = "gap"


def test_stale_cancellation_and_cross_thread_calls_are_inert_or_rejected():
    now = [0.0]
    manager = CaptureRequestManager(clock=lambda: now[0])
    req = request(0)
    assert manager.request(req).accepted
    assert manager.cancel(req.request_id, req.generation + 1) is False
    for generation in (True, 0.0, float("nan")):
        with pytest.raises(ValueError):
            manager.cancel(req.request_id, generation)
        with pytest.raises(ValueError):
            manager.prompt_finished(req.request_id, generation)
    with pytest.raises(ValueError):
        manager.cancel(req.request_id, req.generation, superseded=1)
    with pytest.raises(ValueError):
        manager.cancel(4.5, req.generation)
    assert manager.poll() == ()
    errors = []

    def other_owner():
        try:
            manager.poll()
        except RuntimeError as exc:
            errors.append(str(exc))

    worker = Thread(target=other_owner)
    worker.start()
    worker.join()
    assert errors and "owner thread" in errors[0]
    assert manager.cancel(req.request_id, req.generation, superseded=True)
    assert manager.poll()[0].status == "superseded"


def test_future_source_time_is_rejected_before_capture_state_changes():
    now = [0.0]
    manager = CaptureRequestManager(clock=lambda: now[0])
    req = request(0)
    assert manager.request(req).accepted
    with pytest.raises(ValueError, match="owner clock"):
        manager.feed(0, 0, 1.0, frame(), probability=0.9)
    now[0] = 0.1
    manager.feed(0, 0, 0.1, frame(5), probability=0.9)
    manager.gap()
    assert manager.poll()[0].status == "gap"


def test_outcome_constructor_enforces_complete_and_limit_metadata():
    with pytest.raises(ValueError, match="complete outcomes"):
        CaptureOutcome("r", "t", 0, "command", "complete", None, 1.0)
    audio = CaptureAudio(0, 0, 512, frame(), None, 0.5, 1.0)
    with pytest.raises(ValueError, match="gap metadata"):
        CaptureOutcome("r", "t", 0, "command", "gap", 0.5, 1.0, audio)
    with pytest.raises(ValueError, match="truncation metadata"):
        CaptureOutcome("r", "t", 0, "command", "max_duration", 0.5, 1.0, audio)
