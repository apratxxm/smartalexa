from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest

from assistant.audio_jobs import DecodeJob, DecodeResult
from assistant.async_listener import AcousticProposal, AsyncSttListener


class ManualLane:
    def __init__(self):
        self.jobs = []
        self.epochs = []
        self.suspended = False
        self.inflight = False
        self.closed = False
        self.accept = True

    def submit(self, job):
        if not self.accept or self.suspended or self.closed:
            return False
        self.jobs.append(job)
        return True

    def poll(self):
        return ()

    def invalidate(self, epoch):
        self.epochs.append(epoch)

    def suspend(self, timeout):
        self.suspended = True
        return not self.inflight

    def resume(self, epoch):
        if self.inflight or self.closed:
            return False
        self.suspended = False
        self.epochs.append(epoch)
        return True

    def close(self, timeout):
        self.closed = True
        return not self.inflight


class FakeVAD:
    def __init__(self):
        self.probabilities = deque()
        self.calls = []
        self.resets = 0

    def __call__(self, frame):
        self.calls.append(frame.copy())
        return self.probabilities.popleft()

    def reset(self):
        self.resets += 1


def make_listener(*, overrides=None, verify_text=None, lane=None, vad=None, now=None):
    values = {
        "listener.phrases": ["alexa"], "listener.aliases": {},
        "listener.allow_fuzzy": False, "listener.fuzzy_cutoff": 0.72,
        "listener.mid_phrases": ["alexa"], "listener.mid_fuzzy_cutoff": 0.9,
        "listener.vad_threshold": 0.35, "listener.min_speech_ms": 0,
        "listener.silence_ms": 64, "listener.max_utterance_s": 0.128,
        "listener.pad_ms": 0, "listener.overlap_ms": 64,
        "listener.extend_ms": 128, "listener.peek_first_ms": 32000,
        "listener.peek_every_ms": 32, "listener.peek_until_ms": 10000,
        "listener.peek_window_ms": 32,
    }
    values.update(overrides or {})
    lane = lane or ManualLane()
    vad = vad or FakeVAD()
    now = now or [1.0]
    cfg = SimpleNamespace(get=lambda key, default=None: values.get(key, default))
    listener = AsyncSttListener(cfg, vad, lane, clock=lambda: now[0],
                                verify_text=verify_text)
    return listener, lane, vad, now


def feed(listener, vad, now, probability, *, sample=10, processed=True,
         authorization=None, acoustic_proposal=None):
    frame = np.full(512, sample, dtype=np.int16)
    clean = np.full(512, sample + 1, dtype=np.int16) if processed else None
    vad.probabilities.append(probability)
    now[0] += 0.032
    listener.feed_async(frame, clean, captured_at=now[0],
                        authorization=authorization,
                        acoustic_proposal=acoustic_proposal)


def result(job, transcript, *, latency=4, finished_at=None, error=""):
    return DecodeResult(job, transcript=transcript, latency_ms=latency,
                        error_category=error,
                        finished_at=(job.snapshot.completed_at + 0.01
                                     if finished_at is None else finished_at))


def jobs_for(lane, purpose):
    return [job for job in lane.jobs if job.purpose == purpose]


def test_peek_from_old_segment_cannot_wake_a_new_segment():
    listener, lane, vad, now = make_listener(overrides={
        "listener.peek_first_ms": 32, "listener.peek_window_ms": 32,
        "listener.silence_ms": 32,
    })
    feed(listener, vad, now, 0.9, sample=1)
    old_peek = jobs_for(lane, "peek")[-1]
    old_segment = old_peek.snapshot.segment_id
    feed(listener, vad, now, 0.0, sample=2)
    first_final = jobs_for(lane, "final")[-1]
    feed(listener, vad, now, 0.9, sample=3)
    assert listener._active.segment_id != old_segment
    feed(listener, vad, now, 0.0, sample=4)
    feed(listener, vad, now, 0.0, sample=5)

    listener.apply_result(result(old_peek, "alexa"))
    assert not [event for event in listener.poll_events() if event.kind == "wake"]
    listener.apply_result(result(first_final, "alexa pause"))
    events = listener.poll_events()
    assert [event.kind for event in events] == ["wake", "utterance"]


def test_peek_submits_available_suffix_before_full_window():
    listener, lane, vad, now = make_listener(overrides={
        "listener.peek_first_ms": 64,
        "listener.peek_every_ms": 32000,
        "listener.peek_window_ms": 96,
    })
    feed(listener, vad, now, .9, sample=1)
    assert not jobs_for(lane, "peek")
    feed(listener, vad, now, .9, sample=2)

    peeks = jobs_for(lane, "peek")
    assert len(peeks) == 1
    assert len(peeks[0].snapshot.audio) == 1024


def test_boundary_carries_silence_run_and_splits_delayed_continuation():
    listener, lane, vad, now = make_listener(overrides={
        "listener.max_utterance_s": 0.128,
        "listener.silence_ms": 96,
    })
    for sample, probability in ((1, .9), (2, .9), (3, 0), (4, 0)):
        feed(listener, vad, now, probability, sample=sample)
    boundary = jobs_for(lane, "boundary")[-1]
    assert boundary.snapshot.segment_id == listener._boundary.segment.segment_id
    assert listener._boundary.silence_run == 2

    feed(listener, vad, now, 0, sample=5)
    feed(listener, vad, now, .9, sample=6)
    listener.apply_result(result(boundary, "alexa play song"))

    final = jobs_for(lane, "final")[-1]
    assert final.snapshot.end_reason == "silence"
    assert len(final.snapshot.audio) == 5 * 512
    assert listener._active is not None
    assert [record.start_sample for record in listener._active.records] == [5 * 512]


def test_boundary_without_continuation_restores_prior_silence_run():
    listener, lane, vad, now = make_listener(overrides={
        "listener.max_utterance_s": 0.128,
        "listener.silence_ms": 96,
    })
    for sample, probability in ((1, .9), (2, .9), (3, 0), (4, 0)):
        feed(listener, vad, now, probability, sample=sample)
    boundary = jobs_for(lane, "boundary")[-1]

    listener.apply_result(result(boundary, "alexa play song"))
    feed(listener, vad, now, 0, sample=5)
    assert jobs_for(lane, "final")[-1].snapshot.end_reason == "silence"
    feed(listener, vad, now, .9, sample=6)
    assert listener._active is not None
    assert [record.start_sample for record in listener._active.records] == [5 * 512]


def test_capacity_rejection_does_not_advance_vad_or_capture_state():
    listener, _lane, vad, now = make_listener()
    while listener._has_capacity(2):
        listener._diagnostic("fill")
    frame = np.full(512, 99, dtype=np.int16)
    captured_at = now[0] + 0.032
    sample_counter = listener._sample_counter
    last_captured_at = listener._last_captured_at
    vad_calls = len(vad.calls)
    with pytest.raises(BufferError, match="event queue is full"):
        listener.feed_async(frame, captured_at=captured_at)
    assert len(vad.calls) == vad_calls
    assert listener._sample_counter == sample_counter
    assert listener._last_captured_at == last_captured_at

    listener.poll_events()
    vad.probabilities.append(.9)
    listener.feed_async(frame, captured_at=captured_at)
    assert len(vad.calls) == vad_calls + 1
    assert listener._sample_counter == sample_counter + 512
    assert listener._last_captured_at == captured_at


def test_boundary_abort_skips_already_terminal_pending_peek_segment():
    listener, lane, vad, now = make_listener(overrides={
        "listener.peek_first_ms": 32,
        "listener.peek_window_ms": 32,
        "listener.silence_ms": 64,
        "listener.max_utterance_s": 0.128,
        "listener.extend_ms": 64,
    })
    feed(listener, vad, now, .9, sample=1)
    old_peek = jobs_for(lane, "peek")[-1]
    feed(listener, vad, now, 0, sample=2)
    feed(listener, vad, now, 0, sample=3)
    old_final = jobs_for(lane, "final")[-1]
    listener.apply_result(result(old_final, "alexa pause"))
    assert old_peek.job_id in listener._pending
    assert listener._pending[old_peek.job_id].segment.reserved_events == 0
    listener.poll_events()

    for sample in (10, 11, 12, 13):
        feed(listener, vad, now, .9, sample=sample)
    for sample in (20, 21, 22):
        feed(listener, vad, now, .9, sample=sample)

    events = listener.poll_events()
    assert [event.reason for event in events if event.kind == "rejected"] == [
        "boundary_buffer_overflow"]
    assert not [event for event in events if event.reason == "event_reservation_missing"]


def test_final_result_survives_ordinary_reset_but_flush_invalidates_it():
    accepted, lane, vad, now = make_listener()
    for sample, prob in ((1, .9), (2, .9), (3, 0), (4, 0)):
        feed(accepted, vad, now, prob, sample=sample)
    final_job = jobs_for(lane, "final")[-1]
    assert accepted._active is None  # ordinary endpoint reset keeps the epoch
    accepted.apply_result(result(final_job, "alexa play jazz"))
    events = accepted.poll_events()
    assert [event.kind for event in events] == ["wake", "utterance"]
    delivered = events[-1].utterance
    snapshot_audio = final_job.snapshot.audio.copy()
    delivered.audio[0] = 123
    np.testing.assert_array_equal(final_job.snapshot.audio, snapshot_audio)

    stale, stale_lane, stale_vad, stale_now = make_listener()
    for sample, prob in ((1, .9), (2, .9), (3, 0), (4, 0)):
        feed(stale, stale_vad, stale_now, prob, sample=sample)
    stale_job = jobs_for(stale_lane, "final")[-1]
    old_epoch = stale.stream_epoch
    stale.flush("test reset")
    assert stale.stream_epoch == old_epoch + 1
    assert stale_lane.epochs == [stale.stream_epoch]
    stale.apply_result(result(stale_job, "alexa pause"))
    assert stale.poll_events() == ()


@pytest.mark.parametrize("wake_text", ["alexa play song", "background words"])
def test_forced_boundary_replays_cached_continuation_without_second_vad(wake_text):
    listener, lane, vad, now = make_listener(overrides={
        "listener.max_utterance_s": 0.128,
        "listener.overlap_ms": 64,
    })
    for sample in (10, 11, 12, 13):
        feed(listener, vad, now, .9, sample=sample)
    boundary = jobs_for(lane, "boundary")[-1]
    original_vad_calls = len(vad.calls)
    for sample in (20, 21):
        feed(listener, vad, now, 0.0, sample=sample)
    assert len(vad.calls) == original_vad_calls + 2
    old_segment = boundary.snapshot.segment_id
    listener.apply_result(result(boundary, wake_text))

    if wake_text.startswith("alexa"):
        final = jobs_for(lane, "final")[-1]
        assert final.snapshot.end_reason == "silence"
        assert final.snapshot.truncated is False
        expected = np.concatenate([np.full(512, value, np.int16)
                                   for value in (10, 11, 12, 13, 20, 21)])
        np.testing.assert_array_equal(final.snapshot.audio, expected)
        listener.apply_result(result(final, "alexa play song"))
        events = listener.poll_events()
        utterances = [event.utterance for event in events if event.kind == "utterance"]
        assert len(utterances) == 1
        assert utterances[0].command == "play song"
        assert utterances[0].truncated is False
        assert utterances[0].confirmed_audio_handoff is False
    else:
        final = jobs_for(lane, "final")[-1]
        expected = np.concatenate([np.full(512, value, np.int16)
                                   for value in (12, 13, 20, 21)])
        np.testing.assert_array_equal(final.snapshot.audio, expected)
        assert final.snapshot.overlap_of == old_segment
        assert len(vad.calls) == original_vad_calls + 2


def test_cached_replay_routes_frames_into_a_second_forced_boundary():
    listener, lane, vad, now = make_listener(overrides={
        "listener.max_utterance_s": 0.064,
        "listener.extend_ms": 256,
        "listener.overlap_ms": 32,
    })
    for sample in (1, 2):
        feed(listener, vad, now, .9, sample=sample)
    first = jobs_for(lane, "boundary")[-1]
    for sample in range(10, 15):
        feed(listener, vad, now, .9, sample=sample)
    vad_calls = len(vad.calls)

    listener.apply_result(result(first, "background words"))

    boundaries = jobs_for(lane, "boundary")
    assert len(boundaries) == 2
    assert len(listener._boundary.continuation) == 4
    assert len(vad.calls) == vad_calls


def test_boundary_expiry_rejects_and_invalidates_late_decode_without_action():
    listener, lane, vad, now = make_listener()
    for sample in (1, 2, 3, 4):
        feed(listener, vad, now, .9, sample=sample)
    boundary = jobs_for(lane, "boundary")[-1]
    now[0] += 4.0
    feed(listener, vad, now, .9, sample=5)
    assert listener.stream_epoch == 1
    assert lane.epochs == [1]
    assert vad.resets == 1
    assert [event.reason for event in listener.poll_events()] == ["boundary_deadline_exceeded"]
    listener.apply_result(result(boundary, "alexa play song"))
    assert listener.poll_events() == ()


def test_boundary_silence_after_extension_cap_is_truncated_at_cap():
    listener, lane, vad, now = make_listener(overrides={
        "listener.max_utterance_s": 0.064,
        "listener.extend_ms": 64,
        "listener.silence_ms": 64,
        "listener.pad_ms": 64,
    })
    for sample in (1, 2):
        feed(listener, vad, now, .9, sample=sample)
    boundary = jobs_for(lane, "boundary")[-1]
    for sample, probability in ((10, .9), (11, .9), (12, 0.0), (13, 0.0)):
        feed(listener, vad, now, probability, sample=sample)

    listener.apply_result(result(boundary, "alexa play song"))

    final = jobs_for(lane, "final")[-1]
    assert final.snapshot.end_reason == "extension_limit"
    assert final.snapshot.truncated is True
    expected = np.concatenate([np.full(512, value, np.int16)
                               for value in (1, 2, 10, 11)])
    np.testing.assert_array_equal(final.snapshot.audio, expected)
    listener.apply_result(result(final, "alexa play song"))
    utterance = next(event.utterance for event in listener.poll_events()
                     if event.kind == "utterance")
    assert utterance.command == ""
    assert utterance.truncated is True


def test_boundary_continuation_overflow_rejects_without_final_job():
    listener, lane, vad, now = make_listener(overrides={
        "listener.max_utterance_s": 0.064,
        "listener.extend_ms": 64,
        "listener.pad_ms": 0,
    })
    for sample in (1, 2):
        feed(listener, vad, now, .9, sample=sample)
    assert jobs_for(lane, "boundary")
    for sample in (10, 11, 12):
        feed(listener, vad, now, .9, sample=sample)

    events = listener.poll_events()
    assert [event.reason for event in events if event.kind == "rejected"] == [
        "boundary_buffer_overflow"]
    assert not jobs_for(lane, "final")
    assert listener.stream_epoch == 1
    assert vad.resets == 1


def test_unconfirmed_candidate_uses_command_job_after_reset_and_keeps_primary_text():
    verified = []

    def verify_text(transcript, phrase, mid):
        verified.append((transcript, phrase, mid))
        return "alexa", "pause", False

    listener, lane, vad, now = make_listener(
        overrides={"listener.allow_fuzzy": True, "listener.fuzzy_cutoff": 0.7},
        verify_text=verify_text)
    for sample, prob in ((1, .9), (2, .9), (3, 0), (4, 0)):
        feed(listener, vad, now, prob, sample=sample)
    final = jobs_for(lane, "final")[-1]
    original_audio = final.snapshot.audio.copy()
    listener.apply_result(result(final, "alex pause", latency=17))
    verify = jobs_for(lane, "verify")[-1]
    assert verify.recognizer == "command"
    np.testing.assert_array_equal(verify.snapshot.audio, original_audio)
    listener.apply_result(result(verify, "alexa pause", latency=7))
    events = listener.poll_events()
    utterance = next(event.utterance for event in events if event.kind == "utterance")
    assert verified == [("alexa pause", "alexa", False)]
    assert utterance.transcript == "alex pause"
    assert utterance.confirmation_transcript == "alexa pause"
    assert utterance.confirmation_latency_ms == 7
    assert utterance.latency_ms == 17


def test_alias_and_mid_trigger_require_frozen_per_job_authorization():
    cfg = {"listener.aliases": {"lexa": "alexa"}}
    denied, lane, vad, now = make_listener(overrides=cfg)
    for sample, prob in ((1, .9), (2, .9), (3, 0), (4, 0)):
        feed(denied, vad, now, prob, sample=sample)
    denied_final = jobs_for(lane, "final")[-1]
    denied.apply_result(result(denied_final, "lexa pause"))
    assert next(event.reason for event in denied.poll_events()
                if event.kind == "rejected") == "verification_unavailable"

    allowed, lane2, vad2, now2 = make_listener(overrides=cfg)
    auth = {"alias:alexa": True}
    for sample, prob in ((1, .9), (2, .9), (3, 0), (4, 0)):
        feed(allowed, vad2, now2, prob, sample=sample, authorization=auth)
    allowed_final = jobs_for(lane2, "final")[-1]
    assert allowed_final.snapshot.metadata["authorization"]["alias:alexa"] is True
    allowed.apply_result(result(allowed_final, "lexa pause"))
    assert any(event.kind == "utterance" for event in allowed.poll_events())


def test_final_snapshot_uses_only_last_captured_frame_authorization():
    listener, lane, vad, now = make_listener(
        overrides={"listener.aliases": {"lexa": "alexa"}})
    mutable_grant = {"alias:alexa": True}
    feed(listener, vad, now, .9, sample=1, authorization=mutable_grant)
    # Caller mutation cannot rewrite already-captured authorization evidence.
    mutable_grant["alias:alexa"] = False
    for sample, probability in ((2, .9), (3, 0.0), (4, 0.0)):
        feed(listener, vad, now, probability, sample=sample,
             authorization={"alias:alexa": False})

    final = jobs_for(lane, "final")[-1]
    assert final.snapshot.metadata["authorization"]["alias:alexa"] is False
    listener.apply_result(result(final, "lexa pause"))
    events = listener.poll_events()
    assert not any(event.kind == "utterance" for event in events)
    assert next(event.reason for event in events if event.kind == "rejected") == \
        "verification_unavailable"


@pytest.mark.parametrize(("transcript", "overrides", "expected"), [
    ("lexa pause", {"listener.aliases": {"lexa": "alexa"}},
     "alias_uncorroborated"),
    ("alxa pause", {"listener.allow_fuzzy": True,
                     "listener.fuzzy_cutoff": 0.6}, "fuzzy_uncorroborated"),
    ("background words", {}, "speech_no_trigger"),
])
def test_forced_boundary_rejections_normalize_matcher_reason_tuples(
        transcript, overrides, expected):
    listener, lane, vad, now = make_listener(overrides={
        "listener.max_utterance_s": 0.064, **overrides})
    for sample in (1, 2):
        feed(listener, vad, now, .9, sample=sample)
    boundary = jobs_for(lane, "boundary")[-1]

    listener.apply_result(result(boundary, transcript))

    events = listener.poll_events()
    assert [event.reason for event in events if event.kind == "rejected"] == [expected]
    assert not any(event.kind == "utterance" for event in events)


def test_peek_mid_semantics_use_frame_count_frozen_in_snapshot():
    listener, lane, vad, now = make_listener(overrides={
        "listener.peek_first_ms": 32,
        "listener.peek_every_ms": 32,
        "listener.peek_window_ms": 32,
        "listener.silence_ms": 64,
    })
    feed(listener, vad, now, .9, sample=1)
    first_peek = jobs_for(lane, "peek")[-1]
    listener.apply_result(result(first_peek, "unrelated words"))
    feed(listener, vad, now, .9, sample=2)
    second_peek = jobs_for(lane, "peek")[-1]
    assert second_peek.snapshot.metadata["segment_frame_count"] == 2
    listener.apply_result(result(second_peek, "alexa pause"))
    feed(listener, vad, now, 0.0, sample=3)
    feed(listener, vad, now, 0.0, sample=4)
    final = jobs_for(lane, "final")[-1]
    listener.apply_result(result(final, "alexa pause"))
    utterance = next(event.utterance for event in listener.poll_events()
                     if event.kind == "utterance")
    assert utterance.mid_utterance is True


@pytest.mark.parametrize("overrides", [
    {"listener.max_utterance_s": float("nan")},
    {"listener.extend_ms": -1},
    {"listener.extend_ms": 30000},
    {"listener.peek_window_ms": 40000},
])
def test_oversized_or_invalid_duration_settings_fail_before_capture(overrides):
    with pytest.raises(ValueError):
        make_listener(overrides=overrides)


def test_frame_validation_vad_once_owner_thread_and_timestamp_monotonicity():
    listener, _lane, vad, now = make_listener()
    with pytest.raises(ValueError, match="512 int16"):
        listener.feed_async(np.zeros(256, dtype=np.int16))
    frame = np.zeros(512, dtype=np.int16)
    clean = np.zeros(511, dtype=np.int16)
    with pytest.raises(ValueError, match="processed_frame"):
        listener.feed_async(frame, clean)

    feed(listener, vad, now, 0.0)
    assert len(vad.calls) == 1
    with pytest.raises(ValueError, match="nondecreasing"):
        vad.probabilities.append(0.0)
        listener.feed_async(frame, captured_at=now[0] - 1.0)
    assert len(vad.calls) == 1

    errors = []
    def wrong_owner():
        try:
            listener.poll_events()
        except Exception as exc:
            errors.append(exc)
    thread = __import__("threading").Thread(target=wrong_owner)
    thread.start()
    thread.join(2.0)
    assert isinstance(errors[0], RuntimeError)


def test_suspend_timeout_invalidates_frontend_and_resume_waits_for_quiescence():
    listener, lane, vad, now = make_listener(overrides={
        "listener.peek_first_ms": 32, "listener.peek_window_ms": 32,
    })
    feed(listener, vad, now, .9, sample=1)
    assert jobs_for(lane, "peek")
    lane.inflight = True
    assert listener.suspend(0.0) is False
    assert listener._pending == {}
    assert listener._active is None
    vad_calls = len(vad.calls)
    vad.probabilities.append(0.9)
    with pytest.raises(RuntimeError, match="suspended"):
        listener.feed_async(np.zeros(512, dtype=np.int16))
    assert len(vad.calls) == vad_calls
    assert listener.resume() is False
    lane.inflight = False
    assert listener.resume() is True
    assert listener.stream_epoch == 2
    feed(listener, vad, now, 0.0, sample=2)
    assert len(vad.calls) == 2


def test_close_retries_lane_close_until_quiescent_without_reinvalidating():
    listener, lane, _vad, _now = make_listener()
    lane.inflight = True
    assert listener.close(0.0) is False
    assert listener.close(0.0) is False
    assert lane.epochs == [1]
    lane.inflight = False
    assert listener.close(0.0) is True
    assert lane.epochs == [1]


def test_foreign_same_id_job_does_not_consume_legitimate_pending_result():
    listener, lane, vad, now = make_listener()
    for sample, probability in ((1, .9), (2, .9), (3, 0.0), (4, 0.0)):
        feed(listener, vad, now, probability, sample=sample)
    final = jobs_for(lane, "final")[-1]
    clone = DecodeJob(final.job_id, final.stream_epoch, final.purpose,
                      final.snapshot, final.deadline, final.recognizer)

    listener.apply_result(result(clone, "alexa pause"))
    assert final.job_id in listener._pending
    assert listener.poll_events() == ()

    listener.apply_result(result(final, "alexa pause"))
    assert [event.kind for event in listener.poll_events()] == ["wake", "utterance"]


def test_unaccepted_peek_candidate_does_not_claim_peek_wake_path():
    listener, lane, vad, now = make_listener(overrides={
        "listener.allow_fuzzy": True,
        "listener.fuzzy_cutoff": 0.6,
        "listener.peek_first_ms": 32,
        "listener.peek_every_ms": 32,
        "listener.peek_window_ms": 32,
        "listener.silence_ms": 64,
    })
    feed(listener, vad, now, .9, sample=1)
    peek = jobs_for(lane, "peek")[-1]
    listener.apply_result(result(peek, "alxa pause"))
    assert not any(event.kind == "wake" for event in listener.poll_events())
    feed(listener, vad, now, 0.0, sample=2)
    feed(listener, vad, now, 0.0, sample=3)
    final = jobs_for(lane, "final")[-1]
    listener.apply_result(result(final, "alexa pause"))
    utterance = next(event.utterance for event in listener.poll_events()
                     if event.kind == "utterance")
    assert utterance.wake_path == "final_text"


def test_accepted_peek_wake_keeps_peek_path_for_final_utterance():
    listener, lane, vad, now = make_listener(overrides={
        "listener.peek_first_ms": 32,
        "listener.peek_every_ms": 32,
        "listener.peek_window_ms": 32,
        "listener.silence_ms": 64,
    })
    feed(listener, vad, now, .9, sample=1)
    peek = jobs_for(lane, "peek")[-1]
    listener.apply_result(result(peek, "alexa pause"))
    assert [event.kind for event in listener.poll_events()] == ["wake"]
    feed(listener, vad, now, 0.0, sample=2)
    feed(listener, vad, now, 0.0, sample=3)
    final = jobs_for(lane, "final")[-1]
    listener.apply_result(result(final, "alexa pause"))
    utterance = next(event.utterance for event in listener.poll_events()
                     if event.kind == "utterance")
    assert utterance.wake_path == "peek"


def acoustic_proposal(*, score=0.9, model="alexa", personal_hit=False,
                      personal_score=0.0):
    return AcousticProposal(model, score, personal_hit, personal_score)


def test_acoustic_receipt_uses_current_frame_once_and_never_dispatches_short_command():
    listener, lane, vad, now = make_listener(overrides={
        "wake.confirm_seconds": 0.064,
        "listener.silence_ms": 64,
    })
    feed(listener, vad, now, 0.9, sample=30)
    now[0] += 0.032
    vad.probabilities.append(0.9)
    frame = np.full(512, 31, dtype=np.int16)
    listener.feed_async(frame, captured_at=now[0],
                        acoustic_proposal=acoustic_proposal())
    acoustic = jobs_for(lane, "acoustic_listener")[-1]
    assert acoustic.snapshot.audio.tolist() == [30] * 512 + [31] * 512
    assert len(vad.calls) == 2
    capture_id = listener._capture_id(listener._active)

    listener.apply_result(result(acoustic, "alexa turn off lights"))
    events = listener.poll_events()
    assert [event.kind for event in events] == ["wake"]
    assert events[0].capture_id == capture_id
    assert listener._active.confirmed_audio_handoff is True

    for sample in (32, 33):
        feed(listener, vad, now, 0.0, sample=sample)
    final = jobs_for(lane, "final")[-1]
    assert final.snapshot.audio.tolist() == [30] * 512 + [31] * 512 + [32] * 512 + [33] * 512
    listener.apply_result(result(final, "background music words"))
    utterance = next(event.utterance for event in listener.poll_events()
                     if event.kind == "utterance")
    assert utterance.command == ""
    assert utterance.confirmed_audio_handoff is True
    assert len(vad.calls) == 4


def test_acoustic_confirmation_can_finish_a_sealed_capture_without_borrowing_new_segment():
    listener, lane, vad, now = make_listener(overrides={
        "wake.confirm_seconds": 0.064,
        "listener.silence_ms": 32,
    })
    feed(listener, vad, now, 0.9, sample=40,)
    now[0] += 0.032
    vad.probabilities.append(0.0)
    listener.feed_async(np.full(512, 41, np.int16), captured_at=now[0],
                        acoustic_proposal=acoustic_proposal())
    acoustic = jobs_for(lane, "acoustic_listener")[-1]
    final = jobs_for(lane, "final")[-1]
    old_capture = listener._capture_id(final and listener._pending[final.job_id].segment)
    feed(listener, vad, now, 0.9, sample=50,
         acoustic_proposal=acoustic_proposal())
    new_capture = listener._capture_id(listener._active)
    assert any(event.reason == "acoustic_proposal_busy"
               for event in listener.poll_events())
    assert new_capture != old_capture

    listener.apply_result(result(final, "background words"))
    assert listener.poll_events() == ()  # terminal outcome waits for this proposal
    listener.apply_result(result(acoustic, "alexa"))
    events = listener.poll_events()
    assert [event.kind for event in events] == ["wake", "utterance"]
    assert {event.capture_id for event in events} == {old_capture}
    assert next(event.utterance for event in events
                if event.kind == "utterance").command == ""
    assert listener._active is not None
    assert listener._capture_id(listener._active) == new_capture


def test_acoustic_verification_reuses_snapshot_and_absolute_deadline():
    verified = []

    def verify_text(transcript, phrase, mid):
        verified.append((transcript, phrase, mid))
        return phrase, "dangerous short command", False

    listener, lane, vad, now = make_listener(verify_text=verify_text, overrides={
        "wake.confirm_seconds": 0.064,
    })
    feed(listener, vad, now, 0.9, sample=60,)
    acoustic = jobs_for(lane, "acoustic_listener")[-1] if jobs_for(lane, "acoustic_listener") else None
    if acoustic is None:
        # Proposal must be admitted on an actual supplied frame.
        now[0] += 0.032
        vad.probabilities.append(0.9)
        listener.feed_async(np.full(512, 61, np.int16), captured_at=now[0],
                            acoustic_proposal=acoustic_proposal())
        acoustic = jobs_for(lane, "acoustic_listener")[-1]
    deadline = acoustic.deadline
    listener.apply_result(result(acoustic, "background words"))
    verify = jobs_for(lane, "acoustic_verify")[-1]
    assert verify.deadline == deadline
    assert verify.snapshot is acoustic.snapshot
    listener.apply_result(result(verify, "alexa dangerous short command"))
    assert verified == [("alexa dangerous short command", "alexa", False)]
    events = listener.poll_events()
    assert [event.kind for event in events] == ["wake"]
    assert not any(event.kind == "utterance" for event in events)


def test_acoustic_expiry_makes_late_verification_inert_and_queue_busy_is_reported():
    verified = []

    def verify_text(transcript, phrase, mid):
        verified.append((transcript, phrase, mid))
        return phrase, "", False

    listener, lane, vad, now = make_listener(verify_text=verify_text)
    now[0] += 0.032
    vad.probabilities.append(0.9)
    listener.feed_async(np.full(512, 70, np.int16), captured_at=now[0],
                        acoustic_proposal=acoustic_proposal())
    acoustic = jobs_for(lane, "acoustic_listener")[-1]
    listener.apply_result(result(acoustic, "not a wake"))
    verify = jobs_for(lane, "acoustic_verify")[-1]
    now[0] = verify.deadline
    listener.apply_result(result(verify, "alexa", finished_at=verify.deadline))
    assert not any(event.kind == "wake" for event in listener.poll_events())
    assert verified == []

    busy, busy_lane, busy_vad, busy_now = make_listener(lane=ManualLane())
    busy_lane.accept = False
    busy_now[0] += 0.032
    busy_vad.probabilities.append(0.0)
    busy.feed_async(np.zeros(512, np.int16), captured_at=busy_now[0],
                    acoustic_proposal=acoustic_proposal())
    assert busy._reservation is None
    event = next(event for event in busy.poll_events()
                 if event.reason == "acoustic_queue_busy")
    assert event.diagnostic_snapshot is not None
    assert event.diagnostic_snapshot.metadata["trigger_transcript"] == ""
    assert event.diagnostic_snapshot.metadata["trigger_stage_state"] == "unrun"
    assert "trigger_stt_latency_ms" not in event.diagnostic_snapshot.metadata
    assert event.trigger_transcript is None
    assert event.trigger_stt_latency_ms is None


def test_terminal_result_is_held_until_acoustic_refusal_then_released():
    listener, lane, vad, now = make_listener()
    now[0] += 0.032
    vad.probabilities.append(0.9)
    listener.feed_async(np.full(512, 80, np.int16), captured_at=now[0],
                        acoustic_proposal=acoustic_proposal())
    acoustic = jobs_for(lane, "acoustic_listener")[-1]
    feed(listener, vad, now, 0.0, sample=81)
    feed(listener, vad, now, 0.0, sample=82)
    final = jobs_for(lane, "final")[-1]
    listener.apply_result(result(final, "ordinary conversation"))
    assert listener.poll_events() == ()
    listener.apply_result(result(acoustic, "ordinary conversation"))
    events = listener.poll_events()
    assert [event.kind for event in events if event.kind == "rejected"] == ["rejected"]
    assert any(event.kind == "diagnostic" and event.diagnostic_snapshot is not None
               for event in events)
    rejection = next(event for event in events if event.kind == "rejected")
    assert rejection.diagnostic_snapshot.metadata["diagnostic_provenance"] == "unknown"


def test_acoustic_deadline_releases_held_terminal_and_late_receipt_is_inert():
    listener, lane, vad, now = make_listener()
    now[0] += 0.032
    vad.probabilities.append(0.9)
    listener.feed_async(np.full(512, 85, np.int16), captured_at=now[0],
                        acoustic_proposal=acoustic_proposal())
    acoustic = jobs_for(lane, "acoustic_listener")[-1]
    feed(listener, vad, now, 0.0, sample=86)
    feed(listener, vad, now, 0.0, sample=87)
    final = jobs_for(lane, "final")[-1]
    listener.apply_result(result(final, "ordinary talk"))
    assert listener.poll_events() == ()
    now[0] = acoustic.deadline
    events = listener.poll_events()
    assert not any(event.kind == "wake" for event in events)
    assert any(event.kind == "rejected" for event in events)
    listener.apply_result(result(acoustic, "alexa"))
    assert listener.poll_events() == ()


def test_acoustic_result_requires_exact_job_and_lifecycle_epoch():
    listener, lane, vad, now = make_listener()
    now[0] += 0.032
    vad.probabilities.append(0.9)
    listener.feed_async(np.full(512, 88, np.int16), captured_at=now[0],
                        acoustic_proposal=acoustic_proposal())
    acoustic = jobs_for(lane, "acoustic_listener")[-1]
    clone = DecodeJob(acoustic.job_id, acoustic.stream_epoch, acoustic.purpose,
                      acoustic.snapshot, acoustic.deadline, acoustic.recognizer)
    listener.apply_result(result(clone, "alexa"))
    assert acoustic.job_id in listener._acoustic_pending
    assert listener.poll_events() == ()
    listener.apply_result(result(acoustic, "alexa"))
    assert [event.kind for event in listener.poll_events()] == ["wake"]
    listener.apply_result(result(acoustic, "alexa"))
    assert listener.poll_events() == ()

    stale, stale_lane, stale_vad, stale_now = make_listener()
    stale_now[0] += 0.032
    stale_vad.probabilities.append(0.9)
    stale.feed_async(np.full(512, 89, np.int16), captured_at=stale_now[0],
                     acoustic_proposal=acoustic_proposal())
    late = jobs_for(stale_lane, "acoustic_listener")[-1]
    stale.flush("test acoustic reset")
    stale.apply_result(result(late, "alexa"))
    assert stale.poll_events() == ()


def test_valid_full_text_resolves_acoustic_reservation_and_late_receipt_is_inert():
    listener, lane, vad, now = make_listener()
    now[0] += 0.032
    vad.probabilities.append(0.9)
    listener.feed_async(np.full(512, 95, np.int16), captured_at=now[0],
                        acoustic_proposal=acoustic_proposal())
    acoustic = jobs_for(lane, "acoustic_listener")[-1]
    feed(listener, vad, now, 0.0, sample=96)
    feed(listener, vad, now, 0.0, sample=97)
    final = jobs_for(lane, "final")[-1]
    listener.apply_result(result(final, "alexa play song"))
    events = listener.poll_events()
    actions = [event for event in events if event.kind in ("wake", "utterance")]
    assert [event.kind for event in actions] == ["wake", "utterance"]
    assert actions[-1].utterance.command == "play song"
    listener.apply_result(result(acoustic, "alexa stop"))
    assert listener.poll_events() == ()


def test_weak_candidate_is_unknown_bounded_diagnostic_and_drops_are_counted():
    listener, _lane, vad, now = make_listener(overrides={
        "telemetry.wake_candidates_enabled": True,
        "listener.silence_ms": 32,
    })
    weak_frames = [(sample, 0.1) for sample in range(90, 93)]
    weak_frames.extend((sample, 0.0) for sample in range(93, 104))
    for sample, probability in weak_frames:
        feed(listener, vad, now, probability, sample=sample)
    weak = next(event for event in listener.poll_events()
                if event.reason == "below_primary_vad")
    assert weak.diagnostic_stage == "weak_candidate"
    assert weak.diagnostic_snapshot.metadata["diagnostic_provenance"] == "unknown"
    assert len(weak.diagnostic_snapshot.audio) <= 187 * 512

    for _ in range(20):
        listener._diagnostic_event("overflow", None, stage="test")
    assert listener.diagnostic_drops > 0


def test_acoustic_proposal_validation_rejects_bool_nonfinite_and_bad_hit_types():
    for kwargs in (
        {"stock_score": True},
        {"stock_score": float("nan")},
        {"personal_score": float("inf")},
        {"personal_hit": 1},
        {"stock_model": "x" * 129},
    ):
        values = {"stock_model": "alexa", "stock_score": 0.9,
                  "personal_hit": False, "personal_score": 0.0}
        values.update(kwargs)
        with pytest.raises(ValueError):
            AcousticProposal(**values)


def test_same_capture_proposals_coalesce_without_moving_absolute_deadline():
    listener, lane, vad, now = make_listener()
    feed(listener, vad, now, 0.9, sample=100)
    feed(listener, vad, now, 0.9, sample=101,
         acoustic_proposal=acoustic_proposal())
    reservation = listener._reservation
    deadline = reservation.deadline
    feed(listener, vad, now, 0.9, sample=102,
         acoustic_proposal=acoustic_proposal(score=0.99))
    assert listener._reservation is reservation
    assert listener._reservation.deadline == deadline
    assert sum(job.purpose == "acoustic_listener" for job in lane.jobs) == 1
    assert any(event.reason == "acoustic_proposal_coalesced"
               for event in listener._events)


def test_weak_acoustic_proposal_is_retained_only_as_unknown_diagnostic():
    listener, lane, vad, now = make_listener(overrides={
        "wake.acoustic_activation_threshold": 0.75,
        "wake.personal_threshold": 0.5,
    })
    now[0] += 0.032
    vad.probabilities.append(0.1)
    listener.feed_async(np.full(512, 110, np.int16), captured_at=now[0],
                        acoustic_proposal=acoustic_proposal(score=0.4))
    events = listener.poll_events()
    weak = next(event for event in events if event.reason == "acoustic_weak_proposal")
    assert weak.diagnostic_snapshot.metadata["diagnostic_provenance"] == "unknown"
    assert weak.diagnostic_snapshot.metadata["acoustic_proposal"]["stock_score"] == 0.4
    assert not jobs_for(lane, "acoustic_listener")
    assert not any(event.kind == "wake" for event in events)


def boundary_with_late_proposal(*, overrides=None, verify_text=None):
    values = {
        "wake.confirm_seconds": 0.032,
        "listener.max_utterance_s": 0.064,
        "listener.silence_ms": 64,
        "listener.pad_ms": 0,
        "listener.extend_ms": 128,
        "listener.overlap_ms": 0,
    }
    values.update(overrides or {})
    listener, lane, vad, now = make_listener(overrides=values,
                                              verify_text=verify_text)
    feed(listener, vad, now, 0.9, sample=120)
    feed(listener, vad, now, 0.9, sample=121)
    boundary_job = jobs_for(lane, "boundary")[-1]
    boundary = listener._boundary
    feed(listener, vad, now, 0.9, sample=122,
         acoustic_proposal=acoustic_proposal())
    acoustic_job = jobs_for(lane, "acoustic_listener")[-1]
    return listener, lane, vad, now, boundary_job, acoustic_job, boundary


def test_late_boundary_proposal_defers_result_then_final_contains_receipt_once():
    listener, lane, vad, now, boundary_job, acoustic_job, boundary = \
        boundary_with_late_proposal()
    feed(listener, vad, now, 0.0, sample=123)
    feed(listener, vad, now, 0.0, sample=124)
    calls_before_results = len(vad.calls)
    listener.apply_result(result(boundary_job, "ordinary music"))
    assert listener._boundary is boundary
    assert listener._active is None
    assert not jobs_for(lane, "final")
    assert listener.poll_events() == ()

    listener.apply_result(result(acoustic_job, "alexa"))
    wake_events = [event for event in listener.poll_events() if event.kind == "wake"]
    assert len(wake_events) == 1
    final = jobs_for(lane, "final")[-1]
    expected = np.concatenate([np.full(512, sample, np.int16)
                               for sample in (120, 121, 122, 123, 124)])
    np.testing.assert_array_equal(final.snapshot.audio, expected)
    assert final.snapshot.segment_id == boundary.segment.segment_id
    receipt = final.snapshot.metadata["acoustic_receipt"]
    assert receipt["epoch"] == final.stream_epoch
    assert receipt["capture_id"] == final.snapshot.segment_id
    assert receipt["phrase"] == "alexa"
    assert final.snapshot.start_sample <= receipt["start_sample"]
    assert final.snapshot.end_sample >= receipt["end_sample"]
    assert len(vad.calls) == calls_before_results

    listener.apply_result(result(final, "alexa pause"))
    actions = [event for event in listener.poll_events()
               if event.kind in ("wake", "utterance")]
    assert [event.kind for event in actions] == ["utterance"]
    assert actions[0].utterance.command == "pause"
    assert actions[0].capture_id == wake_events[0].capture_id
    assert actions[0].utterance.audio.tolist() == expected.tolist()


def test_late_boundary_proposal_failure_releases_original_rejection_once():
    listener, lane, _vad, _now, boundary_job, acoustic_job, boundary = \
        boundary_with_late_proposal()
    listener.apply_result(result(boundary_job, "ordinary words"))
    assert listener._boundary is boundary
    assert listener._active is None
    listener.apply_result(result(acoustic_job, "ordinary words"))
    events = listener.poll_events()
    rejected = [event for event in events if event.kind == "rejected"]
    assert len(rejected) == 1
    assert rejected[0].reason == "speech_no_trigger"
    assert not [event for event in events if event.kind == "wake"]
    listener.apply_result(result(acoustic_job, "alexa"))
    assert not [event for event in listener.poll_events() if event.kind == "wake"]


def test_late_boundary_proposal_cannot_extend_original_boundary_deadline():
    listener, _lane, _vad, now, boundary_job, acoustic_job, boundary = \
        boundary_with_late_proposal()
    listener.apply_result(result(boundary_job, "ordinary words"))
    assert listener._boundary is boundary
    now[0] = boundary.deadline
    events = listener.poll_events()
    assert listener.stream_epoch == 1
    assert any(event.kind == "rejected"
               and event.reason == "boundary_deadline_exceeded" for event in events)
    listener.apply_result(result(acoustic_job, "alexa"))
    assert not [event for event in listener.poll_events() if event.kind == "wake"]


def test_late_boundary_receipt_beyond_extension_cap_is_explicit_cutoff():
    listener, lane, vad, now = make_listener(overrides={
        "wake.confirm_seconds": 0.032,
        "listener.max_utterance_s": 0.064,
        "listener.silence_ms": 64,
        "listener.pad_ms": 64,
        "listener.extend_ms": 64,
        "listener.overlap_ms": 0,
    })
    feed(listener, vad, now, 0.9, sample=120)
    feed(listener, vad, now, 0.9, sample=121)
    boundary_job = jobs_for(lane, "boundary")[-1]
    boundary = listener._boundary
    feed(listener, vad, now, 0.9, sample=122)
    feed(listener, vad, now, 0.9, sample=123)
    feed(listener, vad, now, 0.9, sample=124,
         acoustic_proposal=acoustic_proposal())
    acoustic_job = jobs_for(lane, "acoustic_listener")[-1]
    feed(listener, vad, now, 0.9, sample=125)
    listener.apply_result(result(boundary_job, "ordinary words"))
    assert listener._boundary is boundary
    listener.apply_result(result(acoustic_job, "alexa"))
    events = listener.poll_events()
    assert any(event.kind == "wake" for event in events)
    assert any(event.kind == "rejected"
               and event.reason == "acoustic_extension_cutoff" for event in events)
    assert not jobs_for(lane, "final")
    listener.apply_result(result(acoustic_job, "alexa"))
    assert listener.poll_events() == ()


def test_maximum_authorization_snapshot_keeps_decorated_metadata_bounded():
    listener, lane, vad, now = make_listener()
    auth = {"grant_{}".format(index): True for index in range(64)}
    now[0] += 0.032
    vad.probabilities.append(0.9)
    listener.feed_async(np.full(512, 130, np.int16), captured_at=now[0],
                        authorization=auth,
                        acoustic_proposal=acoustic_proposal(personal_hit=True,
                                                            personal_score=0.7))
    acoustic = jobs_for(lane, "acoustic_listener")[-1]
    listener.apply_result(result(acoustic, "", error="recognizer_failed"))
    event = next(event for event in listener.poll_events()
                 if event.reason == "recognizer_failed")
    metadata = event.diagnostic_snapshot.metadata
    assert len(metadata["authorization"]) == 64
    assert metadata["scores"]["alexa_user"] == 0.7
    assert len(metadata["scores"]) == 2
    assert metadata["trigger_stage_state"] == "failed"
    assert metadata["second_stt_stage_state"] == "unrun"
    assert "trigger_stt_latency_ms" not in metadata
    assert event.trigger_transcript is None
    assert event.trigger_stt_latency_ms is None


def test_authorization_over_64_is_rejected_before_vad_admission():
    listener, _lane, vad, now = make_listener()
    auth = {"grant_{}".format(index): True for index in range(65)}
    frame = np.zeros(512, np.int16)
    with pytest.raises(ValueError, match="bounded"):
        listener.feed_async(frame, captured_at=now[0] + 0.032,
                            authorization=auth,
                            acoustic_proposal=acoustic_proposal())
    assert vad.calls == []


def test_acoustic_diagnostic_keeps_both_completed_stage_transcripts_and_latencies():
    def refuse(_transcript, _phrase, _mid):
        return "", "", False

    listener, lane, vad, now = make_listener(verify_text=refuse)
    now[0] += 0.032
    vad.probabilities.append(0.9)
    listener.feed_async(np.full(512, 131, np.int16), captured_at=now[0],
                        acoustic_proposal=acoustic_proposal(score=0.91))
    first = jobs_for(lane, "acoustic_listener")[-1]
    listener.apply_result(result(first, "background phrase", latency=11))
    second = jobs_for(lane, "acoustic_verify")[-1]
    listener.apply_result(result(second, "second stage words", latency=17))
    event = next(event for event in listener.poll_events()
                 if event.reason == "acoustic_verification_refused")
    metadata = event.diagnostic_snapshot.metadata
    assert metadata["trigger_transcript"] == "background phrase"
    assert metadata["second_stt_transcript"] == "second stage words"
    assert metadata["trigger_stt_latency_ms"] == 11
    assert metadata["second_stt_latency_ms"] == 17
    assert metadata["trigger_stage_state"] == "completed"
    assert metadata["second_stt_stage_state"] == "completed"
    assert metadata["scores"] == {"alexa": 0.91}


def test_listener_stage_accepts_registered_phrase_independent_of_stock_fallback_name():
    listener, lane, vad, now = make_listener()
    now[0] += 0.032
    vad.probabilities.append(0.9)
    listener.feed_async(np.full(512, 140, np.int16), captured_at=now[0],
                        acoustic_proposal=acoustic_proposal(
                            model="hey_mycroft", score=0.9))
    first = jobs_for(lane, "acoustic_listener")[-1]
    assert listener._reservation.fallback_target_registered is False
    listener.apply_result(result(first, "alexa turn on the lamp"))
    events = listener.poll_events()
    wake = next(event for event in events if event.kind == "wake")
    assert wake.phrase == "alexa"
    assert not jobs_for(lane, "acoustic_verify")


def test_unregistered_fallback_target_cannot_confirm_unregistered_listener_text():
    listener, lane, vad, now = make_listener()
    now[0] += 0.032
    vad.probabilities.append(0.9)
    listener.feed_async(np.full(512, 141, np.int16), captured_at=now[0],
                        acoustic_proposal=acoustic_proposal(
                            model="hey_mycroft", score=0.9))
    first = jobs_for(lane, "acoustic_listener")[-1]
    listener.apply_result(result(first, "computer play music"))
    events = listener.poll_events()
    assert not [event for event in events if event.kind == "wake"]
    diagnostic = next(event for event in events
                      if event.reason == "acoustic_fallback_target_unregistered")
    assert diagnostic.diagnostic_snapshot.metadata["trigger_transcript"] == \
        "computer play music"
    assert diagnostic.diagnostic_snapshot.metadata["proposal_qualifiers"][
        "fallback_target_registered"] is False
    assert not jobs_for(lane, "acoustic_verify")


@pytest.mark.parametrize(("error", "expired", "cancelled", "reason"), [
    ("full_decode_failed", False, False, "full_decode_failed"),
    ("", True, False, "decode_expired"),
])
def test_failed_full_decode_held_for_acoustic_receipt_never_publishes_utterance(
        error, expired, cancelled, reason):
    listener, lane, vad, now = make_listener(overrides={
        "listener.silence_ms": 64,
    })
    now[0] += 0.032
    vad.probabilities.append(0.9)
    listener.feed_async(np.full(512, 150, np.int16), captured_at=now[0],
                        acoustic_proposal=acoustic_proposal())
    acoustic = jobs_for(lane, "acoustic_listener")[-1]
    feed(listener, vad, now, 0.0, sample=151)
    feed(listener, vad, now, 0.0, sample=152)
    final = jobs_for(lane, "final")[-1]
    failed_full = DecodeResult(
        final, transcript="alexa delete all files", latency_ms=23,
        error_category=error,
        finished_at=final.snapshot.completed_at + 0.01,
        expired=expired, cancelled=cancelled)
    listener.apply_result(failed_full)
    assert listener.poll_events() == ()

    listener.apply_result(result(acoustic, "alexa"))
    events = listener.poll_events()
    wake = next(event for event in events if event.kind == "wake")
    rejections = [event for event in events if event.kind == "rejected"]
    assert len(rejections) == 1
    assert rejections[0].reason == reason
    assert rejections[0].capture_id == wake.capture_id
    assert not [event for event in events if event.kind == "utterance"]
