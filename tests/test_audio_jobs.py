"""Deterministic tests for bounded, serialized offline-ASR job scheduling."""
from threading import Event, Lock
import time

import numpy as np
import pytest

from assistant.audio_jobs import AudioSnapshot, DecodeJob, SerializedAsrLane


def snapshot(value=1, *, epoch=0, start=0, end=8, metadata=None):
    return AudioSnapshot(
        stream_epoch=epoch, segment_id="segment-{}-{}".format(epoch, value), revision=0,
        start_sample=start, end_sample=end, captured_at=1.0, completed_at=2.0,
        audio=np.full(end - start, value, dtype=np.int16),
        processed_audio=np.full(end - start, value + 1, dtype=np.int16),
        end_reason="silence", metadata=metadata or {})


def job(job_id, purpose="final", *, epoch=0, deadline=None, value=1,
        recognizer="main"):
    audio = snapshot(value, epoch=epoch)
    if deadline is None:
        deadline = time.monotonic() + 30.0
    return DecodeJob(job_id, epoch, purpose, audio, deadline, recognizer)


def wait_for_results(lane, count, timeout=1.0):
    deadline = time.monotonic() + timeout
    results = []
    while time.monotonic() < deadline and len(results) < count:
        results.extend(lane.poll())
        if len(results) < count:
            Event().wait(0.005)
    return results


def test_snapshot_owns_pcm_and_deep_freezes_metadata():
    source = np.arange(8, dtype=np.int16)
    processed = source + 10
    metadata = {"labels": ["wake", {"score": 0.9}]}
    snap = AudioSnapshot(0, "seg", 2, 20, 28, 1.0, 2.0,
                         source, processed, metadata=metadata)
    source[:] = -1
    processed[:] = -2
    metadata["labels"].append("changed")
    assert np.array_equal(snap.audio, np.arange(8, dtype=np.int16))
    assert np.array_equal(snap.processed_audio, np.arange(8, dtype=np.int16) + 10)
    assert snap.metadata["labels"] == ("wake", {"score": 0.9})
    with pytest.raises(ValueError):
        snap.audio.setflags(write=True)
    with pytest.raises(TypeError):
        snap.metadata["new"] = True
    with pytest.raises(TypeError):
        snap.metadata["labels"][1]["score"] = 0.0


@pytest.mark.parametrize("changes", [
    {"end_sample": 7},
    {"processed_audio": np.zeros(7, dtype=np.int16)},
    {"captured_at": float("nan")},
    {"completed_at": 0.0},
    {"metadata": {"object": object()}},
])
def test_snapshot_rejects_inconsistent_or_unsafe_fields(changes):
    values = dict(stream_epoch=0, segment_id="seg", revision=0,
                  start_sample=0, end_sample=8, captured_at=1.0, completed_at=2.0,
                  audio=np.zeros(8, dtype=np.int16))
    values.update(changes)
    with pytest.raises(ValueError):
        AudioSnapshot(**values)


def test_snapshot_rejects_pcm_above_thirty_seconds_before_copy():
    with pytest.raises(ValueError, match="samples"):
        snapshot(end=480001)


def test_snapshot_rejects_cyclic_and_too_deep_metadata():
    cyclic = {}
    cyclic["self"] = cyclic
    too_deep = {"nested": [[[[[[[[[["too deep"]]]]]]]]]]}
    for metadata in (cyclic, too_deep):
        with pytest.raises(ValueError, match="depth"):
            snapshot(metadata=metadata)


@pytest.mark.parametrize("kwargs", [
    {"max_priority_pending": 9},
    {"max_results": 33},
])
def test_lane_rejects_unbounded_capacity_configuration(kwargs):
    with pytest.raises(ValueError):
        SerializedAsrLane({"main": lambda audio: ("", 0)}, **kwargs)


def test_decode_job_rejects_invalid_purpose_epoch_and_snapshot():
    snap = snapshot()
    with pytest.raises(ValueError, match="purpose"):
        DecodeJob("bad-purpose", 0, [], snap, 10.0, "main")
    with pytest.raises(ValueError, match="stream_epoch"):
        DecodeJob("bad-epoch", -1, "final", snap, 10.0, "main")
    with pytest.raises(ValueError, match="must match"):
        DecodeJob("wrong-snapshot", 1, "final", snap, 10.0, "main")


@pytest.mark.parametrize("purpose", ["acoustic_listener", "acoustic_verify"])
def test_acoustic_purpose_validation_does_not_run_recognizer(purpose):
    called = []
    lane = SerializedAsrLane({"main": lambda audio: called.append(audio)})
    acoustic_job = job("validated-" + purpose, purpose)
    assert acoustic_job.purpose == purpose
    assert called == []
    assert lane.close(0.0)


@pytest.mark.parametrize("purpose", ["acoustic_listener", "acoustic_verify"])
def test_acoustic_purposes_are_admitted_as_urgent_ahead_of_pending_peek(purpose):
    entered, release = Event(), Event()
    calls = []

    def decode(audio):
        value = int(audio[0])
        calls.append(value)
        if value == 1:
            entered.set()
            release.wait(1.0)
        return "ok", 1

    lane = SerializedAsrLane({"main": decode})
    assert lane.start()
    try:
        assert lane.submit(job("running", "peek", value=1))
        assert entered.wait(1.0)
        assert lane.submit(job("pending-peek", "peek", value=2))
        assert lane.submit(job("acoustic", purpose, value=3))
        release.set()
        results = wait_for_results(lane, 3)
        assert calls == [1, 3, 2]
        assert {r.job.job_id for r in results} == {"running", "pending-peek", "acoustic"}
    finally:
        release.set()
        assert lane.close(1.0)


def test_urgent_fifo_precedes_latest_pending_peek_and_never_overlaps_calls():
    entered, release, all_done = Event(), Event(), Event()
    calls = []
    active = [0]
    maximum_active = [0]
    guard = Lock()

    def decode(audio):
        value = int(audio[0])
        with guard:
            active[0] += 1
            maximum_active[0] = max(maximum_active[0], active[0])
            calls.append(value)
        if value == 1:
            entered.set()
            release.wait(1.0)
        with guard:
            active[0] -= 1
        if len(calls) >= 4:
            all_done.set()
        return str(value), 2

    lane = SerializedAsrLane({"main": decode})
    assert lane._thread is None
    assert lane.start()
    try:
        assert lane.submit(job("running", "peek", value=1))
        assert entered.wait(1.0)
        assert lane.submit(job("old-peek", "peek", value=2))
        assert lane.submit(job("latest-peek", "peek", value=3))
        assert lane.submit(job("urgent", "final", value=4))
        assert lane.submit(job("urgent-next", "boundary", value=5))
        assert not lane.submit(job("urgent-overflow", "verify", value=6))
        release.set()
        assert all_done.wait(1.0)
        results = wait_for_results(lane, 5)
        assert calls == [1, 4, 5, 3]
        assert maximum_active[0] == 1
        assert [(r.job.job_id, r.error_category, r.cancelled) for r in results
                if r.job.job_id == "old-peek"] == [("old-peek", "superseded", True)]
        assert {r.job.job_id for r in results} == {
            "running", "old-peek", "urgent", "urgent-next", "latest-peek"}
    finally:
        release.set()
        assert lane.close(1.0)


@pytest.mark.parametrize("purpose", ["acoustic_listener", "acoustic_verify"])
def test_acoustic_job_expired_before_dispatch_skips_recognizer(purpose):
    entered, release = Event(), Event()
    now = [0.0]
    calls = []

    def decode(audio):
        value = int(audio[0])
        calls.append(value)
        if value == 1:
            entered.set()
            release.wait(1.0)
        return "ok", 1

    lane = SerializedAsrLane({"main": decode}, clock=lambda: now[0])
    assert lane.start()
    try:
        assert lane.submit(job("blocking", deadline=10, value=1))
        assert entered.wait(1.0)
        assert lane.submit(job("expires", purpose, deadline=1, value=2))
        now[0] = 2.0
        release.set()
        results = wait_for_results(lane, 2)
        expired = next(r for r in results if r.job.job_id == "expires")
        assert expired.expired and expired.error_category == "deadline_exceeded"
        assert expired.transcript == "" and calls == [1]
    finally:
        release.set()
        assert lane.close(1.0)


def test_completion_after_deadline_discards_transcript_but_reports_latency():
    entered, release = Event(), Event()
    now = [0.0]

    def decode(_audio):
        entered.set()
        release.wait(1.0)
        return "late transcript", 17

    lane = SerializedAsrLane({"main": decode}, clock=lambda: now[0])
    assert lane.start()
    try:
        assert lane.submit(job("late", deadline=1.0))
        assert entered.wait(1.0)
        now[0] = 2.0
        release.set()
        result = wait_for_results(lane, 1)[0]
        assert result.expired and result.error_category == "deadline_exceeded"
        assert result.transcript == "" and result.latency_ms == 17
    finally:
        release.set()
        assert lane.close(1.0)


@pytest.mark.parametrize("outcome", ["exception", "malformed"])
def test_late_exception_or_malformed_result_is_still_expired(outcome):
    entered, release = Event(), Event()
    now = [0.0]

    def decode(_audio):
        entered.set()
        release.wait(1.0)
        now[0] = 2.0
        if outcome == "exception":
            raise RuntimeError("private failure")
        return "malformed", 1.5

    lane = SerializedAsrLane({"main": decode}, clock=lambda: now[0])
    assert lane.start()
    try:
        assert lane.submit(job("late-invalid", deadline=1.0))
        assert entered.wait(1.0)
        release.set()
        result = wait_for_results(lane, 1)[0]
        assert result.error_category == "deadline_exceeded"
        assert result.expired and result.transcript == ""
    finally:
        release.set()
        assert lane.close(1.0)


@pytest.mark.parametrize("purpose", ["acoustic_listener", "acoustic_verify"])
def test_invalidate_suppresses_late_old_epoch_and_accepts_current_epoch(purpose):
    entered, release, current_done = Event(), Event(), Event()
    calls = []

    def decode(audio):
        value = int(audio[0])
        calls.append(value)
        if value == 1:
            entered.set()
            release.wait(1.0)
        else:
            current_done.set()
        return "decoded-{}".format(value), 3

    lane = SerializedAsrLane({"main": decode})
    assert lane.start()
    try:
        assert lane.submit(job("old", purpose, value=1))
        assert entered.wait(1.0)
        lane.invalidate(1)
        assert lane.submit(job("current", epoch=1, value=2))
        release.set()
        assert current_done.wait(1.0)
        results = wait_for_results(lane, 1)
        assert [r.job.job_id for r in results] == ["current"]
        assert calls == [1, 2]
    finally:
        release.set()
        assert lane.close(1.0)


def test_resume_rejects_epoch_rollback_after_invalidation():
    lane = SerializedAsrLane({"main": lambda audio: ("ok", 1)})
    lane.invalidate(5)
    with pytest.raises(ValueError, match="backwards"):
        lane.resume(4)
    assert lane.resume(5)
    assert lane.resume(6)
    assert lane.close(0.0)


def test_result_reservations_refuse_flood_without_losing_accepted_jobs():
    entered, release = Event(), Event()

    def decode(_audio):
        entered.set()
        release.wait(1.0)
        return "ok", 1

    lane = SerializedAsrLane({"main": decode}, max_results=2)
    assert lane.start()
    try:
        assert lane.submit(job("running", value=1))
        assert entered.wait(1.0)
        assert lane.submit(job("pending-peek", "peek", value=2))
        assert not lane.submit(job("unreserved", "final", value=3))
        assert not lane.submit(job("replacement", "peek", value=4))
        release.set()
        results = wait_for_results(lane, 2)
        assert {result.job.job_id for result in results} == {"running", "pending-peek"}
        assert len(results) == 2
    finally:
        release.set()
        assert lane.close(1.0)


@pytest.mark.parametrize("purpose", ["acoustic_listener", "acoustic_verify"])
def test_suspend_timeout_blocks_submission_then_resumes_after_inflight_finishes(purpose):
    entered, release = Event(), Event()

    def decode(_audio):
        entered.set()
        release.wait(1.0)
        return "done", 1

    lane = SerializedAsrLane({"main": decode})
    assert lane.start()
    try:
        assert lane.submit(job("busy"))
        assert entered.wait(1.0)
        assert lane.submit(job("queued", purpose, value=2))
        assert not lane.suspend(0.01)
        assert not lane.submit(job("during-suspend", epoch=1))
        assert not lane.resume(1)
        assert any(r.job.job_id == "queued" and r.cancelled
                   and r.error_category == "suspended" for r in lane.poll())
        release.set()
        assert wait_for_results(lane, 1)[0].job.job_id == "busy"
        assert lane.resume(1)
        assert lane.poll() == ()
        assert lane.submit(job("after-resume", epoch=1))
        assert wait_for_results(lane, 1)[0].job.job_id == "after-resume"
    finally:
        release.set()
        assert lane.close(1.0)


def test_close_reports_uncancellable_work_until_worker_exits():
    entered, release = Event(), Event()

    def decode(_audio):
        entered.set()
        release.wait(1.0)
        return "done", 1

    lane = SerializedAsrLane({"main": decode})
    assert lane.start()
    assert lane.submit(job("busy"))
    assert entered.wait(1.0)
    assert not lane.close(0.01)
    assert lane._thread is not None and lane._thread.is_alive()
    release.set()
    assert lane.close(1.0)
    assert lane._thread is not None and not lane._thread.is_alive()
    assert not lane.submit(job("after-close"))


def test_recognizer_failure_is_sanitized_and_worker_continues():
    calls = []

    def broken(_audio):
        calls.append("failed")
        raise RuntimeError("private model exception text")

    def healthy(_audio):
        calls.append("healthy")
        return "ok", 1

    lane = SerializedAsrLane({"broken": broken, "healthy": healthy})
    assert lane.start()
    try:
        assert lane.submit(job("bad", recognizer="broken"))
        assert lane.submit(job("good", recognizer="healthy", value=2))
        results = wait_for_results(lane, 2)
        failed = next(r for r in results if r.job.job_id == "bad")
        assert failed.error_category == "decode_error" and failed.transcript == ""
        assert "private" not in failed.error_category
        assert next(r for r in results if r.job.job_id == "good").transcript == "ok"
        assert calls == ["failed", "healthy"]
    finally:
        assert lane.close(1.0)


def test_malformed_recognizer_result_is_terminal_and_worker_continues():
    def malformed(_audio):
        return "text", 1.5

    def healthy(_audio):
        return "ok", 1

    lane = SerializedAsrLane({"bad": malformed, "main": healthy})
    assert lane.start()
    try:
        assert lane.submit(job("bad-result", recognizer="bad"))
        assert lane.submit(job("good-result", recognizer="main", value=2))
        results = wait_for_results(lane, 2)
        bad = next(r for r in results if r.job.job_id == "bad-result")
        assert bad.error_category == "invalid_result" and bad.transcript == ""
        assert next(r for r in results if r.job.job_id == "good-result").transcript == "ok"
    finally:
        assert lane.close(1.0)
