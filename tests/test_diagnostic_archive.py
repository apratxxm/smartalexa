import threading

import numpy as np

from assistant.audio_jobs import AudioSnapshot
from assistant.diagnostic_archive import QueuedDiagnosticArchive


def snapshot(segment="seg", *, metadata=None, processed=False):
    return AudioSnapshot(7, segment, 1, 100, 260, 10.0, 10.01,
                         np.arange(160, dtype=np.int16),
                         np.arange(160, dtype=np.int16) if processed else None,
                         metadata=metadata or {"vad_peak": 0.7, "voiced_frames": 3})


class BlockingTelemetry:
    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls = []

    def record_near_miss(self, pcm, transcript, reason, **kwargs):
        self.entered.set()
        self.release.wait(2)
        self.calls.append((pcm, transcript, reason, kwargs))


def test_submit_is_nonblocking_bounded_and_deduplicates_stage():
    telemetry = BlockingTelemetry()
    archive = QueuedDiagnosticArchive(telemetry, max_pending=1)
    archive.start()
    assert archive.submit(snapshot("first"), "quiet", stage="acoustic_unconfirmed")
    assert telemetry.entered.wait(1)
    assert archive.submit(snapshot("second"), "quiet", stage="acoustic_unconfirmed")
    assert archive.submit(snapshot("second"), "quiet", stage="acoustic_unconfirmed")
    assert not archive.submit(snapshot("third"), "quiet", stage="acoustic_unconfirmed")
    assert archive.counters() == {"accepted": 2, "completed": 0, "archived": 0,
                                  "failed": 0, "suppressed": 0, "disabled": 0,
                                  "partial": 0, "dropped": 1, "duplicates": 1,
                                  "lost_failures": 0}
    assert archive.close(0.01) is False
    telemetry.release.set()
    assert archive.close(1) is True
    assert archive.close(0) is True
    assert len(telemetry.calls) == 2
    kwargs = telemetry.calls[0][3]
    assert kwargs["review_state"] == "unknown" and kwargs["diagnostic_only"] is True
    assert kwargs["processed_alignment"] == "unavailable"
    assert kwargs["source_epoch"] == 7 and kwargs["source_start_sample"] == 100
    assert kwargs["source_end_sample"] == 260 and kwargs["relative_start_sample"] == 0
    assert kwargs["relative_end_sample"] == 160
    assert archive.counters()["completed"] == 2


def test_different_capture_stages_are_independent_and_failures_are_reported():
    class Failing:
        def record_near_miss(self, *_args, **_kwargs):
            raise OSError("disk unavailable")

    archive = QueuedDiagnosticArchive(Failing(), max_failures=1)
    archive.start()
    assert archive.submit(snapshot(), "weak", stage="acoustic_unconfirmed")
    assert archive.submit(snapshot(), "final", stage="final_no_trigger")
    assert archive.close(1)
    counters = archive.counters()
    assert counters["accepted"] == counters["failed"] == 2
    assert counters["lost_failures"] == 1
    assert archive.poll_failures() == ["OSError: disk unavailable"]
    assert archive.poll_failures() == []


def test_submission_rejects_coercions_and_never_accepts_labels():
    archive = QueuedDiagnosticArchive(BlockingTelemetry())
    archive.start()
    try:
        for bad in (True, 1.5, -1):
            try:
                archive.submit(snapshot(), "x", stage="s", latency_ms=bad)
                assert False, "invalid latency accepted"
            except ValueError:
                pass
        try:
            archive.submit(snapshot(), "x", stage="s", review_state="positive")
            assert False, "review label accepted"
        except TypeError:
            pass
    finally:
        assert archive.close(0.1)


def test_frozen_score_mapping_is_projected_and_capture_id_falls_back():
    telemetry = BlockingTelemetry()
    telemetry.release.set()
    archive = QueuedDiagnosticArchive(telemetry)
    archive.start()
    source = snapshot(metadata={"capture_id": "cap-from-metadata",
                                "scores": {"alexa": 0.83, "invalid": 1.4,
                                           "not_number": "0.2"}})
    assert archive.submit(source, "acoustic gap", stage="acoustic_unconfirmed")
    assert archive.close(1)
    kwargs = telemetry.calls[0][3]
    assert kwargs["capture_id"] == "cap-from-metadata"
    assert kwargs["scores"] == {"alexa": 0.83}


def test_blank_reason_and_stage_are_rejected():
    archive = QueuedDiagnosticArchive(BlockingTelemetry())
    archive.start()
    try:
        for reason, stage in ((" ", "stage"), ("reason", "\t")):
            try:
                archive.submit(snapshot(), reason, stage=stage)
                assert False, "blank diagnostic accepted"
            except ValueError:
                pass
    finally:
        assert archive.close(0.1)


def test_suppressed_disabled_and_partial_outcomes_have_distinct_counters():
    from types import SimpleNamespace

    class Outcomes:
        def __init__(self):
            self.results = iter((
                SimpleNamespace(audio_status="suppressed", audio_reason="rate_limited"),
                SimpleNamespace(audio_status="disabled", audio_reason="archive_disabled"),
                SimpleNamespace(audio_status="failed", audio_reason="disk_error"),
                SimpleNamespace(audio_status="saved", processed_audio_status="failed",
                                processed_audio_reason="write_error"),
            ))

        def record_near_miss(self, *_args, **_kwargs):
            return next(self.results)

    archive = QueuedDiagnosticArchive(Outcomes())
    archive.start()
    for index in range(4):
        assert archive.submit(snapshot("seg-{}".format(index), processed=index == 3),
                              "reason", stage="final")
    assert archive.close(1)
    counts = archive.counters()
    assert counts["completed"] == 4 and counts["archived"] == 1
    assert counts["suppressed"] == 1 and counts["disabled"] == 1
    assert counts["failed"] == 1 and counts["partial"] == 1
    assert archive.poll_failures() == ["telemetry audio failed: disk_error"]

