"""Synthetic-only acceptance: no capture streams, playback, or wake model loads."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from assistant.wakebench import bench, record
from assistant.wakebench.capture import BlockSink
from assistant.wakebench.session import (Alignment, FRAME, RATE, chirp, finalize,
    locate, read_audio, read_jsonl, window, write_jsonl, write_wav)


def make_session(tmp_path, name, *, mode="positive", positions=(RATE,),
                 offset=320, drift=1., missing=None, user_invalid=False,
                 flags=(), marker=None):
    directory = tmp_path / name
    directory.mkdir()
    cue = chirp()
    duration = max(6 * RATE, max(positions) + 5 * RATE)
    rng = np.random.default_rng(sum(name.encode()))
    mic = rng.normal(0, 20, duration).astype(np.int16)
    for i, position in enumerate(positions):
        if missing != ("mic", i):
            mic[position:position + len(cue)] += cue
    if marker is not None:
        mic[2 * RATE:2 * RATE + FRAME] = round(marker * 32767)
    # Independent recordings on clocks with known affine offset/drift.
    ref_len = round(duration * drift + offset + 100)
    reference = np.interp((np.arange(ref_len) - offset) / drift,
                          np.arange(duration), mic, left=0, right=0).astype(np.int16)
    if missing is not None and missing[0] == "loopback":
        start = round(positions[missing[1]] * drift + offset)
        reference[max(0, start - 20):start + len(cue) + 20] = 0
    write_wav(directory / "mic.wav", mic)
    write_wav(directory / "loopback.wav", reference)
    write_wav(directory / "cue.wav", cue)
    header = {"type": "session", "version": 1, "session_id": name, "mode": mode,
        "distance": "near", "music": "off", "mic_wav": "mic.wav",
        "loopback_wav": "loopback.wav", "cue_wav": "cue.wav", "mic_rate": RATE,
        "loopback_rate": RATE, "mic_blocks": "mic.blocks.jsonl",
        "loopback_blocks": "loopback.blocks.jsonl", "aborted": False}
    cues = [{"type": "cue", "session_id": name, "cue_id": i,
        "anchor": mode == "negative", "user_invalid": user_invalid,
        "expected_mic_sample": position + 31,
        "expected_loopback_sample": round(position * drift + offset) - 41}
        for i, position in enumerate(positions)]
    for stream in ("mic", "loopback"):
        rows = [{"start_sample": start, "samples": FRAME, "overflow": True,
                 "discontinuity": True, "status": "synthetic input overflow"}
                for name_, start in flags if name_ == stream]
        write_jsonl(directory / f"{stream}.blocks.jsonl", rows)
    write_jsonl(directory / "manifest.jsonl", [header] + cues)
    return directory


def test_chirp_localized_within_two_samples_at_zero_db_snr():
    cue = chirp()
    rms = np.sqrt(np.mean(cue.astype(float) ** 2))
    rng = np.random.default_rng(91)
    audio = rng.normal(0, rms, 5 * RATE)
    onset = 32123
    audio[onset:onset + len(cue)] += cue
    found, confidence = locate(audio, cue, onset + 200)
    assert abs(found - onset) <= 2
    assert confidence > .6


def test_offset_and_linear_drift_recovered(tmp_path):
    positions = (RATE, 4 * RATE, 7 * RATE)
    path = make_session(tmp_path, "drift", positions=positions, offset=731, drift=1.0007)
    rows = finalize(path)
    header, cues = rows[0], rows[1:]
    assert all(abs(c["mic_cue_start"] - p) <= 2 for c, p in zip(cues, positions))
    alignment = Alignment(np.array(header["alignment"]["mic_samples"]),
                          np.array(header["alignment"]["loopback_samples"]))
    probes = np.array([2 * RATE, 5 * RATE, 6 * RATE])
    assert np.max(np.abs(alignment.map(probes) - (probes * 1.0007 + 731))) <= 2
    assert np.max(np.abs(alignment.map(alignment.map(probes), inverse=True) - probes)) < 1e-6
    assert all(c["mic_confidence"] > .9 and c["loopback_confidence"] > .7 for c in cues)


def test_sample_accurate_window_and_manifest_round_trip(tmp_path):
    path = make_session(tmp_path, "window")
    rows = finalize(path)
    cue = rows[1]
    assert cue["cue_offset_sample"] == cue["mic_cue_start"] + len(chirp())
    assert window(cue["cue_offset_sample"]) == (cue["window_start_sample"], cue["window_end_sample"])
    assert cue["window_start_sample"] - cue["cue_offset_sample"] == 2400
    assert cue["window_end_sample"] - cue["cue_offset_sample"] == 56000
    assert read_jsonl(path / "finalized.jsonl") == rows
    assert rows[0]["alignment"]["method"] == "piecewise_linear_chirp"
    assert rows[0]["alignment"]["loopback_samples"][0] - rows[0]["alignment"]["mic_samples"][0] == 320


@pytest.mark.parametrize("stream", ["mic", "loopback"])
def test_missing_cue_invalidates_attempt(tmp_path, stream):
    path = make_session(tmp_path, stream, positions=(RATE, 7 * RATE), missing=(stream, 0))
    rows = finalize(path)
    assert rows[1]["invalid"]
    assert "missing_cue" in rows[1]["invalid_reasons"]
    session = bench.Session.open(path)
    stats = bench.score_session(session, [], .5)
    assert stats["invalid_attempts"] == 1
    assert stats["valid_attempts"] == 1


@pytest.mark.parametrize("stream", ["mic", "loopback"])
def test_discontinuity_in_either_stream_invalidates_window(tmp_path, stream):
    point = 2 * RATE + (320 if stream == "loopback" else 0)
    path = make_session(tmp_path, stream, flags=[(stream, point)])
    rows = finalize(path)
    assert rows[1]["invalid"]
    assert "discontinuity" in rows[1]["invalid_reasons"]
    stats = bench.score_session(bench.Session.open(path), [(2 * RATE, .9)], .5)
    assert stats["invalid_attempts"] == 1
    assert stats["detected_attempts"] == stats["false_wakes"] == 0


def test_user_invalid_counted_and_not_false_wake(tmp_path):
    path = make_session(tmp_path, "user", user_invalid=True)
    session = bench.Session.open(path)
    row = session.cues[0]
    stats = bench.score_session(session, [(row["window_start_sample"], .9)], .5)
    assert stats["invalid_attempts"] == 1
    assert stats["valid_attempts"] == stats["false_wakes"] == 0


def test_cue_hits_ignored_and_boundaries_half_open(tmp_path):
    path = make_session(tmp_path, "boundaries")
    session = bench.Session.open(path)
    c = session.cues[0]
    points = [(c["cue_onset_sample"], .9), (c["cue_offset_sample"] - 1, .9),
              (c["window_start_sample"], .9), (c["window_end_sample"], .9)]
    stats = bench.score_session(session, points, .5, cooldown=0)
    assert stats["detected_attempts"] == stats["valid_attempts"] == 1
    assert stats["false_wakes"] == 1
    ignored = len(chirp()) + c["window_end_sample"] - c["window_start_sample"]
    assert stats["exposure_hours"] == pytest.approx((session.duration - ignored) / RATE / 3600)


def metric_session(mode="negative", duration=RATE * 3600):
    return SimpleNamespace(header={"mode": mode}, duration=duration,
                           spans=lambda: ([], [], [], 0))


def test_false_wake_rate_uses_exposure_hours_and_cooldown():
    session = metric_session()
    stats = bench.score_session(session, [(RATE, .8), (RATE + 512, .9), (4 * RATE, .8)], .5)
    assert stats["false_wakes"] == 2
    assert stats["exposure_hours"] == 1
    assert stats["false_wakes_per_hour"] == 2
    half = bench.score_session(metric_session(duration=RATE * 1800), [(RATE, .8)], .5)
    assert half["false_wakes_per_hour"] == 2


def test_thresholds_use_tuning_negatives_only():
    tuning = [(metric_session(duration=RATE * 30), [(2 * RATE, .6)])]
    threshold = bench.choose_threshold(tuning)
    assert threshold == .61
    # Higher held-out scores become measured errors; they cannot move threshold.
    held_out = bench.score_session(metric_session(duration=RATE * 30), [(2 * RATE, .95)], threshold)
    assert held_out["false_wakes"] == 1
    assert bench.choose_threshold(tuning) == threshold
    with pytest.raises(ValueError, match="negative"):
        bench.choose_threshold([(metric_session("positive"), [])])


def test_overlap_and_renamed_copies_refused_before_reporting(tmp_path):
    a = make_session(tmp_path, "split-a")
    with pytest.raises(ValueError, match="both"):
        bench.validate_split([a], [a / "manifest.jsonl"])
    b = make_session(tmp_path, "split-b")
    rows = read_jsonl(b / "manifest.jsonl")
    rows[0]["session_id"] = "split-a"
    write_jsonl(b / "manifest.jsonl", rows)
    with pytest.raises(ValueError, match="both"):
        bench.validate_split([a], [b])
    rows[0]["session_id"] = "renamed"
    write_jsonl(b / "manifest.jsonl", rows)
    (b / "mic.wav").write_bytes((a / "mic.wav").read_bytes())
    with pytest.raises(ValueError, match="Identical mic audio"):
        bench.validate_split([a], [b])


def test_negative_anchor_spans_excluded_and_missing_anchor_exposure_removed(tmp_path):
    path = make_session(tmp_path, "negative", mode="negative", positions=(RATE, 5 * RATE))
    session = bench.Session.open(path)
    cue_point = session.cues[0]["cue_onset_sample"]
    stats = bench.score_session(session, [(cue_point, .9), (3 * RATE, .9)], .5, cooldown=0)
    assert stats["false_wakes"] == 1
    assert stats["valid_attempts"] == 0
    assert stats["exposure_hours"] == pytest.approx((session.duration - 2 * len(chirp())) / RATE / 3600)
    missing = make_session(tmp_path, "missing-anchor", mode="negative", positions=(RATE, 5 * RATE), missing=("loopback", 0))
    short = bench.Session.open(missing)
    result = bench.score_session(short, [(3 * RATE, .9)], .5)
    assert result["false_wakes"] == 0
    assert result["exposure_hours"] < stats["exposure_hours"]


class FakeDetector:
    def __init__(self, candidate):
        self.closed = False

    def process(self, frame):
        assert frame.dtype == np.int16 and len(frame) == FRAME
        return max(0, float(np.max(frame)) / 32767)

    def reset(self):
        pass

    def close(self):
        self.closed = True


class IdentityEcho:
    def __init__(self, reference):
        self.reference = reference

    def process(self, frame, end):
        ref = self.reference.take_until(end)
        assert len(ref) == FRAME
        return frame


def test_commands_synthetic_end_to_end_and_report(tmp_path, monkeypatch, capsys):
    tuning = make_session(tmp_path, "tuning", mode="negative", marker=.605)
    evaluation = make_session(tmp_path, "evaluation", marker=.9)
    original = bench.evaluate
    monkeypatch.setattr(bench, "evaluate", lambda *a, **kw: original(*a, **kw,
                        detector_factory=FakeDetector, echo_factory=IdentityEcho))
    extra = tmp_path / "future.onnx"
    extra.write_bytes(b"not loaded: synthetic fake detector")
    reports = tmp_path / "reports"
    bench.main(["--tune", str(tuning), "--eval", str(evaluation), "--model", str(extra), "--reports", str(reports)])
    payload = json.loads(next(reports.glob("*.json")).read_text())
    assert len(payload["results"]) == 6  # stock, personal, arbitrary ONNX × AEC
    assert all(r["threshold"] == .61 for r in payload["results"])
    assert all(r["recall"] == 1. for r in payload["results"])
    assert "No live device acceptance" in next(reports.glob("*.md")).read_text()
    record.main(["--finalize", str(evaluation)])
    assert '"finalized": true' in capsys.readouterr().out


def test_real_echo_canceller_runs_offline_without_devices(tmp_path):
    path = make_session(tmp_path, "offline-aec")
    session = bench.Session.open(path)
    # Real livekit streaming AEC, generated WAVs, fake detector; no model/device.
    scores = bench.trace(session, "fake", True, detector_factory=FakeDetector)
    assert scores
    assert all(np.isfinite(score) for _, score in scores)


def test_runtime_detector_reuses_loader_and_frame_chunking(monkeypatch):
    calls = []
    class Loader:
        def __init__(self, cfg):
            calls.append(cfg.get("wake.model"))
            self.last_score = .75
        def process(self, frame):
            calls.append(len(frame))
        def reset(self):
            calls.append("reset")
        def close(self):
            calls.append("close")
    monkeypatch.setattr(bench, "WakeDetector", Loader)
    detector = bench.RuntimeDetector("arbitrary.onnx")
    frame = np.zeros(FRAME, np.int16)
    assert detector.process(frame) is None
    assert detector.process(frame) is None
    assert detector.process(frame) == .75
    detector.reset()
    detector.close()
    assert calls[0] == "arbitrary.onnx"
    assert calls[-2:] == ["reset", "close"]


def test_callback_sidecar_records_flags_and_native_indices(tmp_path):
    sink = BlockSink(tmp_path, "mic", 48000)
    try:
        sink.push(np.zeros(480, np.int16), capture_time=10.)
        sink.push(np.ones(480, np.int16), status="overflow", overflow=True, capture_time=10.02)
    finally:
        sink.close()
    rows = read_jsonl(tmp_path / "mic.blocks.jsonl")
    assert [r["start_sample"] for r in rows] == [0, 480]
    assert not rows[0]["discontinuity"]
    assert rows[1]["overflow"] and rows[1]["discontinuity"]
    assert len(read_audio(tmp_path / "mic.wav")) == 320


def test_dry_run_opens_no_streams_or_creates_files(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(record, "resolve_devices", lambda cfg: {"mic_name": "fake mic", "loopback_name": "fake loopback"})
    monkeypatch.setattr(record, "run", lambda *a: pytest.fail("dry-run attempted recording"))
    output = tmp_path / "must-not-exist"
    record.main(["--dry-run", "--output", str(output)])
    plan = json.loads(capsys.readouterr().out)
    assert plan["resolved_devices"]["mic_name"] == "fake mic"
    assert plan["plan"]["attempts"] == 30
    assert not output.exists()


def test_runtime_mutex_refuses_recording_before_directory_or_capture(tmp_path, monkeypatch):
    from assistant import runtime
    def refused(root):
        raise runtime.AlreadyRunning("Memoir is already running for this project")
    monkeypatch.setattr(runtime, "single_instance", refused)
    monkeypatch.setattr(record, "Capture", lambda *a: pytest.fail("opened capture before lock"))
    with pytest.raises(runtime.AlreadyRunning):
        record.run(SimpleNamespace(output=tmp_path / "no-write"), None, {})
    assert not (tmp_path / "no-write").exists()


def test_aborted_session_refused(tmp_path):
    path = make_session(tmp_path, "aborted")
    rows = read_jsonl(path / "manifest.jsonl")
    rows[0]["aborted"] = True
    write_jsonl(path / "manifest.jsonl", rows)
    with pytest.raises(ValueError, match="Aborted"):
        finalize(path)
