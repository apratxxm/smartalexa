"""Telemetry degradation tests: observability must never break commands."""
import numpy as np
import time

from assistant.config import Config
from assistant.telemetry import Event, Telemetry
from assistant.diagnostic_archive import QueuedDiagnosticArchive
from assistant.audio_jobs import AudioSnapshot


def make_telemetry(tmp_path):
    root = tmp_path / "logs"
    cfg = Config({"telemetry": {
        "jsonl_path": str(root / "telemetry.jsonl"),
        "text_log_path": str(root / "memoir.log"),
        "audio_dir": str(root / "audio"),
        "archive_audio": True,
        "max_audio_mb": 1,
    }, "audio": {"sample_rate": 16000}})
    return Telemetry(cfg)


def test_jsonl_enospc_is_best_effort(tmp_path, monkeypatch):
    telemetry = make_telemetry(tmp_path)
    real_open = open

    def fail_open(path, *args, **kwargs):
        if str(path).endswith("telemetry.jsonl"):
            raise OSError(28, "No space left on device")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", fail_open)
    telemetry.write(Event(outcome="ok"))


def test_audio_enospc_returns_empty_and_removes_partial_file(tmp_path, monkeypatch):
    telemetry = make_telemetry(tmp_path)
    monkeypatch.setattr("assistant.telemetry.shutil.disk_usage",
                        lambda _: type("Usage", (), {"free": 0})())
    event = Event()
    assert telemetry.save_audio(event, np.zeros(1600, dtype=np.int16)) == ""
    assert not list(telemetry.audio_dir.glob("*.wav"))


def test_oversize_clip_does_not_prune_existing_archive(tmp_path):
    telemetry = make_telemetry(tmp_path)
    telemetry.max_audio_mb = 0
    existing = telemetry.audio_dir / "20260101-000000_deadbeef.wav"
    existing.write_bytes(b"keep")
    assert telemetry.save_audio(Event(event_id="newclip"),
                                np.zeros(1600, dtype=np.int16)) == ""
    assert existing.exists()


def test_filename_collision_preserves_original(tmp_path):
    telemetry = make_telemetry(tmp_path)
    event = Event(event_id="collision", ts=0)
    original = telemetry.audio_dir / (time.strftime("%Y%m%d-%H%M%S", time.localtime(0)) + "_collision.wav")
    original.write_bytes(b"original")
    assert telemetry.save_audio(event, np.zeros(160, dtype=np.int16)) == ""
    assert original.read_bytes() == b"original"


def test_mid_write_failure_removes_only_new_file(tmp_path, monkeypatch):
    telemetry = make_telemetry(tmp_path)
    original = telemetry.audio_dir / "20260101-000000_existing.wav"
    original.write_bytes(b"original")

    class BrokenWave:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def setnchannels(self, *_): pass
        def setsampwidth(self, *_): pass
        def setframerate(self, *_): pass
        def writeframes(self, _): raise OSError("disk full")

    monkeypatch.setattr("assistant.telemetry.wave.open", lambda *_: BrokenWave())
    assert telemetry.save_audio(Event(event_id="new"), np.zeros(160, dtype=np.int16)) == ""
    assert original.exists()
    new_name = time.strftime("%Y%m%d-%H%M%S", time.localtime(0)) + "_new.wav"
    assert not (telemetry.audio_dir / new_name).exists()


def test_pruning_only_removes_generated_archive_files(tmp_path):
    telemetry = make_telemetry(tmp_path)
    telemetry.max_audio_mb = 0
    unrelated = telemetry.audio_dir / "third-party.wav"
    unrelated.write_bytes(b"keep")
    generated = telemetry.audio_dir / "20260101-000000_deadbeef.wav"
    generated.write_bytes(b"remove")
    telemetry._prune_audio()
    assert unrelated.exists()
    assert not generated.exists()


def test_candidate_rate_limit_suppresses_audio_but_keeps_every_decision(tmp_path):
    import json
    telemetry = make_telemetry(tmp_path)
    audio = np.ones(1600, np.int16)
    first = telemetry.record_near_miss(audio, "you", "speech_no_trigger", now=100, kind="no_trigger", segment_id="first")
    second = telemetry.record_near_miss(audio, "", "speech_no_trigger", now=101, kind="no_trigger", segment_id="second")
    rows = [json.loads(line) for line in telemetry.jsonl_path.read_text().splitlines()]
    assert len(rows) == 2
    assert first.audio_status == "saved" and first.audio_path
    assert second.audio_status == "suppressed" and second.audio_reason == "rate_limited"
    assert second.audio_path == "" and second.segment_id == "second"
    assert all(row["review_state"] == "unknown" for row in rows)
    assert first.capture_session_id == second.capture_session_id
    assert len(list(telemetry.audio_dir.rglob("*.wav"))) == 1


def test_candidate_raw_and_processed_are_frame_paired_and_hashed(tmp_path):
    import hashlib
    import wave
    telemetry = make_telemetry(tmp_path)
    raw = np.arange(1600, dtype=np.int16)
    processed = raw // 2
    event = telemetry.record_near_miss(raw, "you", "speech_no_trigger", processed_audio=processed,
                                       relative_end_sample=1600, now=10)
    for path, expected, sha in ((event.audio_path, raw, event.audio_sha256),
                                (event.processed_audio_path, processed, event.processed_audio_sha256)):
        full = telemetry.jsonl_path.parent.parent.parent / path
        assert hashlib.sha256(full.read_bytes()).hexdigest() == sha
        with wave.open(str(full)) as wav:
            np.testing.assert_array_equal(np.frombuffer(wav.readframes(1600), np.int16), expected)
    assert event.processed_alignment == "frame_paired"
    assert event.relative_start_sample == 0 and event.relative_end_sample == 1600
    assert event.processed_audio_status == "saved"


def test_candidate_length_cap_does_not_claim_complete_recording(tmp_path):
    telemetry = make_telemetry(tmp_path)
    event = telemetry.record_near_miss(np.zeros(8 * 16000, np.int16), "", "speech_no_trigger",
                                       end_reason="max_utterance", now=0)
    assert event.audio_original_n_frames == 8 * 16000
    assert event.audio_n_frames == event.relative_end_sample == 6 * 16000
    assert event.audio_duration_ms == 6000 and event.recording_truncated
    assert event.end_reason == "max_utterance:archive_length_limit"


def test_candidate_pair_budget_never_prunes_its_just_saved_raw_clip(tmp_path):
    telemetry = make_telemetry(tmp_path)
    telemetry.max_near_miss_mb = 0.01
    audio = np.zeros(4000, np.int16)
    event = telemetry.record_near_miss(audio, "", "candidate", processed_audio=audio.copy(), now=1)
    assert event.audio_status == "suppressed" and event.audio_reason == "archive_cap"
    assert not event.audio_path and not event.processed_audio_path
    assert not list(telemetry.audio_dir.rglob("*.wav"))


def test_candidate_disabled_disk_and_alignment_failures_remain_visible(tmp_path, monkeypatch):
    telemetry = make_telemetry(tmp_path)
    audio = np.zeros(1600, np.int16)
    telemetry.archive_audio = False
    event = telemetry.record_near_miss(audio, "", "candidate", now=0)
    assert event.audio_status == "disabled" and event.audio_reason == "archive_disabled"
    telemetry.archive_audio = True
    monkeypatch.setattr("assistant.telemetry.shutil.disk_usage", lambda _: type("Usage", (), {"free": 0})())
    event = telemetry.record_near_miss(audio, "", "candidate", now=10)
    assert event.audio_status == "failed" and event.audio_reason == "insufficient_disk_space"
    event = telemetry.record_near_miss(audio, "", "candidate", now=20, relative_end_sample=12)
    assert event.audio_status == "unavailable" and event.audio_reason == "invalid_relative_bounds"
    assert len(telemetry.jsonl_path.read_text().splitlines()) == 3


def test_processed_mismatch_does_not_fabricate_aligned_audio(tmp_path):
    telemetry = make_telemetry(tmp_path)
    event = telemetry.record_near_miss(np.zeros(1600, np.int16), "", "candidate", now=1,
                                       processed_audio=np.zeros(1599, np.int16))
    assert event.audio_path and not event.processed_audio_path
    assert event.processed_alignment == "unavailable"
    assert event.processed_audio_reason == "length_or_shape_mismatch"


def test_explicit_sample_time_alignment_is_preserved_for_matching_arrays(tmp_path):
    telemetry = make_telemetry(tmp_path)
    raw = np.zeros(1600, np.int16)
    processed = np.zeros_like(raw)
    event = telemetry.record_near_miss(raw, "", "candidate", now=1,
                                       processed_audio=processed,
                                       processed_alignment="sample_aligned")
    assert event.processed_alignment == "sample_aligned"


def test_unknown_alignment_defaults_to_frame_paired_for_matching_arrays(tmp_path):
    telemetry = make_telemetry(tmp_path)
    raw = np.zeros(1600, np.int16)
    event = telemetry.record_near_miss(raw, "", "candidate", now=1,
                                       processed_audio=raw.copy(),
                                       processed_alignment="unavailable")
    assert event.processed_alignment == "frame_paired"


def test_diagnostic_source_provenance_survives_wav_prefix_crop(tmp_path):
    telemetry = make_telemetry(tmp_path)
    telemetry.candidate_max_frames = 100
    pcm = np.arange(1600, dtype=np.int16)
    event = telemetry.record_near_miss(
        pcm, "", "acoustic_unconfirmed", kind="diagnostic", diagnostic_only=True,
        diagnostic_stage="acoustic_unconfirmed", capture_id="cap", proposal_id="prop",
        source_epoch=9, source_start_sample=400, source_end_sample=2000,
        source_started_at=1.25, source_completed_at=1.35,
        relative_start_sample=0, relative_end_sample=1600, now=1)
    assert event.audio_status == "saved"
    assert event.source_epoch == 9 and event.source_start_sample == 400
    assert event.source_end_sample == 2000 and event.source_started_at == 1.25
    assert event.source_completed_at == 1.35
    assert event.relative_start_sample == 0 and event.relative_end_sample == 1600
    assert event.audio_original_n_frames == 1600 and event.audio_n_frames == 100
    assert event.recording_truncated and event.diagnostic_only


def test_invalid_diagnostic_source_bundle_records_json_without_wav(tmp_path):
    telemetry = make_telemetry(tmp_path)
    pcm = np.zeros(1600, dtype=np.int16)
    event = telemetry.record_near_miss(
        pcm, "", "bad_source", kind="diagnostic", diagnostic_only=True,
        source_epoch=2, source_start_sample=10, source_end_sample=1609,
        source_started_at=1.0, source_completed_at=1.1,
        relative_start_sample=0, relative_end_sample=1600, now=1)
    assert event.audio_status == "unavailable"
    assert event.audio_reason == "invalid_source_interval_metadata"
    assert not event.audio_path
    assert not list(telemetry.audio_dir.rglob("*.wav"))


def test_queued_diagnostic_telemetry_uses_frame_paired_alignment(tmp_path):
    telemetry = make_telemetry(tmp_path)
    source = AudioSnapshot(3, "segment", 1, 20, 180, 1.0, 1.01,
                           np.zeros(160, dtype=np.int16), np.zeros(160, dtype=np.int16),
                           metadata={
                               "acoustic_proposal": {"stock_model": "alexa", "stock_score": 0.82,
                                                     "personal_hit": False, "personal_score": 0.0},
                               "proposal_qualifiers": {"stock_model": "alexa", "personal_hit": False,
                                                       "target_phrase": "hey memoir",
                                                       "fallback_target_registered": True,
                                                       "stock_qualified": True, "personal_qualified": False,
                                                       "stock_threshold": 0.3, "personal_threshold": 0.4,
                                                       "vad_threshold": 0.2},
                               "confirmation_interval": {"epoch": 3, "capture_id": "segment",
                                                          "start_sample": 20, "end_sample": 180},
                               "trigger_stage_state": "completed",
                               "second_stt_stage_state": "unrun",
                               "trigger_transcript_truncated": False,
                               "second_stt_transcript_truncated": True,
                           })
    archive = QueuedDiagnosticArchive(telemetry)
    archive.start()
    assert archive.submit(source, "weak_signal", stage="acoustic_unconfirmed")
    assert archive.close(1)
    import json
    row = json.loads(telemetry.jsonl_path.read_text(encoding="utf-8").splitlines()[0])
    assert row["processed_alignment"] == "frame_paired"
    assert row["review_state"] == "unknown" and row["diagnostic_only"] is True
    assert row["source_start_sample"] == 20 and row["source_end_sample"] == 180
    assert row["diagnostic_metadata_status"] == "structurally_validated"
    assert row["diagnostic_metadata"]["acoustic_proposal"]["stock_score"] == 0.82
    assert row["diagnostic_metadata"]["confirmation_interval"]["capture_id"] == "segment"
    assert row["diagnostic_metadata"]["trigger_stage_state"] == "completed"
    assert row["diagnostic_metadata"]["second_stt_stage_state"] == "unrun"
    assert row["diagnostic_metadata"]["second_stt_transcript_truncated"] is True


def test_invalid_diagnostic_bundle_is_explicitly_unvalidated(tmp_path):
    telemetry = make_telemetry(tmp_path)
    audio = np.zeros(160, dtype=np.int16)
    for bundle in (
        {"proposal_qualifiers": {"stock_threshold": float("nan")}},
        {"confirmation_interval": {"epoch": True, "capture_id": "cap",
                                   "start_sample": 0, "end_sample": 160}},
        {"confirmation_interval": {"epoch": 3, "capture_id": "cap",
                                   "start_sample": 10, "end_sample": 10}},
        {"confirmation_interval": {"epoch": 3, "capture_id": "",
                                   "start_sample": 0, "end_sample": 160}},
    ):
        event = telemetry.record_near_miss(audio, "", "invalid_bundle", kind="diagnostic",
                                           diagnostic_metadata=bundle, now=1)
        assert event.diagnostic_metadata is None
        assert event.diagnostic_metadata_status == "invalid"


def test_legacy_near_miss_has_no_claimed_diagnostic_bundle(tmp_path):
    telemetry = make_telemetry(tmp_path)
    event = telemetry.record_near_miss(np.zeros(160, dtype=np.int16), "", "legacy", now=1)
    assert event.diagnostic_metadata is None
    assert event.diagnostic_metadata_status == "unavailable"


def test_legitimate_generated_suffix_files_count_toward_archive_and_leave_tombstones(tmp_path):
    import json
    telemetry = make_telemetry(tmp_path)
    telemetry.max_audio_mb = 0
    for suffix in ("-retry", "-artist", "-follow-up", "-processed"):
        (telemetry.audio_dir / f"20260101-000000_deadbeef{suffix}.wav").write_bytes(b"old")
    untouched = telemetry.audio_dir / "third-party.wav"
    untouched.write_bytes(b"keep")
    telemetry._prune_audio()
    assert untouched.exists()
    assert len(list(telemetry.audio_dir.glob("*.wav"))) == 1
    rows = [json.loads(line) for line in telemetry.jsonl_path.read_text().splitlines()]
    assert len(rows) == 4 and all(row["outcome"] == "pruned" for row in rows)
    assert all(row["pruned_audio_path"] and row["audio_reason"] == "total_archive_cap" for row in rows)


def _review_row(path, **overrides):
    row = {
        "human_reviewed": True,
        "presence": "negative",
        "wav_sha256": "a" * 64,
        "path": str(path),
        "proposed_label": "uncertain",
    }
    row.update(overrides)
    return row


def _use_review_store(tmp_path, monkeypatch, rows):
    import json
    store = tmp_path / "human_reviews.json"
    store.write_text(json.dumps({"schema": "memoir.wake-human-review.v1", "reviews": rows}),
                     encoding="utf-8")
    monkeypatch.setattr("assistant.telemetry._REVIEW_STORE_PATH", store)
    return store


def test_reviewed_near_miss_counts_toward_candidate_cap_and_is_pinned(tmp_path, monkeypatch):
    telemetry = make_telemetry(tmp_path)
    telemetry.max_near_miss_mb = 0.01
    near = telemetry.audio_dir / "near_miss"
    near.mkdir()
    pinned = near / "20260101-000000_deadbeef.wav"
    pinned.write_bytes(b"p" * 12000)
    _use_review_store(tmp_path, monkeypatch,
                      [_review_row(pinned, presence="positive", proposed_label="uncertain")])

    assert not telemetry._prune_audio(1, "near_miss")
    assert pinned.exists()


def test_reviewed_wav_counts_toward_total_cap_and_blocks_archive_when_no_room(tmp_path, monkeypatch):
    telemetry = make_telemetry(tmp_path)
    pinned = telemetry.audio_dir / "20260101-000000_deadbeef.wav"
    pinned.write_bytes(b"p" * (1024 * 1024))
    _use_review_store(tmp_path, monkeypatch, [_review_row(pinned)])

    event = Event(event_id="newclip")
    assert telemetry.save_audio(event, np.zeros(160, dtype=np.int16)) == ""
    assert event.audio_status == "suppressed" and event.audio_reason == "archive_cap"
    assert pinned.exists()
    assert len(list(telemetry.audio_dir.glob("*.wav"))) == 1


def test_out_of_root_review_path_cannot_pin_matching_archive_name(tmp_path, monkeypatch):
    telemetry = make_telemetry(tmp_path)
    local = telemetry.audio_dir / "20260101-000000_deadbeef.wav"
    local.write_bytes(b"old")
    outside = tmp_path / "outside" / local.name
    outside.parent.mkdir()
    outside.write_bytes(b"reviewed elsewhere")
    _use_review_store(tmp_path, monkeypatch, [_review_row(outside)])
    telemetry.max_audio_mb = 0

    assert telemetry._prune_audio()
    assert not local.exists()
    assert outside.exists()


def test_malformed_review_store_fails_closed_for_deletion(tmp_path, monkeypatch):
    telemetry = make_telemetry(tmp_path)
    old = telemetry.audio_dir / "20260101-000000_deadbeef.wav"
    old.write_bytes(b"old")
    store = tmp_path / "human_reviews.json"
    store.write_text("{broken", encoding="utf-8")
    monkeypatch.setattr("assistant.telemetry._REVIEW_STORE_PATH", store)
    telemetry.max_audio_mb = 0

    assert not telemetry._prune_audio()
    assert old.exists()


def test_malformed_final_presence_label_does_not_pin_unreviewed_audio(tmp_path, monkeypatch):
    telemetry = make_telemetry(tmp_path)
    old = telemetry.audio_dir / "20260101-000000_deadbeef.wav"
    old.write_bytes(b"old")
    _use_review_store(tmp_path, monkeypatch,
                      [_review_row(old, presence="uncertain", human_reviewed=True)])
    telemetry.max_audio_mb = 0

    assert telemetry._prune_audio()
    assert not old.exists()


def test_unreviewed_old_generated_wav_remains_eligible_for_rolling_deletion(tmp_path, monkeypatch):
    telemetry = make_telemetry(tmp_path)
    old = telemetry.audio_dir / "20260101-000000_deadbeef.wav"
    old.write_bytes(b"old")
    _use_review_store(tmp_path, monkeypatch, [])
    telemetry.max_audio_mb = 0

    assert telemetry._prune_audio()
    assert not old.exists()
