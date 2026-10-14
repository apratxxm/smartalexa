import json
import wave

import pytest

from assistant.experimental.wake_train import active_learning as review
from assistant.experimental.wake_train import candidate_training
from assistant.experimental.wake_train import consensus_labels as consensus
from assistant.experimental.wake_train import reviewed_audio_eval as evaluation


def _wav(path, frames=16000, marker=1):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setframerate(16000)
        output.setsampwidth(2)
        output.writeframes(bytes((marker, 0)) * frames)
    return path


def _review_row(path, *, presence="unsure", event="event", session="session-one", **changes):
    identity = review.wav_identity(path)
    row = {
        "path": str(path), **identity, "event_id": event, "audio_field": "audio_path",
        "session_id": session, "source_provenance": "fixture human listening review",
        "reviewer": "fixture-human", "human_reviewed": True,
        "reviewed_at": "2026-10-10T12:00:00+00:00", "presence": presence,
        "addressed_assistant": "unsure", "wake_start_s": None, "wake_end_s": None,
        "entire_clip_negative": presence == "negative",
    }
    if presence == "unsure":
        row["wake_presence"] = "present"
    return {**row, **changes}


def _history(tmp_path, *, positives=4, negatives=4, missing_positive=False):
    audio_root = tmp_path / "audio"
    rows = []
    for index in range(positives):
        path = _wav(audio_root / f"heard-{index}.wav", marker=index + 1)
        rows.append(_review_row(path, event=f"heard-{index}", wake_presence="present"))
    for index in range(negatives):
        path = _wav(audio_root / f"negative-{index}.wav", marker=index + 21)
        rows.append(_review_row(path, presence="negative", event=f"negative-{index}"))
    store_path = tmp_path / "human_reviews.json"
    store = {"schema": review.STORE_SCHEMA, "reviews": rows}
    store_path.write_text(json.dumps(store), encoding="utf-8")
    if missing_positive and positives:
        (audio_root / "heard-0.wav").unlink()
    return audio_root, store_path, store


def _telemetry(tmp_path, *, count=4, now=100000):
    audio_root = tmp_path / "audio"
    events = []
    for index in range(count):
        path = _wav(audio_root / f"near-{index}.wav", marker=index + 41)
        events.append({
            "event_id": f"near-{index}", "ts": now - 100 + index,
            "trigger_source": "near_miss", "candidate_kind": "near_miss",
            "wake_score": 0.8, "capture_session_id": f"near-session-{index % 2}",
            "audio_path": str(path), "end_reason": "wake_rejected",
        })
    telemetry = tmp_path / "telemetry.jsonl"
    telemetry.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")
    return telemetry, audio_root


def test_metadata_selection_caps_deduplicates_and_preserves_store(tmp_path):
    audio_root, store_path, _store = _history(tmp_path, positives=5, negatives=5)
    telemetry, _ = _telemetry(tmp_path, count=5)
    before = store_path.read_bytes()
    report = evaluation.build_report(
        store_path=store_path, telemetry=telemetry, audio_root=audio_root,
        project_root=tmp_path, max_clips=12, now=100000)

    assert report["schema"] == "memoir.wake-reviewed-audio-eval.v1"
    assert report["selection"]["selected_total"] == 12
    assert report["selection"]["selected_by_kind"] == {
        "human_presence_only": 4,
        "recent_unreviewed_near_miss": 4,
        "human_confirmed_negative": 4,
    }
    assert len({row["wav_sha256"] for row in report["records"]}) == 12
    assert sum(row["expected_presence"] is True for row in report["records"]) == 4
    assert sum(row["expected_presence"] is False for row in report["records"]) == 4
    assert sum(row["expected_presence"] is None for row in report["records"]) == 4
    assert all(row["diagnostic_label"] == "unassigned" for row in report["records"])
    assert all(row["training_eligible"] is False for row in report["records"])
    assert report["training_eligible"] is False
    assert report["production_eligible"] is False
    assert report["heldout_claim"] is False
    assert store_path.read_bytes() == before


def test_missing_presence_audio_reports_gap_and_is_not_selected(tmp_path):
    audio_root, store_path, _store = _history(
        tmp_path, positives=1, negatives=1, missing_positive=True)
    telemetry, _ = _telemetry(tmp_path, count=0)
    report = evaluation.build_report(
        store_path=store_path, telemetry=telemetry, audio_root=audio_root,
        project_root=tmp_path, max_clips=12, now=100000)

    assert report["selection"]["presence_positive_gap"] is True
    assert all(row["kind"] != "human_presence_only" for row in report["records"])
    assert report["selection"]["selected_by_kind"] == {"human_confirmed_negative": 1}


def test_duplicate_wav_hash_is_selected_once_across_near_miss_events(tmp_path):
    audio_root, store_path, _store = _history(tmp_path, positives=1, negatives=1)
    telemetry, _ = _telemetry(tmp_path, count=2)
    (audio_root / "near-1.wav").write_bytes((audio_root / "near-0.wav").read_bytes())
    report = evaluation.build_report(
        store_path=store_path, telemetry=telemetry, audio_root=audio_root,
        project_root=tmp_path, max_clips=12, now=100000)

    near = [row for row in report["records"] if row["kind"] == "recent_unreviewed_near_miss"]
    assert len(near) == 1
    assert len({row["wav_sha256"] for row in report["records"]}) == len(report["records"])


def test_fake_serial_teachers_score_only_review_labelled_rows_and_keep_store_read_only(tmp_path):
    audio_root, store_path, _store = _history(tmp_path, positives=2, negatives=2)
    telemetry, _ = _telemetry(tmp_path, count=2)
    before = store_path.read_bytes()
    calls = []

    def fake_stage(stage, records):
        calls.append(stage)
        return {
            "stage": stage,
            "model": {"model": f"fake-{stage}"},
            "providers": {stage: "fixture"},
            "records": [{
                "event_id": item["event_id"], "wav_sha256": item["wav_sha256"],
                "n_frames": item["n_frames"], "sample_rate": item["sample_rate"],
                "text": "Alexa" if index < 4 and stage == "moonshine" else "music",
                "exact_alexa": index < 4 and stage == "moonshine",
                "text_truncated": False, "elapsed_seconds": 0.25,
                "source": "offline_same_wav_inference",
            } for index, item in enumerate(records)],
        }

    report = evaluation.build_report(
        store_path=store_path, telemetry=telemetry, audio_root=audio_root,
        project_root=tmp_path, max_clips=12, run=True, now=100000,
        stage_runner=fake_stage)

    assert calls == ["moonshine", "whisper"]
    assert report["teacher_summary"]["moonshine"]["true_positive"] == 2
    assert report["teacher_summary"]["moonshine"]["false_positive"] == 0
    assert report["teacher_summary"]["whisper"]["false_negative"] == 2
    assert len(report["teacher_disagreement_case_ids"]) == 4
    assert all(row["diagnostic_label"] == "unassigned" for row in report["records"])
    assert store_path.read_bytes() == before
    assert report["labels_written_to_human_store"] is False


def test_changed_wav_identity_after_teacher_stage_fails_closed(tmp_path):
    audio_root, store_path, _store = _history(tmp_path, positives=1, negatives=1)
    telemetry, _ = _telemetry(tmp_path, count=0)
    changed_path = audio_root / "heard-0.wav"

    def changing_stage(stage, records):
        if stage == "whisper":
            _wav(changed_path, frames=16001)
        return {
            "stage": stage,
            "records": [{
                "event_id": item["event_id"], "wav_sha256": item["wav_sha256"],
                "n_frames": item["n_frames"], "sample_rate": item["sample_rate"],
                "text": "Alexa", "exact_alexa": True, "text_truncated": False,
                "source": "offline_same_wav_inference",
                "elapsed_seconds": 0.1,
            } for item in records],
        }

    with pytest.raises(ValueError, match="changed"):
        evaluation.build_report(
            store_path=store_path, telemetry=telemetry, audio_root=audio_root,
            project_root=tmp_path, max_clips=12, run=True, now=100000,
            stage_runner=changing_stage)


def test_truncated_transcript_is_unknown_and_raw_text_never_enters_stdout(tmp_path):
    cases = [{
        "kind": "human_presence_only", "expected_presence": True,
        "source_label_provenance": "human_wake_presence_present", "wav_sha256": "a" * 64,
        "n_frames": 16000, "sample_rate": 16000, "session_id": "one", "capture_session_id": "one",
        "source_event_id": "event", "audio_field": "audio_path", "recorded_split": None,
    }]
    stages = {
        "stage": "moonshine",
        "records": [{"event_id": "eval-01-aaaaaaaa", "wav_sha256": "a" * 64,
                     "n_frames": 16000, "sample_rate": 16000, "source": "offline_same_wav_inference",
                     "text": "PRIVATE RAW TEXT Alexa", "exact_alexa": True, "text_truncated": True,
                     "elapsed_seconds": 0.2}],
    }
    other = {**stages, "stage": "whisper", "records": [dict(stages["records"][0], text="music",
                    exact_alexa=False, text_truncated=False)]}
    rows, summaries, disagreements = evaluation._teacher_diagnostics(
        cases, stages, other, include_transcripts=True)
    assert rows[0]["teacher_outcomes"]["moonshine"]["exact_token_detected"] is None
    assert summaries["moonshine"]["labelled_cases_scored"] == 0
    assert not disagreements
    report = {
        "schema": evaluation.SCHEMA, "mode": "diagnostic_inference",
        "training_eligible": False, "production_eligible": False,
        "labels_written_to_human_store": False,
        "selection": {"selected_total": 1, "selected_by_kind": {"human_presence_only": 1},
                      "presence_positive_gap": False},
        "records": rows, "teacher_summary": summaries,
        "teacher_disagreement_case_ids": [], "heldout_claim": False,
    }
    assert "PRIVATE RAW TEXT" in rows[0]["teacher_outcomes"]["moonshine"]["transcript"]
    summary = evaluation.stdout_summary(report)
    assert "PRIVATE RAW TEXT" not in json.dumps(summary)
    assert set(summary["cases"][0]["teacher_outcomes"]["moonshine"]) == {
        "exact_token_detected", "transcript_truncated", "elapsed_seconds",
    }


def test_candidate_trainer_rejects_diagnostic_schema():
    report = {"schema": evaluation.SCHEMA, "records": [], "human_reviewed": False,
              "production_eligible": False}
    with pytest.raises(ValueError, match="machine-consensus"):
        candidate_training.select_training_rows(report, [])


def test_raw_transcript_opt_in_requires_private_output_authorization():
    with pytest.raises(ValueError, match="private output path"):
        evaluation.build_report(run=True, include_transcripts=True)


def test_output_guard_requires_new_path_under_proposal_root(tmp_path, monkeypatch):
    root = tmp_path / "proposals"
    root.mkdir()
    monkeypatch.setattr(consensus, "OUTPUT_ROOT", root)
    assert evaluation._safe_output(root / "diagnostic.json") == root / "diagnostic.json"
    existing = root / "existing.json"
    existing.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="exists"):
        evaluation._safe_output(existing)
    with pytest.raises(ValueError, match="under experimental/wake_train/proposals"):
        evaluation._safe_output(tmp_path / "outside.json")


def test_run_cli_writes_private_file_then_prints_allowlisted_summary(tmp_path, monkeypatch, capsys):
    root = tmp_path / "proposals"
    root.mkdir()
    output = root / "pilot.json"
    monkeypatch.setattr(consensus, "OUTPUT_ROOT", root)
    report = {
        "schema": evaluation.SCHEMA, "mode": "diagnostic_inference",
        "selection": {"selected_total": 1, "selected_by_kind": {"human_presence_only": 1},
                      "presence_positive_gap": False},
        "records": [{
            "case_id": "eval-01-deadbeef", "wav_sha256": "deadbeef" + "0" * 56,
            "expected_presence": True, "diagnostic_label": "unassigned",
            "teacher_outcomes": {"whisper": {
                "exact_token_detected": True, "transcript_truncated": False,
                "elapsed_seconds": 0.3, "transcript": "PRIVATE CLI TRANSCRIPT",
            }},
        }],
        "teacher_summary": {"whisper": {"labelled_cases_scored": 1}},
        "teacher_disagreement_case_ids": [],
    }
    monkeypatch.setattr(evaluation, "build_report", lambda **_kwargs: report)

    assert evaluation.main(["--run", "--output", str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["records"][0]["teacher_outcomes"]["whisper"]["transcript"] == "PRIVATE CLI TRANSCRIPT"
    stdout = capsys.readouterr().out
    assert "PRIVATE CLI TRANSCRIPT" not in stdout
    printed = json.loads(stdout)
    assert printed["cases"][0]["teacher_outcomes"]["whisper"]["exact_token_detected"] is True

