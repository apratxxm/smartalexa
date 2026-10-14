from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import wave

import numpy as np
import pytest

from assistant.experimental.wake_train import candidate_training as candidate
from assistant.experimental.wake_train import reviewed_training as reviewed


def _wav(path, samples):
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(np.asarray(samples, dtype="<i2").tobytes())


def _row(path, presence, split, group, *, start=0, end=16000, prov=None):
    return {"path": str(path), "wav_sha256": candidate.sha256_file(path), "n_frames": 16000,
            "sample_rate": 16000, "presence": presence, "split": split, "group_id": group,
            "source_start_sample": start, "source_end_sample": end,
            "window_upper_bound": 100, "session_provenance": prov}


def _history_row(path, presence, split, group, digest, *, start=0, end=16000):
    return {"path": str(path), "wav_sha256": digest, "n_frames": 16000, "sample_rate": 16000,
            "presence": presence, "split": split, "group_id": group,
            "source_start_sample": start, "source_end_sample": end,
            "wake_start_sample": start if presence == "positive" else None,
            "wake_end_sample": end if presence == "positive" else None,
            "capture_session_id": group, "segment_id": "", "overlap_of": "",
            "session_provenance": None}


def _ready_plan(tmp_path):
    rows = []
    for label, split, group in (("positive", "train", "pt"), ("positive", "holdout", "ph"),
                                ("negative", "train", "nt"), ("negative", "holdout", "nh")):
        path = tmp_path / f"{label}-{split}.wav"
        _wav(path, np.arange(16000, dtype=np.int16))
        start, end = (2000, 5000) if label == "positive" else (0, 16000)
        row = _row(path, label, split, group, start=start, end=end)
        row.update(selected_sample_count=end - start,
                   retained_window_count=reviewed._retained_window_count(end - start),
                   extractor_hop_count=reviewed._hop_window_count(end - start),
                   temporary_float32_bytes=(reviewed._hop_window_count(end - start)
                                             + reviewed._retained_window_count(end - start))
                   * reviewed.FEATURE_FLOAT32_BYTES,
                   selected_for_training=split == "train")
        rows.append(row)
    selection = {"algorithm": "session_round_robin_sha256_v1", "per_class_limit": 128,
                 "selected_counts": {"positive": 1, "negative": 1},
                 "excluded_records": [{"label": "negative", "wav_sha256": "b" * 64,
                                       "start": 0, "end": 16000, "group_id": "nt"}]}
    report = {"data_eligible": True, "training_ready": False, "training_supported": True,
              "resource_admission_checked": False, "unique_audio_seconds": 4,
              "window_upper_bound": 20, "train_retained_windows": 20,
              "positive_train_count": 1,
              "positive_holdout_count": 1, "negative_train_count": 1,
              "negative_holdout_count": 1, "positive_train_groups": ["pt"],
              "positive_holdout_groups": ["ph"], "negative_train_groups": ["nt"],
              "negative_holdout_groups": ["nh"], "excluded_unsure": 1,
              "missing_historic_recordings": ["missing-old.wav"],
              "review_metadata_rows": 5, "review_store_sha256": None,
              "reviewed_training_selection": selection,
              "blocked_reasons": []}
    # An unsure row is present in the plan, but never becomes a training bag.
    path = tmp_path / "unsure.wav"
    _wav(path, np.zeros(16000, dtype=np.int16))
    rows.append(_row(path, "unsure", "train", "unsure", end=16000))
    return {"rows": rows, "report": report}


def _assets(monkeypatch, tmp_path):
    store = tmp_path / "store.json"
    store.write_text('{"schema":"fixture"}', encoding="utf-8")
    stock = tmp_path / "stock.onnx"
    stock.write_bytes(b"stock")
    front = tmp_path / "front"
    front.mkdir()
    (front / "melspectrogram.onnx").write_bytes(b"mel")
    (front / "embedding_model.onnx").write_bytes(b"embed")
    monkeypatch.setattr(reviewed.candidate, "STOCK_MODEL", stock)
    monkeypatch.setattr(reviewed.candidate, "FRONTEND_DIR", front)
    return store


def _install_plan(monkeypatch, plan, store):
    plan["report"]["review_store_sha256"] = hashlib.sha256(store.read_bytes()).hexdigest()
    monkeypatch.setattr(reviewed, "_inventory", lambda *_: (plan, {}))


def test_training_feeds_exact_positive_samples_and_whole_negative_clip(monkeypatch, tmp_path):
    store = _assets(monkeypatch, tmp_path)
    plan = _ready_plan(tmp_path)
    _install_plan(monkeypatch, plan, store)
    seen = []

    def extract(samples, **kwargs):
        seen.append(np.asarray(samples).copy())
        return np.zeros((2, 16, 96), dtype=np.float32)

    monkeypatch.setattr(reviewed.candidate, "extract_bag", extract)
    def fake_fit(pos, neg, target, **kwargs):
        kwargs["report_extra"] = candidate._validate_report_extra(kwargs["report_extra"])
        return {"positive": len(pos), "negative": len(neg), **kwargs}

    monkeypatch.setattr(reviewed.candidate, "fit_and_export", fake_fit)
    output = tmp_path / "new" / "candidate.onnx"
    result = reviewed.run(store, output, memory_gate=lambda: reviewed.MIN_FREE_BYTES)
    assert len(seen) == 2
    assert np.array_equal(seen[0], np.arange(16000, dtype=np.int16)[2000:5000])
    assert np.array_equal(seen[1], np.arange(16000, dtype=np.int16))
    assert result["positive"] == result["negative"] == 1
    assert result["label_source"] == "human_reviewed_spans"
    assert result["report_extra"]["positive_span_count"] == 1
    assert result["report_extra"]["negative_holdout_excluded"] == 1
    assert result["report_extra"]["missing_historic_audio_excluded"] == 1
    assert result["report_extra"]["reviewed_training_selection"]["selected_counts"] == {
        "positive": 1, "negative": 1}
    assert result["report_extra"]["reviewed_training_selection"]["excluded_records"][0]["label"] == "negative"


def test_holdout_and_unsure_never_reach_feature_extractor(monkeypatch, tmp_path):
    store = _assets(monkeypatch, tmp_path)
    plan = _ready_plan(tmp_path)
    _install_plan(monkeypatch, plan, store)
    assert all(not row["selected_for_training"] for row in plan["rows"] if row["split"] == "holdout")
    calls = []
    factory_calls = []
    resets = []

    class FakeFrontend:
        def reset(self):
            resets.append(True)

    frontend = FakeFrontend()

    def factory():
        factory_calls.append(True)
        return frontend

    def extract(samples, *, feature_factory):
        feature_factory().reset()
        calls.append(len(samples))
        return np.zeros((1, 16, 96), np.float32)

    monkeypatch.setattr(reviewed.candidate, "extract_bag", extract)
    monkeypatch.setattr(reviewed.candidate, "fit_and_export", lambda *a, **k: {})
    reviewed.run(store, tmp_path / "candidate.onnx", memory_gate=lambda: reviewed.MIN_FREE_BYTES,
                 feature_factory=factory)
    assert calls == [3000, 16000]
    assert factory_calls == [True]
    assert len(resets) == 2


def test_blocked_current_data_refuses_before_directory_or_dependencies(monkeypatch, tmp_path):
    store = tmp_path / "empty.json"
    store.write_text("{}", encoding="utf-8")
    plan = {"rows": [], "report": {"data_eligible": False, "training_ready": False,
                                     "blocked_reasons": ["missing train positive session groups"]}}
    _install_plan(monkeypatch, plan, store)
    monkeypatch.setattr(reviewed.candidate, "_check_sources", lambda *_: pytest.fail("must not hash/model-load"))
    outdir = tmp_path / "should-not-exist"
    with pytest.raises(ValueError, match="missing train positive"):
        reviewed.run(store, outdir / "candidate.onnx", memory_gate=lambda: pytest.fail("gate too late"))
    assert not outdir.exists()


def test_span_source_mutation_during_extraction_blocks_fit(monkeypatch, tmp_path):
    store = _assets(monkeypatch, tmp_path)
    plan = _ready_plan(tmp_path)
    _install_plan(monkeypatch, plan, store)
    original = reviewed.candidate._check_sources
    calls = 0

    def mutate(paths):
        nonlocal calls
        calls += 1
        result = original(paths)
        if calls == 1:
            return result
        Path(plan["rows"][0]["path"]).write_bytes(b"changed")
        return original(paths)

    monkeypatch.setattr(reviewed.candidate, "_check_sources", mutate)
    monkeypatch.setattr(reviewed.candidate, "extract_bag", lambda samples, **kwargs: np.zeros((1, 16, 96), np.float32))
    monkeypatch.setattr(reviewed.candidate, "fit_and_export", lambda *a, **k: pytest.fail("must not fit"))
    with pytest.raises(ValueError, match="changed during feature extraction"):
        reviewed.run(store, tmp_path / "candidate.onnx", memory_gate=lambda: reviewed.MIN_FREE_BYTES)


def test_before_publication_guard_rechecks_sources_after_callback(tmp_path):
    source = tmp_path / "review-store.json"
    source.write_text("original", encoding="utf-8")
    hashes = candidate._check_sources([source])

    def mutate_source():
        source.write_text("changed", encoding="utf-8")

    with pytest.raises(ValueError, match="changed before publication"):
        candidate._verify_sources_before_publish(hashes, mutate_source)


def test_preflight_bounds_raw_metadata_before_review_hashing(tmp_path):
    missing = tmp_path / "archive" / "missing.wav"
    missing.parent.mkdir()
    row = {"path": str(missing), "n_frames": 16000}
    store = {"schema": reviewed.review.STORE_SCHEMA, "reviews": [row]}
    reviewed._preflight_store(store, missing.parent)
    oversized = missing.parent / "large.wav"
    oversized.write_bytes(b"x" * (reviewed.MAX_WAV_BYTES + 1))
    store["reviews"] = [{"path": str(oversized), "n_frames": 16000}]
    with pytest.raises(ValueError, match="1 MiB preflight"):
        reviewed._preflight_store(store, missing.parent)
    store["reviews"] = [row] * (reviewed.MAX_STORE_ROWS_FOR_PREFLIGHT + 1)
    with pytest.raises(ValueError, match="512-row"):
        reviewed._preflight_store(store, missing.parent)


def test_cross_partition_session_group_is_rejected(monkeypatch, tmp_path):
    store_path = tmp_path / "store.json"
    store_path.write_text("{}", encoding="utf-8")
    rows = [_history_row(store_path, label, split, group, f"{index + 1:064x}",
                         start=100, end=300) if label == "positive"
            else _history_row(store_path, label, split, group, f"{index + 1:064x}")
            for index, (label, split, group) in enumerate((
                ("positive", "train", "shared"), ("negative", "train", "nt"),
                ("positive", "holdout", "shared"), ("negative", "holdout", "nh")))]
    store = {"schema": reviewed.review.STORE_SCHEMA, "reviews": []}
    monkeypatch.setattr(reviewed.review, "load_preparation_store", lambda _: (store, "digest"))
    monkeypatch.setattr(reviewed.review, "review_history", lambda *_: (rows, rows, []))
    _, report = reviewed.prepare(store_path, audio_root=tmp_path)
    assert "capture session group crosses train and holdout" in report["blocked_reasons"]
    assert report["data_eligible"] is False
    assert report["training_ready"] is False
    assert report["training_supported"] is True
    assert report["resource_admission_checked"] is False


def test_overlapping_segment_identity_across_groups_is_rejected(tmp_path):
    audio_root = tmp_path / "archive"
    audio_root.mkdir()
    rows = []
    for index, session in enumerate(("session-a", "session-b")):
        path = audio_root / f"negative-{index}.wav"
        _wav(path, np.full(16000, index, dtype=np.int16))
        rows.append({"event_id": f"event-{index}", "audio_field": "audio_path",
                     "source_provenance": "human-listening", "reviewer": "reviewer",
                     "reviewed_at": "2026-01-01T00:00:00Z", "human_reviewed": True,
                     "session_id": session, "capture_session_id": session,
                     "segment_id": "shared-segment", "overlap_of": "",
                     "presence": "negative", "addressed_assistant": "no",
                     "entire_clip_negative": True, "path": str(path),
                     "wav_sha256": candidate.sha256_file(path), "n_frames": 16000,
                     "sample_rate": 16000})
    store_path = tmp_path / "reviews.json"
    store_path.write_text(json.dumps({
        "schema": reviewed.review.STORE_SCHEMA, "reviews": rows}), encoding="utf-8")
    with pytest.raises(ValueError, match="segments assigned to conflicting human session groups"):
        reviewed.prepare(store_path, audio_root=audio_root)


def test_parameters_and_candidate_output_safety(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="epochs"):
        reviewed.run("missing", tmp_path / "x.onnx", epochs=True)
    with pytest.raises(ValueError, match="seed"):
        reviewed.run("missing", tmp_path / "x.onnx", seed=-1)
    existing = tmp_path / "existing.onnx"
    existing.write_bytes(b"keep")
    with pytest.raises(ValueError, match="already exists"):
        reviewed.run("missing", existing)


def test_candidate_report_defaults_to_machine_provenance_and_rejects_override():
    assert inspect.signature(candidate.fit_and_export).parameters["label_source"].default == "machine_consensus"
    machine = candidate._label_provenance("machine_consensus")
    human = candidate._label_provenance("human_reviewed_spans")
    assert machine["training_labels"]["positive_human_reviewed"] is False
    assert machine["training_labels"]["positive_source"] == "machine_consensus"
    assert human["training_labels"]["positive_human_reviewed"] is True
    assert human["training_labels"]["all_human_reviewed_positive_rows"] is True
    assert "exact human-reviewed wake spans" in human["disclaimer"]
    assert human["training_labels"]["production_eligible"] is False
    with pytest.raises(ValueError, match="reserved or unknown"):
        candidate._validate_report_extra({"training_labels": {"positive_source": "human_reviewed_spans"}})
    disclosure = {"algorithm": "session_round_robin_sha256_v1", "per_class_limit": 128,
                  "selected_counts": {"positive": 0, "negative": 1},
                  "excluded_records": [{"label": "negative", "wav_sha256": "a" * 64,
                                        "start": 0, "end": 100, "group_id": "session"}]}
    assert candidate._validate_report_extra({"reviewed_training_selection": disclosure})[
        "reviewed_training_selection"] == disclosure
    invalid = {**disclosure, "private_transcript": "not allowed"}
    with pytest.raises(ValueError, match="unknown or missing fields"):
        candidate._validate_report_extra({"reviewed_training_selection": invalid})
    invalid_counts = {**disclosure, "selected_counts": {"positive": True, "negative": 1}}
    with pytest.raises(ValueError, match="selection counts are invalid"):
        candidate._validate_report_extra({"reviewed_training_selection": invalid_counts})


def test_safe_plan_summary_omits_reviewer_notes_and_transcripts(monkeypatch, tmp_path):
    store = tmp_path / "store.json"
    store.write_text("{}", encoding="utf-8")
    row = _history_row("x.wav", "positive", "train", "g", "a" * 64,
                       start=100, end=200)
    row.update(review_note="private", transcript="Alexa")
    monkeypatch.setattr(reviewed.review, "load_preparation_store",
                        lambda _: ({"schema": reviewed.review.STORE_SCHEMA, "reviews": []}, "store-digest"))
    monkeypatch.setattr(reviewed.review, "review_history", lambda *_: ([row], [row], []))
    safe, result = reviewed.prepare(store)
    assert "review_note" not in safe["rows"][0] and "transcript" not in safe["rows"][0]
    assert "review_note" not in str(result) and "Alexa" not in str(result)


def test_149_review_plan_has_data_blockers_without_legacy_caps(monkeypatch, tmp_path):
    path = tmp_path / "positive.wav"
    _wav(path, np.zeros(16000, dtype=np.int16))
    usable = []
    for index in range(149):
        split, group = (("train", "negative-train") if index < 100
                        else ("holdout", "negative-holdout"))
        usable.append(_history_row(path, "negative", split, group, f"{index + 1:064x}"))
    usable.extend(_history_row(path, "unsure", "train", "unsure", f"{index + 1:064x}")
                  for index in range(149, 153))
    store = {"schema": "memoir.wake-human-review.v1",
             "reviews": [{"path": str(path), "n_frames": 16000}] * len(usable)}
    monkeypatch.setattr(reviewed.review, "load_preparation_store", lambda _: (store, "store-hash"))
    monkeypatch.setattr(reviewed.review, "review_history", lambda *_: (store["reviews"], usable, []))
    plan, report = reviewed.prepare("unused", audio_root=tmp_path)
    assert len(plan["rows"]) == 149
    assert report["review_metadata_rows"] == 153
    assert not report["data_eligible"] and report["training_ready"] is False
    assert "missing train positive session groups" in report["blocked_reasons"]
    assert "missing holdout positive session groups" in report["blocked_reasons"]
    assert all("review count exceeds" not in reason and "8192-window" not in reason
               for reason in report["blocked_reasons"])


def test_legacy_prepare_training_keeps_its_100_review_cap():
    store = {"schema": reviewed.review.STORE_SCHEMA, "reviews": [{}] * 101}
    with pytest.raises(ValueError, match="review count exceeds preparation limit"):
        reviewed.review.prepare_training(store)


def test_exact_retained_window_and_peak_temporary_arithmetic():
    for samples in (1, 1279, 1280, 16000, 383999, 384000):
        hop_count = (32000 + samples + 12800 + candidate.HOP - 1) // candidate.HOP
        synthetic_ends = [index * candidate.HOP for index in range(1, hop_count + 1)]
        retained = candidate.retained_window_indices(synthetic_ends, samples)
        assert reviewed._retained_window_count(samples) == len(retained)
        planned = reviewed._plan_row({"presence": "negative", "n_frames": samples})
        assert planned["retained_window_count"] == len(retained)
        assert planned["temporary_float32_bytes"] == (hop_count + len(retained)) * 16 * 96 * 4


def test_session_round_robin_selection_is_deterministic_and_disclosed():
    rows = []
    for index in range(130):
        group = "a" if index < 64 else "b" if index < 128 else "c"
        rows.append({"wav_sha256": f"{index + 1:064x}", "source_start_sample": 0,
                     "source_end_sample": 1000, "group_id": group})
    first, excluded, details = reviewed._session_aware_selection(rows, "positive")
    second, excluded_again, details_again = reviewed._session_aware_selection(rows, "positive")
    assert [reviewed._row_identity(row) for row in first] == [reviewed._row_identity(row) for row in second]
    assert [reviewed._row_identity(row) for row in excluded] == [reviewed._row_identity(row) for row in excluded_again]
    assert details == details_again
    assert len(first) == 128 and len(excluded) == 2
    assert {row["group_id"] for row in first} == {"a", "b", "c"}
    assert details["positive_train_selected_count"] == 128
    assert details["positive_train_excluded_count"] == 2
    assert len(details["positive_train_excluded_sha256"]) == 64
    assert details["positive_train_excluded_groups"] == ["a", "b"]


def test_selection_refuses_more_than_128_independent_groups():
    rows = [{"wav_sha256": f"{index + 1:064x}", "source_start_sample": 0,
             "source_end_sample": 1000, "group_id": f"group-{index}"}
            for index in range(129)]
    with pytest.raises(ValueError, match="more than 128 independent session groups"):
        reviewed._session_aware_selection(rows, "negative")
