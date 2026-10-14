from __future__ import annotations

import numpy as np
import pytest
import wave

from assistant.experimental.wake_train import candidate_training as candidate


def _evidence(text="Alexa"):
    digest = "a" * 64
    return {
        "whisper": {"family": "whisper", "text": text, "exact_alexa": True,
                    "source": "offline_same_wav_inference", "audio_sha256": digest},
        "moonshine": {"family": "moonshine", "text": text, "exact_alexa": True,
                      "source": "offline_same_wav_inference", "audio_sha256": digest},
    }


def test_missing_opt_in_fails_before_reading_files(tmp_path):
    with pytest.raises(ValueError, match="--allow-machine-labels"):
        candidate.run(tmp_path / "missing.json", tmp_path / "candidate.onnx")


def test_candidate_path_refuses_production_and_existing_outputs(tmp_path):
    production = tmp_path / "production"
    production.mkdir()
    with pytest.raises(ValueError, match="production model directory"):
        candidate.review.candidate_output(production / "candidate.onnx", production_root=production)
    target = tmp_path / "candidate.onnx"
    target.write_bytes(b"existing")
    with pytest.raises(ValueError, match="already exists"):
        candidate.review.candidate_output(target, production_root=production)


@pytest.mark.parametrize("text", ["Alex", "Alexandra", "alexa2", "myalexa"])
def test_positive_requires_exact_whole_token(text):
    row = {"label": "positive", "label_source": "machine_consensus", "human_reviewed": False,
           "production_eligible": False, "truncated": False, "evidence": _evidence(text)}
    with pytest.raises(ValueError, match="exact Alexa token"):
        candidate.validate_positive_row(row)


def test_same_wav_offline_evidence_is_mandatory():
    row = {"label": "positive", "label_source": "machine_consensus", "human_reviewed": False,
           "production_eligible": False, "truncated": False, "evidence": _evidence()}
    row["evidence"]["whisper"]["audio_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="exact WAV"):
        candidate.validate_positive_row(row)


def test_transcript_length_and_boolean_epoch_values_are_rejected(tmp_path):
    row = {"label": "positive", "label_source": "machine_consensus", "human_reviewed": False,
           "production_eligible": False, "truncated": False, "evidence": _evidence("Alexa" * 81)}
    with pytest.raises(ValueError, match="400-character"):
        candidate.validate_positive_row(row)
    with pytest.raises(ValueError, match="epochs"):
        candidate.run("unused", tmp_path / "unused.onnx", allow_machine_labels=True, epochs=True)
    row["evidence"]["whisper"]["text"] = "Alexa"
    row["evidence"]["moonshine"]["text"] = "Alexa"
    row["evidence"]["whisper"]["audio_sha256"] = "a" * 64
    row["evidence"]["moonshine"]["source"] = "logged_transcript"
    with pytest.raises(ValueError, match="exact WAV"):
        candidate.validate_positive_row(row)


def test_archived_positive_hash_and_frame_metadata_are_rechecked(tmp_path):
    audio_root = tmp_path / "archive"
    audio_root.mkdir()
    path = audio_root / "clip.wav"
    pcm = np.zeros(16000, dtype="<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(pcm.tobytes())
    digest = candidate.sha256_file(path)
    session = "capture-group-one"
    row = {"event_id": "event-1", "path": str(path), "wav_sha256": digest,
           "n_frames": 16000, "sample_rate": 16000, "capture_session_id": session,
           "session_id": session, "group_id": session, "split": candidate.review.session_split(session),
           "label": "positive", "label_source": "machine_consensus", "human_reviewed": False,
           "production_eligible": False, "truncated": False, "evidence": _evidence()}
    for evidence in row["evidence"].values():
        evidence["audio_sha256"] = digest
    validated = candidate.validate_positive_row(row, audio_root=audio_root)
    assert validated["identity"]["wav_sha256"] == digest
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(np.ones(16000, dtype="<i2").tobytes())
    with pytest.raises(ValueError, match="identity mismatch"):
        candidate.validate_positive_row(row, audio_root=audio_root)


def test_truncation_and_asr_agreement_never_become_human_review():
    row = {"label": "positive", "label_source": "machine_consensus", "human_reviewed": False,
           "production_eligible": False, "truncated": True, "evidence": _evidence()}
    with pytest.raises(ValueError, match="untruncated"):
        candidate.validate_positive_row(row)
    row["truncated"] = False
    row["human_reviewed"] = True
    with pytest.raises(ValueError, match="non-human"):
        candidate.validate_positive_row(row)
    row["human_reviewed"] = False
    row["evidence"]["whisper"]["text_truncated"] = True
    with pytest.raises(ValueError, match="transcript evidence is truncated"):
        candidate.validate_positive_row(row)


def test_report_extra_cannot_override_provenance_or_safety_fields(tmp_path):
    with pytest.raises(ValueError, match="reserved or unknown"):
        candidate.fit_and_export([], [], tmp_path / "candidate.onnx",
                                 report_extra={"production_eligible": True})


def test_bad_label_is_rejected_but_unsure_is_skipped():
    manifest = {"schema": candidate.LABEL_SCHEMA, "human_reviewed": False,
                "production_eligible": False, "records": [{"label": "unsure"}]}
    with pytest.raises(ValueError, match="need at least 3 positive"):
        candidate.select_training_rows(manifest, [])
    manifest["records"] = [{"label": "Alex"}]
    with pytest.raises(ValueError, match="label must"):
        candidate.select_training_rows(manifest, [])


def test_window_selection_uses_consumed_sample_bounds_not_rf_end():
    ends = [32000 + i * 1280 for i in range(1, 40)]
    kept = candidate.retained_window_indices(ends, 16000)
    assert kept
    assert all(32000 + 2560 <= ends[i] <= 32000 + 16000 + 10240 for i in kept)
    assert ends[kept[0]] >= 32000 + 2560
    assert ends[kept[-1]] <= 32000 + 16000 + 10240


def test_streaming_frontend_receives_exact_hops_and_returns_bounded_bag():
    class FakeFeatures:
        def __init__(self):
            self.inputs = []
            self.counter = 0

        def reset(self):
            self.inputs.clear()
            self.counter = 0

        def __call__(self, samples):
            assert samples.shape == (1280,)
            self.inputs.append(samples.copy())
            self.counter += 1

        def get_features(self, count):
            assert count == 16
            return np.full((1, 16, 96), self.counter, dtype=np.float32)

    frontend = FakeFeatures()
    samples = np.arange(16000, dtype=np.int16)
    bag = candidate.extract_bag(samples, feature_factory=lambda: frontend)
    assert len(frontend.inputs) == (32000 + 16000 + 12800 + 1279) // 1280
    assert np.array_equal(frontend.inputs[0], np.zeros(1280, dtype=np.int16))
    assert bag.dtype == np.float32 and bag.shape[1:] == (16, 96)
    assert len(bag) <= candidate.MAX_BAG_WINDOWS


def test_publish_refuses_overwrite_and_cleans_sidecar_after_export_failure(tmp_path):
    temp_model = tmp_path / "temporary.onnx"
    target = tmp_path / "candidate.onnx"
    temp_model.write_bytes(b"new model")
    target.write_bytes(b"keep existing")
    with pytest.raises(FileExistsError):
        candidate._publish_artifacts(temp_model, target, {"source": "test"})
    assert target.read_bytes() == b"keep existing"
    assert not target.with_suffix(".meta.json").exists()


def test_source_race_fails_before_any_artifact_is_published(tmp_path):
    temp_model = tmp_path / "temporary.onnx"
    target = tmp_path / "candidate.onnx"
    temp_model.write_bytes(b"validated model")

    def changed_source():
        raise ValueError("training source changed while fitting")

    with pytest.raises(ValueError, match="source changed"):
        candidate._publish_artifacts(temp_model, target, {"source": "test"}, before_publish=changed_source)
    assert not target.exists()
    assert not target.with_suffix(".meta.json").exists()


def test_low_memory_gate_runs_before_feature_or_export_imports(monkeypatch, tmp_path):
    labels = tmp_path / "labels.json"
    labels.write_text("{}", encoding="utf-8")
    store = tmp_path / "human_reviews.json"
    store.write_text("{}", encoding="utf-8")
    stock = tmp_path / "stock.onnx"
    stock.write_bytes(b"stock")
    frontend = tmp_path / "frontend"
    frontend.mkdir()
    (frontend / "melspectrogram.onnx").write_bytes(b"mel")
    (frontend / "embedding_model.onnx").write_bytes(b"embed")
    monkeypatch.setattr(candidate, "DEFAULT_STORE", store)
    monkeypatch.setattr(candidate, "STOCK_MODEL", stock)
    monkeypatch.setattr(candidate, "FRONTEND_DIR", frontend)
    monkeypatch.setattr(candidate, "_bounded_json", lambda path: ({"bad": True}, "labels-hash"))
    monkeypatch.setattr(candidate.review, "load_preparation_store", lambda path: ({"schema": "x"}, "store-hash"))
    monkeypatch.setattr(candidate.review, "review_history", lambda store: ([], [], []))
    monkeypatch.setattr(candidate, "select_training_rows", lambda manifest, negatives: ([], [], {}))
    def hashes(paths):
        return {str(candidate.Path(path).resolve()): ("labels-hash" if candidate.Path(path).resolve() == labels.resolve()
                else "store-hash" if candidate.Path(path).resolve() == store.resolve() else "model-hash") for path in paths}
    monkeypatch.setattr(candidate, "_check_sources", hashes)
    with pytest.raises(RuntimeError, match="1536 MiB"):
        candidate.run(labels, tmp_path / "candidate.onnx", allow_machine_labels=True,
                      memory_gate=lambda: candidate.MIN_FREE_BYTES - 1)


def test_parse_to_fingerprint_race_aborts_before_gate_or_output(monkeypatch, tmp_path):
    labels = tmp_path / "labels.json"
    labels.write_text("{}", encoding="utf-8")
    original_hash = candidate.sha256_file(labels)
    store = tmp_path / "human_reviews.json"
    store.write_text("{}", encoding="utf-8")
    stock = tmp_path / "stock.onnx"
    stock.write_bytes(b"stock")
    frontend = tmp_path / "frontend"
    frontend.mkdir()
    (frontend / "melspectrogram.onnx").write_bytes(b"mel")
    (frontend / "embedding_model.onnx").write_bytes(b"embed")
    monkeypatch.setattr(candidate, "DEFAULT_STORE", store)
    monkeypatch.setattr(candidate, "STOCK_MODEL", stock)
    monkeypatch.setattr(candidate, "FRONTEND_DIR", frontend)
    original_bounded = candidate._bounded_json
    def change_labels_after_parse(path):
        parsed, digest = original_bounded(path)
        return parsed, digest
    monkeypatch.setattr(candidate, "_bounded_json", change_labels_after_parse)
    original_loader = candidate.review.load_preparation_store
    def mutate_after_parse(path):
        parsed, digest = original_loader(path)
        labels.write_text('{"changed":true}', encoding="utf-8")
        return parsed, digest
    monkeypatch.setattr(candidate.review, "load_preparation_store", mutate_after_parse)
    monkeypatch.setattr(candidate.review, "review_history", lambda parsed: ([], [], []))
    monkeypatch.setattr(candidate, "select_training_rows", lambda parsed, rows: ([], [], {}))
    gate_called = []
    output = tmp_path / "candidate.onnx"
    with pytest.raises(ValueError, match="changed after parsing"):
        candidate.run(labels, output, allow_machine_labels=True,
                      memory_gate=lambda: gate_called.append(True) or candidate.MIN_FREE_BYTES)
    assert candidate.sha256_file(labels) != original_hash
    assert not gate_called
    assert not output.exists()
    assert not output.with_suffix(".meta.json").exists()


def test_holdout_records_are_not_selected_for_extraction(monkeypatch, tmp_path):
    manifest = {"schema": candidate.LABEL_SCHEMA, "human_reviewed": False,
                "production_eligible": False, "records": [{"label": "positive"}] * 5}
    rows = [
        {"label": "positive", "split": "train", "wav_sha256": "1" * 64, "group_id": "g1"},
        {"label": "positive", "split": "train", "wav_sha256": "1" * 64, "group_id": "g1"},
        {"label": "positive", "split": "train", "wav_sha256": "2" * 64, "group_id": "g1"},
        {"label": "positive", "split": "train", "wav_sha256": "3" * 64, "group_id": "g2"},
        {"label": "positive", "split": "holdout", "wav_sha256": "4" * 64, "group_id": "holdout"},
    ]
    source = tmp_path / "source.wav"
    source.write_bytes(b"wav")
    for row in rows:
        row.update(path=source, n_frames=16000)
    monkeypatch.setattr(candidate, "validate_positive_row", lambda row: rows.pop(0))
    negatives = [{"split": "train", "presence": "negative", "wav_sha256": f"{i:064x}",
                  "group_id": "legacy", "path": __file__, "n_frames": 1} for i in range(5)]
    positives, train_neg, counts = candidate.select_training_rows(manifest, negatives)
    assert len(positives) == 3
    assert all(isinstance(row["path"], candidate.Path) for row in positives)
    assert all(isinstance(row["path"], candidate.Path) for row in train_neg)
    assert all(row["split"] == "train" for row in positives)
    assert counts["positive_holdout_excluded"] == 1
    assert counts["train_positive_duplicate_records_excluded"] == 1


def test_holdout_hash_cannot_alias_reviewed_negative_training_audio(monkeypatch, tmp_path):
    manifest = {"schema": candidate.LABEL_SCHEMA, "human_reviewed": False,
                "production_eligible": False, "records": [{"label": "positive"}]}
    shared = "f" * 64
    positive = {"label": "positive", "split": "holdout", "wav_sha256": shared,
                "group_id": "holdout-group", "path": tmp_path / "positive.wav", "n_frames": 16000}
    positive["path"].write_bytes(b"same source")
    monkeypatch.setattr(candidate, "validate_positive_row", lambda row: positive)
    negative_path = tmp_path / "negative.wav"
    negative_path.write_bytes(b"same source")
    negatives = [{"split": "train", "presence": "negative", "wav_sha256": shared,
                  "group_id": "negative-group", "path": str(negative_path), "n_frames": 16000}]
    with pytest.raises(ValueError, match="crosses label, capture group or split"):
        candidate.select_training_rows(manifest, negatives)
