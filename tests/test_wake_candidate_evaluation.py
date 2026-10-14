from __future__ import annotations

import hashlib
import json
import sys
import wave

import numpy as np
import pytest

from assistant.experimental.wake_train import candidate_evaluation as evaluation


def test_overlapping_windows_cannot_match_two_spans():
    spans = [(1.0, 1.4), (1.1, 1.5)]
    events = [{"index": 10, "time_s": 1.3}, {"index": 11, "time_s": 1.4}]
    matches, misses, unmatched = evaluation.match_events(events, spans, 0.64)
    assert len(matches) == 2
    assert len(misses) == 0
    assert unmatched == []
    one, two, extra = evaluation.match_events(events[:1], spans, 0.64)
    assert len(one) == 1 and len(two) == 1 and extra == []


def test_single_activation_never_counts_for_two_overlapping_spans():
    matches, misses, unmatched = evaluation.match_events([{"index": 5, "time_s": 1.3}],
                                                         [(1.0, 1.4), (1.1, 1.5)], 0.64)
    assert len(matches) == 1
    assert len(misses) == 1
    assert unmatched == []


def test_event_clock_is_second_confirming_chunk_and_once_per_run():
    events = evaluation.confirmed_events([0.1, 0.8, 0.9, 0.95, 0.2, 0.7, 0.8],
                                         [-1.0, -0.92, -0.84, -0.76, -0.68, -0.60, -0.52])
    assert [(e["run_start_index"], e["index"], e["time_s"]) for e in events] == [
        (1, 2, -0.84), (5, 6, -0.52)]


def test_refractory_uses_fixed_two_second_confirmation_clock_and_drops_negative():
    events = [{"time_s": -0.1}, {"time_s": 0.0}, {"time_s": 1.99},
              {"time_s": 2.0}, {"time_s": 4.01}]
    assert [e["time_s"] for e in evaluation.apply_refractory(events)] == [0.0, 2.0, 4.01]


@pytest.mark.parametrize("scores", [np.ones((4,)), np.ones((4, 2)), np.ones((3, 1))])
def test_classifier_shape_must_be_exact(scores):
    with pytest.raises(ValueError, match="output shape"):
        evaluation.validate_scores(scores, 4)


@pytest.mark.parametrize("value", [np.nan, np.inf, -0.01, 1.01])
def test_classifier_scores_must_be_finite_probabilities(value):
    with pytest.raises(ValueError, match="finite probabilities"):
        evaluation.validate_scores(np.array([[0.2], [value]], dtype=np.float32), 2)


def test_candidate_metadata_rejects_production_and_heldout_hash_leakage(tmp_path):
    candidate = tmp_path / "new.onnx"
    candidate.write_bytes(b"isolated-candidate")
    model_hash = evaluation.sha256_file(candidate)
    meta = candidate.with_suffix(".meta.json")
    common = {"schema": "memoir.wake-candidate-training-meta.v1", "production_eligible": False,
              "training_labels": {"positive_source": "machine_consensus"},
              "deployment_changed": False, "output_sha256": model_hash}
    meta.write_text(json.dumps({**common,
                                "source_sha256": {"renamed-source.wav": "a" * 64}}), encoding="utf-8")
    with pytest.raises(ValueError, match="leakage"):
        evaluation.load_candidate_metadata(candidate, "a" * 64, model_hash)
    meta.write_text(json.dumps({**common, "production_eligible": True, "source_sha256": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="production_eligible=false"):
        evaluation.load_candidate_metadata(candidate, "a" * 64, model_hash)
    meta.write_text(json.dumps({**common, "source_sha256": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="model hash"):
        evaluation.load_candidate_metadata(candidate, "a" * 64, "c" * 64)


def test_no_overwrite_atomic_json_create(tmp_path):
    output = tmp_path / "report.json"
    output.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        evaluation.atomic_json_create(output, {"new": True})
    assert output.read_text(encoding="utf-8") == "keep"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["report.json"]


def test_module_import_is_cold_and_does_not_import_ml_stack():
    assert "onnxruntime" not in sys.modules
    # NumPy may be imported by pytest itself; importing this module does not
    # eagerly require AudioFeatures, ONNX Runtime, or Torch.
    assert "openwakeword.utils" not in sys.modules
    assert "torch" not in sys.modules


def test_memory_gate_precedes_frontend_factory_and_ml_scoring(monkeypatch, tmp_path):
    # Build tiny valid inputs; all expensive stages are replaced after proving
    # the physical-RAM gate runs first.
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    paths = {name: models_dir / f"{name}.onnx" for name in ("stock", "current", "candidate")}
    for path in paths.values():
        path.write_bytes(b"onnx")
    candidate = paths["candidate"]
    candidate.with_suffix(".meta.json").write_text(json.dumps({
        "schema": "memoir.wake-candidate-training-meta.v1", "production_eligible": False,
        "training_labels": {"positive_source": "machine_consensus"}, "deployment_changed": False,
        "output_sha256": evaluation.sha256_file(candidate), "source_sha256": {"train.wav": "b" * 64}}),
        encoding="utf-8")
    front = tmp_path / "front"
    front.mkdir()
    mel, emb = front / "melspectrogram.onnx", front / "embedding_model.onnx"
    mel.write_bytes(b"mel")
    emb.write_bytes(b"emb")
    wav_path, meta_path = tmp_path / "hold.wav", tmp_path / "hold.json"
    with wave.open(str(wav_path), "wb") as wav:
        wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(16000)
        wav.writeframes(np.zeros(40 * 16000 + 1600, dtype="<i2").tobytes())
    spans = [[i * 1.5 + 0.1, i * 1.5 + 0.2] for i in range(19)]
    meta_path.write_text(json.dumps({"alexa_spans_s": spans, "sample_rate": 16000,
                                    "duration_s": 40.1, "filename": wav_path.name}), encoding="utf-8")
    monkeypatch.setattr(evaluation, "STOCK_MODEL", paths["stock"])
    monkeypatch.setattr(evaluation, "CURRENT_MODEL", paths["current"])
    monkeypatch.setattr(evaluation, "FRONTEND_DIR", front)
    monkeypatch.setattr(evaluation, "HELDOUT_WAV", wav_path)
    monkeypatch.setattr(evaluation, "HELDOUT_META", meta_path)
    output = tmp_path / "report.json"
    with pytest.raises(RuntimeError, match="1536 MiB"):
        evaluation.run(candidate, output, memory_gate=lambda: evaluation.MIN_FREE_BYTES - 1,
                       feature_factory=lambda: pytest.fail("feature construction ran before RAM gate"))
    assert not output.exists()


def test_production_path_guard_rejects_nested_candidate_and_output(tmp_path):
    production = (evaluation.Path(__file__).parents[1] / "assistant" / "models" / "wake").resolve()
    candidate = production / "experimental" / "new.onnx"
    output = tmp_path / "report.json"
    with pytest.raises(ValueError, match="candidate must be outside"):
        evaluation._ensure_safe_paths(candidate, output)
    candidate = tmp_path / "candidate.onnx"
    with pytest.raises(ValueError, match="output must be outside"):
        evaluation._ensure_safe_paths(candidate, production / "reports" / "new.json")


def test_safe_path_guard_protects_candidate_sidecar_and_legacy_report(tmp_path):
    candidate = tmp_path / "candidate.onnx"
    with pytest.raises(ValueError, match="may not overwrite"):
        evaluation._ensure_safe_paths(candidate, candidate.with_suffix(".meta.json"))
    legacy = tmp_path / "alexa_user_stream_test.json"
    with pytest.raises(ValueError, match="may not overwrite"):
        evaluation._ensure_safe_paths(candidate, legacy, [legacy])


def test_parsed_holdout_hashes_must_match_initial_fingerprint(tmp_path):
    wav_path, meta_path = tmp_path / "hold.wav", tmp_path / "hold.json"
    wav_path.write_bytes(b"wav bytes")
    meta_path.write_bytes(b"meta bytes")
    before = evaluation._file_fingerprints([wav_path, meta_path])
    good_wav, good_meta = evaluation.sha256_file(wav_path), evaluation.sha256_file(meta_path)
    evaluation._verify_parsed_holdout(before, wav_path, meta_path, good_wav, good_meta)
    with pytest.raises(ValueError, match="changed between parsing"):
        evaluation._verify_parsed_holdout(before, wav_path, meta_path, "f" * 64, good_meta)


def test_feature_chunk_clock_uses_consumed_samples_minus_prefix():
    class FakeFeatures:
        def __init__(self): self.n = 0
        def reset(self): self.n = 0
        def __call__(self, chunk):
            assert chunk.shape == (evaluation.HOP,)
            self.n += 1
        def get_features(self, count):
            assert count == 16
            return np.full((1, 16, 96), self.n, dtype=np.float32)
    pcm = np.zeros(16000, dtype="<i2").tobytes()
    windows, times = evaluation.extract_features(pcm, 16000, feature_factory=FakeFeatures)
    assert windows.shape[1:] == (16, 96)
    assert times[0] == evaluation.HOP / evaluation.SAMPLE_RATE - 2.0
    assert times[-1] == (len(windows) * evaluation.HOP - evaluation.PREFIX) / evaluation.SAMPLE_RATE


def test_holdout_spans_must_be_ordered_nonoverlapping_and_inside_wav(tmp_path):
    wav_path, meta_path = tmp_path / "hold.wav", tmp_path / "hold.json"
    with wave.open(str(wav_path), "wb") as wav:
        wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(16000)
        wav.writeframes(np.zeros(16000, dtype="<i2").tobytes())
    spans = [[i * 0.01, i * 0.01 + 0.02] for i in range(19)]
    meta_path.write_text(json.dumps({"alexa_spans_s": spans}), encoding="utf-8")
    with pytest.raises(ValueError, match="sorted and non-overlapping"):
        evaluation.load_holdout(wav_path, meta_path)
    spans = [[i * 0.1, i * 0.1 + 0.05] for i in range(19)]
    spans[-1] = [2.0, 2.1]
    meta_path.write_text(json.dumps({"alexa_spans_s": spans}), encoding="utf-8")
    with pytest.raises(ValueError, match="exceeds actual WAV duration"):
        evaluation.load_holdout(wav_path, meta_path)


def test_legacy_rounded_duration_and_actual_metadata_keys_are_checked(tmp_path):
    wav_path, meta_path = tmp_path / "alexas_with_music_20261002.wav", tmp_path / "hold.json"
    with wave.open(str(wav_path), "wb") as wav:
        wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(16000)
        wav.writeframes(np.zeros(642389, dtype="<i2").tobytes())
    spans = [[i * 1.5 + 0.1, i * 1.5 + 0.2] for i in range(19)]
    base = {"alexa_spans_s": spans, "file": wav_path.name, "rate": 16000, "duration_s": 40.1}
    meta_path.write_text(json.dumps(base), encoding="utf-8")
    _, frames, _, _, _ = evaluation.load_holdout(wav_path, meta_path)
    assert frames == 642389
    assert frames / evaluation.SAMPLE_RATE != base["duration_s"]
    for update, message in (({"duration_s": 40.2}, "duration_s"),
                            ({"file": "other.wav"}, "file"),
                            ({"rate": 8000}, "rate")):
        meta_path.write_text(json.dumps({**base, **update}), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            evaluation.load_holdout(wav_path, meta_path)
