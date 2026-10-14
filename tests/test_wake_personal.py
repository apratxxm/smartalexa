"""The personal model proposes wakes without driving the stock tiers."""
import sys
import types
from collections import deque
from pathlib import Path

import numpy as np
import pytest

from assistant.config import Config


class FakeModel:
    def __init__(self, wakeword_models, inference_framework, **kwargs):
        self.models = {Path(path).stem: None for path in wakeword_models}
        self._score_names = {
            key: key.removesuffix("_v0.1") for key in self.models}
        self.wakeword_models = wakeword_models
        self.inference_framework = inference_framework
        self.kwargs = kwargs
        self.queue = []

    def predict(self, chunk):
        scores = self.queue.pop(0)
        return {key: scores.get(name, 0.0) for key, name in self._score_names.items()}

    def reset(self):
        pass


def _detector(monkeypatch, tmp_path, patience=2):
    model_file = tmp_path / "alexa_user.onnx"
    model_file.write_bytes(b"x")
    stock_metadata_path = tmp_path / "alexa_v0.1.tflite"
    feature_metadata_paths = {
        "melspectrogram": tmp_path / "melspectrogram.tflite",
        "embedding": tmp_path / "embedding_model.tflite",
    }
    for path in (tmp_path / "alexa_v0.1.onnx",
                 tmp_path / "melspectrogram.onnx",
                 tmp_path / "embedding_model.onnx"):
        path.write_bytes(b"onnx fixture")
    fake_oww = types.ModuleType("openwakeword")
    fake_oww.MODELS = {"alexa": {"model_path": str(stock_metadata_path)}}
    fake_oww.FEATURE_MODELS = {
        name: {"model_path": str(path)} for name, path in feature_metadata_paths.items()}
    fake_oww.utils = types.SimpleNamespace(
        download_models=lambda *_args, **_kwargs: pytest.fail("runtime download attempted"))
    fake_model_mod = types.ModuleType("openwakeword.model")
    fake_model_mod.Model = FakeModel
    monkeypatch.setitem(sys.modules, "openwakeword", fake_oww)
    monkeypatch.setitem(sys.modules, "openwakeword.model", fake_model_mod)
    from assistant.wake import WakeDetector
    return WakeDetector(Config({"wake": {
        "model": ["alexa"], "personal_model": str(model_file),
        "personal_threshold": 0.5, "personal_patience": patience,
        "threshold_low": 0.15, "threshold_high": 0.6}}))


def _feed(det, scores):
    det._model.queue = [dict(s) for s in scores]
    hits, tiers = [], []
    for _ in scores:
        tier = None
        for _ in range(3):  # 3 x 512 >= one 1280 chunk
            tier = det.process(np.zeros(512, np.int16)) or tier
            if det.personal_hit:
                hits.append(det.personal_score)
            if not det._model.queue:
                break
        tiers.append(tier)
    return hits, tiers


def test_personal_hit_needs_patience_and_fires_once_per_run(monkeypatch, tmp_path):
    det = _detector(monkeypatch, tmp_path)
    hits, tiers = _feed(det, [{"alexa": 0.0, "alexa_user": s} for s in (0.9, 0.2, 0.7, 0.8, 0.9, 0.95)])
    assert hits == [0.8]
    assert all(t is None for t in tiers)  # personal scores never set stock tiers


def test_stock_tier_ignores_personal_scores(monkeypatch, tmp_path):
    det = _detector(monkeypatch, tmp_path, patience=1)
    _, tiers = _feed(det, [{"alexa": 0.7, "alexa_user": 0.0}])
    assert tiers[-1] == "both" and det.last_model == "alexa"


@pytest.mark.parametrize("buffer_size", [400, 1000, 160000])
@pytest.mark.parametrize("n_samples", [0, 1280, 2560, 160000])
def test_recent_melspectrogram_preserves_exact_input_and_output(buffer_size, n_samples):
    from assistant.wake import _stream_recent_melspectrogram
    raw = deque(range(buffer_size), maxlen=160000)
    expected_input = list(raw)[-n_samples-480:]
    seen = []
    def mel(audio):
        seen.append(audio)
        return np.array([[len(audio), sum(audio)], [1, 2]])
    features = types.SimpleNamespace(raw_data_buffer=raw, _get_melspectrogram=mel,
                                    melspectrogram_buffer=np.array([[5, 6], [7, 8]]),
                                    melspectrogram_max_len=3)
    _stream_recent_melspectrogram(features, n_samples)
    assert seen == [expected_input]
    np.testing.assert_array_equal(features.melspectrogram_buffer,
                                  [[7, 8], [len(expected_input), sum(expected_input)], [1, 2]])


def test_recent_melspectrogram_keeps_short_input_guard():
    from assistant.wake import _stream_recent_melspectrogram
    with pytest.raises(ValueError, match="400"):
        _stream_recent_melspectrogram(types.SimpleNamespace(raw_data_buffer=deque(range(399))), 1280)


# Review workflow tests use tiny local PCM fixtures, never real inference/audio devices.
def _review_wav(path, *, frames=16000, channels=1, rate=16000, width=2, value=1):
    import wave
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setframerate(rate)
        wav.setsampwidth(width)
        wav.writeframes(bytes([value, 0]) * (frames * channels * width // 2))
    return path


def _review_row(path, *, presence="positive", session="capture-one", event="event-one", **changes):
    from assistant.experimental.wake_train import active_learning as review
    row = {"path": str(path), **review.wav_identity(path), "event_id": event,
           "audio_field": "audio_path", "session_id": session,
           "source_provenance": "human listened to original archived capture",
           "reviewer": "fixture-human", "human_reviewed": True,
           "reviewed_at": "2026-10-06T01:00:00+00:00", "presence": presence,
           "addressed_assistant": "unsure", "wake_start_s": None, "wake_end_s": None,
           "entire_clip_negative": presence == "negative"}
    if presence == "positive":
        row.update(wake_start_s=0.1, wake_end_s=0.4)
    return {**row, **changes}


def _review_store(*rows):
    from assistant.experimental.wake_train import active_learning as review
    return {"schema": review.STORE_SCHEMA, "reviews": list(rows)}


def test_wake_review_default_never_derives_labels_or_writes(tmp_path, capsys):
    import json
    from assistant.experimental.wake_train import active_learning as review, build_manifest
    wav = _review_wav(tmp_path / "audio.wav")
    telemetry = tmp_path / "telemetry.jsonl"
    events = [{"event_id": "stt", "trigger_source": "stt", "transcript": "Alexa play",
               "audio_path": str(wav)},
              {"event_id": "hotkey", "trigger_source": "hotkey", "audio_path": str(wav)},
              {"event_id": "high-score", "outcome": "false_accept", "wake_scores": {"alexa": .99},
               "audio_path": str(wav), "follow_up_audio_path": str(wav), "follow_up_transcript": ""},
              {"event_id": "missing", "audio_path": str(tmp_path / "pruned.wav")}]
    telemetry.write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")
    before = sorted(p.name for p in tmp_path.iterdir())
    assert build_manifest.main(["--telemetry", str(telemetry), "--audio-root", str(tmp_path)]) == 0
    candidates = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(candidates) == 5
    assert {row["proposed_label"] for row in candidates} == {"uncertain"}
    assert candidates[-1]["unavailable"]
    assert candidates[2]["scores"] == {"alexa": .99}
    assert sorted(p.name for p in tmp_path.iterdir()) == before


def test_wake_review_lists_suppressed_unknown_decision_without_accepting_it_as_audio(tmp_path, capsys):
    import json
    from assistant.experimental.wake_train import active_learning as review
    telemetry = tmp_path / "telemetry.jsonl"
    telemetry.write_text(json.dumps({"event_id": "suppressed", "trigger_source": "near_miss",
                                    "candidate_kind": "below_vad", "audio_path": "",
                                    "audio_status": "suppressed", "audio_reason": "rate_limited",
                                    "capture_session_id": "capture-run-one", "segment_id": "segment-one"}), encoding="utf-8")
    review.print_candidates(telemetry, tmp_path)
    row = json.loads(capsys.readouterr().out)
    assert row["path"] == "" and row["unavailable"] == "rate_limited"
    assert row["proposed_label"] == "uncertain" and row["capture_session_id"] == "capture-run-one"
    with pytest.raises(ValueError, match="missing.*pruned"):
        review.label_candidate(telemetry=telemetry, store_path=tmp_path / "review.json",
                               event_id="suppressed", audio_field="audio_path", presence="positive",
                               session_id="capture-run-one", source_provenance="fixture", reviewer="human",
                               confirm_reviewed=True, wake_start=.1, wake_end=.4, audio_root=tmp_path)
    assert not (tmp_path / "review.json").exists()


@pytest.mark.parametrize("start,end", [(None, .3), (.1, None), (float("nan"), .3),
                                         (.1, float("inf")), (-.1, .3), (.4, .4),
                                         (.5, .4), (.1, 1.1), (.1000001, .4)])
def test_wake_review_positive_span_refuses_invalid_bounds(tmp_path, start, end):
    from assistant.experimental.wake_train import active_learning as review
    wav = _review_wav(tmp_path / "audio.wav")
    with pytest.raises(ValueError, match="span"):
        review.validate_review(_review_row(wav, wake_start_s=start, wake_end_s=end), tmp_path)


@pytest.mark.parametrize("change,match", [({"human_reviewed": False}, "human"),
                                           ({"session_id": ""}, "session_id"),
                                           ({"source_provenance": ""}, "source_provenance"),
                                           ({"wav_sha256": "0" * 64}, "hash mismatch"),
                                           ({"n_frames": 10000}, "frame metadata"),
                                           ({"addressed_assistant": "inferred"}, "addressed_assistant")])
def test_wake_review_refuses_unreviewed_or_stale_metadata(tmp_path, change, match):
    from assistant.experimental.wake_train import active_learning as review
    wav = _review_wav(tmp_path / "audio.wav")
    with pytest.raises(ValueError, match=match):
        review.validate_review(_review_row(wav, **change), tmp_path)


def test_wake_review_span_must_fit_declared_frame_count(tmp_path):
    from assistant.experimental.wake_train import active_learning as review
    wav = _review_wav(tmp_path / "audio.wav")
    with pytest.raises(ValueError, match="span"):
        review.validate_review(_review_row(wav, n_frames=100,
                                           wake_start_s=.1, wake_end_s=.4), tmp_path)


@pytest.mark.parametrize("kwargs", [{"channels": 2}, {"rate": 44100}, {"width": 1}, {"frames": 0}])
def test_wake_review_requires_original_mono_pcm16(tmp_path, kwargs):
    from assistant.experimental.wake_train import active_learning as review
    with pytest.raises(ValueError, match="PCM16|empty"):
        review.wav_identity(_review_wav(tmp_path / "audio.wav", **kwargs))


def test_wake_review_archive_bounds_pruning_and_truncation(tmp_path):
    from assistant.experimental.wake_train import active_learning as review
    root = tmp_path / "archive"
    root.mkdir()
    outside = _review_wav(tmp_path / "outside.wav")
    with pytest.raises(ValueError, match="archival"):
        review.validate_review(_review_row(outside), root)
    wav = _review_wav(root / "audio.wav")
    row = _review_row(wav)
    wav.unlink()
    with pytest.raises(FileNotFoundError):
        review.validate_review(row, root)
    wav = _review_wav(root / "audio.wav")
    wav.write_bytes(wav.read_bytes()[:-2])
    with pytest.raises(ValueError, match="truncated"):
        review.wav_identity(wav)


def test_wake_review_requires_entire_clip_negative(tmp_path):
    from assistant.experimental.wake_train import active_learning as review
    wav = _review_wav(tmp_path / "audio.wav")
    with pytest.raises(ValueError, match="entire-clip"):
        review.validate_review(_review_row(wav, presence="negative", entire_clip_negative=False), tmp_path)
    with pytest.raises(ValueError, match="only valid"):
        review.validate_review(_review_row(wav, presence="negative", wake_start_s=.1), tmp_path)
    positive = review.validate_review(_review_row(wav, addressed_assistant="no"), tmp_path)
    assert positive["presence"] == "positive" and positive["addressed_assistant"] == "no"


def _legacy_negative_row(path, *, event="legacy-event", **changes):
    from assistant.experimental.wake_train import active_learning as review
    values = {"presence": "negative", "session": None, "event": event,
              "session_provenance": review.LEGACY_NEGATIVE_SESSION_PROVENANCE,
              "review_note": "listened to the whole clip; no Alexa spoken",
              "addressed_assistant": "no", "entire_clip_negative": True,
              "capture_session_id": "", "segment_id": "", "overlap_of": ""}
    values.update(changes)
    return _review_row(path, **values)


def test_wake_review_unknown_legacy_negative_mode_is_train_only_and_not_a_session(tmp_path):
    from assistant.experimental.wake_train import active_learning as review
    wav = _review_wav(tmp_path / "legacy.wav")
    row = _legacy_negative_row(wav, split="holdout", group_id="caller-invented")
    validated = review.validated_reviews(_review_store(row), tmp_path)
    assert len(validated) == 1
    assert validated[0]["session_id"] is None
    assert validated[0]["group_id"] == review.LEGACY_NEGATIVE_POOL_ID
    assert validated[0]["split"] == "train"
    report = review.readiness(validated)
    assert report["train_only_legacy_pools"] == [review.LEGACY_NEGATIVE_POOL_ID]
    assert report["train_only_legacy_pool_counts"] == {review.LEGACY_NEGATIVE_POOL_ID: 1}
    assert report["session_groups"]["train"]["negative"] == []
    assert not report["data_ready"] and not report["training_ready"]
    assert any("missing train positive" in reason for reason in report["blocked_reasons"])


@pytest.mark.parametrize("change,match", [
    ({"session_id": "guessed-session"}, "cannot assert"),
    ({"capture_session_id": "real-capture"}, "missing capture metadata"),
    ({"segment_id": "segment"}, "missing capture metadata"),
    ({"overlap_of": "parent"}, "missing capture metadata"),
    ({"presence": "positive", "wake_start_s": .1, "wake_end_s": .4}, "limited to confirmed"),
    ({"presence": "unsure"}, "limited to confirmed"),
    ({"entire_clip_negative": False}, "limited to confirmed"),
    ({"review_note": "  "}, "review_note"),
])
def test_wake_review_unknown_legacy_mode_rejects_unproven_conditions(tmp_path, change, match):
    from assistant.experimental.wake_train import active_learning as review
    row = _legacy_negative_row(_review_wav(tmp_path / "legacy.wav"), **change)
    with pytest.raises(ValueError, match=match):
        review.validate_review(row, tmp_path)


@pytest.mark.parametrize("intent", ["yes", "no", "unsure"])
def test_wake_review_legacy_presence_does_not_infer_addressing_intent(tmp_path, intent):
    from assistant.experimental.wake_train import active_learning as review
    row = _legacy_negative_row(_review_wav(tmp_path / f"legacy-{intent}.wav"), addressed_assistant=intent)
    assert review.validate_review(row, tmp_path)["addressed_assistant"] == intent


def test_wake_review_legacy_capture_missing_metadata_is_normalized(tmp_path):
    from assistant.experimental.wake_train import active_learning as review
    row = _legacy_negative_row(_review_wav(tmp_path / "legacy-none.wav"),
                               capture_session_id=None, segment_id=None, overlap_of=None)
    validated = review.validate_review(row, tmp_path)
    assert (validated["capture_session_id"], validated["segment_id"], validated["overlap_of"]) == ("", "", "")


def test_wake_review_legacy_pool_deduplicates_and_conflicts_with_real_groups(tmp_path):
    import shutil
    from assistant.experimental.wake_train import active_learning as review
    wav = _review_wav(tmp_path / "legacy.wav")
    copy = tmp_path / "legacy-copy.wav"
    shutil.copyfile(wav, copy)
    rows = review.validated_reviews(_review_store(
        _legacy_negative_row(wav), _legacy_negative_row(copy, event="legacy-copy")), tmp_path)
    assert len(rows) == 1
    assert rows[0]["group_id"] == review.LEGACY_NEGATIVE_POOL_ID
    with pytest.raises(ValueError, match="conflicting sessions/groups"):
        review.validated_reviews(_review_store(
            _legacy_negative_row(wav), _review_row(copy, presence="negative", event="real-session",
                                                    entire_clip_negative=True)), tmp_path)


def test_wake_review_unknown_legacy_mode_keeps_archive_hash_frame_and_pcm_guards(tmp_path):
    from assistant.experimental.wake_train import active_learning as review
    wav = _review_wav(tmp_path / "legacy.wav")
    row = _legacy_negative_row(wav)
    for changed, error in ((dict(row, wav_sha256="0" * 64), "hash mismatch"),
                           (dict(row, n_frames=2), "frame metadata")):
        with pytest.raises(ValueError, match=error):
            review.validate_review(changed, tmp_path)
    bad = _review_wav(tmp_path / "bad.wav", rate=44100)
    with pytest.raises(ValueError, match="PCM16"):
        review.validate_review(_legacy_negative_row(bad), tmp_path)
    outside_root = tmp_path / "archive"
    outside_root.mkdir()
    with pytest.raises(ValueError, match="archival"):
        review.validate_review(row, outside_root)


def test_wake_review_recovery_is_atomic_idempotent_and_preserves_existing_rows(tmp_path):
    import json
    from assistant.experimental.wake_train import active_learning as review
    first = _review_wav(tmp_path / "first.wav", value=3)
    second = _review_wav(tmp_path / "second.wav", value=4)
    telemetry = tmp_path / "events.jsonl"
    telemetry.write_text("\n".join(json.dumps({"event_id": event, "audio_path": str(path)})
                                    for event, path in (("one", first), ("two", second))), encoding="utf-8")
    store = tmp_path / "reviews.json"
    base = dict(telemetry=telemetry, store_path=store, reviewer="human",
                source_provenance="user-confirmed full-clip listening review",
                confirm_reviewed=True, entire_clip_negative=True,
                addressed_assistant="no", audio_root=tmp_path)
    with pytest.raises(ValueError, match="candidate is missing"):
        review.recover_confirmed_negatives([
            {"event_id": "one", "audio_field": "audio_path", "note": "whole clip; no Alexa"},
            {"event_id": "missing", "audio_field": "audio_path", "note": "whole clip; no Alexa"}], **base)
    assert not store.exists() and not list(tmp_path.glob("*.lock")) and not list(tmp_path.glob("*.tmp"))

    conflicting = _review_row(_review_wav(tmp_path / "conflict.wav", value=8), event="one")
    store.write_text(json.dumps(_review_store(conflicting)), encoding="utf-8")
    conflict_bytes = store.read_bytes()
    with pytest.raises(ValueError, match="existing review conflicts"):
        review.recover_confirmed_negatives(
            [{"event_id": "one", "audio_field": "audio_path", "note": "whole clip; no Alexa"}], **base)
    assert store.read_bytes() == conflict_bytes and not list(tmp_path.glob("*.lock"))
    real_group_copy = _review_row(first, presence="negative", event="one", session="real-session",
                                  entire_clip_negative=True, addressed_assistant="no")
    store.write_text(json.dumps(_review_store(real_group_copy)), encoding="utf-8")
    group_conflict_bytes = store.read_bytes()
    with pytest.raises(ValueError, match="existing review conflicts"):
        review.recover_confirmed_negatives(
            [{"event_id": "one", "audio_field": "audio_path", "note": "whole clip; no Alexa"}], **base)
    assert store.read_bytes() == group_conflict_bytes and not list(tmp_path.glob("*.lock"))
    store.unlink()

    existing = _review_row(_review_wav(tmp_path / "existing.wav", value=5), presence="negative", event="preexisting", session="real-session",
                           entire_clip_negative=True, addressed_assistant="no")
    store.write_text(json.dumps(_review_store(existing)), encoding="utf-8")
    before = review.load_store(store)
    feedback = [{"event_id": "one", "audio_field": "audio_path", "note": "whole clip; no Alexa"},
                {"event_id": "two", "audio_field": "audio_path", "note": "whole clip; no Alexa"}]
    result = review.recover_confirmed_negatives(feedback, **base)
    saved = review.load_store(store)
    assert result["added"] == 2 and result["stored"] == 3
    assert result["unique_negatives"] == 3
    assert result["train_only_legacy_pool_unique_negatives"] == 2
    assert result["training_started"] is False and result["model_trained"] is False
    assert len(saved["reviews"]) == len(before["reviews"]) + 2
    assert saved["reviews"][0] == before["reviews"][0]
    assert all(row["session_id"] is None and row["presence"] == "negative"
               for row in saved["reviews"][1:])
    again = review.recover_confirmed_negatives(feedback, **base)
    assert again["added"] == 0 and again["stored"] == 3
    assert review.load_store(store) == saved


@pytest.mark.parametrize("confirm,entire", [(False, True), (True, False)])
def test_wake_review_recovery_requires_both_confirmations(tmp_path, confirm, entire):
    from assistant.experimental.wake_train import active_learning as review
    with pytest.raises(ValueError, match="explicit human review"):
        review.recover_confirmed_negatives([{"event_id": "x", "audio_field": "audio_path", "note": "no Alexa"}],
            telemetry=tmp_path / "missing", store_path=tmp_path / "store.json", reviewer="human",
            source_provenance="confirmed", confirm_reviewed=confirm,
            entire_clip_negative=entire, audio_root=tmp_path)


def test_wake_review_recovery_does_not_infer_addressing_intent(tmp_path):
    import json
    from assistant.experimental.wake_train import active_learning as review
    wav = _review_wav(tmp_path / "audio.wav")
    telemetry = tmp_path / "events.jsonl"
    telemetry.write_text(json.dumps({"event_id": "one", "audio_path": str(wav)}), encoding="utf-8")
    store = tmp_path / "reviews.json"
    review.recover_confirmed_negatives(
        [{"event_id": "one", "audio_field": "audio_path", "note": "whole clip reviewed"}],
        telemetry=telemetry, store_path=store, reviewer="human", source_provenance="user review",
        confirm_reviewed=True, entire_clip_negative=True, audio_root=tmp_path)
    assert review.load_store(store)["reviews"][0]["addressed_assistant"] == "unsure"


def test_wake_review_dedup_conflicts_and_event_session_grouping(tmp_path):
    import shutil
    from assistant.experimental.wake_train import active_learning as review
    wav = _review_wav(tmp_path / "audio.wav")
    row = _review_row(wav)
    rows = review.validated_reviews(_review_store(row, dict(row)), tmp_path)
    assert len(rows) == 1
    copy = tmp_path / "copy.wav"
    shutil.copyfile(wav, copy)
    with pytest.raises(ValueError, match="hash.*conflicting sessions"):
        review.validated_reviews(_review_store(row, _review_row(copy, event="other", session="capture-two")), tmp_path)
    with pytest.raises(ValueError, match="positive/negative"):
        review.validated_reviews(_review_store(row, _review_row(wav, presence="negative")), tmp_path)
    with pytest.raises(ValueError, match="overlapping"):
        review.validated_reviews(_review_store(row, _review_row(wav, wake_start_s=.2, wake_end_s=.5)), tmp_path)
    other = _review_wav(tmp_path / "other.wav", value=2)
    with pytest.raises(ValueError, match="same event"):
        review.validated_reviews(_review_store(row, _review_row(other, session="capture-two")), tmp_path)
    # Disjoint spans remain in one session partition; caller split/group cannot cross it.
    rows = review.validated_reviews(_review_store(row, _review_row(wav, wake_start_s=.5, wake_end_s=.7,
                                                                 split="holdout", group_id="fake")), tmp_path)
    assert len(rows) == 2
    assert {r["group_id"] for r in rows} == {"capture-one"}
    assert {r["split"] for r in rows} == {review.session_split("capture-one")}


def test_wake_review_readiness_requires_independent_classes_and_excludes_unknowns(tmp_path):
    from assistant.experimental.wake_train import active_learning as review
    sessions = {side: next(f"session-{i}" for i in range(100) if review.session_split(f"session-{i}") == side)
                for side in ("train", "holdout")}
    rows = []
    for i, (side, presence) in enumerate((s, p) for s in sessions for p in ("positive", "negative")):
        wav = _review_wav(tmp_path / f"audio-{i}.wav", value=i + 1)
        rows.append(_review_row(wav, presence=presence, session=sessions[side], event=f"event-{i}"))
    unknown = _review_row(_review_wav(tmp_path / "unknown.wav", value=9), presence="unsure", event="unknown")
    manifest = review.build_reviewed_manifest(_review_store(*rows, unknown), tmp_path)
    assert len(manifest["positives"]) == 2 and len(manifest["negatives_short"]) == 2
    assert manifest["negatives_long"] == []
    assert manifest["readiness"]["data_ready"] is True
    assert manifest["readiness"]["training_ready"] is False
    assert manifest["readiness"]["model_trained"] is False
    assert manifest["readiness"]["excluded_unsure"] == 1
    assert len(manifest["review_store_sha256"]) == 64
    single_session = review.build_reviewed_manifest(_review_store(*rows[:2]), tmp_path)
    assert not single_session["readiness"]["data_ready"]
    assert any("missing" in reason for reason in single_session["readiness"]["blocked_reasons"])


def test_wake_review_label_command_atomic_store_and_no_write_on_conflict(tmp_path):
    import json
    from assistant.experimental.wake_train import active_learning as review
    wav = _review_wav(tmp_path / "audio.wav")
    telemetry = tmp_path / "events.jsonl"
    telemetry.write_text(json.dumps({"event_id": "one", "audio_path": str(wav)}), encoding="utf-8")
    store = tmp_path / "reviews.json"
    kwargs = dict(telemetry=telemetry, store_path=store, event_id="one", audio_field="audio_path",
                  presence="positive", session_id="capture-one", reviewer="human",
                  source_provenance="listened to original audio", wake_start=.1, wake_end=.4,
                  audio_root=tmp_path)
    with pytest.raises(ValueError, match="confirm-reviewed"):
        review.label_candidate(**kwargs)
    assert not store.exists()
    review.label_candidate(**kwargs, confirm_reviewed=True)
    before = store.read_bytes()
    assert review.load_store(store)["reviews"][0]["human_reviewed"] is True
    with pytest.raises(ValueError, match="positive/negative"):
        review.label_candidate(**{**kwargs, "presence": "negative", "wake_start": None,
                                   "wake_end": None, "entire_clip_negative": True}, confirm_reviewed=True)
    assert store.read_bytes() == before
    assert not list(tmp_path.glob("*.lock")) and not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("name", ["manifest.json", "split_manifest.json", "split_manifest_v1.json", "enroll_split.json"])
def test_wake_review_preserves_legacy_outputs(tmp_path, name):
    from assistant.experimental.wake_train import active_learning as review
    output = tmp_path / name
    output.write_bytes(b"legacy-fixed-content")
    with pytest.raises(ValueError, match="legacy"):
        review.write_new_manifest(output, {})
    assert output.read_bytes() == b"legacy-fixed-content"


def test_wake_review_exports_revalidate_and_do_not_overwrite(tmp_path, capsys):
    import hashlib
    import json
    from assistant.experimental.wake_train import active_learning as review, build_manifest, prepare_clips
    wav = _review_wav(tmp_path / "audio.wav")
    store = tmp_path / "reviews.json"
    store.write_text(json.dumps(_review_store(_review_row(wav))), encoding="utf-8")
    output = tmp_path / "reviewed.json"
    assert build_manifest.main(["--labels", str(store), "--output", str(output), "--audio-root", str(tmp_path)]) == 0
    final = tmp_path / "reviewed-split.json"
    original = json.loads(output.read_text(encoding="utf-8"))
    assert prepare_clips.main(["--manifest", str(output), "--output", str(final), "--audio-root", str(tmp_path)]) == 0
    partitioned = json.loads(final.read_text(encoding="utf-8"))
    assert partitioned["review_store_sha256"] == original["review_store_sha256"]
    assert partitioned["parent_manifest_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert partitioned["derived_reviews_sha256"] != original["review_store_sha256"]
    exported = partitioned["positives"][0]
    assert exported["wake_start_sample"] == 1600 and exported["wake_end_sample"] == 6400
    assert exported["source_provenance"] and exported["wav_sha256"]
    with pytest.raises(ValueError, match="exists"):
        review.write_new_manifest(final, {})
    _review_wav(wav, value=4)
    with pytest.raises(SystemExit):
        prepare_clips.main(["--manifest", str(output), "--output", str(tmp_path / "stale.json"), "--audio-root", str(tmp_path)])
    assert "hash mismatch" in capsys.readouterr().err
    assert not (tmp_path / "stale.json").exists()


def test_wake_review_candidate_exports_never_touch_production_or_existing_files(tmp_path):
    from assistant.experimental.wake_train import active_learning as review
    production = tmp_path / "production"
    production.mkdir()
    live = production / "alexa_user.onnx"
    live.write_bytes(b"live")
    for value in (None, live, production / "new.onnx"):
        with pytest.raises(ValueError, match="required|production"):
            review.candidate_output(value, production)
    experiment = tmp_path / "experiment.onnx"
    experiment.write_bytes(b"prior")
    with pytest.raises(ValueError, match="exists"):
        review.candidate_output(experiment, production)
    assert review.candidate_output(tmp_path / "new.onnx", production) == tmp_path / "new.onnx"
    with pytest.raises(ValueError, match="not wired"):
        review.require_supported_training("test")
    assert live.read_bytes() == b"live" and experiment.read_bytes() == b"prior"


def test_wake_review_candidate_aliases_and_existing_metadata(tmp_path):
    import os
    from assistant.experimental.wake_train import active_learning as review
    production = tmp_path / "production"
    production.mkdir()
    live = production / "alexa_user.onnx"
    live.write_bytes(b"live")
    alias = tmp_path / "hardlink.onnx"
    os.link(live, alias)
    with pytest.raises(ValueError, match="exists"):
        review.candidate_output(alias, production)
    candidate = tmp_path / "new.onnx"
    candidate.with_suffix(".meta.json").write_text("prior-metadata", encoding="utf-8")
    with pytest.raises(ValueError, match="metadata"):
        review.candidate_output(candidate, production)
    assert live.read_bytes() == b"live"


@pytest.mark.parametrize("script", ["extract_features.py", "train_classifier.py"])
def test_wake_review_direct_main_is_guarded_without_loading_ml(tmp_path, script):
    import ast
    from pathlib import Path
    from assistant.experimental.wake_train import active_learning as review
    source = Path(__file__).resolve().parents[1] / "assistant/experimental/wake_train" / script
    tree = ast.parse(source.read_text(encoding="utf-8"))
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    isolated = ast.Module(body=[main], type_ignores=[])
    namespace = {"candidate_output": review.candidate_output,
                 "require_supported_training": review.require_supported_training,
                 "OUT_ONNX": str(tmp_path / "candidate.onnx")}
    exec(compile(isolated, str(source), "exec"), namespace)
    # Actual main body must refuse before resolving ML globals or loading features.
    with pytest.raises(ValueError, match="not wired"):
        namespace["main"]()
    assert not (tmp_path / "candidate.onnx").exists()


@pytest.mark.parametrize("script,arguments,expected", [("extract_features.py", [], "feature extraction blocked"),
                                                        ("train_classifier.py", [], "--out"),
                                                        ("train_classifier.py", ["--out", "candidate.onnx"], "classifier training blocked")])
def test_wake_review_legacy_training_cli_fails_before_model_dependencies(tmp_path, script, arguments, expected):
    import subprocess
    from pathlib import Path
    scripts = Path(__file__).resolve().parents[1] / "assistant/experimental/wake_train"
    # Poison ML imports through isolated cwd/PYTHONPATH to prove gate ordering.
    poison = tmp_path / "torch.py"
    poison.write_text("raise RuntimeError('ML import reached')", encoding="utf-8")
    import os
    env = dict(os.environ, PYTHONPATH=str(tmp_path))
    result = subprocess.run([sys.executable, str(scripts / script), *arguments], cwd=tmp_path,
                            env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode != 0
    assert expected in result.stderr
    assert "ML import reached" not in result.stderr
    assert not (tmp_path / "candidate.onnx").exists()


def _preparation_store(tmp_path, *, unknown=True):
    from assistant.experimental.wake_train import active_learning as review
    sessions = {side: next(f"prepared-session-{i}" for i in range(100)
                           if review.session_split(f"prepared-session-{i}") == side)
                for side in ("train", "holdout")}
    rows = []
    for i, (side, presence) in enumerate((s, p) for s in sessions for p in ("positive", "negative")):
        wav = _review_wav(tmp_path / f"prepared-{i}.wav", value=i + 1)
        rows.append(_review_row(wav, presence=presence, session=sessions[side], event=f"prepared-{i}",
                                capture_session_id=f"real-capture-{side}", segment_id=f"segment-{i}"))
    if unknown:
        rows.append(_review_row(_review_wav(tmp_path / "unknown.wav", value=9), presence="unsure", event="unknown"))
    return _review_store(*rows)


def test_wake_training_preparation_is_exact_bounded_session_split_not_training(tmp_path):
    from assistant.experimental.wake_train import active_learning as review
    store = _preparation_store(tmp_path)
    before = sorted(tmp_path.iterdir())
    prepared = review.prepare_training(store, tmp_path)
    assert prepared["eligible_for_feature_preparation"] is True
    assert prepared["readiness"]["data_ready"] is True
    assert not prepared["training_ready"] and not prepared["feature_extractor_implemented"]
    assert not prepared["feature_extraction_started"] and not prepared["training_started"] and not prepared["model_trained"]
    assert prepared["excluded_unsure"] == 1 and len(prepared["rows"]) == 4
    assert prepared["unique_audio_seconds"] == 5
    assert len(prepared["review_store_sha256"]) == 64
    assert "upstream VAD" in prepared["limitation"] and "not base wake-detector retraining" in prepared["limitation"]
    for row in prepared["rows"]:
        if row["presence"] == "positive":
            assert (row["source_start_sample"], row["source_end_sample"]) == (1600, 6400)
        else:
            assert (row["source_start_sample"], row["source_end_sample"]) == (0, 16000)
            assert row["entire_clip_negative"]
        assert row["group_id"] == row["session_id"]
        assert row["split"] == review.session_split(row["session_id"])
        assert row["window_upper_bound"] > 0 and row["human_reviewed"]
    assert prepared["window_upper_bound"] == sum(row["window_upper_bound"] for row in prepared["rows"])
    assert sorted(tmp_path.iterdir()) == before


def test_wake_training_preparation_counts_legacy_pool_only_as_train_negative(tmp_path):
    from assistant.experimental.wake_train import active_learning as review
    ordinary = _preparation_store(tmp_path, unknown=False)
    legacy = _legacy_negative_row(_review_wav(tmp_path / "legacy-prepared.wav", value=20), event="legacy-prepared")
    ordinary["reviews"] = [ordinary["reviews"][0], ordinary["reviews"][2],
                           ordinary["reviews"][3], legacy]
    prepared = review.prepare_training(ordinary, tmp_path)
    assert prepared["readiness"]["train_only_legacy_pools"] == [review.LEGACY_NEGATIVE_POOL_ID]
    assert prepared["readiness"]["train_only_legacy_pool_counts"] == {review.LEGACY_NEGATIVE_POOL_ID: 1}
    pool_rows = [row for row in prepared["rows"] if row["group_id"] == review.LEGACY_NEGATIVE_POOL_ID]
    assert len(pool_rows) == 1 and pool_rows[0]["presence"] == "negative" and pool_rows[0]["split"] == "train"
    assert pool_rows[0]["session_id"] is None
    assert prepared["readiness"]["session_groups"]["holdout"]["negative"]
    assert prepared["training_ready"] is False and prepared["model_trained"] is False


def test_wake_training_preparation_never_uses_window_budget_as_ground_truth(tmp_path):
    from assistant.experimental.wake_train import active_learning as review
    store = _preparation_store(tmp_path, unknown=False)
    for kwargs, reason in (({"max_windows": 1}, "window budget"), ({"max_total_seconds": 1}, "total-duration")):
        prepared = review.prepare_training(store, tmp_path, **kwargs)
        assert not prepared["eligible_for_feature_preparation"]
        assert any(reason in message for message in prepared["blocked_reasons"])
        assert not prepared["training_ready"]
    empty = review.prepare_training(_review_store(), tmp_path)
    assert not empty["eligible_for_feature_preparation"] and empty["rows"] == []
    assert any("missing train positive" in message for message in empty["blocked_reasons"])
    only_unknown = review.prepare_training(_review_store(_review_row(
        _review_wav(tmp_path / "only-unknown.wav", value=20), presence="unsure")), tmp_path)
    assert not only_unknown["eligible_for_feature_preparation"] and only_unknown["rows"] == []


@pytest.mark.parametrize("kwargs", [{"max_reviews": True}, {"max_reviews": 101}, {"max_reviews": 1.5},
                                   {"max_windows": 8193}, {"max_windows": 0},
                                   {"max_total_seconds": float("inf")}, {"max_total_seconds": 601}])
def test_wake_training_preparation_limits_cannot_be_unbounded(tmp_path, kwargs):
    from assistant.experimental.wake_train import active_learning as review
    with pytest.raises(ValueError, match="limit"):
        review.prepare_training(_review_store(), tmp_path, **kwargs)


def test_wake_training_preparation_refuses_count_size_duration_before_hashing(tmp_path, monkeypatch):
    from assistant.experimental.wake_train import active_learning as review
    store = _preparation_store(tmp_path, unknown=False)
    monkeypatch.setattr(review, "wav_identity", lambda _: pytest.fail("hash reached before bounded metadata/file gate"))
    with pytest.raises(ValueError, match="count"):
        review.prepare_training(store, tmp_path, max_reviews=1)
    oversized = dict(store["reviews"][0], n_frames=24 * 16000 + 1)
    with pytest.raises(ValueError, match="duration"):
        review.prepare_training(_review_store(oversized), tmp_path)
    monkeypatch.setattr(review, "MAX_PREPARATION_WAV_BYTES", 20)
    with pytest.raises(ValueError, match="byte limit"):
        review.prepare_training(_review_store(store["reviews"][0]), tmp_path)


@pytest.mark.parametrize("provenance", ["capture", "segment", "overlap", "pruned-overlap-parent"])
def test_wake_training_preparation_refuses_capture_segment_overlap_leakage(tmp_path, provenance):
    from assistant.experimental.wake_train import active_learning as review
    store = _preparation_store(tmp_path, unknown=False)
    train, holdout = store["reviews"][0], store["reviews"][2]
    assert review.session_split(train["session_id"]) == "train"
    assert review.session_split(holdout["session_id"]) == "holdout"
    if provenance == "capture":
        holdout["capture_session_id"] = train["capture_session_id"]
    elif provenance == "segment":
        holdout["segment_id"] = train["segment_id"]
    elif provenance == "overlap":
        holdout["overlap_of"] = train["segment_id"]
    else:
        train["overlap_of"] = holdout["overlap_of"] = "missing-parent-segment"
    with pytest.raises(ValueError, match="conflicting human session"):
        review.prepare_training(store, tmp_path)


def test_wake_training_preparation_revalidates_hash_spans_and_human_negatives(tmp_path):
    from assistant.experimental.wake_train import active_learning as review
    store = _preparation_store(tmp_path, unknown=False)
    original = store["reviews"][0]
    for changed, error in ((dict(original, wake_start_s=.00001), "span"),
                           (dict(original, wav_sha256="0" * 64), "hash mismatch"),
                           (dict(store["reviews"][1], entire_clip_negative=False), "entire-clip")):
        with pytest.raises(ValueError, match=error):
            review.prepare_training(_review_store(changed), tmp_path)
    copied = dict(original, event_id="different-event", session_id="different-human-session")
    with pytest.raises(ValueError, match="duplicated WAV hash"):
        review.prepare_training(_review_store(original, copied), tmp_path)
    with pytest.raises(ValueError, match="human review store"):
        review.prepare_training({"schema": review.MANIFEST_SCHEMA, "positives": [original]}, tmp_path)


def test_wake_training_preparation_cli_missing_store_reports_unknown_without_writes(tmp_path, capsys):
    import json
    from assistant.experimental.wake_train import active_learning as review
    missing = tmp_path / "missing-reviews.json"
    assert review.main(["--audio-root", str(tmp_path), "prepare-training", "--store", str(missing)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert not report["human_review_store_exists"] and report["input_file_sha256"] is None
    assert not report["eligible_for_feature_preparation"] and not report["training_started"]
    assert not list(tmp_path.iterdir())


def test_wake_training_preparation_cli_preserves_input_fingerprint_no_inference(tmp_path, capsys, monkeypatch):
    import builtins
    import hashlib
    import json
    from assistant.experimental.wake_train import active_learning as review
    store = tmp_path / "human-reviews.json"
    store.write_text(json.dumps(_preparation_store(tmp_path)), encoding="utf-8")
    before = sorted(tmp_path.iterdir())
    original_import = builtins.__import__
    def no_inference(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "onnxruntime", "openwakeword", "wakelib"}:
            pytest.fail("preparation imported inference/training dependencies")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", no_inference)
    assert review.main(["--audio-root", str(tmp_path), "prepare-training", "--store", str(store)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["human_review_store_exists"] and report["input_file_sha256"] == hashlib.sha256(store.read_bytes()).hexdigest()
    assert not report["training_started"] and report["eligible_for_feature_preparation"]
    assert sorted(tmp_path.iterdir()) == before


def test_wake_training_preparation_store_read_is_bounded(tmp_path, monkeypatch):
    from assistant.experimental.wake_train import active_learning as review
    store = tmp_path / "human-reviews.json"
    store.write_bytes(b" " * 21)
    monkeypatch.setattr(review, "MAX_PREPARATION_STORE_BYTES", 20)
    with pytest.raises(ValueError, match="byte limit"):
        review.load_preparation_store(store)
