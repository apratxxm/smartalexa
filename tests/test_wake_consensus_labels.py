import hashlib
import importlib
import sys
import subprocess
import wave
from pathlib import Path

import pytest


MODULE = "assistant.experimental.wake_train.consensus_labels"


def _wav(path, seconds=0.1):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(16000)
        out.writeframes(b"\0\0" * int(16000 * seconds))
    return path


@pytest.fixture
def mod():
    return importlib.import_module(MODULE)


def _row(event_id="evt", digest="a" * 64):
    return {"event_id": event_id, "wav_sha256": digest, "n_frames": 1600,
            "duration_seconds": 0.1, "path": "x.wav", "sample_rate": 16000,
            "capture_session_id": "capture", "session_id": "capture",
            "group_id": "capture", "split": "train", "truncated": False,
            "selection_reason": "accepted_text_hint", "logged_alexa_hint": True}


def _teacher(name, text, *, truncated=False, exact=None, digest="a" * 64):
    exact = ("Alexa" in text) if exact is None else exact
    return {"stage": name, "model": {"repository": name, "revision": "pinned"},
            "records": [{"event_id": "evt", "wav_sha256": digest, "n_frames": 1600,
                         "text": text, "exact_alexa": exact, "text_truncated": truncated,
                         "source": "offline_same_wav_inference", "elapsed_seconds": 0.01}]}


def test_cold_import_does_not_load_teacher_packages():
    code = ("import importlib,sys; importlib.import_module(%r); "
            "assert not any(n in sys.modules for n in "
            "('faster_whisper','moonshine_onnx','onnxruntime'))") % MODULE
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("texts,flags,truncated,label", [
    (("Alexa, lights", "Alexa lights"), (True, True), (False, False), "positive"),
    (("Alexis, lights", "Alexa lights"), (False, True), (False, False), "unsure"),
    (("Alexa command", "no wake word"), (True, False), (False, False), "unsure"),
])
def test_consensus_requires_same_wav_exact_token_and_untruncated(mod, texts, flags, truncated, label):
    moon, whisper = _teacher("moonshine", texts[0], exact=flags[0], truncated=truncated[0]), _teacher(
        "whisper", texts[1], exact=flags[1], truncated=truncated[1])
    report = mod.merge_consensus({"records": [_row()], "total_original_audio_seconds": 0.1,
        "selection_counts": {}, "human_review_store": {}}, moon, whisper)
    row = report["records"][0]
    assert row["label"] == label
    assert row["human_reviewed"] is False and row["production_eligible"] is False
    assert set(row["evidence"]) == {"moonshine", "whisper"}
    assert all(e["family"] == name for name, e in row["evidence"].items())


def test_exact_consensus_is_unsure_when_either_transcript_truncated(mod):
    moon = _teacher("moonshine", "Alexa, lights", exact=True, truncated=True)
    whisper = _teacher("whisper", "Alexa lights", exact=True)
    report = mod.merge_consensus({"records": [_row()], "total_original_audio_seconds": 0.1,
        "selection_counts": {}, "human_review_store": {}}, moon, whisper)
    assert report["records"][0]["label"] == "unsure"


def test_composite_logged_hint_is_not_teacher_evidence(mod):
    assert mod.exact_alexa("Alexa | command")
    report = mod.merge_consensus({"records": [_row()], "total_original_audio_seconds": 0.1,
        "selection_counts": {}, "human_review_store": {}}, _teacher("moonshine", "command", exact=False),
        _teacher("whisper", "command", exact=False))
    assert report["records"][0]["label"] == "unsure"


def test_same_wav_hash_and_duplicate_event_ids_required(mod):
    selection = {"records": [_row()], "total_original_audio_seconds": 0.1,
                 "selection_counts": {}, "human_review_store": {}}
    with pytest.raises(ValueError, match="same selected WAV"):
        mod.merge_consensus(selection, _teacher("moonshine", "Alexa", digest="b" * 64),
                            _teacher("whisper", "Alexa"))
    repeated = _teacher("moonshine", "Alexa")
    repeated["records"].append(dict(repeated["records"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        mod.merge_consensus(selection, repeated, _teacher("whisper", "Alexa"))


def test_trigger_audio_falls_back_when_missing_or_processed(mod, tmp_path):
    audio_root = tmp_path / "archive"
    good = _wav(audio_root / "raw.wav")
    processed = _wav(audio_root / "processed.wav")
    event = {"trigger_audio_path": str(audio_root / "missing.wav"), "audio_path": str(good),
             "processed_audio_path": str(processed)}
    field, value = mod._source_value(event, audio_root=audio_root, project_root=tmp_path)
    assert field == "audio_path" and Path(value) == good
    event["trigger_audio_path"] = str(processed)
    assert mod._source_value(event, audio_root=audio_root, project_root=tmp_path) == ("audio_path", str(good))


def test_changed_source_wav_is_rejected(mod, tmp_path):
    path = _wav(tmp_path / "clip.wav")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    _wav(path, seconds=0.2)
    with pytest.raises(ValueError, match="changed"):
        mod._hash_matches(path, digest, 1600)


def test_stage_payload_limits_and_unique_event_ids(mod):
    with pytest.raises(ValueError, match="duplicate"):
        mod._validate_stage_payload({"records": [_row(), _row()]})
    too_long = _row()
    too_long["n_frames"] = 24 * 16000 + 1
    too_long["duration_seconds"] = too_long["n_frames"] / 16000
    with pytest.raises(ValueError, match="duration"):
        mod._validate_stage_payload({"records": [too_long]})
    inconsistent = _row()
    inconsistent["duration_seconds"] = 24.1
    with pytest.raises(ValueError, match="disagrees"):
        mod._validate_stage_payload({"records": [inconsistent]})
    rows = [_row(str(index)) for index in range(25)]
    with pytest.raises(ValueError, match="record count"):
        mod._validate_stage_payload({"records": rows})


def test_invalid_early_event_does_not_hide_valid_same_group_clip(mod, tmp_path):
    audio_root = tmp_path / "archive"
    good = _wav(audio_root / "good.wav")
    now = 5000.0
    events = [{"ts": now - index, "event_id": f"bad-{index}", "trigger_source": "wake",
        "outcome": "ok", "transcript": "Alexa lights", "capture_session_id": "same",
        "audio_path": str(audio_root / f"missing-{index}.wav")} for index in range(16)]
    events.append({"ts": now - 20, "event_id": "valid", "trigger_source": "wake",
        "outcome": "ok", "transcript": "Alexa lights", "capture_session_id": "same",
        "audio_path": str(good)})
    events.append({"ts": now - 21, "event_id": "unknown", "trigger_source": "wake",
        "outcome": "ok", "transcript": "Alexa lights", "audio_path": str(good)})
    history = {"reviewed_hashes": set(), "exists": False, "sha256": None,
        "metadata": [], "unavailable": []}
    selection = mod.build_selection(events=events, now=now, audio_root=audio_root,
        project_root=tmp_path, history=history)
    assert [row["event_id"] for row in selection["records"]] == ["valid"]
    assert selection["selection_counts"]["unknown_session"] == 1


def test_insufficient_ram_fails_before_teacher_loader(mod, monkeypatch, tmp_path):
    weight = tmp_path / "model.bin"
    weight.write_bytes(b"model")
    monkeypatch.setattr(mod, "_model_files", lambda stage: (tmp_path, {"model.bin": (5, "x")}, {}))
    monkeypatch.setattr(mod, "_validate_model_files", lambda stage: (tmp_path, {"weight_bytes": 5}))
    called = []
    with pytest.raises(MemoryError):
        mod.run_teacher_stage("whisper", {"records": [], "project_root": str(tmp_path),
            "audio_root": str(tmp_path)}, memory_gate=lambda: 1,
            loader=lambda *_: called.append(True))
    assert called == []


def test_stage_runner_is_serial_and_stops_after_first_failure(mod):
    calls = []
    def fail_first(stage, records):
        calls.append(stage)
        raise RuntimeError("first stage failed")
    with pytest.raises(RuntimeError, match="first stage"):
        mod.run_teacher_stages([_row()], temp_parent=".", project_root=".", audio_root=".",
                               stage_runner=fail_first)
    assert calls == ["moonshine"]


def test_teacher_stages_run_in_order(mod):
    calls = []
    def success(stage, records):
        calls.append(stage)
        return {"stage": stage, "records": []}
    moonshine, whisper = mod.run_teacher_stages([], temp_parent=".", project_root=".",
        audio_root=".", stage_runner=success)
    assert calls == ["moonshine", "whisper"]
    assert moonshine["stage"] == "moonshine" and whisper["stage"] == "whisper"


def test_output_refuses_existing_and_outside_paths(mod, tmp_path, monkeypatch):
    root = tmp_path / "proposals"
    monkeypatch.setattr(mod, "OUTPUT_ROOT", root)
    assert mod._safe_output(root / "new.json") == root / "new.json"
    existing = root / "exists.json"
    existing.parent.mkdir(parents=True)
    existing.write_text("{}")
    with pytest.raises(ValueError, match="exists"):
        mod._safe_output(existing)
    with pytest.raises(ValueError, match="under"):
        mod._safe_output(tmp_path / "outside.json")


def test_model_hash_verification_detects_mutation(mod, monkeypatch, tmp_path):
    weight = tmp_path / "model.bin"
    weight.write_bytes(b"original")
    monkeypatch.setattr(mod, "_model_files", lambda stage: (tmp_path, {"model.bin": (8, hashlib.sha256(b"original").hexdigest())}, {}))
    mod._validate_model_unchanged("whisper", {})
    weight.write_bytes(b"modified")
    with pytest.raises(ValueError, match="changed during inference"):
        mod._validate_model_unchanged("whisper", {})
