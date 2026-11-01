import hashlib
import io
import json
from pathlib import Path
import wave

import pytest

from assistant.experimental.wake_train import collection
from assistant.config import Config
from assistant.telemetry import Telemetry as RealTelemetry
from assistant.experimental.wake_train import active_learning


class Clock:
    value = 1000.0

    def __call__(self):
        return self.value


class Telemetry:
    def __init__(self, *, save=True, index=True):
        self.save = save
        self.index = index
        self.events = []

    def save_audio(self, event, pcm, subdir=""):
        self.events.append((event, pcm.copy(), subdir))
        if self.save:
            event.audio_status = "saved"
            event.audio_n_frames = len(pcm)
            event.audio_sha256 = hashlib.sha256(b"stored WAV identity").hexdigest()
            return "private/path.wav"
        event.audio_status = "failed"
        return ""

    def write(self, event):
        return self.index


def wav(frames=4000, *, channels=1, width=2, rate=16000, compression="NONE"):
    stream = io.BytesIO()
    with wave.open(stream, "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(width)
        output.setframerate(rate)
        if compression != "NONE":
            output.setcomptype(compression, "compressed")
        output.writeframes(b"\0" * frames * channels * width)
    return stream.getvalue()


@pytest.fixture
def manager(monkeypatch):
    clock = Clock()
    leases = {}

    def create(_directory, capture_id, *, now=None, seconds=30):
        leases[capture_id] = {"expires": now + seconds, "ack": False, "released": False}

    def ack(_directory, capture_id, *, now=None):
        item = leases.get(capture_id)
        return bool(item and item["ack"] and now < item["expires"] and not item["released"])

    def release(_directory, capture_id):
        item = leases.get(capture_id)
        if item and not item["released"]:
            item["released"] = True
            return True
        return False

    monkeypatch.setattr(collection.runtime, "create_collection_pause", create)
    monkeypatch.setattr(collection.runtime, "collection_pause_acknowledged", ack)
    monkeypatch.setattr(collection.runtime, "release_collection_pause", release)
    telemetry = Telemetry()
    result = collection.CollectionManager(telemetry, "leases", clock=clock)
    result._test_leases = leases
    result._test_clock = clock
    return result


def armed(manager):
    session = manager.create_session("quiet")
    capture = manager.arm(session["session_id"])
    manager._test_leases[capture["capture_id"]]["ack"] = True
    return session, capture


def test_capture_requires_ack_and_is_consumed_once(manager):
    session = manager.create_session("quiet")
    capture = manager.arm(session["session_id"])
    assert manager.ready(capture["capture_id"]) == {"ready": False, "expired": False}
    manager._test_leases[capture["capture_id"]]["ack"] = True
    assert manager.ready(capture["capture_id"])["ready"] is True
    result = manager.submit(capture["capture_id"], wav())
    assert result["saved"] is True and result["indexed"] is True
    assert result["review_state"] == "unknown"
    assert "path" not in result and "audio" not in result
    event = manager.telemetry.events[0][0]
    assert (event.trigger_source, event.candidate_kind, event.outcome) == (
        "wake_collection", "intentional_wake", "pending_review")
    assert event.capture_session_id == session["session_id"]
    assert event.segment_id == capture["capture_id"]
    assert event.input_device == "browser_microphone"
    assert event.end_reason == "explicit_user_collection"
    assert event.review_state == "unknown" and event.transcript == ""
    assert event.wake_score is None and event.relative_start_sample is None
    with pytest.raises(ValueError, match="unknown|replayed"):
        manager.submit(capture["capture_id"], wav())
    assert manager._test_leases[capture["capture_id"]]["released"]


@pytest.mark.parametrize("payload", [
    pytest.param(wav(frames=3999), id="too-short"),
    pytest.param(wav(frames=64001), id="too-long"),
    pytest.param(wav(channels=2), id="stereo"),
    pytest.param(wav(width=1), id="pcm8"),
    pytest.param(wav(rate=44100), id="wrong-rate"),
])
def test_rejects_invalid_wav_and_releases_matching_token(manager, payload):
    _, capture = armed(manager)
    with pytest.raises(ValueError):
        manager.submit(capture["capture_id"], payload)
    assert manager._test_leases[capture["capture_id"]]["released"]
    assert manager.telemetry.events == []


def test_unknown_id_does_not_release_another_active_capture(manager):
    _, capture = armed(manager)
    with pytest.raises(RuntimeError, match="another.*active"):
        manager.arm(manager.create_session("music")["session_id"])
    with pytest.raises(ValueError):
        manager.cancel("f" * 32)
    assert not manager._test_leases[capture["capture_id"]]["released"]
    assert manager.cancel(capture["capture_id"]) == {"cancelled": True}


def test_arm_rejects_malformed_session_ids(manager):
    with pytest.raises(ValueError, match="session_id must be"):
        manager.arm([])


def test_expired_capture_is_released_on_next_operation(manager):
    _, capture = armed(manager)
    manager._test_clock.value += 31
    assert manager.create_session("music")["condition"] == "music"
    assert manager._test_leases[capture["capture_id"]]["released"]


def test_save_or_index_failure_is_explicit_without_retry(manager):
    _, capture = armed(manager)
    manager.telemetry.save = False
    with pytest.raises(RuntimeError, match="WAV save failed"):
        manager.submit(capture["capture_id"], wav())
    assert manager.telemetry.events[0][2] == "collection"
    assert manager._test_leases[capture["capture_id"]]["released"]

    manager.telemetry.save = True
    _, next_capture = armed(manager)
    manager.telemetry.index = False
    with pytest.raises(RuntimeError, match="orphan retained"):
        manager.submit(next_capture["capture_id"], wav())
    assert manager._test_leases[next_capture["capture_id"]]["released"]


def test_session_and_clip_caps(manager):
    sessions = [manager.create_session("noise") for _ in range(8)]
    with pytest.raises(RuntimeError, match="maximum active"):
        manager.create_session("quiet")
    for _ in range(12):
        capture = manager.arm(sessions[0]["session_id"])
        manager._test_leases[capture["capture_id"]]["ack"] = True
        manager.submit(capture["capture_id"], wav())
    with pytest.raises(RuntimeError, match="clip limit"):
        manager.arm(sessions[0]["session_id"])


def test_saved_collection_is_discoverable_from_real_telemetry(tmp_path, monkeypatch):
    clock = Clock()
    leases = {}

    def create(_directory, capture_id, *, now=None, seconds=30):
        leases[capture_id] = {"expires": now + seconds, "ack": False, "released": False}

    def ack(_directory, capture_id, *, now=None):
        item = leases.get(capture_id)
        return bool(item and item["ack"] and now < item["expires"] and not item["released"])

    def release(_directory, capture_id):
        if capture_id in leases:
            leases[capture_id]["released"] = True
            return True
        return False

    monkeypatch.setattr(collection.runtime, "create_collection_pause", create)
    monkeypatch.setattr(collection.runtime, "collection_pause_acknowledged", ack)
    monkeypatch.setattr(collection.runtime, "release_collection_pause", release)
    root = tmp_path
    log_dir = root / "assistant" / "logs"
    cfg = Config({"telemetry": {"jsonl_path": str(log_dir / "telemetry.jsonl"),
                                 "text_log_path": str(log_dir / "memoir.log"),
                                 "audio_dir": str(log_dir / "audio"),
                                 "archive_audio": True, "max_audio_mb": 1},
                  "audio": {"sample_rate": 16000}})
    monkeypatch.setattr(RealTelemetry, "_setup_text_log", lambda _self, _cfg: None)
    telemetry = RealTelemetry(cfg)
    manager = collection.CollectionManager(telemetry, "leases", clock=clock)
    session = manager.create_session("music")
    capture = manager.arm(session["session_id"])
    leases[capture["capture_id"]]["ack"] = True

    result = manager.submit(capture["capture_id"], wav())
    candidates = list(active_learning.iter_candidates(
        telemetry=telemetry.jsonl_path, audio_root=log_dir / "audio", project_root=root))
    assert len(candidates) == 1
    candidate = candidates[0]
    assert result["event_id"] == candidate["event_id"]
    assert candidate["trigger_source"] == "wake_collection"
    assert candidate["capture_session_id"] == session["session_id"]
    assert candidate["candidate_kind"] == "intentional_wake"
    assert Path(candidate["path"]).is_file()
    identity = active_learning.wav_identity(candidate["path"])
    assert identity["n_frames"] == 4000
    assert identity["wav_sha256"]
    assert candidate["proposed_label"] == "uncertain"
    assert not any(key in candidate for key in ("presence", "wake_start_sample", "wake_end_sample"))
    raw_event = json.loads(telemetry.jsonl_path.read_text(encoding="utf-8"))
    assert raw_event["audio_path"] and raw_event["audio_sha256"] == identity["wav_sha256"]
    assert raw_event["decision_source"] == "explicit_collection_music"
