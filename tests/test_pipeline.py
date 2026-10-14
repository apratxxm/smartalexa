"""Tests for the non-audio parts of the pipeline: endpointing logic, telemetry,
and the ducking restore guarantee. No microphone or network required.
"""
import json
import wave

import numpy as np
import pytest

from assistant.audio import Endpointer
from assistant.telemetry import Event, Telemetry


class FakeVAD:
    """Replays a scripted sequence of speech probabilities."""

    def __init__(self, script):
        self.script = list(script)
        self.i = 0

    def reset(self):
        self.i = 0

    def __call__(self, frame):
        value = self.script[self.i] if self.i < len(self.script) else 0.0
        self.i += 1
        return value


class FakeMic:
    def __init__(self):
        self.reads = 0

    def read(self):
        self.reads += 1
        return np.zeros(512, dtype=np.int16)


FRAME_MS = 32.0  # 512 samples @ 16 kHz


def make_endpointer(script, **kwargs):
    params = dict(silence_ms=800, max_seconds=6.0, speech_timeout_ms=1500)
    params.update(kwargs)
    return Endpointer(FakeVAD(script), **params)


class TestEndpointing:
    def test_silence_ends_the_recording(self):
        # 20 frames of speech, then silence past the 800 ms threshold.
        ep = make_endpointer([1.0] * 20 + [0.0] * 60)
        result = ep.record(FakeMic())
        assert result["speech_started"] is True
        expected = 20 + int(800 / FRAME_MS)
        assert result["duration_ms"] == pytest.approx(expected * FRAME_MS, abs=FRAME_MS)

    def test_no_speech_aborts_early_as_false_accept(self):
        """A wake word that fired with nothing said must not run STT."""
        ep = make_endpointer([0.0] * 300)
        result = ep.record(FakeMic())
        assert result["speech_started"] is False
        # Bails at the speech timeout, nowhere near the 6 s cap.
        assert result["duration_ms"] < 2000

    def test_hard_cap_bounds_continuous_speech(self):
        """The roommate case: speech never stops, so the cap must stop it.

        No acoustic method can tell "still addressing the assistant" from "now
        addressing a human", so the cap is the backstop and intent.py truncates.
        """
        ep = make_endpointer([1.0] * 1000, max_seconds=6.0)
        result = ep.record(FakeMic())
        assert result["speech_started"] is True
        assert result["duration_ms"] == pytest.approx(6000, abs=50)

    def test_brief_pause_does_not_end_speech(self):
        """400 ms of silence mid-sentence is below the 800 ms threshold."""
        script = [1.0] * 10 + [0.0] * 12 + [1.0] * 10 + [0.0] * 60
        ep = make_endpointer(script)
        result = ep.record(FakeMic())
        assert result["duration_ms"] > (10 + 12 + 10) * FRAME_MS

    def test_speech_start_is_recorded(self):
        ep = make_endpointer([0.0] * 5 + [1.0] * 10 + [0.0] * 60)
        result = ep.record(FakeMic())
        assert result["speech_start_ms"] == pytest.approx(5 * FRAME_MS, abs=FRAME_MS)


class TestTelemetry:
    @pytest.fixture
    def telemetry(self, tmp_path, monkeypatch):
        from assistant.config import Config

        cfg = Config({
            "telemetry": {
                "jsonl_path": str(tmp_path / "logs" / "telemetry.jsonl"),
                "text_log_path": str(tmp_path / "logs" / "memoir.log"),
                "audio_dir": str(tmp_path / "logs" / "audio"),
                "archive_audio": True,
                "max_audio_mb": 1,
            },
            "audio": {"sample_rate": 16000},
        })
        monkeypatch.setattr(Config, "path", lambda self, key, default="":
                            __import__("pathlib").Path(self.get(key, default)))
        return Telemetry(cfg)

    def test_event_round_trips_as_jsonl(self, telemetry):
        event = Event(trigger_source="wake", wake_tier="both",
                      transcript="play espresso", intent_action="PLAY_TRACK",
                      intent_query="espresso", outcome="ok")
        telemetry.write(event)
        line = telemetry.jsonl_path.read_text(encoding="utf-8").strip()
        record = json.loads(line)
        assert record["transcript"] == "play espresso"
        assert record["wake_tier"] == "both"
        assert record["event_id"]

    def test_event_records_distinct_trigger_and_command_models(self, telemetry):
        telemetry.write(Event(trigger_stt_model="faster-whisper/small.en",
                              command_stt_model="faster-whisper/turbo",
                              stt_model="faster-whisper/turbo",
                              request_phase="complete", outcome="ok"))
        record = json.loads(telemetry.jsonl_path.read_text(encoding="utf-8").strip())
        assert record["trigger_stt_model"] == "faster-whisper/small.en"
        assert record["command_stt_model"] == "faster-whisper/turbo"
        assert record["request_phase"] == "complete"

    def test_archived_audio_is_readable_16k_mono(self, telemetry):
        """The archive is the training corpus; it has to be valid audio."""
        event = Event(trigger_source="wake")
        pcm = (np.random.rand(16000) * 1000).astype(np.int16)
        telemetry.save_audio(event, pcm)
        files = list(telemetry.audio_dir.glob("*.wav"))
        assert len(files) == 1
        with wave.open(str(files[0]), "rb") as wf:
            assert wf.getframerate() == 16000
            assert wf.getnchannels() == 1
            assert wf.getnframes() == 16000

    def test_archive_is_size_capped(self, telemetry):
        """Rolling archive: oldest out first, so it never fills the disk."""
        big = (np.random.rand(300_000) * 1000).astype(np.int16)  # ~600 KB each
        for _ in range(4):
            telemetry.save_audio(Event(), big)
        total = sum(f.stat().st_size for f in telemetry.audio_dir.glob("*.wav"))
        assert total <= 1 * 1024 * 1024


class TestDuckingRestore:
    """A crash must never leave the speaker at 20%."""

    @pytest.fixture
    def fake_volume(self, monkeypatch, tmp_path):
        from assistant import ducking
        from types import SimpleNamespace

        state = {"level": 1.0}
        session = SimpleNamespace(
            InstanceIdentifier="spotify-test-session",
            SimpleAudioVolume=SimpleNamespace(
                GetMasterVolume=lambda: state["level"],
                SetMasterVolume=lambda level, event: state.__setitem__("level", level)))
        monkeypatch.setattr(ducking, "_spotify_sessions", lambda: [session])
        monkeypatch.setattr(ducking.time, "sleep", lambda _: None)
        monkeypatch.setattr(ducking, "_LEASE_ACTIVE", False)
        monkeypatch.setattr(ducking, "_STATE_FILE", tmp_path / ".duck_state.json")
        return state

    def test_restores_on_clean_exit(self, fake_volume):
        from assistant import ducking

        with ducking.ducked(20) as (was, now):
            assert fake_volume["level"] == pytest.approx(0.2)
            assert (was, now) == (100, 20)
        assert fake_volume["level"] == pytest.approx(1.0)

    def test_restores_on_exception(self, fake_volume):
        from assistant import ducking

        with pytest.raises(ValueError):
            with ducking.ducked(20):
                raise ValueError("boom")
        assert fake_volume["level"] == pytest.approx(1.0)

    def test_state_file_cleared_after_restore(self, fake_volume):
        from assistant import ducking

        with ducking.ducked(20):
            assert ducking._STATE_FILE.exists()
        assert not ducking._STATE_FILE.exists()

    def test_repair_restores_orphaned_level(self, fake_volume):
        """Simulates a hard kill mid-duck, which no context manager can cover."""
        from assistant import ducking

        ducking._STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        ducking._write_session_state({"spotify-test-session": 0.85},
                                     {"spotify-test-session": 0.2})
        fake_volume["level"] = 0.2

        ducking.repair_orphaned_duck()

        assert fake_volume["level"] == pytest.approx(0.85)
        assert not ducking._STATE_FILE.exists()

    def test_no_output_endpoint_raises(self, monkeypatch, tmp_path):
        from assistant import ducking

        monkeypatch.setattr(ducking, "_snapshot_sessions", lambda: (_ for _ in ()).throw(RuntimeError("no output")))
        monkeypatch.setattr(ducking, "_STATE_FILE", tmp_path / ".duck_state.json")
        with pytest.raises(RuntimeError, match="no output"):
            with ducking.ducked(20):
                pass
