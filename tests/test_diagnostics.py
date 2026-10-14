"""Diagnostics: stage timeline, event-id logging, near-miss archive, health heartbeat."""
import io
import json
import logging
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from assistant import diag, health, main
from assistant.config import Config
from assistant.listener import SttListener
from assistant.telemetry import Event, Telemetry, format_summary


# ------------------------------------------------------------------ timeline

def test_mark_records_ms_from_origin_and_first_mark_wins():
    event = Event()
    event.set_origin(event._origin - 1.0)            # evidence one second before the event
    event.mark("wake_evidence", event._origin)
    event.mark("stt_done", event._origin + 0.5)
    event.mark("stt_done", event._origin + 9.0)       # ignored: first mark wins
    event.mark("stt_done", event._origin + 0.75, overwrite=True)
    assert event.timeline == {"wake_evidence": 0, "stt_done": 750}


def test_set_origin_ignores_missing_times_and_keeps_earliest():
    event = Event()
    created = event._origin
    event.set_origin(0.0, None, created + 5)
    assert event._origin == created
    event.set_origin(created - 2)
    assert event._origin == created - 2


def test_timeline_is_in_jsonl_but_origin_is_not():
    event = Event()
    event.mark("speech_end")
    record = json.loads(json.dumps(__import__("dataclasses").asdict(event)))
    assert "speech_end" in record["timeline"]
    assert record["wake_path"] == "" and record["wake_scores"] == {}
    assert "_origin" not in record


def test_stage_deltas_are_consecutive_differences_in_time_order():
    timeline = {"stt_done": 1900, "speech_start": 0, "wake_evidence": 0, "speech_end": 900,
                "duck_start": 120}
    assert diag.stage_deltas(timeline) == [
        ("speech_start", 0), ("wake_evidence", 0), ("duck_start", 120),
        ("speech_end", 780), ("stt_done", 1000)]
    assert diag.stage_deltas(None) == []


def test_wake_trace_keeps_peak_per_model_and_resets():
    trace = diag.WakeTrace(hit_floor=0.15)
    trace.note(0.01, "alexa")            # below the floor: ignored
    trace.note(0.4, "alexa")
    trace.note(0.9, "hey_mycroft")
    trace.note(0.7, "alexa")
    first, scores = trace.take()
    assert scores == {"alexa": 0.7, "hey_mycroft": 0.9}
    assert first > 0
    assert trace.take() == (0.0, {})


def test_wake_trace_forgets_old_scores():
    trace = diag.WakeTrace()
    trace.note(0.5, "alexa")
    trace._peaks["alexa"] = (0.5, time.perf_counter() - trace.WINDOW_S - 1)
    assert trace.snapshot() == {}


def _timeline_app():
    app = main.Assistant.__new__(main.Assistant)
    app.cfg = Config({})
    app._wake_trace = diag.WakeTrace()
    app._early_duck_at = 0.0
    return app


def test_open_timeline_uses_earliest_evidence_and_copies_scores():
    app = _timeline_app()
    now = time.perf_counter()
    app._wake_trace.note(0.8, "alexa")
    app._wake_trace.first_at = now - 0.3             # acoustic hit 300 ms ago
    app._early_duck_at = now - 0.2
    utt = SimpleNamespace(speech_started_at=now - 0.5, wake_evidence_at=now - 0.1,
                          completed_at=now - 0.05, trigger_done_at=now)
    event = Event()
    app._open_timeline(event, utt, now)
    assert event.wake_scores == {"alexa": 0.8}
    t = event.timeline
    assert t["speech_start"] == 0                   # the user started talking first
    assert t["wake_evidence"] == 200                # acoustic hit, 200 ms into the speech
    assert t["duck_start"] == 300
    assert t["speech_end"] < t["trigger_stt_done"]
    assert app._early_duck_at == 0.0                # consumed


def test_handle_utterance_fills_timeline_wake_path_and_log_id(monkeypatch):
    from contextlib import nullcontext
    app = _timeline_app()
    app.spotify = Mock()
    app.spotify.token_age_days.return_value = None
    app.spotify.last_resolution = None
    app.stt = Mock(model_name="test")
    app.stt.transcribe.return_value = ("Alexa pause", 5)
    app.feedback = Mock()
    app.listener = SttListener(app.cfg, Mock(), None)
    app._early_lease = None
    app.telemetry = Mock()
    monkeypatch.setattr(main.ducking, "ducked", lambda _: nullcontext((80, 20)))
    now = time.perf_counter()
    pcm = np.zeros(512, dtype=np.int16)
    utt = SimpleNamespace(command="pause", matched_phrase="alexa", transcript="alexa pause",
                          latency_ms=5, audio=pcm, wake_path="peek",
                          speech_started_at=now - 1.0, wake_evidence_at=now - 0.6,
                          completed_at=now - 0.2, trigger_done_at=now - 0.1)
    seen = []
    app.telemetry.write.side_effect = lambda e: seen.append(diag.current_event())

    app.handle_utterance(Mock(), utt)

    event = app.telemetry.write.call_args.args[0]
    assert seen == [event]                           # bound while the summary is written
    assert diag.current_event() is None              # and released afterwards
    assert event.wake_path == "peek"
    for stage in ("speech_start", "wake_evidence", "speech_end", "stt_done",
                  "intent_done", "playback_confirmed", "feedback_done", "handler_done"):
        assert stage in event.timeline, stage
    offsets = [event.timeline[s] for s in ("speech_start", "speech_end", "stt_done",
                                           "intent_done", "playback_confirmed", "handler_done")]
    assert offsets == sorted(offsets)
    assert event.playback_method == "web_api"        # no local route reported by the mock


def test_paused_mic_and_hotkey_wake_paths(monkeypatch):
    from contextlib import nullcontext
    app = _timeline_app()
    app.spotify = Mock()
    app.spotify.token_age_days.return_value = None
    app.stt = Mock(model_name="test")
    app.stt.transcribe.return_value = ("alexa pause", 5)
    app.feedback = Mock()
    app.listener = SttListener(app.cfg, Mock(), None)
    app._early_lease = None
    app.telemetry = Mock()
    monkeypatch.setattr(main.ducking, "ducked", lambda _: nullcontext((80, 20)))
    utt = SimpleNamespace(command="pause", matched_phrase="alexa", transcript="alexa pause",
                          latency_ms=5, audio=np.zeros(512, dtype=np.int16), wake_path="final_text")
    app.handle_utterance(Mock(), utt, input_device="sounddrum")
    assert app.telemetry.write.call_args.args[0].wake_path == "paused_mic"


# ---------------------------------------------------------------- event-id log

def _logger_with_filter(name):
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(diag.EventFormatter(diag.LOG_FORMAT))
    handler.addFilter(diag.EventIdFilter())
    logger = logging.getLogger(name)
    logger.handlers = [handler]
    logger.propagate = False
    logger.setLevel(logging.INFO)
    return logger, stream


def test_log_lines_carry_the_current_event_id_including_child_loggers():
    parent, stream = _logger_with_filter("memoir_test_ids")
    child = logging.getLogger("memoir_test_ids.spotify")
    event = Event(event_id="abc123def456")
    token = diag.bind_event(event)
    try:
        child.info("Resolved %r", "red")
    finally:
        diag.unbind_event(token)
    child.info("after the command")
    lines = stream.getvalue().splitlines()
    assert "memoir_test_ids.spotify: [abc123def456] Resolved 'red'" in lines[0]
    assert "[abc123def456]" not in lines[1]
    assert lines[1].endswith("memoir_test_ids.spotify: after the command")


def test_summary_line_is_not_tagged_twice():
    parent, stream = _logger_with_filter("memoir_test_summary")
    event = Event(event_id="feedfacecafe", trigger_source="stt", outcome="ok")
    token = diag.bind_event(event)
    try:
        parent.info("%s", format_summary(event))
    finally:
        diag.unbind_event(token)
    line = stream.getvalue()
    assert line.count("feedfacecafe") == 1


def test_formatter_tolerates_records_without_the_filter():
    handler = logging.StreamHandler(io.StringIO())
    handler.setFormatter(diag.EventFormatter(diag.LOG_FORMAT))
    record = logging.LogRecord("x", logging.INFO, __file__, 1, "plain", None, None)
    assert "plain" in handler.format(record)


def test_annotate_event_sets_fields_on_the_bound_event_only():
    diag.annotate_event(brain_tool="ignored")        # nothing bound: no-op, no error
    event = Event()
    token = diag.bind_event(event)
    try:
        diag.annotate_event(brain_tool="play_song", not_a_field="x")
    finally:
        diag.unbind_event(token)
    assert event.brain_tool == "play_song" and not hasattr(event, "not_a_field")


def test_brain_decide_annotates_tool_args_latency_and_validation(monkeypatch):
    from assistant.brain import Brain, decide
    brain = Brain("test-model", context=lambda: {"artists": ["Seedhe Maut"], "playlists": []})
    event = Event()
    token = diag.bind_event(event)
    try:
        monkeypatch.setattr(brain, "_chat", lambda m: {
            "tool": "play_song", "arguments": {"title": "Madira", "artist": "Seedhe Maut"}})
        intent = decide("play madira by see the mode", brain)
        assert intent.query == "madira by seedhe maut"
        assert (event.brain_tool, event.brain_outcome) == ("play_song", "accepted")
        assert json.loads(event.brain_args)["title"] == "Madira"
        assert event.brain_latency_ms is not None

        monkeypatch.setattr(brain, "_chat", lambda m: {
            "tool": "play_song", "arguments": {"title": "Bohemian Rhapsody", "artist": ""}})
        decide("play demons", brain)                 # invented title: rules answer
        assert event.brain_outcome.startswith("validated_away:")

        monkeypatch.setattr(brain, "_chat", Mock(side_effect=TimeoutError("busy")))
        decide("play demons", brain)
        assert event.brain_outcome == "unavailable:TimeoutError"
        decide("pause", brain)                       # plain transport never reaches the model
        assert event.brain_outcome == "rules_fast"
    finally:
        diag.unbind_event(token)


# ------------------------------------------------------------- near-miss clips

def make_telemetry(tmp_path, **extra):
    root = tmp_path / "logs"
    cfg = Config({"telemetry": {
        "jsonl_path": str(root / "telemetry.jsonl"),
        "text_log_path": str(root / "memoir.log"),
        "audio_dir": str(root / "audio"),
        "archive_audio": True,
        "max_audio_mb": extra.pop("max_audio_mb", 50),
        **extra,
    }, "audio": {"sample_rate": 16000}})
    return Telemetry(cfg)


def clip(seconds=1.0):
    return (np.random.default_rng(0).standard_normal(int(16000 * seconds)) * 1000).astype(np.int16)


def jsonl(telemetry):
    return [json.loads(line) for line in telemetry.jsonl_path.read_text(encoding="utf-8").splitlines()]


def test_near_miss_is_archived_with_record_and_rate_limited_to_one_per_five_seconds(tmp_path):
    telemetry = make_telemetry(tmp_path)
    first = telemetry.record_near_miss(clip(), "Alexei play red", "alias_uncorroborated",
                                       woke_early=True, scores={"alexa": 0.42}, now=100.0)
    assert first is not None
    suppressed = telemetry.record_near_miss(clip(), "Alexa?", "no_trigger_match", now=104.9)
    assert suppressed.audio_status == "suppressed" and suppressed.audio_reason == "rate_limited"
    assert telemetry.near_miss_suppressed == 1
    assert telemetry.record_near_miss(clip(), "Alexa!", "no_trigger_match", now=105.0) is not None
    wavs = list((telemetry.audio_dir / "near_miss").glob("*.wav"))
    assert len(wavs) == 2
    assert not list(telemetry.audio_dir.glob("*.wav"))        # not mixed into the command archive
    records = jsonl(telemetry)
    assert [r["trigger_source"] for r in records] == ["near_miss", "near_miss", "near_miss"]
    assert records[0]["transcript"] == "Alexei play red"
    assert records[0]["rejection_reason"] == "alias_uncorroborated"
    assert records[0]["woke_early"] is True and records[0]["wake_scores"] == {"alexa": 0.42}
    assert {Path(r["audio_path"]).name for r in records if r["audio_path"]} == {w.name for w in wavs}
    assert all("near_miss" in r["audio_path"] for r in records if r["audio_path"])
    assert records[1]["audio_path"] == "" and records[1]["review_state"] == "unknown"


def test_turbo_rejections_have_their_own_shorter_limit_and_keep_the_turbo_transcript(tmp_path):
    telemetry = make_telemetry(tmp_path)
    assert telemetry.record_near_miss(clip(), "Alexa play", "second_stt_unconfirmed",
                                      second_stt_transcript="Hey Alex, play", kind="second_stt",
                                      now=10.0) is not None
    # A plain near miss right after is not blocked by the turbo clip...
    assert telemetry.record_near_miss(clip(), "Alexei", "no_trigger_match", now=10.1) is not None
    # ...but turbo rejections are limited to one per second.
    suppressed = telemetry.record_near_miss(clip(), "Alexa", "second_stt_unconfirmed", kind="second_stt", now=10.5)
    assert suppressed.audio_status == "suppressed" and suppressed.audio_reason == "rate_limited"
    assert telemetry.record_near_miss(clip(), "Alexa", "second_stt_unconfirmed", kind="second_stt",
                                      now=11.1) is not None
    first = jsonl(telemetry)[0]
    assert first["second_stt_transcript"] == "Hey Alex, play"
    assert first["trigger_confirmation_mode"] == "second_stt"


def test_near_miss_clips_are_size_bounded_oldest_first(tmp_path):
    telemetry = make_telemetry(tmp_path, max_near_miss_mb=1)
    # ~192 KB per 6 s clip: the 6th write must evict the first.
    paths = []
    for i in range(7):
        event = telemetry.record_near_miss(clip(6.0), "Alexei", "no_trigger_match", now=i * 10.0)
        paths.append(event.audio_path)
        time.sleep(0.02)
    wavs = list((telemetry.audio_dir / "near_miss").glob("*.wav"))
    assert len(wavs) == 5                                       # 5 x 192 KB fit under 1 MB
    root = telemetry.jsonl_path.parent.parent.parent
    assert not (root / paths[0]).exists()                       # the oldest went first
    assert (root / paths[-1]).exists()


def test_near_miss_clips_count_toward_the_overall_audio_cap(tmp_path):
    telemetry = make_telemetry(tmp_path, max_audio_mb=1, max_near_miss_mb=256)
    old = telemetry.audio_dir / "20260101-000000_aaaaaaaaaaaa.wav"
    old.write_bytes(b"\0" * (900 * 1024))
    time.sleep(0.02)
    event = telemetry.record_near_miss(clip(6.0), "Alexei", "no_trigger_match", now=1.0)
    assert event.audio_path                                    # saved ...
    assert not old.exists()                                    # ... by evicting the oldest clip


def test_no_audio_means_record_only_when_archiving_is_off(tmp_path):
    telemetry = make_telemetry(tmp_path, archive_audio=False)
    event = telemetry.record_near_miss(clip(), "Alexei", "no_trigger_match", now=1.0)
    assert event.audio_path == ""
    assert not (telemetry.audio_dir / "near_miss").exists() or not list(
        (telemetry.audio_dir / "near_miss").glob("*.wav"))
    assert len(jsonl(telemetry)) == 1


class _Cfg:
    def get(self, key, default=None):
        return {"listener.phrases": ["alexa"], "listener.aliases": {"alexei": "alexa"},
                "listener.min_speech_ms": 300, "listener.silence_ms": 100,
                "listener.peek_first_ms": 700}.get(key, default)


def _drive(listener, speech_frames, silence_frames=10):
    frame = np.zeros(512, dtype=np.int16)
    result = None
    for _ in range(speech_frames):
        result = listener.feed(frame) or result
    for _ in range(silence_frames):
        result = listener.feed(frame) or result
    return result


class _Vad:
    """Speaks for the first N frames it is asked about, then goes quiet."""
    def __init__(self, speech_frames):
        self.left = speech_frames

    def __call__(self, frame):
        self.left -= 1
        return 1.0 if self.left >= 0 else 0.0


def test_listener_hands_near_miss_audio_to_the_callback_with_reason():
    got = []
    listener = SttListener(_Cfg(), _Vad(14), lambda audio: ("Alexei play red", 3),
                           on_near_miss=got.append)
    _drive(listener, 14)
    assert len(got) == 1
    info = got[0]
    assert info["transcript"] == "Alexei play red"
    assert "alias_uncorroborated" in info["reason"]
    assert info["woke_early"] is False and info["kind"] == "near_miss"
    assert info["audio"].size > 0


def test_listener_archives_every_turbo_rejection_with_the_turbo_transcript():
    got = []
    verify = Mock(return_value=("", "", False, 120, "Hey, Alex."))
    listener = SttListener(_Cfg(), _Vad(14), lambda audio: ("Alexei play red", 3),
                           verify_trigger=verify, on_near_miss=got.append)
    _drive(listener, 14)
    assert verify.called
    assert got[0]["kind"] == "second_stt"
    assert got[0]["second_stt_transcript"] == "Hey, Alex."
    assert "second_stt_unconfirmed" in got[0]["reason"]


def test_ambient_speech_is_never_handed_over():
    got = []
    listener = SttListener(_Cfg(), _Vad(14), lambda audio: ("what a lovely day it is", 3),
                           on_near_miss=got.append)
    _drive(listener, 14)
    assert got == []


def test_a_failing_archive_callback_never_breaks_listening():
    listener = SttListener(_Cfg(), _Vad(14), lambda audio: ("Alexei play red", 3),
                           on_near_miss=Mock(side_effect=OSError("disk full")))
    utt = _drive(listener, 14)                             # must not raise
    assert utt is not None and not utt.triggered


def test_utterance_reports_wake_path_and_times():
    listener = SttListener(_Cfg(), _Vad(14), lambda audio: ("Alexa play red", 3))
    utt = _drive(listener, 14)
    assert utt.triggered and utt.wake_path == "final_text"
    assert 0 < utt.speech_started_at <= utt.completed_at <= utt.trigger_done_at
    assert utt.wake_evidence_at == utt.trigger_done_at


def test_early_peek_marks_wake_path_peek_and_its_time():
    woke = []
    listener = SttListener(_Cfg(), _Vad(40), lambda audio: ("Alexa play red", 3),
                           on_wake=lambda: woke.append(time.perf_counter()))
    utt = _drive(listener, 40)
    assert woke and utt.wake_path == "peek"
    assert utt.speech_started_at <= utt.wake_evidence_at < utt.trigger_done_at


def test_main_routes_near_miss_to_telemetry_with_current_scores():
    app = _timeline_app()
    app.telemetry = Mock()
    app._wake_trace.note(0.33, "alexa")
    app._record_near_miss({"audio": clip(), "transcript": "Alexei", "reason": "x",
                           "woke_early": True, "second_stt_transcript": "", "kind": "near_miss"})
    kwargs = app.telemetry.record_near_miss.call_args.kwargs
    assert kwargs["scores"] == {"alexa": 0.33} and kwargs["woke_early"] is True
    assert app._wake_trace.take() == (0.0, {})            # burst consumed


def test_personal_wake_score_and_confirmer_are_not_lost_in_near_miss():
    app = _timeline_app()
    app.telemetry = Mock()
    app.wake = SimpleNamespace(last_score=0.01, last_model="alexa",
                               personal_score=0.93, _personal_key="alexa_user")
    app.listener_stt = SimpleNamespace(model_name="small.en", runtime_info=lambda: "cuda:float16")
    app._trace_wake()
    app._record_near_miss({"audio": clip(), "transcript": "", "reason": "acoustic_unconfirmed",
                           "kind": "acoustic_unconfirmed", "stt_latency_ms": 137})
    kwargs = app.telemetry.record_near_miss.call_args.kwargs
    assert kwargs["scores"] == {"alexa_user": 0.93}
    assert kwargs["trigger_stt_model"] == "small.en"
    assert kwargs["trigger_stt_device"] == "cuda:float16"
    assert kwargs["trigger_stt_latency_ms"] == 137


def test_near_miss_records_level_clipping_and_recognizer(tmp_path):
    telemetry = make_telemetry(tmp_path)
    audio = np.array([-32768, 32767, 0, 0], dtype=np.int16)
    event = telemetry.record_near_miss(audio, "", "acoustic_unconfirmed",
                                      trigger_stt_model="small.en", trigger_stt_device="cuda:float16",
                                      trigger_stt_latency_ms=137)
    assert event.audio_clipped_pct == 50
    assert event.audio_rms_dbfs == pytest.approx(-3.01, abs=0.01)
    record = jsonl(telemetry)[0]
    assert record["trigger_stt_model"] == "small.en"
    assert record["trigger_stt_device"] == "cuda:float16"
    assert record["trigger_stt_latency_ms"] == 137


# ------------------------------------------------------------- health heartbeat

def fake_psutil(monkeypatch, private=3 * 1024 ** 3):
    process = Mock()
    process.memory_info.return_value = SimpleNamespace(rss=1 * 1024 ** 3, private=private, vms=0)
    process.num_threads.return_value = 41
    process.num_handles.return_value = 777
    process.create_time.return_value = time.time() - 3 * 3600 - 125
    monkeypatch.setattr(health.psutil, "Process", lambda: process)
    monkeypatch.setattr(health.psutil, "virtual_memory", lambda: SimpleNamespace(
        available=6 * 1024 ** 3, total=16 * 1024 ** 3))


def test_heartbeat_line_has_every_reading(monkeypatch):
    fake_psutil(monkeypatch)
    monkeypatch.setattr(health, "commit_charge", lambda: (20 * 1024 ** 3, 25 * 1024 ** 3))
    monkeypatch.setattr(health.subprocess, "run", lambda *a, **k: SimpleNamespace(
        stdout="5120, 8188\n"))
    until = time.mktime((2026, 10, 3, 21, 0, 0, 0, 0, -1))
    stats = health.collect({"ollama": lambda: True, "spotify_bans": lambda: {"web": until},
                            "mic_overflows": lambda: 2, "echo_alive": lambda: True,
                            "capture_input_state": lambda: "digital_silence"})
    line = health.format_line(stats, "periodic")
    assert line.startswith("Health [periodic]: ")
    assert stats["capture_input_state"] == "digital_silence"
    assert line.endswith("capture_input_state=digital_silence")
    for token in ("rss=1024MB", "private=3072MB", "ram_free=6144/16384MB",
                  "commit=20480/25600MB(80%)", "gpu=5120/8188MB", "ollama=up",
                  "spotify_ban=web@2026-10-03_21:00", "mic_overflows=2", "echo_ref=alive",
                  "threads=41", "handles=777", "up=3h02m"):
        assert token in line, token


def test_heartbeat_survives_every_probe_failing(monkeypatch):
    monkeypatch.setattr(health.psutil, "Process", Mock(side_effect=OSError("denied")))
    monkeypatch.setattr(health.psutil, "virtual_memory", Mock(side_effect=OSError("denied")))
    monkeypatch.setattr(health, "commit_charge", lambda: None)
    monkeypatch.setattr(health.subprocess, "run", Mock(side_effect=FileNotFoundError("no nvidia-smi")))
    boom = Mock(side_effect=RuntimeError("probe broke"))
    line = health.format_line(health.collect({"ollama": boom, "spotify_bans": boom,
                                              "mic_overflows": boom, "echo_alive": boom}))
    assert "gpu=n/a" in line and "ollama=n/a" in line and "rss=n/a" in line
    assert "spotify_ban=n/a" in line and "echo_ref=n/a" in line
    assert line.endswith("capture_input_state=unknown")


def test_capture_input_health_state_is_whitelisted_and_appended(monkeypatch):
    fake_psutil(monkeypatch)
    monkeypatch.setattr(health, "commit_charge", lambda: None)
    monkeypatch.setattr(health, "gpu_memory", lambda: None)
    stats = health.collect({"capture_input_state": lambda: "private-endpoint-name"})
    line = health.format_line(stats)
    assert stats["capture_input_state"] == "unknown"
    assert line.endswith("capture_input_state=unknown")
    assert "private-endpoint-name" not in line


def test_no_ban_and_silent_echo_reference_are_spelled_out(monkeypatch):
    fake_psutil(monkeypatch)
    monkeypatch.setattr(health, "commit_charge", lambda: None)
    monkeypatch.setattr(health.subprocess, "run", Mock(side_effect=subprocess.TimeoutExpired("x", 2)))
    line = health.format_line(health.collect({"spotify_bans": lambda: {}, "echo_alive": lambda: False,
                                              "ollama": lambda: False}))
    assert "spotify_ban=none" in line and "echo_ref=silent" in line and "ollama=down" in line


def test_gpu_query_is_windowless_bounded_and_parsed(monkeypatch):
    seen = {}

    def run(cmd, **kwargs):
        seen["cmd"], seen["kwargs"] = cmd, kwargs
        return SimpleNamespace(stdout="1234, 8188\n")

    monkeypatch.setattr(health.subprocess, "run", run)
    assert health.gpu_memory() == (1234, 8188)
    assert seen["cmd"][:2] == ["nvidia-smi", "--query-gpu=memory.used,memory.total"]
    assert seen["kwargs"]["timeout"] == 2.0
    assert seen["kwargs"]["creationflags"] == getattr(subprocess, "CREATE_NO_WINDOW", 0)
    monkeypatch.setattr(health.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="garbage"))
    assert health.gpu_memory() is None


def test_heartbeat_includes_actual_stt_placement(monkeypatch):
    fake_psutil(monkeypatch)
    monkeypatch.setattr(health, "commit_charge", lambda: None)
    monkeypatch.setattr(health, "gpu_memory", lambda: None)
    line = health.format_line(health.collect({"command_stt": lambda: "cuda:float16",
                                             "wake_stt": lambda: "cpu:int8"}))
    assert "command_stt=cuda:float16 wake_stt=cpu:int8" in line


class FakeHttpResponse:
    def __init__(self, body):
        self.body = body
        self.read_size = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, size):
        self.read_size = size
        return self.body[:size]


def fake_ollama_ps(monkeypatch, body, seen=None, error=None):
    response = FakeHttpResponse(body)

    class Opener:
        def open(self, request, timeout):
            if seen is not None:
                seen["request"], seen["timeout"] = request, timeout
            if error:
                raise error
            return response

    def build_opener(*handlers):
        if seen is not None:
            seen["handlers"] = handlers
        return Opener()

    monkeypatch.setattr(health.urllib.request, "build_opener", build_opener)
    return response


def test_ollama_model_status_empty_ps_means_unloaded(monkeypatch):
    seen = {}
    fake_ollama_ps(monkeypatch, b'{"models":[]}', seen)
    assert health.ollama_model_status("qwen3:4b") == "unloaded"
    assert seen["request"].full_url == "http://127.0.0.1:11434/api/ps"


def test_ollama_model_status_requires_exact_model(monkeypatch):
    fake_ollama_ps(monkeypatch, json.dumps({"models": [{
        "name": "another-model", "model": "another-model", "size": 100, "size_vram": 100,
    }]}).encode())
    assert health.ollama_model_status("qwen3:4b") == "unloaded"


@pytest.mark.parametrize(("size_vram", "expected"), [
    (100, "loaded_gpu"), (0, "loaded_cpu"), (40, "loaded_partial_gpu"),
])
def test_ollama_model_status_reports_model_allocation(monkeypatch, size_vram, expected):
    fake_ollama_ps(monkeypatch, json.dumps({"models": [{
        "name": "qwen3:4b", "model": "qwen3:4b", "size": 100,
        "size_vram": size_vram,
    }]}).encode())
    assert health.ollama_model_status("qwen3:4b") == expected


@pytest.mark.parametrize("body", [
    b"not json", b"{}", b'{"models":{}}', b'{"models":[null]}',
    b'{"models":[{"name":"qwen3:4b","size":100,"size_vram":"100"}]}',
])
def test_ollama_model_status_malformed_response_is_unavailable(monkeypatch, body):
    fake_ollama_ps(monkeypatch, body)
    assert health.ollama_model_status("qwen3:4b") == "unavailable"


def test_ollama_model_status_bounds_response_and_request(monkeypatch):
    seen = {}
    response = fake_ollama_ps(monkeypatch, b" " * (health.OLLAMA_PS_MAX_BYTES + 1), seen)
    assert health.ollama_model_status("qwen3:4b") == "unavailable"
    assert response.read_size == health.OLLAMA_PS_MAX_BYTES + 1
    assert seen["timeout"] == health.OLLAMA_PS_TIMEOUT_SECONDS == 0.25
    assert isinstance(seen["handlers"][0], urllib.request.ProxyHandler)
    assert seen["handlers"][0].proxies == {}
    redirect = seen["handlers"][1]
    assert redirect.redirect_request(None, None, 302, "redirect", {}, "http://example.com") is None


def test_ollama_model_status_timeout_and_redirect_are_unavailable(monkeypatch):
    fake_ollama_ps(monkeypatch, b"", error=TimeoutError("private detail"))
    assert health.ollama_model_status("qwen3:4b") == "unavailable"
    fake_ollama_ps(monkeypatch, b"", error=urllib.error.HTTPError(
        health.OLLAMA_PS, 302, "redirect", {}, None))
    assert health.ollama_model_status("qwen3:4b") == "unavailable"


def test_collect_reports_model_probe_failure_without_raw_error(caplog):
    def fail():
        raise RuntimeError("secret request detail")

    stats = health.collect({"ollama_model": fail})
    line = health.format_line(stats)
    assert stats["ollama_model"] == "unavailable"
    assert "ollama_model=unavailable" in line
    assert "secret request detail" not in caplog.text
    safe = health.collect({"ollama_model": lambda: "private-model-tag"})
    assert safe["ollama_model"] == "unavailable"
    assert "private-model-tag" not in health.format_line(safe)


def test_snapshot_logs_one_info_line_and_never_raises(monkeypatch):
    logger = logging.getLogger("memoir.health")
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    logger.addHandler(handler)
    old_level = logger.level
    logger.setLevel(logging.INFO)
    try:
        monkeypatch.setattr(health, "collect", Mock(side_effect=MemoryError()))
        health.log_snapshot("fatal")                    # must swallow even MemoryError
        monkeypatch.undo()
        fake_psutil(monkeypatch)
        monkeypatch.setattr(health, "commit_charge", lambda: None)
        monkeypatch.setattr(health, "gpu_memory", lambda: None)
        health.log_snapshot("fatal", {})
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)
    lines = stream.getvalue().splitlines()
    assert lines[0] == "Health [fatal]: unavailable (MemoryError)"
    assert lines[1].startswith("Health [fatal]: rss=")


def test_monitor_beats_on_a_daemon_thread_and_stops(monkeypatch):
    beats = []
    monkeypatch.setattr(health, "log_snapshot", lambda reason, probes=None: beats.append(reason))
    watched = []
    monkeypatch.setattr(health.ducking, "duck_watchdog", lambda: watched.append(1) or 0.0)
    monitor = health.HealthMonitor({"spotify_bans": lambda: watched.append("ban") or {}},
                                   interval_s=0.05, tick_s=0.01)
    monitor.start()
    deadline = time.time() + 2
    while beats.count("periodic") < 2 and time.time() < deadline:
        time.sleep(0.01)
    monitor.stop()
    assert beats[0] == "startup" and beats.count("periodic") >= 2
    assert monitor._thread.daemon
    assert watched and "ban" in watched                 # watchdog ticks between beats


def test_duck_watchdog_warns_once_per_stuck_duck(monkeypatch):
    from assistant import ducking
    logger = logging.getLogger("memoir.ducking")
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    logger.addHandler(handler)
    started = time.monotonic() - 20
    ducking._DUCK_SINCE["lease"] = (started, 80, 30)
    try:
        assert ducking.duck_watchdog() >= 20
        ducking.duck_watchdog()
    finally:
        ducking._DUCK_SINCE.pop("lease", None)
        ducking._DUCK_WARNED.clear()
        logger.removeHandler(handler)
    assert stream.getvalue().count("Duck still active after") == 1
    assert ducking.duck_watchdog() == 0.0


def test_assistant_health_probes_read_mic_echo_and_spotify_guard():
    app = main.Assistant.__new__(main.Assistant)
    app._active_mic = SimpleNamespace(queue_overflow_count=5)
    app.echo = SimpleNamespace(reference=SimpleNamespace(alive=lambda: True))
    app.spotify = SimpleNamespace(sp=SimpleNamespace(ban_status=lambda: {"web": 1.0}))
    app.brain = None
    probes = app.health_probes()
    assert probes["mic_overflows"]() == 5 and probes["echo_alive"]() is True
    assert probes["spotify_bans"]() == {"web": 1.0} and probes["ollama"]() is None
    assert probes["capture_input_state"]() == "unknown"
    app._capture_input_state = "digital_silence"
    assert probes["capture_input_state"]() == "digital_silence"
    app._active_mic = None
    app.echo = None
    assert probes["mic_overflows"]() is None and probes["echo_alive"]() is None
