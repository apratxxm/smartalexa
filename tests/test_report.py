"""assistant.report on a hand-made fixture of telemetry, memoir.log and crash.log."""
import json
import time

import pytest

from assistant import report

BASE = time.mktime((2026, 10, 2, 21, 0, 0, 0, 0, -1))


def stamp(offset_s):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(BASE + offset_s))


def command(offset, **fields):
    record = {"ts": BASE + offset, "event_id": "e%011d" % int(offset), "trigger_source": "stt",
              "outcome": "ok", "transcript": "alexa play red by seedhe maut",
              "intent_action": "PLAY_TRACK", "intent_query": "red by seedhe maut",
              "wake_path": "final_text", "handler_latency_ms": 3000}
    record.update(fields)
    return record


def timeline(stt=600, play=1200, tail=40):
    """A consistent stage timeline (ms from the origin)."""
    t = {"speech_start": 0, "wake_evidence": 0, "speech_end": 900}
    t["stt_done"] = 900 + stt
    t["intent_done"] = t["stt_done"] + 50
    t["resolved"] = t["intent_done"] + 150
    t["playback_confirmed"] = t["resolved"] + play
    t["feedback_done"] = t["playback_confirmed"] + 10
    t["handler_done"] = t["feedback_done"] + tail
    return t


EVENTS = [
    command(60, timeline=timeline(), resolution_path="catalogue", resolution_score=1.02,
            resolution_runner_up="RED CUP by Calm", resolution_margin=0.17,
            playback_method="web_api", radio_count=49, brain_tool="play_song",
            brain_outcome="accepted", brain_latency_ms=520, intent_source="llm",
            matched_rule="brain_play_song", wake_scores={"alexa": 0.81}),
    command(120, wake_path="peek", timeline=timeline(stt=1500, tail=1090),
            resolution_path="search_fielded", resolution_score=0.88, playback_method="desktop_uri",
            brain_tool="play_song", brain_outcome="validated_away:Title 'x' was not heard",
            brain_latency_ms=700, matched_rule="brain_play_song"),
    command(180, trigger_source="hotkey", wake_path="hotkey", outcome="false_accept",
            transcript="", intent_action="", intent_query="", timeline=timeline(tail=2000)),
    command(240, outcome="error", error="Spotify web API is rate-limiting this app",
            timeline=timeline(tail=3000)),
    # Telemetry from before the diagnostics existed: no timeline, no wake_path.
    {"ts": BASE + 300, "event_id": "legacy0000001", "trigger_source": "wake", "outcome": "ok",
     "transcript": "pause", "intent_action": "PAUSE", "intent_query": "", "stt_latency_ms": 470,
     "intent_latency_ms": 5, "spotify_latency_ms": 90, "handler_latency_ms": 900},
    {"ts": BASE + 330, "event_id": "wr0000000001", "trigger_source": "wake_rejected",
     "outcome": "rejected", "rejection_reason": "speech_no_trigger:7,speech_too_short:2"},
    {"ts": BASE + 360, "event_id": "wr0000000002", "trigger_source": "wake_rejected",
     "outcome": "rejected", "rejection_reason": "acoustic_unconfirmed:1,speech_no_trigger:3"},
    {"ts": BASE + 400, "event_id": "nm0000000001", "trigger_source": "near_miss", "outcome": "near_miss",
     "transcript": "Alexei play red", "second_stt_transcript": "Hey Alex, play red",
     "rejection_reason": "second_stt_unconfirmed", "wake_scores": {"alexa": 0.42, "hey_mycroft": 0.1}},
    {"ts": BASE - 7200, "event_id": "old000000001", "trigger_source": "stt", "outcome": "ok"},
]

MEMOIR_LOG = "\n".join([
    "%s,100 INFO    memoir.listener: Wake near-miss rejected: 'Alexei play red' turbo heard 'Hey Alex' "
    "(woke early=True, reason=second_stt_unconfirmed)" % stamp(399),
    "%s,200 INFO    memoir.listener: Wake near-miss rejected: 'Sorry, Alexa.' (woke early=False)" % stamp(410),
    "%s,300 WARNING memoir.brain: Brain unavailable for 'play x' (timed out); using rules" % stamp(121),
    "%s,400 WARNING memoir.brain: Brain answer failed validation for 'play y' (bad); using rules" % stamp(122),
    "%s,500 ERROR   memoir.spotify: Spotify web API ban in effect (HTTP 429 from search, "
    "Retry-After='7200'): calls paused until 2026-10-02 23:00:00 local (2.0 h)" % stamp(100),
    "%s,600 INFO    memoir.spotify: Spotify web API ban lifted (was until 2026-10-02 23:00:00 local); "
    "calls resume" % stamp(500),
    "%s,700 INFO    memoir.health: Health [periodic]: rss=900MB private=2000MB ram_free=3000/16000MB "
    "commit=20000/30000MB(67%%) gpu=5000/8188MB ollama=up spotify_ban=none mic_overflows=0 "
    "echo_ref=alive threads=40 handles=700 up=2h00m" % stamp(450),
    "%s,800 INFO    memoir: [e000000000060] unrelated line" % stamp(61),
    "%s,900 INFO    memoir: cmd id=e000000000060 src=stt outcome=ok" % stamp(62),
    "%s,000 INFO    memoir: too old to count" % stamp(-7300),
]) + "\n"

CRASH_LOG = "\n".join([
    "%s,100 CRITICAL memoir: Assistant stopped on an error" % stamp(200),
    "Traceback (most recent call last):",
    '  File "main.py", line 1, in run',
    "MemoryError",
    "%s,100 CRITICAL memoir: Assistant stopped on an error" % stamp(200),   # handler attached twice
    "Traceback (most recent call last):",
    "MemoryError",
    "%s,200 ERROR   memoir: Spotify web API is rate-limiting this app; retrying in 7200 s" % stamp(250),
    "%s,300 ERROR   memoir: Spotify web API is rate-limiting this app; retrying in 7100 s" % stamp(260),
]) + "\n"


@pytest.fixture()
def logs(tmp_path):
    (tmp_path / "telemetry.jsonl").write_text(
        "\n".join(json.dumps(e) for e in EVENTS) + "\nnot json at all\n", encoding="utf-8")
    (tmp_path / "memoir.log").write_text(MEMOIR_LOG, encoding="utf-8")
    (tmp_path / "crash.log").write_text(CRASH_LOG, encoding="utf-8")
    return tmp_path


def run(capsys, logs, *args):
    report.main(["--logs", str(logs), "--since", stamp(0), "--until", stamp(600), *args])
    return capsys.readouterr().out


def test_parse_when_accepts_the_documented_forms():
    assert report.parse_when("2026-10-02 21:00") == BASE
    assert report.parse_when("2026-10-02 21:00:00") == BASE
    assert report.parse_when("2026-10-02") == time.mktime((2026, 10, 2, 0, 0, 0, 0, 0, -1))
    with pytest.raises(ValueError):
        report.parse_when("yesterday")


def test_load_events_windows_and_skips_bad_lines(logs):
    events = report.load_events(logs / "telemetry.jsonl", BASE, BASE + 600)
    assert len(events) == len(EVENTS) - 1                    # the 2-hours-old one is outside
    assert [e["ts"] for e in events] == sorted(e["ts"] for e in events)


def test_load_log_keeps_traceback_lines_with_their_record(logs):
    records = report.load_log(logs / "crash.log", BASE, BASE + 600)
    critical = [r for r in records if r[1] == "CRITICAL"]
    assert len(critical) == 2 and "MemoryError" in critical[0][3]
    assert all(r[0] >= BASE for r in report.load_log(logs / "memoir.log", BASE, BASE + 600))


def test_percentile_interpolates():
    assert report.percentile([10, 20, 30, 40, 100], 0.5) == 30
    assert report.percentile([10, 20], 0.9) == pytest.approx(19)
    assert report.percentile([], 0.9) == 0.0


def test_command_table_has_one_row_per_command_with_the_key_columns(capsys, logs):
    out = run(capsys, logs)
    table = out.split("COMMANDS")[0]
    rows = [l for l in table.splitlines() if l[:2].isdigit()]
    assert len(rows) == 5                                    # 4 new + 1 legacy; rejections excluded
    first = rows[0]
    assert "final_text" in first and "catalogue:1.02" in first
    assert "'alexa play red by seedhe maut' -> PLAY_TRACK[brain](red b" in first
    assert first.rstrip().endswith("playback_confirmed 1200")   # 2900 - 1700 is the slowest stage
    assert "peek" in rows[1] and rows[1].rstrip().endswith("stt_done 1500")
    assert "hotkey" in rows[2] and "false_accept" in rows[2]
    assert "! Spotify web API is rate-limiting this app" in table   # failure reason under its row
    assert "spotify 90" not in rows[4] and "stt 470" in rows[4]    # legacy: derived from latencies
    assert rows[4].split()[1] == "acoustic"                        # legacy wake source guessed


def test_aggregates(capsys, logs):
    out = run(capsys, logs)
    assert "COMMANDS  5 total, 3 ok, 2 not ok" in out
    assert "error 1" in out and "false_accept 1" in out
    assert "WAKE PATHS  final_text 2, peek 1, hotkey 1, acoustic 1" in out
    # per-stage median / p90 come from the timelines
    stt_line = next(l for l in out.splitlines() if l.strip().startswith("stt_done"))
    assert "n=4" in stt_line and "median    600 ms" in stt_line
    assert "p90" in stt_line
    assert any(l.strip().startswith("TOTAL") for l in out.splitlines())
    assert "NEAR MISSES  2 logged, 1 archived as training clips" in out
    assert ("'Alexei play red' (turbo heard 'Hey Alex, play red') [second_stt_unconfirmed] "
            "oww alexa:0.42,hey_mycroft:0.10") in out
    assert out.count("'Alexei play red'") == 1                  # the archived one is not repeated
    assert "'Sorry, Alexa.' [logged only, no clip]" in out
    assert "speech_no_trigger 10" in out and "speech_too_short 2" in out and "acoustic_unconfirmed 1" in out
    assert "FALSE ACCEPTS  1" in out
    assert "BRAIN  2 commands touched it; 2 fell back to rules (log)" in out
    assert "accepted 1" in out and "validated_away 1" in out and "play_song 2" in out
    assert "validated away: Title 'x' was not heard 1" in out
    assert "RESOLUTION PATHS  catalogue 1, search_fielded 1" in out
    assert "web_api 1" in out and "desktop_uri 1" in out
    assert "ban in effect" in out and "ban lifted" in out


def test_last_health_heartbeat_and_crashes(capsys, logs):
    out = run(capsys, logs)
    assert "LAST HEALTH" in out and "[periodic]" in out
    assert "rss=900MB private=2000MB" in out and "commit=20000/30000MB(67%)" in out
    assert "CRASHES  1 critical, 2 other error line(s) in crash.log" in out    # duplicate collapsed
    assert "CRITICAL Assistant stopped on an error  -> MemoryError" in out
    assert "2x Spotify web API is rate-limiting this app; retrying in N s" in out


def test_window_excludes_old_events_and_limit_truncates(capsys, logs):
    out = run(capsys, logs, "--limit", "2")
    assert "... 3 earlier command(s) not shown" in out
    out = run(capsys, logs)
    assert "old000000001" not in out
    report.main(["--logs", str(logs), "--since", stamp(-7500), "--until", stamp(-7000)])
    assert "COMMANDS  1 total" in capsys.readouterr().out


def test_empty_logs_directory_still_reports(capsys, tmp_path):
    report.main(["--logs", str(tmp_path), "--minutes", "30"])
    out = capsys.readouterr().out
    assert "(no commands in this window)" in out and "COMMANDS  0 total" in out
    assert "LAST HEALTH  (no heartbeat logged in this window)" in out


def test_health_from_before_a_short_window_is_still_shown(capsys, logs):
    report.main(["--logs", str(logs), "--since", stamp(460), "--until", stamp(600)])
    out = capsys.readouterr().out
    assert "before window" in out and "rss=900MB" in out
