"""Terminal report on recent commands: what was heard, where the time went, why.

    python -m assistant.report                          # last 60 minutes
    python -m assistant.report --minutes 240
    python -m assistant.report --since "2026-10-02 21:00"   # local time, to now
    python -m assistant.report --since "2026-10-02 21:00" --until "2026-10-02 22:30"

Reads assistant/logs/telemetry.jsonl, memoir.log and crash.log (read-only).
Prints a per-command table (wake path, heard -> intent, how the song was
resolved, outcome, total time, slowest stage), then aggregates: outcomes,
per-stage median/p90, wake paths, near misses, rejections, false accepts,
brain usage, playback routes, the last health heartbeat and crashes.
Plain text, standard library only. Older telemetry lines (without the
timeline/resolution fields) still load; their columns just show "-".
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional

try:
    from .diag import STAGES, stage_deltas
except ImportError:  # run as a plain script
    from diag import STAGES, stage_deltas  # type: ignore

LOGS = Path(__file__).resolve().parent / "logs"

COMMAND_SOURCES = {"wake", "stt", "hotkey"}
# Stages that only anchor the timeline; they are never "the slowest stage".
ANCHORS = {"speech_start", "wake_evidence"}
LINE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)[,.]\d+ +(\w+) +([\w.]+): ?(.*)$")
# The transcript is logged with %r, so it is a Python string literal.
NEAR_MISS_LINE = re.compile(
    r"""Wake near-miss rejected: ('(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*")""")
HEALTH_LINE = re.compile(r"Health \[(\w+)\]: (.*)$")


# ------------------------------------------------------------------ loading

def parse_when(text: str) -> float:
    """Local "YYYY-MM-DD HH:MM[:SS]" (or just a date) to epoch seconds."""
    text = text.strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%dT%H:%M", "%Y-%m-%d"):
        try:
            return time.mktime(time.strptime(text, fmt))
        except ValueError:
            continue
    raise ValueError("cannot read time {!r}; use e.g. \"2026-10-02 21:00\"".format(text))


def load_events(path: Path, since: float, until: float) -> list:
    """Telemetry records with since <= ts <= until, oldest first. Bad lines are skipped."""
    events = []
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if isinstance(event, dict) and since <= event.get("ts", 0) <= until:
                    events.append(event)
    except OSError:
        pass
    events.sort(key=lambda e: e.get("ts", 0))
    return events


def load_log(path: Path, since: float, until: float) -> list:
    """[(epoch, level, logger, message)] in the window; traceback lines join the
    message of the record they belong to."""
    records = []
    try:
        handle = open(path, encoding="utf-8", errors="replace")
    except OSError:
        return records
    with handle:
        current = None
        for raw in handle:
            raw = raw.rstrip("\r\n")
            match = LINE.match(raw)
            if match:
                try:
                    moment = time.mktime(time.strptime(match.group(1), "%Y-%m-%d %H:%M:%S"))
                except (ValueError, OverflowError):
                    current = None
                    continue
                current = [moment, match.group(2), match.group(3), match.group(4)] \
                    if since <= moment <= until else None
                if current is not None:
                    records.append(current)
            elif current is not None:
                current[3] += "\n" + raw
    return [tuple(r) for r in records]


# --------------------------------------------------------------- statistics

def percentile(values: list, fraction: float) -> float:
    """Linear-interpolated percentile (values need not be sorted)."""
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _ms(value) -> str:
    return "-" if value is None else "{:.0f}".format(value)


def _clip(text, width: int) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= width else text[:max(0, width - 2)] + ".."


def legacy_wake_path(event: dict) -> str:
    """wake_path for telemetry written before the field existed (best guess)."""
    if event.get("wake_path"):
        return event["wake_path"]
    source = event.get("trigger_source")
    if source == "wake":
        return "acoustic"
    if source == "hotkey":
        return "hotkey"
    if source == "stt":
        return {"second_stt": "second_stt", "acoustic": "alias_acoustic"}.get(
            event.get("trigger_confirmation_mode") or "", "final_text")
    return ""


def command_stages(event: dict) -> list:
    """[(stage, ms)] for one command: its timeline, or the legacy latency fields."""
    deltas = [(name, ms) for name, ms in stage_deltas(event.get("timeline"))]
    if deltas:
        return deltas
    legacy = (("stt", event.get("stt_latency_ms")), ("intent", event.get("intent_latency_ms")),
              ("spotify", event.get("spotify_latency_ms")))
    return [(name, ms) for name, ms in legacy if isinstance(ms, (int, float))]


def total_ms(event: dict) -> Optional[float]:
    timeline = event.get("timeline") or {}
    for key in ("handler_done", "feedback_done"):
        if isinstance(timeline.get(key), (int, float)):
            return timeline[key]
    return event.get("handler_latency_ms")


def slowest_stage(event: dict) -> str:
    stages = [(name, ms) for name, ms in command_stages(event) if name not in ANCHORS]
    if not stages:
        return "-"
    name, ms = max(stages, key=lambda item: item[1])
    return "{} {:.0f}".format(name, ms)


def _decided(event: dict) -> str:
    action = event.get("intent_action") or "-"
    query = event.get("intent_query") or ""
    value = event.get("intent_value")
    brain = (event.get("brain_tool") or str(event.get("matched_rule") or "").startswith("brain_"))
    return "{}{}({}{})".format(action, "[brain]" if brain else "", query,
                               "" if value is None or query else value)


def _resolution(event: dict) -> str:
    path = event.get("resolution_path") or ""
    if not path:
        return "-"
    score = event.get("resolution_score")
    return path if score is None else "{}:{:.2f}".format(path, score)


# ------------------------------------------------------------------- report

def _table(events: list, limit: int) -> list:
    commands = [e for e in events if e.get("trigger_source") in COMMAND_SOURCES]
    lines = ["{:8} {:14} {:60} {:28} {:14} {:>6}  {}".format(
        "time", "wake_path", "heard -> decided", "resolution", "outcome", "total", "slowest stage")]
    omitted = max(0, len(commands) - limit)
    if omitted:
        lines.append("... {} earlier command(s) not shown (use --limit)".format(omitted))
    for e in commands[omitted:]:
        heard = e.get("retry_transcript") or e.get("transcript") or ""
        lines.append("{:8} {:14} {:60} {:28} {:14} {:>6}  {}".format(
            time.strftime("%H:%M:%S", time.localtime(e["ts"])),
            legacy_wake_path(e) or "-",
            _clip("{!r} -> {}".format(heard, _decided(e)), 60),
            _clip(_resolution(e), 28),
            _clip(e.get("outcome") or "-", 14),
            _ms(total_ms(e)),
            slowest_stage(e)))
        # A failure is worth its reason on the next line.
        if e.get("outcome") not in ("ok", "false_accept", None, "") and e.get("error"):
            lines.append("{:8} {}".format("", _clip("! " + e["error"], 128)))
    if not commands:
        lines.append("(no commands in this window)")
    return lines


def _counts(counter: Counter, empty: str = "(none)") -> str:
    if not counter:
        return empty
    return ", ".join("{} {}".format(name, count) for name, count in counter.most_common())


def _stage_stats(commands: list) -> list:
    per_stage: dict = {}
    for e in commands:
        for name, ms in command_stages(e):
            if name not in ANCHORS:
                per_stage.setdefault(name, []).append(ms)
    order = {name: i for i, name in enumerate(STAGES)}
    lines = []
    for name in sorted(per_stage, key=lambda n: (order.get(n, len(order)), n)):
        values = per_stage[name]
        lines.append("  {:20} n={:<4} median {:>6.0f} ms   p90 {:>6.0f} ms   max {:>6.0f} ms".format(
            name, len(values), statistics.median(values), percentile(values, 0.9), max(values)))
    return lines


def _reason_counts(events: list) -> Counter:
    counts: Counter = Counter()
    for e in events:
        if e.get("trigger_source") != "wake_rejected":
            continue
        for part in str(e.get("rejection_reason") or "").split(","):
            name, _, number = part.partition(":")
            if name:
                counts[name] += int(number) if number.isdigit() else 1
    return counts


def _crashes(records: list) -> list:
    """Crash entries: CRITICAL records, plus the exception line of any traceback."""
    found, seen = [], set()
    for moment, level, logger, message in records:
        if level not in ("CRITICAL", "ERROR") and "Traceback" not in message:
            continue
        head = message.split("\n", 1)[0]
        tail = [line for line in message.split("\n")[1:] if line.strip()]
        cause = tail[-1].strip() if tail and "Traceback" in message else ""
        key = (int(moment), head, cause)
        if key in seen:        # the crash log's handler is attached twice
            continue
        seen.add(key)
        found.append((moment, level, head, cause))
    return found


def build_report(events: list, log_records: list, crash_records: list, since: float,
                 until: float, limit: int = 50, last_health: Optional[tuple] = None) -> str:
    out = []
    stamp = lambda t: time.strftime("%Y-%m-%d %H:%M", time.localtime(t))
    out.append("Memoir report  {} .. {}".format(stamp(since), stamp(until)))
    out.append("")
    out.extend(_table(events, limit))

    commands = [e for e in events if e.get("trigger_source") in COMMAND_SOURCES]
    ok = sum(e.get("outcome") == "ok" for e in commands)
    outcomes = Counter(e.get("outcome") or "unknown" for e in commands)
    out.append("")
    out.append("COMMANDS  {} total, {} ok, {} not ok".format(len(commands), ok, len(commands) - ok))
    failed = Counter({k: v for k, v in outcomes.items() if k != "ok"})
    out.append("  not ok by outcome: " + _counts(failed))

    out.append("")
    out.append("TIME PER STAGE (ms since the previous stage; origin = first wake evidence or speech start)")
    stage_lines = _stage_stats(commands)
    out.extend(stage_lines or ["  (no timeline data yet)"])
    totals = [t for t in (total_ms(e) for e in commands) if t is not None]
    if totals:
        out.append("  {:20} n={:<4} median {:>6.0f} ms   p90 {:>6.0f} ms   max {:>6.0f} ms".format(
            "TOTAL", len(totals), statistics.median(totals), percentile(totals, 0.9), max(totals)))

    out.append("")
    out.append("WAKE PATHS  " + _counts(Counter(legacy_wake_path(e) or "unknown" for e in commands)))

    near = [e for e in events if e.get("trigger_source") == "near_miss"]
    heard = []
    for _, _, _, message in log_records:
        match = NEAR_MISS_LINE.search(message)
        if match:
            try:
                heard.append(ast.literal_eval(match.group(1)))
            except (ValueError, SyntaxError):
                heard.append(match.group(1))
    out.append("")
    out.append("NEAR MISSES  {} logged, {} archived as training clips".format(len(heard), len(near)))
    archived = {e.get("transcript") for e in near}
    for e in near:
        label = "{!r}".format(e.get("transcript") or "")
        if e.get("second_stt_transcript"):
            label += " (turbo heard {!r})".format(e["second_stt_transcript"])
        label += " [{}]".format(e.get("rejection_reason") or "-")
        scores = e.get("wake_scores") or {}
        if scores:
            label += " oww " + ",".join("{}:{:.2f}".format(m, s) for m, s in sorted(scores.items()))
        out.append("  " + _clip(label, 128))
    for text in [t for t in dict.fromkeys(heard) if t not in archived][:10]:
        out.append("  " + _clip("{!r} [logged only, no clip]".format(text), 128))

    out.append("")
    out.append("REJECTIONS BY REASON  " + _counts(_reason_counts(events)))

    false_accepts = [e for e in commands if e.get("outcome") == "false_accept"]
    out.append("")
    out.append("FALSE ACCEPTS  {} (wake accepted, no command followed)".format(len(false_accepts)))
    for e in false_accepts[:8]:
        out.append("  {} {:14} {!r}".format(time.strftime("%H:%M:%S", time.localtime(e["ts"])),
                                            legacy_wake_path(e) or "-",
                                            _clip(e.get("transcript") or "", 80)))

    brain_events = [e for e in commands if e.get("brain_tool") or e.get("brain_outcome")
                    or str(e.get("matched_rule") or "").startswith("brain_")]
    fallbacks = sum(1 for _, _, _, m in log_records
                    if "Brain unavailable" in m or "Brain answer failed validation" in m)
    out.append("")
    out.append("BRAIN  {} commands touched it; {} fell back to rules (log)".format(
        len(brain_events), fallbacks))
    out.append("  outcomes: " + _counts(Counter(
        (e.get("brain_outcome") or "unrecorded").split(":", 1)[0] for e in brain_events)))
    out.append("  tools: " + _counts(Counter(e.get("brain_tool") for e in brain_events
                                             if e.get("brain_tool"))))
    latencies = [e["brain_latency_ms"] for e in brain_events
                 if isinstance(e.get("brain_latency_ms"), (int, float))]
    if latencies:
        out.append("  latency: median {:.0f} ms, p90 {:.0f} ms".format(
            statistics.median(latencies), percentile(latencies, 0.9)))
    reasons = Counter((e.get("brain_outcome") or "").partition(":")[2][:60]
                      for e in brain_events if (e.get("brain_outcome") or "").startswith("validated_away"))
    if reasons:
        out.append("  validated away: " + _counts(reasons))

    out.append("")
    out.append("RESOLUTION PATHS  " + _counts(Counter(
        e["resolution_path"] for e in commands if e.get("resolution_path"))))
    out.append("PLAYBACK METHODS  " + _counts(Counter(
        e["playback_method"] for e in commands if e.get("playback_method"))))

    bans = [(m, text.split("\n", 1)[0]) for m, _, _, text in log_records
            if "API ban in effect" in text or "API ban lifted" in text]
    out.append("SPOTIFY BANS      " + (
        "; ".join("{} {}".format(time.strftime("%H:%M", time.localtime(m)),
                                 _clip(t.replace("Spotify ", ""), 90)) for m, t in bans[-3:])
        if bans else "(none logged)"))

    out.append("")
    if last_health is None:
        for moment, _, _, message in reversed(log_records):
            match = HEALTH_LINE.search(message)
            if match:
                last_health = (moment, match.group(1), match.group(2))
                break
    if last_health:
        out.append("LAST HEALTH  {} [{}]".format(stamp(last_health[0]), last_health[1]))
        out.append("  " + last_health[2])
    else:
        out.append("LAST HEALTH  (no heartbeat logged in this window)")

    crashes = _crashes(crash_records)
    critical = [c for c in crashes if c[1] == "CRITICAL" or c[3]]
    errors = Counter(_clip(re.sub(r"\d+", "N", c[2]), 100) for c in crashes if c not in critical)
    out.append("")
    out.append("CRASHES  {} critical, {} other error line(s) in crash.log".format(
        len(critical), sum(errors.values())))
    for moment, level, head, cause in critical[-10:]:
        out.append("  {} {} {}{}".format(time.strftime("%H:%M:%S", time.localtime(moment)), level,
                                         _clip(head, 70), "  -> " + _clip(cause, 60) if cause else ""))
    for text, count in errors.most_common(5):
        out.append("  {}x {}".format(count, text))
    return "\n".join(out)


def main(argv: Optional[Iterable[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Summarise recent Memoir commands and health.")
    ap.add_argument("--minutes", type=float, default=60, help="window length (default 60)")
    ap.add_argument("--since", help='window start, local time, e.g. "2026-10-02 21:00"')
    ap.add_argument("--until", help="window end, local time (default: now)")
    ap.add_argument("--limit", type=int, default=50, help="rows in the command table")
    ap.add_argument("--logs", type=Path, default=LOGS, help="directory holding the logs")
    args = ap.parse_args(list(argv) if argv is not None else None)

    until = parse_when(args.until) if args.until else time.time()
    since = parse_when(args.since) if args.since else until - args.minutes * 60
    events = load_events(args.logs / "telemetry.jsonl", since, until)
    records = load_log(args.logs / "memoir.log", since, until)
    crashes = load_log(args.logs / "crash.log", since, until)
    # The newest heartbeat may predate a short window; show it anyway.
    last_health = None
    if not any(HEALTH_LINE.search(m) for _, _, _, m in records):
        older = load_log(args.logs / "memoir.log", since - 7 * 86400, until)
        for moment, _, _, message in reversed(older):
            match = HEALTH_LINE.search(message)
            if match:
                last_health = (moment, match.group(1) + ", before window", match.group(2))
                break
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    print(build_report(events, records, crashes, since, until, args.limit, last_health))


if __name__ == "__main__":
    main()
