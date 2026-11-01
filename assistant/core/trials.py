"""Phase 1 live trial harness (docs/STRATEGY.md, "Phases and gates", phase 1 row).

    python -m assistant.core.trials phase1 [--outage]

The operator says each scripted phrase. The harness tails assistant/logs/core_turns.jsonl
for the next new turn row, shows what was heard and done, and asks y/n/x.
Forbidden effects are detected from the rows and auto-fail the run.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml

from ..config import ROOT

SCRIPT_PATH = ROOT / "docs" / "trials" / "phase1.yaml"
TURNS_PATH = ROOT / "assistant" / "logs" / "core_turns.jsonl"
OUT_DIR = ROOT / "assistant" / "logs" / "trials"

MODES = ("normal", "outage")
CATEGORIES = ("music", "volume", "questions", "negatives")
EXPECTED_COUNTS = {"normal": {"music": 8, "volume": 4, "questions": 4, "negatives": 4},
                   "outage": {"negatives": 4}}
NEGATIVE_KINDS = ("tv", "codex_question", "correction", "truncated")
PREFIX_KINDS = ("correction", "truncated")
EXPECTED_THRESHOLDS = {"music": 7, "volume": 4, "questions": 3, "negatives": 4}
PHASE_GATES = {"phase1b": {"music": (7, 6), "negatives": (3, 3)}, "phase3": {"apps": (5, 5), "onscreen": (5, 5), "codex": (5, 5), "negatives": (5, 5)}, "phase4": {"jam": (3, 3), "questions": (4, 3), "negatives": (3, 3)}}
TIMEOUT_SECONDS = 25
MUSIC_MEDIAN_TARGET_MS = 1500  # STRATEGY target, reported only; not a gate.
TIMINGS = ("wake_to_ack_ms", "speech_end", "stt_ms", "plan_ms",
           "execute_to_observed_ms", "reply_ms")
PROVIDER_NOT_DOWN_MESSAGE = ("Outage run aborted, not scored: the first turn went through the brain "
                             "(path=brain), so the provider was not down. Restart core with "
                             "MEMOIR_CORE_PROVIDER_OUTAGE=1 and rerun.")
CLAIM_RE = re.compile(r"\b(played|playing|paused|pausing|resumed|resuming|skipped|skipping|"
                      r"next track|volume (?:is )?(?:now )?(?:set|at|to|up|down)|"
                      r"turned (?:it )?(?:up|down)|done)\b", re.IGNORECASE)
HEDGE_RE = re.compile(r"\b(can'?t|cannot|couldn'?t|unable|not|asked|confirm|unsure)\b", re.IGNORECASE)


class ScriptError(ValueError):
    pass


@dataclass(frozen=True)
class Line:
    id: str
    mode: str
    category: str
    kind: str
    phrase: str
    note: str = ""
    phase: str = "phase1"
    capability: str = ""


def load_script(path=SCRIPT_PATH, phase="phase1"):
    """Read the script and check it against the phase 1 gate. Raises ScriptError."""
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ScriptError(f"cannot read script {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ScriptError("script must be a mapping")
    if phase != "phase1":
        return _load_extra_script(data, phase)
    if data.get("timeout_seconds") != TIMEOUT_SECONDS:
        raise ScriptError(f"timeout_seconds must be {TIMEOUT_SECONDS}")
    if data.get("thresholds") != EXPECTED_THRESHOLDS:
        raise ScriptError(f"thresholds must match the phase 1 gate: {EXPECTED_THRESHOLDS}")
    raw = data.get("lines")
    if not isinstance(raw, list) or not raw:
        raise ScriptError("lines must be a nonempty list")
    lines, seen = [], set()
    for item in raw:
        if not isinstance(item, dict):
            raise ScriptError("each line must be a mapping")
        missing = [key for key in ("id", "mode", "category", "phrase") if key not in item]
        if missing:
            raise ScriptError(f"line missing fields: {', '.join(missing)}")
        line = Line(id=str(item["id"]), mode=item["mode"], category=item["category"],
                    kind=str(item.get("kind") or item["category"]), phrase=str(item["phrase"]),
                    note=str(item.get("note") or ""))
        if line.id in seen:
            raise ScriptError(f"duplicate line id: {line.id}")
        seen.add(line.id)
        if line.mode not in MODES:
            raise ScriptError(f"{line.id}: mode must be one of {', '.join(MODES)}")
        if line.category not in CATEGORIES:
            raise ScriptError(f"{line.id}: unknown category {line.category}")
        if not line.phrase.strip():
            raise ScriptError(f"{line.id}: empty phrase")
        if line.category == "negatives":
            if line.kind not in NEGATIVE_KINDS:
                raise ScriptError(f"{line.id}: negatives need kind in {', '.join(NEGATIVE_KINDS)}")
        elif line.kind != line.category:
            raise ScriptError(f"{line.id}: kind must equal category for {line.category}")
        lines.append(line)
    for mode in MODES:
        mine = [line for line in lines if line.mode == mode]
        counts = {}
        for line in mine:
            counts[line.category] = counts.get(line.category, 0) + 1
        if counts != EXPECTED_COUNTS[mode]:
            raise ScriptError(f"{mode} script must have {EXPECTED_COUNTS[mode]}, found {counts}")
        kinds = sorted(line.kind for line in mine if line.category == "negatives")
        if kinds != sorted(NEGATIVE_KINDS):
            raise ScriptError(f"{mode} negatives must cover each of {', '.join(NEGATIVE_KINDS)} once")
    return {"timeout_seconds": TIMEOUT_SECONDS, "thresholds": dict(EXPECTED_THRESHOLDS), "lines": lines}


def _load_extra_script(data, phase):
    if phase not in PHASE_GATES:
        raise ScriptError(f"unknown phase: {phase}")
    gate = PHASE_GATES[phase]
    thresholds = {category: required for category, (_, required) in gate.items()}
    if (type(data.get("timeout_seconds")) is not int or data["timeout_seconds"] != TIMEOUT_SECONDS
            or data.get("thresholds") != thresholds or any(type(value) is not int
                for value in (data.get("thresholds") or {}).values())):
        raise ScriptError(f"invalid timeout or thresholds for {phase}: {thresholds}")
    raw = data.get("lines")
    if not isinstance(raw, list) or not raw:
        raise ScriptError("lines must be a nonempty list")
    lines, seen = [], set()
    for item in raw:
        if not isinstance(item, dict) or any(not isinstance(item.get(key), str) or not item[key].strip()
                                             for key in ("id", "mode", "category", "phrase")):
            raise ScriptError("each line needs nonempty id, mode, category and phrase")
        category = item["category"]
        if item["id"] in seen or item["mode"] != "normal" or category not in gate:
            raise ScriptError(f"invalid or duplicate line: {item['id']}")
        kind = item.get("kind", category)
        capability = item.get("capability", "")
        if not isinstance(kind, str) or not isinstance(capability, str) or not isinstance(item.get("note", ""), str):
            raise ScriptError(f"invalid fields: {item['id']}")
        if category == "negatives":
            allowed = {"phase1b": {"action"}, "phase3": {"codex_question", "correction", "hypothetical", "reported", "truncated"},
                       "phase4": {"privacy"}}[phase]
            if kind not in allowed or capability:
                raise ScriptError(f"invalid negative: {item['id']}")
        elif kind != category or (category != "questions" and not capability) or (category == "questions" and capability):
            raise ScriptError(f"invalid positive: {item['id']}")
        permitted = {"phase1b": {"music": {"play_album", "play_artist", "play_playlist", "queue_song", "previous_track", "restart_track", "restart_album"}},
                     "phase3": {"apps": {"open_app", "search_browser"}, "onscreen": {"show_track", "show_album"}, "codex": {"codex_request"}},
                     "phase4": {"jam": {"start_jam"}}}
        if category not in {"questions", "negatives"} and capability not in permitted[phase][category]:
            raise ScriptError(f"invalid capability: {item['id']}")
        seen.add(item["id"])
        lines.append(Line(item["id"], "normal", category, kind, item["phrase"], item.get("note", ""), phase, capability))
    for category, (count, _) in gate.items():
        if sum(line.category == category for line in lines) != count:
            raise ScriptError(f"{phase} requires {count} {category} lines")
    if phase == "phase3" and {line.kind for line in lines if line.category == "negatives"} != {
            "codex_question", "correction", "hypothetical", "reported", "truncated"}:
        raise ScriptError("phase3 must cover all five Codex negative kinds")
    if phase == "phase1b" and len({line.capability for line in lines if line.category == "music"}) != 7:
        raise ScriptError("phase1b must cover all seven music capabilities")
    return {"timeout_seconds": TIMEOUT_SECONDS, "thresholds": thresholds, "lines": lines}


class TurnTail:
    """Reads only rows appended after mark(). Partial trailing lines wait for their newline."""

    def __init__(self, path):
        self.path = Path(path)
        self._offset = 0
        self._buffer = b""
        self._pending = []

    def _size(self):
        try:
            return self.path.stat().st_size
        except FileNotFoundError:
            return 0

    def mark(self):
        self._offset = self._size()
        self._buffer = b""
        self._pending = []

    def _read(self):
        size = self._size()
        if size < self._offset:
            self._offset, self._buffer = 0, b""
        if size <= self._offset:
            return []
        with self.path.open("rb") as stream:
            stream.seek(self._offset)
            chunk = stream.read(size - self._offset)
        self._offset += len(chunk)
        *complete, self._buffer = (self._buffer + chunk).split(b"\n")
        rows = []
        for raw in complete:
            if not raw.strip():
                continue
            try:
                value = json.loads(raw.decode("utf-8"))
            except ValueError:
                continue
            if isinstance(value, dict):
                rows.append(value)
        return rows

    def next_row(self, timeout, *, clock=time.monotonic, sleep=time.sleep, poll=0.25):
        """Next new row, or None when nothing arrives within timeout seconds."""
        if self._pending:
            return self._pending.pop(0)
        deadline = clock() + timeout
        while True:
            self._pending.extend(self._read())
            if self._pending:
                return self._pending.pop(0)
            if clock() >= deadline:
                return None
            sleep(poll)


def claims_success(text):
    return bool(CLAIM_RE.search(text or "")) and not HEDGE_RE.search(text or "")


def row_forbidden(row, line):
    """Forbidden-effect reasons visible in one turn row for the scripted line."""
    reasons = []
    summary = row.get("decision_summary") or {}
    capabilities = list(summary.get("capabilities") or []) if isinstance(summary, dict) else []
    receipts = [item for item in (row.get("receipts") or []) if isinstance(item, dict)]
    capabilities += [str(item.get("capability", "")) for item in receipts]
    if (line.phase != "phase3" or line.category != "codex") and any("codex" in str(name).lower() for name in capabilities):
        reasons.append("codex_triggered")
    if line.phase == "phase4" and line.kind == "privacy" and row.get("knowledge_source"):
        reasons.append("privacy_retrieval")
    if line.category == "negatives" and receipts:
        reasons.append("prefix_executed" if line.kind in PREFIX_KINDS else "unrequested_effect")
    if receipts and not any(item.get("state") == "observed" for item in receipts) \
            and claims_success(row.get("reply")):
        reasons.append("unobserved_success_claim")
    return reasons


def score_categories(lines, outcomes, thresholds):
    result = {}
    for category in thresholds:
        mine = [line for line in lines if line.category == category]
        if not mine:
            continue
        passed = sum(1 for line in mine if outcomes.get(line.id))
        required = thresholds[category]
        result[category] = {"passed": passed, "total": len(mine), "required": required,
                            "ok": passed >= required}
    return result


def nearest_rank(values, quantile):
    ordered = sorted(values)
    return ordered[max(1, math.ceil(quantile * len(ordered))) - 1]


def latency_summary(rows):
    result = {}
    for stage in TIMINGS:
        values = [row["timings"][stage] for row in rows
                  if isinstance(row.get("timings"), dict)
                  and isinstance(row["timings"].get(stage), (int, float))
                  and not isinstance(row["timings"].get(stage), bool)]
        result[stage] = {"n": len(values),
                         "median_ms": statistics.median(values) if values else None,
                         "p95_ms": nearest_rank(values, 0.95) if values else None}
    return result


def _ask_verdict(ask, out):
    while True:
        answer = str(ask("Correct? y = yes, n = no, x = invalid attempt: ")).strip().lower()
        if answer in {"y", "n", "x"}:
            return answer
        out("  Enter y, n or x.")


def _show(row, out):
    summary = row.get("decision_summary") or {}
    kind = summary.get("kind") if isinstance(summary, dict) else None
    caps = ", ".join(summary.get("capabilities") or []) if isinstance(summary, dict) else ""
    receipts = "; ".join(f"{item.get('capability')}={item.get('state')}"
                         for item in (row.get("receipts") or []) if isinstance(item, dict)) or "none"
    out(f"  Heard: {row.get('transcript', '')!r}")
    out(f"  Path: {row.get('path')} · decision: {kind} [{caps}] · receipts: {receipts} · "
        f"outcome: {row.get('outcome')}")
    if row.get("reply"):
        out(f"  Reply: {row['reply']}")
    if row.get("provider_error"):
        out(f"  Provider error: {row['provider_error']}")


def _fmt(value):
    return "-" if value is None else f"{value:.0f}"


def format_scorecard(summary):
    lines = [f"{summary.get('phase', 'phase1').replace('phase', 'Phase ')} scorecard · mode={summary['mode']} · {summary['started_at']}"]
    for name, cat in summary["categories"].items():
        verdict = "PASS" if cat["ok"] else "FAIL"
        lines.append(f"  {name:<10} {cat['passed']}/{cat['total']}  need {cat['required']}  {verdict}")
    forbidden = ", ".join(f"{item['line_id']}={item['reason']}" for item in summary["forbidden"])
    lines.append(f"  forbidden: {forbidden or 'none'}")
    if summary.get("abort_reason") == "provider not down":
        lines.insert(1, f"  {PROVIDER_NOT_DOWN_MESSAGE}")
    elif summary["aborted"]:
        lines.append(f"  run stopped early ({summary.get('abort_reason')}); "
                     "unattempted lines count as misses")
    if summary["interrupted"]:
        lines.append("  run interrupted")
    lines.append("  latency (ms, units as logged)")
    for stage, stat in summary["latency_ms"].items():
        lines.append(f"    {stage:<24} n={stat['n']:<3} median={_fmt(stat['median_ms'])} "
                     f"p95={_fmt(stat['p95_ms'])}")
    target = summary["targets"]["music_execute_to_observed_median_ms"]
    lines.append(f"  music execute_to_observed median {_fmt(target['median'])} "
                 f"(target <= {target['target']}, not a gate): {'met' if target['met'] else 'not met'}")
    lines.append(f"RESULT: {'PASS' if summary['pass'] else 'FAIL'}")
    lines.append(f"  rows: {summary['rows_file']}")
    lines.append(f"  summary: {summary['summary_file']}")
    return "\n".join(lines)


def run_phase(*, mode, script_path=None, phase="phase1", turns_path=TURNS_PATH, out_dir=OUT_DIR,
              ask=input, out=print, clock=time.monotonic, sleep=time.sleep,
              now=datetime.now):
    if mode not in MODES:
        raise ValueError(f"mode must be one of {', '.join(MODES)}")
    script_path = script_path or (SCRIPT_PATH if phase == "phase1" else ROOT / "docs" / "trials" / f"{phase}.yaml")
    script = load_script(script_path, phase)
    lines = [line for line in script["lines"] if line.mode == mode]
    thresholds = script["thresholds"]
    stamp = now().strftime("%Y%m%dT%H%M%S")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / f"{phase}-{stamp}.jsonl"
    summary_path = out_dir / f"{phase}-{stamp}-summary.json"
    env_state = "MEMOIR_CORE_PROVIDER_OUTAGE=1" if mode == "outage" else "MEMOIR_CORE_PROVIDER_OUTAGE unset"
    started = now().isoformat()
    ask(f"Restart core with {env_state}. Press Enter when it is running: ")

    tail = TurnTail(turns_path)
    outcomes = {}
    entries = []
    forbidden = []
    state = {"attempt": 0, "aborted": False, "interrupted": False, "provider_checked": False,
             "abort_reason": None}

    with rows_path.open("w", encoding="utf-8") as stream:
        def record(entry):
            stream.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
            stream.flush()
            entries.append(entry)

        try:
            for position, line in enumerate(lines, 1):
                outcomes[line.id] = False
                invalid_used = False
                while True:
                    state["attempt"] += 1
                    tail.mark()
                    label = f"{line.category} ({line.kind})" if line.category == "negatives" else line.category
                    out(f"\n[{line.id}] line {position}/{len(lines)} · {label}")
                    out(f"  Say: {line.phrase}")
                    if line.note:
                        out(f"  Note: {line.note}")
                    row = tail.next_row(TIMEOUT_SECONDS, clock=clock, sleep=sleep)
                    entry = {"attempt": state["attempt"], "mode": mode, "position": position,
                             "line_id": line.id, "category": line.category, "kind": line.kind,
                             "phrase": line.phrase, "timeout": row is None,
                             "turn_id": (row or {}).get("turn_id"), "verdict": None,
                             "counted": True, "passed": False, "forbidden": [], "row": row}
                    if row is None:
                        if line.category == "negatives" and line.kind != "privacy":
                            entry.update(verdict="auto", passed=True)
                            outcomes[line.id] = True
                            out(f"  No new turn within {TIMEOUT_SECONDS} s: negative passes "
                                "automatically (no effect possible).")
                        else:
                            out(f"  No new turn within {TIMEOUT_SECONDS} s: miss.")
                        record(entry)
                        break
                    entry["forbidden"] = row_forbidden(row, line)
                    _show(row, out)
                    if entry["forbidden"]:
                        out(f"  Forbidden effect: {', '.join(entry['forbidden'])}. Run stops.")
                        forbidden.extend({"line_id": line.id, "reason": reason, "turn_id": row.get("turn_id")}
                                         for reason in entry["forbidden"])
                        record(entry)
                        state["aborted"], state["abort_reason"] = True, "forbidden effect"
                        break
                    if mode == "outage" and not state["provider_checked"]:
                        state["provider_checked"] = True
                        if row.get("path") == "brain":
                            out(f"  {PROVIDER_NOT_DOWN_MESSAGE}")
                            entry["counted"] = False
                            record(entry)
                            state["aborted"], state["abort_reason"] = True, "provider not down"
                            break
                    verdict = _ask_verdict(ask, out)
                    entry["verdict"] = verdict
                    if verdict == "x" and not invalid_used:
                        invalid_used = True
                        entry["counted"] = False
                        record(entry)
                        out("  Invalid attempt: re-asking once.")
                        continue
                    entry["passed"] = verdict == "y" and (line.phase != "phase4" or line.category != "questions"
                        or bool(row.get("knowledge_source"))) and (line.category != "codex" or any(
                        receipt.get("capability") == "codex_request" and receipt.get("state") == "observed"
                        for receipt in row.get("receipts", []) if isinstance(receipt, dict)))
                    outcomes[line.id] = entry["passed"]
                    record(entry)
                    break
                if state["aborted"]:
                    break
        except (KeyboardInterrupt, EOFError):
            state["interrupted"] = True

    categories = score_categories(lines, outcomes, thresholds)
    counted_rows = [entry["row"] for entry in entries if entry["counted"] and entry["row"]]
    music_values = [entry["row"]["timings"].get("execute_to_observed_ms") for entry in entries
                    if entry["counted"] and entry["row"] and entry["category"] == "music"]
    music_values = [value for value in music_values if isinstance(value, (int, float))]
    median = statistics.median(music_values) if music_values else None
    targets = {"music_execute_to_observed_median_ms": {
        "target": MUSIC_MEDIAN_TARGET_MS, "median": median,
        "met": median is not None and median <= MUSIC_MEDIAN_TARGET_MS}}
    if state["abort_reason"] == "provider not down":
        categories = {}  # not scored
    ok = (state["abort_reason"] is None and all(item["ok"] for item in categories.values())
          and not forbidden and not state["aborted"] and not state["interrupted"])
    summary = {"phase": phase, "run": mode, "mode": mode, "provider_env": env_state, "started_at": started,
               "finished_at": now().isoformat(), "script": str(script_path),
               "rows_file": str(rows_path), "summary_file": str(summary_path),
               "attempts": state["attempt"], "categories": categories, "forbidden": forbidden,
               "aborted": state["aborted"], "abort_reason": state["abort_reason"],
               "interrupted": state["interrupted"],
               "latency_ms": latency_summary(counted_rows), "targets": targets, "pass": ok}
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n",
                            encoding="utf-8")
    out(format_scorecard(summary))
    return summary


def _read_summary(path, mode):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("mode") != mode or not isinstance(data.get("pass"), bool):
        raise ValueError(f"{path} is not a {mode} run summary")
    return data


def combine(normal_path, outage_path):
    """Phase 1 verdict: PASS only when the normal and the outage run both pass."""
    normal = _read_summary(normal_path, "normal")
    outage = _read_summary(outage_path, "outage")
    ok = normal["pass"] and outage["pass"]
    text = "\n".join([
        f"PHASE 1 {'PASS' if ok else 'FAIL'}",
        f"  normal run: {'PASS' if normal['pass'] else 'FAIL'} ({normal_path})",
        f"  outage run: {'PASS' if outage['pass'] else 'FAIL'} ({outage_path})"])
    return ok, text


def main(argv=None, **kwargs):
    parser = argparse.ArgumentParser(prog="python -m assistant.core.trials")
    parser.add_argument("--combine", nargs=2, metavar=("NORMAL_SUMMARY", "OUTAGE_SUMMARY"),
                        help="print the combined phase 1 verdict from two run summaries")
    commands = parser.add_subparsers(dest="command")
    phase = commands.add_parser("phase1", help="scripted phase 1 gate trial")
    phase.add_argument("--outage", action="store_true",
                       help="run the four negatives with the provider blocked")
    for name in PHASE_GATES:
        commands.add_parser(name, help=f"scripted {name} gate trial")
    args = parser.parse_args(argv)
    if args.combine:
        try:
            ok, text = combine(*args.combine)
        except (OSError, ValueError) as exc:
            print(f"combine error: {exc}", file=sys.stderr)
            return 2
        print(text)
        return 0 if ok else 1
    if args.command not in ("phase1", *PHASE_GATES):
        parser.error("choose a phase or --combine")
    try:
        summary = run_phase(mode="outage" if getattr(args, "outage", False) else "normal", phase=args.command, **kwargs)
    except ScriptError as exc:
        print(f"script error: {exc}", file=sys.stderr)
        return 2
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
