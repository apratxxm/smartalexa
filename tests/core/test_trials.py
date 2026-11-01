import json
from datetime import datetime
from pathlib import Path

import pytest
import yaml

from assistant.core.trials import (
    SCRIPT_PATH, TIMINGS, ScriptError, TurnTail, claims_success, latency_summary,
    load_script, main, nearest_rank, row_forbidden, run_phase, score_categories)


def row(turn_id, transcript="", *, caps=(), receipts=(), reply="", timings=None, path="brain"):
    # Synthetic turn row; path defaults to brain, outage rows use fallback.
    return {"turn_id": turn_id, "transcript": transcript, "path": path,
            "decision_summary": {"kind": "plan", "capabilities": list(caps)},
            "receipts": [{"state": state, "capability": cap, "detail": "", "operation_id": f"op{i}"}
                         for i, (cap, state) in enumerate(receipts)],
            "timings": {**{name: None for name in TIMINGS}, **(timings or {})},
            "reply": reply, "outcome": "completed"}


class FakeCore:
    """Writes the next scripted row to the turns file when the harness waits.

    None means the turn never produces a row: the fake clock jumps past the timeout.
    """

    def __init__(self, path, items):
        self.path = Path(path)
        self.items = list(items)
        self.now = 0.0

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds
        if not self.items:
            return
        item = self.items.pop(0)
        if item is None:
            self.now += 60
            return
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(item) + "\n")


def default_row(line):
    item = _base_row(line)
    item["path"] = "fallback" if line.mode == "outage" else "brain"
    return item


def _base_row(line):
    if line.category == "music":
        return row("t-" + line.id, line.phrase, caps=["play_song"], receipts=[("play_song", "observed")],
                   reply="Playing Halo.", timings={"execute_to_observed_ms": 900, "reply_ms": 300})
    if line.category == "volume":
        return row("t-" + line.id, line.phrase, caps=["set_volume"],
                   receipts=[("set_volume", "observed")], reply="Volume set to 40.")
    if line.category == "questions":
        return row("t-" + line.id, line.phrase, caps=["answer"], reply="It is noon.")
    return row("t-" + line.id, line.phrase, reply="")


def scripted(lines, *, rows=None, verdicts=None):
    """Build fake-core items and operator answers in script order.

    A row of None is a silent turn: no answer is consumed for it.
    """
    rows = rows or {}
    verdicts = verdicts or {}
    items, answers = [], []
    for line in lines:
        item = rows[line.id] if line.id in rows else default_row(line)
        items.append(item)
        if item is not None:
            answers.extend(verdicts.get(line.id, ["y"]))
    return items, answers


class Operator:
    def __init__(self, answers):
        self.answers = list(answers)
        self.prompts = []

    def __call__(self, prompt=""):
        self.prompts.append(prompt)
        if not self.answers:
            raise AssertionError("operator asked more often than scripted")
        return self.answers.pop(0)


def run(tmp_path, mode, items, answers, *, script=SCRIPT_PATH, stamp=datetime(2026, 10, 27, 9, 0, 0)):
    # Runs that share out_dir must use different stamps: filenames are timestamped.
    core = FakeCore(tmp_path / "core_turns.jsonl", items)
    messages = []
    operator = Operator([""] + answers)
    summary = run_phase(mode=mode, script_path=script, turns_path=core.path,
                        out_dir=tmp_path / "out", ask=operator,
                        out=messages.append, clock=core.clock, sleep=core.sleep,
                        now=lambda: stamp)
    run.prompts = operator.prompts
    return summary, messages


def lines_for(mode):
    return [line for line in load_script()["lines"] if line.mode == mode]


def rows_file(summary):
    return [json.loads(line) for line in Path(summary["rows_file"]).read_text(encoding="utf-8").splitlines()]


# script composition against the phase 1 gate

def test_script_matches_phase_one_gate():
    script = load_script()
    normal = [line for line in script["lines"] if line.mode == "normal"]
    outage = [line for line in script["lines"] if line.mode == "outage"]
    assert script["thresholds"] == {"music": 7, "volume": 4, "questions": 3, "negatives": 4}
    assert script["timeout_seconds"] == 25
    counts = {c: sum(1 for line in normal if line.category == c)
              for c in ("music", "volume", "questions", "negatives")}
    assert counts == {"music": 8, "volume": 4, "questions": 4, "negatives": 4}
    assert sorted(line.kind for line in normal if line.category == "negatives") == \
        sorted(["tv", "codex_question", "correction", "truncated"])
    assert [line.category for line in outage] == ["negatives"] * 4
    assert sorted(line.kind for line in outage) == sorted(["tv", "codex_question", "correction", "truncated"])


def test_script_rejects_wrong_composition(tmp_path):
    data = yaml.safe_load(SCRIPT_PATH.read_text(encoding="utf-8"))
    data["lines"] = [line for line in data["lines"] if line["id"] != "M8"]
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ScriptError, match="normal script must have"):
        load_script(path)


def test_script_rejects_changed_thresholds(tmp_path):
    data = yaml.safe_load(SCRIPT_PATH.read_text(encoding="utf-8"))
    data["thresholds"]["music"] = 6
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ScriptError, match="thresholds"):
        load_script(path)


# scoring thresholds

@pytest.mark.parametrize("category, passing, failing", [
    ("music", 7, 6), ("volume", 4, 3), ("questions", 3, 2), ("negatives", 4, 3)])
def test_thresholds_per_category(category, passing, failing):
    lines = [line for line in lines_for("normal") if line.category == category]
    thresholds = load_script()["thresholds"]

    def outcome(count):
        return {line.id: index < count for index, line in enumerate(lines)}

    assert score_categories(lines, outcome(passing), thresholds)[category]["ok"] is True
    assert score_categories(lines, outcome(failing), thresholds)[category]["ok"] is False


def test_full_normal_run_passes_and_writes_files(tmp_path):
    lines = lines_for("normal")
    items, answers = scripted(lines)
    summary, messages = run(tmp_path, "normal", items, answers)
    assert summary["pass"] is True
    assert summary["forbidden"] == []
    assert summary["categories"]["music"] == {"passed": 8, "total": 8, "required": 7, "ok": True}
    rows = rows_file(summary)
    assert len(rows) == 20
    assert all(entry["passed"] for entry in rows if entry["category"] != "negatives")
    saved = json.loads(Path(summary["summary_file"]).read_text(encoding="utf-8"))
    assert saved["pass"] is True
    assert Path(summary["summary_file"]).name == "phase1-20261027T090000-summary.json"
    assert any("RESULT: PASS" in message for message in messages)
    assert summary["targets"]["music_execute_to_observed_median_ms"]["met"] is True


def test_music_six_of_eight_fails_gate(tmp_path):
    lines = lines_for("normal")
    verdicts = {"M2": ["n"], "M5": ["n"]}
    items, answers = scripted(lines, verdicts=verdicts)
    summary, _ = run(tmp_path, "normal", items, answers)
    assert summary["categories"]["music"]["passed"] == 6
    assert summary["pass"] is False


# timeouts and misses

def test_timeout_is_a_counted_miss_without_operator_prompt(tmp_path):
    lines = lines_for("normal")
    items, answers = scripted(lines, rows={"V1": None})
    summary, _ = run(tmp_path, "normal", items, answers)
    assert summary["categories"]["volume"]["passed"] == 3
    timed_out = [entry for entry in rows_file(summary) if entry["line_id"] == "V1"]
    assert len(timed_out) == 1
    assert timed_out[0]["timeout"] is True
    assert timed_out[0]["verdict"] is None
    assert timed_out[0]["counted"] is True


def test_negative_with_no_turn_passes_automatically(tmp_path):
    lines = lines_for("normal")
    items, answers = scripted(lines, rows={"N1": None})
    summary, messages = run(tmp_path, "normal", items, answers)
    assert summary["categories"]["negatives"]["passed"] == 4
    assert summary["pass"] is True
    entry = next(e for e in rows_file(summary) if e["line_id"] == "N1")
    assert entry["timeout"] is True and entry["passed"] is True and entry["verdict"] == "auto"
    assert any("passes automatically" in message for message in messages)


def test_positive_with_no_turn_stays_a_miss(tmp_path):
    lines = lines_for("normal")
    items, answers = scripted(lines, rows={"M4": None})
    summary, _ = run(tmp_path, "normal", items, answers)
    assert summary["categories"]["music"]["passed"] == 7
    assert summary["categories"]["music"]["ok"] is True  # 7 of 8 meets the gate
    assert summary["pass"] is True


def test_outage_negative_with_no_turn_passes_automatically(tmp_path):
    lines = lines_for("outage")
    items, answers = scripted(lines, rows={"O3": None})
    summary, _ = run(tmp_path, "outage", items, answers, stamp=datetime(2026, 10, 27, 9, 5, 0))
    assert summary["categories"]["negatives"]["passed"] == 4
    assert summary["pass"] is True


# outage run checks the provider really was down

def test_outage_run_aborts_unscored_when_first_turn_used_brain(tmp_path):
    lines = lines_for("outage")
    bad = row("t-O1", lines[0].phrase, reply="Sure.", path="brain")
    items, answers = scripted(lines, rows={"O1": bad})
    summary, messages = run(tmp_path, "outage", items, answers)
    assert summary["pass"] is False
    assert summary["abort_reason"] == "provider not down"
    assert summary["categories"] == {}
    assert any("provider was not down" in message for message in messages)
    entries = rows_file(summary)
    assert len(entries) == 1 and entries[0]["counted"] is False


def test_outage_provider_check_uses_first_turn_row_not_timeout(tmp_path):
    lines = lines_for("outage")
    items, answers = scripted(lines, rows={"O1": None})
    items[1] = row("t-O2", lines[1].phrase, path="brain")
    summary, _ = run(tmp_path, "outage", items, answers)
    assert summary["abort_reason"] == "provider not down"
    assert summary["pass"] is False


def test_outage_run_with_fallback_path_is_scored(tmp_path):
    lines = lines_for("outage")
    items, answers = scripted(lines)
    summary, _ = run(tmp_path, "outage", items, answers)
    assert summary["abort_reason"] is None
    assert summary["categories"]["negatives"]["ok"] is True


# combined verdict

def _run_both(tmp_path, *, outage_fails=False):
    normal_items, normal_answers = scripted(lines_for("normal"))
    normal, _ = run(tmp_path, "normal", normal_items, normal_answers,
                    stamp=datetime(2026, 10, 27, 9, 0, 0))
    outage_answers_override = {"O2": ["n"]} if outage_fails else {}
    outage_items, outage_answers = scripted(lines_for("outage"), verdicts=outage_answers_override)
    outage, _ = run(tmp_path, "outage", outage_items, outage_answers,
                    stamp=datetime(2026, 10, 27, 9, 5, 0))
    return normal["summary_file"], outage["summary_file"]


def test_combine_prints_pass_only_when_both_runs_pass(tmp_path, capsys):
    normal_path, outage_path = _run_both(tmp_path)
    assert main(["--combine", normal_path, outage_path]) == 0
    assert capsys.readouterr().out.startswith("PHASE 1 PASS")


def test_combine_prints_fail_when_outage_fails(tmp_path, capsys):
    normal_path, outage_path = _run_both(tmp_path, outage_fails=True)
    assert main(["--combine", normal_path, outage_path]) == 1
    assert capsys.readouterr().out.startswith("PHASE 1 FAIL")


def test_combine_rejects_swapped_summaries(tmp_path, capsys):
    normal_path, outage_path = _run_both(tmp_path)
    assert main(["--combine", outage_path, normal_path]) == 2
    assert "combine error" in capsys.readouterr().err


# x re-ask once

def test_invalid_attempt_is_reasked_once_and_then_counts(tmp_path):
    lines = lines_for("normal")
    items, answers = [], []
    for line in lines:
        if line.id == "V2":
            items += [default_row(line), default_row(line)]
            answers += ["x", "y"]
        else:
            items.append(default_row(line))
            answers.append("y")
    summary, _ = run(tmp_path, "normal", items, answers)
    entries = [entry for entry in rows_file(summary) if entry["line_id"] == "V2"]
    assert [entry["counted"] for entry in entries] == [False, True]
    assert summary["categories"]["volume"]["passed"] == 4
    assert summary["pass"] is True


def test_second_invalid_attempt_counts_as_miss(tmp_path):
    lines = lines_for("normal")
    items, answers = [], []
    for line in lines:
        if line.id == "V2":
            items += [default_row(line), default_row(line)]
            answers += ["x", "x"]
        else:
            items.append(default_row(line))
            answers.append("y")
    summary, _ = run(tmp_path, "normal", items, answers)
    entries = [entry for entry in rows_file(summary) if entry["line_id"] == "V2"]
    assert [entry["counted"] for entry in entries] == [False, True]
    assert entries[-1]["passed"] is False
    assert summary["categories"]["volume"]["passed"] == 3


def test_unknown_verdict_is_reasked_without_consuming_attempt(tmp_path):
    lines = lines_for("normal")
    items, answers = scripted(lines)
    answers = ["maybe"] + answers
    summary, messages = run(tmp_path, "normal", items, answers)
    assert summary["attempts"] == 20
    assert any("Enter y, n or x" in message for message in messages)


# forbidden effects

def test_codex_capability_auto_fails_and_stops_run(tmp_path):
    lines = lines_for("normal")
    bad = row("t-M3", "Alexa, resume", caps=["codex_request"], receipts=[("codex_request", "requested")])
    items, answers = scripted(lines, rows={"M3": bad})
    summary, _ = run(tmp_path, "normal", items, answers)
    assert summary["pass"] is False
    assert summary["aborted"] is True
    assert summary["forbidden"][0]["reason"] == "codex_triggered"
    assert summary["forbidden"][0]["line_id"] == "M3"
    assert summary["categories"]["music"]["passed"] == 2
    assert len(rows_file(summary)) == 3


def test_correction_with_receipt_is_prefix_execution():
    line = next(l for l in lines_for("normal") if l.id == "N3")
    bad = row("t", line.phrase, caps=["play_song"], receipts=[("play_song", "observed")])
    assert row_forbidden(bad, line) == ["prefix_executed"]


def test_truncated_with_receipt_is_prefix_execution():
    line = next(l for l in lines_for("normal") if l.id == "N4")
    bad = row("t", line.phrase, receipts=[("play_song", "requested")])
    assert row_forbidden(bad, line) == ["prefix_executed"]


def test_tv_negative_with_receipt_is_unrequested_effect():
    line = next(l for l in lines_for("normal") if l.id == "N1")
    bad = row("t", line.phrase, receipts=[("set_volume", "observed")])
    assert row_forbidden(bad, line) == ["unrequested_effect"]


def test_codex_question_negative_with_codex_capability_is_forbidden():
    line = next(l for l in lines_for("normal") if l.id == "N2")
    bad = row("t", line.phrase, caps=["codex_request"])
    assert "codex_triggered" in row_forbidden(bad, line)


def test_success_claim_without_observed_receipt_is_forbidden():
    line = next(l for l in lines_for("normal") if l.id == "M1")
    claim = row("t", line.phrase, receipts=[("play_song", "requested")], reply="Playing Halo now.")
    assert row_forbidden(claim, line) == ["unobserved_success_claim"]
    honest = row("t", line.phrase, receipts=[("play_song", "requested")],
                 reply="I asked Spotify but can't confirm it played.")
    assert row_forbidden(honest, line) == []


def test_claims_success_ignores_hedged_replies():
    assert claims_success("Volume set to 40.")
    assert not claims_success("I asked Windows but can't confirm the volume changed.")


# latency

def test_nearest_rank_and_median():
    values = list(range(1, 21))
    assert nearest_rank(values, 0.95) == 19
    assert nearest_rank([7], 0.95) == 7


def test_latency_summary_reports_median_and_p95_per_stage():
    rows = [row(f"t{i}", timings={"plan_ms": value}) for i, value in enumerate([100, 200, 300, 400])]
    rows.append(row("t9", timings={"plan_ms": None}))
    stats = latency_summary(rows)
    assert stats["plan_ms"] == {"n": 4, "median_ms": 250, "p95_ms": 400}
    assert stats["stt_ms"] == {"n": 0, "median_ms": None, "p95_ms": None}


# tail reads only new rows

def test_tail_ignores_old_rows_and_waits_for_full_line(tmp_path):
    path = tmp_path / "core_turns.jsonl"
    path.write_text(json.dumps({"turn_id": "old"}) + "\n", encoding="utf-8")
    tail = TurnTail(path)
    tail.mark()
    text = json.dumps({"turn_id": "new"}) + "\n"
    with path.open("a", encoding="utf-8") as stream:
        stream.write(text[:5])
    assert tail.next_row(0.0, clock=lambda: 0.0, sleep=lambda s: None) is None
    with path.open("a", encoding="utf-8") as stream:
        stream.write(text[5:])
    assert tail.next_row(1.0, clock=lambda: 0.0, sleep=lambda s: None) == {"turn_id": "new"}


# outage run

def test_outage_run_has_only_negatives_and_records_mode(tmp_path):
    lines = lines_for("outage")
    items, answers = scripted(lines)
    summary, messages = run(tmp_path, "outage", items, answers)
    assert summary["mode"] == "outage"
    assert set(summary["categories"]) == {"negatives"}
    assert summary["pass"] is True
    assert summary["provider_env"] == "MEMOIR_CORE_PROVIDER_OUTAGE=1"
    assert any("MEMOIR_CORE_PROVIDER_OUTAGE=1" in prompt for prompt in run.prompts)


def test_outage_run_fails_on_one_negative_miss(tmp_path):
    lines = lines_for("outage")
    items, answers = scripted(lines, verdicts={"O2": ["n"]})
    summary, _ = run(tmp_path, "outage", items, answers)
    assert summary["categories"]["negatives"]["passed"] == 3
    assert summary["pass"] is False


def test_cli_returns_2_on_script_error(tmp_path, capsys):
    assert main(["phase1"], script_path=tmp_path / "missing.yaml") == 2
    assert "script error" in capsys.readouterr().err


@pytest.fixture(scope="module")
def all_phase_registry():
    from assistant.config import Config
    from assistant.core.registry import PHASE_ONE, Registry

    capabilities = set(PHASE_ONE)
    capabilities.update(name for _, adapter in Registry._discover() for name in adapter.handlers)
    return Registry(Config({"core": {"enabled_phases": ["1", "1b", "3", "4"],
                                     "enabled_capabilities": sorted(capabilities)}}))


@pytest.mark.parametrize("phase", ["phase1", "phase1b", "phase3", "phase4"])
def test_trial_lines_against_real_grammar(phase, all_phase_registry):
    from assistant.conversation_contracts import Step
    from assistant.core.adapters.knowledge import public_topic
    from assistant.core.authority import _without_wake, admit_step, parsed_licenses

    path = SCRIPT_PATH.parent / f"{phase}.yaml"
    for line in load_script(path, phase)["lines"]:
        text = _without_wake(line.phrase)
        _, licenses = parsed_licenses(text, all_phase_registry)
        admitted = [(name, args) for name, _, args in licenses
                    if admit_step(Step("trial", name, args), text, all_phase_registry, lambda: 10)]
        if line.category == "negatives":
            assert not admitted, (phase, line.id, admitted)
            if line.kind == "privacy":
                assert public_topic(line.phrase) is None
        elif line.category == "questions":
            assert not admitted, (phase, line.id, admitted)
            if phase == "phase4":
                assert public_topic(line.phrase)
        else:
            expected = line.capability if phase != "phase1" else (
                "play_song" if line.id in {"M1", "M5"} else
                "pause" if line.id in {"M2", "M6"} else
                "resume" if line.id in {"M3", "M7"} else
                "next_track" if line.id in {"M4", "M8"} else
                "set_volume" if line.id in {"V1", "V4"} else "change_volume")
            assert any(name == expected for name, _ in admitted), (phase, line.id, admitted)


@pytest.mark.parametrize("phase", ["phase1b", "phase3", "phase4"])
def test_extra_phase_gate_and_schema_validation(phase, tmp_path):
    from assistant.core.trials import PHASE_GATES

    path = SCRIPT_PATH.parent / f"{phase}.yaml"
    script = load_script(path, phase)
    outcomes = {line.id: True for line in script["lines"]}
    assert all(item["ok"] for item in score_categories(script["lines"], outcomes, script["thresholds"]).values())
    for category, (_, required) in PHASE_GATES[phase].items():
        category_lines = [line for line in script["lines"] if line.category == category]
        passing = {line.id: index < required - 1 for index, line in enumerate(category_lines)}
        assert not score_categories(script["lines"], passing, script["thresholds"])[category]["ok"]
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["thresholds"][next(iter(data["thresholds"]))] = 0
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ScriptError):
        load_script(invalid, phase)


def test_phase3_codex_and_phase4_privacy_forbidden():
    codex_lines = load_script(SCRIPT_PATH.parent / "phase3.yaml", "phase3")["lines"]
    positive = next(line for line in codex_lines if line.category == "codex")
    negative = next(line for line in codex_lines if line.category == "negatives")
    triggered = row("t", caps=["codex_request"], receipts=[("codex_request", "observed")])
    assert "codex_triggered" not in row_forbidden(triggered, positive)
    assert "codex_triggered" in row_forbidden(triggered, negative)
    privacy = next(line for line in load_script(SCRIPT_PATH.parent / "phase4.yaml", "phase4")["lines"]
                   if line.kind == "privacy")
    private_row = row("t")
    private_row["knowledge_source"] = "source"
    assert "privacy_retrieval" in row_forbidden(private_row, privacy)


@pytest.mark.parametrize("phase", ["phase1b", "phase3", "phase4"])
def test_extra_phase_runs_and_scores(phase, tmp_path):
    path = SCRIPT_PATH.parent / f"{phase}.yaml"
    lines = load_script(path, phase)["lines"]
    items = []
    for line in lines:
        item = row("t-" + line.id, line.phrase)
        if line.capability:
            item = row("t-" + line.id, line.phrase, caps=[line.capability],
                       receipts=[(line.capability, "observed")])
        if phase == "phase4" and line.category == "questions":
            item["knowledge_source"] = "Wikipedia"
        items.append(item)
    core = FakeCore(tmp_path / "turns.jsonl", items)
    summary = run_phase(mode="normal", phase=phase, script_path=path, turns_path=core.path,
                        out_dir=tmp_path / "out", ask=Operator([""] + ["y"] * len(lines)),
                        out=lambda text: None, clock=core.clock, sleep=core.sleep,
                        now=lambda: datetime(2026, 10, 28))
    assert summary["pass"]
    assert summary["phase"] == phase


def test_codex_requires_observed_receipt(tmp_path):
    path = SCRIPT_PATH.parent / "phase3.yaml"
    lines = load_script(path, "phase3")["lines"]
    items = [row("t-" + line.id, caps=[line.capability],
                 receipts=[(line.capability, "requested" if line.category == "codex" else "observed")])
             if line.capability else row("t-" + line.id) for line in lines]
    core = FakeCore(tmp_path / "turns.jsonl", items)
    summary = run_phase(mode="normal", phase="phase3", script_path=path, turns_path=core.path,
                        out_dir=tmp_path / "out", ask=Operator([""] + ["y"] * len(lines)),
                        out=lambda text: None, clock=core.clock, sleep=core.sleep,
                        now=lambda: datetime(2026, 10, 28))
    assert not summary["pass"]
    assert summary["categories"]["codex"]["passed"] == 0
