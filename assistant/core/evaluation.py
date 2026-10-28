"""Planning-only evaluation over the retained brain_eval cases; no executor."""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from datetime import datetime, timezone

from ..config import ROOT
from ..conversation_contracts import Transcript, step_to_intent
from ..intent import Action
from .provider import ProviderError


def _check(case, decision):
    _, expected, needles, *extra = case
    intents = [step_to_intent(step) for step in decision.steps] if decision.kind == "plan" else []
    if not intents:
        return None if expected is Action.UNKNOWN else "action_mismatch"
    first = intents[0]
    if first.action is not expected:
        return "action_mismatch"
    if not all(needle in first.query.casefold() for needle in needles):
        return "query_mismatch"
    slots = extra[0] if extra else {}
    for name, value in slots.items():
        if name == "volume_after_play":
            if len(intents) != 2 or intents[1].action is not Action.VOLUME or intents[1].value != value:
                return "volume_after_play_mismatch"
        elif getattr(first, name, None) != value:
            return "slot_mismatch"
    if len(intents) != (2 if "volume_after_play" in slots else 1):
        return "unexpected_action"
    return None


def evaluate(provider, name="default", *, cases=None, output_dir=None, clock=time.monotonic):
    missing = provider.missing_env()
    if missing:
        raise SystemExit("Cannot run core eval: missing environment variable(s): " + ", ".join(missing))
    if cases is None:
        from ..experimental.brain_eval import CASES
        cases = CASES
    cases = tuple(cases)
    source = "assistant.experimental.brain_eval.CASES"
    if len(cases) != 26:
        print("NOTE: %s currently contains %d cases; evaluating the complete retained set." % (source, len(cases)))
    directory = output_dir or ROOT / "assistant" / "logs"
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = directory / ("core_eval_%s_%s.jsonl" % (name, stamp))
    passed, latencies, usage = 0, [], {}
    with path.open("w", encoding="utf-8") as stream:
        for index, case in enumerate(cases):
            heard = case[0]
            transcript = Transcript(heard, "eval-%d" % (index + 1))
            started, decision, failure = clock(), None, None
            try:
                decision = provider.decide(transcript, deadline=started + provider.timeout,
                                           current_turn=lambda: transcript.turn_id)
                failure = _check(case, decision)
            except ProviderError as exc:
                failure = exc.category + ":" + exc.reason
            except Exception as exc:
                failure = type(exc).__name__
            latency = (clock() - started) * 1000
            latencies.append(latency)
            tokens = dict(provider.last_usage)
            for key, value in tokens.items():
                if type(value) is int:
                    usage[key] = usage.get(key, 0) + value
            ok = failure is None
            passed += ok
            row = {"type": "case", "case_id": index + 1, "transcript": heard,
                   "passed": ok, "failure": failure, "latency_ms": latency,
                   "usage": tokens, "decision": asdict(decision) if decision else None}
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            print("%s case=%d latency_ms=%.1f failure=%s" %
                  ("PASS" if ok else "FAIL", index + 1, latency, failure))
        summary = {"type": "summary", "provider": name, "model": provider.model,
                   "suite_source": source, "attempted": len(cases), "passed": passed,
                   "failed": len(cases) - passed, "unattempted": 0,
                   "usage": usage, "total_latency_ms": sum(latencies)}
        stream.write(json.dumps(summary) + "\n")
    print(json.dumps(summary))
    print("Results: " + str(path))
    return summary
