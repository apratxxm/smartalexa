import json
from types import SimpleNamespace

import pytest

from assistant.conversation_contracts import Decision, Step
from assistant.core.__main__ import _toast, _setup
from assistant.core import __main__ as entry
from assistant.core import volume
from assistant.core.evaluation import evaluate
from assistant.config import Config
from assistant.intent import Action


def test_toast_is_nonblocking_and_spawn_errors_are_swallowed(monkeypatch):
    calls = []
    monkeypatch.setattr(entry, "_TOASTS", [])
    monkeypatch.setattr(entry.subprocess, "run", lambda *a, **k: pytest.fail("blocking toast"))
    def spawn(*args, **kwargs):
        calls.append((args, kwargs))
        raise OSError("desktop unavailable")
    monkeypatch.setattr(entry.subprocess, "Popen", spawn)
    _toast('$(throw "not code")')
    assert len(calls) == 1
    assert "-EncodedCommand" in calls[0][0][0]
    assert "not code" not in " ".join(calls[0][0][0])


def test_only_explicit_alt_selects_groq():
    cfg = Config({"core": {"brain": {"base_url": "https://primary.invalid/v1",
          "model": "primary", "api_key_env": "PRIMARY_KEY",
          "alt": {"base_url": "https://alt.invalid/v1", "model": "alt", "api_key_env": "ALT_KEY"}}}})
    assert _setup(cfg).model == "primary"
    assert _setup(cfg, "alt").model == "alt"


@pytest.mark.parametrize("failure", ["set", "verify"])
def test_volume_write_failures_are_typed_uncertainty(monkeypatch, failure):
    writes = []
    def set_(level, _):
        writes.append(level)
        if failure == "set":
            raise OSError("write may have happened")
    endpoint = SimpleNamespace(SetMasterVolumeLevelScalar=set_,
                               GetMasterVolumeLevelScalar=lambda: 0.9)
    monkeypatch.setattr(volume, "_endpoint", lambda: endpoint)
    with pytest.raises(volume.SystemVolumeVerificationError):
        volume.set_system_volume(0.4)
    assert writes == [0.4]


def test_eval_is_planning_only_retains_usage_and_summary(tmp_path):
    class Planner:
        timeout, model, last_usage = 1, "fake", {}
        def missing_env(self):
            return []
        def decide(self, transcript, **kwargs):
            self.last_usage = {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}
            return Decision("plan", steps=(Step("p", "pause", {}),),
                            evidence_quotes=(transcript.text,))
    summary = evaluate(Planner(), cases=[("pause", Action.PAUSE, [])], output_dir=tmp_path)
    rows = [json.loads(line) for line in next(tmp_path.glob("core_eval_*.jsonl")).read_text().splitlines()]
    assert len(rows) == 2 and rows[0]["passed"] is True and rows[0]["latency_ms"] >= 0
    assert rows[0]["usage"]["total_tokens"] == 3
    assert rows[1] == summary and summary["attempted"] == summary["passed"] == 1
    assert summary["failed"] == summary["unattempted"] == 0


def test_eval_missing_key_exits_before_output_or_provider_call(tmp_path):
    brain = SimpleNamespace(missing_env=lambda: ["CF_API_TOKEN"])
    with pytest.raises(SystemExit, match="CF_API_TOKEN"):
        evaluate(brain, output_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []
