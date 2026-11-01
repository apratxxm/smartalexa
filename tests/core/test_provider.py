import json
import threading
from urllib.error import HTTPError

import pytest

from assistant.conversation_brain import build_messages, _ollama_schema_for
from assistant.conversation_contracts import Transcript
from assistant.core.provider import BrainProvider, ProviderError

WIRE = {"kind": "plan", "steps": [{"id": "p", "capability": "pause",
         "arguments": {}, "after": []}], "evidence_refs": ["c0"]}


class Response:
    def __init__(self, wire=WIRE, **overrides):
        self.data = {"choices": [{"message": {"tool_calls": [{"function": {
            "name": "decision", "arguments": json.dumps(wire)}}]}}],
                     "usage": {"prompt_tokens": 15, "completion_tokens": 7, "total_tokens": 22}}
        self.data.update(overrides)
    def __enter__(self):
        return self
    def __exit__(self, *_):
        pass
    def read(self, limit):
        return json.dumps(self.data).encode()[:limit]


def provider(monkeypatch, opener, **kwargs):
    monkeypatch.setenv("CORE_TEST_KEY", "synthetic-secret")
    return BrainProvider("https://example.invalid/v1", "fake", "CORE_TEST_KEY", opener=opener, **kwargs)


def test_provider_reuses_retained_messages_schema_decoder_and_usage(monkeypatch):
    transcript = Transcript("pause", "t")
    sent = []
    def open_(request, **kwargs):
        sent.append(json.loads(request.data))
        return Response()
    brain = provider(monkeypatch, open_)
    decision = brain.decide(transcript)
    assert sent[0]["messages"] == build_messages(transcript, None)
    assert sent[0]["tools"][0]["function"]["parameters"] == _ollama_schema_for(transcript, None)
    assert decision.steps[0].capability == "pause" and decision.evidence_quotes == ("pause",)
    assert brain.last_usage == {"prompt_tokens": 15, "completion_tokens": 7, "total_tokens": 22}


@pytest.mark.parametrize("status,category", [(429, "rate_limited"), (402, "unavailable"),
                                            (503, "unavailable"), (400, "invalid")])
def test_http_error_categories_without_retry(monkeypatch, status, category):
    calls = []
    def open_(request, **kwargs):
        calls.append(request)
        raise HTTPError(request.full_url, status, "synthetic error", {}, None)
    brain = provider(monkeypatch, open_)
    with pytest.raises(ProviderError) as error:
        brain.decide(Transcript("pause", "t"))
    assert error.value.category == category and len(calls) == 1


def test_refusal_is_terminal_not_provider_outage(monkeypatch):
    brain = provider(monkeypatch, lambda *a, **k: Response(
        choices=[{"message": {"refusal": "Refused"}}]))
    assert brain.decide(Transcript("pause", "t")).kind == "unsupported"


@pytest.mark.parametrize("wire", [
    {"kind": "plan", "steps": [], "evidence_refs": ["c0"]},
    {**WIRE, "evidence_refs": ["invented"]},
    {"kind": "plan", "steps": WIRE["steps"], "evidence_quotes": ["pause"]},
])
def test_invalid_wire_plan_is_typed_invalid(monkeypatch, wire):
    brain = provider(monkeypatch, lambda *a, **k: Response(wire))
    with pytest.raises(ProviderError) as error:
        brain.decide(Transcript("pause", "t"))
    assert error.value.category == "invalid"
    assert brain.last_usage["total_tokens"] == 22


def test_superseded_provider_result_is_discarded_by_turn_id(monkeypatch):
    turn = ["old"]
    def open_(*args, **kwargs):
        turn[0] = "new"
        return Response()
    brain = provider(monkeypatch, open_)
    with pytest.raises(ProviderError) as error:
        brain.decide(Transcript("pause", "old"), current_turn=lambda: turn[0])
    assert error.value.reason == "stale_turn"
    assert brain.last_usage == {}


def test_hard_deadline_discards_late_response_without_leaking_usage(monkeypatch):
    release, finished = threading.Event(), threading.Event()
    def open_(*args, **kwargs):
        try:
            assert release.wait(2)
            return Response()
        finally:
            finished.set()
    brain = provider(monkeypatch, open_, timeout=0.05)
    try:
        with pytest.raises(ProviderError) as error:
            brain.decide(Transcript("pause", "t"))
        assert error.value.reason == "deadline"
    finally:
        release.set()
        assert finished.wait(2)
    assert brain.last_usage == {}


def test_expansion_is_confined_to_base_url(monkeypatch):
    monkeypatch.setenv("CF_ACCOUNT_ID", "synthetic-account")
    brain = BrainProvider("https://example.invalid/accounts/${CF_ACCOUNT_ID}/v1",
                          "${MODEL_STAYS_LITERAL}", "CORE_TEST_KEY")
    assert brain.base_url == "https://example.invalid/accounts/synthetic-account/v1"
    assert brain.model == "${MODEL_STAYS_LITERAL}"
