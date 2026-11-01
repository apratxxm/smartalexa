"""Provider outage switch: MEMOIR_CORE_PROVIDER_OUTAGE=1 fails before any network call."""
from __future__ import annotations

import json
from urllib.error import HTTPError

import pytest

from assistant.conversation_contracts import Transcript
from assistant.core.provider import BrainProvider, ProviderError

WIRE = {"kind": "plan", "steps": [{"id": "p", "capability": "pause",
         "arguments": {}, "after": []}], "evidence_refs": ["c0"]}


class Ok:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def read(self, limit):
        return json.dumps({"choices": [{"message": {"tool_calls": [{"function": {
            "name": "decision", "arguments": json.dumps(WIRE)}}]}}]}).encode()[:limit]


def brain(monkeypatch, opener):
    monkeypatch.setenv("CORE_TEST_KEY", "synthetic-secret")
    return BrainProvider("https://example.invalid/v1", "fake", "CORE_TEST_KEY", opener=opener)


@pytest.mark.parametrize("value", ["1"])
def test_outage_raises_unavailable_without_any_network_call(monkeypatch, value):
    monkeypatch.setenv("MEMOIR_CORE_PROVIDER_OUTAGE", value)
    calls = []

    def opener(request, **kwargs):
        calls.append(request)
        return Ok()

    provider = brain(monkeypatch, opener)
    with pytest.raises(ProviderError) as error:
        provider.decide(Transcript("pause", "t"))
    assert error.value.category == "unavailable" and error.value.reason == "provider_outage"
    assert calls == []


def test_outage_also_blocks_synthesis_without_network(monkeypatch):
    monkeypatch.setenv("MEMOIR_CORE_PROVIDER_OUTAGE", "1")
    calls = []
    provider = brain(monkeypatch, lambda request, **kwargs: calls.append(request) or Ok())
    with pytest.raises(ProviderError) as error:
        provider.synthesize("question", [], turn_id="t")
    assert error.value.category == "unavailable"
    assert calls == []


@pytest.mark.parametrize("value", [None, "0", ""])
def test_switch_off_leaves_the_request_path_unchanged(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("MEMOIR_CORE_PROVIDER_OUTAGE", raising=False)
    else:
        monkeypatch.setenv("MEMOIR_CORE_PROVIDER_OUTAGE", value)
    calls = []

    def opener(request, **kwargs):
        calls.append(request)
        raise HTTPError(request.full_url, 503, "synthetic", {}, None)

    provider = brain(monkeypatch, opener)
    with pytest.raises(ProviderError) as error:
        provider.decide(Transcript("pause", "t"))
    assert len(calls) == 1
    assert error.value.reason == "http_503"
