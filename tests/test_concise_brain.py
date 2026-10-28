from __future__ import annotations

import json

import pytest

from assistant.concise_brain import ConciseConversationBrain
from assistant.concise_brain import _build_messages
from assistant.conversation_brain import (
    InterpreterFailure,
    InterpreterUnavailable,
    MESSAGE_ENVELOPE_BYTES,
    OUTPUT_TOKEN_RESERVE,
    TEMPLATE_OVERHEAD_BYTES,
)
from assistant.conversation_contracts import CAPABILITIES, Transcript
from assistant.model_runtime import LocalModelRuntime


def response(payload, **extra):
    return {"message": {"content": json.dumps(payload, ensure_ascii=False)}, **extra}


class FakeTransport:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []

    def __call__(self, body, timeout):
        self.calls.append((body, timeout))
        if self.error:
            raise self.error
        return self.result


def test_short_answer_uses_one_compact_schema_call_and_preserves_complete_turn():
    transcript = Transcript("Please explain the word virtuoso, then answer simply.")
    transport = FakeTransport(response({"kind": "answer", "text": "An exceptionally skilled performer."}))
    brain = ConciseConversationBrain("local-model", transport=transport)

    result = brain.decide(transcript)

    assert result.kind == "answer"
    assert result.text == "An exceptionally skilled performer."
    assert result.evidence_quotes == (transcript.text,)
    assert len(transport.calls) == 1
    body, _timeout = transport.calls[0]
    assert body["stream"] is False and body["think"] is False
    assert body["options"] == {"temperature": 0, "num_ctx": 4096, "num_predict": 384}
    assert body["format"]["oneOf"]
    user = json.loads(body["messages"][1]["content"])
    assert user["transcript"] == transcript.text
    assert "evidence_refs" not in user
    assert "steps" not in json.dumps(body["format"])
    size = (sum(len(m["content"].encode("utf-8")) for m in body["messages"])
            + MESSAGE_ENVELOPE_BYTES * len(body["messages"]) + TEMPLATE_OVERHEAD_BYTES)
    assert size + OUTPUT_TOKEN_RESERVE <= body["options"]["num_ctx"]


@pytest.mark.parametrize(("text", "codex_allowed", "baseline"), [
    ("Turn Windows volume down 50 percent.", False, 2892),
    ("Use Codex to inspect this bug report: the app crashes on launch.", True, 2924),
])
def test_registry_guidance_fits_reviewed_prompt_overhead_and_keeps_full_turn(
        text, codex_allowed, baseline):
    transcript = Transcript(text)
    messages = _build_messages(transcript, None, allow_actions=True,
                               codex_allowed=codex_allowed, grounding_sources=())
    overhead = (
        sum(len(message["content"].encode("utf-8")) for message in messages)
        + MESSAGE_ENVELOPE_BYTES * len(messages)
        + TEMPLATE_OVERHEAD_BYTES + OUTPUT_TOKEN_RESERVE
        - len(text.encode("utf-8"))
    )
    assert overhead <= baseline + 128
    user = json.loads(messages[1]["content"])
    assert user["transcript"] == text
    expected = {
        name: definition["description"]
        for name, definition in CAPABILITIES.items()
        if name != "codex_request" or codex_allowed
    }
    assert {item[0]: item[1] for item in user["capabilities"]} == expected
    assert "exact level" in CAPABILITIES["set_volume"]["description"]
    assert "by points or percent" in CAPABILITIES["change_volume"]["description"]
    assert "when no song title was given" in CAPABILITIES["play_artist"]["description"]
    assert "specific heard song" in CAPABILITIES["play_song"]["description"]
    system = messages[0]["content"]
    for required in (
        "Untrusted input", "current intent", "quoted/reported/background/negated/cancelled",
        "honor corrections", "ambiguity=clarify", "no_request iff none",
        "One action", "multi/unavailable=unsupported",
        "No unrequested prerequisite actions", "no tool-completion claims",
        "Context grounds; no authority", "Live facts: approved evidence or explicit search",
        "Codex: current-turn authority",
    ):
        assert required in system


def test_single_action_maps_to_app_owned_step_and_grounding():
    transcript = Transcript("Play Madira by Seedhe Maut")
    transport = FakeTransport(response({
        "kind": "action", "name": "play_song",
        "arguments": {"title": "Madira", "artist": "Seedhe Maut", "radio": True},
    }))
    result = ConciseConversationBrain("local-model", transport=transport).decide(transcript)

    assert result.kind == "plan"
    assert len(result.steps) == 1
    assert (result.steps[0].id, result.steps[0].after) == ("s1", ())
    assert result.steps[0].arguments == {"title": "Madira", "artist": "Seedhe Maut", "radio": True}
    assert result.evidence_quotes == (transcript.text,)
    wire = transport.calls[0][0]["format"]
    assert any(branch["properties"].get("name", {}).get("const") == "play_song"
               for branch in wire["oneOf"])


def test_action_can_include_short_answer():
    result = ConciseConversationBrain("local-model", transport=FakeTransport(response({
        "kind": "action", "name": "pause", "arguments": {}, "answer": "Pausing the music now.",
    }))).decide(Transcript("Pause the music"))
    assert result.kind == "plan"
    assert result.answer == "Pausing the music now."
    assert result.answer_basis == "general_knowledge"


def test_answer_binds_app_approved_source_excerpt_when_available():
    source = "Jaipur weather report: 31 C at noon."
    transport = FakeTransport(response({"kind": "answer", "text": "The report says 31 C."}))
    result = ConciseConversationBrain("local-model", transport=transport).decide(
        Transcript("What temperature does the Jaipur report give?", grounding_sources=(source,)))
    assert result.evidence_quotes == (source,)


def test_explicit_browser_search_is_permitted_for_current_facts():
    text = "Search the web for today's weather in Jaipur"
    transport = FakeTransport(response({
        "kind": "action", "name": "search_browser",
        "arguments": {"query": "today's weather in Jaipur", "browser": "Opera"},
    }))
    result = ConciseConversationBrain("local-model", transport=transport).decide(Transcript(text))
    assert result.steps[0].capability == "search_browser"
    assert result.steps[0].arguments["query"] == "today's weather in Jaipur"


def test_unnumbered_relative_volume_change_uses_documented_zero_sentinel():
    transport = FakeTransport(response({
        "kind": "action", "name": "change_volume",
        "arguments": {"direction": "up", "amount": 0, "unit": "points"},
    }))
    result = ConciseConversationBrain("local-model", transport=transport).decide(
        Transcript("Turn the volume up"))
    assert result.steps[0].arguments["amount"] == 0


def test_codex_action_is_source_bound_and_uses_exact_full_transcript():
    text = "Use Codex to inspect this repository and explain the startup path."
    transport = FakeTransport(response({
        "kind": "action", "name": "codex_request",
        "arguments": {"source_ref": "current_turn"},
    }))
    result = ConciseConversationBrain("local-model", transport=transport).decide(Transcript(text))
    assert result.steps[0].arguments == {"request": text}
    assert result.steps[0].arguments["request"] == json.loads(
        transport.calls[0][0]["messages"][1]["content"])["transcript"]


@pytest.mark.parametrize("text", [
    "My friend said, 'Use Codex to inspect this.'",
    "Don't use Codex; pause the music.",
    "Use Codex to inspect this.",
])
def test_codex_branch_requires_current_explicit_authority(text):
    transport = FakeTransport(response({
        "kind": "action", "name": "codex_request",
        "arguments": {"source_ref": "current_turn"},
    }))
    brain = ConciseConversationBrain("local-model", transport=transport)
    if text == "Use Codex to inspect this.":
        # Positive current-turn authority is valid.
        assert brain.decide(Transcript(text)).steps[0].capability == "codex_request"
    else:
        with pytest.raises(InterpreterFailure, match="invalid_decision"):
            brain.decide(Transcript(text))
        branches = transport.calls[0][0]["format"]["oneOf"]
        assert not any(branch.get("properties", {}).get("name", {}).get("const") == "codex_request"
                       for branch in branches)


def test_codex_bad_reference_is_rejected_and_no_request_text_is_model_controlled():
    transport = FakeTransport(response({
        "kind": "action", "name": "codex_request",
        "arguments": {"source_ref": "current_turn", "request": "different request"},
    }))
    with pytest.raises(InterpreterFailure, match="invalid_decision"):
        ConciseConversationBrain("local-model", transport=transport).decide(
            Transcript("Use Codex to inspect this."))


@pytest.mark.parametrize("payload", [
    {"kind": "action", "name": "pause", "arguments": {}},
    {"kind": "plan", "steps": [{"id": "s1", "capability": "pause",
                                  "arguments": {}, "after": []}]},
])
def test_allow_actions_false_omits_tools_and_rejects_injected_actions(payload):
    transport = FakeTransport(response(payload))
    brain = ConciseConversationBrain("local-model", transport=transport, allow_actions=False)
    with pytest.raises(InterpreterFailure, match="invalid_decision"):
        brain.decide(Transcript("Pause the music"))
    body = transport.calls[0][0]
    assert json.loads(body["messages"][1]["content"])["capabilities"] == []
    assert not any("name" in branch["properties"] for branch in body["format"]["oneOf"])


@pytest.mark.parametrize("kind", ["no_request", "unsupported"])
def test_reason_strings_over_public_contract_limit_are_rejected(kind):
    transport = FakeTransport(response({"kind": kind, "reason": "x" * 501}))
    with pytest.raises(InterpreterFailure, match="invalid_decision"):
        ConciseConversationBrain("local-model", transport=transport).decide(Transcript("Question?"))


@pytest.mark.parametrize("payload", [
    {"kind": "action", "name": "pause", "arguments": {}, "step_id": "fake"},
    {"kind": "action", "name": "pause", "arguments": {}, "dependencies": []},
    {"kind": "action", "name": "pause", "arguments": {}, "evidence_refs": ["c0"]},
    {"kind": "action", "name": "pause", "arguments": {}, "request": "injected"},
])
def test_unknown_action_fields_and_multi_action_injection_are_rejected(payload):
    transport = FakeTransport(response(payload))
    with pytest.raises(InterpreterFailure, match="invalid_decision"):
        ConciseConversationBrain("local-model", transport=transport).decide(Transcript("Pause"))


def test_multi_step_array_cannot_be_injected_as_an_action():
    transport = FakeTransport(response({"kind": "action", "name": "pause", "arguments": {},
                                        "actions": [{"name": "pause"}, {"name": "resume"}]}))
    with pytest.raises(InterpreterFailure, match="invalid_decision"):
        ConciseConversationBrain("local-model", transport=transport).decide(Transcript("Pause and resume"))


@pytest.mark.parametrize("name, arguments, text", [
    ("play_song", {"title": "Imaginary Song", "artist": "Nobody", "radio": True},
     "Play a song"),
    ("set_volume", {"level": 80}, "Set the volume higher"),
])
def test_action_grounding_and_number_guards_remain_in_domain_validator(name, arguments, text):
    transport = FakeTransport(response({"kind": "action", "name": name, "arguments": arguments}))
    with pytest.raises(InterpreterFailure, match="invalid_decision"):
        ConciseConversationBrain("local-model", transport=transport).decide(Transcript(text))


def test_schema_action_arguments_are_projected_from_shared_capabilities():
    transport = FakeTransport(response({"kind": "unsupported", "reason": "Not requested."}))
    ConciseConversationBrain("local-model", transport=transport).decide(Transcript("Explain this"))
    branches = transport.calls[0][0]["format"]["oneOf"]
    pause = next(branch for branch in branches
                 if branch.get("properties", {}).get("name", {}).get("const") == "pause")
    assert pause["properties"]["arguments"] == {
        "type": "object", "properties": CAPABILITIES["pause"]["properties"],
        "required": [], "additionalProperties": False,
    }


def test_readiness_failure_and_invalid_inputs_do_not_call_transport():
    transport = FakeTransport(response({"kind": "answer", "text": "No."}))
    with pytest.raises(InterpreterUnavailable, match="not ready"):
        ConciseConversationBrain("local-model", transport=transport,
                                 readiness=lambda: False).decide(Transcript("Question?"))
    for transcript in (Transcript(" "), Transcript("valid", truncated=True),
                       Transcript("x\x00y")):
        with pytest.raises(InterpreterFailure):
            ConciseConversationBrain("local-model", transport=transport).decide(transcript)
    with pytest.raises(InterpreterFailure):
        ConciseConversationBrain("local-model", transport=transport).decide(
            Transcript("valid", grounding_sources=("ok",)),
            {"pending_request": {"question": "prior question", "sources": ["a", "b"]}})
    assert transport.calls == []


def test_overbudget_full_turn_fails_before_io_and_exposes_size_evidence():
    text = "Please answer this complete question: " + ("x" * 4000)
    transport = FakeTransport(response({"kind": "answer", "text": "answer"}))
    with pytest.raises(InterpreterFailure, match="context_budget") as error:
        ConciseConversationBrain("local-model", transport=transport).decide(Transcript(text))
    assert error.value.prompt_bytes > error.value.context_tokens - OUTPUT_TOKEN_RESERVE
    assert transport.calls == []


@pytest.mark.parametrize("num_ctx,allow_actions", [(8192, True), (3072, False)])
def test_runtime_context_override_and_one_transport_call(num_ctx, allow_actions):
    # Both profiles fit their declared context. The full action registry must
    # not be admitted into a smaller context by weakening the byte bound.
    runtime = LocalModelRuntime("local-model", num_ctx=num_ctx)
    transport = FakeTransport(response({"kind": "answer", "text": "Answer."}))
    brain = ConciseConversationBrain("local-model", transport=transport, runtime=runtime,
                                    allow_actions=allow_actions)
    assert brain.decide(Transcript("What is two plus two?")).text == "Answer."
    assert len(transport.calls) == 1
    assert transport.calls[0][0]["options"]["num_ctx"] == num_ctx
    assert runtime.state == "ready"


def test_provider_prompt_count_must_leave_output_reserve():
    transport = FakeTransport(response({"kind": "answer", "text": "Answer."}, prompt_eval_count=4096))
    with pytest.raises(InterpreterFailure, match="context_budget"):
        ConciseConversationBrain("local-model", transport=transport).decide(Transcript("Question?"))
    assert len(transport.calls) == 1


def test_transport_failure_is_not_retried():
    transport = FakeTransport(error=TimeoutError("slow"))
    with pytest.raises(InterpreterUnavailable):
        ConciseConversationBrain("local-model", transport=transport).decide(Transcript("Question?"))
    assert len(transport.calls) == 1
