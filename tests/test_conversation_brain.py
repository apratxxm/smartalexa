"""Mocked boundary tests for the local conversational interpreter."""
from __future__ import annotations

import json
import re

import pytest

from assistant.conversation_brain import (
    ConversationBrain,
    InterpreterFailure,
    InterpreterUnavailable,
    MESSAGE_ENVELOPE_BYTES,
    MODEL_CONTEXT_TOKENS,
    OUTPUT_TOKEN_RESERVE,
    TEMPLATE_OVERHEAD_BYTES,
    OLLAMA_DECISION_SCHEMA,
    _loads_unique,
    _ollama_schema_for,
    _project_ollama_schema,
    _transport,
    build_messages,
)
from assistant.conversation_contracts import CAPABILITIES, DECISION_SCHEMA, Transcript
from assistant.model_runtime import LocalModelRuntime


def _response(payload):
    return {"message": {"content": json.dumps(payload, ensure_ascii=False)}}


def _budget_bytes(messages):
    return (
        sum(len(message["content"].encode("utf-8")) for message in messages)
        + MESSAGE_ENVELOPE_BYTES * len(messages)
        + TEMPLATE_OVERHEAD_BYTES
        + OUTPUT_TOKEN_RESERVE
    )


class FakeTransport:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def __call__(self, body, timeout):
        self.calls.append((body, timeout))
        return _response(self.payload)


def test_one_local_decision_request_uses_schema_and_maps_answer():
    transport = FakeTransport({
        "kind": "answer",
        "answer": "A virtuoso is exceptionally skilled, especially in music.",
        "basis": "general_knowledge",
        "evidence_refs": ["c0"],
    })
    brain = ConversationBrain("local-model:latest", transport=transport)

    result = brain.decide(Transcript("What does virtuoso mean?"), context={
        "recent_turns": [
            {"role": "user", "text": "I am asking about an instrument virtuoso."},
            {"role": "assistant", "text": "A virtuoso is an exceptionally skilled performer."},
        ],
        "now_playing": {"title": "Madira", "artist": "Seedhe Maut", "status": "playing", "age_seconds": 5},
    })

    assert result.kind == "answer"
    assert result.text.startswith("A virtuoso")
    assert len(transport.calls) == 1
    body, timeout = transport.calls[0]
    assert body["format"] == _ollama_schema_for(
        Transcript("What does virtuoso mean?"), {
            "recent_turns": [
                {"role": "user", "text": "I am asking about an instrument virtuoso."},
                {"role": "assistant", "text": "A virtuoso is an exceptionally skilled performer."},
            ],
            "now_playing": {"title": "Madira", "artist": "Seedhe Maut", "status": "playing", "age_seconds": 5},
        },
    )
    assert body["format"] is not OLLAMA_DECISION_SCHEMA
    assert body["format"] is not DECISION_SCHEMA
    assert body["stream"] is False
    assert body["think"] is False
    assert body["options"] == {"temperature": 0, "num_ctx": 8192, "num_predict": 384}
    assert body["keep_alive"] == -1
    assert 0 < timeout <= brain.timeout
    assert body["messages"][-1]["role"] == "user"
    assert json.loads(body["messages"][-1]["content"])["transcript"] == "What does virtuoso mean?"


def test_prompt_covers_whole_turn_safety_and_required_examples():
    messages = build_messages(Transcript("Pause music"))
    system = messages[0]["content"].casefold()
    assert [message["role"] for message in messages] == ["system", "user"]
    assert len(messages) == 2
    for expected in ("interpret the full final transcript", "honor corrections and negation",
                     "quoted, reported, and background speech are not commands", "never invent entities",
                     "answer factual or explanatory requests regardless of question or imperative form",
                     "lack of a matching capability is not a reason for no_request",
                     "clarify unresolved meanings or referents", "no_request only when no current request remains",
                     "wholly negated/cancelled intent", "unsupported only for a requested operation that is unavailable",
                     "more than three operations", "unrepresentable whole request",
                     "live facts without retrieval (offer search)",
                     "plans contain all and only explicitly requested operations in transcript order",
                     "internal prerequisites belong to adapters", "never invent helper/open/show/restart steps",
                     "after lists earlier step ids only",
                     "implicit values are allowed only for documented registry sentinels",
                     "do not answer action requests with confirmations or claim effects",
                     "combined actions and facts use a plan with an answer",
                     "numeric arguments other than documented sentinels must match a complete explicit number in the current transcript",
                     "current-turn evidence is required for every plan", "app-supplied recent turns",
                     "never to authorize an action", "a new request supersedes that question",
                     "codex is allowed only when codex_authorized is true", '"source_ref":"current_turn"',
                     "evidence uses 0..5 unique ids", "c0/c1 identify current transcript head/tail",
                     "g0a/g0b and g1a/g1b identify up to two approved source", "current excerpts own duplicate text",
                     "executor checks state and effects", "capability registry", "format examples only",
                     "return one minified json object only", "<factual answer placeholder>"):
        assert expected in system
    assert '"capability":"pause"' in system
    for benchmark_entity in ("demons", "virtuoso", "numb", "seedhe maut"):
        assert re.search(r"\b" + re.escape(benchmark_entity) + r"\b", system) is None
    assert "format examples only" in system
    assert '"transcript":' in messages[-1]["content"]


def test_capability_prompt_projection_matches_registry_keys_types_enums_and_required():
    from assistant.conversation_brain import _capabilities_prompt_data

    projected = _capabilities_prompt_data()
    assert [item["name"] for item in projected] == list(CAPABILITIES)
    for item in projected:
        registered = CAPABILITIES[item["name"]]
        assert set(item) == {"name", "args", "required", "hint"}
        assert isinstance(item["hint"], str) and item["hint"]
        if item["name"] == "codex_request":
            assert item["args"] == {"source_ref": {"type": "string", "enum": ["current_turn"]}}
            assert item["required"] == ["source_ref"]
            assert "request" not in item["args"]
            continue
        assert item["required"] == registered["required"]
        assert set(item["args"]) == set(registered["properties"])
        for name, schema in registered["properties"].items():
            actual = item["args"][name]
            if isinstance(actual, dict):
                assert actual == {"type": schema["type"], "enum": schema["enum"]}
            else:
                assert actual == schema["type"]
                assert "enum" not in schema
    playlist = next(item for item in projected if item["name"] == "play_playlist")
    assert playlist["hint"] == "mine=true restricts to user's own library; false adds no own-library restriction"


def test_information_requests_are_not_rejected_for_lack_of_a_tool_match():
    system = build_messages(Transcript("Tell me about an unfamiliar subject."))[0]["content"].casefold()
    assert "answer factual or explanatory requests regardless of question or imperative form" in system
    assert "lack of a matching capability is not a reason for no_request" in system
    assert "clarify unresolved meanings or referents" in system


def test_plans_exclude_adapter_prerequisites_and_keep_requested_order():
    system = build_messages(Transcript("Perform the requested operation."))[0]["content"].casefold()
    assert "plans contain all and only explicitly requested operations in transcript order" in system
    assert "internal prerequisites belong to adapters" in system
    assert "never invent helper/open/show/restart steps" in system
    assert "after lists earlier step ids only" in system


def test_corrections_replace_only_superseded_intent_and_keep_positive_remainder():
    system = build_messages(Transcript("Correct the earlier request."))[0]["content"].casefold()
    assert "a correction supersedes only the corrected intent" in system
    assert "interpret any remaining positive request" in system


def test_corrected_transcript_yields_only_the_validated_pause_plan():
    transport = FakeTransport({
        "kind": "plan",
        "steps": [{"id": "s1", "capability": "pause", "arguments": {}, "after": []}],
        "evidence_refs": ["c0"],
    })
    result = ConversationBrain("model", transport=transport).decide(
        Transcript("Play Numb—actually, pause it.")
    )
    assert result.kind == "plan"
    assert len(result.steps) == 1
    assert result.steps[0].capability == "pause"
    # Interpretation returns a plan only; it does not execute any capability.
    assert len(transport.calls) == 1


def test_plan_may_carry_a_brief_stable_answer():
    transport = FakeTransport({
        "kind": "plan",
        "steps": [{"id": "s1", "capability": "pause", "arguments": {}, "after": []}],
        "evidence_refs": ["c0"],
        "answer": "A virtuoso is a highly skilled performer.",
        "answer_basis": "general_knowledge",
    })
    result = ConversationBrain("model", transport=transport).decide(
        Transcript("Pause and define virtuoso.")
    )
    assert result.kind == "plan"
    assert result.steps[0].capability == "pause"
    assert "skilled" in result.answer
    assert result.answer_basis == "general_knowledge"


@pytest.mark.parametrize("transcript", [
    Transcript(""),
    Transcript("x" * 12_001),
    Transcript("hello", truncated=True),
    Transcript("hello\x00there"),
    Transcript("hello\nthere"),
    Transcript("hello\tthere"),
    Transcript("hello\x7fthere"),
])
def test_invalid_transcripts_are_rejected_before_transport(transcript):
    transport = FakeTransport({})
    with pytest.raises(InterpreterFailure):
        ConversationBrain("model", transport=transport).decide(transcript)
    assert transport.calls == []


def test_context_budget_rejects_long_turn_without_trimming_or_calling_model():
    transport = FakeTransport({})
    with pytest.raises(InterpreterFailure, match="context_budget"):
        ConversationBrain("model", transport=transport).decide(Transcript("a" * 12_000))
    assert transport.calls == []


def test_context_budget_rejects_oversize_referents_without_dropping_them():
    transport = FakeTransport({})
    context = {"recent_turns": [
        {"role": "user", "text": "a" * 1000},
        {"role": "assistant", "text": "b" * 1000},
        {"role": "user", "text": "c" * 1000},
    ]}
    with pytest.raises(InterpreterFailure, match="context_budget"):
        ConversationBrain("model", transport=transport).decide(Transcript("Continue that."), context)
    assert transport.calls == []


def test_150_byte_turn_with_small_context_keeps_700_byte_headroom():
    transcript = Transcript(
        "What does virtuoso mean? Use the earlier music example. Keep it brief. "
        "This is a general definition, not a request to search current information."
    )
    context = {
        "recent_turns": [
            {"role": "user", "text": "I was asking about music terms."},
            {"role": "assistant", "text": "I can explain the terms briefly."},
        ],
        "now_playing": {"title": "Madira", "artist": "Seedhe Maut", "status": "playing"},
    }
    assert 140 <= len(transcript.text.encode("utf-8")) <= 160
    estimate = _budget_bytes(build_messages(transcript, context))
    assert MODEL_CONTEXT_TOKENS - estimate >= 700


def test_500_byte_command_and_referents_fit_and_are_preserved():
    transcript_text = (
        "Please play Madira by Seedhe Maut. I want the original track, not an album "
        "or playlist, and I am not asking for radio. Keep my earlier note that I may "
        "ask you to pause later as background only; it is not a pause request now. "
        "Preserve the artist exactly as spoken. Album order was background, not another request. "
        "Play the song title I just said and keep the artist wording as heard. I may "
        "ask later; no search, volume change, app opening, or Codex request. "
        "Please play that one track now."
    )
    assert 450 <= len(transcript_text.encode("utf-8")) <= 550
    context = {
        "recent_turns": [
            {"role": "user", "text": "I prefer the artist spelling I say aloud."},
            {"role": "assistant", "text": "I will preserve the words you heard."},
        ],
        "now_playing": {"title": "Numb", "artist": "Linkin Park", "status": "playing", "age_seconds": 8},
    }
    transport = FakeTransport({
        "kind": "plan",
        "steps": [{"id": "s1", "capability": "play_song", "arguments": {"title": "Madira", "artist": "Seedhe Maut"}, "after": []}],
        "evidence_refs": ["c0"],
    })
    result = ConversationBrain("model", transport=transport).decide(Transcript(transcript_text), context)
    assert result.kind == "plan"
    assert len(transport.calls) == 1
    body, _ = transport.calls[0]
    assert _budget_bytes(body["messages"]) < MODEL_CONTEXT_TOKENS
    sent = json.loads(body["messages"][-1]["content"])
    assert sent["transcript"] == transcript_text
    assert len(sent["context"]["recent_turns"]) == 2
    assert sent["context"]["now_playing"]["title"] == "Numb"


def test_oversized_schema_is_rejected_before_transport(monkeypatch):
    import copy
    import assistant.conversation_brain as conversation_brain

    oversized = copy.deepcopy(OLLAMA_DECISION_SCHEMA)
    oversized["x-test"] = "x" * 32_769
    monkeypatch.setattr(conversation_brain, "OLLAMA_DECISION_SCHEMA", oversized)
    transport = FakeTransport({})
    with pytest.raises(InterpreterFailure, match="context_budget"):
        ConversationBrain("model", transport=transport).decide(Transcript("Pause music"))
    assert transport.calls == []


def test_ollama_schema_projection_removes_only_unsupported_bounds_without_aliasing():
    source = {
        "type": "array", "minItems": 1, "x-vendor": {"maxLength": 9, "kept": True},
        "items": [{"oneOf": [{"type": "string", "minLength": 1,
                                "pattern": "^x$", "maxLength": 4,
                                "const": "x", "uniqueItems": True}],
                   "required": ["value"], "maxItems": 2}],
    }
    original = json.loads(json.dumps(source))
    projected = _project_ollama_schema(source)
    assert projected == {
        "type": "array", "minItems": 1, "x-vendor": {"kept": True},
        "items": [{"oneOf": [{"type": "string", "const": "x"}],
                   "required": ["value"]}],
    }
    assert source == original
    projected["x-vendor"]["kept"] = False
    assert source["x-vendor"]["kept"] is True
    assert "minLength" in json.dumps(DECISION_SCHEMA)
    assert "minLength" not in json.dumps(OLLAMA_DECISION_SCHEMA)
    omitted = {"minLength", "maxLength", "pattern", "maxItems", "uniqueItems"}

    def assert_no_omitted_bounds(node):
        if isinstance(node, dict):
            assert omitted.isdisjoint(node)
            for child in node.values():
                assert_no_omitted_bounds(child)
        elif isinstance(node, list):
            for child in node:
                assert_no_omitted_bounds(child)

    assert_no_omitted_bounds(OLLAMA_DECISION_SCHEMA)
    assert OLLAMA_DECISION_SCHEMA["oneOf"]
    assert "const" in json.dumps(OLLAMA_DECISION_SCHEMA)
    assert "additionalProperties" in json.dumps(OLLAMA_DECISION_SCHEMA)
    assert "minItems" in json.dumps(OLLAMA_DECISION_SCHEMA)


def test_full_validator_rejects_output_over_backend_omitted_length_bound():
    overlong_title = "a" * 121
    transport = FakeTransport({
        "kind": "plan",
        "steps": [{"id": "s1", "capability": "play_song", "arguments": {
            "title": overlong_title, "artist": "",
        }, "after": []}],
        "evidence_refs": ["c0"],
    })
    with pytest.raises(InterpreterFailure, match="invalid_decision") as caught:
        ConversationBrain("model", transport=transport).decide(Transcript(overlong_title))
    assert caught.value.validation_error == "title length must be at most 120"
    assert len(transport.calls) == 1


def test_full_validator_rejects_empty_answer_evidence_even_if_backend_allows_it():
    transport = FakeTransport({
        "kind": "answer", "answer": "A definition.",
        "basis": "general_knowledge", "evidence_refs": [],
    })
    with pytest.raises(InterpreterFailure, match="invalid_decision") as caught:
        ConversationBrain("model", transport=transport).decide(Transcript("Define virtuoso."))
    assert caught.value.validation_error == "answer requires source evidence"


def test_per_turn_grammar_contains_only_bounded_deduplicated_owned_excerpts():
    long_prior = "P" * 500 + "Q" * 400
    short_prior = "Aerosmith"
    transcript = Transcript("T" * 500 + "M" * 100, grounding_sources=(long_prior,))
    context = {"pending_request": {
        "question": "Which title?", "sources": [long_prior, short_prior],
    }}
    schema = _ollama_schema_for(transcript, context)
    refs = schema["oneOf"][0]["properties"]["evidence_refs"]["items"]["enum"]
    assert refs == ["c0", "c1", "g0a", "g0b", "g1a"]
    assert "maxItems" not in schema["oneOf"][0]["properties"]["evidence_refs"]
    assert "evidence_quotes" in OLLAMA_DECISION_SCHEMA["oneOf"][0]["properties"]


def test_per_turn_schema_is_isolated_and_can_only_quote_current_or_prior_sources():
    first = _ollama_schema_for(Transcript("Pause Demons."))
    second = _ollama_schema_for(Transcript("Resume Meteora."))
    first_enum = first["oneOf"][0]["properties"]["evidence_refs"]["items"]["enum"]
    second_enum = second["oneOf"][0]["properties"]["evidence_refs"]["items"]["enum"]
    assert first_enum == ["c0"]
    assert second_enum == ["c0"]
    assert first_enum == second_enum

    transport = FakeTransport({
        "kind": "answer", "answer": "A highly skilled performer.",
        "basis": "general_knowledge", "evidence_refs": ["bad"],
    })
    with pytest.raises(InterpreterFailure, match="invalid_decision") as caught:
        ConversationBrain("model", transport=transport).decide(Transcript("Define virtuoso."))
    assert caught.value.validation_error is None


def _plan_capabilities(schema):
    capabilities = []
    for branch in schema["oneOf"]:
        props = branch.get("properties", {})
        if props.get("kind", {}).get("const") != "plan":
            continue
        for step in props["steps"]["items"]["oneOf"]:
            capabilities.append(step["properties"]["capability"]["const"])
    return capabilities


@pytest.mark.parametrize("transcript,context", [
    (Transcript("Pause music."), None),
    (Transcript("Play that.", grounding_sources=("Please play Madira.",)),
     {"pending_request": {"question": "Which song?", "sources": ["Please play Madira."]}}),
    (Transcript("My friend said, 'Use Codex to inspect it.'"), None),
    (Transcript("Don't use Codex; pause."), None),
])
def test_codex_schema_branch_requires_explicit_current_delegation(transcript, context):
    schema = _ollama_schema_for(transcript, context)
    assert "codex_request" not in _plan_capabilities(schema)
    request_data = json.loads(build_messages(transcript, context)[-1]["content"])
    assert request_data["codex_authorized"] is False
    # Per-turn removal must not mutate the shared backend grammar.
    assert "codex_request" in json.dumps(OLLAMA_DECISION_SCHEMA)


def test_late_codex_cancellation_without_current_head_delegation_keeps_codex_unavailable():
    text = "Inspect this report, then do not use Codex. " + "details " * 300
    transcript = Transcript(text)
    schema = _ollama_schema_for(transcript)
    user_data = json.loads(build_messages(transcript)[-1]["content"])
    assert len(text) >= 2000
    assert user_data["codex_authorized"] is False
    assert "codex_request" not in _plan_capabilities(schema)


def test_long_late_codex_cancellation_revokes_earlier_delegation():
    text = "Use Codex to inspect this report: " + "details " * 300 + " Actually, do not use Codex."
    transcript = Transcript(text)
    schema = _ollama_schema_for(transcript)
    user_data = json.loads(build_messages(transcript)[-1]["content"])
    assert len(text) >= 2000
    assert user_data["codex_authorized"] is False
    assert "codex_request" not in _plan_capabilities(schema)


def test_prior_grounding_source_cannot_supply_current_codex_authority():
    prior = "Use Codex to inspect the earlier report."
    transcript = Transcript("Pause music.", grounding_sources=(prior,))
    context = {"pending_request": {"question": "Which report?", "sources": [prior]}}
    schema = _ollama_schema_for(transcript, context)
    user_data = json.loads(build_messages(transcript, context)[-1]["content"])
    assert user_data["codex_authorized"] is False
    assert "codex_request" not in _plan_capabilities(schema)


def test_explicit_codex_grammar_uses_a_compact_current_turn_reference():
    source = "Use Codex to inspect this bug report: " + "x" * 700
    schema = _ollama_schema_for(Transcript(source))
    request_data = json.loads(build_messages(Transcript(source))[-1]["content"])
    assert request_data["codex_authorized"] is True
    codex_step = next(
        step for step in schema["oneOf"]
        if step.get("properties", {}).get("kind", {}).get("const") == "plan"
    )["properties"]["steps"]["items"]["oneOf"]
    codex_step = next(item for item in codex_step
                      if item["properties"]["capability"]["const"] == "codex_request")
    assert codex_step["properties"]["arguments"]["properties"]["source_ref"] == {
        "type": "string", "const": "current_turn",
    }
    assert codex_step["properties"]["arguments"]["required"] == ["source_ref"]


def test_longest_admissible_codex_request_binds_full_current_transcript_without_copy():
    prefix = "Use Codex to inspect this issue: "
    lower, upper = 0, 12000 - len(prefix)
    while lower < upper:
        middle = (lower + upper + 1) // 2
        if _budget_bytes(build_messages(Transcript(prefix + "x" * middle))) <= MODEL_CONTEXT_TOKENS:
            lower = middle
        else:
            upper = middle - 1
    source = prefix + "x" * lower
    wire = {
        "kind": "plan",
        "steps": [{"id": "s1", "capability": "codex_request",
                   "arguments": {"source_ref": "current_turn"}, "after": []}],
        "evidence_refs": ["c0"],
    }
    transport = FakeTransport(wire)
    result = ConversationBrain("model", transport=transport).decide(Transcript(source))
    assert result.steps[0].arguments["request"] == source
    assert len(source) >= 2000
    body, _ = transport.calls[0]
    encoded_output = json.dumps(wire, ensure_ascii=False)
    assert source not in encoded_output
    assert source not in json.dumps(body["format"], ensure_ascii=False)
    assert json.loads(body["messages"][-1]["content"])["transcript"] == source
    assert _budget_bytes(body["messages"]) == MODEL_CONTEXT_TOKENS
    assert body["format"]["oneOf"][2]["properties"]["steps"]["items"]["oneOf"]

    over_budget = FakeTransport(wire)
    with pytest.raises(InterpreterFailure, match="context_budget"):
        ConversationBrain("model", transport=over_budget).decide(Transcript(source + "x"))
    assert over_budget.calls == []


def test_backend_omits_evidence_maxitems_but_decoder_enforces_five():
    transcript = Transcript("T" * 600, grounding_sources=("P" * 600, "Q" * 600))
    schema = _ollama_schema_for(transcript)
    assert "maxItems" not in schema["oneOf"][0]["properties"]["evidence_refs"]
    transport = FakeTransport({
        "kind": "no_request", "reason": "none",
        "evidence_refs": ["c0", "c1", "g0a", "g0b", "g1a", "g1b"],
    })
    with pytest.raises(InterpreterFailure, match="invalid_decision"):
        ConversationBrain("model", transport=transport).decide(transcript)
    assert len(transport.calls) == 1


@pytest.mark.parametrize("wire", [
    {"kind": "no_request", "reason": "none", "evidence_refs": ["unknown"]},
    {"kind": "no_request", "reason": "none", "evidence_refs": ["c0", "c0"]},
    {"kind": "no_request", "reason": "none", "evidence_refs": [0]},
    {"kind": "no_request", "reason": "none", "evidence_refs": [], "extra": True},
    {"kind": "no_request", "reason": "none", "evidence_refs": [], "evidence_quotes": []},
])
def test_wire_rejects_unknown_duplicate_mixed_and_wrongly_typed_references(wire):
    transport = FakeTransport(wire)
    with pytest.raises(InterpreterFailure, match="invalid_decision"):
        ConversationBrain("model", transport=transport).decide(Transcript("Hello"))
    assert len(transport.calls) == 1


@pytest.mark.parametrize("transcript,arguments", [
    ("My friend said, 'Use Codex to inspect this.'", {"source_ref": "current_turn"}),
    ("Don't use Codex; pause.", {"source_ref": "current_turn"}),
    ("Use Codex to inspect this.", {"source_ref": "previous_turn"}),
    ("Use Codex to inspect this.", {"request": "Use Codex to inspect this."}),
    ("Use Codex to inspect this.", {"source_ref": "current_turn", "request": "x"}),
])
def test_codex_wire_requires_explicit_current_request_reference(transcript, arguments):
    transport = FakeTransport({
        "kind": "plan",
        "steps": [{"id": "s1", "capability": "codex_request", "arguments": arguments, "after": []}],
        "evidence_refs": ["c0"],
    })
    with pytest.raises(InterpreterFailure, match="invalid_decision"):
        ConversationBrain("model", transport=transport).decide(Transcript(transcript))
    assert len(transport.calls) == 1


def test_prior_source_may_ground_plan_arguments_but_not_action_evidence():
    prior = "Please play Madira by See the Moth"
    transport = FakeTransport({
        "kind": "plan",
        "steps": [{"id": "s1", "capability": "play_song", "arguments": {
            "title": "Madira", "artist": "See the Moth",
        }, "after": []}],
        "evidence_refs": ["g0a"],
    })
    with pytest.raises(InterpreterFailure, match="invalid_decision") as caught:
        ConversationBrain("model", transport=transport).decide(
            Transcript("Do that.", grounding_sources=(prior,)),
            {"pending_request": {"question": "Which song?", "sources": [prior]}},
        )
    assert caught.value.validation_error == "plan requires evidence from the current transcript"


def test_readiness_false_prevents_http_call():
    transport = FakeTransport({})
    brain = ConversationBrain("model", transport=transport, readiness=lambda: False)
    with pytest.raises(InterpreterUnavailable):
        brain.decide(Transcript("Pause music"))
    assert transport.calls == []


def test_injected_model_runtime_owns_context_and_residency_policy():
    transport = FakeTransport({
        "kind": "no_request", "reason": "No action requested.", "evidence_refs": [],
    })
    runtime = LocalModelRuntime("model", num_ctx=8192, keep_alive=-1)
    brain = ConversationBrain("model", transport=transport, runtime=runtime)

    result = brain.decide(Transcript("Hello there"))

    assert result.kind == "no_request"
    body, timeout = transport.calls[0]
    assert body["model"] == runtime.model
    assert body["options"]["num_ctx"] == runtime.num_ctx == 8192
    assert body["keep_alive"] == runtime.keep_alive == -1
    assert 0 < timeout <= brain.timeout
    assert runtime.state == "ready"


def test_injected_runtime_context_is_used_for_preflight_budget():
    transport = FakeTransport({})
    runtime = LocalModelRuntime("model", num_ctx=4096)
    brain = ConversationBrain("model", transport=transport, runtime=runtime)
    with pytest.raises(InterpreterFailure, match="context_budget"):
        brain.decide(Transcript("x" * 4100))
    assert transport.calls == []


def test_injected_runtime_model_mismatch_is_rejected_before_transport():
    transport = FakeTransport({})
    runtime = LocalModelRuntime("other-model", num_ctx=4096)
    with pytest.raises(ValueError, match="runtime model"):
        ConversationBrain("model", transport=transport, runtime=runtime)
    assert transport.calls == []


def test_injected_runtime_admission_refusal_becomes_unavailable_without_transport():
    transport = FakeTransport({})
    runtime = LocalModelRuntime("model", num_ctx=8192, cooldown_seconds=60)
    runtime.mark_failed()
    brain = ConversationBrain("model", transport=transport, runtime=runtime)

    with pytest.raises(InterpreterUnavailable, match="runtime refused"):
        brain.decide(Transcript("Pause music"))
    assert transport.calls == []


@pytest.mark.parametrize("model", ["cloud", "remote-cloud-model", "MODEL-CLOUD:latest"])
def test_cloud_model_tags_are_rejected(model):
    with pytest.raises(ValueError):
        ConversationBrain(model)


def test_injected_transport_timeout_is_bounded_by_remaining_deadline():
    transport = FakeTransport({})
    brain = ConversationBrain("model", timeout=0.2, transport=transport)

    def too_slow(body, timeout):
        import time
        assert timeout <= 0.2
        time.sleep(0.25)
        return _response({"kind": "no_request", "reason": "none", "evidence_refs": []})

    brain._transport = too_slow
    with pytest.raises(InterpreterUnavailable, match="deadline"):
        brain.decide(Transcript("Pause music"))


def test_duplicate_fields_and_malformed_inner_json_are_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        _loads_unique('{"kind":"answer","kind":"unsupported"}')

    class MalformedTransport:
        def __call__(self, body, timeout):
            return {"message": {"content": '{"kind":"no_request","kind":"unsupported"}'}}

    with pytest.raises(InterpreterFailure, match="invalid_decision"):
        ConversationBrain("model", transport=MalformedTransport()).decide(Transcript("Hello"))


def test_contract_invalid_output_does_not_fall_back_or_retry():
    transport = FakeTransport({"kind": "plan", "steps": [], "evidence_refs": []})
    with pytest.raises(InterpreterFailure, match="invalid_decision"):
        ConversationBrain("model", transport=transport).decide(Transcript("Pause music"))
    assert len(transport.calls) == 1


def test_context_allowlist_is_small_and_keeps_transcript_unchanged():
    source = "Play Numb by Linkin Park"
    transcript = Transcript("Continue the earlier request as I said.", turn_id="turn-1",
                             grounding_sources=(source,))
    context = {
        "recent_turns": [
            {"role": ["user"], "text": "malformed role"},
            {"role": "user", "text": "older"},
            {"role": "assistant", "text": "recent"},
        ],
        "now_playing": {"title": "Numb", "artist": "Linkin Park", "secret": "omit"},
        "pending_request": {"question": "Which album did you mean?", "sources": [source]},
        "filesystem": "must not enter model context",
    }
    user = json.loads(build_messages(transcript, context)[-1]["content"])
    assert user["transcript"] == transcript.text
    assert user["grounding_sources"] == [source]
    assert user["context"]["recent_turns"] == [
        {"role": "user", "text": "older"},
        {"role": "assistant", "text": "recent"},
    ]
    assert user["context"]["now_playing"] == {"title": "Numb", "artist": "Linkin Park"}
    assert "secret" not in user["context"]["now_playing"]
    assert "filesystem" not in user["context"]
    assert user["context"]["pending_request"] == {"question": "Which album did you mean?"}
    assert source not in json.dumps(user["context"])


def test_pending_reply_grounds_referent_once_and_preserves_heard_artist():
    source = "Please play Madira by See the Moth"
    transport = FakeTransport({
        "kind": "plan",
        "steps": [{"id": "s1", "capability": "play_song", "arguments": {
            "title": "Madira", "artist": "See the Moth",
        }, "after": []}],
        "evidence_refs": ["c0"],
    })
    result = ConversationBrain("model", transport=transport).decide(
        Transcript("Play that.", grounding_sources=(source,)),
        {"pending_request": {"question": "Which song did you mean?", "sources": [source]}},
    )
    assert result.kind == "plan"
    assert result.steps[0].arguments["artist"] == "See the Moth"
    body, _ = transport.calls[0]
    user_text = body["messages"][-1]["content"]
    user = json.loads(user_text)
    assert user["grounding_sources"] == [source]
    assert user["context"]["pending_request"]["question"] == "Which song did you mean?"
    assert user_text.count(source) == 1


def test_default_transport_uses_loopback_and_disables_proxies(monkeypatch):
    calls = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self, limit):
            calls["limit"] = limit
            return b'{"message":{"content":"{}"}}'

    class Opener:
        def open(self, request, timeout):
            calls["url"] = request.full_url
            calls["timeout"] = timeout
            return Response()

    def build_opener(*handlers):
        calls["handlers"] = handlers
        return Opener()

    monkeypatch.setattr("urllib.request.build_opener", build_opener)
    response = _transport({"model": "local"}, 1.25)
    assert response == {"message": {"content": "{}"}}
    assert calls["url"] == "http://127.0.0.1:11434/api/chat"
    assert calls["timeout"] == 1.25
    assert calls["limit"] == 65_537
    assert len(calls["handlers"]) == 2
    assert isinstance(calls["handlers"][0], __import__("urllib.request").request.ProxyHandler)
    assert isinstance(calls["handlers"][1], __import__("assistant.conversation_brain", fromlist=["_NoRedirectHandler"])._NoRedirectHandler)


def test_no_redirect_handler_refuses_redirect_targets():
    from assistant.conversation_brain import _NoRedirectHandler
    from urllib.request import Request

    handler = _NoRedirectHandler()
    assert handler.redirect_request(
        Request("http://127.0.0.1:11434/api/chat"), None, 302, "Found", {},
        "http://example.com/collect",
    ) is None


def test_default_transport_rejects_duplicate_outer_fields(monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self, _limit):
            return b'{"message":{},"message":{}}'

    class Opener:
        def open(self, _request, timeout):
            return Response()

    monkeypatch.setattr("urllib.request.build_opener", lambda *_: Opener())
    with pytest.raises(InterpreterFailure, match="invalid_response_json"):
        _transport({"model": "local"}, 1.0)


@pytest.mark.parametrize("status", [400, 500])
def test_default_transport_classifies_http_errors_and_closes_response(monkeypatch, status):
    import io
    import urllib.error

    class TrackedBody(io.BytesIO):
        was_closed = False

        def close(self):
            self.was_closed = True
            super().close()

    response_body = TrackedBody(b"sensitive provider body")
    http_error = urllib.error.HTTPError(
        "http://127.0.0.1:11434/private-path", status, "private message", {}, response_body,
    )

    class Opener:
        def open(self, _request, timeout):
            raise http_error

    monkeypatch.setattr("urllib.request.build_opener", lambda *_: Opener())
    with pytest.raises(InterpreterFailure, match="provider_http_error") as caught:
        _transport({"model": "private-model"}, 1.0)
    assert caught.value.code == "provider_http_error"
    assert caught.value.http_status == status
    assert response_body.was_closed
    assert "private" not in str(caught.value)
    assert "private-path" not in str(caught.value)


@pytest.mark.parametrize("failure, message", [
    (__import__("socket").timeout("timed out"), "local request timed out"),
    (__import__("urllib.error").error.URLError(OSError("connection refused")), "local interpreter is unreachable"),
])
def test_default_transport_keeps_timeout_and_connectivity_errors_unavailable(monkeypatch, failure, message):
    import urllib.error

    class Opener:
        def open(self, _request, timeout):
            raise failure

    monkeypatch.setattr("urllib.request.build_opener", lambda *_: Opener())
    with pytest.raises(InterpreterUnavailable, match=message):
        _transport({"model": "local"}, 1.0)
