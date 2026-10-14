"""Fake-transport coverage for the private native-function diagnostic."""
from __future__ import annotations

import copy
import hashlib
import json

import pytest

from assistant.conversation_brain import InterpreterFailure, InterpreterUnavailable
from assistant.conversation_contracts import CAPABILITIES, Transcript
from assistant.experimental.native_decision_eval import (
    NativeAdmissionProof,
    NativeDecisionDiagnostic,
    build_native_messages,
    build_native_tools,
    decode_native_response,
)


def _call(name, arguments, index=None):
    function = {"name": name, "arguments": arguments}
    if index is not None:
        function["index"] = index
    return {"type": "function", "function": function}


def _response(*calls, content="", **fields):
    return {"message": {"role": "assistant", "content": content,
                        "tool_calls": list(calls)}, **fields}


def _decision(*calls, transcript="Pause the music.", context=None):
    return decode_native_response(_response(*calls), Transcript(transcript), context)


def test_tools_are_registry_derived_and_codex_is_current_authority_only():
    ordinary = build_native_tools(Transcript("Play Madira."), None)
    by_name = {tool["function"]["name"]: tool["function"] for tool in ordinary}
    assert set(CAPABILITIES) - {"codex_request"} <= set(by_name)
    assert "codex_request" not in by_name
    for name, contract in CAPABILITIES.items():
        if name != "codex_request":
            assert by_name[name]["parameters"] == {
                key: value for key, value in contract.items() if key != "description"}
    codex = {tool["function"]["name"]: tool["function"]
             for tool in build_native_tools(Transcript("Use Codex to inspect this."), None)}
    assert codex["codex_request"]["parameters"] == {
        "type": "object",
        "properties": {"source_ref": {"type": "string", "enum": ["current_turn"]}},
        "required": ["source_ref"], "additionalProperties": False,
    }
    assert "codex_request" not in {
        tool["function"]["name"] for tool in build_native_tools(
            Transcript("My friend said, 'Use Codex to inspect this.'"), None)}


def test_messages_keep_registry_out_of_prompt_text_and_attach_bounded_app_evidence():
    transcript = Transcript("Use the current report.", grounding_sources=("Report title: Alpha",))
    before = copy.deepcopy(transcript)
    messages = build_native_messages(transcript, {"pending_request": {
        "question": "Which report?", "sources": ["Report title: Alpha"]}})
    user = json.loads(messages[1]["content"])
    assert user["transcript"] == transcript.text
    assert user["grounding_sources"] == ["Report title: Alpha"]
    assert user["context"]["pending_request"] == {"question": "Which report?"}
    assert any(item["source"] == "current transcript" and item["excerpt"] == transcript.text
               for item in user["evidence_sources"])
    assert "CAPABILITIES" not in messages[0]["content"]
    assert "pause" not in messages[0]["content"].casefold()
    assert transcript == before


def test_answer_can_accompany_ordered_effects_and_adapter_ids_are_serial():
    decision = _decision(
        _call("respond_answer", {"answer": "A short definition.", "basis": "general_knowledge"}),
        _call("play_song", {"title": "Madira", "artist": "Seedhe Maut"}),
        _call("pause", {}),
        transcript="What does virtuoso mean? Then play Madira by Seedhe Maut and pause.",
    )
    assert decision.kind == "plan"
    assert decision.answer == "A short definition."
    assert decision.answer_basis == "general_knowledge"
    assert [(step.id, step.capability, step.after) for step in decision.steps] == [
        ("s1", "play_song", ()), ("s2", "pause", ("s1",)),
    ]
    assert decision.evidence_quotes == (
        "What does virtuoso mean? Then play Madira by Seedhe Maut and pause.",)


def test_documented_function_index_is_accepted_but_never_used_as_step_id():
    result = _decision(_call("pause", {}, index=3))
    assert [(step.id, step.after) for step in result.steps] == [("s1", ())]


@pytest.mark.parametrize("index", [True, -1, 4, 1.5, "0"])
def test_invalid_function_index_refuses_whole_response(index):
    with pytest.raises(InterpreterFailure, match="invalid_native_decision"):
        _decision(_call("pause", {}, index=index))


@pytest.mark.parametrize(("function", "arguments", "kind", "text"), [
    ("respond_answer", {"answer": "Known fact.", "basis": "general_knowledge"}, "answer", "Known fact."),
    ("respond_clarify", {"question": "Which one?"}, "clarify", "Which one?"),
    ("respond_unsupported", {"reason": "No retrieval is available."}, "unsupported", "No retrieval is available."),
    ("respond_no_request", {"reason": "No current request remains."}, "no_request", "No current request remains."),
])
def test_standalone_metadata_maps_to_public_decision(function, arguments, kind, text):
    decision = _decision(_call(function, arguments), transcript="Please clarify this.")
    assert decision.kind == kind
    assert decision.text == text if kind in {"answer", "clarify"} else decision.reason == text


@pytest.mark.parametrize("calls", [
    (_call("pause", {}), _call("respond_unsupported", {"reason": "No."})),
    (_call("respond_clarify", {"question": "Which?"}), _call("respond_answer", {"answer": "X", "basis": "general_knowledge"})),
    (_call("respond_answer", {"answer": "A", "basis": "general_knowledge"}),
     _call("respond_answer", {"answer": "B", "basis": "general_knowledge"})),
    (_call("pause", {}), _call("resume", {}), _call("next_track", {}), _call("previous_track", {})),
    (_call("made_up", {}),),
])
def test_invalid_outcome_combinations_refuse_the_whole_response(calls):
    with pytest.raises(InterpreterFailure, match="invalid_native_decision"):
        _decision(*calls)


def test_prose_unknown_arguments_bad_types_enums_and_numeric_ungrounding_fail():
    cases = [
        _response(_call("pause", {}), content="Paused."),
        _response(_call("pause", {"extra": True})),
        _response(_call("play_song", {"title": "Madira", "artist": "Seedhe Maut", "radio": "yes"})),
        _response(_call("search_browser", {"query": "news", "browser": "Firefox"})),
        _response(_call("set_volume", {"level": 55})),
        _response(_call("respond_answer", {"answer": "X", "basis": "the transcript"})),
    ]
    for response in cases:
        with pytest.raises(InterpreterFailure, match="invalid_native_decision"):
            decode_native_response(response, Transcript("Play Madira by Seedhe Maut and set volume to 40."), None)


@pytest.mark.parametrize("content", ["  \n\t  ", ""])
def test_empty_or_whitespace_native_content_is_allowed(content):
    result = decode_native_response(_response(_call("pause", {}), content=content),
                                   Transcript("Pause music."), None)
    assert result.steps[0].capability == "pause"


@pytest.mark.parametrize("role", ["user", "system", "tool"])
def test_non_assistant_native_message_role_is_refused(role):
    response = _response(_call("pause", {}))
    response["message"]["role"] = role
    with pytest.raises(InterpreterFailure, match="invalid_native_decision"):
        decode_native_response(response, Transcript("Pause music."), None)


def test_documented_zero_default_is_allowed_but_explicit_values_are_grounded():
    result = _decision(_call("change_volume", {"direction": "up", "amount": 0, "unit": "points"}),
                       transcript="Increase the volume.")
    assert result.steps[0].arguments["amount"] == 0
    with pytest.raises(InterpreterFailure, match="invalid_native_decision"):
        _decision(_call("change_volume", {"direction": "up", "amount": 25, "unit": "points"}),
                  transcript="Increase the volume.")


def test_codex_binds_full_current_turn_and_never_uses_quote_or_prior_authority():
    full = "Use Codex to inspect this report, and include the entire current request verbatim."
    decision = _decision(_call("codex_request", {"source_ref": "current_turn"}), transcript=full)
    assert decision.steps[0].arguments["request"] == full
    invalid = [
        ("My friend said, 'Use Codex to inspect this.'", {"source_ref": "current_turn"}),
        ("Do not use Codex; pause.", {"source_ref": "current_turn"}),
        ("Use Codex to inspect this.", {"source_ref": "previous_turn"}),
        ("Use Codex to inspect this.", {"source_ref": "current_turn", "request": "partial"}),
    ]
    for transcript, args in invalid:
        with pytest.raises(InterpreterFailure, match="invalid_native_decision"):
            _decision(_call("codex_request", args), transcript=transcript)
    pending = {"pending_request": {"question": "Use Codex?", "sources": [full]}}
    with pytest.raises(InterpreterFailure, match="invalid_native_decision"):
        decode_native_response(_response(_call("codex_request", {"source_ref": "current_turn"})),
                              Transcript("Pause music."), pending)


def test_full_transcript_and_source_are_not_mutated_or_truncated():
    transcript = Transcript("Use Codex to inspect this. " + "x" * 500, grounding_sources=("Approved " + "y" * 800,))
    context = {"recent_turns": [{"role": "user", "text": "Earlier"}]}
    response = _response(_call("codex_request", {"source_ref": "current_turn"}))
    copies = copy.deepcopy((transcript, context, response))
    result = decode_native_response(response, transcript, context)
    assert result.steps[0].arguments["request"] == transcript.text
    assert (transcript, context, response) == copies


def _proof(body, model="local-test", **overrides):
    encoded = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    values = {"prompt_tokens": 100,
              "request_sha256": hashlib.sha256(encoded).hexdigest(),
              "context_tokens": 8192, "model": model, "source": "observed provider prompt count"}
    values.update(overrides)
    return NativeAdmissionProof(**values)


class FakeTransport:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def __call__(self, body, timeout):
        self.calls.append((body, timeout))
        return self.response


def test_diagnostic_requires_injected_transport_and_exact_request_admission():
    with pytest.raises(ValueError, match="transport is required"):
        NativeDecisionDiagnostic("local-test", None, lambda *_: None)
    transport = FakeTransport(_response(_call("pause", {})))
    calls = []

    def admission(body, timeout):
        calls.append((copy.deepcopy(body), timeout))
        return _proof(body)

    diagnostic = NativeDecisionDiagnostic("local-test", transport, admission)
    result = diagnostic.decide(Transcript("Pause music."))
    assert result.steps[0].capability == "pause"
    assert calls and transport.calls
    body, timeout = transport.calls[0]
    assert body["stream"] is False and body["think"] is False
    assert body["options"] == {"temperature": 0, "num_ctx": 8192, "num_predict": 384}
    assert body["keep_alive"] == "60s"
    assert 0 < timeout <= 3
    assert diagnostic.last_admission == _proof(body)
    assert diagnostic.admission_status == "experimental_exact_request_preflight"
    # This successful response intentionally omits prompt_eval_count: only the
    # matching injected preflight proves admission, never an inferred cache hit.


@pytest.mark.parametrize("proof_changes", [
    {"prompt_tokens": -1}, {"prompt_tokens": True}, {"prompt_tokens": 7809},
    {"request_sha256": "0" * 64}, {"context_tokens": 4096}, {"context_tokens": True},
    {"model": "other-model"}, {"source": " "},
])
def test_bad_admission_proof_blocks_transport(proof_changes):
    transport = FakeTransport(_response(_call("pause", {})))
    diagnostic = NativeDecisionDiagnostic("local-test", transport,
                                          lambda body, _timeout: _proof(body, **proof_changes))
    with pytest.raises(InterpreterFailure, match="invalid_native_admission_proof"):
        diagnostic.decide(Transcript("Pause music."))
    assert transport.calls == []


def test_preflight_and_transport_timeouts_share_the_diagnostic_deadline():
    transport = FakeTransport(_response(_call("pause", {})))

    def preflight_timeout(_body, remaining):
        assert 0 < remaining <= 0.5
        raise TimeoutError("preflight timeout")

    diagnostic = NativeDecisionDiagnostic("local-test", transport, preflight_timeout, timeout=0.5)
    with pytest.raises(InterpreterUnavailable, match="preflight timed out"):
        diagnostic.decide(Transcript("Pause music."))
    assert transport.calls == []

    def admission(body, _remaining):
        return _proof(body)

    def transport_timeout(_body, remaining):
        assert 0 < remaining <= 0.5
        raise TimeoutError("provider timeout")

    diagnostic = NativeDecisionDiagnostic("local-test", transport_timeout, admission, timeout=0.5)
    with pytest.raises(InterpreterUnavailable, match="transport timed out"):
        diagnostic.decide(Transcript("Pause music."))


def test_response_count_and_oversized_response_are_rejected():
    with pytest.raises(InterpreterFailure, match="invalid_native_decision"):
        decode_native_response(_response(_call("pause", {}), prompt_eval_count=7809),
                              Transcript("Pause music."), None)
    with pytest.raises(InterpreterFailure, match="invalid_native_decision"):
        decode_native_response(_response(_call("pause", {}), prompt_eval_count=True),
                              Transcript("Pause music."), None)
    huge = _response(_call("pause", {}), extra="x" * 65_536)
    with pytest.raises(InterpreterFailure, match="invalid_native_decision"):
        decode_native_response(huge, Transcript("Pause music."), None)


def test_transcript_overflow_and_truncation_are_not_silently_repaired():
    with pytest.raises(InterpreterFailure, match="truncated_transcript"):
        build_native_messages(Transcript("Pause music.", truncated=True), None)
    with pytest.raises(InterpreterFailure, match="transcript_length"):
        build_native_messages(Transcript("x" * 12_001), None)
