from __future__ import annotations

import pytest

from assistant.conversation_contracts import (
    CAPABILITIES, DECISION_SCHEMA, ContractError, Decision, Step, Transcript, step_to_intent,
    validate_decision,
)
from assistant.intent import Action


def _answer(text="What is virtuoso?"):
    return {"kind": "answer", "answer": "A highly skilled performer.",
            "basis": "general_knowledge", "evidence_quotes": [text]}


def _plan(capability, arguments, quote="play Madira by Seedhe Maut", *, after=None):
    return {"kind": "plan", "steps": [{"id": "s1", "capability": capability,
            "arguments": arguments, "after": after or []}], "evidence_quotes": [quote]}


@pytest.mark.parametrize("payload,kind", [
    (_answer(), "answer"),
    ({"kind": "clarify", "question": "Which album?", "evidence_quotes": ["that album"]}, "clarify"),
    (_plan("play_song", {"title": "Madira", "artist": "Seedhe Maut"}), "plan"),
    ({"kind": "no_request", "reason": "Quoted speech.", "evidence_quotes": []}, "no_request"),
    ({"kind": "unsupported", "reason": "No retrieval capability.", "evidence_quotes": []}, "unsupported"),
])
def test_all_decision_branches(payload, kind):
    source = "What is virtuoso?" if kind == "answer" else (
        "play Madira by Seedhe Maut" if kind == "plan" else
        "that album" if kind == "clarify" else "background conversation")
    result = validate_decision(payload, Transcript(source))
    assert isinstance(result, Decision)
    assert result.kind == kind


def test_evidence_quote_schema_matches_branch_requirements():
    branches = DECISION_SCHEMA["oneOf"]
    by_kind = {branch["properties"]["kind"]["const"]: [] for branch in branches}
    for branch in branches:
        by_kind[branch["properties"]["kind"]["const"]].append(
            branch["properties"]["evidence_quotes"])
    assert by_kind["answer"][0]["minItems"] == 1
    assert [schema["minItems"] for schema in by_kind["plan"]] == [1, 1]
    assert "minItems" not in by_kind["clarify"][0]
    assert "minItems" not in by_kind["no_request"][0]
    assert "minItems" not in by_kind["unsupported"][0]


def test_answer_and_plan_require_evidence_but_clarification_may_omit_it():
    with pytest.raises(ContractError):
        validate_decision({**_answer(), "evidence_quotes": []}, Transcript("What is virtuoso?"))
    for attached in (False, True):
        payload = {**_plan("pause", {}, "pause"), "evidence_quotes": []}
        if attached:
            payload.update(answer="A stable fact.", answer_basis="general_knowledge")
        with pytest.raises(ContractError):
            validate_decision(payload, Transcript("pause"))
    assert validate_decision({"kind": "clarify", "question": "Which one?",
                              "evidence_quotes": []}, Transcript("that one")).kind == "clarify"


def test_strict_branch_fields_unknown_tools_and_arguments():
    with pytest.raises(ContractError):
        validate_decision({**_answer(), "extra": 1}, Transcript("What is virtuoso?"))
    with pytest.raises(ContractError):
        validate_decision(_plan("ignore", {"reason": "no"}), Transcript("play Madira by Seedhe Maut"))
    with pytest.raises(ContractError):
        validate_decision(_plan("pause", {"unexpected": True}), Transcript("pause"))
    assert "ignore" not in CAPABILITIES


def test_transcript_bounds_control_and_truncation():
    with pytest.raises(ContractError):
        validate_decision(_answer("x"), Transcript("x" * 12001))
    with pytest.raises(ContractError):
        validate_decision(_answer("x"), Transcript("x", truncated=True))
    with pytest.raises(ContractError):
        validate_decision(_answer("x"), Transcript("x\x00"))
    assert validate_decision(_answer("x"), Transcript("x")).text == "A highly skilled performer."
    with pytest.raises(ContractError):
        validate_decision(_answer("x"), Transcript(123))
    padded = {"kind": "clarify", "question": " " * 501 + "x", "evidence_quotes": []}
    with pytest.raises(ContractError):
        validate_decision(padded, Transcript("x"))


@pytest.mark.parametrize("kind", [[], {}])
def test_invalid_decision_kind_is_contract_error(kind):
    payload = {"kind": kind}
    with pytest.raises(ContractError):
        validate_decision(payload, Transcript("x"))


@pytest.mark.parametrize("level", [True, -1, 101])
def test_volume_rejects_bool_and_out_of_range(level):
    with pytest.raises(ContractError):
        validate_decision(_plan("set_volume", {"level": level}, "set volume to 50"),
                          Transcript("set volume to 50"))


@pytest.mark.parametrize("source,level", [
    ("Set the volume to 50.", 50),
    ("Set volume to zero", 0),
    ("Set volume to fifty", 50),
    ("Set volume to twenty five", 25),
    ("Set volume to one hundred", 100),
    ("Full volume", 100),
    ("Max volume", 100),
    ("Half volume", 50),
    ("Set volume fifty and pause", 50),
    ("Set volume 50 percent and pause", 50),
])
def test_absolute_volume_values_are_grounded_by_supported_number_phrases(source, level):
    decision = validate_decision(_plan("set_volume", {"level": level}, source),
                                 Transcript(source))
    assert decision.steps[0].arguments["level"] == level


@pytest.mark.parametrize("source,level,grounding_sources", [
    ("Pause and define virtuoso", 50, ()),
    ("Set volume", 50, ("50",)),
    ("Set volume to fifty two", 50, ()),
    ("Set volume to fifty two", 2, ()),
    ("Set volume to .50", 50, ()),
    ("Set volume to + .50", 50, ()),
    ("Set volume to 50,5", 50, ()),
    ("Set volume to 50,5", 5, ()),
    ("Set volume to 50.5", 5, ()),
    ("Set volume to fifty point ten five", 10, ()),
    ("Set volume to fifty point ten five", 5, ()),
    ("Set volume to two hundred", 2, ()),
    ("Set volume to two hundred", 100, ()),
    ("Set volume to one hundred fifty", 100, ()),
    ("Set volume to one hundred fifty", 50, ()),
    ("Set volume to one thousand fifty", 100, ()),
    ("Set volume to one thousand fifty", 50, ()),
    ("Set volume to one hundred and fifty", 100, ()),
    ("Set volume to one hundred and fifty", 50, ()),
    ("Set volume to fifty point ten", 10, ()),
    ("Set volume to 1000", 100, ()),
    ("Set volume to 51", 50, ()),
    ("Set volume to -50", 50, ()),
    ("Set volume to +50", 50, ()),
    ("Set volume to minus fifty", 50, ()),
    ("Set volume to negative fifty", 50, ()),
    ("Set volume to 50.5", 50, ()),
])
def test_absolute_volume_rejects_unspoken_or_partial_values(source, level, grounding_sources):
    with pytest.raises(ContractError):
        validate_decision(_plan("set_volume", {"level": level}, source),
                          Transcript(source, grounding_sources=grounding_sources))


def test_relative_volume_requires_spoken_nonzero_amount_but_allows_default_zero():
    default_source = "Lower the volume a little"
    default = validate_decision(_plan("change_volume", {
        "direction": "down", "amount": 0, "unit": "points"}, default_source),
        Transcript(default_source))
    assert default.steps[0].arguments["amount"] == 0

    source = "Lower volume by twenty five percent"
    explicit = validate_decision(_plan("change_volume", {
        "direction": "down", "amount": 25, "unit": "percent"}, source), Transcript(source))
    assert explicit.steps[0].arguments["amount"] == 25

    with pytest.raises(ContractError):
        validate_decision(_plan("change_volume", {
            "direction": "down", "amount": 25, "unit": "percent"}, default_source),
            Transcript(default_source))


def test_valid_compound_number_and_separate_followup_number_are_grounded():
    source = "Set volume to fifty-two. Then pause."
    valid = validate_decision(_plan("set_volume", {"level": 52}, source), Transcript(source))
    assert valid.steps[0].arguments["level"] == 52
    for fragment in (50, 2):
        with pytest.raises(ContractError):
            validate_decision(_plan("set_volume", {"level": fragment}, source), Transcript(source))

    separate = "Set volume to fifty then pause"
    valid_separate = validate_decision(
        _plan("set_volume", {"level": 50}, separate), Transcript(separate))
    assert valid_separate.steps[0].arguments["level"] == 50

    compound = "Set volume 25 then open Spotify and volume 50"
    for level in (25, 50):
        grounded = validate_decision(
            _plan("set_volume", {"level": level}, compound), Transcript(compound))
        assert grounded.steps[0].arguments["level"] == level


def test_plan_dependencies_ids_duplicate_ops_and_step_limit():
    source = "pause and resume"
    payload = {"kind": "plan", "steps": [
        {"id": "a", "capability": "pause", "arguments": {}, "after": []},
        {"id": "b", "capability": "resume", "arguments": {}, "after": ["a"]}],
        "evidence_quotes": [source]}
    assert len(validate_decision(payload, Transcript(source)).steps) == 2
    for refs in [["missing"], ["b"], [[],]]:
        bad = {**payload, "steps": [payload["steps"][0],
               {**payload["steps"][1], "after": refs}]}
        with pytest.raises(ContractError):
            validate_decision(bad, Transcript(source))
    duplicate = {**payload, "steps": [payload["steps"][0],
                {**payload["steps"][0], "id": "b", "after": ["a"]}]}
    with pytest.raises(ContractError):
        validate_decision(duplicate, Transcript(source))
    many = {**payload, "steps": payload["steps"] * 2}
    with pytest.raises(ContractError):
        validate_decision(many, Transcript(source))


def test_grounding_uses_normalized_whole_tokens_and_no_punctuation_rule():
    source = "Please play Madira—by Seedhe Maut tonight"
    accepted = _plan("play_song", {"title": "madira", "artist": "seedhe maut"},
                     "play Madira—by Seedhe Maut")
    assert validate_decision(accepted, Transcript(source)).kind == "plan"
    invented = _plan("play_song", {"title": "Madira Deluxe", "artist": "Seedhe Maut"},
                     "play Madira by Seedhe Maut")
    with pytest.raises(ContractError):
        validate_decision(invented, Transcript(source))
    no_punctuation = _plan("play_song", {"title": "Madira", "artist": "Seedhe Maut"},
                           "play Madira by Seedhe Maut")
    assert validate_decision(no_punctuation, Transcript("play Madira by Seedhe Maut")).kind == "plan"


@pytest.mark.parametrize("capability,title", [
    ("play_song", "Madira"), ("play_album", "N"), ("queue_song", "Madira"),
])
def test_title_only_music_requests_allow_empty_artist(capability, title):
    source = "play " + title
    result = validate_decision(_plan(capability, {"title": title, "artist": ""}, source),
                               Transcript(source))
    assert result.steps[0].arguments["artist"] == ""


def test_album_track_and_show_intents_remain_distinct():
    cases = [
        ("play_song", {"title": "Madira", "artist": "Seedhe Maut"}, Action.PLAY_TRACK),
        ("play_album", {"title": "N", "artist": "Seedhe Maut"}, Action.PLAY_ALBUM),
        ("show_track", {"query": "Madira"}, Action.SHOW_TRACK),
        ("show_album", {"query": "N"}, Action.SHOW_ALBUM),
    ]
    for capability, args, expected in cases:
        intent = step_to_intent(Step("s", capability, args))
        assert intent.action is expected
        assert intent.source == "conversation"
        assert intent.matched_rule == "conversation_" + capability
        if capability == "play_song":
            # Explicit track wording is not represented on Step; coordinator
            # context must supply it rather than inferring from play_song.
            assert intent.explicit_track is False


def test_codex_receives_original_full_transcript_only():
    source = "Please ask Codex to review this entire long prompt: " + "context " * 600
    payload = _plan("codex_request", {"request": source}, "ask Codex")
    result = validate_decision(payload, Transcript(source))
    assert result.steps[0].arguments["request"] == source
    with pytest.raises(ContractError):
        step_to_intent(result.steps[0])
    compound = {**payload, "steps": [payload["steps"][0],
                {"id": "s2", "capability": "pause", "arguments": {}, "after": ["s1"]}]}
    with pytest.raises(ContractError):
        validate_decision(compound, Transcript(source))
    altered = _plan("codex_request", {"request": "ask Codex"}, "ask Codex")
    with pytest.raises(ContractError):
        validate_decision(altered, Transcript(source))


def test_every_executable_capability_maps_to_existing_intent():
    expected = {
        "open_app": Action.OPEN_APP, "search_browser": Action.SEARCH_BROWSER,
        "show_track": Action.SHOW_TRACK, "show_album": Action.SHOW_ALBUM,
        "restart_album": Action.RESTART_ALBUM, "start_jam": Action.START_JAM,
        "play_song": Action.PLAY_TRACK, "play_artist": Action.PLAY_ARTIST,
        "play_album": Action.PLAY_ALBUM, "queue_song": Action.QUEUE,
        "play_playlist": Action.PLAY_PLAYLIST, "pause": Action.PAUSE,
        "resume": Action.RESUME, "next_track": Action.NEXT,
        "previous_track": Action.PREV, "restart_track": Action.RESTART,
        "set_volume": Action.VOLUME, "change_volume": Action.VOLUME,
    }
    samples = {
        "open_app": {"name": "Notepad"}, "search_browser": {"query": "python logging", "browser": "Opera"},
        "show_track": {"query": "Madira"}, "show_album": {"query": "N"},
        "restart_album": {}, "start_jam": {}, "play_song": {"title": "Madira", "artist": "Seedhe Maut", "radio": False},
        "play_artist": {"artist": "Seedhe Maut", "radio": True},
        "play_album": {"title": "N", "artist": "Seedhe Maut"},
        "queue_song": {"title": "Madira", "artist": "Seedhe Maut"},
        "play_playlist": {"name": "Liked Songs", "mine": True}, "pause": {}, "resume": {},
        "next_track": {}, "previous_track": {}, "restart_track": {}, "set_volume": {"level": 40},
        "change_volume": {"direction": "down", "amount": 10, "unit": "percent"},
    }
    assert set(expected) == set(CAPABILITIES) - {"codex_request"}
    for name, action in expected.items():
        assert step_to_intent(Step("s", name, samples[name])).action is action
    assert step_to_intent(Step("s", "play_playlist", samples["play_playlist"])).library_only is True
    assert step_to_intent(Step("s", "change_volume", samples["change_volume"])).volume_mode == "relative_percent"


def test_volume_default_step_and_explicit_percent_mapping():
    default_step = step_to_intent(Step("s", "change_volume", {
        "direction": "up", "amount": 0, "unit": "points"}))
    assert default_step.value is None
    assert default_step.query == "up"
    assert default_step.volume_mode == "absolute"
    percent_step = step_to_intent(Step("s", "change_volume", {
        "direction": "down", "amount": 15, "unit": "percent"}))
    assert percent_step.value == 15
    assert percent_step.query == "down"
    assert percent_step.volume_mode == "relative_percent"


def test_plan_can_carry_stable_answer_pair_and_plain_plan_stays_empty():
    source = "Pause and explain what a virtuoso is"
    plain_payload = _plan("pause", {}, source)
    plain = validate_decision(plain_payload, Transcript(source))
    assert plain.answer == ""
    assert plain.answer_basis == ""

    paired = {**plain_payload, "answer": "A highly skilled performer.",
              "answer_basis": "general_knowledge"}
    decision = validate_decision(paired, Transcript(source))
    assert decision.answer == "A highly skilled performer."
    assert decision.answer_basis == "general_knowledge"
    # Existing five-position Decision construction retains its old field order.
    positional = Decision("plan", "", (), (), "")
    assert positional.answer == "" and positional.answer_basis == ""


@pytest.mark.parametrize("extra", [
    {"answer": "A stable fact."},
    {"answer_basis": "general_knowledge"},
    {"answer": "A stable fact.", "answer_basis": "current_weather"},
    {"answer": 42, "answer_basis": "general_knowledge"},
    {"answer": "x" * 1201, "answer_basis": "general_knowledge"},
])
def test_plan_answer_requires_valid_complete_pair(extra):
    source = "pause and explain virtuoso"
    payload = {**_plan("pause", {}, source), **extra}
    with pytest.raises(ContractError):
        validate_decision(payload, Transcript(source))


@pytest.mark.parametrize("sources", [
    ["Creep"], ("Creep", "Radiohead", "OK Computer"),
    (123,), ("",), ("x" * 1001,), ("bad\x00source",),
])
def test_grounding_sources_are_app_owned_and_bounded(sources):
    with pytest.raises(ContractError):
        validate_decision(_answer("question"), Transcript("question", grounding_sources=sources))


def test_current_turn_can_resolve_prior_title_without_inventing_artist():
    current = "Play it by Radiohead"
    transcript = Transcript(current, grounding_sources=("Creep",))
    payload = _plan("play_song", {"title": "Creep", "artist": "Radiohead"}, current)
    decision = validate_decision(payload, transcript)
    assert decision.steps[0].arguments == {"title": "Creep", "artist": "Radiohead"}

    invented = _plan("play_song", {"title": "Creep Deluxe", "artist": "Radiohead"}, current)
    with pytest.raises(ContractError):
        validate_decision(invented, transcript)


def test_actionable_plan_cannot_rely_only_on_prior_source_quote():
    current = "Pause now"
    payload = _plan("pause", {}, "Creep")
    with pytest.raises(ContractError):
        validate_decision(payload, Transcript(current, grounding_sources=("Creep",)))


def test_codex_request_stays_bound_to_current_text_with_grounding_sources():
    current = "Use Codex to review this request"
    payload = _plan("codex_request", {"request": current}, current)
    result = validate_decision(payload, Transcript(current, grounding_sources=("older prompt",)))
    assert result.steps[0].arguments["request"] == current
    changed = _plan("codex_request", {"request": "older prompt"}, current)
    with pytest.raises(ContractError):
        validate_decision(changed, Transcript(current, grounding_sources=("older prompt",)))

