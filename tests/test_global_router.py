import io
import json

import pytest

from assistant.global_router import LocalGlobalRouter, directly_addressed, validate_route, ground_action


REQUEST = "Open a Codex chat and tell it to check out my Gmail files"
PLAN = {"action": "create_ask", "text": "check out my Gmail files", "title": ""}


def classification(action="ask", direct=True, target_kind="unspecified", target_name=""):
    return {"action": action, "direct_request": direct, "target_kind": target_kind, "target_name": target_name}


@pytest.mark.parametrize("text", ["play Codex", '"Open a Codex chat"', "He said ask Codex to fix it", "Don't open a Codex chat", "Please do not ask Codex anything"])
def test_incidental_quoted_negated_requests_do_not_route(text):
    assert not directly_addressed(text)


def test_exact_transcript_payload_only():
    result = validate_route(REQUEST, PLAN)
    assert result.action == "create_ask"
    assert result.text == "check out my Gmail files"
    for changed in [dict(PLAN, text="check Gmail"), dict(PLAN, action="approve"),
                    dict(PLAN, title="Imaginary"), dict(PLAN, extra="x")]:
        with pytest.raises(ValueError):
            validate_route(REQUEST, changed)


def test_prompt_negation_and_send_it_are_payload():
    assert directly_addressed("Open a Codex chat and tell it don't change files")
    assert directly_addressed("Could you ask Codex don't change files")
    assert validate_route("Could you ask Codex to send it", {
        "action": "ask", "text": "send it", "title": ""}).text == "send it"


@pytest.mark.parametrize("full,partial", [
    ("inspect my current project without changing files", "inspect my current project"),
    ("first ask me before editing anything", "editing anything"),
    ("NOT delete files; only list them", "only list them"),
])
def test_prompt_safety_constraints_cannot_be_truncated(full, partial):
    source = "Could you ask Codex to " + full
    with pytest.raises(ValueError):
        validate_route(source, {"action": "ask", "text": partial, "title": ""})
    assert validate_route(source, {"action": "ask", "text": full, "title": ""}).text == full


@pytest.mark.parametrize("original,proposal", [
    ("investigate missed wake words without changing files.", "investigate missed wake words without changing files"),
    ("Keep foo.py. unchanged?", "Keep foo.py. unchanged"),
    ("Explain why?", "Explain why."),
])
def test_terminal_punctuation_proposals_restore_full_original(original, proposal):
    source = "Could you ask Codex to " + original
    route = validate_route(source, {"action": "ask", "text": proposal, "title": ""})
    assert route.text == original


def test_named_compound_omitted_final_period_restores_source():
    source = "Please open a new Codex chat called Music fixes and tell it to investigate missed wake words without changing files."
    route = validate_route(source, {"action": "create_ask", "text": "investigate missed wake words without changing files", "title": "Music fixes"})
    assert route.text == "investigate missed wake words without changing files."


@pytest.mark.parametrize("original,proposal", [
    ("Keep foo.py. unchanged?", "Keep foopy unchanged"),
    ('Keep the literal "why?"', 'Keep the literal "why"'),
    ("Investigate without changing files.", "Investigate."),
    ("?", ""),
])
def test_punctuation_tolerance_does_not_change_interior_or_drop_words(original, proposal):
    with pytest.raises(ValueError):
        validate_route("Could you ask Codex to " + original, {"action": "ask", "text": proposal, "title": ""})


def test_wrong_creation_action_is_not_repaired_into_ask():
    source = "Can you ask Codex to first ask me before editing anything"
    with pytest.raises(ValueError):
        validate_route(source, {"action": "create_ask", "text": "first ask me before editing anything", "title": ""})


def test_incidental_codex_subject_is_not_addressed():
    assert not directly_addressed("Codex is useful")


def test_compound_cannot_discard_prompt_with_create_classification():
    with pytest.raises(ValueError):
        validate_route(REQUEST, {"action": "create", "text": "", "title": ""})


@pytest.mark.parametrize("source", [
    "Could you ask Codex to create a Codex chat",
    "Please tell Codex open a Codex chat called Build",
])
def test_create_mentioned_in_payload_does_not_authorize_creation(source):
    with pytest.raises(ValueError):
        validate_route(source, {"action": "create", "text": "", "title": ""})


@pytest.mark.parametrize("source,title", [
    ("Open a Codex chat", ""),
    ("Please create a new Codex chat called Build", "Build"),
    ("Could you open a Codex chat called Morning Plan", "Morning Plan"),
])
def test_explicit_creation_wrapper_and_complete_title_are_allowed(source, title):
    assert validate_route(source, {"action": "create", "text": "", "title": title}).title == title
    if title:
        with pytest.raises(ValueError):
            validate_route(source, {"action": "create", "text": "", "title": ""})


def test_router_disables_proxies_and_redirects(monkeypatch):
    from assistant import global_router
    captured = []
    monkeypatch.setattr(global_router.urllib.request, "build_opener", lambda *handlers: captured.extend(handlers))
    LocalGlobalRouter("existing-qwen")
    assert captured[0].proxies == {}
    with pytest.raises(ValueError):
        captured[1].redirect_request(None, None, 302, "", {}, "https://example.com")


def test_oversize_direct_request_refused_without_network():
    router = LocalGlobalRouter("existing-qwen")
    router._opener = Opener(b"")
    assert router.route("Ask Codex " + "x" * 12000).action == "refused"
    assert not router._opener.calls


class Opener:
    def __init__(self, content):
        self.content = content
        self.calls = []

    def open(self, request, timeout):
        self.calls.append((request, timeout))
        return io.BytesIO(self.content)


def test_local_request_is_schema_bounded_and_uses_existing_model():
    router = LocalGlobalRouter("existing-qwen")
    router._opener = Opener(json.dumps({"message": {"content": json.dumps(classification("create_ask", target_kind="new"))}}).encode())
    route = router.route(REQUEST)
    assert route.text == REQUEST
    assert route.inferred
    request, timeout = router._opener.calls[0]
    assert request.full_url == "http://127.0.0.1:11434/api/chat"
    assert timeout == 2.5
    body = json.loads(request.data)
    assert body["model"] == "existing-qwen"
    assert body["format"]["additionalProperties"] is False
    assert set(body["format"]["properties"]) == {"action", "direct_request", "target_kind", "target_name"}
    assert body["options"]["num_predict"] == 128
    assert body["think"] is False
    assert body["options"]["num_ctx"] == 4096
    assert body["keep_alive"] == -1


@pytest.mark.parametrize("body", [b"not json", b"{}", b"x" * 65537], ids=["invalid_json", "missing_envelope", "oversize"])
def test_bad_response_is_safe_refusal(body):
    router = LocalGlobalRouter("existing-qwen")
    router._opener = Opener(body)
    assert router.route(REQUEST).action == "refused"


def test_timeout_is_safe_refusal():
    router = LocalGlobalRouter("existing-qwen")
    class TimeoutOpener:
        def open(self, *args, **kwargs):
            raise TimeoutError()
    router._opener = TimeoutOpener()
    assert router.route(REQUEST).action == "refused"


def test_router_uses_shared_runtime_policy_and_serialized_request_adapter():
    from assistant.model_runtime import LocalModelRuntime
    runtime = LocalModelRuntime("existing-qwen", num_ctx=8192, keep_alive=-1)
    router = LocalGlobalRouter("existing-qwen", runtime=runtime)
    router._opener = Opener(json.dumps({"message": {"content": json.dumps(
        classification("ask", target_kind="selected"))}}).encode())
    assert router.route(REQUEST).action == "ask"
    request, timeout = router._opener.calls[0]
    body = json.loads(request.data)
    assert timeout == 2.5
    assert body["options"]["num_ctx"] == 8192
    assert body["keep_alive"] == -1
    assert runtime.state == "ready"


def test_router_runtime_admission_refusal_does_not_send_request():
    from assistant.model_runtime import LocalModelRuntime
    runtime = LocalModelRuntime("existing-qwen")
    runtime.mark_warming()
    router = LocalGlobalRouter("existing-qwen", runtime=runtime)
    router._opener = Opener(b"")
    assert router.route(REQUEST).action == "refused"
    assert router._opener.calls == []
    assert runtime.state == "warming"


def test_chat_control_aliases_handle_asr_punctuation_without_changing_names():
    from assistant.global_commands import parse_global
    assert parse_global("Open Codex chat current.").text == "current"
    assert parse_global("Open current Codex chat!").action == "open"
    assert parse_global("Select the last Codex chat") is None
    assert parse_global("Select last Codex chat?").text == "last"
    assert parse_global("Open Codex chat Build.v2.").text == "Build.v2."


@pytest.mark.parametrize("proposal", [
    {"action": "ask"}, {"action": "ask", "direct_request": "true"},
    {"action": "ask", "direct_request": 1}, {"action": "ask", "direct_request": None},
    {"action": "approve", "direct_request": True}, {"action": "use_luna", "direct_request": True},
    {"action": "switch_tab", "direct_request": True}, {"action": "send", "direct_request": True},
    {"action": "ask", "direct_request": True, "title": "Injected"},
    {"action": "ask", "direct_request": True, "text": "Shortened"},
    {"action": [], "direct_request": True},
])
def test_classifier_schema_rejects_coercion_arguments_and_forbidden_actions(proposal):
    from assistant.global_router import validate_classification
    with pytest.raises(ValueError):
        validate_classification("Have Codex inspect this without changing anything", dict(proposal, target_kind="unspecified", target_name=""))


@pytest.mark.parametrize("source", [
    "Make a codex chat that will audit my downloads folder",
    "I need a fresh Codex conversation to explain the project",
    "Would you arrange for Codex to inspect this repository?",
    "Set up a Codex conversation for a review, but do NOT edit or delete anything?!",
])
def test_natural_codex_candidates_reach_local_classifier_and_keep_complete_source(source):
    router = LocalGlobalRouter("existing-qwen")
    router._opener = Opener(json.dumps({"message": {"content": json.dumps(classification("create_ask", target_kind="new"))}}).encode())
    command = router.route(source)
    assert command.inferred and command.text == source and command.title == ""
    assert json.loads(router._opener.calls[0][0].data)["messages"][-1]["content"] == source


@pytest.mark.parametrize("source", [
    "Don't create a Codex chat", 'He said "make a Codex chat"',
    "Yesterday I asked Codex for help", "Play the Codex soundtrack",
    'The lyrics say "Have Codex make a conversation"', "Codex is useful",
])
@pytest.mark.parametrize("proposal", [
    {"action": "ignore", "direct_request": False},
    {"action": "create", "direct_request": False},
    {"action": "ignore", "direct_request": True},
])
def test_classifier_ignore_or_nondirect_returns_to_music(source, proposal):
    from assistant.global_router import routing_candidate
    router = LocalGlobalRouter("existing-qwen")
    proposal = dict(proposal, target_kind="new" if proposal["action"] == "create" else "unspecified", target_name="")
    router._opener = Opener(json.dumps({"message": {"content": json.dumps(proposal)}}).encode())
    assert router.route(source) is None
    assert len(router._opener.calls) == int(routing_candidate(source))


def test_classifier_known_music_never_calls_network_without_codex():
    router = LocalGlobalRouter("existing-qwen")
    router._opener = Opener(b"")
    assert router.route("Play music") is None
    assert router.route(None) is None
    assert router._opener.calls == []


@pytest.mark.parametrize("source", [
    "what is virtuoso", "what is Codex", "meaning of a coding assistant",
    "Use GPT 6 Luna",
    "I don't want Codex to inspect my project",
    "She said I'd like a coding assistant to inspect my project",
    "tell my mother to call me", "ask Spotify to play Red",
    "create a new chat in WhatsApp", "start a conversation with my mother",
    'my friend said "Have Codex inspect this"', '"Ask Codex to inspect this"',
    "do not start Codex", "Please do not ask Codex anything",
    "play Codex soundtrack", "Yesterday I asked Codex for help",
    "What if Codex inspected this?", "Could you explain what Codex is?",
])
def test_explicit_codex_admission_rejects_questions_and_nonrequests(source):
    from assistant.global_router import explicit_codex_request, routing_candidate
    assert not explicit_codex_request(source)
    assert not routing_candidate(source)


@pytest.mark.parametrize("source", [
    "Have Codex inspect this",
    "Use Codex to implement a parser",
    "I'd like a coding assistant to inspect my project",
    "I want Codex to inspect my project",
    "I would like the coding assistant to review this project",
    "Create a new Codex chat using GPT-6 Luna to check Gmail orchestration",
    "Ask the Build Notes chat to inspect logs",
    "Could you ask the Build Notes chat to inspect the logs",
    "Make a codex chat that will audit my downloads folder",
    "Would you arrange for Codex to inspect this repository?",
    "I need a fresh Codex conversation to explain the project",
    "Open the current Codex chat to review the failing tests",
    "Please create a new Codex chat",
])
def test_explicit_codex_admission_accepts_source_bound_delegation_and_controls(source):
    from assistant.global_router import explicit_codex_request, routing_candidate
    assert explicit_codex_request(source)
    assert routing_candidate(source)


@pytest.mark.parametrize("source,action,target_kind", [
    ("what is Codex", "ask", "unspecified"),
    ('my friend said "Have Codex inspect this"', "create_ask", "new"),
    ("play Codex soundtrack", "ask", "unspecified"),
    ("tell my mother to call me", "ask", "unspecified"),
    ("create a new chat in WhatsApp", "create", "new"),
])
def test_classifier_cannot_authorize_nonexplicit_source(source, action, target_kind):
    from assistant.global_router import validate_classification
    with pytest.raises(ValueError):
        validate_classification(source, classification(action, True, target_kind=target_kind))


@pytest.mark.parametrize("content", [
    '{"action":"ask","action":"create_ask","direct_request":true}',
    '{"action":"ask","direct_request":false,"direct_request":true}',
])
def test_duplicate_classification_fields_refuse_instead_of_overwriting(content):
    router = LocalGlobalRouter("existing-qwen")
    router._opener = Opener(json.dumps({"message": {"content": content}}).encode())
    assert router.route("Have Codex inspect this without edits").action == "refused"


@pytest.mark.parametrize("source,title", [
    ("Ask the wakeboard transcript chat to check the logs", "wakeboard transcript"),
    ("Send a prompt to check the logs in the Wake World Transcript Chat.", "Wake World Transcript"),
    ("Ask Build Notes chat to inspect this without edits?!", "Build Notes"),
])
def test_named_no_codex_requests_reach_classifier_with_grounded_name(source, title):
    router = LocalGlobalRouter("existing-qwen")
    router._opener = Opener(json.dumps({"message": {"content": json.dumps(classification(target_kind="named", target_name=title.swapcase()))}}).encode())
    route = router.route(source)
    assert route.text == source and route.target_kind == "named" and route.target_name == title
    assert len(router._opener.calls) == 1


@pytest.mark.parametrize("source", [
    "Ask Build Notes to inspect this without edits?!",
    "Send a prompt to my mother",
])
def test_unverified_named_recipient_does_not_reach_classifier(source):
    router = LocalGlobalRouter("existing-qwen")
    router._opener = Opener(b"")
    assert router.route(source) is None
    assert router._opener.calls == []


def test_send_prompt_in_named_chat_is_explicit_and_preserves_source():
    source = "Send a prompt to check the logs in the Wake World Transcript Chat."
    router = LocalGlobalRouter("existing-qwen")
    router._opener = Opener(json.dumps({"message": {"content": json.dumps(
        classification("ask", True, target_kind="named", target_name="Wake World Transcript"))}}).encode())
    command = router.route(source)
    assert command.action == "ask" and command.text == source
    assert command.target_kind == "named" and command.target_name == "Wake World Transcript"
    assert json.loads(router._opener.calls[0][0].data)["messages"][-1]["content"] == source


def test_oversize_explicit_source_refuses_before_candidate_gate():
    router = LocalGlobalRouter("existing-qwen")
    router._opener = Opener(b"")
    assert router.route("Ask Codex " + "x" * 12000).action == "refused"
    assert router._opener.calls == []


@pytest.mark.parametrize("source", ["pause", "play red", "queue Yesterday by the Beatles", "volume 20", "next"])
def test_existing_music_parser_fastpath_bypasses_semantic_classifier(source):
    from assistant.global_router import routing_candidate
    assert not routing_candidate(source)
    assert not routing_candidate("play Codex")


@pytest.mark.parametrize("proposal", [
    classification(target_kind="named", target_name="invented"),
    classification(target_kind="named", target_name=" Build Notes"),
    classification(target_kind="named", target_name="x" * 121),
    classification(target_kind="selected", target_name="Build Notes"),
    classification(target_kind="new"),
    classification("create", target_kind="selected"),
    classification("create_ask", target_kind="unspecified"),
    classification("list", target_kind="named", target_name="Build Notes"),
    classification("ignore", False, target_kind="selected"),
    classification("dictate", target_kind="new"),
    classification("status", target_kind="named", target_name="Build Notes"),
    {"action": "ask", "direct_request": True},
])
def test_strict_target_schema_and_action_invariants(proposal):
    from assistant.global_router import validate_classification
    with pytest.raises(ValueError):
        validate_classification("Ask Build Notes chat to review the files", proposal)


@pytest.mark.parametrize("source", [
    "why are you not hearing me", "play red then reduce volume to 20",
    'He said "ask the Build Notes chat to inspect this"',
    "Yesterday she requested a coding assistant", "The background voice discussed a new chat",
])
def test_unsupported_or_incidental_unknown_requests_honor_ignore(source):
    from assistant.global_router import routing_candidate
    router = LocalGlobalRouter("existing-qwen")
    router._opener = Opener(json.dumps({"message": {"content": json.dumps(classification("ignore", False))}}).encode())
    assert router.route(source) is None
    assert bool(router._opener.calls) == routing_candidate(source)


def test_semantic_navigation_is_grounded_and_cannot_omit_a_task():
    from assistant.global_router import validate_classification
    proposal = classification("open", True, target_kind="named", target_name="Build Notes")
    command = validate_classification("Please show the Build Notes chat", proposal)
    assert command.action == "open" and command.target_name == "Build Notes"
    for source in ["He said open the Build Notes chat", "Open the Build Notes chat and audit the files",
                   "Open the Build Notes chat then send a task",
                   "Open the Build Notes chat to review the failing tests"]:
        with pytest.raises(ValueError):
            validate_classification(source, proposal)


def test_selected_navigation_cannot_swallow_an_infinitive_task():
    from assistant.global_router import validate_classification
    source = "Open the current Codex chat to review the failing tests"
    with pytest.raises(ValueError):
        validate_classification(source, classification("open", True, target_kind="selected"))
    command = validate_classification(source, classification("ask", True, target_kind="selected"))
    assert command.inferred and command.text == source
    assert validate_classification("Open the current Codex chat", classification(
        "open", True, target_kind="selected")).action == "open"


@pytest.mark.parametrize("source", [
    "Create a Codex chat called Build Notes and check the failing tests",
    "Create a Codex chat called Build Notes to inspect the logs",
    "Create a Codex chat called Build Notes for reviewing the logs",
    "Create a Codex chat called Build Notes ready to review the logs",
    "Create a Codex chat called Build Notes. Check the failing tests",
    "Create a Codex chat called Build Notes - inspect the failing tests",
    "Create a Codex chat called Build Notes that checks the failing tests",
    "Create a Codex chat called Build Notes which will inspect the logs",
    "Make a new Codex chat to inspect the logs",
    "Make a new Codex chat for a review",
    "I want a fresh Codex conversation ready to explain this repository",
    "Set up an empty Codex conversation and check the failing tests",
    "Start a Codex chat and leave it blank, then inspect the logs",
    "Start a blank Codex chat. Review the failing tests",
    "Start a Codex chat and don't send a task, but inspect the logs",
    "Would you arrange an empty Codex conversation without sending a task and then inspect the logs?",
    "Arrange a blank Codex chat without submitting a prompt for reviewing the tests",
    "Set up an empty conversation using GPT 6 Luna, then check the failing tests",
    "Make an empty Codex chat without sending a task, ready to explain the repository",
    "Make a new Codex chat. Use GPT 6 Luna and inspect the logs",
    "Make a new Codex chat and check the logs. Use GPT 6 Luna",
])
def test_forced_empty_creation_cannot_discard_task_or_ambiguous_suffix(source):
    from assistant.global_router import CreationClarification, validate_classification
    with pytest.raises(CreationClarification):
        validate_classification(source, classification("create", True, target_kind="new"))
    # A task-bearing route still carries every word unchanged, never a title
    # extraction or synthetic rewrite of the task by the classifier.
    command = validate_classification(source, classification("create_ask", True, target_kind="new"))
    assert command.inferred and command.text == source


@pytest.mark.parametrize("source", [
    "Set up an empty Codex conversation",
    "Start a Codex chat and leave it blank",
    "Make a blank Codex conversation",
    "Create a new Codex chat",
    "Please create a new Codex chat called Build.v2",
    'Create a Codex chat called "Research and Development"',
    'Create a Codex chat called "Ready to review the logs"',
    'Create a Codex chat called "Build Notes for a review"',
    "Could you open a new Codex chat called Build Notes",
    "I want a fresh Codex conversation",
    "Start a Codex chat and don't send any task yet",
    "Would you arrange an empty Codex conversation without sending a task?",
    "Could you arrange a blank Codex chat without submitting any prompt?",
    "Please set up an empty conversation without starting any work for now. Use GPT 6 Luna.",
    "Setup an empty Codex chat without sending a task.",
    "Make a blank Codex conversation without sending a task yet",
    "Set up a blank Codex conversation. Keep it empty for now",
    "Make a new Codex chat. Use GPT 6 Luna.",
    "Make a new Codex chat using GPT-6 Luna",
    "Make a new Codex chat and use GPT 6 Luna",
    "Start a Codex chat and leave it blank. Use GPT 6 Luna",
    "Make a new Codex chat. Use GPT6 Luna. Use GPT-6 Luna.",
])
def test_complete_standalone_or_explicit_blank_creations_preserve_full_source(source):
    from assistant.global_router import validate_classification
    command = validate_classification(source, classification("create", True, target_kind="new"))
    assert command.action == "create" and command.inferred and command.text == source
    assert command.title == "" and command.target_kind == "new"


@pytest.mark.parametrize("source", [
    "Make a new Codex chat. Don't use GPT 6 Luna",
    "Make a new Codex chat. Use GPT 6 Luna. Use GPT 5.6 Luna",
    "Make a new Codex chat. Use GPT 6 Luna to inspect the logs",
])
def test_empty_creation_model_clauses_are_complete_source_constraints_not_tasks(source):
    from assistant.global_router import CreationClarification, validate_classification
    with pytest.raises(CreationClarification):
        validate_classification(source, classification("create", True, target_kind="new"))


@pytest.mark.parametrize("source", [
    "He said create a Codex chat",
    '"Make a new Codex chat"',
    "Don't create a Codex chat",
])
def test_nonpresent_creation_source_cannot_authorize_codex(source):
    from assistant.global_router import validate_classification
    with pytest.raises(ValueError):
        validate_classification(source, classification("create", True, target_kind="new"))


def test_create_only_cannot_discard_task_even_with_explicit_codex_authority():
    from assistant.global_router import CreationClarification, validate_classification
    source = "Have Codex create a conversation only after inspecting the project"
    with pytest.raises(CreationClarification):
        validate_classification(source, classification("create", True, target_kind="new"))


def test_late_unquoted_cancellation_vetoes_full_transcript_and_classifier():
    from assistant.global_router import assess_codex_authority, explicit_codex_request, validate_classification
    source = "Use Codex to review " + ("the complete project history " * 120) + "Actually do not use Codex"
    assessment = assess_codex_authority(source)
    assert assessment.status == "revoked"
    assert assessment.source_spans and assessment.source_spans[-1][0] > 1500
    assert not explicit_codex_request(source)
    with pytest.raises(ValueError):
        validate_classification(source, classification("create_ask", True, target_kind="new"))

    router = LocalGlobalRouter("existing-qwen")
    router._opener = Opener(json.dumps({"message": {"content": json.dumps(
        classification("create_ask", True, target_kind="new"))}}).encode())
    assert router.route(source) is None
    assert router._opener.calls == []


@pytest.mark.parametrize("source", [
    'Use Codex to review the instruction "cancel that Codex request" as text',
    "Use Codex to review this example:\n```text\ncancel that Codex request\n```",
    "Use Codex to review this example: `cancel that Codex request`",
])
def test_cancellation_inside_balanced_quoted_or_code_data_is_ignored(source):
    from assistant.global_router import assess_codex_authority, explicit_codex_request
    assessment = assess_codex_authority(source)
    assert assessment.status == "allowed"
    assert explicit_codex_request(source)


def test_quoted_task_payload_remains_available_to_positive_head_grammar():
    from assistant.global_router import assess_codex_authority, validate_classification
    source = 'Use Codex "inspect my repo"'
    assert assess_codex_authority(source).status == "allowed"
    command = validate_classification(source, classification("ask", True, target_kind="selected"))
    assert command.text == source


@pytest.mark.parametrize("source", [
    'Use Codex to review this "unfinished cancellation example',
    "Use Codex to review this `unfinished cancellation example",
    "Use Codex to review this:\n```text\ncancel that Codex request",
])
def test_unbalanced_quote_or_code_boundary_fails_closed(source):
    from assistant.global_router import assess_codex_authority, explicit_codex_request, validate_classification
    assert assess_codex_authority(source).status == "ambiguous"
    assert not explicit_codex_request(source)
    with pytest.raises(ValueError):
        validate_classification(source, classification("ask", True, target_kind="selected"))


@pytest.mark.parametrize("source,expected", [
    ("I'd like a coding assistant to review my parser", "allowed"),
    ("I don't want Codex to inspect my project", "revoked"),
    ("I’d like Codex to explain this function", "allowed"),
    ("Use Codex to review the project; actually don’t use Codex", "revoked"),
])
def test_apostrophes_in_contractions_are_not_quote_boundaries(source, expected):
    from assistant.global_router import assess_codex_authority
    assert assess_codex_authority(source).status == expected


@pytest.mark.parametrize("source", [
    "Use Codex to review this; actually nevermind",
    "Use Codex to review this; scratch that",
    "Use Codex to review this; don't do that",
])
def test_generic_correction_with_unclear_scope_blocks_classifier(source):
    from assistant.global_router import assess_codex_authority, explicit_codex_request, validate_classification
    assert assess_codex_authority(source).status == "ambiguous"
    assert not explicit_codex_request(source)
    with pytest.raises(ValueError):
        validate_classification(source, classification("ask", True, target_kind="selected"))


def test_mixed_revoke_and_reinstate_is_ambiguous():
    from assistant.global_router import assess_codex_authority, validate_classification
    source = "Use Codex to review this; actually do not use Codex; actually use Codex to review it"
    assert assess_codex_authority(source).status == "ambiguous"
    with pytest.raises(ValueError):
        validate_classification(source, classification("ask", True, target_kind="selected"))


def test_canonical_cancel_control_remains_allowed():
    from assistant.global_router import assess_codex_authority, explicit_codex_request
    assert assess_codex_authority("Cancel Codex task.").status == "allowed"
    assert explicit_codex_request("Cancel Codex task.")


def test_named_navigation_cannot_hide_late_revoke_from_classifier():
    from assistant.global_router import assess_codex_authority, validate_classification
    source = "Open Codex chat Build Notes; actually don't open it"
    assert assess_codex_authority(source).status == "revoked"
    with pytest.raises(ValueError):
        validate_classification(source, classification("ask", True, target_kind="selected"))


def test_undelimited_task_data_cancellation_is_ambiguous_not_allowed_or_revoked():
    from assistant.global_router import assess_codex_authority, validate_classification
    source = "Use Codex to do analysis of text that says do not use Codex"
    assert assess_codex_authority(source).status == "ambiguous"
    with pytest.raises(ValueError):
        validate_classification(source, classification("ask", True, target_kind="selected"))


def test_explicit_blank_creation_allows_only_blank_creation():
    from assistant.global_router import assess_codex_authority, validate_classification
    source = "Create a blank Codex chat; do not send a prompt"
    assessment = assess_codex_authority(source)
    assert assessment.status == "allowed" and assessment.scope == "blank_creation"
    assert validate_classification(source, classification("create", True, target_kind="new")).action == "create"
    with pytest.raises(ValueError):
        validate_classification(source, classification("create_ask", True, target_kind="new"))
    task_suffix = source + "; inspect the logs"
    assert assess_codex_authority(task_suffix).status == "revoked"
    with pytest.raises(ValueError):
        validate_classification(task_suffix, classification("create_ask", True, target_kind="new"))


def test_router_refused_creation_retains_source_for_clarification_without_retry():
    source = "Create a Codex chat called Build Notes and check the failing tests"
    router = LocalGlobalRouter("existing-qwen")
    router._opener = Opener(json.dumps({"message": {"content": json.dumps(
        classification("create", True, target_kind="new"))}}).encode())
    command = router.route(source)
    assert command.action == "refused" and command.text == source
    assert len(router._opener.calls) == 1


@pytest.mark.parametrize("source,title", [
    ("Create a Codex chat called Build Notes and check tests", "Build Notes and check tests"),
    ("Please create a Codex chat called Build Notes for review", "Build Notes for review"),
    ("Could you open a Codex chat called Build Notes to inspect logs", "Build Notes to inspect logs"),
])
def test_legacy_creation_route_cannot_bypass_complete_source_authority(source, title):
    from assistant.global_router import CreationClarification
    with pytest.raises(CreationClarification):
        validate_route(source, {"action": "create", "text": "", "title": title})
