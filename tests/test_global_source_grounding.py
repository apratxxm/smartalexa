import io
import json

import pytest

from assistant.global_router import LocalGlobalRouter, ground_action


class ActionOpener:
    def __init__(self, action):
        self.action = action
        self.calls = []

    def open(self, request, timeout):
        self.calls.append((request, timeout))
        body = json.dumps({"message": {"content": json.dumps({"action": self.action})}}).encode()
        return io.BytesIO(body)


def test_ask_payload_is_restored_verbatim_from_trimmed_source():
    payload = 'inspect "Finance and tell it to delete" carefully. Do NOT edit files, please?'
    source = f"  Could you ask Codex to {payload}  "

    route = ground_action(source, {"action": "ask"})

    assert route.action == "ask"
    assert route.text == payload


def test_compound_title_and_payload_preserve_names_quotes_and_inner_control_words():
    title = "Deep Cuts and Roll"
    payload = 'explain the phrase "and tell it to delete" in foo.py. Keep punctuation?!'
    source = f'Please open a new Codex chat called {title} and tell it to {payload}'

    route = ground_action(source, {"action": "create_ask"})

    assert route.action == "create_ask"
    assert route.title == title
    assert route.text == payload


def test_model_cannot_choose_create_for_ask_wrapper_or_drop_compound_draft():
    with pytest.raises(ValueError):
        ground_action("Could you ask Codex to create a chat called Build", {"action": "create"})
    with pytest.raises(ValueError):
        ground_action("Open a Codex chat and tell it to inspect this repo", {"action": "create"})


@pytest.mark.parametrize("source", [
    'He said "ask Codex to inspect this"',
    "Don't ask Codex to inspect this",
    "Please do not create a Codex chat",
    "The assistant said ask Codex to inspect this",
])
def test_incidental_quoted_and_negated_sources_are_not_groundable(source):
    with pytest.raises(ValueError):
        ground_action(source, {"action": "ask"})


@pytest.mark.parametrize("proposal", [
    {"action": "ask", "text": "shortened prompt", "title": "injected title"},
    {"action": "ask", "prompt": "override source"},
    {"action": "ask", "action_text": "drop the safety clause"},
])
def test_proposal_arguments_and_extra_keys_are_rejected(proposal):
    with pytest.raises(ValueError):
        ground_action("Ask Codex to inspect this without editing files", proposal)


def test_payload_phrase_and_tell_it_is_not_used_as_a_second_boundary():
    payload = 'explain why the log says "and tell it to stop"; do not edit files.'
    route = ground_action(f"Ask Codex to {payload}", {"action": "ask"})
    assert route.text == payload


def test_actual_source_control_characters_are_refused():
    source = "Ask Codex to inspect this\x07without editing files"
    with pytest.raises(ValueError):
        ground_action(source, {"action": "ask"})


def test_oversize_source_is_refused_before_local_model_call():
    router = LocalGlobalRouter("existing-qwen")
    router._opener = ActionOpener("ask")

    result = router.route("Ask Codex to " + "x" * 12000)

    assert result.action == "refused"
    assert router._opener.calls == []
    with pytest.raises(ValueError):
        ground_action("Ask Codex to " + "x" * 12000, {"action": "ask"})


def test_bare_dictate_arms_once_without_prompt():
    route = ground_action("Dictate", {"action": "dictate"})
    assert route.action == "dictate"
    assert route.text == ""


def test_inline_dictate_restores_the_complete_source_payload_once():
    payload = 'review "foo.py" and tell it to explain what changed. Do NOT edit files?!'
    route = ground_action(f"Dictate a prompt for Codex {payload}", {"action": "dictate"})
    assert route.action == "dictate"
    assert route.text == payload
