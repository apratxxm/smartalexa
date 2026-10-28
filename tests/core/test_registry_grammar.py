from types import SimpleNamespace

import pytest

from assistant.core.authority import c1_allows, find_clause


def grammar():
    play = SimpleNamespace(object_kind="text")
    pause = SimpleNamespace(object_kind="none")
    return SimpleNamespace(verbs={"play": [("play_song", play)], "pause": [("pause", pause)]})


@pytest.mark.parametrize("text,obj,suspicious", [
    ("can you play Halo", "Halo", False),
    ("play Simon and Garfunkel", "Simon and Garfunkel", False),
    ("play Don't Stop Believin'", "Don't Stop Believin'", True),
    ("play Halo if I ask", "Halo if I ask", True),
    ("play When I Was Your Man", "When I Was Your Man", True),
])
def test_whole_utterance_objects(text, obj, suspicious):
    clause = find_clause(text, grammar())
    assert clause is not None and clause.object == obj and clause.suspicious is suspicious


@pytest.mark.parametrize("text", ["play Halo and pause", '"play Halo"', "he said play Halo",
                                  "play Halo; pause", "play Halo. Pause.", "pause anything"])
def test_no_partial_or_compound_commands(text):
    assert find_clause(text, grammar()) is None


def test_suspicious_requires_exact_match_and_fails_closed():
    clause = find_clause("play Halo if I ask", grammar())
    assert not c1_allows(SimpleNamespace(mutating=True), clause, {}, lambda: 5)
    assert not c1_allows(SimpleNamespace(mutating=True, exact_match=lambda *_: False), clause, {}, lambda: 5)
    assert c1_allows(SimpleNamespace(mutating=True, exact_match=lambda *_: True), clause, {}, lambda: 5)


def test_with_in_title_requires_and_accepts_exact_match():
    clause = find_clause("play Dancing with a Stranger", grammar())
    assert clause is not None and clause.suspicious
    assert not c1_allows(SimpleNamespace(mutating=True), clause, {}, lambda: 5)
    assert not c1_allows(SimpleNamespace(mutating=True, exact_match=lambda *_: False), clause, {}, lambda: 5)
    assert c1_allows(SimpleNamespace(mutating=True, exact_match=lambda *_: True), clause, {}, lambda: 5)


def test_with_in_title_admits_through_real_registry_only_on_exact_match(monkeypatch):
    from assistant.conversation_contracts import Step
    from assistant.core import authority

    registry = authority._registry_default()
    step = Step("s", "play_song", {"title": "Dancing with a Stranger", "artist": ""})
    text = "play Dancing with a Stranger"
    monkeypatch.setattr(authority.PhaseOneHandler, "exact_match", lambda self, args, guard: False)
    assert not authority.admit_step(step, text, registry, lambda: 5)
    monkeypatch.setattr(authority.PhaseOneHandler, "exact_match", lambda self, args, guard: True)
    assert authority.admit_step(step, text, registry, lambda: 5)
