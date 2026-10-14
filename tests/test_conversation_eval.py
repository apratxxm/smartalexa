"""Offline semantic harness tests; no inference or capability execution."""
from __future__ import annotations

from assistant.conversation_contracts import Decision, Step
from assistant.experimental.conversation_eval import (
    CASES, HELD_OUT_CASES, EvalCase, build_plain_interpretation_messages,
    check_decision, evaluate,
)


class FakeDecider:
    def __init__(self, decisions):
        self.decisions = iter(decisions)
        self.calls = []
        self.timeout = 3.0
        self.timeouts_seen = []

    def decide(self, transcript, context=None):
        self.calls.append((transcript, context))
        self.timeouts_seen.append(self.timeout)
        return next(self.decisions)


def test_held_out_cases_are_synthetic_separate_and_within_case_cap():
    assert len(CASES) + len(HELD_OUT_CASES) <= 39
    assert "definition_virtuoso" in {case.case_id for case in CASES}
    assert "definition_virtuoso" not in {case.case_id for case in HELD_OUT_CASES}
    held = {case.case_id: case for case in HELD_OUT_CASES}
    assert len(held) == 8
    assert held["heldout_unseen_song"].arguments == (
        (0, "title", "Glasshouse"), (0, "artist", "Paper Tigers"))
    volume = held["heldout_volume_then_pause"]
    assert volume.capabilities == ("set_volume", "pause")
    good = Decision("plan", steps=(Step("a", "set_volume", {"level": 40}),
                                    Step("b", "pause", {}, ("a",))))
    bad = Decision("plan", steps=(Step("a", "set_volume", {"level": 40}),
                                   Step("b", "pause", {}, ())))
    assert check_decision(volume, good) is None
    assert check_decision(volume, bad) == "dependency_mismatch"


def test_inertia_lexical_screen_accepts_valid_paraphrases_but_not_unrelated_text():
    case = next(item for item in HELD_OUT_CASES
                if item.case_id == "heldout_negative_codex_definition")
    assert check_decision(case, Decision(
        "answer", text="Inertia is the tendency to remain at rest or maintain constant velocity.")) is None
    assert check_decision(case, Decision(
        "answer", text="Inertia resists changes in motion.")) is None
    assert check_decision(case, Decision(
        "answer", text="A scientific idea about gravity and energy.")) == "answer_not_useful"


def test_plain_interpretation_messages_are_data_only_and_preserve_pending_context():
    import inspect

    source = "Never mind that. Open Opera."
    prior = "Play Red by Seedhe Maut"
    case = EvalCase("plain", source, ("plan",), context={
        "pending_request": {"question": "Which track?", "sources": [prior]},
        "ignored_extra": "not included",
    }, grounding_sources=(prior,))
    messages = build_plain_interpretation_messages(case)
    assert [message["role"] for message in messages] == ["system", "user"]
    assert set(messages[1]) == {"role", "content"}
    user = __import__("json").loads(messages[1]["content"])
    assert user == {"transcript": source, "application_context_data": {
        "pending_request": {"question": "Which track?", "sources": [prior]}}}
    joined = " ".join(message["content"].casefold() for message in messages)
    assert "json schema" not in joined
    assert "tool registry" not in joined
    assert "execute anything" in joined
    assert source not in messages[0]["content"]
    helper_source = inspect.getsource(build_plain_interpretation_messages)
    assert "ConversationBrain" not in helper_source
    assert "_transport" not in helper_source
    assert ".decide(" not in helper_source


def test_plain_messages_accept_only_an_eval_case_without_invoking_a_decider():
    import pytest

    with pytest.raises(TypeError, match="EvalCase"):
        build_plain_interpretation_messages(object())


def test_semantic_checks_use_decider_only_and_report_no_source_text():
    case = EvalCase(
        "synthetic", "Pause and define virtuoso.", ("plan",), ("pause",),
        answer_groups=(("skilled", "accomplished", "expert", "exceptional", "master"),
                       ("performer", "musician", "artist", "music", "art", "craft", "field")),
    )
    decider = FakeDecider([Decision(
        "plan", steps=(Step("s1", "pause", {}, ()),),
        answer="A highly skilled musical performer.", answer_basis="general_knowledge",
    )])
    output = evaluate(decider, cases=(case,))
    assert output == [{
        "case_id": "synthetic", "passed": True, "failure_category": None,
        "kind": "plan", "capabilities": ["pause"], "latency_ms": output[0]["latency_ms"],
    }]
    assert len(decider.calls) == 1
    assert "Pause and define" not in str(output)


def test_pending_source_is_passed_to_decider_and_full_codex_text_is_checked():
    source = "Please play Madira by See the Moth"
    case = EvalCase("pending", "Play that.", ("plan",), ("play_song",),
                    ((0, "artist", "See the Moth"),),
                    context={"pending_request": {"question": "Which song?", "sources": [source]}},
                    grounding_sources=(source,))
    decision = Decision("plan", steps=(Step("s1", "play_song", {
        "title": "Madira", "artist": "See the Moth",
    }, ()),))
    decider = FakeDecider([decision])
    assert evaluate(decider, cases=(case,))[0]["passed"]
    assert decider.calls[0][0].grounding_sources == (source,)
    assert decider.calls[0][1] == case.context


def test_case_limits_and_total_deadline_are_bounded():
    import pytest

    with pytest.raises(ValueError):
        evaluate(FakeDecider([]), max_cases=41)
    with pytest.raises(ValueError):
        evaluate(FakeDecider([]), total_deadline=120.01)
    with pytest.raises(ValueError):
        evaluate(FakeDecider([]), per_case_deadline=3.01)

    now = iter([0.0, 1.0])
    cases = (EvalCase("one", "One", ("answer",)), EvalCase("two", "Two", ("answer",)))
    output = evaluate(FakeDecider([]), cases=cases, total_deadline=0.5,
                      clock=lambda: next(now))
    assert output[0]["failure_category"] == "total_deadline"
    assert output[0]["case_id"] == "one"


def test_report_data_contains_only_ids_categories_and_decision_shape():
    case = EvalCase("weather", "What is the weather now?", ("unsupported",))
    decider = FakeDecider([Decision("unsupported", reason="No live source.")])
    result = evaluate(decider, cases=(case,))[0]
    assert set(result) == {"case_id", "passed", "failure_category", "kind", "capabilities", "latency_ms"}
    assert "reason" not in result
    assert "weather now" not in str(result)


def test_virtuoso_answer_checker_accepts_synonyms_but_rejects_unrelated_text():
    from assistant.experimental.conversation_eval import CASES, check_decision

    case = next(item for item in CASES if item.case_id == "definition_virtuoso")
    assert check_decision(case, Decision("answer", text="An accomplished musician with exceptional technique.")) is None
    assert check_decision(case, Decision("answer", text="An expert in an art or craft.")) is None
    assert check_decision(case, Decision("answer", text="A knowledgeable person.")) == "answer_not_useful"
    assert check_decision(case, Decision("answer", text="A stringed instrument with six sides.")) == "answer_not_useful"


def test_case_deadline_caps_long_brain_timeout_and_restores_it_after_success():
    case = EvalCase("bounded", "Define inertia.", ("answer",))
    decision = Decision("answer", text="Inertia is resistance to change in motion.")
    decider = FakeDecider([decision])
    decider.timeout = 30.0
    result = evaluate(decider, cases=(case,))[0]
    assert decider.timeouts_seen == [3.0]
    assert decider.timeout == 30.0
    assert result["passed"] is True


def test_case_wall_clock_overrun_fails_even_when_decider_returns_valid_answer():
    now = [0.0]

    class SlowDecider(FakeDecider):
        def decide(self, transcript, context=None):
            super().decide(transcript, context)
            now[0] += 3.1
            return Decision("answer", text="Inertia resists change in motion.")

    decider = SlowDecider([None])
    decider.timeout = 30.0
    result = evaluate(decider, cases=(EvalCase("late", "Define inertia.", ("answer",)),),
                      clock=lambda: now[0])[0]
    assert result["passed"] is False
    assert result["failure_category"] == "deadline_exceeded"
    assert decider.timeouts_seen == [3.0]
    assert decider.timeout == 30.0


def test_total_remaining_deadline_shrinks_later_case_timeout():
    now = [0.0]

    class AdvancingDecider(FakeDecider):
        def decide(self, transcript, context=None):
            result = super().decide(transcript, context)
            if len(self.calls) == 1:
                now[0] += 2.0
            return result

    cases = (EvalCase("first", "Define inertia.", ("answer",)),
             EvalCase("second", "Define inertia.", ("answer",)))
    answer = Decision("answer", text="Inertia is resistance to change in motion.")
    decider = AdvancingDecider([answer, answer])
    decider.timeout = 30.0
    results = evaluate(decider, cases=cases, total_deadline=4.0, clock=lambda: now[0])
    assert len(results) == 2 and all(item["passed"] for item in results)
    assert decider.timeouts_seen == [3.0, 2.0]
    assert decider.timeout == 30.0


def test_original_timeout_is_restored_after_interpreter_exception():
    from assistant.conversation_brain import InterpreterUnavailable
    import pytest

    class UnavailableDecider(FakeDecider):
        def decide(self, transcript, context=None):
            super().decide(transcript, context)
            raise InterpreterUnavailable("not ready")

    decider = UnavailableDecider([None])
    decider.timeout = 30.0
    result = evaluate(decider, cases=(EvalCase("unavailable", "Pause.", ("plan",)),))[0]
    assert result["failure_category"] == "interpreter_unavailable"
    assert decider.timeouts_seen == [3.0]
    assert decider.timeout == 30.0


def test_cli_held_out_flag_appends_cases_without_changing_default(monkeypatch, capsys):
    import assistant.conversation_brain as conversation_brain
    import assistant.experimental.conversation_eval as conversation_eval

    captured = []

    class ConstructionOnly:
        def __init__(self, model, timeout):
            captured.append(("brain", model, timeout))

    def fake_evaluate(brain, **kwargs):
        captured.append(kwargs["cases"])
        return []

    monkeypatch.setattr(conversation_brain, "ConversationBrain", ConstructionOnly)
    monkeypatch.setattr(conversation_eval, "evaluate", fake_evaluate)
    assert conversation_eval.main(["--model", "local-model"]) == 0
    assert captured[-1] == CASES
    assert conversation_eval.main(["--model", "local-model", "--held-out"]) == 0
    assert captured[-1] == CASES + HELD_OUT_CASES
    capsys.readouterr()
