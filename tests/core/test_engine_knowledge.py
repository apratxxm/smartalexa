"""Knowledge synthesis in the engine answer path. Retrieval and the provider are faked."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from assistant.conversation_contracts import Decision, Transcript
from assistant.core.adapters.knowledge import Passage
from assistant.core.engine import CoreRuntime
from assistant.core.executor import Executor
from assistant.core.provider import ProviderError
from assistant.config import Config
from assistant.core.registry import Registry

ANSWER = "Paris is the capital of France."
PASSAGE = Passage("Paris", "https://en.wikipedia.org/wiki/Paris", "Paris is the capital of France.")


class Feedback:
    def __init__(self):
        self.spoken = []

    def tick(self):
        pass

    def say(self, text, **kwargs):
        self.spoken.append(text)


class Provider:
    timeout = 18

    def __init__(self, decision, synthesis=None, synth_error=None):
        self.decision, self.synthesis, self.synth_error = decision, synthesis, synth_error
        self.synth_calls = []

    def decide(self, transcript, context=None, **kwargs):
        return self.decision

    def synthesize(self, question, passages, **kwargs):
        self.synth_calls.append((question, [p.title for p in passages], kwargs.get("turn_id")))
        if self.synth_error:
            raise self.synth_error
        return self.synthesis


class Source:
    def __init__(self, passages=(PASSAGE,), error=None):
        self.passages, self.error, self.calls = list(passages), error, 0

    def retrieve(self, question, guard):
        self.calls += 1
        if self.error:
            raise self.error
        return list(self.passages)


def answer(text=ANSWER):
    return Decision("answer", text=text, evidence_quotes=("capital",))


def run(decision, provider_kwargs=None, source=None, knowledge=True, text="what is the capital of France"):
    provider = Provider(decision, **(provider_kwargs or {"synthesis": "Paris is the capital of France."}))
    toasts, feedback = [], Feedback()
    runtime = CoreRuntime(provider, Executor(), feedback, toast=toasts.append,
                          registry=Registry(Config({"core": {"enabled_phases": ["1", "4"]}})),
                          knowledge=knowledge, knowledge_source=source or Source())
    row = runtime.process(Transcript(text, "t1"))
    return row, feedback.spoken, toasts, provider, runtime


def test_knowledge_is_off_by_default_and_retrieval_is_never_called():
    source = Source()
    row, spoken, toasts, provider, _ = run(answer(), knowledge=False, source=source)
    assert source.calls == 0 and provider.synth_calls == []
    assert spoken == [ANSWER]
    assert "knowledge_source" not in row


def test_default_runtime_does_not_enable_knowledge():
    runtime = CoreRuntime(Provider(answer()), Executor(), Feedback())
    assert runtime.knowledge is False


def test_passages_are_synthesised_once_and_the_source_is_named_in_the_toast():
    source = Source()
    row, spoken, toasts, provider, _ = run(answer(), source=source)
    assert source.calls == 1
    assert provider.synth_calls == [("what is the capital of France", ["Paris"], "t1")]
    assert spoken == ["Paris is the capital of France."]
    assert toasts and toasts[0].endswith("· Source: Paris")
    assert row["knowledge_source"] == "Paris"


def test_synthesis_is_bounded_to_one_sentence():
    row, spoken, _, _, _ = run(answer(), {"synthesis": "Paris is the capital. It is large and old."})
    assert spoken == ["Paris is the capital."]


def test_no_passages_speaks_the_model_answer_without_synthesis():
    row, spoken, toasts, provider, _ = run(answer(), source=Source(passages=()))
    assert provider.synth_calls == [] and spoken == [ANSWER]
    assert toasts and "Source:" not in toasts[0]


def test_synthesis_error_speaks_the_original_answer():
    row, spoken, toasts, _, _ = run(answer(), {"synth_error": ProviderError("unavailable", "deadline")})
    assert spoken == [ANSWER]
    assert toasts and "Source:" not in toasts[0]
    assert any(item["stage"] == "synthesis" for item in row["errors"])


def test_retrieval_error_speaks_the_original_answer():
    row, spoken, _, provider, _ = run(answer(), source=Source(error=OSError("offline")))
    assert spoken == [ANSWER] and provider.synth_calls == []
    assert any(item["stage"] == "knowledge" for item in row["errors"])


def test_effect_claim_in_synthesis_is_not_spoken_and_original_answer_stands():
    row, spoken, _, _, _ = run(answer(), {"synthesis": "I played Paris for you."})
    assert spoken == [ANSWER]


def test_original_answer_that_claims_an_effect_is_still_refused_before_retrieval():
    source = Source()
    row, spoken, _, provider, _ = run(answer("I paused the music."), source=source)
    assert source.calls == 0 and provider.synth_calls == []
    assert spoken == ["I didn't do anything."]


@pytest.mark.parametrize("decision", [
    Decision("plan", steps=(), evidence_quotes=("capital",)),
    Decision("clarify", text="Which country?", evidence_quotes=("capital",)),
    Decision("unsupported", reason="Planning unavailable."),
])
def test_only_the_answer_kind_is_synthesised(decision):
    source = Source()
    run(decision, source=source)
    assert source.calls == 0
