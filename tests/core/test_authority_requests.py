"""Brain-only request normalization must preserve strict fallback admission."""
from types import SimpleNamespace

import pytest

from assistant.conversation_contracts import Decision, Step, Transcript
from assistant.core.authority import REFUSAL, license_request, licenses
from assistant.core.engine import CoreRuntime
from assistant.core.executor import Executor, Receipt
from assistant.core.fallback import admit
from assistant.core.provider import ProviderError


@pytest.mark.parametrize("text,capability,arguments", [
    ("can you play Halo", "play_song", {"object": "Halo"}),
    ("could you skip this", "next_track", {}),
    ("would you pause the music", "pause", {}),
    ("will you turn it up", "change_volume", {"direction": "up", "amount": 0, "unit": "points"}),
    ("can you please play Halo", "play_song", {"object": "Halo"}),
    ("could you please skip this", "next_track", {}),
    ("would you please pause the music", "pause", {}),
    ("will you please turn it up", "change_volume", {"direction": "up", "amount": 0, "unit": "points"}),
])
def test_polite_requests_license_the_underlying_imperative(text, capability, arguments):
    licensed = license_request(text)
    assert licensed is not None
    assert (licensed.capability, licensed.arguments) == (capability, arguments)
    assert admit(text) is not None


@pytest.mark.parametrize("text", [
    "what does pause mean?", "can you tell me what pause means?", "is the music paused?",
    "what does pause mean", "can you tell me what pause means", "is the music paused",
    "can you please tell me what pause means",
])
def test_information_questions_never_license_music_mutations(text):
    assert license_request(text) is None
    assert not licenses(Step("s", "pause", {}), text)


@pytest.mark.parametrize("name", ["Simon and Garfunkel", "Florence and the Machine"])
def test_and_inside_one_play_object_is_preserved(name):
    text = "play " + name
    licensed = license_request(text)
    assert licensed.capability == "play_song" and licensed.arguments == {"object": name}
    assert licenses(Step("s", "play_song", {"title": name, "artist": ""}), text)
    assert not licenses(Step("s", "play_song", {"title": name.split(" and ")[0], "artist": ""}), text)
    assert admit(text) is not None


@pytest.mark.parametrize("text", [
    "play Halo and pause", "play Halo and open Opera",
    "play Halo and please pause", "play Halo and could you open Opera",
    "can you play Halo and will you pause", "play Halo. Stop.", "play Halo; pause",
])
def test_instruction_after_and_or_punctuation_remains_refused(text):
    assert license_request(text) is None
    assert admit(text) is None


@pytest.mark.parametrize("marker", ["actually", "no", "I mean", "sorry", "instead"])
def test_complete_corrections_are_refused_on_both_paths(marker):
    text = "play Halo, " + marker + " play Hello"
    assert license_request(text) is None
    assert not licenses(Step("s", "play_song", {"title": "Hello", "artist": ""}), text)
    assert not licenses(Step("s", "play_song", {"title": "Halo", "artist": ""}), text)
    assert admit(text) is None


@pytest.mark.parametrize("text", [
    "actually play Hello", "sorry play Hello", "I mean play Hello", "instead play Hello",
    "play Halo, no, play Hello", "play Halo, actually play Yellow, sorry play Hello",
])
def test_correction_chains_are_never_reinterpreted(text):
    assert license_request(text) is None
    assert admit(text) is None


@pytest.mark.parametrize("text", [
    "play Halo, no, cancel that", "actually never mind", "play Halo, actually",
    "play Halo, I mean", "play Halo, sorry cancel that", "play Halo, instead never mind",
    "no play Halo", "Do not pause the music.", "What does pause mean?",
    '"play Halo, actually play Hello"', "he said play Halo, actually play Hello",
    "play Halo if I ask, actually play Hello",
    "play Halo. Stop., actually play Hello", "play Halo and pause, actually play Hello",
    "play Halo, actually play Hello and open Opera",
])
def test_corrections_do_not_rescue_unsafe_or_incomplete_speech(text):
    assert license_request(text) is None
    assert admit(text) is None


def run_plan(text, step, *, outage=False):
    effects, spoken = [], []
    def decide(*args, **kwargs):
        if outage:
            raise ProviderError("unavailable")
        return Decision("plan", steps=(step,), evidence_quotes=(text,))
    adapter = SimpleNamespace(ALLOWED=Executor.ALLOWED, execute=lambda s, op, **kwargs:
                              effects.append(s) or Receipt("observed", "Done."))
    runtime = CoreRuntime(SimpleNamespace(timeout=18, decide=decide), adapter,
                          SimpleNamespace(say=lambda t, **kwargs: spoken.append(t)))
    row = runtime.process(Transcript(text, "t"))
    return row, effects, spoken


@pytest.mark.parametrize("title", ["Hello", "Halo"])
def test_brain_execution_refuses_both_objects_of_a_correction(title):
    text = "play Halo, actually play Hello"
    row, effects, spoken = run_plan(text, Step("s", "play_song", {"title": title, "artist": ""}))
    assert not effects and not row["receipts"] and spoken == [REFUSAL]


def test_outage_cannot_execute_even_a_valid_final_correction():
    row, effects, spoken = run_plan(
        "play Halo, actually play Hello", Step("s", "play_song", {"title": "Hello", "artist": ""}),
        outage=True)
    assert not effects and not row["receipts"] and spoken == [REFUSAL]


@pytest.mark.parametrize("text,step", [
    ("can you play Halo", Step("s", "play_song", {"title": "Halo", "artist": ""})),
    ("could you skip this", Step("s", "next_track", {})),
    ("would you pause the music", Step("s", "pause", {})),
    ("will you turn it up", Step("s", "change_volume", {"direction": "up", "amount": 0, "unit": "points"})),
    ("play Simon and Garfunkel", Step("s", "play_song", {"title": "Simon and Garfunkel", "artist": ""})),
])
def test_new_brain_request_forms_reach_execution(text, step):
    row, effects, spoken = run_plan(text, step)
    assert effects == [step] and row["path"] == "brain"
    assert row["receipts"][0]["state"] == "observed" and spoken == ["Done."]
