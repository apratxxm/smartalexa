from types import SimpleNamespace
from urllib.error import HTTPError

import pytest

from assistant.config import Config
from assistant.conversation_contracts import Decision, Step, Transcript
from assistant.core.capture import CaptureResult, DeviceFailure, EndReason
from assistant.core.engine import CoreRuntime, _TIMINGS
from assistant.core.executor import Executor, Receipt
from assistant.core.provider import ProviderError
from assistant.core.volume import SystemVolumeVerificationError
from assistant.runtime import single_instance, AlreadyRunning
from assistant.supervise import resolve_runtime_module
from assistant.conversation import ConversationCoordinator
from assistant.core.executor import coordinator_result


class Provider:
    timeout = 18
    def __init__(self, decision=None, error=None):
        self.decision, self.error, self.calls = decision, error, 0
    def decide(self, transcript, context=None, **kwargs):
        self.calls += 1
        if self.error:
            raise self.error
        return self.decision


class Feedback:
    def __init__(self):
        self.spoken = []
    def tick(self):
        pass
    def say(self, text, **kwargs):
        self.spoken.append(text)


def plan(text="play Halo"):
    return Decision("plan", steps=(Step("s1", "play_song", {"title": "Halo", "artist": ""}),),
                    evidence_quotes=(text,))


def runtime(provider, executor=None, **kwargs):
    return CoreRuntime(provider, executor or Executor(), Feedback(), **kwargs)


def music(play):
    return SimpleNamespace(resolve_track=lambda query: query, play_candidate=play)


def test_planning_429_falls_back_through_same_executor():
    effects = []
    provider = Provider(error=ProviderError("rate_limited", "http_429"))
    engine = runtime(provider, Executor(spotify=music(
        lambda query: effects.append(query) or "Playing Halo by Beyoncé.")))
    row = engine.process(Transcript("play Halo", "t"))
    assert row["path"] == "fallback" and provider.calls == 1
    assert len(effects) == 1 and effects[0].casefold() == "halo"
    assert row["receipts"][0]["state"] == "observed"
    assert engine.feedback.spoken == ["Playing Halo by Beyoncé."]


@pytest.mark.parametrize("bad", [ProviderError("invalid", "bad_plan"), "invalid_typed_decision"])
def test_invalid_plan_never_uses_fallback(bad):
    provider = Provider(error=bad if isinstance(bad, Exception) else None,
                        decision=Decision("plan", steps=(Step("s", "play_song", {"query": "Halo"}),),
                                          evidence_quotes=("play Halo",)))
    effects = []
    engine = runtime(provider, Executor(spotify=music(effects.append)))
    row = engine.process(Transcript("play Halo", "t"))
    assert row["path"] != "fallback" and not effects and not row["receipts"]
    assert provider.calls == 1


def test_post_effect_429_reports_unknown_without_repeating_or_fallback():
    effects = []
    def play(query):
        effects.append(query)
        raise HTTPError("https://example.invalid", 429, "rate limited", {}, None)
    provider = Provider(plan())
    engine = runtime(provider, Executor(spotify=music(play)))
    row = engine.process(Transcript("play Halo", "t"))
    assert effects == ["Halo"] and provider.calls == 1 and row["path"] == "brain"
    assert row["receipts"][0]["state"] == "unknown"
    assert engine.feedback.spoken == ["I asked Spotify but can't confirm it played."]


def test_post_effect_synthesis_failure_preserves_receipts_and_reports_locally():
    effects = []
    def synthesize(_):
        raise ProviderError("rate_limited", "http_429")
    provider = Provider(plan())
    engine = runtime(provider, Executor(spotify=music(
        lambda query: effects.append(query) or "Playing Halo.")), reply_builder=synthesize)
    row = engine.process(Transcript("play Halo", "t"))
    assert provider.calls == 1 and effects == ["Halo"]
    assert row["receipts"][0]["state"] == "observed"
    assert engine.feedback.spoken == ["Playing Halo."]
    assert row["errors"] == [{"stage": "reply", "type": "ProviderError",
                              "category": "rate_limited", "reason": "http_429"}]


@pytest.mark.parametrize("state,count", [("observed", 2), ("requested", 1), ("unknown", 1)])
def test_only_observed_receipts_advance_dependent_steps(state, count):
    decision = Decision("plan", steps=(Step("a", "pause", {}),
                                       Step("b", "resume", {}, ("a",))),
                        evidence_quotes=("pause then resume",))
    calls = []
    execute = lambda step, op: coordinator_result(op,
        calls.append(step.capability) or Receipt(state, "Paused."))
    coordinator = ConversationCoordinator(Provider(decision), execute, execute)
    result = coordinator.handle(Transcript("pause then resume", "t"))
    assert calls == ["pause", "resume"][:count]
    assert len(result.results) == count


def test_real_single_instance_refuses_second_holder_and_releases(tmp_path):
    with single_instance(tmp_path):
        with pytest.raises(AlreadyRunning):
            with single_instance(tmp_path):
                pytest.fail("duplicate lock admitted")
    with single_instance(tmp_path):
        pass


@pytest.mark.parametrize("name,module", [("legacy", "assistant.main"), ("core", "assistant.core")])
def test_supervisor_resolves_selected_runtime(name, module):
    assert resolve_runtime_module(Config({"runtime": {"runtime": name}})) == module


def test_volume_write_verification_failure_is_unknown():
    volume = SimpleNamespace(set_system_volume=lambda _, **kwargs: (_ for _ in ()).throw(
        SystemVolumeVerificationError("write was attempted")))
    result = Executor(volume=volume).execute(Step("v", "set_volume", {"level": 40}))
    assert result.state == "unknown"


@pytest.mark.parametrize("error,state", [(LookupError("not found"), "failed"),
                                        (KeyError("verification after sending"), "unknown"),
                                        (RuntimeError("after sending"), "unknown")])
def test_playback_resolution_vs_uncertain_delivery(error, state):
    fail = lambda _: (_ for _ in ()).throw(error)
    spotify = music(fail)
    if type(error) is LookupError:
        spotify.resolve_track = fail
    assert Executor(spotify=spotify).execute(plan().steps[0]).state == state


@pytest.mark.parametrize("kind,text,expected", [
    ("answer", "A virtuoso is an expert musician. Another sentence.", "A virtuoso is an expert musician."),
    ("clarify", "Which song?", "Which song?"),
    ("no_request", "", None),
    ("unsupported", "", "I can't do that yet."),
])
def test_every_decision_kind_has_explicit_feedback(kind, text, expected):
    decision = Decision(kind, text=text, evidence_quotes=("What does virtuoso mean?",),
                        reason="Unavailable.")
    toasts = []
    engine = runtime(Provider(decision), toast=toasts.append)
    row = engine.process(Transcript("What does virtuoso mean?", "t"))
    assert engine.feedback.spoken == ([] if expected is None else [expected])
    assert len(toasts) == (0 if expected is None else 1)
    assert engine.coordinator.pending_clarification is None
    assert set(_TIMINGS) <= row["timings"].keys()


def test_forbidden_capability_rejects_whole_plan_before_any_effect():
    calls = []
    decision = Decision("plan", steps=(Step("a", "pause", {}),
                                      Step("b", "open_app", {"name": "Opera"}, ("a",))),
                        evidence_quotes=("pause and open Opera",))
    engine = runtime(Provider(decision), Executor(spotify=SimpleNamespace(pause=lambda: calls.append(1))))
    row = engine.process(Transcript("pause and open Opera", "t"))
    assert not calls and not row["receipts"]
    assert engine.feedback.spoken == ["I can't do that yet."]


def test_toast_failure_does_not_block_speech_or_next_turn():
    def toast(_):
        raise OSError("desktop unavailable")
    engine = runtime(Provider(Decision("answer", text="An expert musician.",
                                       evidence_quotes=("What is a virtuoso?",))), toast=toast)
    for turn in ("one", "two"):
        row = engine.process(Transcript("What is a virtuoso?", turn))
        assert row["errors"][0]["stage"] == "toast"
    assert engine.feedback.spoken == ["An expert musician.", "An expert musician."]


def test_stt_exception_is_logged_and_next_turn_continues():
    rows = []
    engine = runtime(Provider(plan()), log=rows.append)
    engine.executor = Executor(spotify=music(lambda _: "Playing Halo."))
    class Capture:
        turn_id, wake_score, turns = "", 0.8, 0
        def next_capture(self, acknowledge=None):
            self.turns += 1
            self.turn_id = str(self.turns)
            now = engine.clock()
            return CaptureResult(self.turn_id, [], 1000, EndReason.ENDPOINT, True,
                                 False, self.wake_score, now, now,
                                 {"wake_to_ack_ms": 0, "speech_end": 0})
        def transcribe(self, result):
            if self.turns == 1:
                raise ValueError("bad ASR result")
            return "play Halo", 0
        def reset(self):
            pass
    capture = Capture()
    engine.step(capture)
    engine.step(capture)
    assert len(rows) == 2 and rows[0]["errors"][0]["stage"] == "stt"
    assert rows[1]["receipts"][0]["state"] == "observed"
    assert all(set(_TIMINGS) <= row["timings"].keys() for row in rows)


def test_microphone_failure_is_logged_and_propagated_for_supervisor_restart():
    rows = []
    engine = runtime(Provider(plan()), log=rows.append)
    def fail(**kwargs):
        raise DeviceFailure("mic gone")
    capture = SimpleNamespace(next_capture=fail, reset=lambda: None, turn_id="")
    with pytest.raises(DeviceFailure):
        engine.step(capture)
    assert rows[0]["outcome"] == "device_error"
    assert rows[0]["errors"][0]["stage"] == "capture"
