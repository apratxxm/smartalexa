from types import SimpleNamespace
import threading

from assistant.conversation import ConversationCoordinator
from assistant.conversation_contracts import Decision, Step, ToolResult, Transcript, step_to_intent
from assistant.intent import Action


class FakeInterpreter:
    def __init__(self, decision=None, error=None):
        self.decision = decision
        self.error = error
        self.calls = []

    def decide(self, transcript, context=None):
        self.calls.append((transcript, context))
        if self.error:
            raise self.error
        return self.decision


def _step(name, args=None, *, after=(), step_id="s1"):
    return Step(step_id, name, args or {}, tuple(after))


def _reply_capture_assistant(decisions, captures, replies):
    from contextlib import nullcontext
    from assistant.feedback import KIND_SPEECH
    from assistant.main import Assistant

    class Interpreter:
        def __init__(self):
            self.decisions = list(decisions)
            self.calls = []

        def decide(self, transcript, context=None):
            self.calls.append((transcript, context))
            return self.decisions.pop(0)

    class ReplyEndpointer:
        def __init__(self):
            self.captures = list(captures)
            self.calls = 0
            self.mics = []

        def record(self, mic, after_prompt=True):
            assert after_prompt is False
            self.calls += 1
            self.mics.append(mic)
            return self.captures.pop(0)

    assistant = Assistant.__new__(Assistant)
    assistant.cfg = SimpleNamespace(get=lambda key, default=None: (
        True if key == "conversation.enabled" else default))
    assistant._conversation_recent = []
    assistant._conversation_reply_pointer = ReplyEndpointer()
    assistant._duck_music = lambda *args: nullcontext()
    assistant.feedback = SimpleNamespace(stopped=lambda: None)
    assistant.stt = SimpleNamespace(model_name="fake-main-stt")
    spoken = []
    assistant._say_feedback = lambda mic, text, event: spoken.append(text) or KIND_SPEECH
    reply_values = list(replies)

    def transcribe(mic, audio):
        value = reply_values.pop(0)
        if isinstance(value, Exception):
            raise value
        return value, 17

    assistant._transcribe_captured = transcribe
    children = []
    assistant.telemetry = SimpleNamespace(
        save_audio=lambda event, audio: "mock-conversation-reply.wav",
        write=lambda event: children.append(event))
    executed = []
    interpreter = Interpreter()

    def execute(step, operation_id):
        executed.append((step.capability, assistant._conversation_local.session["event"].trigger_source))
        return ToolResult(operation_id, "completed", {})

    coordinator = ConversationCoordinator(interpreter, execute, lambda *_: None)
    assistant._conversation_instance = lambda: coordinator
    assistant._conversation_test = (coordinator, interpreter, executed, children, spoken)
    return assistant


def test_definition_is_answered_without_global_delegation():
    answer = Decision("answer", text="A virtuoso is a highly skilled performer.",
                      evidence_quotes=("What does virtuoso mean?",))
    interpreter = FakeInterpreter(answer)
    global_calls = []
    coordinator = ConversationCoordinator(interpreter, lambda *_: None,
                                          lambda *args: global_calls.append(args))
    result = coordinator.handle(Transcript("What does virtuoso mean?", "t1"))
    assert result.decision_kind == "answer"
    assert result.text.startswith("A virtuoso")
    assert global_calls == []


def test_correction_uses_only_the_surviving_pause_step():
    decision = Decision("plan", steps=(_step("pause"),), evidence_quotes=("pause",))
    executed = []
    coordinator = ConversationCoordinator(FakeInterpreter(decision),
        lambda step, op: executed.append(step.capability) or ToolResult(op, "completed", {}),
        lambda *_: None)
    result = coordinator.handle(Transcript("Play Red—actually don't, pause", "turn"))
    assert executed == ["pause"]
    assert result.outcome == "completed"


def test_desktop_steps_use_existing_intent_adapters():
    cases = [
        (_step("search_browser", {"query": "weather in Jaipur", "browser": "Opera"}),
         Action.SEARCH_BROWSER, "weather in Jaipur", "Opera"),
        (_step("open_app", {"name": "Notepad"}), Action.OPEN_APP, "Notepad", "Opera"),
    ]
    for step, action, query, browser in cases:
        intent = step_to_intent(step)
        assert intent.action is action
        assert intent.query == query
        assert intent.browser == browser


def test_absolute_and_relative_volume_preserve_intent_modes():
    absolute = step_to_intent(_step("set_volume", {"level": 37}))
    relative = step_to_intent(_step("change_volume", {
        "direction": "down", "amount": 15, "unit": "percent"}))
    assert (absolute.action, absolute.value, absolute.volume_mode) == (Action.VOLUME, 37, "absolute")
    assert (relative.action, relative.value, relative.query, relative.volume_mode) == (
        Action.VOLUME, 15, "down", "relative_percent")


def test_truncation_and_interpreter_failure_never_execute_or_fall_back():
    executed = []
    coordinator = ConversationCoordinator(FakeInterpreter(
        Decision("plan", steps=(_step("pause"),), evidence_quotes=("pause",))),
        lambda *args: executed.append(args), lambda *args: executed.append(args))
    cut_short = coordinator.handle(Transcript("pause", "t2", truncated=True))
    assert cut_short.decision_kind == "clarify"
    assert executed == []

    unavailable = ConversationCoordinator(FakeInterpreter(error=RuntimeError("offline")),
        lambda *args: executed.append(args), lambda *args: executed.append(args))
    failed = unavailable.handle(Transcript("What is a virtuoso?", "t3"))
    assert failed.decision_kind == "unsupported"
    assert failed.outcome == "interpreter_failed"
    assert executed == []


def test_no_request_is_silent():
    coordinator = ConversationCoordinator(FakeInterpreter(Decision(
        "no_request", reason="quoted speech")), lambda *_: None, lambda *_: None)
    result = coordinator.handle(Transcript("He said pause", "t4"))
    assert (result.decision_kind, result.text, result.outcome) == (
        "no_request", "", "no_request")


def test_nonexplicit_codex_decision_is_refused_by_application_guard():
    from assistant.main import Assistant

    assistant = Assistant.__new__(Assistant)
    assistant_calls = []
    assistant._handle_global = lambda *args: assistant_calls.append(args) or True
    event = SimpleNamespace(transcript="What is Codex?", outcome="")
    result = assistant._delegate_conversation_step(
        _step("codex_request", {"request": event.transcript}), "op1", event, None,
        event.transcript)
    assert result.status == "failed"
    assert assistant_calls == []


def test_explicit_codex_request_uses_existing_global_bridge_and_stays_accepted():
    from assistant.main import Assistant

    assistant = Assistant.__new__(Assistant)
    seen = []

    def global_bridge(source, mic, event):
        seen.append(source)
        event.outcome = "global_handled"
        event.global_status = "awaiting_confirmation"
        event.feedback_kind = "speech"
        return True

    assistant._handle_global = global_bridge
    source = "Ask Codex to implement a parser for this project."
    event = SimpleNamespace(transcript=source, outcome="", feedback_kind="", global_status="")
    result = assistant._delegate_conversation_step(
        _step("codex_request", {"request": source}), "op2", event, None, source)
    assert seen == [source]
    assert result.status == "accepted"
    assert result.data["feedback_spoken"] is True


def test_conversation_brain_is_constructed_with_fake_transport_and_readiness(monkeypatch):
    import json
    from assistant import conversation_brain
    from assistant.main import Assistant
    from assistant.telemetry import Event

    observed = []

    def fake_transport(body, timeout):
        observed.append((body, timeout))
        return {"message": {"content": json.dumps({
            "kind": "answer", "answer": "A virtuoso is a highly skilled performer.",
            "basis": "general_knowledge",
            "evidence_quotes": ["What does virtuoso mean?"]})}}

    monkeypatch.setattr(conversation_brain, "_transport", fake_transport)
    assistant = Assistant.__new__(Assistant)
    assistant.cfg = SimpleNamespace(get=lambda key, default=None: {
        "conversation.enabled": True, "conversation.timeout_seconds": 3.0,
        "intent.model": "local-test-model"}.get(key, default))
    assistant.brain = SimpleNamespace(warming=False)
    assistant._conversation_recent = []
    spoken = []
    assistant._say_feedback = lambda mic, text, event: spoken.append(text) or "speech"
    event = Event()
    assert assistant._handle_conversation("What does virtuoso mean?", None, event)
    assert len(observed) == 1
    assert observed[0][0]["model"] == "local-test-model"
    assert assistant._conversation_coordinator.interpreter.model == "local-test-model"
    assert assistant._conversation_coordinator.interpreter.timeout == 3.0
    assert spoken and spoken[0].startswith("A virtuoso")
    assert assistant._conversation_recent == [
        {"role": "user", "text": "What does virtuoso mean?"},
        {"role": "assistant", "text": spoken[0]},
    ]


def test_lazy_conversation_brain_receives_the_shared_runtime(monkeypatch):
    from assistant import conversation_brain
    from assistant.main import Assistant

    runtime = object()
    constructed = []

    class FakeConversationBrain:
        def __init__(self, model, timeout, *, readiness, runtime):
            constructed.append((model, timeout, readiness, runtime))

        def decide(self, *_args, **_kwargs):
            raise AssertionError("Construction only; no inference")

    monkeypatch.setattr(conversation_brain, "ConversationBrain", FakeConversationBrain)
    assistant = Assistant.__new__(Assistant)
    assistant.cfg = SimpleNamespace(get=lambda key, default=None: {
        "intent.model": "local-test-model", "conversation.timeout_seconds": 3.0
    }.get(key, default))
    assistant._model_runtime = runtime
    assistant.brain = SimpleNamespace(warming=False)
    assistant._conversation_coordinator = None

    assistant._conversation_instance()
    assert len(constructed) == 1
    model, timeout, readiness, injected_runtime = constructed[0]
    assert (model, timeout) == ("local-test-model", 3.0)
    assert readiness == assistant._conversation_ready
    assert injected_runtime is runtime


def test_previous_event_ok_does_not_make_unset_step_successful():
    from assistant.main import Assistant

    assistant = Assistant.__new__(Assistant)
    assistant._execute = lambda *args, **kwargs: None
    event = SimpleNamespace(outcome="ok", error="", intent_action="old",
                            intent_query="old", resolution_path="old",
                            resolved_label="old", playback_method="old",
                            spotify_action="old", feedback_text="old", feedback_kind="speech",
                            intent_source="old")
    result = assistant._execute_conversation_step(_step("pause"), "op3", event, None, None)
    assert result.status == "unknown_delivery"
    assert result.data["legacy_outcome"] == ""
    assert result.data["feedback_spoken"] is False


def test_retry_capture_passes_audio_for_disabled_and_enabled_conversation_paths():
    from contextlib import nullcontext
    from assistant.conversation import ConversationResult
    from assistant.intent import Intent
    from assistant.main import Assistant
    from assistant.telemetry import Event

    audio = object()
    for enabled in (False, True):
        assistant = Assistant.__new__(Assistant)
        assistant.cfg = SimpleNamespace(get=lambda key, default=None: (
            enabled if key == "conversation.enabled" else default))
        assistant.listener = None
        assistant._conversation_recent = []
        assistant._duck_music = lambda *args, **kwargs: nullcontext()
        assistant._ask_and_listen = lambda *args: ("pause", {"audio": audio}, 1)
        assistant.telemetry = SimpleNamespace(save_audio=lambda *args: "")
        called = []
        assistant._say_feedback = lambda mic, text, event: "speech"

        def conversation_handle(*args, **kwargs):
            called.append((args[0].text, assistant._conversation_local.session["audio"]))
            return ConversationResult("answer", "Synthetic answer.", "answered", generation=1)

        assistant._conversation_instance = lambda: SimpleNamespace(
            generation=1, handle=conversation_handle)
        assistant._handle_global = lambda *args: False
        assistant._parse_intent = lambda *args: Intent(Action.PAUSE)
        assistant._execute = lambda intent, event, **kwargs: called.append(kwargs["command_audio"])
        event = Event(event_id="retry")
        assistant._retry_unknown(None, event)
        if enabled:
            assert called == [("pause", audio)]
            assert event.feedback_text == "Synthetic answer."
        else:
            assert called == [audio]


def test_truncated_retry_capture_is_not_interpreted_or_executed():
    from contextlib import nullcontext
    from assistant.conversation import ConversationCoordinator
    from assistant.main import Assistant
    from assistant.telemetry import Event

    executed = []
    coordinator = ConversationCoordinator(
        FakeInterpreter(Decision("plan", steps=(_step("pause"),),
                                 evidence_quotes=("pause",))),
        lambda *args: executed.append(args), lambda *args: executed.append(args))
    assistant = Assistant.__new__(Assistant)
    assistant.cfg = SimpleNamespace(get=lambda key, default=None: (
        True if key == "conversation.enabled" else default))
    assistant.listener = None
    assistant._conversation_recent = []
    assistant._conversation_instance = lambda: coordinator
    assistant._duck_music = lambda *args, **kwargs: nullcontext()
    assistant._ask_and_listen = lambda *args: (
        "pause", {"audio": object(), "truncated": True, "end_reason": "max_duration"}, 1)
    assistant.telemetry = SimpleNamespace(save_audio=lambda *args: "")
    assistant._say_feedback = lambda mic, text, event: "speech"
    assistant._handle_global = lambda *args: (_ for _ in ()).throw(AssertionError("legacy route"))
    assistant._parse_intent = lambda *args: (_ for _ in ()).throw(AssertionError("legacy parse"))
    assistant._execute = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("execute"))
    event = Event(event_id="retry-truncated")
    assistant._retry_unknown(None, event)
    assert event.capture_truncated is True
    assert event.conversation_status == "truncated"
    assert executed == []


def test_plan_stops_after_noncompleted_result_and_invalidation_suppresses_speech():
    calls = []
    plan = Decision("plan", steps=(_step("pause", step_id="a"),
                                      _step("resume", after=("a",), step_id="b")),
                    evidence_quotes=("pause then resume",))
    coordinator = ConversationCoordinator(FakeInterpreter(plan),
        lambda step, op: calls.append(step.capability) or ToolResult(op, "accepted", {}),
        lambda *_: None)
    result = coordinator.handle(Transcript("pause then resume", "t5"))
    assert calls == ["pause"]
    assert result.outcome == "partial_accepted"

    holder = {}

    class InvalidatingInterpreter:
        def decide(self, transcript, context=None):
            holder["coordinator"].invalidate()
            return Decision("answer", text="stale answer", evidence_quotes=("answer",))

    invalidated = ConversationCoordinator(InvalidatingInterpreter(), lambda *_: None, lambda *_: None)
    holder["coordinator"] = invalidated
    stale = invalidated.handle(Transcript("answer", "t6"))
    assert stale.outcome == "invalidated"
    assert stale.text == ""


def test_invalidated_during_execution_prevents_following_steps():
    holder = {}
    calls = []

    def execute(step, operation_id):
        calls.append(step.capability)
        holder["coordinator"].invalidate()
        return ToolResult(operation_id, "completed", {})

    decision = Decision("plan", steps=(_step("pause", step_id="a"),
                                        _step("resume", after=("a",), step_id="b")),
                        evidence_quotes=("pause then resume",))
    coordinator = ConversationCoordinator(FakeInterpreter(decision), execute, lambda *_: None)
    holder["coordinator"] = coordinator
    result = coordinator.handle(Transcript("pause then resume", "t7"))
    assert calls == ["pause"]
    assert result.outcome == "invalidated"
    assert result.text == ""
    assert len(result.results) == 1
    assert result.results[0].status == "completed"


def test_failure_and_accepted_results_have_safe_noncompletion_copy():
    for status, expected in (("failed", "Sorry, that action failed."),
                             ("accepted", "The request was accepted and is still pending."),
                             ("unknown_delivery", "I can't confirm whether that action happened.")):
        decision = Decision("plan", steps=(_step("pause"),), evidence_quotes=("pause",))
        coordinator = ConversationCoordinator(FakeInterpreter(decision),
            lambda step, op: ToolResult(op, status, {"feedback_spoken": False}), lambda *_: None)
        result = coordinator.handle(Transcript("pause", "t8"))
        assert result.text == expected


def test_callback_that_mutates_then_raises_is_unknown_and_never_retried():
    side_effects = []
    decision = Decision("plan", steps=(_step("pause", step_id="a"),
                                        _step("resume", after=("a",), step_id="b")),
                        evidence_quotes=("pause then resume",))

    def mutate_then_raise(step, operation_id):
        side_effects.append(step.capability)
        raise RuntimeError("private adapter details")

    coordinator = ConversationCoordinator(FakeInterpreter(decision), mutate_then_raise,
                                          lambda *_: None)
    result = coordinator.handle(Transcript("pause then resume", "uncertain"))
    assert side_effects == ["pause"]
    assert result.outcome == "partial_unknown_delivery"
    assert result.text == "I can't confirm whether that action happened."
    assert len(result.results) == 1
    assert result.results[0].status == "unknown_delivery"


def test_answer_history_uses_new_answer_not_stale_wake_feedback():
    from assistant.conversation import ConversationResult
    from assistant.main import Assistant
    from assistant.telemetry import Event

    assistant = Assistant.__new__(Assistant)
    assistant.cfg = SimpleNamespace(get=lambda key, default=None: (
        True if key == "conversation.enabled" else default))
    assistant._conversation_recent = []
    assistant._conversation_instance = lambda: SimpleNamespace(
        generation=7, handle=lambda *args, **kwargs: ConversationResult(
            "answer", "Fresh answer", "answered", generation=7))
    spoken = []

    def deliver(mic, text, event):
        assert assistant._conversation_recent == []
        spoken.append(text)
        return "speech"

    assistant._say_feedback = deliver
    event = Event(event_id="answer-history")
    event.feedback_text = "Yes sir"
    event.feedback_kind = "speech"
    assert assistant._handle_conversation("What is new?", None, event)
    assert spoken == ["Fresh answer"]
    assert assistant._conversation_recent == [
        {"role": "user", "text": "What is new?"},
        {"role": "assistant", "text": "Fresh answer"},
    ]


def test_no_request_clears_stale_event_feedback_and_stays_silent():
    from assistant.main import Assistant
    from assistant.telemetry import Event

    assistant = Assistant.__new__(Assistant)
    assistant.cfg = SimpleNamespace(get=lambda key, default=None: (
        True if key == "conversation.enabled" else default))
    assistant._conversation_recent = []
    assistant._conversation_instance = lambda: ConversationCoordinator(
        FakeInterpreter(Decision("no_request", reason="quoted speech")),
        lambda *_: None, lambda *_: None)
    spoken = []
    assistant._say_feedback = lambda *args: spoken.append(args) or "speech"
    event = Event(event_id="silent-no-request")
    event.outcome = "ok"
    event.feedback_text = "Yes sir"
    event.feedback_kind = "speech"
    assert assistant._handle_conversation("He said pause", None, event)
    assert event.outcome == "conversation_no_request"
    assert event.feedback_text == event.feedback_kind == ""
    assert spoken == []
    assert assistant._conversation_recent == []


def test_invalidated_receipt_is_retained_but_stale_feedback_and_history_are_suppressed():
    from assistant.main import Assistant
    from assistant.telemetry import Event

    assistant = Assistant.__new__(Assistant)
    assistant.cfg = SimpleNamespace(get=lambda key, default=None: (
        True if key == "conversation.enabled" else default))
    assistant._conversation_recent = []
    assistant._execute = lambda intent, event, **kwargs: setattr(event, "outcome", "ok")
    holder = {}

    def execute(step, operation_id):
        receipt = assistant._execute_conversation_step(step, operation_id, event, None, None)
        holder["coordinator"].invalidate()
        return receipt

    decision = Decision("plan", steps=(_step("pause"),), evidence_quotes=("pause",))
    coordinator = ConversationCoordinator(FakeInterpreter(decision), execute, lambda *_: None)
    holder["coordinator"] = coordinator
    assistant._conversation_instance = lambda: coordinator
    spoken = []
    assistant._say_feedback = lambda *args: spoken.append(args) or "speech"
    event = Event(event_id="invalidated-receipt")
    event.feedback_text = "stale wake text"
    event.feedback_kind = "speech"
    assert assistant._handle_conversation("pause", None, event)
    assert event.conversation_status == "invalidated"
    assert event.conversation_results_summary == "completed"
    assert event.outcome == "conversation_invalidated"
    assert event.feedback_text == event.feedback_kind == ""
    assert spoken == []
    assert assistant._conversation_recent == []


def test_pending_clarification_grounds_play_that_from_current_reply():
    decisions = [
        Decision("clarify", text="Which song do you mean?", evidence_quotes=("Play that",)),
        Decision("plan", steps=(_step("play_song", {"title": "Madira", "artist": ""}),),
                 evidence_quotes=("Madira",)),
    ]

    class Sequenced:
        def __init__(self):
            self.calls = []

        def decide(self, transcript, context=None):
            self.calls.append((transcript, context))
            return decisions.pop(0)

    calls = []
    interpreter = Sequenced()
    coordinator = ConversationCoordinator(interpreter,
        lambda step, op: calls.append(step.arguments) or ToolResult(op, "completed", {}),
        lambda *_: None, clock=lambda: 10.0)
    first = coordinator.handle(Transcript("Play that", "q1"))
    second = coordinator.handle(Transcript("Madira", "q2", grounding_sources=("Injected",)))
    assert first.outcome == "clarification_needed"
    assert second.outcome == "completed"
    assert calls == [{"title": "Madira", "artist": ""}]
    effective, context = interpreter.calls[1]
    assert effective.text == "Madira"
    assert effective.grounding_sources == ("Play that",)
    assert context["pending_request"] == {
        "sources": ["Play that"], "question": "Which song do you mean?"}


def test_pending_clarification_supplies_missing_artist_without_rewriting_reply():
    decisions = [
        Decision("clarify", text="Which artist?", evidence_quotes=("Play Madira",)),
        Decision("plan", steps=(_step("play_song", {
            "title": "Madira", "artist": "Seedhe Maut"}),), evidence_quotes=("Seedhe Maut",)),
    ]
    interpreter = FakeInterpreter()
    interpreter.decide = lambda transcript, context=None: decisions.pop(0)
    executed = []
    coordinator = ConversationCoordinator(interpreter,
        lambda step, op: executed.append(step.arguments) or ToolResult(op, "completed", {}),
        lambda *_: None, clock=lambda: 5.0)
    coordinator.handle(Transcript("Play Madira", "artist-q1"))
    result = coordinator.handle(Transcript("Seedhe Maut", "artist-q2"))
    assert result.outcome == "completed"
    assert executed == [{"title": "Madira", "artist": "Seedhe Maut"}]


def test_new_answer_clears_pending_without_running_old_request():
    decisions = [
        Decision("clarify", text="Which song?", evidence_quotes=("Play that",)),
        Decision("answer", text="A virtuoso is a skilled performer.",
                 evidence_quotes=("What is a virtuoso?",)),
    ]
    called = []
    interpreter = FakeInterpreter()
    interpreter.decide = lambda transcript, context=None: decisions.pop(0)
    coordinator = ConversationCoordinator(interpreter,
        lambda *args: called.append(args), lambda *args: called.append(args))
    coordinator.handle(Transcript("Play that", "new-q1"))
    result = coordinator.handle(Transcript("What is a virtuoso?", "new-q2"))
    assert result.decision_kind == "answer"
    assert result.text.startswith("A virtuoso")
    assert coordinator.pending_clarification is None
    assert called == []


def test_pending_clarification_expiry_cancel_background_and_attempt_budget():
    now = [0.0]
    decisions = [
        Decision("clarify", text="Which song?", evidence_quotes=("Play that",)),
        Decision("no_request", reason="background", evidence_quotes=()),
        Decision("clarify", text="Which artist?", evidence_quotes=("Play Madira",)),
        Decision("clarify", text="Which artist?", evidence_quotes=("Seedhe Maut",)),
        Decision("clarify", text="Which song?", evidence_quotes=("Play that",)),
        Decision("clarify", text="Which song?", evidence_quotes=("Play that",)),
    ]
    interpreter = FakeInterpreter()
    interpreter.decide = lambda transcript, context=None: decisions.pop(0)
    executed = []
    coordinator = ConversationCoordinator(interpreter,
        lambda *args: executed.append(args), lambda *args: executed.append(args),
        clock=lambda: now[0])
    coordinator.handle(Transcript("Play that", "state-q1"))
    quiet = coordinator.handle(Transcript("He said pause", "state-q2"))
    assert quiet.outcome == "no_request"
    assert coordinator.pending_clarification is not None
    coordinator.handle(Transcript("Play Madira", "state-q3"))
    budget = coordinator.handle(Transcript("Seedhe Maut", "state-q4"))
    assert budget.outcome == "clarification_restate"
    assert coordinator.pending_clarification is None
    assert executed == []

    coordinator.handle(Transcript("Play that", "state-expire"))
    now[0] = 31.0
    expired = coordinator.handle(Transcript("Madira", "state-expired-reply"))
    assert expired.outcome == "clarification_expired"
    assert coordinator.pending_clarification is None
    coordinator.handle(Transcript("Play that", "state-cancel"))
    decisions_before_cancel = len(decisions)
    generation_before_cancel = coordinator.generation
    cancelled = coordinator.handle(Transcript("Never mind.", "state-cancel-reply"))
    assert cancelled.outcome == "cancelled"
    assert coordinator.generation == generation_before_cancel + 1
    assert coordinator.pending_clarification is None
    assert len(decisions) == decisions_before_cancel


def test_terminal_invalidity_clears_pending_clarification():
    coordinator = ConversationCoordinator(FakeInterpreter(Decision(
        "clarify", text="Which song?", evidence_quotes=("Play that",))),
        lambda *_: None, lambda *_: None, clock=lambda: 1.0)
    assert coordinator.handle(Transcript("Play that", "clear-first")).outcome == "clarification_needed"
    assert coordinator.pending_clarification is not None
    cut_short = coordinator.handle(Transcript("Madira", "clear-truncated", truncated=True))
    assert cut_short.outcome == "truncated"
    assert coordinator.pending_clarification is None

    coordinator.handle(Transcript("Play that", "clear-second"))
    coordinator.interpreter = FakeInterpreter(error=RuntimeError("offline"))
    failed = coordinator.handle(Transcript("Madira", "clear-error"))
    assert failed.outcome == "interpreter_failed"
    assert coordinator.pending_clarification is None

    coordinator.interpreter = FakeInterpreter(Decision(
        "clarify", text="Which song?", evidence_quotes=("Play that",)))
    coordinator.handle(Transcript("Play that", "clear-third"))
    coordinator.interpreter = FakeInterpreter(Decision(
        "plan", steps=(_step("play_song", {"title": "invented", "artist": ""}),),
        evidence_quotes=("Madira",)))
    invalid = coordinator.handle(Transcript("Madira", "clear-invalid"))
    assert invalid.outcome == "invalid_decision"
    assert coordinator.pending_clarification is None


def test_public_invalidation_clears_pending_and_bumps_generation():
    coordinator = ConversationCoordinator(FakeInterpreter(Decision(
        "clarify", text="Which song?", evidence_quotes=("Play that",))),
        lambda *_: None, lambda *_: None)
    coordinator.handle(Transcript("Play that", "invalidate-pending"))
    before = coordinator.generation
    after = coordinator.invalidate()
    assert after == before + 1
    assert coordinator.pending_clarification is None


def test_conditional_invalidation_preserves_a_newer_pending_clarification():
    interpreter = FakeInterpreter()
    decisions = [
        Decision("clarify", text="Which song?", evidence_quotes=("Play that",)),
        Decision("clarify", text="Which artist?", evidence_quotes=("Play Madira",)),
    ]
    interpreter.decide = lambda *_args, **_kwargs: decisions.pop(0)
    coordinator = ConversationCoordinator(interpreter, lambda *_: None, lambda *_: None)
    old = coordinator.handle(Transcript("Play that", "conditional-old"))
    newer = coordinator.handle(Transcript("Play Madira", "conditional-new"))
    pending = coordinator.pending_clarification
    assert not coordinator.invalidate_if_current(old.generation)
    assert coordinator.pending_clarification is pending
    assert coordinator.invalidate_if_current(newer.generation)
    assert coordinator.pending_clarification is None


def test_expected_generation_atomically_admits_or_rejects_a_clarification_reply():
    decisions = [
        Decision("clarify", text="Which song?", evidence_quotes=("Play that",)),
        Decision("clarify", text="Which title?", evidence_quotes=("Play Madira",)),
    ]
    interpreter = FakeInterpreter()
    def decide(transcript, context=None):
        interpreter.calls.append((transcript, context))
        return decisions.pop(0)
    interpreter.decide = decide
    coordinator = ConversationCoordinator(interpreter, lambda *_: None, lambda *_: None)

    initial = coordinator.handle(Transcript("Play that", "admission-first"))
    assert initial.outcome == "clarification_needed"
    pending = coordinator.pending_clarification
    generation = coordinator.generation

    stale = coordinator.handle(
        Transcript("Play Madira", "admission-stale"),
        expected_generation=generation - 1)
    assert stale.outcome == "invalidated"
    assert coordinator.generation == generation
    assert coordinator.pending_clarification is pending
    assert len(interpreter.calls) == 1

    admitted = coordinator.handle(
        Transcript("Play Madira", "admission-current"),
        expected_generation=generation)
    assert admitted.outcome == "clarification_needed"
    assert coordinator.generation == generation + 1
    assert coordinator.pending_clarification is not pending
    assert len(interpreter.calls) == 2


def test_older_blocked_interpreter_cannot_clear_newer_pending_question():
    entered = threading.Event()
    release = threading.Event()
    old_result = []

    class BlockingInterpreter:
        def decide(self, transcript, context=None):
            if transcript.text == "old request":
                entered.set()
                assert release.wait(2.0)
                return object()  # malformed late decision must not clear new state
            return Decision("clarify", text="Which artist?",
                            evidence_quotes=(transcript.text,))

    coordinator = ConversationCoordinator(BlockingInterpreter(), lambda *_: None,
                                          lambda *_: None)
    worker = threading.Thread(target=lambda: old_result.append(
        coordinator.handle(Transcript("old request", "older"))))
    worker.start()
    assert entered.wait(2.0)
    new_result = coordinator.handle(Transcript("new request", "newer"))
    current_pending = coordinator.pending_clarification
    assert new_result.outcome == "clarification_needed"
    release.set()
    worker.join(2.0)
    assert not worker.is_alive()
    assert old_result[0].outcome == "invalidated"
    assert coordinator.pending_clarification is current_pending
    assert current_pending is not None and current_pending.sources == ("new request",)


def test_compound_answer_is_returned_after_completion_and_with_partial_status():
    decision = Decision("plan", steps=(_step("pause"),), evidence_quotes=("Pause",),
                        answer="A virtuoso is an exceptionally skilled performer.",
                        answer_basis="general_knowledge")
    completed = ConversationCoordinator(FakeInterpreter(decision),
        lambda step, op: ToolResult(op, "completed", {}), lambda *_: None)
    result = completed.handle(Transcript("Pause and define virtuoso", "compound-ok"))
    assert result.outcome == "completed"
    assert result.text == decision.answer

    partial = ConversationCoordinator(FakeInterpreter(decision),
        lambda step, op: ToolResult(op, "unknown_delivery", {"feedback_spoken": False}),
        lambda *_: None)
    result = partial.handle(Transcript("Pause and define virtuoso", "compound-partial"))
    assert result.outcome == "partial_unknown_delivery"
    assert "can't confirm" in result.text
    assert decision.answer in result.text
    assert "completed" not in result.text.casefold()


def test_exact_global_confirmation_keeps_legacy_owner_but_fresh_question_escapes_session():
    from assistant.main import Assistant
    from assistant.telemetry import Event

    class Controller:
        session = SimpleNamespace(active=True, phase="review")
        _pending_choice = None
        _unsent_request = None

        def _valid_unsent(self, held):
            return False

    cases = (("send it", "review", False),
             ("finish dictation", "dictating", False),
             ("what does virtuoso mean", "review", True),
             ("define virtuoso", "dictating", True),
             ("explain what virtuoso means", "dictating", True),
             ("explain my code", "dictating", False),
             ('"what does virtuoso mean?"', "dictating", False),
             ("He asked what does virtuoso mean", "dictating", False),
             ("Summarize this design", "dictating", False))
    for command, phase, expected in cases:
        assistant = Assistant.__new__(Assistant)
        assistant.cfg = SimpleNamespace(get=lambda key, default=None: (
            True if key == "conversation.enabled" else default))
        assistant.global_assistant = Controller()
        assistant.global_assistant.session.phase = phase
        assistant._conversation_instance = lambda: ConversationCoordinator(
            FakeInterpreter(Decision("answer", text="A skilled performer.",
                                     evidence_quotes=(command,))),
            lambda *_: None, lambda *_: None)
        assistant._say_feedback = lambda *args: "speech"
        event = Event(event_id="legacy-control" if command == "send it" else "fresh-question")
        assert assistant._handle_conversation(command, None, event) is expected


def test_listener_feed_queue_overflow_rejects_generated_utterance_and_recovers():
    from assistant.main import Assistant

    class Microphone:
        queue_overflow_count = 0

        def __init__(self):
            self.flushed = 0

        def flush(self):
            self.flushed += 1

    class Listener:
        def __init__(self, mic):
            self.mic = mic
            self.flushed = 0

        def feed(self, frame, cleaned):
            self.mic.queue_overflow_count += 1
            return object()

        def flush(self):
            self.flushed += 1

    class Wake:
        def __init__(self):
            self.resets = 0

        def reset(self):
            self.resets += 1

    mic = Microphone()
    assistant = Assistant.__new__(Assistant)
    assistant.mode = "stt"
    assistant.listener = Listener(mic)
    assistant.wake = Wake()
    assistant._early_lease = None
    assistant._record_wake_rejection = lambda reason, mic=None: None
    utterance, gap, count = assistant._feed_listener_guarded(mic, object(), object())
    assert utterance is None
    assert gap is True
    assert count == 1
    assert mic.flushed == assistant.listener.flushed == assistant.wake.resets == 1


def test_malformed_pending_reply_never_executes_or_spawns_new_pending_state():
    decisions = [
        Decision("clarify", text="Which song?", evidence_quotes=("Play that",)),
        Decision("plan", steps=(_step("play_song", {"title": "invented", "artist": ""}),),
                 evidence_quotes=("Madira",)),
    ]
    calls = []
    interpreter = FakeInterpreter()
    interpreter.decide = lambda transcript, context=None: decisions.pop(0)
    coordinator = ConversationCoordinator(interpreter,
        lambda *args: calls.append(args), lambda *args: calls.append(args), clock=lambda: 1.0)
    coordinator.handle(Transcript("Play that", "malformed-q1"))
    result = coordinator.handle(Transcript("Madira", "malformed-q2"))
    assert result.outcome == "invalid_decision"
    assert calls == []
    assert coordinator.pending_clarification is None


def test_compound_answer_is_suppressed_when_execution_generation_is_stale():
    holder = {}
    decision = Decision("plan", steps=(_step("pause"),), evidence_quotes=("Pause",),
                        answer="A virtuoso is an exceptionally skilled performer.",
                        answer_basis="general_knowledge")

    def execute(step, operation_id):
        holder["coordinator"].invalidate()
        return ToolResult(operation_id, "completed", {})

    coordinator = ConversationCoordinator(FakeInterpreter(decision), execute, lambda *_: None)
    holder["coordinator"] = coordinator
    result = coordinator.handle(Transcript("Pause and define virtuoso", "compound-stale"))
    assert result.outcome == "invalidated"
    assert result.text == ""
    assert len(result.results) == 1


def test_quota_exhaustion_is_reported_without_retry_or_generic_rate_limit_copy():
    from assistant.main import Assistant, SPOTIFY_QUOTA_MESSAGE
    from assistant.telemetry import Event
    from assistant.spotify import SpotifyQuotaExceeded

    assistant = Assistant.__new__(Assistant)
    assistant._execute = lambda *args, **kwargs: (_ for _ in ()).throw(SpotifyQuotaExceeded())
    spoken = []
    assistant._say_feedback = lambda mic, text, event: spoken.append(text) or "speech"
    event = Event(event_id="quota")
    result = assistant._execute_conversation_step(_step("play_song", {
        "title": "Madira", "artist": ""}), "quota-op", event, object(), None)
    assert result.status == "failed"
    assert result.data["legacy_outcome"] == "spotify_quota_exceeded"
    assert spoken == [SPOTIFY_QUOTA_MESSAGE]
    assert "retry" not in SPOTIFY_QUOTA_MESSAGE.casefold()

    assistant.spotify = SimpleNamespace(quota_status=lambda: {"state": "exhausted"})
    assert assistant.health_probes()["spotify_quota"]() == {"state": "exhausted"}


def test_desktop_delivery_unknown_is_distinct_from_refusal_and_stops_plan(monkeypatch):
    from assistant import computer_tools
    from assistant.main import Assistant
    from assistant.telemetry import Event

    for error_type, expected_status, expected_outcome in (
            (computer_tools.ComputerControlError, "failed", "partial_failed"),
            (computer_tools.ComputerDeliveryUnknown, "unknown_delivery",
             "partial_unknown_delivery")):
        assistant = Assistant.__new__(Assistant)
        event = Event(event_id=f"desktop-{expected_status}")
        spoken = []
        executed = []
        assistant._say_feedback = lambda mic, text, event: spoken.append(text) or "speech"

        def open_app(_name):
            raise error_type("bounded desktop feedback")

        monkeypatch.setattr(computer_tools, "open_app", open_app)

        def run_intent(intent, current_event, **kwargs):
            executed.append(intent.action)
            if intent.action is Action.OPEN_APP:
                assistant._execute_desktop(intent, current_event, kwargs.get("mic"))
            else:
                current_event.outcome = "ok"

        assistant._execute = run_intent
        plan = Decision("plan", steps=(
            _step("open_app", {"name": "Notepad"}, step_id="desktop"),
            _step("pause", after=("desktop",), step_id="pause"),
        ), evidence_quotes=("open Notepad then pause",))
        coordinator = ConversationCoordinator(FakeInterpreter(plan),
            lambda step, operation_id: assistant._execute_conversation_step(
                step, operation_id, event, object(), None), lambda *_: None)
        result = coordinator.handle(Transcript("open Notepad then pause", event.event_id))
        assert result.results[0].status == expected_status
        assert result.results[0].data["legacy_outcome"] == (
            "desktop_refused" if expected_status == "failed" else "desktop_unknown_delivery")
        assert result.outcome == expected_outcome
        assert executed == [Action.OPEN_APP]
        assert spoken == ["bounded desktop feedback"]


def test_clarification_captures_wake_free_reply_and_logs_child_event():
    from assistant.main import Assistant
    from assistant.telemetry import Event

    assistant = _reply_capture_assistant([
        Decision("clarify", text="Which artist?", evidence_quotes=("Play Madira",)),
        Decision("plan", steps=(_step("play_song", {
            "title": "Madira", "artist": "Seedhe Maut"}),),
            evidence_quotes=("Seedhe Maut",)),
    ], [{"audio": object(), "speech_started": True, "duration_ms": 900,
        "end_reason": "silence", "truncated": False, "wait_duration_ms": 1200}],
        ["Seedhe Maut"])
    event = Event(event_id="clarify-parent", trigger_source="hotkey")
    mic = object()
    assert assistant._handle_conversation("Play Madira", mic, event)
    coordinator, interpreter, executed, children, spoken = assistant._conversation_test
    assert spoken == ["Which artist?"]
    assert assistant._conversation_reply_pointer.calls == 1
    assert assistant._conversation_reply_pointer.mics == [mic]
    assert executed == [("play_song", "conversation_reply")]
    assert event.outcome == "conversation_clarification_needed"
    assert event.conversation_status == "clarification_needed"
    assert len(children) == 1
    child = children[0]
    assert child.trigger_source == "conversation_reply"
    assert child.conversation_parent_event_id == event.event_id
    assert child.transcript == "Seedhe Maut"
    assert child.stt_model == "fake-main-stt"
    assert child.capture_wait_ms == 1200
    assert child.audio_path == "mock-conversation-reply.wav"
    assert child.conversation_status == "completed"
    assert len(interpreter.calls) == 2


def test_clarification_cancellation_timeout_and_truncation_stop_without_effects():
    from assistant.main import Assistant
    from assistant.telemetry import Event

    clarify = Decision("clarify", text="Which song?", evidence_quotes=("Play that",))
    canceled = _reply_capture_assistant([clarify], [
        {"audio": object(), "speech_started": True, "duration_ms": 500,
         "end_reason": "silence", "truncated": False}], ["cancel"])
    cancel_event = Event(event_id="reply-cancel")
    assert canceled._handle_conversation("Play that", object(), cancel_event)
    _, _, executed, cancel_children, cancel_spoken = canceled._conversation_test
    assert executed == []
    assert canceled._conversation_reply_pointer.calls == 1
    assert cancel_children[0].conversation_status == "cancelled"
    assert "cancel" in cancel_spoken[-1].casefold()

    timeout = _reply_capture_assistant([clarify], [
        {"audio": object(), "speech_started": False, "duration_ms": 12000,
         "end_reason": "speech_timeout", "truncated": False}], [])
    timeout_event = Event(event_id="reply-timeout")
    timeout._handle_conversation("Play that", object(), timeout_event)
    _, _, timeout_effects, timeout_children, timeout_spoken = timeout._conversation_test
    assert timeout_effects == []
    assert timeout_children[0].outcome == "conversation_reply_timeout"
    assert timeout_spoken[-1] == "Ask me again when ready."
    assert timeout._conversation_test[0].pending_clarification is None

    truncated = _reply_capture_assistant([clarify], [
        {"audio": object(), "speech_started": True, "duration_ms": 8000,
         "end_reason": "utterance_limit", "truncated": True}], [])
    truncated_event = Event(event_id="reply-truncated")
    truncated._handle_conversation("Play that", object(), truncated_event)
    _, truncated_interpreter, truncated_effects, truncated_children, _ = truncated._conversation_test
    assert truncated_effects == []
    assert truncated._conversation_reply_pointer.calls == 1
    assert truncated_children[0].outcome == "conversation_reply_truncated"
    assert len(truncated_interpreter.calls) == 1
    assert truncated._conversation_test[0].pending_clarification is None


def test_clarification_stt_failure_and_spoken_failure_never_start_action_capture():
    from assistant.feedback import KIND_ERR
    from assistant.main import Assistant
    from assistant.telemetry import Event

    clarify = Decision("clarify", text="Which artist?", evidence_quotes=("Play Madira",))
    failed_stt = _reply_capture_assistant([clarify], [
        {"audio": object(), "speech_started": True, "duration_ms": 700,
         "end_reason": "silence", "truncated": False}], [RuntimeError("decode failed")])
    failed_event = Event(event_id="reply-stt-error")
    failed_stt._handle_conversation("Play Madira", object(), failed_event)
    _, _, effects, children, spoken = failed_stt._conversation_test
    assert effects == []
    assert children[0].outcome == "conversation_reply_stt_failed"
    assert "couldn't process" in spoken[-1]
    assert failed_stt._conversation_test[0].pending_clarification is None
    assert len(failed_stt._conversation_test[1].calls) == 1

    empty = _reply_capture_assistant([clarify], [{
        "audio": object(), "speech_started": True, "duration_ms": 600,
        "end_reason": "silence", "truncated": False}], [""])
    empty_event = Event(event_id="reply-empty-stt")
    empty._handle_conversation("Play Madira", object(), empty_event)
    assert empty._conversation_test[2] == []
    assert empty._conversation_test[3][0].outcome == "conversation_reply_empty"
    assert len(empty._conversation_test[1].calls) == 1
    assert empty._conversation_test[0].pending_clarification is None

    silent = _reply_capture_assistant([clarify], [
        {"audio": object(), "speech_started": True, "duration_ms": 700,
         "end_reason": "silence", "truncated": False}], ["Seedhe Maut"])
    silent._say_feedback = lambda mic, text, event: KIND_ERR
    silent_event = Event(event_id="reply-prompt-not-spoken")
    silent._handle_conversation("Play Madira", object(), silent_event)
    assert silent._conversation_reply_pointer.calls == 0
    assert silent._conversation_test[2] == []
    assert silent_event.outcome == "conversation_clarification_speech_failed"
    assert silent._conversation_test[0].pending_clarification is None


def test_clarification_uses_at_most_two_captures_and_stops_if_generation_stales():
    from assistant.main import Assistant
    from assistant.telemetry import Event

    assistant = _reply_capture_assistant([
        Decision("clarify", text="Which item?", evidence_quotes=("Play that",)),
        Decision("clarify", text="Which artist?", evidence_quotes=("Madira",)),
        Decision("answer", text="A virtuoso is a skilled performer.",
                 evidence_quotes=("virtuoso",)),
    ], [
        {"audio": object(), "speech_started": True, "duration_ms": 500,
         "end_reason": "silence", "truncated": False},
        {"audio": object(), "speech_started": True, "duration_ms": 600,
         "end_reason": "silence", "truncated": False}],
        ["Madira", "virtuoso"])
    event = Event(event_id="reply-two")
    assistant._handle_conversation("Play that", object(), event)
    assert assistant._conversation_reply_pointer.calls == 2
    assert len(assistant._conversation_test[3]) == 2

    stale = _reply_capture_assistant([
        Decision("clarify", text="Which artist?", evidence_quotes=("Play Madira",))], [], [])
    from assistant.feedback import KIND_SPEECH

    def invalidate_while_speaking(mic, text, current_event):
        stale._conversation_test[0].invalidate()
        return KIND_SPEECH

    stale._say_feedback = invalidate_while_speaking
    stale_event = Event(event_id="reply-stale")
    stale._handle_conversation("Play Madira", object(), stale_event)
    assert stale._conversation_reply_pointer.calls == 0
    assert stale_event.outcome == "conversation_invalidated"


def test_reply_endpointer_is_lazy_bounded_and_shares_main_vad(monkeypatch):
    from assistant import main
    from assistant.main import Assistant

    observed = []

    class FakeEndpointer:
        def __init__(self, vad, **kwargs):
            observed.append((vad, kwargs))

    monkeypatch.setattr(main, "Endpointer", FakeEndpointer)
    vad = object()
    assistant = Assistant.__new__(Assistant)
    assistant.vad = vad
    assistant.endpointer = SimpleNamespace(
        vad=vad, sample_rate=16000, frame_ms=32.0, threshold=0.44,
        echo_window_frames=10, echo_max_frames=6)
    assistant.cfg = SimpleNamespace(get=lambda key, default=None: default)
    first = assistant._conversation_reply_endpointer()
    second = assistant._conversation_reply_endpointer()
    assert first is second
    assert len(observed) == 1
    assert observed[0][0] is vad
    assert observed[0][1] == {
        "sample_rate": 16000, "frame_length": 512, "silence_ms": 600,
        "max_seconds": 12.0, "speech_timeout_ms": 3500, "threshold": 0.44,
        "echo_window_ms": 320, "echo_max_ms": 192, "utterance_seconds": 8.0}


def test_assistant_without_microphone_or_endpointer_keeps_clarification_unit_only():
    from assistant.feedback import KIND_SPEECH
    from assistant.main import Assistant
    from assistant.telemetry import Event

    assistant = Assistant.__new__(Assistant)
    assistant.cfg = SimpleNamespace(get=lambda key, default=None: (
        True if key == "conversation.enabled" else default))
    assistant._conversation_recent = []
    assistant._conversation_instance = lambda: ConversationCoordinator(
        FakeInterpreter(Decision("clarify", text="Which song?",
                                 evidence_quotes=("Play that",))),
        lambda *_: None, lambda *_: None)
    assistant._say_feedback = lambda *args: KIND_SPEECH
    event = Event(event_id="unit-only-clarification")
    assert assistant._handle_conversation("Play that", None, event)
    assert not hasattr(assistant, "_conversation_reply_pointer")
    assert event.outcome == "conversation_clarification_needed"


def test_reply_capture_and_stt_failures_cannot_clear_or_speak_over_newer_question():
    from assistant.conversation_contracts import Transcript
    from assistant.telemetry import Event

    clarify_old = Decision("clarify", text="Which artist?",
                           evidence_quotes=("Play Madira",))
    clarify_new = Decision("clarify", text="Which title?",
                           evidence_quotes=("Play new request",))
    capturing = _reply_capture_assistant([clarify_old, clarify_new], [], [])
    coordinator = capturing._conversation_test[0]

    def mutate_then_capture_raise(mic, after_prompt=True):
        coordinator.handle(Transcript("Play new request", "newer-during-capture"))
        raise RuntimeError("capture failed")

    capturing._conversation_reply_pointer.record = mutate_then_capture_raise
    capture_event = Event(event_id="old-capture-failure")
    capturing._handle_conversation("Play Madira", object(), capture_event)
    assert capturing._conversation_test[4] == ["Which artist?"]
    capture_pending = capturing._conversation_test[0].pending_clarification
    assert capture_pending.question == "Which title?"
    assert capture_pending.sources[-1] == "Play new request"
    assert capturing._conversation_test[3][0].outcome == "conversation_invalidated"
    assert capturing._conversation_test[3][0].feedback_text == ""

    stt = _reply_capture_assistant([clarify_old, clarify_new], [{
        "audio": object(), "speech_started": True, "duration_ms": 500,
        "end_reason": "silence", "truncated": False}], [])
    coordinator = stt._conversation_test[0]

    def mutate_then_stt_raise(mic, audio):
        coordinator.handle(Transcript("Play new request", "newer-during-stt"))
        raise RuntimeError("decoder failed")

    stt._transcribe_captured = mutate_then_stt_raise
    stt_event = Event(event_id="old-stt-failure")
    stt._handle_conversation("Play Madira", object(), stt_event)
    assert stt._conversation_test[4] == ["Which artist?"]
    stt_pending = stt._conversation_test[0].pending_clarification
    assert stt_pending.question == "Which title?"
    assert stt_pending.sources[-1] == "Play new request"
    assert stt._conversation_test[3][0].outcome == "conversation_invalidated"
    assert stt._conversation_test[3][0].feedback_text == ""


def test_failed_clarification_speech_does_not_invalidate_newer_pending_question():
    from assistant.feedback import KIND_ERR
    from assistant.conversation_contracts import Transcript
    from assistant.telemetry import Event

    assistant = _reply_capture_assistant([
        Decision("clarify", text="Which artist?", evidence_quotes=("Play Madira",)),
        Decision("clarify", text="Which title?", evidence_quotes=("Play new request",)),
    ], [], [])
    coordinator = assistant._conversation_test[0]
    spoken = []

    def new_question_then_speech_failure(mic, text, event):
        spoken.append(text)
        if text == "Which artist?":
            coordinator.handle(Transcript("Play new request", "newer-during-prompt"))
        return KIND_ERR

    assistant._say_feedback = new_question_then_speech_failure
    event = Event(event_id="old-prompt-failure")
    assistant._handle_conversation("Play Madira", object(), event)
    assert spoken == ["Which artist?"]
    assert assistant._conversation_reply_pointer.calls == 0
    assert coordinator.pending_clarification.question == "Which title?"
    assert coordinator.pending_clarification.sources[-1] == "Play new request"
    assert event.outcome == "conversation_invalidated"


def test_adapter_receipt_must_match_current_operation_id():
    decision = Decision("plan", steps=(_step("pause"),), evidence_quotes=("pause",))
    coordinator = ConversationCoordinator(FakeInterpreter(decision),
        lambda step, op: ToolResult("stale-operation", "completed", {}), lambda *_: None)
    result = coordinator.handle(Transcript("pause", "t9"))
    assert result.outcome == "partial_unknown_delivery"
    assert result.results[0].operation_id.startswith("t9:1:")
    assert result.results[0].operation_id.endswith(":s1")
    assert result.results[0].status == "unknown_delivery"


def test_reused_event_id_gets_distinct_application_operation_ids():
    decision = Decision("plan", steps=(_step("pause"),), evidence_quotes=("pause",))
    operation_ids = []
    coordinator = ConversationCoordinator(FakeInterpreter(decision),
        lambda step, op: operation_ids.append(op) or ToolResult(op, "completed", {}),
        lambda *_: None)
    coordinator.handle(Transcript("pause", "same-event"))
    coordinator.handle(Transcript("pause", "same-event"))
    assert len(operation_ids) == 2
    assert operation_ids[0] != operation_ids[1]


def test_invalid_typed_plan_is_rejected_before_any_execution():
    steps = tuple(_step("pause", step_id=f"s{i}") for i in range(4))
    decision = Decision("plan", steps=steps, evidence_quotes=("pause then stop",))
    called = []
    coordinator = ConversationCoordinator(FakeInterpreter(decision),
        lambda *args: called.append(args), lambda *args: called.append(args))
    result = coordinator.handle(Transcript("pause then stop", "t10"))
    assert result.decision_kind == "unsupported"
    assert result.outcome == "invalid_decision"
    assert called == []


def test_interpreter_error_after_invalidation_is_silent():
    holder = {}

    class InvalidatingFailure:
        def decide(self, transcript, context=None):
            holder["coordinator"].invalidate()
            raise RuntimeError("private transport detail")

    coordinator = ConversationCoordinator(InvalidatingFailure(), lambda *_: None, lambda *_: None)
    holder["coordinator"] = coordinator
    result = coordinator.handle(Transcript("What is this?", "t11"))
    assert result.outcome == "invalidated"
    assert result.text == ""


def test_main_conversation_flag_defaults_to_legacy_path():
    from assistant.main import Assistant

    assistant = Assistant.__new__(Assistant)
    assistant.cfg = SimpleNamespace(get=lambda key, default=None: default)
    assert assistant._handle_conversation("pause", None, SimpleNamespace()) is False


def test_conversation_construction_failure_is_handled_and_clears_adapter():
    from assistant.main import Assistant
    from assistant.telemetry import Event

    assistant = Assistant.__new__(Assistant)
    assistant.cfg = SimpleNamespace(get=lambda key, default=None: (
        True if key == "conversation.enabled" else default))
    assistant._conversation_instance = lambda: (_ for _ in ()).throw(RuntimeError("private detail"))
    assistant._conversation_recent = []
    assistant._say_feedback = lambda mic, text, event: "speech"
    event = Event(event_id="construction-fails")
    assert assistant._handle_conversation("pause", None, event)
    assert event.outcome == "conversation_interpreter_unavailable"
    assert event.error == "RuntimeError"
    assert assistant._conversation_local.session is None
    assert assistant._conversation_recent == []


def test_both_final_transcript_entry_points_call_conversation_before_legacy_routing():
    import ast
    from pathlib import Path

    tree = ast.parse(Path("assistant/main.py").read_text(encoding="utf-8"))
    methods = {node.name: node for node in ast.walk(tree)
               if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    for name in ("handle", "handle_utterance"):
        calls = [node.func.attr for node in ast.walk(methods[name])
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
        assert calls.index("_handle_conversation") < calls.index("_handle_global")
