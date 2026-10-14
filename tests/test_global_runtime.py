import contextlib
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from assistant import main
from assistant.config import Config
from assistant.global_assistant import GlobalAssistant


def session_bridge():
    bridge = Mock(active_turn=None, owned_threads=set())
    bridge.drain_events.return_value = []
    bridge.bound_model.return_value = ""
    bridge.last_created.return_value = None
    return bridge


def app_for_session(monkeypatch, segments, *, initial=""):
    app = main.Assistant.__new__(main.Assistant)
    app.cfg = Config({"codex": {"model": "gpt-6-luna"}})
    app._stop = threading.Event()
    app.vad = Mock()
    app.listener = Mock()
    app.telemetry = Mock()
    app.feedback = Mock()
    app._duck_music = lambda *_: contextlib.nullcontext()
    mic = Mock()
    mic.self_echo = ""
    mic.flush.side_effect = lambda: setattr(mic, "self_echo", "")
    def say(text, **_):
        mic.self_echo = text
        return main.KIND_SPEECH
    app.feedback.say.side_effect = say
    app._settle_after_prompt = lambda source: source.flush()
    bridge = Mock(active_turn=None, owned_threads=set())
    bridge.drain_events.return_value = []
    bridge.bound_model.return_value = ""
    bridge.last_created.return_value = None
    controller = GlobalAssistant(app.cfg, main.ROOT, bridge=bridge,
                                 navigator=lambda ident: "Requested the chat in Codex.")
    session = controller.session
    session.select("thr_voice", "Voice task")
    session.begin(initial)
    controller.submit_confirmed = Mock(return_value="Sending the confirmed prompt.")
    app.global_assistant = controller
    if initial:
        # Establish the real frontend precondition: this exact identity was
        # reviewed successfully before any confirmation recording starts.
        assert app._global_say(mic, controller.review_message(), main.Event(), review=True)
        app.feedback.say.reset_mock()
    app.stt = Mock()
    app.stt.transcribe.side_effect = [(text, 1) for text in segments]
    pointer = Mock()
    def capture(source, **_):
        assert not source.self_echo, "Prompt playback was not flushed before capture"
        return {"audio": np.zeros(512, dtype=np.int16), "speech_started": True}
    pointer.record.side_effect = capture
    monkeypatch.setattr(main, "Endpointer", Mock(return_value=pointer))
    return app, mic, pointer


def test_dictation_does_not_use_music_parser_and_sends_only_after_review(monkeypatch):
    app, mic, pointer = app_for_session(monkeypatch, ['Keep "play red" unchanged.', "Finish.", "Send it."])
    app._parse_intent = Mock(side_effect=AssertionError("Music parsing must not see dictation"))
    event = main.Event()
    app._run_global_session(mic, event)
    confirmed = app.global_assistant.submit_confirmed.call_args.args[0]
    assert confirmed.prompt == 'Keep "play red" unchanged.'
    assert confirmed.target_id == "thr_voice"
    assert confirmed.confirmation_id
    assert event.outcome == "global_dispatched"
    assert [call.kwargs["after_prompt"] for call in pointer.record.call_args_list] == [False, False, False]
    assert [call.kwargs["silence_ms"] for call in main.Endpointer.call_args_list] == [1000, 1000, 600]
    assert not app.global_assistant.session.active


def test_readback_audio_is_flushed_before_confirmation(monkeypatch):
    app, mic, pointer = app_for_session(monkeypatch, ["Read it back.", "Cancel."], initial="Say send it in an example.")
    app._run_global_session(mic, main.Event())
    assert pointer.record.call_count == 2
    assert all(not call.kwargs["after_prompt"] for call in pointer.record.call_args_list)
    app.global_assistant.submit_confirmed.assert_not_called()
    assert mic.flush.call_count >= 3


def test_finished_dictation_reads_exact_prompt_before_confirmation(monkeypatch):
    prompt = 'Check foo.py. Keep "send it" as literal text.'
    app, mic, pointer = app_for_session(monkeypatch, [prompt, "Finish.", "Send it."])
    app._run_global_session(mic, main.Event())
    spoken = [call.args[0] for call in app.feedback.say.call_args_list]
    review = next(text for text in spoken if text.startswith("Send to Voice task using gpt-6-luna:"))
    assert prompt in review and review.endswith("Ready.")
    assert spoken.index(review) < spoken.index("Sending the confirmed prompt. Ready.")
    confirmed = app.global_assistant.submit_confirmed.call_args.args[0]
    assert confirmed.prompt == prompt
    assert pointer.record.call_args_list[-1].kwargs["after_prompt"] is False


def test_silent_dictation_cancels_without_submission(monkeypatch):
    app, mic, pointer = app_for_session(monkeypatch, [])
    pointer.record.side_effect = None
    pointer.record.return_value = {"audio": np.zeros(512, dtype=np.int16), "speech_started": False}
    event = main.Event()
    app._run_global_session(mic, event)
    assert event.outcome == "global_timeout"
    app.global_assistant.submit_confirmed.assert_not_called()
    assert not app.global_assistant.session.active


def test_capture_exception_clears_draft_and_does_not_send(monkeypatch):
    app, mic, pointer = app_for_session(monkeypatch, [], initial="must not send")
    pointer.record.side_effect = RuntimeError("mic disconnected")
    with pytest.raises(RuntimeError):
        app._run_global_session(mic, main.Event())
    assert app.global_assistant.session.draft == ""
    app.global_assistant.submit_confirmed.assert_not_called()


def test_global_route_is_optional_and_never_intercepts_music():
    app = main.Assistant.__new__(main.Assistant)
    assert not app._handle_global("pause", Mock(), main.Event())
    app.global_assistant = Mock()
    app.global_assistant.handle.return_value = None
    assert not app._handle_global("play red", Mock(), main.Event())


def test_creation_wait_drains_microphone_and_returns_review_on_audio_owner(monkeypatch):
    from assistant.global_assistant import GlobalReply
    app, mic, _ = app_for_session(monkeypatch, [])
    app.global_assistant.continue_creation = Mock(side_effect=[None, GlobalReply("Exact prompt. Say send it.", True, review=True)])
    reply = app._await_global_creation(mic, main.Event(), "token")
    assert reply.session
    assert "Exact prompt" in reply.message
    mic.read.assert_called_once_with(timeout_s=0.1)
    mic.flush.assert_called()
    app.feedback.say.assert_not_called()
    app.global_assistant.submit_confirmed.assert_not_called()


def test_creation_wait_timeout_never_starts_late_capture(monkeypatch):
    ticks = iter([0.0, 0.1, 9.0])
    app, mic, pointer = app_for_session(monkeypatch, [])
    monkeypatch.setattr(main.time, "monotonic", lambda: next(ticks))
    app.global_assistant.continue_creation = Mock(return_value=None)
    event = main.Event()
    assert app._await_global_creation(mic, event, "token") is None
    assert event.outcome == "global_pending"
    assert "Ask for Codex status" in app.feedback.say.call_args.args[0]
    pointer.record.assert_not_called()
    app.global_assistant.submit_confirmed.assert_not_called()


def test_creation_wait_shutdown_does_not_speak_or_capture(monkeypatch):
    app, mic, pointer = app_for_session(monkeypatch, [])
    app._stop.set()
    app.global_assistant.continue_creation = Mock()
    assert app._await_global_creation(mic, main.Event(), "token") is None
    app.global_assistant.continue_creation.assert_not_called()
    app.feedback.say.assert_not_called()
    pointer.record.assert_not_called()


def test_global_creation_reads_back_before_opening_confirmation(monkeypatch):
    from assistant.global_assistant import GlobalReply
    app, mic, _ = app_for_session(monkeypatch, [])
    app.global_assistant.session.begin("Exact source.")
    app.global_assistant.handle = Mock(return_value=GlobalReply("Creating the chat.", await_token="token"))
    app._await_global_creation = Mock(return_value=GlobalReply("For Voice task: Exact source. Say send it.", True, review=True))
    app._run_global_session = Mock()
    event = main.Event()
    assert app._handle_global("Create a Codex chat and tell it Exact source.", mic, event)
    app._await_global_creation.assert_called_once_with(mic, event, "token")
    assert [call.args[0] for call in app.feedback.say.call_args_list] == [
        "Creating the chat. Ready.", "For Voice task: Exact source. Say send it. Ready."]
    app._run_global_session.assert_called_once_with(mic, event)


def test_dictation_records_confirmation_and_source_text(monkeypatch):
    app, mic, _ = app_for_session(monkeypatch, ['Keep punctuation unchanged?!', "Finish.", "Cancel."])
    app._run_global_session(mic, main.Event())
    records = [call.args[0] for call in app.telemetry.write.call_args_list]
    assert [record.transcript for record in records] == ['Keep punctuation unchanged?!', 'Finish.', 'Cancel.']
    assert all(record.trigger_source == "global_dictation" for record in records)
    assert all(record.global_target == "thr_voice" for record in records)
    assert records[-1].global_action == "cancel"


@pytest.mark.parametrize("ending", ["unheard", "timeout", "exception", "shutdown", "cancel", "start_again"])
def test_main_session_cleanup_restores_inferred_new_target_on_every_exit(monkeypatch, tmp_path, ending):
    from assistant.global_assistant import GlobalAssistant
    from assistant.global_commands import GlobalCommand
    segments = [""] if ending == "unheard" else ["start again"] if ending == "start_again" else ["cancel"]
    app, mic, pointer = app_for_session(monkeypatch, segments)
    bridge = session_bridge()
    router = SimpleNamespace(route=lambda text: GlobalCommand("create_ask", text, inferred=True,
                                                             target_kind="new"))
    controller = GlobalAssistant(app.cfg, tmp_path, bridge=bridge, router=router)
    controller.session.select("thr_previous", "Previous")
    source = "Make a Codex conversation to audit the folder without changing files"
    entry = controller.handle(source)
    assert entry.session and source in entry.message
    app.global_assistant = controller
    if ending == "timeout":
        pointer.record.side_effect = None
        pointer.record.return_value = {"audio": np.zeros(512, dtype=np.int16), "speech_started": False}
    elif ending == "exception":
        pointer.record.side_effect = RuntimeError("capture failed")
    elif ending == "shutdown":
        app._stop.set()
    if ending == "exception":
        with pytest.raises(RuntimeError):
            app._run_global_session(mic, main.Event())
    else:
        app._run_global_session(mic, main.Event())
    if ending in {"timeout", "unheard"}:
        assert controller.session.active
        assert controller._pending_plan is not None
        assert controller.session.target_id != "thr_previous"
        assert "I have kept your request" in app.feedback.say.call_args.args[0]
    else:
        assert controller.session.target_id == "thr_previous"
        assert controller._pending_plan is None and not controller.session.active
    bridge.create_thread.assert_not_called()
    bridge.start_turn.assert_not_called()
    bridge.queue_existing.assert_not_called()
    mic.flush.assert_called()


def test_main_create_only_acceptance_is_recorded_as_dispatch(monkeypatch):
    app, mic, _ = app_for_session(monkeypatch, ["send it"], initial="Original empty-chat request")
    app.global_assistant.submit_confirmed.return_value = "Creating the confirmed chat."
    event = main.Event()
    app._run_global_session(mic, event)
    assert event.outcome == "global_dispatched"


def test_handle_inferred_request_reads_complete_original_before_confirmation(monkeypatch, tmp_path):
    from assistant.global_assistant import GlobalAssistant
    from assistant.global_commands import GlobalCommand
    app, mic, _ = app_for_session(monkeypatch, ["cancel"])
    bridge = session_bridge()
    controller = GlobalAssistant(app.cfg, tmp_path, bridge=bridge,
                                 router=SimpleNamespace(route=lambda text: GlobalCommand("create_ask", text, inferred=True,
                                                                                       target_kind="new")))
    app.global_assistant = controller
    source = 'Make a Codex chat to audit "Downloads". Do not change or delete any files?!'
    event = main.Event()
    assert app._handle_global(source, mic, event)
    first_readback = app.feedback.say.call_args_list[0].args[0]
    assert source in first_readback and "a new Codex chat" in first_readback
    assert controller.session.target_id == ""
    bridge.create_thread.assert_not_called()
    bridge.start_turn.assert_not_called()


def test_failed_review_playback_never_opens_confirmation_capture(monkeypatch):
    from assistant.global_assistant import GlobalReply
    app, mic, pointer = app_for_session(monkeypatch, [], initial="Keep this exact request.")
    app.global_assistant.handle = Mock(return_value=GlobalReply("Review this request.", True, review=True))
    app.feedback.say.side_effect = None
    app.feedback.say.return_value = "beep_err"
    event = main.Event()
    assert app._handle_global("Ask Codex to review", mic, event)
    pointer.record.assert_not_called()
    assert event.outcome == "global_feedback_failed"
    assert not app.global_assistant.session.active
    app.global_assistant.submit_confirmed.assert_not_called()


def test_neutral_tail_cannot_confirm_and_real_send_remains_possible(monkeypatch):
    app, mic, pointer = app_for_session(monkeypatch, ["Ready.", "Send it."], initial="Exact request")
    event = main.Event()
    app._run_global_session(mic, event)
    records = [call.args[0] for call in app.telemetry.write.call_args_list]
    assert records[0].global_action == "error"
    assert records[1].global_action == "submit"
    assert app.global_assistant.submit_confirmed.call_count == 1
    assert app.global_assistant.submit_confirmed.call_args.args[0].prompt == "Exact request"
    assert all(call.args[0].endswith("Ready.") for call in app.feedback.say.call_args_list)
    assert all(call.kwargs["after_prompt"] is False for call in pointer.record.call_args_list)


@pytest.mark.parametrize("unheard", [False, True])
def test_quiet_review_preserves_plan_and_next_wake_resumes(monkeypatch, tmp_path, unheard):
    from assistant.global_assistant import GlobalAssistant
    from assistant.global_commands import GlobalCommand
    app, mic, pointer = app_for_session(monkeypatch, [""] if unheard else [])
    bridge = session_bridge()
    controller = GlobalAssistant(app.cfg, tmp_path, bridge=bridge,
        router=SimpleNamespace(route=lambda text: GlobalCommand("ask", text, inferred=True,
                                                                target_kind="selected")))
    controller.session.select("thr_previous", "Previous")
    source = "Have Codex explain this exact request"
    entry = controller.handle(source)
    app.global_assistant = controller
    expected_draft = controller.session.draft
    assert controller._pending_plan.source == source
    assert source in expected_draft
    assert app._global_say(mic, entry.message, main.Event(), review=True)
    app.feedback.say.reset_mock()
    pointer.record.side_effect = None
    pointer.record.return_value = {"audio": np.zeros(512, dtype=np.int16),
                                   "speech_started": unheard}
    event = main.Event()
    app._run_global_session(mic, event)
    assert event.outcome == "global_review_waiting"
    assert controller.session.active and controller._pending_plan is not None
    assert controller.session.draft == expected_draft
    app.feedback.say.assert_called_once()
    assert "I have kept your request. Say Alexa to continue." in app.feedback.say.call_args.args[0]
    app._run_global_session = Mock()
    assert app._handle_global("", mic, main.Event())
    app._run_global_session.assert_called_once()
    assert app._run_global_session.call_args.args[0] is mic
    assert controller.session.draft == expected_draft
    bridge.start_turn.assert_not_called()


def test_global_capture_builds_full_wait_plus_utterance_budget(monkeypatch):
    app, mic, _ = app_for_session(monkeypatch, ["Cancel."], initial="Exact request")
    app._run_global_session(mic, main.Event())
    kwargs = main.Endpointer.call_args.kwargs
    assert kwargs["speech_timeout_ms"] == 15000
    assert kwargs["utterance_seconds"] == 20
    assert kwargs["max_seconds"] == 35
    assert kwargs["silence_ms"] == 600


def test_bare_alexa_resumes_retained_review_before_music_capture(monkeypatch):
    from assistant.global_assistant import GlobalReply
    app, mic, _ = app_for_session(monkeypatch, [], initial="Exact request")
    app.global_assistant.handle = Mock(return_value=GlobalReply("Exact request. Say send it.", True, review=True))
    app._run_global_session = Mock()
    app.endpointer = Mock()
    app.spotify = Mock()
    app._early_lease = None
    app._open_timeline = Mock()
    utt = SimpleNamespace(command="", matched_phrase="alexa", transcript="Alexa",
                          latency_ms=1, audio=np.zeros(512, dtype=np.int16))
    app.handle_utterance(mic, utt)
    app.global_assistant.handle.assert_called_once_with("")
    app._run_global_session.assert_called_once()
    app.endpointer.record.assert_not_called()


def test_feedback_clock_preserves_long_review_and_records_actual_timings(monkeypatch):
    app, mic, _ = app_for_session(monkeypatch, [], initial="Exact request")
    clock = [0.0]
    session = app.global_assistant.session
    session.clock = lambda: clock[0]
    session.begin("Exact request")
    before = session.deadline
    def speak(*_, **__):
        clock[0] += 95
        return main.KIND_SPEECH
    app.feedback.say.side_effect = speak
    event = main.Event()
    assert app._global_say(mic, "Review this request.", event)
    assert session.active and session.deadline == before + 95
    assert event.feedback_latency_ms is not None
    assert "feedback_start" in event.timeline and "feedback_done" in event.timeline
    assert "feedback_settled" in event.timeline


def test_failed_clarification_playback_does_not_listen(monkeypatch):
    app, mic, _ = app_for_session(monkeypatch, [])
    app.endpointer = Mock()
    app.feedback.say.side_effect = None
    app.feedback.say.return_value = "beep_err"
    text, captured, latency = app._ask_and_listen(mic, "Which artist?")
    assert text == "" and latency == 0 and not captured["speech_started"]
    app.endpointer.record.assert_not_called()


def test_busy_music_work_discards_intentionally_and_unwinds_on_error(monkeypatch):
    from assistant.audio import Microphone
    app, _, _ = app_for_session(monkeypatch, [])
    app.spotify = Mock()
    app.spotify.ensure_client_running.return_value = False
    mic = Microphone()
    def slow_pause():
        for _ in range(600):
            mic._callback(np.ones((512, 1), dtype=np.int16), 512, None, None)
        raise RuntimeError("pause failed")
    app.spotify.pause.side_effect = slow_pause
    with pytest.raises(RuntimeError, match="pause failed"):
        app._execute(main.Intent(main.Action.PAUSE), main.Event(), mic=mic)
    assert mic.intentional_discard_count == 600
    assert mic.queue_overflow_count == 0 and mic._discard_depth == 0
    mic._callback(np.ones((512, 1), dtype=np.int16), 512, None, None)
    assert np.any(mic.read())


def test_idle_notice_waits_for_quiet_and_never_speaks_during_capture(monkeypatch):
    app, mic, _ = app_for_session(monkeypatch, [])
    app.global_assistant.session.reset()
    app.global_assistant.take_notice = Mock(return_value="The Codex request completed.")
    app._busy = threading.Lock()
    app._trigger = threading.Event()
    app._paused_mic = None
    app.wake = Mock()
    app.listener._in_speech = False
    app.listener.last_vad_probability = 0.0
    clock = [0.0]
    monkeypatch.setattr(main.time, "monotonic", lambda: clock[0])
    assert not app._speak_global_notice_if_quiet(mic)
    clock[0] = 1.1
    app.listener._in_speech = True
    assert not app._speak_global_notice_if_quiet(mic)
    app.global_assistant.take_notice.assert_not_called()
    app.listener._in_speech = False
    assert not app._speak_global_notice_if_quiet(mic)
    clock[0] = 2.2
    assert app._speak_global_notice_if_quiet(mic)
    app.global_assistant.take_notice.assert_called_once()
    app.wake.reset.assert_called_once()
    assert app.feedback.say.call_count == 1


@pytest.mark.parametrize("command", ["Send it.", "Cancel."])
def test_acoustic_review_captures_buffered_inline_control_before_any_readback(monkeypatch, command):
    from assistant.audio import Microphone
    app, _, _ = app_for_session(monkeypatch, [], initial="Exact pending request")
    mic = Microphone()
    human = np.full((512, 1), 7000, dtype=np.int16)
    mic._callback(human, 512, None, None)
    app.wake = Mock(last_score=0.9)
    app.spotify = Mock()
    app.listener = None
    app._early_lease = None
    app._open_timeline = Mock()
    app._acknowledge = Mock(return_value="beep_wake")
    app.feedback.speaks_on_wake = False
    sequence = []
    app.endpointer = Mock()
    def capture(source, **_):
        sequence.append("capture")
        audio = source.read()
        np.testing.assert_array_equal(audio, human[:, 0])
        return {"audio": audio, "speech_started": True, "duration_ms": 32,
                "speech_start_ms": 0, "speech_end_ms": 32}
    app.endpointer.record.side_effect = capture
    app.stt.transcribe.side_effect = lambda audio: (command, 1)
    def route(text, source, event):
        assert text == command
        sequence.append("control")
        event.outcome = "global_cancelled" if command == "Cancel." else "global_handled"
        return True
    app._handle_global = Mock(side_effect=route)
    app.handle(mic, "wake", main.TIER_BOTH)
    assert sequence == ["capture", "control"]
    app._handle_global.assert_called_once()
    app.feedback.say.assert_not_called()
    assert mic.intentional_discard_count == 0


def test_unspoken_pending_review_reads_back_before_send_can_dispatch(monkeypatch, tmp_path):
    from assistant.global_assistant import GlobalAssistant
    from assistant.global_commands import GlobalCommand
    app, mic, pointer = app_for_session(monkeypatch, ["Cancel."])
    bridge = session_bridge()
    controller = GlobalAssistant(app.cfg, tmp_path, bridge=bridge,
        router=SimpleNamespace(route=lambda text: GlobalCommand("ask", text, inferred=True,
                                                                target_kind="selected")))
    controller.session.select("thr_previous", "Previous")
    source = "Have Codex explain this exact request"
    assert controller.handle(source).session
    app.global_assistant = controller
    event = main.Event()
    assert app._handle_global("Send it.", mic, event)
    assert pointer.record.call_count == 1
    assert source in app.feedback.say.call_args_list[0].args[0]
    bridge.start_turn.assert_not_called()
    bridge.queue_existing.assert_not_called()
    assert not controller.session.active


def test_error_and_notice_feedback_do_not_arm_an_unspoken_plan(monkeypatch, tmp_path):
    from assistant.global_assistant import GlobalAssistant
    from assistant.global_commands import GlobalCommand
    app, mic, _ = app_for_session(monkeypatch, [])
    bridge = session_bridge()
    controller = GlobalAssistant(app.cfg, tmp_path, bridge=bridge,
        router=SimpleNamespace(route=lambda text: GlobalCommand("ask", text, inferred=True,
                                                                target_kind="selected")))
    controller.session.select("thr_previous", "Previous")
    assert controller.handle("Have Codex explain this exact request").session
    app.global_assistant = controller
    assert app._global_say(mic, "I have kept your request.", main.Event())
    assert controller.consume("Send it.").action == "review"
    assert app._global_say(mic, controller.review_message(), main.Event(), review=True)
    reply = controller.consume("Send it.")
    assert reply.action == "submit" and reply.confirmation_id
    bridge.start_turn.assert_not_called()
    bridge.queue_existing.assert_not_called()


def test_blocking_list_busy_feedback_cannot_arm_an_unspoken_review(monkeypatch, tmp_path):
    from assistant.global_commands import GlobalCommand
    app, mic, _ = app_for_session(monkeypatch, [])
    bridge = session_bridge()
    started, release = threading.Event(), threading.Event()
    def list_threads():
        started.set()
        assert release.wait(timeout=5), "Test did not release its mocked list operation"
        return []
    bridge.list_threads.side_effect = list_threads
    controller = GlobalAssistant(app.cfg, tmp_path, bridge=bridge,
        router=SimpleNamespace(route=lambda text: GlobalCommand("ask", text, inferred=True,
                                                                target_kind="selected")))
    controller.session.select("thr_previous", "Previous")
    source = "Have Codex explain this exact request"
    assert controller.handle(source).session
    app.global_assistant = controller
    app._run_global_session = Mock()
    try:
        assert not controller.handle("List Codex chats.").review
        assert started.wait(timeout=5)
        assert app._handle_global("Send it.", mic, main.Event())
        busy_speech = app.feedback.say.call_args.args[0]
        assert "already running" in busy_speech and source not in busy_speech
        assert controller._review_armed is None
        bridge.start_turn.assert_not_called()
    finally:
        release.set()
    result = controller._results.get(timeout=5)
    controller._results.put_nowait(result)
    controller.poll()
    assert controller.consume("Send it.").action == "review"
    assert app._handle_global("Send it.", mic, main.Event())
    assert source in app.feedback.say.call_args.args[0]
    assert controller.consume("Send it.").action == "submit"
    bridge.start_turn.assert_not_called()
    bridge.queue_existing.assert_not_called()
