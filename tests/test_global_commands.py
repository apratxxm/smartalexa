from assistant.global_commands import GlobalCommand, VoiceSession, parse_global


def test_global_parser_is_anchored_and_preserves_prompt_text():
    assert parse_global("ask Codex Fix THIS, please!") == GlobalCommand(
        "ask", "Fix THIS, please!"
    )
    assert parse_global("he said ask Codex Fix this") is None
    assert parse_global("play ask Codex") is None


def test_global_parser_chat_and_tab_commands():
    assert parse_global("Create a new Codex chat.") == GlobalCommand("create")
    assert parse_global("use Luna for this Codex chat.") == GlobalCommand("use_luna")
    assert parse_global("new Codex chat called Morning") == GlobalCommand(
        "create", "Morning"
    )
    assert parse_global("list Codex chats") == GlobalCommand("list")
    assert parse_global("switch to Codex chat current") == GlobalCommand(
        "select", "current"
    )
    assert parse_global("previous browser tab") == GlobalCommand(
        "switch_tab", direction="previous"
    )


def test_unquoted_title_connectors_cannot_swallow_a_followon_task():
    for source in [
        "Create a Codex chat called Build Notes and check the failing tests",
        "Create a Codex chat called Build Notes then inspect the results",
        "Open Codex chat Build Notes to review the failing tests",
    ]:
        assert parse_global(source) is None
    assert parse_global('Create a Codex chat called "Research and Development"') == GlobalCommand(
        "create", "Research and Development")
    assert parse_global('Open Codex chat "Research and Development"') == GlobalCommand(
        "open", "Research and Development")


def test_short_global_controls_accept_terminal_asr_punctuation_only():
    assert parse_global("Codex status.") == GlobalCommand("status")
    assert parse_global("Create a Codex chat!") == GlobalCommand("create")
    assert parse_global("next browser tab?") == GlobalCommand(
        "switch_tab", direction="next"
    )
    command = parse_global('Ask Codex Fix foo.py. Keep "play red" unchanged.')
    assert command == GlobalCommand(
        "ask", 'Fix foo.py. Keep "play red" unchanged.'
    )
    assert parse_global("review Codex approval.") == GlobalCommand("review_approval")
    assert parse_global("approve Codex action!") == GlobalCommand("approve")
    assert parse_global("deny Codex action?") == GlobalCommand("deny")
    assert parse_global("dictate Fix foo.py. for Codex.") == GlobalCommand(
        "dictate", "Fix foo.py."
    )
    assert parse_global('"approve Codex action"') is None
    assert parse_global("please approve Codex action") is None


def test_explicit_created_chat_navigation_is_complete_and_never_consumes_a_task():
    assert parse_global("Open the chat you just made.") == GlobalCommand("open", "last_created")
    assert parse_global("Show the last created Codex chat") == GlobalCommand("open", "last_created")
    assert parse_global("He said open the chat you just made") is None
    assert parse_global("Open the chat you just made and audit the logs") is None
    assert parse_global("Open Codex chat Build and review the logs") is None


def test_creation_model_preserves_the_requested_version_and_rejects_conflicts():
    import pytest
    from assistant.global_commands import creation_model
    assert creation_model("Make a new Codex chat. Use GPT-6 Luna.") == "gpt-6-luna"
    assert creation_model("Create a chat using GPT-5.6 Luna") == "gpt-5.6-luna"
    assert creation_model("Create a chat using GPT-6 Astra") == "gpt-6-astra"
    with pytest.raises(ValueError):
        creation_model("Use GPT-6 Luna and use GPT-5.6 Luna")
    with pytest.raises(ValueError):
        creation_model("Create a chat. Do not use GPT-6 Luna")
    assert creation_model("Create a chat. Use GPT-6") == "gpt-6"


def test_session_requires_target_and_invalidates_draft_on_target_change():
    session = VoiceSession(clock=lambda: 0)
    assert session.begin().action == "error"
    session.select("thr_one", "Build chat")
    assert session.begin().action == "dictating"
    session.consume("keep this")
    session.select("thr_two", "Other chat")
    assert session.draft == ""
    assert not session.active


def test_session_reviews_and_submits_exact_draft():
    now = [0.0]
    session = VoiceSession(clock=lambda: now[0])
    session.select("thr_one", "Build chat")
    session.begin()
    session.consume("Fix THIS")
    session.consume("please send it")  # embedded control words are payload
    session.consume("finish")
    assert session.phase == "review"
    response = session.consume("send the prompt")
    assert response.action == "submit"
    assert response.prompt == "Fix THIS please send it"
    assert response.target_id == "thr_one"
    assert session.target_id == "thr_one"
    assert session.draft == ""


def test_session_timeout_and_limits_never_submit():
    now = [0.0]
    session = VoiceSession(clock=lambda: now[0], timeout_seconds=2, max_chars=5)
    session.select("thr_one", "Build chat")
    session.begin()
    assert session.consume("123456").action == "limit"
    assert session.consume("send it").action in {"expired", "error"}

    session.begin()
    session.consume("ok")
    now[0] = 3
    assert not session.active
    assert session.consume("finish").action == "expired"


def test_review_rejects_unrecognized_speech_and_empty_finish():
    session = VoiceSession(clock=lambda: 0)
    session.select("thr_one", "Build chat")
    session.begin()
    assert session.consume("finish").action == "error"
    session.consume("prompt")
    session.consume("finish")
    response = session.consume("yes")
    assert response.action == "error"
    assert "send it" in response.message


def test_session_controls_accept_terminal_punctuation_but_not_quotes():
    session = VoiceSession(clock=lambda: 0)
    session.select("thr_one", "Build chat")
    session.begin()
    session.consume("A quoted phrase")
    assert session.consume("Finish.").action == "review"
    assert session.consume('"send it"').action == "error"
    assert session.phase == "review"
    assert session.consume("Send it.").action == "submit"


def test_session_control_words_inside_dictation_remain_payload():
    session = VoiceSession(clock=lambda: 0)
    session.select("thr_one", "Build chat")
    session.begin()
    session.consume("Please write code that says send it.")
    assert session.draft == "Please write code that says send it."


def test_submit_confirmation_is_bound_and_one_time():
    session = VoiceSession(clock=lambda: 0)
    session.select("thr_one", "One")
    session.begin()
    session.consume("exact payload")
    session.consume("finish")
    reply = session.consume("send it")
    assert reply.confirmation_id
    assert session.take_confirmation(reply) is True
    assert session.take_confirmation(reply) is False


def test_confirmation_deadline_accessor_does_not_consume_or_regrant_token():
    now = [0.0]
    session = VoiceSession(clock=lambda: now[0], timeout_seconds=90)
    session.select("thr_one", "One")
    session.begin("exact payload")
    now[0] = 89
    reply = session.consume("send it")
    assert session.confirmation_expires_at == 90
    assert session.confirmation_expires_at == 90
    assert session.take_confirmation(reply)
    assert session.confirmation_expires_at is None


def test_forged_confirmation_does_not_consume_valid_one():
    session = VoiceSession(clock=lambda: 0)
    session.select("thr_one", "One")
    session.begin("exact payload")
    reply = session.consume("send it")
    forged = type(reply)(
        reply.action,
        prompt="changed payload",
        target_id=reply.target_id,
        confirmation_id=reply.confirmation_id,
    )
    assert session.take_confirmation(forged) is False
    assert session.take_confirmation(reply) is True


def test_confirmation_expires_and_clear_or_target_change_invalidates_it():
    now = [0.0]
    session = VoiceSession(clock=lambda: now[0], timeout_seconds=2)
    session.select("thr_one", "One")
    session.begin("payload")
    reply = session.consume("send it")
    now[0] = 2
    assert session.take_confirmation(reply) is False

    session.begin("payload")
    reply = session.consume("send it")
    session.reset()
    assert session.take_confirmation(reply) is False

    session.begin("payload")
    reply = session.consume("send it")
    session.select("thr_two", "Two")
    assert session.take_confirmation(reply) is False


def test_same_target_selection_does_not_mutate_pending_confirmation():
    session = VoiceSession(clock=lambda: 0)
    session.select("thr_one", "One")
    session.begin("payload")
    reply = session.consume("send it")
    session.select("thr_one", "One")
    assert session.take_confirmation(reply) is True


def test_prompt_bound_and_segment_limits_have_hard_bounds():
    import pytest

    with pytest.raises(ValueError):
        VoiceSession(max_chars=12001)
    with pytest.raises(ValueError):
        VoiceSession(max_segments=31)


def test_unbound_legacy_start_again_keeps_original_session_deadline():
    now = [0.0]
    session = VoiceSession(clock=lambda: now[0], timeout_seconds=2)
    session.select("thr_one", "One")
    session.begin("original")
    now[0] = 1.5
    assert session.consume("start again").action == "dictating"
    now[0] = 2.0
    assert not session.active
    assert session.consume("send it").action != "submit"


def test_feedback_elapsed_extends_reply_budget_without_dispatch():
    now = [0.0]
    session = VoiceSession(clock=lambda: now[0], timeout_seconds=10)
    session.select("thr_one", "One")
    session.begin("Exact draft")
    token = session.begin_feedback()
    now[0] = 35
    assert session.active and session.deadline == 45
    assert session.consume("send it").action == "error"
    assert session.end_feedback(token, True)
    assert session.deadline == 45
    assert not session.end_feedback(token, True)
    now[0] = 44
    assert session.active
    now[0] = 45
    assert not session.active


def test_feedback_hard_cap_remains_absolute_during_speech():
    now = [0.0]
    session = VoiceSession(clock=lambda: now[0], timeout_seconds=90)
    session.select("thr_one", "One")
    session.begin("Exact draft")
    token = session.begin_feedback()
    now[0] = 269
    assert session.active and session.deadline == 270
    now[0] = 270
    assert not session.active
    assert not session.end_feedback(token, True)
    assert session.draft == ""


def test_feedback_failure_target_change_and_reset_cannot_resurrect_draft():
    for mode in ["failure", "select", "reset"]:
        session = VoiceSession(clock=lambda: 0)
        session.select("thr_one", "One")
        session.begin("Never dispatch")
        token = session.begin_feedback()
        if mode == "select":
            session.select("thr_two", "Two")
        elif mode == "reset":
            session.reset()
        assert not session.end_feedback(token, mode != "failure")
        assert not session.active and not session.draft
