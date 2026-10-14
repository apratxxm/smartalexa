import logging
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from assistant.global_assistant import GlobalAssistant, GlobalReply
from assistant.global_commands import GlobalCommand, VoiceSession, parse_global, _control_text


class FakeBridge:
    def __init__(self):
        self.owned_threads = set()
        self.active_turn = None
        self.calls = []
        self.create_started = threading.Event()
        self.create_done = threading.Event()
        self.create_release = threading.Event()
        self.block_create = False
        self.list_started = threading.Event()
        self.list_release = threading.Event()
        self.block_list = False
        self.submit_started = threading.Event()
        self.submit_done = threading.Event()
        self.cancel_done = threading.Event()
        self.submit_release = threading.Event()
        self.block_submit = False

    def drain_events(self):
        self.calls.append(("drain_events",))
        return []

    def create_thread(self, title, *, model=None):
        self.calls.append(("create", title) if model is None else ("create", title, model))
        self.create_started.set()
        if self.block_create:
            assert self.create_release.wait(2)
        result = {"id": "thr_created", "name": title}
        if model:
            result["model"] = model
        self.create_done.set()
        return result

    def list_threads(self):
        self.calls.append(("list",))
        self.list_started.set()
        if self.block_list:
            assert self.list_release.wait(2)
        return []

    def start_turn(self, thread_id, prompt):
        self.calls.append(("start", thread_id, prompt))
        self.submit_started.set()
        if self.block_submit:
            assert self.submit_release.wait(2)
        result = {"status": "accepted"}
        self.submit_done.set()
        return result

    def queue_existing(self, thread_id, prompt):
        self.calls.append(("queue", thread_id, prompt))
        self.submit_done.set()
        return {"status": "queued"}

    def cancel_turn(self):
        self.calls.append(("cancel",))
        self.cancel_done.set()
        return True

    def close(self):
        self.calls.append(("close",))


def make_assistant(tmp_path, bridge=None, navigator=None, tabs=None):
    cfg = {
        "codex.model": "gpt-5.6-luna",
        "codex.dictation_timeout_seconds": 90,
        "codex.max_prompt_chars": 12000,
        "codex.job_timeout_seconds": 900,
    }
    assistant = GlobalAssistant(cfg, Path(tmp_path), bridge=bridge or FakeBridge(),
                                navigator=navigator or (lambda ident: "Requested the chat in Codex."), tabs=tabs)
    # These unit setups model completed frontend readback before direct send.
    # Unspoken-review tests explicitly use VoiceSession.consume to bypass this.
    consume = assistant.session.consume
    def reviewed_consume(text):
        if (assistant.session.phase == "review" and _control_text(text) in
                {"send it", "send the prompt", "submit prompt", "sendit", "sendtheprompt", "submitprompt"}
                and assistant.session.active and assistant._review_armed != assistant._review_identity()):
            token = assistant.begin_feedback(review=True)
            assert token and assistant.end_feedback(token, True)
        return consume(text)
    assistant.session.consume = reviewed_consume
    return assistant


def finish_prompt(assistant, text="Fix the bug"):
    assistant.session.select("thr_target", "Target")
    assistant.session.begin()
    assistant.session.consume(text)
    assistant.session.consume("finish")
    return assistant.session.consume("send it")


def test_nonmusic_text_is_not_routed(tmp_path):
    bridge = FakeBridge()
    assistant = make_assistant(tmp_path, bridge)
    assert assistant.handle("play red") is None
    assert [call for call in bridge.calls if call[0] not in {"drain_events"}] == []


def test_inline_ask_only_arms_review_until_explicit_send(tmp_path):
    bridge = FakeBridge()
    bridge.owned_threads.add("thr_target")
    assistant = make_assistant(tmp_path, bridge)
    assistant.session.select("thr_target", "Target")
    reply = assistant.handle("Ask Codex Fix foo.py. Keep PLAY unchanged.")
    assert reply.session is True
    assert assistant.session.phase == "review"
    assert not any(call[0] in {"start", "queue"} for call in bridge.calls)
    candidate = assistant.session.consume("Send it.")
    assert candidate.action == "submit"
    assert candidate.prompt == "Fix foo.py. Keep PLAY unchanged."
    assistant.submit_confirmed(candidate)
    assert bridge.submit_started.wait(2)
    assert ("start", "thr_target", "Fix foo.py. Keep PLAY unchanged.") in bridge.calls


def test_submit_target_and_exact_payload_are_bound(tmp_path):
    bridge = FakeBridge()
    bridge.owned_threads.add("thr_target")
    assistant = make_assistant(tmp_path, bridge)
    candidate = finish_prompt(assistant, 'Fix foo.py. Keep "play red" unchanged.')
    assert candidate.prompt == 'Fix foo.py. Keep "play red" unchanged.'
    assert candidate.target_id == "thr_target"
    assistant.submit_confirmed(candidate)
    assert bridge.submit_started.wait(2)
    assert ("start", "thr_target", candidate.prompt) in bridge.calls


def test_create_is_queued_async_and_does_not_block_audio_owner(tmp_path):
    bridge = FakeBridge()
    bridge.block_create = True
    assistant = make_assistant(tmp_path, bridge)
    started = time.monotonic()
    reply = assistant.handle("Create a Codex chat.")
    elapsed = time.monotonic() - started
    assert reply.session is True
    assert elapsed < 1.0
    assert not bridge.create_started.is_set()
    assistant.submit_confirmed(assistant.session.consume("send it"))
    assert bridge.create_started.wait(2)
    assert assistant.session.target_id == ""
    bridge.create_release.set()
    assert bridge.create_done.wait(2)
    await_worker(assistant)


def test_poll_selects_created_chat_for_audio_owner(tmp_path):
    bridge = FakeBridge()
    assistant = make_assistant(tmp_path, bridge)
    assistant.handle("create Codex chat called Build")
    assistant.submit_confirmed(assistant.session.consume("send it"))
    assert bridge.create_started.wait(2)
    assert bridge.create_done.wait(2)
    await_worker(assistant)
    assert assistant.session.target_id == "thr_created"
    assert assistant.session.target_name == "Build"


def test_one_pending_send_rejects_a_second_send(tmp_path):
    bridge = FakeBridge()
    bridge.owned_threads.add("thr_target")
    bridge.block_submit = True
    assistant = make_assistant(tmp_path, bridge)
    first = finish_prompt(assistant, "first")
    assert "Sending" in assistant.submit_confirmed(first)
    assert bridge.submit_started.wait(2)
    second = assistant.submit_confirmed(first)
    assert "nothing sent" in second.lower()
    bridge.submit_release.set()


def test_nonowned_target_queues_existing_instead_of_resuming(tmp_path):
    bridge = FakeBridge()
    assistant = make_assistant(tmp_path, bridge)
    candidate = finish_prompt(assistant)
    assistant.submit_confirmed(candidate)
    assert bridge.submit_done.wait(2)
    assert not any(call[0] == "start" for call in bridge.calls)


def test_submit_confirmation_cannot_be_replayed_or_forged(tmp_path):
    bridge = FakeBridge()
    bridge.owned_threads.add("thr_target")
    assistant = make_assistant(tmp_path, bridge)
    candidate = finish_prompt(assistant, "once only")
    assistant.submit_confirmed(candidate)
    assert bridge.submit_started.wait(2)
    bridge.submit_release.set()
    assert bridge.submit_done.wait(2)
    await_worker(assistant)
    assert "expired" in assistant.submit_confirmed(candidate).lower()
    forged = replace(candidate, prompt="modified")
    assert "expired" in assistant.submit_confirmed(forged).lower()
    assert len([call for call in bridge.calls if call[0] == "start"]) == 1


def test_busy_retry_cannot_resubmit_consumed_confirmation(tmp_path):
    bridge = FakeBridge()
    bridge.owned_threads.add("thr_target")
    bridge.block_submit = True
    assistant = make_assistant(tmp_path, bridge)
    candidate = finish_prompt(assistant, "busy once")
    assistant.submit_confirmed(candidate)
    assert bridge.submit_started.wait(2)
    assert "nothing sent" in assistant.submit_confirmed(candidate).lower()
    bridge.submit_release.set()
    assert bridge.submit_done.wait(2)
    assert len([call for call in bridge.calls if call[0] == "start"]) == 1


def test_selecting_another_target_invalidates_pending_draft(tmp_path):
    assistant = make_assistant(tmp_path)
    assistant.session.select("thr_one", "One")
    assistant.session.begin()
    assistant.session.consume("private draft")
    assistant.session.select("thr_two", "Two")
    assert assistant.session.draft == ""
    assert assistant.session.phase == "idle"


def test_status_uses_cached_notice_without_bridge_rpc(tmp_path):
    bridge = FakeBridge()
    assistant = make_assistant(tmp_path, bridge)
    bridge.calls.clear()
    assistant._notice = "Cached status"
    reply = assistant.handle("Codex status.")
    assert reply.message == "Cached status"
    assert not any(call[0] in {"create", "list", "start", "queue", "cancel"} for call in bridge.calls)


def test_cancel_pending_create_is_not_treated_as_coding_job(tmp_path):
    bridge = FakeBridge()
    bridge.block_create = True
    assistant = make_assistant(tmp_path, bridge)
    assistant.handle("create Codex chat")
    assistant.submit_confirmed(assistant.session.consume("send it"))
    assert bridge.create_started.wait(2)
    response = assistant.handle("cancel Codex task")
    assert "Cancellation requested" in response.message
    assert not any(call[0] == "cancel" for call in bridge.calls)
    bridge.create_release.set()
    assert bridge.create_done.wait(2)
    await_worker(assistant)
    assert not bridge.cancel_done.is_set()


def test_cancel_pending_submit_but_not_list(tmp_path):
    bridge = FakeBridge()
    bridge.owned_threads.add("thr_target")
    bridge.block_submit = True
    assistant = make_assistant(tmp_path, bridge)
    candidate = finish_prompt(assistant)
    assistant.submit_confirmed(candidate)
    assert bridge.submit_started.wait(2)
    response = assistant.handle("stop Codex")
    assert "pending" in response.message
    bridge.submit_release.set()
    assert bridge.submit_done.wait(2)
    await_worker(assistant)
    assert bridge.cancel_done.wait(2)

    other = FakeBridge()
    other.block_list = True
    second = make_assistant(tmp_path, other)
    second.handle("list Codex chats")
    assert other.list_started.wait(2)
    second.handle("cancel Codex")
    assert not any(call[0] == "cancel" for call in other.calls)
    other.list_release.set()


class CompoundRouter:
    def route(self, text):
        return GlobalCommand("create_ask", "check out my Gmail files")


def compound_assistant(tmp_path, bridge):
    assistant = make_assistant(tmp_path, bridge)
    assistant._router = CompoundRouter()
    return assistant


def drain_creation(assistant, bridge):
    if assistant.session.phase == "review" and not assistant._busy:
        return
    assert bridge.create_done.wait(2)
    # create_done precedes worker queue insertion; drain until completion.
    deadline = time.monotonic() + 2
    while assistant._busy and time.monotonic() < deadline:
        assistant.poll()
        time.sleep(.001)
    assert not assistant._busy


def test_compound_request_creates_then_stages_until_one_explicit_send(tmp_path):
    bridge = FakeBridge()
    assistant = compound_assistant(tmp_path, bridge)
    assistant.handle("Open a Codex chat and tell it to check out my Gmail files")
    drain_creation(assistant, bridge)
    assert assistant.session.draft == "check out my Gmail files"
    assert assistant.session.phase == "review"
    assert not any(c[0] in {"start", "queue"} for c in bridge.calls)
    reply = assistant.handle("Codex status")
    assert reply.session
    assert "check out my Gmail files" in reply.message
    assert "Say send it, start again, or cancel" in reply.message
    candidate = assistant.session.consume("send it")
    assistant.submit_confirmed(candidate)
    assert bridge.submit_done.wait(2)
    assert "Nothing sent" in assistant.submit_confirmed(candidate)


def test_compound_title_not_swallowed_by_deterministic_create():
    assert parse_global("create Codex chat called Build and tell it Fix the bug") is None


def test_cancellation_discards_canonical_prompt_before_creating_any_chat(tmp_path):
    bridge = FakeBridge()
    bridge.block_create = True
    assistant = compound_assistant(tmp_path, bridge)
    assistant.handle("Open a Codex chat and tell it to check out my Gmail files")
    assert not bridge.create_started.is_set()
    assistant.handle("cancel Codex")
    bridge.create_release.set()
    assert not assistant.session.active
    assert not assistant.handle("Codex status").session
    assert assistant.session.draft == ""
    assert writes(bridge) == []


def test_target_change_invalidates_pending_creation_even_changed_back(tmp_path):
    bridge = FakeBridge()
    bridge.block_create = True
    assistant = compound_assistant(tmp_path, bridge)
    assistant.session.select("thr_one", "One")
    assistant.handle("Open a Codex chat and tell it to check out my Gmail files")
    assert not bridge.create_started.is_set()
    assistant.session.select("thr_two", "Two")
    assistant.session.select("thr_one", "One")
    bridge.create_release.set()
    assert not assistant.session.active
    assert assistant.session.target_id == "thr_one"
    assert writes(bridge) == []


def test_semantic_failures_do_not_become_music(tmp_path):
    assistant = make_assistant(tmp_path)
    assert assistant.handle("Open a Codex chat and tell it to check out my Gmail files") is not None
    assistant._router = SimpleNamespace(route=lambda _: None)
    assert assistant.handle("He said open Codex chat") is None


def test_inline_ask_without_target_stages_after_creation(tmp_path):
    bridge = FakeBridge()
    assistant = make_assistant(tmp_path, bridge)
    assistant.handle("Ask Codex send it")
    drain_creation(assistant, bridge)
    assert assistant.session.draft == "send it"
    assert assistant.session.phase == "review"
    assert not any(c[0] in {"start", "queue"} for c in bridge.calls)


def test_cancelled_or_expired_staged_review_not_repeated_on_status(tmp_path):
    for mode in ["cancel", "expire"]:
        bridge = FakeBridge()
        assistant = compound_assistant(tmp_path, bridge)
        assistant.handle("Open a Codex chat and tell it to check out my Gmail files")
        drain_creation(assistant, bridge)
        if mode == "cancel":
            assistant.session.consume("cancel")
        else:
            assistant.session._deadline = 0
        status = assistant.handle("Codex status")
        assert not status.session
        assert "send it" not in status.message
        assert "No prompt awaiting confirmation" in status.message


def test_pending_creation_expires_and_explicit_selection_discards_draft(tmp_path):
    for mode in ["expire", "select"]:
        bridge = FakeBridge()
        bridge.block_create = True
        assistant = compound_assistant(tmp_path, bridge)
        assistant.handle("Open a Codex chat and tell it to check out my Gmail files")
        assert not bridge.create_started.is_set()
        if mode == "expire":
            assistant._pending_plan = replace(assistant._pending_plan, expires_at=0)
            assistant.poll()
        else:
            assistant.session.select("thr_other", "Other")
        bridge.create_release.set()
        assert not assistant.session.active
        assert not assistant.handle("Codex status").session
        assert writes(bridge) == []


def with_successful_frontend_review(assistant):
    """Simulate the frontend reading actual review, never generic status cues."""
    handle = assistant.handle
    continuation = assistant.continue_creation
    def reviewed(reply):
        if reply and reply.review and reply.session and assistant.session.phase == "review":
            token = assistant.begin_feedback(review=True)
            assert token and assistant.end_feedback(token, True)
        return reply
    assistant.handle = lambda text: reviewed(handle(text))
    assistant.continue_creation = lambda token: reviewed(continuation(token))
    return assistant


def inferred_assistant(tmp_path, *, action="create_ask", bridge=None, selected=True):
    assistant = make_assistant(tmp_path, bridge)
    if selected:
        assistant.session.select("thr_previous", "Previous")
    assistant._router = SimpleNamespace(route=lambda text: GlobalCommand(action, text, inferred=True,
                                       target_kind="new" if action in {"create", "create_ask"} else "unspecified"))
    return with_successful_frontend_review(assistant)


def await_worker(assistant):
    """Poll as the owner until the worker result is published and consumed."""
    deadline = time.monotonic() + 2
    wait = threading.Event()
    while assistant._busy and time.monotonic() < deadline:
        assistant.poll()
        if assistant._busy:
            wait.wait(min(.001, max(0, deadline - time.monotonic())))
    assert not assistant._busy


def writes(bridge):
    return [call for call in bridge.calls if call[0] in {"create", "start", "queue", "cancel"}]


@pytest.mark.parametrize("source", [
    "Make a codex chat that will audit my downloads folder",
    "I need a fresh Codex conversation that can explain my repository",
    "Set up a Codex conversation to review this; do not edit or delete anything?!",
    '  Have a new Codex chat investigate "foo.py". Ask me before editing!  ',
])
def test_inferred_new_task_stages_original_capture_before_any_write(tmp_path, source):
    assistant = inferred_assistant(tmp_path)
    reply = assistant.handle(source)
    assert reply.session and source in reply.message
    final_prompt = assistant.session.draft
    assert final_prompt.endswith("Original voice request:\n" + source)
    assert "do not repeat that outer creation request" in final_prompt
    assert assistant._pending_plan.source == source
    assert assistant._pending_plan.final_prompt == final_prompt
    assert final_prompt not in reply.message
    assert assistant.review_message(full=True) == final_prompt
    temporary = assistant.session.target_id
    assert temporary.startswith("thr_voice_plan_")
    assert writes(assistant.bridge) == []
    assert assistant.session.consume('"send it"').action == "error"
    candidate = assistant.session.consume("send it")
    assert candidate.plan_id and candidate.prompt == final_prompt
    assert assistant.submit_confirmed(candidate) == "Sending the confirmed prompt."
    assert assistant.session.target_id == "thr_previous"
    await_worker(assistant)
    assert writes(assistant.bridge) == [("create", "Memoir voice task"), ("start", "thr_created", final_prompt)]
    assert temporary not in repr(writes(assistant.bridge))
    assert assistant.session.target_id == "thr_created"
    assert "Nothing sent" in assistant.submit_confirmed(candidate)
    assert len([c for c in writes(assistant.bridge) if c[0] == "create"]) == 1


def test_inferred_create_only_reads_source_creates_once_without_starting_turn(tmp_path):
    assistant = inferred_assistant(tmp_path, action="create", selected=False)
    source = "Would you arrange an empty Codex conversation without sending a task?"
    reply = assistant.handle(source)
    assert source in reply.message and "empty" in reply.message
    assert writes(assistant.bridge) == []
    confirmation = assistant.session.consume("send it")
    assert assistant.submit_confirmed(confirmation) == "Creating the confirmed chat."
    assert assistant.session.target_id == ""
    await_worker(assistant)
    assert writes(assistant.bridge) == [("create", "Memoir voice task")]
    assert assistant.session.target_id == "thr_created"
    assert "No prompt sent" in assistant._notice


@pytest.mark.parametrize("selected", [False, True])
@pytest.mark.parametrize("abandon", ["cancel", "expiry", "reset", "start_again", "close", "selection", "selection_back"])
def test_inferred_plan_abandonment_restores_prior_or_preserves_new_selection(tmp_path, selected, abandon):
    assistant = inferred_assistant(tmp_path, selected=selected)
    assistant.handle("Make a Codex conversation that can explain the files")
    candidate = None
    if abandon == "cancel":
        assistant.session.consume("cancel")
    elif abandon == "expiry":
        assistant.session._deadline = 0
        assert not assistant.session.active
    elif abandon == "reset":
        assistant.end_session()
    elif abandon == "start_again":
        reply = assistant.session.consume("start again")
        assert reply.action == "cancel" and "revised request" in reply.message
        assert not assistant.session.active
    elif abandon == "close":
        assistant.close()
    else:
        candidate = assistant.session.consume("send it")
        assistant.session.select("thr_newer", "Newer")
        if abandon == "selection_back":
            assistant.session.select("thr_previous", "Previous")
    expected = "thr_newer" if abandon == "selection" else "thr_previous" if selected or abandon == "selection_back" else ""
    assert assistant.session.target_id == expected
    assert assistant._pending_plan is None
    assert writes(assistant.bridge) == []
    if candidate:
        assert "Nothing sent" in assistant.submit_confirmed(candidate)
        assert writes(assistant.bridge) == []


def test_cancel_global_pending_plan_never_calls_bridge_cancel(tmp_path):
    assistant = inferred_assistant(tmp_path)
    assistant.handle("Make a Codex conversation to explain this project")
    reply = assistant.handle("cancel Codex")
    assert "cancelled" in reply.message
    assert assistant.session.target_id == "thr_previous"
    assert writes(assistant.bridge) == []
    final_prompt = assistant.session.draft


@pytest.mark.parametrize("change", ["prompt", "target_id", "confirmation_id", "plan_id", "revision", "expiry"])
def test_inferred_confirmation_rejects_mismatch_and_restores_prior(tmp_path, change):
    assistant = inferred_assistant(tmp_path)
    assistant.handle("Make a Codex conversation to explain this project without edits")
    candidate = assistant.session.consume("send it")
    if change == "revision":
        assistant.session.target_revision += 1
    elif change == "expiry":
        assistant.session.clock = lambda: assistant._pending_plan.expires_at + 1
    else:
        candidate = replace(candidate, **{change: "changed"})
    assert "Nothing sent" in assistant.submit_confirmed(candidate)
    assert writes(assistant.bridge) == []
    # A new explicit revision must never be overwritten by cleanup.
    if change != "revision":
        assert assistant.session.target_id == "thr_previous"


@pytest.mark.parametrize("guard", ["busy", "active"])
def test_busy_guard_does_not_consume_inferred_confirmation(tmp_path, guard):
    assistant = inferred_assistant(tmp_path)
    assistant.handle("Make a Codex conversation to explain this project")
    candidate = assistant.session.consume("send it")
    if guard == "busy":
        assistant._busy = True
    else:
        assistant.bridge.active_turn = ("thr_other", "turn_other")
    assert "already running" in assistant.submit_confirmed(candidate)
    assert assistant.session._confirmation is not None
    assert writes(assistant.bridge) == []
    assistant._busy = False
    assistant.bridge.active_turn = None
    assert assistant.submit_confirmed(candidate) == "Sending the confirmed prompt."
    await_worker(assistant)
    assert len([call for call in writes(assistant.bridge) if call[0] == "create"]) == 1


def test_forged_temporary_target_never_reaches_bridge_navigation_or_permissions(tmp_path):
    navigation = []
    assistant = make_assistant(tmp_path, navigator=lambda target: navigation.append(target))
    assistant.session.select("thr_voice_plan_forged", "Forged")
    assistant.session.begin("payload")
    candidate = assistant.session.consume("send it")
    assert "Nothing sent" in assistant.submit_confirmed(candidate)
    assistant.handle("open Codex chat current")
    assistant.handle("use Luna for this Codex chat")
    assert navigation == [] and not assistant._luna_permission
    assert writes(assistant.bridge) == []


def test_completed_inferred_creation_never_overwrites_new_explicit_selection(tmp_path):
    bridge = FakeBridge()
    bridge.block_create = True
    assistant = inferred_assistant(tmp_path, bridge=bridge)
    assistant.handle("Make a Codex conversation to explain this project")
    assistant.submit_confirmed(assistant.session.consume("send it"))
    assert bridge.create_started.wait(2)
    assistant.session.select("thr_newer", "Newer")
    bridge.create_release.set()
    await_worker(assistant)
    assert assistant._created_thread_id == "thr_created"
    assert assistant.session.target_id == "thr_newer"


@pytest.mark.parametrize("failure", ["turn_error", "unknown", "creation_unknown"])
def test_inferred_creation_and_submission_failures_retain_created_identity_without_retry(tmp_path, failure):
    from assistant.codex_bridge import DeliveryUnknown
    bridge = FakeBridge()
    if failure == "creation_unknown":
        def fail_create(title):
            bridge.calls.append(("create", title))
            raise DeliveryUnknown("private details")
        bridge.create_thread = fail_create
    else:
        def fail_turn(target, prompt):
            bridge.calls.append(("start", target, prompt))
            raise DeliveryUnknown("private details") if failure == "unknown" else RuntimeError("private details")
        bridge.start_turn = fail_turn
    assistant = inferred_assistant(tmp_path, bridge=bridge)
    assistant.handle("Make a Codex conversation to explain this project")
    assistant.submit_confirmed(assistant.session.consume("send it"))
    await_worker(assistant)
    assert "private details" not in assistant._notice
    assert len([call for call in writes(bridge) if call[0] == "create"]) == 1
    if failure == "creation_unknown":
        assert "creation could not be confirmed" in assistant._notice
        assert assistant._created_thread_id == "" and assistant.session.target_id == "thr_previous"
    else:
        assert assistant._created_thread_id == "thr_created"
        assert assistant.session.target_id == "thr_created"
        assert "Chat created" in assistant._notice
        if failure == "unknown":
            assert "Delivery is unknown" in assistant._notice


def test_inferred_existing_ask_preserves_model_guard_and_exact_transcript(tmp_path):
    from assistant.codex_bridge import CodexError
    assistant = inferred_assistant(tmp_path, action="ask")
    source = "Have Codex inspect the folder without changing anything"
    def guarded_queue(target, prompt, **kwargs):
        assistant.bridge.calls.append(("guarded_queue", target, prompt, kwargs))
        raise CodexError("model details", code="model_permission")
    assistant.bridge.queue_existing = guarded_queue
    reply = assistant.handle(source)
    final_prompt = assistant.session.draft
    assert source in reply.message
    assert assistant.session.target_id == "thr_previous"
    assert writes(assistant.bridge) == []
    assistant.submit_confirmed(assistant.session.consume("send it"))
    await_worker(assistant)
    assert ("guarded_queue", "thr_previous", final_prompt, {}) in assistant.bridge.calls
    assert "exact task is held" in assistant._notice
    assert assistant._unsent_request and assistant._unsent_request.prompt == final_prompt
    assert "model details" not in assistant._notice
    assert assistant._luna_permission == set()


def test_controller_sends_broad_codex_candidates_to_router_and_honors_ignore(tmp_path):
    assistant = make_assistant(tmp_path)
    seen = []
    assistant._router = SimpleNamespace(route=lambda text: seen.append(text))
    for source in ["He said make a Codex chat", "Play Codex soundtrack", "Make a Codex conversation"]:
        assert assistant.handle(source) is None
    assert len(seen) == 3
    assert assistant.handle("play music") is None
    assert len(seen) == 3
    assert writes(assistant.bridge) == []


@pytest.mark.parametrize("action", ["approve", "use_luna", "switch_tab", "send"])
def test_controller_refuses_inferred_forbidden_actions_even_from_custom_router(tmp_path, action):
    assistant = inferred_assistant(tmp_path, action=action)
    reply = assistant.handle("Maybe Codex can do this")
    assert "Nothing sent" in reply.message
    assert writes(assistant.bridge) == [] and not assistant._luna_permission


@pytest.mark.parametrize("selected", [False, True])
def test_inferred_dictate_arms_empty_real_target_or_requests_selection(tmp_path, selected):
    assistant = inferred_assistant(tmp_path, action="dictate", selected=selected)
    reply = assistant.handle("I'd like to dictate a prompt for Codex")
    assert writes(assistant.bridge) == []
    assert assistant.session.draft == "" and assistant._pending_plan is None
    if selected:
        assert reply.session and assistant.session.phase == "dictating"
        assert assistant.session.target_id == "thr_previous"
        payload = 'Review foo.py. Do not edit "send it".'
        assistant.session.consume(payload)
        assistant.session.consume("finish")
        candidate = assistant.session.consume("send it")
        assert candidate.prompt == payload and candidate.plan_id == ""
        assistant.submit_confirmed(candidate)
        await_worker(assistant)
        assert writes(assistant.bridge) == [("queue", "thr_previous", payload)]
    else:
        assert not reply.session and "Create or select" in reply.message
        assert not assistant.session.active and assistant.session.target_id == ""


def test_cancelling_confirmed_creation_before_it_finishes_prevents_start_turn(tmp_path):
    bridge = FakeBridge()
    bridge.block_create = True
    assistant = inferred_assistant(tmp_path, bridge=bridge)
    assistant.handle("Make a Codex conversation to audit files without edits")
    assistant.submit_confirmed(assistant.session.consume("send it"))
    assert bridge.create_started.wait(2)
    assistant.handle("cancel Codex")
    bridge.create_release.set()
    await_worker(assistant)
    assert writes(bridge) == [("create", "Memoir voice task")]
    assert assistant._created_thread_id == "thr_created"
    assert assistant._notice.startswith("Chat created; prompt cancelled before sending.")


def test_closing_confirmed_creation_before_it_finishes_prevents_start_turn(tmp_path):
    bridge = FakeBridge()
    bridge.block_create = True
    assistant = inferred_assistant(tmp_path, bridge=bridge)
    assistant.handle("Make a Codex conversation to audit files without edits")
    assistant.submit_confirmed(assistant.session.consume("send it"))
    assert bridge.create_started.wait(2)
    assistant.close()
    bridge.create_release.set()
    assert bridge.create_done.wait(2)
    assert writes(bridge) == [("create", "Memoir voice task")]
    assert assistant.session.target_id == "thr_previous"


def test_luna_permission_during_new_plan_never_silently_changes_previous_target(tmp_path):
    assistant = inferred_assistant(tmp_path)
    assistant.handle("Make a Codex conversation to audit the files")
    synthetic = assistant.session.target_id
    reply = assistant.handle("use Luna for this Codex chat")
    assert "No permission changed" in reply.message
    assert assistant.session.target_id == "thr_previous"
    assert synthetic not in assistant._luna_permission and "thr_previous" not in assistant._luna_permission
    assert writes(assistant.bridge) == []


def test_explicit_luna_permission_during_existing_plan_remains_bound_to_real_target(tmp_path):
    assistant = inferred_assistant(tmp_path, action="ask")
    assistant.handle("Have Codex audit the files")
    reply = assistant.handle("use Luna for this Codex chat")
    assert "Luna is permitted" in reply.message
    assert assistant._luna_permission == {"thr_previous"}
    assert assistant._pending_plan is None and assistant.session.target_id == "thr_previous"
    assert writes(assistant.bridge) == []


def test_expiring_new_plan_during_permission_request_does_not_grant_previous_target(tmp_path):
    assistant = inferred_assistant(tmp_path)
    assistant.handle("Make a Codex conversation to audit the files")
    expiry = assistant._pending_plan.expires_at
    assistant.session.clock = lambda: expiry + 1
    reply = assistant.handle("use Luna for this Codex chat")
    assert "No permission changed" in reply.message
    assert assistant.session.target_id == "thr_previous"
    assert assistant._luna_permission == set()


def test_inferred_new_chat_context_is_precomposed_before_review_and_keeps_nested_task(tmp_path):
    assistant = inferred_assistant(tmp_path)
    source = "Make a Codex conversation to explain how to create a chat from Python. Do not edit files?!"
    expected = ("The outer request to create this Codex chat has already been fulfilled. "
                "Perform the task requested below in this chat; do not repeat that outer creation request."
                "\n\nOriginal voice request:\n" + source)
    reply = assistant.handle(source)
    assert assistant.session.draft == expected
    assert source in reply.message and expected not in reply.message
    assert reply.message.startswith("Create a new Codex chat and send")
    assert assistant.review_message(full=True) == expected
    assert assistant._pending_plan.source == source
    assert assistant._pending_plan.final_prompt == expected
    assert "do not create another chat" not in expected
    status = assistant.handle("Codex status")
    assert status.session and source in status.message and expected not in status.message
    candidate = assistant.session.consume("send it")
    assistant.submit_confirmed(candidate)
    await_worker(assistant)
    assert ("start", "thr_created", expected) in writes(assistant.bridge)


def test_raw_source_cannot_replace_reviewed_composed_payload_at_confirmation(tmp_path):
    assistant = inferred_assistant(tmp_path)
    source = "Make a Codex conversation to inspect this without edits"
    assistant.handle(source)
    candidate = assistant.session.consume("send it")
    assert candidate.prompt != source
    assert "Nothing sent" in assistant.submit_confirmed(replace(candidate, prompt=source))
    assert writes(assistant.bridge) == [] and assistant.session.target_id == "thr_previous"


def test_composed_prompt_length_overflow_refuses_without_truncation_or_synthetic_target(tmp_path):
    assistant = inferred_assistant(tmp_path)
    source = "Make a Codex conversation " + "x" * (12000 - len("Make a Codex conversation "))
    assert len(source) == 12000
    reply = assistant.handle(source)
    assert not reply.session and "too long" in reply.message
    assert assistant.session.target_id == "thr_previous" and assistant.session.draft == ""
    assert assistant._pending_plan is None and writes(assistant.bridge) == []


@pytest.mark.parametrize("control", ["\n", "\t", "\x07"])
def test_source_controls_refused_before_composition(tmp_path, control):
    assistant = inferred_assistant(tmp_path)
    reply = assistant.handle("Make a Codex conversation to inspect" + control + "do not edit")
    assert not reply.session and "Nothing sent" in reply.message
    assert control not in reply.message
    assert assistant.session.target_id == "thr_previous" and assistant._pending_plan is None
    assert writes(assistant.bridge) == []


@pytest.mark.parametrize("verb", ["open", "select"])
@pytest.mark.parametrize("alias", ["current", "this"])
@pytest.mark.parametrize("expired", [False, True])
def test_current_navigation_during_synthetic_plan_never_retargets_previous_chat(tmp_path, verb, alias, expired):
    assistant = inferred_assistant(tmp_path)
    opened = []
    assistant._navigator = opened.append
    assistant.handle("Make a Codex conversation to inspect the files")
    if expired:
        expiry = assistant._pending_plan.expires_at
        assistant.session.clock = lambda: expiry + 1
    reply = assistant.handle(f"{verb} Codex chat {alias}")
    assert "has not been created yet" in reply.message
    assert opened == [] and assistant.session.target_id == "thr_previous"
    assert assistant._pending_plan is None and writes(assistant.bridge) == []
    assert not any(call[0] == "list" for call in assistant.bridge.calls)


@pytest.mark.parametrize("field", ["target_id", "prompt", "confirmation_id", "plan_id"])
def test_malformed_confirmation_field_types_refuse_without_controller_crash(tmp_path, field):
    assistant = make_assistant(tmp_path)
    candidate = finish_prompt(assistant)
    assert "Nothing sent" in assistant.submit_confirmed(replace(candidate, **{field: None}))
    assert writes(assistant.bridge) == []


def targeted_assistant(tmp_path, *, kind="selected", name="", bridge=None, selected=True):
    assistant = make_assistant(tmp_path, bridge)
    if selected:
        assistant.session.select("thr_previous", "Previous")
    assistant._router = SimpleNamespace(route=lambda text: GlobalCommand("ask", text, inferred=True,
                                                                        target_kind=kind, target_name=name))
    return with_successful_frontend_review(assistant)


def test_selected_request_without_real_selection_never_creates_chat(tmp_path):
    assistant = targeted_assistant(tmp_path, selected=False)
    source = "Send a prompt in the Codex chat to check the log without editing"
    reply = assistant.handle(source)
    assert not reply.session and "No Codex chat is selected" in reply.message
    assert writes(assistant.bridge) == [] and not assistant.session.active
    assert assistant._unresolved_request.text == source


@pytest.mark.parametrize("source,title", [
    ("Ask the wakeboard transcript chat to check the logs", "wakeboard transcript"),
    ("Send a prompt to check the logs in the Wake World Transcript Chat.", "Wake World Transcript"),
])
def test_named_without_codex_resolves_exact_project_title_readonly_before_review(tmp_path, source, title):
    bridge = FakeBridge()
    bridge.list_threads = lambda: [{"id": "thr_named", "name": title.upper(), "cwd": str(tmp_path)},
                                   {"id": "thr_foreign", "name": title, "cwd": str(tmp_path.parent)}]
    assistant = targeted_assistant(tmp_path, kind="named", name=title, bridge=bridge)
    reply = assistant.handle(source)
    assert reply.await_token and not reply.session and writes(bridge) == []
    await_worker(assistant)
    completed = assistant.continue_creation(reply.await_token)
    assert completed.session and source in completed.message
    assert assistant.session.target_id == "thr_named" and assistant.session.draft.endswith(source)
    final_prompt = assistant.session.draft
    assert writes(bridge) == []
    assert not assistant.continue_creation(reply.await_token).session
    assistant.handle("Send it.")
    await_worker(assistant)
    assert writes(bridge) == [("queue", "thr_named", final_prompt)]


def test_named_title_ignores_whitespace_and_case_but_requires_full_name_and_confirmation(tmp_path):
    bridge = FakeBridge()
    bridge.list_threads = lambda: [
        {"id": "thr_token_tracker", "name": "token tracker", "cwd": str(tmp_path)},
        {"id": "thr_continuation", "name": "token tracker continuation", "cwd": str(tmp_path)},
    ]
    assistant = targeted_assistant(tmp_path, kind="named", name="TokenTracker", bridge=bridge)

    initial = assistant.handle("Ask TokenTracker to inspect logs without changes")
    await_worker(assistant)
    reply = assistant.continue_creation(initial.await_token)

    assert reply.review and reply.session
    assert assistant.session.target_id == "thr_token_tracker"
    assert assistant._pending_choice is None
    assert writes(bridge) == []


def test_named_whitespace_collision_offers_only_colliding_chats(tmp_path):
    bridge = FakeBridge()
    bridge.list_threads = lambda: [
        {"id": "thr_unrelated", "name": "Newest unrelated chat", "cwd": str(tmp_path)},
        {"id": "thr_spaced", "name": "Token Tracker", "cwd": str(tmp_path)},
        {"id": "thr_joined", "name": "tokentracker", "cwd": str(tmp_path)},
    ]
    assistant = targeted_assistant(tmp_path, kind="named", name="TokenTracker", bridge=bridge)

    initial = assistant.handle("Ask TokenTracker to inspect logs without changes")
    await_worker(assistant)
    reply = assistant.continue_creation(initial.await_token)

    assert not reply.session and "one exact chat" in reply.message
    assert [row["id"] for row in assistant._pending_choice.rows] == ["thr_spaced", "thr_joined"]
    assert "Newest unrelated chat" not in reply.message
    assert assistant.session.target_id == "thr_previous"
    assert writes(bridge) == []


def test_named_title_matching_preserves_punctuation(tmp_path):
    bridge = FakeBridge()
    bridge.list_threads = lambda: [
        {"id": "thr_c", "name": "C", "cwd": str(tmp_path)},
        {"id": "thr_cpp", "name": "C++", "cwd": str(tmp_path)},
    ]
    assistant = targeted_assistant(tmp_path, kind="named", name="C++", bridge=bridge)

    initial = assistant.handle("Ask C++ to inspect the build logs without changes")
    await_worker(assistant)
    reply = assistant.continue_creation(initial.await_token)

    assert reply.review and reply.session
    assert assistant.session.target_id == "thr_cpp"
    assert writes(bridge) == []


@pytest.mark.parametrize("verb", ["Open", "Select"])
@pytest.mark.parametrize(("requested", "rows", "expected"), [
    ("TokenTracker", [{"id": "thr_token", "name": "token tracker"}], "thr_token"),
    ("C++", [{"id": "thr_c", "name": "C"}, {"id": "thr_cpp", "name": "C++"}], "thr_cpp"),
])
def test_explicit_chat_lookup_uses_same_whitespace_case_and_punctuation_key(
        tmp_path, verb, requested, rows, expected):
    bridge = FakeBridge()
    bridge.list_threads = lambda: [dict(row, cwd=str(tmp_path)) for row in rows]
    opened = []
    assistant = make_assistant(
        tmp_path, bridge,
        navigator=lambda ident: opened.append(ident) or "Requested the chat in Codex.")

    reply = assistant.handle(f"{verb} Codex chat {requested}")
    assert not reply.session
    await_worker(assistant)

    assert assistant.session.target_id == expected
    assert opened == ([expected] if verb == "Open" else [])
    assert writes(bridge) == []


@pytest.mark.parametrize("verb", ["Open", "Select"])
@pytest.mark.parametrize("start_selected", [False, True])
def test_explicit_chat_lookup_refuses_whitespace_title_collisions(tmp_path, verb, start_selected):
    bridge = FakeBridge()
    bridge.list_threads = lambda: [
        {"id": "thr_spaced", "name": "Token Tracker", "cwd": str(tmp_path)},
        {"id": "thr_joined", "name": "tokentracker", "cwd": str(tmp_path)},
    ]
    opened = []
    assistant = make_assistant(tmp_path, bridge, navigator=opened.append)
    original_target = "thr_previous" if start_selected else ""
    if start_selected:
        assistant.session.select(original_target, "Previous")

    assistant.handle(f"{verb} Codex chat TokenTracker")
    await_worker(assistant)

    assert assistant.session.target_id == original_target
    assert not opened and writes(bridge) == []


def test_named_choice_exact_title_uses_same_whitespace_key(tmp_path):
    bridge = FakeBridge()
    bridge.list_threads = lambda: [
        {"id": "thr_token", "name": "Token Tracker", "cwd": str(tmp_path)},
        {"id": "thr_other", "name": "Other Chat", "cwd": str(tmp_path)},
    ]
    assistant = targeted_assistant(tmp_path, kind="named", name="Unmatched", bridge=bridge)

    initial = assistant.handle("Ask Unmatched to inspect logs without changes")
    await_worker(assistant)
    choices = assistant.continue_creation(initial.await_token)
    assert assistant._pending_choice is not None and not writes(bridge)

    selected = assistant.handle("TokenTracker")

    assert selected.review and selected.session
    assert assistant.session.target_id == "thr_token"
    assert not writes(bridge)


@pytest.mark.parametrize(("error_code", "message"), [
    ("auth", "Codex needs your ChatGPT sign-in. No API key will be used."),
    ("quota", "Codex reached its usage limit. I won't retry or upgrade models."),
    ("model", "The permitted Luna model isn't available. I won't choose an expensive model."),
    ("scope", "That chat or action is outside the permitted project."),
    ("model_permission", "That chat's model can't be safely preserved. Say use Luna for this Codex chat, then dictate again."),
    ("transport", "Codex connection is unavailable. Nothing sent."),
    ("server-secret-code", "Codex couldn't complete that request. Check its sign-in or model availability."),
])
def test_named_lookup_failure_is_categorized_without_exposing_error_payload(tmp_path, caplog, error_code, message):
    from assistant.codex_bridge import CodexError

    bridge = FakeBridge()
    bridge.list_threads = lambda: (_ for _ in ()).throw(
        CodexError("private server payload and auth detail", code=error_code))
    assistant = targeted_assistant(tmp_path, kind="named", name="Build Notes", bridge=bridge)
    logger = logging.getLogger("memoir.global")
    prior_propagate = logger.propagate
    logger.propagate = False
    logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.INFO, logger=logger.name):
            initial = assistant.handle("Ask Build Notes to inspect logs without changes")
            await_worker(assistant)
            reply = assistant.continue_creation(initial.await_token)

        assert reply.status == "failed"
        assert reply.message == message
        assert assistant._notice == message
        assert "private server payload" not in caplog.text
        assert "auth detail" not in caplog.text
        log_messages = [record.getMessage() for record in caplog.records]
        logged_codes = [message.split("code=", 1)[-1]
                        for message in log_messages if "Codex operation failed:" in message]
        assert logged_codes == [error_code if error_code in {
            "auth", "quota", "model", "scope", "model_permission", "transport", "protocol", "receipt"
        } else "other"]
    finally:
        logger.removeHandler(caplog.handler)
        logger.propagate = prior_propagate


def test_global_reply_status_default_preserves_existing_positional_arguments():
    reply = GlobalReply("message", True, "token", True)
    assert reply.status == ""


@pytest.mark.parametrize("mode", ["cancel", "expire", "selection", "selection_back", "revise", "close"])
def test_named_lookup_never_stages_after_cancel_expiry_revision_or_close(tmp_path, mode):
    bridge = FakeBridge()
    bridge.block_list = True
    original_list = bridge.list_threads
    def catalogue():
        original_list()
        return [{"id": "thr_named", "name": "Build Notes", "cwd": str(tmp_path)}]
    bridge.list_threads = catalogue
    assistant = targeted_assistant(tmp_path, kind="named", name="Build Notes", bridge=bridge)
    reply = assistant.handle("Ask the Build Notes chat to inspect this without edits")
    assert bridge.list_started.wait(2)
    if mode == "cancel":
        assistant.handle("cancel Codex")
    elif mode == "expire":
        assistant._pending_lookup = replace(assistant._pending_lookup, expires_at=0)
    elif mode in {"selection", "selection_back"}:
        assistant.session.select("thr_other", "Other")
        if mode == "selection_back":
            assistant.session.select("thr_previous", "Previous")
    elif mode == "revise":
        assistant.handle("Ask the Build Notes chat to inspect a different thing")
    else:
        assistant.close()
    bridge.list_release.set()
    if mode != "close":
        await_worker(assistant)
    result = assistant.continue_creation(reply.await_token)
    assert not result.session and not assistant.session.active
    assert assistant.session.target_id != "thr_named" and writes(bridge) == []


def test_operation_kind_is_read_only_and_tracks_background_lookup(tmp_path):
    bridge = FakeBridge()
    bridge.block_list = True
    bridge.list_threads = lambda: (bridge.list_started.set(), bridge.list_release.wait(2), [])[2]
    assistant = targeted_assistant(tmp_path, kind="named", name="Build Notes", bridge=bridge)

    reply = assistant.handle("Ask Build Notes to inspect logs without changes")
    assert reply.await_token and bridge.list_started.wait(2)
    assert assistant.operation_kind == "resolve_named"
    with pytest.raises(AttributeError):
        assistant.operation_kind = "submit"

    bridge.list_release.set()
    await_worker(assistant)
    assert assistant.operation_kind == ""


@pytest.mark.parametrize("names", [["Build Notes", "Build Notes"], ["Build Notes Chat", "Another Chat"]])
def test_named_nonunique_or_weak_title_offers_actual_choices_without_fuzzy_selection(tmp_path, names):
    bridge = FakeBridge()
    bridge.list_threads = lambda: [{"id": f"thr_{i}", "name": name, "cwd": str(tmp_path)} for i, name in enumerate(names)]
    assistant = targeted_assistant(tmp_path, kind="named", name="Build Notes", bridge=bridge)
    reply = assistant.handle("Ask the Build Notes chat to inspect this without edits")
    await_worker(assistant)
    result = assistant.continue_creation(reply.await_token)
    assert not result.session and "one exact chat" in result.message
    assert all(name in result.message for name in names)
    assert assistant.session.target_id == "thr_previous" and writes(bridge) == []
    assert assistant._unresolved_request.text.endswith("without edits")


def test_named_handoff_invalidated_if_target_changes_after_lookup_completion(tmp_path):
    bridge = FakeBridge()
    bridge.list_threads = lambda: [{"id": "thr_named", "name": "Build Notes", "cwd": str(tmp_path)}]
    assistant = targeted_assistant(tmp_path, kind="named", name="Build Notes", bridge=bridge)
    reply = assistant.handle("Ask Build Notes to inspect this without edits")
    await_worker(assistant)
    assistant.session.select("thr_other", "Other")
    assert not assistant.continue_creation(reply.await_token).session
    assert assistant.session.target_id == "thr_other" and writes(bridge) == []


def test_controller_feedback_extends_both_plan_and_session_before_confirmation(tmp_path):
    assistant = inferred_assistant(tmp_path)
    now = [0.0]
    assistant.session.clock = lambda: now[0]
    assistant.session.timeout_seconds = 10
    source = "Make a Codex conversation to review this without edits"
    reply = assistant.handle(source)
    assert source in reply.message and "outer request" not in reply.message
    deadline = assistant._pending_plan.expires_at
    token = assistant.begin_feedback()
    now[0] = 40
    assistant.poll()
    assert assistant.session.active and assistant._pending_plan is not None
    assert assistant.end_feedback(token, True)
    assert assistant._pending_plan.expires_at == deadline + 40 == assistant.session.deadline
    assert assistant.review_message(full=True) == assistant._pending_plan.final_prompt
    now[0] = 49
    candidate = assistant.consume("send it")
    assert candidate.action == "submit"
    assistant.submit_confirmed(candidate)
    await_worker(assistant)
    assert len([call for call in writes(assistant.bridge) if call[0] == "create"]) == 1


@pytest.mark.parametrize("mode", ["failure", "cap", "cancel", "selection"])
def test_controller_feedback_invalidations_restore_prior_and_never_resurrect(tmp_path, mode):
    assistant = inferred_assistant(tmp_path)
    now = [0.0]
    assistant.session.clock = lambda: now[0]
    assistant.handle("Make a Codex conversation to inspect this")
    token = assistant.begin_feedback()
    if mode == "cap":
        now[0] = 270
    elif mode == "cancel":
        assistant.end_session()
    elif mode == "selection":
        assistant.session.select("thr_other", "Other")
    assert not assistant.end_feedback(token, mode != "failure")
    assert assistant._pending_plan is None and not assistant.session.active
    assert assistant.session.target_id == ("thr_other" if mode == "selection" else "thr_previous")
    assert writes(assistant.bridge) == []


def test_preserved_review_next_wake_controls_send_exactly_once(tmp_path):
    assistant = targeted_assistant(tmp_path)
    source = "Send a prompt in the Codex chat to inspect logs; do not delete anything?!"
    assistant.handle(source)
    final_prompt = assistant.session.draft
    original_deadline = assistant.session.deadline
    assert assistant.keep_review_waiting()
    assert assistant.session.deadline == original_deadline
    assert source in assistant.handle("status").message
    assert assistant.handle("read it back").message == final_prompt
    assert assistant.handle("send it").message == "Sending the confirmed prompt."
    await_worker(assistant)
    assert writes(assistant.bridge) == [("queue", "thr_previous", final_prompt)]
    assistant._router = SimpleNamespace(route=lambda _: None)
    assert assistant.handle("send it") is None
    assert len(writes(assistant.bridge)) == 1


def test_partial_dictation_preserved_as_review_but_empty_dictation_is_not(tmp_path):
    assistant = make_assistant(tmp_path)
    assistant.session.select("thr_previous", "Previous")
    assistant.session.begin()
    assert not assistant.keep_review_waiting()
    assistant.consume("Exact partial task without edits")
    deadline = assistant.session.deadline
    assert assistant.keep_review_waiting() and assistant.session.phase == "review"
    assert assistant.session.deadline == deadline
    assert "Exact partial task without edits" in assistant.handle("status").message
    assert writes(assistant.bridge) == []


def test_notices_are_fixed_one_use_correlated_and_never_speak_native_payloads(tmp_path):
    bridge = FakeBridge()
    assistant = make_assistant(tmp_path, bridge)
    bridge.owned_threads.add("thr_owned")
    assistant._current_job = ("thr_owned", "turn_current")
    events = [{"kind": "completed", "thread_id": "thr_foreign", "text": "PRIVATE ONE"},
              {"kind": "completed", "thread_id": "thr_owned", "turn_id": "turn_old", "text": "PRIVATE TWO"},
              {"kind": "error", "thread_id": "thr_owned", "turn_id": "turn_current", "message": "PRIVATE THREE"}]
    bridge.drain_events = lambda: events[:]
    assistant.poll()
    events.clear()
    assert assistant.take_notice() == "Codex reported an error."
    assert assistant.take_notice() is None
    assistant.session.select("thr_owned", "Owned")
    assistant.session.begin("Task")
    token = assistant.begin_feedback(review=True)
    assert assistant.end_feedback(token, True)
    assistant.submit_confirmed(assistant.consume("send it"))
    await_worker(assistant)
    assert assistant.take_notice() == "Codex accepted the prompt."
    assert assistant.take_notice() is None


def test_non_session_feedback_has_no_token_and_success_is_boolean(tmp_path):
    assistant = make_assistant(tmp_path)
    assert assistant.begin_feedback() is None
    assert assistant.end_feedback(None, True)
    assert not assistant.end_feedback(None, False)
    assert not assistant.end_feedback(None, "true")


@pytest.mark.parametrize("selection", ["second", "Actual Review Chat"])
def test_named_choice_carries_original_task_without_repeating_or_dispatching(tmp_path, selection):
    bridge = FakeBridge()
    bridge.list_threads = lambda: [{"id": "thr_a", "name": "Other Chat", "cwd": str(tmp_path)},
                                   {"id": "thr_b", "name": "Actual Review Chat", "cwd": str(tmp_path)}]
    assistant = targeted_assistant(tmp_path, kind="named", name="Misheard Review Chat", bridge=bridge)
    source = "Ask Misheard Review Chat to inspect logs without deleting or editing anything?!"
    initial = assistant.handle(source)
    await_worker(assistant)
    choices = assistant.continue_creation(initial.await_token)
    assert "Say first, second" in choices.message and not choices.session
    assert writes(bridge) == []
    review = assistant.handle(selection)
    assert review.session and "Actual Review Chat" in review.message and source in review.message
    assert assistant.session.target_id == "thr_b"
    assert assistant._pending_plan.source == source
    assert writes(bridge) == []
    final_prompt = assistant.session.draft
    assert final_prompt.endswith(source) and "do not repeat that outer dispatch request" in final_prompt
    assistant.handle("send it")
    await_worker(assistant)
    assert writes(bridge) == [("queue", "thr_b", final_prompt)]


@pytest.mark.parametrize("mode", ["expiry", "revision", "cancel", "close"])
def test_named_choice_bound_to_deadline_revision_and_cancellation(tmp_path, mode):
    bridge = FakeBridge()
    bridge.list_threads = lambda: [{"id": "thr_b", "name": "Actual Review", "cwd": str(tmp_path)}]
    assistant = targeted_assistant(tmp_path, kind="named", name="Misheard Review", bridge=bridge)
    initial = assistant.handle("Ask Misheard Review to inspect logs without deleting")
    await_worker(assistant)
    assistant.continue_creation(initial.await_token)
    if mode == "expiry":
        assistant._pending_choice = replace(assistant._pending_choice, expires_at=0)
    elif mode == "revision":
        assistant.session.select("thr_newer", "Newer")
    elif mode == "cancel":
        assistant.handle("cancel")
        assistant._router = SimpleNamespace(route=lambda _: None)
    else:
        assistant.close()
    reply = assistant.handle("first")
    assert reply is None or not reply.session
    assert assistant.session.target_id != "thr_b" and writes(bridge) == []


def test_bare_wake_resumes_only_active_review_without_changing_deadline(tmp_path):
    assistant = targeted_assistant(tmp_path)
    source = "Send a prompt in the Codex chat to inspect this without edits"
    assistant.handle(source)
    deadline = assistant.session.deadline
    assert assistant.keep_review_waiting()
    resumed = assistant.handle("")
    assert resumed.session and source in resumed.message
    assert assistant.session.deadline == deadline and writes(assistant.bridge) == []
    assistant.end_session()
    assert assistant.handle("") is None


def test_old_turn_event_cannot_clear_new_job_budget_and_fatal_error_is_fixed_notice(tmp_path):
    from assistant.codex_bridge import CodexError
    bridge = FakeBridge()
    assistant = make_assistant(tmp_path, bridge)
    assistant._current_job = ("thr_owned", "turn_new")
    assistant._job_started = time.monotonic()
    events = [{"kind": "completed", "thread_id": "thr_owned", "turn_id": "turn_old", "text": "secret"}]
    bridge.drain_events = lambda: events[:]
    assistant.poll()
    assert assistant._job_started is not None and not assistant._notices
    events[:] = [{"kind": "error", "message": "private transport failure"}]
    bridge._fatal = CodexError("private failure")
    assistant.poll()
    events.clear()
    notice = assistant.take_notice()
    assert "Job status is unknown" in notice and "private" not in notice
    assert assistant._job_started is None


def test_failed_choice_feedback_invalidates_held_task_before_choice_control(tmp_path):
    bridge = FakeBridge()
    bridge.list_threads = lambda: [{"id": "thr_b", "name": "Actual Review", "cwd": str(tmp_path)}]
    assistant = targeted_assistant(tmp_path, kind="named", name="Misheard Review", bridge=bridge)
    initial = assistant.handle("Ask Misheard Review to inspect logs without deleting")
    await_worker(assistant)
    assistant.continue_creation(initial.await_token)
    assert assistant.begin_feedback() is None
    assert not assistant.end_feedback(None, False)
    assistant._router = SimpleNamespace(route=lambda _: None)
    assert assistant.handle("first") is None
    assert assistant._pending_choice is None and writes(bridge) == []


def test_late_named_lookup_requires_successful_actual_review_before_wake_send(tmp_path):
    bridge = FakeBridge()
    bridge.list_threads = lambda: [{"id": "thr_named", "name": "Build Notes", "cwd": str(tmp_path)}]
    assistant = targeted_assistant(tmp_path, kind="named", name="Build Notes", bridge=bridge)
    source = "Ask Build Notes to inspect the logs without changes"
    initial = GlobalAssistant.handle(assistant, source)
    await_worker(assistant)
    # Simulate the bounded audio wait ending before the async result returned.
    assert assistant.session.active and assistant._review_armed is None
    response = GlobalAssistant.handle(assistant, "send it")
    assert response.session and source in response.message
    assert writes(bridge) == [] and assistant.session._confirmation is None
    generic = assistant.begin_feedback()
    assert assistant.end_feedback(generic, True)
    assert assistant._review_armed is None
    assert GlobalAssistant.handle(assistant, "send it").session
    token = assistant.begin_feedback(review=True)
    assert assistant.end_feedback(token, True)
    sent = GlobalAssistant.handle(assistant, "send it")
    assert sent.message == "Sending the confirmed prompt."
    await_worker(assistant)
    assert len(writes(bridge)) == 1 and writes(bridge)[0][0] == "queue"


def test_direct_session_confirmation_cannot_bypass_missing_review_arm(tmp_path):
    assistant = inferred_assistant(tmp_path)
    GlobalAssistant.handle(assistant, "Make a Codex conversation to inspect the folder")
    candidate = VoiceSession.consume(assistant.session, "send it")
    assert "not been read back successfully" in assistant.submit_confirmed(candidate)
    assert writes(assistant.bridge) == [] and assistant.session.target_id == "thr_previous"


def test_named_cancel_restores_prelookup_selection_without_overwriting_newer_choice(tmp_path):
    for newer in [False, True]:
        bridge = FakeBridge()
        bridge.list_threads = lambda: [{"id": "thr_named", "name": "Build Notes", "cwd": str(tmp_path)}]
        assistant = targeted_assistant(tmp_path, kind="named", name="Build Notes", bridge=bridge)
        initial = assistant.handle("Ask Build Notes to inspect without changing files")
        await_worker(assistant)
        assistant.continue_creation(initial.await_token)
        assert assistant.session.target_id == "thr_named"
        if newer:
            assistant.session.select("thr_newer", "Newer")
        else:
            assistant.consume("cancel")
        assert assistant.session.target_id == ("thr_newer" if newer else "thr_previous")
        assert writes(bridge) == []


def test_named_choice_cancel_restores_selection_before_choice(tmp_path):
    bridge = FakeBridge()
    bridge.list_threads = lambda: [{"id": "thr_named", "name": "Actual Notes", "cwd": str(tmp_path)}]
    assistant = targeted_assistant(tmp_path, kind="named", name="Misheard Notes", bridge=bridge)
    initial = assistant.handle("Ask Misheard Notes to inspect without changing files")
    await_worker(assistant)
    assistant.continue_creation(initial.await_token)
    review = assistant.handle("first")
    assert review.session and assistant.session.target_id == "thr_named"
    assistant.consume("cancel")
    assert assistant.session.target_id == "thr_previous" and writes(bridge) == []


def test_unspoken_canonical_creation_review_also_requires_actual_feedback(tmp_path):
    bridge = FakeBridge()
    assistant = compound_assistant(tmp_path, bridge)
    initial = assistant.handle("Open a Codex chat and tell it to check out my Gmail files")
    drain_creation(assistant, bridge)
    assert assistant._pending_plan is not None and assistant.session.phase == "review"
    assert assistant._pending_plan.final_prompt == "check out my Gmail files"
    assert assistant._review_armed is None
    reply = assistant.handle("send it")
    assert reply.session and writes(bridge) == []
    token = assistant.begin_feedback(review=True)
    assert assistant.end_feedback(token, True)
    assert assistant.handle("send it").message == "Sending the confirmed prompt."
    await_worker(assistant)
    assert len([call for call in writes(bridge) if call[0] == "start"]) == 1


def test_segmented_dictation_requires_successful_review_and_generic_cue_cannot_arm(tmp_path):
    assistant = make_assistant(tmp_path)
    assistant.session.select("thr_previous", "Previous")
    assistant.session.begin()
    assistant.consume("Inspect logs without changing files")
    assistant.consume("finish")
    assert assistant.consume("send it").action == "review"
    assert writes(assistant.bridge) == []
    generic = assistant.begin_feedback()
    assert assistant.end_feedback(generic, True)
    assert assistant.consume("send it").action == "review"
    token = assistant.begin_feedback(review=True)
    assert assistant.end_feedback(token, True)
    candidate = assistant.consume("send it")
    assert candidate.action == "submit"
    assert assistant.submit_confirmed(candidate) == "Sending the confirmed prompt."
    await_worker(assistant)
    assert writes(assistant.bridge) == [("queue", "thr_previous", "Inspect logs without changing files")]


def test_busy_error_cue_never_claims_review_or_arms_unspoken_task(tmp_path):
    assistant = inferred_assistant(tmp_path)
    GlobalAssistant.handle(assistant, "Make a Codex chat to inspect this")
    assistant._busy = True
    reply = GlobalAssistant.handle(assistant, "send it")
    assert reply.session and not reply.review and "already running" in reply.message
    token = assistant.begin_feedback(review=reply.review)
    assert assistant.end_feedback(token, True)
    assert assistant._review_armed is None and writes(assistant.bridge) == []
    assistant._busy = False
    actual = GlobalAssistant.handle(assistant, "send it")
    assert actual.review and actual.session and "inspect this" in actual.message


def test_redictating_same_text_after_start_again_requires_fresh_review(tmp_path):
    assistant = make_assistant(tmp_path)
    assistant.session.select("thr_previous", "Previous")
    assistant.session.begin("Exact task")
    token = assistant.begin_feedback(review=True)
    assert assistant.end_feedback(token, True)
    assistant.consume("start again")
    assistant.consume("Exact task")
    assistant.consume("finish")
    assert assistant._review_armed is None
    assert assistant.consume("send it").action == "review"
    assert writes(assistant.bridge) == []


def test_named_staging_overflow_restores_selection_before_lookup(tmp_path):
    bridge = FakeBridge()
    bridge.list_threads = lambda: [{"id": "thr_named", "name": "Build Notes", "cwd": str(tmp_path)}]
    assistant = targeted_assistant(tmp_path, kind="named", name="Build Notes", bridge=bridge)
    source = "Ask Build Notes to inspect " + "x" * 11850
    initial = assistant.handle(source)
    await_worker(assistant)
    reply = assistant.continue_creation(initial.await_token)
    assert not reply.session and "too long" in reply.message
    assert assistant.session.target_id == "thr_previous" and writes(bridge) == []


def test_named_dictation_cancel_restores_prelookup_target(tmp_path):
    bridge = FakeBridge()
    bridge.list_threads = lambda: [{"id": "thr_named", "name": "Build Notes", "cwd": str(tmp_path)}]
    assistant = make_assistant(tmp_path, bridge)
    assistant.session.select("thr_previous", "Previous")
    assistant._router = SimpleNamespace(route=lambda source: GlobalCommand("dictate", source, inferred=True,
                                                                         target_kind="named", target_name="Build Notes"))
    initial = assistant.handle("I want to dictate a task for Build Notes")
    await_worker(assistant)
    reply = assistant.continue_creation(initial.await_token)
    assert reply.session and not reply.review and assistant.session.target_id == "thr_named"
    assistant.consume("cancel")
    assert assistant.session.target_id == "thr_previous" and writes(bridge) == []


def permission_assistant(tmp_path):
    from assistant.codex_bridge import CodexError
    assistant = targeted_assistant(tmp_path)
    def guarded(target, prompt, **kwargs):
        assistant.bridge.calls.append(("guarded", target, prompt, kwargs))
        if not kwargs.get("allow_model_change"):
            raise CodexError("private model metadata", code="model_permission")
        return {"status": "queued"}
    assistant.bridge.queue_existing = guarded
    return assistant


def test_known_unsent_permission_failure_holds_exact_task_and_restages_after_consent(tmp_path):
    assistant = permission_assistant(tmp_path)
    source = "Send a prompt in the Codex chat to inspect logs. Do not edit or delete anything?!"
    review = assistant.handle(source)
    original = assistant.session.draft
    original_cap = assistant.session.hard_deadline
    original_expiry = assistant.session.deadline
    assistant.submit_confirmed(assistant.consume("send it"))
    # The main audio cleanup must not invalidate the legitimate pending result.
    assistant.end_session()
    await_worker(assistant)
    assert assistant._unsent_request and assistant._unsent_request.prompt == original
    assert assistant._unsent_request.source == source
    assert assistant._unsent_request.expires_at == original_expiry
    assert "exact task is held" in assistant._notice and "private" not in assistant._notice
    reply = assistant.handle("use Luna for this Codex chat")
    assert reply.session and reply.review and source in reply.message
    assert assistant.session.draft == original
    assert assistant.session.hard_deadline == min(original_expiry, original_cap)
    assert assistant._unsent_request is None
    assert not any(call[0] in {"queue", "start", "create"} for call in assistant.bridge.calls)
    assert len([call for call in assistant.bridge.calls if call[0] == "guarded"]) == 1
    assistant.handle("send it")
    await_worker(assistant)
    guarded = [call for call in assistant.bridge.calls if call[0] == "guarded"]
    assert guarded == [("guarded", "thr_previous", original, {}),
                       ("guarded", "thr_previous", original, {"allow_model_change": True})]


def test_permission_result_after_original_confirmation_deadline_is_discarded(tmp_path):
    from assistant.codex_bridge import CodexError
    assistant = permission_assistant(tmp_path)
    now = [0.0]
    assistant.session.clock = lambda: now[0]
    started, release = threading.Event(), threading.Event()
    def delayed_permission_failure(target, prompt):
        assistant.bridge.calls.append(("guarded", target, prompt, {}))
        started.set()
        assert release.wait(2)
        raise CodexError("private model state", code="model_permission")
    assistant.bridge.queue_existing = delayed_permission_failure
    assistant.handle("Send a prompt in the Codex chat to inspect logs without edits")
    assert assistant.session.deadline == 90
    now[0] = 89
    assistant.submit_confirmed(assistant.consume("send it"))
    assert started.wait(2)
    assistant.end_session()
    now[0] = 92
    release.set()
    await_worker(assistant)
    assert assistant._unsent_request is None
    reply = assistant.handle("use Luna for this Codex chat")
    assert not reply.review and not assistant.session.active
    assert len([call for call in assistant.bridge.calls if call[0] == "guarded"]) == 1
    assert writes(assistant.bridge) == []


def test_permission_consent_near_original_deadline_cannot_extend_review(tmp_path):
    assistant = permission_assistant(tmp_path)
    now = [0.0]
    assistant.session.clock = lambda: now[0]
    assistant.handle("Send a prompt in the Codex chat to inspect logs without edits")
    now[0] = 88
    assistant.submit_confirmed(assistant.consume("send it"))
    await_worker(assistant)
    assert assistant._unsent_request.expires_at == 90
    assert assistant._unsent_request.hard_deadline == 270
    now[0] = 89
    reply = assistant.handle("use Luna for this Codex chat")
    assert reply.review and reply.session
    assert assistant.session.deadline == assistant.session.hard_deadline == 90
    assert assistant._pending_plan.expires_at == 90
    # Readback suspension cannot exceed the held request's original expiry.
    token = assistant.begin_feedback(review=True)
    now[0] = 90
    assert not assistant.end_feedback(token, True)
    assert not assistant.session.active and assistant._pending_plan is None
    assert assistant.consume("send it").action != "submit"
    assert len([call for call in assistant.bridge.calls if call[0] == "guarded"]) == 1
    assert writes(assistant.bridge) == []


@pytest.mark.parametrize("control", ["cancel", "cancel that", "start again", "cancel Codex"])
def test_idle_permission_recovery_local_controls_clear_task_without_remote_cancel(tmp_path, control):
    assistant = permission_assistant(tmp_path)
    assistant.handle("Send a prompt in the Codex chat to inspect logs without edits")
    assistant.submit_confirmed(assistant.consume("send it"))
    await_worker(assistant)
    assert assistant._unsent_request and assistant.session.phase == "idle"
    generation = assistant._request_generation
    assistant._router = SimpleNamespace(route=lambda _: None)
    reply = assistant.handle(control)
    assert "Held request cancelled" in reply.message and not reply.session
    assert assistant._unsent_request is None and assistant._request_generation == generation + 1
    assert writes(assistant.bridge) == []
    consent = assistant.handle("use Luna for this Codex chat")
    assert not consent.review and not assistant.session.active
    assert len([call for call in assistant.bridge.calls if call[0] == "guarded"]) == 1
    assert writes(assistant.bridge) == []


def test_idle_permission_recovery_does_not_intercept_bare_music_stop(tmp_path):
    assistant = permission_assistant(tmp_path)
    assistant.handle("Send a prompt in the Codex chat to inspect logs without edits")
    assistant.submit_confirmed(assistant.consume("send it"))
    await_worker(assistant)
    held, generation = assistant._unsent_request, assistant._request_generation
    assistant._router = SimpleNamespace(route=lambda _: None)
    assert assistant.handle("stop") is None
    assert assistant._unsent_request is held and assistant._request_generation == generation
    assert writes(assistant.bridge) == []


@pytest.mark.parametrize("mode", ["cancel", "new_request", "selection", "expiry", "hardcap"])
def test_permission_recovery_never_resurrects_changed_cancelled_or_expired_request(tmp_path, mode):
    assistant = permission_assistant(tmp_path)
    assistant.handle("Send a prompt in the Codex chat to inspect logs without edits")
    assistant.submit_confirmed(assistant.consume("send it"))
    await_worker(assistant)
    held = assistant._unsent_request
    assert held
    if mode == "cancel":
        assistant.handle("cancel Codex")
        await_worker(assistant)
    elif mode == "new_request":
        assistant.handle("Send a prompt in the Codex chat to do another task")
    elif mode == "selection":
        assistant.session.select("thr_newer", "Newer")
    elif mode == "expiry":
        assistant.session.clock = lambda: held.expires_at
    else:
        assistant.session.clock = lambda: held.hard_deadline
    reply = assistant.handle("use Luna for this Codex chat")
    assert not reply.review
    assert not assistant.session.active and assistant._unsent_request is None
    assert len([call for call in assistant.bridge.calls if call[0] == "guarded"]) == 1


@pytest.mark.parametrize("error_kind", ["unknown", "auth", "quota", "scope", "generic"])
def test_only_known_unsent_model_permission_is_recoverable(tmp_path, error_kind):
    from assistant.codex_bridge import CodexError, DeliveryUnknown
    assistant = targeted_assistant(tmp_path)
    def fail(target, prompt):
        if error_kind == "unknown":
            raise DeliveryUnknown("private delivery state")
        if error_kind == "generic":
            raise RuntimeError("private generic state")
        raise CodexError("private error state", code=error_kind)
    assistant.bridge.queue_existing = fail
    assistant.handle("Send a prompt in the Codex chat to inspect logs")
    assistant.submit_confirmed(assistant.consume("send it"))
    await_worker(assistant)
    assert assistant._unsent_request is None
    reply = assistant.handle("use Luna for this Codex chat")
    assert not reply.review and not assistant.session.active
    assert "private" not in assistant._notice


@pytest.mark.parametrize("mode", ["cancel", "new_request", "changed_back", "hardcap"])
def test_permission_failure_completed_but_not_polled_cannot_revive_newer_intent(tmp_path, mode):
    from assistant.codex_bridge import CodexError
    assistant = targeted_assistant(tmp_path)
    failed = threading.Event()
    def fail(target, prompt):
        failed.set()
        raise CodexError("private model state", code="model_permission")
    assistant.bridge.queue_existing = fail
    assistant.handle("Send a prompt in the Codex chat to inspect logs")
    cap = assistant.session.hard_deadline
    assistant.submit_confirmed(assistant.consume("send it"))
    assert failed.wait(2)
    if mode == "cancel":
        assistant.handle("cancel Codex")
    elif mode == "new_request":
        assistant.handle("Send a prompt in the Codex chat to inspect other logs")
    elif mode == "changed_back":
        assistant.session.select("thr_other", "Other")
        assistant.session.select("thr_previous", "Previous")
    else:
        assistant.session.clock = lambda: cap
    await_worker(assistant)
    assert assistant._unsent_request is None


def test_restored_permission_draft_requires_new_confirmation_not_old_send_token(tmp_path):
    assistant = permission_assistant(tmp_path)
    assistant.handle("Send a prompt in the Codex chat to inspect logs")
    old = assistant.consume("send it")
    assistant.submit_confirmed(old)
    await_worker(assistant)
    reply = assistant.handle("use Luna for this Codex chat")
    assert reply.review and reply.session
    assert len([call for call in assistant.bridge.calls if call[0] == "guarded"]) == 1
    assert "Nothing sent" in assistant.submit_confirmed(old)
    assert len([call for call in assistant.bridge.calls if call[0] == "guarded"]) == 1


@pytest.mark.parametrize("can_submit", [True, False])
def test_confirmed_creation_naming_warning_preserves_chat_and_never_duplicates(tmp_path, can_submit):
    bridge = FakeBridge()
    create = bridge.create_thread
    def warning(title):
        return dict(create(title), name="Untitled Codex chat",
                    warning="The requested chat name could not be confirmed.", can_submit=can_submit)
    bridge.create_thread = warning
    assistant = inferred_assistant(tmp_path, bridge=bridge)
    assistant.handle("Make a Codex conversation to explain this project")
    candidate = assistant.session.consume("send it")
    assert assistant.submit_confirmed(candidate) == "Sending the confirmed prompt."
    await_worker(assistant)
    assert assistant._created_thread_id == "thr_created"
    assert assistant.session.target_id == "thr_created"
    assert len([call for call in writes(bridge) if call[0] == "create"]) == 1
    assert bool([call for call in writes(bridge) if call[0] == "start"]) is can_submit
    assert "name could not be confirmed" in assistant._notice
    if not can_submit:
        assert "No prompt sent" in assistant._notice
    assert "Nothing sent" in assistant.submit_confirmed(candidate)
    assert len([call for call in writes(bridge) if call[0] == "create"]) == 1


def test_plain_create_review_binds_opening_then_emits_actual_native_receipt(tmp_path):
    opened = []
    assistant = make_assistant(tmp_path, navigator=lambda ident: opened.append(ident) or "Requested the chat in Codex.")
    source = "Create a new Codex chat."
    reply = assistant.handle(source)
    assert reply.review and reply.session and source in reply.message
    assert "gpt-5.6-luna" in reply.message and "request its Codex view" in reply.message
    assert not writes(assistant.bridge) and not opened
    candidate = assistant.session.consume("send it")
    assistant.submit_confirmed(candidate)
    await_worker(assistant)
    receipt = assistant.take_receipt()
    assert receipt["id"] == "thr_created" and not receipt["id"].startswith("thr_voice_plan_")
    assert receipt["model"] == "gpt-5.6-luna" and receipt["submitted"] is False
    assert receipt["turn_id"] is None and "prompt" not in receipt
    assert opened == ["thr_created"] and writes(assistant.bridge) == [("create", "Memoir voice task")]
    assert "Nothing sent" in assistant.submit_confirmed(candidate)


def test_requested_six_luna_is_reviewed_and_passed_without_substitution(tmp_path):
    assistant = inferred_assistant(tmp_path, action="create", selected=False)
    source = "Make a new Codex chat. Use GPT-6 Luna."
    reply = assistant.handle(source)
    assert source in reply.message and "using gpt-6-luna" in reply.message
    assistant.submit_confirmed(assistant.session.consume("send it"))
    await_worker(assistant)
    assert writes(assistant.bridge) == [("create", "Memoir voice task", "gpt-6-luna")]
    assert assistant.take_receipt()["model"] == "gpt-6-luna"


def test_expensive_model_request_is_refused_without_creating_or_substituting(tmp_path):
    assistant = inferred_assistant(tmp_path, action="create", selected=False)
    reply = assistant.handle("Make a new Codex chat. Use GPT-6 Astra.")
    assert not reply.session and "not a permitted Luna" in reply.message
    assert not writes(assistant.bridge)


def test_open_just_created_recovers_only_validated_navigation_identity(tmp_path):
    bridge = FakeBridge()
    receipt = {"id": "thr_saved", "name": "Saved", "model": "gpt-6-luna", "submitted": False, "turn_id": None}
    bridge.last_created = lambda: dict(receipt)
    bridge.read_thread = lambda ident: {"id": ident, "cwd": str(tmp_path), "name": "Saved"}
    opened = []
    assistant = make_assistant(tmp_path, bridge, navigator=lambda ident: opened.append(ident) or "Requested the chat in Codex.")
    assert not bridge.owned_threads and not assistant.session.target_id
    assistant.handle("Open the chat you just made")
    await_worker(assistant)
    assert opened == ["thr_saved"] and assistant.session.target_id == "thr_saved"
    assert not bridge.owned_threads and not writes(bridge)


def test_saved_created_chat_outside_scope_never_opens_or_transfers_ownership(tmp_path):
    bridge = FakeBridge()
    bridge.last_created = lambda: {"id": "thr_saved", "name": "Saved", "model": "gpt-6-luna"}
    bridge.read_thread = lambda ident: {"id": ident, "cwd": str(tmp_path.parent)}
    opened = []
    assistant = make_assistant(tmp_path, bridge, navigator=opened.append)
    assistant.handle("Open the chat you just made")
    await_worker(assistant)
    assert not opened and not bridge.owned_threads and not writes(bridge)


def test_named_explicit_open_finishes_in_one_request_without_sending(tmp_path):
    bridge = FakeBridge()
    bridge.list_threads = lambda: [{"id": "thr_named", "name": "Build", "cwd": str(tmp_path)}]
    opened = []
    assistant = make_assistant(tmp_path, bridge, navigator=lambda ident: opened.append(ident) or "Requested the chat in Codex.")
    assistant.handle("Open Codex chat Build")
    await_worker(assistant)
    assert opened == ["thr_named"] and assistant.session.target_id == "thr_named"
    assert not writes(bridge)


def test_late_confirmed_creation_does_not_override_newer_selection_or_open(tmp_path):
    bridge = FakeBridge()
    bridge.block_create = True
    opened = []
    assistant = make_assistant(tmp_path, bridge, navigator=opened.append)
    assistant.handle("Create a Codex chat")
    assistant.submit_confirmed(assistant.session.consume("send it"))
    assert bridge.create_started.wait(2)
    assistant.session.select("thr_newer", "Newer explicit choice")
    bridge.create_release.set()
    await_worker(assistant)
    assert not opened and assistant.session.target_id == "thr_newer"
    assert assistant.take_receipt()["id"] == "thr_created"


def test_followup_in_owned_six_chat_reviews_its_bound_model(tmp_path):
    bridge = FakeBridge()
    bridge.owned_threads.add("thr_six")
    bridge.bound_model = lambda ident: "gpt-6-luna" if ident == "thr_six" else ""
    assistant = make_assistant(tmp_path, bridge)
    assistant.session.select("thr_six", "Six")
    reply = assistant.handle("ask Codex Check the logs")
    assert reply.session and "using gpt-6-luna" in reply.message
    assert assistant._pending_plan.model == "gpt-6-luna"
    assert not writes(bridge)


def test_explicit_open_plus_named_task_keeps_entire_source_until_confirmation(tmp_path):
    bridge = FakeBridge()
    bridge.list_threads = lambda: [{"id": "thr_named", "name": "Build", "cwd": str(tmp_path)}]
    opened = []
    assistant = make_assistant(tmp_path, bridge, navigator=opened.append)
    source = "Open the Build chat and review the logs without changing them"
    assistant._router = SimpleNamespace(route=lambda text: GlobalCommand("ask", text, inferred=True,
                                        target_kind="named", target_name="Build"))
    reply = assistant.handle(source)
    await_worker(assistant)
    continuation = assistant.continue_creation(reply.await_token)
    assert continuation.session and source in continuation.message
    assert assistant._pending_plan.final_prompt.endswith(source)
    assert assistant._pending_plan.show_chat and not writes(bridge) and not opened
    expected = assistant._pending_plan.final_prompt
    assistant.submit_confirmed(assistant.session.consume("send it"))
    await_worker(assistant)
    assert opened == ["thr_named"]
    assert writes(bridge) == [("queue", "thr_named", expected)]
    notice = assistant.take_notice()
    assert "returned no receipt" in notice and "Requested the chat in Codex" not in notice
    assert assistant.session.target_id == "thr_named"


def test_compound_named_creation_keeps_the_task_out_of_empty_chat_title(tmp_path):
    bridge = FakeBridge()
    assistant = make_assistant(tmp_path, bridge)
    source = "Create a Codex chat called Build Notes and check the failing tests"
    routed = []
    def route(text):
        routed.append(text)
        return GlobalCommand("create_ask", text, inferred=True, target_kind="new")
    assistant._router = SimpleNamespace(route=route)
    reply = assistant.handle(source)
    assert routed == [source] and reply.session and source in reply.message
    expected = assistant._pending_plan.final_prompt
    assert expected.endswith(source) and not writes(bridge)
    confirmation = assistant.session.consume("send it")
    assistant.submit_confirmed(confirmation)
    await_worker(assistant)
    assert writes(bridge)[-1] == ("start", "thr_created", expected)
    assistant.submit_confirmed(confirmation)
    assert len([call for call in writes(bridge) if call[0] == "create"]) == 1


def test_open_current_with_task_requires_review_of_the_whole_source(tmp_path):
    bridge = FakeBridge()
    opened = []
    assistant = make_assistant(tmp_path, bridge, navigator=opened.append)
    assistant.session.select("thr_current", "Current")
    source = "Open the current Codex chat to review the failing tests"
    assistant._router = SimpleNamespace(route=lambda text: GlobalCommand(
        "ask", text, inferred=True, target_kind="selected"))
    reply = assistant.handle(source)
    assert reply.session and source in reply.message
    expected = assistant._pending_plan.final_prompt
    assert expected.endswith(source) and not writes(bridge) and not opened
    assistant.submit_confirmed(assistant.session.consume("send it"))
    await_worker(assistant)
    assert writes(bridge) == [("queue", "thr_current", expected)]
    assert opened == ["thr_current"]


def test_cancelled_async_open_cannot_navigate_on_late_result(tmp_path):
    bridge = FakeBridge()
    entered, release = threading.Event(), threading.Event()
    def rows():
        entered.set()
        assert release.wait(2)
        return [{"id": "thr_named", "name": "Build", "cwd": str(tmp_path)}]
    bridge.list_threads = rows
    opened = []
    assistant = make_assistant(tmp_path, bridge, navigator=opened.append)
    assistant.handle("Open Codex chat Build")
    assert entered.wait(2)
    assistant.handle("cancel Codex")
    release.set()
    await_worker(assistant)
    assert not opened and not assistant.session.target_id and not writes(bridge)


def test_configured_executable_is_threaded_to_bridge_without_startup_work(tmp_path, monkeypatch):
    import assistant.global_assistant as module
    constructed = []
    bridge = FakeBridge()
    monkeypatch.setattr(module, "CodexBridge", lambda root, **options: constructed.append(options) or bridge)
    module.GlobalAssistant({"codex.executable": "explicit-codex", "codex.model": "gpt-6-luna"},
                           tmp_path, navigator=lambda ident: "Requested")
    assert constructed == [{"model": "gpt-6-luna", "executable": "explicit-codex"}]
    assert not bridge.calls
