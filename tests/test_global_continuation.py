"""Canonical task creation stages locally before any Codex mutation."""
import threading
import time
from pathlib import Path

from assistant.global_assistant import GlobalAssistant
from assistant.global_commands import GlobalCommand


class FakeBridge:
    def __init__(self, *, error=False):
        self.started = threading.Event()
        self.release = threading.Event()
        self.error = error
        self.calls = []
        self.active_turn = None
        self.owned_threads = set()

    def create_thread(self, title):
        self.calls.append(("create", title))
        self.started.set()
        assert self.release.wait(2)
        if self.error:
            raise RuntimeError("private bridge details")
        self.owned_threads.add("thr_created")
        return {"id": "thr_created", "name": title}

    def start_turn(self, target, prompt):
        self.calls.append(("start", target, prompt))
        return {"id": "turn_created", "status": "accepted"}

    def drain_events(self):
        return []

    def close(self):
        pass


class CompoundRouter:
    def route(self, text):
        return GlobalCommand("create_ask", "Inspect only the selected project.", title="Exact Title")


def make_assistant(tmp_path, *, error=False):
    bridge = FakeBridge(error=error)
    cfg = {"codex.dictation_timeout_seconds": 90, "codex.max_prompt_chars": 12000,
           "codex.model": "gpt-6-luna"}
    assistant = GlobalAssistant(cfg, Path(tmp_path), bridge=bridge, router=CompoundRouter(),
                                navigator=lambda ident: "Requested the chat in Codex.")
    return assistant, bridge


def settle(assistant):
    deadline = time.monotonic() + 2
    while assistant._busy and time.monotonic() < deadline:
        assistant.poll()
        time.sleep(.005)
    assert not assistant._busy


def confirm_review(assistant):
    token = assistant.begin_feedback(review=True)
    assert assistant.end_feedback(token, True)
    return assistant.consume("send it")


def test_canonical_task_and_title_are_frozen_before_any_creation(tmp_path):
    assistant, bridge = make_assistant(tmp_path)
    initial = assistant.handle("Open a Codex chat and tell it to inspect the project")
    assert initial.session and initial.review and not initial.await_token
    assert assistant.session.draft == "Inspect only the selected project."
    assert assistant._pending_plan.title == "Exact Title"
    assert bridge.calls == []
    assert assistant.consume("send it").action == "review"
    candidate = confirm_review(assistant)
    assert assistant.submit_confirmed(candidate) == "Sending the confirmed prompt."
    assert bridge.started.wait(2)
    assert assistant.session.target_id == ""
    bridge.release.set()
    settle(assistant)
    assert bridge.calls == [("create", "Exact Title"), ("start", "thr_created", "Inspect only the selected project.")]
    assert assistant.session.target_id == "thr_created"
    assert "Nothing sent" in assistant.submit_confirmed(candidate)


def test_old_or_wrong_creation_continuation_token_cannot_authorize_task(tmp_path):
    assistant, bridge = make_assistant(tmp_path)
    initial = assistant.handle("Open a Codex chat and tell it to inspect the project")
    assert not initial.await_token
    assert not assistant.continue_creation("wrong-token").session
    assert bridge.calls == [] and assistant.session.phase == "review"


def test_canonical_cancel_and_expiry_prevent_creation_and_restore_prior(tmp_path):
    for mode in ["cancel", "expiry"]:
        assistant, bridge = make_assistant(tmp_path)
        assistant.session.select("thr_original", "Original")
        assistant.handle("Open a Codex chat and tell it to inspect the project")
        if mode == "cancel":
            assistant.consume("cancel")
        else:
            assistant.session._deadline = 0
            assert not assistant.session.active
        assert bridge.calls == [] and not assistant.session.active
        assert assistant.session.target_id == "thr_original"


def test_changed_away_and_back_target_invalidates_frozen_canonical_task(tmp_path):
    assistant, bridge = make_assistant(tmp_path)
    assistant.handle("Open a Codex chat and tell it to inspect the project")
    candidate = confirm_review(assistant)
    assistant.session.select("thr_other", "Other")
    assistant.session.clear_selection()
    assert "Nothing sent" in assistant.submit_confirmed(candidate)
    assert bridge.calls == []


def test_canonical_create_error_after_confirmation_is_uncertain_without_private_details(tmp_path):
    assistant, bridge = make_assistant(tmp_path, error=True)
    assistant.handle("Open a Codex chat and tell it to inspect the project")
    assistant.submit_confirmed(confirm_review(assistant))
    assert bridge.started.wait(2)
    bridge.release.set()
    settle(assistant)
    assert "creation could not be confirmed" in assistant._notice
    assert "private bridge details" not in assistant._notice
    assert len(bridge.calls) == 1 and not assistant.session.active
