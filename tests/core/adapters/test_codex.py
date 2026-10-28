import pytest

from assistant.core.adapters.base import Clause, Receipt
from assistant.core.adapters.codex import ADAPTER, CodexRequest
import assistant.core.adapters.codex as codex_module
from assistant.codex_bridge import CodexError, DeliveryUnknown, DeadlineExpired
from assistant.core.playback import TurnExpired


handler = ADAPTER.handlers["codex_request"]


@pytest.mark.parametrize("verb,obj", [
    ("ask codex to", "fix the login bug"),
    ("ask codex to", "fix the login bug please"),
    ("tell codex to", "repair the broken tests"),
    ("get codex to", "refactor the parser code"),
    ("have codex", "inspect the application logs"),
    ("send codex", "investigate the failing build"),
    ("give codex", "review the database migration"),
    ("ask codex to", "Fix  the Login BUG"),
    ("ask codex to", "fix my login page"),
    ("ask codex to", "fix the login bug in the settings page"),
    ("ask codex to", "fix the login page for me"),
])
def test_exact_licensed_object(verb, obj):
    assert handler.license(Clause(verb + " " + obj, verb, obj)) == {"request": obj}


@pytest.mark.parametrize("text,obj", [
    ("Ask Codex to fix the tests, but don't send it yet", "fix the tests, but don't send it yet"),
    ("Ask Codex to fix the tests? What does that mean?", "fix the tests? What does that mean?"),
    ('"ask codex to fix the tests"', "fix the tests"),
    ("he said ask codex to fix the tests", "fix the tests"),
    ("ask codex to fix the tests later", "fix the tests later"),
    ("ask codex to fix the tests not yet", "fix the tests not yet"),
    ("ask codex to fix the tests wait", "fix the tests wait"),
    ("ask codex to fix the tests hold on", "fix the tests hold on"),
    ("ask codex to fix the tests never mind", "fix the tests never mind"),
    ("ask codex to fix the tests or not", "fix the tests or not"),
    ("ask codex to maybe fix the tests", "maybe fix the tests"),
    ("ask codex to fix the tests should I", "fix the tests should I"),
    ("ask codex to fix the tests what is this", "fix the tests what is this"),
    ("ask codex to fix the tests can you explain", "fix the tests can you explain"),
    ("ask codex to fix the tests just kidding", "fix the tests just kidding"),
    ("ask codex to fix the tests cancel", "fix the tests cancel"),
    ("ask codex to do not send this", "do not send this"),
    ("ask codex to fix this?", "fix this?"),
    ("ask codex to fix tests", "fix tests"),
    ("ask codex to fix the tests if I ask", "fix the tests if I ask"),
    ("ask codex to fix the tests unless I stop", "fix the tests unless I stop"),
    ("ask codex to fix the tests; then stop", "fix the tests; then stop"),
])
def test_refuses_ambiguity(text, obj):
    assert handler.license(Clause(text, "ask codex to", obj)) is None


@pytest.mark.parametrize("form", [
    "i", "im", "i'm", "ive", "i've", "id", "i'd", "i'll", "ill",
    "I\u2019m", "I\u2019ve", "I\u2019d", "I\u2019ll",
    "joke", "joking", "jest", "example", "sample", "take that back",
    "take it back", "take back", "retract", "not serious", "seriously",
    "kidding", "that was", "it was", "was just",
    "I take that back", "that was a joke", "was just an example",
    "I introduced",
])
def test_refuses_structural_withdrawal_forms(form):
    obj = "fix the tests " + form
    assert handler.license(Clause("ask codex to " + obj, "ask codex to", obj)) is None


@pytest.mark.parametrize("text", ["what did codex say", "is codex done?",
                                  "tell me about codex", "codex, refactor the parser"])
def test_nonmatching_or_not_yet_supported_verbs(text):
    assert handler.license(Clause(text, text, "refactor the parser")) is None


def test_replies_are_fixed():
    assert [handler.reply(Receipt(state), {"request": "untrusted"}) for state in
            ("observed", "requested", "unknown", "failed")] == [
                "Sent to Codex.", "I asked Codex but can't confirm it got the request.",
                "I asked Codex but can't confirm it got the request.", "I couldn't reach Codex."]


class FakeBridge:
    def __init__(self, root):
        self.root = root
        self.active_turn = None
        self.mutation_uncertain = False
        self.calls = []
        self.create_error = None
        self.turn_error = None

    def create_thread(self, *, title, sandbox, guard):
        self.calls.append(("create", title))
        if self.create_error:
            raise self.create_error
        return {"id": "thread1"}

    def start_turn(self, ident, request, *, guard):
        self.calls.append(("send", ident, request))
        if self.turn_error:
            raise self.turn_error
        return {"id": "turn1"}


def setup_handler(tmp_path, *, foreground=True):
    bridge = FakeBridge(tmp_path)
    calls = []
    def raise_window(guard):
        calls.append("foreground")
        return foreground
    config_path = tmp_path / "config.toml"
    config_path.write_text('sandbox_mode = "workspace-write"')
    adapter = CodexRequest(project_dir=tmp_path, bridge_factory=lambda root: bridge, raise_window=raise_window, config_path=config_path)
    return adapter, bridge, calls


@pytest.mark.parametrize("foreground,state", [(True, "observed"), (False, "requested")])
def test_confirmed_submission_and_foreground(tmp_path, foreground, state):
    adapter, bridge, calls = setup_handler(tmp_path, foreground=foreground)
    guards = []
    def guard():
        guards.append(len(bridge.calls))
        return 10
    text = "Fix  the login bug with careful tests please"
    args = adapter.license(Clause("ask codex to " + text, "ask codex to", text))
    assert adapter.execute(args, guard).state == state
    assert guards == [0, 1]
    assert bridge.calls == [("create", "Fix the login bug with careful"), ("send", "thread1", codex_module._VOICE_PREAMBLE + text)]
    assert calls == ["foreground"]


@pytest.mark.parametrize("value", [None, "", "relative/path"])
def test_missing_or_invalid_project_fails_without_bridge(value):
    adapter = CodexRequest(project_dir=value if value is not None else "", bridge_factory=lambda _: pytest.fail("bridge created"))
    assert adapter.execute({"request": "fix the tests"}, lambda: 1).state == "failed"


@pytest.mark.parametrize("attribute,value", [("active_turn", ("thread", "turn")), ("mutation_uncertain", True)])
def test_busy_does_not_dispatch(tmp_path, attribute, value):
    adapter, bridge, _ = setup_handler(tmp_path)
    setattr(bridge, attribute, value)
    receipt = adapter.execute({"request": "fix the tests"}, lambda: 1)
    assert receipt.state == "failed" and not bridge.calls
    assert adapter.reply(receipt, {}) == "Codex is still busy with the last request."


def test_expired_after_creation_does_not_submit(tmp_path):
    adapter, bridge, _ = setup_handler(tmp_path)
    budgets = iter([1, 0])
    assert adapter.execute({"request": "fix the tests"}, lambda: next(budgets)).state == "unknown"
    assert bridge.calls == [("create", "fix the tests")]


@pytest.mark.parametrize("error", [DeliveryUnknown(), RuntimeError("error")])
def test_creation_error_never_retries(tmp_path, error):
    adapter, bridge, _ = setup_handler(tmp_path)
    bridge.create_error = error
    assert adapter.execute({"request": "fix the tests"}, lambda: 1).state == "unknown"
    assert bridge.calls == [("create", "fix the tests")]


def test_deadline_during_creation_is_failed(tmp_path):
    adapter, bridge, calls = setup_handler(tmp_path)
    bridge.create_error = DeadlineExpired("before dispatch")
    receipt = adapter.execute({"request": "fix the tests"}, lambda: 1)
    assert (receipt.state, receipt.detail) == ("failed", "deadline_expired")
    assert bridge.calls == [("create", "fix the tests")]
    assert calls == []


def test_deadline_in_start_turn_is_unknown_after_creation(tmp_path):
    adapter, bridge, calls = setup_handler(tmp_path)
    bridge.turn_error = DeadlineExpired("before turn dispatch")
    receipt = adapter.execute({"request": "fix the tests"}, lambda: 1)
    assert (receipt.state, receipt.detail) == ("unknown", "expired_after_creation")
    assert len(bridge.calls) == 2
    assert calls == []


@pytest.mark.parametrize("error,state", [(DeliveryUnknown(), "unknown"), (CodexError("rejected"), "unknown"), (RuntimeError(), "unknown")])
def test_submission_error_never_retries(tmp_path, error, state):
    adapter, bridge, calls = setup_handler(tmp_path)
    bridge.turn_error = error
    assert adapter.execute({"request": "fix the tests"}, lambda: 1).state == state
    assert len(bridge.calls) == 2 and calls == []


def test_foreground_exception_is_unknown_and_never_resends(tmp_path):
    adapter, bridge, _ = setup_handler(tmp_path)
    def fail(guard):
        raise RuntimeError("focus")
    adapter._raise_window = fail
    assert adapter.execute({"request": "fix the tests"}, lambda: 1).state == "unknown"
    assert len(bridge.calls) == 2


def test_guard_expired_before_creation(tmp_path):
    adapter, bridge, _ = setup_handler(tmp_path)
    def expired():
        raise TurnExpired()
    assert adapter.execute({"request": "fix the tests"}, expired).state == "failed"
    assert not bridge.calls


@pytest.mark.parametrize("contents,allowed", [
    ('sandbox_mode = "read-only"', False),
    ('sandbox_mode = "workspace-write"', True),
    ('sandbox_mode = "danger-full-access"', True),
    ('sandbox_mode = "invalid"', False),
    ('[other]\nsandbox_mode = "workspace-write"', False),
    ('sandbox_mode = [', False),
    (None, False),
])
def test_sandbox_default_ceiling(tmp_path, contents, allowed):
    adapter, bridge, _ = setup_handler(tmp_path)
    if contents is None:
        adapter._config_path.unlink()
    else:
        adapter._config_path.write_text(contents)
    receipt = adapter.execute({"request": "fix the login bug"}, lambda: 10)
    assert bool(bridge.calls) is allowed
    assert receipt.state == ("observed" if allowed else "failed")


def test_effect_words_do_not_block_codex_name():
    assert "codex" not in handler.effect_words
    assert {"submitted", "forwarded", "sent", "asked", "delegated", "started"} <= set(handler.effect_words)
