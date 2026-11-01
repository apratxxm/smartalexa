"""Contract tests for the phase 4 Jam and knowledge adapters. All I/O is faked."""
from __future__ import annotations

import json
import socket
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlsplit

import pytest

from assistant import computer_tools
from assistant.computer_tools import ComputerControlError, ComputerDeliveryUnknown
from assistant.core.adapters import jam, knowledge
from assistant.core.adapters.base import Adapter, Clause, Receipt, TurnExpired
from assistant.core.adapters.knowledge import KnowledgeSource, Passage


# ---------- Jam ----------

def _clause(verb, obj=""):
    return Clause(text=f"{verb} {obj}".strip(), verb=verb, object=obj)


def _guard_recording(events, value=10.0):
    def guard():
        events.append("guard")
        return value
    return guard


def test_jam_adapter_registration_shape():
    assert isinstance(jam.ADAPTER, Adapter)
    assert jam.ADAPTER.phase == "4"
    assert set(jam.ADAPTER.handlers) == {"start_jam"}
    handler = jam.ADAPTER.handlers["start_jam"]
    assert handler.mutating is True
    assert handler.object_kind == "none"
    assert set(handler.effect_words) == {"started", "starting", "showing", "jam"}


@pytest.mark.parametrize("verb", ["start a jam", "start jam", "show the jam code", "show the jam qr"])
def test_jam_license_accepts_each_verb_without_object(verb):
    handler = jam.START_JAM
    assert handler.license(_clause(verb)) == {}
    # Pure: same answer on repeat, no state.
    assert handler.license(_clause(verb)) == {}


@pytest.mark.parametrize("clause", [
    _clause("start a jam", "tonight"),
    _clause("play"),
    _clause("show the jam"),
    _clause("stop jam"),
])
def test_jam_license_refuses_objects_and_other_verbs(clause):
    assert jam.START_JAM.license(clause) is None


def test_jam_guard_is_called_before_helper_launch(monkeypatch):
    events = []

    def fake_helper(*, deadline):
        events.append("helper")
        return "Jam QR is on screen."

    monkeypatch.setattr(computer_tools, "start_spotify_jam", fake_helper)
    receipt = jam.START_JAM.execute({}, _guard_recording(events))
    assert events == ["guard", "helper"]
    assert receipt.state == "observed"


def test_jam_expired_guard_fails_without_launch(monkeypatch):
    calls = []
    monkeypatch.setattr(computer_tools, "start_spotify_jam", lambda *, deadline: calls.append(1) or "ok")

    def stale():
        raise TurnExpired("stale")

    receipt = jam.START_JAM.execute({}, stale)
    assert receipt.state == "failed"
    assert calls == []


def test_jam_zero_budget_fails_without_launch(monkeypatch):
    calls = []
    monkeypatch.setattr(computer_tools, "start_spotify_jam", lambda *, deadline: calls.append(1) or "ok")
    receipt = jam.START_JAM.execute({}, lambda: 0.0)
    assert receipt.state == "failed"
    assert calls == []


def test_jam_nonempty_args_fail_without_launch(monkeypatch):
    calls = []
    monkeypatch.setattr(computer_tools, "start_spotify_jam", lambda *, deadline: calls.append(1) or "ok")
    receipt = jam.START_JAM.execute({"query": "x"}, _guard_recording([]))
    assert receipt.state == "failed"
    assert calls == []


def test_jam_helper_ok_maps_to_observed_and_hides_helper_text(monkeypatch):
    monkeypatch.setattr(computer_tools, "start_spotify_jam", lambda *, deadline: "Helper said SECRET-TEXT")
    receipt = jam.START_JAM.execute({}, _guard_recording([]))
    assert receipt.state == "observed"
    assert "SECRET-TEXT" not in receipt.detail
    assert "SECRET-TEXT" not in receipt.content
    assert "SECRET-TEXT" not in receipt.error
    assert jam.START_JAM.reply(receipt, {}) == "Jam started; the QR code is on screen."


def test_jam_unverified_helper_result_maps_to_unknown(monkeypatch):
    def helper(*, deadline):
        raise ComputerDeliveryUnknown("raw helper text")

    monkeypatch.setattr(computer_tools, "start_spotify_jam", helper)
    receipt = jam.START_JAM.execute({}, _guard_recording([]))
    assert receipt.state == "unknown"
    reply = jam.START_JAM.reply(receipt, {})
    assert reply == "I asked Spotify to start a Jam but can't confirm it."
    assert "raw helper text" not in reply


def test_jam_helper_start_refusal_maps_to_failed(monkeypatch):
    def helper(*, deadline):
        raise ComputerControlError("could not be started")

    monkeypatch.setattr(computer_tools, "start_spotify_jam", helper)
    receipt = jam.START_JAM.execute({}, _guard_recording([]))
    assert receipt.state == "failed"


def test_jam_unexpected_exception_after_launch_maps_to_unknown(monkeypatch):
    def helper(*, deadline):
        raise RuntimeError("boom")

    monkeypatch.setattr(computer_tools, "start_spotify_jam", helper)
    receipt = jam.START_JAM.execute({}, _guard_recording([]))
    assert receipt.state == "unknown"


def test_jam_reply_templates_only():
    handler = jam.START_JAM
    assert handler.reply(Receipt("observed"), {}) == "Jam started; the QR code is on screen."
    assert handler.reply(Receipt("requested"), {}) == "I asked Spotify to start a Jam but can't confirm it."
    assert handler.reply(Receipt("unknown"), {}) == "I asked Spotify to start a Jam but can't confirm it."
    assert handler.reply(Receipt("failed"), {}) == "I couldn't start the Spotify Jam."


def test_jam_reply_never_claims_success_without_observed():
    for state in ("requested", "unknown", "failed"):
        reply = jam.START_JAM.reply(Receipt(state), {}).lower()
        assert "started;" not in reply and "qr code is on screen" not in reply


# ---------- Knowledge ----------

class FakeOpen:
    """Stands in for the urllib transport; routes by URL and records each request."""

    def __init__(self, search=None, summary=None, error=None):
        self.search = search
        self.summary = summary
        self.error = error
        self.requests = []

    def __call__(self, request, timeout):
        self.requests.append((request.full_url, timeout))
        if self.error is not None:
            raise self.error
        url = request.full_url
        if "/w/rest.php/v1/search/page" in url:
            body = self.search
        elif "/api/rest_v1/page/summary/" in url:
            body = self.summary
        else:
            raise AssertionError("unexpected URL " + url)
        if body is None:
            return b"{}"
        return body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")


def _search_body(title="Ada Lovelace", excerpt="<span>mathematician</span>"):
    return {"pages": [{"id": 1, "key": title.replace(" ", "_"), "title": title, "excerpt": excerpt}]}


def _summary_body(extract="Ada Lovelace was an English mathematician."):
    return {"title": "Ada Lovelace", "extract": extract}


def _retrieve(open_url, question="who was Ada Lovelace", budget=10.0):
    return KnowledgeSource(open_url=open_url).retrieve(question, lambda: budget)


def test_knowledge_is_not_registered_as_capability():
    assert knowledge.ADAPTER.phase == "4"
    assert knowledge.ADAPTER.handlers == {}
    assert isinstance(knowledge.KNOWLEDGE, KnowledgeSource)
    assert callable(knowledge.KNOWLEDGE.retrieve)


def test_knowledge_happy_path_returns_one_bounded_passage():
    fake = FakeOpen(search=_search_body(), summary=_summary_body())
    result = _retrieve(fake)
    assert result == [Passage(
        "Ada Lovelace",
        "https://en.wikipedia.org/wiki/Ada_Lovelace",
        "Ada Lovelace was an English mathematician.",
    )]


def test_knowledge_requests_only_the_two_wikipedia_shapes():
    fake = FakeOpen(search=_search_body(), summary=_summary_body())
    _retrieve(fake)
    assert len(fake.requests) == 2
    search_url, summary_url = (url for url, _ in fake.requests)
    search = urlsplit(search_url)
    assert search.scheme == "https" and search.netloc == "en.wikipedia.org"
    assert search.path == "/w/rest.php/v1/search/page"
    assert parse_qs(search.query) == {"q": ["who was Ada Lovelace"], "limit": ["3"]}
    summary = urlsplit(summary_url)
    assert summary.netloc == "en.wikipedia.org"
    assert summary.path == "/api/rest_v1/page/summary/Ada_Lovelace"


def test_knowledge_query_is_url_encoded():
    fake = FakeOpen(search={"pages": []})
    _retrieve(fake, question="a&b=c/d?e #f")
    search_url = fake.requests[0][0]
    assert "a%26b%3Dc%2Fd%3Fe%20%23f" in search_url
    assert "&b=c" not in search_url
    assert "#" not in search_url


def test_knowledge_title_with_slash_and_question_mark_stays_on_host():
    evil = "Foo/evil.example?x"
    fake = FakeOpen(search=_search_body(title=evil), summary=_summary_body())
    result = _retrieve(fake)
    summary_url = fake.requests[1][0]
    assert urlsplit(summary_url).netloc == "en.wikipedia.org"
    assert "Foo%2Fevil.example%3Fx" in summary_url
    assert "Foo/evil" not in summary_url
    assert len(result) == 1
    assert urlsplit(result[0].url).netloc == "en.wikipedia.org"
    assert result[0].url.startswith("https://en.wikipedia.org/wiki/")
    assert "?" not in result[0].url.split("/wiki/", 1)[1]


def test_knowledge_title_with_control_characters_is_refused():
    fake = FakeOpen(search=_search_body(title="Bad\x00Title"), summary=_summary_body())
    result = _retrieve(fake)
    # The cleaned title is still a valid path element; the URL must remain fixed-host.
    for passage in result:
        assert urlsplit(passage.url).netloc == "en.wikipedia.org"
        assert "\x00" not in passage.title


def test_knowledge_makes_at_most_two_requests_with_timeout_cap():
    fake = FakeOpen(search=_search_body(), summary=_summary_body())
    _retrieve(fake, budget=10.0)
    assert len(fake.requests) == 2
    assert all(timeout == 2.5 for _, timeout in fake.requests)


def test_knowledge_timeout_is_remaining_budget_when_smaller():
    fake = FakeOpen(search=_search_body(), summary=_summary_body())
    _retrieve(fake, budget=1.0)
    assert all(timeout == 1.0 for _, timeout in fake.requests)


def test_knowledge_no_budget_makes_no_request():
    fake = FakeOpen(search=_search_body(), summary=_summary_body())
    assert _retrieve(fake, budget=0.0) == []
    assert fake.requests == []


def test_knowledge_guard_raising_returns_empty_without_request():
    fake = FakeOpen(search=_search_body(), summary=_summary_body())

    def stale():
        raise TurnExpired("stale")

    assert KnowledgeSource(open_url=fake).retrieve("who was Ada", stale) == []
    assert fake.requests == []


@pytest.mark.parametrize("question", ["", "   ", "bad\x07question", "x" * 301, None, 42])
def test_knowledge_invalid_question_is_refused_without_request(question):
    fake = FakeOpen(search=_search_body(), summary=_summary_body())
    assert KnowledgeSource(open_url=fake).retrieve(question, lambda: 10.0) == []
    assert fake.requests == []


@pytest.mark.parametrize("error", [
    OSError("network down"),
    socket.timeout("slow"),
    TimeoutError("slow"),
    HTTPError("https://en.wikipedia.org/", 500, "err", None, None),
])
def test_knowledge_transport_errors_return_empty(error):
    assert _retrieve(FakeOpen(error=error)) == []


def test_knowledge_bad_json_returns_empty():
    assert _retrieve(FakeOpen(search=b"not json")) == []


def test_knowledge_oversized_response_returns_empty():
    huge = b"{" + b" " * (128 * 1024 + 10) + b"}"
    assert _retrieve(FakeOpen(search=huge)) == []


def test_knowledge_no_pages_returns_empty_and_skips_summary():
    fake = FakeOpen(search={"pages": []}, summary=_summary_body())
    assert _retrieve(fake) == []
    assert len(fake.requests) == 1


def test_knowledge_missing_summary_extract_returns_empty():
    fake = FakeOpen(search=_search_body(), summary={"title": "Ada Lovelace"})
    assert _retrieve(fake) == []


def test_knowledge_strips_html_and_caps_text():
    long_extract = "<b>Hi</b> <script>alert(1)</script>" + ("word " * 1000)
    fake = FakeOpen(search=_search_body(), summary=_summary_body(extract=long_extract))
    (passage,) = _retrieve(fake)
    assert "<" not in passage.text and ">" not in passage.text
    assert passage.text.startswith("Hi alert(1)")
    assert len(passage.text) <= 1200


def test_knowledge_strips_html_from_title():
    fake = FakeOpen(search=_search_body(title="<i>Ada</i> Lovelace"), summary=_summary_body())
    (passage,) = _retrieve(fake)
    assert passage.title == "Ada Lovelace"
    assert "<" not in passage.url


def test_knowledge_empty_extract_returns_empty():
    fake = FakeOpen(search=_search_body(), summary=_summary_body(extract="   <br>  "))
    assert _retrieve(fake) == []


def test_knowledge_redirects_are_not_followed():
    handler = knowledge._NoRedirect()
    assert handler.redirect_request(None, None, 301, "moved", {}, "https://evil.example/") is None
