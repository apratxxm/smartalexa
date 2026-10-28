"""Phase 4 public-knowledge retrieval for answers. Non-mutating; not a capability.

Only the question text is sent to Wikipedia: one search request and at most one
summary request for the top hit. No other hosts, no redirects, no private context.
"""
from __future__ import annotations

from dataclasses import dataclass
import html
import json
import re
from typing import Callable
from urllib.parse import quote
from urllib.request import HTTPRedirectHandler, HTTPSHandler, ProxyHandler, Request, build_opener

from .base import Adapter, Guard

_SEARCH_BASE = "https://en.wikipedia.org/w/rest.php/v1/search/page"
_SUMMARY_BASE = "https://en.wikipedia.org/api/rest_v1/page/summary/"
_ARTICLE_BASE = "https://en.wikipedia.org/wiki/"
_USER_AGENT = "Memoir/0.1 (local assistant; public knowledge lookup)"
_MAX_TIMEOUT_SECONDS = 2.5
_MAX_RESPONSE_BYTES = 128 * 1024
_MAX_QUESTION_CHARS = 300
_MAX_TITLE_CHARS = 200
_MAX_TEXT_CHARS = 1200
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_TAG = re.compile(r"<[^>]*>")


_PRIVATE = re.compile(r"\b(?:i|me|my|mine|we|our|us|you|your|password|code|pin|address|phone|account|secret|otp|one[\s-]+time|2fa|mfa|api|key|token|passphrase|recovery|seed|private|ssn|cvv|iban|login|username|credentials)\b", re.I)
_QUESTION = re.compile(r"^(?:what\s+(?:is|are|was|were)|who\s+(?:is|was|were)|where\s+(?:is|was)|when\s+(?:was|did)|how\s+(?:does|do|did|many|much)|tell\s+me\s+about|define|explain)\s+(.+)$", re.I)
_MEANING = re.compile(r"^what\s+does\s+(.+?)\s+mean$", re.I)


def public_topic(text: str) -> str | None:
    if not isinstance(text, str):
        return None
    text = text.strip()
    text = re.sub(r"^(?:(?:hey|hi|okay|ok)\s+)?(?:memoir|alexa)[,\s]+", "", text, flags=re.I)
    text = re.sub(r"^(?:(?:please|kindly)\s*[, ]\s*)+", "", text, flags=re.I)
    text = text.rstrip(" \t.!?")
    text = re.sub(r"(?:[,\s]+(?:please|thanks))+$", "", text, flags=re.I).rstrip(" \t.!?")
    privacy_text = re.sub(r"^tell\s+me\s+about\b", "", text, flags=re.I)
    if _PRIVATE.search(privacy_text):
        return None
    for token in re.findall(r"\w+", privacy_text):
        if (re.search(r"\d{4,}", token)
                or (any(c.isalpha() for c in token)
                    and any(c.isdigit() or c == "_" for c in token))):
            return None
    match = _MEANING.fullmatch(text) or _QUESTION.fullmatch(text)
    return match.group(1).strip() if match and match.group(1).strip() else None


@dataclass(frozen=True)
class Passage:
    title: str
    url: str
    text: str


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # any 3xx becomes HTTPError, so the host cannot change


_OPENER = build_opener(ProxyHandler({}), HTTPSHandler(), _NoRedirect())


def _default_open(request: Request, timeout: float) -> bytes:
    response = _OPENER.open(request, timeout=timeout)
    try:
        return response.read(_MAX_RESPONSE_BYTES + 1)
    finally:
        response.close()


def _clean(value: str, limit: int) -> str:
    text = _TAG.sub("", value)
    text = html.unescape(text)
    text = _CONTROL.sub(" ", text)
    return " ".join(text.split())[:limit]


def _title_path(title: str) -> str:
    return quote(title.replace(" ", "_"), safe="")


class KnowledgeSource:
    """Retrieve up to one bounded Wikipedia passage for a question."""

    def __init__(self, open_url: Callable[[Request, float], bytes] | None = None):
        self._open = open_url or _default_open

    def retrieve(self, question: str, guard: Guard) -> list[Passage]:
        try:
            query = self._question(question)
            if query is None:
                return []
            title = self._search(query, guard)
            if title is None:
                return []
            text = self._summary(title, guard)
            if not text:
                return []
            return [Passage(title, _ARTICLE_BASE + _title_path(title), text)]
        except Exception:
            return []

    def _question(self, question) -> str | None:
        if not isinstance(question, str):
            return None
        query = question.strip()
        if not 1 <= len(query) <= _MAX_QUESTION_CHARS or _CONTROL.search(query):
            return None
        return query

    def _budget(self, guard: Guard) -> float | None:
        remaining = guard()
        if remaining <= 0:
            return None
        return min(float(remaining), _MAX_TIMEOUT_SECONDS)

    def _get_json(self, url: str, guard: Guard):
        timeout = self._budget(guard)
        if timeout is None:
            return None
        request = Request(url, headers={"User-Agent": _USER_AGENT, "Accept": "application/json"},
                          method="GET")
        body = self._open(request, timeout)
        if not isinstance(body, bytes) or len(body) > _MAX_RESPONSE_BYTES:
            return None
        return json.loads(body.decode("utf-8"))

    def _search(self, query: str, guard: Guard) -> str | None:
        url = _SEARCH_BASE + "?q=" + quote(query, safe="") + "&limit=3"
        payload = self._get_json(url, guard)
        pages = payload.get("pages") if isinstance(payload, dict) else None
        if not isinstance(pages, list):
            return None
        for page in pages:
            raw = page.get("title") if isinstance(page, dict) else None
            if not isinstance(raw, str):
                continue
            title = _clean(raw, _MAX_TITLE_CHARS)
            if title:
                return title
        return None

    def _summary(self, title: str, guard: Guard) -> str:
        payload = self._get_json(_SUMMARY_BASE + _title_path(title), guard)
        extract = payload.get("extract") if isinstance(payload, dict) else None
        if not isinstance(extract, str):
            return ""
        return _clean(extract, _MAX_TEXT_CHARS)


KNOWLEDGE = KnowledgeSource()

# No capability handlers: knowledge is consulted by the engine, not registered as an action.
ADAPTER = Adapter("knowledge", "4", handlers={})

__all__ = ["ADAPTER", "KNOWLEDGE", "KnowledgeSource", "Passage"]
