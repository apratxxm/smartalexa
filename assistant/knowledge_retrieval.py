"""Bounded Wikipedia retrieval for explicit public-knowledge queries.

Calling this adapter sends the supplied query to Wikipedia. Callers must never
implicitly include captured audio, conversation history, credentials, or other
private context. Returned text is untrusted data and must not be executed as
instructions. This module does not synthesize answers or verify current facts.
"""
from __future__ import annotations

import json
import math
import socket
import time
import unicodedata
from dataclasses import dataclass
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import (HTTPSHandler, HTTPRedirectHandler, ProxyHandler,
                            Request, build_opener)


_ENDPOINT = "https://en.wikipedia.org/w/api.php"
_HOST = "en.wikipedia.org"
_MAX_RESPONSE_BYTES = 128 * 1024
_MAX_TIMEOUT_SECONDS = 3.0
_MAX_SOURCES = 3
_MAX_EXCERPT_CHARS = 1200
_USER_AGENT = "Memoir/0.1 (public knowledge retrieval; contact: local assistant)"


@dataclass(frozen=True)
class Source:
    id: str
    title: str
    url: str
    excerpt: str

    def __post_init__(self):
        if not isinstance(self.id, str) or self.id not in {"w1", "w2", "w3"}:
            raise ValueError("source id must be w1, w2, or w3")
        if (not isinstance(self.title, str) or not self.title.strip() or
                len(self.title) > 200 or _contains_control(self.title)):
            raise ValueError("source title must be bounded control-free text")
        if _safe_source_url(self.url) is None:
            raise ValueError("source URL must be a fixed-host Wikipedia article URL")
        if (not isinstance(self.excerpt, str) or not self.excerpt.strip() or
                len(self.excerpt) > _MAX_EXCERPT_CHARS or
                any(unicodedata.category(char) in {"Cc", "Cf"} and char != "\n"
                    for char in self.excerpt)):
            raise ValueError("source excerpt must be bounded plain text")


@dataclass(frozen=True)
class RetrievalResult:
    status: str
    sources: tuple[Source, ...] = ()

    def __post_init__(self):
        if not isinstance(self.status, str) or self.status not in {
                "ok", "not_found", "unavailable", "timeout"}:
            raise ValueError("invalid retrieval status")
        if not isinstance(self.sources, tuple) or len(self.sources) > _MAX_SOURCES:
            raise ValueError("sources must be an immutable tuple of at most three records")
        if any(not isinstance(source, Source) for source in self.sources):
            raise ValueError("sources must contain Source records")
        if (self.status == "ok") != bool(self.sources):
            raise ValueError("only ok results may contain sources, and ok requires a source")
        ids = [source.id for source in self.sources]
        urls = [source.url for source in self.sources]
        if len(set(ids)) != len(ids) or len(set(urls)) != len(urls):
            raise ValueError("source ids and URLs must be unique")


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _contains_control(value: str) -> bool:
    return any(unicodedata.category(char) == "Cc" for char in value)


def _safe_source_url(value) -> str | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = urlsplit(value)
        if (parsed.scheme != "https" or parsed.hostname != _HOST or
                parsed.username is not None or parsed.password is not None or
                parsed.port is not None or parsed.fragment or parsed.query or
                not parsed.path.startswith("/wiki/") or _contains_control(value)):
            return None
    except (ValueError, UnicodeError):
        return None
    return value


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class WikipediaRetriever:
    """Perform one fixed-host, bounded MediaWiki search request.

    ``transport`` is an internal testing seam: it receives the request and the
    remaining per-call timeout and returns a response with ``read``/``close``.
    Production uses a proxy-free, redirect-rejecting HTTPS opener.
    """

    def __init__(self, *, transport: Callable | None = None,
                 monotonic: Callable[[], float] = time.monotonic):
        self._transport = transport
        self._monotonic = monotonic

    def lookup(self, query, timeout_seconds=3.0) -> RetrievalResult:
        if (not isinstance(query, str) or isinstance(query, bool) or
                _contains_control(query)):
            raise ValueError("query must be control-free text")
        query = query.strip()
        if not 1 <= len(query) <= 300:
            raise ValueError("query must contain 1 to 300 characters")
        if (isinstance(timeout_seconds, bool) or
                not isinstance(timeout_seconds, (int, float)) or
                not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
            raise ValueError("timeout_seconds must be a finite positive number")

        budget = min(float(timeout_seconds), _MAX_TIMEOUT_SECONDS)
        started = self._monotonic()
        params = {
            "action": "query", "format": "json", "formatversion": "2",
            "generator": "search", "gsrsearch": query, "gsrnamespace": "0",
            "gsrlimit": "3", "exlimit": "3", "prop": "extracts|info", "inprop": "url",
            "exintro": "1", "explaintext": "1", "exchars": "1200",
            "redirects": "1",
        }
        request = Request(_ENDPOINT + "?" + urlencode(params), headers={
            "User-Agent": _USER_AGENT, "Accept": "application/json",
        }, method="GET")

        try:
            remaining = budget - (self._monotonic() - started)
            if remaining <= 0:
                return RetrievalResult("timeout")
            if self._transport is not None:
                response = self._transport(request, remaining)
            else:
                opener = build_opener(ProxyHandler({}), HTTPSHandler(), _NoRedirect())
                response = opener.open(request, timeout=remaining)
            try:
                remaining = budget - (self._monotonic() - started)
                if remaining <= 0:
                    return RetrievalResult("timeout")
                body = response.read(_MAX_RESPONSE_BYTES + 1)
            finally:
                response.close()
        except (socket.timeout, TimeoutError):
            return RetrievalResult("timeout")
        except HTTPError as exc:
            exc.close()
            return RetrievalResult("unavailable")
        except URLError as exc:
            reason = exc.reason
            if isinstance(reason, (socket.timeout, TimeoutError)):
                return RetrievalResult("timeout")
            return RetrievalResult("unavailable")
        except (OSError, ValueError):
            return RetrievalResult("unavailable")

        if self._monotonic() - started > budget:
            return RetrievalResult("timeout")
        if not isinstance(body, bytes) or len(body) > _MAX_RESPONSE_BYTES:
            return RetrievalResult("unavailable")
        try:
            payload = json.loads(body.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            return RetrievalResult("unavailable")
        if self._monotonic() - started > budget:
            return RetrievalResult("timeout")
        if not isinstance(payload, dict):
            return RetrievalResult("unavailable")
        if "error" in payload:
            return RetrievalResult("unavailable")
        query_data = payload.get("query")
        pages = query_data.get("pages") if isinstance(query_data, dict) else None
        if pages is None:
            return RetrievalResult("not_found")
        if not isinstance(pages, list):
            return RetrievalResult("unavailable")

        ranked_pages = []
        for position, page in enumerate(pages):
            rank = page.get("index") if isinstance(page, dict) else None
            valid_rank = isinstance(rank, int) and not isinstance(rank, bool) and rank > 0
            ranked_pages.append((0, rank, position, page) if valid_rank else
                                (1, position, position, page))
        ranked_pages.sort(key=lambda item: item[:3])

        sources = []
        seen_ids, seen_urls = set(), set()
        for _, _, _, page in ranked_pages:
            if not isinstance(page, dict):
                continue
            page_id, title, excerpt = page.get("pageid"), page.get("title"), page.get("extract")
            if (not isinstance(page_id, int) or isinstance(page_id, bool) or page_id <= 0 or
                    not isinstance(title, str) or not title.strip() or len(title) > 200 or
                    _contains_control(title) or not isinstance(excerpt, str)):
                continue
            url = _safe_source_url(page.get("fullurl"))
            if not url or page_id in seen_ids or url in seen_urls:
                continue
            excerpt = excerpt.replace("\r\n", "\n").replace("\r", "\n")
            excerpt = excerpt.translate({0x85: "\n", 0x2028: "\n", 0x2029: "\n"})
            excerpt = "".join(char for char in excerpt
                              if unicodedata.category(char) not in {"Cc", "Cf"} or char == "\n")
            excerpt = excerpt[:_MAX_EXCERPT_CHARS]
            if not excerpt.strip():
                continue
            seen_ids.add(page_id)
            seen_urls.add(url)
            sources.append(Source(f"w{len(sources) + 1}", title.strip(), url, excerpt))
            if len(sources) == _MAX_SOURCES:
                break
        if sources:
            return RetrievalResult("ok", tuple(sources))
        return RetrievalResult("not_found")
