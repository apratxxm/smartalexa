"""Spotify playback control via the Web API, driving the running desktop client.

The audio path is untouched: Spotify -> Windows default output -> Bluetooth
speaker. We only send transport commands.
"""
from __future__ import annotations

import argparse
from datetime import timezone
from email.utils import parsedate_to_datetime
import json
import logging
import math
import os
import re
import socket
from contextlib import contextmanager
from difflib import SequenceMatcher
from collections import deque
import threading
import time
from typing import Optional

import spotipy
from spotipy.oauth2 import SpotifyOAuth

from . import catalogue
from .config import ROOT, load, secret
from .resolution import (AmbiguousMatch, Candidate, MusicConfirmationRequired,
                          _VERSION_TERMS, _clearly_best, _norm,
                          _strong_match, album_key, best_available_candidate, collapse_reissues,
                          eligible_media, rank_albums,
                          rank_tracks, title_key)

log = logging.getLogger("memoir.spotify")

SCOPES = (
    "user-modify-playback-state user-read-playback-state "
    "playlist-read-private user-top-read"
)
CACHE_PATH = ROOT / ".cache"
# Last known good playback device. The desktop client disappears from
# /me/player/devices once it has been idle for a while, even though it is
# running and perfectly able to play -- so an empty device list is not proof
# that Spotify is closed, and we should try the remembered id before giving up.
DEVICE_CACHE = ROOT / "assistant" / "logs" / ".device.json"



# This app's search endpoint rejects limit > 10 with "Invalid limit" (a
# Development-mode restriction). Library endpoints still accept 50, which is why
# it never surfaced there. Paginate search with offset instead of raising this.
SEARCH_PAGE = 10
# start_playback accepts up to 100 uris; the requested song plus a radio.
RADIO_MAX_TRACKS = 49


def _spoken_consonants(value: str) -> str:
    """Compare a rough phonetic outline when ASR spells an artist by sound."""
    return "".join(ch for ch in _norm(value) if ch.isalpha() and ch not in "aeiouh")



TRACK_CACHE = ROOT / "assistant" / "logs" / ".track_cache.json"
# A ban outlives the process: restarts used to re-probe a 24 h ban every
# 5 minutes, and each refused call counts against the app.
BACKOFF_PATH = ROOT / "assistant" / "logs" / ".spotify_backoff.json"

# Words people (or the recognizer) put between a title and its artist.
_CONNECTORS = {"by", "from", "with", "of", "i", "is", "and", "feat", "featuring", "ft", "buy", "bye"}

class SpotifyRateLimited(RuntimeError):
    """An actual Spotify HTTP 429 or an active persisted 429 cooldown."""


class SpotifyQuotaExceeded(SpotifyRateLimited):
    """Spotify development-mode quota is exhausted until an operator clears it."""

    def __init__(self):
        super().__init__("Spotify API quota is exhausted; local controls may still work.")


class SpotifyRequestDeferred(RuntimeError):
    """A local request budget deferred work without contacting Spotify."""


class _SearchBudgetExhausted(dict):
    """Empty response sentinel that must not be cached as a Spotify result."""


_request_context = threading.local()
_BACKOFF_LOCK = threading.RLock()


@contextmanager
def request_context(source="catalogue", background=True):
    """Attribute requests in this thread; restore the prior context on exit."""
    old = getattr(_request_context, "value", None)
    _request_context.value = (str(source), bool(background))
    try:
        yield
    finally:
        if old is None:
            try:
                del _request_context.value
            except AttributeError:
                pass
        else:
            _request_context.value = old


class _RequestBudget:
    """Conservative process-local rolling admission policy, not a Spotify limit."""
    def __init__(self, window=30.0, total=20, background=15, clock=None, sleeper=None):
        self.window, self.total, self.background = window, total, background
        self.clock = clock or time.monotonic
        self.sleeper = sleeper or time.sleep
        self.lock = threading.Lock()
        self.attempts = deque()
        self.last_foreground = float("-inf")

    def admit(self, source, background, wait):
        deadline = self.clock() + (min(max(wait, 0.0), 1.0) if not background else 0.0)
        while True:
            now = self.clock()
            with self.lock:
                if not background:
                    # Foreground demand gets priority even when this attempt is denied.
                    self.last_foreground = now
                while self.attempts and now - self.attempts[0][0] >= self.window:
                    self.attempts.popleft()
                bg_count = sum(1 for _, is_bg in self.attempts if is_bg)
                if background and now - self.last_foreground < 2.0:
                    reason = "recent foreground request"
                elif len(self.attempts) < self.total and (not background or bg_count < self.background):
                    self.attempts.append((now, background))
                    return
                else:
                    reason = "rolling request budget"
                next_at = (self.attempts[0][0] + self.window) if self.attempts else now
            if background or now >= deadline:
                raise SpotifyRequestDeferred("Spotify request deferred by local {}".format(reason))
            self.sleeper(min(max(next_at - now, 0.01), deadline - now))


_SHARED_REQUEST_BUDGET = _RequestBudget()


class _RateLimitGuard:
    """Wraps the spotipy client: after a 429, fail instantly instead of calling.

    HTTP 429 cooldowns retain their player/web scopes for compatibility. The
    request budget below is a conservative local policy, not a known Spotify threshold.
    """

    PROBE_SECONDS = 300
    request_context = staticmethod(request_context)
    PLAYER = {"current_playback", "currently_playing", "devices", "start_playback",
              "pause_playback", "next_track", "previous_track", "seek_track",
              "volume", "transfer_playback", "add_to_queue", "shuffle", "repeat",
              "current_user_recently_played", "queue"}

    def __init__(self, client, path=None, budget=None, clock=None, sleeper=None,
                 window=30.0, total=20, background=15, foreground_wait=1.0) -> None:
        self._client = client
        self.blocked = {"player": 0.0, "web": 0.0}
        self.quota = {"player": None, "web": None}
        self.quota_revision = {"player": 0, "web": 0}
        # Ban end-times already written to the log, so each ban is logged once.
        self._announced = {"player": 0.0, "web": 0.0}
        self._path = path
        self._seen_mtime = None
        self._budget = budget or (_SHARED_REQUEST_BUDGET if clock is None and sleeper is None
                                  and window == 30.0 and total == 20 and background == 15
                                  else _RequestBudget(window, total, background, clock, sleeper))
        self._foreground_wait = min(max(float(foreground_wait), 0.0), 1.0)
        self._sync()

    def _sync(self) -> None:
        """Adopt a ban recorded by another process (e.g. the catalogue build)."""
        if self._path is None:
            return
        try:
            mtime = self._path.stat().st_mtime
        except OSError:
            return
        if mtime == self._seen_mtime:
            return
        self._seen_mtime = mtime
        try:
            saved = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if not isinstance(saved, dict):
            return
        for scope in self.blocked:
            try:
                value = float(saved.get(scope, 0) or 0)
                if math.isfinite(value):
                    self.blocked[scope] = max(self.blocked[scope], value)
            except (TypeError, ValueError):
                pass
            self._announce(scope, "from the backoff file")
        self._adopt_quota(saved)

    def _adopt_quota(self, saved: dict) -> None:
        revisions = saved.get("quota_revision", {})
        if not isinstance(revisions, dict):
            log.warning("Malformed Spotify quota revision metadata; retaining local quarantine state")
            revisions = {}
        entries = saved.get("quota", {})
        if not isinstance(entries, dict):
            entries = {}
        for scope in self.quota:
            raw_revision = revisions.get(scope, 0)
            if isinstance(raw_revision, bool) or not isinstance(raw_revision, int) or raw_revision < 0:
                log.warning("Malformed Spotify quota revision metadata; retaining local quarantine state")
                disk_revision = 0
            else:
                disk_revision = raw_revision
            disk_entry = self._quota_entry(entries.get(scope))
            if disk_revision > self.quota_revision[scope]:
                self.quota[scope] = disk_entry
                self.quota_revision[scope] = disk_revision
            elif disk_revision == self.quota_revision[scope]:
                self.quota[scope] = self._merge_quota(self.quota[scope], disk_entry)

    @staticmethod
    def _local_time(moment: float) -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(moment))

    def _announce(self, scope: str, source: str, level: int = logging.WARNING) -> None:
        """Log a ban once: its scope, how we learned of it, when it lifts (local)."""
        until = self.blocked[scope]
        if until <= time.time() or self._announced[scope] == until:
            return
        self._announced[scope] = until
        log.log(level, "Spotify %s API ban in effect (%s): calls paused until %s local (%.1f h)",
                scope, source, self._local_time(until), (until - time.time()) / 3600)

    def _expire(self, scope: str) -> None:
        """Log once when a ban that was announced has run out."""
        if self._announced[scope] and self.blocked[scope] <= time.time():
            log.info("Spotify %s API ban lifted (was until %s local); calls resume",
                     scope, self._local_time(self._announced[scope]))
            self._announced[scope] = 0.0

    def ban_status(self) -> dict:
        """{scope: unix time the ban lifts} for active bans; also logs expiries."""
        self._sync()
        now = time.time()
        for scope in self.blocked:
            self._expire(scope)
        return {scope: until for scope, until in self.blocked.items() if until > now}

    @staticmethod
    def _quota_entry(value):
        if not isinstance(value, dict) or value.get("reason") != "QUOTA_EXCEEDED":
            return None
        if "retry_until" not in value:
            log.warning("Malformed Spotify quota expiry metadata; retaining indefinite quarantine")
            return {"reason": "QUOTA_EXCEEDED", "retry_until": None}
        until = value["retry_until"]
        if until is None:
            return {"reason": "QUOTA_EXCEEDED", "retry_until": None}
        try:
            until = float(until) if not isinstance(until, bool) else float("nan")
        except (TypeError, ValueError, OverflowError):
            until = float("nan")
        if not math.isfinite(until):
            # A known quota marker with corrupt expiry must fail closed.
            log.warning("Malformed Spotify quota expiry metadata; retaining indefinite quarantine")
            return {"reason": "QUOTA_EXCEEDED", "retry_until": None}
        return {"reason": "QUOTA_EXCEEDED", "retry_until": until}

    @staticmethod
    def _merge_quota(left, right):
        left, right = _RateLimitGuard._quota_entry(left), _RateLimitGuard._quota_entry(right)
        if left is None:
            return right
        if right is None:
            return left
        if left["retry_until"] is None or right["retry_until"] is None:
            return {"reason": "QUOTA_EXCEEDED", "retry_until": None}
        return {"reason": "QUOTA_EXCEEDED", "retry_until": max(left["retry_until"], right["retry_until"])}

    def quota_status(self) -> dict:
        """Active quota quarantines: scope -> reason and provider expiry (or None)."""
        self._sync()
        now = time.time()
        return {scope: dict(entry) for scope, entry in self.quota.items()
                if entry is not None and (entry["retry_until"] is None or entry["retry_until"] > now)}

    def clear_quota(self, scope: str) -> None:
        """Explicitly clear one operator-managed quota quarantine."""
        if not isinstance(scope, str) or scope not in self.quota:
            raise ValueError("scope must be 'player' or 'web'")
        previous = self.quota[scope]
        previous_revision = self.quota_revision[scope]
        if not self._save(clear_quota_scope=scope):
            self.quota[scope] = previous
            self.quota_revision[scope] = previous_revision
            raise RuntimeError("Could not persist the Spotify quota clear")

    def _save(self, clear_quota_scope=None, quota_scope=None, quota_entry=None) -> bool:
        with _BACKOFF_LOCK:
            try:
                saved = {}
                if self._path is not None:
                    try:
                        existing = json.loads(self._path.read_text(encoding="utf-8"))
                        if isinstance(existing, dict):
                            saved = existing
                    except (OSError, ValueError):
                        pass
                self._adopt_quota(saved)
                for scope in self.blocked:
                    try:
                        value = float(saved.get(scope, 0) or 0)
                        if math.isfinite(value):
                            self.blocked[scope] = max(self.blocked[scope], value)
                    except (TypeError, ValueError, OverflowError):
                        pass
                if clear_quota_scope is not None:
                    self.quota_revision[clear_quota_scope] += 1
                    self.quota[clear_quota_scope] = None
                elif quota_scope is not None:
                    self.quota_revision[quota_scope] += 1
                    self.quota[quota_scope] = self._quota_entry(quota_entry)
                if self._path is None:
                    return True
                payload = dict(self.blocked)
                payload["quota"] = {scope: entry for scope, entry in self.quota.items() if entry is not None}
                payload["quota_revision"] = dict(self.quota_revision)
                tmp = self._path.with_name("{}.{}.{}.tmp".format(
                    self._path.name, os.getpid(), threading.get_ident()))
                tmp.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
                os.replace(tmp, self._path)
                self._seen_mtime = self._path.stat().st_mtime
                return True
            except OSError as exc:
                log.debug("Could not save Spotify backoff: %s", exc)
                return False

    @staticmethod
    def _retry_after_epoch(value, now):
        if value is None:
            return None
        value = str(value).strip()
        try:
            seconds = int(value)
            if seconds < 0:
                return None
            until = now + seconds
        except OverflowError:
            return None
        except ValueError:
            try:
                retry_at = parsedate_to_datetime(value)
                if retry_at.tzinfo is None:
                    retry_at = retry_at.replace(tzinfo=timezone.utc)
                until = retry_at.timestamp()
            except (TypeError, ValueError, OverflowError):
                return None
        return float(until) if math.isfinite(until) else None

    def _raise_quota_if_active(self, scope: str) -> None:
        entry = self.quota_status().get(scope)
        if entry is not None:
            raise SpotifyQuotaExceeded()

    @property
    def blocked_until(self) -> float:
        return self.blocked["player"]

    def __getattr__(self, name):
        attr = getattr(self._client, name)
        if not callable(attr):
            return attr
        scope = "player" if name in self.PLAYER else "web"

        def guarded(*args, **kwargs):
            source, background = getattr(_request_context, "value", ("assistant", False))
            if source not in {"assistant", "catalogue", "manual_catalogue", "startup"}:
                source = "assistant"
            self._sync()
            self._raise_quota_if_active(scope)
            remaining = self.blocked[scope] - time.time()
            if remaining > 0:
                log.info("Spotify request method=%s source=%s mode=%s elapsed=0 status=blocked result=cooldown network=no",
                         name, source, "background" if background else "foreground")
                raise SpotifyRateLimited(
                    "Spotify {} API is rate-limiting this app; retrying in {:.0f} s".format(
                        scope, remaining))
            self._expire(scope)
            try:
                self._budget.admit(source, background, self._foreground_wait)
            except SpotifyRequestDeferred as exc:
                log.info("Spotify request method=%s source=%s mode=%s elapsed=0 status=deferred result=local_budget network=no",
                         name, source, "background" if background else "foreground")
                raise
            # Admission does not imply dispatch: another call may have observed 429.
            self._sync()
            self._raise_quota_if_active(scope)
            remaining = self.blocked[scope] - time.time()
            if remaining > 0:
                log.info("Spotify request method=%s source=%s mode=%s elapsed=0 status=blocked result=cooldown network=no",
                         name, source, "background" if background else "foreground")
                raise SpotifyRateLimited("Spotify {} API is rate-limiting this app; retrying in {:.0f} s".format(scope, remaining))
            started = time.monotonic()
            try:
                result = attr(*args, **kwargs)
                log.info("Spotify request method=%s source=%s mode=%s elapsed=%.3f status=success result=ok",
                         name, source, "background" if background else "foreground", time.monotonic() - started)
                return result
            except spotipy.SpotifyException as exc:
                status = getattr(exc, "http_status", "error")
                log.info("Spotify request method=%s source=%s mode=%s elapsed=%.3f status=%s result=http_error",
                         name, source, "background" if background else "foreground", time.monotonic() - started, status)
                if exc.http_status != 429:
                    raise
                retry = None
                try:
                    retry = next((value for key, value in (exc.headers or {}).items()
                                  if str(key).casefold() == "retry-after"), None)
                except (AttributeError, TypeError):
                    pass
                now = time.time()
                retry_until = self._retry_after_epoch(retry, now)
                if exc.reason == "QUOTA_EXCEEDED":
                    self._save(quota_scope=scope, quota_entry={
                        "reason": "QUOTA_EXCEEDED", "retry_until": retry_until})
                    raise SpotifyQuotaExceeded() from exc
                wait = self.PROBE_SECONDS
                if retry_until is not None:
                    wait = max(0, int(retry_until - now))
                self.blocked[scope] = now + wait
                self._save()
                self._announce(scope, "HTTP 429 from {}, Retry-After={}".format(
                    name, "missing" if retry is None else repr(retry)), logging.ERROR)
                raise SpotifyRateLimited("Spotify {} API is rate-limiting this app".format(scope)) from exc
            except Exception as exc:
                log.info("Spotify request method=%s source=%s mode=%s elapsed=%.3f status=error result=%s",
                         name, source, "background" if background else "foreground",
                         time.monotonic() - started, type(exc).__name__)
                raise
        return guarded


class SpotifyAuthError(RuntimeError):
    """Raised distinctly so the failure can be spoken, not just beeped."""


def _title_diag(wanted: str, ranked: list) -> tuple:
    """(title similarity of the winner, runner-up label, margin) for telemetry."""
    if not ranked:
        return None, "", None
    target = _norm(wanted)
    top = SequenceMatcher(None, target, _norm(ranked[0].name)).ratio()
    if len(ranked) < 2:
        return top, "", None
    second = SequenceMatcher(None, target, _norm(ranked[1].name)).ratio()
    return top, ranked[1].label, top - second


def _catalogue_hit(query: str, index):
    """catalogue.best's decision, plus the song it beat, from one scoring pass.

    best() returns only the winner, and scoring twice would add ~70 ms to every
    command. tests/test_resolution_diagnostics.py pins this to catalogue.best.
    Returns (entry, score, rival Scored or None) or None.
    """
    if index is None:
        return None
    ranked = catalogue.match(query, index, limit=0)
    if not ranked:
        return None
    top = ranked[0]
    if top.score < top.threshold or top.title_sim < catalogue.MIN_TITLE_ACCEPT:
        return None
    rival = next((s for s in ranked[1:] if s.song_key != top.song_key), None)
    if rival is not None and top.score - rival.score < catalogue.MARGIN:
        return None
    return top.entry, top.score, rival


class PlaybackVerificationError(RuntimeError):
    """Spotify accepted a request but did not report the intended state."""


class TrackLookupError(LookupError):
    """No confident song match; only this lookup failure permits album fallback."""


def _matches_local_candidate(candidate: Candidate, snapshot, require_playing: bool = True) -> bool:
    """Match SMTC metadata to the requested catalogue entry without fuzzy matches."""
    wanted_title = _norm(candidate.name)
    wanted_artist = _norm(candidate.artist)
    if not snapshot or not wanted_title or not wanted_artist:
        return False
    return (
        _norm(snapshot.title) == wanted_title
        and _matches_local_credits(candidate, snapshot.artist)
        and (not require_playing or snapshot.playing is True)
    )


def _matches_local_credits(candidate: Candidate, native: str) -> bool:
    credits = [row.get("name", "") for row in (candidate.metadata or {}).get("artists") or []
               if isinstance(row, dict) and row.get("name")] or [candidate.artist]
    # Artists can contain punctuation themselves. Try the full canonical
    # spelling first; otherwise require a complete delimited credit set.
    if len(credits) == 1:
        return _norm(native) == _norm(credits[0])
    if " ".join(_norm(native).split()) == " ".join(_norm(", ".join(credits)).split()):
        return True
    canonical = {_norm(credit) for credit in credits}
    supplied = {_norm(part) for part in re.split(r"[,;&]", native) if _norm(part)}
    return supplied == canonical


class Spotify:
    def __init__(self) -> None:
        self._cfg = load()
        self._device_id: Optional[str] = None
        self._device_name: str = ""
        # A command must never start on one Connect device and verify or undo
        # work on whichever device happens to become active a moment later.
        self._command_device_id: Optional[str] = None
        self._command_device_name: str = ""
        self._command_depth = 0
        self._auth = SpotifyOAuth(
            client_id=secret("SPOTIPY_CLIENT_ID"),
            client_secret=secret("SPOTIPY_CLIENT_SECRET"),
            redirect_uri=secret("SPOTIPY_REDIRECT_URI"),
            scope=SCOPES,
            cache_path=str(CACHE_PATH),
            open_browser=True,
        )
        # Never sleep through a 429: spotipy's default retry honours
        # Retry-After, which once froze startup for a 16.5 hour ban.
        client = spotipy.Spotify(
            auth_manager=self._auth, requests_timeout=8, retries=0, status_retries=0)
        # urllib3 treats any 429 carrying Retry-After as retryable even with
        # zero retries, so spotipy raised a header-less "Max Retries" error and
        # the real ban length (85823 s once) was lost. Return the response.
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        adapter = HTTPAdapter(max_retries=Retry(
            total=0, read=False, status_forcelist=(),
            respect_retry_after_header=False, raise_on_status=False))
        client._session.mount("https://", adapter)
        client._session.mount("http://", adapter)
        self.sp = _RateLimitGuard(client, BACKOFF_PATH)
        self._known_artists = []
        # How the last lookup / playback was answered; main copies these into
        # the command's telemetry (see _note_resolution, play_candidate).
        self.last_resolution = None
        self.last_playback = None

    @property
    def rate_limited(self) -> bool:
        if isinstance(self.sp, _RateLimitGuard) and self.sp.quota_status().get("player"):
            return True
        return time.time() < getattr(self.sp, "blocked_until", 0.0)

    @property
    def web_rate_limited(self) -> bool:
        if isinstance(self.sp, _RateLimitGuard):
            return "web" in self.sp.ban_status() or bool(self.sp.quota_status().get("web"))
        blocked = getattr(self.sp, "blocked", None)
        return isinstance(blocked, dict) and time.time() < blocked.get("web", 0.0)

    # ---------- auth ----------

    def token_age_days(self) -> Optional[float]:
        """Days since cache modification, for diagnostics only (not expiry)."""
        if not CACHE_PATH.exists():
            return None
        try:
            return (time.time() - CACHE_PATH.stat().st_mtime) / 86400.0
        except OSError:
            return None

    def check_auth(self) -> None:
        """Fail loudly and specifically. A generic API error here is unhelpful."""
        try:
            self.sp.current_user()
        except spotipy.SpotifyException as exc:
            if exc.http_status in (401, 403):
                raise SpotifyAuthError(
                    "Spotify authorization failed. Re-authorize if needed: "
                    f"delete {CACHE_PATH} and run 'python -m assistant.spotify --auth'."
                ) from exc
            raise

    # ---------- device ----------

    @staticmethod
    def _client_running() -> bool:
        """Is the Spotify desktop app already running?

        Checked before launching: an idle client is missing from the Connect
        device list but very much alive, and spawning a second copy does not
        help.
        """
        # tasklist returns "Access denied" on this machine even while several
        # Spotify.exe processes are running. psutil can read process names here.
        import psutil

        for process in psutil.process_iter(["name"]):
            try:
                if (process.info["name"] or "").casefold() == "spotify.exe":
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return False

    def _remember(self, device_id: str, name: str) -> None:
        try:
            DEVICE_CACHE.parent.mkdir(parents=True, exist_ok=True)
            DEVICE_CACHE.write_text(json.dumps({"id": device_id, "name": name}))
        except OSError:
            pass

    def _remembered(self) -> tuple:
        try:
            d = json.loads(DEVICE_CACHE.read_text())
            return d.get("id", ""), d.get("name", "")
        except (OSError, json.JSONDecodeError):
            return "", ""

    def _wake_device(self, device_id: str) -> bool:
        """Nudge an idle client back onto Connect without starting playback."""
        try:
            self.sp.transfer_playback(device_id=device_id, force_play=False)
            return True
        except spotipy.SpotifyException as exc:
            log.debug("Could not transfer playback to %s: %s", device_id, exc)
            return False

    def _launch_client(self) -> None:
        """Start Spotify via its URI protocol handler.

        The Microsoft Store build has no stable exe path, so the protocol
        handler is the only reliable launcher.
        """
        log.info("Spotify desktop client is closed; opening it.")
        try:
            os.startfile("spotify:")
        except OSError as exc:
            raise RuntimeError("Could not open Spotify via its Windows URI handler") from exc

    def ensure_client_running(self) -> bool:
        """Open a closed desktop client once for an actual recognized command.

        Returns whether a launch was requested. A short grace window avoids
        opening duplicates while Windows is still starting Spotify.exe.
        """
        if self._client_running():
            return False
        now = time.monotonic()
        if now - getattr(self, "_launch_requested_at", 0.0) < 15.0:
            return True
        self._launch_client()
        self._launch_requested_at = now
        return True

    def wait_for_client(self, timeout_seconds: float = 8.0) -> None:
        """Bound the first command's wait for a newly launched desktop app."""
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if self._client_running():
                # The process appears before Spotify Connect is ready.
                time.sleep(1.0)
                return
            time.sleep(0.25)
        raise RuntimeError("Spotify was requested but did not start")

    def device(self, retry_launch: bool = True) -> str:
        """Resolve the playback device, reviving an idle client if needed."""
        pinned = getattr(self, "_command_device_id", None)
        if pinned:
            return pinned
        if self._cfg.get("spotify.target", "laptop") == "active":
            devices = self.sp.devices().get("devices", [])
            chosen = next((d for d in devices if d.get("is_active")
                           and not d.get("is_restricted")), None)
            if chosen is None:
                # Pausing for a while can leave Connect with no active target
                # even though this laptop is still listed. A bare "play" must
                # be able to resume it, but never guess a phone or another PC.
                local_name = socket.gethostname().casefold()
                local = [d for d in devices if d.get("name", "").strip().casefold() == local_name
                         and d.get("type", "").casefold() == "computer"
                         and not d.get("is_restricted")]
                if len(local) == 1:
                    chosen = local[0]
                else:
                    raise PlaybackVerificationError("No active controllable Spotify device")
            self._device_id, self._device_name = chosen["id"], chosen["name"]
            return self._device_id
        if self._device_id:
            return self._device_id

        hostname = socket.gethostname().lower()
        configured = str(self._cfg.get("spotify.device_name", "") or self._cfg.get("device_name", "")).strip().lower()
        target = configured or hostname

        def pick(devices):
            local = [d for d in devices if d.get("type", "").lower() in ("computer", "speaker")
                     and not any(x in d.get("name", "").lower() for x in ("phone", "iphone", "android"))]
            return next((d for d in local if d.get("name", "").strip().lower() == target), None)

        chosen = pick(self.sp.devices().get("devices", []))

        if chosen is None:
            # Empty list: try the remembered device before assuming it is closed.
            cached_id, cached_name = self._remembered()
            if cached_id and cached_name.strip().lower() == target and self._wake_device(cached_id):
                time.sleep(1)
                chosen = pick(self.sp.devices().get("devices", []))
                if chosen is None:
                    log.info("Reusing remembered device %r", cached_name)
                    self._device_id, self._device_name = cached_id, cached_name
                    return cached_id

        if chosen is None and retry_launch:
            if self._client_running():
                log.warning(
                    "Spotify is running but not visible on Connect. Play "
                    "something in the app once to register it.")
            else:
                self._launch_client()
            for _ in range(10):
                time.sleep(1)
                chosen = pick(self.sp.devices().get("devices", []))
                if chosen:
                    break

        if chosen is None:
            raise RuntimeError(
                "No Spotify device available. Open Spotify and play something "
                "once so it registers with Spotify Connect.")

        self._device_id = chosen["id"]
        self._device_name = chosen["name"]
        self._remember(self._device_id, self._device_name)
        log.info("Using Spotify device %r", self._device_name)
        return self._device_id

    @property
    def device_name(self) -> str:
        return getattr(self, "_command_device_name", "") or self._device_name

    @contextmanager
    def command_target(self):
        """Pin one Spotify Connect target for a complete command.

        In ``target = active`` mode Spotify may switch active devices between
        API calls. Nested operations share this pin, so a verification or a
        volume restoration cannot affect a different device.
        """
        if getattr(self, "_command_depth", 0):
            self._command_depth += 1
            try:
                yield self._command_device_id
            finally:
                self._command_depth -= 1
            return

        target = self.device()
        self._command_device_id = target
        self._command_device_name = getattr(self, "_device_name", "")
        self._command_depth = 1
        try:
            yield target
        finally:
            self._command_depth = 0
            self._command_device_id = None
            self._command_device_name = ""

    def _call(self, fn, *args, **kwargs):
        """Run one targeted mutation; never retry a command on another device."""
        return fn(*args, device_id=self.device(), **kwargs)

    # ---------- transport ----------

    def _verify(self, predicate, description: str) -> dict:
        """Bounded read-only checks; never repeat a playback mutation here."""
        target = self.device()
        for attempt in range(4):
            state = self.sp.current_playback()
            if state and (state.get("device") or {}).get("id") == target and predicate(state):
                return state
            if attempt < 3:
                time.sleep(0.25)
        raise PlaybackVerificationError("Could not verify " + description + " on the selected laptop")

    def _local_enabled(self) -> bool:
        cfg = getattr(self, "_cfg", None)
        return bool(cfg is not None and cfg.get("spotify.local_transport", True))

    def _note_resolution(self, path: str, score=None, runner_up: str = "", margin=None) -> None:
        """Remember how the last lookup was answered, for telemetry."""
        self.last_resolution = {
            "path": path,
            "score": None if score is None else round(float(score), 3),
            "runner_up": runner_up,
            "margin": None if margin is None else round(float(margin), 3),
        }

    def _local(self, action) -> bool:
        """Run a transport action on the local app via Windows media controls.

        Instant, and no Web API call at all. Returns False when there is no
        local Spotify media session, so the caller can fall back to the API.
        """
        if not self._local_enabled():
            return False
        try:
            from . import local_media
        except ImportError:
            return False
        if local_media.status() is None:
            return False
        if not action(local_media):
            raise PlaybackVerificationError("Spotify did not respond to the media command")
        self.last_playback = {"method": "local_media"}
        return True

    def play_resume(self) -> None:
        if self._local(lambda m: m.play()):
            return
        with self.command_target():
            self._call(self.sp.start_playback)
            self._verify(lambda s: s.get("is_playing") is True, "resume")

    def pause(self) -> None:
        if self._local(lambda m: m.pause()):
            return
        with self.command_target():
            # Read the target, not whichever other device happens to be playing.
            state = self.sp.current_playback()
            if not (state and (state.get("device") or {}).get("id") == self.device()
                    and state.get("is_playing") is False):
                self._call(self.sp.pause_playback)
            self._verify(lambda s: s.get("is_playing") is False, "pause")

    def next(self) -> None:
        if self._local(lambda m: m.next_track()):
            return
        with self.command_target():
            self._change_track(self.sp.next_track, "next track")

    def previous(self) -> None:
        if self._local(lambda m: m.previous_track()):
            return
        with self.command_target():
            self._change_track(self.sp.previous_track, "previous track")

    def restart_track(self) -> None:
        """Seek the current track to its beginning without changing the queue."""
        if self._local(lambda m: m.restart()):
            return
        with self.command_target():
            before = self.sp.current_playback()
            if not before or (before.get("device") or {}).get("id") != self.device():
                raise PlaybackVerificationError("No playback on the selected laptop")
            uri = (before.get("item") or {}).get("uri")
            if not uri:
                raise PlaybackVerificationError("No current track to restart")
            self._call(self.sp.seek_track, 0)
            self._verify(lambda state: (
                (state.get("item") or {}).get("uri") == uri
                and (state.get("progress_ms") or 0) <= 2500
            ), "restart track")

    def restart_album(self) -> str:
        """Restart only a freshly proven album context on the pinned device.

        A track's album metadata and native media title cannot establish its
        playback context. Missing API proof therefore fails without mutation.
        Shuffle and volume preferences are never changed here.
        """
        self.last_playback = None
        with self.command_target():
            before = self.sp.current_playback()
            target = self.device()
            context = (before or {}).get("context") or {}
            uri = context.get("uri")
            if (not before or (before.get("device") or {}).get("id") != target
                    or context.get("type") != "album" or not isinstance(uri, str)
                    or not re.fullmatch(r"spotify:album:[A-Za-z0-9]+", uri)):
                raise PlaybackVerificationError("The selected device has no verified current album context")
            rows = self.sp.album_tracks(uri, limit=1)
            items = rows.get("items") if isinstance(rows, dict) else None
            first = items[0] if isinstance(items, list) and items else None
            track_uri = first.get("uri") if isinstance(first, dict) else None
            if (not isinstance(track_uri, str) or not re.fullmatch(r"spotify:track:[A-Za-z0-9]+", track_uri)
                    or first.get("type") not in (None, "track")
                    or first.get("is_playable") is False or first.get("is_local") is True
                    or first.get("restrictions")):
                raise PlaybackVerificationError("The album's first track could not be verified")
            # A lookup must not let a newer context/device inherit this request.
            current = self.sp.current_playback()
            if (not current or (current.get("device") or {}).get("id") != target
                    or (current.get("context") or {}).get("type") != "album"
                    or (current.get("context") or {}).get("uri") != uri):
                raise PlaybackVerificationError("The album playback context changed before restart")
            self._call(self.sp.start_playback, context_uri=uri, offset={"position": 0}, position_ms=0)
            self._verify(lambda state: state.get("is_playing") is True
                         and (state.get("context") or {}).get("type") == "album"
                         and (state.get("context") or {}).get("uri") == uri
                         and (state.get("item") or {}).get("uri") == track_uri
                         and type(state.get("progress_ms")) is int
                         and 0 <= state["progress_ms"] <= 2500,
                         "album restart at its first track")
        self.radio_played = False
        self.last_playback = {"method": "web_api_album_restart_verified", "radio": 0}
        return "Restarted the current album."

    def _change_track(self, fn, description: str) -> None:
        before = self.sp.current_playback()
        if not before or (before.get("device") or {}).get("id") != self.device():
            raise PlaybackVerificationError("No playback on the selected laptop")
        uri = (before.get("item") or {}).get("uri")
        progress = before.get("progress_ms") or 0
        if not uri:
            raise PlaybackVerificationError("No current track to verify navigation")
        self._call(fn)
        # Previous may restart the current track; repeating the only queued
        # track may do the same. A substantial progress reset is observable.
        self._verify(lambda s: bool((s.get("item") or {}).get("uri")) and (
            (s.get("item") or {}).get("uri") != uri
            or ((s.get("progress_ms") or 0) + 1000 < progress)
        ), description)

    def is_playing(self) -> bool:
        with self.command_target():
            state = self.sp.current_playback()
            return bool(state and (state.get("device") or {}).get("id") == self.device()
                        and state.get("is_playing"))

    def playback_snapshot(self) -> tuple:
        """Read target playback/volume together, avoiding duplicate API calls."""
        playing = None
        if self._local_enabled():
            try:
                from . import local_media
                playing = local_media.is_playing()
            except ImportError:
                pass
        if playing is not None:
            # Telemetry only; not worth an API call per command.
            return playing, None
        with self.command_target():
            state = self.sp.current_playback()
            if not state or (state.get("device") or {}).get("id") != self.device():
                return None, None
            return bool(state.get("is_playing")), (state.get("device") or {}).get("volume_percent")

    def get_volume(self) -> Optional[int]:
        with self.command_target():
            state = self.sp.current_playback()
            if not state or (state.get("device") or {}).get("id") != self.device():
                return None
            if (state.get("device") or {}).get("supports_volume") is False:
                return None
            return (state.get("device") or {}).get("volume_percent")

    def set_volume(self, percent: int) -> None:
        percent = max(0, min(100, int(percent)))
        with self.command_target():
            state = self.sp.current_playback()
            device = (state or {}).get("device") or {}
            if device.get("id") != self.device():
                raise PlaybackVerificationError("No playback on the selected device")
            if device.get("supports_volume") is False:
                raise PlaybackVerificationError("Selected Spotify device does not support volume control")
            self._call(self.sp.volume, percent)
            self._verify(lambda s: (s.get("device") or {}).get("volume_percent") == percent, "volume")

    # ---------- content ----------

    def set_known_artists(self, names: list) -> None:
        """Artists the user actually listens to, used to locate the title/artist
        boundary when recognition loses the word "by"."""
        self._known_artists = [n for n in (names or []) if n and n.strip()]

    def _split_on_known_artist(self, query: str) -> tuple:
        """Recover "<title> <artist>" when the connector was mis-transcribed.

        Production output for a real command was "Play Nakhshab, I Seedhe Maut":
        the artist is recovered perfectly but "by" became "I", so a rpartition
        on " by " finds no artist and the catalogue fallback never runs. Rather
        than guess at connector spellings, test whether any suffix of the query
        matches an artist the user listens to -- a small closed set, so a match
        is strong evidence of where the boundary is.
        """
        known = getattr(self, "_known_artists", None)
        if not known:
            return "", ""
        words = query.split()
        best = ("", "", 0.0)
        for cut in range(1, len(words)):
            tail = _norm(" ".join(words[cut:]))
            if not tail:
                continue
            for artist in known:
                ratio = SequenceMatcher(None, tail, _norm(artist)).ratio()
                if ratio > best[2]:
                    best = (" ".join(words[:cut]), artist, ratio)
        # A high bar: a weak suffix match would invent an artist constraint and
        # exclude the correct song entirely.
        if best[2] >= 0.85:
            # The spoken connector survives as the title's last word when it
            # is not "by": "red from seedhe maut" -> "red", not "red from".
            title_words = best[0].split()
            while len(title_words) > 1 and _norm(title_words[-1]) in _CONNECTORS:
                title_words.pop()
            return " ".join(title_words), best[1]
        return "", ""

    @contextmanager
    def resolution_budget(self, max_searches: int = 3):
        """Bound foreground searches across a complete music-resolution chain.

        Nested calls on this Spotify instance reuse the outer allowance and
        exact query/page results. Playback and background work do not use it.
        """
        local = getattr(self, "_resolution_budget_thread", None)
        if local is None:
            local = self._resolution_budget_thread = threading.local()
        active = getattr(local, "state", None)
        if active is not None:
            yield active
            return
        state = {"max_searches": max(0, int(max_searches)), "attempts": 0,
                 "pages": {}}
        local.state = state
        try:
            yield state
        finally:
            try:
                del local.state
            except AttributeError:
                pass

    def _search_page(self, query: str, kind: str, *, limit: int = SEARCH_PAGE,
                     offset: int = 0) -> dict:
        """Issue one search page, honoring the active foreground resolution cap."""
        _source, background = getattr(_request_context, "value", ("assistant", False))
        local = getattr(self, "_resolution_budget_thread", None)
        state = getattr(local, "state", None) if local is not None else None
        if state is not None and not background:
            key = (str(query), str(kind), int(limit), int(offset))
            if key in state["pages"]:
                return state["pages"][key]
            if state["attempts"] >= state["max_searches"]:
                log.debug("Music resolution search budget exhausted after %d searches",
                          state["attempts"])
                return _SearchBudgetExhausted({str(kind) + "s": {"items": []}})
            # Count the call attempt, including a real error/429; never disguise it.
            state["attempts"] += 1
            kwargs = {"q": query, "type": kind, "limit": limit}
            if offset:
                kwargs["offset"] = offset
            result = self.sp.search(**kwargs)
            state["pages"][key] = result
            return result
        kwargs = {"q": query, "type": kind, "limit": limit}
        if offset:
            kwargs["offset"] = offset
        return self.sp.search(**kwargs)

    def _artist_catalogue(self, artist_name: str, pages: int = 3) -> list:
        """Tracks credited to an artist, as raw search items.

        Used only as a fallback when a fielded search found nothing. Results are
        filtered to entries actually crediting the artist, so a loose search
        match cannot smuggle in somebody else's song.
        """
        artist_key = _norm(artist_name)
        if not artist_key:
            return []
        items, seen = [], set()
        for page in range(pages):
            try:
                res = self._search_page('artist:"{}"'.format(artist_name.strip()),
                                        "track", limit=SEARCH_PAGE,
                                        offset=page * SEARCH_PAGE)
            except spotipy.SpotifyException as exc:
                log.debug("Artist catalogue page %d failed: %s", page, exc)
                break
            page_items = res.get("tracks", {}).get("items", [])
            if not page_items:
                break
            for track in page_items:
                credited = [_norm(a.get("name", "")) for a in track.get("artists") or []]
                if artist_key not in credited:
                    continue
                key = track.get("uri", "")
                if key and key not in seen:
                    seen.add(key)
                    items.append(track)
        return items

    def artist_hints_for_exact_title(self, query: str) -> list[str]:
        """Narrow ASR retry hints only when one exact-title artist is plausible.

        The first, unbiased transcript remains authoritative unless it failed
        resolution. Never use a popular but unrelated search hit as evidence.
        """
        title, separator, heard_artist = query.rpartition(" by ")
        if not separator or not _norm(title) or not _norm(heard_artist):
            return []
        try:
            items = self._search_page("track:{}".format(title.strip()),
                                      "track", limit=SEARCH_PAGE).get("tracks", {}).get("items", [])
        except spotipy.SpotifyException as exc:
            log.debug("Could not fetch exact-title artist hints: %s", exc)
            return []
        exact = [item for item in items if _norm(item.get("name", "")) == _norm(title)]
        primary = {_norm((item.get("artists") or [{}])[0].get("name", ""))
                   for item in exact}
        primary.discard("")
        if len(primary) != 1:
            return []
        names = []
        for item in exact:
            for artist in item.get("artists") or []:
                name = artist.get("name", "").strip()
                if name and name.casefold() not in {n.casefold() for n in names}:
                    names.append(name)
        if not names or SequenceMatcher(None, _norm(heard_artist),
                                        _norm(names[0])).ratio() < 0.60:
            return []
        return names[:4]

    # ---------- local catalogue ----------

    def set_catalogue(self, index, pin: bool = True) -> None:
        """Use ``index`` for catalogue-grounded resolution.

        ``pin`` keeps it in place; unpinned (the background refresher) it stays
        replaceable by a newer file on disk.
        """
        self._catalogue = index
        self._catalogue_pinned = bool(pin) and index is not None
        try:
            self._catalogue_mtime = catalogue.INDEX_PATH.stat().st_mtime
        except OSError:
            self._catalogue_mtime = None

    def _catalogue_index(self):
        """The loaded index, lazily (re)read from disk when the file changes."""
        if getattr(self, "_catalogue_pinned", False):
            return self._catalogue
        try:
            mtime = catalogue.INDEX_PATH.stat().st_mtime
        except OSError:
            return getattr(self, "_catalogue", None)
        if mtime != getattr(self, "_catalogue_mtime", None):
            self._catalogue = catalogue.load()
            self._catalogue_mtime = mtime
        return getattr(self, "_catalogue", None)

    def catalogue_index(self):
        """The local track index (or None), for radio and other read-only users."""
        return self._catalogue_index()

    def resolve_local(self, query: str) -> Optional[Candidate]:
        """A cached or catalogue-confident match for ``query`` without any network call."""
        return self.cached_track(query) or self._resolve_from_catalogue(query)

    def _resolve_from_catalogue(self, query: str) -> Optional[Candidate]:
        """A confident local match, or None to run the search-based path.

        Never raises: this is an accelerator in front of the existing logic,
        so any failure here must leave that logic exactly as it was.
        """
        try:
            hit = _catalogue_hit(query, self._catalogue_index())
        except Exception:
            log.exception("Catalogue lookup failed; using search")
            return None
        if not hit:
            return None
        entry, score, rival = hit
        artists = entry.get("artists") or []
        rival_artists = (rival.entry.get("artists") or []) if rival is not None else []
        self._note_resolution(
            "catalogue", score,
            "{} by {}".format(rival.entry.get("name", ""), rival_artists[0] if rival_artists else "")
            if rival is not None else "",
            score - rival.score if rival is not None else None)
        log.info("Resolved %r from catalogue -> %r (score %.2f)", query,
                 "{} by {}".format(entry["name"], artists[0] if artists else ""), score)
        return Candidate(entry["uri"], entry["name"], artists[0] if artists else "",
                         metadata={"uri": entry["uri"], "name": entry["name"],
                                   "artists": [{"name": a} for a in artists],
                                   "album": {"name": entry.get("album", "")},
                                   # Lets callers tell an exact hit from a repaired one.
                                   "match_score": score})

    def _local_confirmation_proposals(self, query: str, index) -> list[Candidate]:
        """Build local-only choices after the confident catalogue gate declines."""
        if index is None:
            return []
        exact_proposals = []
        near_proposals = []
        boundary = re.split(r"\s+by\s+", query, maxsplit=1, flags=re.I)
        requested_title = boundary[0]

        def add(entry, *, score=None, title_sim=None, artist_sim=None):
            if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
                return None
            artists = [name for name in entry.get("artists") or []
                       if isinstance(name, str) and name.strip()]
            candidate = Candidate(
                entry.get("uri", ""), entry.get("name", ""),
                artists[0] if artists else "",
                {**entry, "artists": [{"name": name} for name in artists],
                 "resolution_proposal": True,
                 **({"match_score": score} if score is not None else {}),
                 **({"title_similarity": title_sim} if title_sim is not None else {}),
                 **({"artist_similarity": artist_sim} if artist_sim is not None else {})})
            if eligible_media(candidate, "track", requested_title):
                return candidate
            return None

        # Exact title is useful even when the supplied artist is wrong or
        # unfamiliar, but remains a proposal and can never satisfy resolution.
        if len(boundary) == 2:
            wanted = title_key(requested_title, requested_title)
            heard_artist = album_key(boundary[1])
            for entry in index.entries:
                if isinstance(entry, dict) and isinstance(entry.get("name"), str) \
                        and title_key(entry["name"], requested_title) == wanted:
                    candidate = add(entry, title_sim=1.0)
                    if candidate is not None:
                        credits = [album_key(name) for name in entry.get("artists", [])
                                   if isinstance(name, str)]
                        similarity = max((SequenceMatcher(None, heard_artist, name).ratio()
                                          for name in credits), default=0.0)
                        exact_proposals.append((similarity, candidate))

        # A recognizer may add a word to the title while preserving a strong
        # artist sound. Keep this deliberately separate from auto-acceptance.
        try:
            ranked = catalogue.match(query, index, limit=0)
        except Exception:
            log.exception("Could not assemble local music proposals")
            ranked = []
        for scored in ranked:
            if scored.title_sim >= 0.60 and scored.artist_sim >= 0.88:
                candidate = add(scored.entry, score=scored.score,
                                title_sim=scored.title_sim, artist_sim=scored.artist_sim)
                if candidate is not None:
                    near_proposals.append(candidate)

        # Reissues do not create duplicate choices. Keep the best-ranked
        # occurrence and cap the prompt at three local candidates.
        exact_proposals.sort(key=lambda row: row[0], reverse=True)
        proposals = [candidate for _, candidate in exact_proposals] + near_proposals
        unique = collapse_reissues(proposals, requested_title)
        return unique[:3]

    def resolve_track(self, query: str) -> Candidate:
        local = self._resolve_from_catalogue(query)
        if local is not None:
            return local
        proposals = self._local_confirmation_proposals(query, self._catalogue_index())
        if proposals:
            self._note_resolution("local_confirmation")
            raise MusicConfirmationRequired(query, proposals)
        # "<track> by <artist>" becomes a fielded query. Spotify matches
        # track:/artist: far more precisely than one free-text blob, which
        # matters when the STT has already mangled part of the name.
        search = query
        if " by " in query:
            track, _, artist = query.rpartition(" by ")
            if track.strip() and artist.strip():
                search = "track:{} artist:{}".format(track.strip(),
                                                     artist.strip())
        via = "search_fielded" if search != query else "search_free"
        items = self._search_music(search, "track")
        if not items and search != query:
            via = "search_free"
            items = self._search_music(query, "track")
        title, separator, artist = query.rpartition(" by ")
        if not separator:
            title, artist = query, ""
            # "by" is routinely lost or heard as "I" / a comma. Recover the
            # boundary from the artist list before giving up on the artist.
            recovered_title, recovered_artist = self._split_on_known_artist(query)
            if recovered_artist:
                log.info("Recovered artist boundary: %r -> title=%r artist=%r",
                         query, recovered_title, recovered_artist)
                title, artist = recovered_title, recovered_artist
                via += "+artist_boundary"
        ranked = rank_tracks(items, title if artist else query, artist if artist else None)
        if artist and not ranked:
            # A title can be heard clearly while a name is spelt by sound:
            # "Sidaimoth" vs "Seedhe Maut". Search by the exact title, then
            # allow only one artist whose consonant outline closely matches.
            # A weaker outline can be accepted only when the original spelling
            # also agrees and is clearly ahead of every competing artist with
            # the same title. Never substitute popularity for name evidence.
            title_items = self._search_music("track:{}".format(title.strip()), "track")
            heard = _spoken_consonants(artist)
            plausible = []
            if len(heard) >= 3:
                for item in title_items:
                    if _norm(item.get("name", "")) != _norm(title):
                        continue
                    for credited in item.get("artists") or []:
                        name = credited.get("name", "")
                        outline = _spoken_consonants(name)
                        if len(outline) >= 3:
                            sound = SequenceMatcher(None, heard, outline).ratio()
                            if sound >= 0.75:
                                spelling = SequenceMatcher(None, _norm(artist),
                                                           _norm(name)).ratio()
                                plausible.append((item, name, sound, spelling))
            # Preserve the stricter path for strong sound matches. When sound
            # alone ties, spelling is not enough to break that tie safely.
            close = [(item, name) for item, name, sound, _ in plausible
                     if sound >= 0.8]
            if not close and plausible:
                # The slightly weaker phonetic case needs corroboration from
                # the unstripped letters and a decisive margin over other
                # plausible exact-title artists. This distinguishes a heard
                # "See The Moth" from "Aerosmith" without a name alias.
                by_artist = {}
                for _, name, _, spelling in plausible:
                    key = _norm(name)
                    by_artist[key] = max(by_artist.get(key, 0.0), spelling)
                winner = max(by_artist, key=by_artist.get)
                runner_up = max((score for key, score in by_artist.items()
                                 if key != winner), default=0.0)
                if by_artist[winner] >= 0.68 and by_artist[winner] - runner_up >= 0.15:
                    close = [(item, name) for item, name, _, _ in plausible
                             if _norm(name) == winner]
            distinct_artists = {_norm(name) for _, name in close}
            if len(distinct_artists) == 1:
                corrected = close[0][1]
                ranked = rank_tracks([item for item, _ in close], title, corrected)
                if ranked:
                    log.info("Recovered spoken artist %r as %r for exact title %r",
                             artist, corrected, title)
                    artist = corrected
                    via += "+exact_title_artist"
        if artist and not ranked:
            # Both words can be spelt by sound: "Madeira by Seedemoth" is
            # "Madira by Seedhe Maut". The exact-title search above cannot
            # see that track. A broad title search may, but only accept one
            # near-title track whose primary artist strongly matches the
            # spoken consonants AND the original spelling.
            broad = self._search_music(title.strip(), "track")
            heard_sound = _spoken_consonants(artist)
            plausible = {}
            if len(heard_sound) >= 3:
                for item in broad:
                    name = item.get("name", "")
                    credited = (item.get("artists") or [{}])[0].get("name", "")
                    if (SequenceMatcher(None, _norm(title), _norm(name)).ratio() < 0.85
                            or SequenceMatcher(None, heard_sound,
                                               _spoken_consonants(credited)).ratio() < 0.85
                            or SequenceMatcher(None, _norm(artist),
                                               _norm(credited)).ratio() < 0.65):
                        continue
                    plausible.setdefault((_norm(name), _norm(credited)), (item, credited))
            if len(plausible) == 1:
                item, corrected = next(iter(plausible.values()))
                ranked = rank_tracks([item], title, corrected)
                if ranked:
                    log.info("Recovered spoken title/artist %r as %r",
                             query, ranked[0].label)
                    artist = corrected
                    via += "+broad_title_artist"
        # A weak top hit means search returned something loosely similar but may
        # never have surfaced the right song at all: "Nachshavai by Seedhe Maut"
        # returned only "Nanchaku" (0.556) and played it, while the intended
        # "Naksha" (0.625) was absent from the results entirely. Absolute
        # similarity cannot separate those cases -- a wrong match scored 0.667
        # where a right one scored 0.625 -- so let the artist's real catalogue
        # compete and decide on margin instead.
        if artist and (not ranked or not _strong_match(title, ranked[0])):
            # Recognition mangles titles far more often than artist names, and
            # Spotify's own spelling may differ from what was said. Measured on
            # a real recording: "Nakhshab"/"Nachshavai" by Seedhe Maut both
            # return nothing from search, while ranking the same words against
            # that artist's catalogue puts "Naksha" first.
            #
            # This is a general mechanism, not a per-song alias: the candidate
            # set is whatever that artist actually released, and the existing
            # rank_tracks thresholds and ambiguity rules still apply.
            catalogue = self._artist_catalogue(artist)
            if catalogue:
                merged = {t.get("uri"): t for t in items if t.get("uri")}
                merged.update({t.get("uri"): t for t in catalogue if t.get("uri")})
                ranked = collapse_reissues(rank_tracks(list(merged.values()), title, artist), title)
                if ranked:
                    # Everything reaching here is a guess about a title the
                    # recogniser already got wrong, so require the winner to be
                    # clearly ahead. Without this, "Nachshavai" silently played
                    # "Nanchaku" instead of "Naksha" -- a wrong action, which is
                    # worse than asking.
                    if not _clearly_best(title, ranked):
                        self._note_resolution(via + ">artist_catalogue:ambiguous",
                                              *_title_diag(title, ranked))
                        raise AmbiguousMatch(query, ranked[:3])
                    # A clear winner inside the artist's own catalogue is a
                    # deliberate confidence decision, so it returns here rather
                    # than falling through to the exact-title rule below --
                    # which would reject every recovered spelling by definition.
                    log.info("Resolved %r via %s's catalogue -> %r",
                             title.strip(), artist.strip(), ranked[0].name)
                    self._note_resolution(via + ">artist_catalogue", *_title_diag(title, ranked))
                    return ranked[0]
        if not ranked:
            self._note_resolution(via + ":not_found")
            raise TrackLookupError("No track found for " + repr(query))
        # Same song on several releases is not a real choice.
        requested_title = title if artist else query
        ranked = collapse_reissues(ranked, requested_title)
        # Spotify search often returns one loosely related item for a title it
        # did not actually find. A lone result is not evidence of correctness:
        # "See the Moth" once resolved to "See The Light" (0.72 similarity).
        if not artist and SequenceMatcher(None, title_key(query, query),
                                           title_key(ranked[0].name, query)).ratio() < 0.85:
            self._note_resolution(via + ":low_confidence", *_title_diag(query, ranked))
            raise TrackLookupError("No confident track match for " + repr(query))
        if len(ranked) > 1:
            # An exact title beats irrelevant fuzzy results. Same-title
            # alternatives remain ambiguous unless the artist was supplied.
            requested = title_key(requested_title, requested_title)
            exact_title = title_key(ranked[0].name, requested_title) == requested
            credits = (ranked[0].metadata or {}).get("artists") or [{"name": ranked[0].artist}]
            exact_artist = bool(artist and any(_norm(row.get("name", "")) == _norm(artist)
                                              for row in credits))
            competing_same_title = title_key(ranked[1].name, requested_title) == requested
            rival_credits = (ranked[1].metadata or {}).get("artists") or [{"name": ranked[1].artist}]
            rival_exact_artist = bool(artist and any(_norm(row.get("name", "")) == _norm(artist)
                                                    for row in rival_credits))
            if not (exact_title and (not competing_same_title or exact_artist and not rival_exact_artist)):
                self._note_resolution(via + ":ambiguous", *_title_diag(requested, ranked))
                raise AmbiguousMatch(query, ranked[:3])
        self._note_resolution(via, *_title_diag(title if artist else query, ranked))
        return ranked[0]

    def _search_music(self, query: str, kind: str) -> list[dict]:
        """Remember actual search responses in memory, without creating preference aliases."""
        result = self._search_page(query, kind, limit=SEARCH_PAGE)
        if isinstance(result, _SearchBudgetExhausted):
            return []
        rows = [{**row, "spotify_relevance": i} if isinstance(row, dict) else row
                for i, row in enumerate((result.get(kind + "s") or {}).get("items") or [])]
        cache = getattr(self, "_music_search_cache", None)
        if not isinstance(cache, dict):
            cache = self._music_search_cache = {}
        cache[(kind, query)] = rows
        while len(cache) > 100:
            cache.pop(next(iter(cache)))
        return rows

    @staticmethod
    def _media_candidates(rows: list[dict], kind: str) -> list[Candidate]:
        out = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            credits = [artist.get("name", "") for artist in row.get("artists") or []
                       if isinstance(artist, dict) and isinstance(artist.get("name"), str)]
            display = ", ".join(credits) if kind == "album" else (credits[0] if credits else "")
            out.append(Candidate(row.get("uri", ""), row.get("name", ""), display, row))
        return out

    def _saved_music_candidates(self, query: str, kind: str) -> list[Candidate]:
        cache = getattr(self, "_music_search_cache", None) or {}
        rows = cache.get((kind, query), [])
        candidates = self._media_candidates(rows, kind)
        if kind != "track":
            return candidates
        index = self._catalogue_index()
        entries = getattr(index, "entries", []) if index is not None else []
        for row in entries if isinstance(entries, list) else []:
            if not isinstance(row, dict):
                continue
            credits = [name for name in row.get("artists") or [] if isinstance(name, str)]
            metadata = {**row, "artists": [{"name": name} for name in credits]}
            candidates.append(Candidate(row.get("uri", ""), row.get("name", ""),
                                        credits[0] if credits else "", metadata))
        for row in self._track_cache().values():
            if isinstance(row, dict):
                metadata = {**row, "artists": row.get("artists") or [{"name": row.get("artist", "")}]}
                candidates.append(Candidate(row.get("uri", ""), row.get("name", ""),
                                            row.get("artist", ""), metadata))
        return candidates

    def _partial_saved_artist(self, candidates: list[Candidate], title: str, artist: str) -> str:
        """One damaged first name can identify one known credit on an exact saved title."""
        words = _norm(artist).split()
        if len(words) != 1 or len(words[0]) < 4 or not words[0].isalpha():
            return ""
        known = {album_key(name) for name in getattr(self, "_known_artists", [])}
        matches = {}
        for candidate in candidates:
            if (not eligible_media(candidate, "track", title)
                    or " ".join(_norm(candidate.name).split()) != " ".join(_norm(title).split())):
                continue
            for credit in (candidate.metadata or {}).get("artists") or []:
                name = credit.get("name", "") if isinstance(credit, dict) else ""
                parts = _norm(name).split()
                if (not parts or len(parts[0]) < 4 or not parts[0].isalpha()
                        or SequenceMatcher(None, words[0], parts[0]).ratio() < 0.65
                        or SequenceMatcher(None, _spoken_consonants(words[0]),
                                           _spoken_consonants(parts[0])).ratio() < 0.80):
                    continue
                matches[album_key(name)] = name
        return next(iter(matches.values())) if len(matches) == 1 and next(iter(matches)) in known else ""

    def resolve_best_available(self, query: str, kind: str = "track",
                               candidates: tuple[Candidate, ...] = (), *, offline_only: bool = False) -> Candidate:
        """Choose available source evidence without confusing a ban with a missing song."""
        if kind not in {"track", "album"}:
            raise ValueError("Unsupported music entity")
        title, separator, artist = query.rpartition(" by ")
        if not separator:
            title, artist = query, ""
            if kind == "track":
                recovered_title, recovered_artist = self._split_on_known_artist(query)
                words = query.split()
                credible = {album_key(name) for cut in range(1, len(words))
                            for name in getattr(self, "_known_artists", [])
                            if SequenceMatcher(None, album_key(" ".join(words[cut:])),
                                               album_key(name)).ratio() >= 0.90}
                if recovered_artist and credible == {album_key(recovered_artist)}:
                    title, artist = recovered_title, recovered_artist
        if not title.strip():
            raise LookupError("Empty music query")
        chosen = best_available_candidate(list(candidates), title, artist, kind)
        if chosen is not None:
            self._note_resolution("best_available_bound_" + kind, *_title_diag(title, [chosen]))
            return chosen
        banned = self.web_rate_limited
        if offline_only or banned:
            saved = self._saved_music_candidates(query, kind)
            cached_rows = (getattr(self, "_music_search_cache", None) or {}).get((kind, query), [])
            chosen = best_available_candidate(self._media_candidates(cached_rows, kind), title, artist, kind)
            if chosen is not None:
                self._note_resolution("best_available_cached_search_" + kind, *_title_diag(title, [chosen]))
                return chosen
            if not artist:
                wanted = set(title_key(title, title).split()) - {
                    "the", "a", "an", "of", "and", "song", "track", "music", "by", "to", "in", "my"}
                # Edition constraints describe the recording, not its song title.
                # Literal titles such as "Live" still pass the strong-match gate.
                wanted -= {term.casefold() for term in _VERSION_TERMS.findall(title)} | {"take"}
                saved = [candidate for candidate in saved if isinstance(candidate.name, str) and (
                    _strong_match(title, candidate)
                    or any(len(word) >= 3 and word.isalpha()
                           for word in wanted.intersection(title_key(candidate.name, title).split())))]
            chosen = best_available_candidate(saved, title, artist, kind,
                                              rank_title=True)
            repaired_artist = ""
            if chosen is None and artist and kind == "track":
                repaired_artist = self._partial_saved_artist(saved, title, artist)
                if repaired_artist:
                    exact_title_rows = [candidate for candidate in saved
                                        if isinstance(candidate.name, str)
                                        and " ".join(_norm(candidate.name).split()) == " ".join(_norm(title).split())]
                    chosen = best_available_candidate(exact_title_rows, title, repaired_artist, kind,
                                                      rank_title=True)
            if chosen is None:
                self._note_resolution("best_available_saved_" + kind + ":unavailable")
                if banned:
                    raise SpotifyRateLimited("Spotify search is temporarily unavailable")
                raise LookupError("No eligible saved music for " + repr(query))
            path = "best_available_saved_" + kind + ("+partial_artist" if repaired_artist else "")
            self._note_resolution(path, *_title_diag(title, [chosen]))
            return chosen
        cached = (getattr(self, "_music_search_cache", None) or {}).get((kind, query))
        rows = cached if cached is not None else self._search_music(query, kind)
        chosen = best_available_candidate(self._media_candidates(rows, kind), title, artist, kind)
        if chosen is None:
            self._note_resolution("best_available_search_" + kind + ":not_found")
            raise LookupError("No eligible music for " + repr(query))
        self._note_resolution("best_available_search_" + kind, *_title_diag(title, [chosen]))
        return chosen

    def resolve_album(self, query: str, *, offline_only: bool = False) -> Candidate:
        """Resolve a near-exact album, never an artist catalogue or song hypothesis."""
        title, separator, artist = query.rpartition(" by ")
        if not separator:
            title, artist = query, ""
        search = "album:{} artist:{}".format(title.strip(), artist.strip()) if artist else title.strip()
        if not title.strip():
            raise LookupError("Empty album title")
        cache = getattr(self, "_music_search_cache", None) or {}
        lookup = (lambda text: list(cache.get(("album", text), []))) if offline_only else (
            lambda text: self._search_music(text, "album"))
        rows = lookup(search)
        if artist and not rows:
            rows = lookup(query)
        # A collaborator need not be indexed by Spotify's artist field. The
        # unfielded title search still applies the exact credited-artist gate.
        if artist:
            if not rank_albums(rows, title, artist):
                rows = rows + lookup(title.strip())
        ranked = rank_albums(rows, title, artist)
        if not ranked:
            self._note_resolution("album_search:not_found")
            raise LookupError("No confident album match for " + repr(query))
        score, winner = ranked[0]
        second = ranked[1][0] if len(ranked) > 1 else None
        margin = score - second if second is not None else None
        if second is not None and margin < 0.10:
            self._note_resolution("album_search:ambiguous", score, ranked[1][1].label, margin)
            raise AmbiguousMatch(query, [candidate for _, candidate in ranked[:3]])
        self._note_resolution("album_search", score,
                              ranked[1][1].label if second is not None else "", margin)
        return winner

    def play_album(self, candidate: Candidate) -> str:
        """Start the album context, honoring the player's existing shuffle preference."""
        if not candidate.uri.startswith("spotify:album:"):
            raise ValueError("Album context required")
        self.radio_played = False
        self.last_playback = None
        # Native title/album/credit metadata cannot prove album context: a
        # playlist can play that same recording. Rate limits therefore propagate.
        with self.command_target():
            self._call(self.sp.start_playback, context_uri=candidate.uri,
                       offset={"position": 0}, position_ms=0)
            self._verify(lambda state: state.get("is_playing") is True and
                         (state.get("context") or {}).get("uri") == candidate.uri,
                         "requested album")
        self.last_playback = {"method": "web_api_album_verified", "radio": 0}
        return candidate.label

    def _now_playing(self, candidate: Candidate, timeout: float) -> bool:
        """Confirm through Windows' media info, not by polling the Web API."""
        from . import local_media
        return local_media._wait_for(
            lambda: _matches_local_candidate(candidate, local_media.snapshot()),
            timeout=timeout)

    def _play_uri_locally(self, candidate: Candidate) -> bool:
        """Open the track in the desktop app, which starts playing it.

        Used while the player API is unavailable (rate limit). Opening a link
        brings Spotify to the front, so its window is kept see-through, focus
        is handed straight back, and it is re-minimised once the song plays.
        """
        if not self._local_enabled() or not candidate.uri.startswith("spotify:track:"):
            return False
        try:
            from . import local_media
        except ImportError:
            return False
        current = local_media.snapshot()
        if current is None and not self._client_running():
            return False
        if _matches_local_candidate(candidate, current, require_playing=False):
            if current.playing:
                return True
            # A paused same-track session is only success after SMTC reports
            # the requested title and artist in the playing state.
            if local_media.play() and self._now_playing(candidate, timeout=1.5):
                return True
        import threading
        # Measured: while something is playing, opening a track link only
        # navigates; when paused, it starts that track (~2.5 s).
        local_media.pause()
        with local_media.InvisibleSpotify() as window:
            try:
                os.startfile(candidate.uri)
            except OSError as exc:
                log.warning("Could not open %s in the desktop app: %s", candidate.uri, exc)
                return False
            threading.Thread(target=window.return_focus, daemon=True).start()
            if self._now_playing(candidate, timeout=8.0):
                return True
        log.warning("Desktop app did not start %r", candidate.label)
        return False

    def play_candidate(self, candidate: Candidate, then: Optional[list] = None) -> str:
        """Start one track: Web API first (no window, fastest), else the app link.

        ``then`` is a radio: track URIs sent in the same start_playback call, so
        the song plays first and the radio follows. Verification still checks
        only the first track. The desktop-link fallback opens a single track and
        so cannot carry a radio.
        """
        started = time.perf_counter()
        follow = [u for u in dict.fromkeys(then or [])
                  if isinstance(u, str) and u.startswith("spotify:track:")
                  and u != candidate.uri][:RADIO_MAX_TRACKS]
        uris = [candidate.uri] + follow
        # True only when the radio really went out; callers record it for variety.
        self.radio_played = False
        self.last_playback = None
        # Why the Web API did not carry this playback, for telemetry.
        skipped = ""
        if self._local_enabled():
            api_error = None
            accepted = False
            try:
                with self.command_target():
                    self._call(self.sp.start_playback, uris=uris)
                    accepted = True
                    if self._now_playing(candidate, timeout=4.0):
                        self.radio_played = bool(follow)
                        self.last_playback = {"method": "web_api", "radio": len(follow),
                                              "ms": int((time.perf_counter() - started) * 1000)}
                        log.info("Playing %r via Web API in %.0f ms (+%d radio)", candidate.label,
                                 (time.perf_counter() - started) * 1000, len(follow))
                        return candidate.label
                    # Native metadata can miss a newly accepted command. A read-only
                    # player check on this same pinned target is the final proof.
                    log.info("Native confirmation missed accepted playback; checking track URI on device %r",
                             self.device_name)
                    try:
                        self._verify(lambda s: s.get("is_playing") is True and (
                            (s.get("item") or {}).get("uri") == candidate.uri
                            or ((s.get("item") or {}).get("linked_from") or {}).get("uri") == candidate.uri
                        ), "requested track")
                    except Exception:
                        log.warning("Could not verify accepted playback on device %r for track %s",
                                    self.device_name, candidate.uri)
                        raise
                    self.radio_played = bool(follow)
                    self.last_playback = {"method": "web_api_verified", "radio": len(follow),
                                          "native_confirmation": "missed",
                                          "ms": int((time.perf_counter() - started) * 1000)}
                    return candidate.label
            except SpotifyRequestDeferred as exc:
                if accepted:
                    raise
                api_error = exc
                skipped = "local_request_deferred"
            except SpotifyRateLimited as exc:
                if accepted:
                    raise
                api_error = exc
                skipped = "rate_limited"
            except (spotipy.SpotifyException, PlaybackVerificationError, RuntimeError, OSError) as exc:
                if accepted:
                    raise
                api_error = exc
                skipped = type(exc).__name__
                log.info("Web API playback unavailable (%s); opening the track in the app", exc)
            try:
                desktop_started = self._play_uri_locally(candidate)
            except Exception as exc:
                if api_error is None:
                    raise
                log.info("Desktop playback fallback failed (%s)", exc)
                raise api_error from exc
            if desktop_started:
                self.last_playback = {"method": "desktop_uri", "radio": 0,
                                      "radio_skipped": len(follow), "web_api_skipped": skipped,
                                      "ms": int((time.perf_counter() - started) * 1000)}
                if follow:
                    # Opening a track link plays that one track, then continues in
                    # its album context, which is acceptable for this fallback.
                    log.info("Radio skipped: the desktop app link plays a single track")
                log.info("Playing %r via the desktop app in %.0f ms", candidate.label,
                         (time.perf_counter() - started) * 1000)
                return candidate.label
            if api_error is not None:
                raise api_error
            # An accepted request always returns or raises through the pinned
            # verification path above; it must never reach another mutation.
            raise PlaybackVerificationError("Could not verify requested track after accepted playback")
        with self.command_target():
            self._call(self.sp.start_playback, uris=uris)
            self._verify(lambda s: s.get("is_playing") is True and (
                (s.get("item") or {}).get("uri") == candidate.uri
                or ((s.get("item") or {}).get("linked_from") or {}).get("uri") == candidate.uri
            ), "requested track")
        self.radio_played = bool(follow)
        self.last_playback = {"method": "web_api_verified", "radio": len(follow),
                              "web_api_skipped": skipped,
                              "ms": int((time.perf_counter() - started) * 1000)}
        return candidate.label

    def queue(self, candidate: Candidate) -> None:
        """Add one track to the end of the user's queue on the pinned device.

        Player-scope call, so a rate limit raises SpotifyRateLimited; the desktop
        app has no way to queue from outside, so there is no local fallback.
        """
        with self.command_target():
            self._call(self.sp.add_to_queue, candidate.uri)

    # ---------- remembered tracks ----------

    def cached_track(self, query: str) -> Optional[Candidate]:
        """Return only an exact title-and-artist cache match."""
        hit = self._track_cache().get(_norm(query))
        separators = list(re.finditer(r"\s+by\s+", query, flags=re.IGNORECASE))
        if not hit or not separators:
            return None
        separator = separators[-1]
        title, artist = query[:separator.start()].strip(), query[separator.end():].strip()
        if not title or not artist:
            return None
        credits = hit.get("artists") or [{"name": hit.get("artist", "")}]
        if (_norm(title) != _norm(hit.get("name", ""))
                or not any(_norm(artist) == _norm(row.get("name", "")) for row in credits)):
            return None
        self._note_resolution("track_cache", 1.0)
        return Candidate(hit["uri"], hit["name"], hit["artist"],
                         {"artists": credits} if hit.get("artists") else None)

    def remember_track(self, query: str, candidate: Candidate) -> None:
        if candidate.approximate:
            return
        cache = self._track_cache()
        key = _norm(query)
        if not key or not candidate.uri:
            return
        cache[key] = {"uri": candidate.uri, "name": candidate.name, "artist": candidate.artist}
        credits = (candidate.metadata or {}).get("artists")
        if credits:
            cache[key]["artists"] = credits
        while len(cache) > 500:
            cache.pop(next(iter(cache)))
        try:
            TRACK_CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
        except OSError as exc:
            log.debug("Could not save track cache: %s", exc)

    def _track_cache(self) -> dict:
        cache = getattr(self, "_tracks", None)
        if cache is None:
            try:
                cache = json.loads(TRACK_CACHE.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                cache = {}
            self._tracks = cache
        return cache

    def resolve_artist_track(self, name: str) -> Candidate:
        """A popular song by the named artist, for "play some <artist>"."""
        target = _norm(name)
        try:
            items = (self._search_page(name, "artist", limit=5)
                     .get("artists", {}).get("items", []))
        except SpotifyRateLimited:
            # Search is banned, but the local index still knows this
            # artist's songs; the user's own top/playlist tracks come first.
            candidate = self._artist_from_catalogue(name)
            if candidate is None:
                self._note_resolution("artist_search:rate_limited")
                raise
            self._note_resolution("artist_catalogue_offline")
            return candidate
        scored = sorted(((SequenceMatcher(None, target, _norm(a.get("name", ""))).ratio(), a)
                         for a in items if a), key=lambda pair: -pair[0])
        if not scored or scored[0][0] < 0.8:
            self._note_resolution("artist_search:not_found", scored[0][0] if scored else None)
            raise LookupError("No artist found for " + repr(name))
        artist = scored[0][1]
        tracks = []
        try:
            tracks = self.sp.artist_top_tracks(artist["id"]).get("tracks", [])
        except spotipy.SpotifyException as exc:
            log.debug("Top tracks unavailable (%s); using the catalogue", exc)
        if not tracks:
            tracks = self._artist_catalogue(artist["name"], pages=1)
        if not tracks:
            raise LookupError("No tracks found for " + repr(name))
        top = tracks[0]
        self._note_resolution(
            "artist_search", scored[0][0],
            scored[1][1].get("name", "") if len(scored) > 1 else "",
            scored[0][0] - scored[1][0] if len(scored) > 1 else None)
        return Candidate(top["uri"], top.get("name", ""), artist["name"], metadata=top)

    def _artist_from_catalogue(self, name: str) -> Optional[Candidate]:
        index = self._catalogue_index()
        if index is None:
            return None
        target = _norm(name)
        rank = {"top": 0, "saved": 0, "playlist": 1}
        best = None
        for entry in getattr(index, "entries", None) or []:
            credited = entry.get("artists") or []
            if not credited or SequenceMatcher(None, target, _norm(credited[0])).ratio() < 0.9:
                continue
            order = min((rank.get(s, 2) for s in entry.get("sources") or []), default=2)
            if best is None or order < best[0]:
                best = (order, entry)
        if best is None:
            return None
        entry = best[1]
        log.info("Artist %r from catalogue (search rate-limited) -> %r", name, entry.get("name"))
        return Candidate(entry["uri"], entry.get("name", ""), entry["artists"][0], metadata=entry)

    def play_track(self, query: str) -> str:
        return self.play_candidate(self.resolve_track(query))

    def play_playlist(self, name: str, library_only: bool = False, before_play=None) -> str:
        """Search the user's own playlists first.

        "my chill playlist" should hit your library, not a stranger's.

        Exact matches take precedence, then unique whole-token containment.
        Fuzzy matches require clarification rather than automatic playback.
        library_only restricts selection to saved playlists, including followed
        playlists; it does not mean only playlists created by this account.
        """
        import difflib

        target = _norm(name)
        if not target:
            raise LookupError("Empty playlist name")
        playlists = []
        page = self.sp.current_user_playlists(limit=50)
        while page:
            playlists.extend(pl for pl in page.get("items", []) if pl)
            page = self.sp.next(page) if page.get("next") else None

        def resolve(items):
            rows = [Candidate(p.get("uri", ""), p.get("name", ""), metadata=p)
                    for p in items if p and p.get("uri") and _norm(p.get("name", ""))]
            exact = [c for c in rows if _norm(c.name) == target]
            if exact:
                if len({c.uri for c in exact}) == 1:
                    return exact[0]
                raise AmbiguousMatch(name, exact)
            wanted = set(target.split())
            contained = [c for c in rows if wanted and
                         (wanted <= set(_norm(c.name).split()) or
                          set(_norm(c.name).split()) <= wanted)]
            if contained:
                if len({c.uri for c in contained}) == 1:
                    return contained[0]
                raise AmbiguousMatch(name, contained)
            fuzzy = [c for c in rows if difflib.SequenceMatcher(None, target, _norm(c.name)).ratio() >= 0.6]
            if fuzzy:
                raise AmbiguousMatch(name, fuzzy[:5])
            return None

        best = resolve(playlists)
        if best is None and not library_only:
            found = self._search_page(name, "playlist", limit=10)
            best = resolve(found.get("playlists", {}).get("items", []))
        if best is None:
            raise LookupError("No playlist found for " + repr(name))

        if before_play is not None:
            before_play()
        with self.command_target():
            self._call(self.sp.start_playback, context_uri=best.uri)
            self._verify(lambda s: s.get("is_playing") is True and
                         (s.get("context") or {}).get("uri") == best.uri,
                         "requested playlist")
        return best.name

    def playlist_names(self) -> list:
        """Just the user's playlist names, for query disambiguation."""
        names = []
        try:
            page = self.sp.current_user_playlists(limit=50)
            while page:
                names.extend(pl["name"] for pl in page.get("items", []) if pl)
                page = self.sp.next(page) if page.get("next") else None
        except spotipy.SpotifyException as exc:
            log.warning("Could not fetch playlist names: %s", exc)
        return names

    def artist_names(self) -> list:
        """The user's top artists, for repairing mangled names in queries."""
        try:
            return [a["name"] for a in
                    self.sp.current_user_top_artists(limit=50).get("items", [])]
        except spotipy.SpotifyException as exc:
            log.warning("Could not fetch artist names: %s", exc)
            return []

    def vocabulary(self) -> list:
        """Playlist and artist names, used to bias STT decoding.

        Proper nouns are ASR's weak spot, and every 'play X' command is one.
        """
        words = []
        try:
            # Paginated, like playlist_names(). Fetching only the first
            # page silently dropped playlists past the 50th from the
            # decoding hints.
            words.extend(self.playlist_names())
            top = self.sp.current_user_top_artists(limit=50)
            words.extend(a["name"] for a in top.get("items", []))
        except spotipy.SpotifyException as exc:
            log.warning("Could not fetch vocabulary hints: %s", exc)
        return words


def main() -> None:
    """Checkpoint CLI: everything downstream is decoration once this works."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description="Spotify layer smoke test")
    ap.add_argument("--auth", action="store_true", help="Run interactive authorization")
    ap.add_argument("--play", metavar="QUERY", help="Search and play a track")
    ap.add_argument("--playlist", metavar="NAME", help="Play a playlist by name")
    ap.add_argument("--pause", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--next", action="store_true")
    ap.add_argument("--previous", action="store_true")
    ap.add_argument("--volume", type=int, metavar="PCT")
    ap.add_argument("--status", action="store_true", help="Device, volume, token age")
    args = ap.parse_args()

    sp = Spotify()
    sp.check_auth()

    if args.auth or args.status:
        print("Authorized. Device:", sp.device(), "(" + sp.device_name + ")")
        print("Volume:", sp.get_volume())
        age = sp.token_age_days()
        if age is not None:
            print(
                    "Token cache age {:.1f} days (diagnostic only)".format(
                    age
                )
            )
    if args.play:
        print("Playing:", sp.play_track(args.play))
    if args.playlist:
        print("Playing playlist:", sp.play_playlist(args.playlist))
    if args.pause:
        sp.pause()
        print("Paused")
    if args.resume:
        sp.play_resume()
        print("Resumed")
    if args.next:
        sp.next()
        print("Skipped")
    if args.previous:
        sp.previous()
        print("Back")
    if args.volume is not None:
        sp.set_volume(args.volume)
        print("Volume:", args.volume)


if __name__ == "__main__":
    main()
