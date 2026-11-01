"""Phase 1b music adapter: albums, artists, playlists, queueing and track navigation.

Mutations go only through the existing assistant/spotify.py methods. Resolution is
read-only and happens before the guard that precedes a mutation. A receipt is
observed only where spotify.py verifies the effect; queueing has no read-back here,
so it is requested. Replies are fixed templates and never include heard text.
"""
from __future__ import annotations

import contextlib
import json
import re
import time

from ..playback import guarded_playback, remaining
from .base import Adapter, Clause, Guard, Receipt, TurnExpired

_TOKEN = re.compile(r"[^\W_]+")
_EXACT_TTL_SECONDS = 10.0


class _NotDispatched(Exception):
    """A refusal raised before any mutation was sent."""


def _norm(value) -> tuple:
    return tuple(_TOKEN.findall(str(value).casefold()))


def _query(args: dict) -> str:
    artist = args.get("artist", "")
    return args["title"] + (" by " + artist if artist else "")


class MusicHandler:
    mutating = True
    area = "music"
    executor = None                 # set by the engine; its Executor owns the Spotify client
    verbs: tuple = ()
    object_kind = "none"
    effect_words: tuple = ()
    bounded = True                  # False where spotify.py takes its local-media path
    needs_lookup = True
    replies: dict = {}
    _exact_cache = None

    def _args(self, obj: str):
        raise NotImplementedError

    def license(self, clause: Clause) -> dict | None:
        if not isinstance(clause, Clause) or clause.verb not in self.verbs:
            return None
        return self._args(clause.object)

    def reply(self, receipt: Receipt, args: dict) -> str:
        return self.replies.get(receipt.state, self.replies["unknown"])

    def _spotify(self):
        if self.executor is None:
            raise RuntimeError("music adapter has no executor")
        if getattr(self.executor, "spotify", None) is None:
            from ...spotify import Spotify
            self.executor.spotify = Spotify()
        return self.executor.spotify

    def _context(self, spotify, guard: Guard):
        if not self.bounded:
            return contextlib.nullcontext()
        return guarded_playback(spotify, guard)

    def _lookup(self, spotify, args: dict):
        """Read-only resolution. Never mutates."""
        return None

    def _matches(self, found, args: dict) -> bool:
        return False

    def _mutate(self, spotify, target, args: dict):
        raise NotImplementedError

    def _success(self) -> str:
        return "observed"

    def _take_lookup(self, spotify, args: dict):
        key = json.dumps(args, sort_keys=True)
        cache, self._exact_cache = self._exact_cache, None
        if cache and cache[0] == key and time.monotonic() - cache[2] <= _EXACT_TTL_SECONDS:
            return cache[1]
        return self._lookup(spotify, args)

    def execute(self, args: dict, guard: Guard) -> Receipt:
        stage = "resolve"
        try:
            remaining(guard)
            spotify = self._spotify()
            with self._context(spotify, guard):
                target = self._take_lookup(spotify, args) if self.needs_lookup else None
                remaining(guard)  # Last guard point before the mutation.
                stage = "dispatch"
                label = self._mutate(spotify, target, args)
        except _NotDispatched as exc:
            return Receipt("failed", error=type(exc).__name__)
        except TurnExpired:
            return Receipt("failed" if stage == "resolve" else "unknown", error="TurnExpired")
        except Exception as exc:
            return Receipt("failed" if stage == "resolve" else "unknown", error=type(exc).__name__)
        return Receipt(self._success(), detail=str(label or ""))


class ExactMusicHandler(MusicHandler):
    """Handlers a suspicious clause may license only through an exact resolution match (C1)."""

    def exact_match(self, args: dict, guard: Guard) -> bool:
        self._exact_cache = None
        try:
            remaining(guard)
            spotify = self._spotify()
            with self._context(spotify, guard):
                found = self._lookup(spotify, args)
            matched = self._matches(found, args) is True
        except Exception:
            return False
        if matched and self.needs_lookup:
            self._exact_cache = (json.dumps(args, sort_keys=True), found, time.monotonic())
        return matched


class PlayAlbum(ExactMusicHandler):
    capability = "play_album"
    verbs = ("play the album",)
    object_kind = "text"
    effect_words = ("playing", "played")
    replies = {
        "observed": "Playing the album.",
        "requested": "I asked Spotify but can't confirm it played.",
        "unknown": "I asked Spotify but can't confirm it played.",
        "failed": "I couldn't play that album.",
    }

    def _args(self, obj):
        return {"title": obj, "artist": ""} if obj else None

    def _lookup(self, spotify, args):
        return spotify.resolve_album(_query(args))

    def _matches(self, found, args):
        return _norm(found.name) == _norm(args["title"]) and (
            not args["artist"] or _norm(found.artist) == _norm(args["artist"]))

    def _mutate(self, spotify, target, args):
        return spotify.play_album(target)


class PlayArtist(ExactMusicHandler):
    capability = "play_artist"
    verbs = ("play music by",)
    object_kind = "text"
    effect_words = ("playing", "played")
    replies = {
        "observed": "Playing music by that artist.",
        "requested": "I asked Spotify but can't confirm it played.",
        "unknown": "I asked Spotify but can't confirm it played.",
        "failed": "I couldn't play that artist.",
    }

    def _args(self, obj):
        return {"artist": obj} if obj else None

    def _lookup(self, spotify, args):
        return spotify.resolve_artist_track(args["artist"])

    def _matches(self, found, args):
        return _norm(found.artist) == _norm(args["artist"])

    def _mutate(self, spotify, target, args):
        # Plays the artist's top track; spotify.py verifies playback before it returns.
        return spotify.play_candidate(target)


class PlayPlaylist(ExactMusicHandler):
    capability = "play_playlist"
    verbs = ("play my playlist",)
    object_kind = "text"
    effect_words = ("playing", "played")
    needs_lookup = False
    replies = {
        "observed": "Playing that playlist.",
        "requested": "I asked Spotify but can't confirm it played.",
        "unknown": "I asked Spotify but can't confirm it played.",
        "failed": "I couldn't play that playlist.",
    }

    def _args(self, obj):
        return {"name": obj, "mine": True} if obj else None

    def _lookup(self, spotify, args):
        return spotify.playlist_names()  # Read-only; returns [] on a failed read.

    def _matches(self, found, args):
        return any(_norm(name) == _norm(args["name"]) for name in found)

    def _mutate(self, spotify, target, args):
        try:
            # spotify.py resolves the name before its first mutation and raises LookupError there.
            return spotify.play_playlist(args["name"], library_only=args["mine"])
        except LookupError as exc:
            raise _NotDispatched(type(exc).__name__) from exc


class QueueSong(ExactMusicHandler):
    capability = "queue_song"
    verbs = ("queue",)
    object_kind = "text"
    effect_words = ("queued",)
    replies = {
        "observed": "Queued the song.",
        "requested": "I asked Spotify to queue it but can't confirm it was added.",
        "unknown": "I asked Spotify to queue it but can't confirm it was added.",
        "failed": "I couldn't queue that song.",
    }

    def _args(self, obj):
        return {"title": obj, "artist": ""} if obj else None

    def _lookup(self, spotify, args):
        return spotify.resolve_track(_query(args))

    def _matches(self, found, args):
        return _norm(found.name) == _norm(args["title"]) and (
            not args["artist"] or _norm(found.artist) == _norm(args["artist"]))

    def _mutate(self, spotify, target, args):
        spotify.queue(target)  # No read-back exists here, so the receipt stays requested.
        return target.label

    def _success(self):
        return "requested"


class _NoObject(MusicHandler):
    object_kind = "none"

    def _args(self, obj):
        return {} if obj == "" else None


class PreviousTrack(_NoObject):
    capability = "previous_track"
    verbs = ("previous track",)
    effect_words = ("skipped",)
    # spotify.py's local-media path does not accept guarded controls for previous/restart.
    bounded = False
    replies = {
        "observed": "Went back to the previous track.",
        "requested": "I asked Spotify but can't confirm it changed tracks.",
        "unknown": "I asked Spotify but can't confirm it changed tracks.",
        "failed": "I couldn't go to the previous track.",
    }

    def _mutate(self, spotify, target, args):
        spotify.previous()


class RestartTrack(_NoObject):
    capability = "restart_track"
    verbs = ("restart the track",)
    effect_words = ("restarted",)
    bounded = False
    replies = {
        "observed": "Restarted the track.",
        "requested": "I asked Spotify but can't confirm the track restarted.",
        "unknown": "I asked Spotify but can't confirm the track restarted.",
        "failed": "I couldn't restart the track.",
    }

    def _mutate(self, spotify, target, args):
        spotify.restart_track()


class RestartAlbum(_NoObject):
    capability = "restart_album"
    verbs = ("restart the album",)
    effect_words = ("restarted",)
    replies = {
        "observed": "Restarted the album.",
        "requested": "I asked Spotify but can't confirm the album restarted.",
        "unknown": "I asked Spotify but can't confirm the album restarted.",
        "failed": "I couldn't restart the album.",
    }

    def _mutate(self, spotify, target, args):
        spotify.restart_album()


ADAPTER = Adapter("music", "1b", {
    "play_album": PlayAlbum(),
    "play_artist": PlayArtist(),
    "play_playlist": PlayPlaylist(),
    "queue_song": QueueSong(),
    "previous_track": PreviousTrack(),
    "restart_track": RestartTrack(),
    "restart_album": RestartAlbum(),
})

__all__ = ["ADAPTER", "MusicHandler", "ExactMusicHandler"]
