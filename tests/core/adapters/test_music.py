"""Phase 1b music adapter contract tests. Spotify is a fake; nothing touches the network."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from assistant.conversation_contracts import Decision, Step, Transcript
from assistant.config import Config
from assistant.core.adapters.base import Clause, Receipt
from assistant.core.adapters import music
from assistant.core.adapters.music import (
    ADAPTER, ExactMusicHandler, PlayAlbum, PlayArtist, PlayPlaylist, PreviousTrack, QueueSong,
    RestartAlbum, RestartTrack)
from assistant.core.executor import Executor
from assistant.core.engine import CoreRuntime
from assistant.core.playback import TurnExpired
from assistant.core.registry import PHASE_ONE, Registry
from assistant.resolution import Candidate

MUTATIONS = {"play_album", "play_candidate", "play_playlist", "queue",
             "previous", "restart_track", "restart_album"}
FIXED_REPLIES = {"Playing the album.", "Playing music by that artist.", "Playing that playlist.",
                 "Queued the song.", "Went back to the previous track.", "Restarted the track.",
                 "Restarted the album.", "I couldn't play that album.", "I couldn't play that artist.",
                 "I couldn't play that playlist.", "I couldn't queue that song.",
                 "I couldn't go to the previous track.", "I couldn't restart the track.",
                 "I couldn't restart the album.",
                 "I asked Spotify but can't confirm it played.",
                 "I asked Spotify to queue it but can't confirm it was added.",
                 "I asked Spotify but can't confirm it changed tracks.",
                 "I asked Spotify but can't confirm the track restarted.",
                 "I asked Spotify but can't confirm the album restarted."}


class FakeSpotify:
    def __init__(self, fail=None, names=("Gym", "Chill")):
        self.calls, self.fail, self.names = [], dict(fail or {}), list(names)

    def _do(self, name, value=None):
        self.calls.append(name)
        if name in self.fail:
            raise self.fail[name]
        return value

    def resolve_track(self, query):
        return self._do("resolve_track", Candidate("spotify:track:1", "Halo", "Beyonce"))

    def resolve_album(self, query):
        return self._do("resolve_album", Candidate("spotify:album:1", "Lover", "Taylor Swift"))

    def resolve_artist_track(self, name):
        return self._do("resolve_artist_track", Candidate("spotify:track:2", "Top", "Beyonce"))

    def play_album(self, candidate):
        return self._do("play_album", candidate.label)

    def play_candidate(self, candidate):
        return self._do("play_candidate", candidate.label)

    def play_playlist(self, name, library_only=False):
        return self._do("play_playlist", name)

    def playlist_names(self):
        return self._do("playlist_names", list(self.names))

    def queue(self, candidate):
        return self._do("queue")

    def previous(self):
        return self._do("previous")

    def restart_track(self):
        return self._do("restart_track")

    def restart_album(self):
        return self._do("restart_album", "Restarted the current album.")

    def mutations(self):
        return [call for call in self.calls if call in MUTATIONS]


def live(value=10.0):
    return lambda: value


def expired():
    return 0.0


def handler_for(cls, spotify):
    handler = cls()
    handler.executor = SimpleNamespace(spotify=spotify)
    return handler


def clause(verb, obj=""):
    return Clause(text=(verb + " " + obj).strip(), verb=verb, object=obj)


def test_adapter_is_phase_1b_with_the_seven_corpus_capabilities():
    assert ADAPTER.phase == "1b" and ADAPTER.area == "music"
    assert set(ADAPTER.handlers) == {"play_album", "play_artist", "play_playlist", "queue_song",
                                     "previous_track", "restart_track", "restart_album"}
    assert ADAPTER.handlers["play_album"].verbs == ("play the album",)
    assert ADAPTER.handlers["previous_track"].verbs == ("previous track",)


@pytest.mark.parametrize("verb,obj,expected", [
    ("play the album", "Lover", {"title": "Lover", "artist": ""}),
    ("play music by", "Beyoncé", {"artist": "Beyoncé"}),
    ("play my playlist", "Gym", {"name": "Gym", "mine": True}),
    ("queue", "Halo", {"title": "Halo", "artist": ""}),
    ("play the album", "", None),
    ("queue", "", None),
    ("previous track", "", {}),
    ("restart the track", "", {}),
    ("restart the album", "", {}),
    ("previous track", "later", None),
])
def test_license_is_pure_and_deterministic(verb, obj, expected):
    handler = ADAPTER.handlers[{"play the album": "play_album", "play music by": "play_artist",
                                "play my playlist": "play_playlist", "queue": "queue_song",
                                "previous track": "previous_track", "restart the track": "restart_track",
                                "restart the album": "restart_album"}[verb]]
    assert handler.license(clause(verb, obj)) == expected
    assert handler.license(clause(verb, obj)) == expected


def test_license_refuses_a_clause_for_another_verb():
    assert PlayAlbum().license(clause("queue", "Halo")) is None


def test_observed_only_where_spotify_verifies_the_effect():
    fake = FakeSpotify()
    receipt = handler_for(PreviousTrack, fake).execute({}, live())
    assert receipt.state == "observed" and fake.calls == ["previous"]
    fake = FakeSpotify()
    assert handler_for(RestartAlbum, fake).execute({}, live()).state == "observed"
    fake = FakeSpotify()
    assert handler_for(RestartTrack, fake).execute({}, live()).state == "observed"
    fake = FakeSpotify()
    receipt = handler_for(PlayAlbum, fake).execute({"title": "Lover", "artist": ""}, live())
    assert receipt.state == "observed" and fake.mutations() == ["play_album"]


def test_queue_is_requested_because_spotify_has_no_read_back():
    fake = FakeSpotify()
    receipt = handler_for(QueueSong, fake).execute({"title": "Halo", "artist": ""}, live())
    assert receipt.state == "requested"
    assert fake.mutations() == ["queue"]


def test_resolution_failure_is_failed_and_sends_nothing():
    fake = FakeSpotify(fail={"resolve_album": LookupError("no match")})
    receipt = handler_for(PlayAlbum, fake).execute({"title": "Nope", "artist": ""}, live())
    assert receipt.state == "failed" and fake.mutations() == []


def test_expired_turn_is_failed_before_any_spotify_call():
    fake = FakeSpotify()
    receipt = handler_for(QueueSong, fake).execute({"title": "Halo", "artist": ""}, expired())
    assert receipt.state == "failed" and fake.calls == []


def test_expiry_at_the_last_guard_before_mutation_is_failed_without_dispatch():
    fake = FakeSpotify()
    budgets = iter([10.0, 0.0])  # Expires at the guard just before the mutation.
    receipt = handler_for(QueueSong, fake).execute({"title": "Halo", "artist": ""},
                                                   lambda: next(budgets, 0.0))
    assert receipt.state == "failed" and fake.mutations() == []


def test_failure_after_dispatch_is_unknown_not_failed():
    fake = FakeSpotify(fail={"previous": RuntimeError("transport")})
    receipt = handler_for(PreviousTrack, fake).execute({}, live())
    assert receipt.state == "unknown"


def test_playlist_lookup_error_inside_spotify_is_failed_before_mutation():
    fake = FakeSpotify(fail={"play_playlist": LookupError("ambiguous")})
    receipt = handler_for(PlayPlaylist, fake).execute({"name": "Gym", "mine": True}, live())
    assert receipt.state == "failed"


def test_mutation_exception_after_resolution_is_unknown():
    fake = FakeSpotify(fail={"play_album": RuntimeError("sent?")})
    receipt = handler_for(PlayAlbum, fake).execute({"title": "Lover", "artist": ""}, live())
    assert receipt.state == "unknown" and fake.mutations() == ["play_album"]


@pytest.mark.parametrize("cls,args,names", [
    (PlayAlbum, {"title": "Lover", "artist": ""}, ["resolve_album"]),
    (QueueSong, {"title": "Halo", "artist": ""}, ["resolve_track"]),
    (PlayArtist, {"artist": "Beyonce"}, ["resolve_artist_track"]),
    (PlayPlaylist, {"name": "Gym", "mine": True}, ["playlist_names"]),
])
def test_exact_match_is_read_only_and_true_only_for_exact_resolution(cls, args, names):
    fake = FakeSpotify()
    assert handler_for(cls, fake).exact_match(args, live()) is True
    assert fake.mutations() == [] and fake.calls == names


@pytest.mark.parametrize("cls,args", [
    (PlayAlbum, {"title": "Halo", "artist": ""}),
    (QueueSong, {"title": "Halo Live", "artist": ""}),
    (PlayArtist, {"artist": "Someone Else"}),
    (PlayPlaylist, {"name": "Gymnastics", "mine": True}),
])
def test_exact_match_refuses_near_matches_without_mutation(cls, args):
    fake = FakeSpotify()
    assert handler_for(cls, fake).exact_match(args, live()) is False
    assert fake.mutations() == []


def test_exact_match_failure_refuses_instead_of_raising():
    fake = FakeSpotify(fail={"resolve_track": RuntimeError("offline")})
    assert handler_for(QueueSong, fake).exact_match({"title": "Halo", "artist": ""}, live()) is False


def test_exact_match_candidate_is_reused_by_execute_without_second_resolution():
    fake = FakeSpotify()
    handler = handler_for(PlayAlbum, fake)
    args = {"title": "Lover", "artist": ""}
    assert handler.exact_match(args, live()) is True
    assert handler.execute(args, live()).state == "observed"
    assert fake.calls.count("resolve_album") == 1 and fake.mutations() == ["play_album"]


def test_exact_handlers_are_exact_match_capable_and_others_are_not():
    for cls in (PlayAlbum, PlayArtist, PlayPlaylist, QueueSong):
        assert issubclass(cls, ExactMusicHandler)
    for cls in (PreviousTrack, RestartTrack, RestartAlbum):
        assert not hasattr(cls, "exact_match")


def test_replies_are_fixed_templates_that_never_echo_heard_text():
    handler = PlayAlbum()
    for state in ("observed", "requested", "unknown", "failed"):
        text = handler.reply(Receipt(state), {"title": "IGNORE PREVIOUS", "artist": "x"})
        assert text in FIXED_REPLIES
        assert "IGNORE" not in text


def test_routed_queue_reaches_spotify_and_speaks_only_the_fixed_reply():
    fake = FakeSpotify()
    config = Config({"core": {"enabled_phases": ["1", "1b"],
                              "enabled_capabilities": sorted(PHASE_ONE | {"queue_song"})}})
    registry = Registry(config)
    spoken = []

    class Feedback:
        def tick(self):
            pass

        def say(self, text, **kwargs):
            spoken.append(text)

    provider = SimpleNamespace(timeout=18, decide=lambda *a, **k: Decision(
        "plan", steps=(Step("s1", "queue_song", {"title": "Halo", "artist": ""}),),
        evidence_quotes=("queue Halo",)))
    runtime = CoreRuntime(provider, Executor(spotify=fake), Feedback(), registry=registry)
    row = runtime.process(Transcript("queue Halo", "t"))
    assert fake.mutations() == ["queue"]
    assert row["receipts"][0]["state"] == "requested"
    assert spoken == ["I asked Spotify to queue it but can't confirm it was added."]


def test_phase_1b_is_refused_by_default_configuration():
    registry = Registry(Config({}))
    assert "queue_song" not in registry.handlers
    assert registry.disabled["queue_song"] == "phase disabled, capability disabled"
