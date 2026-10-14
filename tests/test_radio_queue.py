"""Local radio after a song, the queue action, and the just/only opt-out."""
import random
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from assistant import catalogue, main
from assistant.brain import Brain, decide
from assistant.catalogue import Index, radio
from assistant.intent import Action, parse
from assistant.resolution import Candidate
from assistant.spotify import Spotify, SpotifyRateLimited


def entry(n, artists, album="", sources=("artist",), playlists=None):
    e = {"uri": "spotify:track:%s" % n, "name": "Song %s" % n, "artists": list(artists),
         "album": album or "Album %s" % n, "sources": list(sources)}
    if playlists:
        e["playlists"] = list(playlists)
    return e


def index_of(entries):
    return Index({"tracks": entries})


def world():
    tracks = [entry("seed", ["A"], "S", ("top",), ["p1"])]
    tracks += [entry("a%d" % i, ["A"], "Alb%d" % (i // 5)) for i in range(12)]
    tracks += [entry("ab", ["A", "B"], "AB", ("saved",))]
    tracks += [entry("b%d" % i, ["B"], "B%d" % (i // 5), ("saved",) if i < 4 else ("artist",))
               for i in range(12)]
    tracks += [entry("p%d" % i, ["Z"], "Z%d" % i, ("playlist",), ["p1"]) for i in range(6)]
    return index_of(tracks)


def test_radio_excludes_seed_and_reissues_and_is_deterministic():
    tracks = [entry("seed", ["A"]), entry("x", ["A"]), dict(entry("y", ["A"]), name="Song x")]
    out = radio(index_of(tracks), "spotify:track:seed", 10, rng=random.Random(1))
    assert out == ["spotify:track:x"] or out == ["spotify:track:y"]
    a = radio(world(), "spotify:track:seed", 30, rng=random.Random(4))
    assert a == radio(world(), "spotify:track:seed", 30, rng=random.Random(4))
    assert "spotify:track:seed" not in a and len(a) == len(set(a)) == 30


def test_radio_mixes_artist_collaborators_and_playlist():
    out = radio(world(), "spotify:track:seed", 20, rng=random.Random(2))
    kinds = {u.split(":")[-1][0] for u in out}
    assert {"a", "b", "p"} <= kinds


def test_radio_avoids_runs_and_album_pileups():
    out = radio(world(), "spotify:track:seed", 20, rng=random.Random(3))
    by_uri = {e["uri"]: e for e in world().entries}
    primaries = [by_uri[u]["artists"][0] for u in out]
    assert all(not (primaries[i] == primaries[i + 1] == primaries[i + 2])
               for i in range(len(primaries) - 2))
    albums = [by_uri[u]["album"] for u in out]
    assert max(albums.count(a) for a in set(albums)) <= 3


def test_radio_unknown_seed_falls_back_to_artist_then_empty():
    ghost = {"uri": "spotify:track:ghost", "artists": ["B"]}
    assert radio(world(), ghost, 5, rng=random.Random(1))
    assert radio(world(), {"uri": "spotify:track:ghost", "artists": ["Nobody"]}, 5) == []
    assert radio(world(), "Nobody", 5) == []
    assert radio(Mock(), "spotify:track:seed") == []


def test_recent_tracks_are_deprioritised_not_excluded():
    idx = index_of([entry("seed", ["A"])] + [entry("a%d" % i, ["A"]) for i in range(6)])
    recent = ["spotify:track:a%d" % i for i in range(3)]
    out = radio(idx, "spotify:track:seed", 3, rng=random.Random(1), recent=recent)
    assert not set(out) & set(recent)
    assert set(radio(idx, "spotify:track:seed", 6, rng=random.Random(1), recent=recent)) == {
        "spotify:track:a%d" % i for i in range(6)}


def test_radio_history_is_rolling_and_atomic(tmp_path):
    path = tmp_path / "h.json"
    catalogue.remember_radio(["u%d" % i for i in range(250)], path)
    catalogue.remember_radio(["u%d" % i for i in range(200, 400)], path)
    hist = catalogue.radio_history(path)
    assert len(hist) == 300 and hist[-1] == "u399" and len(set(hist)) == 300
    assert not (tmp_path / "h.json.tmp").exists()


def test_radio_is_fast_on_3000_tracks():
    import time
    tracks = [entry(i, ["Art%d" % (i % 150), "Art%d" % ((i * 7) % 150)], "Alb%d" % (i // 8))
              for i in range(3000)]
    started = time.perf_counter()
    assert radio(index_of(tracks), "spotify:track:5", 49, rng=random.Random(1))
    assert time.perf_counter() - started < 0.5


def test_dedupe_merges_playlist_ids():
    merged = catalogue.dedupe([entry("x", ["A"], sources=("playlist",), playlists=["p1"]),
                               entry("x", ["A"], sources=("playlist",), playlists=["p2"]),
                               entry("y", ["A"])])
    assert [e.get("playlists") for e in merged] == [["p1", "p2"], None]


# ------------------------------------------------------------ opt-out rules

@pytest.mark.parametrize("text,query", [
    ("just play naksha", "naksha"), ("play only naksha", "naksha"),
    ("play naksha only", "naksha"), ("just play naksha at 20% volume", "naksha")])
def test_just_and_only_turn_radio_off(text, query):
    intent = parse(text)
    assert (intent.action, intent.query, intent.radio) == (Action.PLAY_TRACK, query, False)


@pytest.mark.parametrize("text", ["play naksha", "play just the two of us", "play only"])
def test_radio_stays_on_otherwise(text):
    assert parse(text).radio is True


# ------------------------------------------------------------- queue rules

@pytest.mark.parametrize("text,query", [
    ("add naksha by seedhe maut to the queue", "naksha by seedhe maut"),
    ("add naksha to my queue", "naksha"), ("put naksha in the queue", "naksha"),
    ("put naksha on the queue please", "naksha"), ("queue naksha", "naksha"),
    ("queue up naksha", "naksha"), ("play naksha next", "naksha"),
    ("play naksha after this", "naksha"), ("after this play naksha", "naksha"),
    ("after this, play naksha by seedhe maut", "naksha by seedhe maut"),
    ("next play naksha", "naksha"), ("queue naksha and madira", "naksha and madira")])
def test_queue_phrasings(text, query):
    intent = parse(text)
    assert (intent.action, intent.query) == (Action.QUEUE, query)


@pytest.mark.parametrize("text", ["play next", "next", "next song", "play the next song",
                                  "skip", "play next track"])
def test_next_stays_next(text):
    assert parse(text).action is Action.NEXT


@pytest.mark.parametrize("text", [
    "queue", "queue up", "queue it", "queue this song", "add this song to the queue",
    "add to queue", "play it next", "play my gym playlist next", "clear the queue",
    "what's in the queue", "add naksha to the queue and then pause",
    "queue naksha and set volume to 20"])
def test_queue_negatives_are_never_queue(text):
    assert parse(text).action is not Action.QUEUE


def test_title_with_next_still_plays_normally():
    assert parse("play next to me by new west").query == "next to me by new west"


# ------------------------------------------------------------------- brain

CONTEXT = {"artists": ["Seedhe Maut"], "playlists": [], "now_playing": ""}


def brain():
    return Brain("test-model", context=lambda: CONTEXT)


def test_brain_queue_song_is_grounded_like_play_song():
    call = {"tool": "queue_song", "arguments": {"title": "Madira", "artist": "Seedhe Maut"}}
    intent = brain().to_intent(call, "add madira by see the mode to the queue")
    assert (intent.action, intent.query, intent.matched_rule) == (
        Action.QUEUE, "madira by seedhe maut", "brain_queue_song")
    with pytest.raises(ValueError):
        brain().to_intent({"tool": "queue_song", "arguments": {"title": "Numb", "artist": ""}},
                          "queue madira")


def test_brain_radio_false_needs_just_or_only_heard():
    off = {"tool": "play_song", "arguments": {"title": "Numb", "artist": "", "radio": False}}
    assert brain().to_intent(off, "just play numb").radio is False
    assert brain().to_intent(off, "play numb").radio is True
    only_title = {"tool": "play_song", "arguments": {"title": "Only Girl", "artist": "",
                                                     "radio": False}}
    assert brain().to_intent(only_title, "play only girl").radio is True


def test_explicit_queue_is_fast_and_skips_the_brain():
    model = Mock()
    assert decide("queue naksha", model).action is Action.QUEUE
    model.decide.assert_not_called()


# ---------------------------------------------------------------- spotify

def make_spotify(monkeypatch):
    obj = Spotify.__new__(Spotify)
    obj._cfg = SimpleNamespace(get=lambda key, default=None: default)
    obj.command_target = lambda: __import__("contextlib").nullcontext()
    obj._call = Mock()
    obj.sp = SimpleNamespace(start_playback=Mock(), add_to_queue=Mock())
    obj._now_playing = Mock(return_value=True)
    return obj


def test_play_candidate_sends_song_then_radio_in_one_call(monkeypatch):
    obj = make_spotify(monkeypatch)
    then = ["spotify:track:r%d" % i for i in range(60)]
    obj.play_candidate(Candidate("spotify:track:s", "S"), then=then)
    obj._call.assert_called_once()
    uris = obj._call.call_args.kwargs["uris"]
    assert uris[0] == "spotify:track:s" and len(uris) == 50
    assert obj.radio_played is True
    obj._now_playing.assert_called_once()


def test_desktop_fallback_skips_radio(monkeypatch):
    obj = make_spotify(monkeypatch)
    obj._call.side_effect = SpotifyRateLimited("limited")
    obj._play_uri_locally = Mock(return_value=True)
    obj.play_candidate(Candidate("spotify:track:s", "S"), then=["spotify:track:r"])
    obj._play_uri_locally.assert_called_once()
    assert obj.radio_played is False


def test_queue_goes_through_the_guarded_call():
    obj = make_spotify(None)
    obj.queue(Candidate("spotify:track:q", "Q"))
    obj._call.assert_called_once_with(obj.sp.add_to_queue, "spotify:track:q")


# -------------------------------------------------------------------- main

def app_for(monkeypatch, playing):
    monkeypatch.setattr("assistant.local_media.is_playing", lambda: playing)
    app = main.Assistant.__new__(main.Assistant)
    app.spotify = Mock()
    app.spotify.ensure_client_running.return_value = False
    app.spotify.resolve_local.return_value = None
    app.feedback = Mock()
    app._paused_mic = None
    app._music_playing = False
    return app


def exact(name="Naksha", artist="Seedhe Maut"):
    return Candidate("spotify:track:n", name, artist, metadata={"match_score": 1.02})


def test_song_queue_lookup_failure_never_probes_album(monkeypatch):
    from assistant.spotify import TrackLookupError
    app = app_for(monkeypatch, True)
    app.spotify.cached_track.return_value = None
    app.spotify.resolve_track.side_effect = TrackLookupError("missing song")
    app.spotify.resolve_best_available.side_effect = LookupError("no eligible saved track")
    app._recover_track_from_audio = Mock(return_value=None)
    event = main.Event()
    app._execute(parse("queue keychain lalu by arpit bala"), event)
    assert event.outcome == "not_found"
    app.spotify.resolve_album.assert_not_called()
    app.spotify.play_album.assert_not_called()
    app.spotify.queue.assert_not_called()


@pytest.mark.parametrize("banned", [False, True])
def test_best_available_queue_preserves_song_kind_and_never_caches_or_speaks_approximation(monkeypatch, banned):
    from assistant.spotify import TrackLookupError
    app = app_for(monkeypatch, True)
    app.spotify.cached_track.return_value = None
    app.spotify.web_rate_limited = banned
    app.spotify.resolve_local.return_value = None
    app.spotify.resolve_track.side_effect = TrackLookupError("missing song")
    app._recover_track_from_audio = Mock(return_value=None)
    candidate = Candidate("spotify:track:taaron", "Taaron Se", "Arpit Bala", {"resolution_approximate": True})
    app.spotify.resolve_best_available.return_value = candidate
    event = main.Event()
    app._execute(parse("queue tarot by arpit bala"), event)
    app.spotify.queue.assert_called_once_with(candidate)
    app.spotify.play_candidate.assert_not_called()
    app.spotify.resolve_album.assert_not_called()
    app.spotify.remember_track.assert_not_called()
    app.feedback.say.assert_not_called()
    app.feedback.ok.assert_called_once()
    assert event.outcome == "ok" and event.resolved_label == candidate.label
    if banned:
        app.spotify.resolve_track.assert_not_called()
        app.spotify.resolve_best_available.assert_called_once_with("tarot by arpit bala", "track", offline_only=True)
    else:
        app.spotify.resolve_best_available.assert_called_once_with("tarot by arpit bala", "track")


def test_confirmed_queue_still_queues_immutable_song_without_search(monkeypatch):
    app = app_for(monkeypatch, True)
    candidate = exact("Taaron Se", "Arpit Bala")
    event = main.Event()
    app._execute(parse("queue tarot by arpit bala"), event, confirmed_candidate=candidate)
    app.spotify.queue.assert_called_once_with(candidate)
    app.spotify.play_candidate.assert_not_called()
    app.spotify.resolve_track.assert_not_called()
    app.spotify.remember_track.assert_not_called()
    assert event.spotify_action == "QUEUE" and event.outcome == "ok"


def test_queue_exact_match_beeps_and_queues(monkeypatch):
    app = app_for(monkeypatch, True)
    app.spotify.cached_track.return_value = None
    app.spotify.resolve_track.return_value = exact()
    event = main.Event()
    app._execute(parse("queue naksha by seedhe maut"), event)
    app.spotify.queue.assert_called_once()
    app.spotify.play_candidate.assert_not_called()
    app.feedback.ok.assert_called_once()
    app.feedback.say.assert_not_called()
    assert event.outcome == "ok" and event.spotify_action == "QUEUE"


def test_queue_repaired_match_is_spoken_not_beeped(monkeypatch):
    app = app_for(monkeypatch, True)
    app.spotify.cached_track.return_value = None
    app.spotify.resolve_track.return_value = exact()
    app._execute(parse("queue naksha by seedemoth"), main.Event())
    app.feedback.ok.assert_not_called()
    assert app.feedback.say.call_args.args[0] == "Queued Naksha by Seedhe Maut."
    assert app.feedback.say.call_args.kwargs["blocking"] is False


def test_queue_when_nothing_plays_plays_now_with_radio(monkeypatch):
    app = app_for(monkeypatch, False)
    app.spotify.cached_track.return_value = None
    app.spotify.resolve_track.return_value = exact()
    monkeypatch.setattr(main.catalogue, "radio", lambda *a, **k: ["spotify:track:r1"])
    app._execute(parse("queue naksha by seedhe maut"), main.Event())
    app.spotify.queue.assert_not_called()
    app.spotify.play_candidate.assert_called_once_with(exact(), then=["spotify:track:r1"])


def test_queue_rate_limited_says_sorry(monkeypatch):
    app = app_for(monkeypatch, True)
    app.spotify.cached_track.return_value = None
    app.spotify.resolve_track.return_value = exact()
    app.spotify.queue.side_effect = SpotifyRateLimited("player")
    event = main.Event()
    app._execute(parse("queue naksha by seedhe maut"), event)
    assert app.feedback.say.call_args.args[0] == main.SORRY
    assert event.outcome == "error"


def test_queue_two_songs_only_when_both_resolve_exactly(monkeypatch):
    app = app_for(monkeypatch, True)
    one = Candidate("spotify:track:1", "Naksha", "X", metadata={"match_score": 1.0})
    two = Candidate("spotify:track:2", "Madira", "X", metadata={"match_score": 1.0})
    lookup = {"naksha": one, "madira": two}
    app.spotify.resolve_local.side_effect = lambda q: lookup.get(q)
    app._execute(parse("queue naksha and madira"), main.Event())
    assert [c.args[0] for c in app.spotify.queue.call_args_list] == [one, two]

    app = app_for(monkeypatch, True)
    app.spotify.resolve_local.side_effect = lambda q: lookup.get(q) if q == "naksha" else None
    app.spotify.cached_track.return_value = None
    app.spotify.resolve_track.return_value = one
    app._execute(parse("queue naksha and madira"), main.Event())
    app.spotify.resolve_track.assert_called_once_with("naksha and madira")
    app.spotify.queue.assert_called_once()


def test_play_next_of_a_real_title_plays_it_now(monkeypatch):
    app = app_for(monkeypatch, True)
    title = Candidate("spotify:track:t", "Thank U, Next", "Ariana Grande",
                      metadata={"match_score": 1.02})
    app.spotify.resolve_local.side_effect = lambda q: title if q == "thank u next" else None
    app._execute(parse("play thank u next"), main.Event())
    app.spotify.queue.assert_not_called()
    app.spotify.play_candidate.assert_called_once()


def test_radio_failure_never_blocks_playback(monkeypatch):
    app = app_for(monkeypatch, True)
    app.spotify.cached_track.return_value = None
    app.spotify.resolve_track.return_value = exact()
    monkeypatch.setattr(main.catalogue, "radio", Mock(side_effect=RuntimeError("boom")))
    app._execute(parse("play naksha by seedhe maut"), main.Event())
    app.spotify.play_candidate.assert_called_once_with(exact())


def test_just_play_sends_no_radio_and_records_history(monkeypatch, tmp_path):
    app = app_for(monkeypatch, True)
    app.spotify.cached_track.return_value = None
    app.spotify.resolve_track.return_value = exact()
    radio_fn = Mock(return_value=["spotify:track:r1"])
    monkeypatch.setattr(main.catalogue, "radio", radio_fn)
    app._execute(parse("just play naksha by seedhe maut"), main.Event())
    radio_fn.assert_not_called()
    app.spotify.play_candidate.assert_called_once_with(exact())
    app.spotify.radio_played = True
    app._execute(parse("play naksha by seedhe maut"), main.Event())
    assert catalogue.radio_history() == ["spotify:track:r1"]
