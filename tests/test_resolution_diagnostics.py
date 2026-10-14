"""How a song was resolved and played is recorded, and bans/playback routes are logged."""
import io
import json
import logging
import time
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import spotipy

from assistant import catalogue, main, spotify as spotify_module
from assistant.catalogue import Index
from assistant.resolution import AmbiguousMatch, Candidate
from assistant.spotify import (PlaybackVerificationError, Spotify,
                               SpotifyRateLimited, _RateLimitGuard, _catalogue_hit)
from test_catalogue import FIXTURE

HEARD = ["madeira by seedemoth", "red by seedhe maut", "naksha", "numb", "hello",
         "stronger", "kho gaye by prateek kuhad", "creep by radiohead", "play nothing at all",
         "thank you next by ariana", "bohemian rhapsody", "uss din by seedhe maut",
         "bhussi by kshmr", "boss life", "channa mereya by arijit"]


@pytest.fixture(scope="module")
def index():
    return Index({"tracks": FIXTURE, "built_at": time.time(), "complete": True})


@pytest.mark.parametrize("heard", HEARD)
def test_single_pass_catalogue_hit_agrees_with_catalogue_best(index, heard):
    """spotify.py scores once to also get the runner-up; it must decide exactly as best()."""
    expected = catalogue.best(heard, index)
    got = _catalogue_hit(heard, index)
    assert (got is None) == (expected is None)
    if expected is not None:
        assert got[0] is expected[0] and got[1] == expected[1]


def sp_with(index, search=None):
    sp = Spotify.__new__(Spotify)
    sp.sp = SimpleNamespace(search=search or Mock(side_effect=AssertionError("search called")))
    sp.set_catalogue(index)
    return sp


def item(name, artist, uri=None):
    return {"name": name, "uri": uri or "spotify:track:" + name.replace(" ", "_"),
            "artists": [{"name": artist}]}


def test_catalogue_hit_records_path_score_runner_up_and_margin(index):
    sp = sp_with(index)
    sp.resolve_track("red by seedhe maut")
    diag = sp.last_resolution
    assert diag["path"] == "catalogue"
    assert diag["score"] == pytest.approx(1.02)
    assert diag["runner_up"]                      # the song it beat
    assert diag["margin"] >= catalogue.MARGIN


def test_track_cache_hit_is_labelled(tmp_path, monkeypatch):
    monkeypatch.setattr(spotify_module, "TRACK_CACHE", tmp_path / "cache.json")
    sp = Spotify.__new__(Spotify)
    sp._tracks = {"red by seedhe maut": {"uri": "spotify:track:r", "name": "Red", "artist": "Seedhe Maut"}}
    assert sp.cached_track("Red by Seedhe Maut") is not None
    assert sp.last_resolution["path"] == "track_cache"


def test_fielded_search_records_title_similarity_and_runner_up():
    search = Mock(return_value={"tracks": {"items": [
        item("Demons", "Imagine Dragons"), item("Demons Run", "Imagine Dragons")]}})
    sp = Spotify.__new__(Spotify)
    sp.sp = SimpleNamespace(search=search)
    sp.resolve_track("demons by imagine dragons")
    diag = sp.last_resolution
    assert diag["path"] == "search_fielded"
    assert diag["score"] == 1.0 and diag["runner_up"] == "Demons Run by Imagine Dragons"
    assert 0 < diag["margin"] < 1


def test_free_text_search_and_empty_fielded_fallback_are_distinguished():
    sp = Spotify.__new__(Spotify)
    sp.sp = SimpleNamespace(search=Mock(return_value={"tracks": {"items": [item("Numb", "Linkin Park")]}}))
    sp.resolve_track("numb")
    assert sp.last_resolution["path"] == "search_free"

    answers = [{"tracks": {"items": []}}, {"tracks": {"items": [item("Numb", "Linkin Park")]}}]
    sp.sp = SimpleNamespace(search=Mock(side_effect=answers))
    sp.resolve_track("numb by linkin park")
    assert sp.last_resolution["path"] == "search_free"


def test_resolution_budget_reuses_nested_pages_and_stops_after_limit():
    search = Mock(side_effect=[{"tracks": {"items": []}},
                              {"tracks": {"items": []}},
                              {"tracks": {"items": [item("Third", "Artist")]}}])
    sp = sp_with(Index({"tracks": []}), search)
    with sp.resolution_budget(2) as outer:
        sp._search_music("first", "track")
        with sp.resolution_budget(1) as nested:
            assert nested is outer
            sp._search_music("first", "track")  # exact page reuse
            sp._search_music("second", "track")
        sp._search_music("third", "track")  # exhausted; safely empty
        assert outer["attempts"] == 2
    assert search.call_count == 2
    assert sp._search_music("third", "track")[0]["name"] == "Third"
    assert search.call_count == 3


def test_resolution_budget_covers_track_hints_artist_pages_and_album_fallback():
    search = Mock(return_value={"tracks": {"items": []}, "albums": {"items": []}})
    sp = sp_with(Index({"tracks": []}), search)
    query = "missing by unknown act"
    with sp.resolution_budget(3) as budget:
        with pytest.raises(LookupError):
            sp.resolve_track(query)
        assert sp.artist_hints_for_exact_title(query) == []
        with pytest.raises(LookupError):
            sp.resolve_album(query)
        assert budget["attempts"] == 3
    assert search.call_count == 3


def test_resolution_budget_does_not_limit_background_searches():
    from assistant.spotify import request_context
    search = Mock(return_value={"tracks": {"items": []}})
    sp = sp_with(Index({"tracks": []}), search)
    with sp.resolution_budget(0) as budget:
        with request_context("catalogue", background=True):
            assert sp._search_music("background", "track") == []
        assert budget["attempts"] == 0
    search.assert_called_once_with(q="background", type="track", limit=10)


def test_failed_lookup_records_why():
    sp = Spotify.__new__(Spotify)
    sp.sp = SimpleNamespace(search=Mock(return_value={"tracks": {"items": []}}))
    with pytest.raises(LookupError):
        sp.resolve_track("zzzz qqqq")
    assert sp.last_resolution["path"] == "search_free:not_found"


def test_ambiguous_lookup_records_path_and_runner_up():
    search = Mock(return_value={"tracks": {"items": [
        item("Hello", "Adele"), item("Hello", "Lionel Richie")]}})
    sp = Spotify.__new__(Spotify)
    sp.sp = SimpleNamespace(search=search)
    with pytest.raises(AmbiguousMatch):
        sp.resolve_track("hello")
    assert sp.last_resolution["path"].endswith(":ambiguous")
    assert sp.last_resolution["runner_up"] in ("Hello by Adele", "Hello by Lionel Richie")


# ------------------------------------------------------------------ playback

def make_player(**overrides):
    obj = Spotify.__new__(Spotify)
    obj._cfg = SimpleNamespace(get=lambda key, default=None: default)
    obj.command_target = lambda: nullcontext()
    obj._call = Mock(side_effect=lambda fn, *args, **kwargs: fn(*args, **kwargs))
    obj.sp = SimpleNamespace(start_playback=Mock(), current_playback=Mock())
    obj._device_name = ""
    obj._verify = Mock(return_value={"is_playing": True})
    obj._now_playing = Mock(return_value=True)
    obj._play_uri_locally = Mock(return_value=True)
    for name, value in overrides.items():
        setattr(obj, name, value)
    return obj


CANDIDATE = Candidate("spotify:track:s", "S", "A")


def test_web_api_playback_records_method_and_radio_count():
    obj = make_player()
    obj.play_candidate(CANDIDATE, then=["spotify:track:r1", "spotify:track:r2"])
    assert obj.last_playback["method"] == "web_api" and obj.last_playback["radio"] == 2


def test_desktop_uri_fallback_records_why_the_web_api_was_skipped():
    obj = make_player(_play_uri_locally=Mock(return_value=True))
    obj._call.side_effect = SpotifyRateLimited("banned")
    obj.play_candidate(CANDIDATE, then=["spotify:track:r1"])
    assert obj.last_playback["method"] == "desktop_uri"
    assert obj.last_playback["web_api_skipped"] == "rate_limited"
    assert obj.last_playback["radio"] == 0 and obj.last_playback["radio_skipped"] == 1


def test_accepted_web_api_playback_uses_read_only_verification_after_native_miss():
    obj = make_player()
    obj._now_playing = Mock(return_value=False)
    obj.play_candidate(CANDIDATE)
    assert obj.last_playback["method"] == "web_api_verified"
    assert obj.last_playback["native_confirmation"] == "missed"
    obj._call.assert_called_once()
    obj.sp.start_playback.assert_called_once_with(uris=[CANDIDATE.uri])
    obj._verify.assert_called_once()
    obj._play_uri_locally.assert_not_called()


def test_failed_read_only_verification_propagates_without_second_mutation():
    obj = make_player()
    obj._now_playing = Mock(return_value=False)
    obj._verify.side_effect = PlaybackVerificationError("read-only verification failed")
    with pytest.raises(PlaybackVerificationError, match="read-only verification failed"):
        obj.play_candidate(CANDIDATE)
    obj._call.assert_called_once()
    obj.sp.start_playback.assert_called_once_with(uris=[CANDIDATE.uri])
    obj._play_uri_locally.assert_not_called()


def test_local_transport_is_recorded(monkeypatch):
    from assistant import local_media
    sp = Spotify.__new__(Spotify)
    sp._cfg = SimpleNamespace(get=lambda key, default=None: default)
    sp.sp = Mock()
    monkeypatch.setattr(local_media, "status", lambda: local_media.PLAYING)
    monkeypatch.setattr(local_media, "pause", lambda: True)
    sp.pause()
    assert sp.last_playback == {"method": "local_media"}


# ---------------------------------------------------------- ban logging (once)

def capture_spotify_log():
    logger = logging.getLogger("memoir.spotify")
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    logger.addHandler(handler)
    previous = logger.level
    logger.setLevel(logging.DEBUG)
    return stream, lambda: (logger.removeHandler(handler), logger.setLevel(previous))


def test_ban_start_is_logged_once_with_scope_and_local_lift_time():
    stream, restore = capture_spotify_log()
    try:
        client = Mock()
        client.search.side_effect = spotipy.SpotifyException(
            429, -1, "limited", headers={"Retry-After": "7200"})
        guard = _RateLimitGuard(client)
        for _ in range(3):
            with pytest.raises(SpotifyRateLimited):
                guard.search(q="x")
    finally:
        restore()
    lines = [l for l in stream.getvalue().splitlines() if "ban in effect" in l]
    assert len(lines) == 1
    lifts = time.strftime("%Y-%m-%d %H:%M", time.localtime(time.time() + 7200))
    assert lines[0].startswith("ERROR") and "web API ban" in lines[0]
    assert "Retry-After='7200'" in lines[0] and lifts[:13] in lines[0]


def test_ban_adopted_from_another_process_is_logged_and_expiry_is_logged_once(tmp_path):
    path = tmp_path / "backoff.json"
    path.write_text(json.dumps({"player": time.time() + 0.2, "web": 0}), encoding="utf-8")
    stream, restore = capture_spotify_log()
    try:
        guard = _RateLimitGuard(Mock(), path)
        assert set(guard.ban_status()) == {"player"}
        time.sleep(0.3)
        assert guard.ban_status() == {}
        guard.ban_status()
        guard.current_playback()          # an expired ban must not be re-announced
    finally:
        restore()
    text = stream.getvalue()
    assert text.count("player API ban in effect (from the backoff file)") == 1
    assert text.count("player API ban lifted") == 1


# --------------------------------------------------------- main copies it all

def make_app():
    app = main.Assistant.__new__(main.Assistant)
    app.cfg = SimpleNamespace(get=lambda key, default=None: default)
    app.spotify = Mock()
    app.spotify.ensure_client_running.return_value = False
    app.feedback = Mock()
    return app


def test_execute_copies_resolution_and_playback_into_the_event():
    app = make_app()
    cand = Candidate("spotify:track:red", "RED", "Seedhe Maut")
    app.spotify.resolve_track.return_value = cand
    app.spotify.resolve_track.side_effect = lambda q: setattr(
        app.spotify, "last_resolution",
        {"path": "catalogue", "score": 1.02, "runner_up": "RED CUP by Calm", "margin": 0.17}) or cand
    app.spotify.play_candidate.side_effect = lambda c, then=None: setattr(
        app.spotify, "last_playback", {"method": "web_api", "radio": 3})
    app._radio_after = Mock(return_value=["spotify:track:a"] * 3)
    event = main.Event()
    app._execute(main.parse("play red by seedhe maut"), event)
    assert event.resolution_path == "catalogue" and event.resolution_score == 1.02
    assert event.resolution_runner_up == "RED CUP by Calm" and event.resolution_margin == 0.17
    assert event.resolved_label == "RED by Seedhe Maut"
    assert event.playback_method == "web_api" and event.radio_count == 3
    for stage in ("resolved", "playback_confirmed", "feedback_done"):
        assert stage in event.timeline
    assert event.timeline["resolved"] <= event.timeline["playback_confirmed"] <= event.timeline["feedback_done"]


def test_stale_resolution_from_an_earlier_command_is_not_copied():
    app = make_app()
    app.spotify.last_resolution = {"path": "catalogue", "score": 1.0, "runner_up": "", "margin": None}
    app.spotify.resolve_track.return_value = Candidate("spotify:track:x", "X", "Y")
    event = main.Event()
    app._execute(main.parse("play x by y"), event)
    assert event.resolution_path == ""            # the mock never resolved anything this time


def test_failed_plain_lookup_keeps_its_reason_when_recovery_succeeds():
    from assistant.listener import SttListener
    from assistant.config import Config
    app = make_app()
    app.stt = Mock(model_name="faster-whisper/test")
    app.stt.transcribe.side_effect = [("Play red by Seedhe Maut", 1), ("Play RED by Seedhe Maut", 1)]
    app.listener = SttListener(Config({}), None, None)
    app.spotify.artist_hints_for_exact_title.return_value = ["Seedhe Maut"]
    cand = Candidate("spotify:track:red", "RED", "Seedhe Maut")

    def resolve(query):
        if "see them all" in query:
            app.spotify.last_resolution = {"path": "search_fielded:not_found", "score": None,
                                           "runner_up": "", "margin": None}
            raise LookupError("not found")
        app.spotify.last_resolution = {"path": "catalogue", "score": 1.02, "runner_up": "", "margin": None}
        return cand

    app.spotify.resolve_track.side_effect = resolve
    import numpy as np
    event = main.Event()
    app._execute(main.parse("play red by see them all"), event,
                 command_audio=np.ones(16000, dtype=np.int16))
    assert event.outcome == "ok"
    assert event.resolution_path == "targeted_asr>catalogue"
    assert event.heard_query == "red by see them all"
    assert event.intent_query == "red by seedhe maut"


def test_not_found_records_the_failed_path():
    app = make_app()

    def resolve(query):
        app.spotify.last_resolution = {"path": "search_free:not_found", "score": None,
                                       "runner_up": "", "margin": None}
        raise LookupError("nothing")

    app.spotify.resolve_track.side_effect = resolve
    app.stt = Mock(model_name="x")
    event = main.Event()
    app._execute(main.parse("play zzzz qqqq"), event)
    assert event.outcome == "not_found" and event.resolution_path == "search_free:not_found"
    assert "resolved" not in event.timeline


def test_queue_records_count_and_route():
    app = make_app()
    app._nothing_playing = Mock(return_value=False)
    cand = Candidate("spotify:track:q", "Q", "A")
    app.spotify.resolve_track.return_value = cand
    event = main.Event()
    app._execute(main.parse("add q to the queue"), event)
    assert event.queued == 1 and event.playback_method == "queue_api"


def test_transport_commands_record_their_route():
    app = make_app()
    event = main.Event()
    app._execute(main.Intent(main.Action.PAUSE), event)
    assert event.playback_method == "web_api"          # no local route reported
    app.spotify.pause.side_effect = lambda: setattr(app.spotify, "last_playback", {"method": "local_media"})
    event = main.Event()
    app._execute(main.Intent(main.Action.PAUSE), event)
    assert event.playback_method == "local_media"
