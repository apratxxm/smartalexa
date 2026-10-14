import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from assistant.catalogue import Index
from assistant.resolution import (AmbiguousMatch, MusicConfirmationRequired,
                                  rank_tracks)
from assistant.spotify import Spotify


@pytest.fixture(autouse=True)
def isolate_process_request_budget(monkeypatch):
    from assistant import spotify
    monkeypatch.setattr(spotify, "_SHARED_REQUEST_BUDGET", spotify._RequestBudget())


def track(name, artist, uri=None, artists=None):
    return {"name": name, "uri": uri or "spotify:track:" + name, "artists":
            [{"name": a} for a in (artists or [artist])]}


def album(name="Keychain Laalu", artists=("Arpit Bala",), uri="spotify:album:release"):
    return {"name": name, "uri": uri, "artists": [{"name": a} for a in artists]}


def test_best_available_no_artist_uses_first_eligible_spotify_result_not_closest_spelling():
    sp = Spotify.__new__(Spotify)
    rows = [track("The Intended Song", "Artist", "spotify:track:first"),
            track("Forgotten Name", "Other", "spotify:track:second")]
    sp.sp = SimpleNamespace(search=Mock(return_value={"tracks": {"items": rows}}))
    hit = sp.resolve_best_available("forgotten name")
    assert hit.uri == "spotify:track:first" and hit.approximate is True
    sp.sp.search.assert_called_once_with(q="forgotten name", type="track", limit=10)
    assert sp.last_resolution["path"] == "best_available_search_track"


def test_best_available_uses_bound_candidate_relevance_before_another_search():
    from assistant.resolution import Candidate
    sp = Spotify.__new__(Spotify)
    sp.sp = SimpleNamespace(search=Mock(side_effect=AssertionError("Bound choices need no search")))
    bound = (Candidate("spotify:track:closest", "Forgotten Name", "Other", {"spotify_relevance": 1}),
             Candidate("spotify:track:first", "The Intended Song", "Artist", {"spotify_relevance": 0}))
    assert sp.resolve_best_available("forgotten name", candidates=bound).uri == "spotify:track:first"
    sp.sp.search.assert_not_called()


def test_best_available_artist_is_hard_constraint_across_all_credits_and_title_ranks():
    sp = Spotify.__new__(Spotify)
    rows = [track("Tarot", "Wrong", "spotify:track:wrong"),
            track("Distant", "Producer", "spotify:track:distant", artists=["Producer", "Arpit Bala"]),
            track("Taaron Se", "Arpit Bala", "spotify:track:taaron")]
    sp.sp = SimpleNamespace(search=Mock(return_value={"tracks": {"items": rows}}))
    assert sp.resolve_best_available("tarot by arpit bala").uri == "spotify:track:taaron"
    with pytest.raises(LookupError):
        sp.resolve_best_available("tarot by Aerosmith")


def test_best_available_artist_near_spelling_requires_unique_canonical_credit():
    from assistant.resolution import Candidate
    sp = Spotify.__new__(Spotify)
    sp.sp = SimpleNamespace(search=Mock(return_value={"tracks": {"items": []}}))
    one = Candidate("spotify:track:one", "Song", "Mira Sol", {"artists": [{"name": "Mira Sol"}]})
    assert sp.resolve_best_available("song by MiraSoll", candidates=(one,)).uri == one.uri
    one = Candidate("spotify:track:one", "Song", "Mira Solara", {"artists": [{"name": "Mira Solara"}]})
    two = Candidate("spotify:track:two", "Song", "Mira Solaro", {"artists": [{"name": "Mira Solaro"}]})
    with pytest.raises(LookupError):
        sp.resolve_best_available("song by MiraSolare", candidates=(one, two))


@pytest.mark.parametrize("bad", [
    {"uri": "spotify:local:artist:album:song"}, {"uri": "spotify:album:album"},
    {"uri": "spotify:track:"}, {"uri": "https://example.invalid/track"},
    {"is_local": True}, {"is_playable": False}, {"restrictions": {"reason": "market"}},
    {"type": "episode"}, {"name": ""}, {"name": None},
])
def test_best_available_filters_unplayable_and_wrong_entity_results(bad):
    sp = Spotify.__new__(Spotify)
    row = {**track("Song", "Artist", "spotify:track:bad"), **bad}
    valid = track("Song Live", "Artist", "spotify:track:valid")
    sp.sp = SimpleNamespace(search=Mock(return_value={"tracks": {"items": [row, valid]}}))
    assert sp.resolve_best_available("song live").uri == "spotify:track:valid"


def test_best_available_preserves_explicit_recording_edition_and_album_category():
    sp = Spotify.__new__(Spotify)
    sp.sp = SimpleNamespace(search=Mock(return_value={"albums": {"items": [
        album("Blue Lights", ("Mira Sol",), "spotify:album:studio"),
        album("Blue Lights Live", ("Mira Sol",), "spotify:album:live")]}}))
    assert sp.resolve_best_available("blue lights live by mira sol", "album").uri == "spotify:album:live"
    assert sp.sp.search.call_args.kwargs["type"] == "album"


def test_best_available_approximation_never_enters_permanent_track_cache(tmp_path, monkeypatch):
    from assistant.resolution import Candidate
    cache_path = tmp_path / "cache.json"
    monkeypatch.setattr("assistant.spotify.TRACK_CACHE", cache_path)
    sp = Spotify.__new__(Spotify)
    sp._tracks = {}
    hit = Candidate("spotify:track:taaron", "Taaron Se", "Arpit Bala", {"resolution_approximate": True})
    sp.remember_track("tarot by arpit bala", hit)
    assert sp._tracks == {} and not cache_path.exists()


def _banned_saved_spotify(entries, artists=()):
    from assistant.catalogue import Index
    from assistant.spotify import _RateLimitGuard
    sp = Spotify.__new__(Spotify)
    client = Mock()
    client.search.side_effect = AssertionError("Never retry a banned endpoint")
    sp.sp = _RateLimitGuard(client)
    sp.sp.blocked["web"] = time.time() + 3600
    sp._tracks = {}
    sp._known_artists = list(artists)
    sp.set_catalogue(Index({"tracks": entries}))
    return sp, client


def saved_track(name, artists, uri="spotify:track:saved"):
    return {"uri": uri, "name": name, "artists": list(artists)}


def test_known_web_ban_uses_saved_artist_track_without_calling_search():
    sp, client = _banned_saved_spotify([saved_track("Taaron Se", ["Arpit Bala"])])
    hit = sp.resolve_best_available("tarot by arpit bala", offline_only=True)
    assert hit.uri == "spotify:track:saved" and hit.approximate
    assert sp.last_resolution["path"] == "best_available_saved_track"
    client.search.assert_not_called()


def test_saved_exact_title_partial_known_artist_repair_is_generic_and_approximate():
    sp, client = _banned_saved_spotify([saved_track("RED", ["Seedhe Maut", "Hisab"])], ["Seedhe Maut"])
    hit = sp.resolve_best_available("red by sedebh", offline_only=True)
    assert hit.label == "RED by Seedhe Maut" and hit.approximate
    assert sp.last_resolution["path"].endswith("+partial_artist")
    client.search.assert_not_called()


def test_partial_saved_artist_selects_only_the_exact_title_that_grounded_repair():
    sp, client = _banned_saved_spotify([
        saved_track("RED - 2024 Remastered", ["Seedhe Maut"], "spotify:track:remaster"),
        saved_track("RED", ["Seedhe Maut"], "spotify:track:exact")], ["Seedhe Maut"])
    hit = sp.resolve_best_available("red by sedebh", offline_only=True)
    assert hit.uri == "spotify:track:exact" and hit.approximate
    assert sp.last_resolution["path"].endswith("+partial_artist")
    client.search.assert_not_called()


@pytest.mark.parametrize("query,rows,known", [
    ("red by sedebh", [saved_track("RED", ["Seedhe Maut"]),
        saved_track("RED", ["Seedhe Other"], "spotify:track:rival")], ["Seedhe Maut"]),
    ("red by aerosmith", [saved_track("RED", ["Seedhe Maut"])], ["Seedhe Maut"]),
    ("red by sed", [saved_track("RED", ["Seedhe Maut"])], ["Seedhe Maut"]),
    ("red by sedebh other", [saved_track("RED", ["Seedhe Maut"])], ["Seedhe Maut"]),
    ("redd by sedebh", [saved_track("RED", ["Seedhe Maut"])], ["Seedhe Maut"]),
    ("red by sedebh", [saved_track("RED", ["Seedhe Maut"])], []),
])
def test_partial_saved_artist_refuses_ambiguous_unrelated_short_or_unsupported_source(query, rows, known):
    from assistant.spotify import SpotifyRateLimited
    sp, client = _banned_saved_spotify(rows, known)
    with pytest.raises(SpotifyRateLimited, match="temporarily unavailable"):
        sp.resolve_best_available(query, offline_only=True)
    client.search.assert_not_called()


@pytest.mark.parametrize("query", ["zxqv nonsensical", "glorp blaf", "song track music"])
def test_banned_no_artist_gibberish_cannot_choose_nearest_saved_track(query):
    from assistant.spotify import SpotifyRateLimited
    sp, client = _banned_saved_spotify([saved_track("Black Swan", ["Artist"]),
        saved_track("Some Track Of Music", ["Artist"], "spotify:track:other")])
    with pytest.raises(SpotifyRateLimited, match="temporarily unavailable"):
        sp.resolve_best_available(query, offline_only=True)
    client.search.assert_not_called()


@pytest.mark.parametrize("query,name", [("zxqv live", "Yesterday Live"),
                                       ("zxqv remix", "Yesterday Remix"),
                                       ("zxqv 2024 remastered", "Yesterday - 2024 Remastered"),
                                       ("zxqv take 123", "Yesterday Take 123")])
def test_banned_edition_only_overlap_does_not_ground_an_unrelated_saved_title(query, name):
    from assistant.spotify import SpotifyRateLimited
    sp, client = _banned_saved_spotify([saved_track(name, ["Artist"])])
    with pytest.raises(SpotifyRateLimited, match="temporarily unavailable"):
        sp.resolve_best_available(query, offline_only=True)
    client.search.assert_not_called()


@pytest.mark.parametrize("name", ["Live", "Remix", "Take 123"])
def test_banned_exact_literal_edition_word_title_remains_eligible(name):
    sp, client = _banned_saved_spotify([saved_track(name, ["Artist"])])
    assert sp.resolve_best_available(name, offline_only=True).name == name
    client.search.assert_not_called()


def test_banned_no_artist_partial_title_with_meaningful_word_can_choose_saved_track():
    sp, client = _banned_saved_spotify([saved_track("Bridge Over Troubled Water", ["Artist"])])
    assert sp.resolve_best_available("bridge water", offline_only=True).name == "Bridge Over Troubled Water"
    client.search.assert_not_called()


def test_banned_cached_search_reuses_actual_relevance_without_claiming_catalogue_order():
    sp, client = _banned_saved_spotify([])
    sp._music_search_cache = {("track", "forgotten sound"): [
        track("Actual Retrieved Song", "Artist", "spotify:track:cached")]}
    hit = sp.resolve_best_available("forgotten sound", offline_only=True)
    assert hit.uri == "spotify:track:cached"
    assert sp.last_resolution["path"] == "best_available_cached_search_track"
    client.search.assert_not_called()


def test_banned_album_resolution_can_use_strict_cached_metadata_without_search():
    sp, client = _banned_saved_spotify([])
    sp._music_search_cache = {("album", "keychain lalu by arpit bala"): [album()]}
    assert sp.resolve_album("keychain lalu by arpit bala", offline_only=True).uri == "spotify:album:release"
    client.search.assert_not_called()


def test_saved_track_album_name_does_not_invent_an_album_entity():
    from assistant.spotify import SpotifyRateLimited
    row = {**saved_track("A Song", ["Artist"]), "album": "Blue Lights"}
    sp, client = _banned_saved_spotify([row])
    with pytest.raises(SpotifyRateLimited, match="temporarily unavailable"):
        sp.resolve_best_available("blue lights by artist", "album", offline_only=True)
    client.search.assert_not_called()


@pytest.mark.parametrize("error", [RuntimeError("transport offline"),])
def test_best_available_new_search_transport_error_propagates(error):
    sp = Spotify.__new__(Spotify)
    sp.sp = SimpleNamespace(search=Mock(side_effect=error))
    with pytest.raises(RuntimeError, match="transport offline"):
        sp.resolve_best_available("forgotten")
    assert sp.sp.search.call_count == 1


@pytest.mark.parametrize("query", ["keychain lalu by arpit bala", "KeyChainLalu by Arpit Bala",
                                   "Key Chain Lalu by Arpit Bala", "keychain lalu"])
def test_album_match_joins_words_without_an_entity_alias(query):
    sp = Spotify.__new__(Spotify)
    sp.sp = SimpleNamespace(search=Mock(return_value={"albums": {"items": [album()]}}))
    assert sp.resolve_album(query).uri == "spotify:album:release"
    assert sp.last_resolution["path"] == "album_search"


def test_album_fielded_empty_falls_back_to_plain_query():
    sp = Spotify.__new__(Spotify)
    sp.sp = SimpleNamespace(search=Mock(side_effect=[{"albums": {"items": []}},
        {"albums": {"items": [album()]}}]))
    assert sp.resolve_album("keychain lalu by arpit bala").name == "Keychain Laalu"
    assert sp.sp.search.call_args_list[1].kwargs["q"] == "keychain lalu by arpit bala"
    assert sp.sp.search.call_count == 2


def test_album_featured_credit_is_accepted_and_wrong_artist_rejected():
    sp = Spotify.__new__(Spotify)
    row = album(artists=("Producer", "Arpit Bala"))
    sp.sp = SimpleNamespace(search=Mock(return_value={"albums": {"items": [row]}}))
    result = sp.resolve_album("keychain lalu by arpit bala")
    assert result.artist == "Producer, Arpit Bala"
    with pytest.raises(LookupError):
        sp.resolve_album("keychain lalu by unrelated artist")


def test_album_duplicate_uri_collapses_but_distinct_near_equal_releases_are_ambiguous():
    sp = Spotify.__new__(Spotify)
    sp.sp = SimpleNamespace(search=Mock(return_value={"albums": {"items": [album(), album()]}}))
    assert sp.resolve_album("keychain lalu").uri == "spotify:album:release"
    sp.sp.search.return_value = {"albums": {"items": [album(), album(uri="spotify:album:other")]}}
    with pytest.raises(AmbiguousMatch):
        sp.resolve_album("keychain lalu")


@pytest.mark.parametrize("row,query", [
    (album("Completely Different"), "keychain lalu"),
    (album("A Very Long Album Title"), "a very long album title live"),
    (album("A Very Long Album Title - Remastered 2009"), "a very long album title 2015 remaster"),
    (album(uri="spotify:track:wrong"), "keychain lalu"),
])
def test_album_weak_titles_editions_and_wrong_entity_uris_do_not_play(row, query):
    sp = Spotify.__new__(Spotify)
    sp.sp = SimpleNamespace(search=Mock(return_value={"albums": {"items": [row]}}))
    with pytest.raises(LookupError):
        sp.resolve_album(query)


def test_album_context_is_verified_and_does_not_change_shuffle():
    from contextlib import nullcontext
    from assistant.resolution import Candidate
    sp = Spotify.__new__(Spotify)
    sp.sp = Mock()
    sp.command_target = lambda: nullcontext()
    sp.device = lambda: "laptop"
    sp.sp.current_playback.return_value = {"device": {"id": "laptop"}, "is_playing": True,
        "context": {"uri": "spotify:album:release"}, "shuffle_state": True}
    result = Candidate("spotify:album:release", "Keychain Laalu", "Arpit Bala", album())
    assert sp.play_album(result) == result.label
    sp.sp.start_playback.assert_called_once_with(context_uri=result.uri, device_id="laptop",
                                               offset={"position": 0}, position_ms=0)
    sp.sp.shuffle.assert_not_called()
    sp.sp.repeat.assert_not_called()
    assert sp.radio_played is False and sp.last_playback["radio"] == 0


@pytest.mark.parametrize("state", [
    {"is_playing": False, "context": {"uri": "spotify:album:release"}, "device": {"id": "laptop"}},
    {"is_playing": True, "context": {"uri": "spotify:album:other"}, "device": {"id": "laptop"}},
    {"is_playing": True, "item": {"album": {"uri": "spotify:album:release"}}, "device": {"id": "laptop"}},
    {"is_playing": True, "context": {"uri": "spotify:album:release"}, "device": {"id": "remote"}},
])
def test_album_api_acknowledgement_without_pinned_context_proof_fails(state, monkeypatch):
    from contextlib import nullcontext
    from assistant.resolution import Candidate
    from assistant.spotify import PlaybackVerificationError
    sp = Spotify.__new__(Spotify)
    sp.sp = Mock()
    sp.command_target = lambda: nullcontext()
    sp.device = lambda: "laptop"
    sp.sp.current_playback.return_value = state
    monkeypatch.setattr("assistant.spotify.time.sleep", lambda _: None)
    with pytest.raises(PlaybackVerificationError):
        sp.play_album(Candidate("spotify:album:release", "Keychain Laalu", "Arpit Bala", album()))
    sp.sp.start_playback.assert_called_once()


@pytest.mark.parametrize("rate_limit_stage", ["mutation", "verification"])
def test_album_player_rate_limit_propagates_even_when_native_album_metadata_matches(monkeypatch, rate_limit_stage):
    from contextlib import nullcontext
    from assistant import local_media
    from assistant.resolution import Candidate
    from assistant.spotify import SpotifyRateLimited
    sp = Spotify.__new__(Spotify)
    sp.sp = Mock()
    if rate_limit_stage == "mutation":
        sp.sp.start_playback.side_effect = SpotifyRateLimited("banned")
    else:
        sp.sp.current_playback.side_effect = SpotifyRateLimited("banned")
    sp.command_target = lambda: nullcontext()
    sp.device = lambda: "laptop"
    snapshot = Mock(return_value=SimpleNamespace(title="A Track", artist="Arpit Bala",
                    playing=True, album_title="Keychain Laalu"))
    monkeypatch.setattr(local_media, "snapshot", snapshot)
    pause = Mock()
    monkeypatch.setattr(local_media, "pause", pause)
    opened = Mock()
    monkeypatch.setattr("assistant.spotify.os.startfile", opened)
    candidate = Candidate("spotify:album:release", "Keychain Laalu", "Arpit Bala", album())
    with pytest.raises(SpotifyRateLimited):
        sp.play_album(candidate)
    snapshot.assert_not_called()
    pause.assert_not_called()
    opened.assert_not_called()
    sp.sp.start_playback.assert_called_once()
    assert sp.last_playback is None


def test_featured_track_credit_is_hard_constraint_and_complete_credits_stay_distinct():
    from assistant.resolution import collapse_reissues
    rows = [track("Maharani", "E Karun", artists=["E Karun", "Arpit Bala"]),
            track("Maharani", "E Karun", uri="spotify:track:other", artists=["E Karun", "Other"])]
    assert len(rank_tracks(rows, "Maharani", "Arpit Bala")) == 1
    assert not rank_tracks(rows, "Maharani", "Unrelated")
    assert len(collapse_reissues(rank_tracks(rows, "Maharani", "E Karun"))) == 2


@pytest.mark.parametrize("credits,expected", [("E Karun, Arpit Bala", True),
    ("E Karun & Arpit Bala", True), ("E Karun, Arpit Bala Tribute", False),
    ("Arpit Bala", False), ("E Karun", False)])
def test_track_native_joined_credits_are_exact_sets(credits, expected):
    from assistant.resolution import Candidate
    from assistant.spotify import _matches_local_candidate
    row = track("Maharani", "E Karun", artists=["E Karun", "Arpit Bala"])
    candidate = Candidate(row["uri"], row["name"], "E Karun", row)
    assert _matches_local_candidate(candidate, SimpleNamespace(
        title="Maharani", artist=credits, playing=True)) is expected


def test_featured_credit_search_selects_the_right_track_and_distinct_credit_sets_need_choice():
    sp = Spotify.__new__(Spotify)
    rows = [track("Maharani", "E Karun", artists=["E Karun", "Arpit Bala"]),
            track("Maharani", "Other", uri="spotify:track:unrelated", artists=["Other"])]
    sp.sp = SimpleNamespace(search=Mock(return_value={"tracks": {"items": rows}}))
    sp._resolve_from_catalogue = Mock(return_value=None)
    assert sp.resolve_track("Maharani by Arpit Bala").artist == "E Karun"
    rows.append(track("Maharani", "E Karun", uri="spotify:track:guest",
                      artists=["E Karun", "Arpit Bala", "Guest"]))
    with pytest.raises(AmbiguousMatch):
        sp.resolve_track("Maharani by Arpit Bala")


def test_featured_artist_cache_preserves_canonical_credit_metadata(tmp_path, monkeypatch):
    from assistant.resolution import Candidate
    monkeypatch.setattr("assistant.spotify.TRACK_CACHE", tmp_path / "cache.json")
    sp = Spotify.__new__(Spotify)
    sp._tracks = {}
    row = track("Maharani", "E Karun", artists=["E Karun", "Arpit Bala"])
    candidate = Candidate(row["uri"], row["name"], "E Karun", row)
    sp.remember_track("Maharani by Arpit Bala", candidate)
    hit = sp.cached_track("Maharani by Arpit Bala")
    assert hit.uri == candidate.uri
    assert hit.metadata["artists"] == row["artists"]
    assert sp.cached_track("Maharani by Other") is None


@pytest.mark.parametrize("include_album", [True, False])
def test_media_snapshot_album_title_is_backward_compatible(include_album, monkeypatch):
    from assistant import local_media
    props = SimpleNamespace(title="A Track", artist="Artist")
    if include_album:
        props.album_title = "An Album"
    async def properties():
        return props
    async def session():
        return SimpleNamespace(get_playback_info=lambda: SimpleNamespace(playback_status=local_media.PLAYING),
                               try_get_media_properties_async=properties)
    monkeypatch.setattr(local_media, "_session", session)
    snapshot = local_media.snapshot()
    assert snapshot.playing is True and snapshot.title == "A Track"
    assert snapshot.album_title == ("An Album" if include_album else "")


def test_artist_is_a_hard_constraint():
    rows = rank_tracks([track("Hello", "Adele"), track("Hello", "Lionel Richie")],
                       "Hello", "Lionel Richie")
    assert [x.artist for x in rows] == ["Lionel Richie"]


def test_client_detection_uses_process_names_not_blocked_tasklist(monkeypatch):
    import psutil
    monkeypatch.setattr(psutil, "process_iter", lambda _: [
        SimpleNamespace(info={"name": "Other.exe"}),
        SimpleNamespace(info={"name": "Spotify.exe"}),
    ])
    assert Spotify._client_running() is True


def test_recognized_command_launches_closed_client_only_once(monkeypatch):
    sp = Spotify.__new__(Spotify)
    sp._client_running = Mock(return_value=False)
    launch = Mock()
    sp._launch_client = launch
    assert sp.ensure_client_running() is True
    assert sp.ensure_client_running() is True
    launch.assert_called_once()
    sp._client_running.return_value = True
    assert sp.ensure_client_running() is False


def test_windows_uri_handler_is_used_to_open_spotify(monkeypatch):
    import assistant.spotify as spotify_module
    startfile = Mock()
    monkeypatch.setattr(spotify_module.os, "startfile", startfile)
    Spotify.__new__(Spotify)._launch_client()
    startfile.assert_called_once_with("spotify:")


def test_title_only_query_is_not_treated_as_artist():
    sp = Spotify.__new__(Spotify)
    sp.sp = SimpleNamespace(search=lambda **kw: {"tracks": {"items": [
        track("Demons", "Imagine Dragons", "spotify:track:demons")
    ]}})
    assert sp.resolve_track("demons").uri == "spotify:track:demons"


def test_catalogue_reissues_do_not_create_false_ambiguity():
    sp = Spotify.__new__(Spotify)
    sp.sp = SimpleNamespace(search=lambda **kw: {"tracks": {"items": []}})
    sp._artist_catalogue = lambda _: [
        track('Naksha', 'Seedhe Maut', 'spotify:track:album'),
        track('Naksha', 'Seedhe Maut', 'spotify:track:single'),
        track('Nanchaku', 'Seedhe Maut'),
    ]
    assert sp.resolve_track('nakhsha by seedhe maut').name == 'Naksha'


@pytest.mark.parametrize("query", ["yesterday", "yesterday by the beatles"])
def test_plain_song_request_accepts_remaster_releases(query):
    sp = Spotify.__new__(Spotify)
    rows = [track("Yesterday - Remastered 2009", "The Beatles", "spotify:track:a"),
            track("Yesterday (2015 Remaster)", "The Beatles", "spotify:track:b"),
            track("Yesterday - Live", "The Beatles", "spotify:track:live")]
    sp.sp = SimpleNamespace(search=Mock(return_value={"tracks": {"items": rows}}))
    sp._artist_catalogue = Mock(side_effect=AssertionError("Exact title needs no catalogue fallback"))
    assert sp.resolve_track(query).uri == "spotify:track:a"


def test_remaster_grouping_preserves_explicit_edition_and_other_recordings():
    from assistant.resolution import Candidate, collapse_reissues, title_key
    names = ["Song - Remastered 2009", "Song (2015 Remaster)", "Song - Live",
             "Song - Remix", "Song - Acoustic", "Song (Mono Remastered)", "Song - Remastered Live"]
    candidates = [Candidate(str(i), name, "Artist") for i, name in enumerate(names)]
    assert len(collapse_reissues(candidates)) == len(names) - 1
    assert len(collapse_reissues(candidates, "Song 2015 Remaster")) == len(names)
    assert title_key("Remastered") == "remastered"
    assert title_key("Song - Live (2009 Remaster)") == "song live"
    rows = [track(name, "Artist", str(i)) for i, name in enumerate(names)]
    assert rank_tracks(rows, "Song 2015 Remaster")[0].uri == "1"


def test_remasters_by_competing_artists_remain_ambiguous():
    sp = Spotify.__new__(Spotify)
    sp.sp = SimpleNamespace(search=Mock(return_value={"tracks": {"items": [
        track("Yesterday - Remastered 2009", "The Beatles"),
        track("Yesterday (2015 Remaster)", "Another Artist") ]}}))
    with pytest.raises(AmbiguousMatch):
        sp.resolve_track("yesterday")


def test_reissues_with_different_collaborators_remain_distinct():
    from assistant.resolution import collapse_reissues
    rows = [track("Song", "Artist", artists=["Artist", "Guest A"]),
            track("Song - Remastered 2009", "Artist", artists=["Artist", "Guest B"])]
    assert len(collapse_reissues(rank_tracks(rows, "Song"))) == 2


@pytest.mark.parametrize("qualifier", ["live", "acoustic", "remix", "2023 mix", "take 1", "2015 remaster"])
def test_requested_recording_version_is_not_replaced_by_studio(qualifier):
    assert not rank_tracks([track("Yesterday - Remastered 2009", "The Beatles")],
                           "Yesterday " + qualifier, "The Beatles")


def test_exact_title_hints_are_narrow_and_not_a_global_artist_alias():
    sp = Spotify.__new__(Spotify)
    sp.sp = SimpleNamespace(search=lambda **kw: {"tracks": {"items": [
        track("RED", "Seedhe Maut", artists=["Seedhe Maut", "Hisab"]),
        track("Redbone", "Childish Gambino"),
    ]}})
    assert sp.artist_hints_for_exact_title("red by see them all") == ["Seedhe Maut", "Hisab"]
    assert sp.artist_hints_for_exact_title("red by aerosmith") == []
    assert sp.artist_hints_for_exact_title("red") == []


def test_exact_title_hints_reject_competing_artists():
    sp = Spotify.__new__(Spotify)
    sp.sp = SimpleNamespace(search=lambda **kw: {"tracks": {"items": [
        track("RED", "Seedhe Maut"), track("RED", "Another Artist"),
    ]}})
    assert sp.artist_hints_for_exact_title("red by see them all") == []


def test_single_loose_title_result_is_not_played():
    sp = Spotify.__new__(Spotify)
    sp.sp = SimpleNamespace(search=lambda **kw: {"tracks": {"items": [
        track("See The Light", "Lim Hyunsik")
    ]}})
    with pytest.raises(LookupError, match="No confident track match"):
        sp.resolve_track("See the Moth")


def test_misheard_artist_recovers_only_from_exact_title_and_matching_sound():
    red = track("RED", "Seedhe Maut")
    sp = Spotify.__new__(Spotify)
    def search(**kw):
        return {"tracks": {"items": [red] if kw['q'] == 'track:red' else []}}
    sp.sp = SimpleNamespace(search=search)
    assert sp.resolve_track('red by sidaimoth').label == 'RED by Seedhe Maut'
    with pytest.raises(LookupError):
        sp.resolve_track('red by aerosmith')


def local_music_spotify(rows):
    sp = Spotify.__new__(Spotify)
    sp.sp = SimpleNamespace(search=Mock(side_effect=AssertionError("local proposal must precede Spotify search")))
    sp.set_catalogue(Index({"tracks": rows, "complete": True}))
    return sp


def test_local_near_title_with_grounded_artist_requires_confirmation():
    red = {"uri": "spotify:track:red", "name": "RED",
           "artists": ["Seedhe Maut", "Hisab"], "album": "Kshama",
           "sources": ["artist"]}
    sp = local_music_spotify([red])
    with pytest.raises(MusicConfirmationRequired) as caught:
        sp.resolve_track("layer red by siddhimoth")
    assert caught.value.query == "layer red by siddhimoth"
    assert [candidate.label for candidate in caught.value.candidates] == [
        "RED by Seedhe Maut"]
    assert sp.last_resolution["path"] == "local_confirmation"
    sp.sp.search.assert_not_called()


def test_local_exact_title_with_unrecognized_artist_is_confirmation_only():
    red = {"uri": "spotify:track:red", "name": "RED",
           "artists": ["Seedhe Maut", "Hisab"], "album": "Kshama"}
    sp = local_music_spotify([red])
    with pytest.raises(MusicConfirmationRequired) as caught:
        sp.resolve_track("red by sirian north")
    assert [candidate.label for candidate in caught.value.candidates] == [
        "RED by Seedhe Maut"]
    assert sp.last_resolution["path"] == "local_confirmation"
    sp.sp.search.assert_not_called()


def test_local_proposals_never_autoaccept_negative_or_unrequested_editions():
    rows = [{"uri": "spotify:track:red", "name": "RED",
             "artists": ["Seedhe Maut"], "album": "Kshama"}]
    sp = local_music_spotify(rows)
    with pytest.raises(MusicConfirmationRequired) as caught:
        sp.resolve_track("not red by seedhe maut")
    assert [candidate.name for candidate in caught.value.candidates] == ["RED"]
    sp.sp.search.assert_not_called()
    assert sp._local_confirmation_proposals("red live by seedhe maut", sp.catalogue_index()) == []
    assert sp._local_confirmation_proposals("red remix by seedhe maut", sp.catalogue_index()) == []

    live_artist = {"uri": "spotify:track:echo", "name": "Echo",
                   "artists": ["Live"], "album": "Studio"}
    live_artist_sp = local_music_spotify([live_artist])
    proposals = live_artist_sp._local_confirmation_proposals(
        "echo by Live", live_artist_sp.catalogue_index())
    assert [candidate.label for candidate in proposals] == ["Echo by Live"]

    live = {"uri": "spotify:track:redlive", "name": "RED - Live",
            "artists": ["Seedhe Maut"], "album": "Live"}
    editions = Spotify.__new__(Spotify)
    editions.sp = SimpleNamespace(search=Mock(side_effect=AssertionError("no search")))
    editions.set_catalogue(Index({"tracks": rows + [live], "complete": True}))
    proposals = editions._local_confirmation_proposals("red live by another artist",
                                                       editions.catalogue_index())
    assert [candidate.name for candidate in proposals] == ["RED - Live"]


def test_exact_title_proposals_preserve_distinct_artists_and_cap_choices():
    rows = [{"uri": "spotify:track:red" + str(i), "name": "RED",
             "artists": ["Artist " + str(i)], "album": "Release"}
            for i in range(5)]
    sp = local_music_spotify(rows)
    with pytest.raises(MusicConfirmationRequired) as caught:
        sp.resolve_track("red by unknown act")
    assert len(caught.value.candidates) == 3
    assert len({candidate.artist for candidate in caught.value.candidates}) == 3
    assert {candidate.artist for candidate in caught.value.candidates} <= {
        "Artist " + str(i) for i in range(5)}


def test_weaker_artist_sound_needs_spelling_evidence_and_clear_margin():
    sp = Spotify.__new__(Spotify)
    rows = [track('RED', 'Aerosmith'), track('RED', 'Seedhe Maut')]
    sp.sp = SimpleNamespace(search=lambda **kw: {
        'tracks': {'items': rows if kw['q'] == 'track:red' else []}})
    assert sp.resolve_track('red by See The Moth').label == 'RED by Seedhe Maut'
    assert sp.resolve_track('red by Aerosmith').label == 'RED by Aerosmith'


def test_weaker_artist_sound_rejects_close_competing_spelling():
    sp = Spotify.__new__(Spotify)
    rows = [track('RED', 'Seedhe Maut'), track('RED', 'Sid Moth')]
    sp.sp = SimpleNamespace(search=lambda **kw: {
        'tracks': {'items': rows if kw['q'] == 'track:red' else []}})
    with pytest.raises(LookupError):
        sp.resolve_track('red by See The Moth')


def test_artist_recovery_rejects_competing_exact_title_artists():
    sp = Spotify.__new__(Spotify)
    def search(**kw):
        rows = [track('RED', 'Seedhe Maut'), track('RED', 'Sid Moth')]
        return {"tracks": {"items": rows if kw['q'] == 'track:red' else []}}
    sp.sp = SimpleNamespace(search=search)
    with pytest.raises(LookupError):
        sp.resolve_track('red by sidaimoth')


def test_both_title_and_artist_can_be_recovered_without_song_alias():
    madira = track('Madira', 'Seedhe Maut')
    sp = Spotify.__new__(Spotify)
    sp.sp = SimpleNamespace(search=lambda **kw: {
        'tracks': {'items': [madira] if kw['q'] == 'madeira' else []}})
    sp._artist_catalogue = Mock(return_value=[])

    assert sp.resolve_track('madeira by seedemoth').label == 'Madira by Seedhe Maut'
    with pytest.raises(LookupError):
        sp.resolve_track('madeira by aerosmith')


def test_double_spelling_recovery_refuses_competing_artists():
    sp = Spotify.__new__(Spotify)
    rows = [track('Madira', 'Seedhe Maut'), track('Madira', 'Seedemoth')]
    sp.sp = SimpleNamespace(search=lambda **kw: {
        'tracks': {'items': rows if kw['q'] == 'madeira' else []}})
    sp._artist_catalogue = Mock(return_value=[])

    with pytest.raises(LookupError):
        sp.resolve_track('madeira by seedemoth')


def test_snapshot_uses_one_request_and_rejects_foreign_device():
    from unittest.mock import Mock
    sp = Spotify.__new__(Spotify); sp.sp = Mock(); sp.device = lambda: 'laptop'
    sp.sp.current_playback.return_value = {'is_playing': True, 'device': {'id': 'laptop', 'volume_percent': 50}}
    assert sp.playback_snapshot() == (True, 50)
    sp.sp.current_playback.assert_called_once()
    sp.sp.current_playback.return_value['device']['id'] = 'phone'
    assert sp.playback_snapshot() == (None, None)


def test_completely_unrelated_single_result_is_rejected():
    assert rank_tracks([track("Seedhe Maut", "Artist")], "Aerosmith") == []


def test_close_title_requires_confirmation():
    rows = rank_tracks([track("Bohemian Rhapsody", "Queen"),
                        track("Bohemian Rhapsodies", "Other")], "Bohemian Rhapsody")
    assert rows[0].name == "Bohemian Rhapsody"
    # A caller must not silently choose a fuzzy title when it is not exact.
    with pytest.raises(AmbiguousMatch):
        raise AmbiguousMatch("Bohemian Rhapsod", rows)


def test_unicode_and_multiple_artists_are_preserved():
    rows = rank_tracks([track("दिल से", "A.R. Rahman", artists=["A.R. Rahman", "Alka Yagnik"])],
                       "दिल से", "A.R. Rahman")
    assert rows[0].artist == "A.R. Rahman"
    assert rows[0].metadata["artists"][1]["name"] == "Alka Yagnik"


def test_play_track_uses_candidate_and_never_first_result(monkeypatch):
    sp = Spotify.__new__(Spotify)
    sp.sp = SimpleNamespace(search=lambda **kw: {"tracks": {"items": [
        track("Same", "Wrong", "spotify:track:wrong"), track("Same", "Right", "spotify:track:right")
    ]}})
    played = []
    sp.play_candidate = lambda c: played.append(c.uri) or c.label
    assert sp.play_track("Same by Right") == "Same by Right"
    assert played == ["spotify:track:right"]


def test_device_does_not_choose_arbitrary_remote(monkeypatch):
    sp = Spotify.__new__(Spotify)
    sp._cfg = SimpleNamespace(get=lambda key, default=None: "My Laptop" if key == "spotify.device_name" else default)
    sp._device_id = sp._device_name = ""
    sp.sp = SimpleNamespace(devices=lambda: {"devices": [
        {"id": "phone", "name": "Jane's Phone", "type": "Smartphone", "is_active": True},
        {"id": "laptop", "name": "My Laptop", "type": "Computer", "is_active": False},
    ]})
    monkeypatch.setattr("assistant.spotify.socket.gethostname", lambda: "other")
    monkeypatch.setattr(sp, "_remember", lambda *args: None)
    assert sp.device() == "laptop"


def test_other_active_computer_is_not_a_target(monkeypatch):
    sp = Spotify.__new__(Spotify)
    sp._cfg = SimpleNamespace(get=lambda key, default=None: default)
    sp._device_id = None
    sp.sp = SimpleNamespace(devices=lambda: {'devices': [
        {'id': 'remote', 'name': 'Someone Else', 'type': 'Computer', 'is_active': True}]})
    monkeypatch.setattr('assistant.spotify.socket.gethostname', lambda: 'This Laptop')
    sp._remembered = lambda: ('cached-other', 'Someone Else')
    sp._wake_device = lambda _: pytest.fail('must not transfer to foreign cache')
    with pytest.raises(RuntimeError, match='No Spotify device'):
        sp.device(retry_launch=False)


def test_active_target_can_resume_unique_idle_local_laptop(monkeypatch):
    sp = Spotify.__new__(Spotify)
    sp._cfg = SimpleNamespace(get=lambda key, default=None: "active" if key == "spotify.target" else default)
    sp._device_id = sp._device_name = ""
    sp.sp = SimpleNamespace(devices=lambda: {"devices": [
        {"id": "laptop", "name": "APRATXM", "type": "Computer", "is_active": False,
         "is_restricted": False},
        {"id": "phone", "name": "Phone", "type": "Smartphone", "is_active": False,
         "is_restricted": False},
    ]})
    monkeypatch.setattr('assistant.spotify.socket.gethostname', lambda: 'APRATXM')
    assert sp.device() == 'laptop'


def test_active_target_refuses_other_idle_computer(monkeypatch):
    from assistant.spotify import PlaybackVerificationError
    sp = Spotify.__new__(Spotify)
    sp._cfg = SimpleNamespace(get=lambda key, default=None: "active" if key == "spotify.target" else default)
    sp._device_id = sp._device_name = ""
    sp.sp = SimpleNamespace(devices=lambda: {"devices": [
        {"id": "other", "name": "Someone Else", "type": "Computer", "is_active": False,
         "is_restricted": False},
    ]})
    monkeypatch.setattr('assistant.spotify.socket.gethostname', lambda: 'APRATXM')
    with pytest.raises(PlaybackVerificationError, match='No active controllable'):
        sp.device()


def test_active_target_is_pinned_through_mutation_and_verification(monkeypatch):
    sp = Spotify.__new__(Spotify)
    # Web API path under test: keep the real local media controls out of it.
    sp._cfg = SimpleNamespace(get=lambda key, default=None: {
        "spotify.target": "active", "spotify.local_transport": False}.get(key, default))
    sp._device_id = sp._device_name = ""
    sp._command_device_id = None
    sp._command_device_name = ""
    sp._command_depth = 0
    sp.sp = SimpleNamespace(
        devices=Mock(return_value={"devices": [
            {"id": "first", "name": "First", "is_active": True, "is_restricted": False},
            {"id": "later", "name": "Later", "is_active": False, "is_restricted": False},
        ]}),
        start_playback=Mock(),
        current_playback=Mock(return_value={
            "device": {"id": "first"}, "is_playing": True,
        }),
    )
    monkeypatch.setattr("assistant.spotify.time.sleep", lambda _: None)

    sp.play_resume()

    sp.sp.start_playback.assert_called_once_with(device_id="first")
    assert sp.sp.devices.call_count == 1


def test_restart_seeks_current_track_on_pinned_device():
    sp = Spotify.__new__(Spotify)
    sp.device = lambda: "laptop"
    sp._device_name = "Laptop"
    sp._command_depth = 0
    sp._command_device_id = None
    state = {"device": {"id": "laptop"}, "item": {"uri": "spotify:track:one"},
             "progress_ms": 42000}
    sp.sp = SimpleNamespace(current_playback=Mock(side_effect=[state, {**state, "progress_ms": 500}]),
                            seek_track=Mock())
    sp.restart_track()
    sp.sp.seek_track.assert_called_once_with(0, device_id="laptop")


def test_restart_refuses_when_no_current_track():
    from assistant.spotify import PlaybackVerificationError
    sp = Spotify.__new__(Spotify)
    sp.device = lambda: "laptop"
    sp._device_name = "Laptop"
    sp._command_depth = 0
    sp._command_device_id = None
    sp.sp = SimpleNamespace(current_playback=Mock(return_value={
        "device": {"id": "laptop"}, "item": None}), seek_track=Mock())
    with pytest.raises(PlaybackVerificationError, match="No current track"):
        sp.restart_track()
    sp.sp.seek_track.assert_not_called()


def _restart_album_player():
    sp = Spotify.__new__(Spotify)
    sp.device = lambda: "laptop"
    sp._device_name = "Laptop"
    sp._command_depth = 0
    sp._command_device_id = None
    state = {"device": {"id": "laptop", "volume_percent": 33},
             "context": {"type": "album", "uri": "spotify:album:release"},
             "item": {"uri": "spotify:track:later"}, "progress_ms": 42000,
             "shuffle_state": True, "is_playing": True}
    sp.sp = Mock()
    sp.sp.album_tracks.return_value = {"items": [{"uri": "spotify:track:first", "is_playable": True}]}
    return sp, state


def test_restart_album_requires_actual_context_and_verifies_first_track_without_shuffle_changes():
    sp, state = _restart_album_player()
    sp.sp.current_playback.side_effect = [state, state,
        {**state, "item": {"uri": "spotify:track:first"}, "progress_ms": 500}]
    assert sp.restart_album() == "Restarted the current album."
    sp.sp.album_tracks.assert_called_once_with("spotify:album:release", limit=1)
    sp.sp.start_playback.assert_called_once_with(context_uri="spotify:album:release",
        offset={"position": 0}, position_ms=0, device_id="laptop")
    sp.sp.shuffle.assert_not_called()
    sp.sp.volume.assert_not_called()
    assert sp.last_playback["method"] == "web_api_album_restart_verified"


@pytest.mark.parametrize("context,device", [
    (None, "laptop"), ({"type": "playlist", "uri": "spotify:album:release"}, "laptop"),
    ({"type": "album", "uri": "spotify:track:wrong"}, "laptop"),
    ({"type": "album", "uri": "spotify:album:release"}, "remote"),
])
def test_restart_album_never_infers_context_from_track_album_metadata(context, device):
    from assistant.spotify import PlaybackVerificationError
    sp, state = _restart_album_player()
    sp.sp.current_playback.return_value = {**state, "context": context, "device": {"id": device},
        "item": {"uri": "spotify:track:first", "album": {"uri": "spotify:album:release"}}}
    with pytest.raises(PlaybackVerificationError):
        sp.restart_album()
    sp.sp.album_tracks.assert_not_called()
    sp.sp.start_playback.assert_not_called()


def test_restart_album_context_change_during_lookup_does_not_mutate():
    from assistant.spotify import PlaybackVerificationError
    sp, state = _restart_album_player()
    sp.sp.current_playback.side_effect = [state, {**state, "context": {"type": "album", "uri": "spotify:album:new"}}]
    with pytest.raises(PlaybackVerificationError):
        sp.restart_album()
    sp.sp.start_playback.assert_not_called()


@pytest.mark.parametrize("first", [None, {"uri": "spotify:album:wrong"},
    {"uri": "spotify:track:first", "is_playable": False},
    {"uri": "spotify:track:first", "type": "episode"},
    {"uri": "spotify:track:first", "is_local": True}])
def test_restart_album_invalid_first_track_never_mutates(first):
    from assistant.spotify import PlaybackVerificationError
    sp, state = _restart_album_player()
    sp.sp.current_playback.return_value = state
    sp.sp.album_tracks.return_value = {"items": [first]}
    with pytest.raises(PlaybackVerificationError):
        sp.restart_album()
    sp.sp.start_playback.assert_not_called()


@pytest.mark.parametrize("changes", [
    {"item": {"uri": "spotify:track:later"}, "progress_ms": 500},
    {"item": {"uri": "spotify:track:first"}, "progress_ms": None},
    {"item": {"uri": "spotify:track:first"}, "progress_ms": 12000},
    {"item": {"uri": "spotify:track:first"}, "progress_ms": 500, "device": {"id": "remote"}},
])
def test_restart_album_ack_without_first_track_progress_device_proof_is_uncertain(changes, monkeypatch):
    from assistant.spotify import PlaybackVerificationError
    sp, state = _restart_album_player()
    after = {**state, **changes}
    sp.sp.current_playback.side_effect = [state, state, after, after, after, after]
    monkeypatch.setattr("assistant.spotify.time.sleep", lambda _: None)
    with pytest.raises(PlaybackVerificationError):
        sp.restart_album()
    sp.sp.start_playback.assert_called_once()
    assert sp.last_playback is None


@pytest.mark.parametrize('status', [404, 403])
def test_stale_or_restricted_mutation_is_not_retried(status):
    import spotipy
    sp = Spotify.__new__(Spotify)
    sp._device_id, sp._device_name = 'old', 'laptop'
    sp.device = lambda: sp._device_id or 'new'
    calls = []
    def action(**kwargs):
        calls.append(kwargs['device_id'])
        if len(calls) == 1:
            raise spotipy.SpotifyException(status, -1, 'test error')
    with pytest.raises(spotipy.SpotifyException):
        sp._call(action)
    assert calls == ['old']


def test_rate_limit_fails_fast_and_stops_calling(monkeypatch):
    import spotipy
    from assistant.spotify import SpotifyRateLimited, _RateLimitGuard
    client = Mock()
    client.devices.side_effect = spotipy.SpotifyException(429, -1, "limit", headers={})
    guard = _RateLimitGuard(client)
    with pytest.raises(SpotifyRateLimited):
        guard.devices()
    with pytest.raises(SpotifyRateLimited):
        guard.devices()
    assert client.devices.call_count == 1   # the second call never reached Spotify


def test_transport_prefers_local_media_controls(monkeypatch):
    from assistant import local_media
    sp = Spotify.__new__(Spotify)
    sp._cfg = SimpleNamespace(get=lambda key, default=None: default)
    sp.sp = Mock()
    monkeypatch.setattr(local_media, "status", lambda: local_media.PLAYING)
    paused = Mock(return_value=True)
    monkeypatch.setattr(local_media, "pause", paused)
    sp.pause()
    paused.assert_called_once()
    sp.sp.pause_playback.assert_not_called()
    sp.sp.current_playback.assert_not_called()


def test_transport_falls_back_to_api_without_a_local_session(monkeypatch):
    from assistant import local_media
    sp = Spotify.__new__(Spotify)
    sp._cfg = SimpleNamespace(get=lambda key, default=None: default)
    monkeypatch.setattr(local_media, "status", lambda: None)
    called = []
    sp.command_target = lambda: __import__("contextlib").nullcontext()
    sp.device = lambda: "laptop"
    sp.sp = Mock()
    sp.sp.current_playback.return_value = {"device": {"id": "laptop"}, "is_playing": False}
    sp.pause()
    sp.sp.current_playback.assert_called()


def test_player_rate_limit_does_not_block_search():
    import spotipy
    from assistant.spotify import SpotifyRateLimited, _RateLimitGuard
    client = Mock()
    client.devices.side_effect = spotipy.SpotifyException(429, -1, "limit", headers={})
    client.search.return_value = {"tracks": {"items": []}}
    guard = _RateLimitGuard(client)
    with pytest.raises(SpotifyRateLimited):
        guard.devices()
    assert guard.search(q="x") == {"tracks": {"items": []}}


def test_rate_limited_player_api_falls_back_to_desktop_app(monkeypatch):
    from contextlib import nullcontext
    from assistant import local_media
    from assistant.resolution import Candidate
    from assistant.spotify import SpotifyRateLimited
    sp = Spotify.__new__(Spotify)
    sp._cfg = SimpleNamespace(get=lambda key, default=None: default)
    sp.sp = Mock()
    sp.sp.start_playback.side_effect = SpotifyRateLimited("banned")
    sp.command_target = lambda: nullcontext()
    sp.device = lambda: "laptop"
    opened = []
    monkeypatch.setattr("assistant.spotify.os.startfile", opened.append)
    monkeypatch.setattr(local_media, "status", lambda: local_media.PAUSED)
    monkeypatch.setattr(local_media, "spotify_in_front", lambda: True)
    monkeypatch.setattr(local_media, "title", lambda: "Madira - Seedhe Maut" if opened else "KODAK - King")
    monkeypatch.setattr(local_media, "snapshot", lambda: SimpleNamespace(
        title="Madira" if opened else "KODAK",
        artist="Seedhe Maut" if opened else "King",
        playing=bool(opened)))
    cand = Candidate("spotify:track:abc", "Madira", "Seedhe Maut")
    assert sp.play_candidate(cand) == cand.label
    assert opened == ["spotify:track:abc"]


def test_player_api_plays_without_opening_the_app(monkeypatch):
    from contextlib import nullcontext
    from assistant import local_media
    from assistant.resolution import Candidate
    sp = Spotify.__new__(Spotify)
    sp._cfg = SimpleNamespace(get=lambda key, default=None: default)
    sp.sp = Mock()
    sp.command_target = lambda: nullcontext()
    sp.device = lambda: "laptop"
    opened = []
    monkeypatch.setattr("assistant.spotify.os.startfile", opened.append)
    monkeypatch.setattr(local_media, "title", lambda: "Madira - Seedhe Maut")
    monkeypatch.setattr(local_media, "snapshot", lambda: SimpleNamespace(
        title="Madira", artist="Seedhe Maut", playing=True))
    cand = Candidate("spotify:track:abc", "Madira", "Seedhe Maut")
    assert sp.play_candidate(cand) == cand.label
    sp.sp.start_playback.assert_called_once_with(uris=["spotify:track:abc"], device_id="laptop")
    sp.sp.current_playback.assert_not_called()
    assert opened == []


def test_accepted_api_native_miss_verifies_uri_on_same_pinned_target(monkeypatch, caplog):
    from contextlib import contextmanager
    import logging
    from assistant.resolution import Candidate

    logger = logging.getLogger("memoir.spotify")
    monkeypatch.setattr(logger, "handlers", [*logger.handlers, caplog.handler])
    monkeypatch.setattr(logger, "propagate", False)
    caplog.set_level("INFO", logger="memoir.spotify")
    sp = Spotify.__new__(Spotify)
    sp._device_name = "laptop"
    sp._cfg = SimpleNamespace(get=lambda key, default=None: default)
    sp.sp = Mock()
    active = []

    @contextmanager
    def target():
        active.append("laptop")
        try:
            yield "laptop"
        finally:
            active.pop()

    sp.command_target = target
    sp.device = lambda: active[-1] if active else "unexpected-device-selection"
    sp._now_playing = lambda candidate, timeout: False
    sp.sp.current_playback.return_value = {"device": {"id": "laptop"}, "is_playing": True,
        "item": {"uri": "spotify:track:other", "linked_from": {"uri": "spotify:track:abc"}}}
    monkeypatch.setattr(sp, "_play_uri_locally", Mock(side_effect=AssertionError("no desktop fallback")))
    candidate = Candidate("spotify:track:abc", "Madira", "Seedhe Maut")

    assert sp.play_candidate(candidate) == candidate.label
    sp.sp.start_playback.assert_called_once_with(uris=[candidate.uri], device_id="laptop")
    assert sp.sp.current_playback.call_count == 1
    sp._play_uri_locally.assert_not_called()
    assert sp.last_playback["method"] == "web_api_verified"
    assert sp.last_playback["native_confirmation"] == "missed"
    assert "web_api_skipped" not in sp.last_playback
    assert "Native confirmation missed accepted playback" in caplog.text


@pytest.mark.parametrize("state", [
    {"device": {"id": "laptop"}, "is_playing": True, "item": {"uri": "spotify:track:wrong"}},
    {"device": {"id": "remote"}, "is_playing": True, "item": {"uri": "spotify:track:abc"}},
    {"device": {"id": "laptop"}, "is_playing": False, "item": {"uri": "spotify:track:abc"}},
])
def test_accepted_api_wrong_uri_device_or_paused_never_retries_playback(monkeypatch, state):
    from contextlib import nullcontext
    from assistant.resolution import Candidate
    from assistant.spotify import PlaybackVerificationError

    sp = Spotify.__new__(Spotify)
    sp._device_name = "laptop"
    sp._cfg = SimpleNamespace(get=lambda key, default=None: default)
    sp.sp = Mock()
    sp.command_target = lambda: nullcontext()
    sp.device = lambda: "laptop"
    sp._now_playing = lambda candidate, timeout: False
    sp.sp.current_playback.return_value = state
    desktop = Mock(side_effect=AssertionError("accepted request cannot fall back"))
    monkeypatch.setattr(sp, "_play_uri_locally", desktop)
    monkeypatch.setattr("assistant.spotify.time.sleep", lambda _: None)

    with pytest.raises(PlaybackVerificationError):
        sp.play_candidate(Candidate("spotify:track:abc", "Madira", "Seedhe Maut"))
    sp.sp.start_playback.assert_called_once()
    assert sp.sp.current_playback.call_count == 4
    desktop.assert_not_called()


@pytest.mark.parametrize("failure", ["rate_limited", "deferred"])
def test_accepted_api_verification_failure_never_falls_back(monkeypatch, failure, caplog):
    from contextlib import nullcontext
    import logging
    from assistant.resolution import Candidate
    from assistant.spotify import SpotifyRateLimited, SpotifyRequestDeferred

    logger = logging.getLogger("memoir.spotify")
    monkeypatch.setattr(logger, "handlers", [*logger.handlers, caplog.handler])
    monkeypatch.setattr(logger, "propagate", False)
    caplog.set_level("WARNING", logger="memoir.spotify")
    sp = Spotify.__new__(Spotify)
    sp._device_name = "laptop"
    sp._cfg = SimpleNamespace(get=lambda key, default=None: default)
    sp.sp = Mock()
    sp.command_target = lambda: nullcontext()
    sp.device = lambda: "laptop"
    sp._now_playing = lambda candidate, timeout: False
    error = (SpotifyRateLimited("Bearer secret-token response-body") if failure == "rate_limited"
             else SpotifyRequestDeferred("budget secret-token response-body"))
    sp.sp.current_playback.side_effect = error
    desktop = Mock(side_effect=AssertionError("accepted request cannot fall back"))
    monkeypatch.setattr(sp, "_play_uri_locally", desktop)

    with pytest.raises(type(error)):
        sp.play_candidate(Candidate("spotify:track:abc", "Madira", "Seedhe Maut"))
    sp.sp.start_playback.assert_called_once()
    desktop.assert_not_called()
    assert "Could not verify accepted playback on device 'laptop' for track spotify:track:abc" in caplog.text
    assert "secret-token" not in caplog.text
    assert "response-body" not in caplog.text


def test_rejected_api_and_failed_desktop_fallback_propagates_without_second_mutation(monkeypatch):
    from contextlib import nullcontext
    from assistant.resolution import Candidate
    from assistant.spotify import SpotifyRateLimited

    sp = Spotify.__new__(Spotify)
    sp._cfg = SimpleNamespace(get=lambda key, default=None: default)
    sp.sp = Mock()
    error = SpotifyRateLimited("banned")
    sp.sp.start_playback.side_effect = error
    sp.command_target = lambda: nullcontext()
    sp.device = lambda: "laptop"
    desktop = Mock(return_value=False)
    monkeypatch.setattr(sp, "_play_uri_locally", desktop)

    with pytest.raises(SpotifyRateLimited) as caught:
        sp.play_candidate(Candidate("spotify:track:abc", "Madira", "Seedhe Maut"))
    assert caught.value is error
    sp.sp.start_playback.assert_called_once()
    desktop.assert_called_once()


@pytest.mark.parametrize("failure", ["rate_limited", "deferred"])
def test_rejected_api_can_still_succeed_through_desktop_fallback(monkeypatch, failure):
    from contextlib import nullcontext
    from assistant.resolution import Candidate
    from assistant.spotify import SpotifyRateLimited, SpotifyRequestDeferred

    sp = Spotify.__new__(Spotify)
    sp._cfg = SimpleNamespace(get=lambda key, default=None: default)
    sp.sp = Mock()
    error = SpotifyRateLimited("banned") if failure == "rate_limited" else SpotifyRequestDeferred("budget")
    sp.sp.start_playback.side_effect = error
    sp.command_target = lambda: nullcontext()
    sp.device = lambda: "laptop"
    monkeypatch.setattr(sp, "_play_uri_locally", Mock(return_value=True))
    candidate = Candidate("spotify:track:abc", "Madira", "Seedhe Maut")

    assert sp.play_candidate(candidate) == candidate.label
    sp.sp.start_playback.assert_called_once()
    assert sp.last_playback["method"] == "desktop_uri"
    assert sp.last_playback["radio_skipped"] == 0
    assert sp.last_playback["web_api_skipped"] in {"rate_limited", "local_request_deferred"}


@pytest.mark.parametrize(
    ("headers", "expected_wait"),
    [({}, 300), ({"retry-after": "16000"}, 16000),
     ({"rEtRy-AfTeR": "not-a-delay"}, 300)],
)
def test_rate_limit_honors_retry_after_or_uses_fallback(monkeypatch, headers, expected_wait):
    import spotipy
    from assistant.spotify import SpotifyRateLimited, _RateLimitGuard

    monkeypatch.setattr("assistant.spotify.time.time", lambda: 1000.0)
    client = Mock()
    client.devices.side_effect = spotipy.SpotifyException(429, -1, "limit", headers=headers)
    guard = _RateLimitGuard(client)

    with pytest.raises(SpotifyRateLimited):
        guard.devices()

    assert guard.blocked_until == 1000.0 + expected_wait
    client.devices.assert_called_once()


def test_rate_limit_parses_http_date_retry_after(monkeypatch):
    from datetime import datetime, timezone
    from email.utils import format_datetime
    import spotipy
    from assistant.spotify import SpotifyRateLimited, _RateLimitGuard

    monkeypatch.setattr("assistant.spotify.time.time", lambda: 1000.0)
    retry_date = format_datetime(datetime.fromtimestamp(2000, timezone.utc), usegmt=True)
    client = Mock()
    client.devices.side_effect = spotipy.SpotifyException(
        429, -1, "limit", headers={"Retry-AfTeR": retry_date})
    guard = _RateLimitGuard(client)

    with pytest.raises(SpotifyRateLimited):
        guard.devices()

    assert guard.blocked_until == 2000.0


@pytest.mark.parametrize(
    "snapshot",
    [
        None,
        SimpleNamespace(title="Madira (Live)", artist="Seedhe Maut", playing=True),
        SimpleNamespace(title="Madira", artist="Other Artist", playing=True),
        SimpleNamespace(title="Madira", artist="Seedhe Maut", playing=False),
        SimpleNamespace(title="Madira", artist="", playing=True),
    ],
)
def test_now_playing_requires_exact_title_artist_and_playing(monkeypatch, snapshot):
    from assistant import local_media
    from assistant.resolution import Candidate

    sp = Spotify.__new__(Spotify)
    monkeypatch.setattr(local_media, "snapshot", lambda: snapshot)
    monkeypatch.setattr(local_media, "_wait_for", lambda predicate, timeout: predicate())

    assert not sp._now_playing(Candidate("spotify:track:abc", "Madira", "Seedhe Maut"), 1)


def test_now_playing_accepts_exact_playing_snapshot(monkeypatch):
    from assistant import local_media
    from assistant.resolution import Candidate

    sp = Spotify.__new__(Spotify)
    monkeypatch.setattr(local_media, "snapshot", lambda: SimpleNamespace(
        title="MADIRA", artist="Seedhe Maut", playing=True))
    monkeypatch.setattr(local_media, "_wait_for", lambda predicate, timeout: predicate())

    assert sp._now_playing(Candidate("spotify:track:abc", "Madira", "Seedhe Maut"), 1)


@pytest.mark.parametrize(
    ("current", "play_result"),
    [
        (SimpleNamespace(title="Madira", artist="Seedhe Maut", playing=False), True),
        (SimpleNamespace(title="Madira", artist="Other Artist", playing=True), False),
    ],
)
def test_local_playback_does_not_accept_paused_or_wrong_artist(monkeypatch, current, play_result):
    from contextlib import nullcontext
    import threading
    from assistant import local_media
    from assistant.resolution import Candidate

    sp = Spotify.__new__(Spotify)
    sp._local_enabled = lambda: True
    sp._client_running = Mock(return_value=True)
    opened = []
    monkeypatch.setattr("assistant.spotify.os.startfile", opened.append)
    monkeypatch.setattr(local_media, "snapshot", lambda: current)
    monkeypatch.setattr(local_media, "play", Mock(return_value=play_result))
    monkeypatch.setattr(local_media, "pause", Mock(return_value=True))
    monkeypatch.setattr(local_media, "_wait_for", lambda predicate, timeout: predicate())
    monkeypatch.setattr(local_media, "InvisibleSpotify", lambda: nullcontext(
        SimpleNamespace(return_focus=Mock())))
    monkeypatch.setattr(threading, "Thread", lambda **kwargs: SimpleNamespace(start=Mock()))

    candidate = Candidate("spotify:track:abc", "Madira", "Seedhe Maut")
    assert not sp._play_uri_locally(candidate)
    assert opened == [candidate.uri]


def test_cached_track_requires_exact_requested_title_and_artist():
    from assistant.resolution import Candidate, _norm

    sp = Spotify.__new__(Spotify)
    sp._tracks = {
        _norm("Red by Seedhe Maut"): {
            "uri": "spotify:track:abc", "name": "Red", "artist": "Seedhe Maut"},
        _norm("Red from Seedhemov"): {
            "uri": "spotify:track:wrong", "name": "Red", "artist": "Seedhe Maut"},
    }

    assert sp.cached_track("Red by Seedhe Maut") == Candidate(
        "spotify:track:abc", "Red", "Seedhe Maut")
    assert sp.cached_track("Red from Seedhemov") is None
    assert sp.cached_track("Red by Seedhemov") is None


def test_rate_limit_ban_is_shared_through_the_backoff_file(tmp_path):
    import json
    from assistant.spotify import _RateLimitGuard, SpotifyRateLimited
    path = tmp_path / "backoff.json"
    path.write_text(json.dumps({"player": 0, "web": time.time() + 3600}), encoding="utf-8")
    client = Mock()
    guard = _RateLimitGuard(client, path)
    with pytest.raises(SpotifyRateLimited):
        guard.search(q="x")
    client.search.assert_not_called()
    guard.current_playback()
    client.current_playback.assert_called_once()


def test_rate_limit_ban_is_saved_for_the_next_process(tmp_path):
    import json
    import spotipy
    from assistant.spotify import _RateLimitGuard, SpotifyRateLimited
    path = tmp_path / "backoff.json"
    client = Mock()
    client.search.side_effect = spotipy.SpotifyException(429, -1, "limited", headers={"Retry-After": "86000"})
    with pytest.raises(SpotifyRateLimited):
        _RateLimitGuard(client, path).search(q="x")
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["web"] > time.time() + 85000
    with pytest.raises(SpotifyRateLimited):
        _RateLimitGuard(Mock(), path).search(q="x")


def test_quota_exceeded_with_retry_after_is_distinct_and_expires(monkeypatch, tmp_path):
    import spotipy
    from assistant.spotify import SpotifyQuotaExceeded, _RateLimitGuard
    now = [1000.0]
    monkeypatch.setattr("assistant.spotify.time.time", lambda: now[0])
    path = tmp_path / "backoff.json"
    client = Mock()
    client.search.side_effect = spotipy.SpotifyException(
        429, -1, "private URL and body", reason="QUOTA_EXCEEDED", headers={"Retry-After": "30"})
    guard = _RateLimitGuard(client, path)
    with pytest.raises(SpotifyQuotaExceeded, match="Spotify API quota is exhausted; local controls may still work"):
        guard.search(q="x")
    assert guard.quota_status() == {"web": {"reason": "QUOTA_EXCEEDED", "retry_until": 1030.0}}
    with pytest.raises(SpotifyQuotaExceeded):
        guard.search(q="x")
    assert client.search.call_count == 1
    now[0] = 1030.0
    assert _RateLimitGuard(Mock(), path).quota_status() == {}


def test_quota_without_valid_retry_after_persists_until_explicit_clear(tmp_path):
    import json
    import spotipy
    from assistant.spotify import SpotifyQuotaExceeded, _RateLimitGuard
    path = tmp_path / "backoff.json"
    client = Mock()
    client.search.side_effect = spotipy.SpotifyException(
        429, -1, "private URL and body", reason="QUOTA_EXCEEDED", headers={})
    with pytest.raises(SpotifyQuotaExceeded):
        _RateLimitGuard(client, path).search(q="x")
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["quota"]["web"] == {"reason": "QUOTA_EXCEEDED", "retry_until": None}
    resumed = _RateLimitGuard(Mock(), path)
    assert resumed.quota_status() == {"web": {"reason": "QUOTA_EXCEEDED", "retry_until": None}}
    with pytest.raises(SpotifyQuotaExceeded):
        resumed.search(q="x")
    resumed.clear_quota("web")
    resumed.search(q="x")
    with pytest.raises(ValueError):
        resumed.clear_quota("unknown")


@pytest.mark.parametrize("bad_until", ["tomorrow", float("nan"), float("inf"), True])
def test_corrupt_recognized_quota_expiry_fails_closed(tmp_path, bad_until):
    import json
    from assistant.spotify import SpotifyQuotaExceeded, _RateLimitGuard
    path = tmp_path / "backoff.json"
    path.write_text(json.dumps({"player": 0, "web": 0, "quota": {
        "web": {"reason": "QUOTA_EXCEEDED", "retry_until": bad_until}}}), encoding="utf-8")
    client = Mock()
    guard = _RateLimitGuard(client, path)
    assert guard.quota_status()["web"]["retry_until"] is None
    with pytest.raises(SpotifyQuotaExceeded):
        guard.search(q="x")
    client.search.assert_not_called()


def test_missing_recognized_quota_expiry_fails_closed(tmp_path):
    import json
    from assistant.spotify import SpotifyQuotaExceeded, _RateLimitGuard
    path = tmp_path / "backoff.json"
    path.write_text(json.dumps({"player": 0, "web": 0,
                                "quota": {"web": {"reason": "QUOTA_EXCEEDED"}}}),
                    encoding="utf-8")
    client = Mock()
    guard = _RateLimitGuard(client, path)
    assert guard.quota_status()["web"]["retry_until"] is None
    with pytest.raises(SpotifyQuotaExceeded):
        guard.search(q="x")
    client.search.assert_not_called()


def test_unknown_quota_reason_and_scope_are_ignored(tmp_path):
    import json
    from assistant.spotify import _RateLimitGuard
    path = tmp_path / "backoff.json"
    path.write_text(json.dumps({"player": 0, "web": 0, "quota": {
        "web": {"reason": "RATE_LIMIT", "retry_until": None},
        "other": {"reason": "QUOTA_EXCEEDED", "retry_until": None}}}), encoding="utf-8")
    guard = _RateLimitGuard(Mock(), path)
    assert guard.quota_status() == {}


def test_malformed_revision_cannot_clear_known_quota_state(tmp_path):
    import json
    import spotipy
    from assistant.spotify import SpotifyQuotaExceeded, _RateLimitGuard
    path = tmp_path / "backoff.json"
    first, observer = _RateLimitGuard(Mock(), path), _RateLimitGuard(Mock(), path)
    first._client.search.side_effect = spotipy.SpotifyException(
        429, -1, "limited", reason="QUOTA_EXCEEDED", headers={})
    with pytest.raises(SpotifyQuotaExceeded):
        first.search(q="x")
    assert observer.quota_status()["web"]["reason"] == "QUOTA_EXCEEDED"
    path.write_text(json.dumps({"player": 0, "web": 0, "quota": {},
                                "quota_revision": {"player": 0, "web": "broken"}}),
                    encoding="utf-8")
    observer._seen_mtime = None
    assert observer.quota_status()["web"]["retry_until"] is None
    with pytest.raises(SpotifyQuotaExceeded):
        observer.search(q="x")


def test_unknown_reason_429_keeps_retry_after_cooldown_and_legacy_schema(tmp_path):
    import json
    import spotipy
    from assistant.spotify import SpotifyRateLimited, _RateLimitGuard
    path = tmp_path / "backoff.json"
    client = Mock()
    client.search.side_effect = spotipy.SpotifyException(
        429, -1, "limit", reason="SOMETHING_ELSE", headers={"Retry-After": "12"})
    guard = _RateLimitGuard(client, path)
    with pytest.raises(SpotifyRateLimited) as caught:
        guard.search(q="x")
    assert type(caught.value) is SpotifyRateLimited
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["web"] > time.time() + 10
    assert saved["quota"] == {}
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps({"player": 0, "web": time.time() + 60}), encoding="utf-8")
    legacy_guard = _RateLimitGuard(Mock(), legacy)
    assert "web" in legacy_guard.ban_status()
    assert legacy_guard.quota_revision == {"player": 0, "web": 0}


def test_quota_marker_on_non_429_does_not_become_quota_exception():
    import spotipy
    from assistant.spotify import SpotifyQuotaExceeded, _RateLimitGuard
    client = Mock()
    client.search.side_effect = spotipy.SpotifyException(
        403, -1, "forbidden", reason="QUOTA_EXCEEDED", headers={})
    with pytest.raises(spotipy.SpotifyException) as caught:
        _RateLimitGuard(client).search(q="x")
    assert not isinstance(caught.value, SpotifyQuotaExceeded)


def test_quota_metadata_survives_other_guard_write(tmp_path):
    import json
    import spotipy
    from assistant.spotify import SpotifyQuotaExceeded, SpotifyRateLimited, _RateLimitGuard
    path = tmp_path / "backoff.json"
    first = _RateLimitGuard(Mock(), path)
    second = _RateLimitGuard(Mock(), path)
    first._client.search.side_effect = spotipy.SpotifyException(
        429, -1, "limited", reason="QUOTA_EXCEEDED", headers={})
    with pytest.raises(SpotifyQuotaExceeded):
        first.search(q="x")
    second._client.current_playback.side_effect = spotipy.SpotifyException(
        429, -1, "limited", headers={"Retry-After": "30"})
    with pytest.raises(SpotifyRateLimited):
        second.current_playback()
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["quota"]["web"]["retry_until"] is None


def test_quota_clear_propagates_to_guard_with_stale_cached_quota(tmp_path):
    import json
    import spotipy
    from assistant.spotify import SpotifyQuotaExceeded, _RateLimitGuard
    path = tmp_path / "backoff.json"
    first, second = _RateLimitGuard(Mock(), path), _RateLimitGuard(Mock(), path)
    first._client.search.side_effect = spotipy.SpotifyException(
        429, -1, "limited", reason="QUOTA_EXCEEDED", headers={})
    with pytest.raises(SpotifyQuotaExceeded):
        first.search(q="x")
    assert second.quota_status()["web"]["reason"] == "QUOTA_EXCEEDED"
    assert first.quota_revision["web"] == second.quota_revision["web"] == 1
    second.clear_quota("web")
    assert second.quota_revision["web"] == 2
    assert first.quota_status() == {}
    first._client.search.side_effect = None
    first.search(q="x")
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["quota"] == {} and saved["quota_revision"]["web"] == 2


def test_stale_other_scope_save_does_not_resurrect_cleared_quota(tmp_path):
    import json
    import spotipy
    from assistant.spotify import SpotifyQuotaExceeded, _RateLimitGuard
    path = tmp_path / "backoff.json"
    stale, operator = _RateLimitGuard(Mock(), path), _RateLimitGuard(Mock(), path)
    stale._client.search.side_effect = spotipy.SpotifyException(
        429, -1, "limited", reason="QUOTA_EXCEEDED", headers={})
    with pytest.raises(SpotifyQuotaExceeded):
        stale.search(q="x")
    operator.clear_quota("web")
    stale.blocked["player"] = time.time() + 30
    assert stale._save() is True
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["quota"] == {} and saved["quota_revision"]["web"] == 2
    assert _RateLimitGuard(Mock(), path).quota_status() == {}


def test_new_quota_after_explicit_clear_is_persisted_at_next_revision(tmp_path):
    import json
    import spotipy
    from assistant.spotify import SpotifyQuotaExceeded, _RateLimitGuard
    path = tmp_path / "backoff.json"
    guard = _RateLimitGuard(Mock(), path)
    guard._client.search.side_effect = spotipy.SpotifyException(
        429, -1, "limited", reason="QUOTA_EXCEEDED", headers={})
    with pytest.raises(SpotifyQuotaExceeded):
        guard.search(q="x")
    guard.clear_quota("web")
    guard._client.search.side_effect = spotipy.SpotifyException(
        429, -1, "limited again", reason="QUOTA_EXCEEDED", headers={})
    with pytest.raises(SpotifyQuotaExceeded):
        guard.search(q="x")
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["quota_revision"]["web"] == 3
    assert saved["quota"]["web"]["retry_until"] is None


def test_local_budget_deferral_remains_distinct_from_quota():
    from assistant.spotify import SpotifyQuotaExceeded, SpotifyRequestDeferred, _RateLimitGuard, _RequestBudget
    client = Mock()
    guard = _RateLimitGuard(client, budget=_RequestBudget(total=0), foreground_wait=0)
    with pytest.raises(SpotifyRequestDeferred):
        guard.search(q="x")
    assert not isinstance(SpotifyRequestDeferred("budget"), SpotifyQuotaExceeded)
    client.search.assert_not_called()


def test_quota_quarantine_counts_as_existing_rate_limited_compatibility_state():
    import spotipy
    from assistant.spotify import Spotify, SpotifyQuotaExceeded, _RateLimitGuard
    client = Mock()
    client.devices.side_effect = spotipy.SpotifyException(
        429, -1, "limited", reason="QUOTA_EXCEEDED", headers={})
    guard = _RateLimitGuard(client)
    with pytest.raises(SpotifyQuotaExceeded):
        guard.devices()
    app = Spotify.__new__(Spotify)
    app.sp = guard
    assert app.rate_limited is True
    assert app.web_rate_limited is False
    client.search.side_effect = spotipy.SpotifyException(
        429, -1, "limited", reason="QUOTA_EXCEEDED", headers={})
    with pytest.raises(SpotifyQuotaExceeded):
        guard.search(q="x")
    assert app.web_rate_limited is True


def test_shared_injected_budget_caps_concurrent_guard_attempts():
    import threading
    from assistant.spotify import _RateLimitGuard, _RequestBudget, SpotifyRequestDeferred

    budget = _RequestBudget(clock=lambda: 100.0)
    clients = [Mock() for _ in range(24)]
    guards = [_RateLimitGuard(client, budget=budget, foreground_wait=0) for client in clients]
    barrier = threading.Barrier(len(guards))
    outcomes = []

    def call(guard):
        barrier.wait()
        try:
            guard.search(q="sensitive query")
            outcomes.append("sent")
        except SpotifyRequestDeferred:
            outcomes.append("deferred")

    workers = [threading.Thread(target=call, args=(guard,)) for guard in guards]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=2)
        assert not worker.is_alive()
    assert outcomes.count("sent") == 20
    assert outcomes.count("deferred") == 4
    assert sum(client.search.call_count for client in clients) == 20


def test_background_budget_reserve_and_recent_foreground_deferral():
    from assistant.spotify import (_RateLimitGuard, _RequestBudget,
                                   SpotifyRequestDeferred, request_context)

    now = [100.0]
    budget = _RequestBudget(clock=lambda: now[0])
    client = Mock()
    guard = _RateLimitGuard(client, budget=budget)
    with request_context("catalogue", background=True):
        for _ in range(15):
            guard.search(q="catalogue")
        with pytest.raises(SpotifyRequestDeferred, match="local"):
            guard.search(q="catalogue")
    assert client.search.call_count == 15

    priority_budget = _RequestBudget(clock=lambda: now[0])
    fg_client = Mock()
    fg = _RateLimitGuard(fg_client, budget=priority_budget, foreground_wait=0)
    fg.search(q="foreground")
    bg_client = Mock()
    bg = _RateLimitGuard(bg_client, budget=priority_budget)
    with request_context("catalogue", background=True):
        with pytest.raises(SpotifyRequestDeferred, match="recent foreground"):
            bg.search(q="catalogue")
    bg_client.search.assert_not_called()
    now[0] += 2.01
    with request_context("catalogue", background=True):
        bg.search(q="catalogue")
    bg_client.search.assert_called_once()


def test_request_context_restores_previous_values_after_exception():
    from assistant.spotify import _request_context, request_context

    with request_context("assistant", background=False):
        with pytest.raises(ValueError):
            with request_context("catalogue", background=True):
                assert _request_context.value == ("catalogue", True)
                raise ValueError("expected")
        assert _request_context.value == ("assistant", False)
    assert not hasattr(_request_context, "value")


def test_spotify_request_logs_never_include_query_arguments_or_exception_text(caplog, monkeypatch):
    import logging
    from assistant.spotify import _RateLimitGuard, _RequestBudget, request_context

    logger = logging.getLogger("memoir.spotify")
    monkeypatch.setattr(logger, "handlers", [*logger.handlers, caplog.handler])
    monkeypatch.setattr(logger, "propagate", False)

    secret_query = "private track query 8f73"
    private_error = "private response body 1a2b"
    client = Mock()
    client.search.side_effect = RuntimeError(private_error)
    guard = _RateLimitGuard(client, budget=_RequestBudget(clock=lambda: 1.0))
    with caplog.at_level(logging.INFO, logger="memoir.spotify"):
        with request_context("catalogue", background=True):
            with pytest.raises(RuntimeError):
                guard.search(q=secret_query, access_token="secret-token")
    output = "\n".join(record.getMessage() for record in caplog.records)
    assert "method=search" in output and "source=catalogue" in output
    assert "mode=background" in output and "RuntimeError" in output
    assert secret_query not in output and private_error not in output and "secret-token" not in output
