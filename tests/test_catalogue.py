"""Catalogue-grounded resolution: phonetic matching, building, integration.

Everything is offline: a hand-made index stands in for the user's Spotify
world, and the spotipy client is faked. The heard strings are real recogniser
output for tracks the user confirmed.
"""
import json
import os
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import spotipy

from assistant import catalogue
from assistant.catalogue import Index, best, forms_for, phonetic_key
from assistant.resolution import Candidate
from assistant.spotify import Spotify

SM = ["Seedhe Maut"]


def e(name, artists, source="playlist", album=""):
    return {"uri": "spotify:track:" + name.replace(" ", "_") + "_" + artists[0].replace(" ", "_"),
            "name": name, "artists": list(artists), "album": album, "sources": [source]}


FIXTURE = [
    # Seedhe Maut: the real neighbours that make this hard (similar names).
    e("Madira", ["Seedhe Maut", "Hurricane"]), e("Naksha", SM), e("RED", SM + ["Hisab"]),
    e("Nanchaku", SM), e("Natkhat", SM, "artist"), e("Khush Nahi", SM), e("Shaayar", SM),
    e("Maar Kaat", SM + ["Hurricane"]), e("Meri Baggi", SM), e("Na Jaaye", SM),
    e("Round 3", SM), e("Nayaab", SM), e("Nazarbhattu Freestyle", SM, "artist"),
    e("Uss Din", SM),
    e("3 Cheez", ["The Siege", "Natiq"]), e("Shayad Maut", ["The Siege"]),
    e("RED CUP", ["Calm", "DL91 Era"]), e("Mudda", ["Yung Sammy"], "artist"),
    e("Bhussi", ["KSHMR", "Seedhe Maut"]), e("Fark", ["Farak"]), e("Khwab", ["Farak"], "artist"),
    # The rest of a plausible library.
    e("Creep", ["Radiohead"]), e("Bohemian Rhapsody", ["Queen"]),
    e("Champagne Poetry", ["Drake"], "artist"), e("Stronger", ["Kanye West"]),
    e("Stronger", ["Kelly Clarkson"], "artist"), e("Runaway", ["Kanye West"]),
    e("Shield", ["Pink Floyd"], "artist"), e("Houdini", ["Karan Kanchan", "Rawal"]),
    e("Thank U, Next", ["Ariana Grande"], "artist"), e("Tinka", ["Bharat Chauhan"]),
    e("Boss Life", ["Rawal"], "artist"), e("Yellow", ["Coldplay"]), e("Hello", ["Adele"]),
    e("Hello", ["Lionel Richie"]), e("Midnight Thoughts", ["Prateek Kuhad"]),
    e("Kasoor", ["Prateek Kuhad"]), e("Tum Hi Ho", ["Arijit Singh"]),
    e("Channa Mereya", ["Arijit Singh"]), e("Ghost", ["Joji"], "artist"),
]


@pytest.fixture(scope="module")
def index():
    return Index({"tracks": FIXTURE, "built_at": time.time(), "complete": True})


def resolve(index, heard):
    hit = best(heard, index)
    return None if hit is None else (hit[0]["name"], hit[0]["artists"][0])


def flats(text):
    return {f.flat for f in forms_for(text)}


def test_unqualified_song_uses_cached_remaster_without_losing_edition():
    rows = [e("Yesterday - Remastered 2009", ["The Beatles"]),
            e("Yesterday (2015 Remaster)", ["The Beatles"]),
            e("Yesterday - Take 1", ["The Beatles"]),
            e("Yesterday - 2023 Mix", ["The Beatles"]),
            e("Blackbird / Yesterday", ["The Beatles"])]
    idx = Index({"tracks": rows})
    for query in ("Yesterday", "Yesterday by the Beatles"):
        assert best(query, idx)[0] is rows[0]
    assert best("Yesterday 2015 Remaster", idx)[0] is rows[1]
    assert rows[0]["name"] == "Yesterday - Remastered 2009"


def test_remaster_alias_does_not_merge_distinct_artists_or_performances():
    rows = [e("Yesterday - Remastered 2009", ["The Beatles"]),
            e("Yesterday (2015 Remaster)", ["Another Artist"])]
    assert best("Yesterday", Index({"tracks": rows})) is None
    live = e("Yesterday - Live", ["The Beatles"])
    idx = Index({"tracks": [rows[0], live]})
    assert best("Yesterday Live by the Beatles", idx)[0] is live


@pytest.mark.parametrize("qualifier", ["live", "acoustic", "instrumental", "remix", "take 1", "2023 mix"])
def test_recording_qualifier_cannot_be_consumed_as_filler(qualifier):
    studio = e("Yesterday - Remastered 2009", ["The Beatles"])
    assert best(f"Yesterday {qualifier} by the Beatles", Index({"tracks": [studio]})) is None


def test_artist_named_live_does_not_require_a_live_recording():
    entry = e("Lightning Crashes", ["Live"])
    assert best("Lightning Crashes by Live", Index({"tracks": [entry]}))[0] is entry


def test_remaster_identity_keeps_non_latin_artists_distinct():
    entries = [e("Song - Remastered 2009", ["अजय"]),
               e("Song (2015 Remaster)", ["विजय"])]
    assert best("Song", Index({"tracks": entries})) is None


# ------------------------------------------------------------------ phonetics

class TestPhonetics:
    def test_digraphs_and_vowels_fold(self):
        assert phonetic_key("Seedhe") == "sid"
        assert phonetic_key("cheese") == phonetic_key("Cheez")
        assert phonetic_key("phone") == phonetic_key("fone")
        assert phonetic_key("Naksha") == "naksa"

    def test_accents_punctuation_and_doubles(self):
        assert phonetic_key("Beyoncé") == phonetic_key("beyonce")
        assert phonetic_key("A.R. Rahman") == phonetic_key("a r rahman")
        assert phonetic_key("Hello!!") == phonetic_key("helo")

    def test_ch_stays_distinct_from_k(self):
        assert phonetic_key("chat") != phonetic_key("kat")

    def test_digits_expand_to_english_and_hindi(self):
        assert phonetic_key("teen cheez").replace(" ", "") in flats("3 Cheez")
        assert phonetic_key("three cheez").replace(" ", "") in flats("3 Cheez")
        assert phonetic_key("do saal").replace(" ", "") in flats("2 Saal")
        assert phonetic_key("two saal").replace(" ", "") in flats("2 Saal")

    @pytest.mark.parametrize("digit,hindi,english", [
        ("0", "shunya", "zero"), ("1", "ek", "one"), ("4", "char", "four"),
        ("5", "paanch", "five"), ("6", "chhe", "six"), ("7", "saat", "seven"),
        ("8", "aath", "eight"), ("9", "nau", "nine"), ("10", "das", "ten"),
    ])
    def test_every_numeral_has_both_readings(self, digit, hindi, english):
        got = flats(digit + " raat")
        assert phonetic_key(hindi + " raat").replace(" ", "") in got
        assert phonetic_key(english + " raat").replace(" ", "") in got

    def test_two_digit_numbers_are_spelt_in_english(self):
        assert phonetic_key("twenty one savage").replace(" ", "") in flats("21 Savage")


# ---------------------------------------------------- real mangled transcripts

class TestPositives:
    @pytest.mark.parametrize("heard", [
        "madira by see the mode", "madira by see them all", "madira with sidemotri",
        "madeira by seedemoth",
    ])
    def test_madira(self, index, heard):
        assert resolve(index, heard) == ("Madira", "Seedhe Maut")

    @pytest.mark.parametrize("heard", [
        "nakshawa is idemot", "nakhshab i seedhe maut", "nachshavai by seedhe maut",
    ])
    def test_naksha(self, index, heard):
        assert resolve(index, heard) == ("Naksha", "Seedhe Maut")

    @pytest.mark.parametrize("heard", ["red from seedemoth", "red by sidaimoth"])
    def test_red(self, index, heard):
        assert resolve(index, heard) == ("RED", "Seedhe Maut")

    @pytest.mark.parametrize("heard", [
        "thin cheese by the seat", "teen cheese by cg", "3 cheez by the siege",
    ])
    def test_teen_cheez(self, index, heard):
        assert resolve(index, heard) == ("3 Cheez", "The Siege")

    def test_numeral_spoken_in_hindi_or_english_matches_digit_title(self, index):
        assert resolve(index, "teen cheez") == ("3 Cheez", "The Siege")
        assert resolve(index, "three cheese by the siege") == ("3 Cheez", "The Siege")

    @pytest.mark.parametrize("heard", [
        "creep by radiohead", "champagne poetry by drake", "bohemian rhapsody",
    ])
    def test_exact_english_requests_resolve(self, index, heard):
        assert resolve(index, heard) is not None

    @pytest.mark.parametrize("connector", ["by", "from", "with", "of", "is", "i", "buy",
                                           "bye", "bai", "mi", "the", ""])
    def test_any_one_word_connector_is_tolerated(self, index, connector):
        heard = " ".join(w for w in ["red", connector, "seedhe maut"] if w)
        assert resolve(index, heard) == ("RED", "Seedhe Maut")

    def test_artist_decides_between_same_titles(self, index):
        assert resolve(index, "stronger by kelly clarkson") == ("Stronger", "Kelly Clarkson")
        assert resolve(index, "stronger by kanye west") == ("Stronger", "Kanye West")
        assert resolve(index, "hello by lionel richie") == ("Hello", "Lionel Richie")

    def test_featured_artist_counts_as_artist(self, index):
        assert resolve(index, "madira by hurricane") == ("Madira", "Seedhe Maut")


# ------------------------------------------------------------------ negatives

class TestNegatives:
    @pytest.mark.parametrize("heard", [
        "shield up shield up shield up", "boss", "james song to something played by kanye west",
        "change strong to me and your ghost", "thank you", "oh no no no", "for khair",
    ])
    def test_observed_noise_is_not_resolved(self, index, heard):
        assert resolve(index, heard) is None

    @pytest.mark.parametrize("heard", [
        "blinding lights by the weeknd", "shape of you by ed sheeran",
        "no role modelz by j cole", "naksha by drake", "madira by taylor swift",
        "red by taylor swift", "levitating by dua lipa", "kesariya by arijit singh",
        "sunflower by post malone", "tum hi ho by pritam", "channa by seedhe maut",
        "creep by tlc", "bohemian like you", "good morning", "play something nice",
        "what time is it", "stairway to heaven",
    ])
    def test_plausible_absent_songs_fall_through(self, index, heard):
        assert resolve(index, heard) is None

    def test_wrong_artist_is_never_a_match_for_a_known_title(self, index):
        # "Creep" exists, but not by this artist: search should decide.
        assert resolve(index, "creep by metallica") is None

    def test_empty_and_overlong_input(self, index):
        assert best("", index) is None
        assert best("a " * 30, index) is None
        assert best("anything", None) is None

    def test_near_tie_between_different_songs_is_withheld(self):
        twin = Index({"tracks": [e("Kala Chashma", ["Badshah"]),
                                 e("Kala Chasma", ["Amar Arshi"])]})
        # Same-sounding titles, no artist given: nothing separates them.
        assert best("kala chashma", twin) is None

    def test_reissue_copies_are_not_a_competing_song(self):
        reissue = Index({"tracks": [e("Naksha", SM), dict(e("Naksha", SM, "artist"),
                                                          uri="spotify:track:other")]})
        assert best("naksha by seedhe maut", reissue)[0]["name"] == "Naksha"

    def test_matching_is_fast_enough_for_a_large_index(self):
        # ~6000 tracks is a realistic upper size; this must not stall a command.
        big = [e("Song %s %d" % (w, i), ["Artist %d" % (i % 50)], "artist")
               for i, w in enumerate(["alpha", "beta", "gamma", "delta"] * 1500)]
        big_index = Index({"tracks": big + FIXTURE})
        started = time.perf_counter()
        assert resolve(big_index, "madira by see them all") == ("Madira", "Seedhe Maut")
        assert time.perf_counter() - started < 3.0


# ---------------------------------------------------------------------- build

class FakeSpotify:
    """Just the GET endpoints build() may touch, with spotipy's paging shape."""

    def __init__(self, limited=False):
        self.calls = []
        self.limited = limited

    @staticmethod
    def page(items, limit, offset):
        return {"items": items[offset:offset + limit],
                "next": "x" if offset + limit < len(items) else None}

    @staticmethod
    def tr(n, name, artist="Seedhe Maut", aid="A1"):
        return {"uri": "spotify:track:%s" % n, "name": name, "is_local": False,
                "artists": [{"name": artist, "id": aid}]}

    def current_user_saved_tracks(self, limit, offset):
        self.calls.append("saved")
        raise spotipy.SpotifyException(403, -1, "Insufficient client scope")

    def current_user_top_tracks(self, limit, offset, time_range):
        self.calls.append("top")
        return self.page([self.tr("top1", "Naksha")], limit, offset)

    def current_user(self):
        return {"id": "me"}

    def current_user_playlists(self, limit, offset):
        mine = {"id": "p1", "owner": {"id": "me"}}
        theirs = {"id": "p2", "owner": {"id": "someone"}}
        return self.page([mine, theirs], limit, offset)

    def playlist_items(self, pid, limit, offset):
        self.calls.append("playlist:" + pid)
        rows = [{"item": self.tr("pl1", "Houdini", "Rawal", "A2")},
                {"item": {"uri": "spotify:episode:1", "name": "pod"}},
                {"item": self.tr("pl1", "Houdini", "Rawal", "A2")}]
        return self.page(rows, limit, offset)

    def search(self, q, type, limit):
        self.calls.append("search")
        return {"artists": {"items": [{"id": "A9", "name": "Nobody"}]}}

    def artist_albums(self, aid, limit, offset, include_groups):
        self.calls.append("albums:" + aid)
        assert limit <= 10          # Development-mode apps reject more
        albums = [{"id": "al1", "name": "Nayaab", "total_tracks": 2},
                  {"id": "al2", "name": "Nayaab (Deluxe)", "total_tracks": 3}]
        return self.page(albums, limit, offset)

    def album_tracks(self, aid, limit, offset):
        self.calls.append("tracks:" + aid)
        if self.limited and aid == "al2":
            raise spotipy.SpotifyException(429, -1, "slow down", headers={"Retry-After": "30"})
        rows = [self.tr(aid + "a", "Naksha"), self.tr(aid + "b", "RED")]
        return self.page(rows, limit, offset)


class TestBuild:
    def test_build_collects_library_and_discography(self):
        sp = FakeSpotify()
        data = catalogue.build(sp, ["Seedhe Maut", "Nobody At All"], delay=0,
                               include_discographies=True, max_requests=50)
        names = {(t["name"], t["artists"][0]) for t in data["tracks"]}
        assert ("Naksha", "Seedhe Maut") in names and ("Houdini", "Rawal") in names
        assert not any(t["name"] == "pod" for t in data["tracks"])      # episodes dropped
        assert data["complete"] is True
        # The artist id comes from the user's own tracks, not a (wrong-prone) search.
        assert "albums:A1" in sp.calls
        assert data["artists_skipped"] == {"Nobody At All": "not found"}
        # Followed playlists are unreadable in Development mode: never requested.
        assert "playlist:p2" not in sp.calls
        # Saved tracks needed a scope the token lacks: asked once, not fatal.
        assert sp.calls.count("saved") == 1

    def test_default_build_reads_library_without_artist_discographies(self):
        sp = FakeSpotify()
        def unused_artists():
            raise AssertionError("artist iterable consumed in library-only mode")
            yield "unreachable"
        catalogue.build(sp, unused_artists(), delay=0, max_requests=50)
        assert not any(call.startswith(("search", "albums:", "tracks:"))
                       for call in sp.calls)

    def test_dedupe_by_uri_and_reissue_and_sources_merge(self):
        rows = [dict(e("Naksha", SM, "artist"), uri="u1"),
                dict(e("Naksha", SM, "top"), uri="u2"),       # library copy wins
                dict(e("Naksha", SM, "playlist"), uri="u2"),  # same uri: merge sources
                e("Naksha - Live", SM, "artist")]
        out = catalogue.dedupe(rows)
        assert len(out) == 2
        naksha = next(r for r in out if r["name"] == "Naksha")
        assert naksha["uri"] == "u2" and set(naksha["sources"]) >= {"top", "playlist", "artist"}

    def test_429_stops_cleanly_and_is_resumable(self):
        sp = FakeSpotify(limited=True)
        data = catalogue.build(sp, ["Seedhe Maut"], delay=0,
                               include_discographies=True, max_requests=50)
        assert data["complete"] is False
        assert data["retry_not_before"] > time.time() + 3600
        assert data["artists_done"] == []
        assert sp.calls[-1] == "tracks:al2"      # not one request after the 429
        assert data["tracks"], "partial work is kept"
        resumed = catalogue.build(FakeSpotify(), ["Seedhe Maut"], delay=0, previous=data,
                                  include_discographies=True, max_requests=50)
        assert resumed["complete"] is True and resumed["artists_done"] == ["Seedhe Maut"]

    def test_rate_limit_guard_error_also_stops(self):
        from assistant.spotify import SpotifyRateLimited

        class Limited(FakeSpotify):
            def current_user_top_tracks(self, **kw):
                raise SpotifyRateLimited("blocked")

        data = catalogue.build(Limited(), ["Seedhe Maut"], delay=0)
        assert data["complete"] is False

    def test_requests_are_throttled(self):
        sp = FakeSpotify()
        started = time.monotonic()
        data = catalogue.build(sp, ["Seedhe Maut"], delay=0.02)
        assert time.monotonic() - started >= 0.02 * (data["requests"] - 1) * 0.9

    def test_save_is_atomic_and_load_roundtrips(self, tmp_path):
        path = tmp_path / "c.json"
        catalogue.save({"built_at": time.time(), "complete": True, "tracks": FIXTURE}, path)
        assert not list(tmp_path.glob("*.tmp"))
        loaded = catalogue.load(path)
        assert len(loaded) == len(FIXTURE) and loaded.complete
        assert catalogue.load(tmp_path / "missing.json") is None
        (tmp_path / "bad.json").write_text("{not json")
        assert catalogue.load(tmp_path / "bad.json") is None

    def test_refresh_only_when_missing_or_stale(self, tmp_path):
        path = tmp_path / "c.json"

        def write(age_hours, complete):
            catalogue.save({"built_at": time.time() - age_hours * 3600,
                            "complete": complete, "tracks": FIXTURE}, path)

        assert catalogue.needs_refresh(24, path)
        write(0, True)
        assert not catalogue.needs_refresh(24, path)
        write(25, True)
        assert catalogue.needs_refresh(24, path)
        write(0.5, False)
        assert not catalogue.needs_refresh(24, path)    # partial: retry hourly, not at once
        write(2, False)
        assert catalogue.needs_refresh(24, path)
        # After a 429 even a stale partial index waits out the backoff.
        catalogue.save({"built_at": time.time() - 2 * 3600, "complete": False,
                        "retry_not_before": time.time() + 3600, "tracks": FIXTURE}, path)
        assert not catalogue.needs_refresh(24, path)

    def test_refresh_in_background_skips_fresh_and_rate_limited(self, tmp_path):
        path = tmp_path / "c.json"
        catalogue.save({"built_at": time.time(), "complete": True, "tracks": FIXTURE}, path)
        assert catalogue.refresh_in_background(SimpleNamespace(), [], path=path,
                                               enabled=True) is None
        assert catalogue.refresh_in_background(
            SimpleNamespace(rate_limited=True), [], path=tmp_path / "none.json",
            enabled=True) is None

    def test_background_refresh_is_disabled_unless_explicitly_enabled(self, tmp_path):
        holder = SimpleNamespace(sp=Mock(side_effect=AssertionError("Spotify called")))
        assert catalogue.refresh_in_background(holder, ["artist"], path=tmp_path / "none.json") is None
        assert not (tmp_path / "none.json").exists()

    def test_bounded_library_build_preserves_complete_previous_index(self):
        previous = {"complete": True, "tracks": list(FIXTURE)}
        data = catalogue.build(FakeSpotify(), [], delay=0, previous=previous,
                               max_requests=1)
        assert data["complete"] is False
        assert data["stop_reason"] == "request_budget"
        assert data["requests"] == 1
        assert {t["uri"] for t in data["tracks"]} >= {t["uri"] for t in FIXTURE}
        assert "retry_not_before" not in data

    def test_local_defer_stops_without_rate_limit_cooldown(self):
        from assistant.spotify import SpotifyRequestDeferred

        class Deferred(FakeSpotify):
            def current_user_saved_tracks(self, **kwargs):
                raise SpotifyRequestDeferred("rolling budget")

        data = catalogue.build(Deferred(), [], delay=0, max_requests=10)
        assert data["stop_reason"] == "deferred"
        assert data["complete"] is False
        assert "retry_not_before" not in data

    def test_refresh_in_background_builds_and_hands_over_the_index(self, tmp_path, monkeypatch):
        path = tmp_path / "c.json"
        monkeypatch.setattr(catalogue, "PAGE_DELAY", 0)
        holder = SimpleNamespace(sp=FakeSpotify(), rate_limited=False, set_catalogue=Mock())
        thread = catalogue.refresh_in_background(holder, ["Seedhe Maut"], path=path,
                                                 enabled=True)
        assert thread is not None and thread.daemon
        thread.join(10)
        assert path.exists() and holder.set_catalogue.called
        assert holder.set_catalogue.call_args.kwargs.get("pin") is False
        # A second call while fresh does nothing.
        assert catalogue.refresh_in_background(holder, ["Seedhe Maut"], path=path,
                                               enabled=True) is None


# ----------------------------------------------------- Spotify.resolve_track

def demons_search():
    return Mock(return_value={"tracks": {"items": [
        {"name": "Demons", "uri": "spotify:track:demons",
         "artists": [{"name": "Imagine Dragons"}]}]}})


def spotify_with(index, search=None):
    sp = Spotify.__new__(Spotify)
    sp.sp = SimpleNamespace(search=search or Mock(side_effect=AssertionError("search called")))
    sp.set_catalogue(index)
    return sp


class TestResolveTrackIntegration:
    def test_catalogue_hit_returns_candidate_without_any_search(self, index):
        sp = spotify_with(index)
        cand = sp.resolve_track("madeira by seedemoth")
        assert isinstance(cand, Candidate)
        assert (cand.name, cand.artist) == ("Madira", "Seedhe Maut")
        assert cand.uri.startswith("spotify:track:")
        assert cand.label == "Madira by Seedhe Maut"
        assert cand.metadata["artists"][0]["name"] == "Seedhe Maut"
        sp.sp.search.assert_not_called()

    def test_low_confidence_falls_through_to_the_existing_path(self, index):
        search = demons_search()
        sp = spotify_with(index, search)
        assert sp.resolve_track("demons").uri == "spotify:track:demons"
        search.assert_called()

    def test_out_of_library_artist_request_uses_search(self, index):
        search = Mock(return_value={"tracks": {"items": [
            {"name": "Blinding Lights", "uri": "spotify:track:bl",
             "artists": [{"name": "The Weeknd"}]}]}})
        sp = spotify_with(index, search)
        assert sp.resolve_track("blinding lights by the weeknd").uri == "spotify:track:bl"

    def test_catalogue_failure_never_breaks_resolution(self, index, monkeypatch):
        sp = spotify_with(index, demons_search())
        monkeypatch.setattr(catalogue, "best", Mock(side_effect=RuntimeError("boom")))
        assert sp.resolve_track("demons").uri == "spotify:track:demons"

    def test_no_index_means_old_behaviour(self):
        sp = Spotify.__new__(Spotify)
        sp.sp = SimpleNamespace(search=demons_search())
        assert sp.resolve_track("demons").uri == "spotify:track:demons"

    def test_index_reloads_when_the_file_changes(self, tmp_path, monkeypatch):
        path = tmp_path / "idx.json"
        monkeypatch.setattr(catalogue, "INDEX_PATH", path)
        sp = Spotify.__new__(Spotify)
        sp.sp = SimpleNamespace(search=Mock(side_effect=AssertionError("search called")))
        catalogue.save({"built_at": time.time(), "tracks": [e("Naksha", SM)]}, path)
        assert sp.resolve_track("naksha by seedhe maut").name == "Naksha"
        catalogue.save({"built_at": time.time(), "tracks": [e("Madira", SM)]}, path)
        os.utime(path, (time.time() + 5, time.time() + 5))
        assert sp.resolve_track("madira by seedhe maut").name == "Madira"


def test_known_artists_reads_vocabulary_as_utf8(tmp_path):
    path = tmp_path / "v.json"
    path.write_text(json.dumps({"artists": ["Seedhe Maut", "Beyoncé", 7]}), encoding="utf-8")
    assert catalogue.known_artists(path) == ["Seedhe Maut", "Beyoncé"]
    assert catalogue.known_artists(tmp_path / "none.json") == []


def test_artist_request_uses_catalogue_while_search_is_rate_limited():
    from assistant import catalogue as cat
    from assistant.spotify import Spotify, SpotifyRateLimited
    index = cat.Index({"tracks": [
        {"uri": "spotify:track:a", "name": "Old", "artists": ["Seedhe Maut"], "sources": ["artist"]},
        {"uri": "spotify:track:b", "name": "Naksha", "artists": ["Seedhe Maut"], "sources": ["top"]},
        {"uri": "spotify:track:c", "name": "Other", "artists": ["Drake"], "sources": ["top"]},
    ]})
    sp = Spotify.__new__(Spotify)
    sp.sp = Mock()
    sp.sp.search.side_effect = SpotifyRateLimited("banned")
    sp.set_catalogue(index)
    candidate = sp.resolve_artist_track("Seedhe Maut")
    assert candidate.uri == "spotify:track:b"
