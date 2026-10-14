"""Recovering the title/artist boundary, and collapsing reissues.

Both come from verbatim production transcripts. Recognition reliably recovers
the artist name but loses the connector: "play Nakhsha by Seedhe Maut" was
transcribed "Play Nakhshab, I Seedhe Maut" (small.en) and "Play Nagshabay,
Seedhe Maut." (large-v3-turbo). Without the word "by", track resolution had no
artist and the catalogue fallback never ran.

Offline: no Spotify calls, no network.
"""
import pytest

from assistant.resolution import Candidate, collapse_reissues
from assistant.spotify import Spotify

ARTISTS = ["Seedhe Maut", "Imagine Dragons", "Queen", "Dhanji", "Arijit Singh"]


@pytest.fixture
def resolver():
    """A Spotify instance without constructing one (no auth, no network)."""
    sp = object.__new__(Spotify)
    sp.set_known_artists(ARTISTS)
    return sp


class TestArtistBoundary:
    @pytest.mark.parametrize("query,title,artist", [
        ("nakhshab i seedhe maut", "nakhshab", "Seedhe Maut"),
        ("nagshabay seedhe maut", "nagshabay", "Seedhe Maut"),
        ("demons imagine dragons", "demons", "Imagine Dragons"),
        ("bohemian rhapsody queen", "bohemian rhapsody", "Queen"),
    ])
    def test_boundary_found_without_the_word_by(self, resolver, query, title, artist):
        assert resolver._split_on_known_artist(query) == (title, artist)

    @pytest.mark.parametrize("query", [
        "bohemian rhapsody",          # no artist present at all
        "some song nobody made",      # tail matches no known artist
        "demons",                     # single word, nothing to split
    ])
    def test_no_boundary_invented_when_absent(self, resolver, query):
        """A weak suffix match would add an artist constraint that excludes the
        correct song outright, so the bar is deliberately high."""
        assert resolver._split_on_known_artist(query) == ("", "")

    def test_no_artists_known_is_safe(self):
        sp = object.__new__(Spotify)
        sp.set_known_artists([])
        assert sp._split_on_known_artist("nakhshab i seedhe maut") == ("", "")


class TestCollapseReissues:
    def test_same_song_on_two_releases_collapses(self):
        """Spotify lists album, single and remaster separately; asking the user
        to pick between two identical labels is an unanswerable question."""
        ranked = [
            Candidate("spotify:track:a", "Bohemian Rhapsody", "Queen"),
            Candidate("spotify:track:b", "Bohemian Rhapsody", "Queen"),
        ]
        assert len(collapse_reissues(ranked)) == 1

    def test_different_artists_are_kept(self):
        ranked = [
            Candidate("spotify:track:a", "Demons", "Imagine Dragons"),
            Candidate("spotify:track:b", "Demons", "Joji"),
        ]
        assert len(collapse_reissues(ranked)) == 2

    def test_best_ranked_survives(self):
        ranked = [
            Candidate("spotify:track:best", "RED", "Seedhe Maut"),
            Candidate("spotify:track:other", "RED", "Seedhe Maut"),
        ]
        assert collapse_reissues(ranked)[0].uri == "spotify:track:best"

    def test_empty_is_empty(self):
        assert collapse_reissues([]) == []
