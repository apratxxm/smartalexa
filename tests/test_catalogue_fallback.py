"""Artist-scoped title recovery, and the confidence gate that guards it.

Built from real recogniser output for the user-confirmed recording
"play Nakhsha by Seedhe Maut". Spotify spells the song "Naksha"; the recogniser
produced "Nakhshab" and "Nachshavai" on separate runs.

These are offline: no Spotify calls, no audio, no model.
"""
import pytest

from assistant.resolution import (Candidate, _clearly_best, _strong_match,
                                  rank_tracks)

# A slice of Seedhe Maut's real catalogue, in Spotify's own spelling.
CATALOGUE = ["Naksha", "Nanchaku", "Natkhat", "RED", "Khush Nahi", "Shaayar"]


def items(names, artist="Seedhe Maut"):
    return [{"uri": "spotify:track:{}".format(i), "name": n,
             "artists": [{"name": artist}]} for i, n in enumerate(names)]


class TestCatalogueRanking:
    @pytest.mark.parametrize("heard", ["Nakhshab", "Nakhsha", "Nachshavai"])
    def test_naksha_outranks_its_neighbours(self, heard):
        """Every observed mangling must put the intended song first.

        Absolute similarity cannot do this on its own: "Nakhshab"->"Natkhat"
        scores 0.667 while the correct "Nachshavai"->"Naksha" scores only 0.625.
        Ranking inside the artist's catalogue is what separates them.
        """
        ranked = rank_tracks(items(CATALOGUE), heard, "Seedhe Maut")
        assert ranked, "expected at least one candidate"
        assert ranked[0].name == "Naksha"

    def test_unrelated_title_matches_nothing(self):
        ranked = rank_tracks(items(CATALOGUE), "zzzznotarealsong", "Seedhe Maut")
        assert ranked == []

    def test_other_artists_are_never_returned(self):
        """A supplied artist is a hard constraint, not a preference."""
        mixed = items(["Naksha"], artist="Seedhe Maut") + items(
            ["Naksha"], artist="Some Other Act")
        ranked = rank_tracks(mixed, "Naksha", "Seedhe Maut")
        assert {c.artist for c in ranked} == {"Seedhe Maut"}


class TestConfidenceGate:
    """Uncertain matches must ask, never guess. A wrong action is the one
    outcome worse than no action."""

    def test_clear_winner_plays(self):
        ranked = rank_tracks(items(CATALOGUE), "Nakhshab", "Seedhe Maut")
        assert _clearly_best("Nakhshab", ranked) is True

    def test_near_tie_is_withheld(self):
        """"Nachshavai" sits 0.069 from Naksha and 0.556 from Nanchaku -- too
        close to act on. This exact case silently played the wrong song."""
        ranked = rank_tracks(items(CATALOGUE), "Nachshavai", "Seedhe Maut")
        assert ranked[0].name == "Naksha"
        assert _clearly_best("Nachshavai", ranked) is False

    def test_empty_is_not_confident(self):
        assert _clearly_best("anything", []) is False

    def test_single_candidate_needs_real_similarity(self):
        lonely = [Candidate("spotify:track:x", "Nanchaku", "Seedhe Maut")]
        assert _clearly_best("Nachshavai", lonely) is False
        good = [Candidate("spotify:track:y", "Naksha", "Seedhe Maut")]
        assert _clearly_best("Nakhsha", good) is True


class TestStrongMatch:
    """Decides whether the artist's catalogue is consulted at all, so a verbatim
    title stays fast and only uncertain ones pay the extra lookups."""

    @pytest.mark.parametrize("heard,name", [
        ("RED", "RED"), ("demons", "Demons"), ("dream on", "Dream On"),
    ])
    def test_verbatim_titles_skip_the_lookup(self, heard, name):
        assert _strong_match(heard, Candidate("u", name, "x")) is True

    @pytest.mark.parametrize("heard,name", [
        ("Nachshavai", "Nanchaku"), ("Nakhshab", "Natkhat"),
        ("three-mon", "Dream On"),
    ])
    def test_mangled_titles_trigger_the_lookup(self, heard, name):
        assert _strong_match(heard, Candidate("u", name, "x")) is False
