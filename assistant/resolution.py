"""Deterministic Spotify entity matching (no playback side effects)."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher


def _norm(value: str) -> str:
    # Keep letters from all scripts; ASCII-only normalization made Hindi and
    # accented names impossible to match. Transliteration is deliberately not
    # guessed here: Spotify search already handles the original spelling.
    value = unicodedata.normalize("NFKC", value or "").casefold()
    return "".join(ch if ch.isalnum() else " " for ch in value).strip()


_REMASTER = r"(?:(?:19|20)\d{2}\s+)?remaster(?:ed)?(?:\s+(?:19|20)\d{2})?"
_REMASTER_SUFFIX = re.compile(
    rf"\s*(?:[-–—]\s*{_REMASTER}|\({_REMASTER}\)|\[{_REMASTER}\])\s*$", re.I)
_VERSION_TERMS = re.compile(
    r"\b(live|remix(?:ed)?|acoustic|demo|instrumental|karaoke|mono|stereo|remaster(?:ed)?|mix)\b", re.I)


def edition_compatible(name: str, request: str) -> bool:
    """An explicit recording qualifier is a constraint, never filler.

    Scope qualifiers to the title before an explicit artist boundary. This
    prevents a catalogue hypothesis treating 'live' as the connector in
    'Yesterday live by the Beatles', while allowing an artist named Live.
    """
    requested = re.split(r"\s+by\s+", request, flags=re.I)[0]
    normalize = lambda term: "remaster" if term.startswith("remaster") else "remix" if term.startswith("remix") else term
    wanted = {normalize(term.casefold()) for term in _VERSION_TERMS.findall(requested)}
    offered = {normalize(term.casefold()) for term in _VERSION_TERMS.findall(name)}
    if not wanted.issubset(offered):
        return False
    take = re.search(r"\btake\s+(\d+)\b", requested, re.I)
    if take and not re.search(r"\btake\s+" + re.escape(take[1]) + r"\b", name, re.I):
        return False
    if wanted:
        years = set(re.findall(r"\b(?:19|20)\d{2}\b", requested))
        if not years.issubset(set(re.findall(r"\b(?:19|20)\d{2}\b", name))):
            return False
    return True


def title_key(name: str, request: str = "") -> str:
    """Ignore a release's remaster label unless the user requested an edition.

    Only an entire, separated remaster suffix is metadata. Live performances,
    mixes, acoustic versions and titles containing 'Remastered' stay distinct.
    The original title and URI remain intact for playback and feedback.
    """
    if not re.search(r"\bremaster(?:ed)?\b", request, re.I):
        name = _REMASTER_SUFFIX.sub("", name)
    return " ".join(_norm(name).split())


@dataclass(frozen=True)
class Candidate:
    uri: str
    name: str
    artist: str = ""
    metadata: dict | None = None

    @property
    def label(self) -> str:
        return f"{self.name} by {self.artist}" if self.artist else self.name

    @property
    def approximate(self) -> bool:
        return (self.metadata or {}).get("resolution_approximate") is True


class AmbiguousMatch(LookupError):
    def __init__(self, query: str, candidates: list[Candidate]):
        self.query, self.candidates = query, candidates
        labels = ", ".join(c.label for c in candidates[:5])
        super().__init__(f"Several matches for {query!r}: {labels}")


class MusicConfirmationRequired(LookupError):
    """Local music suggestions that need an explicit user choice before use."""

    def __init__(self, query: str, candidates: list[Candidate]):
        self.query, self.candidates = query, candidates[:3]
        labels = ", ".join(candidate.label for candidate in self.candidates)
        super().__init__(f"Confirm a match for {query!r}: {labels}")


def eligible_media(candidate: Candidate, kind: str, title: str) -> bool:
    """A fallback still respects recording edition, entity kind, and playability."""
    if not isinstance(candidate, Candidate) or kind not in {"track", "album"}:
        return False
    meta = candidate.metadata or {}
    return bool(isinstance(candidate.uri, str)
                and re.fullmatch(r"spotify:" + kind + r":[A-Za-z0-9]+", candidate.uri)
                and isinstance(candidate.name, str) and candidate.name.strip()
                and meta.get("is_local") is not True and meta.get("is_playable") is not False
                and not meta.get("restrictions")
                and meta.get("type", kind) == kind
                and edition_compatible(candidate.name, title))


def best_available_candidate(candidates: list[Candidate], title: str, artist: str = "",
                             kind: str = "track", *, rank_title: bool = False,
                             minimum_title: float = 0.0) -> Candidate | None:
    """Spotify order without an artist; title rank inside a grounded credited artist."""
    pool, seen = [], set()
    for candidate in candidates:
        if eligible_media(candidate, kind, title) and candidate.uri not in seen:
            seen.add(candidate.uri)
            similarity = SequenceMatcher(None, title_key(title, title),
                                         title_key(candidate.name, title)).ratio()
            if similarity >= minimum_title:
                pool.append((similarity, candidate))
    if artist:
        heard = album_key(artist)
        def credits(candidate):
            rows = (candidate.metadata or {}).get("artists") or [{"name": candidate.artist}]
            return [row["name"] for row in rows if isinstance(row, dict)
                    and isinstance(row.get("name"), str) and row["name"].strip()]
        exact = [row for row in pool if any(album_key(name) == heard for name in credits(row[1]))]
        if exact:
            pool = exact
        else:
            names = {album_key(name) for _, candidate in pool for name in credits(candidate)
                     if SequenceMatcher(None, heard, album_key(name)).ratio() >= 0.90}
            if len(names) != 1:
                return None
            pool = [row for row in pool if any(album_key(name) in names for name in credits(row[1]))]
    if not pool:
        return None
    if artist or rank_title:
        pool.sort(key=lambda row: row[0], reverse=True)
    elif all(isinstance((candidate.metadata or {}).get("spotify_relevance"), int)
             for _, candidate in pool):
        pool.sort(key=lambda row: row[1].metadata["spotify_relevance"])
    winner = pool[0][1]
    return Candidate(winner.uri, winner.name, winner.artist,
                     {**(winner.metadata or {}), "resolution_approximate": True})


def album_key(value: str) -> str:
    """Join Unicode words for album spelling/spacing, preserving edition words."""
    return "".join(_norm(value).split())


def rank_albums(items: list[dict], title: str, artist: str = "") -> list[tuple[float, Candidate]]:
    """Only near-verbatim albums; every credited artist can satisfy the constraint."""
    wanted = album_key(title)
    if not wanted:
        return []
    ranked, seen = [], set()
    for item in items:
        if not isinstance(item, dict):
            continue
        uri, name = item.get("uri", ""), item.get("name", "")
        if (not isinstance(uri, str) or not uri.startswith("spotify:album:") or uri in seen
                or not isinstance(name, str) or not name.strip()):
            continue
        seen.add(uri)
        if not edition_compatible(name, title):
            continue
        credits = [row["name"] for row in item.get("artists") or []
                   if isinstance(row, dict) and isinstance(row.get("name"), str) and row["name"].strip()]
        title_score = SequenceMatcher(None, wanted, album_key(name)).ratio()
        artist_score = max((SequenceMatcher(None, album_key(artist), album_key(credit)).ratio()
                            for credit in credits), default=0.0) if artist else 1.0
        if title_score < 0.9 or artist_score < 0.9:
            continue
        ranked.append((title_score, Candidate(uri, name, ", ".join(credits), item)))
    return sorted(ranked, key=lambda row: row[0], reverse=True)


# Confidence gate for catalogue-fallback matches. These are guesses about a
# title the recogniser already mangled, so the winner must both resemble what
# was heard and be clearly ahead of the runner-up. Tuned against real output:
# "Nakhshab"/"Nakhsha" -> Naksha pass; "Nachshavai" is too close to Nanchaku to
# play without asking.
CATALOGUE_MIN_RATIO = 0.60
CATALOGUE_MIN_MARGIN = 0.10


# A near-verbatim title needs no second opinion; anything less is worth checking
# against the artist's real catalogue before playing.
STRONG_MATCH_RATIO = 0.90


def _strong_match(title: str, candidate: Candidate) -> bool:
    """True when the heard title matches this candidate almost exactly."""
    return SequenceMatcher(None, title_key(title, title),
                           title_key(candidate.name, title)).ratio() >= STRONG_MATCH_RATIO


def collapse_reissues(ranked: list[Candidate], request: str = "") -> list[Candidate]:
    """Merge candidates that are the same song on different releases.

    Spotify returns a separate track per album, single and remaster, so a plain
    search for "bohemian rhapsody" yielded two entries both labelled "Bohemian
    Rhapsody by Queen". Asking the user to choose between identical labels is
    an unanswerable question; same title AND same artist means the choice does
    not matter, so keep the best-ranked one. Unrequested remaster labels are
    release metadata; alternate performances and credited artists still differ.
    """
    seen, unique = set(), []
    for candidate in ranked:
        credits = (candidate.metadata or {}).get("artists") or []
        artist_key = tuple(_norm(row.get("name", "")) for row in credits) or (_norm(candidate.artist),)
        key = (title_key(candidate.name, request), artist_key)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def _clearly_best(title: str, ranked: list[Candidate]) -> bool:
    """True when the top candidate is a confident winner, not a coin flip."""
    if not ranked:
        return False
    wanted = title_key(title, title)
    best = SequenceMatcher(None, wanted, title_key(ranked[0].name, title)).ratio()
    if best < CATALOGUE_MIN_RATIO:
        return False
    if len(ranked) == 1:
        return True
    second = SequenceMatcher(None, wanted, title_key(ranked[1].name, title)).ratio()
    return (best - second) >= CATALOGUE_MIN_MARGIN


def rank_tracks(items: list[dict], title: str, artist: str | None = None) -> list[Candidate]:
    wanted_title, wanted_artist = title_key(title, title), _norm(artist or "")
    scored = []
    for item in items:
        name = item.get("name", "")
        if not edition_compatible(name, title):
            continue
        artists = item.get("artists") or []
        aname = artists[0].get("name", "") if artists else ""
        ts = SequenceMatcher(None, wanted_title, title_key(name, title)).ratio()
        ass = max((SequenceMatcher(None, wanted_artist, _norm(row.get("name", ""))).ratio()
                   for row in artists), default=0.0) if wanted_artist else 1.0
        # A supplied artist is a hard constraint: never play a similarly named
        # song by somebody else.
        if wanted_artist and ass < 0.72:
            continue
        score = 0.72 * ts + 0.28 * ass
        # Do not turn Spotify's broad-search fallback into an unrelated play.
        if ts < 0.55:
            continue
        scored.append((score, Candidate(item.get("uri", ""), name, aname, item)))
    return [c for _, c in sorted(scored, key=lambda x: x[0], reverse=True)]
