"""Catalogue-grounded track resolution for mangled speech.

Speech recognition spells Hindi/Punjabi names by sound ("see them all" for
"Seedhe Maut"), which no amount of string cleverness against Spotify's *search*
can recover: search is an open-set problem and the right song is simply not in
the results. This module turns it into a closed-set problem instead. We keep a
local index of the user's own Spotify world (liked/top/playlist tracks plus the
full discographies of their known artists) and score the heard words against
that small, relevant set with a Hindi-aware phonetic comparison.

Building the index is slow and rate-limited, so it is never done on the command
path: ``build`` runs from a background thread (or the offline script) and
``match``/``best`` only read the persisted result.
"""
from __future__ import annotations

import itertools
import json
import logging
import os
import random
import re
import threading
import time
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Callable, Iterable, Optional

from .config import ROOT
from .resolution import edition_compatible, title_key as recording_title_key

log = logging.getLogger("memoir.catalogue")

INDEX_PATH = ROOT / "assistant" / "logs" / ".catalogue.json"
RADIO_HISTORY_PATH = ROOT / "assistant" / "logs" / ".radio_history.json"
VOCABULARY_PATH = ROOT / "assistant" / "logs" / ".vocabulary.json"

LIBRARY_SOURCES = ("saved", "top", "playlist")

# ---------------------------------------------------------------- phonetics

# Digits are spoken in Hindi as often as English in song titles ("3 Cheez" is
# read "teen cheez"), so each numeral expands to every plausible spoken form.
_NUMERALS = {
    "0": ("shunya", "zero"), "1": ("ek", "one"), "2": ("do", "two"),
    "3": ("teen", "three"), "4": ("char", "four"), "5": ("paanch", "five"),
    "6": ("chhe", "six"), "7": ("saat", "seven"), "8": ("aath", "eight"),
    "9": ("nau", "nine"), "10": ("das", "ten"),
}
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
_MAX_VARIANTS = 8


def _ascii_fold(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "").casefold()
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def _words(text: str) -> list[str]:
    """Lowercase ASCII words; apostrophes vanish ("don't" -> "dont")."""
    text = _ascii_fold(text).replace("'", "").replace("’", "")
    return re.findall(r"[a-z0-9]+", text)


def _number_words(token: str) -> tuple:
    if token in _NUMERALS:
        return _NUMERALS[token]
    if token.isdigit() and token[0] != "0" and len(token) <= 2:
        n = int(token)
        english = _TENS[n // 10] + ((" " + _ONES[n % 10]) if n % 10 else "")
        return (english.strip(),)
    return ()


def _spoken_variants(words: list[str]) -> list[list[str]]:
    """Every reading of the words with digits spelled out (digits kept too)."""
    options = []
    for word in words:
        if word.isdigit():
            spoken = [alt.split() for alt in _number_words(word)]
            options.append(spoken + [[word]])
        else:
            options.append([[word]])
    variants = []
    for combo in itertools.product(*options):
        variants.append([w for part in combo for w in part])
        if len(variants) >= _MAX_VARIANTS:
            break
    return variants


# Digraphs first so "dh"/"th" never survive as a lone consonant + "h".
_DIGRAPHS = (("ph", "f"), ("kh", "k"), ("gh", "g"), ("dh", "d"),
             ("bh", "b"), ("th", "t"), ("sh", "s"), ("jh", "j"), ("ck", "k"),
             ("qu", "ku"))
_VOWEL_FOLDS = (("ee", "i"), ("ea", "i"), ("ie", "i"), ("oo", "u"),
                ("ou", "u"), ("ue", "u"), ("aa", "a"), ("ai", "e"), ("ae", "e"), ("ei", "e"), ("au", "o"), ("aw", "o"), ("oa", "o"),
                ("ow", "o"))
_SINGLE = {"z": "s", "v": "w", "q": "k", "x": "ks", "y": "i"}


def _word_key(word: str) -> str:
    """Phonetic spelling of one lowercase ASCII word (see phonetic_key)."""
    # Hard/soft c before the "ch" digraph is resolved first so that "ch" -> "c"
    # (its own sound) can never be confused with a plain k-sound c.
    chars, i = [], 0
    while i < len(word):
        ch = word[i]
        if ch == "c":
            nxt = word[i + 1] if i + 1 < len(word) else ""
            if nxt == "h":
                chars.append("C")
                i += 2
                continue
            if nxt == "k":
                chars.append("k")
                i += 2
                continue
            chars.append("s" if nxt in ("e", "i", "y") else "k")
        else:
            chars.append(ch)
        i += 1
    word = "".join(chars)
    for src, dst in _DIGRAPHS:
        word = word.replace(src, dst)
    # A trailing silent "e" ("cheese", "seedhe", "love"): drop after a consonant
    # in a word that has an earlier vowel. Two letter words ("me", "he") stay.
    if len(word) > 3 and word.endswith("e") and word[-2] not in "aeiou" \
            and any(ch in "aeiou" for ch in word[:-1]):
        word = word[:-1]
    # "ay"/"ey" are vowels only before a consonant ("day"), not in "shayad".
    word = re.sub(r"ay(?![aeiou])", "e", word)
    word = re.sub(r"ey(?![aeiou])", "i", word)
    for src, dst in _VOWEL_FOLDS:
        word = word.replace(src, dst)
    # After the vowel folds: a "w" that came from "v" must not merge with "a".
    word = "".join(_SINGLE.get(ch, ch) for ch in word)
    word = re.sub(r"(.)\1+", r"\1", word)
    # "ch" internally uses an upper-case C; lowering at the end keeps it apart
    # from plain k-sound c (already mapped to k above).
    return word.replace("C", "c")


def phonetic_key(text: str) -> str:
    """Sound-alike normalisation for Latin-script Hindi/English names.

    Spaces between words are kept; the matcher also compares with spaces
    removed because the recogniser splits and joins words freely
    ("sidemotri" vs "seedhe maut").
    """
    return " ".join(k for k in (_word_key(w) for w in _words(text)) if k)


_SKELETON_FOLD = str.maketrans({"b": "p", "d": "t", "g": "k", "c": "k"})


def consonant_skeleton(text_or_key: str, *, is_key: bool = False) -> str:
    """Consonants only, with voicing folded (b/p, d/t, g/k): ASR confuses them
    constantly and Hindi aspirates already collapsed in the key."""
    key = text_or_key if is_key else phonetic_key(text_or_key)
    out = []
    for ch in key:
        if ch in "aeiou hw" or not ch.isalpha():
            continue
        ch = ch.translate(_SKELETON_FOLD)
        if not out or out[-1] != ch:
            out.append(ch)
    return "".join(out)


@dataclass(frozen=True)
class Form:
    """One precomputed reading of a phrase, ready for comparison."""
    key: str          # phonetic key with word spaces
    flat: str         # same, spaces removed
    skeleton: str


def _form(words: list[str]) -> Form:
    key = " ".join(k for k in (_word_key(w) for w in words) if k)
    return Form(key, key.replace(" ", ""), consonant_skeleton(key, is_key=True))


def forms_for(text: str) -> list[Form]:
    """Forms for every numeral reading of ``text`` (deduplicated)."""
    seen, forms = set(), []
    for variant in _spoken_variants(_words(text)):
        form = _form(variant)
        if form.flat and form.flat not in seen:
            seen.add(form.flat)
            forms.append(form)
    return forms


class _Compare:
    """Blend of key and skeleton similarity against one heard Form.

    difflib's seq2 caches its index, so the heard side is set once and each
    candidate swept past it; cheap upper bounds skip hopeless candidates.
    """

    W_KEY, W_SKEL = 0.6, 0.4

    def __init__(self, heard: Form) -> None:
        self.heard = heard
        self._key = SequenceMatcher(None, "", heard.flat, autojunk=False)
        self._skel = SequenceMatcher(None, "", heard.skeleton, autojunk=False)
        self._spaced = SequenceMatcher(None, "", heard.key, autojunk=False)

    def sim(self, cand: Form, floor: float = 0.0) -> float:
        if not cand.flat:
            return 0.0
        key = self._key
        key.set_seq1(cand.flat)
        use_skel = len(cand.skeleton) >= 2 and len(self.heard.skeleton) >= 2
        if use_skel:
            skel = self._skel
            skel.set_seq1(cand.skeleton)
            # Cheap upper bounds first: most of the index is nowhere near.
            if self.W_KEY * key.real_quick_ratio() + self.W_SKEL * skel.real_quick_ratio() < floor:
                return 0.0
            if self.W_KEY * key.quick_ratio() + self.W_SKEL * skel.quick_ratio() < floor:
                return 0.0
        elif key.quick_ratio() * 0.95 < floor:
            return 0.0
        key_ratio = key.ratio()
        if cand.key != cand.flat or self.heard.key != self.heard.flat:
            spaced = self._spaced
            spaced.set_seq1(cand.key)
            key_ratio = max(key_ratio, spaced.ratio())
        if not use_skel:
            # Too little consonant evidence to weigh: key alone, slightly damped.
            return key_ratio * 0.95
        return self.W_KEY * key_ratio + self.W_SKEL * skel.ratio()


# -------------------------------------------------------------- index model


def _norm_title(text: str) -> str:
    return " ".join(_words(text))


class Index:
    """The persisted track list plus the per-entry phonetic forms."""

    def __init__(self, data: dict) -> None:
        self.data = data
        self.entries: list[dict] = data.get("tracks", [])
        self.built_at: float = float(data.get("built_at", 0) or 0)
        self.complete: bool = bool(data.get("complete", True))
        # Distinct strings are compared once, then fanned out to their entries.
        self._title_forms: dict[str, list[Form]] = {}
        self._unqualified_title_forms: dict[str, list[Form]] = {}
        self._title_entries: dict[str, list[int]] = {}
        self._artist_forms: dict[str, list[Form]] = {}
        for i, entry in enumerate(self.entries):
            title_key = _norm_title(entry.get("name", ""))
            self._title_entries.setdefault(title_key, []).append(i)
            if title_key not in self._title_forms:
                self._title_forms[title_key] = forms_for(entry.get("name", ""))
                # Keep edition labels in the index. Only unqualified requests
                # may also match the recording title without a remaster suffix.
                base = recording_title_key(entry.get("name", ""))
                self._unqualified_title_forms[title_key] = forms_for(base)
            for name in entry.get("artists", []):
                akey = _norm_title(name)
                if akey not in self._artist_forms:
                    self._artist_forms[akey] = forms_for(name)

    def __len__(self) -> int:
        return len(self.entries)

    def artist_keys(self, entry: dict) -> list[str]:
        return [_norm_title(name) for name in entry.get("artists", [])]


def dedupe(entries: Iterable[dict]) -> list[dict]:
    """Merge by uri, then collapse reissues (same title, same primary artist).

    Library-sourced copies win the collapse: that is the version the user
    actually saved, and its sources record why the song matters to them.
    """
    by_uri: dict[str, dict] = {}
    for entry in entries:
        uri = entry.get("uri")
        if not uri:
            continue
        have = by_uri.get(uri)
        if have is None:
            by_uri[uri] = dict(entry, sources=list(entry.get("sources", [])))
            if "playlists" in entry:
                by_uri[uri]["playlists"] = list(entry["playlists"])
        else:
            for src in entry.get("sources", []):
                if src not in have["sources"]:
                    have["sources"].append(src)
            _merge_playlists(have, entry)
    by_song: dict[tuple, dict] = {}
    for entry in by_uri.values():
        artists = entry.get("artists") or [""]
        key = (_norm_title(entry.get("name", "")), _norm_title(artists[0]))
        have = by_song.get(key)
        if have is None:
            by_song[key] = entry
            continue
        keep, drop = have, entry
        if (_is_library(entry) and not _is_library(have)):
            keep, drop = entry, have
        for src in drop["sources"]:
            if src not in keep["sources"]:
                keep["sources"].append(src)
        _merge_playlists(keep, drop)
        by_song[key] = keep
    return list(by_song.values())


def _merge_playlists(keep: dict, drop: dict) -> None:
    """Union the playlist ids two copies of one song were found in.

    The key only exists on entries that came from a playlist, so indexes built
    before this field (or with no playlist hits) keep their exact shape.
    """
    extra = [pid for pid in drop.get("playlists", []) if pid not in keep.get("playlists", [])]
    if extra:
        keep["playlists"] = list(keep.get("playlists", [])) + extra


def _is_library(entry: dict) -> bool:
    return any(src in LIBRARY_SOURCES for src in entry.get("sources", []))


def load(path=None) -> Optional[Index]:
    """Read the persisted index, or None when missing/corrupt/empty."""
    path = path or INDEX_PATH
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or not data.get("tracks"):
        return None
    return Index(data)


def save(data: dict, path=None) -> None:
    """Atomic write: a reader (or a crash mid-write) never sees half a file."""
    path = path or INDEX_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def known_artists(path=None) -> list[str]:
    """The user's artist list, read-only, from the vocabulary file."""
    try:
        data = json.loads((path or VOCABULARY_PATH).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [a for a in data.get("artists", []) if isinstance(a, str) and a.strip()]


# ------------------------------------------------------------------ matching

# Calibrated against real mangled transcripts plus negatives (see
# tests/test_catalogue.py and assistant/experimental/catalogue_eval.py).
ACCEPT_WITH_ARTIST = 0.78    # combined title+artist score
ACCEPT_TITLE_ONLY = 0.90     # whole-span title similarity, no artist evidence
# An artist span this short ("cg") is recogniser debris, not evidence either
# way, so a near-exact multi-word title in the user's library may stand on its
# own. One-word titles never do: "creep by tlc" is a real request for a
# different Creep, which search must find.
WEAK_ARTIST_FLAT = 3
WEAK_ARTIST_TITLE = 0.97
WEAK_ARTIST_FACTOR = 0.93
MIN_WEAK_TITLE_CHARS = 5
MIN_TITLE = 0.62             # candidate pool floor (cheap pruning)
# A perfect artist must not rescue a poor title: "channa by Seedhe Maut" is
# simply not "Nayaab", however well the artist matches.
MIN_TITLE_ACCEPT = 0.75
MIN_ARTIST = 0.58
TITLE_WEIGHT = 0.6
MARGIN = 0.04                # over the best *different* song
LIBRARY_PRIOR = 0.02         # tie-breakers only
MAX_WORDS = 12
# Real titles/artists are short; skipping absurd splits keeps long, noisy
# utterances ("james song to something played by kanye west") cheap.
MAX_TITLE_WORDS = 6
MAX_ARTIST_WORDS = 5


@dataclass
class Scored:
    entry: dict
    score: float
    title_sim: float
    artist_sim: float
    hypothesis: tuple = field(default=("", "", ""))   # (title, connector, artist)
    threshold: float = 0.0                            # score needed to accept

    @property
    def song_key(self) -> tuple:
        artists = self.entry.get("artists") or [""]
        return (recording_title_key(self.entry.get("name", ""), self.hypothesis[0]),
                tuple(_artist_key(artist) for artist in artists))


def _hypotheses(words: list[str]) -> list[tuple]:
    """Every (title, connector, artist) split of the heard words.

    The connector is any single word ("by" arrives as "buy", "i", "is", "bai"),
    or absent; the title-only reading has no artist at all.
    """
    n = len(words)
    out = [(words, [], [])] if n <= MAX_TITLE_WORDS else []
    for cut in range(1, min(n, MAX_TITLE_WORDS + 1)):
        if n - cut <= MAX_ARTIST_WORDS:
            out.append((words[:cut], [], words[cut:]))
        if cut + 1 < n and n - cut - 1 <= MAX_ARTIST_WORDS:
            out.append((words[:cut], [words[cut]], words[cut + 1:]))
    return out


def match(heard: str, index: Index, limit: int = 5) -> list[Scored]:
    """Score every index entry against the heard words, best first."""
    words = _words(heard)
    if not words or len(words) > MAX_WORDS or not index.entries:
        return []
    hyps = _hypotheses(words)

    title_cache: dict[tuple, dict[str, float]] = {}
    artist_cache: dict[tuple, dict[str, float]] = {}

    def title_sims(span: tuple) -> dict:
        if span not in title_cache:
            comps = [_Compare(f) for f in forms_for(" ".join(span))]
            sims = {}
            title_forms = (index._title_forms if re.search(r"\bremaster(?:ed)?\b", heard, re.I)
                           else index._unqualified_title_forms)
            for tkey, cand_forms in title_forms.items():
                best = 0.0
                for comp in comps:
                    for cf in cand_forms:
                        best = max(best, comp.sim(cf, MIN_TITLE))
                if best >= MIN_TITLE:
                    sims[tkey] = best
            title_cache[span] = sims
        return title_cache[span]

    def artist_sims(span: tuple) -> dict:
        if span not in artist_cache:
            comps = [_Compare(f) for f in forms_for(" ".join(span))]
            sims = {}
            for akey, cand_forms in index._artist_forms.items():
                best = 0.0
                for comp in comps:
                    for cf in cand_forms:
                        best = max(best, comp.sim(cf, MIN_ARTIST))
                if best >= MIN_ARTIST:
                    sims[akey] = best
            artist_cache[span] = sims
        return artist_cache[span]

    best: dict[int, Scored] = {}

    def offer(i: int, scored: Scored) -> None:
        if not edition_compatible(scored.entry.get("name", ""), heard):
            return
        if i not in best or scored.score > best[i].score:
            best[i] = scored

    for title_words, conn_words, artist_words in hyps:
        tspan = tuple(title_words)
        tsims = title_sims(tspan)
        if not tsims:
            continue
        if not artist_words:
            # Title only: the whole utterance is the title. Needs near-verbatim
            # similarity because nothing else corroborates it.
            for tkey, sim in tsims.items():
                for i in index._title_entries[tkey]:
                    entry = index.entries[i]
                    prior = LIBRARY_PRIOR if _is_library(entry) else 0.0
                    offer(i, Scored(entry, sim + prior, sim, 0.0,
                                    (" ".join(title_words), "", ""), ACCEPT_TITLE_ONLY))
            continue
        weak = (len(title_words) >= 2
                and len("".join(_words(" ".join(artist_words)))) <= WEAK_ARTIST_FLAT
                and not any(ch.isdigit() for w in artist_words for ch in w))
        if weak:
            for tkey, sim in tsims.items():
                if sim < WEAK_ARTIST_TITLE or len(tkey.replace(" ", "")) < MIN_WEAK_TITLE_CHARS:
                    continue
                for i in index._title_entries[tkey]:
                    entry = index.entries[i]
                    if _is_library(entry):
                        offer(i, Scored(entry, sim * WEAK_ARTIST_FACTOR + LIBRARY_PRIOR, sim, 0.0,
                                        (" ".join(title_words), " ".join(conn_words),
                                         " ".join(artist_words)), ACCEPT_TITLE_ONLY))
        asims = artist_sims(tuple(artist_words))
        if not asims:
            continue
        for tkey, tsim in tsims.items():
            for i in index._title_entries[tkey]:
                entry = index.entries[i]
                a_best = max((asims.get(a, 0.0) for a in index.artist_keys(entry)),
                             default=0.0)
                if a_best < MIN_ARTIST:
                    continue
                score = TITLE_WEIGHT * tsim + (1 - TITLE_WEIGHT) * a_best
                score += LIBRARY_PRIOR if _is_library(entry) else 0.0
                offer(i, Scored(entry, score, tsim, a_best,
                                (" ".join(title_words), " ".join(conn_words),
                                 " ".join(artist_words)), ACCEPT_WITH_ARTIST))
    ranked = sorted(best.values(), key=lambda s: -s.score)
    return ranked[:limit] if limit else ranked


def best(heard: str, index: Optional[Index]) -> Optional[tuple]:
    """(entry, score) for a confident match, else None.

    Wrong acceptance is far worse than no match (no match falls back to
    Spotify search), so this needs both an absolute score and a margin over the
    best *different* song.
    """
    if index is None:
        return None
    ranked = match(heard, index, limit=0)
    if not ranked:
        return None
    top = ranked[0]
    if top.score < top.threshold or top.title_sim < MIN_TITLE_ACCEPT:
        return None
    rival = next((s for s in ranked[1:] if s.song_key != top.song_key), None)
    if rival is not None and top.score - rival.score < MARGIN:
        return None
    return top.entry, top.score


# -------------------------------------------------------------------- radio

# Spotify's recommendation and radio endpoints are closed to Development-mode
# apps, so "what to play after this song" is answered from the local index.
RADIO_HISTORY_LIMIT = 300
# Share of a radio taken from: the seed's own artists, their collaborators
# (the "scene"), and tracks that sit in the same playlists as the seed.
RADIO_SHARES = (0.35, 0.45, 0.20)
RADIO_RUN_CAP = 2       # consecutive tracks by one primary artist
RADIO_ALBUM_CAP = 3     # tracks from one album in a single radio
RADIO_SEED_ARTISTS = 3  # credited artists treated as "the same artist"
# A track credited to many artists (cypher, soundtrack, compilation) says
# little about who works together; it would link unrelated scenes.
RADIO_MAX_COLLAB_CREDITS = 6


def radio_history(path=None) -> list[str]:
    """URIs of recent radio tracks we sent, oldest first. Never raises."""
    try:
        data = json.loads((path or RADIO_HISTORY_PATH).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    uris = data.get("uris") if isinstance(data, dict) else None
    return [u for u in uris if isinstance(u, str)] if isinstance(uris, list) else []


def remember_radio(uris: Iterable[str], path=None, limit: int = RADIO_HISTORY_LIMIT) -> None:
    """Append sent radio URIs to the rolling history (atomic write).

    A repeat moves to the newest end, so a song heard often keeps being
    deprioritised instead of aging out.
    """
    path = path or RADIO_HISTORY_PATH
    fresh = [u for u in uris if isinstance(u, str) and u]
    if not fresh:
        return
    merged = [u for u in radio_history(path) if u not in set(fresh)] + list(dict.fromkeys(fresh))
    save({"uris": merged[-limit:]}, path)


def _artist_key(name: str) -> str:
    # Non-Latin names fold to nothing; fall back to the raw text so they still
    # group (two different Devanagari artists must not share the key "").
    return _norm_title(name) or (name or "").casefold().strip()


class _RadioTables:
    """Per-index lookup tables for radio, built once and reused."""

    def __init__(self, entries: list) -> None:
        self.entries = entries
        self.keys: list[list[str]] = []
        self.song: list[tuple] = []
        self.album: list[str] = []
        self.lib: list[bool] = []
        self.by_uri: dict[str, int] = {}
        collab: dict[str, set] = {}
        canon: dict[tuple, int] = {}
        for i, entry in enumerate(entries):
            keys = [k for k in (_artist_key(a) for a in entry.get("artists") or []) if k]
            self.keys.append(keys)
            primary = keys[0] if keys else ""
            self.song.append((_norm_title(entry.get("name", "")), primary))
            self.album.append(_norm_title(entry.get("album", "")))
            self.lib.append(_is_library(entry))
            if entry.get("uri"):
                self.by_uri.setdefault(entry["uri"], i)
            if len(keys) <= RADIO_MAX_COLLAB_CREDITS:
                for k in keys:
                    collab.setdefault(k, set()).update(keys)
            if not keys or not entry.get("uri"):
                continue
            # Reissues of one song collapse to a single candidate, preferring
            # the copy the user actually has in their library.
            have = canon.get(self.song[i])
            if have is None or (self.lib[i] and not self.lib[have]):
                canon[self.song[i]] = i
        for k, peers in collab.items():
            peers.discard(k)
        self.collab = collab
        self.valid = sorted(canon.values())
        self.by_artist: dict[str, list[int]] = {}
        self.by_playlist: dict[str, list[int]] = {}
        for i in self.valid:
            for k in dict.fromkeys(self.keys[i]):
                self.by_artist.setdefault(k, []).append(i)
            for pid in entries[i].get("playlists") or []:
                self.by_playlist.setdefault(pid, []).append(i)


def _radio_tables(index) -> Optional[_RadioTables]:
    entries = getattr(index, "entries", None)
    if not isinstance(entries, list) or not entries:
        return None
    tables = getattr(index, "_radio_tables", None)
    if tables is None or tables.entries is not entries:
        tables = _RadioTables(entries)
        try:
            index._radio_tables = tables
        except AttributeError:
            pass
    return tables


def _seed_artist_keys(tables: _RadioTables, names: Iterable) -> list[str]:
    """Index artist keys for heard/credited names (exact, else a close spelling)."""
    import difflib
    found: list[str] = []
    for name in names:
        if isinstance(name, dict):
            name = name.get("name", "")
        key = _artist_key(name) if isinstance(name, str) else ""
        if not key:
            continue
        if key not in tables.by_artist:
            close = difflib.get_close_matches(key, list(tables.by_artist), n=1, cutoff=0.9)
            key = close[0] if close else ""
        if key and key not in found:
            found.append(key)
    return found


def radio(index, seed, length: int = 30, rng: Optional[random.Random] = None,
          recent: Iterable[str] = ()) -> list[str]:
    """Track URIs to play after ``seed``, from the local index (seed excluded).

    ``seed`` is a track URI, an index entry / dict with ``uri`` and ``artists``,
    or an artist name. A seed track missing from the index falls back to its
    credited artist(s); if those are unknown too the result is empty.

    Three pools, each deduped by URI and by (title, primary artist):
      same artist   other tracks by the seed's credited artists, library first;
      scene         tracks by artists who co-credit with them on any indexed
                    track (one hop), library first, interleaved across artists;
      co-playlist   tracks sharing a playlist with the seed (needs the
                    ``playlists`` field written by newer index builds).
    They are mixed ~35/45/20, a short pool's share goes to the others, then the
    order is shuffled with no more than ``RADIO_RUN_CAP`` in a row by one artist
    and ``RADIO_ALBUM_CAP`` from one album (relaxed only to reach ``length``).
    ``recent`` URIs are ranked last in their pool, never excluded.
    """
    tables = _radio_tables(index)
    if tables is None or length <= 0:
        return []
    rng = rng or random.Random()
    entries = tables.entries

    seed_uri, hint = "", []
    if isinstance(seed, dict):
        seed_uri, hint = seed.get("uri") or "", list(seed.get("artists") or [])
    elif isinstance(seed, str):
        if seed.startswith("spotify:"):
            seed_uri = seed
        else:
            hint = [seed]
    seed_i = tables.by_uri.get(seed_uri) if seed_uri else None
    if seed_i is not None:
        seed_keys = tables.keys[seed_i][:RADIO_SEED_ARTISTS]
        seed_song = tables.song[seed_i]
        playlists = set(entries[seed_i].get("playlists") or [])
    else:
        seed_keys = _seed_artist_keys(tables, hint)[:RADIO_SEED_ARTISTS]
        seed_song = None
        playlists = {pid for k in seed_keys for i in tables.by_artist.get(k, [])
                     for pid in entries[i].get("playlists") or []}
    if not seed_keys:
        return []

    recent_set = set(recent)
    seed_set = set(seed_keys)

    def eligible(i: int) -> bool:
        return entries[i]["uri"] != seed_uri and tables.song[i] != seed_song

    def rank(i: int, extra: int = 0) -> tuple:
        # Fresh before recently-sent, then library before discography-only.
        return (entries[i]["uri"] in recent_set, not tables.lib[i], extra, rng.random())

    taken: set[int] = set()
    same_ids = {i for k in seed_keys for i in tables.by_artist.get(k, []) if eligible(i)}
    same = sorted(same_ids, key=lambda i: rank(i, tables.keys[i][0] not in seed_set))
    taken |= same_ids

    peers = {c for k in seed_keys for c in tables.collab.get(k, ())} - seed_set
    scene_ids = {i for c in peers for i in tables.by_artist.get(c, [])
                 if i not in taken and eligible(i)}
    scene = _interleave(sorted(scene_ids, key=rank), lambda i: tables.keys[i][0],
                        lambda i: (entries[i]["uri"] in recent_set, not tables.lib[i]), rng)
    taken |= scene_ids

    shared: dict[int, int] = {}
    for pid in playlists:
        for i in tables.by_playlist.get(pid, []):
            if i not in taken and eligible(i):
                shared[i] = shared.get(i, 0) + 1
    co_playlist = sorted(shared, key=lambda i: rank(i, -shared[i]))

    pools = [same, scene, co_playlist]
    pulled = _pull(pools, tables, length, RADIO_SHARES)
    merged = _mix(pulled, rng)
    primary = [tables.keys[i][0] for i in merged]
    ordered: list[int] = []
    rest = list(range(len(merged)))
    while rest:
        # Greedy: first remaining track that does not make a third in a row.
        pick = next((j for j in rest if not (
            len(ordered) >= RADIO_RUN_CAP
            and all(tables.keys[ordered[-n]][0] == primary[j]
                    for n in range(1, RADIO_RUN_CAP + 1)))), rest[0])
        rest.remove(pick)
        ordered.append(merged[pick])
    return [entries[i]["uri"] for i in ordered]


def _interleave(items: list, artist_of, tier_of, rng: random.Random) -> list:
    """Round-robin across artists within each tier, so one collaborator with a
    huge discography cannot fill the scene share."""
    out: list = []
    tiers: dict = {}
    for item in items:
        tiers.setdefault(tier_of(item), []).append(item)
    for tier in sorted(tiers):
        queues: dict[str, list] = {}
        for item in tiers[tier]:
            queues.setdefault(artist_of(item), []).append(item)
        order = list(queues)
        rng.shuffle(order)
        while order:
            for artist in list(order):
                out.append(queues[artist].pop(0))
                if not queues[artist]:
                    order.remove(artist)
    return out


def _pull(pools: list, tables: _RadioTables, length: int, shares: tuple) -> list[list[int]]:
    """Take ``length`` tracks across the pools by share, honouring the album cap."""
    albums: dict[str, int] = {}
    remaining = [list(pool) for pool in pools]
    pulled: list[list[int]] = [[] for _ in pools]

    def grab(p: int, count: int, strict: bool) -> int:
        got, skipped = 0, []
        for pos, i in enumerate(remaining[p]):
            if got >= count:
                skipped.extend(remaining[p][pos:])
                break
            album = tables.album[i]
            if strict and album and albums.get(album, 0) >= RADIO_ALBUM_CAP:
                skipped.append(i)
                continue
            pulled[p].append(i)
            albums[album] = albums.get(album, 0) + 1
            got += 1
        remaining[p] = skipped
        return got

    def topup(strict: bool) -> None:
        # D'Hondt: hand out the shortfall in proportion to the shares, to
        # whichever pool still has something to give.
        closed: set[int] = set()
        while sum(len(x) for x in pulled) < length:
            open_pools = [p for p in range(len(pools)) if remaining[p] and p not in closed
                          and (not strict or _has_room(remaining[p], tables, albums))]
            if not open_pools:
                return
            p = max(open_pools, key=lambda q: shares[q] / (len(pulled[q]) + 1))
            if grab(p, 1, strict) == 0:
                closed.add(p)

    targets = [int(round(length * share)) for share in shares[:-1]]
    targets.append(max(0, length - sum(targets)))
    for p, count in enumerate(targets):
        grab(p, count, True)
    topup(True)
    topup(False)
    return pulled


def _has_room(items: list, tables: _RadioTables, albums: dict) -> bool:
    return any(not tables.album[i] or albums.get(tables.album[i], 0) < RADIO_ALBUM_CAP
               for i in items)


def _mix(pulled: list[list[int]], rng: random.Random) -> list[int]:
    """Random interleave of the pools, each keeping its own preference order."""
    queues = [list(p) for p in pulled]
    out: list[int] = []
    while any(queues):
        total = sum(len(q) for q in queues)
        roll = rng.random() * total
        for q in queues:
            if roll < len(q):
                out.append(q.pop(0))
                break
            roll -= len(q)
    return out


# -------------------------------------------------------------------- build

# 2.35 requests/s for 300 s earned an 85823 s ban in Development mode. The
# background build has all day; it must never cost the live assistant search.
PAGE_DELAY = 2.0
CAP_SAVED = 2000
CAP_PLAYLIST_EACH = 200
CAP_PLAYLIST_TOTAL = 3000
CAP_ARTIST_TRACKS = 300
CAP_ARTIST_ALBUMS = 60
ALBUM_PAGE = 10             # Development-mode apps reject a larger album limit
RETRY_INCOMPLETE_HOURS = 1.0
# Spotify's Development-mode ban has lasted 16-24 h, and the guard in spotify.py
# cannot see the real Retry-After (spotipy hides it), so after a 429 do not even
# probe again for a day: each refused call still counts against the app.
RATE_LIMIT_BACKOFF_HOURS = 24.0


class _Stop(Exception):
    """Rate limited or otherwise unable to continue: keep what we have."""


DEFAULT_MANUAL_REQUESTS = 50
DEFAULT_BACKGROUND_REQUESTS = 20


def _entry(track: dict, source: str, album: str = "") -> Optional[dict]:
    if not isinstance(track, dict):
        return None
    uri = track.get("uri") or ""
    if not uri.startswith("spotify:track:") or track.get("is_local"):
        return None
    out = {
        "uri": uri,
        "name": track.get("name", ""),
        "artists": [a.get("name", "") for a in track.get("artists") or [] if a.get("name")],
        "album": album or (track.get("album") or {}).get("name", ""),
        "sources": [source],
    }
    if track.get("popularity") is not None:
        out["popularity"] = track["popularity"]
    return out if out["name"] and out["artists"] else None


def build(sp, artists: Iterable[str], *, delay: Optional[float] = None,
          previous: Optional[dict] = None, checkpoint: Optional[Callable] = None,
          progress: Optional[Callable[[str], None]] = None,
          include_discographies: bool = False,
          max_requests: int = DEFAULT_MANUAL_REQUESTS,
          source: str = "manual_catalogue") -> dict:
    """Build a bounded library index, preserving prior results.

    GET-only and throttled. On HTTP 429 it stops at once and returns what it
    has, marked incomplete -- never a retry loop against a banned app. Pass an
    incomplete ``previous`` to resume past the artists already covered.
    """
    import spotipy
    from contextlib import nullcontext
    from .spotify import SpotifyRateLimited, SpotifyRequestDeferred
    from . import spotify as spotify_module

    say = progress or (lambda msg: log.info("catalogue: %s", msg))
    delay = PAGE_DELAY if delay is None else delay
    artists = ([a for a in artists if a and a.strip()]
               if include_discographies else [])
    state = {"requests": 0, "errors": 0, "limited": False,
             "deferred": False, "stop_reason": ""}
    # Never replace a useful complete index with a smaller partial crawl.
    entries: list[dict] = list((previous or {}).get("tracks", []))
    done: list[str] = []
    skipped: dict[str, str] = {}
    if previous and not previous.get("complete", True):
        done = list(previous.get("artists_done", []))
        skipped = dict(previous.get("artists_skipped", {}))
    started = time.time()
    last_call = [0.0]

    def call(fn, *args, **kwargs):
        if state["requests"] >= max(0, int(max_requests)):
            state["stop_reason"] = "request_budget"
            raise _Stop("request budget reached")
        wait = delay - (time.monotonic() - last_call[0])
        if wait > 0:
            time.sleep(wait)
        last_call[0] = time.monotonic()
        try:
            context_factory = getattr(spotify_module, "request_context", None)
            context = (context_factory(source=source, background=True)
                       if callable(context_factory) else nullcontext())
            with context:
                state["requests"] += 1
                result = fn(*args, **kwargs)
            state["errors"] = 0
            return result
        except SpotifyRequestDeferred as exc:
            state["deferred"] = True
            state["stop_reason"] = "deferred"
            raise _Stop(str(exc)) from exc
        except SpotifyRateLimited as exc:
            state["limited"] = True
            state["stop_reason"] = "rate_limited"
            raise _Stop(str(exc)) from exc
        except spotipy.SpotifyException as exc:
            if exc.http_status == 429:
                state["limited"] = True
                state["stop_reason"] = "rate_limited"
                raise _Stop("HTTP 429") from exc
            # 400/403/404 deny one resource (a followed playlist, a missing
            # scope), not the whole crawl; only server/network trouble counts.
            if exc.http_status not in (400, 403, 404):
                state["errors"] += 1
                if state["errors"] >= 8:
                    raise _Stop("repeated errors: %s" % exc) from exc
            log.debug("catalogue: %s failed: %s", getattr(fn, "__name__", fn), exc)
            return None
        except Exception as exc:  # network blips must not kill the crawl
            state["errors"] += 1
            if state["errors"] >= 8:
                raise _Stop("repeated errors: %s" % exc) from exc
            log.debug("catalogue: %s failed: %s", getattr(fn, "__name__", fn), exc)
            return None

    artist_ids: dict[str, str] = {}   # normalised name -> id, learned from tracks

    def learn_ids(track: dict) -> None:
        for a in track.get("artists") or []:
            if a.get("id") and a.get("name"):
                artist_ids.setdefault(_norm_title(a["name"]), a["id"])

    def pages(fn, *args, cap: int, page: int, **kwargs):
        """Yield items from an offset-paginated endpoint up to ``cap`` items."""
        offset, got = 0, 0
        while got < cap:
            res = call(fn, *args, limit=min(page, cap - got), offset=offset, **kwargs)
            items = (res or {}).get("items") or []
            if not items:
                return
            for item in items:
                yield item
            got += len(items)
            offset += len(items)
            if not (res or {}).get("next"):
                return

    def own_entries(source: str, items, key=None, playlist: Optional[str] = None):
        for item in items:
            track = key(item) if key else item
            if isinstance(track, dict):
                learn_ids(track)
                e = _entry(track, source)
                if e:
                    if playlist:
                        # Radio uses playlist co-occurrence as a taste signal.
                        e["playlists"] = [playlist]
                    yield e

    try:
        # --- the user's library (saved tracks need a scope the token may lack)
        saved = list(own_entries("saved", pages(
            sp.current_user_saved_tracks, cap=CAP_SAVED, page=50),
            key=lambda it: it.get("track")))
        say("saved tracks: %d" % len(saved))
        entries += saved
        for window in ("short_term", "medium_term", "long_term"):
            top = list(own_entries("top", pages(
                sp.current_user_top_tracks, cap=50, page=50, time_range=window)))
            entries += top
        say("top tracks added; entries=%d" % len(entries))

        me = call(sp.current_user) or {}
        my_id = me.get("id")
        # Only the user's own lists are readable: Development-mode apps get 403
        # for followed playlists, which would just burn requests.
        playlists = [p for p in pages(sp.current_user_playlists, cap=100, page=50)
                     if (p.get("owner") or {}).get("id") == my_id]
        total = 0
        for pl in playlists:
            if total >= CAP_PLAYLIST_TOTAL:
                break
            room = min(CAP_PLAYLIST_EACH, CAP_PLAYLIST_TOTAL - total)
            got = list(own_entries("playlist", pages(
                sp.playlist_items, pl["id"], cap=room, page=50),
                key=lambda it: it.get("item") or it.get("track"), playlist=pl["id"]))
            total += len(got)
            entries += got
        say("playlist tracks: %d from %d playlists" % (total, len(playlists)))

        # Discographies are available only through an explicit manual opt-in.
        if include_discographies:
          # --- discographies of the artists the user listens to
          for name in artists:
            if name in done or name in skipped:
                continue
            artist_id = artist_ids.get(_norm_title(name))
            if not artist_id:
                found = call(sp.search, q=name, type="artist", limit=5) or {}
                for cand in (found.get("artists") or {}).get("items") or []:
                    if cand and _norm_title(cand.get("name", "")) == _norm_title(name):
                        artist_id = cand["id"]
                        break
            if not artist_id:
                skipped[name] = "not found"
                continue
            albums, seen_albums = [], set()
            for alb in pages(sp.artist_albums, artist_id, cap=CAP_ARTIST_ALBUMS,
                             page=ALBUM_PAGE, include_groups="album,single"):
                akey = (_norm_title(alb.get("name", "")), alb.get("total_tracks"))
                if akey not in seen_albums:
                    seen_albums.add(akey)
                    albums.append(alb)
            count = 0
            for alb in albums:
                if count >= CAP_ARTIST_TRACKS:
                    break
                for track in pages(sp.album_tracks, alb["id"], cap=100, page=50):
                    e = _entry(track, "artist", album=alb.get("name", ""))
                    if e:
                        entries.append(e)
                        count += 1
            done.append(name)
            say("%s: %d albums, %d tracks" % (name, len(albums), count))
            if checkpoint:
                checkpoint(_package(entries, done, skipped, state, started, False))
    except _Stop as exc:
        log.warning("catalogue build stopped early: %s", exc)
        say("stopped early: %s" % exc)
    complete = (not state["limited"] and not state["deferred"]
                and not state["stop_reason"] and state["errors"] < 8
                and (not include_discographies
                     or all(a in done or a in skipped for a in artists)))
    package = _package(entries, done, skipped, state, started, complete)
    if (not state["limited"] and previous
            and float(previous.get("retry_not_before", 0) or 0) > time.time()):
        package["retry_not_before"] = previous["retry_not_before"]
    return package


def _package(entries, done, skipped, state, started, complete) -> dict:
    tracks = dedupe(entries)
    package = {
        "version": 1,
        "built_at": time.time(),
        "complete": complete,
        "build_seconds": round(time.time() - started, 1),
        "requests": state["requests"],
        "artists_done": list(done),
        "artists_skipped": dict(skipped),
        "stop_reason": state.get("stop_reason", ""),
        "tracks": tracks,
    }
    if state["limited"]:
        package["retry_not_before"] = time.time() + RATE_LIMIT_BACKOFF_HOURS * 3600
    return package


_refresh_lock = threading.Lock()
_refreshing = False


def needs_refresh(max_age_hours: float = 24.0, path=None) -> bool:
    path = path or INDEX_PATH
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return True
    if float(data.get("retry_not_before", 0) or 0) > time.time():
        return False
    age = (time.time() - float(data.get("built_at", 0) or 0)) / 3600.0
    if not data.get("complete", True):
        return age >= RETRY_INCOMPLETE_HOURS
    return age >= max_age_hours


def refresh_in_background(spotify_obj, artists: Iterable[str],
                          max_age_hours: float = 24.0, path=None,
                          *, enabled: bool = False) -> Optional[threading.Thread]:
    """Rebuild the index on a daemon thread when it is missing or stale."""
    global _refreshing
    if not enabled:
        return None
    path = path or INDEX_PATH
    if (not needs_refresh(max_age_hours, path) or getattr(spotify_obj, "rate_limited", False)
            or getattr(spotify_obj, "web_rate_limited", False) is True):
        return None
    with _refresh_lock:
        if _refreshing:
            return None
        _refreshing = True

    def run() -> None:
        global _refreshing
        try:
            previous = None
            try:
                previous = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass
            data = build(spotify_obj.sp, artists, previous=previous,
                         include_discographies=False,
                         max_requests=DEFAULT_BACKGROUND_REQUESTS,
                         source="catalogue",
                         checkpoint=lambda d: save(d, path))
            if data["tracks"]:
                save(data, path)
                setter = getattr(spotify_obj, "set_catalogue", None)
                if setter:
                    setter(load(path), pin=False)
                log.info("Catalogue index built: %d tracks, %d requests, complete=%s",
                         len(data["tracks"]), data["requests"], data["complete"])
        except Exception:
            log.exception("Catalogue refresh failed")
        finally:
            with _refresh_lock:
                _refreshing = False

    thread = threading.Thread(target=run, name="catalogue-refresh", daemon=True)
    thread.start()
    return thread


def _main() -> None:
    """``python -m assistant.catalogue``: build the index once, throttled."""
    import argparse
    import sys

    from .spotify import Spotify

    parser = argparse.ArgumentParser(description="Build the local track index")
    parser.add_argument("--delay", type=float, default=None)
    parser.add_argument("--fresh", action="store_true", help="ignore a partial index")
    parser.add_argument("--ignore-backoff", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--max-requests", type=int, default=DEFAULT_MANUAL_REQUESTS,
                        help="maximum Spotify requests (default: %(default)s)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--library-only", action="store_true",
                      help="index library tracks without artist discographies (default)")
    mode.add_argument("--include-discographies", action="store_true",
                      help="also crawl artist albums within the request budget")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    previous = None
    if not args.fresh:
        try:
            previous = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
    wait = float((previous or {}).get("retry_not_before", 0) or 0) - time.time()
    if args.ignore_backoff:
        raise SystemExit("--ignore-backoff is no longer supported; stored Spotify cooldown remains active")
    if wait > 0:
        raise SystemExit("Spotify rate-limited the last build; not retrying for %.1f h" % (wait / 3600))
    if args.max_requests < 1:
        raise SystemExit("--max-requests must be at least 1")
    data = build(Spotify().sp, known_artists(), delay=args.delay, previous=previous,
                 include_discographies=args.include_discographies,
                 max_requests=args.max_requests,
                 source="manual_catalogue",
                 checkpoint=save, progress=lambda m: print(m, flush=True))
    save(data)
    print("tracks=%d artists_done=%d skipped=%s requests=%d seconds=%s complete=%s" % (
        len(data["tracks"]), len(data["artists_done"]), data["artists_skipped"],
        data["requests"], data["build_seconds"], data["complete"]))


if __name__ == "__main__":
    _main()
