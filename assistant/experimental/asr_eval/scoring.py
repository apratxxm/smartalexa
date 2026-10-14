"""Command-level scoring shared by every model runner.

Parses transcripts with the project's own rule parser (assistant.intent.parse)
and checks action + key entity (song/playlist/artist name), not word error
rate. Never imports anything from assistant/.env or assistant/.cache.
"""
from __future__ import annotations

import difflib
import re
import sys
from pathlib import Path

REPO_ROOT = Path(r"A:\pisha\memoir")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from assistant.intent import parse, Action  # noqa: E402

_WAKE_WORDS = ("alexa", "alexei", "alexia", "elexa", "bemor", "memor", "memore",
               "memoir", "hey memoir", "jarvis")


def strip_wake_word(text: str) -> str:
    """Strip a leading wake word / '<wake> | <command>' separator produced by
    the trigger+command two-pass transcription in production telemetry."""
    if "|" in text:
        # transcript format used by trigger_source=="stt": "<wake> | <command>"
        parts = text.split("|", 1)
        return parts[1].strip()
    t = text.strip()
    low = t.lower()
    for w in _WAKE_WORDS:
        if low.startswith(w):
            rest = t[len(w):]
            rest = re.sub(r"^[\s,.:;!?]+", "", rest)
            return rest
    return t


def normalize_name(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _split_title_artist(query: str) -> tuple:
    q = normalize_name(query)
    for sep in (" by ", " from ", " with "):
        if sep in q:
            title, artist = q.split(sep, 1)
            return title.strip(), artist.strip()
    return q, ""


def _fuzzy_match(a: str, b: str, cutoff: float = 0.72) -> bool:
    if not a or not b:
        return a == b
    if a == b or a in b or b in a:
        return True
    ratio = difflib.SequenceMatcher(None, a, b).ratio()
    return ratio >= cutoff


def name_matches(predicted_query: str, expected_query: str) -> bool:
    """Song title (+ artist, if present) match, allowing spelling variants
    like madhira/madira, seedhe maut/seedemoth, etc."""
    p_title, p_artist = _split_title_artist(predicted_query)
    e_title, e_artist = _split_title_artist(expected_query)
    if not _fuzzy_match(p_title, e_title):
        return False
    if e_artist:
        return _fuzzy_match(p_artist, e_artist)
    return True


def expected_intent(expected_transcript: str):
    return parse(strip_wake_word(expected_transcript))


def score_clip(predicted_transcript: str, expected_transcript: str) -> dict:
    pred = parse(strip_wake_word(predicted_transcript))
    exp = expected_intent(expected_transcript)

    command_ok = pred.action == exp.action
    name_ok = None
    if exp.action in (Action.PLAY_TRACK, Action.PLAY_PLAYLIST, Action.PLAY_ARTIST):
        name_ok = command_ok and name_matches(pred.query, exp.query)
    elif exp.action == Action.VOLUME:
        name_ok = command_ok and (pred.value == exp.value)
    else:
        name_ok = command_ok

    return {
        "predicted_transcript": predicted_transcript,
        "predicted_action": pred.action.value,
        "predicted_query": pred.query,
        "predicted_value": pred.value,
        "expected_transcript": expected_transcript,
        "expected_action": exp.action.value,
        "expected_query": exp.query,
        "expected_value": exp.value,
        "command_ok": bool(command_ok),
        "name_ok": bool(name_ok),
    }


_COMMANDY_ACTIONS = {Action.PAUSE, Action.RESUME, Action.NEXT, Action.PREV,
                     Action.RESTART, Action.VOLUME, Action.PLAY_PLAYLIST,
                     Action.PLAY_TRACK, Action.PLAY_ARTIST}


def is_hallucination(predicted_transcript: str) -> bool:
    """True if a non-speech/music clip was transcribed into something that
    parses as an actionable command (a false command, not just noise text)."""
    if not predicted_transcript.strip():
        return False
    pred = parse(strip_wake_word(predicted_transcript))
    return pred.action in _COMMANDY_ACTIONS


# ---------------------------------------------------------------------------
# Secondary, "production-realistic" name metric.  The assistant does not take
# the transcript literally: spotify.py recovers the title/artist boundary from
# the user's known artists and compares artists by consonant outline
# (_spoken_consonants), so "Seedemoth" still resolves to Seedhe Maut.  This
# metric mimics that so ASR spelling-by-sound is not counted as a failure.
# ---------------------------------------------------------------------------
_CONNECTORS = {"by", "from", "with", "of", "i", "is", "and", "feat", "featuring", "ft", "buy", "bye"}


def _skel(s: str) -> str:
    return "".join(ch for ch in normalize_name(s) if ch.isalpha() and ch not in "aeiouh")


def _skel_match(a: str, b: str, cutoff: float = 0.75) -> bool:
    sa, sb = _skel(a), _skel(b)
    if not sa or not sb:
        return normalize_name(a) == normalize_name(b)
    return sa == sb or difflib.SequenceMatcher(None, sa, sb).ratio() >= cutoff


def _split_known(query: str, known: list) -> tuple:
    words = normalize_name(query).split()
    best = ("", "", 0.0)
    for cut in range(1, len(words)):
        tail = " ".join(words[cut:])
        for artist in known:
            sa, sb = _skel(tail), _skel(artist)
            r = difflib.SequenceMatcher(None, sa, sb).ratio() if sa and sb else 0.0
            if r > best[2]:
                best = (" ".join(words[:cut]), artist, r)
    if best[2] >= 0.8:
        tw = best[0].split()
        while len(tw) > 1 and tw[-1] in _CONNECTORS:
            tw.pop()
        return " ".join(tw), best[1]
    return "", ""


def name_matches_phonetic(pred_query: str, exp_query: str, known: list) -> bool:
    p_title, p_artist = _split_title_artist(pred_query)
    e_title, e_artist = _split_title_artist(exp_query)
    if e_artist and not p_artist:
        p_title, p_artist = _split_known(pred_query, known)
    if e_artist and not p_artist:
        return False
    if not (_fuzzy_match(p_title, e_title) or _skel_match(p_title, e_title)):
        return False
    if e_artist:
        return _skel_match(p_artist, e_artist)
    return True


def name_ok_phonetic(pred_action: str, pred_query: str, pred_value, exp_action: str,
                     exp_query: str, exp_value, known: list) -> bool:
    if str(pred_action).upper() != str(exp_action).upper():
        return False
    if str(exp_action).upper() in ("PLAY_TRACK", "PLAY_PLAYLIST", "PLAY_ARTIST"):
        return name_matches_phonetic(pred_query or "", exp_query or "", known)
    if str(exp_action).upper() == "VOLUME":
        return pred_value == exp_value
    return True
