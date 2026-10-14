"""Text -> Intent. Pure functions, no I/O, fully unit-testable.

This module also decides *where the command ends*. The microphone cannot tell
"still talking to the assistant" from "now talking to a human" -- the voice, the
room and the volume are identical -- so the parser handles it instead: it scans
left to right, stops at the first complete intent, and bounds the query argument
at the first conversational marker.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Action(str, Enum):
    PAUSE = "PAUSE"
    RESUME = "RESUME"
    NEXT = "NEXT"
    PREV = "PREV"
    RESTART = "RESTART"
    VOLUME = "VOLUME"
    PLAY_PLAYLIST = "PLAY_PLAYLIST"
    PLAY_TRACK = "PLAY_TRACK"
    PLAY_ALBUM = "PLAY_ALBUM"
    PLAY_ARTIST = "PLAY_ARTIST"
    OPEN_APP = "OPEN_APP"
    SEARCH_BROWSER = "SEARCH_BROWSER"
    SHOW_TRACK = "SHOW_TRACK"
    SHOW_ALBUM = "SHOW_ALBUM"
    RESTART_ALBUM = "RESTART_ALBUM"
    START_JAM = "START_JAM"
    QUEUE = "QUEUE"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class Intent:
    action: Action
    query: str = ""
    value: Optional[int] = None      # for VOLUME
    matched_rule: str = ""
    source: str = "rule"             # rule | llm | none
    library_only: bool = False       # explicit personal-playlist constraint
    volume_mode: str = "absolute"   # absolute | relative_percent | relative_points
    volume_after_play: Optional[int] = None
    # Keep playing similar songs after a requested track/artist. Only "just
    # play X" / "play only X" / "play X only" turn it off.
    radio: bool = True
    # A source-grounded song/track type cue forbids changing the entity to an album.
    explicit_track: bool = False
    browser: str = "Opera"


# Filler and wake-word residue stripped before matching. Whisper-family models
# routinely transcribe the wake word itself plus hesitation sounds.
# The wake word is itself mis-transcribed ("pay memoir", "a memoir", "hey, my
# war"), so strip any leading token(s) up to and including the memoir-ish word
# rather than matching a fixed "hey memoir".
_WAKE_RESIDUE = r"^\s*\w*\s*(?:memoir|memoirs|memoire|mem\s*or|my\s*war|jarvis)\b"
_FILLER = r"(?:uh|um|erm|uhh|oh|like|please|can you|could you|i want to|i wanna|sir)"

# Markers that signal the speaker has stopped addressing the assistant. Used to
# bound an open-ended query so trailing conversation never reaches Spotify.
_END_MARKERS = (
    "yeah", "yea", "no", "nah", "so", "anyway", "wait", "actually",
    "okay", "ok", "alright", "and then", "but ", "because",
)

# Words that mean "just resume", not a search query.
# "play the song" once searched Spotify for a track literally named "the song".
_GENERIC_MUSIC = {"music", "something", "anything", "songs", "a song",
                  "the music", "some music", "my music", "tunes",
                  "the song", "this song", "that song", "some songs", "some song",
                  "the songs", "the track", "this track", "that track", "a track",
                  "my songs", "my song"}

# Verb-position mishearings of "play", repaired before parsing.
_PLAY_MISHEARD = r"^(?:lay|place|plate|pray|clay|played|plays|blame|flame|plain|plane)\b"

_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19,
    "zero": 0, "ten": 10, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100,
    "full": 100, "max": 100, "half": 50,
}


def normalize(text: str) -> str:
    """Lowercase, strip wake-word residue and filler, repair the verb."""
    t = text.lower().strip()
    t = re.sub(r"[^\w\s%']", " ", t)
    t = re.sub(_WAKE_RESIDUE, " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    # "play" is regularly transcribed as lay/place/plate/pray. Rewriting
    # it here keeps the query intact -- mapping these straight to an
    # action instead threw away everything after the verb.
    t = re.sub(_PLAY_MISHEARD, "play", t)
    return t


def _command_text(text: str) -> str:
    """Remove only leading hesitation/request framing, never query words."""
    t = text
    # Fillers are syntax, not a global stop-word list: ``like`` may be part of
    # a song title, so only consume them before the command verb.
    while True:
        updated = re.sub(r"^\s*" + _FILLER + r"(?:\s+|$)", "", t, count=1)
        if updated == t:
            return t.strip()
        t = updated


def explicit_track_requested(text: str) -> bool:
    """Conservatively retain a heard song/track constraint, outside artist credits."""
    scope = re.split(r"\s+(?:by|from|of|with)\s+", normalize(text), maxsplit=1)[0]
    return bool(re.search(r"\b(?:song|track)\b", scope))


def _track_type_query(query: str) -> str:
    """Only called/named marks a type boundary; other words may be a real title."""
    return re.sub(r"^(?:(?:the|a|this|that|my)\s+)?(?:song|track)\s+(?:called|named)\s+",
                  "", query, count=1).strip()


def _truncate_at_marker(text: str) -> str:
    """Cut the query at the first sign the speaker turned to someone else.

    "play something chill yeah no i told him tuesday" -> "something chill"
    """
    lowest = len(text)
    for marker in _END_MARKERS:
        # Match whole marker words; ``no`` must not truncate ``now``.
        marker_text = marker.strip()
        match = re.search(r"\s+" + re.escape(marker_text) + r"\b", text)
        if match:
            lowest = min(lowest, match.start())
    return text[:lowest].strip()


def _strip_article(value: str) -> str:
    """Drop a leading article from a playlist lookup ("the focus" -> "focus").

    Playlist names are matched against the user's library, where "the" is
    noise. Deliberately NOT applied to track queries: "play the smiths"
    must keep its article.
    """
    return re.sub(r"^(?:the|a|an)\s+", "", value).strip()


def _clean_query(raw: str) -> str:
    # A hesitation immediately after the verb is framing; later occurrences
    # belong to the requested title and must be retained.
    q = _command_text(raw.strip())
    q = _truncate_at_marker(q)
    # Trailing service names are noise: "play bohemian rhapsody on spotify"
    q = re.sub(r"\b(?:on|from|in|off of)\s+spotify\b.*$", "", q).strip()
    q = re.sub(r"\bfor me\b.*$", "", q).strip()
    return q


# What "queue <this>" can never mean: the song already playing or a
# placeholder. Such a request is not actionable, so it must not search Spotify
# for a track called "it".
_QUEUE_PLACEHOLDER = re.compile(
    r"(?:(?:the|a|this|that|another|some|my)\s+)?"
    r"(?:it|one|up|song|songs|track|tracks|music|something|anything|else|"
    r"(?:the\s+)?next(?:\s+(?:song|track|one))?)")
# The tail of an add-to-queue request may carry politeness, never a second command.
_QUEUE_TAIL = re.compile(r"(?:please|for me|now|thanks|thank you|yeah|ok|okay)\b.*")
# Another command after "and"/"then" is a compound request we do not run.
_COMPOUND_TAIL = re.compile(
    r"\b(?:and|then)\s+(?:pause|stop|skip|next|previous|resume|play|set|turn|volume|change)\b")


def _queue_query(raw: str) -> str:
    """The song named in a queue request, or "" when nothing actionable was said."""
    q = _clean_query(raw)
    if (not q or q in _GENERIC_MUSIC or _QUEUE_PLACEHOLDER.fullmatch(q)
            or re.search(r"\bplaylist\b", q) or _COMPOUND_TAIL.search(q)):
        return ""
    return q


def _parse_queue(t: str) -> Optional[Intent]:
    """"add X to the queue", "queue (up) X", "play X next/after this", "after this play X".

    Checked before the transport rules: "next play X" must not be a skip. A
    request that names no song is UNKNOWN rather than falling through to play.
    """
    m = re.match(
        r"^(?:add|put|throw|stick)\s+(?P<q>.+?)\s+(?:to|in|into|on|onto)\s+"
        r"(?:(?:the|my|our|a)\s+)?(?:play\s*)?queue\b\s*(?P<tail>.*)$", t)
    if m and (not m.group("tail") or _QUEUE_TAIL.fullmatch(m.group("tail"))):
        rule = "queue_add"
    else:
        m = re.match(r"^queue(?:\s+up)?\s+(?P<q>.+)$", t)
        rule = "queue_verb"
    if m is None:
        m = re.match(
            r"^play\s+(?P<q>.+?)\s+(?:next|(?:right\s+)?after\s+(?:this|that|the\s+current)"
            r"(?:\s+(?:one|song|track))?)$", t)
        rule = "queue_play_next"
    if m is None:
        m = re.match(
            r"^(?:after\s+(?:this|that|the\s+current)(?:\s+(?:one|song|track))?|next)\s+"
            r"play\s+(?P<q>.+)$", t)
        rule = "queue_after_this"
    if m is None:
        return None
    q = _queue_query(m.group("q"))
    if not q:
        return Intent(Action.UNKNOWN, source="none")
    return Intent(Action.QUEUE, query=q, matched_rule=rule)


def capability_domain(text: str) -> bool:
    """Desktop request admission, not permission to execute its arguments."""
    if not isinstance(text, str) or re.search(r"\b(?:codex|chat|conversation|thread)\b", text, re.I):
        return False
    value = text.strip()
    framing = r"(?:(?:please|can you|could you|would you|will you)\s+)*(?:please\s+)?(?:(?:i(?:'d| would)?\s+(?:like|want)\s+to)\s+)?"
    return bool(re.match(framing + r"(?:open|launch|show|view|see|search|look up|find|bring up|pull up)\b", value, re.I)
                or re.match(framing + r"(?:restart|replay|start over|start)\s+(?:the\s+)?(?:current|this)\s+album\b", value, re.I)
                or re.match(framing + r"(?:start|create|host)\s+(?:(?:a|the)\s+)?(?:spotify\s+)?jam\b", value, re.I)
                or re.match(framing + r"start\b.*\b(?:app|application)\b", value, re.I))


def capability_request(text: str, *, semantic: bool = False) -> Optional[Intent]:
    """Ground bounded desktop tools in a complete present request.

    These queries retain punctuation/case; they never use music's conversational
    tail truncation. Unsupported compounds are refused as a whole.
    """
    if not isinstance(text, str) or not text.strip() or len(text) > 400:
        return None
    source = text.strip()
    if re.search(r"[\x00-\x1f\x7f]|;", source):
        return None
    lead = r"(?:(?:please|can you|could you|would you|will you)\s+)*(?:please\s+)?"
    if semantic:
        lead += r"(?:(?:i(?:'d| would)?\s+(?:like|want)\s+to)\s+)?"

    # Jam controls are a single idempotent operation: start one if needed, or
    # display the existing invite. Keep the optional QR display phrasing
    # bounded so a second command cannot hitch a ride on this capability.
    jam_object = r"(?:(?:a|the)\s+)?(?:spotify\s+)?jam"
    qr = r"(?:the\s+)?qr(?:\s+code)?"
    jam_provider = r"(?:\s+(?:in|on)\s+spotify)?"
    jam_suffix = (
        r"(?:\s+and\s+show\s+(?:me\s+)?(?:the\s+)?"
        r"(?:qr(?:\s+code)?(?:\s+(?:of|for)\s+(?:the\s+)?(?:spotify\s+)?jam)?|"
        r"invite\s+code)(?:\s+on\s+screen)?)?"
    )
    match = re.fullmatch(lead + r"(?:start|create|host)\s+" + jam_object
                         + jam_provider + jam_suffix + r"[.!?]*", source, re.I)
    if match:
        return Intent(Action.START_JAM, matched_rule="start_spotify_jam")
    jam_show = (
        lead + r"(?:show|open)(?:\s+me)?\s+(?:(?:the\s+)?(?:spotify\s+)?jam\s+"
        r"(?:qr(?:\s+code)?|invite\s+code)(?:\s+(?:in|on)\s+spotify)?|"
        + qr + r"\s+(?:of|for)\s+(?:the\s+)?(?:spotify\s+)?jam)"
        + r"(?:\s+on\s+screen)?[.!?]*"
    )
    match = re.fullmatch(jam_show, source, re.I)
    if match:
        return Intent(Action.START_JAM, matched_rule="show_spotify_jam_invite")

    def has_command_tail(query: str) -> bool:
        # A whole quoted query is unambiguously data, even when it contains
        # words that would otherwise mark a second spoken command.
        if re.fullmatch(r"(?:\"[^\"]+\"|'[^']+')", query.strip()):
            return False
        return bool(re.search(
            r"\b(?:and|then)\s+(?:(?:please|also|can you|could you|would you|"
            r"do not|don't|never)\s+)*(?:pause|stop|skip|next|previous|resume|play|set|turn|"
            r"volume|change|open|launch|close|delete|send|start|restart|show|view|find|"
            r"bring\s+up|pull\s+up|queue|submit|ask|tell)\b", query, re.I))

    # Opening Opera and starting a search is one browser operation. Parse it
    # before generic app opening so the search query is never swallowed as an
    # application name.
    match = re.fullmatch(
        lead + r"open\s+(?:(?:a\s+)?tab\s+in\s+)?opera(?:\s+gx)?\s+and\s+"
        r"search(?:\s+for)?\s+(.+)", source, re.I)
    if match:
        query = match.group(1).strip()
        if not query or has_command_tail(query):
            return None
        return Intent(Action.SEARCH_BROWSER, query=query, browser="Opera", matched_rule="search_browser")

    # Search queries and song titles can contain conjunctions as ordinary
    # text. Other capability types keep the conservative compound guard below.
    search_verbs = r"search(?:\s+for)?" if not semantic else r"(?:search(?:\s+for)?|look up|find)"
    match = re.fullmatch(
        lead + search_verbs + r"\s+(.+?)\s+(?:in|on|using|with)\s+"
        r"(?:the\s+)?(opera(?:\s+gx)?)(?:\s+browser)?[.!?]*", source, re.I)
    if match:
        query = match.group(1).strip()
        if query and not has_command_tail(query):
            return Intent(Action.SEARCH_BROWSER, query=query, browser="Opera", matched_rule="search_browser")
        return None
    match = re.fullmatch(
        lead + r"(?:in|on|using)\s+(opera(?:\s+gx)?)(?:\s+browser)?\s+" + search_verbs + r"\s+(.+)",
        source, re.I)
    if match:
        query = match.group(2).strip()
        if query and not has_command_tail(query):
            return Intent(Action.SEARCH_BROWSER, query=query, browser="Opera", matched_rule="search_browser")
        return None

    show_verbs = r"(?:show|open)" if not semantic else r"(?:show|open|see|view|bring up|pull up)"
    match = re.fullmatch(
        lead + show_verbs + r"\s+(?:me\s+)?(?:the\s+)?(song|track|album)\s+"
        r"(?:(?:called|named)\s+)?(.+)", source, re.I)
    if match:
        query = re.sub(r"\s+(?:in|on)\s+spotify[.!?]*$", "", match.group(2), flags=re.I).strip()
        if not query or has_command_tail(query):
            return None
        return Intent(Action.SHOW_ALBUM if match.group(1).casefold() == "album" else Action.SHOW_TRACK,
                      query=query, matched_rule="show_spotify")

    if re.search(r"\b(?:and|then)\b", source, re.I):
        return None
    value = re.sub(r"[.!?]+$", "", source).strip()
    app_verbs = r"(?:open|launch)" if not semantic else r"(?:open|launch|bring up|pull up)"
    match = re.fullmatch(lead + app_verbs + r"\s+(?:(?:the\s+)?(?:app|application)\s+)?(.+?)(?:\s+(?:app|application))?", value, re.I)
    if match:
        query = match.group(1).strip()
        # Codex/chat management retains its reviewed controller boundary.
        if not re.search(r"\b(?:codex|chat|conversation|song|track|album)\b", query, re.I):
            return Intent(Action.OPEN_APP, query=query, matched_rule="open_app")
    match = re.fullmatch(lead + r"start\s+(?:the\s+)?(?:(?:app|application)\s+(.+)|(.+?)\s+(?:app|application))", value, re.I)
    if match:
        return Intent(Action.OPEN_APP, query=(match.group(1) or match.group(2)).strip(), matched_rule="open_app")
    if semantic:
        match = re.fullmatch(lead + r"(?:get\s+(.+?)\s+(?:up|running|open)|bring\s+(.+?)\s+up)", value, re.I)
        if match:
            query = (match.group(1) or match.group(2)).strip()
            if not re.search(r"\b(?:codex|chat|conversation|song|track|album)\b", query, re.I):
                return Intent(Action.OPEN_APP, query=query, matched_rule="open_app")
    match = re.fullmatch(lead + r"(?:restart|replay|start over|start)\s+(?:the\s+)?(?:current|this)\s+album(?:\s+(?:from|at)\s+the\s+(?:start|beginning))?", value, re.I)
    if match:
        return Intent(Action.RESTART_ALBUM, matched_rule="restart_album", radio=False)
    return None


def parse(text: str) -> Intent:
    """Ordered rule table. First complete match wins; the tail is discarded."""
    if re.match(r"^\s*(?:set\s+)?volume\s+-\d", text.lower()):
        return Intent(Action.UNKNOWN, source="none")
    capability = capability_request(text)
    if capability is not None:
        return capability
    if capability_domain(text):
        return Intent(Action.UNKNOWN, source="none", matched_rule="unsupported_desktop")
    t = _command_text(normalize(text))
    if not t:
        return Intent(Action.UNKNOWN, source="none")
    # "just play X" asks for exactly one song (no radio after it). The word is
    # only an opt-out before the verb; "play just the two of us" is a title.
    radio, opt_out = True, ""
    m = re.match(r"^just\s+((?:play|start|put on|throw on)\b.*)$", t)
    if m:
        t, radio, opt_out = m.group(1), False, "_just"
    # A playback request may also specify the resulting Windows volume. Keep
    # the two arguments separate so "full volume" is not an artist suffix.
    play_volume = None
    if re.match(r"^(?:play|start|put on|throw on)\b", t):
        volume_tail = re.search(
            r"\s+(?:(?:at|on|with|and)\s+)?"
            r"(?P<level>full|max|half|\d{1,3})\s*(?:%|percent)?\s+volume$",
            t,
        )
        if volume_tail is None:
            volume_tail = re.search(
                r"\s+volume\s+(?:to\s+)?(?P<level>full|max|half|\d{1,3})\s*(?:%|percent)$",
                t,
            )
        if volume_tail:
            volume_intent = _parse_volume("volume " + volume_tail.group('level'))
            play_volume = volume_intent.value if volume_intent is not None else None
            if play_volume is None:
                return Intent(Action.UNKNOWN, source="none")
            if play_volume > 100:
                return Intent(Action.UNKNOWN, source="none")
            t = t[:volume_tail.start()].strip()
    if re.match(r"^(?:do not|don t|dont|never|not)\b", t):
        return Intent(Action.UNKNOWN, source="none")
    if re.match(r"^(?:pause|stop|skip|next|previous|resume)\b.*\b(?:then|and)\s+(?:pause|stop|skip|next|previous|resume|play)\b", t):
        return Intent(Action.UNKNOWN, source="none")

    # "play only X" / "play X only" also mean one song. "only" can open a real
    # title ("Only Girl"), so the rule is tagged and the executor first checks
    # whether the library holds the title with the word kept.
    m = re.match(r"^(play(?:ed|ing|s)?|start|put on|throw on)\s+only\s+(.+)$", t)
    if m:
        t, radio, opt_out = "{} {}".format(m.group(1), m.group(2)), False, "_only_before"
    else:
        m = re.match(r"^(play(?:ed|ing|s)?|start|put on|throw on)\s+(.+?)\s+only$", t)
        if m:
            t, radio, opt_out = "{} {}".format(m.group(1), m.group(2)), False, "_only_after"

    queued = _parse_queue(t)
    if queued is not None:
        return queued

    # "play next" / "play the next song" are the skip command; as a track
    # query they searched Spotify for a song literally called "next".
    if re.fullmatch(r"play\s+(?:the\s+)?next(?:\s+(?:song|track|one))?", t):
        return Intent(Action.NEXT, matched_rule="next")

    # --- transport: closed vocabulary, checked first ---
    if re.match(r"^(?:pause|stop|shut up|be quiet|silence)\b", t):
        return Intent(Action.PAUSE, matched_rule="pause")

    if re.match(r"^(?:next|skip|forward)\b", t):
        return Intent(Action.NEXT, matched_rule="next")

    if re.match(r"^(?:previous|go back|last song|back a song|rewind)\b", t):
        return Intent(Action.PREV, matched_rule="previous")

    if re.fullmatch(r"(?:restart|replay|start over)(?:\s+(?:the|this|current)\s+(?:song|track))?", t):
        return Intent(Action.RESTART, matched_rule="restart")

    # --- volume, before the play rules so "turn it up" is not a play query ---
    vol = _parse_volume(t)
    if vol is not None:
        return vol

    # A named replacement is a track request, not an unqualified skip.
    m = re.fullmatch(
        r"(?:change|switch)\s+(?:(?:the|this|my)\s+)?(?:song|track)\s+to\s+(.+)",
        t,
    )
    if m:
        query = _clean_query(m.group(1))
        if query:
            return Intent(Action.PLAY_TRACK, query=query, matched_rule="change_track", explicit_track=True)

    if re.match(r"^(?:resume|continue|unpause|keep going|carry on|keep playing)\b", t):
        return Intent(Action.RESUME, matched_rule="resume")

    m = re.match(r"^(?:play(?:ed|ing|s)?|start|put on|throw on)\s+"
                 r"(?:(?:the|a|this|that|my)\s+)?(?:song|track)\s+(?:called|named)\s+(.+)$", t)
    if m:
        query = _clean_query(m.group(1))
        if query:
            return Intent(Action.PLAY_TRACK, query=query, matched_rule="explicit_track" + opt_out,
                          volume_after_play=play_volume, radio=radio, explicit_track=True)

    # Explicit albums remain albums even when lookup fails. Queue stays songs.
    m = re.match(r"^(?:play(?:ed|ing|s)?|start|put on|throw on)\s+(?:the\s+)?album\s+(.+)$", t)
    if m is None:
        m = re.match(r"^(?:play(?:ed|ing|s)?|start|put on|throw on)\s+(.+?)\s+album(?:\s+by\s+(.+))?$", t)
    if m:
        query = _clean_query(m.group(1))
        if m.lastindex == 2 and m.group(2):
            query += " by " + _clean_query(m.group(2))
        if query:
            return Intent(Action.PLAY_ALBUM, query=query, matched_rule="play_album" + opt_out,
                          volume_after_play=play_volume, radio=False)

    # "start"/"put on" are natural synonyms for "play" and were being
    # rejected outright: "start the music" went to UNKNOWN.
    m = re.match(r"^(?:start|put on|throw on)\s+(.*)$", t)
    if m:
        rest = _clean_query(m.group(1))
        if not explicit_track_requested(text):
            rest = re.sub(r"^(?:the|some|my)\s+", "", rest).strip()
        if not rest or rest in _GENERIC_MUSIC:
            return Intent(Action.RESUME, matched_rule="start_bare", volume_after_play=play_volume)
        return Intent(Action.PLAY_TRACK, query=_track_type_query(rest), matched_rule="start_track" + opt_out,
                      volume_after_play=play_volume, radio=radio,
                      explicit_track=explicit_track_requested(text))

    # --- playlist, before the generic play fallthrough ---
    m = re.match(r"^play(?:ed|ing|s)?\s+(?:(my|the)\s+)?playlist\s+(.+)$", t)
    if m:
        q = _strip_article(_clean_query(m.group(2)))
        if q:
            return Intent(Action.PLAY_PLAYLIST, query=q, matched_rule="playlist_prefix",
                          library_only=m.group(1) == 'my', volume_after_play=play_volume)

    # "my" is routinely misheard as "by"/"me" -- verbatim from telemetry:
    # "play my beyond playlist" came back as "Play by beyond playlist".
    # Safe to absorb here because this rule requires the literal word
    # "playlist", so "play red by seedhe maut" cannot reach it.
    m = re.match(r"^play(?:ed|ing|s)?\s+(?:(my|by|mah|me)\s+)?(.+?)\s+playlist\b", t)
    if m:
        q = _strip_article(_clean_query(m.group(2)))
        if q:
            return Intent(Action.PLAY_PLAYLIST, query=q, matched_rule="playlist_suffix",
                          library_only=bool(m.group(1)), volume_after_play=play_volume)

    # Only when the possessive is the FIRST word after "play". "play red by
    # seedhe maut" starts "play red", so it never reaches this rule and stays a
    # track search -- which is what "<track> by <artist>" should be.
    m = re.match(r"^play(?:ed|ing|s)?\s+(?:my|by|mah|me)\s+(.+)$", t)
    if m and _clean_query(t.split(None, 1)[1]) not in _GENERIC_MUSIC:
        q = _clean_query(m.group(1))
        if q:
            return Intent(Action.PLAY_PLAYLIST, query=q, matched_rule="playlist_my", library_only=True,
                          volume_after_play=play_volume)

    # --- bare play = resume; play <X> = track ---
    m = re.match(r"^play(?:ed|ing|s)?\b\s*(.*)$", t)
    if m:
        rest = _clean_query(m.group(1))
        if not rest or rest in _GENERIC_MUSIC:
            return Intent(Action.RESUME, matched_rule="play_bare", volume_after_play=play_volume)
        return Intent(Action.PLAY_TRACK, query=_track_type_query(rest), matched_rule="play_track" + opt_out,
                      volume_after_play=play_volume, radio=radio,
                      explicit_track=explicit_track_requested(text))

    return _fuzzy_transport(t)


# Single-word transport commands have no redundancy: one phoneme error kills
# them, and the ASR genuinely produces "polls"/"paws" for "pause" and
# "pain"/"pay" for "play". Longer commands survive because context carries them.
# Near-miss matching is safe here only because this vocabulary is tiny and
# closed -- it is deliberately NOT applied to song or playlist queries.
_TRANSPORT_KEYWORDS = {
    "pause": Action.PAUSE,
    "stop": Action.PAUSE,
    "next": Action.NEXT,
    "skip": Action.NEXT,
    "previous": Action.PREV,
    "resume": Action.RESUME,
    "play": Action.RESUME,
}
# Phonetic confusions observed in real Moonshine output. Character-similarity
# scoring cannot catch these -- "polls" vs "pause" scores 0.46 -- because they
# are acoustic near-misses, not spelling near-misses. This table is meant to
# grow from logged UNKNOWN transcripts rather than from guesswork.
_PHONETIC_ALIASES = {
    "polls": Action.PAUSE,
    "paws": Action.PAUSE,
    "pours": Action.PAUSE,
    "pos": Action.PAUSE,
    "cause": Action.PAUSE,
    "pain": Action.RESUME,
    "sip": Action.NEXT,
    "ship": Action.NEXT,
}

_FUZZY_CUTOFF = 0.8
_FUZZY_MAX_TOKENS = 3


def _fuzzy_transport(t: str) -> Intent:
    """Last resort before UNKNOWN: match a near-miss on a short transport command."""
    import difflib

    tokens = [tok for tok in t.split() if tok not in {"hey", "memoir", "a", "the"}]
    # A fuzzy word inside a sentence is not authorization for that action.
    if len(tokens) != 1:
        return Intent(Action.UNKNOWN, source="none")

    for token in tokens:
        if token in _PHONETIC_ALIASES:
            return Intent(
                _PHONETIC_ALIASES[token],
                matched_rule="alias_" + token,
                source="fuzzy",
            )

    for token in tokens:
        if len(token) < 4:  # too short to match reliably
            continue
        match = difflib.get_close_matches(
            token, _TRANSPORT_KEYWORDS, n=1, cutoff=_FUZZY_CUTOFF
        )
        if match:
            return Intent(
                _TRANSPORT_KEYWORDS[match[0]],
                matched_rule="fuzzy_" + match[0],
                source="fuzzy",
            )

    return Intent(Action.UNKNOWN, source="none")


def _parse_volume(t: str) -> Optional[Intent]:
    def number(raw: str) -> Optional[int]:
        raw = re.sub(r"\s+percent$", "", raw.strip().replace("%", "")).strip()
        if re.fullmatch(r"\d{1,3}", raw):
            return int(raw)
        words = raw.split()
        if words == ["one", "hundred"]:
            return 100
        if len(words) == 1:
            return _NUMBER_WORDS.get(words[0])
        if len(words) == 2:
            tens, units = (_NUMBER_WORDS.get(w) for w in words)
            if tens is not None and units is not None and tens in range(20, 100, 10) and units in range(1, 10):
                return tens + units
        return None

    m = re.fullmatch(r"(full|max|half)\s+volume", t)
    if m:
        return Intent(Action.VOLUME, value=number(m.group(1)),
                      matched_rule="volume_absolute", volume_mode="absolute")

    # Explicit relative changes.  Keep the direction in query for the
    # existing executor, and distinguish percentage from percentage-point
    # changes in volume_mode.
    m = re.match(r"(?:increase|decrease|raise|lower|reduce|set|put|change)\s+(?:the\s+)?volume\s+(?:to|at)\s+(.+?)(?:\s+percent|\s*%)?$", t)
    if m:
        value = number(m.group(1))
        if value is not None and 0 <= value <= 100:
            return Intent(Action.VOLUME, value=value, matched_rule="volume_absolute", volume_mode="absolute")
        return Intent(Action.UNKNOWN, source="none")
    # "turn down volume to 20%", "turn it up to 80", "bring the volume down to
    # thirty": the target level is absolute even though a direction is spoken.
    m = re.match(r"(?:turn|bring|put|take|get|make)(?:\s+(?:it|the|volume|up|down))+\s+to\s+(.+?)(?:\s+percent|\s*%)?$", t)
    if m:
        value = number(m.group(1))
        if value is not None and 0 <= value <= 100:
            return Intent(Action.VOLUME, value=value, matched_rule="volume_absolute", volume_mode="absolute")
        return Intent(Action.UNKNOWN, source="none")
    m = re.match(r"(?:turn\s+)?(?:the\s+)?volume\s+(up|down)\s+by\s+(.+?)\s+(?:points?|steps?)$", t)
    if m:
        value = number(m.group(2))
        if value is not None and 0 <= value <= 100:
            return Intent(Action.VOLUME, value=value, query=m.group(1), matched_rule="volume_relative_points", volume_mode="relative_points")
        return Intent(Action.UNKNOWN, source="none")
    m = re.match(r"(increase|decrease|raise|lower|reduce|bring|turn|take)\s+(?:(up|down)\s+)?(?:the\s+)?volume\s+(?:(up|down)\s+)?by\s+(.+?)(?:\s+percent|\s*%)?$", t)
    if m:
        value = number(m.group(4))
        if value is not None and 0 <= value <= 100:
            explicit = m.group(2) or m.group(3)
            direction = explicit or ("up" if m.group(1) in {"increase", "raise"} else "down")
            return Intent(Action.VOLUME, value=value, query=direction, matched_rule="volume_relative_percent", volume_mode="relative_percent")
        return Intent(Action.UNKNOWN, source="none")

    m = re.fullmatch(r"(increase|decrease|raise|lower|reduce)\s+(?:the\s+)?volume(?:\s+(up|down))?", t)
    if m:
        direction = m.group(2) or ("up" if m.group(1) in {"increase", "raise"} else "down")
        return Intent(Action.VOLUME, query=direction, matched_rule="volume_step")

    if re.match(r"(?:volume\s+up|turn\s+it\s+up|louder|turn\s+up)\b", t):
        return Intent(Action.VOLUME, value=None, query="up", matched_rule="volume_up")
    if re.match(r"(?:volume\s+down|turn\s+it\s+down|quieter|softer|turn\s+down)\b", t):
        return Intent(Action.VOLUME, value=None, query="down", matched_rule="volume_down")

    m = re.match(r"(?:set\s+)?volume\s+(?:to\s+)?(.+)$", t)
    if m:
        value = number(m.group(1))
        if value is not None and 0 <= value <= 100:
            return Intent(Action.VOLUME, value=value, matched_rule="volume_absolute", volume_mode="absolute")
        return Intent(Action.UNKNOWN, source="none")

    return None
