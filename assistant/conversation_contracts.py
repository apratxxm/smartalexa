"""Typed, capability-neutral contracts for the conversation interpreter.

This module intentionally owns a local capability registry. It does not import
the legacy brain, perform I/O, authorize operations, or execute requests.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .intent import Action, Intent, _NUMBER_WORDS


@dataclass(frozen=True)
class Transcript:
    """Final turn plus app-approved prior user text for entity grounding only."""
    text: str
    turn_id: str = ""
    truncated: bool = False
    grounding_sources: tuple[str, ...] = ()


@dataclass(frozen=True)
class Step:
    id: str
    capability: str
    arguments: dict[str, Any]
    after: tuple[str, ...] = ()


@dataclass(frozen=True)
class Decision:
    kind: str
    text: str = ""
    steps: tuple[Step, ...] = ()
    evidence_quotes: tuple[str, ...] = ()
    reason: str = ""
    answer: str = ""
    answer_basis: str = ""


@dataclass(frozen=True)
class ToolResult:
    operation_id: str
    status: str
    data: dict[str, Any]
    error: str = ""


class ContractError(ValueError):
    """A model output or typed value violates the conversation contract."""


def _schema(description: str, properties: dict[str, dict[str, Any]],
            required: tuple[str, ...] = ()) -> dict[str, Any]:
    return {
        "description": description,
        "type": "object",
        "properties": properties,
        "required": list(required),
        "additionalProperties": False,
    }


_STR = lambda maximum, minimum=1: {"type": "string", "minLength": minimum, "maxLength": maximum}
_TITLE_ARTIST = {
    "title": _STR(120),
    "artist": _STR(80, 0),
}

# Names and argument shapes follow assistant.brain.TOOLS, except the legacy
# ignore tool and compound volume_after_play field are deliberately absent.
CAPABILITIES: dict[str, dict[str, Any]] = {
    "open_app": _schema("Open an installed application by its heard name; never pass a path, app ID, or command.", {"name": _STR(120)}, ("name",)),
    "search_browser": _schema("Search a heard query in an Opera browser tab.", {"query": _STR(300), "browser": {"type": "string", "enum": ["Opera"]}}, ("query", "browser")),
    "show_track": _schema("Show a song query in Spotify without playing it.", {"query": _STR(300)}, ("query",)),
    "show_album": _schema("Show an album query in Spotify without playing it.", {"query": _STR(300)}, ("query",)),
    "restart_album": _schema("Restart the currently playing album context; does not search a named album.", {}),
    "start_jam": _schema("Start a Spotify Jam or show its invite QR/code when explicitly requested.", {}),
    "play_song": _schema("Play a specific heard song; artist may be empty if not spoken.", {**_TITLE_ARTIST, "radio": {"type": "boolean"}}, ("title", "artist")),
    "play_artist": _schema("Play music by a heard artist when no song title was given.", {"artist": _STR(80), "radio": {"type": "boolean"}}, ("artist",)),
    "play_album": _schema("Play a specifically requested album, preserving album context.", {**_TITLE_ARTIST}, ("title", "artist")),
    "queue_song": _schema("Queue a specific heard song without interrupting current playback.", _TITLE_ARTIST, ("title", "artist")),
    "play_playlist": _schema("Play a heard playlist; mine preserves the personal-library constraint.", {"name": _STR(100), "mine": {"type": "boolean"}}, ("name", "mine")),
    "pause": _schema("Pause music.", {}),
    "resume": _schema("Resume music.", {}),
    "next_track": _schema("Skip to the next track.", {}),
    "previous_track": _schema("Go to the previous track.", {}),
    "restart_track": _schema("Restart the current track.", {}),
    "set_volume": _schema("Set Windows master volume to an exact level from 0 to 100.", {"level": {"type": "integer", "minimum": 0, "maximum": 100}}, ("level",)),
    "change_volume": _schema("Change Windows master volume by points or percent of its current level; amount 0 means use the configured small default step.", {"direction": {"type": "string", "enum": ["up", "down"]}, "amount": {"type": "integer", "minimum": 0, "maximum": 100}, "unit": {"type": "string", "enum": ["points", "percent"]}}, ("direction", "amount", "unit")),
    "codex_request": _schema("Delegate the full original transcript to Codex; integration must separately authorize it.", {"request": _STR(12000)}, ("request",)),
}

def _branch(required: tuple[str, ...], properties: dict[str, Any]) -> dict[str, Any]:
    return {"type": "object", "required": list(required), "properties": properties,
            "additionalProperties": False}


_QUOTES = {"type": "array", "items": _STR(500), "maxItems": 5}
_REQUIRED_QUOTES = {**_QUOTES, "minItems": 1}
_PLAN_STEPS = {
    "type": "array", "minItems": 1, "maxItems": 3,
    "items": {"oneOf": [{
        "type": "object", "required": ["id", "capability", "arguments", "after"],
        "properties": {
            "id": {"type": "string", "pattern": "^[A-Za-z0-9_-]{1,24}$"},
            "capability": {"const": name},
            "arguments": schema,
            "after": {"type": "array", "items": {"type": "string", "pattern": "^[A-Za-z0-9_-]{1,24}$"}, "uniqueItems": True},
        }, "additionalProperties": False,
    } for name, schema in CAPABILITIES.items()]},
}
DECISION_SCHEMA: dict[str, Any] = {
    "oneOf": [
        _branch(("kind", "answer", "basis", "evidence_quotes"), {
            "kind": {"const": "answer"}, "answer": _STR(1200),
            "basis": {"const": "general_knowledge"}, "evidence_quotes": _REQUIRED_QUOTES}),
        _branch(("kind", "question", "evidence_quotes"), {
            "kind": {"const": "clarify"}, "question": _STR(500), "evidence_quotes": _QUOTES}),
        _branch(("kind", "steps", "evidence_quotes"), {
            "kind": {"const": "plan"}, "steps": _PLAN_STEPS,
            "evidence_quotes": _REQUIRED_QUOTES}),
        _branch(("kind", "steps", "evidence_quotes", "answer", "answer_basis"), {
            "kind": {"const": "plan"}, "steps": _PLAN_STEPS,
            "evidence_quotes": _REQUIRED_QUOTES, "answer": _STR(1200),
            "answer_basis": {"const": "general_knowledge"}}),
        _branch(("kind", "reason", "evidence_quotes"), {
            "kind": {"const": "no_request"}, "reason": _STR(500), "evidence_quotes": _QUOTES}),
        _branch(("kind", "reason", "evidence_quotes"), {
            "kind": {"const": "unsupported"}, "reason": _STR(500), "evidence_quotes": _QUOTES}),
    ]
}


_ID = re.compile(r"^[A-Za-z0-9_-]{1,24}$")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_GROUND_FIELDS = {"title", "artist", "name", "query"}


def _normalized_tokens(value: str) -> tuple[str, ...]:
    # Punctuation and whitespace are equivalent separators for grounding.
    chars = []
    for char in value.casefold():
        chars.append(char if char.isalnum() else " ")
    return tuple(" ".join("".join(chars).split()).split())


def _is_grounded(value: str, transcript: str, grounding_sources: tuple[str, ...] = ()) -> bool:
    needle = _normalized_tokens(value)
    if not needle:
        return False
    for source in (transcript, *grounding_sources):
        haystack = _normalized_tokens(source)
        if any(haystack[i:i + len(needle)] == needle
               for i in range(len(haystack) - len(needle) + 1)):
            return True
    return False


def _number_grounded(value: int, current_transcript: str) -> bool:
    """Match only a complete, supported numeric run in the current turn."""
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 100:
        return False
    scales = {"thousand", "million", "billion", "trillion", "quadrillion", "quintillion", "dozen", "score"}
    connectors = {"point", "dot", "decimal", "percent", "minus", "negative", "plus", "positive"}
    punctuation = set(".,+-/:_") | set("\u2212\u2010\u2011\u2012\u2013\u2014\u2015") | set("$€£%")
    candidates = set(_NUMBER_WORDS) | scales | connectors
    tokens = []
    for match in re.finditer(r"\d+|[^\W\d_]+|_|[^\w\s]", current_transcript, re.IGNORECASE):
        raw = match.group(0)
        word = raw.casefold()
        if raw.isdigit():
            kind = "digits"
        elif raw.isalpha():
            kind = "word"
        else:
            kind = "punct"
        tokens.append((word, kind, match.start(), match.end()))

    def is_number_component(token: tuple[str, str, int, int]) -> bool:
        word, kind, _, _ = token
        return kind == "digits" or (kind == "word" and (word in _NUMBER_WORDS or word in scales))

    def is_candidate(index: int) -> bool:
        token = tokens[index]
        word, kind, _, _ = token
        if word == "and" and kind == "word":
            left = index - 1
            if left >= 0 and tokens[left][0] in {"percent", "%"}:
                left -= 1
            right = index + 1
            return (left >= 0 and right < len(tokens)
                    and is_number_component(tokens[left])
                    and is_number_component(tokens[right]))
        return (kind == "digits" or (kind == "word" and word in candidates)
                or (kind == "punct" and word in punctuation))

    def parse_run(run: list[tuple[str, str, int, int]]) -> int | None:
        words = [token[0] for token in run]
        # Sentence punctuation and a terminal percent sign do not extend the number.
        while words and words[-1] in {".", ","}:
            words.pop()
        if words and words[-1] in {"%", "percent"}:
            words.pop()
        while words and words[-1] in {".", ","}:
            words.pop()
        if not words:
            return None
        digit_tokens = [token for token in run if token[1] == "digits"]
        if digit_tokens:
            if len(words) != 1 or len(digit_tokens) != 1 or words[0] != digit_tokens[0][0]:
                return None
            if len(words[0]) > 3:
                return None
            number = int(words[0])
            return number if 0 <= number <= 100 else None

        # A hyphen is valid only between two number words. All other internal
        # punctuation/connectors invalidate the entire run, preventing fragments.
        normalized_words = []
        for index, token in enumerate(words):
            if token == "-":
                if (index == 0 or index + 1 >= len(words)
                        or words[index - 1] not in _NUMBER_WORDS
                        or words[index + 1] not in _NUMBER_WORDS):
                    return None
                continue
            if token not in _NUMBER_WORDS:
                return None
            normalized_words.append(token)
        if len(normalized_words) == 1:
            return _NUMBER_WORDS[normalized_words[0]]
        if len(normalized_words) == 2:
            first, second = (_NUMBER_WORDS[word] for word in normalized_words)
            if normalized_words == ["one", "hundred"]:
                return 100
            if 20 <= first <= 90 and first % 10 == 0 and 1 <= second <= 9:
                return first + second
        return None

    index = 0
    while index < len(tokens):
        if not is_candidate(index):
            index += 1
            continue
        end = index + 1
        while end < len(tokens) and is_candidate(end):
            gap = current_transcript[tokens[end - 1][3]:tokens[end][2]]
            if gap and not gap.isspace():
                break
            end += 1
        run = tokens[index:end]
        left_joined = (index > 0 and tokens[index - 1][1] in {"digits", "word"}
                       and current_transcript[tokens[index - 1][3]:tokens[index][2]] == "")
        right_joined = (end < len(tokens) and tokens[end][1] in {"digits", "word"}
                        and current_transcript[tokens[end - 1][3]:tokens[end][2]] == "")
        number = None if left_joined or right_joined else parse_run(run)
        if number == value:
            return True
        index = end
    return False


def _valid_string(value: Any, field: str, maximum: int, minimum: int = 1) -> str:
    if not isinstance(value, str):
        raise ContractError(f"{field} must be a string")
    if _CONTROL.search(value):
        raise ContractError(f"{field} contains control characters")
    if len(value) > maximum:
        raise ContractError(f"{field} length must be at most {maximum}")
    value = value.strip()
    if not minimum <= len(value) <= maximum:
        raise ContractError(f"{field} length must be {minimum}..{maximum}")
    return value


def _check_arguments(capability: str, args: Any, transcript: str,
                     grounding_sources: tuple[str, ...] = ()) -> dict[str, Any]:
    schema = CAPABILITIES[capability]
    if not isinstance(args, dict):
        raise ContractError("step arguments must be an object")
    props = schema["properties"]
    if set(args) - set(props):
        raise ContractError("unknown capability argument")
    missing = set(schema["required"]) - set(args)
    if missing:
        raise ContractError("missing capability argument")
    result: dict[str, Any] = {}
    for key, raw in args.items():
        definition = props[key]
        kind = definition.get("type")
        if kind == "integer":
            if isinstance(raw, bool) or not isinstance(raw, int):
                raise ContractError(f"{key} must be an integer")
            if not definition.get("minimum", raw) <= raw <= definition.get("maximum", raw):
                raise ContractError(f"{key} is out of range")
            if capability == "set_volume" and key == "level" and not _number_grounded(raw, transcript):
                raise ContractError("volume level is not grounded in the current transcript")
            if capability == "change_volume" and key == "amount" and raw != 0 \
                    and not _number_grounded(raw, transcript):
                raise ContractError("volume amount is not grounded in the current transcript")
        elif kind == "boolean":
            if not isinstance(raw, bool):
                raise ContractError(f"{key} must be a boolean")
        elif kind == "string":
            if capability == "codex_request" and key == "request":
                if not isinstance(raw, str) or raw != transcript:
                    raise ContractError("Codex request must equal the original transcript")
                result[key] = raw
                continue
            if "enum" in definition:
                if not isinstance(raw, str) or raw not in definition["enum"]:
                    raise ContractError(f"{key} is not an allowed value")
                value = raw
            else:
                value = _valid_string(raw, key, definition["maxLength"], definition.get("minLength", 1))
                if (key in _GROUND_FIELDS and value != ""
                        and not _is_grounded(value, transcript, grounding_sources)):
                    raise ContractError(f"{key} is not grounded in the transcript")
            result[key] = value
            continue
        else:
            raise ContractError("invalid capability schema")
        result[key] = raw
    return result


def validate_decision(payload: dict[str, Any], transcript: Transcript) -> Decision:
    """Validate structured model output against the source turn and registry."""
    if not isinstance(transcript, Transcript) or not isinstance(transcript.text, str):
        raise ContractError("transcript must contain text")
    if transcript.truncated:
        raise ContractError("truncated transcripts cannot be interpreted")
    if len(transcript.text) > 12000 or not transcript.text.strip():
        raise ContractError("transcript must contain 1..12000 characters")
    if _CONTROL.search(transcript.text):
        raise ContractError("transcript contains control characters")
    sources = transcript.grounding_sources
    if not isinstance(sources, tuple) or len(sources) > 2:
        raise ContractError("grounding_sources must be a tuple of at most two strings")
    for source in sources:
        if not isinstance(source, str):
            raise ContractError("grounding sources must be strings")
        if not 1 <= len(source) <= 1000:
            raise ContractError("grounding source length must be 1..1000 characters")
        if _CONTROL.search(source):
            raise ContractError("grounding source contains control characters")
    source = transcript.text
    if not isinstance(payload, dict):
        raise ContractError("decision must be an object")
    kind = payload.get("kind")
    if not isinstance(kind, str):
        raise ContractError("decision kind must be a string")
    expected = {
        "answer": {"kind", "answer", "basis", "evidence_quotes"},
        "clarify": {"kind", "question", "evidence_quotes"},
        "plan": {"kind", "steps", "evidence_quotes"},
        "no_request": {"kind", "reason", "evidence_quotes"},
        "unsupported": {"kind", "reason", "evidence_quotes"},
    }.get(kind)
    if kind == "plan":
        base = {"kind", "steps", "evidence_quotes"}
        with_answer = base | {"answer", "answer_basis"}
        if set(payload) not in (base, with_answer):
            raise ContractError("unknown decision kind or wrong branch fields")
    elif expected is None or set(payload) != expected:
        raise ContractError("unknown decision kind or wrong branch fields")
    quotes = payload["evidence_quotes"]
    if not isinstance(quotes, list) or len(quotes) > 5:
        raise ContractError("evidence_quotes must be a list of at most five quotes")
    checked_quotes = tuple(_valid_string(q, "evidence quote", 500) for q in quotes)
    allowed_evidence = (source, *sources)
    if any(not any(q in allowed for allowed in allowed_evidence) for q in checked_quotes):
        raise ContractError("evidence quote must occur literally in the transcript")
    if kind in {"answer", "plan"} and not checked_quotes:
        raise ContractError(f"{kind} requires source evidence")
    if kind == "plan" and not any(q in source for q in checked_quotes):
        raise ContractError("plan requires evidence from the current transcript")

    if kind == "answer":
        if payload["basis"] != "general_knowledge":
            raise ContractError("answer basis must be general_knowledge")
        text = _valid_string(payload["answer"], "answer", 1200)
        return Decision(kind=kind, text=text, evidence_quotes=checked_quotes)
    if kind == "clarify":
        text = _valid_string(payload["question"], "question", 500)
        return Decision(kind=kind, text=text, evidence_quotes=checked_quotes)
    if kind in {"no_request", "unsupported"}:
        reason = _valid_string(payload["reason"], "reason", 500)
        return Decision(kind=kind, reason=reason, evidence_quotes=checked_quotes)

    raw_steps = payload["steps"]
    if not isinstance(raw_steps, list) or not 1 <= len(raw_steps) <= 3:
        raise ContractError("plan must contain one to three steps")
    steps: list[Step] = []
    seen_ids: set[str] = set()
    mutation_keys: set[tuple[str, tuple[tuple[str, str], ...]]] = set()
    for raw in raw_steps:
        if not isinstance(raw, dict) or set(raw) != {"id", "capability", "arguments", "after"}:
            raise ContractError("step has missing or unknown fields")
        step_id = _valid_string(raw["id"], "step id", 24)
        if not _ID.fullmatch(step_id) or step_id in seen_ids:
            raise ContractError("step IDs must be unique safe identifiers")
        capability = _valid_string(raw["capability"], "capability", 40)
        if capability not in CAPABILITIES:
            raise ContractError("unknown capability")
        after = raw["after"]
        if (not isinstance(after, list) or any(not isinstance(ref, str) for ref in after)
                or len(set(after)) != len(after)):
            raise ContractError("after must be a list of unique prior step IDs")
        if any(not isinstance(ref, str) or ref not in seen_ids for ref in after):
            raise ContractError("after may reference only prior step IDs")
        args = _check_arguments(capability, raw["arguments"], source, sources)
        # Codex is a separate bridge and must be the sole explicit action.
        if capability == "codex_request" and len(raw_steps) != 1:
            raise ContractError("codex_request must be a singleton plan")
        identity = (capability, tuple(sorted((k, repr(v)) for k, v in args.items())))
        mutations = {"open_app", "start_jam", "play_song", "play_artist", "play_album",
                     "queue_song", "play_playlist", "pause", "resume", "next_track",
                     "previous_track", "restart_track", "restart_album", "set_volume",
                     "change_volume"}
        if capability in mutations and identity in mutation_keys:
            raise ContractError("plan repeats an identical operation")
        mutation_keys.add(identity)
        steps.append(Step(step_id, capability, args, tuple(after)))
        seen_ids.add(step_id)
    if "answer" not in payload:
        return Decision(kind="plan", steps=tuple(steps), evidence_quotes=checked_quotes)
    answer = _valid_string(payload["answer"], "answer", 1200)
    if (not isinstance(payload["answer_basis"], str)
            or payload["answer_basis"] != "general_knowledge"):
        raise ContractError("answer_basis must be general_knowledge")
    return Decision(kind="plan", steps=tuple(steps), evidence_quotes=checked_quotes,
                    answer=answer, answer_basis="general_knowledge")


def step_to_intent(step: Step) -> Intent:
    """Map executable legacy capabilities to the existing Intent envelope."""
    if step.capability == "codex_request":
        raise ContractError("Codex requests require the separate authorization bridge")
    args = step.arguments
    name = step.capability
    common = {"source": "conversation", "matched_rule": "conversation_" + name}
    if name in {"open_app", "search_browser", "show_track", "show_album", "play_song", "play_artist", "play_album", "queue_song", "play_playlist"}:
        if name == "open_app":
            return Intent(Action.OPEN_APP, query=args["name"], **common)
        if name == "search_browser":
            return Intent(Action.SEARCH_BROWSER, query=args["query"], browser="Opera", **common)
        if name == "show_track":
            return Intent(Action.SHOW_TRACK, query=args["query"], **common)
        if name == "show_album":
            return Intent(Action.SHOW_ALBUM, query=args["query"], **common)
        if name in {"play_song", "play_album", "queue_song"}:
            query = args["title"] + (" by " + args["artist"] if args["artist"] else "")
            action = {"play_song": Action.PLAY_TRACK, "play_album": Action.PLAY_ALBUM, "queue_song": Action.QUEUE}[name]
            return Intent(action, query=query, radio=args.get("radio", True), **common)
        if name == "play_artist":
            return Intent(Action.PLAY_ARTIST, query=args["artist"], radio=args.get("radio", True), **common)
        return Intent(Action.PLAY_PLAYLIST, query=args["name"], library_only=args["mine"], **common)
    if name == "restart_album":
        return Intent(Action.RESTART_ALBUM, **common)
    if name == "start_jam":
        return Intent(Action.START_JAM, **common)
    if name == "pause":
        return Intent(Action.PAUSE, **common)
    if name == "resume":
        return Intent(Action.RESUME, **common)
    if name == "next_track":
        return Intent(Action.NEXT, **common)
    if name == "previous_track":
        return Intent(Action.PREV, **common)
    if name == "restart_track":
        return Intent(Action.RESTART, **common)
    if name == "set_volume":
        return Intent(Action.VOLUME, value=args["level"], volume_mode="absolute", **common)
    if name == "change_volume":
        if args["amount"] == 0:
            return Intent(Action.VOLUME, query=args["direction"], **common)
        unit = "relative_percent" if args["unit"] == "percent" else "relative_points"
        return Intent(Action.VOLUME, value=args["amount"], query=args["direction"], volume_mode=unit, **common)
    raise ContractError("unknown capability")
