"""Whole-utterance authority, independent of the model and legacy parser."""
from __future__ import annotations
import re
from dataclasses import dataclass

from ..conversation_contracts import Step, step_to_intent
from .adapters.base import Clause

REFUSAL = "I didn't catch a clear instruction."
NOT_DONE = "I didn't do anything."
MUTATING = frozenset({"play_song", "pause", "resume", "next_track", "set_volume", "change_volume"})
# Bare words that ordinary search queries may contain. computer.py refuses only the
# multi-word show phrases and the other mutating verbs; the core search check mirrors that.
_SEARCH_BARE_VERBS = frozenset({"stop", "show", "show me"})
_EFFECT = re.compile(r"\b(?:paused|resumed|played|playing|started|starting|stopped|skipped|skipping|lowered|raised|turned|set|changed|muted|unmuted|increased|decreased|"
                     r"queued|queueing|restarted|restarting|rewound|opened|opening|launched|searched|searching|showing|showed|sent|delegated|submitted|forwarded|requested|created|displayed|generated|done)\b", re.I)
_WORDS = dict(zip(
    ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen").split(),
    range(20)))
_WORDS.update(dict(zip("twenty thirty forty fifty sixty seventy eighty ninety".split(), range(20, 100, 10))))

_FILLER_PREFIX = re.compile(r"^(?:(?:um|uh)\s+)*(?:(?:can|could|would|will)\s+you\s+)?(?:please\s+)?", re.I)
_FILLER_SUFFIX = re.compile(r"(?:\s+(?:please|thanks|thank you|um|uh))+$", re.I)
_SUSPICIOUS = re.compile(
    r"\b(?:don['’]?t|do\s+not|not|never|no|actually|cancel|wait|stop|instead|rather|"
    r"sorry|correction|i\s+mean|never\s+mind|nevermind|but|however|if|unless|until|"
    r"when|whenever|once|as\s+soon\s+as|in\s+case|said|says|told|reported|"
    r"before|after|while|except|otherwise|followed|also|plus|with|alongside|yet)\b", re.I)
_STOPWORDS = frozenset("a an the song track album music some my me it to on for of and or".split())


def _phrase_pattern(phrase):
    return r"(?<!\w)" + r"\s+".join(re.escape(part) for part in phrase.split()) + r"(?!\w)"


def find_clause(text, registry, *, capabilities=None):
    """Parse one complete registered command, preserving the spoken object."""
    if not isinstance(text, str) or not text.strip() or len(text) > 12000:
        return None
    if any(ord(character) < 32 for character in text) or any(character in text for character in '"`“”'):
        return None
    original = text.strip()
    if original[-1:] in {".", "!", "?"}:
        original = original[:-1].rstrip()
    if not original or any(character in original for character in ".!?;:/-\u2013\u2014\u2019"):
        return None
    if re.search(r"(?<!\w)['‘’]", original):
        return None
    prefix = _FILLER_PREFIX.match(original)
    body = original[prefix.end():].strip()
    body = re.sub(r",\s*(?=(?:please|thanks|thank you|um|uh)\s*$)", " ", body, flags=re.I)
    body = _FILLER_SUFFIX.sub("", body).strip()
    if not body or "," in body:
        return None
    enabled_verbs = tuple(registry.verbs)
    eligible = []
    for verb in sorted(enabled_verbs, key=lambda value: (-len(value), value)):
        match = re.match(_phrase_pattern(verb), body, re.I)
        if match is None or match.start() != 0:
            continue
        entries = [(name, handler) for name, handler in registry.verbs[verb]
                   if capabilities is None or name in capabilities]
        if entries:
            eligible.append((verb, match.end(), entries))
    if not eligible:
        return None
    verb, end, entries = eligible[0]
    obj = body[end:].strip()
    if obj and not body[end].isspace():
        return None
    kinds = {handler.object_kind for _, handler in entries}
    if obj:
        if not kinds.intersection({"text", "number"}) and not any(name == "resume" for name, _ in entries):
            return None
        words = _tokens(obj)
        if not words:
            return None
        if all(name not in {"pause", "resume", "next_track", "set_volume", "change_volume"} for name, _ in entries) and not any(word not in _STOPWORDS for word in words):
            return None
    if obj:
        for other in getattr(registry, "safety_verbs", enabled_verbs):
            connector = r"(?:\b(?:and|or|then)\b|[&+,;])\s*(?:(?:please|can\s+you|could\s+you|would\s+you|will\s+you)\s+)*"
            if re.search(connector + _phrase_pattern(other), obj, re.I):
                return None
    suspicious = (bool(_SUSPICIOUS.search(obj)) or any(
        re.search(_phrase_pattern(other), obj, re.I) for other in enabled_verbs)) and not all(
        name in {"pause", "resume", "next_track", "set_volume", "change_volume"} for name, _ in entries)
    return Clause(text=text, verb=verb, object=obj,
                  polite=bool(prefix.end() or body != original[prefix.end():].strip()),
                  suspicious=suspicious)


def c1_allows(handler, clause, arguments, guard):
    if not clause.suspicious or not handler.mutating or getattr(handler, "c1_exempt", False):
        return True
    matcher = getattr(handler, "exact_match", None)
    if not callable(matcher):
        return False
    try:
        return matcher(arguments, guard) is True
    except Exception:
        return False


def _tokens(value):
    return tuple(re.findall(r"[^\W_]+", value.casefold()))


def _number(value):
    value = value.casefold().strip().replace("-", " ")
    if re.fullmatch(r"\d{1,3}", value):
        n = int(value)
    elif value in {"hundred", "one hundred", "full", "max", "maximum"}:
        n = 100
    elif value == "half":
        n = 50
    else:
        parts = value.split()
        if len(parts) == 1:
            n = _WORDS.get(value, -1)
        elif len(parts) == 2 and _WORDS.get(parts[0], -1) in range(20, 100, 10) and _WORDS.get(parts[1], -1) in range(1, 10):
            n = _WORDS[parts[0]] + _WORDS[parts[1]]
        else:
            n = -1
    return n if 0 <= n <= 100 else None


@dataclass(frozen=True)
class License:
    capability: str
    arguments: dict



class PhaseOneHandler:
    mutating = True
    effect_words = ()
    _VERBS = {
        "play_song": ("play", "put on", "start"),
        "pause": ("pause", "hold", "stop"),
        "resume": ("resume", "continue", "unpause", "play", "start"),
        "next_track": ("next", "skip", "skip to the next track"),
        "set_volume": ("set", "put", "turn", "volume", "windows volume", "system volume", "master volume"),
        "change_volume": ("turn", "make it louder", "make it quieter", "louder", "quieter"),
    }
    def __init__(self, name):
        self.name = name
        self.verbs = self._VERBS[name]
        self.object_kind = "none" if name == "resume" else ("number" if name == "change_volume" else "text")
        self.executor = None

    def license(self, clause):
        obj, verb = clause.object, clause.verb
        lower = " ".join(obj.casefold().split())
        if self.name == "play_song":
            if not obj or lower in {"music", "the music"} or all(token in _STOPWORDS for token in _tokens(obj)):
                return None
            title = re.sub(r"^(?:(?:the )?(?:song|track) (?:(?:called|named) )?)", "", obj, flags=re.I)
            return {"title": title, "artist": ""} if title else None
        if self.name == "pause":
            allowed = {"", "music", "the music", "playback", "the playback", "song", "the song", "track", "the track"}
            return {} if lower in allowed and (verb != "stop" or lower in {"music", "the music"}) else None
        if self.name == "resume":
            return {} if lower in {"", "music", "the music", "playback", "the playback"} else None
        if self.name == "next_track":
            return {} if lower in {"", "song", "the song", "this song", "track", "the track", "this track", "one", "the one", "this one", "this", "that", "it"} else None
        text = (verb + " " + obj).strip()
        if self.name == "set_volume":
            match = re.fullmatch(r"(?:(?:set|put|turn)\s+)?(?:the\s+)?(?:(?:windows|system|master)\s+)?volume\s+(?:(?:to|at)\s+)?(.+?)(?:\s*(?:%|percent))?", text, re.I)
            number = _number(match[1]) if match else None
            return {"level": number} if number is not None else None
        match = re.fullmatch(r"turn\s+(?:(?:the\s+)?(?:(?:windows|system|master)\s+)?volume|it)\s+(up|down)(?:\s+(.+?)(?:\s+(points|percent))?)?", text, re.I)
        if match:
            amount = _number(match[2]) if match[2] else 0
            return {"direction": match[1].lower(), "amount": amount, "unit": (match[3] or "points").lower()} if amount is not None else None
        if text.casefold() in {"make it louder", "make it quieter", "louder", "quieter"}:
            return {"direction": "up" if text.casefold().endswith("louder") else "down", "amount": 0, "unit": "points"}
        return None

    def exact_match(self, args, guard):
        matcher = getattr(self.executor, "exact_match", None)
        return callable(matcher) and matcher(args, guard)


def _registry_default():
    from ..config import Config
    from .registry import Registry
    return Registry(Config({}))


def parsed_licenses(text, registry, *, capabilities=None):
    clause = find_clause(text, registry, capabilities=capabilities)
    if clause is None:
        return None, ()
    results = []
    for name, handler in registry.verbs[clause.verb]:
        if capabilities is not None and name not in capabilities:
            continue
        try:
            args = handler.license(clause)
        except Exception:
            continue
        if args is not None:
            results.append((name, handler, args))
    return clause, results


def _equal_args(capability, expected, actual):
    if capability == "play_song":
        try:
            query = step_to_intent(Step("match", capability, actual)).query
            return " ".join(query.casefold().split()) == " ".join(expected["title"].casefold().split())
        except (KeyError, TypeError, ValueError):
            return False
    return expected == actual


def _search_query_refused(registry, query):
    """C1 exemption limit: a search query may not contain another mutating capability's verb phrase."""
    for verb, entries in registry.verbs.items():
        if verb in _SEARCH_BARE_VERBS:
            continue
        if any(getattr(handler, "mutating", False) for _, handler in entries) and re.search(
                _phrase_pattern(verb), query, re.I):
            return True
    return False


def _without_wake(text):
    return re.sub(r"^(?:(?:hey|hi|okay|ok)\s+)?(?:memoir|alexa)[,\s]+", "", text.strip(), flags=re.I)


def _normalized_request(text):
    return " ".join(_without_wake(text).casefold().split())


def licensed_codex_args(transcript, registry):
    _, results = parsed_licenses(_without_wake(transcript), registry)
    return next((args for name, _, args in results if name == "codex_request"), None)


def admit_step(step, transcript, registry, guard):
    source = _without_wake(transcript) if step.capability == "codex_request" else transcript
    clause, results = parsed_licenses(source, registry)
    if clause is None:
        return False
    for name, handler, args in results:
        matches = _equal_args(name, args, step.arguments)
        if name == "codex_request" and isinstance(step.arguments, dict):
            request = step.arguments.get("request")
            matches = (set(step.arguments) == {"request"} and isinstance(request, str)
                       and _normalized_request(request) in {
                           _normalized_request(args["request"]), _normalized_request(transcript)})
        if name != step.capability or not matches:
            continue
        if name == "search_browser" and _search_query_refused(registry, clause.object):
            return False
        return c1_allows(handler, clause, args, guard)
    return False


def license_request(text, *, brain=True):
    clause, results = parsed_licenses(text, _registry_default())
    if not results:
        return None
    name, _, args = results[0]
    if name == "play_song":
        args = {"object": args["title"]}
    return License(name, args)


def licenses(step, transcript, *, brain=True):
    clause, results = parsed_licenses(transcript, _registry_default())
    return any(name == step.capability and _equal_args(name, args, step.arguments)
               for name, handler, args in results)

def claims_effect(text, handlers=()):
    text = text or ""
    if _EFFECT.search(text) or re.search(
        r"\b(?:is|are)\s+(?:ready|(?:now\s+)?on\s+(?:the\s+)?screen|(?:now\s+)?(?:open|playing|paused|running))\b|\bi(?:['\u2019]ve|\s+have)\s+(?:just\s+)?\w+ed\b", text, re.I
    ):
        return True
    if re.search(
        r"\b(?:is|are)\s+(?:on it|working on it|handling it|taking care of it|up|(?:now\s+)?(?:live|started|underway|going))\b|"
        r"\bi(?:['\u2019]m|\s+am|\s+will|['\u2019]ll)\s+(?:on it|handling|starting|opening|playing|sending|asking)\b",
        text, re.I
    ):
        return True
    effect_verbs = {"playing", "starting", "stopping", "skipping", "opening",
                    "sending", "asking", "showing", "searching", "queueing",
                    "queuing", "sent", "set", "done", "made", "took", "ran", "began"}
    return any(re.search(_phrase_pattern(word), text, re.I)
               for handler in handlers for word in getattr(handler, "effect_words", ())
               if word and (word.casefold().endswith(("ed", "en")) or word.casefold() in effect_verbs))

