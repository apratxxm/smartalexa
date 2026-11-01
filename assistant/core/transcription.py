"""One raw command decode with local-only word-timestamp wake trimming."""
from __future__ import annotations
import math
import time
from dataclasses import dataclass
import numpy as np
from ..stt import Transcriber


class NoWakeBoundary(ValueError):
    def __init__(self, reason="no_wake_boundary", prefix_length=0):
        self.reason, self.prefix_length = reason, prefix_length
        super().__init__("no wake boundary")


@dataclass(frozen=True)
class Word:
    text: str
    start: float
    end: float


@dataclass(frozen=True)
class TimedTranscript:
    words: tuple[Word, ...]
    latency_ms: float


class TimestampedTranscriber(Transcriber):
    def transcribe(self, pcm, *, hotwords=None):
        if pcm.ndim != 1 or pcm.dtype != np.int16:
            raise ValueError("STT requires one-dimensional int16 PCM")
        if not pcm.size:
            return TimedTranscript((), 0)
        audio = pcm.astype(np.float32) / 32768.0
        started = time.monotonic()
        # Reuse the retained, already-loaded command backend. Do not invoke
        # Transcriber.transcribe first: that would decode the same turn twice.
        segments, _ = self._backend._model.transcribe(
            audio, language="en", beam_size=1, best_of=1, temperature=0.0,
            hotwords=hotwords, condition_on_previous_text=False, vad_filter=False,
            word_timestamps=True, without_timestamps=False,
            no_speech_threshold=0.6, compression_ratio_threshold=2.4)
        words = tuple(Word(w.word, w.start, w.end) for s in segments for w in (s.words or ()))
        return TimedTranscript(words, (time.monotonic() - started) * 1000)


def trim_to_wake(result, aliases=(), *, wake_at=None, audio_started_at=None):
    if not isinstance(result, TimedTranscript):
        raise NoWakeBoundary()
    if (wake_at is None or audio_started_at is None or not math.isfinite(wake_at)
            or not math.isfinite(audio_started_at) or wake_at < audio_started_at):
        raise NoWakeBoundary("unknown_wake_timeline")
    normalize = lambda text: "".join(c for c in text.casefold() if c.isalnum())
    accepted = {"alexa", *(normalize(alias) for alias in aliases)}
    previous = 0.0
    for word in result.words:
        if (not isinstance(word.text, str) or not math.isfinite(word.start)
                or not math.isfinite(word.end) or word.start < previous or word.end < word.start):
            raise NoWakeBoundary("invalid_word_timestamps")
        previous = word.start
    tokens = [i for i, word in enumerate(result.words) if normalize(word.text) in accepted]
    qualifying = [i for i in tokens if abs(audio_started_at + result.words[i].end - wake_at) <= 0.6 + 1e-9]
    if len(qualifying) != 1:
        raise NoWakeBoundary("ambiguous_wake_boundary" if qualifying else "uncorroborated_wake_boundary")
    index = qualifying[0]
    prefix_length = len(" ".join(w.text.strip() for w in result.words[:index + 1]))
    if any(i < index for i in tokens):
        raise NoWakeBoundary("earlier_wake_token", prefix_length)
    # Construct output only after the complete local boundary check succeeds.
    text = " ".join(w.text.strip() for w in result.words[index + 1:]).strip()
    return text, prefix_length
