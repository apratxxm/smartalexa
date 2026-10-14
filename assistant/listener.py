"""VAD-gated utterance capture followed by configured local speech recognition.

Text triggers are checked at the utterance prefix. The separate wake detector
may duck music early, but does not authorize playback in this mode. Endpoint
silence, decoding, acknowledgement and service calls all contribute to latency;
historical model timings are not guarantees for the current audio environment.
"""
from __future__ import annotations

import difflib
import logging
import re
import time
import uuid
from collections import deque
from typing import List, Optional

import numpy as np

from .audio import Microphone, SileroVAD
from .config import Config

log = logging.getLogger("memoir.listener")


class Utterance:
    """One detected stretch of speech, with what was heard in it."""

    def __init__(self, audio: np.ndarray, transcript: str, latency_ms: int,
                 completed_at: float = 0.0) -> None:
        self.audio = audio
        self.transcript = transcript
        self.latency_ms = latency_ms
        self.completed_at = completed_at
        self.triggered = False
        self.matched_phrase = ""
        self.command = ""          # what followed the trigger, if anything
        self.mid_utterance = False # trigger came after other speech (lyrics)
        # Diagnostics (perf_counter times, 0.0 = unknown). wake_path says which
        # detector first caught the wake: second_stt | alias_acoustic | peek | final_text.
        self.wake_path = ""
        self.wake_evidence_at = 0.0
        self.speech_started_at = 0.0
        self.trigger_done_at = 0.0
        self.processed_audio = None
        self.segment_id = ""
        self.end_reason = ""
        self.truncated = False
        self.confirmed_audio_handoff = False


class SttListener:
    """Continuously listens; yields utterances that begin with the trigger."""

    def __init__(self, cfg: Config, vad: SileroVAD, transcribe,
                 on_speech_start=None, on_speech_end=None, on_wake=None,
                 authorize_trigger=None, on_reject=None, verify_trigger=None,
                 on_near_miss=None, on_candidate=None) -> None:
        self.vad = vad
        self._transcribe = transcribe
        # Called the instant speech begins / ends. Used to duck the music
        # before the command is captured: we cannot wait until we know it was
        # a command, because by then it has already been recorded over music.
        self._on_speech_start = on_speech_start
        self._on_speech_end = on_speech_end
        # Called once, mid-utterance, the moment the wake phrase is heard, so
        # the music can drop while the user is still saying the command.
        self._on_wake = on_wake
        # Acoustic corroboration is required for aliases and for wake words
        # found after preceding speech. Without a provider these fail closed.
        self._authorize_trigger = authorize_trigger
        self._on_reject = on_reject
        self._verify_trigger = verify_trigger
        # Called with a dict for a wake-like utterance that was refused, so the
        # clip can be archived for wake-word retraining.
        self._on_near_miss = on_near_miss
        self._on_candidate = on_candidate
        self._pending_rejections = []
        self._current_duration_ms = 0
        self.phrases = [p.lower() for p in cfg.get(
            "listener.phrases", ["memoir", "jarvis", "alexa"])]
        self.fuzzy_cutoff = cfg.get("listener.fuzzy_cutoff", 0.72)
        self.allow_fuzzy = cfg.get("listener.allow_fuzzy", False)
        # Observed mishearings, straight from the live monitor. Character
        # similarity cannot bridge these ("remember" vs "memoir" scores ~0.5),
        # so they are listed explicitly and grow from the logs.
        self.aliases = {a.lower(): t.lower() for a, t in
                        (cfg.get("listener.aliases", {}) or {}).items()}
        self.log_ignored = cfg.get("listener.log_ignored", False)
        # The wake gate may be looser than command capture: a far or soft
        # "Alexa" (VAD 0.1-0.6) never became a segment, so it was never
        # transcribed at all. The text match still has to say "Alexa".
        self.vad_threshold = cfg.get("listener.vad_threshold",
                                     cfg.get("capture.vad_threshold", 0.35))

        sample_rate = self.sample_rate = 16000
        frame_ms = 512 * 1000.0 / sample_rate
        self.min_frames = int(cfg.get("listener.min_speech_ms", 300) / frame_ms)
        self.silence_frames = int(cfg.get(
            "listener.silence_ms", cfg.get("capture.silence_ms", 800)
        ) / frame_ms)
        self.max_frames = int(cfg.get("listener.max_utterance_s", 6.0)
                              * 1000 / frame_ms)
        self.pad_frames = int(cfg.get("listener.pad_ms", 300) / frame_ms)
        # Vocal music keeps the VAD "in speech" indefinitely, so utterances
        # are cut at max_utterance_s and the wake word lands mid-chunk or
        # across the cut. Consecutive forced chunks overlap by this much.
        self.overlap_frames = int(cfg.get("listener.overlap_ms", 1500) / frame_ms)
        # A wake word found in a forced chunk with a command after it may have
        # been cut mid-command; keep listening up to this long before acting.
        self.extend_frames = int(cfg.get("listener.extend_ms", 4000) / frame_ms)
        # Phrases trusted when found after other speech (e.g. sung lyrics).
        # "computer" is a common lyric word, so it must open the utterance.
        self.mid_phrases = [p.lower() for p in cfg.get(
            "listener.mid_phrases", ["alexa", "jarvis", "memoir"])]
        self.mid_fuzzy_cutoff = cfg.get("listener.mid_fuzzy_cutoff", 0.9)  # "alexia" yes, the name "alex" no
        # Early wake check: transcribe the utterance so far at these points.
        self.peek_first = int(cfg.get("listener.peek_first_ms", 900) / frame_ms)
        self.peek_every = max(1, int(cfg.get("listener.peek_every_ms", 1000) / frame_ms))
        self.peek_until = int(cfg.get("listener.peek_until_ms", 10000) / frame_ms)
        self.peek_window = max(1, int(cfg.get("listener.peek_window_ms", 1600) / frame_ms))

        # Rolling pre-speech padding, so we never clip the first syllable of the
        # trigger word -- which is exactly the part we need to recognise.
        self._pre: List[np.ndarray] = []
        self._speech: List[np.ndarray] = []
        self._pre_processed = []
        self._speech_processed = []
        self._segment_id = ""
        self._overlap_of = ""
        self._segment_vad_peak = 0.0
        self._peek_errors = []
        # Diagnostics alone: no transcription, authorization or music ducking.
        self._weak_enabled = bool(cfg.get("telemetry.wake_candidates_enabled", False))
        self._weak_floor = float(cfg.get("telemetry.wake_candidate_vad_threshold", 0.08))
        self._weak_min_frames = 3  # 96 ms at 512/16k
        self._weak_silence_frames = 11  # 352 ms
        self._weak_max_frames = 187  # <= 6 seconds
        self._weak_pre = deque(maxlen=9)  # <= 300 ms
        self._weak_frames = []
        self._weak_voiced = 0
        self._weak_silence = 0
        self._weak_peak = 0.0
        self._silence_run = 0
        self._in_speech = False
        self._voiced_frames = 0
        self._extending = False
        self._woke = False
        self._woke_phrase = ""
        self._woke_source = ""
        self._woke_mid = False
        self._woke_at = 0.0
        self._confirmed_audio_handoff = False
        self._speech_started_at = 0.0
        self.last_vad_probability = 0.0
        self.segments_completed = 0
        self.discarded_short_segments = 0
        self._next_peek = 0

    # ---------- trigger matching ----------

    def _authorized(self, source: str, phrase: str) -> bool:
        return bool(self._authorize_trigger and self._authorize_trigger(source, phrase))

    def _reject(self, reason: str, duration_ms: int = 0) -> None:
        self._pending_rejections.append((reason, duration_ms or self._current_duration_ms))

    def _unconfirmed_candidate(self, transcript: str) -> tuple:
        candidate = SttListener.__new__(SttListener)
        candidate.phrases = self.phrases
        candidate.aliases = self.aliases
        candidate.fuzzy_cutoff = self.fuzzy_cutoff
        candidate.mid_fuzzy_cutoff = self.mid_fuzzy_cutoff
        candidate.allow_fuzzy = self.allow_fuzzy
        candidate.mid_phrases = self.mid_phrases
        candidate._authorize_trigger = lambda source, phrase: True
        candidate._on_reject = None
        candidate._pending_rejections = []
        candidate._current_duration_ms = self._current_duration_ms
        return candidate.locate_trigger(transcript)

    def match_trigger(self, transcript: str) -> tuple:
        """Return (matched_phrase, remaining_command) or ("", "").

        Only the first few words are considered: the trigger has to open the
        utterance, so ordinary conversation that happens to contain the word
        later does not fire.
        """
        text = re.sub(r"[^\w\s]", " ", transcript.lower())
        words = [w for w in text.split() if w]
        if not words:
            return "", ""

        # Match complete token prefixes only; allow the common optional
        # "hey" before a configured phrase, but never search arbitrary text.
        candidates = [(0, words)]
        if words and words[0] == "hey":
            candidates.append((1, words[1:]))
        configured_aliases = [(raw, target) for raw, target in self.aliases.items()
                              if target in self.phrases]
        canonical = [(p, p) for p in self.phrases]

        # Exact configured phrases always win over spelling similarity. In
        # particular, aliases must not shadow a canonical wake phrase.
        for _, head_words in candidates:
            for raw, target in canonical + configured_aliases:
                parts = raw.split()
                if len(head_words) < len(parts):
                    continue
                heard = " ".join(head_words[:len(parts)])
                if heard == raw:
                    if raw != target and not self._authorized("alias", target):
                        self._reject("alias_uncorroborated")
                        continue
                    self._match_source = "acoustic" if raw != target else "text_exact"
                    return target, " ".join(head_words[len(parts):]).strip()

        if self.allow_fuzzy:
            for _, head_words in candidates:
                if not head_words:
                    continue
                heard = head_words[0]
                for raw, target in canonical:
                    if (len(raw.split()) == 1
                            and difflib.SequenceMatcher(None, heard, raw).ratio()
                            >= self.fuzzy_cutoff):
                        if not self._authorized("fuzzy", target):
                            self._reject("fuzzy_uncorroborated")
                            continue
                        self._match_source = "acoustic"
                        return target, " ".join(head_words[1:]).strip()
        return "", ""

    def locate_trigger(self, transcript: str) -> tuple:
        """(phrase, command, mid): the trigger at the start, or after other speech.

        Over music the transcript usually opens with lyrics ("...tonight alexa
        play red"), so a prefix-only rule missed most real attempts. Later
        positions use a stricter spelling match and only trusted phrases.
        """
        phrase, command = self.match_trigger(transcript)
        if phrase:
            return phrase, command, False
        text = re.sub(r"[^\w\s]", " ", transcript.lower())
        words = [w for w in text.split() if w]
        strict = SttListener.__new__(SttListener)
        strict.phrases = [p for p in self.mid_phrases if p in self.phrases]
        strict.aliases = {a: t for a, t in self.aliases.items()
                           if t in strict.phrases and t in self.phrases}
        strict.fuzzy_cutoff = self.mid_fuzzy_cutoff
        strict.allow_fuzzy = self.allow_fuzzy
        strict._authorize_trigger = self._authorize_trigger
        strict._on_reject = self._on_reject
        strict._current_duration_ms = self._current_duration_ms
        strict._pending_rejections = self._pending_rejections
        for start in range(1, len(words)):
            phrase, command = strict.match_trigger(" ".join(words[start:]))
            if phrase:
                if not command:
                    # A bare name embedded in conversation is not an invocation.
                    # Qualified bare acoustic wakes use strict short-window
                    # prefix confirmation in the assistant instead.
                    continue
                if self._authorized("mid", phrase):
                    self._match_source = "acoustic"
                    return phrase, command, True
                self._reject("mid_uncorroborated")
                return "", "", False
        return "", "", False

    # ---------- the loop ----------

    def feed(self, frame: np.ndarray, vad_frame: Optional[np.ndarray] = None) -> Optional[Utterance]:
        """Feed one 512-sample frame. Returns an Utterance when one completes.

        vad_frame, when given, is the echo-cancelled copy: speech is judged on
        it (so music does not count as the user talking), while the raw frame
        is what gets transcribed.
        """
        self.last_vad_probability = float(self.vad(frame if vad_frame is None else vad_frame))
        is_speech = self.last_vad_probability >= self.vad_threshold
        processed = (vad_frame if isinstance(vad_frame, np.ndarray)
                     and vad_frame is not frame and vad_frame.shape == frame.shape else None)
        self._feed_weak_candidate(frame, processed, is_speech)

        if not self._in_speech:
            self._pre.append(frame)
            self._pre_processed.append(processed)
            if len(self._pre) > self.pad_frames:
                self._pre.pop(0)
                self._pre_processed.pop(0)
            if is_speech:
                self._in_speech = True
                self._speech = list(self._pre)
                self._speech_processed = list(self._pre_processed)
                # With zero padding the current frame was just trimmed from
                # pre-roll; speech onset still belongs to the utterance.
                if not self._pre:
                    self._speech.append(frame)
                    self._speech_processed.append(processed)
                self._segment_id = uuid.uuid4().hex[:16]
                self._segment_vad_peak = self.last_vad_probability
                self._silence_run = 0
                self._voiced_frames = 1
                self._next_peek = self.peek_first
                self._speech_started_at = time.perf_counter()
                pending = getattr(self, "_pending_wake", None)
                self._pending_wake = None
                if pending is not None and time.perf_counter() - pending[1] <= 2.0:
                    self._woke = True
                    self._woke_phrase = pending[0]
                    self._woke_at = pending[1]
                    self._woke_source = "acoustic"
                    self._woke_mid = True
                if self._on_speech_start:
                    self._on_speech_start()
            return None

        self._speech.append(frame)
        self._speech_processed.append(processed)
        self._segment_vad_peak = max(self._segment_vad_peak, self.last_vad_probability)
        if is_speech:
            self._voiced_frames += 1
        self._silence_run = 0 if is_speech else self._silence_run + 1

        self._peek()
        limit = self.max_frames + (self.extend_frames if self._extending else 0)
        ended = self._silence_run >= self.silence_frames
        forced = not ended and len(self._speech) >= limit
        if not (ended or forced):
            return None

        audio = np.concatenate(self._speech)
        processed_audio = self._aligned_processed(self._speech_processed)
        segment_id = self._segment_id
        end_reason = "silence" if ended else "extension_limit" if self._extending else "max_utterance"
        self.segments_completed += 1
        # VAD can undercount soft wake speech under noise. Do not let that
        # count veto a segment that captured enough audio for recognition;
        # this does not claim the whole segment is speech. No-onset and truly
        # tiny captures still stay below the duration floor.
        if (self._voiced_frames < self.min_frames
                and len(self._speech) < self.min_frames and not self._extending):
            self.discarded_short_segments += 1
            self._emit_candidate(audio, processed_audio, "speech_too_short", "short_segment",
                                 end_reason=end_reason, truncated=forced)
            if self._on_reject:
                self._on_reject("speech_too_short", int(audio.size * 1000 / self.sample_rate))
            self._reset()
            if self._on_speech_end:
                self._on_speech_end()
            return None

        completed_at = time.perf_counter()
        self._current_duration_ms = int(audio.size * 1000 / self.sample_rate)
        self._match_source = ""
        try:
            transcript, latency = self._transcribe(audio)
        except Exception:
            self._emit_candidate(audio, processed_audio, "trigger_decode_error", "decode_error",
                                 end_reason=end_reason, truncated=forced)
            raise
        trigger_done_at = time.perf_counter()
        # _reset() below clears these; the Utterance and near-miss record need them.
        woke_early, woke_at = self._woke, self._woke_at
        speech_started_at = self._speech_started_at
        confirmed_audio_handoff = self._confirmed_audio_handoff
        phrase, command, mid = self.locate_trigger(transcript)
        confirmation_mode = getattr(self, "_match_source", "text_exact") if phrase else ""
        confirmation_latency_ms = None
        confirmation_transcript = ""
        turbo_rejected, turbo_rejection = False, ""
        if not phrase and self._verify_trigger is not None:
            proposed, _, proposed_mid = self._unconfirmed_candidate(transcript)
            if not proposed and self._woke:
                proposed, proposed_mid = self._woke_phrase, self._woke_mid
            if proposed:
                verified = self._verify_trigger(audio, proposed, proposed_mid)
                trigger_done_at = time.perf_counter()
                if verified and verified[0]:
                    phrase, command, mid, confirmation_latency_ms = verified[:4]
                    confirmation_transcript = verified[4] if len(verified) > 4 else ""
                    confirmation_mode = "second_stt"
                else:
                    self._reject("second_stt_unconfirmed")
                    confirmation_latency_ms = verified[3] if verified and len(verified) > 3 else None
                    turbo_rejected = True
                    turbo_rejection = verified[4] if verified and len(verified) > 4 else ""
        if not phrase:
            reasons = sorted({reason for reason, _ in self._pending_rejections}) + self._peek_errors
            if woke_early:
                reasons.append("woke_early_not_confirmed")
            self._emit_candidate(audio, processed_audio, ",".join(reasons) or "speech_no_trigger",
                                 "second_stt" if turbo_rejected else "no_trigger",
                                 transcript=transcript, second_stt_transcript=turbo_rejection,
                                 stt_latency_ms=latency, second_stt_latency_ms=confirmation_latency_ms,
                                 woke_early=woke_early, end_reason=end_reason, truncated=forced)
            if self._on_candidate is None:
                self._log_near_miss(transcript, audio, woke_early, turbo_rejection, turbo_rejected)
        if not phrase and self._on_reject:
            for reason, duration_ms in set(self._pending_rejections):
                self._on_reject(reason, duration_ms)
        if forced and phrase and command and not self._extending:
            # The wake word is here but the command may still be going on:
            # keep this audio and listen until the user stops.
            self._extending = True
            return None

        if forced and not phrase:
            # Still in (music) speech: continue with an overlap so a wake
            # word straddling this cut is heard whole in the next chunk.
            keep = self._speech[-self.overlap_frames:] if self.overlap_frames else []
            keep_processed = self._speech_processed[-self.overlap_frames:] if self.overlap_frames else []
            self._reset()
            if keep:
                self._in_speech = True
                self._speech = list(keep)
                self._speech_processed = list(keep_processed)
                self._segment_id = uuid.uuid4().hex[:16]
                self._overlap_of = segment_id
                # The new chunk starts with the overlap, not at the first cut.
                self._speech_started_at = time.perf_counter() - len(keep) * 512 / self.sample_rate
        else:
            self._reset()
            if self._on_speech_end:
                self._on_speech_end()

        utt = Utterance(audio, transcript, latency, completed_at)
        utt.processed_audio = processed_audio
        utt.segment_id = segment_id
        utt.end_reason = end_reason
        utt.truncated = forced
        utt.confirmed_audio_handoff = confirmed_audio_handoff
        utt.confirmation_mode = confirmation_mode if phrase else ""
        utt.confirmation_latency_ms = confirmation_latency_ms
        utt.confirmation_transcript = confirmation_transcript
        utt.speech_started_at = speech_started_at
        utt.trigger_done_at = trigger_done_at
        if phrase:
            # How the wake was first caught, and when (see handle_utterance).
            if confirmation_mode == "second_stt":
                utt.wake_path = "second_stt"
            elif confirmation_mode == "acoustic":
                utt.wake_path = "alias_acoustic"
            elif woke_early:
                utt.wake_path = "peek"
            else:
                utt.wake_path = "final_text"
            utt.wake_evidence_at = woke_at if woke_early and woke_at else trigger_done_at
            if forced:
                # Still talking at the extension cap: never act on a command
                # that may be cut short; acknowledge and ask for it instead.
                command = ""
            # When loud playback joins two attempts into one VAD utterance,
            # "Alexa Alexa" is still just a wake request, not a command.
            repeated_wake, remainder = self.match_trigger(command)
            if repeated_wake and not remainder:
                command = ""
            utt.triggered = True
            utt.matched_phrase = phrase
            utt.command = command
            utt.mid_utterance = mid
        else:
            if self.log_ignored:
                rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
                dbfs = 20 * np.log10(max(rms, 1.0) / 32768.0)
                log.info("Ignored speech (no wake): %r, %.1f dBFS, %.2f s%s",
                         transcript, dbfs, audio.size / self.sample_rate,
                         " [cut]" if forced else "")
            else:
                log.debug("Ignored (no trigger): %r", transcript)
        return utt

    def _peek(self) -> None:
        if self._on_wake is None or self._woke or not self._next_peek:
            return
        n = len(self._speech)
        if n < self._next_peek or n > self.peek_until:
            return
        self._next_peek = n + self.peek_every
        window = self._speech[-self.peek_window:]
        self._current_duration_ms = int(len(window) * 512 * 1000 / self.sample_rate)
        text, _ = self._transcribe(np.concatenate(window))
        heard, _, mid = self.locate_trigger(text)
        if heard:
            self._woke = True
            self._woke_at = time.perf_counter()
            self._woke_phrase = heard
            self._woke_source = getattr(self, "_match_source", "text_exact")
            self._woke_mid = mid or n > self.peek_window
            try:
                self._on_wake()
            except Exception as exc:
                self._peek_errors.append("early_wake_action_failed")
                log.warning("Early wake action failed: %s", exc)

    @staticmethod
    def _aligned_processed(frames):
        if not frames or any(frame is None for frame in frames):
            return None
        return np.concatenate(frames)

    def _emit_candidate(self, audio, processed, reason, kind, **metadata):
        if self._on_candidate is None:
            return
        info = {"audio": audio, "processed_audio": processed, "reason": reason, "kind": kind,
                "transcript": "", "second_stt_transcript": "", "review_state": "unknown",
                "segment_id": self._segment_id, "overlap_of": self._overlap_of,
                "sample_rate": self.sample_rate, "relative_start_sample": 0,
                "relative_end_sample": int(audio.size),
                "processed_alignment": "frame_paired" if processed is not None else "unavailable",
                "vad_peak": self._segment_vad_peak, "vad_threshold": self.vad_threshold,
                "voiced_frames": self._voiced_frames, "decision_source": "listener", **metadata}
        try:
            self._on_candidate(info)
        except Exception as exc:
            log.warning("Wake candidate logging failed: %s", exc)

    def _feed_weak_candidate(self, raw, processed, primary_speech):
        if not self._weak_enabled or self._on_candidate is None:
            return
        # Hand off to the primary lane whenever it has a segment; never duplicate it.
        if primary_speech or self._in_speech:
            self._weak_frames = []
            self._weak_pre.clear()
            self._weak_voiced = self._weak_silence = 0
            self._weak_peak = 0.0
            return
        weak = self.last_vad_probability >= self._weak_floor
        pair = (raw, processed)
        if not self._weak_frames:
            if not weak:
                self._weak_pre.append(pair)
                return
            self._weak_frames = list(self._weak_pre) + [pair]
            self._weak_pre.clear()
            self._weak_voiced, self._weak_silence = 1, 0
            self._weak_peak = self.last_vad_probability
            return
        self._weak_frames.append(pair)
        self._weak_voiced += int(weak)
        self._weak_silence = 0 if weak else self._weak_silence + 1
        self._weak_peak = max(self._weak_peak, self.last_vad_probability)
        forced = len(self._weak_frames) >= self._weak_max_frames
        if not forced and self._weak_silence < self._weak_silence_frames:
            return
        if self._weak_voiced >= self._weak_min_frames:
            audio = np.concatenate([item[0] for item in self._weak_frames])
            clean = self._aligned_processed([item[1] for item in self._weak_frames])
            self._emit_candidate(audio, clean, "below_primary_vad", "below_vad",
                                 segment_id=uuid.uuid4().hex[:16], vad_peak=self._weak_peak,
                                 voiced_frames=self._weak_voiced,
                                 end_reason="diagnostic_limit" if forced else "diagnostic_silence",
                                 truncated=forced, diagnostic_only=True)
        self._weak_frames = []
        self._weak_voiced = self._weak_silence = 0
        self._weak_peak = 0.0

    def _wake_like(self, text: str) -> bool:
        words = re.sub(r"[^\w\s]", " ", (text or "").lower()).split()
        targets = [p for p in self.phrases if " " not in p]
        return any(difflib.SequenceMatcher(None, w, t).ratio() >= 0.6
                   for w in words for t in targets)

    def _log_near_miss(self, transcript: str, audio=None, woke_early: bool = False,
                       turbo_transcript: str = "", turbo_rejected: bool = False) -> None:
        """Log (and hand over for archiving) a rejected utterance that sounds
        like a wake phrase.

        Ambient speech stays out of the log; these are the misses worth
        tracing ("Alexei", "Elixir", "Alex up" that verification refused). The
        audio goes to ``on_near_miss`` so it can become wake-word training data.
        """
        if not (woke_early or turbo_rejected or self._wake_like(transcript) or self._wake_like(turbo_transcript)):
            return
        reasons = sorted({reason for reason, _ in self._pending_rejections})
        if woke_early:
            reasons.append("woke_early_not_confirmed")
        reason = ",".join(reasons) or "no_trigger_match"
        log.info("Wake near-miss rejected: %r%s (woke early=%s, reason=%s)", transcript,
                 " turbo heard %r" % turbo_transcript if turbo_transcript else "",
                 woke_early, reason)
        if self._on_near_miss is None or audio is None:
            return
        try:
            self._on_near_miss({
                "audio": audio, "transcript": transcript, "reason": reason,
                "woke_early": woke_early, "second_stt_transcript": turbo_transcript,
                "kind": "second_stt" if turbo_rejected else "near_miss"})
        except Exception as exc:  # archiving must never break listening
            log.warning("Near-miss archive failed: %s", exc)

    def note_wake(self, phrase: str) -> None:
        """An acoustic hit confirmed by text arrived while this utterance is
        still open: treat it as woken, as an early peek would. The final
        decision still goes through turbo verification if its text lacks it."""
        if not self._in_speech:
            # Acoustic confirmation happens before feed(current_frame). Keep
            # it across speech onset instead of silently losing that command.
            self._pending_wake = (phrase, time.perf_counter())
        elif not self._woke:
            self._woke = True
            self._woke_phrase = phrase
            self._woke_source = "acoustic"
            self._woke_mid = True
            self._woke_at = time.perf_counter()

    def confirm_wake(self, phrase: str, raw_history, processed_history=None) -> bool:
        """Seed a text-confirmed acoustic wake with audio already consumed by capture.

        ``raw_history`` is flattened int16 PCM containing complete 512-sample
        frames before the frame that the caller will feed next. This method
        only records audio and wake evidence; VAD, transcription and command
        execution remain in ``feed``.
        """
        canonical = phrase.strip().lower() if isinstance(phrase, str) else ""
        if not canonical or canonical not in self.phrases:
            return False

        def valid_frames(value):
            if value is None:
                return None
            if not isinstance(value, np.ndarray) or value.ndim != 1 or value.dtype != np.int16:
                return None
            if value.size == 0 or value.size % 512:
                return None
            frame_count = value.size // 512
            if frame_count > min(self.peek_window, self.max_frames):
                return None
            return [value[i * 512:(i + 1) * 512].copy()
                    for i in range(frame_count)]

        raw_frames = valid_frames(raw_history)
        if raw_frames is None:
            return False

        # An active segment already owns this audio. Attach the confirmation
        # without replacing or duplicating any of its frames.
        if self._in_speech:
            self.note_wake(canonical)
            self._confirmed_audio_handoff = True
            return True

        clean_frames = valid_frames(processed_history)
        if clean_frames is None or len(clean_frames) != len(raw_frames):
            clean_frames = [None] * len(raw_frames)

        # Clear stale idle/pre-roll and segment state without resetting VAD.
        self._reset()
        self._speech = raw_frames
        self._speech_processed = clean_frames
        self._in_speech = True
        self._voiced_frames = max(1, self.min_frames)
        self._segment_id = uuid.uuid4().hex[:16]
        self._speech_started_at = time.perf_counter() - (
            len(raw_frames) * 512 / self.sample_rate
        )
        self._woke = True
        self._woke_phrase = canonical
        self._woke_source = "acoustic"
        self._woke_mid = True
        self._woke_at = time.perf_counter()
        self._confirmed_audio_handoff = True
        return True

    def _reset(self) -> None:
        self._pending_wake = None
        self._in_speech = False
        self._speech = []
        self._pre = []
        self._speech_processed = []
        self._pre_processed = []
        self._segment_id = ""
        self._overlap_of = ""
        self._segment_vad_peak = 0.0
        self._pending_rejections = []
        self._peek_errors = []
        self._weak_frames = []
        self._weak_pre.clear()
        self._weak_voiced = self._weak_silence = 0
        self._silence_run = 0
        self._voiced_frames = 0
        self._extending = False
        self._woke = False
        self._woke_phrase = ""
        self._woke_source = ""
        self._woke_mid = False
        self._woke_at = 0.0
        self._confirmed_audio_handoff = False
        self._next_peek = 0

    def flush(self) -> None:
        """Drop any partial utterance -- e.g. after we spoke a prompt."""
        self._reset()
        self.vad.reset()


def main() -> None:
    """Live monitor: shows every utterance heard and whether it triggered."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from .config import load
    from .stt import Transcriber

    cfg = load()
    stt = Transcriber(cfg.get("stt.model", "faster-whisper/small.en"))
    listener = SttListener(cfg, SileroVAD(), stt.transcribe)

    print("")
    print("Listening. Trigger phrases: {}".format(", ".join(listener.phrases)))
    print("Say e.g. \"memoir play red by seedhe maut\". Ctrl-C to stop.")
    print("")
    with Microphone(
        sample_rate=16000,
        frame_length=512,
        device=cfg.get("audio.input_device") or None,
        capture_rate=cfg.get("audio.capture_rate"),
        exclusive=cfg.get("audio.exclusive_mode", False),
    ) as mic:
        try:
            while True:
                utt = listener.feed(mic.read())
                if utt is None:
                    continue
                if utt.triggered:
                    print("  TRIGGER [{}] -> command {!r}   ({} ms)".format(
                        utt.matched_phrase, utt.command, utt.latency_ms))
                else:
                    print("  ignored: {!r}".format(utt.transcript[:70]))
        except KeyboardInterrupt:
            print("")


if __name__ == "__main__":
    main()
