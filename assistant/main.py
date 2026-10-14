"""Memoir: always-on voice control for Spotify.

The loop is deliberately dull. Everything interesting -- how confident the wake
word was, what was heard, what was inferred, what Spotify did -- is written to
telemetry, because most of the remaining design questions can only be settled
with data from the actual room.
"""
from __future__ import annotations

import logging
import math
import re
import signal
import sys
import threading
import time
from contextlib import contextmanager, nullcontext

import numpy as np

from . import catalogue, ducking, health
from .audio import Endpointer, Microphone, SileroVAD
from .config import ROOT, load
from .diag import EventFormatter, EventIdFilter, WakeTrace, bind_event, unbind_event
from .runtime import AlreadyRunning, single_instance
from .feedback import Feedback, KIND_SPEECH
from .intent import Action, Intent, parse
from .resolution import Candidate, MusicConfirmationRequired, _norm
from .spotify import (AmbiguousMatch, Spotify, SpotifyAuthError, SpotifyQuotaExceeded,
                      SpotifyRateLimited,
                      SpotifyRequestDeferred, TrackLookupError)
from .stt import Transcriber
from .telemetry import Event, Telemetry
from .listener import SttListener
from .async_runtime import AsyncCaptureRuntime, validate_async_config
from .async_listener import AcousticProposal
from .power import KeepAwakeGuard
from .wake import TIER_BOTH, WakeDetector, WakeWordError

log = logging.getLogger("memoir")

VOLUME_STEP = 15
MUSIC_CLARIFICATION_SECONDS = 30.0


class ClarificationExpired(TimeoutError):
    """A music choice has no authority after its absolute monotonic deadline."""


def _check_music_deadline(deadline: float | None) -> None:
    if deadline is not None and time.monotonic() >= deadline:
        raise ClarificationExpired("Music clarification expired")


class _WarmAwareRouter:
    """Guard only model routing; deterministic global commands do not use this wrapper."""
    def __init__(self, router, brain, runtime=None):
        self.router, self.brain, self.runtime = router, brain, runtime

    def route(self, text):
        if self.brain.warming:
            raise RuntimeError("Local model is still warming")
        if self.runtime is not None and not self.runtime.can_attempt():
            raise RuntimeError("Local model runtime is not accepting requests")
        return self.router.route(text)


# How long a duck triggered by the wake model is held while we wait for the
# command. Restored early once the utterance is handled.
DUCK_HOLD_SECONDS = 6.0
PAUSED_MIC_STALL_SECONDS = 3.0
# Failure speech stays short: the user asked for "Sorry." rather than
# instructions, and every extra word is time before they can try again.
SORRY = "Sorry."
PAUSE_FLAG = ROOT / "assistant" / "logs" / ".listening_paused"
NOT_FOUND_MESSAGE = "Sorry, I couldn't find that."
AMBIGUOUS_MESSAGE = "Sorry. Say the song and artist."
RATE_LIMITED_MESSAGE = "Spotify search is temporarily unavailable."
SPOTIFY_QUOTA_MESSAGE = "Spotify API quota is exhausted. Local controls may still work."
# One start_playback call carries the song plus at most this many radio tracks.
RADIO_LIMIT = 49
# A catalogue hit scores 1.00-1.02 when the whole heard title(+artist) matched.
QUEUE_EXACT_SCORE = 0.98


def _is_active_session_direct_question(command: str) -> bool:
    """Recognize only direct question-shaped language while a session is active.

    Punctuation is often omitted by speech recognition, so the opening words
    carry the signal. Quoted/reported text and imperative ``explain my code``
    remain session payload rather than an interruption.
    """
    if not isinstance(command, str):
        return False
    text = command.strip()
    if not text or text[0] in {'"', "'", "“", "‘"}:
        return False
    if re.match(r"(?i)^(he|she|they|you)\s+(said|asked|mentioned)\b", text):
        return False
    words = text.split()
    if len(words) >= 3 and re.match(r"(?i)^(what|why|how|when|where|who|which)\b", text):
        return True
    if re.match(r"(?i)^define\s+\S+", text):
        return True
    return re.match(
        r"(?i)^explain\s+(what|why|how|when|where|who|which|the\s+meaning\s+of|the\s+term)\b",
        text,
    ) is not None


def _perf_attr(obj, name: str) -> float:
    """A perf_counter timestamp from an utterance-like object, else 0.0."""
    value = getattr(obj, name, 0.0) if obj is not None else 0.0
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0


def _text_attr(obj, name: str) -> str:
    value = getattr(obj, name, "") if obj is not None else ""
    return value if isinstance(value, str) else ""


def _first_sentence(text: str) -> str:
    """Drop a decoding loop: "Play Red by X. Play Red by X. Play Red" -> the
    first copy. Text that does not repeat itself is returned unchanged, so
    titles such as "Mr. Brightside" survive."""
    text = text.strip()
    parts = [p.strip() for p in re.split(r"(?<=[.!?,])\s+", text) if p.strip()]
    if len(parts) > 1:
        first = _norm(parts[0])
        if first and any(_norm(p) and (_norm(p) == first or first.startswith(_norm(p)))
                         for p in parts[1:]):
            return parts[0].rstrip(".!?,")
    return text


class Assistant:
    def __init__(self) -> None:
        self.cfg = load()
        self.mode = self.cfg.get("listener.mode", "stt")
        self.duck_trigger = self.cfg.get("ducking.duck_trigger", "wakeword")
        self.async_enabled = validate_async_config(self.cfg)
        self.telemetry = Telemetry(self.cfg)
        self.feedback = Feedback(self.cfg)
        self._stop = threading.Event()
        self._busy = threading.Lock()
        self._trigger = threading.Event()
        self._trigger_source = "hotkey"
        self._paused_mic = None
        self._paused_listener = None
        self._paused_mic_last_frame_at = 0.0
        self._music_playing = True  # fail closed until the first playback check
        self._paused_mic_retry_at = 0.0

        # A level orphaned by a previous hard kill is the one case the ducking
        # context manager cannot cover, so repair it before anything else.
        ducking.repair_orphaned_duck()

        self.spotify = Spotify()
        try:
            self.spotify.check_auth()
        except SpotifyQuotaExceeded:
            log.error("%s", SPOTIFY_QUOTA_MESSAGE)
        except SpotifyRequestDeferred as exc:
            log.info("Spotify request deferred locally: %s", exc)
        except SpotifyRateLimited as exc:
            # Pause/play/next still work through Windows media controls.
            log.error("%s; local controls still work, song search waits.", exc)
        except Exception as exc:
            # Authentication/network health is not microphone health. A DNS
            # outage must not crash-loop the whole local listener.
            log.warning("Spotify startup check unavailable (%s); local listening continues", exc)

        self.feedback.prime()   # synthesise the wake prompt once, not per command
        # In STT mode only the high acoustic tier ducks early. Exact text
        # triggering remains independent for acoustic misses.
        self.wake = (
            WakeDetector(self.cfg)
            if self.mode == "wakeword" or self.duck_trigger == "wakeword"
            else None
        )
        self.vad = SileroVAD(16000)
        self.stt = Transcriber(self.cfg.get("stt.model", "faster-whisper/small.en"), 16000)
        self.listener_stt = (
            Transcriber(self.cfg.get("listener.stt_model"), 16000)
            if self.mode == "stt" and self.cfg.get("listener.stt_model")
            else self.stt
        )
        if self.listener_stt is not self.stt:
            self.listener_stt.set_wake_mode()
        self._speech_duck = None      # volume held while someone is speaking
        self._duck_lock = threading.Lock()
        self._duck_expires = 0.0
        self._early_lease = ducking.DuckLease(
            self.cfg.get("ducking.duck_to_percent", 20), DUCK_HOLD_SECONDS
        ) if self.duck_trigger == "wakeword" else None
        self._speech_lease = None
        self._last_strong_wake = None
        self._last_reject_log_at = float("-inf")
        self._last_reject_archive_at = float("-inf")
        self._rejection_counts = {}
        # Diagnostics: openWakeWord scores around the current utterance, and
        # when the early duck began (see _open_timeline).
        self._wake_trace = WakeTrace(self.cfg.get("wake.threshold_low", 0.15))
        self._early_duck_at = 0.0
        self._reject_scores = {}
        self.listener = (
            SttListener(
                self.cfg, self.vad, self.listener_stt.transcribe,
                on_speech_start=(self._duck_for_speech
                                 if self.duck_trigger == "speech" else None),
                on_speech_end=(self._undock_after_speech
                               if self.duck_trigger == "speech" else None),
                on_wake=self._duck_on_wake,
                authorize_trigger=self._authorize_text_trigger,
                on_reject=self._record_text_rejection,
                verify_trigger=self._verify_text_trigger,
                on_near_miss=self._record_near_miss,
                on_candidate=self._record_wake_candidate)
            if self.mode == "stt" else None
        )
        self.async_capture = (
            AsyncCaptureRuntime(
                self.cfg, self.vad,
                {"listener": self.listener_stt.transcribe,
                 "command": self.stt.transcribe},
                self._match_verified_trigger, self.telemetry)
            if self.async_enabled else None
        )
        self._async_cued_segments = set()
        self.echo = None
        self._echo_checked_at = 0.0
        self._capture_input_state = "unknown"
        self._capture_zero_windows = 0
        self._capture_silence_warning_emitted = False
        self.vocabulary = self._load_vocabulary()
        self.brain = None
        self._model_runtime = None
        if (self.cfg.get("intent.backend") == "brain"
                or self.cfg.get("conversation.enabled", False) is True):
            from .brain import Brain
            from .model_runtime import LocalModelRuntime
            model = self.cfg.get("intent.model", "qwen3:4b-instruct-2507-q4_K_M")
            num_ctx = 8192 if self.cfg.get("conversation.enabled", False) is True else 4096
            self._model_runtime = LocalModelRuntime(
                model, num_ctx=num_ctx, keep_alive=-1, cooldown_seconds=30)
            self.brain = Brain(model,
                               self.cfg.get("intent.timeout_seconds", 4.0),
                               context=self._brain_context, warmup_pending=True,
                               runtime=self._model_runtime)
            # Loading the model can take a while; never hold up listening.
            threading.Thread(target=self._warm_brain, daemon=True).start()
        self._conversation_coordinator = None
        self._conversation_recent = []
        if self.cfg.get("stt.bias_from_spotify", True):
            self.stt.set_vocabulary(self.vocabulary["hints"])
        # Recognition drops or mangles the word "by" ("Play Nakhshab, I
        # Seedhe Maut"), leaving track resolution with no artist to work
        # from. The artist list lets it find that boundary itself.
        self.spotify.set_known_artists(self.vocabulary["artists"])
        # Song names are matched by sound against everything these artists
        # released, so "madira by see them all" finds Seedhe Maut's "Madira".
        # The index loads lazily; this keeps it fresh, slowly, off the
        # command path, and never while Spotify is rate-limiting the app.
        if self.cfg.get("spotify.catalogue_auto_refresh", False):
            threading.Thread(target=self._keep_catalogue_fresh, daemon=True).start()

        self.endpointer = Endpointer(
            self.vad,
            sample_rate=16000,
            frame_length=512,
            silence_ms=self.cfg.get("capture.silence_ms", 800),
            max_seconds=self.cfg.get("capture.max_seconds", 23.5),
            speech_timeout_ms=self.cfg.get("capture.speech_timeout_ms", 1500),
            threshold=self.cfg.get("capture.vad_threshold", 0.5),
            utterance_seconds=self.cfg.get("capture.utterance_seconds", 20.0),
        )
        # The Codex client is lazy: no subprocess, remote request, or additional
        # local model is loaded until an explicitly addressed global command.
        self.global_assistant = None
        if self.cfg.get("codex.enabled", False):
            try:
                from .global_assistant import GlobalAssistant
                from .global_router import LocalGlobalRouter
                self.global_assistant = GlobalAssistant(self.cfg, ROOT,
                    router=_WarmAwareRouter(
                        LocalGlobalRouter(self.brain.model, runtime=self._model_runtime),
                        self.brain, runtime=self._model_runtime)
                    if self.brain is not None else None)
                log.info("Global voice tools enabled: Codex %s, lazy ChatGPT connection, project Memoir",
                         self.cfg.get("codex.model", "gpt-5.6-luna"))
            except Exception as exc:
                self._global_fault = True
                log.warning("Global voice tools unavailable (%s); audio and music remain active", type(exc).__name__)

    @staticmethod
    def _discard_while_busy(mic):
        """Explicit non-listening work; never wrap a reply recording."""
        discard = getattr(type(mic), "discard_capture", None)
        return mic.discard_capture() if callable(discard) else nullcontext()

    def _say_feedback(self, mic, message, event=None):
        """Block until speech and its device tail finish, then release capture."""
        started = time.perf_counter()
        finished = None
        if event is not None:
            event.feedback_text = message
            event.mark("feedback_start", overwrite=True)
        try:
            with self._discard_while_busy(mic):
                kind = self.feedback.say(message, force=True, blocking=True)
                finished = time.perf_counter()
                if event is not None:
                    event.mark("feedback_done", at=finished, overwrite=True)
                if mic is not None:
                    self._settle_after_prompt(mic)
                    if event is not None:
                        event.mark("feedback_settled", overwrite=True)
            listener = getattr(self, "listener", None)
            if listener is not None:
                listener.flush()
            if event is not None:
                event.feedback_kind = kind
            return kind
        finally:
            if event is not None:
                elapsed = int(((finished or time.perf_counter()) - started) * 1000)
                event.feedback_latency_ms = (event.feedback_latency_ms or 0) + elapsed
                if finished is None:
                    event.mark("feedback_done", overwrite=True)

    def _global_say(self, mic, message, event, *, review=False):
        """Only the audio owner speaks; the review clock pauses during feedback."""
        if not message:
            return False
        controller = getattr(self, "global_assistant", None)
        begin = getattr(controller, "begin_feedback", None)
        end = getattr(controller, "end_feedback", None)
        token = begin(review=review) if callable(begin) else None
        success = False
        valid = True
        try:
            with self._duck_music(event, self.cfg.get("ducking.duck_to_percent", 20)):
                # A late device tail must not be the send instruction itself.
                text = message.rstrip() + " Ready."
                success = self._say_feedback(mic, text, event) == KIND_SPEECH
        finally:
            if callable(end):
                valid = end(token, success)
        return success and valid

    def _transcribe_captured(self, mic, audio):
        """Decode a completed recording while intentionally not listening."""
        with self._discard_while_busy(mic):
            return self.stt.transcribe(audio)

    @staticmethod
    def _capture_result(event, captured):
        event.capture_end_reason = captured.get("end_reason", "")
        event.capture_truncated = captured.get("truncated", False) is True
        wait_ms = captured.get("wait_duration_ms")
        event.capture_wait_ms = None
        if (isinstance(wait_ms, (int, float)) and not isinstance(wait_ms, bool)
                and math.isfinite(wait_ms) and wait_ms >= 0):
            event.capture_wait_ms = int(wait_ms)

    def _handle_global(self, command, mic, event):
        controller = getattr(self, "global_assistant", None)
        # Refuse a cut-off prefix even when Codex is disabled: a truncated
        # music or desktop request is no safer to execute than a partial prompt.
        if getattr(event, "capture_truncated", False) is True:
            if controller is not None:
                controller.end_session()
            self._global_say(mic, "That request was cut short. Nothing sent. Please repeat the full request.", event)
            event.outcome = "capture_truncated"
            return True
        if controller is None:
            if getattr(self, "_global_fault", False):
                from .global_commands import parse_global
                if parse_global(command) is not None:
                    self._global_say(mic, "Codex voice tools are unavailable. Music control is still available.", event)
                    event.outcome = "global_unavailable"
                    return True
            return False
        # Desktop capabilities share the existing brain. Do not classify the
        # same app/search request through Codex and then through that brain.
        # A held Codex interaction retains priority until it ends or is cancelled.
        from .global_assistant import _PendingChoice, _UnsentRequest
        held = getattr(controller, "_unsent_request", None)
        global_context = (
            getattr(getattr(controller, "session", None), "active", False) is True
            or isinstance(getattr(controller, "_pending_choice", None), _PendingChoice)
            or isinstance(held, _UnsentRequest) and controller._valid_unsent(held) is True)
        if not global_context and not re.search(r"\b(?:codex|chat|conversation|thread)\b", command, re.I):
            from .computer_tools import desktop_domain
            if desktop_domain(command):
                return False
        from .desktop_controls import NavigationError
        brain = getattr(self, "brain", None)
        if getattr(brain, "warming", False) is True:
            from .global_commands import parse_global
            from .global_router import routing_candidate
            if not global_context and parse_global(command) is None and routing_candidate(command):
                self._global_say(mic, "The local model is still warming. Try that request again shortly.", event)
                event.outcome = "global_warming"
                return True
        try:
            with self._discard_while_busy(mic):
                reply = controller.handle(command)
        except NavigationError as exc:
            self._global_say(mic, str(exc), event)
            event.outcome = "global_refused"
            return True
        if reply is None:
            return False
        event.intent_action = "GLOBAL"
        event.intent_source = "global"
        event.request_phase = "global"
        event.global_action = "create_pending" if reply.await_token else ("session" if reply.session else "control")
        spoken = self._global_say(mic, reply.message, event,
                                  review=reply.review)
        event.outcome = "global_handled"
        if reply.await_token:
            reply = self._await_global_creation(mic, event, reply.await_token)
            if reply is None:
                return True
            event.global_action = "review" if reply.session else "create_ended"
            event.global_target = controller.session.target_id if reply.session else ""
            event.global_prompt_chars = len(controller.session.draft) if reply.session else 0
            spoken = self._global_say(mic, reply.message, event,
                                      review=reply.review)
            if getattr(reply, "status", "") == "failed":
                event.global_status = "failed"
                event.outcome = "global_failed"
        if reply.session:
            if spoken:
                self._run_global_session(mic, event)
            else:
                controller.end_session()
                event.outcome = "global_feedback_failed"
        return True

    def _await_global_creation(self, mic, event, token):
        """Continue this explicit interaction briefly; never start late capture."""
        deadline = time.monotonic() + 8.0
        controller = self.global_assistant
        with self._duck_music(event, self.cfg.get("ducking.duck_to_percent", 20)):
            while not self._stop.is_set() and time.monotonic() < deadline:
                reply = controller.continue_creation(token)
                if reply is not None:
                    mic.flush()
                    if self.listener is not None:
                        self.listener.flush()
                    return reply
                try:
                    mic.read(timeout_s=0.1)
                except TimeoutError:
                    continue
        if not self._stop.is_set():
            self._global_say(mic, "Still working. Ask for Codex status shortly. Nothing sent.", event)
            event.outcome = "global_pending"
        mic.flush()
        if self.listener is not None:
            self.listener.flush()
        return None

    def _run_global_session(self, mic, event):
        """Bounded local dictation, separate from wake and music normalization.

        Intentional command mode owns capture for this session. Remote work is
        dispatched only after explicit send confirmation, and never waited on.
        """
        controller = self.global_assistant
        session = controller.session
        start_ms = self.cfg.get("codex.segment_start_timeout_ms", 15000)
        utterance_seconds = self.cfg.get("codex.segment_seconds", 20.0)
        preserve_review = False
        try:
            while session.active and not self._stop.is_set():
                # Short confirmations need less trailing silence than dictated
                # tasks. Neither setting reduces the full speech budget.
                silence_ms = self.cfg.get(
                    "codex.confirmation_silence_ms" if session.phase == "review"
                    else "codex.dictation_silence_ms",
                    600 if session.phase == "review" else 1000)
                pointer = Endpointer(
                    self.vad, sample_rate=16000, frame_length=512,
                    silence_ms=silence_ms,
                    max_seconds=start_ms / 1000 + utterance_seconds,
                    utterance_seconds=utterance_seconds,
                    speech_timeout_ms=start_ms,
                    threshold=self.cfg.get("capture.vad_threshold", 0.35))
                with self._duck_music(event, self.cfg.get("ducking.duck_to_percent", 20)):
                    event.mark("global_capture_start", overwrite=True)
                    captured = pointer.record(mic, after_prompt=False)
                    event.mark("global_capture_done", overwrite=True)
                self._capture_result(event, captured)
                if event.capture_truncated:
                    controller.end_session()
                    self._global_say(mic, "That segment was cut short. Nothing sent. Please start again.", event)
                    event.outcome = "capture_truncated"
                    return
                if not captured["speech_started"]:
                    preserve_review = controller.keep_review_waiting()
                    message = ("I have kept your request. Say Alexa to continue." if preserve_review
                               else "Dictation timed out. Nothing sent.")
                    if not self._global_say(mic, message, event):
                        preserve_review = False
                    event.outcome = "global_review_waiting" if preserve_review else "global_timeout"
                    return
                extra = Event(event_id=event.event_id + "-dictation-" + str(time.monotonic_ns()))
                self.telemetry.save_audio(extra, captured["audio"])
                with self._discard_while_busy(mic):
                    text, _ = self.stt.transcribe(captured["audio"])
                extra.trigger_source = "global_dictation"
                extra.request_phase = session.phase
                extra.transcript = text
                extra.global_target = session.target_id
                extra.audio_duration_ms = int(len(captured["audio"]) / 16)
                if not text.strip():
                    extra.outcome = "global_unheard"
                    self.telemetry.write(extra)
                    preserve_review = controller.keep_review_waiting()
                    message = ("I have kept your request. Say Alexa to continue." if preserve_review
                               else "I couldn't hear that. Nothing sent.")
                    if not self._global_say(mic, message, event):
                        preserve_review = False
                    event.outcome = "global_review_waiting" if preserve_review else "global_unheard"
                    return
                reply = controller.consume(text)
                extra.global_action = reply.action
                extra.outcome = "global_" + reply.action
                extra.global_prompt_chars = len(reply.prompt or session.draft)
                self.telemetry.write(extra)
                if reply.action == "submit":
                    event.global_target = reply.target_id
                    event.global_prompt_chars = len(reply.prompt)
                    message = controller.submit_confirmed(reply)
                    self._global_say(mic, message, event)
                    event.outcome = "global_dispatched" if message in {"Sending the confirmed prompt.", "Creating the confirmed chat."} else "global_refused"
                    return
                if reply.action == "readback":
                    message = (controller.review_message(full=True)
                               + ". Say send it, start again, or cancel.")
                elif reply.action == "review" and session.draft:
                    message = controller.review_message(full=False)
                else:
                    message = reply.message
                if message and not self._global_say(
                        mic, message, event, review=reply.action in {"review", "readback"}):
                    event.outcome = "global_feedback_failed"
                    return
            if not self._stop.is_set():
                self._global_say(mic, "Dictation ended. Nothing sent.", event)
                event.outcome = "global_cancelled"
        finally:
            # Quiet review returns to wake mode; every other exit invalidates it.
            cleanup = getattr(controller, "end_session", None)
            if not preserve_review or self._stop.is_set():
                if callable(cleanup):
                    cleanup()
                else:
                    session.reset()
            mic.flush()
            if self.listener is not None:
                self.listener.flush()

    def _warm_brain(self) -> None:
        try:
            self.brain.warm_up()
        except Exception as exc:
            log.warning("Brain not ready (%s); rules answer until it is", exc)
        try:
            from .computer_tools import warm_catalog
            warm_catalog()
        except Exception as exc:
            log.debug("Desktop catalogue warmup unavailable (%s)", type(exc).__name__)

    def _listening_paused(self) -> bool:
        """True while logs/.listening_paused exists (wake-word recording
        sessions say "Alexa" dozens of times). Checked twice a second."""
        now = time.monotonic()
        if now - getattr(self, "_pause_checked_at", 0.0) >= 0.5:
            self._pause_checked_at = now
            was = getattr(self, "_paused_flag", False)
            self._paused_flag = PAUSE_FLAG.exists()
            if self._paused_flag != was:
                log.info("Listening %s", "paused" if self._paused_flag else "resumed")
                if self.mode == "stt" and self.listener is not None:
                    self.listener.flush()
                if self.wake is not None:
                    self.wake.reset()
        return getattr(self, "_paused_flag", False)

    def _keep_catalogue_fresh(self) -> None:
        if not self.cfg.get("spotify.catalogue_auto_refresh", False):
            return
        from . import catalogue
        while not self._stop.is_set():
            try:
                catalogue.refresh_in_background(self.spotify, self.vocabulary["artists"],
                                                enabled=True)
            except Exception as exc:
                log.warning("Catalogue refresh could not start: %s", exc)
            self._stop.wait(3600)

    def _brain_context(self) -> dict:
        voc = getattr(self, "vocabulary", None) or {}
        artists = voc.get("artists") or []
        playlists = [h for h in (voc.get("hints") or []) if h not in set(artists)]
        # Intent validation calls this too. Never perform blocking device or
        # network discovery while constructing/validating an LLM request.
        return {"artists": artists, "playlists": playlists, "now_playing": ""}

    def _conversation_ready(self) -> bool:
        """Share the intent brain's warmup gate; never warm a second model."""
        brain = getattr(self, "brain", None)
        if brain is not None and getattr(brain, "warming", False) is True:
            return False
        runtime = getattr(self, "_model_runtime", None)
        return runtime is None or runtime.can_attempt()

    def _conversation_instance(self):
        coordinator = getattr(self, "_conversation_coordinator", None)
        if coordinator is None:
            from .conversation_brain import ConversationBrain
            from .conversation import ConversationCoordinator
            cfg = getattr(self, "cfg", None)
            get = getattr(cfg, "get", lambda *_: "")
            interpreter = ConversationBrain(
                get("intent.model", "qwen3:4b-instruct-2507-q4_K_M"),
                timeout=get("conversation.timeout_seconds", 3.0),
                readiness=self._conversation_ready,
                runtime=getattr(self, "_model_runtime", None))
            coordinator = ConversationCoordinator(
                interpreter,
                self._run_conversation_step,
                self._run_conversation_delegate)
            self._conversation_coordinator = coordinator
        return coordinator

    def _run_conversation_step(self, step, operation_id):
        session = getattr(getattr(self, "_conversation_local", None), "session", None)
        if session is None:
            from .conversation_contracts import ToolResult
            return ToolResult(operation_id, "unknown_delivery", {}, "missing turn adapter")
        return self._execute_conversation_step(step, operation_id, session["event"],
                                               session["mic"], session["audio"])

    def _run_conversation_delegate(self, step, operation_id):
        session = getattr(getattr(self, "_conversation_local", None), "session", None)
        if session is None:
            from .conversation_contracts import ToolResult
            return ToolResult(operation_id, "unknown_delivery", {}, "missing turn adapter")
        from .conversation_contracts import ToolResult
        try:
            return self._delegate_conversation_step(step, operation_id, session["event"],
                                                    session["mic"], session["source"])
        except Exception:
            return ToolResult(operation_id, "unknown_delivery", {
                "feedback_spoken": bool(session["event"].feedback_kind)})

    def _conversation_context(self):
        return {"recent_turns": list(getattr(self, "_conversation_recent", [])[-3:])}

    def _conversation_reply_endpointer(self):
        """Lazily create a bounded reply recorder sharing the main VAD."""
        pointer = getattr(self, "_conversation_reply_pointer", None)
        if pointer is not None:
            return pointer
        base = getattr(self, "endpointer", None)
        vad = getattr(self, "vad", getattr(base, "vad", None))
        if base is None or vad is None:
            return None
        rate = getattr(base, "sample_rate", 16000)
        frame_length = max(1, int(round(rate * getattr(base, "frame_ms", 32.0) / 1000)))
        cfg = getattr(self, "cfg", None)
        get = getattr(cfg, "get", lambda _key, default=None: default)
        pointer = Endpointer(
            vad, sample_rate=rate, frame_length=frame_length,
            silence_ms=600, max_seconds=12.0, speech_timeout_ms=3500,
            threshold=getattr(base, "threshold", get("capture.vad_threshold", 0.5)),
            echo_window_ms=int(getattr(base, "echo_window_frames", 0)
                               * getattr(base, "frame_ms", 32.0)),
            echo_max_ms=int(getattr(base, "echo_max_frames", 0)
                            * getattr(base, "frame_ms", 32.0)),
            utterance_seconds=8.0)
        self._conversation_reply_pointer = pointer
        return pointer

    def _record_conversation_turn(self, coordinator, generation, user_text, assistant_text):
        recent = getattr(self, "_conversation_recent", None)
        if recent is not None and coordinator.generation == generation:
            recent.extend(({"role": "user", "text": user_text[:1000]},
                           {"role": "assistant", "text": assistant_text[:1000]}))
            del recent[:-3]

    def _mark_conversation_reply_invalidated(self, event, *, parent=False):
        event.feedback_text = ""
        event.feedback_kind = ""
        event.conversation_status = "invalidated"
        event.outcome = "conversation_invalidated"
        event.mark("handler_done")
        if not parent:
            self.telemetry.write(event)

    def _invalidate_conversation_reply_if_current(self, coordinator, generation,
                                                  event, *, parent=False):
        if not coordinator.invalidate_if_current(generation):
            self._mark_conversation_reply_invalidated(event, parent=parent)
            return None
        return generation + 1

    def _continue_conversation_clarification(self, parent_event, source, mic,
                                             coordinator, initial_result):
        """Speak each clarification and capture at most two wake-free replies."""
        pointer = self._conversation_reply_endpointer()
        if pointer is None or mic is None:
            return
        result = initial_result
        prompt_event = parent_event
        prompt_source = source
        for reply_index in range(2):
            generation = result.generation
            if coordinator.generation != generation:
                parent_event.feedback_text = ""
                parent_event.feedback_kind = ""
                parent_event.outcome = "conversation_invalidated"
                parent_event.conversation_status = "invalidated"
                return
            duck_to = getattr(getattr(self, "cfg", None), "get",
                              lambda _key, default=None: default)(
                                  "ducking.duck_to_percent", 20)
            captured = None
            child = None
            capture_error = None
            with self._duck_music(prompt_event, duck_to):
                prompt_event.feedback_text = result.text
                try:
                    prompt_event.feedback_kind = self._say_feedback(
                        mic, result.text, prompt_event)
                except Exception:
                    prompt_event.feedback_kind = ""
                if prompt_event.feedback_kind != KIND_SPEECH:
                    next_generation = self._invalidate_conversation_reply_if_current(
                        coordinator, generation, prompt_event,
                        parent=prompt_event is parent_event)
                    if next_generation is None:
                        return
                    if prompt_event is parent_event:
                        parent_event.outcome = "conversation_clarification_speech_failed"
                        parent_event.conversation_status = "clarification_speech_failed"
                    else:
                        prompt_event.outcome = "conversation_clarification_speech_failed"
                        prompt_event.conversation_status = "clarification_speech_failed"
                        prompt_event.mark("handler_done")
                        self.telemetry.write(prompt_event)
                    return
                if coordinator.generation != generation:
                    self._mark_conversation_reply_invalidated(
                        prompt_event, parent=prompt_event is parent_event)
                    return
                self._record_conversation_turn(coordinator, generation,
                                               prompt_source, result.text)
                if prompt_event is not parent_event:
                    prompt_event.mark("handler_done")
                    self.telemetry.write(prompt_event)
                child = Event(trigger_source="conversation_reply",
                              input_device=getattr(parent_event, "input_device", ""),
                              conversation_parent_event_id=parent_event.event_id)
                child.request_phase = "conversation_reply_capture"
                began = time.perf_counter()
                child.set_origin(began)
                child.mark("conversation_reply_start", began)
                child_token = bind_event(child)
                try:
                    child.mark("capture_start")
                    captured = pointer.record(mic, after_prompt=False)
                    child.mark("capture_done")
                    if captured.get("speech_started"):
                        self.feedback.stopped()
                except Exception as exc:
                    capture_error = type(exc).__name__
                finally:
                    unbind_event(child_token)
            if child is None:
                return
            if capture_error is not None:
                next_generation = self._invalidate_conversation_reply_if_current(
                    coordinator, generation, child)
                if next_generation is None:
                    return
                if coordinator.generation != next_generation:
                    self._mark_conversation_reply_invalidated(child)
                    return
                child.error = capture_error
                child.conversation_kind = "unsupported"
                child.conversation_status = "capture_failed"
                child.outcome = "conversation_reply_capture_failed"
                child.feedback_text = "Ask me again when ready."
                child_token = bind_event(child)
                try:
                    with self._duck_music(child, duck_to):
                        child.feedback_kind = self._say_feedback(
                            mic, child.feedback_text, child)
                finally:
                    unbind_event(child_token)
                self.telemetry.write(child)
                return
            self._capture_result(child, captured)
            child.audio_duration_ms = int(captured.get("duration_ms", 0) or 0)
            audio = captured.get("audio")
            child.audio_path = self.telemetry.save_audio(child, audio)
            child.mark("reply_capture_saved")
            if coordinator.generation != generation:
                child.conversation_kind = "clarify"
                child.conversation_status = "invalidated"
                child.outcome = "conversation_invalidated"
                child.mark("handler_done")
                self.telemetry.write(child)
                return
            if child.capture_truncated:
                next_generation = self._invalidate_conversation_reply_if_current(
                    coordinator, generation, child)
                if next_generation is None:
                    return
                if coordinator.generation != next_generation:
                    self._mark_conversation_reply_invalidated(child)
                    return
                child.conversation_kind = "clarify"
                child.conversation_status = "truncated"
                child.feedback_text = "That reply was cut short. Ask me again when ready."
                child_token = bind_event(child)
                try:
                    with self._duck_music(child, duck_to):
                        child.feedback_kind = self._say_feedback(
                            mic, child.feedback_text, child)
                finally:
                    unbind_event(child_token)
                child.outcome = "conversation_reply_truncated"
                child.mark("handler_done")
                self.telemetry.write(child)
                return
            if not captured.get("speech_started"):
                next_generation = self._invalidate_conversation_reply_if_current(
                    coordinator, generation, child)
                if next_generation is None:
                    return
                if coordinator.generation != next_generation:
                    self._mark_conversation_reply_invalidated(child)
                    return
                child.conversation_kind = "clarify"
                child.conversation_status = "reply_timeout"
                child.feedback_text = "Ask me again when ready."
                child_token = bind_event(child)
                try:
                    with self._duck_music(child, duck_to):
                        child.feedback_kind = self._say_feedback(
                            mic, child.feedback_text, child)
                finally:
                    unbind_event(child_token)
                child.outcome = "conversation_reply_timeout"
                child.mark("handler_done")
                self.telemetry.write(child)
                return
            if coordinator.generation != generation:
                child.conversation_kind = "clarify"
                child.conversation_status = "invalidated"
                child.outcome = "conversation_invalidated"
                child.mark("handler_done")
                self.telemetry.write(child)
                return
            try:
                child.request_phase = "conversation_reply_transcribe"
                child.mark("stt_start")
                child_token = bind_event(child)
                try:
                    reply, child.stt_latency_ms = self._transcribe_captured(mic, audio)
                finally:
                    unbind_event(child_token)
                child.mark("stt_done")
            except Exception as exc:
                next_generation = self._invalidate_conversation_reply_if_current(
                    coordinator, generation, child)
                if next_generation is None:
                    return
                if coordinator.generation != next_generation:
                    self._mark_conversation_reply_invalidated(child)
                    return
                child.error = type(exc).__name__
                child.conversation_kind = "unsupported"
                child.conversation_status = "reply_stt_failed"
                child.feedback_text = "I couldn't process that reply. Ask me again when ready."
                child_token = bind_event(child)
                try:
                    with self._duck_music(child, duck_to):
                        child.feedback_kind = self._say_feedback(
                            mic, child.feedback_text, child)
                finally:
                    unbind_event(child_token)
                child.outcome = "conversation_reply_stt_failed"
                child.mark("handler_done")
                self.telemetry.write(child)
                return
            if not isinstance(reply, str):
                reply = ""
            if coordinator.generation != generation:
                child.conversation_kind = "clarify"
                child.conversation_status = "invalidated"
                child.outcome = "conversation_invalidated"
                child.mark("handler_done")
                self.telemetry.write(child)
                return
            child.transcript = reply
            child.stt_model = self.stt.model_name
            child.command_stt_model = self.stt.model_name
            if not reply.strip():
                next_generation = self._invalidate_conversation_reply_if_current(
                    coordinator, generation, child)
                if next_generation is None:
                    return
                if coordinator.generation != next_generation:
                    self._mark_conversation_reply_invalidated(child)
                    return
                child.conversation_kind = "clarify"
                child.conversation_status = "reply_empty"
                child.feedback_text = "Ask me again when ready."
                child_token = bind_event(child)
                try:
                    with self._duck_music(child, duck_to):
                        child.feedback_kind = self._say_feedback(
                            mic, child.feedback_text, child)
                finally:
                    unbind_event(child_token)
                child.outcome = "conversation_reply_empty"
                child.mark("handler_done")
                self.telemetry.write(child)
                return
            adapter = getattr(self, "_conversation_local", None)
            if adapter is None:
                adapter = self._conversation_local = threading.local()
            adapter.session = {"event": child, "mic": mic, "audio": audio,
                               "source": reply}
            child_token = bind_event(child)
            if coordinator.generation != generation:
                adapter.session = None
                unbind_event(child_token)
                self._mark_conversation_reply_invalidated(child)
                return
            dispatch_generation = generation + 1
            try:
                from .conversation_contracts import Transcript
                result = coordinator.handle(
                    Transcript(reply, turn_id=child.event_id,
                               truncated=child.capture_truncated),
                    context=self._conversation_context(),
                    expected_generation=generation)
            except Exception as exc:
                next_generation = self._invalidate_conversation_reply_if_current(
                    coordinator, dispatch_generation, child)
                if next_generation is None:
                    return
                if coordinator.generation != next_generation:
                    self._mark_conversation_reply_invalidated(child)
                    return
                child.error = type(exc).__name__
                child.conversation_kind = "unsupported"
                child.conversation_status = "reply_interpreter_failed"
                child.outcome = "conversation_reply_interpreter_failed"
                child.feedback_text = "I couldn't process that reply. Ask me again when ready."
                with self._duck_music(child, duck_to):
                    child.feedback_kind = self._say_feedback(mic, child.feedback_text, child)
                self.telemetry.write(child)
                return
            finally:
                adapter.session = None
                unbind_event(child_token)
            child.conversation_kind = result.decision_kind
            child.conversation_status = result.outcome
            child.conversation_latency_ms = result.latency_ms
            child.conversation_step_count = len(result.results)
            child.conversation_results_summary = ",".join(
                item.status for item in result.results)[:120]
            child.mark("conversation_done")
            if result.outcome == "invalidated" or coordinator.generation != result.generation:
                child.conversation_status = "invalidated"
                child.outcome = "conversation_invalidated"
                child.mark("handler_done")
                self.telemetry.write(child)
                return
            if result.text:
                child.feedback_text = result.text
            child.outcome = "conversation_" + result.outcome
            if result.decision_kind == "no_request":
                child.feedback_text = ""
                child.feedback_kind = ""
                child.outcome = "conversation_no_request"
                child.mark("handler_done")
                self.telemetry.write(child)
                return
            if result.text:
                prompt_event = child
                prompt_source = reply
                # The next loop iteration speaks this result and, only for a
                # live clarification, captures one more bounded reply.
                if result.outcome == "clarification_needed" and reply_index == 0:
                    continue
                child_token = bind_event(child)
                try:
                    with self._duck_music(child, duck_to):
                        child.feedback_kind = self._say_feedback(mic, result.text, child)
                finally:
                    unbind_event(child_token)
                if coordinator.generation != result.generation:
                    child.feedback_text = ""
                    child.feedback_kind = ""
                    child.conversation_status = "invalidated"
                    child.outcome = "conversation_invalidated"
                    child.mark("handler_done")
                    self.telemetry.write(child)
                    return
            self._record_conversation_turn(coordinator, result.generation,
                                           reply, result.text or child.feedback_text)
            child.mark("handler_done")
            self.telemetry.write(child)
            return

    def _execute_conversation_step(self, step, operation_id, event, mic, audio):
        from .conversation_contracts import ToolResult, step_to_intent
        intent = step_to_intent(step)
        # Reuse the turn event for telemetry, but clear operation evidence so
        # an earlier successful step cannot make a later no-op look complete.
        # Transitional receipt coverage is limited to this adapter's live
        # event outcome; there is not yet a generic durable job store.
        event.outcome = ""
        event.error = ""
        event.feedback_text = ""
        event.feedback_kind = ""
        event.global_status = ""
        event.intent_action = ""
        event.intent_query = ""
        event.resolution_path = ""
        event.resolved_label = ""
        event.playback_method = ""
        event.spotify_action = ""
        event.intent_action = intent.action.value
        event.intent_query = intent.query
        event.intent_source = "conversation"
        try:
            self._execute(intent, event, command_audio=audio, mic=mic)
        except MusicConfirmationRequired as exc:
            self._handle_music_confirmation(mic, event, exc)
        except AmbiguousMatch as exc:
            self._handle_ambiguous_track(mic, event, exc)
        except SpotifyQuotaExceeded:
            event.outcome = "spotify_quota_exceeded"
            event.error = "SpotifyQuotaExceeded"
            event.feedback_text = SPOTIFY_QUOTA_MESSAGE
            event.feedback_kind = self._say_feedback(mic, SPOTIFY_QUOTA_MESSAGE, event) if mic is not None else ""
        except SpotifyRequestDeferred as exc:
            event.outcome = "request_deferred"
            event.error = "request_deferred"
            message = "Spotify requests are busy. Please try again shortly."
            event.feedback_text = message
            event.feedback_kind = self._say_feedback(mic, message, event) if mic is not None else ""
        except Exception as exc:
            # A generic exception may follow a side effect. Preserve an
            # already-recorded receipt; otherwise report delivery uncertainty.
            if event.outcome not in {"ok", "requested", "global_handled", "global_pending",
                                      "global_review_waiting", "not_found", "desktop_refused",
                                      "request_deferred", "ambiguous"}:
                event.outcome = "unknown_delivery"
                event.error = type(exc).__name__
        legacy = event.outcome
        if legacy == "ok":
            status = "completed"
        elif legacy in {"requested", "global_handled", "global_pending", "global_review_waiting"}:
            status = "accepted"
        elif legacy == "request_deferred":
            status = "deferred"
        elif legacy in {"unknown_delivery", "desktop_unknown_delivery"}:
            status = "unknown_delivery"
        elif legacy == "spotify_quota_exceeded":
            status = "failed"
        elif legacy in {"desktop_refused", "error", "not_found", "ambiguous"}:
            status = "failed"
        else:
            status = "unknown_delivery"
        return ToolResult(operation_id, status,
                          {"legacy_outcome": legacy, "action": intent.action.value,
                           "feedback_spoken": bool(event.feedback_kind)})

    def _delegate_conversation_step(self, step, operation_id, event, mic, source):
        from .conversation_contracts import ToolResult
        from .global_router import explicit_codex_request
        event.outcome = ""
        event.error = ""
        event.feedback_text = ""
        event.feedback_kind = ""
        event.global_status = ""
        if not explicit_codex_request(source):
            return ToolResult(operation_id, "failed", {"legacy_outcome": "global_refused",
                                                         "feedback_spoken": False})
        # Preserve the existing global bridge, confirmation flow and complete
        # original transcript. The interpreter never supplies a prompt payload.
        if not self._handle_global(source, mic, event):
            return ToolResult(operation_id, "unknown_delivery", {"feedback_spoken": False})
        outcome = getattr(event, "outcome", "")
        status = ("failed" if outcome in {"global_refused", "global_failed", "global_unavailable",
                                          "global_feedback_failed"}
                  else "accepted" if outcome.startswith("global_") else "unknown_delivery")
        return ToolResult(operation_id, status, {"legacy_outcome": outcome,
                                                   "global_status": getattr(event, "global_status", ""),
                                                   "feedback_spoken": bool(event.feedback_kind)})

    def _handle_conversation(self, command, mic, event, command_audio=None) -> bool:
        """Handle one final transcript when the explicitly gated path is enabled."""
        get = getattr(getattr(self, "cfg", None), "get", None)
        if not callable(get) or get("conversation.enabled", False) is not True:
            return False
        # The legacy global state machine remains sole owner of active review,
        # exact controls, and explicit delegation. Ordinary questions can
        # still leave an unrelated active dictation/review session.
        controller = getattr(self, "global_assistant", None)
        if controller is not None:
            from .global_assistant import _PendingChoice, _UnsentRequest
            from .global_commands import _control_text, parse_global
            from .global_router import explicit_codex_request
            held = getattr(controller, "_unsent_request", None)
            session = getattr(controller, "session", None)
            active = getattr(session, "active", False) is True
            phase = getattr(session, "phase", "")
            if explicit_codex_request(command) or parse_global(command) is not None:
                return False
            direct_question = _is_active_session_direct_question(command)
            pending_choice = isinstance(getattr(controller, "_pending_choice", None), _PendingChoice)
            held_request = (isinstance(held, _UnsentRequest)
                            and controller._valid_unsent(held) is True)
            if (pending_choice or held_request) and not direct_question:
                return False
            if active:
                control = _control_text(command)
                owned_controls = {"send it", "send the prompt", "submit prompt", "sendit",
                                  "sendtheprompt", "submitprompt", "read it back", "read back",
                                  "readitback", "cancel", "cancel that", "stop", "stop that",
                                  "start again", "startagain", "edit prompt", "editprompt",
                                  "status", "codex status", "finish", "finish dictation",
                                  "finishdictation"}
                if control in owned_controls:
                    return False
                words = command.strip().split()
                if phase == "review" and len(words) <= 3 and not direct_question:
                    # Keep ambiguous short confirmations/replies with the
                    # existing confirmation owner; a full question is central.
                    return False
                if phase == "dictating":
                    if not direct_question:
                        return False
        # A wake acknowledgement or prior operation's feedback is not evidence
        # for this newly interpreted turn.
        event.feedback_text = ""
        event.feedback_kind = ""
        event.outcome = ""
        from .conversation_contracts import Transcript
        adapter = getattr(self, "_conversation_local", None)
        if adapter is None:
            adapter = self._conversation_local = threading.local()
        adapter.session = {"event": event, "mic": mic, "audio": command_audio,
                           "source": command}
        try:
            coordinator = self._conversation_instance()
            result = coordinator.handle(
                Transcript(command, turn_id=getattr(event, "event_id", ""),
                           truncated=getattr(event, "capture_truncated", False) is True),
                context=self._conversation_context())
        except Exception as exc:
            # Construction/readiness failures are handled by this route too;
            # do not fall through to the legacy parser for the same utterance.
            event.conversation_kind = "unsupported"
            event.conversation_status = "interpreter_unavailable"
            event.conversation_latency_ms = None
            event.conversation_step_count = 0
            event.conversation_results_summary = ""
            event.request_phase = "conversation"
            event.error = type(exc).__name__
            message = "I couldn't start conversation interpretation. Please try again."
            event.feedback_text = message
            event.feedback_kind = self._say_feedback(mic, message, event)
            event.outcome = "conversation_interpreter_unavailable"
            return True
        finally:
            adapter.session = None
        event.conversation_kind = result.decision_kind
        event.conversation_status = result.outcome
        event.conversation_latency_ms = result.latency_ms
        event.conversation_step_count = len(result.results)
        event.conversation_results_summary = ",".join(r.status for r in result.results)[:120]
        event.request_phase = "conversation"
        event.mark("conversation_done", at=time.perf_counter())
        if result.outcome == "invalidated" or coordinator.generation != result.generation:
            event.conversation_status = "invalidated"
            event.feedback_text = ""
            event.feedback_kind = ""
            event.outcome = "conversation_invalidated"
            return True
        if result.decision_kind == "no_request":
            event.feedback_text = ""
            event.feedback_kind = ""
            event.outcome = "conversation_no_request"
            return True
        event.outcome = "conversation_" + result.outcome
        if result.outcome == "clarification_needed" and result.text and mic is not None:
            pointer = self._conversation_reply_endpointer()
            if pointer is not None:
                self._continue_conversation_clarification(
                    event, command, mic, coordinator, result)
                return True
        if result.text:
            event.feedback_text = result.text
            event.feedback_kind = self._say_feedback(mic, result.text, event)
        recent = getattr(self, "_conversation_recent", None)
        if recent is not None and result.outcome != "truncated":
            # Prefer the interpreter's response to any old wake/status text;
            # append only after delivery has been attempted and generation
            # remains current.
            if coordinator.generation == result.generation:
                assistant_text = result.text or event.feedback_text
                recent.extend(({"role": "user", "text": command[:1000]},
                               {"role": "assistant", "text": assistant_text[:1000]}))
                del recent[:-3]
        return True

    def _load_vocabulary(self) -> dict:
        """Artist and playlist names, from Spotify or the last saved copy.

        Saved to disk so a Spotify API outage or rate limit does not strip the
        assistant of the names it needs to recognise.
        """
        import json
        path = ROOT / "assistant" / "logs" / ".vocabulary.json"
        def valid_copy(value):
            if not isinstance(value, dict):
                return None
            hints, artists = value.get("hints", []), value.get("artists", [])
            if (not isinstance(hints, list) or not isinstance(artists, list)
                    or any(not isinstance(x, str) for x in hints + artists)):
                return None
            cleaned = {"hints": hints, "artists": artists}
            return cleaned if hints or artists else None
        cached = None
        try:
            cached = valid_copy(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            pass
        if cached and (cached.get("hints") or cached.get("artists")):
            try:
                if time.time() - path.stat().st_mtime < 7 * 24 * 3600:
                    return cached
            except OSError:
                pass
        if cached and (getattr(self.spotify, "rate_limited", False)
                       or getattr(self.spotify, "web_rate_limited", False) is True):
            return cached
        try:
            fresh = {"hints": self.spotify.vocabulary(),
                     "artists": self.spotify.artist_names()}
            if fresh["hints"] or fresh["artists"]:
                path.write_text(json.dumps(fresh, ensure_ascii=False), encoding="utf-8")
                return fresh
        except Exception as exc:
            log.warning("Could not refresh Spotify names (%s); using the saved copy", exc)
        return cached or {"hints": [], "artists": []}

    # ---------- hotkeys ----------

    def _start_hotkeys(self, mic: Microphone) -> None:
        if not self.cfg.get("hotkey.enabled", True):
            return
        from pynput import keyboard

        def fire() -> None:
            self._trigger_source = "hotkey"
            self._trigger.set()

        def log_miss() -> None:
            """Dump a user-reported miss for review, not as a training label.

            You cannot log a detection that never fired, so this is the only way
            to collect negative-space data on wake words that were said and
            missed.
            """
            event = Event(trigger_source="miss", outcome="miss_reported")
            event.audio_path = self.telemetry.save_audio(event, mic.ring_contents())
            self.telemetry.write(event)
            self.feedback.ok()

        hotkeys = {
            self.cfg.get("hotkey.combo", "<ctrl>+<alt>+<space>"): fire,
            self.cfg.get("hotkey.miss_combo", "<ctrl>+<alt>+m"): log_miss,
        }
        try:
            listener = keyboard.GlobalHotKeys(hotkeys)
            listener.daemon = True
            listener.start()
        except ValueError as exc:
            # A malformed combo must not take the whole assistant down: the
            # wake word is the primary path and works without any hotkey.
            log.error(
                "Hotkeys disabled -- bad combo in config.toml (%s). "
                "pynput needs angle brackets for named keys, e.g. "
                "'<ctrl>+<alt>+<space>'.", exc
            )
            return
        log.info("Hotkeys active: %s", ", ".join(hotkeys))

    @contextmanager
    def _duck_music(self, event: Event, to_percent: int):
        """Temporarily lower local Windows output, never Spotify volume.

        A remote Connect speaker is outside the laptop master-volume path.
        """
        if not self._can_duck_local():
            event.duck_from = event.duck_to = None
            yield
            return
        with ducking.ducked(to_percent) as (was, now):
            event.duck_from, event.duck_to = was, now
            if now is not None:
                event.mark("duck_start")
            yield

    def _can_duck_local(self) -> bool:
        return (self.cfg.get("spotify.target", "laptop") != "active"
                or ducking.spotify_is_playing_local())

    def _watch_playback(self, stop: threading.Event) -> None:
        """Poll off the capture thread so Spotify latency cannot delay a wake."""
        while not stop.is_set():
            try:
                # Local only. Asking the Web API once a second is what got
                # this app rate-limited by Spotify for 16.5 hours.
                from . import local_media
                playing = local_media.is_playing()
                if playing is None:
                    playing = ducking.spotify_is_playing_local()
                self._music_playing = bool(playing)
            except Exception as exc:
                # An unknown state must never leave the Bluetooth headset mic
                # open over potentially active stereo playback.
                self._music_playing = True
                log.debug("Playback-state mic check failed: %s", exc)
            stop.wait(1.0)

    def _close_paused_mic(self) -> None:
        mic = getattr(self, "_paused_mic", None)
        self._paused_mic = None
        self._paused_listener = None
        self._paused_mic_last_frame_at = 0.0
        if mic is not None:
            try:
                mic.__exit__(None, None, None)
            except Exception as exc:
                log.warning("Could not close SoundDrum microphone cleanly: %s", exc)
            log.info("SoundDrum microphone closed; laptop microphone remains active")

    def _release_paused_mic_for_playback(self) -> None:
        """A Bluetooth headset input can force its output into hands-free
        mode. Release it just before stereo playback starts -- not earlier,
        so a failed lookup does not switch the audio path under its own
        spoken feedback."""
        self._music_playing = True
        self._paused_mic_retry_at = time.monotonic() + 2.0
        self._close_paused_mic()

    def _sync_paused_mic(self) -> None:
        name = self.cfg.get("audio.paused_input_device")
        if not name or self.mode != "stt":
            return
        if self._music_playing:
            self._close_paused_mic()
            return
        if self._paused_mic is not None or time.monotonic() < self._paused_mic_retry_at:
            return
        secondary = None
        try:
            secondary = Microphone(
                sample_rate=16000,
                frame_length=512,
                ring_seconds=self.cfg.get("audio.ring_buffer_seconds", 30),
                device=name,
                capture_rate=self.cfg.get("audio.paused_capture_rate", 16000),
                exclusive=False,
            )
            secondary.__enter__()
            secondary_listener = SttListener(
                self.cfg, SileroVAD(16000), self.listener_stt.transcribe,
                on_candidate=lambda info: self._record_wake_candidate(
                    dict(info, input_device="sounddrum")))
        except Exception as exc:
            if secondary is not None:
                secondary.__exit__(None, None, None)
            self._paused_mic_retry_at = time.monotonic() + 10.0
            now = time.monotonic()
            if now - getattr(self, "_paused_mic_last_warning", 0.0) >= 300.0:
                log.warning("SoundDrum microphone unavailable; using laptop only: %s", exc)
                self._paused_mic_last_warning = now
            else:
                log.debug("SoundDrum microphone still unavailable: %s", exc)
            return
        self._paused_mic = secondary
        self._paused_listener = secondary_listener
        self._paused_mic_last_frame_at = time.monotonic()
        self._paused_mic_last_warning = 0.0
        log.info("Listening on SoundDrum and laptop microphones while paused")

    def _read_paused_mic_frame(self) -> np.ndarray | None:
        """Poll without holding up the primary microphone's capture loop."""
        try:
            frame = self._paused_mic.read(timeout_s=0.0)
        except TimeoutError:
            if time.monotonic() - self._paused_mic_last_frame_at >= PAUSED_MIC_STALL_SECONDS:
                log.warning("SoundDrum microphone produced no audio for %.1f seconds",
                            PAUSED_MIC_STALL_SECONDS)
                self._close_paused_mic()
                self._paused_mic_retry_at = time.monotonic() + 10.0
            return None
        self._paused_mic_last_frame_at = time.monotonic()
        return frame

    # ---------- diagnostics ----------

    def _trace_wake(self) -> None:
        """Per audio frame: remember openWakeWord's score for this utterance."""
        trace = getattr(self, "_wake_trace", None)
        score = getattr(self.wake, "last_score", 0.0)
        if trace is not None and isinstance(score, float):
            trace.note(score, getattr(self.wake, "last_model", ""))
        personal_score = getattr(self.wake, "personal_score", None)
        personal_key = getattr(self.wake, "_personal_key", "")
        if trace is not None and isinstance(personal_score, float) and isinstance(personal_key, str) and personal_key:
            trace.note(personal_score, personal_key)

    def _note_early_duck(self) -> None:
        now = time.perf_counter()
        at = getattr(self, "_early_duck_at", 0.0)
        if not at or now - at > 15.0:
            self._early_duck_at = now

    def _take_early_duck(self) -> float:
        at = getattr(self, "_early_duck_at", 0.0)
        self._early_duck_at = 0.0
        return at if at and time.perf_counter() - at < 15.0 else 0.0

    def _open_timeline(self, event: Event, utt=None, handler_started: float = 0.0) -> None:
        """Anchor the command's timeline at the earliest wake evidence.

        Evidence is the first acoustic score, the listener's early peek (or its
        final transcript) and, for a hotkey, the key press. The origin is
        whichever came first of that and the user starting to speak, so every
        later stage is a non-negative offset.
        """
        trace = getattr(self, "_wake_trace", None)
        acoustic_at, scores = trace.take() if trace is not None else (0.0, {})
        event.wake_scores = scores
        duck_at = self._take_early_duck()
        speech_start = _perf_attr(utt, "speech_started_at")
        evidence = [t for t in (acoustic_at, _perf_attr(utt, "wake_evidence_at")) if t]
        first = min(evidence) if evidence else handler_started
        event.set_origin(first, speech_start, duck_at)
        event.mark("wake_evidence", first)
        for stage, moment in (("speech_start", speech_start), ("duck_start", duck_at),
                              ("speech_end", _perf_attr(utt, "completed_at")),
                              ("trigger_stt_done", _perf_attr(utt, "trigger_done_at"))):
            if moment:
                event.mark(stage, moment)

    @staticmethod
    def _close_timeline(event: Event) -> None:
        event.mark("handler_done")
        if event.feedback_kind:
            event.mark("feedback_done")

    def health_probes(self) -> dict:
        """Readings only the assistant can take, for the health heartbeat."""
        def ollama():
            if getattr(self, "brain", None) is None:
                return None
            from .supervise import _ollama_ready
            return _ollama_ready()

        def spotify_bans():
            guard = getattr(getattr(self, "spotify", None), "sp", None)
            status = getattr(guard, "ban_status", None)
            result = status() if callable(status) else None
            return result if isinstance(result, dict) else None

        def mic_overflows():
            mic = getattr(self, "_active_mic", None)
            return None if mic is None else int(getattr(mic, "queue_overflow_count", 0))

        def echo_alive():
            echo = getattr(self, "echo", None)
            return None if echo is None else bool(echo.reference.alive())

        def spotify_quota():
            query = getattr(getattr(self, "spotify", None), "quota_status", None)
            if not callable(query):
                return None
            status = query()
            return status if isinstance(status, dict) else None

        return {"ollama": ollama, "spotify_bans": spotify_bans,
                "ollama_model": lambda: health.ollama_model_status(self.brain.model) if getattr(self, "brain", None) else None,
                "mic_overflows": mic_overflows, "echo_alive": echo_alive,
                "capture_input_state": lambda: getattr(self, "_capture_input_state", "unknown"),
                "command_stt": lambda: self.stt.runtime_info(),
                "wake_stt": lambda: self.listener_stt.runtime_info(),
                "spotify_quota": spotify_quota}

    def start_health_monitor(self) -> "health.HealthMonitor":
        """Heartbeat + watchdog on a daemon thread; never touches the audio loop."""
        monitor = health.HealthMonitor(
            self.health_probes(),
            interval_s=float(self.cfg.get("telemetry.health_interval_s", health.HEARTBEAT_SECONDS)))
        monitor.start()
        return monitor

    # ---------- the one interesting path ----------

    def handle(self, mic: Microphone, source: str, tier: str = "") -> None:
        handler_started = time.perf_counter()
        event = Event(
            trigger_source=source,
            wake_tier=tier,
            wake_score=self.wake.last_score if source == "wake" else None,
            request_phase="trigger",
        )
        preroll = mic.preroll(self.cfg.get("capture.preroll_seconds", 1.5))

        duck_to = self.cfg.get("ducking.duck_to_percent", 20)
        pause_first = (
            tier == TIER_BOTH
            and self.cfg.get("ducking.pause_on_high_confidence", False)
        )

        token = bind_event(event)
        try:
            self._duck_on_wake()
            self._open_timeline(event, handler_started=handler_started)
            event.wake_path = "hotkey" if source == "hotkey" else "acoustic"
            controller = getattr(self, "global_assistant", None)
            if (source == "hotkey" and controller is not None and controller.session.active
                    and controller.session.phase == "review"
                    and self._handle_global("", mic, event)):
                return
            event.request_phase = "capture"
            if pause_first:
                self.spotify.pause()
                event.duck_to = 0
                event.feedback_kind = self._acknowledge(mic)
                captured = self.endpointer.record(
                    mic, after_prompt=self.feedback.speaks_on_wake)
                if captured["speech_started"]:
                    self.feedback.stopped()
            else:
                with self._duck_music(event, duck_to):
                    # Acknowledge INSIDE the duck, so the prompt is not fighting
                    # the music, and before recording, so we never transcribe
                    # our own voice.
                    event.feedback_kind = self._acknowledge(mic)
                    captured = self.endpointer.record(
                        mic, after_prompt=self.feedback.speaks_on_wake)
                    if captured["speech_started"]:
                        self.feedback.stopped()

            event.mark("speech_end", overwrite=True)
            self._capture_result(event, captured)
            event.audio_duration_ms = captured["duration_ms"]
            event.vad_speech_start_ms = captured["speech_start_ms"]
            event.vad_speech_end_ms = captured["speech_end_ms"]

            # Pre-roll is what makes the archive usable for wake-word training:
            # a clip starting at detection holds the command, not the trigger.
            full_audio = np.concatenate([preroll, captured["audio"]])
            event.audio_path = self.telemetry.save_audio(event, full_audio)

            if not captured["speech_started"]:
                # Tell the user something happened. A wake beep followed by
                # silence is indistinguishable from being ignored, and leaves
                # them repeating a command the assistant already gave up on.
                event.outcome = "false_accept"
                # Nothing was said after the wake: the music coming back up is the
                # answer. A spoken "Sorry." here was mostly heard after false wakes.
                mic.flush()
                return

            event.stt_model = self.stt.model_name
            event.command_stt_model = self.stt.model_name
            event.request_phase = "transcribe"
            # This decode may reveal a repeated wake. Keep the user's immediate
            # follow-up queued until that decision is known.
            event.transcript, event.stt_latency_ms = self.stt.transcribe(captured["audio"])
            event.transcript, captured, event.stt_latency_ms = self._past_repeated_wake(
                mic, event, event.transcript, captured, event.stt_latency_ms)
            event.mark("stt_done", overwrite=True)
            if not event.transcript:
                event.outcome = "false_accept"
                return

            if self._handle_conversation(event.transcript, mic, event, captured["audio"]):
                return
            if self._handle_global(event.transcript, mic, event):
                return
            intent = self._parse_intent(event.transcript, event)
            event.request_phase = "intent"
            event.intent_action = intent.action.value
            event.intent_query = intent.query
            event.intent_volume_after_play = intent.volume_after_play
            event.intent_library_only = intent.library_only
            event.intent_source = intent.source
            event.matched_rule = intent.matched_rule

            if intent.action is Action.UNKNOWN:
                self._retry_unknown(mic, event)
                return

            self._execute(intent, event, command_audio=captured["audio"], mic=mic)
            event.request_phase = "complete"

        except MusicConfirmationRequired as exc:
            self._handle_music_confirmation(mic, event, exc)
        except AmbiguousMatch as exc:
            self._handle_ambiguous_track(mic, event, exc)
        except SpotifyQuotaExceeded:
            event.outcome = "spotify_quota_exceeded"
            event.error = "SpotifyQuotaExceeded"
            event.feedback_text = SPOTIFY_QUOTA_MESSAGE
            event.feedback_kind = self._say_feedback(mic, event.feedback_text, event)
        except SpotifyRequestDeferred as exc:
            event.outcome = "request_deferred"
            event.error = str(exc)
            event.feedback_text = "Spotify requests are busy. Please try again shortly."
            event.feedback_kind = self._say_feedback(mic, event.feedback_text, event)
            log.info("Spotify request deferred locally: %s", exc)
        except SpotifyRateLimited as exc:
            event.outcome = "error"
            event.error = str(exc)
            event.feedback_text = RATE_LIMITED_MESSAGE
            event.feedback_kind = self._say_feedback(mic, event.feedback_text, event)
            log.error("%s", exc)
        except SpotifyAuthError as exc:
            # The one failure a beep genuinely cannot communicate.
            event.outcome = "error"
            event.error = str(exc)
            event.feedback_kind = self._say_feedback(mic, "Spotify needs you to sign in.", event)
            log.error("%s", exc)
        except Exception as exc:
            event.outcome = "error"
            event.error = "{}: {}".format(type(exc).__name__, exc)
            event.feedback_kind = self._say_feedback(mic, SORRY, event)
            log.exception("Command failed")
        finally:
            try:
                mic.flush()
            finally:
                try:
                    if self.listener is not None:
                        self.listener.flush()
                finally:
                    try:
                        lease = getattr(self, "_early_lease", None)
                        if lease is not None:
                            lease.release()
                    finally:
                        try:
                            event.token_age_days = self.spotify.token_age_days()
                            event.handler_latency_ms = int((time.perf_counter() - handler_started) * 1000)
                            self._close_timeline(event)
                            self.telemetry.write(event)
                        finally:
                            unbind_event(token)

    def _authorize_text_trigger(self, source: str, phrase: str) -> bool:
        """Require a matching recent acoustic hit for noncanonical text wakes."""
        evidence = getattr(self, "_last_strong_wake", None)
        if evidence is None:
            return False
        heard_at, score, model = evidence
        expected = phrase.replace(" ", "_")
        return (time.monotonic() - heard_at <= 2.0
                and score >= self.cfg.get("wake.threshold_high", 0.60)
                and model == expected)

    def _async_authorization(self) -> dict:
        """Freeze the legacy corroboration decision onto the current frame."""
        if self.listener is None:
            return {}
        return {"{}:{}".format(source, phrase): True
                for source in ("alias", "fuzzy", "mid")
                for phrase in set(self.listener.phrases)
                if self._authorize_text_trigger(source, phrase)}

    def _reset_async_capture_hints(self, mic) -> None:
        mic.flush()
        if self.wake is not None:
            self.wake.reset()
        self._last_strong_wake = None
        self._confirmed_acoustic_history = None
        cues = getattr(self, "_async_cued_segments", None)
        if cues is not None:
            cues.clear()
        if self.listener is not None:
            self.listener.flush()

    def _run_async_owned_callback(self, mic, callback, *, reason: str) -> bool:
        """Run a legacy mic/STT owner only after the lane acknowledges quiescence."""
        runtime = getattr(self, "async_capture", None)
        if runtime is not None:
            runtime.counters["handoff_attempts"] += 1
        if runtime is None or not runtime.suspend():
            log.warning("Skipped %s: asynchronous recognizer handoff was not acknowledged",
                        reason)
            self._reset_async_capture_hints(mic)
            self._record_wake_rejection("async_handler_busy", mic=mic)
            return False
        try:
            callback()
            return True
        finally:
            self._reset_async_capture_hints(mic)
            if not runtime.resume_if_quiescent():
                log.error("Asynchronous capture remains fenced after %s", reason)

    def _resume_async_after_fence(self, mic) -> bool:
        runtime = getattr(self, "async_capture", None)
        if runtime is None or runtime.accepting:
            return True
        if not runtime.resume_if_quiescent():
            return False
        self._reset_async_capture_hints(mic)
        return False

    def _dispatch_async_events(self, mic, events, *, last_trigger, cooldown):
        runtime = self.async_capture
        utterances = []
        for event in events:
            if event.kind == "wake":
                if event.segment_id not in self._async_cued_segments:
                    self._duck_on_wake()
                    self._async_cued_segments.add(event.segment_id)
            elif event.kind == "diagnostic":
                runtime.queue_diagnostic(event)
                if event.diagnostic_snapshot is None:
                    self._record_wake_rejection(event.reason or "async_diagnostic", mic=mic)
            elif event.kind == "rejected":
                self._async_cued_segments.discard(event.segment_id)
                runtime.queue_diagnostic(event)
                self._record_wake_rejection(event.reason or "async_rejected", mic=mic)
            elif event.kind == "utterance" and event.utterance is not None:
                utterances.append(event)
        if not utterances:
            return last_trigger
        if time.monotonic() - last_trigger < cooldown:
            for capture in utterances:
                self._async_cued_segments.discard(capture.segment_id)
                self._record_async_capture_discard(capture, "async_cooldown")
            return last_trigger
        capture = utterances[0]
        utt = capture.utterance
        segment_id = capture.segment_id
        wake_cued = segment_id in self._async_cued_segments
        self._async_cued_segments.discard(segment_id)
        for discarded in utterances[1:]:
            self._async_cued_segments.discard(discarded.segment_id)
            self._record_async_capture_discard(discarded, "async_batch_limit")
        try:
            handled = self._run_async_owned_callback(
                mic, lambda: self.handle_utterance(
                    mic, utt, wake_cued=wake_cued), reason="utterance handler")
        except Exception:
            self._record_async_capture_discard(capture, "async_handler_raised")
            raise
        if handled:
            return time.monotonic()
        self._record_async_capture_discard(capture, "async_handler_busy")
        return last_trigger

    def _record_async_capture_discard(self, capture, reason):
        """Record an explicitly discarded completed capture without its audio."""
        runtime = self.async_capture
        runtime.counters["completed_captures_discarded"] += 1
        telemetry = getattr(self, "telemetry", None)
        if telemetry is None:
            log.warning("Discarded async capture %s: %s", capture.capture_id or
                        capture.segment_id, reason)
            return
        event = Event(
            trigger_source="async_listener", input_device="laptop",
            outcome="capture_discarded", request_phase="trigger",
            rejection_reason=reason, rejection_count=1,
            segment_id=capture.segment_id,
            capture_id=capture.capture_id or capture.segment_id,
            diagnostic_stage="dispatch", diagnostic_only=True,
            review_state="unknown")
        try:
            telemetry.write(event)
        except Exception as exc:
            log.warning("Could not record discarded async capture %s (%s)",
                        event.capture_id, type(exc).__name__)

    def _try_async_global_notice(self, mic, *, last_trigger, cooldown):
        runtime = self.async_capture
        controller = getattr(self, "global_assistant", None)
        notices = getattr(controller, "_notices", None)
        if (runtime is None or not runtime.accepting or controller is None
                or not notices or time.monotonic() - last_trigger < cooldown):
            return
        probability = runtime.last_vad_probability
        active = (runtime.frontend._active is not None
                  or runtime.frontend._boundary is not None)
        busy = getattr(self, "_busy", None)
        lease = getattr(self, "_early_lease", None)
        trigger = getattr(self, "_trigger", None)
        recent_wake = getattr(self, "_last_strong_wake", None)
        unsafe = (active or not isinstance(probability, (int, float))
                  or probability >= self.cfg.get("capture.vad_threshold", 0.35)
                  or controller.session.active
                  or (busy is not None and busy.locked())
                  or (trigger is not None and trigger.is_set())
                  or (lease is not None and getattr(lease, "_owned", False))
                  or getattr(self, "_paused_mic", None) is not None
                  or (recent_wake is not None
                      and time.monotonic() - recent_wake[0] < 2.0))
        if unsafe:
            self._global_quiet_since = None
            return
        now = time.monotonic()
        quiet_since = getattr(self, "_global_quiet_since", None)
        if quiet_since is None:
            self._global_quiet_since = now
            return
        if now - quiet_since < 1.0:
            return
        self._run_async_owned_callback(
            mic, lambda: self._speak_global_notice_if_quiet(
                mic, async_suspended=True), reason="global notice")
        self._global_quiet_since = time.monotonic()

    def _run_async_frame(self, mic, frame, *, last_trigger, last_confirmation,
                         last_early_log, cooldown):
        if not self._resume_async_after_fence(mic):
            self.async_capture.counters["fenced_frames"] += 1
            return last_trigger, last_confirmation, last_early_log

        captured_at = time.monotonic()
        self._check_echo_reference()
        cleaned_frame = mic.vad_frame(frame)
        proposal = None
        personal = False
        tier = ""
        if self.wake is not None:
            tier = self.wake.process(frame)
            self._trace_wake()
            personal = getattr(self.wake, "personal_hit", False) is True
            if tier == TIER_BOTH:
                self._last_strong_wake = (
                    time.monotonic(), self.wake.last_score,
                    getattr(self.wake, "last_model", ""))
            if personal:
                self._last_strong_wake = (
                    time.monotonic(), self.wake.personal_score, "alexa")
            if tier == TIER_BOTH and self._early_lease is not None:
                now = time.monotonic()
                if now - last_early_log >= 2.0:
                    log.info("Early wake hint awaiting confirmation: model=%s score=%s",
                             getattr(self.wake, "last_model", ""),
                             getattr(self.wake, "last_score", ""))
                    last_early_log = now

            frontend = self.async_capture.frontend
            segment = frontend._active
            if frontend._boundary is not None:
                segment = frontend._boundary.segment
            already_woke = bool(segment is not None and segment.wake_emitted)
            confirm_due = (time.monotonic() - last_confirmation >= 1.0
                           and not already_woke)
            stock_qualified = (tier == TIER_BOTH
                               and self.wake.last_score >= self.cfg.get(
                                   "wake.acoustic_activation_threshold", 0.75))
            if ((stock_qualified or personal)
                    and time.monotonic() - last_trigger >= cooldown
                    and confirm_due):
                last_confirmation = time.monotonic()
                proposal = AcousticProposal(
                    getattr(self.wake, "last_model", "") or
                    ("alexa" if personal else "unknown"),
                    float(getattr(self.wake, "last_score", 0.0) or 0.0),
                    personal,
                    float(getattr(self.wake, "personal_score", 0.0) or 0.0))
            if self._early_lease is not None and self._early_lease.expired():
                self._early_lease.release()

        try:
            self.async_capture.feed(
                frame, cleaned_frame if cleaned_frame is not frame else None,
                captured_at=captured_at, authorization=self._async_authorization(),
                acoustic_proposal=proposal)
        except BufferError:
            log.warning("Async capture queue is full; fencing the discontinuous audio")
            self._recover_capture_queue_gap(mic)
            return last_trigger, last_confirmation, last_early_log
        self._record_capture_diagnostics(
            mic, frame, cleaned_frame, self.async_capture.last_vad_probability)
        events = self.async_capture.poll_events()
        last_trigger = self._dispatch_async_events(
            mic, events, last_trigger=last_trigger, cooldown=cooldown)
        self._try_async_global_notice(mic, last_trigger=last_trigger,
                                      cooldown=cooldown)
        return last_trigger, last_confirmation, last_early_log

    def _recover_capture_queue_gap(self, mic) -> None:
        """Break all recognizer state after callback PCM continuity is lost."""
        mic.flush()
        runtime = getattr(self, "async_capture", None)
        if runtime is not None:
            runtime.flush("capture_queue_gap")
        elif self.mode == "stt" and self.listener is not None:
            self.listener.flush()
        if self.wake is not None:
            self.wake.reset()
        self._last_strong_wake = None
        self._confirmed_acoustic_history = None
        cues = getattr(self, "_async_cued_segments", None)
        if cues is not None:
            cues.clear()
        lease = getattr(self, "_early_lease", None)
        if lease is not None:
            lease.release()
        self._record_wake_rejection("capture_queue_gap", mic=mic)

    def _feed_listener_guarded(self, mic, frame, cleaned_frame):
        """Reject a completed utterance if its synchronous feed crossed a drop."""
        before = getattr(mic, "queue_overflow_count", None)
        utterance = self.listener.feed(frame, cleaned_frame)
        after = getattr(mic, "queue_overflow_count", None)
        valid = lambda value: isinstance(value, int) and not isinstance(value, bool)
        if valid(before) and valid(after) and after > before:
            self._recover_capture_queue_gap(mic)
            return None, True, after
        return utterance, False, after

    @staticmethod
    def _match_verified_trigger(transcript: str, phrase: str, mid: bool) -> tuple:
        """Pure exact-command matching shared by legacy and async verification."""
        words = re.sub(r"[^\w\s]", " ", transcript.lower()).split()
        marker = phrase.split()
        starts = (range(len(words) - len(marker) + 1) if mid
                  else ((0, 1) if words[:1] == ["hey"] else (0,)))
        for start in starts:
            if words[start:start + len(marker)] != marker:
                continue
            command = " ".join(words[start + len(marker):]).strip()
            # A name mentioned inside conversation is not a standalone wake.
            # Acoustic shortcut verification handles bare prefix wakes on its
            # short window; a mid-utterance match needs an actual command.
            embedded = start > 0 and not (start == 1 and words[:1] == ["hey"])
            if embedded and (not command or parse(command).action is Action.UNKNOWN):
                continue
            return phrase, command, bool(mid and embedded)
        return "", "", False

    def _verify_text_trigger(self, audio: np.ndarray, phrase: str,
                             mid: bool) -> tuple:
        """Use the command recognizer once on a final ambiguous utterance."""
        started = time.perf_counter()
        try:
            transcript, latency = self.stt.transcribe(audio)
        except Exception as exc:
            log.warning("Could not verify ambiguous wake: %s", exc)
            latency = int((time.perf_counter() - started) * 1000)
            self._last_verify_latency_ms = latency
            return "", "", False, latency, ""
        self._last_verify_latency_ms = latency
        phrase, command, verified_mid = self._match_verified_trigger(
            transcript, phrase, mid)
        return phrase, command, verified_mid, latency, transcript

    def _record_text_rejection(self, reason: str, duration_ms: int) -> None:
        self._record_wake_rejection(reason, duration_ms=duration_ms,
                                    mic=getattr(self, "_active_mic", None))

    def _record_capture_diagnostics(self, mic, raw, cleaned, probability) -> None:
        """Summarize capture health every ten seconds without another VAD pass."""
        now = time.monotonic()
        stats = getattr(self, "_capture_diagnostics", None)
        if stats is None:
            stats = self._capture_diagnostics = {
                "started": now, "frames": 0, "raw_rms": 0.0, "raw_peak": 0.0,
                "raw_valid_frames": 0, "clean_rms": 0.0, "vad_peak": 0.0, "voiced": 0,
                "wake_peak": 0.0, "wake_model": ""}
        stats["frames"] += 1
        for name, audio in (("raw", raw), ("clean", cleaned)):
            if (not isinstance(audio, np.ndarray) or not audio.size
                    or audio.dtype.kind not in "biuf"):
                continue
            values = audio.astype(np.float64)
            if not np.isfinite(values).all():
                continue
            if name == "raw":
                stats["raw_valid_frames"] += 1
            stats[name + "_rms"] = max(stats[name + "_rms"], float(np.sqrt(np.mean(values ** 2))))
            if name == "raw":
                stats["raw_peak"] = max(stats["raw_peak"], float(np.max(np.abs(values))))
        if isinstance(probability, (float, int)):
            stats["vad_peak"] = max(stats["vad_peak"], probability)
            stats["voiced"] += int(probability >= self.cfg.get("capture.vad_threshold", 0.35))
        score = getattr(self.wake, "last_score", 0.0) if self.wake is not None else 0.0
        if isinstance(score, (float, int)) and score >= stats["wake_peak"]:
            stats["wake_peak"] = score
            stats["wake_model"] = getattr(self.wake, "last_model", "")
        if now - stats["started"] < 10.0:
            return
        zero_windows = getattr(self, "_capture_zero_windows", 0)
        warning_emitted = getattr(self, "_capture_silence_warning_emitted", False)
        valid_raw_frames = stats["raw_valid_frames"]
        raw_peak = stats["raw_peak"]
        if valid_raw_frames and raw_peak > 0:
            self._capture_input_state = "signal_present"
            self._capture_zero_windows = 0
            if warning_emitted:
                log.info("Capture input signal returned; raw microphone PCM is nonzero again")
                self._capture_silence_warning_emitted = False
        elif valid_raw_frames >= 128 and raw_peak == 0:
            self._capture_input_state = "digital_silence"
            self._capture_zero_windows = zero_windows + 1
            if self._capture_zero_windows >= 2 and not warning_emitted:
                log.warning(
                    "Raw microphone input is zero-valued across consecutive windows, "
                    "preventing wake recognition; check microphone mute, permission, "
                    "or the input processing chain")
                self._capture_silence_warning_emitted = True
        else:
            self._capture_input_state = "unknown"
            self._capture_zero_windows = 0
            self._capture_silence_warning_emitted = warning_emitted
        echo = getattr(mic, "echo", None)
        reference_alive = None
        if echo is not None:
            try:
                reference_alive = bool(echo.reference.alive())
            except Exception:
                reference_alive = False
        log.info(
            "Capture health: frames=%d raw_rms_max=%.1f raw_peak=%.0f clean_rms_max=%.1f "
            "clean_vad_peak=%.3f voiced_frames=%d wake_peak=%.3f wake_model=%s "
            "queue_overflows_total=%s prompt_discarded_total=%s echo_attached=%s echo_active=%s reference_alive=%s "
            "segments_total=%s short_discarded_total=%s",
            stats["frames"], stats["raw_rms"], stats["raw_peak"], stats["clean_rms"],
            stats["vad_peak"], stats["voiced"], stats["wake_peak"], stats["wake_model"],
            getattr(mic, "queue_overflow_count", 0), getattr(mic, "intentional_discard_count", 0), echo is not None,
            bool(getattr(echo, "active", False)), reference_alive,
            getattr(self.listener, "segments_completed", 0),
            getattr(self.listener, "discarded_short_segments", 0))
        self._capture_diagnostics = None

    def _record_wake_rejection(self, reason: str, duration_ms: int = 0,
                               mic: Microphone | None = None) -> None:
        """Coalesce counts only; exact rejected PCM comes from the listener.

        A later microphone snapshot cannot represent all earlier rejections.
        """
        # This acoustic burst is spent; keep its scores for the record we may
        # write, and do not let them leak into the next command.
        trace = getattr(self, "_wake_trace", None)
        if trace is not None:
            kept = getattr(self, "_reject_scores", None)
            if kept is None:
                kept = self._reject_scores = {}
            for model, score in trace.take()[1].items():
                kept[model] = max(kept.get(model, 0.0), score)
        now = time.monotonic()
        counts = getattr(self, "_rejection_counts", None)
        if counts is None:
            counts = self._rejection_counts = {}
        counts[reason] = counts.get(reason, 0) + 1
        log_due = now - getattr(self, "_last_reject_log_at", float("-inf")) >= 30.0
        if not log_due:
            return
        event = Event(trigger_source="wake_rejected", outcome="rejected",
                      request_phase="trigger", input_device="laptop",
                      wake_score=getattr(self.wake, "last_score", None) if self.wake else None,
                      wake_model=getattr(self.wake, "last_model", "") if self.wake else "",
                      rejection_reason=",".join(f"{key}:{value}" for key, value in sorted(counts.items())),
                      rejection_count=sum(counts.values()), audio_duration_ms=duration_ms,
                      wake_scores=dict(getattr(self, "_reject_scores", None) or {}),
                      trigger_confirmation_mode="second_stt" if "second_stt_unconfirmed" in counts else "",
                      trigger_confirmation_latency_ms=(getattr(self, "_last_verify_latency_ms", None)
                                                       if "second_stt_unconfirmed" in counts else None))
        self.telemetry.write(event)
        self._rejection_counts = {}
        self._reject_scores = {}
        self._last_reject_log_at = now

    def _record_wake_candidate(self, info: dict) -> None:
        """Record an unknown candidate without consuming live wake evidence."""
        telemetry = getattr(self, "telemetry", None)
        if telemetry is None:
            return
        trace = getattr(self, "_wake_trace", None)
        scores = trace.snapshot() if trace is not None else {}
        names = ("processed_audio", "segment_id", "overlap_of", "sample_rate",
                 "relative_start_sample", "relative_end_sample", "processed_alignment",
                 "vad_peak", "vad_threshold", "voiced_frames", "decision_source",
                 "end_reason", "truncated", "diagnostic_only")
        details = {name: info[name] for name in names if name in info}
        telemetry.record_near_miss(
            info["audio"], info.get("transcript", ""), info["reason"],
            input_device=info.get("input_device", "laptop"), scores=scores,
            woke_early=info.get("woke_early", False),
            kind=info.get("kind", "wake_candidate"),
            second_stt_transcript=info.get("second_stt_transcript", ""),
            trigger_stt_latency_ms=info.get("stt_latency_ms"),
            second_stt_latency_ms=info.get("second_stt_latency_ms"), **details)

    def _record_near_miss(self, info: dict) -> None:
        """Archive a wake-like utterance the listener refused (retraining data).

        Rate limiting and the size cap live in Telemetry.record_near_miss."""
        trace = getattr(self, "_wake_trace", None)
        scores = trace.take()[1] if trace is not None else {}
        telemetry = getattr(self, "telemetry", None)
        if telemetry is None:
            return
        recognizer = getattr(self, "listener_stt", None)
        model_name = getattr(recognizer, "model_name", "")
        runtime_info = getattr(recognizer, "runtime_info", None)
        device = runtime_info() if callable(runtime_info) else ""
        details = {name: info[name] for name in (
            "processed_audio", "processed_alignment", "decision_source",
            "relative_start_sample", "relative_end_sample", "sample_rate") if name in info}
        telemetry.record_near_miss(
            info["audio"], info["transcript"], info["reason"],
            woke_early=info.get("woke_early", False), scores=scores,
            second_stt_transcript=info.get("second_stt_transcript", ""),
            kind=info.get("kind", "near_miss"),
            trigger_stt_model=model_name if isinstance(model_name, str) else "",
            trigger_stt_device=device if isinstance(device, str) else "",
            trigger_stt_latency_ms=info.get("stt_latency_ms"),
            second_stt_latency_ms=info.get("second_stt_latency_ms"), **details)

    def _confirm_acoustic_wake(self, mic: Microphone) -> bool:
        """An acoustic hit acts only if the words just heard contain the wake
        phrase. Scores cannot separate them: room conversation scored 0.94
        and recorded six seconds of other people talking."""
        self._confirmed_acoustic_history = None
        if self.listener is None:
            return True
        seconds = self.cfg.get("wake.confirm_seconds", 1.6)
        processed_audio = None
        if callable(getattr(type(mic), "preroll_pair", None)):
            audio, processed_audio = mic.preroll_pair(seconds)
        else:
            audio = mic.preroll(seconds)
        try:
            text, latency_ms = self.listener_stt.transcribe(audio)
        except Exception as exc:
            log.warning("Could not confirm acoustic wake: %s", exc)
            return False
        # The window can open on the tail of an earlier word ("...to. Alexa"),
        # so the wake may sit mid-window; the acoustic hit authorizes that.
        phrase, command, _ = self.listener.locate_trigger(text)
        wake = getattr(self, "wake", None)
        personal = getattr(wake, "personal_hit", False) is True
        personal_score = getattr(wake, "personal_score", 0.0)
        stock_score = getattr(wake, "last_score", 0.0)
        stock_model = getattr(wake, "last_model", "")
        qualified = (personal and isinstance(personal_score, (float, int))
                     and personal_score >= self.cfg.get("wake.personal_threshold", 0.5)) or (
                         isinstance(stock_score, (float, int))
                         and stock_score >= self.cfg.get("wake.acoustic_activation_threshold", 0.75))
        second_text = ""
        second_latency = None
        # small.en must not be the sole veto after an acoustic proposal. Turbo
        # already resides on CUDA: reuse this exact window, without hotwords or
        # arbitrary-position matching. Never infer the word from a score alone.
        if not phrase and qualified and getattr(self, "stt", None) is not None and self.stt is not self.listener_stt:
            target = "alexa" if personal else stock_model.replace("_", " ") if isinstance(stock_model, str) else ""
            if target in self.listener.phrases:
                verified = self._verify_text_trigger(audio, target, False)
                phrase, command, _, second_latency, second_text = verified
                if phrase:
                    log.info("Acoustic wake confirmed by command model: %r (small heard %r, %d ms)",
                             second_text, text, second_latency)
        if phrase:
            # A short confirmer can hear just the wake even when the same PCM
            # already contains the first command syllables. Preserve that PCM
            # for the listener; the bool alone must not discard it as a bare wake.
            self._confirmed_acoustic_history = (phrase, audio, processed_audio)
            self._duck_on_wake()
            if command:
                # A one-breath request already contains command audio. Let the
                # utterance path finish it (acknowledging here would flush it),
                # but make sure that utterance counts as woken and the music
                # drops now -- a soft "Alexa" may be missing from its text.
                note = getattr(self.listener, "note_wake", None)
                if note is not None:
                    note(phrase)
            return not command
        log.info("Acoustic wake not confirmed by speech (%s, score %.2f): %r",
                 "personal" if personal else getattr(wake, "last_model", ""),
                 float(getattr(wake, "personal_score" if personal else "last_score", 0.0) or 0.0),
                 text)
        # These are unlabelled proposals, not proven false accepts: a real
        # Alexa misheard by the confirmer must not become a training negative.
        self._record_near_miss({"audio": audio, "transcript": text,
                                "reason": "acoustic_unconfirmed",
                                "kind": "acoustic_unconfirmed", "stt_latency_ms": latency_ms,
                                "second_stt_transcript": second_text,
                                "second_stt_latency_ms": second_latency,
                                "processed_audio": processed_audio,
                                "processed_alignment": ("frame_paired"
                                    if processed_audio is not None else "unavailable"),
                                "decision_source": "acoustic_confirmation"})
        return False

    def _duck_on_wake(self) -> None:
        """Lower local music for an accepted command exchange."""
        if getattr(self, "_early_lease", None) is None or not self._can_duck_local():
            return
        try:
            if self._early_lease.acquire():
                self._note_early_duck()
                log.info("Command listening; music ducked")
        except Exception as exc:
            log.debug("Early duck failed: %s", exc)

    def _check_echo_reference(self) -> None:
        """Follow playback route changes and recover a silent loopback."""
        echo = getattr(self, "echo", None)
        now = time.monotonic()
        if echo is None or now - getattr(self, "_echo_checked_at", 0.0) < 3.0:
            return
        self._echo_checked_at = now
        route_changed = getattr(echo.reference, "route_changed", None)
        if (route_changed is not None
                and now - getattr(self, "_echo_route_checked_at", 0.0) >= 10.0):
            self._echo_route_checked_at = now
            try:
                if route_changed():
                    log.info("Speaker playback route changed; reopening reference")
                    echo.reference.stop()
                    echo.reference.start()
                    self._echo_reopened_at = now
                    return
            except Exception as exc:
                log.warning("Could not switch the speaker reference: %s", exc)
        if echo.reference.alive() or not ducking.spotify_is_playing_local():
            return
        if now - getattr(self, "_echo_reopened_at", 0.0) < 10.0:
            return
        self._echo_reopened_at = now
        log.info("Speaker reference silent during playback; reopening it")
        try:
            echo.reference.stop()
            echo.reference.start()
        except Exception as exc:
            log.warning("Could not reopen the speaker reference: %s", exc)

    def _duck_for_speech(self) -> None:
        """Duck the moment ANY speech starts, before we know it is a command.

        With the STT approach the first utterance is captured at full music
        volume -- which is why "pause" came back as "buzz". We cannot wait for
        a wake word to tell us to duck, so we duck on speech and restore if it
        turns out not to have been for us. The music dips for a second when you
        talk; that is the price of catching the command at all.
        """
        if self.duck_trigger == "none" or not self._can_duck_local():
            return
        with self._duck_lock:
            if self._speech_lease is not None:
                return
            lease = ducking.DuckLease(
                self.cfg.get("ducking.duck_to_percent", 20),
                self.cfg.get("capture.max_seconds", 6.0) + 2.0,
            )
            try:
                lease.acquire()
                self._speech_lease = lease
            except Exception as exc:
                log.debug("Speech duck failed: %s", exc)

    def _undock_after_speech(self) -> None:
        """Restore the level ducked by _duck_for_speech, unless a command is
        running (which holds its own duck)."""
        with self._duck_lock:
            lease = self._speech_lease
            self._speech_lease = None
        if lease is not None:
            lease.release()

    def _ask_and_listen(self, mic: Microphone, prompt: str) -> tuple:
        """Say something, then listen for one answer. Return text and capture.

        The caller is expected to already hold the duck, so the music stays
        lowered for the whole exchange instead of jumping back up between the
        question and the reply.
        """
        # Keep the tail neutral, then retain even a very short immediate reply.
        kind = self._say_feedback(mic, prompt.rstrip() + " Ready.")
        if kind != KIND_SPEECH:
            return "", {"audio": np.array([], dtype=np.int16),
                        "speech_started": False, "duration_ms": 0}, 0
        captured = self.endpointer.record(mic, after_prompt=False)
        if not captured["speech_started"]:
            return "", captured, 0
        self.feedback.stopped()
        text, latency = self._transcribe_captured(mic, captured["audio"])
        return text, captured, latency

    def _handle_ambiguous_track(self, mic: Microphone, event: Event,
                                exc: AmbiguousMatch) -> None:
        """One grounded clarification; a reply never selects outside the supplied choices."""
        event.outcome = "ambiguous"
        event.error = str(exc)
        title = event.intent_query.strip()
        original = getattr(exc, "intent", None)
        if not isinstance(original, Intent):
            original = Intent(Action(event.intent_action) if event.intent_action else Action.UNKNOWN, query=title,
                              volume_after_play=event.intent_volume_after_play)
        album_choices = bool(exc.candidates) and all(
            isinstance(candidate, Candidate) and candidate.uri.startswith("spotify:album:")
            for candidate in exc.candidates[:3])
        if (original.action in (Action.PLAY_TRACK, Action.QUEUE) and " by " in title.lower()
                or original.action is Action.PLAY_ALBUM
                or original.action is Action.PLAY_TRACK and album_choices):
            kind = "spotify:album:" if album_choices else "spotify:track:"
            choices = tuple(candidate for candidate in exc.candidates[:3]
                            if isinstance(candidate, Candidate)
                            and candidate.uri.startswith(kind)
                            and not (original.action is Action.PLAY_ALBUM and kind != "spotify:album:")
                            and not (original.explicit_track and kind == "spotify:album:")
                            and not (original.action is Action.QUEUE and kind == "spotify:album:"))
            if choices:
                self._confirm_music_candidate(mic, event, original, choices)
                return
        if event.intent_action == Action.PLAY_TRACK.value and title and " by " not in title.lower():
            deadline = time.monotonic() + MUSIC_CLARIFICATION_SECONDS
            try:
                with self._duck_music(event, self.cfg.get("ducking.duck_to_percent", 20)):
                    answer, captured, latency = self._ask_and_listen(
                        mic, "Which artist?")
            except Exception as listen_exc:
                event.outcome = "error"
                event.error = "{}: {}".format(type(listen_exc).__name__, listen_exc)
                event.feedback_text = SORRY
                event.feedback_kind = self._say_feedback(mic, event.feedback_text, event)
                log.exception("Artist clarification failed")
                return
            event.clarification_transcript = answer
            event.clarification_stt_latency_ms = latency
            extra = Event(ts=event.ts, event_id=event.event_id + "-artist")
            event.clarification_audio_path = self.telemetry.save_audio(extra, captured["audio"])
            artist = re.sub(r"^(?:by|the artist is|it's)\s+", "", answer.strip(), flags=re.I)
            full = parse(artist)
            if full.action is Action.PLAY_TRACK and " by " in full.query:
                repeated_title, _, repeated_artist = full.query.rpartition(" by ")
                if _norm(repeated_title) == _norm(title):
                    artist = repeated_artist.strip()
            if artist and parse(artist).action is Action.UNKNOWN:
                try:
                    _check_music_deadline(deadline)
                    from dataclasses import replace
                    clarified = replace(original, query=f"{title} by {artist}")
                    event.intent_query = clarified.query
                    self._execute(clarified, event, command_audio=captured["audio"], mic=mic,
                                  clarification_deadline=deadline)
                    if event.outcome == "ok":
                        event.error = ""
                        event.request_phase = "complete"
                        return
                    if event.outcome == "not_found":
                        event.error = ""
                        return
                except ClarificationExpired:
                    event.outcome = "ambiguous"
                    event.feedback_text = "That choice expired."
                    event.feedback_kind = self._say_feedback(mic, event.feedback_text, event)
                    return
                except (AmbiguousMatch, LookupError):
                    pass
                except SpotifyAuthError as auth_exc:
                    event.outcome = "error"
                    event.error = str(auth_exc)
                    event.feedback_text = "Spotify needs you to sign in."
                    event.feedback_kind = self._say_feedback(mic, event.feedback_text, event)
                    return
                except SpotifyQuotaExceeded:
                    event.outcome = "spotify_quota_exceeded"
                    event.error = "SpotifyQuotaExceeded"
                    event.feedback_text = SPOTIFY_QUOTA_MESSAGE
                    event.feedback_kind = self._say_feedback(mic, event.feedback_text, event)
                    return
                except SpotifyRequestDeferred as execute_exc:
                    event.outcome = "request_deferred"
                    event.error = str(execute_exc)
                    event.feedback_text = "Spotify requests are busy. Please try again shortly."
                    event.feedback_kind = self._say_feedback(mic, event.feedback_text, event)
                    return
                except Exception as execute_exc:
                    event.outcome = "error"
                    event.error = "{}: {}".format(type(execute_exc).__name__, execute_exc)
                    event.feedback_text = SORRY
                    event.feedback_kind = self._say_feedback(mic, event.feedback_text, event)
                    log.exception("Clarified command failed")
                    return
        event.outcome = "ambiguous"
        event.feedback_text = AMBIGUOUS_MESSAGE
        event.feedback_kind = self._say_feedback(mic, event.feedback_text, event)

    def _handle_music_confirmation(self, mic, event, exc):
        original = getattr(exc, "intent", None)
        if not isinstance(original, Intent):
            original = Intent(Action(event.intent_action), query=event.intent_query,
                              volume_after_play=event.intent_volume_after_play,
                              library_only=event.intent_library_only)
        self._note_resolution(event)
        self._confirm_music_candidate(mic, event, original, tuple(exc.candidates))

    def _confirm_music_candidate(self, mic: Microphone, event: Event, original: Intent,
                                 choices: tuple[Candidate, ...]) -> None:
        """Confirm immutable Spotify choices once, keeping the original play/queue options."""
        deadline = time.monotonic() + MUSIC_CLARIFICATION_SECONDS
        def label(candidate):
            credits = [row.get("name", "") for row in (candidate.metadata or {}).get("artists") or []
                       if isinstance(row, dict) and row.get("name")]
            return "{} by {}".format(candidate.name, ", ".join(credits)) if credits else candidate.label

        labels = tuple(label(candidate) for candidate in choices)
        album_kind = choices[0].uri.startswith("spotify:album:")
        described = tuple("album " + label if album_kind else label for label in labels)
        prompt = ("Did you mean {}?".format(described[0]) if len(choices) == 1 else
                  "Which one? " + ". ".join("{}: {}".format(i, label)
                                            for i, label in enumerate(described, 1)))
        try:
            with self._duck_music(event, self.cfg.get("ducking.duck_to_percent", 20)):
                answer, captured, latency = self._ask_and_listen(mic, prompt)
            event.clarification_transcript = answer
            event.clarification_stt_latency_ms = latency
            extra = Event(ts=event.ts, event_id=event.event_id + "-candidate")
            event.clarification_audio_path = self.telemetry.save_audio(extra, captured["audio"])
            key = " ".join(_norm(answer).split())
            chosen = None
            if key and not re.match(r"^(?:no|nope|nah|cancel|stop|never|don t|dont)\b", key):
                if key in {"yes", "yeah", "yep", "correct", "that s right"}:
                    chosen = choices[0] if len(choices) == 1 else None
                else:
                    ordinals = {"1": 0, "one": 0, "first": 0, "the first one": 0,
                                "2": 1, "two": 1, "second": 1, "the second one": 1,
                                "3": 2, "three": 2, "third": 2, "the third one": 2}
                    index = ordinals.get(key)
                    if index is not None and index < len(choices):
                        chosen = choices[index]
                    else:
                        repeated = parse(answer)
                        generic_reply = re.sub(r"^(?:play|start|put on|throw on)\s+", "", answer.strip(), flags=re.I)
                        generic_key = " ".join(_norm(generic_reply).split())
                        exact_entity = any(generic_key in {" ".join(_norm(candidate.name).split()),
                                                          " ".join(_norm(full_label).split())}
                                           for candidate, full_label in zip(choices, labels))
                        disagrees = not exact_entity and (
                            repeated.action is Action.PLAY_ALBUM and not album_kind
                            or repeated.explicit_track and album_kind)
                        requested = generic_reply if exact_entity else (
                            repeated.query if repeated.action in (Action.PLAY_TRACK, Action.PLAY_ALBUM) else answer)
                        requested_key = " ".join(_norm(requested).split())
                        matches = [candidate for candidate, full_label in zip(choices, labels) if requested_key in {
                            " ".join(_norm(candidate.name).split()),
                            " ".join(_norm(full_label).split())}]
                        chosen = matches[0] if len(matches) == 1 and not disagrees else None
            if chosen is not None:
                _check_music_deadline(deadline)
                event.matched_rule += "+confirmed_candidate"
                self._execute(original, event, mic=mic, confirmed_candidate=chosen,
                              clarification_deadline=deadline)
                if event.outcome == "ok":
                    event.error = ""
                    event.request_phase = "complete"
                return
        except ClarificationExpired:
            event.outcome = "ambiguous"
            event.feedback_text = "That choice expired."
        except SpotifyAuthError as auth_exc:
            event.outcome, event.error = "error", str(auth_exc)
            event.feedback_text = "Spotify needs you to sign in."
        except SpotifyQuotaExceeded:
            event.outcome = "spotify_quota_exceeded"
            event.error = "SpotifyQuotaExceeded"
            event.feedback_text = SPOTIFY_QUOTA_MESSAGE
        except SpotifyRequestDeferred as rate_exc:
            event.outcome, event.error = "request_deferred", str(rate_exc)
            event.feedback_text = "Spotify requests are busy. Please try again shortly."
        except SpotifyRateLimited as rate_exc:
            event.outcome, event.error = "error", str(rate_exc)
            event.feedback_text = RATE_LIMITED_MESSAGE
        except Exception as execute_exc:
            event.outcome = "error"
            event.error = "{}: {}".format(type(execute_exc).__name__, execute_exc)
            event.feedback_text = SORRY
            log.exception("Candidate clarification failed")
        else:
            event.outcome = "ambiguous"
            event.feedback_text = "No selection made."
        event.feedback_kind = self._say_feedback(mic, event.feedback_text, event)

    def _retry_unknown(self, mic: Microphone, event: Event) -> None:
        """One explicit retry, then return to wake-word mode without guessing."""
        event.request_phase = "retry_capture"
        prompt = "Repeat, please."
        with self._duck_music(event, self.cfg.get("ducking.duck_to_percent", 20)):
            retry_text, captured, latency = self._ask_and_listen(mic, prompt)
        self._capture_result(event, captured)
        event.retry_transcript = retry_text
        event.retry_stt_latency_ms = latency
        event.mark("retry_stt_done")
        retry_event = Event(ts=event.ts, event_id=event.event_id + "-retry")
        event.retry_audio_path = self.telemetry.save_audio(retry_event, captured["audio"])
        if retry_text:
            # The prompt asks for a full repeat. Users naturally say
            # "Alexa, pause" again, so strip that validated prefix here too.
            retry_wake, retry_command = (self.listener.match_trigger(retry_text)
                                         if self.listener is not None else ("", ""))
            command = retry_command if retry_wake else retry_text
            if self._handle_conversation(command, mic, event, captured["audio"]):
                return
            if self._handle_global(command, mic, event):
                return
            intent = self._parse_intent(command, event)
            event.intent_action = intent.action.value
            event.intent_query = intent.query
            event.intent_volume_after_play = intent.volume_after_play
            event.intent_library_only = intent.library_only
            event.intent_source = intent.source
            event.matched_rule = intent.matched_rule
            if intent.action is not Action.UNKNOWN:
                self._execute(intent, event, command_audio=captured["audio"], mic=mic)
                event.request_phase = "complete"
                return
        event.outcome = "unknown_intent"
        event.feedback_text = SORRY
        event.feedback_kind = self._say_feedback(mic, event.feedback_text, event)
        mic.flush()

    def _strip_wakes(self, text: str) -> tuple:
        """(heard_wake, rest): leading wake phrases removed ("Alexa Alexa play x")."""
        heard = False
        while self.listener is not None and text:
            wake, rest = self.listener.match_trigger(text)
            if not wake:
                break
            heard, text = True, rest
        return heard, text

    def _past_repeated_wake(self, mic, event, text, captured, latency):
        """The follow-up was "Alexa" again: the user wasn't sure we heard.

        Live (22:41): every repeat was taken as the command, answered with
        "Repeat, please" and then "Sorry". Instead keep listening, silently
        apart from the tick, and strip the extra wake words from a command.
        Returns ("", ...) when nothing but wake words ever came.
        """
        for _ in range(2):
            heard, rest = self._strip_wakes(text)
            if not heard:
                return text, captured, latency
            if rest:
                return rest, captured, latency
            log.info("Wake repeated as the command; still listening")
            with self._duck_music(event, self.cfg.get("ducking.duck_to_percent", 20)):
                self.feedback.tick()
                captured = self.endpointer.record(mic, after_prompt=False)
            self._capture_result(event, captured)
            if not captured["speech_started"]:
                return "", captured, latency
            text, extra = self.stt.transcribe(captured["audio"])
            latency = (latency or 0) + extra
        heard, rest = self._strip_wakes(text)
        return (rest if heard else text), captured, latency

    def _acknowledge(self, mic: Microphone) -> str:
        """Say "Yes sir?" (or beep), then discard what the mic heard meanwhile.

        The flush is essential: playback and capture overlap in time, so without
        it the recording opens with the assistant's own prompt and the VAD ends
        the utterance before the user has said anything.
        """
        if not self.feedback.speaks_on_wake:
            # No spoken prompt to wait for: the music ducking is the cue, and
            # nothing is flushed, so words said right after "Alexa" are kept.
            return self.feedback.tick()
        kind = self.feedback.wake()
        self.feedback.listening()
        self._settle_after_prompt(mic)
        return kind

    def _settle_after_prompt(self, mic: Microphone) -> None:
        # Bluetooth keeps playing the end of the prompt after PortAudio reports
        # playback complete. 120 ms was too short: recordings opened on the
        # prompt's own tail. The endpointer also rejects any leftover burst.
        time.sleep(self.cfg.get("capture.playback_settle_ms", 350) / 1000)
        mic.flush()

    def _speech_only(self, audio: np.ndarray, pad_ms: int = 200) -> np.ndarray:
        """The stretch from first to last speech frame, with a little padding."""
        vad = getattr(self, "_trim_vad", None)
        if vad is None:
            vad = self._trim_vad = SileroVAD(16000)
        vad.reset()
        frames = range(0, audio.size - 511, 512)
        voiced = [i for i in frames if vad(audio[i:i + 512]) >= 0.35]
        if not voiced:
            return audio
        pad = pad_ms * 16
        return audio[max(0, voiced[0] - pad):min(audio.size, voiced[-1] + 512 + pad)]

    def _recover_with_name_hints(self, query: str, audio, event: Event):
        """Second listen for a song that was not found, biased to your artists.

        The command model runs without name hints (they once turned "pause"
        into "Buzz"), so it spells "Seedhe Maut" as "see the mode". Re-reading
        the same audio with the artist list as hints recovered all four such
        real clips. Only used after a plain lookup failed, and the result must
        still resolve on Spotify, so it cannot replace a correct hearing.
        """
        artists = (getattr(self, "vocabulary", None) or {}).get("artists") or []
        if audio is None or not getattr(audio, "size", 0) or not artists:
            return None
        hotwords = ", ".join(artists[:40])
        # Hints plus silence make Whisper loop ("K.R.A., K.R.A., ..."), so
        # decode only the spoken part and keep only the first sentence.
        audio = self._speech_only(audio)
        tried = []
        for model in (getattr(self, "listener_stt", None), self.stt):
            if model is None or not model.model_name.startswith("faster-whisper/"):
                continue
            try:
                text, latency = model.transcribe(audio, hotwords=hotwords)
            except Exception as exc:
                log.warning("Name-hinted decode failed: %s", exc)
                continue
            event.recovery_stt_latency_ms = (event.recovery_stt_latency_ms or 0) + latency
            text = _first_sentence(text)
            tried.append(text)
            wake, command, _ = (self.listener.locate_trigger(text)
                                if self.listener else ("", "", False))
            alternate = parse(command if wake else text)
            if alternate.action is not Action.PLAY_TRACK or _norm(alternate.query) == _norm(query):
                continue
            try:
                candidate = self.spotify.resolve_track(alternate.query)
            except LookupError:
                continue
            event.recovery_transcript = " | ".join(tried)
            log.info("Recovered song with name hints: %r -> %r", query, alternate.query)
            return candidate, alternate.query
        event.recovery_transcript = " | ".join(tried)
        return None

    def _recover_track_from_audio(self, query: str, audio: np.ndarray | None,
                                  event: Event):
        """Retry a failed artist name with exact-title catalogue hints only."""
        if audio is None or not audio.size or not self.stt.model_name.startswith("faster-whisper/"):
            return None
        hints = self.spotify.artist_hints_for_exact_title(query)
        if not hints:
            return None
        title, _, heard_artist = query.rpartition(" by ")
        orders = [hints + [heard_artist], [heard_artist] + list(reversed(hints))]
        recovered = []
        for order in orders:
            hotwords = ", ".join(order)
            try:
                text, latency = self.stt.transcribe(audio, hotwords=hotwords)
            except Exception as exc:
                log.warning("Targeted artist ASR failed; keeping original not-found result: %s", exc)
                return None
            event.recovery_stt_latency_ms = (event.recovery_stt_latency_ms or 0) + latency
            event.recovery_hotwords = hotwords
            recovered.append(text)
            wake, command = self.listener.match_trigger(text) if self.listener else ("", "")
            alternate = parse(command if wake else text)
            alt_title, separator, alt_artist = alternate.query.rpartition(" by ")
            if (alternate.action is not Action.PLAY_TRACK or not separator
                    or _norm(alt_title) != _norm(title)
                    or _norm(alt_artist) != _norm(hints[0])):
                event.recovery_transcript = " | ".join(recovered)
                return None
        event.recovery_transcript = " | ".join(recovered)
        try:
            candidate = self.spotify.resolve_track(alternate.query)
        except LookupError:
            return None
        if _norm(candidate.artist) != _norm(hints[0]):
            return None
        log.info("Recovered track artist with targeted ASR: %r -> %r",
                 query, alternate.query)
        return candidate, alternate.query

    @staticmethod
    def _was_repaired(query: str, candidate: Candidate, recovered: bool = False) -> bool:
        """Did resolution change what was heard, so the user should be told?

        True when targeted ASR recovery was used, the catalogue match score was
        below ``QUEUE_EXACT_SCORE``, or the title/artist differ from the heard
        words after ``_norm``. An artist is checked against every credited artist.
        """
        if recovered:
            return True
        meta = candidate.metadata or {}
        score = meta.get("match_score")
        if isinstance(score, (int, float)) and score < QUEUE_EXACT_SCORE:
            return True
        def words(text: str) -> str:
            # _norm keeps a gap where punctuation was ("Thank U, Next").
            return " ".join(_norm(text).split())

        title, separator, artist = query.rpartition(" by ")
        if not separator:
            title, artist = query, ""
        if words(title) != words(candidate.name):
            return True
        if not artist.strip():
            return False
        credited = [candidate.artist] + [a.get("name", "") if isinstance(a, dict) else str(a)
                                         for a in meta.get("artists") or []]
        return words(artist) not in {words(name) for name in credited}

    def _literal_title(self, intent: Intent) -> Candidate | None:
        """A library song whose title contains words a rule stripped as a modifier.

        "play only girl", "play thank u next": "only"/"next" may be part of the
        title. Only an exact local hit counts, so a real "play X next" still queues.
        """
        rule = intent.matched_rule
        if rule.endswith("_only_before"):
            literal = "only " + intent.query
        elif rule.endswith("_only_after"):
            literal = intent.query + " only"
        elif rule == "queue_play_next":
            literal = intent.query + " next"
        else:
            return None
        try:
            hit = self.spotify.resolve_local(literal)
        except Exception:
            log.exception("Local title check failed")
            return None
        if isinstance(hit, Candidate) and not self._was_repaired(literal, hit):
            return hit
        return None

    def _radio_after(self, candidate: Candidate, enabled: bool = True) -> list[str]:
        """Track URIs to play after ``candidate``, from the local index.

        Never raises: radio is an extra, and must not stand between the user and
        the song they asked for.
        """
        cfg = getattr(self, "cfg", None)
        uri = getattr(candidate, "uri", "")
        if not enabled or not uri or (cfg is not None and not cfg.get("spotify.radio_after_song", True)):
            return []
        try:
            length = min(int(cfg.get("spotify.radio_length", RADIO_LIMIT)) if cfg else RADIO_LIMIT,
                         RADIO_LIMIT)
            names = [a.get("name", "") if isinstance(a, dict) else str(a)
                     for a in (candidate.metadata or {}).get("artists") or []]
            seed = {"uri": uri, "artists": names or [candidate.artist]}
            urls = catalogue.radio(self.spotify.catalogue_index(), seed, length=length,
                                   recent=catalogue.radio_history())
        except Exception:
            log.exception("Radio unavailable; playing the song alone")
            return []
        if isinstance(urls, list) and urls:
            log.info("Radio after %r: %d tracks", getattr(candidate, "label", uri), len(urls))
            return urls
        return []

    def _play_with(self, candidate: Candidate, then: list[str]) -> None:
        """Start ``candidate`` followed by ``then``; remember a radio that really went out."""
        sp = self.spotify
        if not then:
            sp.play_candidate(candidate)
            return
        sp.play_candidate(candidate, then=then)
        if getattr(sp, "radio_played", False) is True:
            try:
                catalogue.remember_radio(then)
            except Exception:
                log.exception("Could not record radio history")

    def _resolve_requested_track(self, query: str, event: Event,
                                 command_audio: np.ndarray | None) -> Candidate:
        """Cache, then catalogue/search, then exact-title ASR recovery."""
        sp = self.spotify
        sp.last_resolution = None
        try:
            candidate = sp.cached_track(query)
            if not isinstance(candidate, Candidate):
                if getattr(sp, "web_rate_limited", False) is True:
                    candidate = sp.resolve_local(query)
                    if not isinstance(candidate, Candidate):
                        candidate = sp.resolve_best_available(query, "track", offline_only=True)
                else:
                    candidate = sp.resolve_track(query)
        except (AmbiguousMatch, MusicConfirmationRequired):
            self._note_resolution(event)
            raise
        except LookupError:
            # Keep why the plain lookup failed even if recovery then succeeds.
            self._note_resolution(event)
            # A catalog hit is not proof of what the user said. Broad
            # artist hints could rewrite both title and artist. Only
            # the exact-title, agreeing-decodes recovery is allowed.
            recovered = self._recover_track_from_audio(query, command_audio, event)
            if recovered is None:
                raise
            event.heard_query = query
            candidate, event.intent_query = recovered
            event.matched_rule += "+targeted_asr"
            self._note_resolution(event, candidate, prefix="targeted_asr>")
            return candidate
        self._note_resolution(event, candidate)
        return candidate

    def _resolve_available_track(self, intent: Intent, event: Event,
                                  command_audio: np.ndarray | None = None,
                                  allow_album: bool = False) -> Candidate:
        # Share one search budget through recovery and album fallback instead
        # of resetting the allowance inside every successive lookup.
        factory = getattr(type(self.spotify), "resolution_budget", None)
        scope = (self.spotify.resolution_budget(max_searches=3)
                 if callable(factory) else nullcontext())
        with scope:
            try:
                return self._resolve_available_track_bounded(intent, event, command_audio, allow_album)
            except MusicConfirmationRequired as exc:
                exc.intent = intent
                raise

    def _resolve_available_track_bounded(self, intent: Intent, event: Event,
                                        command_audio: np.ndarray | None,
                                        allow_album: bool) -> Candidate:
        """Confident first; then existing choices or best available of the requested kind."""
        sp = self.spotify
        # An existing ban cannot justify choosing an approximate saved song
        # ahead of a strictly matched album for an unspecified play request.
        if allow_album and not intent.explicit_track and getattr(sp, "web_rate_limited", False) is True:
            confident = sp.cached_track(intent.query)
            if not isinstance(confident, Candidate):
                confident = sp.resolve_local(intent.query)
            if isinstance(confident, Candidate):
                self._note_resolution(event, confident)
                return confident
            try:
                album = sp.resolve_album(intent.query, offline_only=True)
            except AmbiguousMatch as exc:
                album = sp.resolve_best_available(intent.query, "album", tuple(exc.candidates), offline_only=True)
            except LookupError:
                album = None
            if isinstance(album, Candidate):
                self._note_resolution(event, album)
                return album
        try:
            return self._resolve_requested_track(intent.query, event, command_audio)
        except AmbiguousMatch as exc:
            candidate = sp.resolve_best_available(intent.query, "track", tuple(exc.candidates))
        except TrackLookupError:
            if allow_album and not intent.explicit_track:
                try:
                    candidate = sp.resolve_album(intent.query)
                except AmbiguousMatch as exc:
                    candidate = sp.resolve_best_available(intent.query, "album", tuple(exc.candidates))
                except LookupError:
                    candidate = sp.resolve_best_available(intent.query, "track")
            else:
                candidate = sp.resolve_best_available(intent.query, "track")
        self._note_resolution(event, candidate)
        return candidate

    def _note_resolution(self, event: Event, candidate: Candidate | None = None,
                         prefix: str = "") -> None:
        """Copy how Spotify answered the last lookup into the command's telemetry."""
        diag = getattr(self.spotify, "last_resolution", None)
        if isinstance(diag, dict):
            event.resolution_path = prefix + str(diag.get("path") or "")
            event.resolution_score = diag.get("score")
            event.resolution_runner_up = diag.get("runner_up") or ""
            event.resolution_margin = diag.get("margin")
        elif prefix:
            event.resolution_path = prefix.rstrip(">")
        if isinstance(candidate, Candidate):
            event.resolved_label = candidate.label
            event.mark("resolved")

    def _note_playback(self, event: Event) -> None:
        """Which route carried the playback (local media keys, Web API, desktop link)."""
        diag = getattr(self.spotify, "last_playback", None)
        if isinstance(diag, dict):
            event.playback_method = str(diag.get("method") or "")
            if isinstance(diag.get("radio"), int):
                event.radio_count = diag["radio"]
        elif not event.playback_method:
            event.playback_method = "web_api"   # transport call that had no local route
        event.mark("playback_confirmed")

    def _nothing_playing(self) -> bool:
        """Local media session says Spotify is not playing (no player API call)."""
        try:
            from . import local_media
            # Unknown media state is not proof that music is stopped. A queue
            # request must not unexpectedly replace playback on that inference.
            return local_media.is_playing() is False
        except Exception:
            log.exception("Could not read local playback state; assuming music is playing")
            return False

    def _split_queue_request(self, query: str) -> list[Candidate]:
        """Two songs for "X and Y", only if the whole is not a title and both halves are exact.

        Local lookups only. Anything less certain is treated as one title, since
        "love and hate" must not become two unrelated songs.
        """
        parts = [part.strip() for part in re.split(r"\s+and\s+", query)]
        if len(parts) != 2 or not all(parts):
            return []
        sp = self.spotify
        try:
            whole = sp.resolve_local(query)
            if isinstance(whole, Candidate) and not self._was_repaired(query, whole):
                return []
            found = [sp.resolve_local(part) for part in parts]
        except Exception:
            log.exception("Local lookup for a two-song queue failed")
            return []
        if (not all(isinstance(c, Candidate) for c in found)
                or any(self._was_repaired(part, c) for part, c in zip(parts, found))
                or found[0].uri == found[1].uri):
            return []
        return found

    def _execute(self, intent: Intent, event: Event,
                 command_audio: np.ndarray | None = None,
                 mic: Microphone | None = None,
                 confirmed_candidate: Candidate | None = None,
                 clarification_deadline: float | None = None) -> None:
        # This method never listens. Clarification/retry capture belongs to its
        # callers, after this scope unwinds even if a lookup raises.
        source = mic if mic is not None else getattr(self, "_active_mic", None)
        _check_music_deadline(clarification_deadline)
        if confirmed_candidate is not None:
            album_choice = confirmed_candidate.uri.startswith("spotify:album:")
            if ((intent.action is Action.PLAY_ALBUM and not album_choice)
                    or (album_choice and (intent.action is Action.QUEUE or intent.explicit_track))):
                raise ValueError("Confirmed media kind disagrees with the request")
        try:
            with self._discard_while_busy(source):
                if intent.action in (Action.OPEN_APP, Action.SEARCH_BROWSER,
                                     Action.SHOW_TRACK, Action.SHOW_ALBUM, Action.START_JAM):
                    self._execute_desktop(intent, event, source)
                    return
                self._execute_music(intent, event, command_audio, source, confirmed_candidate,
                                    clarification_deadline)
        except AmbiguousMatch as exc:
            exc.intent = intent
            raise

    def _execute_desktop(self, intent: Intent, event: Event, mic=None) -> None:
        from . import computer_tools
        controller = getattr(self, "global_assistant", None)
        if callable(getattr(type(controller), "retire_navigation", None)):
            # A newer desktop request owns the view. A delayed Codex receipt
            # must not take focus, but its job and receipt still remain valid.
            controller.retire_navigation()
        event.request_phase = "execute"
        event.spotify_action = ""
        event.playback_method = "desktop_request"
        try:
            if intent.action is Action.OPEN_APP:
                message = computer_tools.open_app(intent.query)
            elif intent.action is Action.SEARCH_BROWSER:
                message = computer_tools.search_browser(intent.query, intent.browser)
            elif intent.action is Action.START_JAM:
                message = computer_tools.start_spotify_jam()
            else:
                message = computer_tools.show_spotify(
                    intent.query, "album" if intent.action is Action.SHOW_ALBUM else "track")
            event.outcome = "ok" if intent.action is Action.START_JAM else "requested"
        except computer_tools.ComputerDeliveryUnknown as exc:
            message = str(exc)
            event.outcome = "desktop_unknown_delivery"
        except computer_tools.ComputerControlError as exc:
            message = str(exc)
            event.outcome = "desktop_refused"
        event.feedback_text = message
        event.feedback_kind = (self._say_feedback(mic, message, event) if mic is not None
                               else self.feedback.say(message, force=True, blocking=False))
        event.mark("feedback_done")

    def _execute_music(self, intent: Intent, event: Event,
                       command_audio: np.ndarray | None = None,
                       mic: Microphone | None = None,
                       confirmed_candidate: Candidate | None = None,
                       clarification_deadline: float | None = None) -> None:
        event.request_phase = "execute"
        _check_music_deadline(clarification_deadline)
        t0 = time.perf_counter()
        sp = self.spotify
        action = intent.action
        sp.last_resolution = sp.last_playback = None   # read back below for telemetry
        reported = action            # what actually ran, for telemetry
        queued_message = ""          # spoken instead of the beep when a queue was repaired

        # A recognized voice command should open the desktop client even for
        # system-volume commands, which otherwise never touch Spotify's API.
        # Do not launch for an unrecognized utterance or a bare wake word.
        launched = action is not Action.UNKNOWN and sp.ensure_client_running()

        def ready_to_play() -> None:
            # Track and playlist lookups run on the Web API while a freshly
            # launched client starts; only playback itself has to wait.
            if launched and action is not Action.VOLUME:
                sp.wait_for_client()
            _check_music_deadline(clarification_deadline)
            self._release_paused_mic_for_playback()

        if action in (Action.RESUME, Action.NEXT, Action.PREV, Action.RESTART, Action.RESTART_ALBUM):
            ready_to_play()
        elif action is Action.PAUSE and launched:
            sp.wait_for_client()

        if action is Action.PAUSE:
            sp.pause()
        elif action is Action.RESTART_ALBUM:
            sp.restart_album()
        elif action is Action.RESUME:
            sp.play_resume()
            if intent.volume_after_play is not None:
                self._set_volume(Intent(Action.VOLUME, value=intent.volume_after_play))
        elif action is Action.NEXT:
            sp.next()
        elif action is Action.PREV:
            sp.previous()
        elif action is Action.RESTART:
            sp.restart_track()
        elif action is Action.VOLUME:
            self._set_volume(intent)
        elif action is Action.PLAY_PLAYLIST:
            try:
                sp.play_playlist(intent.query, library_only=intent.library_only,
                                 before_play=ready_to_play)
                if intent.volume_after_play is not None:
                    self._set_volume(Intent(Action.VOLUME, value=intent.volume_after_play))
            except (AmbiguousMatch, MusicConfirmationRequired):
                raise
            except LookupError:
                message = NOT_FOUND_MESSAGE
                event.outcome = "not_found"
                event.feedback_text = message
                event.feedback_kind = self._say_feedback(mic, message, event)
                return
        elif action is Action.PLAY_ARTIST:
            try:
                candidate = sp.resolve_artist_track(intent.query)
            except LookupError:
                self._note_resolution(event)
                event.outcome = "not_found"
                event.feedback_text = NOT_FOUND_MESSAGE
                event.feedback_kind = self._say_feedback(mic, NOT_FOUND_MESSAGE, event)
                return
            self._note_resolution(event, candidate)
            then = self._radio_after(candidate, intent.radio)
            ready_to_play()
            self._play_with(candidate, then)
            self._note_playback(event)
            if intent.volume_after_play is not None:
                self._set_volume(Intent(Action.VOLUME, value=intent.volume_after_play))
        elif action is Action.PLAY_ALBUM:
            try:
                if confirmed_candidate is not None:
                    candidate = confirmed_candidate
                elif getattr(sp, "web_rate_limited", False) is True:
                    try:
                        candidate = sp.resolve_album(intent.query, offline_only=True)
                    except AmbiguousMatch as exc:
                        candidate = sp.resolve_best_available(intent.query, "album", tuple(exc.candidates), offline_only=True)
                    except LookupError:
                        candidate = sp.resolve_best_available(intent.query, "album", offline_only=True)
                else:
                    try:
                        candidate = sp.resolve_album(intent.query)
                    except AmbiguousMatch as exc:
                        candidate = sp.resolve_best_available(intent.query, "album", tuple(exc.candidates))
                    except LookupError:
                        candidate = sp.resolve_best_available(intent.query, "album")
                if confirmed_candidate is not None:
                    event.matched_rule += "+confirmed_album"
            except (AmbiguousMatch, MusicConfirmationRequired):
                raise
            except LookupError:
                self._note_resolution(event)
                event.outcome = "not_found"
                event.feedback_text = NOT_FOUND_MESSAGE
                event.feedback_kind = self._say_feedback(mic, NOT_FOUND_MESSAGE, event)
                return
            self._note_resolution(event, candidate)
            ready_to_play()
            sp.play_album(candidate)
            self._note_playback(event)
            if intent.volume_after_play is not None:
                self._set_volume(Intent(Action.VOLUME, value=intent.volume_after_play))
        elif action is Action.PLAY_TRACK:
            # Preserve the parsed query; entity resolution belongs to Spotify.
            try:
                literal = self._literal_title(intent) if confirmed_candidate is None else None
                if confirmed_candidate is not None:
                    candidate = confirmed_candidate
                    self._note_resolution(event, candidate, prefix="confirmed_candidate>")
                elif literal is not None:
                    candidate = literal
                    event.matched_rule += "+literal_title"
                    self._note_resolution(event, candidate, prefix="literal_title>")
                else:
                    candidate = self._resolve_available_track(intent, event, command_audio, allow_album=True)
                if isinstance(candidate, Candidate) and candidate.uri.startswith("spotify:album:"):
                    reported = Action.PLAY_ALBUM
                    ready_to_play()
                    sp.play_album(candidate)
                    self._note_playback(event)
                    if intent.volume_after_play is not None:
                        self._set_volume(Intent(Action.VOLUME, value=intent.volume_after_play))
                else:
                    resolved_ms = int((time.perf_counter() - t0) * 1000)
                    then = self._radio_after(candidate, intent.radio or literal is not None)
                    ready_to_play()
                    self._play_with(candidate, then)
                    self._note_playback(event)
                    # Confirmations never teach the misheard query as a preference.
                    if (not event.recovery_transcript and literal is None and confirmed_candidate is None
                            and getattr(candidate, "approximate", False) is not True):
                        sp.remember_track(intent.query, candidate)
                    log.info("Song timing: found in %d ms, playing after %d ms",
                             resolved_ms, int((time.perf_counter() - t0) * 1000))
                    if intent.volume_after_play is not None:
                        self._set_volume(Intent(Action.VOLUME, value=intent.volume_after_play))
            except (AmbiguousMatch, MusicConfirmationRequired):
                raise
            except LookupError:
                # No readback of the misheard name and no second recording:
                # repeating a wrong transcript aloud was slow and grating, and
                # the retry mostly captured noise. Say Alexa again instead.
                message = NOT_FOUND_MESSAGE
                event.outcome = "not_found"
                event.feedback_text = message
                event.feedback_kind = self._say_feedback(mic, message, event)
                return
        elif action is Action.QUEUE:
            try:
                literal = self._literal_title(intent) if confirmed_candidate is None else None
                if confirmed_candidate is not None:
                    candidates = [confirmed_candidate]
                    self._note_resolution(event, confirmed_candidate, prefix="confirmed_candidate>")
                elif literal is not None:
                    candidates = [literal]
                    event.matched_rule += "+literal_title"
                    self._note_resolution(event, literal, prefix="literal_title>")
                else:
                    candidates = self._split_queue_request(intent.query)
                    if candidates:
                        self._note_resolution(event, candidates[0], prefix="split_queue>")
                        event.resolved_label = " + ".join(c.label for c in candidates)
                    else:
                        candidates = [self._resolve_available_track(intent, event, command_audio)]
            except (AmbiguousMatch, MusicConfirmationRequired):
                raise
            except LookupError:
                event.outcome = "not_found"
                event.feedback_text = NOT_FOUND_MESSAGE
                event.feedback_kind = self._say_feedback(mic, NOT_FOUND_MESSAGE, event)
                return
            if literal is not None or self._nothing_playing():
                # Nothing to queue behind: play it now, with the radio, like
                # "play X". Further songs of "X and Y" lead the radio.
                reported = Action.PLAY_TRACK
                first = candidates[0]
                then = ([c.uri for c in candidates[1:]]
                        + self._radio_after(first, intent.radio))[:RADIO_LIMIT]
                ready_to_play()
                self._play_with(first, then)
                self._note_playback(event)
                if (len(candidates) == 1 and not event.recovery_transcript and literal is None
                        and confirmed_candidate is None and getattr(first, "approximate", False) is not True):
                    sp.remember_track(intent.query, first)
            else:
                try:
                    for candidate in candidates:
                        _check_music_deadline(clarification_deadline)
                        sp.queue(candidate)
                except SpotifyQuotaExceeded:
                    event.outcome = "spotify_quota_exceeded"
                    event.error = "SpotifyQuotaExceeded"
                    event.feedback_text = SPOTIFY_QUOTA_MESSAGE
                    event.feedback_kind = self._say_feedback(mic, SPOTIFY_QUOTA_MESSAGE, event)
                    return
                except SpotifyRateLimited as exc:
                    # No local way to queue; this is not a search problem, so
                    # not the "search is blocked" message.
                    event.outcome = "error"
                    event.error = str(exc)
                    event.feedback_text = SORRY
                    event.feedback_kind = self._say_feedback(mic, SORRY, event)
                    return
                event.queued = len(candidates)
                event.playback_method = "queue_api"
                log.info("Queued %s", ", ".join(c.label for c in candidates))
                if len(candidates) == 1:
                    if (not event.recovery_transcript and confirmed_candidate is None
                            and getattr(candidates[0], "approximate", False) is not True):
                        sp.remember_track(intent.query, candidates[0])
                    if (getattr(candidates[0], "approximate", False) is not True
                            and self._was_repaired(intent.query, candidates[0],
                                          "+targeted_asr" in event.matched_rule)):
                        queued_message = "Queued {}.".format(candidates[0].label)

        if action is Action.VOLUME:
            event.spotify_action = ""
            event.playback_method = "windows_volume"
            event.mark("playback_confirmed")
        else:
            event.spotify_action = reported.value
            event.spotify_latency_ms = int((time.perf_counter() - t0) * 1000)
            event.spotify_device = sp.device_name
            self._note_playback(event)
        event.outcome = "ok"
        if queued_message:
            # The beep cannot say which song was queued; a repaired match deserves a name.
            event.feedback_text = queued_message
            event.feedback_kind = (self._say_feedback(mic, queued_message, event)
                                   if mic is not None else
                                   self.feedback.say(queued_message, force=True, blocking=False))
        else:
            event.feedback_kind = self.feedback.ok()
        event.mark("feedback_done")

    def _set_volume(self, intent: Intent) -> None:
        if intent.volume_mode == "absolute" and intent.value is not None:
            if type(intent.value) is not int or not 0 <= intent.value <= 100:
                raise ValueError("Invalid absolute volume")
            ducking.set_command_volume(intent.value / 100)
            return
        current = ducking.get_system_volume() * 100
        if intent.query not in ("up", "down"):
            raise ValueError("Invalid volume direction")
        amount = VOLUME_STEP if intent.value is None else intent.value
        if type(amount) is not int or not 0 <= amount <= 100:
            raise ValueError("Invalid relative volume")
        if intent.volume_mode == "relative_percent":
            amount = current * amount / 100
        elif intent.volume_mode not in ("absolute", "relative_points"):
            raise ValueError("Invalid volume mode")
        target = current + (amount if intent.query == "up" else -amount)
        ducking.set_command_volume(max(0, min(100, int(target + 0.5))) / 100)

    def _parse_intent(self, text: str, event: Event) -> Intent:
        # Text has already been captured. Bound the callback backlog during
        # potentially slow local inference without wrapping any reply capture.
        with self._discard_while_busy(getattr(self, "_active_mic", None)):
            return self._parse_intent_text(text, event)

    def _parse_intent_text(self, text: str, event: Event) -> Intent:
        started = time.perf_counter()
        try:
            if self.cfg.get("intent.backend", "rules") == "rules":
                result = parse(text)
            elif self.cfg.get("intent.backend") == "brain":
                from .brain import decide
                result = decide(text, getattr(self, "brain", None))
            elif self.cfg.get("intent.backend") == "local_llm":
                from .intent_llm import LocalIntentParser
                parser = getattr(self, "_intent_parser", None)
                if parser is None:
                    parser = self._intent_parser = LocalIntentParser(
                        self.cfg.get("intent.model", "qwen3:0.6b"),
                        self.cfg.get("intent.timeout_seconds", 4.0))
                # Exact transport commands need no model call or extra delay.
                exact = {"pause": Action.PAUSE, "stop": Action.PAUSE,
                         "resume": Action.RESUME, "next": Action.NEXT,
                         "previous": Action.PREV}
                command = text.strip().casefold().rstrip(".!?")
                result = (Intent(exact[command], matched_rule="exact_transport")
                          if command in exact else parser.parse(text))
            else:
                raise ValueError("Unknown intent backend")
            event.intent_value = result.value
            event.intent_volume_mode = result.volume_mode
            return result
        finally:
            event.intent_latency_ms = int((time.perf_counter() - started) * 1000)
            event.mark("intent_done", overwrite=True)

    # ---------- loop ----------

    def _drain_global_receipts(self, controller):
        """Log native acceptance separately from the earlier local plan."""
        if not callable(getattr(type(controller), "take_receipt", None)):
            return
        for _ in range(8):
            receipt = controller.take_receipt()
            if not isinstance(receipt, dict):
                break
            event = Event(trigger_source="global_receipt", request_phase="global",
                          global_action="create_receipt", global_target=receipt.get("id", ""),
                          global_model=receipt.get("model", ""),
                          global_status=receipt.get("status", ""),
                          global_turn_id=receipt.get("turn_id") or "",
                          outcome="global_" + receipt.get("status", "unknown"))
            self.telemetry.write(event)

    def _speak_global_notice_if_quiet(self, mic, utterance=None, *, async_suspended=False):
        """Owner-only notices after a full quiet second in the STT loop."""
        controller = getattr(self, "global_assistant", None)
        runtime = getattr(self, "async_capture", None)
        listener = (runtime.frontend if async_suspended and runtime is not None
                    else getattr(self, "listener", None))
        now = time.monotonic()
        probability = (runtime.last_vad_probability if async_suspended and runtime is not None
                       else getattr(listener, "last_vad_probability", None))
        busy = getattr(self, "_busy", None)
        trigger = getattr(self, "_trigger", None)
        lease = getattr(self, "_early_lease", None)
        recent_wake = getattr(self, "_last_strong_wake", None)
        in_speech = (getattr(listener, "_active", None) is not None
                     or getattr(listener, "_boundary", None) is not None
                     if async_suspended and listener is not None
                     else getattr(listener, "_in_speech", True))
        unsafe = (controller is None or listener is None or utterance is not None
                  or (not async_suspended and in_speech)
                  or (not async_suspended and (not isinstance(probability, (int, float))
                      or probability >= self.cfg.get("capture.vad_threshold", 0.35)))
                  or controller.session.active
                  or (busy is not None and busy.locked())
                  or (trigger is not None and trigger.is_set())
                  or (lease is not None and getattr(lease, "_owned", False))
                  or getattr(self, "_paused_mic", None) is not None
                  or (recent_wake is not None and now - recent_wake[0] < 2.0))
        if unsafe:
            self._global_quiet_since = None
            return False
        quiet_since = getattr(self, "_global_quiet_since", None)
        if quiet_since is None:
            self._global_quiet_since = now
            return False
        if now - quiet_since < 1.0:
            return False
        try:
            notice = controller.take_notice()
        except Exception as exc:
            log.debug("Global notice unavailable (%s)", type(exc).__name__)
            return False
        if not notice:
            return False
        event = Event(trigger_source="global_notice", request_phase="global")
        try:
            success = self._global_say(mic, notice, event)
            event.outcome = "global_notice" if success else "global_feedback_failed"
        except Exception as exc:
            event.outcome = "global_feedback_failed"
            event.error = type(exc).__name__
            log.warning("Global notice feedback failed (%s)", type(exc).__name__)
        finally:
            self._global_quiet_since = None
            if not async_suspended:
                listener.flush()
            if self.wake is not None:
                self.wake.reset()
            self._last_strong_wake = None
        self.telemetry.write(event)
        return True

    def run(self) -> None:
        self.async_capture = getattr(self, "async_capture", None)
        if self.async_capture is not None and not hasattr(self, "_async_cued_segments"):
            self._async_cued_segments = set()
        cooldown = self.cfg.get("wake.cooldown_seconds", 2.0)
        last_trigger = 0.0
        last_early_log = 0.0
        last_confirmation = float("-inf")
        mic_recovery_attempts = 0
        observed_overflows = 0

        playback_stop = threading.Event()
        playback_thread = None

        async_context = self.async_capture if self.async_capture is not None else nullcontext()
        with async_context, KeepAwakeGuard(self.cfg.get("runtime.keep_awake_on_ac", False)) as power_guard, Microphone(
            sample_rate=16000,
            frame_length=512,
            ring_seconds=self.cfg.get("audio.ring_buffer_seconds", 30),
            device=self.cfg.get("audio.input_device") or None,
            capture_rate=self.cfg.get("audio.capture_rate"),
            exclusive=self.cfg.get("audio.exclusive_mode", False),
        ) as mic:
            self._start_hotkeys(mic)
            if self.mode == "stt" and self.cfg.get("audio.echo_cancellation", False):
                from .echo import open_echo_canceller
                self.echo = open_echo_canceller(
                    self.cfg.get("audio.echo_reference", "") or "")
                mic.echo = self.echo
            if self.mode == "stt":
                log.info("Memoir is listening. Say: %s",
                         ", ".join(self.listener.phrases))
            else:
                log.info("Memoir is listening.")

            try:
                self._active_mic = mic
                self._async_was_paused = False
                if self.cfg.get("audio.paused_input_device") and self.mode == "stt":
                    playback_thread = threading.Thread(
                        target=self._watch_playback, args=(playback_stop,), daemon=True)
                    playback_thread.start()
                while not self._stop.is_set():
                    global_controller = getattr(self, "global_assistant", None)
                    if global_controller is not None:
                        try:
                            global_controller.poll()
                            self._drain_global_receipts(global_controller)
                        except Exception as exc:
                            if not getattr(self, "_global_poll_fault_logged", False):
                                log.warning("Global result handling failed (%s); audio continues", type(exc).__name__)
                                self._global_poll_fault_logged = True
                    try:
                        frame = mic.read()
                    except TimeoutError:
                        if self._stop.is_set():
                            break
                        if self._early_lease is not None:
                            self._early_lease.release()
                        if self.async_capture is not None:
                            self.async_capture.flush("microphone_timeout")
                            self._reset_async_capture_hints(mic)
                        elif self.mode == "stt":
                            self.listener.flush()
                        if self.async_capture is None:
                            if self.wake is not None:
                                self.wake.reset()
                            self._last_strong_wake = None
                        while True:
                            if self._stop.is_set():
                                break
                            if mic_recovery_attempts >= 2:
                                raise
                            mic_recovery_attempts += 1
                            log.warning("Microphone stalled; reopening it (attempt %d/2)",
                                        mic_recovery_attempts)
                            try:
                                mic.reopen()
                            except Exception:
                                if self._stop.is_set():
                                    break
                                if mic_recovery_attempts >= 2:
                                    raise
                                log.warning("Microphone reopen failed; retrying", exc_info=True)
                                continue
                            break
                        continue
                    mic_recovery_attempts = 0
                    overflow_count = getattr(mic, "queue_overflow_count", 0)
                    if isinstance(overflow_count, int) and not isinstance(overflow_count, bool):
                        dropped = overflow_count > observed_overflows
                        observed_overflows = overflow_count
                        if dropped:
                            # Missing callback periods break PCM continuity. Do not
                            # join a previous wake/command to audio after that gap.
                            self._recover_capture_queue_gap(mic)
                            continue
                    listening_paused = self._listening_paused()
                    power_guard.update(listening=not listening_paused)
                    if self.async_capture is not None:
                        if listening_paused and not self._async_was_paused:
                            self.async_capture.flush("listening_paused")
                            self._reset_async_capture_hints(mic)
                            self._async_was_paused = True
                        elif not listening_paused and self._async_was_paused:
                            self.async_capture.flush("listening_resumed")
                            self._reset_async_capture_hints(mic)
                            self._async_was_paused = False
                            continue
                    if listening_paused:
                        if self.async_capture is not None:
                            self.async_capture.counters["paused_frames"] += 1
                        continue
                    self._sync_paused_mic()
                    if self._trigger.is_set():
                        self._trigger.clear()
                        if time.monotonic() - last_trigger >= cooldown:
                            last_trigger = time.monotonic()
                            if self.async_capture is not None:
                                self._run_async_owned_callback(
                                    mic, lambda: self.handle(mic, "hotkey"),
                                    reason="hotkey handler")
                            else:
                                if self.mode == "stt":
                                    self.listener.flush()
                                self.handle(mic, "hotkey")
                                if self.wake is not None:
                                    self.wake.reset()
                                self._last_strong_wake = None
                        continue
                    if self.async_capture is not None:
                        (last_trigger, last_confirmation, last_early_log) = self._run_async_frame(
                            mic, frame, last_trigger=last_trigger,
                            last_confirmation=last_confirmation,
                            last_early_log=last_early_log, cooldown=cooldown)
                        continue
                    if self.mode == "stt":
                        if getattr(self, "_paused_mic", None) is not None:
                            paused_frame = self._read_paused_mic_frame()
                            paused_utt = (self._paused_listener.feed(paused_frame)
                                          if paused_frame is not None else None)
                            if paused_utt is not None and paused_utt.triggered:
                                if time.monotonic() - last_trigger >= cooldown:
                                    last_trigger = time.monotonic()
                                    source_mic = self._paused_mic
                                    self.handle_utterance(source_mic, paused_utt, input_device="sounddrum")
                                    if self.wake is not None:
                                        self.wake.reset()
                                    self._last_strong_wake = None
                                    mic.flush()
                                    self.listener.flush()
                                    if self._paused_listener is not None:
                                        self._paused_listener.flush()
                                    continue
                        if self.wake is not None:
                            tier = self.wake.process(frame)
                            self._trace_wake()
                            personal = getattr(self.wake, "personal_hit", False) is True
                            if tier == TIER_BOTH:
                                self._last_strong_wake = (
                                    time.monotonic(), self.wake.last_score,
                                    getattr(self.wake, "last_model", ""))
                            if personal:
                                # Also corroborates aliases ("Alexei") and
                                # mid-utterance wakes in the text path.
                                self._last_strong_wake = (
                                    time.monotonic(), self.wake.personal_score, "alexa")
                            if tier == TIER_BOTH and self._early_lease is not None:
                                now = time.monotonic()
                                if now - last_early_log >= 2.0:
                                    log.info("Early wake hint awaiting confirmation: model=%s score=%s",
                                             getattr(self.wake, "last_model", ""),
                                             getattr(self.wake, "last_score", ""))
                                    last_early_log = now
                            # The user says Alexa and goes straight on to the
                            # command; the music ducking is the cue. A strong acoustic hit
                            # should not depend on small.en spelling the wake
                            # word correctly over music. Keep the text path
                            # below for weaker acoustic detections.
                            confirm_due = (time.monotonic() - last_confirmation >= 1.0
                                           and getattr(self.listener, "_woke", False) is not True)
                            if (((tier == TIER_BOTH
                                    and self.wake.last_score >= self.cfg.get(
                                        "wake.acoustic_activation_threshold", 0.75))
                                    or personal)
                                    and time.monotonic() - last_trigger >= cooldown
                                    and confirm_due):
                                # A rejected peak otherwise schedules another
                                # synchronous decode on the very next frame.
                                last_confirmation = time.monotonic()
                                confirmed = self._confirm_acoustic_wake(mic)
                                history = getattr(self, "_confirmed_acoustic_history", None)
                                handoff = getattr(type(self.listener), "confirm_wake", None)
                                if history is not None and callable(handoff):
                                    phrase, raw_history, clean_history = history
                                    # The current read is fed below exactly once. The
                                    # retained prefix covers only earlier consumed frames.
                                    size = mic.frame_length
                                    seeded = self.listener.confirm_wake(
                                        phrase, raw_history[:-size],
                                        clean_history[:-size] if clean_history is not None else None)
                                    self._confirmed_acoustic_history = None
                                    confirmed = False
                                    if not seeded:
                                        self._record_wake_rejection("acoustic_history_invalid", mic=mic)
                                elif not confirmed and history is None:
                                    self._record_wake_rejection("acoustic_unconfirmed", mic=mic)
                            else:
                                confirmed = False
                            if confirmed:
                                last_trigger = time.monotonic()
                                self.listener.flush()
                                self.handle(mic, "wake", tier or ("personal" if personal else ""))
                                self.wake.reset()
                                self._last_strong_wake = None
                                if getattr(self, "_paused_mic", None) is not None:
                                    self._paused_mic.flush()
                                    self._paused_listener.flush()
                                continue
                            if self._early_lease is not None and self._early_lease.expired():
                                self._early_lease.release()
                        self._check_echo_reference()
                        cleaned_frame = mic.vad_frame(frame)
                        utt, feed_gap, feed_overflows = self._feed_listener_guarded(
                            mic, frame, cleaned_frame)
                        if feed_gap:
                            if isinstance(feed_overflows, int) and not isinstance(feed_overflows, bool):
                                observed_overflows = max(observed_overflows, feed_overflows)
                            continue
                        self._record_capture_diagnostics(
                            mic, frame, cleaned_frame,
                            getattr(self.listener, "last_vad_probability", None))
                        if utt is not None and not utt.triggered:
                            self._record_wake_rejection("speech_no_trigger", mic=mic,
                                duration_ms=int(utt.audio.size * 1000 / 16000))
                        if utt is None or not utt.triggered:
                            self._speak_global_notice_if_quiet(mic, utt)
                            continue
                        if time.monotonic() - last_trigger < cooldown:
                            continue
                        last_trigger = time.monotonic()
                        self.handle_utterance(mic, utt)
                        if self.wake is not None:
                            self.wake.reset()
                        self._last_strong_wake = None
                        self.listener.flush()
                        if getattr(self, "_paused_mic", None) is not None:
                            self._paused_mic.flush()
                            self._paused_listener.flush()
                        continue
                    tier = self.wake.process(frame)
                    if not tier:
                        continue
                    if time.monotonic() - last_trigger < cooldown:
                        continue
                    last_trigger = time.monotonic()
                    self.handle(mic, "wake", tier)
                    self.wake.reset()
            finally:
                self._active_mic = None
                playback_stop.set()
                if getattr(self, "echo", None) is not None:
                    self.echo.reference.stop()
                self._close_paused_mic()
                if self._early_lease is not None:
                    self._early_lease.release()

    def handle_utterance(self, mic: Microphone, utt, input_device: str = "laptop",
                         *, wake_cued: bool = False) -> None:
        """Act on an utterance that already contains its own command.

        The whole point of the STT path: "memoir play red by seedhe maut" is one
        breath, already captured, so there is nothing to record and no
        wake-beep round trip. Small.en only gates the wake phrase; the larger
        model reads the same audio before an action is selected. Only a bare
        trigger ("memoir") needs a follow-up recording.
        """
        handler_started = time.perf_counter()
        event = Event(
            trigger_source="stt",
            input_device=input_device,
            wake_tier=utt.matched_phrase,
            transcript=utt.transcript,
            trigger_transcript=utt.transcript,
            trigger_stt_model=getattr(self, "listener_stt", self.stt).model_name,
            trigger_stt_latency_ms=utt.latency_ms,
            trigger_confirmation_mode=getattr(utt, "confirmation_mode", ""),
            trigger_confirmation_latency_ms=getattr(utt, "confirmation_latency_ms", None),
            request_phase="trigger",
        )
        token = bind_event(event)
        try:
            if not wake_cued:
                self._duck_on_wake()
            self._open_timeline(event, utt, handler_started)
            event.wake_path = ("paused_mic" if input_device == "sounddrum"
                               else _text_attr(utt, "wake_path") or "final_text")
            command = utt.command
            command_audio = utt.audio
            retained_decode = None
            if (not command and getattr(utt, "confirmed_audio_handoff", False) is True
                    and not getattr(utt, "truncated", False)):
                # A wake-only fast transcript does not prove that the retained
                # PCM contains no command. Check it once before asking again.
                retained_decode = self.stt.transcribe(utt.audio)
                retained_wake, retained_command, _ = self.listener.locate_trigger(retained_decode[0])
                if retained_wake and retained_command:
                    command = retained_command
            if not command:
                controller = getattr(self, "global_assistant", None)
                if (controller is not None and controller.session.active
                        and controller.session.phase == "review"
                        and self._handle_global("", mic, event)):
                    return
                # Bare trigger: ask, then listen, exactly like the wake path.
                trigger_event = Event(ts=event.ts)
                event.trigger_audio_path = self.telemetry.save_audio(trigger_event, utt.audio)
                duck_to = self.cfg.get("ducking.duck_to_percent", 20)
                event.request_phase = "capture"
                with self._duck_music(event, duck_to):
                    event.feedback_kind = self._acknowledge(mic)
                    if getattr(utt, "completed_at", 0):
                        event.wake_to_ack_ms = int(
                            (time.perf_counter() - utt.completed_at) * 1000)
                    captured = self.endpointer.record(
                        mic, after_prompt=self.feedback.speaks_on_wake)
                    if captured["speech_started"]:
                        self.feedback.stopped()
                event.mark("follow_up_end")
                self._capture_result(event, captured)
                event.audio_duration_ms = captured["duration_ms"]
                event.audio_path = self.telemetry.save_audio(
                    event, captured["audio"])
                if not captured["speech_started"]:
                    event.outcome = "false_accept"
                    # Nothing was said after the wake: the music coming back up is the
                    # answer. A spoken "Sorry." here was mostly heard after false wakes.
                    mic.flush()
                    return
                follow_up, event.stt_latency_ms = self.stt.transcribe(captured["audio"])
                follow_up, captured, event.stt_latency_ms = self._past_repeated_wake(
                    mic, event, follow_up, captured, event.stt_latency_ms)
                event.mark("stt_done", overwrite=True)
                if not follow_up:
                    event.outcome = "false_accept"
                    return
                command_audio = captured["audio"]
                event.stt_model = self.stt.model_name
                event.command_stt_model = self.stt.model_name
                event.transcript = "{} | {}".format(utt.transcript, follow_up)
                command = follow_up
            else:
                event.capture_end_reason = getattr(utt, "end_reason", "")
                event.capture_truncated = getattr(utt, "truncated", False) is True
                event.audio_path = self.telemetry.save_audio(event, utt.audio)
                event.trigger_audio_path = event.audio_path
                event.audio_duration_ms = int(utt.audio.size * 1000 / 16000)
                event.request_phase = "transcribe"
                verified_transcript = getattr(utt, "confirmation_transcript", "")
                if retained_decode is not None:
                    event.transcript, event.stt_latency_ms = retained_decode
                    event.mark("stt_done")
                elif (getattr(utt, "confirmation_mode", "") == "second_stt"
                        and verified_transcript):
                    event.transcript = verified_transcript
                    event.stt_latency_ms = utt.confirmation_latency_ms
                    # The listener's second-STT verification already decoded it.
                    event.mark("stt_done", at=_perf_attr(utt, "trigger_done_at") or None)
                else:
                    # The trigger recognizer can invent a trailing command;
                    # if this decode says only Alexa, buffered follow-up is
                    # still part of the addressed interaction.
                    event.transcript, event.stt_latency_ms = self.stt.transcribe(utt.audio)
                    event.mark("stt_done")
                event.stt_model = self.stt.model_name
                event.command_stt_model = self.stt.model_name
                # Small.en has already established that this was addressed to
                # us. It must not also decide the action: on two real clips it
                # heard "Alexa, boss" where turbo heard "Alexa, pause".
                # If turbo keeps the wake phrase, remove it. If it drops the
                # phrase but hears an explicit command, parse that command.
                turbo_wake, turbo_command, _ = self.listener.locate_trigger(event.transcript)
                command = turbo_command if turbo_wake else event.transcript
                if not turbo_wake and getattr(utt, "mid_utterance", False):
                    # Over music the audio opens with lyrics; without a wake
                    # word to anchor on, the whole transcript is not a command.
                    command = utt.command
                if turbo_wake and not turbo_command:
                    # The fast gate sometimes invents a trailing word from a
                    # bare "Alexa" (observed: "Alex up"). Trust the command
                    # model's wake-only result and capture the next utterance
                    # after the normal acknowledgement, not a "Repeat" retry.
                    event.request_phase = "capture"
                    with self._duck_music(event, self.cfg.get("ducking.duck_to_percent", 20)):
                        event.feedback_kind = self._acknowledge(mic)
                        if getattr(utt, "completed_at", 0):
                            event.wake_to_ack_ms = int(
                                (time.perf_counter() - utt.completed_at) * 1000)
                        captured = self.endpointer.record(
                            mic, after_prompt=self.feedback.speaks_on_wake)
                        if captured["speech_started"]:
                            self.feedback.stopped()
                    extra = Event(ts=event.ts, event_id=event.event_id + "-follow-up")
                    event.follow_up_audio_path = self.telemetry.save_audio(extra, captured["audio"])
                    if not captured["speech_started"]:
                        event.outcome = "false_accept"
                        # Nothing was said after the wake: the music coming back up is the
                        # answer. A spoken "Sorry." here was mostly heard after false wakes.
                        return
                    follow_up, latency = self.stt.transcribe(captured["audio"])
                    self._capture_result(event, captured)
                    follow_up, captured, latency = self._past_repeated_wake(
                        mic, event, follow_up, captured, latency)
                    event.mark("stt_done", overwrite=True)
                    if not follow_up:
                        event.outcome = "false_accept"
                        return
                    event.stt_latency_ms = (event.stt_latency_ms or 0) + latency
                    event.transcript = "{} | {}".format(event.transcript, follow_up)
                    command = follow_up
                    command_audio = captured["audio"]
                elif not turbo_wake and parse(command).action is Action.UNKNOWN:
                    # Small.en already validated the wake at the start of this
                    # very audio. Turbo may render only that first token as a
                    # different word (observed: "elixir pause"). Discard one
                    # leading token only when the remainder is a full rule-
                    # backed command; a fuzzy guess such as "cause" is unsafe.
                    _, separator, tail = command.partition(" ")
                    if separator:
                        tail_intent = parse(tail)
                        if tail_intent.action is not Action.UNKNOWN and tail_intent.source == "rule":
                            command = tail

            if self._handle_conversation(command, mic, event, command_audio):
                return
            if self._handle_global(command, mic, event):
                return
            intent = self._parse_intent(command, event)
            event.request_phase = "intent"
            event.intent_action = intent.action.value
            event.intent_query = intent.query
            event.intent_volume_after_play = intent.volume_after_play
            event.intent_library_only = intent.library_only
            event.intent_source = intent.source
            event.matched_rule = intent.matched_rule

            if intent.action is Action.UNKNOWN:
                if getattr(utt, "mid_utterance", False):
                    # The wake word came in the middle of other speech and no
                    # command followed: most likely conversation, not us.
                    # Stay quiet instead of saying "Repeat, please".
                    event.outcome = "false_accept"
                    log.info("Mid-sentence wake word without a command; ignored")
                    return
                self._retry_unknown(mic, event)
                return

            self._execute(intent, event, command_audio=command_audio, mic=mic)
            event.request_phase = "complete"

        except MusicConfirmationRequired as exc:
            self._handle_music_confirmation(mic, event, exc)
        except AmbiguousMatch as exc:
            self._handle_ambiguous_track(mic, event, exc)
        except SpotifyQuotaExceeded:
            event.outcome = "spotify_quota_exceeded"
            event.error = "SpotifyQuotaExceeded"
            event.feedback_text = SPOTIFY_QUOTA_MESSAGE
            event.feedback_kind = self._say_feedback(mic, event.feedback_text, event)
        except SpotifyRequestDeferred as exc:
            event.outcome = "request_deferred"
            event.error = str(exc)
            event.feedback_text = "Spotify requests are busy. Please try again shortly."
            event.feedback_kind = self._say_feedback(mic, event.feedback_text, event)
            log.info("Spotify request deferred locally: %s", exc)
        except SpotifyRateLimited as exc:
            event.outcome = "error"
            event.error = str(exc)
            event.feedback_text = RATE_LIMITED_MESSAGE
            event.feedback_kind = self._say_feedback(mic, event.feedback_text, event)
            log.error("%s", exc)
        except SpotifyAuthError as exc:
            event.outcome = "error"
            event.error = str(exc)
            event.feedback_kind = self._say_feedback(mic, "Spotify needs you to sign in.", event)
            log.error("%s", exc)
        except Exception as exc:
            event.outcome = "error"
            event.error = "{}: {}".format(type(exc).__name__, exc)
            event.feedback_kind = self._say_feedback(mic, SORRY, event)
            log.exception("Command failed")
        finally:
            try:
                mic.flush()
            finally:
                try:
                    lease = getattr(self, "_early_lease", None)
                    if lease is not None:
                        lease.release()
                finally:
                    try:
                        event.token_age_days = self.spotify.token_age_days()
                        event.handler_latency_ms = int((time.perf_counter() - handler_started) * 1000)
                        self._close_timeline(event)
                        self.telemetry.write(event)
                    finally:
                        unbind_event(token)

    def shutdown(self, *_) -> None:
        log.info("Shutting down.")
        self._stop.set()
        controller = getattr(self, "global_assistant", None)
        if controller is not None:
            controller.close()
        if self._early_lease is not None:
            self._early_lease.release()
        if self._speech_lease is not None:
            self._speech_lease.release()
        if self.wake is not None:
            self.wake.close()


def _record_crashes() -> None:
    """pythonw has no console: an uncaught exception used to vanish, leaving
    only "exit code 1" in Task Scheduler (seen after a reboot). Keep them."""
    crash_log = ROOT / "assistant" / "logs" / "crash.log"
    handler = logging.FileHandler(crash_log, encoding="utf-8")
    handler.setFormatter(EventFormatter(
        "%(asctime)s %(levelname)s %(name)s: %(event_tag)s%(message)s"))
    handler.addFilter(EventIdFilter())
    handler.setLevel(logging.ERROR)
    logging.getLogger().addHandler(handler)
    log.addHandler(handler)

    def hook(kind, value, tb):
        log.critical("Uncaught exception", exc_info=(kind, value, tb))
    sys.excepthook = hook
    threading.excepthook = lambda args: log.critical(
        "Uncaught exception in thread %s", args.thread and args.thread.name,
        exc_info=(args.exc_type, args.exc_value, args.exc_traceback))


def _run_instance() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _record_crashes()
    try:
        assistant = Assistant()
    except (WakeWordError, SpotifyAuthError, RuntimeError) as exc:
        # Startup failures must be loud. A silently dead assistant is
        # indistinguishable from one that simply isn't hearing you.
        log.error("Startup failed:\n%s", exc)
        sys.exit(1)
    except Exception:
        # Typically MemoryError / CUDA OOM while the models load.
        health.log_snapshot("fatal_startup")
        raise

    signal.signal(signal.SIGINT, assistant.shutdown)
    try:
        monitor = assistant.start_health_monitor()
    except Exception:  # the heartbeat is a diagnostic; it must never stop the assistant
        monitor = None
        log.warning("Health heartbeat unavailable", exc_info=True)
    try:
        assistant.run()
    except KeyboardInterrupt:
        pass
    except Exception:
        health.log_snapshot("fatal", assistant.health_probes())
        log.critical("Assistant stopped on an error", exc_info=True)
        raise
    finally:
        if monitor is not None:
            monitor.stop()
        assistant.shutdown()


def main() -> None:
    try:
        with single_instance(ROOT):
            _run_instance()
    except AlreadyRunning as exc:
        logging.getLogger("memoir").warning("%s", exc)


if __name__ == "__main__":
    main()
