"""Sol round-2 probes: strict consumption, observation, timing and dispatch."""
import io
import json
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from assistant.config import Config
from assistant.conversation_contracts import Decision, Step, Transcript
from assistant.core.authority import NOT_DONE, REFUSAL, license_request, licenses
from assistant.core.capture import CaptureResult, CaptureRuntime, EndReason
from assistant.core.engine import CoreRuntime
from assistant.core.executor import Executor, Receipt
from assistant.core.fallback import admit
from assistant.core.provider import BrainProvider
from assistant.core.transcription import NoWakeBoundary, TimedTranscript, Word, trim_to_wake
from assistant.core import volume
from assistant import local_media
from assistant.resolution import Candidate
from assistant.spotify import Spotify, SpotifyRequestDeferred


@pytest.mark.parametrize("text", [
    "play Halo & pause", "play Halo or pause", "play Halo + pause", "play Halo / pause",
    "play Halo; pause", "play Halo: pause", "play Halo, pause",
    "play Halo then pause", "play Halo plus pause", "play Halo also pause",
    "play Halo with pause", "play Halo followed by pause", "play Halo before pause",
    "play Halo!!", "play Halo?!", "play Halo...", "play X-Y",
])
def test_f1_fallback_rejects_connectors_and_nonterminal_punctuation(text):
    assert admit(text) is None


@pytest.mark.parametrize("ending", ["", ".", "!", "?"])
def test_f1_one_terminal_mark_and_internal_apostrophe_are_not_discarded_content(ending):
    admitted = admit("play O'Clock" + ending)
    assert admitted is not None
    assert admitted.steps[0].arguments["title"].casefold() == "o'clock"


def test_f1_complete_consumption_does_not_strip_object_punctuation():
    assert not licenses(Step("s", "play_song", {"title": "OClock", "artist": ""}),
                        "play O'Clock", brain=False)
    assert admit("play O’Clock") is None  # The retained parser removes this punctuation.


def run_decision(text, decision, *, clock=None, executor=None):
    effects, spoken = [], []
    adapter = executor or SimpleNamespace(ALLOWED=Executor.ALLOWED, execute=lambda s, op, **kwargs:
        effects.append(s) or Receipt("observed", "Done."))
    runtime = CoreRuntime(SimpleNamespace(timeout=18, decide=lambda *a, **k: decision),
        adapter, SimpleNamespace(say=lambda t, **kwargs: spoken.append(t)),
        **({"clock": clock} if clock else {}))
    return runtime.process(Transcript(text, "t")), effects, spoken


@pytest.mark.parametrize("title", ["No Pause", "Actually Pause", "Sorry Pause", "Cancel", "Never Mind", "Wait"])
def test_f2_marker_words_are_only_literal_play_objects(title):
    text = "play " + title
    song = Step("s", "play_song", {"title": title, "artist": ""})
    for brain in (True, False):
        assert license_request(text, brain=brain).capability == "play_song"
        assert licenses(song, text, brain=brain)
        assert not licenses(Step("s", "pause", {}), text, brain=brain)
        assert not licenses(Step("s", "resume", {}), text, brain=brain)
    effects = []
    adapter = SimpleNamespace(ALLOWED=Executor.ALLOWED, exact_match=lambda *_: True,
        execute=lambda step, op, **kwargs: effects.append(step) or Receipt("observed", "Done."))
    row, _, _ = run_decision(text, Decision("plan", steps=(song,), evidence_quotes=(text,)), executor=adapter)
    assert effects == [song] and row["receipts"][0]["state"] == "observed"


@pytest.mark.parametrize("text", [
    "play Halo, actually no play Hello", "play Halo, actually play Hello",
    "actually play Hello", "no play Hello", "sorry play Hello", "instead play Hello",
    "I mean play Hello", "cancel", "never mind", "wait",
])
def test_f2_corrections_never_create_another_instruction(text):
    assert license_request(text) is None and admit(text) is None


@pytest.mark.parametrize("kind,text", [
    ("answer", "I resumed Spotify."), ("answer", "I lowered the volume to 40."),
    ("clarify", "Paused. Which song next?"),
])
def test_f6_answer_and_clarification_effect_claims_never_reach_speech(kind, text):
    source = "What happened?"
    decision = Decision(kind, text=text, evidence_quotes=(source,))
    row, effects, spoken = run_decision(source, decision)
    assert not effects and not row["receipts"] and spoken == ["I didn't do anything."]


@pytest.mark.parametrize("verb", [
    "paused", "resumed", "played", "playing", "started", "stopped", "skipped", "skipping",
    "lowered", "raised", "turned", "set", "changed", "muted", "unmuted", "increased", "decreased",
])
def test_f6_generic_effect_forms_are_guarded_in_clarifications(verb):
    source = "What happened?"
    decision = Decision("clarify", text="I " + verb + ". Which song next?", evidence_quotes=(source,))
    _, _, spoken = run_decision(source, decision)
    assert spoken == [NOT_DONE]


@pytest.mark.parametrize("question", ["Which song? Which artist?", "Which song? " + "Which artist? " * 30])
def test_f6_clarification_preserves_multiple_sentences_with_a_200_character_cap(question):
    source = "Tell me about a board."
    decision = Decision("clarify", text=question, evidence_quotes=(source,))
    _, _, spoken = run_decision(source, decision)
    assert spoken == [" ".join(question.split())[:200]]


def test_f4_earlier_aliased_name_refuses_without_sending_or_logging_private_speech(monkeypatch):
    sent, rows = [], []
    monkeypatch.setenv("CORE_R2_KEY", "synthetic")
    def request(req, **kwargs):
        sent.append(req.data)
        return io.BytesIO(b"{}")
    provider = BrainProvider("https://example.invalid/v1", "fake", "CORE_R2_KEY",
                             opener=request, clock=lambda: 100.0)
    words = (Word("Tell", 0, 0.1), Word("Alexis", 0.1, 0.3),
             Word("confidential_secret_8675309", 0.3, 1.6),
             Word("Alexa", 1.6, 1.8), Word("pause", 1.8, 2.0))
    capture = CaptureRuntime(Config({"listener": {"aliases": {"alexis": "alexa"}}}))
    capture.turn_id, capture.wake_score = "t", 0.9
    capture.stt = SimpleNamespace(transcribe=lambda pcm: TimedTranscript(words, 1))
    result = CaptureResult("t", np.ones(48000, dtype=np.int16), 1800,
                           EndReason.ENDPOINT, True, False, 0.9, 100, 100, audio_started_at=98.2)
    capture.next_capture, capture.reset = lambda **kwargs: result, lambda: None
    runtime = CoreRuntime(provider, Executor(), SimpleNamespace(tick=lambda: None, say=lambda *a, **k: None),
                          log=rows.append, clock=lambda: 100.0)
    runtime.step(capture)
    assert sent == [] and len(rows) == 1 and rows[0]["transcript"] == ""
    assert rows[0]["wake_boundary"]["reason"] == "earlier_wake_token"
    assert set(rows[0]["wake_boundary"]) == {"reason", "prefix_length"}
    encoded = json.dumps(rows)
    assert "Alexis" not in encoded and "confidential_secret_8675309" not in encoded


@pytest.mark.parametrize("words,reason", [
    ((Word("Alexa", 0, 0.2), Word("pause", 0.2, 0.3)), "uncorroborated_wake_boundary"),
    ((Word("Alexa", 1.0, 1.4), Word("Alexis", 1.4, 1.8)), "ambiguous_wake_boundary"),
])
def test_f4_requires_exactly_one_timing_corroborated_token(words, reason):
    with pytest.raises(NoWakeBoundary) as error:
        trim_to_wake(TimedTranscript(words, 1), {"alexis": "alexa"},
                     wake_at=100.0, audio_started_at=98.2)
    assert error.value.reason == reason


def test_f3_native_play_then_metadata_loss_never_pauses_or_opens_a_uri(monkeypatch):
    events = []
    class Session:
        dispatched = False
        def get_playback_info(self):
            return SimpleNamespace(playback_status=local_media.PLAYING if self.dispatched else local_media.PAUSED)
        async def try_get_media_properties_async(self):
            return None if self.dispatched else SimpleNamespace(title="Halo", artist="Beyonce", album_title="")
        def get_timeline_properties(self):
            return None
        async def try_play_async(self):
            events.append("native_play_dispatched")
            self.dispatched = True
            return True
        async def try_pause_async(self):
            events.append("native_pause_dispatched")
            return True
    session = Session()
    async def get_session():
        return session
    monkeypatch.setattr(local_media, "_session", get_session)
    monkeypatch.setattr(local_media, "snapshot", lambda: local_media.MediaSnapshot("Halo", "Beyonce", False))
    monkeypatch.setattr(local_media, "InvisibleSpotify", lambda: nullcontext(SimpleNamespace(return_focus=lambda: None)))
    monkeypatch.setattr("assistant.core.playback.open_external_target", lambda uri: events.append("desktop_start"))
    spotify = Spotify.__new__(Spotify)
    spotify._cfg, spotify.device = Config({}), lambda: "pinned"
    spotify.command_target = lambda: nullcontext()
    spotify.resolve_track = lambda query: Candidate("spotify:track:halo", "Halo", "Beyonce")
    def deferred(**kwargs):
        raise SpotifyRequestDeferred("proven local budget refusal")
    import socket
    spotify.sp = SimpleNamespace(start_playback=deferred, devices=lambda: {"devices": [
        {"id": "pinned", "name": socket.gethostname(), "type": "Computer"}]})
    from assistant.core.playback import guarded_playback
    from assistant.spotify import PlaybackVerificationError
    with guarded_playback(spotify, lambda: 5):
        with pytest.raises(PlaybackVerificationError):
            spotify._play_uri_locally(Candidate("spotify:track:halo", "Halo", "Beyonce"))
    assert events == ["native_play_dispatched"]


def test_f7_endpoint_discovery_past_deadline_never_writes_volume(monkeypatch):
    now, writes = [0.0], []
    endpoint = SimpleNamespace(SetMasterVolumeLevelScalar=lambda *a: writes.append(a),
                               GetMasterVolumeLevelScalar=lambda: 0.4)
    def discover():
        now[0] = 20
        return endpoint
    monkeypatch.setattr(volume, "_endpoint", discover)
    text = "set volume to 40"
    decision = Decision("plan", steps=(Step("s", "set_volume", {"level": 40}),), evidence_quotes=(text,))
    row, _, _ = run_decision(text, decision, clock=lambda: now[0], executor=Executor(volume=volume))
    assert not writes and row["receipts"][0]["state"] == "failed"
    assert row["receipts"][0]["detail"] == "That turn expired."
