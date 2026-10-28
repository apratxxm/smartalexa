"""Sol F1-F7 probes, exercised through real admission and adapter boundaries."""
import io
import json
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from assistant.config import Config
from assistant.conversation_contracts import Decision, Step, Transcript
from assistant.core.authority import REFUSAL, NOT_DONE, licenses, license_request
from assistant.core.capture import CaptureRuntime, CaptureResult, EndReason
from assistant.core.engine import CoreRuntime
from assistant.core.executor import Executor
from assistant.core.fallback import admit
from assistant.core.provider import BrainProvider, ProviderError
from assistant.core.transcription import TimestampedTranscriber, Word
from assistant.resolution import Candidate
from assistant.spotify import Spotify, _RateLimitGuard
from assistant import local_media


def plan(text, capability="play_song", arguments=None):
    args = {"title": "Halo", "artist": ""} if arguments is None else arguments
    return Decision("plan", steps=(Step("s", capability, args),), evidence_quotes=(text,))


def engine(decision=None, *, error=None, spotify=None, clock=None):
    def decide(*args, **kwargs):
        if error:
            raise error
        return decision
    provider = SimpleNamespace(timeout=18, decide=decide)
    spoken, effects = [], []
    if spotify is None:
        spotify = SimpleNamespace(resolve_track=lambda q: q,
            play_candidate=lambda q: effects.append(q) or "Playing Halo.",
            pause=lambda: effects.append("pause"))
    runtime = CoreRuntime(provider, Executor(spotify), SimpleNamespace(
        say=lambda text, **kwargs: spoken.append(text), tick=lambda: None),
        **({"clock": clock} if clock else {}))
    return runtime, spoken, effects


@pytest.mark.parametrize("text", [
    "play Halo, no, cancel that", "play Halo. Stop.", "play Halo; pause",
    "play Halo if I ask", "play Halo unless I cancel", "play Halo, wait",
])
def test_f1_fallback_never_executes_a_discarded_prefix(text):
    assert admit(text) is None
    runtime, spoken, effects = engine(error=ProviderError("unavailable"))
    row = runtime.process(Transcript(text, "t"))
    assert effects == [] and row["receipts"] == []
    assert spoken == [REFUSAL]


@pytest.mark.parametrize("text", [
    "Do not pause the music.", "What does pause mean?", "He said pause the music.",
    '"pause the music"', "pause if I ask", "pause. Resume.",
])
def test_f2_model_evidence_does_not_grant_authority(text):
    runtime, spoken, effects = engine(plan(text, "pause", {}))
    row = runtime.process(Transcript(text, "t"))
    assert not effects and not row["receipts"]
    assert spoken == [REFUSAL]


@pytest.mark.parametrize("text,cap,args", [
    ("play Halo", "play_song", {"title": "Other", "artist": ""}),
    ("play Halo by Beyonce", "play_song", {"title": "Halo", "artist": ""}),
    ("set volume to 40", "set_volume", {"level": 80}),
    ("turn it down", "change_volume", {"direction": "up", "amount": 0, "unit": "points"}),
])
def test_f2_arguments_must_match_the_licensed_clause(text, cap, args):
    assert not licenses(Step("s", cap, args), text)


@pytest.mark.parametrize("text,cap", [
    ("play Halo", "play_song"), ("put on Halo", "play_song"), ("start Halo", "play_song"),
    ("pause", "pause"), ("stop the music", "pause"), ("hold the music", "pause"),
    ("resume", "resume"), ("continue music", "resume"), ("unpause", "resume"),
    ("next", "next_track"), ("skip this one", "next_track"),
    ("set Windows volume to forty", "set_volume"), ("turn it down", "change_volume"),
    ("make it louder", "change_volume"), ("quieter", "change_volume"),
])
def test_authority_generous_capability_lexicon(text, cap):
    assert license_request(text).capability == cap


def test_f3_lost_start_response_never_triggers_desktop_replay(monkeypatch, tmp_path):
    monkeypatch.setattr("assistant.core.executor.guarded_playback", lambda *args: nullcontext())
    events = []
    spotify = Spotify.__new__(Spotify)
    spotify._cfg = Config({})
    spotify.device = lambda: "pinned"
    spotify.command_target = lambda: nullcontext()
    spotify.resolve_track = lambda query: Candidate("spotify:track:halo", "Halo")
    def start(**kwargs):
        events.append("web_start_dispatched")
        raise OSError("response lost")
    spotify.sp = _RateLimitGuard(SimpleNamespace(start_playback=start), path=tmp_path / "backoff.json")
    spotify._play_uri_locally = Mock(side_effect=lambda candidate: events.append("desktop_start"))
    result = Executor(spotify).execute(Step("s", "play_song", {"title": "Halo", "artist": ""}))
    assert events == ["web_start_dispatched"]
    assert result.state == "unknown"
    spotify._play_uri_locally.assert_not_called()


def capture_for_words(words, *, aliases=None):
    capture = CaptureRuntime(Config({"listener": {"aliases": aliases or {}}}))
    model_calls = []
    def decode(audio, **kwargs):
        model_calls.append((audio, kwargs))
        return [SimpleNamespace(words=[SimpleNamespace(word=w, start=i / 5, end=(i + 1) / 5)
                                      for i, w in enumerate(words)])], None
    capture.stt = TimestampedTranscriber.__new__(TimestampedTranscriber)
    capture.stt._backend = SimpleNamespace(_model=SimpleNamespace(transcribe=decode))
    capture.turn_id, capture.wake_score = "t", 0.9
    result = CaptureResult("t", np.ones(32000, dtype=np.int16), 1000,
                           EndReason.ENDPOINT, True, False, 0.9, 100, 100, audio_started_at=99)
    capture.next_capture = lambda **kwargs: result
    capture.reset = lambda: None
    return capture, model_calls


@pytest.mark.parametrize("wake,aliases", [("Alexa,", {}), ("LEXA!", {"lexa": "alexa"})])
def test_f4_private_prefix_never_reaches_provider_payload_or_turn_log(monkeypatch, wake, aliases):
    sent, rows = [], []
    monkeypatch.setenv("CORE_PROBE_KEY", "synthetic")
    def request(req, **kwargs):
        sent.append(json.loads(req.data))
        wire = {"kind": "plan", "steps": [{"id": "s", "capability": "play_song",
                "arguments": {"title": "Halo", "artist": ""}, "after": []}], "evidence_refs": ["c0"]}
        return io.BytesIO(json.dumps({"choices": [{"message": {"tool_calls": [
            {"function": {"name": "decision", "arguments": json.dumps(wire)}}]}}]}).encode())
    provider = BrainProvider("https://example.invalid/v1", "fake", "CORE_PROBE_KEY", opener=request, clock=lambda: 100.0)
    runtime = CoreRuntime(provider, Executor(SimpleNamespace(
        resolve_track=lambda query: query, play_candidate=lambda candidate: "Playing Halo.")),
        SimpleNamespace(tick=lambda: None, say=lambda *a, **k: None), log=rows.append, clock=lambda: 100.0)
    capture, calls = capture_for_words(["private", "bank", "details", wake, "play", "Halo"], aliases=aliases)
    runtime.step(capture)
    assert len(sent) == len(rows) == len(calls) == 1
    encoded = json.dumps(sent) + json.dumps(rows)
    assert "private" not in encoded and "bank" not in encoded and "details" not in encoded
    assert rows[0]["transcript"] == "play Halo" and rows[0]["trimmed_prefix_length"] > 0
    assert calls[0][1]["word_timestamps"] is True and calls[0][1]["without_timestamps"] is False
    assert json.loads(sent[0]["messages"][1]["content"])["transcript"] == "play Halo"


def test_f4_missing_wake_boundary_sends_nothing():
    runtime, _, effects = engine(plan("play Halo"))
    sent = []
    runtime.provider.decide = lambda *a, **k: sent.append(a) or plan("play Halo")
    capture, calls = capture_for_words(["private", "bank", "details", "play", "Halo"])
    row = runtime.step(capture)
    assert not sent and not effects and row["transcript"] == ""
    assert row["outcome"] == "capture_rejected: no wake boundary"
    assert len(calls) == 1
    assert "private" not in json.dumps(row)


@pytest.mark.parametrize("use_core_guards", [True, False])
def test_f5_missing_next_metadata_is_uncertain_on_one_pinned_session(monkeypatch, use_core_guards):
    if not use_core_guards:
        monkeypatch.setattr("assistant.core.executor.guarded_playback", lambda *args: nullcontext())
    sessions, mutations = [], []
    class Session:
        changed = False
        def get_playback_info(self):
            return SimpleNamespace(playback_status=local_media.PLAYING)
        async def try_get_media_properties_async(self):
            return None if self.changed else SimpleNamespace(title="Halo", artist="Beyonce", album_title="")
        def get_timeline_properties(self):
            return None
        async def try_skip_next_async(self):
            self.changed = True
            mutations.append(self)
            return True
    async def session():
        current = Session()
        sessions.append(current)
        return current
    monkeypatch.setattr(local_media, "_session", session)
    spotify = Spotify.__new__(Spotify)
    spotify._cfg = Config({})
    spotify.sp = SimpleNamespace(next_track=lambda **kwargs: pytest.fail("must not replay through Web API"))
    result = Executor(spotify).execute(Step("s", "next_track", {}))
    assert result.state == "unknown"
    # One preflight status lookup plus one pinned transport session.
    assert len(sessions) == 2 and mutations == [sessions[1]]


@pytest.mark.parametrize("kind,text", [("answer", "Paused."), ("clarify", "Which song?")])
def test_f6_action_acknowledgements_are_application_owned(kind, text):
    decision = Decision(kind, text=text, evidence_quotes=("pause",))
    runtime, spoken, effects = engine(decision)
    row = runtime.process(Transcript("pause", "t"))
    assert not effects and not row["receipts"] and spoken == [NOT_DONE]


@pytest.mark.parametrize("answer", ["Paused.", "Playing Halo.", "Skipped.", "Volume set to 40.",
                                   "The volume changed.", "Volume turned down."])
def test_f6_unobserved_answer_effect_claim_is_refused(answer):
    runtime, spoken, _ = engine(Decision("answer", text=answer, evidence_quotes=("What happened?",)))
    row = runtime.process(Transcript("What happened?", "t"))
    assert not row["receipts"] and spoken == [NOT_DONE]


def test_f7_resolution_finishing_after_deadline_never_dispatches():
    now, effects = [0.0], []
    def resolve(query):
        now[0] = 20.0
        return Candidate("spotify:track:halo", "Halo")
    runtime, _, _ = engine(plan("play Halo"), clock=lambda: now[0],
        spotify=SimpleNamespace(resolve_track=resolve, play_candidate=effects.append))
    row = runtime.process(Transcript("play Halo", "t"), deadline=18)
    assert not effects and row["receipts"][0]["state"] == "failed"
    assert row["receipts"][0]["detail"] == "That turn expired."


def test_f7_guard_rechecks_device_waits_and_restores_owned_hooks(tmp_path):
    from assistant.core.playback import guarded_playback, TurnExpired
    now, dispatches = [0.0], []
    send = lambda request, **kwargs: dispatches.append(kwargs)
    session = SimpleNamespace(send=send)
    raw = SimpleNamespace(_session=session, start_playback=lambda **kwargs: dispatches.append(kwargs))
    spotify = Spotify.__new__(Spotify)
    spotify.sp = _RateLimitGuard(raw, path=tmp_path / "backoff.json")
    def device():
        now[0] = 20
        return "pinned"
    spotify.device = device
    with guarded_playback(spotify, lambda: 18 - now[0]):
        with pytest.raises(TurnExpired):
            spotify._call(spotify.sp.start_playback)
    assert dispatches == [] and session.send is send and "_call" not in vars(spotify)


def test_f7_http_waits_use_remaining_budget_and_restore_session(tmp_path):
    from assistant.core.playback import guarded_playback
    waits = []
    send = lambda request, **kwargs: waits.append(kwargs["timeout"])
    session = SimpleNamespace(send=send)
    spotify = Spotify.__new__(Spotify)
    spotify.sp = _RateLimitGuard(SimpleNamespace(_session=session), path=tmp_path / "backoff.json")
    with guarded_playback(spotify, lambda: 0.25):
        session.send(object(), timeout=(10, 20))
    assert waits == [(0.25, 0.25)] and session.send is send
