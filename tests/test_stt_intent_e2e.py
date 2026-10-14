"""End-to-end STT -> intent, using Piper to synthesize the spoken commands.

This exercises the real decoding path against real audio without needing anyone
to sit at the microphone, and it catches the failure mode unit tests cannot:
phrasings that parse fine as text but that the ASR mangles.

Synthetic TTS is not a substitute for testing in the actual room -- it has no
speaker bleed, no distance, no room reverb -- so treat a pass here as necessary,
not sufficient. The archived WAVs from real use are the real evaluation set.

Skipped automatically if Piper, its voice, or Moonshine are unavailable.
"""
import io
import wave

import numpy as np
import pytest

from assistant.intent import Action, parse

VOICE = "assistant/models/en_US-lessac-medium.onnx"

pytest.importorskip("piper", reason="piper-tts not installed")


@pytest.fixture(scope="module")
def voice():
    from pathlib import Path

    from piper.voice import PiperVoice

    if not Path(VOICE).exists():
        pytest.skip("Piper voice not downloaded")
    return PiperVoice.load(VOICE)


@pytest.fixture(scope="module")
def stt():
    """The model actually configured for production.

    Previously hardcoded to Moonshine, so the suite kept passing while
    the daemon ran faster-whisper -- the tests validated a model that
    was no longer in use.
    """
    from assistant.config import load
    from assistant.stt import Transcriber

    return Transcriber(load().get("stt.model", "moonshine/base"))


def speak(voice, text: str) -> np.ndarray:
    """Synthesize text and return int16 PCM resampled to 16 kHz mono."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        voice.synthesize_wav(text, wf)
    buf.seek(0)
    with wave.open(buf, "rb") as wf:
        rate = wf.getframerate()
        pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)

    if rate != 16000:
        n = int(len(pcm) * 16000 / rate)
        pcm = np.interp(
            np.linspace(0, len(pcm) - 1, n), np.arange(len(pcm)), pcm
        ).astype(np.int16)
    return pcm


@pytest.mark.parametrize(
    "spoken,expected_action",
    [
        ("Hey Memoir, pause.", Action.PAUSE),
        ("Hey Memoir, skip.", Action.NEXT),
        ("Hey Memoir, go back.", Action.PREV),
        ("Hey Memoir, play music.", Action.RESUME),
        ("Hey Memoir, turn it up.", Action.VOLUME),
        ("Hey Memoir, set volume to forty.", Action.VOLUME),
        ("Hey Memoir, play Bohemian Rhapsody.", Action.PLAY_TRACK),
        ("Hey Memoir, play my focus playlist.", Action.PLAY_PLAYLIST),
    ],
)
def test_spoken_command_reaches_the_right_intent(voice, stt, spoken, expected_action):
    """Report first-attempt failure rather than hiding it behind retries."""
    transcript, _ = stt.transcribe(speak(voice, spoken))
    assert parse(transcript).action is expected_action, (
        "spoken {!r} expected {}; heard {!r}".format(spoken, expected_action, transcript)
    )


def test_command_latency_is_within_budget(voice, stt):
    """Guards the persistent-model path.

    Moonshine's module-level transcribe() re-resolves the model from HuggingFace
    on every call, costing ~3.2 s per command instead of ~50 ms. Latency is
    asserted here, on a warmed model, rather than inside the correctness tests
    where a cold first call makes it flaky.
    """
    pcm = speak(voice, "Hey Memoir, play Bohemian Rhapsody.")
    stt.transcribe(pcm)  # warm
    latencies = [stt.transcribe(pcm)[1] for _ in range(3)]
    assert min(latencies) < 500, "STT latency regressed: {} ms".format(latencies)
