"""Benchmark every pretrained wake word against YOUR voice, and prove the audio.

Each recording is scored against all four pretrained models AND transcribed with
Moonshine. The transcript is the control: if Moonshine reads your words back
correctly, the capture path is good and any wake-word failure is the model
mismatching your voice, not bad audio. Without that control, "score is zero" is
ambiguous -- which is what wasted several rounds of debugging here.

Level is reported as the loudest 32 ms frame, not the whole-window average: a
short phrase inside a 3 second window averages down to near silence and looks
like a capture failure when it isn't.

Recordings are kept in assistant/logs/audio/bench/ -- they are the positive
examples if we end up training a custom model.

    python scripts/wake_bench.py
    python scripts/wake_bench.py --repeats 5 --phrase hey_jarvis
"""
from __future__ import annotations

import argparse
import logging
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from assistant.audio import Microphone  # noqa: E402
from assistant.config import load  # noqa: E402

PHRASES = {
    "hey_jarvis": "Hey Jarvis",
    "alexa": "Alexa",
    "hey_mycroft": "Hey Mycroft",
    "hey_rhasspy": "Hey Rhasspy",
}
OUT_DIR = ROOT / "assistant" / "logs" / "audio" / "bench"
TRIGGER = 0.5
CR = chr(13)   # carriage return, for in-place progress lines


def frame_peak_db(pcm: np.ndarray) -> float:
    """Loudest 32 ms frame, in dBFS. The honest measure for a short phrase."""
    best = 1.0
    for i in range(0, max(len(pcm) - 512, 1), 512):
        best = max(best, float(np.sqrt((pcm[i:i + 512].astype(np.float64) ** 2).mean())))
    return 20 * np.log10(best / 32768.0)


def listen_window(mic, seconds: float, label: str) -> np.ndarray:
    """One open recording window. No countdown, no calibration.

    Timing the user against a countdown proved unreliable -- speech landed
    outside the window and looked like a capture failure. A longer open window
    with several repeats removes the timing variable entirely.
    """
    print("")
    print('  Window open {:.0f}s -- say "{}" three times, pausing between.'
          .format(seconds, label))
    frames = []
    total = int(seconds * 16000 / 512)
    step = max(total // 20, 1)
    for i in range(total):
        frames.append(mic.read())
        if i % step == 0:
            print(CR + "    listening " + "." * (i // step + 1), end="", flush=True)
    print(CR + "    captured" + " " * 32)
    return np.concatenate(frames)


def score_all(models: dict, pcm: np.ndarray) -> dict:
    out = {}
    for name, model in models.items():
        model.reset()
        best = 0.0
        for i in range(0, len(pcm) - 1280, 1280):
            best = max(best, float(model.predict(pcm[i:i + 1280])[name]))
        out[name] = best
    return out


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--phrase", choices=list(PHRASES),
                    help="test only this one")
    args = ap.parse_args()

    phrases = {args.phrase: PHRASES[args.phrase]} if args.phrase else PHRASES

    print("Loading models (wake + STT)...")
    from openwakeword.model import Model

    from assistant.stt import Transcriber

    models = {n: Model(wakeword_models=[n], inference_framework="onnx")
              for n in PHRASES}
    stt = Transcriber("moonshine/base")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = load()
    results = {n: [] for n in phrases}

    print("\n" + "=" * 70)
    print("Speak normally, from where you'd actually be in the room.")
    print("A score above {:.2f} would trigger.".format(TRIGGER))
    print("The TRANSCRIPT line is the control: if it reads back what you said,")
    print("the audio is fine and a zero score is the model, not the mic.")
    print("=" * 70)

    with Microphone(
        sample_rate=16000,
        frame_length=512,
        device=cfg.get("audio.input_device") or None,
        capture_rate=cfg.get("audio.capture_rate"),
        exclusive=cfg.get("audio.exclusive_mode", False),
    ) as mic:
        for name, phrase in phrases.items():
            for attempt in range(args.repeats):
                try:
                    pcm = listen_window(mic, args.seconds, phrase)
                except Exception as exc:
                    # Toggling a Windows sound setting resets the device and
                    # kills the stream mid-run; don't lose the whole session.
                    print("    audio stream error ({}); skipping".format(
                        str(exc)[:60]))
                    continue

                scores = score_all(models, pcm)
                results[name].append(scores[name])
                transcript, _ = stt.transcribe(pcm)

                path = OUT_DIR / "{}_{}.wav".format(name, attempt)
                with wave.open(str(path), "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(16000)
                    wf.writeframes(pcm.tobytes())

                print("    level {:.1f} dBFS | TRANSCRIPT {!r}".format(
                    frame_peak_db(pcm), transcript))
                print("    scores: " + "  ".join(
                    "{}={:.3f}".format(k, v) for k, v in scores.items())
                    + ("   <-- WOULD TRIGGER" if scores[name] >= TRIGGER else ""))

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    ranked = sorted(results.items(),
                    key=lambda kv: max(kv[1]) if kv[1] else 0, reverse=True)
    for name, scores in ranked:
        best = max(scores) if scores else 0.0
        verdict = ("USABLE" if best >= TRIGGER
                   else "marginal" if best >= 0.2 else "no response")
        print("  {:<14} best {:.3f}   {}   -> {}".format(
            name, best, [round(s, 3) for s in scores], verdict))

    winner, best_scores = ranked[0]
    if best_scores and max(best_scores) >= TRIGGER:
        print('\nSet in assistant/config.toml:  [wake]  model = "{}"'.format(winner))
    else:
        print("\nNo pretrained model responds to your voice.")
        print("If the transcripts above read correctly, the audio is good and")
        print("the answer is a custom model trained on your own recordings,")
        print("which are saved in {}".format(OUT_DIR))


if __name__ == "__main__":
    main()
