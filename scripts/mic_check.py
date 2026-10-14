"""Record, play back, and analyse -- the definitive check on capture quality.

Every automated proxy for "is this audio good" has misled us here: level looked
fine, spectrum looked fine, and the wake model scored 0.96 on one clip. Yet a
full spoken sentence transcribes to nothing. The one check never run is the
obvious one -- listening to it.

This records you, plays it straight back, and prints the objective numbers
alongside. If the playback sounds like you, the capture is good and the fault is
in the models. If it sounds broken, chopped, robotic or silent, the fault is in
the capture path and no amount of model work will fix it.

    python scripts/mic_check.py
    python scripts/mic_check.py --exclusive     # compare the other audio path
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from assistant.audio import Microphone  # noqa: E402
from assistant.config import load  # noqa: E402

OUT = ROOT / "assistant" / "logs" / "miccheck.wav"


def analyse(pcm: np.ndarray, sr: int = 16000) -> None:
    x = pcm.astype(np.float64)
    frames = [x[i:i + 512] for i in range(0, len(x) - 512, 512)]
    rms = np.array([np.sqrt((f ** 2).mean()) for f in frames])
    db = 20 * np.log10(np.maximum(rms, 1) / 32768)

    print("  duration        : {:.1f}s".format(len(pcm) / sr))
    print("  peak sample     : {}  (clipped frames: {})".format(
        int(np.abs(pcm).max()), int(np.sum(np.abs(pcm) >= 32000))))
    print("  loudest frame   : {:.1f} dBFS".format(db.max()))
    print("  quietest frame  : {:.1f} dBFS".format(db.min()))
    print("  dynamic range   : {:.1f} dB  (speech needs >20 dB between words "
          "and silence)".format(db.max() - db.min()))

    # Zero runs: a stuttering or gap-ridden stream is the classic symptom of a
    # capture path dropping blocks, and is invisible to level and spectrum.
    zeros = (pcm == 0).astype(np.int8)
    runs, cur = [], 0
    for z in zeros:
        if z:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    long_runs = [r for r in runs if r >= 64]
    print("  zero-runs >=4ms : {}  (total {} samples, {:.1f}% of signal)".format(
        len(long_runs), sum(long_runs), 100 * sum(long_runs) / max(len(pcm), 1)))
    if long_runs:
        print("    ^ gaps in the stream: the capture path is dropping audio")

    n = len(x) // 2 * 2
    spec = np.abs(np.fft.rfft(x[:n] * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, 1 / sr)
    tot = spec.sum() + 1e-9
    bands = [(0, 300), (300, 1000), (1000, 3000), (3000, 8000)]
    print("  spectrum        : " + "  ".join(
        "{}-{}Hz={:.0f}%".format(lo, hi, 100 * spec[(freqs >= lo) & (freqs < hi)].sum() / tot)
        for lo, hi in bands))
    print("    (speech is usually strongest in 300-3000 Hz; a signal dominated")
    print("     by 0-300 Hz is rumble or handling noise, not voice)")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--exclusive", dest="exclusive", action="store_true")
    ap.add_argument("--shared", dest="exclusive", action="store_false")
    ap.set_defaults(exclusive=None)
    args = ap.parse_args()

    cfg = load()
    exclusive = (cfg.get("audio.exclusive_mode", False)
                 if args.exclusive is None else args.exclusive)

    with Microphone(
        sample_rate=16000,
        frame_length=512,
        device=cfg.get("audio.input_device") or None,
        capture_rate=cfg.get("audio.capture_rate"),
        exclusive=exclusive,
    ) as mic:
        print("")
        print("Recording {:.0f}s. Say: \"testing one two three, play my gym "
              "playlist\"".format(args.seconds))
        for n in (3, 2, 1):
            print("  {}...".format(n), end="", flush=True)
            time.sleep(1.0)
        print("  SPEAK NOW")
        total = int(args.seconds * 16000 / 512)
        frames = [mic.read() for _ in range(total)]
    pcm = np.concatenate(frames)

    with wave.open(str(OUT), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(pcm.tobytes())

    print("")
    print("ANALYSIS")
    analyse(pcm)

    import sounddevice as sd

    print("")
    print("PLAYBACK -- 16 kHz, what the models actually receive.")
    print("  This is EXPECTED to sound duller than Sound Recorder: 16 kHz has")
    print("  8 kHz of bandwidth vs 48 kHz. Silero, openWakeWord and Moonshine")
    print("  all require 16 kHz, so this dullness is by design, not a fault.")
    try:
        sd.play(pcm, 16000)
        sd.wait()
    except Exception as exc:
        print("  playback failed ({}); open {} manually".format(exc, OUT))

    print("")
    print("TRANSCRIBING...")
    try:
        from assistant.stt import Transcriber

        text, ms = Transcriber("moonshine/base").transcribe(pcm)
        print("  transcript: {!r}  ({} ms)".format(text, ms))
    except Exception as exc:
        print("  stt failed: {}".format(exc))

    print("")
    print("Saved to {}".format(OUT))
    print("")
    print("THE TRANSCRIPT ABOVE IS THE REAL VERDICT -- not how it sounds.")
    print("If it reads back your sentence, the pipeline is good enough for the")
    print("models, however dull the 16 kHz playback is to your ear.")


if __name__ == "__main__":
    main()
