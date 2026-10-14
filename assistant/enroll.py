"""Record wake-word training clips of the user's own voice, on cue.

    python -m assistant.enroll              # full session, about 6 minutes
    python -m assistant.enroll --quick      # half the prompts

Live use only ever archived the wake words the assistant already HEARD, so
the misses -- the clips a personal model needs most -- were never saved, and
a month of use produced only 130 usable "Alexa" clips. This collects a
balanced set in minutes: quiet and over music, near and far, soft and loud,
"Alexa" alone and "Alexa, <command>", plus near-words and free talk as
negatives.

The running assistant pauses listening for the whole session (it would
otherwise react to every prompt) and resumes on exit, even after Ctrl+C.
Clips stay on this machine under A:/wake-data/enroll/<session>/.
"""
from __future__ import annotations

import argparse
import json
import time
import wave
from datetime import datetime
from pathlib import Path

import numpy as np

from .audio import Microphone, SileroVAD
from .config import ROOT, load

OUT_ROOT = Path("A:/wake-data/enroll")
PAUSE_FLAG = ROOT / "assistant" / "logs" / ".listening_paused"
RATE = 16000

# (label, instruction, seconds, count). Counts are for the full session.
QUIET = [
    ("pos_normal", 'Say "Alexa" normally', 2.2, 8),
    ("pos_soft", 'Say "Alexa" softly, almost to yourself', 2.2, 5),
    ("pos_loud", 'Say "Alexa" loudly, as if across a noisy room', 2.2, 4),
    ("pos_fast", 'Say "Alexa" quickly', 2.0, 4),
    ("pos_far", 'Step 2-3 m away (or turn away) and say "Alexa"', 2.4, 5),
    ("pos_question", 'Say "Alexa?" as a question', 2.2, 3),
    ("pos_command", 'Say "Alexa" then any command in one breath ("Alexa, pause", "Alexa, play Naksha")', 4.5, 6),
]
MUSIC = [
    ("pos_music_normal", 'Music on. Say "Alexa" normally', 2.2, 8),
    ("pos_music_loud", 'Music on. Say "Alexa" loudly', 2.2, 5),
    ("pos_music_far", 'Music on. From 2-3 m away, say "Alexa"', 2.4, 5),
    ("pos_music_command", 'Music on. Say "Alexa" then a command in one breath', 4.5, 6),
]
NEAR = [
    ("neg_near", 'Say "Alex"', 2.0, 1), ("neg_near", 'Say "election"', 2.0, 1),
    ("neg_near", 'Say "Alexander"', 2.2, 1), ("neg_near", 'Say "relax"', 2.0, 1),
    ("neg_near", 'Say "Alaska"', 2.0, 1), ("neg_near", 'Say "a lexus"', 2.0, 1),
    ("neg_near", 'Say "Electra"', 2.0, 1), ("neg_near", 'Say "Elixir"', 2.0, 1),
]
FREE_TALK_S = 60.0


def _write(path: Path, pcm: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(RATE)
        wf.writeframes(pcm.astype(np.int16).tobytes())


def _record(mic: Microphone, seconds: float) -> np.ndarray:
    mic.flush()
    frames = [mic.read() for _ in range(int(seconds * RATE / 512))]
    return np.concatenate(frames)


def _speech_peak(vad: SileroVAD, pcm: np.ndarray) -> float:
    vad.reset()
    return max((vad(pcm[i:i + 512]) for i in range(0, pcm.size - 511, 512)), default=0.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true", help="half the prompts")
    args = ap.parse_args()

    from . import local_media
    from .feedback import BEEP_TICK, Feedback

    cfg = load()
    feedback = Feedback(cfg)
    vad = SileroVAD(RATE)
    session = OUT_ROOT / datetime.now().strftime("%Y%m%d-%H%M%S")
    manifest = []
    was_playing = bool(local_media.is_playing())

    def plan(items):
        return [(label, text, sec, max(1, count // 2) if args.quick else count)
                for label, text, sec, count in items]

    PAUSE_FLAG.write_text("enroll", encoding="utf-8")
    time.sleep(1.0)  # the assistant checks the flag twice a second
    print("\nThe assistant is paused while we record. Ctrl+C stops at any time.\n"
          "After each beep, say the prompt once. Clips are saved under", session, "\n")
    try:
        with Microphone(sample_rate=RATE, frame_length=512,
                        device=cfg.get("audio.input_device") or None,
                        capture_rate=cfg.get("audio.capture_rate"),
                        exclusive=cfg.get("audio.exclusive_mode", False)) as mic:

            def prompt_set(items, part):
                total = sum(count for *_, count in items)
                done = 0
                for label, text, seconds, count in items:
                    for _ in range(count):
                        done += 1
                        for attempt in range(2):
                            print("[{} {}/{}] {}".format(part, done, total, text))
                            time.sleep(0.6)
                            feedback._beep(*BEEP_TICK)
                            pcm = _record(mic, seconds)
                            peak = _speech_peak(vad, pcm)
                            if peak >= 0.5 or attempt == 1:
                                break
                            print("   didn't catch that -- once more")
                        name = "{}_{:03d}.wav".format(label, len(manifest))
                        _write(session / name, pcm)
                        manifest.append({"file": name, "label": label, "prompt": text,
                                         "vad_peak": round(float(peak), 3),
                                         "music": label.startswith("pos_music")})

            local_media.pause()
            input("Part 1 of 4: quiet room. Press Enter to start.")
            prompt_set(plan(QUIET), "quiet")

            local_media.play()
            input("\nPart 2 of 4: with music. Spotify is playing -- set it to your usual "
                  "listening volume, then press Enter.")
            prompt_set(plan(MUSIC), "music")
            local_media.pause()

            input("\nPart 3 of 4: words that are NOT the wake word. Press Enter.")
            prompt_set(NEAR, "near")

            seconds = FREE_TALK_S / (2 if args.quick else 1)
            input("\nPart 4 of 4: talk normally for {:.0f} s about anything -- "
                  "do NOT say Alexa. Press Enter, then talk.".format(seconds))
            feedback._beep(*BEEP_TICK)
            pcm = _record(mic, seconds)
            _write(session / "neg_free_talk.wav", pcm)
            manifest.append({"file": "neg_free_talk.wav", "label": "neg_free", "music": False})
            print("Done.")
    except KeyboardInterrupt:
        print("\nStopped; keeping the clips recorded so far.")
    finally:
        PAUSE_FLAG.unlink(missing_ok=True)
        if manifest:
            session.mkdir(parents=True, exist_ok=True)
            (session / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
        try:
            (local_media.play if was_playing else local_media.pause)()
        except Exception:
            pass
    positives = sum(1 for m in manifest if m["label"].startswith("pos"))
    print("Saved {} clips ({} wake words) to {}. The assistant is listening again."
          .format(len(manifest), positives, session))


if __name__ == "__main__":
    main()
