#!/usr/bin/env python3
"""Drive synth_winrt.ps1 (Windows OneCore voices incl. en-IN Heera/Ravi; local,
offline) for positives / adversarial negatives / generic negatives, then trim
and convert to 16 kHz mono wav next to the Piper clips (prefix 'w_')."""
import subprocess
import wave
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from audio_io import write_wav_mono16k, SR
from synth_tts import POS_PHRASES, NEG_TRAIN_ADV, NEG_TRAIN_GENERIC, NEG_HOLDOUT

HERE = Path(__file__).parent
BASE = Path(r"A:\wake-data\synthetic")
RAW = BASE / "winrt_raw"
JOBS = [  # (tag, phrases, variants per voice/phrase, dest dir)
    ("pos", ["Alexa", "Alexa.", "Alexa!", "Alexa?", "Alexa,", "alexa"], 6, "positives"),
    ("adv", NEG_TRAIN_ADV, 3, "negatives_train"),
    ("gen", NEG_TRAIN_GENERIC, 2, "negatives_train"),
    ("hold", NEG_HOLDOUT, 2, "negatives_holdout"),
]


def read_any(p):
    with wave.open(str(p), "rb") as w:
        sr, nch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16 if sw == 2 else np.uint8)
    x = x.astype(np.float32) / (32768.0 if sw == 2 else 128.0)
    if nch > 1:
        x = x.reshape(-1, nch).mean(1)
    return x, sr


def main():
    for tag, phrases, nvar, dest in JOBS:
        out = RAW / tag
        out.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(["powershell", "-NoProfile", "-File", str(HERE / "synth_winrt.ps1"), "-OutDir", str(out),
                            "-Phrases", "|".join(phrases), "-Tag", tag, "-MaxVariants", str(nvar)],
                           capture_output=True, text=True)
        print(tag, r.stdout.strip()[-300:], r.stderr.strip()[-200:])
        d = BASE / dest
        d.mkdir(parents=True, exist_ok=True)
        n = 0
        for p in sorted(out.glob("*.wav")):
            x, sr = read_any(p)
            if sr != SR:
                from math import gcd
                g = gcd(sr, SR)
                x = resample_poly(x, SR // g, sr // g).astype(np.float32)
            idx = np.where(np.abs(x) > 0.01)[0]
            if len(idx) == 0:
                continue
            x = x[max(0, idx[0] - int(0.08 * SR)): idx[-1] + int(0.12 * SR)]
            pk = np.max(np.abs(x))
            if pk > 0:
                x = x * (0.5 / pk)  # these voices are loud; normalise to a moderate level
            write_wav_mono16k(str(d / f"w_{p.stem}.wav"), x)
            n += 1
        print(tag, "->", dest, n, "clips")


if __name__ == "__main__":
    main()
