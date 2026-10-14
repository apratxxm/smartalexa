#!/usr/bin/env python3
"""Synthetic clips from the local Piper voice (en_US-lessac-medium), fully offline.

SUPPLEMENT only (single non-target voice): synthetic positives are
down-weighted in training; hard negatives are the more valuable half.

Outputs under A:\\wake-data\\synthetic\\ (16 kHz mono wav):
  positives/            standalone "Alexa" utterances (trailing silence trimmed)
  negatives_train/      near-miss words ("Alex", "elixir", "relax", ...) + generic phrases
  negatives_holdout/    DIFFERENT near-miss words/phrases, used only by eval.py
"""
from __future__ import annotations

import itertools
import random
from pathlib import Path

import numpy as np

from audio_io import write_wav_mono16k, SR

VOICE_MODEL = Path(r"A:\pisha\memoir\assistant\models\en_US-lessac-medium.onnx")
BASE = Path(r"A:\wake-data\synthetic")

POS_PHRASES = ["Alexa", "Alexa.", "Alexa!", "Alexa?", "alexa", "Alexa,", "Hey Alexa", "Okay Alexa"]

NEG_TRAIN_ADV = [
    "Alex", "Alex,", "Alexis", "Alexander", "a lexer", "elixir", "relax", "Alexei", "electra",
    "exam", "flex", "Lexi", "Alexia", "lexicon", "a Lexus", "Alessa", "Alisa", "Elisa", "Alexa's",
    "Alaska", "Texas", "elect", "Eleanor", "Aleksa", "a sec", "okay, next", "hey, listen",
]
NEG_TRAIN_GENERIC = [
    "play some music", "what time is it", "turn it off", "stop the music", "set a timer",
    "next up", "play the next song", "pause", "volume up", "turn the volume down",
    "what's the weather like today", "tell me a joke", "I will be right back", "okay",
    "thank you very much", "that sounds good", "can you hear me", "let's go", "see you tomorrow",
    "play something by Drake", "how are you doing", "I think that's enough", "good morning",
    "send a message to my friend", "open the door", "it is a little bit late", "never mind",
    "play Seedhe Maut", "skip this one", "who is singing this",
]
NEG_HOLDOUT = [
    "Alexandria", "election", "Alexei", "a letter", "elixirs", "Lexus", "Alexis,", "reflex",
    "a lexical", "Alexander the Great", "relaxing", "a lecture", "exit", "excuse me",
    "play the song again", "what is the date", "make it louder", "go to sleep",
    "that was amazing", "call my mother",
]


def main():
    from piper.voice import PiperVoice
    from piper.config import SynthesisConfig

    for d in ("positives", "negatives_train", "negatives_holdout"):
        (BASE / d).mkdir(parents=True, exist_ok=True)
    voice = PiperVoice.load(str(VOICE_MODEL))
    rng = random.Random(1337)

    def synth(text, length_scale, noise_scale, noise_w):
        cfg = SynthesisConfig(length_scale=length_scale, noise_scale=noise_scale,
                              noise_w_scale=noise_w, volume=1.0)
        chunks = list(voice.synthesize(text, cfg))
        if not chunks:
            return None
        a = np.concatenate([c.audio_float_array for c in chunks]).astype(np.float32)
        sr = chunks[0].sample_rate
        if sr != SR:
            n_out = int(len(a) / sr * SR)
            a = np.interp(np.linspace(0, len(a), n_out, endpoint=False),
                          np.arange(len(a)), a).astype(np.float32)
        return a

    def trim(a, pad=0.08):
        idx = np.where(np.abs(a) > 0.01)[0]
        if len(idx) == 0:
            return a
        return a[max(0, idx[0] - int(pad * SR)): idx[-1] + int(pad * SR)]

    def prosody():
        return (rng.choice([0.8, 0.9, 1.0, 1.1, 1.25, 1.4]),
                rng.choice([0.4, 0.667, 0.85, 1.0]),
                rng.choice([0.5, 0.8, 1.0]))

    def run(phrases, outdir, tag, n_var, trim_it=True):
        n = 0
        for pi, ph in enumerate(phrases):
            seen = set()
            for vi in range(n_var):
                pr = prosody()
                if pr in seen:
                    continue
                seen.add(pr)
                a = synth(ph, *pr)
                if a is None:
                    continue
                a = trim(a) if trim_it else a
                write_wav_mono16k(str(outdir / f"{tag}_{pi:03d}_{vi}.wav"), a)
                n += 1
        return n

    n_pos = run(POS_PHRASES, BASE / "positives", "pos", 10)
    n_adv = run(NEG_TRAIN_ADV, BASE / "negatives_train", "adv", 4)
    n_gen = run(NEG_TRAIN_GENERIC, BASE / "negatives_train", "gen", 3)
    n_hold = run(NEG_HOLDOUT, BASE / "negatives_holdout", "hold", 3)
    print(f"synthesized positives={n_pos} adv_neg_train={n_adv} generic_neg_train={n_gen} holdout_neg={n_hold}")


if __name__ == "__main__":
    main()
