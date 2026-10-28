"""Pure, seeded plan for synthetic clips: phrases, prosody and splits. No audio I/O."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = ROOT / "assistant" / "models"
WORK_DIR = Path(__file__).resolve().parent / "_work"
CANDIDATES_DIR = ROOT / "assistant" / "models" / "wake" / "candidates"
SEED = 1337
SR = 16000

POS_TRAIN = ("Alexa", "Alexa.", "Alexa!", "alexa", "Alexa,", "Hey Alexa", "Okay Alexa",
             "Alexa, pause", "Alexa play some music", "Alexa, stop the timer")
POS_HOLDOUT = ("Alexa?", "Alexa, what time is it", "Alexa, volume up")
NEG_CONF_TRAIN = ("Alex", "Alex,", "Alexis", "Alexander", "a lexer", "elixir", "relax a", "Alexei",
                  "electra", "exam", "flex", "Lexi", "Alexia", "lexicon", "a Lexus", "Alessa", "Alisa",
                  "Elisa", "Alexa's", "Alaska", "Texas", "elect", "Eleanor", "Aleksa", "a sec", "election")
NEG_CONF_HOLDOUT = ("Alexandria", "elixirs", "Lexus", "reflex", "a lexical", "relaxing", "a lecture",
                    "exit", "excuse me")
NEG_GEN_TRAIN = ("play some music", "turn it off", "stop the music", "set a timer", "next up",
                 "play the next song", "pause", "turn the volume down", "what's the weather like today",
                 "tell me a joke", "I will be right back", "okay", "thank you very much", "that sounds good",
                 "can you hear me", "let's go", "see you tomorrow", "play something by Drake",
                 "how are you doing", "I think that's enough", "good morning", "send a message to my friend",
                 "open the door", "it is a little bit late", "never mind", "skip this one", "who is singing this")
NEG_GEN_HOLDOUT = ("play the song again", "what is the date", "make it louder", "go to sleep",
                   "that was amazing", "call my mother")

LENGTH_SCALES = (0.8, 0.9, 1.0, 1.1, 1.25, 1.4)
NOISE_SCALES = (0.4, 0.667, 0.85, 1.0)
NOISE_W = (0.5, 0.8, 1.0)
RATE_FACTORS = (0.92, 0.96, 1.0, 1.04, 1.08)  # resampling: speed and pitch move together
VARIANTS = {  # (split, kind) -> variants per phrase
    ("train", "positive"): 12, ("holdout", "positive"): 6,
    ("train", "neg_confusable"): 4, ("holdout", "neg_confusable"): 3,
    ("train", "neg_generic"): 3, ("holdout", "neg_generic"): 3,
}


@dataclass(frozen=True)
class SynthItem:
    item_id: str
    kind: str          # positive | neg_confusable | neg_generic
    split: str         # train | holdout
    text: str
    voice: str         # "<voice stem>#<speaker id>"
    length_scale: float
    noise_scale: float
    noise_w: float
    rate: float
    augment_seed: int


def discover_voices(models_dir: Path = MODELS_DIR) -> dict[str, tuple[Path, int]]:
    """Piper voices (onnx + onnx.json pair). Returns {"stem#speaker": (onnx_path, speaker_id)}."""
    voices: dict[str, tuple[Path, int]] = {}
    for onnx in sorted(Path(models_dir).glob("*.onnx")):
        cfg_path = onnx.with_name(onnx.name + ".json")
        if not cfg_path.is_file():
            continue
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        if "espeak" not in cfg and "phoneme_id_map" not in cfg:
            continue
        speakers = int(cfg.get("num_speakers", 1))
        for sid in range(max(1, speakers)):
            voices[f"{onnx.stem}#{sid}"] = (onnx, sid)
    return voices


def voice_split(voice_keys: list[str]) -> tuple[list[str], list[str], bool]:
    """(train_voices, holdout_voices, independent). With one voice, holdout is text-only."""
    keys = sorted(voice_keys)
    if len(keys) >= 2:
        return keys[:-1], keys[-1:], True
    return keys, keys, False


def build_plan(voice_keys: list[str], *, seed: int = SEED) -> list[SynthItem]:
    """Deterministic list of synthetic items. Same seed and voices give the same plan."""
    if not voice_keys:
        raise ValueError("at least one Piper voice is required")
    rng = random.Random(seed)
    train_voices, hold_voices, independent = voice_split(voice_keys)
    specs = [
        ("positive", "train", POS_TRAIN, train_voices),
        ("positive", "holdout", POS_TRAIN if independent else POS_HOLDOUT, hold_voices),
        ("neg_confusable", "train", NEG_CONF_TRAIN, train_voices),
        ("neg_confusable", "holdout", NEG_CONF_HOLDOUT, hold_voices),
        ("neg_generic", "train", NEG_GEN_TRAIN, train_voices),
        ("neg_generic", "holdout", NEG_GEN_HOLDOUT, hold_voices),
    ]
    items: list[SynthItem] = []
    for kind, split, phrases, voices in specs:
        n_var = VARIANTS[(split, kind)]
        for pi, text in enumerate(phrases):
            seen: set[tuple] = set()
            vi = 0
            while vi < n_var:
                prosody = (rng.choice(LENGTH_SCALES), rng.choice(NOISE_SCALES),
                           rng.choice(NOISE_W), rng.choice(RATE_FACTORS))
                if prosody in seen:
                    continue
                seen.add(prosody)
                voice = voices[(pi + vi) % len(voices)]
                items.append(SynthItem(
                    item_id=f"{kind}-{split}-{pi:03d}-{vi:02d}", kind=kind, split=split, text=text,
                    voice=voice, length_scale=prosody[0], noise_scale=prosody[1],
                    noise_w=prosody[2], rate=prosody[3], augment_seed=rng.randrange(2 ** 31)))
                vi += 1
    return items


def plan_counts(items: list[SynthItem]) -> dict:
    out: dict[str, int] = {}
    for it in items:
        key = f"{it.kind}_{it.split}"
        out[key] = out.get(key, 0) + 1
    return out
