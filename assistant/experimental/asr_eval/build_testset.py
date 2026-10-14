#!/usr/bin/env python3
"""Build the labeled command-level test set for the ASR comparison.

Reads (read-only):
  - assistant/logs/telemetry.jsonl
  - assistant/logs/real-commands-20260910.jsonl
  - assistant/logs/.vocabulary.json
  - assistant/logs/aec/*/mic16k.wav, assistant/logs/aec/live-*/raw.wav

Writes:
  - assistant/experimental/asr_eval/testset.json

Never touches anything under .venv, .env, or .cache. No audio is uploaded;
this script only reads local wav files and writes a JSON manifest of paths
and expected labels.
"""
from __future__ import annotations

import json
import wave
from pathlib import Path

REPO_ROOT = Path(r"A:\pisha\memoir")
LOGS = REPO_ROOT / "assistant" / "logs"
TELEMETRY = LOGS / "telemetry.jsonl"
REAL_COMMANDS = LOGS / "real-commands-20260910.jsonl"
VOCAB = LOGS / ".vocabulary.json"
AEC_DIR = LOGS / "aec"
OUT = Path(__file__).with_name("testset.json")

SAMPLE_RATE = 16000

# --- explicit known-intent clips given in the task -------------------------
EXPLICIT = {
    "dc44b90f4d44": "play Madira by Seedhe Maut",
    "75eaddb9f0da": "play Madira by Seedhe Maut",
    "7c3c9c5df293": "play Madira by Seedhe Maut",
    "0aefd88aa39b": "play Madira by Seedhe Maut",
    "e3638eedb17a": "play Red by Seedhe Maut",
    "b2c99bfc4359": "play Red by Seedhe Maut",
    "5f21b8130d3f": "play Naksha by Seedhe Maut",
    "cb028d864b08": "play 3 Cheez by The Siege",
    "0e9d6720c158": "pause",
    "370786df91f6": "pause",
    "ef672847ae6d": "volume 70",
    "65a929b6449a": "volume 15",
    "d0037412d046": "play Creep by Radiohead",
}

# --- curated additional confident labels from outcome=="ok" events ---------
# Chosen only where the logged transcript unambiguously supports one command
# text (simple transport/volume commands, or "Red by Seedhe Maut" family --
# the only vocabulary artist that phonetically matches "Seedemoth/Seedemov/
# see the mod/see the north", already confirmed by the task for two sibling
# events). Ambiguous/garbled events are left out on purpose.
CURATED_OK = {
    "bd8097106186": "play Red by Seedhe Maut",
    "fb2e21df4f94": "play Red by Seedhe Maut",
    "32bec86eea22": "play Red by Seedhe Maut",
    "2d1b947918ca": "play Red by Seedhe Maut",
    "84d1eaf268d8": "play Red by Seedhe Maut",
    "4de8cb47e9fc": "play Red by Seedhe Maut",
    "2c7980b64c6a": "play Red by Seedhe Maut",
    "4f582eec1cd7": "play Red by Seedhe Maut",
    "c2529385e67c": "play Red by Seedhe Maut",
    "25b2da9b3be2": "play Naksha by Seedhe Maut",
    "7883f097683c": "play champagne poetry by Drake",
    "0af40fac3656": "play my beyond playlist",
    "f7edfb053eb2": "play my beyond playlist",
    "ee212667ca2f": "play my dhh playlist",
    "a9effb986f20": "play my dhh playlist",
    "30ba795de3a8": "pause",
    "22136bfc28bc": "pause",
    "1fff418d1bc6": "pause",
    "0c8bec35fff7": "pause",
    "bebab762f1e8": "pause",
    "e049fdecf9db": "pause",
    "a9e02925d5b2": "pause",
    "97968251de05": "volume 70",
    "820b422741cc": "volume 65",
    "9beaa3cc8497": "volume 30",
    "82ecf2141346": "volume 60",
    "4f8492dfd079": "volume 50",
    "33230781267e": "volume 60",
    "77270782abe7": "volume 50",
    "fe19a21db0c1": "volume 50",
    "de6de573c7c9": "volume 20",
    "ad5278b3f08e": "skip",
}


def load_events() -> dict:
    events = {}
    with open(TELEMETRY, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            events[e.get("event_id")] = e
    return events


def resolve_telemetry_audio(rel: str) -> Path:
    # telemetry audio_path values look like "assistant\\logs\\audio\\....wav",
    # relative to the memoir repo root.
    return REPO_ROOT / rel.replace("\\", "/")


def wav_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as f:
        return f.getnframes() / float(f.getframerate())


def build_command_clips() -> list:
    events = load_events()
    clips = []

    def add(event_id: str, expected_transcript: str, source: str):
        e = events.get(event_id)
        if e is None:
            print(f"WARN: event {event_id} not found in telemetry, skipping")
            return
        rel = e.get("audio_path")
        if not rel:
            print(f"WARN: event {event_id} has no audio_path, skipping")
            return
        path = resolve_telemetry_audio(rel)
        if not path.exists():
            print(f"WARN: audio missing for {event_id}: {path}")
            return
        clips.append({
            "id": event_id,
            "audio_path": str(path),
            "expected_transcript": expected_transcript,
            "source": source,
            "duration_s": round(wav_seconds(path), 3),
        })

    for eid, txt in EXPLICIT.items():
        add(eid, txt, "explicit")
    for eid, txt in CURATED_OK.items():
        add(eid, txt, "curated_ok")

    # real-commands-20260910.jsonl: audio_path there is relative to
    # assistant/logs/, e.g. "audio/....wav" (NOT the memoir root).
    if REAL_COMMANDS.exists():
        with open(REAL_COMMANDS, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                rel = r["audio_path"]
                path = LOGS / rel
                if not path.exists():
                    print(f"WARN: real-commands audio missing: {path}")
                    continue
                txt = r.get("expected_transcript") or ""
                clips.append({
                    "id": Path(rel).stem,
                    "audio_path": str(path),
                    "expected_transcript": txt,
                    "source": "real_commands_confirmed",
                    "duration_s": round(wav_seconds(path), 3),
                })

    return clips


def build_negative_clips(max_windows: int = 16) -> list:
    """3s non-overlapping windows sliced from real mic/room audio (with music
    playing), used only to measure ASR hallucination -- i.e. false commands
    invented from non-command audio."""
    clips = []
    win_s = 3.0
    win = int(win_s * SAMPLE_RATE)
    files = sorted(AEC_DIR.glob("*/mic16k.wav")) + sorted(AEC_DIR.glob("live-*/raw.wav"))
    for wav_path in files:
        with wave.open(str(wav_path), "rb") as f:
            if f.getframerate() != SAMPLE_RATE or f.getsampwidth() != 2:
                continue
            n = f.getnframes()
        starts = list(range(0, max(0, n - win + 1), win))
        for start in starts:
            clips.append({
                "id": f"{wav_path.parent.name}_{wav_path.stem}_{start}",
                "audio_path": str(wav_path),
                "offset_samples": start,
                "num_samples": win,
                "source": "aec_negative",
            })
            if len(clips) >= max_windows:
                return clips
    return clips


def main() -> None:
    command_clips = build_command_clips()
    negative_clips = build_negative_clips()

    with open(VOCAB, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    manifest = {
        "sample_rate": SAMPLE_RATE,
        "command_clips": command_clips,
        "negative_clips": negative_clips,
        "hotwords_artists": vocab.get("artists", []),
        "hotwords_hints": vocab.get("hints", []),
    }
    OUT.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"command clips: {len(command_clips)}")
    print(f"negative clips: {len(negative_clips)}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
