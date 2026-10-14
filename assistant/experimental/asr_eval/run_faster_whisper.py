#!/usr/bin/env python3
"""Evaluate a faster-whisper (CTranslate2) model on the ASR eval test set,
with and without hotwords, exactly matching the decoding settings used in
production (assistant/stt.py:_FasterWhisperBackend.transcribe): greedy
(beam_size=1, best_of=1, temperature=0.0), condition_on_previous_text=False,
vad_filter=False, without_timestamps=True, no_speech_threshold=0.6,
compression_ratio_threshold=2.4, language="en".

Run with A:\\pisha\\memoir\\.venv-asr (NOT the production .venv, NOT .env/.cache).

Usage:
  A:\\pisha\\memoir\\.venv-asr\\Scripts\\python.exe run_faster_whisper.py ^
      --model-path <path> --tag large-v3-turbo --device cpu --compute-type int8
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from scoring import score_clip, is_hallucination  # noqa: E402

TESTSET = HERE / "testset.json"


def read_wav_int16(path: Path, offset_samples: int = 0, num_samples: int = None) -> np.ndarray:
    with wave.open(str(path), "rb") as f:
        sr = f.getframerate()
        assert sr == 16000, f"expected 16kHz, got {sr} for {path}"
        if offset_samples:
            f.setpos(offset_samples)
        n = num_samples if num_samples is not None else (f.getnframes() - offset_samples)
        raw = f.readframes(n)
    samples = np.frombuffer(raw, dtype=np.int16)
    return samples


def to_float32(samples_int16: np.ndarray) -> np.ndarray:
    return samples_int16.astype(np.float32) / 32768.0


def build_hotwords(artists: list, max_names: int = 60) -> str:
    return ", ".join(artists[:max_names])


def transcribe(model, audio_f32: np.ndarray, hotwords: str | None) -> tuple:
    # Host RAM is shared with the running assistant; a transient
    # "mkl_malloc: failed to allocate memory" is retried (not counted in latency).
    for attempt in range(6):
        try:
            return _transcribe_once(model, audio_f32, hotwords)
        except RuntimeError as exc:
            if "alloc" not in str(exc).lower() or attempt == 5:
                raise
            print(f"  retry after: {exc}", flush=True)
            time.sleep(3)


def _transcribe_once(model, audio_f32: np.ndarray, hotwords: str | None) -> tuple:
    t0 = time.perf_counter()
    segments, _ = model.transcribe(
        audio_f32,
        language="en",
        beam_size=1,
        best_of=1,
        temperature=0.0,
        hotwords=hotwords,
        condition_on_previous_text=False,
        vad_filter=False,
        without_timestamps=True,
        no_speech_threshold=0.6,
        compression_ratio_threshold=2.4,
    )
    text = " ".join(s.text for s in segments).strip()
    ms = (time.perf_counter() - t0) * 1000.0
    return text, ms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--tag", required=True, help="short name for the results file, e.g. large-v3-turbo")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--compute-type", default="int8")
    ap.add_argument("--num-threads", type=int, default=4)
    ap.add_argument("--variants", default="plain,hotwords",
                    help="comma list of plain,hotwords (resume support)")
    args = ap.parse_args()
    wanted = set(args.variants.split(","))

    from faster_whisper import WhisperModel

    manifest = json.loads(TESTSET.read_text(encoding="utf-8"))
    command_clips = manifest["command_clips"]
    negative_clips = manifest["negative_clips"]
    artists = manifest["hotwords_artists"]
    hotwords_str = build_hotwords(artists)

    print(f"Loading {args.tag} from {args.model_path} on {args.device}/{args.compute_type} ...")
    t0 = time.perf_counter()
    model = WhisperModel(args.model_path, device=args.device, compute_type=args.compute_type,
                         local_files_only=True, cpu_threads=args.num_threads)
    # warmup
    list(model.transcribe(np.zeros(16000, dtype=np.float32), language="en")[0])
    print(f"Loaded in {time.perf_counter()-t0:.1f}s")

    for hotwords_on in (False, True):
        if ("hotwords" if hotwords_on else "plain") not in wanted:
            continue
        hw = hotwords_str if hotwords_on else None
        variant = f"{args.tag}{'+hotwords' if hotwords_on else ''}"
        print(f"\n=== {variant} ===")

        results = []
        latencies = []
        for clip in command_clips:
            audio = to_float32(read_wav_int16(Path(clip["audio_path"])))
            text, ms = transcribe(model, audio, hw)
            latencies.append(ms)
            r = score_clip(text, clip["expected_transcript"])
            r["id"] = clip["id"]
            r["source"] = clip["source"]
            r["latency_ms"] = round(ms, 1)
            results.append(r)

        neg_results = []
        for clip in negative_clips:
            audio = to_float32(read_wav_int16(Path(clip["audio_path"]),
                                              clip.get("offset_samples", 0),
                                              clip.get("num_samples")))
            text, ms = transcribe(model, audio, hw)
            latencies.append(ms)
            neg_results.append({
                "id": clip["id"],
                "predicted_transcript": text,
                "latency_ms": round(ms, 1),
                "hallucinated": is_hallucination(text),
            })

        n = len(results)
        command_acc = sum(r["command_ok"] for r in results) / n if n else 0.0
        name_acc = sum(r["name_ok"] for r in results) / n if n else 0.0
        halluc = sum(r["hallucinated"] for r in neg_results)
        med_latency = float(np.median(latencies)) if latencies else 0.0

        out = {
            "model": args.tag,
            "variant": variant,
            "hotwords": hotwords_on,
            "device": args.device,
            "compute_type": args.compute_type,
            "n_command_clips": n,
            "command_accuracy": round(command_acc, 4),
            "name_accuracy": round(name_acc, 4),
            "n_negative_clips": len(neg_results),
            "hallucinations": halluc,
            "median_latency_ms": round(med_latency, 1),
            "per_clip": results,
            "per_negative_clip": neg_results,
        }
        out_path = HERE / f"results_{variant.replace('+', '_')}.json"
        out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"command_accuracy={command_acc:.3f} name_accuracy={name_acc:.3f} "
              f"hallucinations={halluc}/{len(neg_results)} median_latency_ms={med_latency:.1f}")
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
