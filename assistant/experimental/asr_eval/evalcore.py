"""Shared evaluation loop: run a transcribe(audio_f32)->(text, ms) callable over
the command clips + music-only negative clips and write results_<variant>.json."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from audio_utils import read_wav_int16, to_float32
from scoring import score_clip, is_hallucination

HERE = Path(__file__).parent
REPO = Path(r"A:\pisha\memoir")


def load_testset() -> dict:
    return json.loads((HERE / "testset.json").read_text(encoding="utf-8"))


def _p(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO / p


def evaluate(variant: str, transcribe, meta: dict, manifest: dict | None = None) -> dict:
    manifest = manifest or load_testset()
    results, neg_results, lat = [], [], []
    for clip in manifest["command_clips"]:
        audio = to_float32(read_wav_int16(_p(clip["audio_path"])))
        text, ms = transcribe(audio)
        lat.append(ms)
        r = score_clip(text, clip["expected_transcript"])
        r.update(id=clip["id"], source=clip["source"], latency_ms=round(ms, 1))
        results.append(r)
    for clip in manifest["negative_clips"]:
        audio = to_float32(read_wav_int16(_p(clip["audio_path"]), clip.get("offset_samples", 0),
                                          clip.get("num_samples")))
        text, ms = transcribe(audio)
        lat.append(ms)
        neg_results.append({"id": clip["id"], "predicted_transcript": text,
                            "latency_ms": round(ms, 1), "hallucinated": is_hallucination(text)})
    n = len(results)
    out = dict(meta)
    out.update(
        variant=variant,
        n_command_clips=n,
        command_accuracy=round(sum(r["command_ok"] for r in results) / n, 4),
        name_accuracy=round(sum(r["name_ok"] for r in results) / n, 4),
        n_negative_clips=len(neg_results),
        hallucinations=sum(r["hallucinated"] for r in neg_results),
        median_latency_ms=round(float(np.median(lat)), 1),
        per_clip=results, per_negative_clip=neg_results,
    )
    path = HERE / f"results_{variant.replace('+', '_')}.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{variant}: cmd={out['command_accuracy']:.3f} name={out['name_accuracy']:.3f} "
          f"halluc={out['hallucinations']}/{out['n_negative_clips']} "
          f"med_ms={out['median_latency_ms']:.0f}", flush=True)
    return out
