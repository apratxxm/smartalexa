#!/usr/bin/env python3
"""Evaluate the official native-Transformers Qwen3-ASR checkpoint locally.

This runner deliberately keeps model imports behind CLI, checkpoint, manifest,
audio, and report-path validation. It never downloads model files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
import wave
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable

HERE = Path(__file__).parent
TESTSET = HERE / "testset.json"
sys.path.insert(0, str(HERE))
from scoring import is_hallucination, score_clip  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="existing local model directory")
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--language", choices=("auto", "en"), default="auto",
                        help="auto-detect (default) or constrain transcription to English")
    parser.add_argument("--max-clips", type=int, default=64,
                        help="maximum total cases, 1..64; commands precede negatives")
    parser.add_argument("--deadline-seconds", type=float, default=900.0,
                        help="whole-run deadline including local model load")
    parser.add_argument("--threads", type=int, default=4,
                        help="CPU intra-op threads (1..4)")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--report", help="optional JSONL path; must not already exist")
    return parser


def _validate_args(args: argparse.Namespace, manifest_path: Path) -> tuple[Path, dict, bytes, Path | None]:
    model_path = Path(args.model_path).expanduser()
    if not model_path.is_dir():
        raise ValueError(f"--model-path must be an existing local directory: {model_path}")
    if not 1 <= args.max_clips <= 64:
        raise ValueError("--max-clips must be between 1 and 64")
    if not 1 <= args.threads <= 4:
        raise ValueError("--threads must be between 1 and 4")
    if (isinstance(args.deadline_seconds, bool)
            or not math.isfinite(args.deadline_seconds)
            or not 0 < args.deadline_seconds <= 900):
        raise ValueError("--deadline-seconds must be finite and between 0 and 900")
    if not 1 <= args.max_new_tokens <= 128:
        raise ValueError("--max-new-tokens must be between 1 and 128")

    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    if manifest.get("sample_rate") != 16000:
        raise ValueError("evaluation manifest must declare 16000 Hz audio")
    commands = manifest.get("command_clips")
    negatives = manifest.get("negative_clips")
    if not isinstance(commands, list) or not isinstance(negatives, list):
        raise ValueError("evaluation manifest must contain command_clips and negative_clips lists")
    cases = commands + negatives
    if not cases or len(cases) > 64:
        raise ValueError("evaluation corpus must contain 1..64 total cases")
    seen: set[str] = set()
    for case in cases:
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id or case_id in seen:
            raise ValueError("each evaluation case must have a unique nonempty id")
        seen.add(case_id)
        if not isinstance(case.get("audio_path"), str):
            raise ValueError(f"case {case_id} has no audio_path")
        if case in commands and not isinstance(case.get("expected_transcript"), str):
            raise ValueError(f"command case {case_id} has no expected_transcript")
        _validate_wav(case)

    report_path = Path(args.report).expanduser() if args.report else None
    if report_path is not None:
        if not report_path.parent.is_dir():
            raise ValueError(f"report parent directory does not exist: {report_path.parent}")
        if report_path.exists():
            raise FileExistsError(f"report already exists: {report_path}")
    return model_path, manifest, raw, report_path


def _validate_wav(case: dict) -> None:
    path = Path(case["audio_path"])
    if not path.is_file():
        raise FileNotFoundError(f"audio for case {case['id']} does not exist: {path}")
    with wave.open(str(path), "rb") as wav:
        if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (16000, 1, 2):
            raise ValueError(f"case {case['id']} must be 16 kHz mono 16-bit PCM: {path}")
        total = wav.getnframes()
    offset = case.get("offset_samples", 0)
    length = case.get("num_samples")
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0 or offset > total:
        raise ValueError(f"case {case['id']} has invalid offset_samples")
    if length is not None and (isinstance(length, bool) or not isinstance(length, int)
                               or length < 0 or offset + length > total):
        raise ValueError(f"case {case['id']} has invalid num_samples")
    if (length if length is not None else total - offset) <= 0:
        raise ValueError(f"case {case['id']} selects no audio samples")
    selected = length if length is not None else total - offset
    if selected > 30 * 16000:
        raise ValueError(f"case {case['id']} exceeds the 30-second comparison limit")
    with wave.open(str(path), "rb") as wav:
        wav.setpos(offset)
        if len(wav.readframes(selected)) != selected * 2:
            raise ValueError(f"case {case['id']} has truncated PCM data")


def read_audio(case: dict):
    import numpy as np

    path = Path(case["audio_path"])
    offset = case.get("offset_samples", 0)
    length = case.get("num_samples")
    with wave.open(str(path), "rb") as wav:
        wav.setpos(offset)
        raw = wav.readframes(length if length is not None else wav.getnframes() - offset)
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def load_dependencies():
    """Import heavyweight inference libraries only after all preflight checks."""
    import torch
    import transformers
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    return torch, transformers, AutoProcessor, AutoModelForMultimodalLM


def _case_lists(manifest: dict, max_clips: int) -> list[tuple[str, dict]]:
    cases = [("command", item) for item in manifest["command_clips"]]
    cases.extend(("negative", item) for item in manifest["negative_clips"])
    return cases[:max_clips]


def _device_sync(torch: Any, device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def _infer(torch: Any, processor: Any, model: Any, audio: Any,
           language: str | None, max_new_tokens: int) -> str:
    inputs = processor.apply_transcription_request(audio=audio, language=language)
    if hasattr(inputs, "to"):
        inputs = inputs.to(model.device, model.dtype)
    context = torch.inference_mode() if hasattr(torch, "inference_mode") else torch.no_grad()
    with context:
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens)
    # The official native model returns input IDs followed by generated IDs.
    # Its parser needs the generated language/ASR markers, not prompt tokens.
    generated_ids = generated[:, inputs["input_ids"].shape[1]:]
    decoded = processor.decode(generated_ids, return_format="transcription_only")
    if isinstance(decoded, (list, tuple)):
        decoded = decoded[0] if decoded else ""
    if hasattr(decoded, "text"):
        decoded = decoded.text
    return str(decoded).strip()


def _source_counts(cases: list[tuple[str, dict]]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {"command": {}, "negative": {}}
    for kind, case in cases:
        source = str(case.get("source", "unknown"))
        counts[kind][source] = counts[kind].get(source, 0) + 1
    return counts


def run(args: argparse.Namespace, *, dependency_loader: Callable = load_dependencies,
        manifest_path: Path = TESTSET, now: Callable = time.perf_counter,
        audio_reader: Callable = read_audio, emit: Callable = print) -> dict:
    model_path, manifest, raw_manifest, report_path = _validate_args(args, manifest_path)
    cases = _case_lists(manifest, args.max_clips)
    language = None if args.language == "auto" else args.language
    run_started = now()
    report = report_path.open("x", encoding="utf-8") if report_path else None

    def write(row: dict) -> None:
        line = json.dumps(row, ensure_ascii=False)
        emit(line)
        if report is not None:
            report.write(line + "\n")
            report.flush()

    meta = {
        "event": "run_start",
        "backend": "transformers-native",
        "model": str(model_path),
        "device": args.device,
        "language": language or "auto",
        "max_new_tokens": args.max_new_tokens,
        "requested_threads": args.threads,
        "manifest_sha256": hashlib.sha256(raw_manifest).hexdigest(),
        "n_command_clips": min(len(manifest["command_clips"]), args.max_clips),
        "n_negative_clips": max(0, min(len(manifest["command_clips"]) + len(manifest["negative_clips"]), args.max_clips) - len(manifest["command_clips"])),
        "source_counts": _source_counts(cases),
        "max_clips": args.max_clips,
        "deadline_seconds": args.deadline_seconds,
        "deadline_scope": "checked before each case; an in-flight local load or generation call cannot be interrupted",
        "comparison_note": "Qwen generation and decoding settings differ from faster-whisper; compare task metrics cautiously.",
    }
    try:
        torch, transformers, processor_class, model_class = dependency_loader()
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        torch.set_num_threads(args.threads)
        torch_dtype = torch.float16 if args.device == "cuda" else torch.float32
        cold_started = now()
        processor = processor_class.from_pretrained(
            str(model_path), local_files_only=True, trust_remote_code=False)
        model = model_class.from_pretrained(str(model_path), dtype=torch_dtype,
                                            device_map=args.device,
                                            local_files_only=True, trust_remote_code=False)
        model.eval()
        _device_sync(torch, args.device)
        cold_load_s = max(0.0, now() - cold_started)
        meta.update({
            "python": sys.version.split()[0],
            "torch": getattr(torch, "__version__", "unknown"),
            "torch_cuda": getattr(getattr(torch, "version", None), "cuda", None),
            "transformers": getattr(transformers, "__version__", "unknown"),
            "cold_load_seconds": round(cold_load_s, 3),
            "case_timing_note": "Per-case latency includes the first request; no hidden warmup inference was run.",
        })
        write(meta)
        case_latencies: list[float] = []
        failures = 0
        attempted = 0
        negative_scored = 0
        command_scores: list[dict] = []
        hallucinations = 0
        for kind, case in cases:
            case_id = case["id"]
            if now() - run_started >= args.deadline_seconds:
                failures += 1
                write({"event": "case", "case_id": case_id, "kind": kind,
                       "source": case.get("source"), "latency_ms": None,
                       "error": "deadline exceeded before inference", "failure": True})
                continue
            start = now()
            attempted += 1
            try:
                audio = audio_reader(case)
                _device_sync(torch, args.device)
                transcript = _infer(torch, processor, model, audio, language,
                                    args.max_new_tokens)
                _device_sync(torch, args.device)
                elapsed_ms = max(0.0, (now() - start) * 1000.0)
                case_latencies.append(elapsed_ms)
                if kind == "command":
                    score = score_clip(transcript, case["expected_transcript"])
                    score["id"] = case_id
                    score["source"] = case.get("source")
                    score["latency_ms"] = round(elapsed_ms, 1)
                    command_scores.append(score)
                    write({"event": "case", "case_id": case_id, "kind": kind,
                           "source": case.get("source"), "transcript": transcript,
                           "score": score, "latency_ms": round(elapsed_ms, 1),
                           "error": None, "failure": False})
                else:
                    negative_scored += 1
                    hallucinated = is_hallucination(transcript)
                    hallucinations += int(hallucinated)
                    write({"event": "case", "case_id": case_id, "kind": kind,
                           "source": case.get("source"), "transcript": transcript,
                           "hallucinated": hallucinated,
                           "latency_ms": round(elapsed_ms, 1),
                           "error": None, "failure": False})
            except Exception as exc:
                elapsed_ms = max(0.0, (now() - start) * 1000.0)
                case_latencies.append(elapsed_ms)
                failures += 1
                write({"event": "case", "case_id": case_id, "kind": kind,
                       "source": case.get("source"), "latency_ms": round(elapsed_ms, 1),
                       "error": f"{type(exc).__name__}: {exc}", "failure": True})
                if isinstance(exc, MemoryError) or "out of memory" in str(exc).casefold():
                    break

        summary = {
            "event": "run_summary",
            "cases_total": len(cases),
            "cases_attempted": attempted,
            "complete": attempted == len(cases) and failures == 0,
            "cases_failed": failures,
            "command_scored": len(command_scores),
            "command_accuracy": round(sum(s["command_ok"] for s in command_scores) / len(command_scores), 4) if command_scores else None,
            "name_accuracy": round(sum(s["name_ok"] for s in command_scores) / len(command_scores), 4) if command_scores else None,
            "negative_cases": sum(kind == "negative" for kind, _ in cases),
            "negative_scored": negative_scored,
            "accuracy_denominator": "successfully scored commands; incomplete runs cannot establish overall accuracy",
            "hallucinations": hallucinations,
            "median_latency_ms": round(statistics.median(case_latencies), 1) if case_latencies else None,
            "latency_cases_count": len(case_latencies),
        }
        write(summary)
        return summary
    except Exception as exc:
        write({"event": "run_error", "error": f"{type(exc).__name__}: {exc}"})
        raise
    finally:
        if report is not None:
            report.close()


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
