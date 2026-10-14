"""Leakage-safe, fixed-protocol evaluation of isolated wake classifiers.

This module never trains or promotes a model.  The single legacy recording is
held out by hash and every decision rule is fixed before scoring.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import wave
from io import BytesIO

SAMPLE_RATE = 16000
HOP = 1280
PREFIX = 2 * SAMPLE_RATE
SUFFIX = int(1.5 * SAMPLE_RATE)
THRESHOLD = 0.5
PATIENCE = 2
REFRACTORY_SECONDS = 2.0
PRIMARY_TAIL = 0.64
SECONDARY_TAIL = 1.5
MIN_FREE_BYTES = 1536 * 1024 * 1024
MAX_MODEL_BYTES = 16 * 1024 * 1024
MAX_AUDIO_BYTES = 2 * 1024 * 1024
MAX_META_BYTES = 64 * 1024
MAX_FRONTEND_BYTES = 64 * 1024 * 1024
MAX_DURATION = 60
BATCH_SIZE = 64
STOCK_MODEL = Path(r"A:/wake-data/models/alexa_v0.1.onnx")
CURRENT_MODEL = Path(__file__).parents[2] / "models" / "wake" / "alexa_user.onnx"
FRONTEND_DIR = Path(r"A:/wake-data/models")
HELDOUT_WAV = Path(r"A:/wake-data/user_recordings/alexas_with_music_20261002.wav")
HELDOUT_META = Path(r"A:/wake-data/user_recordings/alexas_with_music_20261002.json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def physical_free_bytes() -> int:
    """Read physical free RAM on Windows without importing numeric/ML stacks."""
    if os.name == "nt":
        import ctypes
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        state = MEMORYSTATUSEX()
        state.dwLength = ctypes.sizeof(state)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state)):
            raise OSError("GlobalMemoryStatusEx failed")
        return int(state.ullAvailPhys)
    import psutil
    return int(psutil.virtual_memory().available)


def load_holdout(wav_path: Path, meta_path: Path):
    if wav_path.stat().st_size > MAX_AUDIO_BYTES:
        raise ValueError("heldout WAV exceeds 2 MiB limit")
    if meta_path.stat().st_size > MAX_META_BYTES:
        raise ValueError("heldout metadata exceeds 64 KiB limit")
    with meta_path.open("rb") as stream:
        raw_meta = stream.read(MAX_META_BYTES + 1)
    if len(raw_meta) > MAX_META_BYTES:
        raise ValueError("heldout metadata exceeds 64 KiB limit")
    meta = json.loads(raw_meta)
    spans = meta.get("alexa_spans_s") if isinstance(meta, dict) else None
    if not isinstance(spans, list) or len(spans) != 19:
        raise ValueError("heldout metadata must contain exactly 19 manually provided spans")
    normalized = []
    for span in spans:
        if (not isinstance(span, (list, tuple)) or len(span) != 2
                or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in span)):
            raise ValueError("invalid heldout span")
        start, end = map(float, span)
        if start < 0 or end <= start:
            raise ValueError("invalid heldout span bounds")
        normalized.append((start, end))
    if normalized != sorted(normalized) or any(normalized[i][1] > normalized[i + 1][0]
                                                for i in range(len(normalized) - 1)):
        raise ValueError("heldout spans must be sorted and non-overlapping")
    with wav_path.open("rb") as stream:
        raw_wav = stream.read(MAX_AUDIO_BYTES + 1)
    if len(raw_wav) > MAX_AUDIO_BYTES:
        raise ValueError("heldout WAV exceeds 2 MiB limit")
    with wave.open(BytesIO(raw_wav), "rb") as wav:
        if (wav.getnchannels(), wav.getframerate(), wav.getsampwidth(), wav.getcomptype()) != (1, SAMPLE_RATE, 2, "NONE"):
            raise ValueError("heldout audio must be mono PCM16 at 16 kHz")
        frames = wav.getnframes()
        if frames <= 0 or frames > MAX_DURATION * SAMPLE_RATE:
            raise ValueError("heldout WAV duration must be at most 60 seconds")
        payload = wav.readframes(frames)
    if len(payload) != frames * 2:
        raise ValueError("truncated heldout WAV")
    duration = frames / SAMPLE_RATE
    if any(end > duration for _, end in normalized):
        raise ValueError("heldout span exceeds actual WAV duration")
    if isinstance(meta, dict):
        for key in ("sample_rate", "wav_sample_rate", "rate"):
            if key in meta and meta[key] != SAMPLE_RATE:
                raise ValueError(f"heldout metadata {key} disagrees with WAV")
        if "duration_s" in meta:
            legacy_duration = meta["duration_s"]
            rounded_duration = round(duration, 1)
            if (isinstance(legacy_duration, bool) or not isinstance(legacy_duration, (int, float))
                    or not math.isfinite(legacy_duration)
                    or abs(float(legacy_duration) - rounded_duration) > 1 / SAMPLE_RATE):
                raise ValueError("heldout metadata duration_s disagrees with rounded WAV duration")
        for key in ("wav_duration_s",):
            if key in meta and (isinstance(meta[key], bool) or not isinstance(meta[key], (int, float)) or
                                not math.isfinite(meta[key]) or abs(float(meta[key]) - duration) > 1 / SAMPLE_RATE):
                raise ValueError(f"heldout metadata {key} disagrees with WAV")
        for key in ("filename", "wav_filename", "audio_file", "file"):
            if key in meta and Path(str(meta[key])).name != wav_path.name:
                raise ValueError(f"heldout metadata {key} disagrees with WAV")
    return payload, frames, normalized, hashlib.sha256(raw_meta).hexdigest(), hashlib.sha256(raw_wav).hexdigest()


def load_candidate_metadata(candidate: Path, heldout_wav_sha256: str, candidate_sha256: str):
    sidecar = candidate.with_suffix(".meta.json")
    if sidecar.stat().st_size > MAX_META_BYTES:
        raise ValueError("candidate metadata exceeds 64 KiB limit")
    with sidecar.open("rb") as stream:
        raw = stream.read(MAX_META_BYTES + 1)
    meta = json.loads(raw)
    if (not isinstance(meta, dict) or meta.get("schema") != "memoir.wake-candidate-training-meta.v1"
            or meta.get("production_eligible") is not False):
        raise ValueError("candidate metadata must be valid training metadata with production_eligible=false")
    labels = meta.get("training_labels")
    if not isinstance(labels, dict) or labels.get("positive_source") != "machine_consensus":
        raise ValueError("candidate metadata does not prove expected isolated training provenance")
    if meta.get("output_sha256") != candidate_sha256 or meta.get("deployment_changed") is not False:
        raise ValueError("candidate model hash or deployment provenance does not match metadata")
    source_hashes = meta.get("source_sha256")
    if not isinstance(source_hashes, dict):
        raise ValueError("candidate metadata must include training source_sha256 hashes")
    if any(not isinstance(value, str) or len(value) != 64 or
           any(ch not in "0123456789abcdef" for ch in value.casefold()) for value in source_hashes.values()):
        raise ValueError("candidate training source hashes must be SHA-256 hex values")
    if heldout_wav_sha256.casefold() in {value.casefold() for value in source_hashes.values()}:
        raise ValueError("heldout WAV hash appears in candidate training sources; leakage rejected")
    return meta, hashlib.sha256(raw).hexdigest()


def validate_scores(scores, expected_count: int):
    import numpy as np
    arr = np.asarray(scores)
    if arr.shape != (expected_count, 1):
        raise ValueError(f"classifier output shape must be ({expected_count}, 1)")
    flat = arr[:, 0]
    if not np.isfinite(flat).all() or (flat < 0).any() or (flat > 1).any():
        raise ValueError("classifier outputs must be finite probabilities in [0, 1]")
    return flat.astype(np.float32, copy=False)


def confirmed_events(scores, timestamps, *, threshold=THRESHOLD, patience=PATIENCE):
    """Emit once per contiguous above-threshold run, at its confirmation chunk."""
    events, start = [], None
    for i, score in enumerate(scores):
        if score >= threshold:
            if start is None:
                start = i
            if i == start + patience - 1:
                events.append({"index": i, "run_start_index": start, "time_s": float(timestamps[i]),
                               "score": float(score)})
        else:
            start = None
    return events


def apply_refractory(events, seconds=REFRACTORY_SECONDS):
    kept, last = [], -math.inf
    for event in events:
        timestamp = event["time_s"]
        if timestamp < 0:
            continue
        if timestamp - last >= seconds:
            kept.append(event)
            last = timestamp
    return kept


def match_events(events, spans, tail):
    """Greedily match ordered events to earliest-deadline eligible unmatched span."""
    available = set(range(len(spans)))
    matches, unmatched_events = [], []
    for event in sorted(events, key=lambda item: (item["time_s"], item["index"])):
        eligible = [i for i in available if spans[i][0] <= event["time_s"] <= spans[i][1] + tail]
        if not eligible:
            unmatched_events.append(event)
            continue
        idx = min(eligible, key=lambda i: (spans[i][1] + tail, spans[i][0], i))
        available.remove(idx)
        matches.append({"event": event, "span_index": idx,
                        "latency_from_span_end_s": event["time_s"] - spans[idx][1]})
    return matches, sorted(available), unmatched_events


def extract_features(pcm_bytes, frames, *, feature_factory=None):
    import numpy as np
    samples = np.frombuffer(pcm_bytes, dtype="<i2")
    if len(samples) != frames:
        raise ValueError("PCM frame count mismatch")
    stream = np.concatenate((np.zeros(PREFIX, np.int16), samples,
                             np.zeros(SUFFIX, np.int16)))
    if feature_factory is None:
        from openwakeword.utils import AudioFeatures
        feature_factory = lambda: AudioFeatures(
            melspec_model_path=str(FRONTEND_DIR / "melspectrogram.onnx"),
            embedding_model_path=str(FRONTEND_DIR / "embedding_model.onnx"),
            inference_framework="onnx", ncpu=1, device="cpu")
    frontend = feature_factory()
    if hasattr(frontend, "reset"):
        frontend.reset()
    features, times = [], []
    for start in range(0, len(stream), HOP):
        hop = stream[start:start + HOP]
        if len(hop) < HOP:
            hop = np.pad(hop, (0, HOP - len(hop)))
        frontend(hop)
        item = np.asarray(frontend.get_features(16))
        if item.shape == (1, 16, 96):
            item = item[0]
        if item.shape != (16, 96) or not np.isfinite(item).all():
            raise ValueError("frontend must return finite (16, 96) windows")
        features.append(item.astype(np.float32, copy=True))
        # AudioFeatures consumes a complete padded hop, including at EOF.
        consumed = start + HOP
        times.append((consumed - PREFIX) / SAMPLE_RATE)
    return np.stack(features), np.asarray(times, dtype=np.float64)


def score_session(model_path: Path, windows):
    import numpy as np
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = opts.inter_op_num_threads = 1
    session = ort.InferenceSession(str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])
    input_info, output_info = session.get_inputs()[0], session.get_outputs()[0]
    inp, out = input_info.name, output_info.name
    shape = input_info.shape
    if len(shape) != 3 or shape[1:] != [16, 96]:
        raise ValueError("classifier input must accept (batch, 16, 96) windows")
    batch_size = shape[0] if isinstance(shape[0], int) and shape[0] > 0 else BATCH_SIZE
    results = []
    for start in range(0, len(windows), batch_size):
        batch = np.asarray(windows[start:start + batch_size], dtype=np.float32)
        results.extend(validate_scores(session.run([out], {inp: batch})[0], len(batch)))
    return np.asarray(results, dtype=np.float32)


def _file_fingerprints(paths):
    return {str(Path(path).resolve(strict=True)): sha256_file(Path(path)) for path in paths}


def _verify_parsed_holdout(before, wav_path, meta_path, wav_digest, metadata_digest):
    if (wav_digest != before[str(Path(wav_path).resolve(strict=True))]
            or metadata_digest != before[str(Path(meta_path).resolve(strict=True))]):
        raise ValueError("heldout WAV or metadata changed between parsing and initial fingerprint")


def _ensure_safe_paths(candidate: Path, output: Path, protected_paths=()):
    production_root = (Path(__file__).parents[2] / "models" / "wake").resolve()
    candidate, output = Path(candidate).resolve(), Path(output).resolve()
    protected = {Path(path).resolve() for path in protected_paths}
    protected.add(candidate.with_suffix(".meta.json"))
    if candidate.is_relative_to(production_root):
        raise ValueError("candidate must be outside assistant/models/wake")
    if output.is_relative_to(production_root):
        raise ValueError("output must be outside assistant/models/wake")
    if output == candidate or output in protected:
        raise ValueError("output may not overwrite a model, metadata, frontend, or heldout input")


def atomic_json_create(path: Path, report):
    path = Path(path)
    parent = path.parent
    fd, tmp_name = tempfile.mkstemp(prefix=".wake-eval-", suffix=".json", dir=parent)
    temp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def run(candidate_path, output_path, *, memory_gate=None, feature_factory=None):
    candidate_path, output_path = Path(candidate_path).resolve(strict=True), Path(output_path).resolve()
    if output_path.exists():
        raise FileExistsError(f"output already exists: {output_path}")
    if output_path.suffix.casefold() != ".json":
        raise ValueError("output must be a new .json file")
    if output_path == candidate_path or output_path == candidate_path.with_suffix(".meta.json"):
        raise ValueError("output may not overwrite candidate model or metadata")
    models = {"stock": STOCK_MODEL, "current": CURRENT_MODEL, "candidate": candidate_path}
    mel, embedding = FRONTEND_DIR / "melspectrogram.onnx", FRONTEND_DIR / "embedding_model.onnx"
    protected = {p.resolve() for p in (*models.values(), mel, embedding, HELDOUT_WAV, HELDOUT_META,
                                      candidate_path.with_suffix(".meta.json"),
                                      CURRENT_MODEL.with_name("alexa_user_stream_test.json"))}
    _ensure_safe_paths(candidate_path, output_path, protected)
    if any(not p.is_file() for p in (*models.values(), mel, embedding, HELDOUT_WAV, HELDOUT_META)):
        raise FileNotFoundError("required model, frontend, or fixed heldout input is missing")
    for name, path in models.items():
        if path.stat().st_size > MAX_MODEL_BYTES:
            raise ValueError(f"{name} classifier exceeds 16 MiB limit")
    for name, path in (("melspectrogram", mel), ("embedding_model", embedding)):
        if path.stat().st_size > MAX_FRONTEND_BYTES:
            raise ValueError(f"{name} frontend exceeds 64 MiB limit")
    # Verify metadata and fixed data before any ML dependency import.
    fingerprint_paths = [*models.values(), mel, embedding, HELDOUT_WAV, HELDOUT_META,
                         candidate_path.with_suffix(".meta.json")]
    before = _file_fingerprints(fingerprint_paths)
    pcm, frames, spans, metadata_digest, parsed_wav_digest = load_holdout(HELDOUT_WAV, HELDOUT_META)
    heldout_digest = before[str(HELDOUT_WAV.resolve(strict=True))]
    _verify_parsed_holdout(before, HELDOUT_WAV, HELDOUT_META, parsed_wav_digest, metadata_digest)
    candidate_model_digest = before[str(candidate_path)]
    candidate_meta, candidate_meta_digest = load_candidate_metadata(candidate_path, heldout_digest,
                                                                       candidate_model_digest)
    if candidate_meta_digest != before[str(candidate_path.with_suffix('.meta.json').resolve(strict=True))]:
        raise ValueError("candidate metadata changed between parsing and initial fingerprint")
    gate = memory_gate or physical_free_bytes
    free = int(gate())
    if free < MIN_FREE_BYTES:
        raise RuntimeError("at least 1536 MiB physical free RAM is required before ML imports")
    import numpy as np
    np.random.seed(1337)
    windows, timestamps = extract_features(pcm, frames, feature_factory=feature_factory)
    model_scores = {name: score_session(path, windows) for name, path in models.items()}
    raw, kept, primary, secondary = {}, {}, {}, {}
    for name, scores in model_scores.items():
        events = confirmed_events(scores, timestamps)
        # Include only frames up to the fixed recording plus the fixed suffix.
        events = [item for item in events if 0 <= item["time_s"] <= frames / SAMPLE_RATE + SUFFIX / SAMPLE_RATE]
        raw[name] = events
        kept[name] = apply_refractory(events)
        primary[name] = match_events(kept[name], spans, PRIMARY_TAIL)
        secondary[name] = match_events(kept[name], spans, SECONDARY_TAIL)
    after = _file_fingerprints(fingerprint_paths)
    if before != after:
        raise ValueError("a model, frontend, heldout input, or metadata changed during evaluation")
    results = {}
    for name in models:
        matches, misses, unmatched = primary[name]
        smatches, smisses, sunmatched = secondary[name]
        matched_spans = {item["span_index"] for item in matches}
        duplicates = [e for e in unmatched if any(
            i in matched_spans and a <= e["time_s"] <= b + PRIMARY_TAIL
            for i, (a, b) in enumerate(spans))]
        late = [e for e in unmatched if e not in duplicates and any(
            a <= e["time_s"] <= b + SECONDARY_TAIL and e["time_s"] > b + PRIMARY_TAIL
            for a, b in spans)]
        results[name] = {
            "raw_confirmed_run_events": raw[name], "refractory_kept_events": kept[name],
            "primary": {"allowance_s": PRIMARY_TAIL, "hits": len(matches), "misses": len(misses),
                        "unmatched_activations": len(unmatched), "matches": matches,
                        "unmatched_span_indices": misses, "latencies_from_span_end_s": [m["latency_from_span_end_s"] for m in matches]},
            "secondary_descriptive": {"allowance_s": SECONDARY_TAIL, "hits": len(smatches),
                                       "misses": len(smisses), "unmatched_activations": len(sunmatched)},
            "late_activation_diagnostic": late,
            "duplicate_activation_diagnostic": duplicates,
            "unmatched_music_activations_are_not_confirmed_human_negatives": True,
        }
    report = {
        "schema": "memoir.wake-heldout-evaluation.v1", "production_eligible": False,
        "training_performed": False, "promotion_claim": False,
        "caveat": "19 manually provided legacy spans from one small single-room recording; not general accuracy evidence.",
        "fixed_parameters": {"sample_rate": SAMPLE_RATE, "hop_samples": HOP, "prefix_zero_samples": PREFIX,
                             "suffix_zero_samples": SUFFIX, "threshold": THRESHOLD, "patience_chunks": PATIENCE,
                             "refractory_seconds": REFRACTORY_SECONDS, "primary_tail_seconds": PRIMARY_TAIL,
                             "secondary_tail_seconds": SECONDARY_TAIL, "batch_size": BATCH_SIZE,
                             "seed": 1337,
                             "max_considered_time_s": frames / SAMPLE_RATE + SUFFIX / SAMPLE_RATE},
        "heldout": {"wav_sha256": heldout_digest, "metadata_sha256": metadata_digest,
                    "duration_s": frames / SAMPLE_RATE, "span_count": len(spans), "spans_s": spans},
        "candidate_metadata_sha256": candidate_meta_digest,
        "model_sha256": {name: before[str(path.resolve(strict=True))] for name, path in models.items()},
        "frontend_sha256": {"melspectrogram": before[str(mel.resolve(strict=True))],
                             "embedding_model": before[str(embedding.resolve(strict=True))]},
        "input_sha256_after": {"wav": after[str(HELDOUT_WAV.resolve(strict=True))],
                               "metadata": after[str(HELDOUT_META.resolve(strict=True))]},
        "results": results,
    }
    if before != _file_fingerprints(fingerprint_paths):
        raise ValueError("a model, frontend, heldout input, or metadata changed before report publication")
    atomic_json_create(output_path, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    run(args.candidate, args.output)


if __name__ == "__main__":
    main()
