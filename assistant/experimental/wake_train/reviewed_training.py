"""Human-reviewed span candidate experiment, isolated from Memoir production."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

try:
    from . import active_learning as review
    from . import candidate_training as candidate
except ImportError:
    import active_learning as review
    import candidate_training as candidate

MAX_TRAIN_BAGS_PER_CLASS = 128
MAX_TOTAL_SECONDS = 600
MAX_WINDOWS = 8192
MAX_CLIP_SECONDS = 24
MAX_WAV_BYTES = 1024 * 1024
MAX_STORE_ROWS_FOR_PREFLIGHT = 512
MIN_FREE_BYTES = 1536 * 1024 * 1024
FEATURE_FLOAT32_BYTES = 16 * 96 * 4


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _preflight_store(store, audio_root):
    if (not isinstance(store, dict) or store.get("schema") != review.STORE_SCHEMA
            or not isinstance(store.get("reviews"), list)):
        raise ValueError("invalid human review store schema")
    rows = store["reviews"]
    if len(rows) > MAX_STORE_ROWS_FOR_PREFLIGHT:
        raise ValueError("review store exceeds 512-row preflight limit")
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("invalid review row")
        frames = row.get("n_frames")
        if not _is_int(frames) or not 0 < frames <= MAX_CLIP_SECONDS * 16000:
            raise ValueError("reviewed clip exceeds 24-second preflight limit")
        try:
            path = review.audio_path(row.get("path"), audio_root)
        except FileNotFoundError:
            # Missing historical recordings are counted by review_history.
            continue
        if path.stat().st_size > MAX_WAV_BYTES:
            raise ValueError("reviewed WAV exceeds 1 MiB preflight limit")


def _hop_window_count(sample_count):
    padded_samples = 32000 + sample_count + 12800
    return (padded_samples + candidate.HOP - 1) // candidate.HOP


def _retained_window_count(sample_count):
    hop_count = _hop_window_count(sample_count)
    ends = [index * candidate.HOP for index in range(1, hop_count + 1)]
    return len(candidate.retained_window_indices(ends, sample_count))


def _plan_row(row):
    start, end = ((row["wake_start_sample"], row["wake_end_sample"])
                  if row["presence"] == "positive" else (0, row["n_frames"]))
    selected_samples = end - start
    retained = _retained_window_count(selected_samples)
    hops = _hop_window_count(selected_samples)
    return {**row, "source_start_sample": start, "source_end_sample": end,
            "selected_sample_count": selected_samples, "retained_window_count": retained,
            "extractor_hop_count": hops,
            "temporary_float32_bytes": (hops + retained) * FEATURE_FLOAT32_BYTES}


def _row_identity(row):
    return (row["wav_sha256"], row["source_start_sample"], row["source_end_sample"], row["group_id"])


def _session_aware_selection(rows, label):
    by_group = {}
    for row in rows:
        by_group.setdefault(row["group_id"], []).append(row)
    if len(by_group) > MAX_TRAIN_BAGS_PER_CLASS:
        raise ValueError(f"{label} training spans more than 128 independent session groups")
    for group_rows in by_group.values():
        group_rows.sort(key=lambda row: (row["wav_sha256"], row["source_start_sample"],
                                         row["source_end_sample"]))
    groups = sorted(by_group)
    selected = []
    depth = 0
    while len(selected) < MAX_TRAIN_BAGS_PER_CLASS:
        added = False
        for group in groups:
            group_rows = by_group[group]
            if depth < len(group_rows):
                selected.append(group_rows[depth])
                added = True
                if len(selected) == MAX_TRAIN_BAGS_PER_CLASS:
                    break
        if not added:
            break
        depth += 1
    selected_ids = {_row_identity(row) for row in selected}
    excluded = [row for row in rows if _row_identity(row) not in selected_ids]
    descriptors = sorted(({"wav_sha256": row["wav_sha256"],
                           "start": row["source_start_sample"], "end": row["source_end_sample"],
                           "group_id": row["group_id"]} for row in excluded),
                         key=lambda item: (item["group_id"], item["wav_sha256"], item["start"], item["end"]))
    exclusion_hash = hashlib.sha256(json.dumps(descriptors, sort_keys=True,
                                               separators=(",", ":")).encode()).hexdigest()
    details = {f"{label}_train_input_count": len(rows),
               f"{label}_train_selected_count": len(selected),
               f"{label}_train_excluded_count": len(excluded),
               f"{label}_train_excluded_sha256": exclusion_hash,
               f"{label}_train_excluded_groups": sorted({row["group_id"] for row in excluded}),
               f"{label}_train_selected_groups": sorted({row["group_id"] for row in selected}),
               f"{label}_train_excluded_records": [{"label": label, **item} for item in descriptors]}
    return selected, excluded, details


def _inventory(store_path, audio_root):
    store, store_hash = review.load_preparation_store(store_path)
    _preflight_store(store, audio_root)
    metadata, usable, unavailable = review.review_history(store, audio_root)
    review.validate_session_provenance(usable)
    for row in usable:
        group_id = row["group_id"]
        if not isinstance(group_id, str) or not group_id.strip() or len(group_id) > 128:
            raise ValueError("reviewed session group id must be a non-empty string up to 128 characters")
    rows = [_plan_row(row) for row in usable if row["presence"] != "unsure"]
    unique_sources = {row["wav_sha256"]: row["n_frames"] for row in usable}
    total_seconds = sum(unique_sources.values()) / 16000
    # Require each label on both sides of a whole-session partition. Unknown
    # legacy negative pools are train-only and can never supply holdout data.
    groups = {split: {label: set() for label in ("positive", "negative")}
              for split in ("train", "holdout")}
    reasons = []
    for row in rows:
        if row["presence"] in ("positive", "negative"):
            groups[row["split"]][row["presence"]].add(row["group_id"])
    for split in ("train", "holdout"):
        for label in ("positive", "negative"):
            if not groups[split][label]:
                reasons.append(f"missing {split} {label} session groups")
    train_groups = groups["train"]["positive"] | groups["train"]["negative"]
    holdout_groups = groups["holdout"]["positive"] | groups["holdout"]["negative"]
    if train_groups & holdout_groups:
        reasons.append("capture session group crosses train and holdout")
    train_pos_all = [row for row in rows if row["presence"] == "positive" and row["split"] == "train"]
    train_neg_all = [row for row in rows if row["presence"] == "negative" and row["split"] == "train"]
    holdout_rows = [row for row in rows if row["split"] == "holdout"]
    train_pos, excluded_pos, selection_pos = _session_aware_selection(train_pos_all, "positive")
    train_neg, excluded_neg, selection_neg = _session_aware_selection(train_neg_all, "negative")
    selected_ids = {_row_identity(row) for row in train_pos + train_neg}
    for row in rows:
        row["selected_for_training"] = row["split"] == "train" and _row_identity(row) in selected_ids
    train_windows = sum(row["retained_window_count"] for row in train_pos + train_neg)
    holdout_windows = sum(row["retained_window_count"] for row in holdout_rows)
    if total_seconds > MAX_TOTAL_SECONDS:
        reasons.append("unique reviewed audio exceeds 600-second limit")
    if train_windows > MAX_WINDOWS:
        reasons.append("retained training windows exceed 8192-window limit")
    peak_temp_bytes = max((row["temporary_float32_bytes"] for row in train_pos + train_neg), default=0)
    selection = {**selection_pos, **selection_neg}
    excluded_records = (selection_pos["positive_train_excluded_records"]
                        + selection_neg["negative_train_excluded_records"])
    report = {
        "schema": "memoir.wake-reviewed-training-plan.v1",
        "data_eligible": not reasons,
        "training_ready": False,
        "training_supported": True,
        "resource_admission_checked": False,
        "feature_extraction_started": False,
        "training_started": False,
        "model_trained": False,
        "blocked_reasons": list(dict.fromkeys(reasons)),
        "review_metadata_rows": len(metadata),
        "usable_recordings": len(usable),
        "missing_historic_recordings": [str(Path(row["path"])) for row in unavailable],
        "excluded_unsure": sum(r["presence"] == "unsure" for r in usable),
        "positive_train_count": len(train_pos_all),
        "positive_holdout_count": sum(r["presence"] == "positive" and r["split"] == "holdout" for r in rows),
        "negative_train_count": len(train_neg_all),
        "negative_holdout_count": sum(r["presence"] == "negative" and r["split"] == "holdout" for r in rows),
        "positive_train_groups": sorted(groups["train"]["positive"]),
        "positive_holdout_groups": sorted(groups["holdout"]["positive"]),
        "negative_train_groups": sorted(groups["train"]["negative"]),
        "negative_holdout_groups": sorted(groups["holdout"]["negative"]),
        "review_store_sha256": store_hash,
        "unique_audio_seconds": total_seconds,
        "window_upper_bound": train_windows,
        "train_retained_windows": train_windows,
        "holdout_planned_retained_windows": holdout_windows,
        "retained_train_float32_bytes": train_windows * FEATURE_FLOAT32_BYTES,
        "peak_train_clip_temporary_float32_bytes": peak_temp_bytes,
        "train_selection": {key: value for key, value in selection.items()
                            if not key.endswith("_excluded_records")},
        "reviewed_training_selection": {
            "algorithm": "session_round_robin_sha256_v1",
            "per_class_limit": MAX_TRAIN_BAGS_PER_CLASS,
            "selected_counts": {"positive": len(train_pos), "negative": len(train_neg)},
            "excluded_records": excluded_records,
        },
        "resource_limits": {"train_bags_per_class": MAX_TRAIN_BAGS_PER_CLASS,
                             "max_clip_seconds": MAX_CLIP_SECONDS,
                             "max_total_seconds": MAX_TOTAL_SECONDS,
                             "max_windows": MAX_WINDOWS,
                             "minimum_free_bytes": MIN_FREE_BYTES},
        "limitation": review.PROPOSAL_LIMITATION,
    }
    # The plan contains only fields used by the extractor, never reviewer
    # notes or transcripts.
    safe_rows = [{key: row[key] for key in (
        "path", "wav_sha256", "n_frames", "sample_rate", "presence", "split", "group_id",
        "source_start_sample", "source_end_sample", "selected_sample_count", "retained_window_count",
        "extractor_hop_count", "temporary_float32_bytes", "selected_for_training",
        "session_provenance", "capture_session_id", "segment_id", "overlap_of") if key in row} for row in rows]
    return {"rows": safe_rows, "report": report}, store


def prepare(store_path, *, audio_root=review.AUDIO_ROOT):
    """Return a read-only eligibility plan and a transcript-free summary."""
    plan, _ = _inventory(store_path, audio_root)
    return plan, plan["report"]


def run(store_path, output_path, *, epochs=8, seed=1337, memory_gate=None, feature_factory=None):
    """Train from selected reviewed spans; injected feature factories are called once and reused.

    The reused frontend must provide reset() when it keeps state; extract_bag
    resets it before each clip.
    """
    if not _is_int(epochs) or not 1 <= epochs <= 8:
        raise ValueError("epochs must be between 1 and 8")
    if not _is_int(seed) or not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be an unsigned 32-bit integer")
    target = review.candidate_output(output_path)
    plan, store = _inventory(store_path, review.AUDIO_ROOT)
    report = plan["report"]
    if not report["data_eligible"]:
        raise ValueError("reviewed training blocked: " + "; ".join(report["blocked_reasons"]))
    rows = plan["rows"]
    train_pos = [r for r in rows if r["presence"] == "positive" and r["split"] == "train"
                 and r["selected_for_training"]]
    train_neg = [r for r in rows if r["presence"] == "negative" and r["split"] == "train"
                 and r["selected_for_training"]]
    if report["unique_audio_seconds"] > MAX_TOTAL_SECONDS:
        raise ValueError("unique reviewed audio exceeds 600-second limit")
    if report["train_retained_windows"] > MAX_WINDOWS:
        raise ValueError("retained training windows exceed 8192-window limit")

    # All source bytes, clip dimensions, and hashes are checked before imports
    # of numpy/ML or creation of the candidate output directory.
    source_paths = {Path(store_path).resolve(strict=True), candidate.STOCK_MODEL.resolve(strict=True),
                    (candidate.FRONTEND_DIR / "melspectrogram.onnx").resolve(strict=True),
                    (candidate.FRONTEND_DIR / "embedding_model.onnx").resolve(strict=True)}
    for row in rows:
        path = Path(row["path"]).resolve(strict=True)
        if path.stat().st_size > MAX_WAV_BYTES or row["n_frames"] > MAX_CLIP_SECONDS * 16000:
            raise ValueError("reviewed WAV exceeds per-source byte or duration limit")
        source_paths.add(path)
    source_hashes = candidate._check_sources(source_paths)
    for row in rows:
        if source_hashes[str(Path(row["path"]).resolve())] != row["wav_sha256"]:
            raise ValueError("training WAV changed after review identity check")
    _, store_digest = review.load_preparation_store(store_path)
    if store_digest != report["review_store_sha256"]:
        raise ValueError("human review store changed after parsing")
    if memory_gate is None:
        from .machine_proposals import physical_free_bytes
        memory_gate = physical_free_bytes
    if memory_gate() < MIN_FREE_BYTES:
        raise RuntimeError("need at least 1536 MiB physical memory before loading ML dependencies")

    # Import only after every dataset and memory admission gate has passed.
    import numpy as np
    bags_pos, bags_neg, total_windows = [], [], 0
    span_records = []
    frontend_holder = []
    def create_frontend():
        if feature_factory is not None:
            return feature_factory()
        from openwakeword.utils import AudioFeatures
        return AudioFeatures(
            melspec_model_path=str(candidate.FRONTEND_DIR / "melspectrogram.onnx"),
            embedding_model_path=str(candidate.FRONTEND_DIR / "embedding_model.onnx"),
            inference_framework="onnx", ncpu=1, device="cpu")

    def reused_frontend_factory():
        if not frontend_holder:
            frontend_holder.append(create_frontend())
        return frontend_holder[0]

    for row, is_positive in ([(r, True) for r in train_pos] + [(r, False) for r in train_neg]):
        samples = candidate._read_pcm(Path(row["path"]))
        if is_positive:
            first, last = row["source_start_sample"], row["source_end_sample"]
            selected = samples[first:last]
            if not len(selected):
                raise ValueError("reviewed positive span is empty")
            span_records.append({"wav_sha256": row["wav_sha256"], "start": first, "end": last,
                                 "group_id": row["group_id"], "split": row["split"]})
        else:
            selected = samples
        np.random.seed(seed ^ int(row["wav_sha256"][:8], 16))
        bag = candidate.extract_bag(selected, feature_factory=reused_frontend_factory)
        total_windows += len(bag)
        if total_windows > MAX_WINDOWS:
            raise ValueError("extracted feature windows exceed 8192-window limit")
        (bags_pos if is_positive else bags_neg).append(bag)
    if candidate._check_sources(source_paths) != source_hashes:
        raise ValueError("training source changed during feature extraction")
    span_digest = hashlib.sha256(json.dumps(sorted(span_records, key=lambda x: (x["wav_sha256"], x["start"])),
                                                sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    target.parent.mkdir(parents=True, exist_ok=True)
    extra = {"positive_span_plan_sha256": span_digest, "positive_span_count": len(span_records),
             "train_positive_count": len(train_pos), "train_negative_count": len(train_neg),
             "positive_holdout_excluded": report["positive_holdout_count"],
             "negative_holdout_excluded": report["negative_holdout_count"],
             "unsure_excluded": report["excluded_unsure"],
             "missing_historic_audio_excluded": len(report["missing_historic_recordings"]),
             "human_review_metadata_rows": report["review_metadata_rows"],
             "reviewed_training_selection": report["reviewed_training_selection"]}
    return candidate.fit_and_export(
        bags_pos, bags_neg, target, seed=seed, epochs=epochs, source_hashes=source_hashes,
        positive_groups=report["positive_train_groups"], report_extra=extra,
        label_source="human_reviewed_spans")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)
    if args.prepare_only:
        _, result = prepare(args.store)
    else:
        result = run(args.store, args.out, epochs=args.epochs, seed=args.seed)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
