#!/usr/bin/env python3
"""Bounded diagnostic comparison on reviewed and recent archived wake audio.

This report is intentionally separate from consensus labels, training manifests,
and production eligibility. Human annotations are evaluation expectations only;
machine output never changes those annotations or the review store.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import wave

try:
    from . import active_learning as review
    from . import consensus_labels as consensus
except ImportError:  # direct script execution
    import active_learning as review
    import consensus_labels as consensus

SCHEMA = "memoir.wake-reviewed-audio-eval.v1"
MAX_PER_GROUP = 4
MAX_CLIPS = 12
DEFAULT_STORE = consensus.DEFAULT_STORE
DEFAULT_TELEMETRY = review.TELEMETRY
OUTPUT_ROOT = consensus.OUTPUT_ROOT


def _safe_output(path):
    """Use consensus' existing new-file-only pilot output guard."""
    return consensus._safe_output(path)


def _review_cases(history):
    """Return verified active human-presence and strict-negative rows only."""
    rows = []
    for row in history["usable"]:
        if row.get("human_reviewed") is not True:
            continue
        if row.get("presence") == "unsure" and row.get("wake_presence") == "present":
            kind, expected, provenance = "human_presence_only", True, "human_wake_presence_present"
        elif row.get("presence") == "negative" and row.get("entire_clip_negative") is True:
            kind, expected, provenance = "human_confirmed_negative", False, "human_entire_clip_negative"
        else:
            continue
        rows.append({
            "kind": kind,
            "expected_presence": expected,
            "source_label_provenance": provenance,
            "source_event_id": row["event_id"],
            "audio_field": row["audio_field"],
            "path": row["path"],
            "wav_sha256": row["wav_sha256"],
            "n_frames": row["n_frames"],
            "sample_rate": row["sample_rate"],
            "duration_seconds": row["n_frames"] / row["sample_rate"],
            "session_id": row.get("session_id"),
            "capture_session_id": row.get("capture_session_id") or row.get("session_id"),
            "session_partition_assignment": row.get("split"),
            "session_partition_basis": "active_learning.session_split",
        })
    return rows


def _recent_unreviewed_cases(*, telemetry, history, now=None,
                             audio_root=review.AUDIO_ROOT,
                             project_root=review.PROJECT_ROOT,
                             identity_fn=review.wav_identity,
                             exclude_hashes=()):
    events, stats = consensus._bounded_jsonl(telemetry, now=now)
    _accepted, fresh, _unknown = consensus._candidate_pool(events)
    reviewed_hashes = history["reviewed_hashes"]
    seen = set(reviewed_hashes) | set(exclude_hashes)
    selected = []
    counts = Counter()
    for event in fresh:
        if len(selected) >= MAX_PER_GROUP:
            break
        if event.get("recording_truncated") is True or event.get("capture_truncated") is True:
            counts["truncated"] += 1
            continue
        field, value = consensus._source_value(
            event, audio_root=audio_root, project_root=project_root)
        if not field or not value:
            counts["missing_or_invalid_audio"] += 1
            continue
        try:
            path = review.audio_path(value, audio_root, project_root)
            identity = identity_fn(path)
        except (OSError, ValueError, TypeError, KeyError, wave.Error, EOFError):
            counts["missing_or_invalid_audio"] += 1
            continue
        digest = identity["wav_sha256"]
        if digest in seen:
            counts["reviewed_or_duplicate_hash"] += 1
            continue
        expected_digest = consensus._logged_audio_hash(event, field, value)
        if expected_digest and expected_digest.casefold() != digest.casefold():
            counts["logged_hash_mismatch"] += 1
            continue
        if identity["sample_rate"] != 16000 or identity["n_frames"] / 16000 > consensus.MAX_CLIP_SECONDS:
            counts["invalid_duration_or_rate"] += 1
            continue
        event_id = event.get("event_id")
        if not isinstance(event_id, str) or not event_id:
            counts["missing_event_id"] += 1
            continue
        seen.add(digest)
        selected.append({
            "kind": "recent_unreviewed_near_miss",
            "expected_presence": None,
            "source_label_provenance": "unreviewed_telemetry_candidate",
            "source_event_id": event_id,
            "audio_field": field,
            "path": str(path),
            "wav_sha256": digest,
            "n_frames": identity["n_frames"],
            "sample_rate": identity["sample_rate"],
            "duration_seconds": identity["n_frames"] / identity["sample_rate"],
            "session_id": event.get("capture_session_id"),
            "capture_session_id": event.get("capture_session_id"),
            "session_partition_assignment": (
                review.session_split(event["capture_session_id"])
                if isinstance(event.get("capture_session_id"), str) and event["capture_session_id"]
                else None),
            "session_partition_basis": "active_learning.session_split" if event.get("capture_session_id") else "unknown_session",
            "trigger_source": event.get("trigger_source"),
            "candidate_kind": event.get("candidate_kind"),
            "end_reason": event.get("end_reason") or event.get("rejection_reason"),
            "event_ts": event.get("ts"),
        })
    return selected, {**stats, **dict(counts)}


def _bounded_selection(history, *, telemetry, max_clips, now=None,
                       audio_root=review.AUDIO_ROOT,
                       project_root=review.PROJECT_ROOT,
                       identity_fn=review.wav_identity):
    if isinstance(max_clips, bool) or not isinstance(max_clips, int) or not 1 <= max_clips <= MAX_CLIPS:
        raise ValueError("max_clips must be an integer from 1 through 12")

    human = _review_cases(history)
    sort_key = lambda row: (str(row.get("session_id") or ""), row["source_event_id"])
    reviewed_hashes = {row["wav_sha256"] for row in human}

    def unique_limited(rows, seen_hashes):
        selected = []
        for row in sorted(rows, key=sort_key):
            digest = row["wav_sha256"]
            if digest in seen_hashes:
                continue
            seen_hashes.add(digest)
            selected.append(row)
            if len(selected) == MAX_PER_GROUP:
                break
        return selected

    seen_hashes = set()
    positive = unique_limited(
        (row for row in human if row["kind"] == "human_presence_only"), seen_hashes)
    near_misses, telemetry_stats = _recent_unreviewed_cases(
        telemetry=telemetry, history=history, now=now, audio_root=audio_root,
        project_root=project_root, identity_fn=identity_fn,
        exclude_hashes=reviewed_hashes | seen_hashes)
    seen_hashes.update(row["wav_sha256"] for row in near_misses)
    negatives = unique_limited(
        (row for row in human if row["kind"] == "human_confirmed_negative"), seen_hashes)

    # Preserve the requested priority when the caller lowers the overall cap.
    chosen = (positive + near_misses + negatives)[:max_clips]
    groups = {row["kind"]: sum(item["kind"] == row["kind"] for item in chosen)
              for row in chosen}
    positive_groups = {row.get("capture_session_id") or row.get("session_id")
                       for row in chosen if row["kind"] == "human_presence_only"}
    return chosen, {
        "requested_max_clips": max_clips,
        "selected_total": len(chosen),
        "selected_by_kind": dict(groups),
        "selected_reviewed_presence_only": sum(row["kind"] == "human_presence_only" for row in chosen),
        "selected_reviewed_negatives": sum(row["kind"] == "human_confirmed_negative" for row in chosen),
        "selected_recent_unreviewed_near_misses": sum(row["kind"] == "recent_unreviewed_near_miss" for row in chosen),
        "positive_capture_group_count": len(positive_groups),
        "positive_clips_share_capture_group": bool(positive_groups) and len(positive_groups) == 1,
        "presence_positive_gap": not positive,
        "telemetry_scan": telemetry_stats,
        "heldout_claim": False,
    }


def _stage_record(case, index):
    return {
        "event_id": f"eval-{index:02d}-{case['wav_sha256'][:8]}",
        "path": case["path"],
        "wav_sha256": case["wav_sha256"],
        "n_frames": case["n_frames"],
        "sample_rate": case["sample_rate"],
        "duration_seconds": case["duration_seconds"],
    }


def _teacher_diagnostics(case_rows, moonshine, whisper, *, include_transcripts=False):
    stages = {"moonshine": moonshine, "whisper": whisper}
    expected_case_ids = {
        f"eval-{index:02d}-{row['wav_sha256'][:8]}"
        for index, row in enumerate(case_rows, start=1)
    }
    by_stage = {}
    for name, stage in stages.items():
        if not isinstance(stage, dict) or stage.get("stage") != name or not isinstance(stage.get("records"), list):
            raise ValueError(f"invalid {name} diagnostic stage result")
        by_id = {}
        for item in stage["records"]:
            case_id = item.get("event_id")
            if not isinstance(case_id, str) or case_id in by_id:
                raise ValueError(f"duplicate or missing {name} case identity")
            by_id[case_id] = item
        if set(by_id) != expected_case_ids:
            raise ValueError(f"{name} results do not exactly cover selected diagnostic cases")
        by_stage[name] = by_id

    rows = []
    confusion = {name: Counter() for name in stages}
    disagreements = []
    for index, source in enumerate(case_rows, start=1):
        case_id = f"eval-{index:02d}-{source['wav_sha256'][:8]}"
        row = {
            "case_id": case_id,
            "kind": source["kind"],
            "wav_sha256": source["wav_sha256"],
            "expected_presence": source["expected_presence"],
            "expected_presence_source": source["source_label_provenance"],
            "diagnostic_label": "unassigned",
            "training_eligible": False,
            "session_id": source.get("session_id"),
            "capture_session_id": source.get("capture_session_id"),
            "session_partition_assignment": source.get("session_partition_assignment"),
            "session_partition_basis": source.get("session_partition_basis"),
            "source_event_id": source["source_event_id"],
            "audio_field": source["audio_field"],
            "teacher_outcomes": {},
        }
        for name in stages:
            item = by_stage[name].get(case_id)
            if (item is None or item.get("wav_sha256") != source["wav_sha256"]
                    or item.get("n_frames") != source["n_frames"]
                    or item.get("sample_rate") != source["sample_rate"]
                    or item.get("source") != "offline_same_wav_inference"):
                raise ValueError(f"{name} result does not match selected WAV identity")
            text = item.get("text")
            if not isinstance(text, str):
                raise ValueError(f"{name} result has no transcript text")
            if type(item.get("text_truncated")) is not bool:
                raise ValueError(f"{name} result has invalid transcript truncation metadata")
            exact = consensus.exact_alexa(text)
            if item.get("exact_alexa") is not exact:
                raise ValueError(f"{name} exact-token result disagrees with transcript")
            truncated = item.get("text_truncated") is True
            detected = None if truncated else exact
            outcome = {
                "exact_token_detected": detected,
                "transcript_truncated": truncated,
                "elapsed_seconds": item.get("elapsed_seconds"),
            }
            if include_transcripts:
                outcome["transcript"] = text
            row["teacher_outcomes"][name] = outcome
            expected = source["expected_presence"]
            if expected is not None and detected is not None:
                if detected and expected:
                    confusion[name]["true_positive"] += 1
                elif detected and not expected:
                    confusion[name]["false_positive"] += 1
                elif not detected and expected:
                    confusion[name]["false_negative"] += 1
                else:
                    confusion[name]["true_negative"] += 1
        left = row["teacher_outcomes"]["moonshine"]["exact_token_detected"]
        right = row["teacher_outcomes"]["whisper"]["exact_token_detected"]
        if left is not None and right is not None and left != right:
            disagreements.append(case_id)
        rows.append(row)

    summaries = {}
    for name, counts in confusion.items():
        summaries[name] = {
            **{key: counts[key] for key in ("true_positive", "false_positive", "true_negative", "false_negative")},
            "labelled_cases_scored": sum(counts.values()),
            "unlabelled_cases_excluded": sum(row["expected_presence"] is None for row in case_rows),
            "truncated_labelled_cases_excluded": sum(
                row["expected_presence"] is not None and row["teacher_outcomes"][name]["transcript_truncated"]
                for row in rows),
        }
    return rows, summaries, disagreements


def build_report(*, store_path=DEFAULT_STORE, telemetry=DEFAULT_TELEMETRY,
                 audio_root=review.AUDIO_ROOT, project_root=review.PROJECT_ROOT,
                 max_clips=MAX_CLIPS, run=False, include_transcripts=False,
                 private_output_authorized=False,
                 now=None, history_loader=consensus.load_human_history,
                 stage_runner=None, identity_fn=review.wav_identity):
    if include_transcripts and (not run or not private_output_authorized):
        raise ValueError("raw transcripts require --run, an explicit flag, and a private output path")
    history = history_loader(store_path, audio_root=audio_root, project_root=project_root)
    selected, selection = _bounded_selection(
        history, telemetry=telemetry, max_clips=max_clips, now=now,
        audio_root=audio_root, project_root=project_root, identity_fn=identity_fn)
    cases = []
    for index, source in enumerate(selected, start=1):
        identity = identity_fn(source["path"])
        if identity["wav_sha256"] != source["wav_sha256"] or identity["n_frames"] != source["n_frames"]:
            raise ValueError("selected WAV identity changed before evaluation")
        cases.append({**source, "case_id": f"eval-{index:02d}-{identity['wav_sha256'][:8]}"})

    report = {
        "schema": SCHEMA,
        "mode": "diagnostic_inference" if run else "metadata_only",
        "training_eligible": False,
        "production_eligible": False,
        "labels_written_to_human_store": False,
        "selection": selection,
        "records": [{
            "case_id": case["case_id"],
            "kind": case["kind"],
            "wav_sha256": case["wav_sha256"],
            "expected_presence": case["expected_presence"],
            "expected_presence_source": case["source_label_provenance"],
            "diagnostic_label": "unassigned",
            "training_eligible": False,
            "session_id": case.get("session_id"),
            "capture_session_id": case.get("capture_session_id"),
            "session_partition_assignment": case.get("session_partition_assignment"),
            "session_partition_basis": case.get("session_partition_basis"),
            "source_event_id": case["source_event_id"],
            "audio_field": case["audio_field"],
        } for case in cases],
        "teacher_summary": None,
        "teacher_disagreement_case_ids": [],
        "heldout_claim": False,
        "interpretation_limit": "Diagnostic exact-token comparison on selected reviewed recordings only; unreviewed candidates have no expected label, and this report is not training or held-out generalization evidence.",
    }
    if not run:
        return report

    if not selected:
        raise ValueError("no valid archived recordings are available for diagnostic inference")
    stage_records = [_stage_record(case, index) for index, case in enumerate(cases, start=1)]
    for case, stage_record in zip(cases, stage_records):
        if case["case_id"] != stage_record["event_id"]:
            # Case IDs are derived from the same ordered hash tuple.
            raise ValueError("internal diagnostic case identity mismatch")
    if stage_runner is None:
        moonshine, whisper = consensus.run_teacher_stages(
            stage_records, temp_parent=OUTPUT_ROOT,
            project_root=project_root, audio_root=audio_root)
    else:
        moonshine, whisper = consensus.run_teacher_stages(
            stage_records, temp_parent=OUTPUT_ROOT,
            project_root=project_root, audio_root=audio_root,
            stage_runner=stage_runner)

    # Recheck exact source bytes after both isolated stages and make sure the
    # active human history did not change while inference was running.
    for case in cases:
        consensus._hash_matches(case["path"], case["wav_sha256"], case["n_frames"])
    consensus.assert_human_history_unchanged(
        history, store_path, audio_root=audio_root, project_root=project_root)

    evaluated, summaries, disagreements = _teacher_diagnostics(
        cases, moonshine, whisper, include_transcripts=include_transcripts)
    report["records"] = evaluated
    report["teacher_summary"] = summaries
    report["teacher_disagreement_case_ids"] = disagreements
    report["model_provenance"] = {
        name: {key: value for key, value in result.items() if key != "records"}
        for name, result in (("moonshine", moonshine), ("whisper", whisper))
    }
    return report


def stdout_summary(report):
    """Privacy-minimal summary; never include audio paths or ASR transcripts."""
    return {
        "schema": report["schema"],
        "mode": report["mode"],
        "training_eligible": False,
        "production_eligible": False,
        "labels_written_to_human_store": False,
        "selected_total": report["selection"]["selected_total"],
        "selected_by_kind": report["selection"]["selected_by_kind"],
        "presence_positive_gap": report["selection"]["presence_positive_gap"],
        "cases": [{
            "case_id": row["case_id"],
            "wav_sha256_prefix": row["wav_sha256"][:8],
            "diagnostic_label": row.get("diagnostic_label", "unassigned"),
            "expected_presence": row["expected_presence"],
        "teacher_outcomes": ({
            name: {
                key: outcome.get(key)
                for key in ("exact_token_detected", "transcript_truncated", "elapsed_seconds")
            }
            for name, outcome in (row.get("teacher_outcomes") or {}).items()
        } or None),
        } for row in report["records"]],
        "teacher_summary": report["teacher_summary"],
        "teacher_disagreement_case_ids": report["teacher_disagreement_case_ids"],
        "heldout_claim": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", default=str(DEFAULT_STORE))
    parser.add_argument("--telemetry", default=str(DEFAULT_TELEMETRY))
    parser.add_argument("--audio-root", default=str(review.AUDIO_ROOT))
    parser.add_argument("--project-root", default=str(review.PROJECT_ROOT))
    parser.add_argument("--max-clips", type=int, default=MAX_CLIPS)
    parser.add_argument("--run", action="store_true", help="run the two pinned local CPU teachers serially")
    parser.add_argument("--output", help="new private JSON path under experimental/wake_train/proposals")
    parser.add_argument("--include-transcripts", action="store_true",
                        help="include raw ASR text only in the private --run output")
    args = parser.parse_args(argv)
    try:
        if args.output and not args.run:
            raise ValueError("--output is only permitted with --run; default output is metadata-only stdout")
        if args.run and not args.output:
            raise ValueError("--run requires an explicit new --output under proposals")
        if args.include_transcripts and (not args.run or not args.output):
            raise ValueError("--include-transcripts requires --run and a private output file")
        target = _safe_output(args.output) if args.output else None
        report = build_report(
            store_path=args.store, telemetry=args.telemetry,
            audio_root=args.audio_root, project_root=args.project_root,
            max_clips=args.max_clips, run=args.run,
            include_transcripts=args.include_transcripts,
            private_output_authorized=target is not None)
        if target is not None:
            target = _safe_output(target)
            review._atomic_json(target, report)
        print(json.dumps(stdout_summary(report), ensure_ascii=False, allow_nan=False))
        return 0
    except (OSError, ValueError, TypeError, KeyError, MemoryError, RuntimeError, wave.Error) as exc:
        parser.exit(2, f"wake reviewed-audio evaluation refused: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
