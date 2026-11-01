"""Tune wake thresholds on A; report only held-out B evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from assistant.config import Config, ROOT, load
from assistant.echo import EchoCanceller
from assistant.wake import CHUNK, WakeDetector
from .session import Alignment, FRAME, RATE, finalize, manifest_path, read_audio, read_jsonl

REPORTS = Path(__file__).parent / "reports"


def merge(spans, end):
    result = []
    for a, b in sorted((max(0, int(a)), min(end, int(b))) for a, b in spans if b > a):
        if b <= a:
            continue
        if result and a <= result[-1][1]:
            result[-1][1] = max(b, result[-1][1])
        else:
            result.append([a, b])
    return result


def inside(point, spans):
    return any(a <= point < b for a, b in spans)


@dataclass
class Session:
    path: Path
    header: dict
    cues: list[dict]
    mic: np.ndarray
    reference: np.ndarray
    alignment: Alignment

    @classmethod
    def open(cls, path):
        path = manifest_path(Path(path))
        # Rebuild from source so edits to a user invalid flag or sidecar cannot
        # silently reuse an earlier finalization.
        rows = finalize(path)
        header = next(r for r in rows if r["type"] == "session")
        data = header.get("alignment")
        if not data:
            raise ValueError(f"No confident alignment anchors: {path}")
        return cls(path, header, [r for r in rows if r["type"] == "cue"],
                   read_audio(path.parent / header["mic_wav"]),
                   read_audio(path.parent / header["loopback_wav"]),
                   Alignment(np.array(data["mic_samples"]), np.array(data["loopback_samples"])))

    @property
    def duration(self):
        return len(self.mic) // FRAME * FRAME

    @property
    def condition(self):
        return f"{self.header['mode']}; distance={self.header.get('distance') or 'unspecified'}; music={self.header.get('music') or 'unspecified'}"

    def spans(self):
        cues, excluded = [], list(self.header["discontinuities"])
        windows = []
        invalid = 0
        for index, row in enumerate(self.cues):
            start = row.get("cue_onset_sample", row["expected_mic_sample"])
            end = row.get("cue_offset_sample", start + self.header["cue_samples"])
            cues.append([start, end])
            if row.get("anchor"):
                if row["invalid"]:
                    # A missing anchor makes its surrounding segment unusable.
                    previous = self.cues[index - 1].get("cue_onset_sample", 0) if index else 0
                    following = self.cues[index + 1].get("cue_offset_sample", self.duration) if index + 1 < len(self.cues) else self.duration
                    excluded.append([previous, following])
                continue
            a, b = row["window_start_sample"], row["window_end_sample"]
            if row["invalid"] or b > self.duration:
                invalid += 1
                # When mic cue location is unknown, protect the search range as
                # well as the expected attempt window from false-wake scoring.
                pad = 2 * RATE if row.get("mic_cue_start") is None else 0
                excluded.append([a - pad, b + pad])
            else:
                windows.append([a, b])
        # No fabricated/padded speaker reference can count as AEC evidence.
        mic_start, mic_end = self.alignment.map([0, len(self.reference) - 1], inverse=True)
        excluded.extend([[0, int(np.ceil(mic_start))], [int(np.floor(mic_end)), self.duration]])
        excluded = merge(excluded, self.duration)
        # An out-of-reference region invalidates the entire positive attempt.
        retained = []
        for a, b in windows:
            if any(x < b and y > a for x, y in excluded):
                invalid += 1
                excluded.append([a, b])
            else:
                retained.append([a, b])
        return merge(cues, self.duration), merge(excluded, self.duration), retained, invalid


class RecordedReference:
    """EchoCanceller-compatible reference on a virtual 16 kHz sample clock."""
    generation = 0

    def __init__(self, session):
        self.session = session
        self.cursor = 0

    def take_until(self, seconds):
        end = round(seconds * RATE)
        positions = self.session.alignment.map(np.arange(self.cursor, end))
        self.cursor = end
        audio = self.session.reference
        lower = np.floor(positions).astype(np.int64)
        fraction = positions - lower
        valid = (positions >= 0) & (positions <= len(audio) - 1)
        low = np.clip(lower, 0, len(audio) - 1)
        high = np.clip(lower + 1, 0, len(audio) - 1)
        samples = audio[low] * (1 - fraction) + audio[high] * fraction
        return np.where(valid, samples / 32768, 0).astype(np.float32)

    def alive(self):
        return True


class RuntimeDetector:
    def __init__(self, candidate):
        # Reuse loading, feature models, optimization, and chunking verbatim.
        self.detector = WakeDetector(Config({"wake": {"model": candidate,
            "threshold_low": 0., "threshold_high": 1., "personal_model": ""}}))
        self.pending = 0

    def process(self, frame):
        self.detector.process(frame)
        self.pending += len(frame)
        if self.pending < CHUNK:
            return None
        self.pending -= CHUNK
        return self.detector.last_score

    def reset(self):
        self.detector.reset()
        self.pending = 0

    def close(self):
        self.detector.close()


def trace(session, candidate, aec, detector_factory=RuntimeDetector, echo_factory=EchoCanceller):
    detector = detector_factory(candidate)
    reference = RecordedReference(session)
    echo = echo_factory(reference) if aec else None
    _, gaps, _, _ = session.spans()
    scores = []
    was_gap = False
    try:
        for start in range(0, session.duration, FRAME):
            end = start + FRAME
            gap = any(a < end and b > start for a, b in gaps)
            if gap:
                was_gap = True
                continue
            if was_gap:
                detector.reset()
                reference.cursor = start
                reference.generation += 1
                # AEC must not bridge missing or invalid capture history.
                echo = echo_factory(reference) if aec else None
                was_gap = False
            frame = session.mic[start:end]
            if echo is not None:
                frame = echo.process(frame, end / RATE)
            score = detector.process(frame)
            if score is not None:
                score = float(score)
                if not np.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError(f"Detector returned invalid probability: {score}")
                scores.append((end - 1, score))
    finally:
        detector.close()
    return scores


def hits(scores, threshold, cooldown):
    last = -float("inf")
    for sample, score in scores:
        if score >= threshold and sample - last >= cooldown * RATE:
            last = sample
            yield sample


def score_session(session, scores, threshold, cooldown=2.):
    cues, excluded, windows, invalid = session.spans()
    ignored = merge(cues + excluded, session.duration)
    detected = set()
    false = 0
    # Ignored spans cannot suppress a later eligible detection via cooldown.
    eligible = [(p, s) for p, s in scores if not inside(p, ignored)]
    for point in hits(eligible, threshold, cooldown):
        matches = [i for i, (a, b) in enumerate(windows) if a <= point < b]
        if matches:
            detected.update(matches)
        else:
            false += 1
    non_exposure = merge(ignored + windows, session.duration)
    samples = session.duration - sum(b - a for a, b in non_exposure)
    hours = samples / RATE / 3600
    return {"valid_attempts": len(windows), "invalid_attempts": invalid,
            "detected_attempts": len(detected), "recall": len(detected) / len(windows) if windows else None,
            "false_wakes": false, "exposure_hours": hours,
            "false_wakes_per_hour": false / hours if hours > 0 else None}


def choose_threshold(tuning, cooldown=2., target=1.):
    if not np.isfinite(cooldown) or cooldown < 0:
        raise ValueError("Cooldown must be finite and nonnegative")
    if not any(s.header["mode"] == "negative" for s, _ in tuning):
        raise ValueError("Threshold tuning requires negative sessions")
    # This fixed grid and disabled sentinel never inspect evaluation scores.
    choices = []
    for threshold in [i / 100 for i in range(1, 101)] + [float(np.nextafter(1., 2.))]:
        stats = [(s, score_session(s, scores, threshold, cooldown)) for s, scores in tuning]
        negatives = [m for s, m in stats if s.header["mode"] == "negative"]
        hours = sum(m["exposure_hours"] for m in negatives)
        if hours <= 0:
            raise ValueError("No usable tuning-negative exposure")
        rate = sum(m["false_wakes"] for m in negatives) / hours
        if rate <= target:
            attempts = sum(m["valid_attempts"] for _, m in stats)
            recall = sum(m["detected_attempts"] for _, m in stats) / attempts if attempts else 0
            choices.append((-recall, threshold))
    return min(choices)[1]


def validate_split(tune_paths, eval_paths):
    tune_paths = [manifest_path(Path(p)).resolve() for p in tune_paths]
    eval_paths = [manifest_path(Path(p)).resolve() for p in eval_paths]
    def identity(path):
        headers = [r for r in read_jsonl(path) if r.get("type") == "session"]
        if len(headers) != 1 or not headers[0].get("session_id"):
            raise ValueError(f"Missing or ambiguous session_id: {path}")
        return headers[0]["session_id"]
    tune_ids, eval_ids = [identity(p) for p in tune_paths], [identity(p) for p in eval_paths]
    if set(tune_paths) & set(eval_paths) or set(tune_ids) & set(eval_ids):
        raise ValueError("A session appears in both --tune and --eval; report refused")
    if len(set(tune_ids)) != len(tune_ids) or len(set(eval_ids)) != len(eval_ids):
        raise ValueError("Duplicate sessions within a split; exposure would be counted twice")
    # Renaming/copying a session_id must not evade the held-out boundary.
    def audio_hash(path):
        header = next(r for r in read_jsonl(path) if r["type"] == "session")
        digest = hashlib.sha256()
        with (path.parent / header["mic_wav"]).open("rb") as src:
            for chunk in iter(lambda: src.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    if set(audio_hash(p) for p in tune_paths) & set(audio_hash(p) for p in eval_paths):
        raise ValueError("Identical mic audio appears in tune and eval; report refused")
    return tune_paths, eval_paths


def aggregate(metrics):
    result = {key: sum(m[key] for m in metrics) for key in
              ("valid_attempts", "invalid_attempts", "detected_attempts", "false_wakes", "exposure_hours")}
    result["recall"] = result["detected_attempts"] / result["valid_attempts"] if result["valid_attempts"] else None
    result["false_wakes_per_hour"] = result["false_wakes"] / result["exposure_hours"] if result["exposure_hours"] else None
    return result


def evaluate(tune, evaluation, candidates, *, cooldown=2., detector_factory=RuntimeDetector,
             echo_factory=EchoCanceller):
    result = []
    for candidate in candidates:
        for aec in (False, True):
            tuning = [(s, trace(s, candidate, aec, detector_factory, echo_factory)) for s in tune]
            threshold = choose_threshold(tuning, cooldown)
            # The threshold is frozen before any held-out audio is processed.
            by_condition = {}
            for session in evaluation:
                scores = trace(session, candidate, aec, detector_factory, echo_factory)
                by_condition.setdefault(session.condition, []).append(score_session(session, scores, threshold, cooldown))
            for condition, metrics in sorted(by_condition.items()):
                result.append({"candidate": str(candidate), "aec": aec, "condition": condition,
                               "threshold": threshold, **aggregate(metrics)})
    return result


def write_report(output, result, tune, evaluation, cooldown):
    output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    payload = {"tune_sessions": [s.header["session_id"] for s in tune],
        "eval_sessions": [s.header["session_id"] for s in evaluation],
        "sample_rate": RATE, "frame_samples": FRAME, "target_tuning_negative_false_wakes_per_hour": 1,
        "event_policy": f"threshold exceedance with {cooldown:g}s cooldown; continuous runtime features; reset at excluded gaps",
        "exposure_policy": "recorded samples outside cues, invalid windows, discontinuities and valid positive windows",
        "proof_limit": "Held-out offline replay; rates are observed counts/exposure, not statistical guarantees. No live device acceptance.",
        "results": result}
    json_path = output / f"{stamp}.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = ["# Wake benchmark", "", payload["proof_limit"], "",
        f"Tune: {', '.join(payload['tune_sessions'])}. Eval: {', '.join(payload['eval_sessions'])}.",
        f"Runtime: {RATE} Hz / {FRAME} samples; {payload['event_policy']}.",
        "Thresholds use tuning negatives (≤1 observed false wake/hour); evaluation never selects thresholds.",
        "", "| Candidate | AEC | Condition | Recall (hits/valid) | Invalid | False wakes / hours | Threshold |",
        "|---|---|---|---|---|---|---|"]
    for row in result:
        recall = f"{row['recall']:.1%}" if row["recall"] is not None else "n/a"
        lines.append(f"| {row['candidate']} | {'on' if row['aec'] else 'off'} | {row['condition']} | {recall} ({row['detected_attempts']}/{row['valid_attempts']}) | {row['invalid_attempts']} | {row['false_wakes']} / {row['exposure_hours']:.6f} h | {row['threshold']:.6g} |")
    md_path = output / f"{stamp}.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path, json_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tune", type=Path, nargs="+", required=True)
    parser.add_argument("--eval", type=Path, nargs="+", required=True)
    parser.add_argument("--model", type=Path, action="append", default=[])
    parser.add_argument("--reports", type=Path, default=REPORTS)
    args = parser.parse_args(argv)
    tune_paths, eval_paths = validate_split(args.tune, args.eval)
    candidates = ["alexa", str(ROOT / "assistant/models/wake/alexa_user.onnx")]
    for path in args.model:
        path = path.resolve()
        if path.suffix.lower() != ".onnx" or not path.is_file():
            parser.error(f"Extra model must be an existing .onnx: {path}")
        candidates.append(str(path))
    if len(set(candidates)) != len(candidates):
        parser.error("Duplicate model candidates")
    tuning, evaluation = [Session.open(p) for p in tune_paths], [Session.open(p) for p in eval_paths]
    cooldown = float(load().get("wake.cooldown_seconds", 2.))
    results = evaluate(tuning, evaluation, candidates, cooldown=cooldown)
    for path in write_report(args.reports, results, tuning, evaluation, cooldown):
        print(path)


if __name__ == "__main__":
    main()
