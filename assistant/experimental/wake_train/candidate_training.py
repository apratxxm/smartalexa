"""Bounded, opt-in weak-label wake-word training experiment.

This module is deliberately isolated from Memoir's production model selection
and runtime. Machine-consensus positives are weak labels, not human review.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import wave

try:
    from . import active_learning as review
except ImportError:
    import active_learning as review

LABEL_SCHEMA = "memoir.wake-machine-consensus.v1"
MAX_INPUT_BYTES = 2 * 1024 * 1024
MAX_RECORDS = 24
MAX_CLIP_SECONDS = 24
MAX_WAV_BYTES = 1024 * 1024
MAX_TOTAL_SECONDS = 600
MAX_WINDOWS = 8192
MAX_BAG_WINDOWS = 160
MIN_FREE_BYTES = 1536 * 1024 * 1024
HOP = 1280
SAMPLE_RATE = 16000
DEFAULT_STORE = Path(__file__).with_name("human_reviews.json")
FRONTEND_DIR = Path(r"A:/wake-data/models")
STOCK_MODEL = Path(r"A:/wake-data/models/alexa_v0.1.onnx")
USER_RECORDINGS = Path(r"A:/wake-data/user_recordings")
TOKEN = re.compile(r"(?<!\w)alexa(?!\w)", re.IGNORECASE)
DISCLAIMER = (
    "Positive labels are weak machine-consensus labels from Whisper and Moonshine. "
    "Transcript agreement does not establish acoustic ground truth or user intent; "
    "capture sessions are not independent rooms; this experiment has no accuracy "
    "claim and is not eligible for production."
)
REPORT_EXTRA_KEYS = frozenset({
    "unsure_excluded", "positive_holdout_excluded", "train_positive_duplicate_records_excluded",
    "train_positive_groups", "train_negative_count", "train_negative_groups",
    "train_positive_count", "training_seconds", "missing_historic_audio_excluded",
    "human_review_metadata_rows", "positive_span_plan_sha256", "positive_span_count",
    "negative_holdout_excluded", "reviewed_training_selection",
})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _bounded_json(path: Path):
    with Path(path).open("rb") as stream:
        payload = stream.read(MAX_INPUT_BYTES + 1)
    if len(payload) > MAX_INPUT_BYTES:
        raise ValueError("input JSON exceeds 2 MiB limit")
    return json.loads(payload), hashlib.sha256(payload).hexdigest()


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _transcript_has_token(value):
    return isinstance(value, str) and TOKEN.search(value) is not None


def _validate_report_extra(report_extra):
    if report_extra is None:
        return {}
    if not isinstance(report_extra, dict) or set(report_extra) - REPORT_EXTRA_KEYS:
        raise ValueError("report_extra contains reserved or unknown provenance fields")
    result = dict(report_extra)
    if "reviewed_training_selection" in result:
        _validate_reviewed_training_selection(result["reviewed_training_selection"])
    return result


def _validate_reviewed_training_selection(value):
    keys = {"algorithm", "per_class_limit", "selected_counts", "excluded_records"}
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError("reviewed training selection has unknown or missing fields")
    if value["algorithm"] != "session_round_robin_sha256_v1":
        raise ValueError("reviewed training selection algorithm is invalid")
    if not _is_int(value["per_class_limit"]) or value["per_class_limit"] != 128:
        raise ValueError("reviewed training selection limit must be 128")
    counts = value["selected_counts"]
    if (not isinstance(counts, dict) or set(counts) != {"positive", "negative"}
            or any(not _is_int(count) or not 0 <= count <= 128 for count in counts.values())):
        raise ValueError("reviewed training selection counts are invalid")
    excluded = value["excluded_records"]
    if not isinstance(excluded, list) or len(excluded) > 512:
        raise ValueError("reviewed training exclusions exceed 512 records")
    record_keys = {"label", "wav_sha256", "start", "end", "group_id"}
    for row in excluded:
        if not isinstance(row, dict) or set(row) != record_keys:
            raise ValueError("reviewed training exclusion has unknown or missing fields")
        digest, start, end, group = row["wav_sha256"], row["start"], row["end"], row["group_id"]
        if row["label"] not in ("positive", "negative"):
            raise ValueError("reviewed training exclusion label is invalid")
        if (not isinstance(digest, str) or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)):
            raise ValueError("reviewed training exclusion hash must be lowercase SHA-256")
        if (not _is_int(start) or not _is_int(end) or not 0 <= start < end <= MAX_CLIP_SECONDS * SAMPLE_RATE):
            raise ValueError("reviewed training exclusion sample bounds are invalid")
        if not isinstance(group, str) or not group.strip() or len(group) > 128:
            raise ValueError("reviewed training exclusion group id is invalid")


def _label_provenance(label_source):
    if label_source == "human_reviewed_spans":
        return {
            "disclaimer": ("Positive labels are exact human-reviewed wake spans and negative labels are whole clips "
                           "confirmed negative by human review. Session groups are human attested; this experiment "
                           "has no accuracy claim and is not eligible for production."),
            "training_labels": {"positive_source": label_source, "positive_human_reviewed": True,
                                "all_human_reviewed_positive_rows": True,
                                "negative_source": "authoritative_human_review_store",
                                "negative_human_reviewed": True, "production_eligible": False},
        }
    if label_source == "machine_consensus":
        return {
            "disclaimer": DISCLAIMER,
            "training_labels": {"positive_source": label_source, "positive_human_reviewed": False,
                                "all_human_reviewed_positive_rows": False,
                                "negative_source": "authoritative_human_review_store",
                                "negative_human_reviewed": True, "production_eligible": False},
        }
    raise ValueError("unsupported candidate label source")


def validate_positive_row(row, *, audio_root=review.AUDIO_ROOT):
    """Strictly validate one positive weak-label row and its archived WAV."""
    if not isinstance(row, dict):
        raise ValueError("label record must be an object")
    if row.get("label") == "unsure":
        return None
    if row.get("label") != "positive":
        if row.get("label") not in ("negative", "unsure"):
            raise ValueError("label must be positive, negative or unsure")
        if row.get("label") == "negative":
            raise ValueError("machine-consensus manifest cannot supply negative training labels")
        return None
    if (row.get("label_source") != "machine_consensus"
            or row.get("human_reviewed") is not False
            or row.get("production_eligible") is not False):
        raise ValueError("positive machine labels must remain explicitly non-human and non-production")
    if row.get("truncated") is not False:
        raise ValueError("positive source must explicitly be untruncated")
    evidence = row.get("evidence")
    if not isinstance(evidence, dict):
        raise ValueError("positive machine label requires two independent transcript records")
    sources = []
    texts = []
    for key in ("whisper", "moonshine"):
        item = evidence.get(key)
        if not isinstance(item, dict):
            raise ValueError("positive machine label requires Whisper and Moonshine evidence")
        family, text = item.get("family"), item.get("text")
        if not isinstance(family, str) or not isinstance(text, str):
            raise ValueError("machine evidence family and transcript are required")
        if item.get("text_truncated") is True or row.get("text_truncated") is True:
            raise ValueError("positive label transcript evidence is truncated")
        if len(text) > 400:
            raise ValueError("machine transcript exceeds 400-character limit")
        if family.strip().casefold() != key:
            raise ValueError("machine evidence family must identify its exact inference model")
        if item.get("exact_alexa") is not True or not _transcript_has_token(text):
            raise ValueError("positive label evidence does not contain an exact Alexa token")
        if (item.get("source") != "offline_same_wav_inference"
                or item.get("audio_sha256") != row.get("wav_sha256")):
            raise ValueError("machine evidence must come from offline inference on this exact WAV")
        sources.append(family.strip().casefold())
        texts.append(text)
    if sources[0] == sources[1]:
        raise ValueError("positive labels require distinct Whisper and Moonshine families")
    if not isinstance(row.get("event_id"), str) or not row["event_id"].strip():
        raise ValueError("positive record event_id is required")
    capture = row.get("capture_session_id")
    if (not isinstance(capture, str) or not capture.strip() or len(capture) > 128
            or row.get("session_id") != capture or row.get("group_id") != capture):
        raise ValueError("positive row needs one consistent capture session and group")
    if row.get("split") not in ("train", "holdout"):
        raise ValueError("positive row split must be train or holdout")
    if row["split"] != review.session_split(capture):
        raise ValueError("positive split does not match the deterministic whole-session partition")
    path_value = row.get("path")
    path = review.audio_path(path_value, audio_root)
    if path.is_relative_to(USER_RECORDINGS.resolve()):
        raise ValueError("user_recordings audio is forbidden in training")
    if path.stat().st_size > MAX_WAV_BYTES:
        raise ValueError("positive WAV exceeds 1 MiB limit")
    frames = row.get("n_frames")
    rate = row.get("sample_rate")
    digest = row.get("wav_sha256")
    if not _is_int(frames) or not 0 < frames <= MAX_CLIP_SECONDS * SAMPLE_RATE:
        raise ValueError("positive WAV frame count exceeds 24-second limit")
    if not _is_int(rate) or rate != SAMPLE_RATE:
        raise ValueError("positive sample rate must be 16000 Hz")
    identity = review.wav_identity(path)
    if identity["wav_sha256"] != digest or identity["n_frames"] != frames or identity["sample_rate"] != rate:
        raise ValueError("positive WAV identity mismatch")
    return {**row, "path": path, "identity": identity, "families": sources}


def select_training_rows(manifest, negatives):
    if (not isinstance(manifest, dict) or manifest.get("schema") != LABEL_SCHEMA
            or manifest.get("human_reviewed") is not False
            or manifest.get("production_eligible") is not False
            or not isinstance(manifest.get("records"), list)):
        raise ValueError("invalid machine-consensus labels manifest")
    records = manifest["records"]
    if len(records) > MAX_RECORDS:
        raise ValueError("machine-consensus manifest exceeds 24 records")
    positives, skipped = [], 0
    for row in records:
        if isinstance(row, dict) and row.get("label") == "unsure":
            skipped += 1
            continue
        validated = validate_positive_row(row)
        if validated:
            positives.append(validated)
    # Normalize review-history paths (the authoritative API returns strings)
    # and validate identity across every positive/negative row before filtering
    # holdout. Aliasing across splits/groups/labels is always a hard failure.
    all_neg = [dict(r, path=Path(r["path"]).resolve(strict=True))
               for r in negatives if r.get("presence") == "negative"]
    for row in positives:
        row["path"] = Path(row["path"]).resolve(strict=True)
    hashes = {}
    for row, label in ([(r, "positive") for r in positives] + [(r, "negative") for r in all_neg]):
        digest = row["wav_sha256"]
        identity = (label, row["group_id"], row["split"])
        old = hashes.setdefault(digest, identity)
        if old != identity:
            raise ValueError("duplicate WAV crosses label, capture group or split")
    holdout_pos = [r for r in positives if r["split"] == "holdout"]
    train_pos = [r for r in positives if r["split"] == "train"]
    train_neg = [r for r in all_neg if r.get("split") == "train"]
    if len(train_pos) + len(train_neg) > 44:
        raise ValueError("training source count exceeds 44 clips")
    for row in train_pos + train_neg:
        if row["path"].stat().st_size > MAX_WAV_BYTES or row["n_frames"] > MAX_CLIP_SECONDS * SAMPLE_RATE:
            raise ValueError("training WAV exceeds per-source byte or duration limit")
    # Repeated records for the same archived clip within one group do not
    # increase the number of independent positive clips.
    duplicate_train_positives = len(train_pos) - len({r["wav_sha256"] for r in train_pos})
    train_pos = list({r["wav_sha256"]: r for r in train_pos}.values())
    pos_groups = {r["group_id"] for r in train_pos}
    if len(train_pos) < 3 or len(pos_groups) < 2:
        raise ValueError("need at least 3 positive TRAIN clips from 2 capture-session groups")
    if len(train_neg) < 5:
        raise ValueError("need at least 5 verified human-reviewed negative TRAIN clips")
    total = sum(r["n_frames"] for r in train_pos + train_neg) / SAMPLE_RATE
    if total > MAX_TOTAL_SECONDS:
        raise ValueError("training audio exceeds 600-second limit")
    return train_pos, train_neg, {"unsure_excluded": skipped, "positive_holdout_excluded": len(holdout_pos),
                                  "train_positive_duplicate_records_excluded": duplicate_train_positives,
                                  "train_positive_groups": sorted(pos_groups), "train_negative_count": len(train_neg),
                                  "train_negative_groups": sorted({r["group_id"] for r in train_neg}),
                                  "train_positive_count": len(train_pos), "training_seconds": total}


def _read_pcm(path):
    with wave.open(str(path), "rb") as wav:
        if (wav.getnchannels(), wav.getframerate(), wav.getsampwidth(), wav.getcomptype()) != (1, SAMPLE_RATE, 2, "NONE"):
            raise ValueError("training audio must be mono PCM16 at 16 kHz")
        frames = wav.getnframes()
        if not 0 < frames <= MAX_CLIP_SECONDS * SAMPLE_RATE:
            raise ValueError("training WAV exceeds 24-second limit")
        payload = wav.readframes(frames)
    import numpy as np
    samples = np.frombuffer(payload, dtype="<i2").copy()
    if samples.size != frames:
        raise ValueError("truncated WAV frame data")
    return samples


def retained_window_indices(consumed_ends, original_frames):
    """Use windows after 160 ms of the padded stream through 640 ms suffix."""
    lower = 32000 + 2560
    upper = 32000 + int(original_frames) + 10240
    return [i for i, end in enumerate(consumed_ends) if lower <= end <= upper][::2]


def select_bag_windows(features, consumed_ends, original_frames, *, max_windows=MAX_BAG_WINDOWS):
    import numpy as np
    if len(features) != len(consumed_ends):
        raise ValueError("feature/end index count mismatch")
    keep = retained_window_indices(consumed_ends, original_frames)
    if not keep:
        raise ValueError("audio produced no in-range embedding windows")
    if len(keep) > max_windows:
        slots = np.linspace(0, len(keep) - 1, max_windows, dtype=np.int64)
        keep = [keep[int(i)] for i in slots]
    selected = np.asarray([features[i] for i in keep], dtype=np.float32)
    if selected.ndim != 3 or selected.shape[1:] != (16, 96) or not np.isfinite(selected).all():
        raise ValueError("embedding windows must be finite float32 (N,16,96)")
    return selected


def extract_bag(samples, *, feature_factory=None):
    """Create one bounded MIL bag through 1280-sample streaming frontend hops."""
    import numpy as np
    if feature_factory is None:
        from openwakeword.utils import AudioFeatures
        feature_factory = lambda: AudioFeatures(
            melspec_model_path=str(FRONTEND_DIR / "melspectrogram.onnx"),
            embedding_model_path=str(FRONTEND_DIR / "embedding_model.onnx"),
            inference_framework="onnx", ncpu=1, device="cpu")
    frontend = feature_factory()
    if hasattr(frontend, "reset"):
        frontend.reset()
    pcm = np.concatenate((np.zeros(32000, np.int16), np.asarray(samples, dtype=np.int16), np.zeros(12800, np.int16)))
    features, ends = [], []
    for start in range(0, len(pcm), HOP):
        hop = pcm[start:start + HOP]
        if len(hop) < HOP:
            hop = np.pad(hop, (0, HOP - len(hop)))
        frontend(hop)
        arr = np.asarray(frontend.get_features(16))[0]
        if arr.ndim == 3:
            arr = arr[0]
        if arr.shape == (16, 96) and np.isfinite(arr).all():
            features.append(arr.astype(np.float32, copy=True))
            ends.append(start + HOP)
        else:
            raise ValueError(f"frontend returned unexpected feature shape {arr.shape}")
    return select_bag_windows(features, ends, len(samples))


def _check_sources(paths):
    return {str(Path(path).resolve(strict=True)): sha256_file(Path(path).resolve(strict=True)) for path in paths}


def _verify_sources_before_publish(source_hashes, before_publish=None):
    if before_publish is not None:
        before_publish()
    if source_hashes is not None and _check_sources(map(Path, source_hashes)) != source_hashes:
        raise ValueError("training source changed before publication")


def _publish_artifacts(temp_model, target, report, *, before_publish=None):
    """Atomically publish sidecar then model, rolling back only created paths."""
    target = Path(target)
    meta_path = target.with_suffix(".meta.json")
    parent = target.parent
    if before_publish is not None:
        before_publish()
    meta_fd, meta_name = tempfile.mkstemp(prefix=".wake-candidate-meta-", suffix=".json", dir=parent)
    meta_temp = Path(meta_name)
    created_meta = created_model = False
    try:
        with os.fdopen(meta_fd, "w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(meta_temp, meta_path)
        created_meta = True
        os.link(temp_model, target)
        created_model = True
    except Exception:
        if created_model:
            target.unlink(missing_ok=True)
        if created_meta:
            meta_path.unlink(missing_ok=True)
        raise
    finally:
        meta_temp.unlink(missing_ok=True)


def _load_stock_weights(onnx_path, net, torch):
    import onnx
    from onnx import numpy_helper
    model = onnx.load(str(onnx_path), load_external_data=False)
    mapping = {x.name: numpy_helper.to_array(x) for x in model.graph.initializer
               if x.name in {"1.weight", "1.bias", "3.weight", "3.bias", "5.weight", "5.bias"}}
    expected = {"1.weight": (128, 1536), "1.bias": (128,), "3.weight": (128, 128),
                "3.bias": (128,), "5.weight": (1, 128), "5.bias": (1,)}
    if set(mapping) != set(expected) or any(tuple(mapping[k].shape) != shape for k, shape in expected.items()):
        raise ValueError("stock ONNX classifier has unexpected initializer names or shapes")
    with torch.no_grad():
        for layer, prefix in ((net.fc1, "1"), (net.fc2, "3"), (net.fc3, "5")):
            layer.weight.copy_(torch.from_numpy(mapping[prefix + ".weight"].copy()))
            layer.bias.copy_(torch.from_numpy(mapping[prefix + ".bias"].copy()))


def fit_and_export(pos_bags, neg_bags, target, *, seed=1337, epochs=8, source_hashes=None,
                   report_extra=None, positive_groups=None, before_publish=None,
                   label_source="machine_consensus"):
    if not _is_int(epochs) or not 1 <= epochs <= 8:
        raise ValueError("epochs must be between 1 and 8")
    report_extra = _validate_report_extra(report_extra)
    if label_source not in ("machine_consensus", "human_reviewed_spans"):
        raise ValueError("unsupported candidate label source")
    import numpy as np
    import torch
    from torch import nn
    import onnx
    import onnxruntime as ort
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    np.random.seed(seed)

    class SmallMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(16 * 96, 128)
            self.fc2 = nn.Linear(128, 128)
            self.fc3 = nn.Linear(128, 1)

        def forward(self, x):
            x = x.reshape(x.shape[0], -1)
            return torch.sigmoid(self.fc3(torch.relu(self.fc2(torch.relu(self.fc1(x))))))

    baseline = SmallMLP().cpu()
    _load_stock_weights(STOCK_MODEL, baseline, torch)
    baseline.eval()
    parity_rng = np.random.default_rng(seed ^ 0x51A7)
    parity_input = parity_rng.standard_normal((31, 16, 96), dtype=np.float32)
    parity_input = np.concatenate((parity_input, np.zeros((1, 16, 96), np.float32)), axis=0)
    ort_options = ort.SessionOptions()
    ort_options.intra_op_num_threads = 1
    ort_options.inter_op_num_threads = 1
    stock_session = ort.InferenceSession(str(STOCK_MODEL), sess_options=ort_options,
                                         providers=["CPUExecutionProvider"])
    stock_expected = stock_session.run([stock_session.get_outputs()[0].name],
                                       {stock_session.get_inputs()[0].name: parity_input})[0]
    with torch.no_grad():
        baseline_actual = baseline(torch.from_numpy(parity_input)).numpy()
    initial_parity = float(np.max(np.abs(stock_expected - baseline_actual)))
    if (stock_expected.shape != (32, 1) or not np.isfinite(stock_expected).all()
            or (stock_expected < 0).any() or (stock_expected > 1).any()
            or not np.isfinite(baseline_actual).all() or initial_parity > 1e-4):
        raise ValueError("stock ONNX and imported classifier parity failed")
    del stock_session
    candidate = copy.deepcopy(baseline)
    optimizer = torch.optim.Adam(candidate.parameters(), lr=1e-5)
    rng = np.random.default_rng(seed)
    if not pos_bags or not neg_bags:
        raise ValueError("both positive and negative bags are required")
    if 2 * max(len(pos_bags), len(neg_bags)) > 256:
        raise ValueError("bag count exceeds bounded training steps")
    losses = []
    for epoch in range(epochs):
        pos_order = rng.permutation(len(pos_bags))
        neg_order = rng.permutation(len(neg_bags))
        epoch_losses = []
        candidate.train()
        for index in range(max(len(pos_bags), len(neg_bags))):
            pair = ((pos_bags[int(pos_order[index % len(pos_order)])], 1),
                    (neg_bags[int(neg_order[index % len(neg_order)])], 0))
            for bag, label in pair:
                x = torch.from_numpy(np.asarray(bag, dtype=np.float32))
                with torch.no_grad():
                    baseline_probs = baseline(x).detach()
                probs = candidate(x)
                bag_prob = probs.max()
                y = torch.tensor(float(label), dtype=torch.float32)
                loss = nn.functional.binary_cross_entropy(bag_prob.clamp(1e-6, 1 - 1e-6), y)
                loss = loss + 0.5 * torch.mean((probs - baseline_probs) ** 2)
                if not torch.isfinite(loss):
                    raise ValueError("non-finite training loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(loss.detach()))
        losses.append(float(np.mean(epoch_losses)))
    if any(not torch.isfinite(p).all() for p in candidate.parameters()):
        raise ValueError("non-finite trained model weights")
    candidate.eval()
    parent = Path(target).parent
    fd, temp_name = tempfile.mkstemp(prefix=".wake-candidate-", suffix=".onnx", dir=parent)
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        dummy = torch.zeros((2, 16, 96), dtype=torch.float32)
        torch.onnx.export(candidate, dummy, str(temp_path), input_names=["input"], output_names=["output"],
                          dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
                          opset_version=13, dynamo=False)
        onnx.checker.check_model(onnx.load(str(temp_path)))
        session = ort.InferenceSession(str(temp_path), sess_options=ort_options,
                                       providers=["CPUExecutionProvider"])
        sample = parity_input
        with torch.no_grad():
            expected = candidate(torch.from_numpy(sample)).numpy()
        actual = session.run(["output"], {"input": sample})[0]
        if (actual.shape != (32, 1) or not np.isfinite(actual).all()
                or (actual < 0).any() or (actual > 1).any()
                or float(np.max(np.abs(actual - expected))) > 1e-4):
            raise ValueError("export parity or output validation failed")
        if source_hashes is not None and _check_sources(map(Path, source_hashes)) != source_hashes:
            raise ValueError("training source changed while fitting")
        provenance = _label_provenance(label_source)
        report = {
            "schema": "memoir.wake-candidate-training-meta.v1",
            "weak_label_disclaimer": provenance["disclaimer"],
            "training_labels": provenance["training_labels"],
            "positive_bags": len(pos_bags), "negative_bags": len(neg_bags),
            "positive_groups": sorted(positive_groups or []), "positive_windows": int(sum(len(x) for x in pos_bags)),
            "negative_windows": int(sum(len(x) for x in neg_bags)), "seed": seed, "epochs": epochs,
            "epoch_mean_losses": losses, "source_sha256": source_hashes or {},
            "stock_import_parity_max_abs_diff": initial_parity,
            "candidate_export_parity_max_abs_diff": float(np.max(np.abs(actual - expected))),
            "output_sha256": sha256_file(temp_path), "deployment_changed": False,
            "production_eligible": False, "accuracy_claim": False,
        }
        report.update(report_extra)
        def verify_before_publish():
            _verify_sources_before_publish(source_hashes, before_publish)

        _publish_artifacts(temp_path, target, report, before_publish=verify_before_publish)
        return report
    finally:
        temp_path.unlink(missing_ok=True)


def run(labels_path, output_path, *, allow_machine_labels=False, epochs=8, seed=1337,
        memory_gate=None, feature_factory=None):
    if not allow_machine_labels:
        raise ValueError("--allow-machine-labels is required; no training started")
    target = review.candidate_output(output_path)
    if not _is_int(epochs) or not 1 <= epochs <= 8:
        raise ValueError("epochs must be between 1 and 8")
    label_path = Path(labels_path).resolve(strict=True)
    label_manifest, labels_hash = _bounded_json(label_path)
    store_path = DEFAULT_STORE
    store, store_hash = review.load_preparation_store(store_path)
    # Review history is authoritative for all human negatives; changed hashes
    # fail closed while missing historic recordings are counted and excluded.
    metadata, usable, unavailable = review.review_history(store)
    positives, negatives, counts = select_training_rows(label_manifest, usable)
    source_paths = {label_path, Path(store_path).resolve(strict=True),
                    STOCK_MODEL.resolve(strict=True), (FRONTEND_DIR / "melspectrogram.onnx").resolve(strict=True),
                    (FRONTEND_DIR / "embedding_model.onnx").resolve(strict=True)}
    source_paths.update(row["path"] for row in positives + negatives)
    source_hashes = _check_sources(source_paths)
    if (labels_hash != source_hashes.get(str(label_path))
            or store_hash != source_hashes.get(str(Path(store_path).resolve(strict=True)))):
        raise ValueError("labels or human review store changed after parsing")
    for row in positives + negatives:
        if source_hashes.get(str(row["path"])) != row["wav_sha256"]:
            raise ValueError("training WAV changed after its review/label identity check")
    if memory_gate is None:
        from .machine_proposals import physical_free_bytes
        memory_gate = physical_free_bytes
    if memory_gate() < MIN_FREE_BYTES:
        raise RuntimeError("need at least 1536 MiB physical memory before loading ML dependencies")
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    # Lazy ML dependency imports happen only after all eligibility/resource gates.
    import numpy as np
    bags_pos, bags_neg = [], []
    total_windows = 0
    for row, is_positive in ([(row, True) for row in positives] + [(row, False) for row in negatives]):
        samples = _read_pcm(row["path"])
        np.random.seed(seed ^ int(row["wav_sha256"][:8], 16))
        bag = extract_bag(samples, feature_factory=feature_factory)
        total_windows += len(bag)
        if total_windows > MAX_WINDOWS:
            raise ValueError("extracted feature windows exceed 8192-window limit")
        (bags_pos if is_positive else bags_neg).append(bag)
    if _check_sources(source_paths) != source_hashes:
        raise ValueError("training source changed during feature extraction")
    report = fit_and_export(bags_pos, bags_neg, target, seed=seed, epochs=epochs,
                            source_hashes=source_hashes, positive_groups=counts["train_positive_groups"],
                            report_extra={**counts, "missing_historic_audio_excluded": len(unavailable),
                                          "human_review_metadata_rows": len(metadata)})
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--allow-machine-labels", action="store_true", required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args(argv)
    report = run(args.labels, args.out, allow_machine_labels=args.allow_machine_labels,
                 epochs=args.epochs, seed=args.seed)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
