"""python -m assistant.wakebench.train --dry-run | --run [--tag T] [--epochs N] [--seed S]

--dry-run prints the plan and writes nothing. --run writes only under _work/ and a new
assistant/models/wake/candidates/<date>-<tag>/ directory. It never touches runtime config.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import sys
import time
from collections import Counter

import numpy as np

from assistant.wakebench.train import data as D
from assistant.wakebench.train import fit as F
from assistant.wakebench.train import plan as P
from assistant.wakebench.train import synth as S

RAM_FLOOR_BYTES = 1536 * 1024 * 1024       # brief: stop cleanly below 1.5 GB free
WALL_LIMIT_S = 90 * 60
N_GENERIC_TRAIN = 300
N_GENERIC_HOLDOUT = 120
THRESHOLDS = (0.3, 0.5, 0.7)
DISCLAIMER = ("Offline synthetic-speech candidate. Positives are Piper TTS renderings; negatives are "
              "human-reviewed archive clips, synthetic confusables and generic embedding windows. "
              "No accuracy claim, not production-eligible, not deployed.")


class StopRun(RuntimeError):
    pass


def free_ram_bytes() -> int:
    from assistant.experimental.wake_train.machine_proposals import physical_free_bytes

    return int(physical_free_bytes())


def leakage_checks(items, arch_rows, voice_keys) -> dict:
    """Split-leakage booleans. True means train and holdout do not overlap on that key."""
    def positives(split):
        return [it for it in items if it.split == split and it.kind == "positive"]

    def negatives(split):
        return [it for it in items if it.split == split and it.kind != "positive"]

    _, _, independent = P.voice_split(voice_keys)
    text_pos = {it.text for it in positives("train")} & {it.text for it in positives("holdout")}
    text_neg = {it.text for it in negatives("train")} & {it.text for it in negatives("holdout")}
    voice_pos = {it.voice for it in positives("train")} & {it.voice for it in positives("holdout")}
    groups = {r["group_id"] for r in arch_rows if r["split"] == "train"} & \
             {r["group_id"] for r in arch_rows if r["split"] == "holdout"}
    return {
        "positive_text_overlap": len(text_pos),
        "negative_text_overlap": len(text_neg),
        "positive_voice_overlap": len(voice_pos),
        "archive_group_overlap": len(groups),
        "independent_voice_validation": independent,
        "ok": (not text_neg and not groups
               and (not voice_pos if independent else not text_pos)),
    }


def _plan_summary(keys, items, arch_rows, unavailable) -> dict:
    counts = P.plan_counts(items)
    arch = Counter(r["split"] for r in arch_rows)
    est_bags = (counts.get("positive_train", 0) * 2 + counts.get("positive_holdout", 0)
                + counts.get("neg_confusable_train", 0) + counts.get("neg_confusable_holdout", 0)
                + counts.get("neg_generic_train", 0) + counts.get("neg_generic_holdout", 0))
    _, _, independent = P.voice_split(keys)
    return {
        "voices": keys,
        "voices_independent_holdout": independent,
        "split_basis": ("held-out voice (speaker) for validation" if independent else
                        "held-out phrase text, single voice (no held-out voice available)"),
        "synthetic_items": counts,
        "synthetic_items_total": len(items),
        "phrases": {
            "positive_train": list(P.POS_TRAIN),
            "positive_holdout": list(P.POS_TRAIN if independent else P.POS_HOLDOUT),
            "confusable_train": list(P.NEG_CONF_TRAIN),
            "confusable_holdout": list(P.NEG_CONF_HOLDOUT),
            "generic_train": list(P.NEG_GEN_TRAIN),
            "generic_holdout": list(P.NEG_GEN_HOLDOUT),
        },
        "archive_negative_clips": {"train": arch["train"], "holdout": arch["holdout"],
                                   "unavailable_audio_excluded": unavailable},
        "generic_bags": {"train": N_GENERIC_TRAIN, "holdout": N_GENERIC_HOLDOUT,
                         "windows_per_bag": D.WINDOWS_PER_BAG, "source": str(D.GENERIC)},
        "augmentations": {"gain_db": list(S.GAIN_DB), "reverb_prob": S.REVERB_PROB,
                          "reverb": "generated exponential-decay RIR, RT60 0.15-0.5 s",
                          "noise_prob": S.NOISE_PROB, "snr_db": list(S.SNR_CHOICES),
                          "noise_source": "archive room bank (human-reviewed negatives, same split), "
                                          "synthetic coloured noise only when a bank is empty",
                          "speed_pitch_rates": list(P.RATE_FACTORS),
                          "length_scales": list(P.LENGTH_SCALES), "noise_scales": list(P.NOISE_SCALES),
                          "noise_w": list(P.NOISE_W), "music_beds": "none used"},
        "frontend_bags_estimate": est_bags,
        "time_estimate_note": "rough guess, not measured: ~0.3 s per Piper render, ~0.05 s per bag, plus fit",
        "time_estimate_minutes": round((len(items) * 0.3 + est_bags * 0.05) / 60 + 3, 1),
    }


def _ram_status() -> dict:
    try:
        free = free_ram_bytes()
    except Exception as exc:  # report, do not guess
        return {"error": str(exc)}
    return {"free_gib": round(free / 2 ** 30, 2), "floor_gib": 1.5, "run_allowed": free >= RAM_FLOOR_BYTES}


def dry_run(seed: int, log=print) -> dict:
    voices = P.discover_voices()
    keys = sorted(voices)
    if not keys:
        raise StopRun("no Piper voices found under assistant/models")
    items = P.build_plan(keys, seed=seed)
    arch_rows, unavailable = D.archive_negative_rows()
    summary = _plan_summary(keys, items, arch_rows, unavailable)
    summary["leakage_checks"] = leakage_checks(items, arch_rows, keys)
    summary["stock_init"] = str(F.ct.STOCK_MODEL)
    summary["seed"] = seed
    summary["ram"] = _ram_status()
    log(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def _frontend_factory():
    cache = {}

    def factory():
        if "fe" not in cache:
            from openwakeword.utils import AudioFeatures

            cache["fe"] = AudioFeatures(melspec_model_path=str(F.ct.FRONTEND_DIR / "melspectrogram.onnx"),
                                        embedding_model_path=str(F.ct.FRONTEND_DIR / "embedding_model.onnx"),
                                        inference_framework="onnx", ncpu=1, device="cpu")
        return cache["fe"]

    return factory


def run(seed: int, epochs: int, tag: str, log=print):
    t0 = time.monotonic()

    def guard(stage: str):
        if time.monotonic() - t0 > WALL_LIMIT_S:
            raise StopRun(f"90-minute wall limit exceeded at {stage}")
        free = free_ram_bytes()
        if free < RAM_FLOOR_BYTES:
            raise StopRun(f"free RAM {free / 2 ** 30:.2f} GiB below 1.5 GiB floor at {stage}")

    guard("start")
    voices = P.discover_voices()
    keys = sorted(voices)
    items = P.build_plan(keys, seed=seed)
    rng = np.random.default_rng(seed)
    arch_rows, unavailable = D.archive_negative_rows()
    leak = leakage_checks(items, arch_rows, keys)
    if not leak["ok"]:
        raise StopRun(f"split leakage check failed: {leak}")

    P.WORK_DIR.mkdir(parents=True, exist_ok=True)
    with (P.WORK_DIR / "plan.jsonl").open("w", encoding="utf-8") as out:
        for it in items:
            out.write(json.dumps(it.__dict__, ensure_ascii=False) + "\n")

    archive = [(row, D.read_pcm16(row["path"])) for row in arch_rows]
    banks = {s: D.room_bank_from([pcm for r, pcm in archive if r["split"] == s]) for s in ("train", "holdout")}
    log(f"room bank segments train={len(banks['train'])} holdout={len(banks['holdout'])}")
    factory = _frontend_factory()
    from piper.voice import PiperVoice

    guard("feature_extraction")
    voice_cache: dict = {}
    syn: dict = {}
    skipped: Counter = Counter()
    augment_sources: Counter = Counter()
    for k, it in enumerate(items):
        if k % 40 == 0:
            guard(f"synthesis {k}/{len(items)}")
        path, sid = voices[it.voice]
        if path not in voice_cache:
            voice_cache[path] = PiperVoice.load(str(path))
        audio = S.render_piper(voice_cache[path], sid, it.text, length_scale=it.length_scale,
                               noise_scale=it.noise_scale, noise_w=it.noise_w, rate=it.rate)
        copies = 2 if (it.kind == "positive" and it.split == "train") else 1
        sub = np.random.default_rng(it.augment_seed)
        for _ in range(copies):
            aug, info = S.augment(audio, sub, banks[it.split])
            augment_sources[info.get("noise_source", "none")] += 1
            try:
                bag = F.ct.extract_bag(S.to_pcm16(aug), feature_factory=factory)
            except ValueError:
                skipped[it.kind] += 1
                continue
            syn.setdefault((it.kind, it.split), []).append(bag.astype(np.float16))
    log("synthetic bags: " + json.dumps({f"{k[0]}_{k[1]}": len(v) for k, v in syn.items()})
        + f" skipped={dict(skipped)}")

    arch_bags: dict = {"train": [], "holdout": []}
    for row, pcm in archive:
        guard("archive features")
        arch_bags[row["split"]].append(F.ct.extract_bag(pcm, feature_factory=factory).astype(np.float16))

    gen = D.GenericWindows()
    gen_bags = {"train": [b.astype(np.float16) for b in gen.bags("train", N_GENERIC_TRAIN, rng)],
                "holdout": [b.astype(np.float16) for b in gen.bags("holdout", N_GENERIC_HOLDOUT, rng)]}

    pos_train = syn.get(("positive", "train"), [])
    pos_hold = syn.get(("positive", "holdout"), [])
    neg_train = (syn.get(("neg_confusable", "train"), []) + syn.get(("neg_generic", "train"), [])
                 + arch_bags["train"] + gen_bags["train"])
    neg_hold = {"synthetic_confusable": syn.get(("neg_confusable", "holdout"), []),
                "synthetic_generic": syn.get(("neg_generic", "holdout"), []),
                "archive": arch_bags["holdout"], "generic": gen_bags["holdout"]}
    if not pos_train or not pos_hold or not neg_train or not any(neg_hold.values()):
        raise StopRun("a required bag set is empty; refusing to train")
    log(f"train pos={len(pos_train)} neg={len(neg_train)}; holdout pos={len(pos_hold)} "
        + json.dumps({k: len(v) for k, v in neg_hold.items()}))

    guard("training")
    _, parity = F.stock_parity(seed)
    model, losses, baseline = F.fit(pos_train, neg_train, seed=seed, epochs=epochs, log=log)

    def eval_model(m):
        pos = F.max_probs(m, pos_hold)
        per = {name: F.max_probs(m, bags) for name, bags in neg_hold.items() if bags}
        neg = np.concatenate(list(per.values()))
        return {
            "recall_at": {str(t): float(np.mean(pos >= t)) for t in THRESHOLDS},
            "false_accept_bag_rate_at": {str(t): float(np.mean(neg >= t)) for t in THRESHOLDS},
            "false_accept_by_subset_at_0.5": {n: float(np.mean(v >= 0.5)) for n, v in per.items()},
            "holdout_positive_bags": int(len(pos)), "holdout_negative_bags": int(len(neg)),
        }

    guard("evaluation")
    cand_metrics = eval_model(model)
    base_metrics = eval_model(baseline)

    guard("export")
    out_dir = P.CANDIDATES_DIR / f"{dt.date.today():%Y%m%d}-{tag}"
    if out_dir.exists():
        raise StopRun(f"candidate directory already exists: {out_dir}")
    out_dir.mkdir(parents=True)
    try:
        temp = out_dir / ".alexa_synth.tmp.onnx"
        export_parity = F.export_onnx(model, temp, seed)
        model_path = out_dir / "alexa_synth.onnx"
        F.publish_no_overwrite(temp, model_path)
        report = {
            "schema": "memoir.wake-synthetic-candidate-report.v1",
            "candidate_dir": str(out_dir),
            "model": {"file": "alexa_synth.onnx", "sha256": F.sha256_file(model_path),
                      "bytes": model_path.stat().st_size},
            "disclaimer": DISCLAIMER, "deployment_changed": False, "production_eligible": False,
            "accuracy_claim": False, "runtime_config_touched": False, "alexa_user_replaced": False,
            "split_basis": _plan_summary(keys, items, arch_rows, unavailable)["split_basis"],
            "leakage_checks": leak,
            "data_counts": {
                "synthetic_positive_bags": {"train": len(pos_train), "holdout": len(pos_hold)},
                "synthetic_negative_bags": {f"{k[0]}_{k[1]}": len(v) for k, v in syn.items()
                                            if k[0] != "positive"},
                "archive_negative_clips": {"train": len(arch_bags["train"]),
                                           "holdout": len(arch_bags["holdout"]),
                                           "unavailable_audio_excluded": unavailable},
                "generic_negative_bags": {"train": len(gen_bags["train"]), "holdout": len(gen_bags["holdout"])},
                "skipped_items": dict(skipped),
            },
            "provenance": {
                "synthetic": {"tts": "Piper en_US-lessac-medium (single voice, single speaker)",
                              "voices": keys, "items": len(items),
                              "augmentation_noise_sources": dict(augment_sources)},
                "archive": {"source": "assistant/experimental/wake_train/human_reviews.json, presence=negative",
                            "label_basis": "human listening review (whole clip negative)",
                            "positives_used": False, "split_key": "capture group via session_split"},
                "generic": {"source": str(D.GENERIC), "split": "contiguous time blocks, 85/15"},
                "stock_initialisation": {"path": str(F.ct.STOCK_MODEL),
                                         "sha256": F.sha256_file(F.ct.STOCK_MODEL)},
            },
            "training": {"seed": seed, "epochs": epochs, "lr": F.LR, "epoch_mean_losses": losses,
                         "stock_import_parity_max_abs_diff": parity,
                         "export_parity_max_abs_diff": export_parity,
                         "wall_seconds": round(time.monotonic() - t0, 1)},
            "validation": {"candidate": cand_metrics, "stock_baseline": base_metrics,
                           "thresholds": list(THRESHOLDS), "unit": "clip/bag level, not per hour of audio"},
        }
        (out_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                                             encoding="utf-8")
        (out_dir / "report.md").write_text(_report_md(report), encoding="utf-8")
        log(f"candidate written: {model_path}")
        return out_dir
    except Exception:
        shutil.rmtree(out_dir, ignore_errors=True)
        raise


def _report_md(r: dict) -> str:
    c, b = r["validation"]["candidate"], r["validation"]["stock_baseline"]
    lines = [
        "# Synthetic alexa candidate", "", r["disclaimer"], "",
        f"- Model: {r['model']['file']} sha256 {r['model']['sha256']} ({r['model']['bytes']} bytes)",
        f"- Split basis: {r['split_basis']}",
        f"- Leakage checks: {json.dumps(r['leakage_checks'])}",
        f"- Deployment changed: {r['deployment_changed']}; production eligible: {r['production_eligible']}",
        "", "## Data counts", "", "```json", json.dumps(r["data_counts"], indent=2), "```",
        "", "## Provenance", "", "```json", json.dumps(r["provenance"], indent=2), "```",
        "", "## Validation (held-out, clip/bag level)", "", "| metric | candidate | stock baseline |", "|---|---|---|",
    ]
    for t in r["validation"]["thresholds"]:
        lines.append(f"| recall @ {t} | {c['recall_at'][str(t)]:.3f} | {b['recall_at'][str(t)]:.3f} |")
    for t in r["validation"]["thresholds"]:
        lines.append(f"| false-accept bag rate @ {t} | {c['false_accept_bag_rate_at'][str(t)]:.3f} | "
                     f"{b['false_accept_bag_rate_at'][str(t)]:.3f} |")
    lines += ["", f"Training epoch mean losses: {r['training']['epoch_mean_losses']}",
              f"Parity: stock import {r['training']['stock_import_parity_max_abs_diff']:.2e}, "
              f"export {r['training']['export_parity_max_abs_diff']:.2e}", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m assistant.wakebench.train")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="print the plan; write nothing")
    mode.add_argument("--run", action="store_true", help="train one candidate")
    parser.add_argument("--tag", default="synthtts-lessac")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=P.SEED)
    args = parser.parse_args(argv)
    try:
        if args.dry_run:
            dry_run(args.seed)
            return 0
        if not 1 <= args.epochs <= 8:
            parser.error("--epochs must be 1..8")
        print(run(args.seed, args.epochs, args.tag))
        return 0
    except StopRun as exc:
        print(f"STOPPED: {exc}", file=sys.stderr)
        return 2
