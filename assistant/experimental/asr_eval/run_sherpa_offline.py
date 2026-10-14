#!/usr/bin/env python3
"""Evaluate a sherpa-onnx OFFLINE transducer (Parakeet-TDT via nemo_transducer, or
a plain icefall transducer) with greedy search and with modified_beam_search +
hotwords (artist names).  Run with A:\pisha\memoir\.venv-asr.

  python run_sherpa_offline.py --model-dir A:\asr-models\sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8 \
      --tag parakeet-tdt-0.6b-v2-int8 --model-type nemo_transducer --scores 1.5,3.0
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from evalcore import evaluate, load_testset  # noqa: E402
from sherpa_hotwords import build_hotwords_file  # noqa: E402


def first(model_dir: Path, *pats) -> str:
    for pat in pats:
        hits = sorted(model_dir.glob(pat))
        if hits:
            return str(hits[0])
    raise FileNotFoundError(pats)


def build(model_dir, model_type, hotwords_file="", score=1.5, threads=4, paths=4):
    import sherpa_onnx
    kw = dict(encoder=first(model_dir, "encoder*.int8.onnx", "encoder*.onnx"),
              decoder=first(model_dir, "decoder*.int8.onnx", "decoder*.onnx"),
              joiner=first(model_dir, "joiner*.int8.onnx", "joiner*.onnx"),
              tokens=str(model_dir / "tokens.txt"), num_threads=threads, sample_rate=16000,
              feature_dim=80, provider="cpu", model_type=model_type)
    if hotwords_file:
        kw.update(decoding_method="modified_beam_search", max_active_paths=paths,
                  hotwords_file=hotwords_file, hotwords_score=score)
    else:
        kw.update(decoding_method="greedy_search")
    return sherpa_onnx.OfflineRecognizer.from_transducer(**kw)


def preprocess(audio, mode):
    if "norm" in mode:
        peak = float(np.max(np.abs(audio))) or 1.0
        audio = audio * (0.9 / peak)
    if "pad" in mode:
        z = np.zeros(8000, dtype=np.float32)  # 0.5 s of silence each side
        audio = np.concatenate([z, audio, z])
    return audio.astype(np.float32)


def make_transcribe(rec, mode="none"):
    def f(audio):
        audio = preprocess(audio, mode)
        t0 = time.perf_counter()
        s = rec.create_stream()
        s.accept_waveform(16000, audio)
        rec.decode_stream(s)
        return s.result.text.strip(), (time.perf_counter() - t0) * 1000.0
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model-type", default="nemo_transducer")
    ap.add_argument("--scores", default="1.5,3.0", help="hotwords_score values to run")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--paths", type=int, default=4)
    ap.add_argument("--with-hints", action="store_true")
    ap.add_argument("--no-plain", action="store_true")
    ap.add_argument("--preproc", default="none", help="none|norm|pad|normpad")
    a = ap.parse_args()
    md = Path(a.model_dir)
    m = load_testset()
    phrases = m["hotwords_artists"] + (m["hotwords_hints"] if a.with_hints else [])
    hw, skipped = build_hotwords_file(phrases, HERE / f"hotwords_{a.tag}.txt", md / "tokens.txt")
    print("hotwords skipped (untokenisable):", skipped, flush=True)
    tag = a.tag
    a.tag = tag  # variant names below use a.tag (suffix added by caller)
    meta = dict(model=a.tag, preproc=a.preproc, device="cpu", num_threads=a.threads, decoder_hw="modified_beam_search")
    if not a.no_plain:
        rec = build(md, a.model_type, threads=a.threads)
        transcribe = make_transcribe(rec, a.preproc)
        transcribe(np.zeros(16000, dtype=np.float32))
        evaluate(a.tag, transcribe, dict(meta, hotwords=False, decoder_hw="greedy_search"), m)
    for sc in [float(x) for x in a.scores.split(",") if x]:
        rec = build(md, a.model_type, str(hw), sc, a.threads, a.paths)
        transcribe = make_transcribe(rec, a.preproc)
        transcribe(np.zeros(16000, dtype=np.float32))
        evaluate(f"{a.tag}+hotwords{sc:g}", transcribe, dict(meta, hotwords=True, hotwords_score=sc), m)


if __name__ == "__main__":
    main()
