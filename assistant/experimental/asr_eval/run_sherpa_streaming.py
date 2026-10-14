#!/usr/bin/env python3
"""Evaluate a sherpa-onnx STREAMING zipformer transducer, greedy vs
modified_beam_search + hotwords. Audio fed in 512-sample frames (like the mic
loop) then flushed. Latency = time from end-of-audio feed to final result is
NOT separated here; reported latency is total decode time for the clip (the
streaming decode runs faster than real time so tail latency is much smaller).
Run with A:\pisha\memoir\.venv-asr."""
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
from run_sherpa_offline import first  # noqa: E402

SR = 16000


def build(md: Path, hotwords_file="", score=1.5, threads=2, paths=4, int8=True):
    import sherpa_onnx
    suf = "*.int8.onnx" if int8 else "*.onnx"
    kw = dict(tokens=str(md / "tokens.txt"),
              encoder=first(md, "encoder" + suf if int8 else "encoder-epoch*[0-9].onnx", "encoder*.onnx"),
              decoder=first(md, "decoder" + suf if int8 else "decoder-epoch*[0-9].onnx", "decoder*.onnx"),
              joiner=first(md, "joiner" + suf if int8 else "joiner-epoch*[0-9].onnx", "joiner*.onnx"),
              num_threads=threads, sample_rate=SR, feature_dim=80,
              enable_endpoint_detection=False, provider="cpu")
    if hotwords_file:
        kw.update(decoding_method="modified_beam_search", max_active_paths=paths,
                  hotwords_file=hotwords_file, hotwords_score=score)
    else:
        kw.update(decoding_method="greedy_search")
    return sherpa_onnx.OnlineRecognizer.from_transducer(**kw)


def make_transcribe(rec):
    def f(audio):
        t0 = time.perf_counter()
        s = rec.create_stream()
        for pos in range(0, len(audio), 512):
            s.accept_waveform(SR, audio[pos:pos + 512])
            while rec.is_ready(s):
                rec.decode_stream(s)
        s.accept_waveform(SR, np.zeros(int(0.5 * SR), dtype=np.float32))
        s.input_finished()
        while rec.is_ready(s):
            rec.decode_stream(s)
        return rec.get_result(s).strip() if isinstance(rec.get_result(s), str) else rec.get_result(s).text.strip(), \
            (time.perf_counter() - t0) * 1000.0
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--scores", default="1.5,3.0")
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    md = Path(a.model_dir)
    m = load_testset()
    hw, skipped = build_hotwords_file(m["hotwords_artists"], HERE / f"hotwords_{a.tag}.txt", md / "tokens.txt")
    print("skipped:", skipped, flush=True)
    meta = dict(model=a.tag, device="cpu", num_threads=a.threads)
    rec = build(md, threads=a.threads)
    t = make_transcribe(rec)
    t(np.zeros(SR, dtype=np.float32))
    evaluate(a.tag, t, dict(meta, hotwords=False, decoder_hw="greedy_search"), m)
    for sc in [float(x) for x in a.scores.split(",")]:
        rec = build(md, str(hw), sc, a.threads)
        t = make_transcribe(rec)
        t(np.zeros(SR, dtype=np.float32))
        evaluate(f"{a.tag}+hotwords{sc:g}", t, dict(meta, hotwords=True, hotwords_score=sc), m)


if __name__ == "__main__":
    main()
