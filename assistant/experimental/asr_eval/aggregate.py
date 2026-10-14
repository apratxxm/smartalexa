#!/usr/bin/env python3
"""Collect results_*.json into summary.json + a table. Re-derives the secondary
phonetic-name metric from the stored transcripts (no re-decoding)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from scoring import name_ok_phonetic  # noqa: E402

known = json.loads((HERE / "testset.json").read_text(encoding="utf-8"))["hotwords_artists"]


def main() -> None:
    rows = []
    for p in sorted(HERE.glob("results_*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        pc = d["per_clip"]
        ph = sum(name_ok_phonetic(r["predicted_action"], r["predicted_query"], r["predicted_value"],
                                  r["expected_action"], r["expected_query"], r["expected_value"], known)
                 for r in pc) / len(pc)
        # music hallucination also reported as "any non-empty text" count
        nonempty = sum(bool(r["predicted_transcript"].strip()) for r in d["per_negative_clip"])
        rows.append({
            "variant": d["variant"], "hotwords": d.get("hotwords"),
            "hotwords_score": d.get("hotwords_score"), "device": d.get("device", "cpu"),
            "n_command_clips": d["n_command_clips"],
            "command_accuracy": d["command_accuracy"], "name_accuracy": d["name_accuracy"],
            "name_accuracy_phonetic": round(ph, 4),
            "hallucinated_commands": d["hallucinations"], "nonempty_text_on_music": nonempty,
            "n_negative_clips": d["n_negative_clips"], "median_latency_ms": d["median_latency_ms"],
        })
    print(f"{'variant':<40}{'cmd':>7}{'name':>7}{'name_ph':>9}{'hall':>7}{'text':>7}{'med_ms':>9}")
    for r in rows:
        print(f"{r['variant']:<40}{r['command_accuracy']:>7.3f}{r['name_accuracy']:>7.3f}"
              f"{r['name_accuracy_phonetic']:>9.3f}{r['hallucinated_commands']:>4}/{r['n_negative_clips']:<2}"
              f"{r['nonempty_text_on_music']:>4}/{r['n_negative_clips']:<2}{r['median_latency_ms']:>9.0f}")
    (HERE / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
