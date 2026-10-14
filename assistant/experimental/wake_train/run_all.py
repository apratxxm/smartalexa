#!/usr/bin/env python3
"""Convenience driver: runs the full pipeline end to end in order.

    A:\\pisha\\memoir\\.venv-wake\\Scripts\\python.exe run_all.py

Equivalent to running build_manifest.py, prepare_clips.py, synth_tts.py,
extract_features.py, train_classifier.py, eval.py in sequence. Each step
also works standalone (e.g. to re-run just eval.py after inspecting
results), which is how this was actually developed and run.
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
STEPS = [
    "build_manifest.py",
    "prepare_clips.py",
    "synth_tts.py",
    "extract_features.py",
    "train_classifier.py",
    "eval.py",
]


def main():
    for step in STEPS:
        print(f"\n===== {step} =====")
        r = subprocess.run([sys.executable, str(HERE / step)])
        if r.returncode != 0:
            print(f"FAILED at {step} (exit {r.returncode})")
            sys.exit(r.returncode)


if __name__ == "__main__":
    main()
