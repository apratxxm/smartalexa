"""Seeded determinism of the synthetic plan, augmentation and RIR (no Piper, no torch)."""
from __future__ import annotations

import json

import numpy as np

from assistant.wakebench.train import plan as P
from assistant.wakebench.train import synth as S


KEYS = ["en_US-lessac-medium#0"]
MULTI = ["voice_a#0", "voice_b#0", "voice_c#0"]


def test_plan_is_identical_for_same_seed_and_differs_for_other_seed():
    first = P.build_plan(KEYS, seed=1337)
    again = P.build_plan(KEYS, seed=1337)
    other = P.build_plan(KEYS, seed=2024)
    assert first == again
    assert [it.augment_seed for it in first] != [it.augment_seed for it in other]


def test_plan_prosody_unique_per_phrase_and_counts_match_variants():
    items = P.build_plan(KEYS, seed=1337)
    seen = {}
    for it in items:
        key = (it.kind, it.split, it.text)
        prosody = (it.length_scale, it.noise_scale, it.noise_w, it.rate)
        assert prosody not in seen.setdefault(key, set())
        seen[key].add(prosody)
    counts = P.plan_counts(items)
    assert counts["positive_train"] == len(P.POS_TRAIN) * P.VARIANTS[("train", "positive")]
    assert counts["positive_holdout"] == len(P.POS_HOLDOUT) * P.VARIANTS[("holdout", "positive")]


def test_augmentation_is_seeded_and_bounded():
    clip = (np.sin(np.linspace(0, 80 * np.pi, 32000)) * 0.3).astype(np.float32)
    bank = [np.random.default_rng(5).standard_normal(16000).astype(np.float32) * 0.01]
    a1, i1 = S.augment(clip, np.random.default_rng(99), bank)
    a2, i2 = S.augment(clip, np.random.default_rng(99), bank)
    assert np.array_equal(a1, a2) and i1 == i2
    assert a1.dtype == np.float32 and np.isfinite(a1).all()
    assert float(np.max(np.abs(a1))) <= 1.0 + 1e-6


def test_augmentation_uses_fallback_noise_only_when_bank_empty():
    clip = np.ones(16000, np.float32) * 0.1
    seen = set()
    for seed in range(40):
        _, info = S.augment(clip, np.random.default_rng(seed), [])
        if "noise_source" in info:
            seen.add(info["noise_source"])
    assert seen <= {"synthetic_colored_fallback"}


def test_rir_is_seeded_and_normalised():
    h1 = S.synthetic_rir(np.random.default_rng(3))
    h2 = S.synthetic_rir(np.random.default_rng(3))
    assert np.array_equal(h1, h2)
    assert abs(float(np.sum(h1 ** 2)) - 1.0) < 1e-5


def test_resample_rate_scales_duration():
    x = np.arange(16000, dtype=np.float32)
    assert len(S.resample_rate(x, 1.0)) == 16000
    assert abs(len(S.resample_rate(x, 1.25)) - 12800) <= 1


def test_discover_voices_reads_piper_pairs_only(tmp_path):
    (tmp_path / "voice_x.onnx").write_bytes(b"0")
    (tmp_path / "voice_x.onnx.json").write_text(json.dumps({"num_speakers": 2, "espeak": {}}))
    (tmp_path / "silero.onnx").write_bytes(b"0")          # no json: not a Piper voice
    voices = P.discover_voices(tmp_path)
    assert sorted(voices) == ["voice_x#0", "voice_x#1"]
