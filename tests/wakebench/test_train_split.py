"""Split leakage: no voice, phrase, archive session or generic region on both train and holdout."""
from __future__ import annotations

import numpy as np

from assistant.wakebench.train import data as D
from assistant.wakebench.train import plan as P
from assistant.wakebench.train.cli import leakage_checks


MULTI = ["voice_a#0", "voice_b#0", "voice_c#0"]


def _voices_by_split(items, split):
    return {it.voice for it in items if it.split == split and it.kind == "positive"}


def test_multi_voice_holdout_voice_never_in_train():
    items = P.build_plan(MULTI, seed=1337)
    train_voices = _voices_by_split(items, "train")
    hold_voices = _voices_by_split(items, "holdout")
    assert hold_voices and train_voices
    assert train_voices.isdisjoint(hold_voices)
    assert all(it.voice in MULTI for it in items)
    assert P.voice_split(MULTI)[2] is True


def test_single_voice_fallback_is_text_held_out_and_flagged():
    items = P.build_plan(["only#0"], seed=1337)
    train_text = {it.text for it in items if it.split == "train" and it.kind == "positive"}
    hold_text = {it.text for it in items if it.split == "holdout" and it.kind == "positive"}
    assert train_text.isdisjoint(hold_text)
    assert P.voice_split(["only#0"])[2] is False
    checks = leakage_checks(items, [], ["only#0"])
    assert checks["ok"] is True
    assert checks["independent_voice_validation"] is False


def test_negative_phrases_disjoint_between_splits():
    items = P.build_plan(MULTI, seed=1337)
    neg_train = {it.text for it in items if it.split == "train" and it.kind != "positive"}
    neg_hold = {it.text for it in items if it.split == "holdout" and it.kind != "positive"}
    assert neg_train.isdisjoint(neg_hold)


def test_archive_sessions_cannot_straddle_splits():
    items = P.build_plan(MULTI, seed=1337)
    rows = [{"group_id": "sessA", "split": "train"}, {"group_id": "sessA", "split": "holdout"}]
    checks = leakage_checks(items, rows, MULTI)
    assert checks["archive_group_overlap"] == 1
    assert checks["ok"] is False


def test_generic_windows_are_contiguous_and_region_disjoint(tmp_path):
    rows = np.arange(5000, dtype=np.float32)[:, None] * np.ones((1, 96), np.float32)
    path = tmp_path / "generic.npy"
    np.save(path, rows)
    gen = D.GenericWindows(path)
    rng = np.random.default_rng(0)
    train = gen.bags("train", 20, rng)
    hold = gen.bags("holdout", 20, rng)
    train_idx = {int(v) for bag in train for v in bag[..., 0].ravel()}
    hold_idx = {int(v) for bag in hold for v in bag[..., 0].ravel()}
    assert train_idx and hold_idx
    assert max(train_idx) < min(hold_idx)
    for bag in train[:3]:
        first = bag[:, :, 0]
        assert np.array_equal(np.diff(first[0]), np.ones(D.FRAMES - 1))        # frames inside a window
        assert np.array_equal(np.diff(first[:, 0]), np.ones(first.shape[0] - 1))  # windows step by one frame
    assert train[0].shape == (D.WINDOWS_PER_BAG, D.FRAMES, 96)


def test_room_bank_only_takes_quiet_segments():
    loud = np.full(16000 * 2, 20000, np.int16)
    quiet = (np.random.default_rng(1).standard_normal(16000 * 2) * 5).astype(np.int16)
    bank = D.room_bank_from([loud, quiet])
    assert len(bank) == 1
    assert bank[0].size == int(0.5 * 16000)
