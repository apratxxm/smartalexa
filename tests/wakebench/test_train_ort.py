"""Static-batch ONNX feeding: the stock model has a fixed batch of 1 and must receive samples one at a time."""
from __future__ import annotations

import numpy as np
import pytest

from assistant.wakebench.train import fit as F


class FakeInput:
    def __init__(self, name, shape):
        self.name = name
        self.shape = shape


class FakeSession:
    """Mimics onnxruntime: rejects any batch that does not match the declared static dim."""

    def __init__(self, batch_dim):
        self._dim = batch_dim
        self.calls = []

    def get_inputs(self):
        return [FakeInput("onnx::Flatten_0", [self._dim, 16, 96])]

    def run(self, outputs, feed):
        x = feed["onnx::Flatten_0"] if "onnx::Flatten_0" in feed else feed["input"]
        if isinstance(self._dim, int) and x.shape[0] != self._dim:
            raise RuntimeError(f"Got invalid dimensions for input: Got: {x.shape[0]} Expected: {self._dim}")
        self.calls.append(x.shape[0])
        # output depends on each sample only, so chunking must reproduce the full-batch answer
        return [x.reshape(x.shape[0], -1).sum(axis=1, keepdims=True).astype(np.float32)]


def _batch(n):
    return np.arange(n * 16 * 96, dtype=np.float32).reshape(n, 16, 96) / 1e4


def test_static_batch_of_one_is_fed_per_sample_and_matches_full_result():
    sess = FakeSession(batch_dim=1)
    x = _batch(32)
    out = F.run_static_batch(sess, "onnx::Flatten_0", "out", x)
    assert out.shape == (32, 1)
    assert sess.calls == [1] * 32
    expected = x.reshape(32, -1).sum(axis=1, keepdims=True)
    assert np.allclose(out, expected)


def test_dynamic_batch_gets_whole_batch_in_one_call():
    sess = FakeSession(batch_dim="batch")
    out = F.run_static_batch(sess, "onnx::Flatten_0", "out", _batch(32))
    assert out.shape == (32, 1)
    assert sess.calls == [32]


def test_fake_static_batch_one_rejects_whole_batch():
    sess = FakeSession(batch_dim=1)
    with pytest.raises(RuntimeError):
        sess.run(["out"], {"onnx::Flatten_0": _batch(32)})
