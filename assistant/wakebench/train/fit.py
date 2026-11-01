"""Stock-initialised MIL fine-tune, export and parity checks.

Reuses from assistant.experimental.wake_train.candidate_training: extract_bag (front-end
features), _load_stock_weights, STOCK_MODEL. The fit loop below mirrors fit_and_export's
optimizer, loss and distillation term, but fit_and_export hard-codes machine-consensus
provenance in its report, so it cannot emit synthetic/archive provenance.
"""
from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path

import numpy as np

from assistant.experimental.wake_train import candidate_training as ct

LR = 1e-5
PARITY_SEED_XOR = 0x51A7


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _build(torch):
    from torch import nn

    class SmallMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(16 * 96, 128)
            self.fc2 = nn.Linear(128, 128)
            self.fc3 = nn.Linear(128, 1)

        def forward(self, x):
            x = x.reshape(x.shape[0], -1)
            return torch.sigmoid(self.fc3(torch.relu(self.fc2(torch.relu(self.fc1(x))))))

    return SmallMLP


def run_static_batch(session, feed: str, out: str, x: np.ndarray) -> np.ndarray:
    """Run an ONNX session honouring a static batch dim (e.g. the stock model's fixed batch of 1).

    If the first input dim is a fixed int, feed it in chunks of that size and concatenate.
    Dynamic (symbolic) batch dims get the whole batch in one call.
    """
    dim = session.get_inputs()[0].shape[0]
    if isinstance(dim, int) and dim >= 1 and x.shape[0] != dim:
        parts = [session.run([out], {feed: x[i:i + dim]})[0] for i in range(0, x.shape[0], dim)]
        return np.concatenate(parts, axis=0)
    return session.run([out], {feed: x})[0]


def _parity_input(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed ^ PARITY_SEED_XOR)
    x = rng.standard_normal((31, 16, 96), dtype=np.float32)
    return np.concatenate((x, np.zeros((1, 16, 96), np.float32)), axis=0)


def _ort_options():
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    return opts


def stock_parity(seed: int) -> tuple[object, float]:
    """Stock ONNX vs imported torch classifier must agree before any training."""
    import torch
    import onnxruntime as ort

    SmallMLP = _build(torch)
    baseline = SmallMLP().cpu()
    ct._load_stock_weights(ct.STOCK_MODEL, baseline, torch)
    baseline.eval()
    x = _parity_input(seed)
    session = ort.InferenceSession(str(ct.STOCK_MODEL), sess_options=_ort_options(),
                                   providers=["CPUExecutionProvider"])
    expected = run_static_batch(session, session.get_inputs()[0].name,
                                 session.get_outputs()[0].name, x)
    with torch.no_grad():
        actual = baseline(torch.from_numpy(x)).numpy()
    diff = float(np.max(np.abs(expected - actual)))
    if expected.shape != (32, 1) or not np.isfinite(expected).all() or diff > 1e-4:
        raise ValueError(f"stock ONNX and imported classifier parity failed (max diff {diff})")
    return baseline, diff


def fit(pos_bags, neg_bags, *, seed: int, epochs: int, log=print):
    """Returns (candidate_model, mean_losses_per_epoch, baseline_model). Bags are float16 (W,16,96)."""
    import torch
    from torch import nn

    if not 1 <= epochs <= 8:
        raise ValueError("epochs must be between 1 and 8")
    if not pos_bags or not neg_bags:
        raise ValueError("both positive and negative bags are required")
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    baseline, _ = stock_parity(seed)
    baseline.eval()
    candidate = copy.deepcopy(baseline)
    optimizer = torch.optim.Adam(candidate.parameters(), lr=LR)
    rng = np.random.default_rng(seed)
    losses = []
    for epoch in range(epochs):
        pos_order = rng.permutation(len(pos_bags))
        neg_order = rng.permutation(len(neg_bags))
        candidate.train()
        epoch_losses = []
        for index in range(max(len(pos_bags), len(neg_bags))):
            pair = ((pos_bags[int(pos_order[index % len(pos_order)])], 1.0),
                    (neg_bags[int(neg_order[index % len(neg_order)])], 0.0))
            for bag, label in pair:
                x = torch.from_numpy(np.asarray(bag, dtype=np.float32))
                with torch.no_grad():
                    base_probs = baseline(x).detach()
                probs = candidate(x)
                bag_prob = probs.max()
                y = torch.tensor(label, dtype=torch.float32)
                loss = nn.functional.binary_cross_entropy(bag_prob.clamp(1e-6, 1 - 1e-6), y)
                loss = loss + 0.5 * torch.mean((probs - base_probs) ** 2)
                if not torch.isfinite(loss):
                    raise ValueError("non-finite training loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(loss.detach()))
        losses.append(float(np.mean(epoch_losses)))
        log(f"epoch {epoch + 1}/{epochs} mean_loss={losses[-1]:.5f}")
    if any(not torch.isfinite(p).all() for p in candidate.parameters()):
        raise ValueError("non-finite trained weights")
    candidate.eval()
    return candidate, losses, baseline


def max_probs(model, bags) -> np.ndarray:
    import torch

    out = []
    model.eval()
    with torch.no_grad():
        for bag in bags:
            x = torch.from_numpy(np.asarray(bag, dtype=np.float32))
            out.append(float(model(x).max()))
    return np.asarray(out, dtype=np.float64)


def export_onnx(model, path: Path, seed: int) -> float:
    """Export to ONNX (opset 13, dynamic batch) and return max abs parity diff vs torch."""
    import onnx
    import onnxruntime as ort
    import torch

    dummy = torch.zeros((2, 16, 96), dtype=torch.float32)
    torch.onnx.export(model, dummy, str(path), input_names=["input"], output_names=["output"],
                      dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
                      opset_version=13, dynamo=False)
    onnx.checker.check_model(onnx.load(str(path)))
    session = ort.InferenceSession(str(path), sess_options=_ort_options(), providers=["CPUExecutionProvider"])
    x = _parity_input(seed)
    with torch.no_grad():
        expected = model(torch.from_numpy(x)).numpy()
    actual = run_static_batch(session, "input", "output", x)
    if actual.shape != (32, 1) or not np.isfinite(actual).all() or (actual < 0).any() or (actual > 1).any():
        raise ValueError("exported ONNX output invalid")
    diff = float(np.max(np.abs(actual - expected)))
    if diff > 1e-4:
        raise ValueError(f"export parity failed (max diff {diff})")
    return diff


def publish_no_overwrite(temp: Path, target: Path) -> None:
    if target.exists():
        raise FileExistsError(f"refusing to overwrite existing candidate: {target}")
    os.replace(temp, target)
