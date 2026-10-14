#!/usr/bin/env python3
"""Train the (16, 96) -> sigmoid wake-word classifier and export ONNX.

Architecture mirrors openwakeword.train.Model(model_type="dnn")
(flatten -> linear -> layernorm -> relu -> FCN block -> linear -> sigmoid),
so the exported graph is a drop-in wakeword_models=[...] entry.

Data:
  * train_data.npz (extract_features.py): real/synthetic positives, local
    negatives, tail/pad/music negatives, 63% of the openWakeWord validation
    feature set -- split in TRAIN / inner-VAL (whole clips / blocks).
  * ACAV100M negative-feature chunks (A:\\wake-data\\acav\\chunk_*.bin, float16
    (N,16,96) windows, ~280 h of general audio) -- last 2% of each chunk is
    inner-VAL, rest TRAIN.  Sampled per epoch + hard-negative mining.

Model selection uses ONLY inner-VAL (never the held-out eval split):
    score = stream-level recall(real val positives @0.5) - 3*mean(FP of local/music/tail/
            synthetic/startup negative kinds @0.5) - 100*FP(generic windows @0.5)
CPU only (GPU is shared with the assistant/Ollama).
"""
from __future__ import annotations

import glob
import json
import time
from pathlib import Path

try:
    from .active_learning import candidate_output, require_supported_training
except ImportError:
    from active_learning import candidate_output, require_supported_training

# CLI safety happens before the ML dependencies or large feature arrays load.
if __name__ == "__main__":
    import argparse
    safety = argparse.ArgumentParser(description="Legacy training is blocked pending reviewed-data integration")
    safety.add_argument("--out", required=True, help="new candidate ONNX outside production models")
    safety_args, _ = safety.parse_known_args()
    try:
        candidate_output(safety_args.out)
        require_supported_training("classifier training")
    except ValueError as exc:
        safety.exit(2, f"training refused: {exc}\n")

import numpy as np
import torch
from torch import nn

FEATURES = Path(r"A:\wake-data\features")
ACAV_DIR = Path(r"A:\wake-data\acav")
OUT_MODEL = Path(r"A:\pisha\memoir\assistant\models\wake")

SEED = 1337
LAYER_DIM = 128
DROPOUT = 0.1
BATCH = 512
LR = 1e-3
WD = 1e-4
POS_CLASS_W = 2.0
GEN_PER_EPOCH = 120_000
ROUND_EPOCHS = [8, 6, 6, 6, 6, 6, 6]
ROUND_LR = [1e-3, 5e-4, 3e-4, 2e-4, 1e-4, 1e-4, 1e-4]
HARD_THR = 0.15
HARD_CAP = 60_000
ACAV_WIN_PER_CHUNK = 160_000
EMA_DECAY = 0.998
LABEL_SMOOTH = 0.0
SAVE_STATE = None
ARCH = "ln"
REALIGN = False
INIT_STOCK = False
OUT_ONNX = None  # explicit separate candidate output required
torch.set_num_threads(6)
NAMES = {0: "real_pos", 1: "synth_pos", 2: "generic", 3: "local", 4: "synth_neg", 5: "tail", 6: "pad", 7: "music", 8: "startup"}


class FCNBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fcn_layer = nn.Linear(dim, dim)
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.relu(self.layer_norm(self.fcn_layer(x)))


class Net(nn.Module):
    def __init__(self, layer_dim=LAYER_DIM, dropout=DROPOUT):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(16 * 96, layer_dim)
        self.relu1 = nn.ReLU()
        self.layernorm1 = nn.LayerNorm(layer_dim)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([FCNBlock(layer_dim)])
        self.last_layer = nn.Linear(layer_dim, 1)
        self.last_act = nn.Sigmoid()

    def forward(self, x):
        x = self.drop(self.relu1(self.layernorm1(self.layer1(self.flatten(x)))))
        for b in self.blocks:
            x = b(x)
        return self.last_act(self.last_layer(x))


class StockMLP(nn.Module):
    """Same topology as the stock openWakeWord alexa_v0.1 model
    (flatten -> 128 -> relu -> 128 -> relu -> 1 -> sigmoid)."""

    def __init__(self, layer_dim=128, dropout=0.0):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 96, layer_dim)
        self.fc2 = nn.Linear(layer_dim, layer_dim)
        self.fc3 = nn.Linear(layer_dim, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.drop(torch.relu(self.fc1(self.flatten(x))))
        x = self.drop(torch.relu(self.fc2(x)))
        return torch.sigmoid(self.fc3(x))


def load_stock_into(net, path=r"A:/wake-data/models/alexa_v0.1.onnx"):
    import onnx
    from onnx import numpy_helper
    w = {i.name: numpy_helper.to_array(i) for i in onnx.load(path).graph.initializer}
    with torch.no_grad():
        for mod, k in ((net.fc1, "1"), (net.fc2, "3"), (net.fc3, "5")):
            mod.weight.copy_(torch.from_numpy(w[k + ".weight"])); mod.bias.copy_(torch.from_numpy(w[k + ".bias"]))


def make_net(layer_dim=None, dropout=None):
    ld = LAYER_DIM if layer_dim is None else layer_dim
    dp = DROPOUT if dropout is None else dropout
    if ARCH == "stock":
        n = StockMLP(ld, dp)
        if INIT_STOCK:
            load_stock_into(n)
        return n
    return Net(ld, dp)


def bce(p, y):
    return nn.functional.binary_cross_entropy(p.clamp(1e-6, 1 - 1e-6), y, reduction="none")


@torch.no_grad()
def predict_np(model, X, bs=16384):
    """X: numpy/memmap (N,16,96) float16/32 -> numpy scores."""
    model.eval()
    out = np.empty(len(X), np.float32)
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(np.asarray(X[i:i + bs], dtype=np.float32))
        out[i:i + bs] = model(xb).squeeze(-1).numpy()
    return out


def main(tag="final"):
    candidate_output(OUT_ONNX)
    require_supported_training("classifier training")
    global_cfg = dict(layer_dim=LAYER_DIM, dropout=DROPOUT, wd=WD, ema=EMA_DECAY, pcw=POS_CLASS_W,
                      ls=LABEL_SMOOTH, lr_scale=1.0)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    d = np.load(FEATURES / "train_data.npz")
    X, y, w, kind, val = d["X"], d["y"], d["w"], d["kind"], d["val"]
    is_gen = kind == 2
    # --- "local" pool: everything except generic features ---
    loc_tr = (~is_gen) & (~val)
    Xl = torch.from_numpy(X[loc_tr]); yl = torch.from_numpy(y[loc_tr]); wl = torch.from_numpy(w[loc_tr])
    # --- generic pools: valset-train windows + ACAV chunks (float16, memmapped) ---
    gen_valset_tr = np.where(is_gen & ~val)[0]
    Xg_vs = X[gen_valset_tr]
    chunks = sorted(glob.glob(str(ACAV_DIR / "chunk_*.bin")))
    acav_tr, acav_val = [], []
    for c in chunks:
        n = Path(c).stat().st_size // 3072
        mm = np.memmap(c, dtype=np.float16, mode="r", shape=(n, 16, 96))
        cut = int(n * 0.98)
        acav_tr.append(mm[:cut]); acav_val.append(mm[cut:])
    n_acav_tr = sum(len(a) for a in acav_tr); n_acav_val = sum(len(a) for a in acav_val)
    print(f"local train windows {int(loc_tr.sum())} (pos {int(((y==1)&loc_tr).sum())}); "
          f"generic: valset {len(Xg_vs)} + ACAV {n_acav_tr} train windows ({len(chunks)} chunks), "
          f"ACAV inner-val {n_acav_val}")

    def gen_get(idx):
        """idx: sorted global indices into [valset | acav chunks] -> float32 tensor."""
        out = np.empty((len(idx), 16, 96), np.float16)
        offs = np.cumsum([0, len(Xg_vs)] + [len(a) for a in acav_tr])
        srcs = [Xg_vs] + acav_tr
        for si, src in enumerate(srcs):
            m = (idx >= offs[si]) & (idx < offs[si + 1])
            if m.any():
                out[m] = np.asarray(src[idx[m] - offs[si]], dtype=np.float16)
        return torch.from_numpy(out)

    n_gen = len(Xg_vs) + n_acav_tr

    def sample_gen(n):
        """random contiguous 256-window slices (cheap memmap reads) across the generic pool."""
        starts = rng.integers(0, n_gen - 256, size=n // 256)
        idx = np.unique((starts[:, None] + np.arange(256)[None, :]).reshape(-1))
        return idx

    # --- inner-val sets ---
    vm = val
    Xv = torch.from_numpy(X[vm]); yv = y[vm]; kv = kind[vm]
    n_windows_total = int(len(X))
    del X, d
    import gc; gc.collect()
    # generic val: valset-val windows (kind 2 within vm) + ACAV val slice (subsample 20k)
    acav_val_cat = np.concatenate([np.asarray(a[::max(1, len(a) // 5000)], dtype=np.float32) for a in acav_val]) \
        if acav_val else np.zeros((0, 16, 96), np.float32)
    Xav = torch.from_numpy(acav_val_cat)
    print("inner-val:", {NAMES[k]: int((kv == k).sum()) for k in NAMES}, "acav_val_sub", len(Xav))

    def val_metrics(model):
        model.eval()
        with torch.no_grad():
            pv = np.concatenate([model(Xv[i:i + 8192].float()).squeeze(-1).numpy() for i in range(0, len(Xv), 8192)])
            pa = model(Xav).squeeze(-1).numpy() if len(Xav) else np.zeros(0)
        rec = float((pv[kv == 0] > 0.5).mean())
        pr = pv[kv == 0]
        rec_stream = float((pr[: len(pr) // 8 * 8].reshape(-1, 8).max(axis=1) > 0.5).mean())
        fp = {NAMES[k]: float((pv[kv == k] > 0.5).mean()) for k in (3, 4, 5, 6, 7, 8) if (kv == k).any()}
        gen_scores = pv[kv == 2]  # valset-val windows only (ACAV val may overlap stock-model training data)
        fpg = float((gen_scores > 0.5).mean())
        gs = np.sort(gen_scores)[::-1]
        rec_at = {}
        for fpt in (1e-3, 3e-4):
            ix = int(len(gs) * fpt)
            thr = max(float(gs[ix]) if ix < len(gs) else 0.0, 0.05)
            rec_at[fpt] = float((pr[: len(pr) // 8 * 8].reshape(-1, 8).max(axis=1) > thr).mean())
        # objective: stream recall at matched generic-negative FP (1e-3 and 3e-4 per window), minus
        # penalty for user-speech / music / near-word / start-up false positives at 0.5
        score = 0.5 * (rec_at[1e-3] + rec_at[3e-4]) - 1.5 * float(np.mean(list(fp.values())))
        recs = float((pv[kv == 1] > 0.5).mean())
        return score, {"rec_at_fp1e-3": rec_at[1e-3], "rec_at_fp3e-4": rec_at[3e-4], "rec_real": rec, "rec_real_stream": rec_stream, "rec_synth": recs, "fp_generic": fpg, **{f"fp_{k}": v for k, v in fp.items()}}

    # ---- hard local negatives: windows the v1 ensemble scores > 0.3 get x3 weight ----
    import onnxruntime as ort
    s_v1 = ort.InferenceSession(r"A:/pisha/memoir/assistant/models/wake/alexa_user_v1.onnx", providers=["CPUExecutionProvider"])
    Xl_np = Xl.numpy()
    p_v1 = np.concatenate([s_v1.run(None, {"input": Xl_np[i:i + 8192].astype(np.float32)})[0].reshape(-1)
                           for i in range(0, len(Xl_np), 8192)])
    neg_loc = (yl.numpy() < 0.5)
    hard_v1 = neg_loc & (p_v1 > 0.3)
    wl_base = wl.clone()
    wl_base[torch.from_numpy(hard_v1)] *= 3.0
    wl[:] = wl_base
    print(f"v1 ensemble: {int(hard_v1.sum())} of {int(neg_loc.sum())} local negative windows scored >0.3 (x3 weight)", flush=True)
    neg_loc_t = torch.from_numpy(neg_loc)
    real_idx = torch.where(torch.from_numpy(kind[loc_tr] == 0))[0]
    assert len(real_idx) % 8 == 0
    model = make_net()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    ema = make_net(); ema.load_state_dict(model.state_dict()); ema.eval()
    for p_ in ema.parameters():
        p_.requires_grad_(False)
    best = {"score": -1e9, "state": None}
    hard_X = hard_y = None
    hist = []
    n_steps = 0
    t0 = time.time()
    n_hard_hist = []
    for rnd, n_ep in enumerate(ROUND_EPOCHS):
        for g_ in opt.param_groups:
            g_['lr'] = ROUND_LR[rnd]
        if REALIGN and rnd > 0:
            # EM-style re-alignment: within each real positive stream keep only the 3 consecutive
            # windows around the best-scoring one as positives; ignore (weight 0) the rest
            S = predict_np(ema, Xl[real_idx]).reshape(-1, 8)
            sm = np.stack([np.convolve(np.pad(r, 1, mode="edge"), np.ones(3) / 3, mode="valid") for r in S])
            b = sm.argmax(1)
            keep = np.abs(np.arange(8)[None, :] - b[:, None]) <= 1
            wl[real_idx] = torch.from_numpy(np.where(keep.reshape(-1), 1.0, 0.0).astype(np.float32))
            print(f"[round {rnd}] realigned positives: kept {int(keep.sum())}/{keep.size} windows", flush=True)
        if rnd > 0:
            pl = torch.from_numpy(predict_np(ema, Xl))
            dyn = neg_loc_t & (pl > 0.3)
            wl[:] = wl_base
            wl[dyn] *= 2.0
            print(f"[round {rnd}] local negatives >0.3 up-weighted x2: {int(dyn.sum())}", flush=True)
            # ---- hard-negative mining over ALL train generic windows + local negatives ----
            ps = []
            hard_idx = []
            Xg_all_scores = []
            for si, src in enumerate([Xg_vs] + acav_tr):
                ps.append(predict_np(model, src))
            sg = np.concatenate(ps)
            order = np.where(sg > HARD_THR)[0]
            if len(order) > HARD_CAP:
                order = order[np.argsort(-sg[order])[:HARD_CAP]]
            order = np.sort(order)
            hard_X = gen_get(order)
            n_hard_hist.append(int(len(order)))
            print(f"[round {rnd}] mined {len(order)} hard generic negatives (>{HARD_THR}); "
                  f"generic train FP@0.5 = {(sg > 0.5).mean():.5f}  time {time.time()-t0:.0f}s", flush=True)
        for ep in range(n_ep):
            gi = sample_gen(GEN_PER_EPOCH)
            Xgen = gen_get(gi)
            srcs = [(Xl, yl, wl), (Xgen, torch.zeros(len(Xgen)), torch.ones(len(Xgen)))]
            if hard_X is not None:
                srcs.append((hard_X, torch.zeros(len(hard_X)), torch.ones(len(hard_X))))
            lens = [len(s[0]) for s in srcs]
            offs = np.cumsum([0] + lens)
            n = int(offs[-1])
            model.train()
            perm = torch.randperm(n)
            tot = 0.0
            for i in range(0, n, BATCH):
                bi = perm[i:i + BATCH]
                xb = torch.empty(len(bi), 16, 96); ye_b = torch.empty(len(bi)); we_b = torch.empty(len(bi))
                for (Xs, ys_, ws_), o0, o1 in zip(srcs, offs[:-1], offs[1:]):
                    m_ = (bi >= o0) & (bi < o1)
                    if m_.any():
                        li = bi[m_] - o0
                        xb[m_] = Xs[li].float(); ye_b[m_] = ys_[li]; we_b[m_] = ws_[li]
                p = model(xb).squeeze(-1)
                cw = torch.where(ye_b > 0.5, torch.full_like(p, POS_CLASS_W), torch.ones_like(p))
                yb = ye_b * (1 - LABEL_SMOOTH) + 0.5 * LABEL_SMOOTH if LABEL_SMOOTH > 0 else ye_b
                loss = (bce(p, yb) * cw * we_b).mean()
                opt.zero_grad(); loss.backward(); opt.step()
                n_steps += 1
                with torch.no_grad():
                    for pe, pm in zip(ema.state_dict().values(), model.state_dict().values()):
                        if pe.dtype.is_floating_point:
                            dec = min(EMA_DECAY, (1 + n_steps) / (10 + n_steps)); pe.mul_(dec).add_(pm, alpha=1 - dec)
                        else:
                            pe.copy_(pm)
                tot += loss.item() * len(bi)
            score, m = val_metrics(ema)
            hist.append({"round": rnd, "epoch": ep, "train_loss": tot / n, "score": score, **m})
            print(f"r{rnd} ep{ep} n={n} loss={tot/n:.4f} score={score:.3f} " +
                  " ".join(f"{k}={v:.4f}" for k, v in m.items()), flush=True)
            if score > best["score"]:
                best = {"score": score, "state": {k: v.clone() for k, v in ema.state_dict().items()},
                        "round": rnd, "epoch": ep, "metrics": m}
    train_time = time.time() - t0
    model = make_net()
    model.load_state_dict(best["state"])
    model.eval()
    print(f"best inner-val score {best['score']:.3f} at round {best['round']} epoch {best['epoch']}: {best['metrics']}")

    if SAVE_STATE:
        torch.save({'state': best['state'], 'dim': LAYER_DIM, 'dropout': DROPOUT, 'metrics': best['metrics'],
                    'round': best['round'], 'epoch': best['epoch'], 'seed': SEED, 'arch': ARCH}, SAVE_STATE)
    onnx_path = candidate_output(OUT_ONNX)
    torch.onnx.export(model, torch.rand(1, 16, 96), str(onnx_path), input_names=["input"],
                      output_names=["output"], dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
                      opset_version=13, dynamo=False)
    import onnxruntime as ort
    s = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    a = s.run(None, {"input": Xv[:512].float().numpy()})[0].reshape(-1)
    b = model(Xv[:512].float()).detach().squeeze(-1).numpy()
    parity = float(np.abs(a - b).max())
    print("exported", onnx_path, onnx_path.stat().st_size, "bytes; onnx/torch max diff", parity)

    meta = {"train_time_s": train_time, "n_local_train_windows": int(loc_tr.sum()),
            "n_positive_windows_train": int(((y == 1) & loc_tr).sum()),
            "n_generic_train_windows": int(n_gen), "acav_chunks": len(chunks),
            "acav_train_windows": int(n_acav_tr), "acav_hours": float(n_acav_tr * 1.28 / 3600),
            "best_round": best["round"], "best_epoch": best["epoch"], "best_val_score": best["score"],
            "best_val_metrics": best["metrics"], "hard_negatives_per_round": n_hard_hist,
            "layer_dim": LAYER_DIM, "dropout": DROPOUT, "pos_class_weight": POS_CLASS_W,
            "seed": SEED, "onnx_torch_parity_max_abs_diff": parity}
    Path(OUT_ONNX).with_suffix(".meta.json").write_text(json.dumps({**meta, "history": hist}, indent=1), encoding="utf-8")
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=LAYER_DIM)
    ap.add_argument("--dropout", type=float, default=DROPOUT)
    ap.add_argument("--wd", type=float, default=WD)
    ap.add_argument("--ema", type=float, default=EMA_DECAY)
    ap.add_argument("--pcw", type=float, default=POS_CLASS_W)
    ap.add_argument("--ls", type=float, default=LABEL_SMOOTH)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--save-state", default=None)
    ap.add_argument("--arch", default="ln", choices=["ln", "stock"])
    ap.add_argument("--init-stock", action="store_true")
    ap.add_argument("--realign", action="store_true")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--out", required=True, help="new candidate ONNX outside production models")
    a = ap.parse_args()
    LAYER_DIM, DROPOUT, WD, EMA_DECAY, POS_CLASS_W, LABEL_SMOOTH, SEED, OUT_ONNX =         a.dim, a.dropout, a.wd, a.ema, a.pcw, a.ls, a.seed, a.out
    SAVE_STATE = a.save_state
    ARCH, INIT_STOCK, REALIGN = a.arch, a.init_stock, a.realign
    if a.lr:
        ROUND_LR = [a.lr * f for f in (1, .5, .3, .2, .1, .1, .1)]
    main()
