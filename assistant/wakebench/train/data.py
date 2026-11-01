"""Archive negatives (human-reviewed, never positives), generic embedding windows and room bank."""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from assistant.experimental.wake_train import active_learning as review
from assistant.wakebench.train.plan import ROOT, SR

STORE = ROOT / "assistant" / "experimental" / "wake_train" / "human_reviews.json"
GENERIC = Path(r"A:\wake-data\negatives\validation_set_features.npy")
GENERIC_CUT = 0.85     # contiguous time blocks: first 85% train, last 15% holdout
WINDOWS_PER_BAG = 24
FRAMES = 16


def archive_negative_rows() -> tuple[list[dict], int]:
    """Human-reviewed whole-clip negatives with verified WAV identity.

    Split is by capture group (session), never by clip, so one session cannot be on both sides.
    Returns (rows, unavailable_count).
    """
    store, _ = review.load_preparation_store(STORE)
    _, usable, unavailable = review.review_history(store)
    rows = []
    for row in usable:
        if row.get("presence") != "negative":
            continue
        group = str(row.get("group_id") or row.get("capture_session_id") or row["event_id"])
        rows.append({"path": Path(row["path"]), "sha256": row["wav_sha256"], "group_id": group,
                     "split": review.session_split(group), "n_frames": int(row["n_frames"])})
    return rows, len(unavailable)


def read_pcm16(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wav:
        if (wav.getnchannels(), wav.getframerate(), wav.getsampwidth()) != (1, SR, 2):
            raise ValueError(f"archive WAV must be mono PCM16 at 16 kHz: {path.name}")
        payload = wav.readframes(wav.getnframes())
    return np.frombuffer(payload, dtype="<i2").copy()


def room_bank_from(clips: list[np.ndarray], *, seg_sec: float = 0.5, max_floor: float = 0.03) -> list[np.ndarray]:
    """Quietest 0.5 s segment of each clip, kept only if it is actually quiet (local assets only)."""
    frame = 320
    need = int(seg_sec * SR / frame)
    bank = []
    for clip in clips:
        x = clip.astype(np.float32) / 32768.0
        n = len(x) // frame
        if n < need:
            continue
        r = np.sqrt(np.mean(x[: n * frame].reshape(n, frame) ** 2, axis=1))
        means = np.convolve(r, np.ones(need) / need, mode="valid")
        best = int(np.argmin(means))
        if means[best] <= max_floor:
            bank.append(x[best * frame: (best + need) * frame].copy())
    return bank


class GenericWindows:
    """Reads contiguous 80 ms openWakeWord embedding rows via memmap (bounded RAM)."""

    def __init__(self, path: Path = GENERIC):
        self.path = Path(path)
        self.rows = np.load(self.path, mmap_mode="r")
        if self.rows.ndim != 2 or self.rows.shape[1] != 96:
            raise ValueError("generic embeddings must be (M, 96)")
        self.cut = int(self.rows.shape[0] * GENERIC_CUT)
        self.span = WINDOWS_PER_BAG + FRAMES - 1

    def bounds(self, split: str) -> tuple[int, int]:
        return (0, self.cut) if split == "train" else (self.cut, self.rows.shape[0])

    def bags(self, split: str, n: int, rng: np.random.Generator) -> list[np.ndarray]:
        lo, hi = self.bounds(split)
        if hi - lo < self.span:
            raise ValueError(f"generic {split} region too short")
        out = []
        for _ in range(n):
            s = int(rng.integers(lo, hi - self.span + 1))
            block = np.asarray(self.rows[s: s + self.span], dtype=np.float32)
            out.append(np.stack([block[i: i + FRAMES] for i in range(WINDOWS_PER_BAG)]))
        return out


def split_is_disjoint(*groups: set) -> bool:
    seen: set = set()
    for g in groups:
        if seen & g:
            return False
        seen |= g
    return True
