"""Build a sherpa-onnx hotwords file from phrases, tokenising directly against
the model's tokens.txt (SentencePiece pieces with the U+2581 word marker), so no
bpe.model / sentencepiece install is needed.  Each phrase becomes one line of
space-separated pieces (min-piece-count Viterbi segmentation)."""
from __future__ import annotations

import re
from pathlib import Path

WB = "▁"


def load_pieces(tokens_path: Path) -> set:
    pieces = set()
    for line in Path(tokens_path).read_text(encoding="utf-8").splitlines():
        parts = line.rsplit(" ", 1)
        if len(parts) == 2:
            pieces.add(parts[0])
    return pieces


def _segment_word(word: str, pieces: set):
    s = WB + word
    n = len(s)
    best = [None] * (n + 1)
    best[0] = []
    for i in range(n):
        if best[i] is None:
            continue
        for j in range(i + 1, min(n, i + 16) + 1):
            p = s[i:j]
            if p in pieces and (best[j] is None or len(best[i]) + 1 < len(best[j])):
                best[j] = best[i] + [p]
    return best[n]


def tokenize_phrase(phrase: str, pieces: set):
    for k, v in (("21", "twenty one"),):
        phrase = phrase.replace(k, v)
    clean = re.sub(r"[^A-Za-z0-9\s]", " ", phrase)
    words = clean.split()
    if not words:
        return None
    for variant in (str.lower, str.upper, lambda x: x):
        out = []
        ok = True
        for w in words:
            seg = _segment_word(variant(w), pieces)
            if seg is None:
                ok = False
                break
            out.extend(seg)
        if ok:
            return out
    return None


def build_hotwords_file(phrases: list, out_path: Path, tokens_path: Path) -> tuple:
    pieces = load_pieces(tokens_path)
    lines, skipped = [], []
    for ph in dict.fromkeys(p.strip() for p in phrases if p.strip()):
        t = tokenize_phrase(ph, pieces)
        if t:
            lines.append(" ".join(t))
        else:
            skipped.append(ph)
    Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return Path(out_path), skipped
