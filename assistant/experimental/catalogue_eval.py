"""Offline check of catalogue resolution against the real local index.

No network, no audio, no models: reads assistant/logs/.catalogue.json and
prints, for each heard string, what ``catalogue.best`` would resolve.

    .venv/Scripts/python -m assistant.experimental.catalogue_eval
"""
from __future__ import annotations

import sys
import time

from assistant import catalogue

# (heard, expected "title|primary artist" or None). Heard strings are real ASR output.
POSITIVE = [
    ("madira by see the mode", "Madira|Seedhe Maut"),
    ("madira by see them all", "Madira|Seedhe Maut"),
    ("madira with sidemotri", "Madira|Seedhe Maut"),
    ("madeira by seedemoth", "Madira|Seedhe Maut"),
    ("nakshawa is idemot", "Naksha|Seedhe Maut"),
    ("nakhshab i seedhe maut", "Naksha|Seedhe Maut"),
    ("nachshavai by seedhe maut", "Naksha|Seedhe Maut"),
    ("red from seedemoth", "RED|Seedhe Maut"),
    ("red by sidaimoth", "RED|Seedhe Maut"),
    ("thin cheese by the seat", "3 Cheez|The Siege"),
    ("teen cheese by cg", "3 Cheez|The Siege"),
    ("3 cheez by the siege", "3 Cheez|The Siege"),
    ("champagne poetry by drake", "Champagne Poetry|Drake"),
]
# Either resolved by the index (legitimately in the user's world) or None; both fine.
OPEN = [
    ("creep by radiohead", None),
    ("bohemian rhapsody", None),
    ("for khair", None),
    # Absent from my guesses but genuinely in the user's library: resolving is right.
    ("kesariya by arijit singh", None),
    ("good morning", None),
]
NEGATIVE = [
    ("shield up shield up shield up", None),
    ("boss", None),
    ("james song to something played by kanye west", None),
    ("change strong to me and your ghost", None),
    ("thank you", None),
    ("oh no no no", None),
    # Plausible but absent songs (invented, none by these artists in the index).
    ("blinding lights by the weeknd", None),
    ("shape of you by ed sheeran", None),
    ("levitating by dua lipa", None),
    ("sunflower by post malone", None),
    ("counting stars by onerepublic", None),
    ("tum hi ho by pritam", None),
    ("channa by seedhe maut", None),
    ("naksha by drake", None),
    ("teen taal by seedhe maut", None),
    ("ocean eyes by billie eilish", None),
    ("mad world by gary jules", None),
    ("bad guy by billie eilish", None),
    ("play something nice", None),
]


def main() -> int:
    index = catalogue.load()
    if index is None:
        print("no index at", catalogue.INDEX_PATH)
        return 2
    print("index: %d tracks, complete=%s\n" % (len(index), index.complete))
    wrong = 0
    print("%-46s %-34s %-6s %-6s %s" % ("heard", "resolved", "score", "ms", "verdict"))
    for group, cases in (("positive", POSITIVE), ("open", OPEN), ("negative", NEGATIVE)):
        print("--", group)
        for heard, expected in cases:
            started = time.perf_counter()
            hit = catalogue.best(heard, index)
            ms = (time.perf_counter() - started) * 1000
            got = None if hit is None else "%s|%s" % (hit[0]["name"], hit[0]["artists"][0])
            if group == "open":
                verdict = "ok"
            else:
                verdict = "ok" if got == expected else "WRONG"
            wrong += verdict == "WRONG"
            label = "None" if got is None else got.replace("|", " / ")
            print("%-46s %-34s %-6s %-6.0f %s" % (
                heard[:46], label[:34], "-" if hit is None else "%.2f" % hit[1], ms, verdict))
    print("\nwrong:", wrong)
    return 1 if wrong else 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
