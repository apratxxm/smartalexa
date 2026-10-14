"""Launch registered external targets outside Memoir's managed process tree."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess


_MANAGED_PROCESS_TREE = "MEMOIR_MANAGED_PROCESS_TREE"
_CREATE_BREAKAWAY_FROM_JOB = 0x01000000


def external_creation_flags() -> int:
    """Return launch flags for external apps, preserving legacy unmanaged behavior."""
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    if os.name == "nt" and os.environ.get(_MANAGED_PROCESS_TREE) == "1":
        flags |= _CREATE_BREAKAWAY_FROM_JOB
    return flags


def _trusted_explorer() -> tuple[Path, Path]:
    root_value = os.environ.get("WINDIR") or os.environ.get("SystemRoot")
    if os.name != "nt" or not root_value:
        raise OSError("Windows URI launching is unavailable")
    root = Path(root_value)
    if not root.is_absolute():
        raise OSError("Windows URI launcher could not be verified")
    try:
        canonical_root = root.resolve(strict=True)
        candidate = canonical_root / "explorer.exe"
        binary = candidate.resolve(strict=True)
    except (OSError, ValueError) as exc:
        raise OSError("Windows URI launcher could not be verified") from exc
    if (not candidate.is_absolute() or binary != candidate
            or canonical_root not in binary.parents or not binary.is_file()):
        raise OSError("Windows URI launcher could not be verified")
    return binary, canonical_root


def open_external_target(target: str) -> None:
    """Open one already-authorized app URI/target without shell composition.

    Unmanaged processes retain the exact ``os.startfile`` behavior. In a
    managed Windows process, Explorer brokers the literal target from an
    argv entry and is explicitly allowed to outlive the assistant job.
    """
    if (not isinstance(target, str) or not target or len(target) > 2048
            or any(ord(char) < 32 or 0x7F <= ord(char) <= 0x9F for char in target)):
        raise ValueError("Invalid external app target")
    if os.name != "nt" or os.environ.get(_MANAGED_PROCESS_TREE) != "1":
        os.startfile(target)  # type: ignore[attr-defined]
        return

    explorer, windows_root = _trusted_explorer()
    subprocess.Popen(
        [str(explorer), target], stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        cwd=str(windows_root), creationflags=external_creation_flags(),
    )
