"""Windows single-instance guard, acquired before audio or model startup."""
from __future__ import annotations

import ctypes
import hashlib
import json
import math
import os
import re
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from ctypes import wintypes


class AlreadyRunning(RuntimeError):
    pass


_COLLECTION_SCHEMA = "memoir.collection-pause.v1"
_CAPTURE_ID = re.compile(r"^[0-9a-f]{32}$")
_MAX_COLLECTION_PAUSES = 32
_MAX_LEASE_BYTES = 4096


class CollectionPauseOverflow(ValueError):
    """The private collection namespace exceeded its bounded lease count."""


def _collection_dir(lease_dir: Path, *, create: bool = False) -> Path:
    """Return a private, non-linked lease directory without replacing contents."""
    path = Path(lease_dir)
    if create:
        path.mkdir(parents=True, exist_ok=True)
    elif not path.exists():
        return path.absolute()
    try:
        if path.is_symlink() or not path.is_dir():
            raise ValueError("collection pause directory must be a real directory")
        resolved = path.resolve(strict=True)
        if resolved != path.absolute():
            raise ValueError("collection pause directory escaped its path")
    except OSError as exc:
        raise ValueError("invalid collection pause directory") from exc
    return resolved


def _read_collection_lease(directory: Path, path: Path, capture_id: str,
                           clock: float, *, require_active: bool = True) -> dict | None:
    try:
        if path.is_symlink() or path.parent.resolve(strict=True) != directory:
            return None
        with path.open("rb") as stream:
            raw = stream.read(_MAX_LEASE_BYTES + 1)
        if len(raw) > _MAX_LEASE_BYTES:
            return None
        item = json.loads(raw.decode("utf-8"))
        if not isinstance(item, dict) or item.get("schema") != _COLLECTION_SCHEMA:
            return None
        if item.get("capture_id") != capture_id:
            return None
        issued, expires = item.get("issued_at"), item.get("expires_at")
        if any(isinstance(v, bool) or not isinstance(v, (int, float))
               or not math.isfinite(v) for v in (issued, expires)):
            return None
        if not 0 < expires - issued <= 30:
            return None
        if issued > clock or (require_active and clock >= expires):
            return None
        return item
    except (OSError, UnicodeError, ValueError, TypeError, OverflowError, RecursionError):
        return None


def active_collection_pauses(lease_dir: Path, *, now: float | None = None) -> list[dict]:
    """Return valid unexpired private collection leases; malformed entries are ignored."""
    try:
        directory = _collection_dir(lease_dir)
        clock = time.time() if now is None else now
        if isinstance(clock, bool) or not isinstance(clock, (int, float)) or not math.isfinite(clock):
            return []
        result = []
        for path in directory.iterdir():
            if not re.fullmatch(r"[0-9a-f]{32}\.json", path.name):
                continue
            item = _read_collection_lease(directory, path, path.stem, clock)
            if item is not None:
                if len(result) == _MAX_COLLECTION_PAUSES:
                    raise CollectionPauseOverflow("too many active collection pauses")
                result.append(item)
        return result
    except CollectionPauseOverflow:
        raise
    except OSError:
        return []


def _ack_matches(directory: Path, capture_id: str, lease: dict) -> bool:
    path = directory / f"{capture_id}.ack"
    try:
        if path.is_symlink() or path.parent.resolve(strict=True) != directory:
            return False
        with path.open("rb") as stream:
            raw = stream.read(_MAX_LEASE_BYTES + 1)
        if len(raw) > _MAX_LEASE_BYTES:
            return False
        ack = json.loads(raw.decode("utf-8"))
        return (isinstance(ack, dict)
                and ack.get("schema") == _COLLECTION_SCHEMA
                and ack.get("capture_id") == capture_id
                and ack.get("issued_at") == lease["issued_at"]
                and ack.get("expires_at") == lease["expires_at"]
                and not isinstance(ack.get("issued_at"), bool)
                and not isinstance(ack.get("expires_at"), bool))
    except (OSError, UnicodeError, ValueError, TypeError, OverflowError, RecursionError):
        return False


def create_collection_pause(lease_dir: Path, capture_id: str, *, now: float | None = None,
                            seconds: float = 30) -> None:
    if not isinstance(capture_id, str) or not _CAPTURE_ID.fullmatch(capture_id):
        raise ValueError("capture_id must be 32 lowercase hexadecimal characters")
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or not 0 < seconds <= 30:
        raise ValueError("seconds must be finite and between 0 and 30")
    issued = time.time() if now is None else now
    if isinstance(issued, bool) or not isinstance(issued, (int, float)) or not math.isfinite(issued):
        raise ValueError("now must be finite")
    directory = _collection_dir(lease_dir, create=True)
    target = directory / f"{capture_id}.json"
    payload = json.dumps({"schema": _COLLECTION_SCHEMA, "capture_id": capture_id,
                          "issued_at": issued, "expires_at": issued + seconds},
                         separators=(",", ":")).encode("utf-8")
    if len(payload) > _MAX_LEASE_BYTES:
        raise ValueError("collection pause lease is too large")
    fd, temporary = tempfile.mkstemp(prefix=".lease-", dir=directory)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, target)  # atomic publication, fails on collisions
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def collection_pause_acknowledged(lease_dir: Path, capture_id: str, *, now: float | None = None) -> bool:
    if not isinstance(capture_id, str) or not _CAPTURE_ID.fullmatch(capture_id):
        return False
    try:
        directory = _collection_dir(lease_dir)
        clock = time.time() if now is None else now
        lease = _read_collection_lease(directory, directory / f"{capture_id}.json", capture_id, clock)
        ack = directory / f"{capture_id}.ack"
        return lease is not None and _ack_matches(directory, capture_id, lease)
    except (OSError, ValueError):
        return False


def acknowledge_collection_pauses(lease_dir: Path, *, now: float | None = None) -> None:
    for lease in active_collection_pauses(lease_dir, now=now):
        capture_id = lease["capture_id"]
        try:
            directory = _collection_dir(lease_dir)
            marker = directory / f"{capture_id}.ack"
            if marker.exists() or marker.is_symlink():
                continue
            payload = json.dumps({"schema": _COLLECTION_SCHEMA,
                                  "capture_id": capture_id,
                                  "issued_at": lease["issued_at"],
                                  "expires_at": lease["expires_at"]},
                                 separators=(",", ":")).encode("utf-8")
            fd, temporary = tempfile.mkstemp(prefix=".ack-", dir=directory)
            try:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.link(temporary, marker)
            finally:
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass
        except (OSError, ValueError):
            continue


def release_collection_pause(lease_dir: Path, capture_id: str) -> bool:
    if not isinstance(capture_id, str) or not _CAPTURE_ID.fullmatch(capture_id):
        return False
    try:
        directory = _collection_dir(lease_dir)
        lease_path = directory / f"{capture_id}.json"
        lease = _read_collection_lease(directory, lease_path, capture_id, time.time(),
                                       require_active=False)
        if lease is None:
            return False
        ack_path = directory / f"{capture_id}.ack"
        if ack_path.is_symlink():
            return False
        if ack_path.exists() and not _ack_matches(directory, capture_id, lease):
            return False
        lease_path.unlink()
        try:
            ack_path.unlink()
        except FileNotFoundError:
            pass
        return True
    except (OSError, ValueError):
        return False


def pause_requested(path: Path, *, now: float | None = None,
                    lease_dir: Path | None = None) -> bool:
    """Honor manual flags indefinitely; only validated probe leases may expire.

    Reading never deletes a flag, so expiry cannot remove a later owner's pause.
    Unknown formats and read failures remain paused rather than bypassing intent.
    """
    private_pause = False
    if lease_dir is not None:
        try:
            private_pause = bool(active_collection_pauses(lease_dir, now=now))
        except (OSError, ValueError):
            private_pause = True
    try:
        with path.open("r", encoding="utf-8") as stream:
            raw = stream.read(4097)
    except FileNotFoundError:
        return private_pause
    except (OSError, UnicodeError):
        return True
    if len(raw) > 4096:
        return True
    try:
        lease = json.loads(raw)
        if not isinstance(lease, dict) or lease.get("schema") != "memoir.probe-pause.v1":
            return True
        if not isinstance(lease.get("owner"), str) or not lease["owner"].strip():
            return True
        issued, expires = lease["issued_at"], lease["expires_at"]
        if any(isinstance(v, bool) or not isinstance(v, (int, float))
               or not math.isfinite(v) for v in (issued, expires)):
            return True
        clock = time.time() if now is None else now
        if not 0 < expires - issued <= 900 or issued > clock:
            return True
        manual = clock < expires
        return manual or private_pause
    except (ValueError, KeyError, TypeError, OverflowError, RecursionError):
        return True


@contextmanager
def single_instance(project: Path):
    # Scope to this checkout and Windows login session. The kernel closes the
    # handle after a crash; no stale lock file can prevent a later restart.
    identity = hashlib.sha256(str(project.resolve()).casefold().encode()).hexdigest()[:20]
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
    kernel.CreateMutexW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    handle = kernel.CreateMutexW(None, False, "Local\\MemoirAssistant-" + identity)
    error = ctypes.get_last_error()
    if not handle:
        raise ctypes.WinError(error)
    try:
        if error == 183:  # ERROR_ALREADY_EXISTS
            raise AlreadyRunning("Memoir is already running for this project")
        yield
    finally:
        kernel.CloseHandle(handle)
