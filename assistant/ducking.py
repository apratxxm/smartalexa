"""Temporary music ducking with crash recovery.

Ducking lowers only Spotify's own Windows mixer session (the per-app slider in
the Volume Mixer). It used to lower the Windows master volume, which also
lowered Memoir's voice and then snapped everything back up at once. Windows
master volume is changed only by an explicit voice volume command, and the
Spotify Connect volume is never changed here.
"""
from __future__ import annotations

import atexit
import json
import logging
import math
from numbers import Real
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

log = logging.getLogger("memoir.ducking")


class SystemVolumeVerificationError(RuntimeError):
    """The requested Windows output level could not be verified."""

# Written before ducking, deleted after restoring. If it exists at startup, a
# previous run died while ducked and the level needs repairing.
_STATE_FILE = Path(__file__).resolve().parent / "logs" / ".system_duck_state.json"
_LEASE_LOCK = threading.RLock()
_LEASE_ORIGINAL = None
_LEASE_ACTIVE = False
_DUCK_RESTORE_STACK = []
# Ducks in progress: key -> (monotonic start, original %, ducked %). Read by the
# health watchdog (duck_watchdog); start/restore are logged at DEBUG, anomalies
# (restore failed, a duck held too long) at WARNING.
_DUCK_SINCE: dict = {}
_DUCK_WARNED: set = set()
DUCK_STUCK_SECONDS = 15.0

# Short fades: a step change in the music is what read as "abrupt".
DUCK_FADE_MS = 120
RESTORE_FADE_MS = 450
_FADE_STEPS = 9


def duck_watchdog(threshold_s: float = DUCK_STUCK_SECONDS) -> float:
    """Warn once per duck that has been held longer than ``threshold_s``.

    A command-length duck lasts seconds; a long one means a restore never ran.
    Returns the age of the oldest duck (0.0 when none). Safe from any thread.
    """
    now = time.monotonic()
    oldest = 0.0
    for key, (started, original, target) in list(_DUCK_SINCE.items()):
        age = now - started
        oldest = max(oldest, age)
        if age > threshold_s and (key, started) not in _DUCK_WARNED:
            _DUCK_WARNED.add((key, started))
            log.warning("Duck still active after %.0f s (%s%% -> %s%%); music may be left quiet",
                        age, original, target)
    return oldest


def _endpoint():
    """Return the Windows default output endpoint volume interface."""
    try:
        from pycaw.pycaw import AudioUtilities
    except ImportError:  # non-Windows dev machine
        raise RuntimeError("pycaw is required for Windows output volume")
    device = AudioUtilities.GetSpeakers()
    if device is None or device.EndpointVolume is None:
        raise RuntimeError("No Windows output endpoint is available")
    return device.EndpointVolume


def _spotify_sessions() -> list:
    """Every Windows audio session owned by Spotify.exe."""
    try:
        from pycaw.pycaw import AudioUtilities
        import psutil
    except ImportError:
        return []
    found = []
    for session in AudioUtilities.GetAllSessions():
        try:
            process = session.Process
            if process and process.name().lower() == "spotify.exe":
                found.append(session)
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError, AttributeError):
            continue
    return found


def spotify_is_playing_local() -> bool:
    """Read-only hint for whether ducking Spotify's session will help."""
    return any(getattr(s, "State", 0) == 1 for s in _spotify_sessions())


def get_system_volume() -> float:
    """Current default output level as a scalar in 0.0-1.0."""
    return float(_endpoint().GetMasterVolumeLevelScalar())


def set_system_volume(level: float) -> None:
    """Set and verify the Windows output level; never call Spotify's volume API."""
    if isinstance(level, bool) or not isinstance(level, Real):
        raise ValueError("System volume must be a finite real number.")
    try:
        numeric_level = float(level)
    except OverflowError:
        numeric_level = 0.0 if level < 0 else 1.0
    if not math.isfinite(numeric_level):
        raise ValueError("System volume must be a finite real number.")
    target = max(0.0, min(1.0, numeric_level))

    endpoint = _endpoint()
    endpoint.SetMasterVolumeLevelScalar(target, None)
    actual = endpoint.GetMasterVolumeLevelScalar()
    if isinstance(actual, bool) or not isinstance(actual, Real):
        raise SystemVolumeVerificationError("Windows output volume change could not be verified.")
    try:
        actual = float(actual)
    except (TypeError, ValueError, OverflowError):
        raise SystemVolumeVerificationError("Windows output volume change could not be verified.") from None
    delta = abs(actual - target)
    outside_tolerance = delta > 0.01 and not math.isclose(delta, 0.01, rel_tol=1e-12, abs_tol=1e-12)
    if not math.isfinite(actual) or not 0.0 <= actual <= 1.0 or outside_tolerance:
        raise SystemVolumeVerificationError("Windows output volume change could not be verified.")


def get_music_volume() -> Optional[float]:
    """Spotify's mixer-session level, or None when Spotify has no session."""
    sessions = _spotify_sessions()
    if not sessions:
        return None
    return max(float(s.SimpleAudioVolume.GetMasterVolume()) for s in sessions)


def set_music_volume(level: float, fade_ms: int = 0) -> None:
    """Set Spotify's mixer-session level, fading over fade_ms."""
    level = max(0.0, min(1.0, level))
    sessions = _spotify_sessions()
    if not sessions:
        return
    start = max(float(s.SimpleAudioVolume.GetMasterVolume()) for s in sessions)
    steps = _FADE_STEPS if fade_ms > 0 and abs(level - start) > 0.01 else 1
    for i in range(1, steps + 1):
        value = start + (level - start) * i / steps
        for s in sessions:
            try:
                s.SimpleAudioVolume.SetMasterVolume(value, None)
            except OSError as exc:
                log.debug("Spotify session volume write failed: %s", exc)
        if i < steps:
            time.sleep(fade_ms / 1000 / steps)


def _session_identifier(session) -> Optional[str]:
    """Return the Windows audio-session identity needed for safe recovery."""
    for name in ("InstanceIdentifier", "Identifier"):
        value = getattr(session, name, None)
        if value:
            return str(value)
    return None


def _session_levels() -> dict[str, tuple[object, float]]:
    """Current Spotify sessions keyed by stable Windows session identity."""
    levels = {}
    for session in _spotify_sessions():
        identifier = _session_identifier(session)
        if identifier is None:
            continue
        try:
            levels[identifier] = (session, float(session.SimpleAudioVolume.GetMasterVolume()))
        except (OSError, AttributeError, RuntimeError) as exc:
            log.debug("Could not read Spotify session %s: %s", identifier, exc)
    return levels


def _snapshot_sessions() -> dict[str, float]:
    return {identifier: level for identifier, (_, level) in _session_levels().items()}


def _write_session_state(original: dict[str, float], ducked: dict[str, float]) -> None:
    _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    sessions = [{"id": identifier, "original": original[identifier], "ducked": ducked[identifier]}
                for identifier in original]
    _STATE_FILE.write_text(json.dumps({"version": 2, "target": "spotify_sessions",
                                      "sessions": sessions}))


def _fade_sessions(targets: dict[str, float], fade_ms: int = 0,
                   expected: Optional[dict[str, float]] = None,
                   written: Optional[dict[str, float]] = None) -> set[str]:
    """Set only owned sessions; expected guards against overwriting user changes."""
    current = _session_levels()
    eligible = {key for key, value in targets.items()
                if key in current and (expected is None or
                   abs(current[key][1] - expected.get(key, current[key][1])) <= 0.015)}
    starts = {key: current[key][1] for key in eligible}
    steps = _FADE_STEPS if fade_ms > 0 and any(abs(targets[k] - starts[k]) > 0.01 for k in eligible) else 1
    last = dict(starts)
    for step in range(1, steps + 1):
        sessions_now = _session_levels()
        for key in tuple(eligible):
            item = sessions_now.get(key)
            if item is None or abs(item[1] - last[key]) > 0.015:
                eligible.remove(key)
                continue
            value = starts[key] + (targets[key] - starts[key]) * step / steps
            try:
                item[0].SimpleAudioVolume.SetMasterVolume(value, None)
                last[key] = value
                if written is not None:
                    written[key] = value
            except (OSError, AttributeError, RuntimeError) as exc:
                log.warning("Could not set Spotify session %s: %s", key, exc)
                raise
        if step < steps and eligible:
            time.sleep(fade_ms / 1000 / steps)
    return eligible


def _restore_sessions(original: dict[str, float], ducked: dict[str, float],
                      fade_ms: int = 0) -> set[str]:
    """Restore matching sessions only while they remain at our ducked level."""
    current = _session_levels()
    eligible = {key: value for key, value in original.items()
                if key in current and abs(current[key][1] - ducked[key]) <= 0.015}
    written: dict[str, float] = {}
    try:
        _fade_sessions(eligible, fade_ms, expected=ducked, written=written)
    except (OSError, AttributeError, RuntimeError):
        # Recovery must recognize our last successful intermediate writes,
        # otherwise a partial restore looks like a user moved the slider.
        ducked.update(written)
        _write_session_state(original, ducked)
        raise
    # A vanished instance cannot be restored and must not hold the recovery
    # file forever. Keep only live sessions that still appear ducked after a
    # write (for example, an audio driver that accepted but ignored it).
    after = _session_levels()
    return {key for key in eligible if key in after
            and abs(after[key][1] - ducked[key]) <= 0.015
            and abs(after[key][1] - original[key]) > 0.015}


def set_command_volume(level: float) -> None:
    """Apply a spoken volume command to the Windows output level.

    Ducking works on Spotify's session instead, so a duck ending can no longer
    undo a volume the user asked for mid-command.
    """
    set_system_volume(level)


def _write_state(original: float) -> None:
    _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _STATE_FILE.write_text(json.dumps({"original": original, "target": "spotify_session"}))


def _clear_state() -> None:
    _STATE_FILE.unlink(missing_ok=True)


def repair_orphaned_duck() -> None:
    """Restore a level left behind by a hard kill. Call once at startup."""
    if not _STATE_FILE.exists():
        return
    try:
        state = json.loads(_STATE_FILE.read_text())
        if state.get("version") != 2 or state.get("target") != "spotify_sessions":
            log.warning("Discarding legacy duck state without bound Spotify session identities; "
                        "Windows master volume was not changed.")
            _clear_state()
            return
        entries = state.get("sessions", [])
        original = {entry["id"]: float(entry["original"]) for entry in entries}
        ducked = {entry["id"]: float(entry["ducked"]) for entry in entries}
        pending_ids = _restore_sessions(original, ducked, RESTORE_FADE_MS)
        pending = {key: value for key, value in original.items() if key in pending_ids}
        if pending:
            _write_session_state(pending, {key: ducked[key] for key in pending})
            return
    except (json.JSONDecodeError, KeyError, TypeError, ValueError, OSError, RuntimeError) as exc:
        log.error("Could not repair orphaned duck state: %s", exc)
    else:
        _clear_state()


@contextmanager
def ducked(to_percent: int) -> Iterator[tuple]:
    """Duck Spotify's session TO to_percent of its level for the block.

    Yields (original_percent, ducked_percent) for telemetry; both are None
    when Spotify has no local audio session to duck.
    """
    # An early wake lease owns restoration across this nested operation.
    # Never overwrite its recovery file or restore to the already-ducked level.
    with _LEASE_LOCK:
        if _LEASE_ACTIVE:
            current = get_music_volume()
            yield (None if current is None else int(round(current * 100)), to_percent)
            return
    original = _snapshot_sessions()
    if not original:
        yield (None, None)
        return

    targets = {key: min(level, max(0.0, min(1.0, to_percent / 100.0)))
               for key, level in original.items()}
    original_pct = int(round(max(original.values()) * 100))
    ducked_pct = int(round(max(targets.values()) * 100))

    restore = {"original": original, "ducked": targets}
    written: dict[str, float] = {}
    with _LEASE_LOCK:
        _DUCK_RESTORE_STACK.append(restore)
    # Belt and braces: if the process is terminated in a way that skips the
    # finally block but still runs atexit hooks, we still restore.
    def restore_at_exit():
        _restore_sessions(restore["original"], restore["ducked"])
    atexit.register(restore_at_exit)
    try:
        _write_session_state(original, targets)
        _fade_sessions(targets, DUCK_FADE_MS, expected=original, written=written)
        _DUCK_SINCE[id(restore)] = (time.monotonic(), original_pct, ducked_pct)
        log.debug("Duck %d%% -> %d%%", original_pct, ducked_pct)
        yield (original_pct, ducked_pct)
    finally:
        # If restoration raises after some session writes, retain the entire
        # record. A later repair checks each current level before writing, so
        # already-restored or user-adjusted sessions will be left alone.
        expected = {key: value for key, value in written.items()
                    if abs(value - original[key]) > 0.015}
        pending = set(expected)
        try:
            pending = _restore_sessions(
                {key: original[key] for key in expected}, expected,
                RESTORE_FADE_MS)
        except Exception as exc:
            log.warning("Duck restore failed (%s); levels kept for repair", exc)
            raise
        finally:
            began = _DUCK_SINCE.pop(id(restore), None)
            if pending:
                log.warning("Duck restore incomplete: %d Spotify session(s) still ducked", len(pending))
            elif began is not None:
                log.debug("Duck restored to %d%% after %.1f s", began[1], time.monotonic() - began[0])
            atexit.unregister(restore_at_exit)
            with _LEASE_LOCK:
                _DUCK_RESTORE_STACK.remove(restore)
                if pending:
                    _write_session_state(
                        {key: value for key, value in restore["original"].items() if key in pending},
                        {key: value for key, value in expected.items() if key in pending})
                elif _DUCK_RESTORE_STACK:
                    top = _DUCK_RESTORE_STACK[-1]
                    _write_session_state(top["original"], top["ducked"])
                else:
                    _clear_state()


class DuckLease:
    """Serialized early duck held across an utterance capture."""
    def __init__(self, to_percent: int, timeout_seconds: float = 6.0):
        self.to_percent = to_percent
        self.timeout_seconds = timeout_seconds
        self._owned = False
        self._started = 0.0

    def acquire(self) -> bool:
        global _LEASE_ORIGINAL, _LEASE_ACTIVE
        with _LEASE_LOCK:
            if self._owned:
                return True
            if _LEASE_ACTIVE:
                return False
            if _STATE_FILE.exists():
                repair_orphaned_duck()
                if _STATE_FILE.exists():
                    return False
            original = _snapshot_sessions()
            _LEASE_ACTIVE = True
            _LEASE_ORIGINAL = original or None  # None means ownership with no session.
            state_written = False
            written: dict[str, float] = {}
            try:
                if original:
                    target = {key: min(level, max(0.0, min(1.0, self.to_percent / 100.0)))
                              for key, level in original.items()}
                    _write_session_state(original, target)
                    state_written = True
                    _fade_sessions(target, DUCK_FADE_MS, expected=original, written=written)
            except Exception:
                _LEASE_ORIGINAL = None
                _LEASE_ACTIVE = False
                # Record the levels actually reached by a partial fade, then
                # try to return those sessions to their original levels now.
                changed = {key: value for key, value in written.items()
                           if abs(value - original[key]) > 0.015}
                if state_written and changed:
                    _write_session_state({key: original[key] for key in changed}, changed)
                    try:
                        pending = _restore_sessions(
                            {key: original[key] for key in changed}, changed)
                    except Exception:
                        # Keep the recovery record for the next startup.
                        pass
                    else:
                        if not pending:
                            _clear_state()
                else:
                    _clear_state()
                raise
            self._owned = True
            self._started = time.monotonic()
            levels = max(original.values(), default=None)
            _DUCK_SINCE["lease"] = (self._started,
                                    None if levels is None else int(round(levels * 100)),
                                    self.to_percent)
            log.debug("Duck lease acquired: %s%% -> %s%%", _DUCK_SINCE["lease"][1], self.to_percent)
            return True

    def expired(self) -> bool:
        return self._owned and time.monotonic() - self._started >= self.timeout_seconds

    def release(self) -> None:
        global _LEASE_ORIGINAL, _LEASE_ACTIVE
        with _LEASE_LOCK:
            if not self._owned:
                return
            try:
                if _LEASE_ORIGINAL is not None:
                    ducked = {key: min(level, max(0.0, min(1.0, self.to_percent / 100.0)))
                              for key, level in _LEASE_ORIGINAL.items()}
                    pending = _restore_sessions(_LEASE_ORIGINAL, ducked, RESTORE_FADE_MS)
                    if pending:
                        log.warning("Duck lease restore incomplete: %d Spotify session(s) still ducked",
                                    len(pending))
                        _write_session_state({key: _LEASE_ORIGINAL[key] for key in pending},
                                             {key: ducked[key] for key in pending})
                    else:
                        _clear_state()
            except Exception as exc:
                log.warning("Duck lease restore failed (%s); levels kept for repair", exc)
                raise
            finally:
                began = _DUCK_SINCE.pop("lease", None)
                if began is not None:
                    log.debug("Duck lease released after %.1f s", time.monotonic() - began[0])
                # Keep the recovery file if restoration fails, but never leave
                # an in-memory owner stranded after its handler has exited.
                _LEASE_ORIGINAL = None
                _LEASE_ACTIVE = False
                self._owned = False

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *_):
        self.release()


def main() -> None:
    """Smoke test: duck Spotify for three seconds, then restore."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    repair_orphaned_duck()
    print("Windows output volume: {:.0f}%".format(get_system_volume() * 100))
    print("Spotify session volume:", get_music_volume())
    with ducked(20) as (was, now):
        print("Ducked {} -> {}. Restoring in 3s...".format(was, now))
        time.sleep(3)
    print("Restored Spotify to", get_music_volume())
    print("Windows output volume: {:.0f}%".format(get_system_volume() * 100))


if __name__ == "__main__":
    main()
