"""Machine-health heartbeat: the readings we lacked before MemoryError / CUDA OOM.

One INFO line (``Health [reason]: key=value ...``) at startup, every 10 minutes,
and just before a fatal error is re-raised. It runs on a daemon thread, so a slow
``nvidia-smi`` or Ollama probe can never stall the audio loop; every reading is
best effort and shows ``n/a`` when unavailable.

The same thread is the watchdog for two slow anomalies that deserve a line of
their own: a duck held for more than 15 s, and a Spotify ban expiring.
"""
from __future__ import annotations

import ctypes
import json
import logging
import subprocess
import threading
import time
import urllib.request
from typing import Callable, Optional

import psutil

from . import ducking

log = logging.getLogger("memoir.health")

HEARTBEAT_SECONDS = 600.0
WATCH_TICK_SECONDS = 5.0
GPU_QUERY = ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"]
OLLAMA_PS = "http://127.0.0.1:11434/api/ps"
OLLAMA_PS_TIMEOUT_SECONDS = 0.25
OLLAMA_PS_MAX_BYTES = 64 * 1024
OLLAMA_MODEL_STATES = frozenset({
    "unloaded", "loaded_gpu", "loaded_cpu", "loaded_partial_gpu", "unavailable",
})
CAPTURE_INPUT_STATES = frozenset({"unknown", "digital_silence", "signal_present"})
_MB = 1024 * 1024


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def ollama_model_status(model: str) -> str:
    """Return residency for one exact local Ollama model without loading it."""
    if not isinstance(model, str) or not model.strip():
        return "unavailable"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
    request = urllib.request.Request(OLLAMA_PS, headers={"Accept": "application/json"})
    try:
        with opener.open(request, timeout=OLLAMA_PS_TIMEOUT_SECONDS) as response:
            raw = response.read(OLLAMA_PS_MAX_BYTES + 1)
        if len(raw) > OLLAMA_PS_MAX_BYTES:
            return "unavailable"
        payload = json.loads(raw)
        models = payload.get("models") if isinstance(payload, dict) else None
        if not isinstance(models, list):
            return "unavailable"
        for loaded in models:
            if not isinstance(loaded, dict):
                return "unavailable"
            names = [loaded.get("name"), loaded.get("model")]
            if not any(isinstance(name, str) for name in names):
                return "unavailable"
            if any(name is not None and not isinstance(name, str) for name in names):
                return "unavailable"
            if model not in names:
                continue
            size, size_vram = loaded.get("size"), loaded.get("size_vram")
            if (type(size) is not int or size <= 0 or
                    type(size_vram) is not int or size_vram < 0):
                return "unavailable"
            if size_vram == 0:
                return "loaded_cpu"
            if size_vram >= size:
                return "loaded_gpu"
            return "loaded_partial_gpu"
        return "unloaded"
    except Exception:
        # Keep URLs, response bodies, and exception text out of the health log.
        return "unavailable"


class _MemoryStatusEx(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def commit_charge() -> Optional[tuple]:
    """(used, limit) bytes of the Windows commit charge, the budget a
    MemoryError exhausts. psutil's swap figures stand in elsewhere."""
    try:
        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(_MemoryStatusEx)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError("GlobalMemoryStatusEx failed")
        limit = int(status.ullTotalPageFile)
        return limit - int(status.ullAvailPageFile), limit
    except (AttributeError, OSError, ValueError):
        try:
            swap = psutil.swap_memory()
            return int(swap.used), int(swap.total)
        except Exception:
            return None


def gpu_memory(timeout: float = 2.0) -> Optional[tuple]:
    """(used MB, total MB) of the first GPU, or None. Never raises, never shows a window."""
    try:
        done = subprocess.run(
            GPU_QUERY, capture_output=True, text=True, timeout=timeout,
            stdin=subprocess.DEVNULL, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        used, total = (int(part.strip()) for part in done.stdout.strip().splitlines()[0].split(",")[:2])
        return used, total
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def _process_stats() -> dict:
    process = psutil.Process()
    info = process.memory_info()
    stats = {"rss_mb": info.rss // _MB,
             # "private" is the commit this process holds; rss only its resident part.
             "private_mb": getattr(info, "private", getattr(info, "vms", 0)) // _MB,
             "threads": process.num_threads(),
             "uptime_s": int(time.time() - process.create_time())}
    handles = getattr(process, "num_handles", None)
    if handles is not None:
        stats["handles"] = handles()
    return stats


def collect(probes: Optional[dict] = None) -> dict:
    """Gather every reading; any single failure just leaves its key out."""
    probes = probes or {}
    stats: dict = {}

    def attempt(name: str, fn: Callable) -> None:
        try:
            value = fn()
        except Exception as exc:
            log.debug("health probe %s failed: %s", name, exc)
            return
        if value is not None:
            stats[name] = value

    def memory() -> None:
        ram = psutil.virtual_memory()
        stats["ram_free_mb"] = int(ram.available) // _MB
        stats["ram_total_mb"] = int(ram.total) // _MB

    try:
        stats.update(_process_stats())
    except Exception as exc:
        log.debug("health probe process failed: %s", exc)
    try:
        memory()
    except Exception as exc:
        log.debug("health probe memory failed: %s", exc)
    commit = commit_charge()
    if commit:
        stats["commit_used_mb"], stats["commit_limit_mb"] = commit[0] // _MB, commit[1] // _MB
    gpu = gpu_memory()
    if gpu:
        stats["gpu_used_mb"], stats["gpu_total_mb"] = gpu
    attempt("ollama", probes.get("ollama") or (lambda: None))
    model_probe = probes.get("ollama_model")
    if model_probe is not None:
        try:
            status = model_probe()
            stats["ollama_model"] = status if status in OLLAMA_MODEL_STATES else "unavailable"
        except Exception:
            # This diagnostic must never expose request details or disrupt health collection.
            stats["ollama_model"] = "unavailable"
    attempt("spotify_bans", probes.get("spotify_bans") or (lambda: None))
    attempt("mic_overflows", probes.get("mic_overflows") or (lambda: None))
    attempt("echo_alive", probes.get("echo_alive") or (lambda: None))
    attempt("capture_input_state", probes.get("capture_input_state") or (lambda: "unknown"))
    capture_input_state = stats.get("capture_input_state")
    if not isinstance(capture_input_state, str) or capture_input_state not in CAPTURE_INPUT_STATES:
        stats["capture_input_state"] = "unknown"
    attempt("command_stt", probes.get("command_stt") or (lambda: None))
    attempt("wake_stt", probes.get("wake_stt") or (lambda: None))
    return stats


def _pair(used, total, unit: str = "MB") -> str:
    if used is None or total is None:
        return "n/a"
    return "{}/{}{}".format(used, total, unit)


def format_line(stats: dict, reason: str = "periodic") -> str:
    """``Health [reason]: ...`` -- key=value tokens, parsed again by assistant.report."""
    commit = _pair(stats.get("commit_used_mb"), stats.get("commit_limit_mb"))
    if commit != "n/a" and stats.get("commit_limit_mb"):
        commit += "({:.0f}%)".format(100.0 * stats["commit_used_mb"] / stats["commit_limit_mb"])
    bans = stats.get("spotify_bans")
    if isinstance(bans, dict) and bans:
        ban = ",".join("{}@{}".format(scope, time.strftime("%Y-%m-%d_%H:%M", time.localtime(until)))
                       for scope, until in sorted(bans.items()))
    else:
        ban = "none" if isinstance(bans, dict) else "n/a"

    def yes_no(key: str, yes: str, no: str) -> str:
        value = stats.get(key)
        return "n/a" if value is None else (yes if value else no)

    uptime = stats.get("uptime_s")
    capture_input_state = stats.get("capture_input_state")
    if not isinstance(capture_input_state, str) or capture_input_state not in CAPTURE_INPUT_STATES:
        capture_input_state = "unknown"
    return ("Health [{}]: rss={}MB private={}MB ram_free={} commit={} gpu={} ollama={} "
            "spotify_ban={} mic_overflows={} echo_ref={} threads={} handles={} up={}").format(
        reason,
        stats.get("rss_mb", "n/a"), stats.get("private_mb", "n/a"),
        _pair(stats.get("ram_free_mb"), stats.get("ram_total_mb")),
        commit,
        _pair(stats.get("gpu_used_mb"), stats.get("gpu_total_mb")),
        yes_no("ollama", "up", "down"),
        ban,
        stats.get("mic_overflows", "n/a"),
        yes_no("echo_alive", "alive", "silent"),
        stats.get("threads", "n/a"), stats.get("handles", "n/a"),
        "n/a" if uptime is None else "{}h{:02d}m".format(uptime // 3600, uptime % 3600 // 60)) + (
            " command_stt={} wake_stt={} ollama_model={} capture_input_state={}".format(
                stats.get("command_stt", "n/a"), stats.get("wake_stt", "n/a"),
                stats.get("ollama_model", "n/a"), capture_input_state))


def log_snapshot(reason: str, probes: Optional[dict] = None) -> None:
    """Log one health line now, on the calling thread. Used right before a
    fatal error is re-raised, so it must survive a machine that is out of memory."""
    try:
        log.info("%s", format_line(collect(probes), reason))
    except BaseException as exc:  # never mask the error that is already in flight
        try:
            log.info("Health [%s]: unavailable (%s)", reason, type(exc).__name__)
        except BaseException:
            pass


class HealthMonitor:
    """Daemon thread: a heartbeat every ``interval_s`` plus a slow-anomaly watchdog."""

    def __init__(self, probes: Optional[dict] = None, interval_s: float = HEARTBEAT_SECONDS,
                 tick_s: float = WATCH_TICK_SECONDS) -> None:
        self.probes = probes or {}
        self.interval_s = interval_s
        self.tick_s = tick_s
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="memoir-health", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _watch(self) -> None:
        try:
            ducking.duck_watchdog()
        except Exception as exc:
            log.debug("duck watchdog failed: %s", exc)
        bans = self.probes.get("spotify_bans")
        if bans is not None:
            try:
                bans()  # also writes the "ban lifted" line when one expires
            except Exception as exc:
                log.debug("ban watch failed: %s", exc)

    def _run(self) -> None:
        log_snapshot("startup", self.probes)
        next_beat = time.monotonic() + self.interval_s
        while not self._stop.wait(self.tick_s):
            self._watch()
            if time.monotonic() >= next_beat:
                next_beat = time.monotonic() + self.interval_s
                log_snapshot("periodic", self.probes)
