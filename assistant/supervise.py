"""Keep the assistant running: restart it whenever it exits on an error.

Task Scheduler's "restart on failure" only covers a task that fails to
launch, not a program that starts and later exits with an error -- which is
how the assistant stayed dead after a reboot. This loop is the task action.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from .config import ROOT, load

log = logging.getLogger("memoir.supervise")
OLLAMA_HEALTH = "http://127.0.0.1:11434/api/version"
OLLAMA_READY_SECONDS = 3.0
DEPENDENCY_CHECK_SECONDS = 30.0
_ollama_process: subprocess.Popen | None = None


def _ollama_ready() -> bool:
    # The assistant sends transcripts only to this loopback server. Ignore
    # machine proxy settings for the health check as the brain does.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(OLLAMA_HEALTH, timeout=0.25) as response:
            return response.status == 200
    except (OSError, TimeoutError):
        return False


def _ollama_executable(configured_path: str | None = None) -> str | None:
    if configured_path is not None:
        if (not isinstance(configured_path, str) or not configured_path.strip()):
            log.warning("Configured Ollama executable path is invalid")
            return None
        configured = Path(configured_path)
        if not configured.is_absolute() or not configured.is_file():
            log.warning("Configured Ollama executable path is invalid")
            return None
        return str(configured)
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        installed = Path(local_app_data) / "Programs" / "Ollama" / "ollama.exe"
        if installed.is_file():
            return str(installed)
    return shutil.which("ollama")


def ensure_local_ollama() -> None:
    """Start the local server if configured local consumers need it.

    A failed dependency must not keep the microphone from starting. The brain
    retains its own command-time fallback and retry behavior.
    """
    global _ollama_process
    config = load()
    backend = config.get("intent.backend")
    if backend not in ("brain", "local_llm") and config.get("conversation.enabled") is not True:
        return
    if _ollama_ready():
        return
    # A slow-starting server must not cause another serve process each check.
    if _ollama_process is not None:
        if _ollama_process.poll() is None:
            return
        _ollama_process = None
    configured_path = config.get("intent.ollama_executable")
    executable = (_ollama_executable(configured_path) if configured_path is not None
                  else _ollama_executable())
    if not executable:
        log.warning("Ollama is unavailable and no installed executable was found")
        return
    environment = os.environ.copy()
    environment["OLLAMA_HOST"] = "127.0.0.1:11434"
    environment["OLLAMA_NUM_PARALLEL"] = "1"
    environment["OLLAMA_MAX_LOADED_MODELS"] = "1"
    environment["OLLAMA_NO_CLOUD"] = "1"
    # Bound compatible llama-server prompt-cache reservations; this limits cache
    # capacity, not memory already allocated by the model/runtime. The smaller
    # cache may retain less prompt history for reuse, without implying a change
    # to model accuracy.
    environment["LLAMA_ARG_CACHE_RAM"] = "256"
    environment["LLAMA_ARG_CTX_CHECKPOINTS"] = "0"
    try:
        _ollama_process = subprocess.Popen(
            [executable, "serve"], cwd=str(ROOT), env=environment,
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except OSError as exc:
        log.warning("Could not start local Ollama: %s", exc)
        return
    deadline = time.monotonic() + OLLAMA_READY_SECONDS
    while time.monotonic() < deadline:
        if _ollama_ready():
            log.info("Local Ollama is ready")
            return
        time.sleep(0.25)
    log.warning("Local Ollama is still starting; assistant will start listening")


def main() -> None:
    logging.basicConfig(
        filename=str(ROOT / "assistant" / "logs" / "supervise.log"),
        level=logging.INFO, format="%(asctime)s %(message)s")
    backoff = 5.0
    while True:
        ensure_local_ollama()
        started = time.monotonic()
        child = subprocess.Popen([sys.executable, "-m", "assistant.main"], cwd=str(ROOT))
        while True:
            try:
                code = child.wait(timeout=DEPENDENCY_CHECK_SECONDS)
                break
            except subprocess.TimeoutExpired:
                ensure_local_ollama()
        ran = time.monotonic() - started
        if code == 0:
            log.info("Assistant exited normally after %.0f s; not restarting", ran)
            return
        # Quick repeated failures (e.g. audio devices not ready at logon)
        # back off up to a minute; a long healthy run resets the delay.
        backoff = 5.0 if ran > 300 else min(backoff * 2, 60.0)
        log.warning("Assistant exited with code %s after %.0f s; restarting in %.0f s",
                    code, ran, backoff)
        time.sleep(backoff)


if __name__ == "__main__":
    main()
