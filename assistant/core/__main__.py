"""Opt-in phase-one runtime, offline selfcheck, and planning-only evaluation."""
from __future__ import annotations

import argparse
import base64
import importlib
import json
import logging
import os
import re
import subprocess
import time

from ..config import ROOT, load
from ..runtime import single_instance
from .capture import CaptureRuntime
from .engine import CoreRuntime
from .executor import Executor
from .provider import BrainProvider
from .registry import Registry, RegistryError

_TOASTS = []


def _setup(cfg, provider="default", *, clock=time.monotonic):
    key = "core.brain" + (".alt" if provider == "alt" else "")
    brain = cfg.get(key, {})
    return BrainProvider(brain.get("base_url"), brain.get("model"), brain.get("api_key_env"),
                         float(brain.get("turn_deadline_seconds",
                               cfg.get("core.brain.turn_deadline_seconds", 18))), clock=clock)


def _toast(text):
    """Best-effort asynchronous WinRT notification with inert user text."""
    try:
        encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
        script = r"""
$ErrorActionPreference='Stop'
[Windows.UI.Notifications.ToastNotificationManager,Windows.UI.Notifications,ContentType=WindowsRuntime] > $null
[Windows.Data.Xml.Dom.XmlDocument,Windows.Data.Xml.Dom.XmlDocument,ContentType=WindowsRuntime] > $null
$t=[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('PAYLOAD'))
$x=New-Object Windows.Data.Xml.Dom.XmlDocument
$x.LoadXml('<toast duration="short"><visual><binding template="ToastGeneric"><text>' + [Security.SecurityElement]::Escape($t) + '</text></binding></visual></toast>')
$n=[Windows.UI.Notifications.ToastNotification]::new($x)
[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('Memoir').Show($n)
""".replace("PAYLOAD", encoded)
        command = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
        _TOASTS[:] = [child for child in _TOASTS if child.poll() is None]
        # Do not accumulate notification helpers if the desktop is unavailable.
        if len(_TOASTS) >= 2:
            return
        _TOASTS.append(subprocess.Popen(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-EncodedCommand", command],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)))
    except Exception:
        logging.debug("Core toast unavailable", exc_info=True)


def _log(row):
    path = ROOT / "assistant" / "logs" / "core_turns.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def _validate_capture(cfg):
    if cfg.get("audio.sample_rate", 16000) != 16000 or cfg.get("audio.frame_length", 512) != 512:
        raise ValueError("core capture requires 16000 Hz / 512-sample frames")
    model = cfg.get("stt.model", "")
    if not isinstance(model, str) or not model.startswith("faster-whisper/"):
        raise ValueError("core requires the configured faster-whisper command model")
    limit = cfg.get("core.capture.max_seconds", 8)
    if isinstance(limit, bool) or not isinstance(limit, (int, float)) or not 0 < limit <= 30:
        raise ValueError("core.capture.max_seconds must be in (0, 30]")
    if cfg.get("runtime.runtime", "legacy") not in {"legacy", "core"}:
        raise ValueError("runtime.runtime must be legacy or core")


def selfcheck(provider="default"):
    cfg = load()
    _configured_registry(cfg)
    _validate_capture(cfg)
    # Imports only: no constructors, devices, sockets, synthesis, or log files.
    for name in ("audio", "echo", "wake", "stt", "spotify", "feedback", "conversation_brain", "core.evaluation"):
        importlib.import_module("assistant." + name)
    for name in ("default", "alt"):
        brain = _setup(cfg, name)
        env_names = [brain.api_key_env, *re.findall(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", brain.base_url_template)]
        states = " ".join("%s=%s" % (key, "set" if os.getenv(key) else "unset") for key in env_names)
        print("provider=%s selected=%s model=%s %s" % (name, name == provider, brain.model, states))
        if brain.missing_env():
            print("WARNING: missing environment: " + ", ".join(brain.missing_env()))
    print("microphone=%s output=%s command_stt=%s" %
          (cfg.get("audio.input_device", "default"), cfg.get("feedback.output_device", "default"), cfg.get("stt.model")))
    try:
        _registry_report(cfg)
    except RegistryError as exc:
        print("selfcheck: registry refused to start: %s" % exc)
        raise SystemExit(1)
    print("selfcheck: wiring valid; no devices or network opened")


def _registry_report(cfg):
    """List every capability with its enabled state or the registry's reason. Raises RegistryError."""
    from ..conversation_contracts import CAPABILITIES
    from .registry import PHASES, PHASE_ONE, Registry
    discovered = Registry._discover()
    registry = Registry(cfg, adapters=discovered)
    phase_of = {name: adapter.phase for _, adapter in discovered for name in adapter.handlers}
    print("phases enabled=%s known=%s" % (
        ",".join(cfg.get("core.enabled_phases", ["1"])), ",".join(sorted(PHASES))))
    for name in sorted(set(CAPABILITIES) | {"answer"}):
        phase = "1" if name in PHASE_ONE or name == "answer" else phase_of.get(name, "none")
        if name in registry.handlers or (name == "answer" and registry.answer_enabled):
            print("capability=%s phase=%s state=enabled" % (name, phase))
        else:
            reason = registry.disabled.get(name, "no adapter")
            print("capability=%s phase=%s state=disabled reason=%s" % (name, phase, reason))


def _configured_registry(cfg):
    try:
        return Registry(cfg)
    except RegistryError as exc:
        raise SystemExit("registry refused to start: %s" % exc) from exc


def run(provider="default"):
    from ..feedback import Feedback
    cfg = load()
    registry = _configured_registry(cfg)
    _validate_capture(cfg)
    brain = _setup(cfg, provider)
    feedback = Feedback(cfg)
    runtime = CoreRuntime(brain, Executor(), feedback, toast=_toast, log=_log,
                          registry=registry, knowledge=cfg.get("core.knowledge", False) is True)
    with CaptureRuntime(cfg) as capture:
        try:
            while True:
                runtime.step(capture)
        except KeyboardInterrupt:
            return


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--selfcheck", action="store_true")
    mode.add_argument("--eval", action="store_true")
    parser.add_argument("--provider", choices=("default", "alt"), default="default")
    args = parser.parse_args()
    if args.selfcheck:
        selfcheck(args.provider)
    elif args.eval:
        from .evaluation import evaluate
        cfg = load()
        _configured_registry(cfg)
        evaluate(_setup(cfg, args.provider), args.provider)
    else:
        logging.basicConfig(level=logging.INFO)
        with single_instance(ROOT):
            run(args.provider)


if __name__ == "__main__":
    main()
