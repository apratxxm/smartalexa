"""Fail-closed discovery and configuration of capability adapters."""
from __future__ import annotations

import importlib
from pathlib import Path

from .adapters.base import Adapter

PHASE_ONE = frozenset({"play_song", "pause", "resume", "next_track",
                       "set_volume", "change_volume", "answer"})
PHASES = frozenset({"1", "1b", "3", "4"})


class RegistryError(ValueError):
    pass


def _names(value, label):
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise RegistryError(f"{label} must be a list of nonempty strings")
    if len(value) != len(set(value)):
        raise RegistryError(f"{label} contains duplicates")
    return set(value)


class Registry:
    def __init__(self, config, *, adapters=None):
        phases = _names(config.get("core.enabled_phases", ["1"]), "enabled_phases")
        capabilities = _names(config.get("core.enabled_capabilities", sorted(PHASE_ONE)),
                              "enabled_capabilities")
        if phases - PHASES:
            raise RegistryError("unknown phases: " + ", ".join(sorted(phases - PHASES)))
        self.enabled_phases = frozenset(phases)
        discovered = self._discover() if adapters is None else adapters
        from .authority import PhaseOneHandler
        supplied = {name for _, adapter in discovered for name in adapter.handlers}
        builtins = {name: PhaseOneHandler(name) for name in PHASE_ONE - {"answer"} - supplied}
        discovered = [("builtin.phase_one", Adapter("music", "1", builtins))] + list(discovered)
        self.handlers = {}
        self.disabled = {}
        self.verbs = {}
        known = {"answer"}
        entries = []
        for module_name, adapter in discovered:
            if not isinstance(adapter, Adapter):
                raise RegistryError(f"{module_name}: missing or invalid ADAPTER")
            for name, handler in adapter.handlers.items():
                if name in known:
                    raise RegistryError(f"duplicate capability: {name}")
                known.add(name)
                if getattr(handler, "c1_exempt", False) and not (
                        module_name == "assistant.core.adapters.computer" and
                        adapter.area == "computer" and name == "search_browser"):
                    raise RegistryError(f"unauthorized C1 exemption: {name}")
                entries.append((name, adapter.phase, handler))
        self.safety_verbs = tuple({verb for _, _, handler in entries for verb in handler.verbs})
        unknown = capabilities - known
        if unknown:
            raise RegistryError("unknown capabilities: " + ", ".join(sorted(unknown)))
        for name, phase, handler in entries:
            reasons = []
            if phase not in phases:
                reasons.append("phase disabled")
            if name not in capabilities:
                reasons.append("capability disabled")
            if reasons:
                self.disabled[name] = ", ".join(reasons)
                continue
            self.handlers[name] = handler
            for verb in handler.verbs:
                existing = self.verbs.setdefault(verb, [])
                if any(previous.object_kind == handler.object_kind for _, previous in existing):
                    raise RegistryError(f"conflicting verb: {verb}")
                existing.append((name, handler))
        if "answer" in capabilities and "1" in phases:
            self.answer_enabled = True
        else:
            self.answer_enabled = False
            self.disabled["answer"] = "phase or capability disabled"

    @staticmethod
    def _discover():
        directory = Path(__file__).parent / "adapters"
        adapters = []
        for path in sorted(directory.glob("*.py")):
            if path.name in {"base.py", "__init__.py"}:
                continue
            module_name = "assistant.core.adapters." + path.stem
            try:
                module = importlib.import_module(module_name)
                adapter = module.ADAPTER
            except Exception as exc:
                raise RegistryError(f"adapter import failed: {module_name}: {type(exc).__name__}: {exc}") from exc
            adapters.append((module_name, adapter))
        return adapters
