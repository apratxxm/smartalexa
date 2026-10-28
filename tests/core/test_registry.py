from types import SimpleNamespace

import pytest

from assistant.config import Config
from assistant.core.adapters.base import Adapter
from assistant.core.registry import Registry, RegistryError


def handler(verb, kind="text", **attributes):
    return SimpleNamespace(verbs=(verb,), object_kind=kind, **attributes)


def config(names, phases=None):
    return Config({"core": {"enabled_phases": phases or ["1"], "enabled_capabilities": names}})


def test_registry_enables_only_allowed_phase_and_capabilities():
    adapters = [("music", Adapter("music", "1", {"pause": handler("pause", "none")})),
                ("jam", Adapter("jam", "4", {"start_jam": handler("jam", "none")}))]
    registry = Registry(config(["pause", "start_jam"]), adapters=adapters)
    assert set(registry.handlers) == {"pause"}
    assert registry.disabled["start_jam"] == "phase disabled"


def test_registry_rejects_unknown_and_duplicate_capabilities():
    adapter = Adapter("music", "1", {"pause": handler("pause", "none")})
    with pytest.raises(RegistryError, match="unknown capabilities"):
        Registry(config(["missing"]), adapters=[("music", adapter)])
    with pytest.raises(RegistryError, match="duplicate capability"):
        Registry(config(["pause"]), adapters=[("first", adapter), ("second", adapter)])


def test_registry_rejects_ambiguous_verb_and_exemption():
    first = Adapter("music", "1", {"pause": handler("control", "none")})
    second = Adapter("music", "1", {"resume": handler("control", "none")})
    with pytest.raises(RegistryError, match="conflicting verb"):
        Registry(config(["pause", "resume"]), adapters=[("first", first), ("second", second)])
    exempt = Adapter("music", "1", {"pause": handler("pause", "none", c1_exempt=True)})
    with pytest.raises(RegistryError, match="unauthorized C1 exemption"):
        Registry(config(["pause"]), adapters=[("music", exempt)])


def test_registry_rejects_import_errors(monkeypatch):
    def fail_import(name):
        raise ImportError("synthetic adapter failure")

    monkeypatch.setattr("assistant.core.registry.importlib.import_module", fail_import)
    with pytest.raises(RegistryError, match="adapter import failed.*ImportError"):
        Registry(config([]))
