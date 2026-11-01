"""Exercise the labelled corpus against registered handlers and core authority."""
from pathlib import Path

import pytest
import yaml

from assistant.config import Config
from assistant.conversation_contracts import Step
from assistant.core.authority import admit_step, find_clause
from assistant.core.registry import PHASE_ONE, Registry


CORPUS = Path(__file__).parent / "corpus" / "authority.yaml"
# Only actual conflicts with the current contract belong here, never missing handlers.
KNOWN_CORPUS_DISAGREEMENTS = {}


def _cases():
    if not CORPUS.exists():
        return [pytest.param(None, marks=pytest.mark.skip(reason="authority corpus absent"))]
    rows = yaml.safe_load(CORPUS.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("authority corpus must be a list")
    return [pytest.param(row, id=f"{i}: {row['text']}") for i, row in enumerate(rows)]


@pytest.fixture(scope="module")
def registry():
    discovered = Registry._discover()
    names = set(PHASE_ONE)
    for _, adapter in discovered:
        names.update(adapter.handlers)
    cfg = Config({"core": {"enabled_phases": ["1", "1b", "3", "4"],
                           "enabled_capabilities": sorted(names)}})
    return Registry(cfg, adapters=discovered)


@pytest.mark.parametrize("case", _cases())
def test_authority_corpus(case, registry, monkeypatch):
    text = case["text"]
    expected = case["expect"]
    assert expected in {"admit", "refuse"}
    if text in KNOWN_CORPUS_DISAGREEMENTS:
        expected = "refuse" if expected == "admit" else "admit"
    capability = case.get("capability")
    if expected == "admit" and capability not in registry.handlers:
        pytest.skip(f"no registered handler for {capability}; corpus case not exercised")

    clause = find_clause(text, registry)
    admitted = []
    if clause is not None:
        for name, handler in registry.verbs[clause.verb]:
            args = handler.license(clause)
            if args is None:
                continue
            if clause.suspicious and not getattr(handler, "c1_exempt", False):
                exact = bool(case.get("c1") and name == capability)
                monkeypatch.setattr(handler, "exact_match", lambda *_, result=exact: result,
                                    raising=False)
            if admit_step(Step("corpus", name, args), text, registry, lambda: 5):
                admitted.append((name, args))
    if expected == "refuse":
        assert not admitted
        return
    matching = [args for name, args in admitted if name == capability]
    assert matching, f"expected {capability}, got {admitted!r}"
    for key, value in case.get("args", {}).items():
        assert matching[0].get(key) == value
    if case.get("c1"):
        assert clause.suspicious
