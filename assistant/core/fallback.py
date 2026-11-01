"""Phase-one fallback proposals from the shared whole-utterance grammar."""
from ..conversation_contracts import Decision, Step
from .authority import _registry_default, parsed_licenses, c1_allows
from .registry import PHASE_ONE


def admit(text, registry=None, guard=lambda: float("inf")):
    registry = registry or _registry_default()
    clause, results = parsed_licenses(text, registry, capabilities=PHASE_ONE - {"answer"})
    if clause is None or not results:
        return None
    name, handler, args = results[0]
    if not c1_allows(handler, clause, args, guard):
        return None
    return Decision("plan", steps=(Step("fallback", name, args),), evidence_quotes=(text.strip(),))
