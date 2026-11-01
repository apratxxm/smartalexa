from types import SimpleNamespace

import pytest

from assistant.core.capture import CaptureResult, EndReason, admissible
from assistant.core.fallback import admit
from assistant.core.executor import Executor, Receipt, coordinator_result


@pytest.mark.parametrize("reason,continuous,truncated,turn", [
    (EndReason.ENDPOINT, True, True, "t"),
    (EndReason.HARD_LIMIT, True, False, "t"),
    (EndReason.OVERFLOW, True, False, "t"),
    (EndReason.ENDPOINT, False, False, "t"),
    (EndReason.ENDPOINT, None, False, "t"),
    (EndReason.ENDPOINT, True, False, "old"),
])
def test_incomplete_or_stale_capture_rejected(reason, continuous, truncated, turn):
    result = CaptureResult(turn, [], 1000, reason, continuous, truncated)
    assert not admissible(result, "t")


@pytest.mark.parametrize("text", [
    '"play Halo"', "play Halo, actually don't play anything", "he said play Halo",
    "don't play Halo", "play Halo and open Opera", "“play Halo”", "‘play Halo’",
    "`play Halo`", "'play Halo'", "play 'Halo'", "play “Halo”", "play ‘Halo’",
])
def test_fallback_refuses_ambiguous_or_compound_speech(text):
    assert admit(text) is None


@pytest.mark.parametrize("text,cap", [("play Halo", "play_song"),
                                        ("set volume to 40", "set_volume")])
def test_fallback_admits_explicit_phase_one_requests(text, cap):
    assert admit(text).steps[0].capability == cap


def test_receipt_mapping_is_single_boundary():
    assert coordinator_result("x", Receipt("observed")).status == "completed"
    assert coordinator_result("x", Receipt("requested")).status == "accepted"
    assert coordinator_result("x", Receipt("failed")).status == "failed"
    assert coordinator_result("x", Receipt("unknown")).status == "unknown_delivery"


def test_volume_uses_master_adapter_only():
    calls = []
    volume = SimpleNamespace(set_system_volume=lambda value, **kwargs: calls.append(value))
    spotify = SimpleNamespace(set_volume=lambda *_: pytest.fail("Spotify volume called"))
    receipt = Executor(spotify=spotify, volume=volume).execute(
        SimpleNamespace(capability="set_volume", arguments={"level": 40}))
    assert receipt.state == "observed" and calls == [0.4]


@pytest.mark.parametrize("cap", ["codex_request", "open_app", "search_browser", "start_jam"])
def test_phase_one_executor_denies_non_allowlisted_capabilities(cap):
    assert Executor().execute(SimpleNamespace(capability=cap, arguments={})).state == "failed"
