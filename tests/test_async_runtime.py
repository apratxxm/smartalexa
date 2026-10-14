from collections import deque
from types import SimpleNamespace
from unittest.mock import Mock
import time

import numpy as np
import pytest

from assistant import main
from assistant.async_listener import ListenerEvent
from assistant.async_runtime import AsyncCaptureRuntime, validate_async_config
from assistant.config import Config
from assistant.wake import TIER_BOTH


class FakeLane:
    def __init__(self, _recognizers):
        self.recognizers = dict(_recognizers)
        self.started = False
        self.closed = False
        self.suspended = False
        self.inflight = False
        self.poll_calls = 0
        self.start_calls = 0
        self.close_calls = 0
        self.close_results = deque()

    def start(self):
        self.start_calls += 1
        self.started = True
        error = getattr(self, "start_error", None)
        if error is not None:
            raise error
        return True

    def poll(self):
        self.poll_calls += 1
        return ()

    def suspend(self, _timeout):
        self.suspended = True
        return not self.inflight

    def resume(self, _epoch):
        if self.inflight or self.closed:
            return False
        self.suspended = False
        return True

    def invalidate(self, _epoch):
        pass

    def close(self, _timeout):
        self.close_calls += 1
        self.closed = True
        if self.close_results:
            return self.close_results.popleft()
        return getattr(self, "close_result", True)


class FakeFrontend:
    def __init__(self, _cfg, vad, lane, *, verify_text):
        self.vad = vad
        self.lane = lane
        self.verify_text = verify_text
        self.feed_calls = []
        self.events = deque()
        self.poll_calls = 0
        self.epoch = 0

    def feed_async(self, frame, processed_frame=None, **kwargs):
        self.vad(processed_frame if processed_frame is not None else frame)
        self.feed_calls.append((frame.copy(), processed_frame, kwargs))

    def poll_events(self):
        self.poll_calls += 1
        self.lane.poll()
        values = tuple(self.events)
        self.events.clear()
        return values

    def suspend(self, timeout):
        acknowledged = self.lane.suspend(timeout)
        self.epoch += 1
        return acknowledged

    def resume(self):
        if not self.lane.resume(self.epoch + 1):
            return False
        self.epoch += 1
        return True

    def flush(self, _reason):
        self.epoch += 1
        self.events.clear()

    def close(self, timeout):
        return self.lane.close(timeout)


class FakeArchive:
    def __init__(self, _telemetry):
        self.started = False
        self.closed = False
        self.items = []

    def start(self):
        error = getattr(self, "start_error", None)
        if error is not None:
            raise error
        self.started = True

    def submit(self, *args, **kwargs):
        self.items.append((args, kwargs))
        return True

    def close(self, _timeout):
        self.closed = True
        return True


def runtime(vad=None, frontend_factory=FakeFrontend, archive_factory=FakeArchive):
    vad = vad or Mock(return_value=0.2)
    cfg = Config({"listener": {"mode": "stt"}})
    value = AsyncCaptureRuntime(
        cfg, vad, {"listener": lambda audio: ("", 1), "command": lambda audio: ("", 1)},
        lambda transcript, phrase, mid: (phrase, "pause", False), object(),
        lane_factory=FakeLane, frontend_factory=frontend_factory,
        archive_factory=archive_factory)
    return value, vad


@pytest.mark.parametrize("data", [
    {"listener": {"async_enabled": False, "mode": "wakeword"}},
    {"listener": {"mode": "stt"}},
])
def test_async_config_defaults_off_and_legacy_modes_remain_allowed(data):
    assert validate_async_config(Config(data)) is False


@pytest.mark.parametrize("data,match", [
    ({"listener": {"async_enabled": "true", "mode": "stt"}}, "boolean"),
    ({"listener": {"async_enabled": True, "mode": "wakeword"}}, "mode='stt'"),
    ({"listener": {"async_enabled": True, "mode": "stt"},
      "audio": {"paused_input_device": "secondary"}}, "paused_input_device"),
    ({"listener": {"async_enabled": True, "mode": "stt"},
      "ducking": {"duck_trigger": "speech"}}, "speech-triggered"),
    ({"listener": {"async_enabled": True, "mode": "stt"},
      "ducking": {"pause_on_high_confidence": True}}, "pause_on_high_confidence"),
])
def test_async_config_rejects_unsupported_combinations(data, match):
    with pytest.raises(ValueError, match=match):
        validate_async_config(Config(data))


def test_runtime_feeds_once_and_frontend_is_the_only_lane_result_drainer():
    value, vad = runtime()
    value.start()
    frame = np.zeros(512, dtype=np.int16)
    assert value.feed(frame, None, captured_at=1.0, authorization={}) is True
    assert value.last_vad_probability == 0.2
    assert value.poll_events() == ()
    assert value.frontend.poll_calls == 1
    assert value.lane.poll_calls == 1
    assert vad.call_count == 1

    assert value.suspend() is True
    assert value.feed(frame, None, captured_at=2.0, authorization={}) is False
    assert vad.call_count == 1
    assert value.counters["feed_while_suspended"] == 1
    assert value.resume_if_quiescent() is True


def test_unacknowledged_suspend_keeps_admission_closed_until_resume_is_safe():
    value, vad = runtime()
    value.start()
    value.lane.inflight = True
    assert value.suspend(timeout=0) is False
    assert value.feed(np.zeros(512, dtype=np.int16), None,
                      captured_at=1.0, authorization={}) is False
    assert value.resume_if_quiescent() is False
    assert vad.call_count == 0
    value.lane.inflight = False
    assert value.resume_if_quiescent() is True


def test_owned_diagnostic_is_enqueued_with_stage_capture_and_latencies():
    value, _vad = runtime()
    value.start()
    snapshot = object()
    event = ListenerEvent(
        "diagnostic", reason="acoustic_unconfirmed", capture_id="capture-1",
        proposal_id="proposal-1", diagnostic_snapshot=snapshot,
        diagnostic_stage="confirmation", trigger_transcript="maybe alexa",
        second_stt_transcript="alexa pause", trigger_stt_latency_ms=14,
        second_stt_latency_ms=27)

    assert value.queue_diagnostic(event) is True
    args, kwargs = value.archive.items[0]
    assert args[0] is snapshot
    assert args[1] == "acoustic_unconfirmed"
    assert kwargs["stage"] == "confirmation"
    assert kwargs["capture_id"] == "capture-1"
    assert kwargs["proposal_id"] == "proposal-1"
    assert kwargs["transcript"] == "maybe alexa"
    assert kwargs["second_stt_transcript"] == "alexa pause"
    assert kwargs["latency_ms"] == 14
    assert kwargs["second_latency_ms"] == 27


def test_shutdown_reports_lane_and_archive_status_separately():
    value, _vad = runtime()
    value.start()
    assert value.close(timeout=0) == {"lane_closed": True, "archive_closed": True}


def test_shutdown_reports_an_unjoined_lane_truthfully():
    value, _vad = runtime()
    value.start()
    value.lane.close_result = False
    assert value.close(timeout=0) == {"lane_closed": False, "archive_closed": True}
    value.lane.close_result = True
    assert value.close(timeout=0) == {"lane_closed": True, "archive_closed": True}
    assert value.lane.start_calls == 1
    assert value.lane.close_calls == 2


def test_failed_lane_start_closes_archive_and_does_not_claim_admission():
    value, _vad = runtime()
    value.lane.start = Mock(return_value=False)
    with pytest.raises(RuntimeError, match="did not start"):
        value.start()
    assert value.closed is True
    assert value.accepting is False
    assert value.lane.closed is True
    assert value.archive.closed is True
    assert value.close(timeout=0) == {"lane_closed": True, "archive_closed": True}


def test_archive_start_failure_closes_lane_and_archive_truthfully():
    value, _vad = runtime()
    value.archive.start_error = RuntimeError("archive unavailable")
    with pytest.raises(RuntimeError, match="archive unavailable"):
        value.start()
    assert value.closed is True
    assert value.accepting is False
    assert value.lane.closed is True
    assert value.archive.closed is True


def test_lane_start_exception_closes_both_resources_and_preserves_error():
    value, _vad = runtime()
    value.lane.start_error = RuntimeError("lane partial start")
    with pytest.raises(RuntimeError, match="lane partial start"):
        value.start()
    assert value.closed is True
    assert value.accepting is False
    assert value.lane.started is True
    assert value.lane.closed is True
    assert value.archive.closed is True
    assert value.close_status == {"lane_closed": True, "archive_closed": True}


class FakeWake:
    personal_hit = False
    personal_score = 0.0
    last_score = 0.9
    last_model = "alexa"

    def process(self, _frame):
        return TIER_BOTH


def test_async_frame_passes_acoustic_proposal_with_current_frame_once():
    frame = np.zeros(512, dtype=np.int16)
    clean = np.ones(512, dtype=np.int16)
    app = main.Assistant.__new__(main.Assistant)
    app.async_capture = SimpleNamespace(
        accepting=True, frontend=SimpleNamespace(_active=None, _boundary=None),
        last_vad_probability=0.2)
    calls = []

    def feed(raw, processed, **kwargs):
        calls.append((raw, processed, kwargs))
        app.async_capture.last_vad_probability = 0.2

    app.async_capture.feed = feed
    app.async_capture.poll_events = Mock(return_value=())
    app.wake = FakeWake()
    app.listener = SimpleNamespace(phrases=["alexa"])
    app.cfg = Config({"wake": {"acoustic_activation_threshold": 0.75,
                                "threshold_high": 0.60}})
    app._last_strong_wake = None
    app._early_lease = None
    app._check_echo_reference = Mock()
    app._trace_wake = Mock()
    app._authorize_text_trigger = Mock(return_value=False)
    app._record_capture_diagnostics = Mock()
    app._dispatch_async_events = Mock(return_value=0.0)
    app._try_async_global_notice = Mock()
    app._confirm_acoustic_wake = Mock(side_effect=AssertionError("sync confirmation called"))
    mic = SimpleNamespace(vad_frame=Mock(return_value=clean))

    main.Assistant._run_async_frame(
        app, mic, frame, last_trigger=0.0, last_confirmation=float("-inf"),
        last_early_log=0.0, cooldown=2.0)

    assert len(calls) == 1
    assert calls[0][0] is frame
    assert calls[0][1] is clean
    assert calls[0][2]["acoustic_proposal"].stock_model == "alexa"
    app._confirm_acoustic_wake.assert_not_called()
    assert app.async_capture.poll_events.call_count == 1


def _dispatch_app():
    app = main.Assistant.__new__(main.Assistant)
    app.async_capture = SimpleNamespace(
        counters={"completed_captures_discarded": 0},
        queue_diagnostic=Mock(return_value=True))
    app._async_cued_segments = set()
    app._duck_on_wake = Mock()
    app._record_wake_rejection = Mock()
    app.handle_utterance = Mock()
    app.telemetry = Mock()

    def run_callback(_mic, callback, **_kwargs):
        callback()
        return True

    app._run_async_owned_callback = Mock(side_effect=run_callback)
    return app


def test_nonterminal_diagnostic_keeps_prior_wake_cue_for_utterance():
    app = _dispatch_app()
    utterance = SimpleNamespace(segment_id="seg-1", command="pause")
    events = [
        ListenerEvent("wake", segment_id="seg-1"),
        ListenerEvent("diagnostic", reason="acoustic_superseded_by_text",
                      segment_id="seg-1", capture_id="cap-1",
                      diagnostic_snapshot=object()),
        ListenerEvent("utterance", segment_id="seg-1", capture_id="cap-1",
                      utterance=utterance),
    ]

    mic = object()
    main.Assistant._dispatch_async_events(
        app, mic, events, last_trigger=time.monotonic() - 5.0, cooldown=2.0)

    app._duck_on_wake.assert_called_once_with()
    app.handle_utterance.assert_called_once_with(
        mic, utterance, wake_cued=True)


def test_batch_dispatch_runs_one_capture_and_records_the_rest_by_identity():
    app = _dispatch_app()
    first = SimpleNamespace(segment_id="seg-1", command="pause")
    second = SimpleNamespace(segment_id="seg-2", command="next")
    events = [
        ListenerEvent("wake", segment_id="seg-1"),
        ListenerEvent("utterance", segment_id="seg-1", capture_id="cap-1",
                      utterance=first),
        ListenerEvent("wake", segment_id="seg-2"),
        ListenerEvent("utterance", segment_id="seg-2", capture_id="cap-2",
                      utterance=second),
    ]
    mic = object()

    main.Assistant._dispatch_async_events(
        app, mic, events, last_trigger=time.monotonic() - 5.0, cooldown=2.0)

    app.handle_utterance.assert_called_once_with(mic, first, wake_cued=True)
    assert app.async_capture.counters["completed_captures_discarded"] == 1
    discarded = app.telemetry.write.call_args.args[0]
    assert discarded.capture_id == "cap-2"
    assert discarded.segment_id == "seg-2"
    assert discarded.rejection_reason == "async_batch_limit"
    assert discarded.diagnostic_only is True
    assert "seg-2" not in app._async_cued_segments


def test_cooldown_records_every_completed_capture_without_dispatch():
    app = _dispatch_app()
    one = SimpleNamespace(segment_id="seg-1", command="pause")
    two = SimpleNamespace(segment_id="seg-2", command="next")
    app._async_cued_segments.update(("seg-1", "seg-2"))
    events = [
        ListenerEvent("utterance", segment_id="seg-1", capture_id="cap-1",
                      utterance=one),
        ListenerEvent("utterance", segment_id="seg-2", capture_id="cap-2",
                      utterance=two),
    ]

    main.Assistant._dispatch_async_events(
        app, object(), events, last_trigger=time.monotonic(), cooldown=2.0)

    app._run_async_owned_callback.assert_not_called()
    app.handle_utterance.assert_not_called()
    assert app.async_capture.counters["completed_captures_discarded"] == 2
    assert [call.args[0].capture_id for call in app.telemetry.write.call_args_list] == [
        "cap-1", "cap-2"]
    assert not app._async_cued_segments


def test_unacknowledged_handoff_never_invokes_partial_command_handler():
    app = main.Assistant.__new__(main.Assistant)
    runtime_value, _vad = runtime()
    runtime_value.start()
    runtime_value.lane.inflight = True
    app.async_capture = runtime_value
    app.listener = Mock()
    app.wake = Mock()
    app._record_wake_rejection = Mock()
    app.handle_utterance = Mock()
    partial_utt = SimpleNamespace(command="pause", transcript="alexa pause")
    handler = Mock(side_effect=lambda: app.handle_utterance(mic, partial_utt))
    mic = Mock()

    accepted = main.Assistant._run_async_owned_callback(
        app, mic, handler, reason="utterance handler")

    assert accepted is False
    handler.assert_not_called()
    app.handle_utterance.assert_not_called()
    assert runtime_value.accepting is False
    assert runtime_value.counters["handoff_attempts"] == 1
    app._record_wake_rejection.assert_called_once_with("async_handler_busy", mic=mic)


def test_acknowledged_handoff_uses_shared_recognizer_only_after_quiescence():
    app = main.Assistant.__new__(main.Assistant)
    runtime_value, _vad = runtime()
    runtime_value.start()
    app.async_capture = runtime_value
    app.listener = Mock()
    app.wake = None
    app._last_strong_wake = None
    app._confirmed_acoustic_history = None
    app._reset_async_capture_hints = Mock()
    mic = Mock()
    called = []

    def legacy_decode():
        assert runtime_value.lane.suspended is True
        assert runtime_value.lane.inflight is False
        assert runtime_value.lane.recognizers["listener"](np.zeros(512, dtype=np.int16)) == ("", 1)
        called.append(True)

    assert main.Assistant._run_async_owned_callback(
        app, mic, legacy_decode, reason="test handler") is True
    assert called == [True]
    assert runtime_value.accepting is True


def test_acknowledged_handoff_resumes_fresh_epoch_after_handler_exception():
    app = main.Assistant.__new__(main.Assistant)
    runtime_value, _vad = runtime()
    runtime_value.start()
    app.async_capture = runtime_value
    app.listener = Mock()
    app.wake = None
    app._last_strong_wake = None
    app._confirmed_acoustic_history = None
    app._reset_async_capture_hints = Mock()
    mic = Mock()

    with pytest.raises(RuntimeError, match="handler failed"):
        main.Assistant._run_async_owned_callback(
            app, mic, lambda: (_ for _ in ()).throw(RuntimeError("handler failed")),
            reason="test handler")

    assert runtime_value.accepting is True
    assert runtime_value.frontend.epoch >= 2
    app._reset_async_capture_hints.assert_called_once_with(mic)


def test_capture_gap_invalidates_async_epoch_and_discards_queued_stale_event():
    app = main.Assistant.__new__(main.Assistant)
    runtime_value, _vad = runtime()
    runtime_value.start()
    runtime_value.frontend.events.append(ListenerEvent("rejected", reason="old"))
    app.async_capture = runtime_value
    app.mode = "stt"
    app.listener = Mock()
    app.wake = Mock()
    app._last_strong_wake = (1.0, 0.9, "alexa")
    app._confirmed_acoustic_history = ("alexa", np.zeros(512, np.int16), None)
    app._early_lease = Mock()
    app._record_wake_rejection = Mock()
    mic = Mock()
    old_epoch = runtime_value.frontend.epoch

    main.Assistant._recover_capture_queue_gap(app, mic)

    assert runtime_value.frontend.epoch == old_epoch + 1
    assert runtime_value.poll_events() == ()
    mic.flush.assert_called_once()
    app.wake.reset.assert_called_once()
    app._record_wake_rejection.assert_called_once_with("capture_queue_gap", mic=mic)
