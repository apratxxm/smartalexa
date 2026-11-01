import numpy as np
import pytest

from assistant.audio import Endpointer
from assistant.config import Config
from assistant.core.capture import CaptureRuntime, EndReason
from assistant.core.transcription import TimedTranscript, Word


class Clock:
    def __init__(self):
        self.now = 100.0
    def __call__(self):
        self.now += 0.001
        return self.now


class Mic:
    """Deliberately lacks preroll/vad_frame: the core must own those pieces."""
    def __init__(self, **kwargs):
        self.index = 0
        self.last_clean, self.echo = None, object()
        self.queue_overflow_count = self._callback_status_count = self.intentional_discard_count = 0
        self.event = None
    def __enter__(self):
        return self
    def __exit__(self, *_):
        pass
    def flush(self):
        pass
    def read(self, timeout_s=3.0):
        index = self.index
        self.index += 1
        if self.event:
            self.event(self, index)
        raw = np.full(512, index, np.int16)
        self.last_clean = raw + 1000
        return raw


class Wake:
    def __init__(self, cfg):
        self.count, self.last_score = 0, 0.8
    def process(self, frame):
        assert frame[0] >= 1000  # Only the cleaned branch reaches wake.
        self.count += 1
        return self.count == 35
    def reset(self):
        pass
    def close(self):
        pass


class VAD:
    def __init__(self, rate):
        pass
    def reset(self):
        pass
    def __call__(self, frame):
        assert frame[0] >= 1000
        return float(1045 <= frame[0] <= 1047)


class STT:
    def __init__(self, name, rate):
        self.calls = []
    def transcribe(self, pcm):
        assert pcm.dtype == np.int16 and pcm.max() < 1000
        self.calls.append(pcm.copy())
        return TimedTranscript((Word("Alexa", 0.8, 1.0), Word("play", 1.0, 1.1), Word("Halo", 1.1, 1.3)), 1)


def capture_runtime():
    return CaptureRuntime(Config({"audio": {"echo_cancellation": False},
                                  "stt": {"model": "faster-whisper/fake"}}),
                          mic_factory=Mic, wake_factory=Wake, vad_factory=VAD,
                          endpointer_factory=Endpointer, stt_factory=STT, clock=Clock())


def test_preroll_and_all_endpointer_reads_are_one_contiguous_raw_stream():
    with capture_runtime() as runtime:
        result = runtime.next_capture(lambda: None)
        assert result.complete and result.preroll_ms >= 1000
        assert abs(result.audio_started_at + result.preroll_ms / 1000 - result.wake_at) < 1e-9
        # Wake hits frame 34; bounded pre-roll contains frames 2..34.
        # The real Endpointer trims early silence, but core retains ALL reads.
        values = result.pcm.reshape(-1, 512)
        np.testing.assert_array_equal(values[:, 0], np.arange(2, runtime.mic.index))
        assert np.all(values == values[:, :1])
        assert values[32, 0] == 34 and values[33, 0] == 35
        assert runtime.transcribe(result) == ("play Halo", 1)
        assert len(runtime.stt.calls) == 1
        np.testing.assert_array_equal(runtime.stt.calls[0], result.pcm)


@pytest.mark.parametrize("kind,at", [("overflow", 20), ("status", 40), ("missing", 0)])
def test_preroll_and_capture_health_fail_closed(kind, at):
    with capture_runtime() as runtime:
        def event(mic, index):
            if index != at:
                return
            if kind == "overflow":
                mic.queue_overflow_count += 1
            elif kind == "status":
                mic._callback_status_count += 1
            else:
                del mic.intentional_discard_count
        runtime.mic.event = event
        result = runtime.next_capture()
        assert not result.complete
        assert result.continuous is not True
        if kind == "overflow":
            assert result.end_reason == EndReason.OVERFLOW
        with pytest.raises(ValueError, match="incomplete"):
            runtime.transcribe(result)
        assert runtime.stt.calls == []
