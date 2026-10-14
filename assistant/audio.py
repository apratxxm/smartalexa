"""Microphone input, ring buffer, and neural voice-activity detection.

Two things here are load-bearing:

1. The input device is pinned by name, never "default". Windows can silently
   route input from a Bluetooth speaker's headset profile, which collapses
   capture to 8 kHz *and* degrades music output on the same device.

2. The ring buffer continuously retains the last N seconds. That gives us
   pre-roll for archived triggers (a clip starting at detection contains the
   command but not the wake word, making it useless for wake-word training) and
   makes miss-logging possible at all -- you cannot log a detection that never
   fired.

VAD is Silero (a ~1 MB neural model), not webrtcvad: the latter is a 2011 GMM
that degrades badly with music bleeding into the mic, which is exactly our
environment.
"""
from __future__ import annotations

import argparse
import collections
from contextlib import contextmanager
import logging
import threading
import time
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger("memoir.audio")

SILERO_URL = (
    "https://raw.githubusercontent.com/snakers4/silero-vad/master/"
    "src/silero_vad/data/silero_vad.onnx"
)
SILERO_PATH = Path(__file__).resolve().parent / "models" / "silero_vad.onnx"


def ensure_silero() -> Path:
    """Download the Silero VAD model once, on first use."""
    if SILERO_PATH.exists():
        return SILERO_PATH
    SILERO_PATH.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading Silero VAD model (~1 MB)...")
    urllib.request.urlretrieve(SILERO_URL, SILERO_PATH)
    return SILERO_PATH


CONTEXT_SIZE = 64   # samples of history Silero v5 expects, at 16 kHz


class SileroVAD:
    """Speech probability per 512-sample frame at 16 kHz.

    Silero v5 is NOT stateless per chunk: besides the recurrent state, it
    expects the last CONTEXT_SIZE samples of the *previous* chunk prepended to
    the current one, so the model receives 576 samples and sees a continuous
    signal across frame boundaries.

    Omitting that context does not fail loudly -- it returns plausible-looking
    but badly depressed probabilities. Here it capped out around 0.24 on audio
    that Moonshine transcribed perfectly ("Alexa, play my beyond playlist"),
    which silently discarded every command as "no speech detected".
    """

    def __init__(self, sample_rate: int = 16000) -> None:
        import onnxruntime

        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self.session = onnxruntime.InferenceSession(
            str(ensure_silero()), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.sample_rate = sample_rate
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(CONTEXT_SIZE, dtype=np.float32)

    def __call__(self, frame: np.ndarray) -> float:
        """frame: int16 samples, length 512."""
        chunk = frame.astype(np.float32) / 32768.0
        audio = np.concatenate([self._context, chunk]).reshape(1, -1)
        self._context = chunk[-CONTEXT_SIZE:].copy()
        out, self._state = self.session.run(
            None,
            {
                "input": audio,
                "state": self._state,
                "sr": np.array(self.sample_rate, dtype=np.int64),
            },
        )
        return float(out[0][0])


def device_candidates(name: Optional[str], exclusive: bool = False) -> list:
    """Resolve a device name to candidate indices, best first.

    A plain name is ambiguous: Windows exposes the same microphone once per host
    API (MME, DirectSound, WASAPI), and sounddevice refuses to guess. We prefer
    WASAPI first, always. On this machine DirectSound's *blocking read* path
    returns digital silence forever -- sd.rec() works on the same device only
    because it is callback-based -- so a resolver that prefers DirectSound pins
    us to a silently broken combination. WASAPI only opens at the hardware rate,
    which is why capture_rate exists and why we resample.

    Indices are resolved at startup rather than stored in config, because they
    shift whenever a device is added or removed.

    Returns a list because the preferred host API can still fail to open; the
    caller walks the list.
    """
    if not name:
        return [None]
    import sounddevice as sd

    hostapis = sd.query_hostapis()
    priority = {"Windows WASAPI": 0, "Windows DirectSound": 1, "MME": 2}
    matches = [
        (priority.get(hostapis[dev["hostapi"]]["name"], 9), idx)
        for idx, dev in enumerate(sd.query_devices())
        if dev["max_input_channels"] > 0 and name.lower() in dev["name"].lower()
    ]
    if not matches:
        raise RuntimeError(
            "Input device {!r} not found. Run 'python -m assistant.audio --list'.".format(name)
        )
    matches.sort()
    # Never substitute another physical microphone for the configured one.
    return [idx for _, idx in matches]


class Microphone:
    """Continuous 16 kHz mono capture with a rolling ring buffer."""

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_length: int = 512,
        ring_seconds: int = 30,
        device: Optional[str] = None,
        capture_rate: Optional[int] = None,
        exclusive: bool = False,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        # Capture may run at the device's native rate and be downsampled here.
        # Needed because WASAPI (and exclusive mode especially) will only open
        # at the hardware rate, while Silero and openWakeWord both want 16 kHz.
        self.capture_rate = capture_rate or sample_rate
        self.exclusive = exclusive
        self._requested_device = device
        if self.capture_rate % sample_rate:
            raise ValueError(
                "capture_rate {} must be an integer multiple of {}".format(
                    self.capture_rate, sample_rate
                )
            )
        self._decim = self.capture_rate // sample_rate
        self._read_size = frame_length * self._decim
        self._candidates = device_candidates(device, exclusive)
        self.device = None
        max_frames = int(ring_seconds * sample_rate / frame_length)
        self._ring: collections.deque = collections.deque(maxlen=max_frames)
        self._clean_ring: collections.deque = collections.deque(maxlen=max_frames)
        self._lock = threading.Lock()
        self._stream = None

        # Audio-thread handoff. The callback only appends; all work happens in
        # read(), so the audio thread never blocks. Overflow is counted and
        # reported because the bounded queue drops its oldest period.
        self._queue: collections.deque = collections.deque(maxlen=512)
        self.queue_overflow_count = 0
        self._reported_overflow_count = 0
        self._last_overflow_warning = None
        self._discard_depth = 0
        self.intentional_discard_count = 0
        self._pending = np.array([], dtype=np.int16)
        self._output_pending = np.array([], dtype=np.float32)
        self._resampler = None
        # Capture time (perf_counter) of the newest sample taken off the queue,
        # so each frame can be paired with the speaker reference by time.
        self._last_chunk_time = 0.0
        self.echo = None          # optional EchoCanceller
        self.last_clean = None    # echo-cancelled copy of the last frame read
        if self._decim > 1:
            import soxr

            self._resampler = soxr.ResampleStream(
                self.capture_rate, sample_rate, 1, dtype="float32"
            )

    def _callback(self, indata, frames, time_info, status) -> None:
        """Audio thread: copy in and get out. No processing here."""
        if status:
            log.debug("Input status: %s", status)
        if self._discard_depth:
            self.intentional_discard_count += 1
            return
        import time as _time
        if len(self._queue) == self._queue.maxlen:
            self.queue_overflow_count += 1
        self._queue.append((_time.perf_counter(), indata[:, 0].copy()))

    def reopen(self) -> None:
        """Recover the same microphone after sleep without reloading models."""
        self.__exit__()
        self.flush()
        self._candidates = device_candidates(self._requested_device, self.exclusive)
        if self._decim > 1:
            import soxr
            self._resampler = soxr.ResampleStream(
                self.capture_rate, self.sample_rate, 1, dtype="float32")
        self.__enter__()

    def __enter__(self) -> "Microphone":
        import sounddevice as sd

        extra = sd.WasapiSettings(exclusive=True) if self.exclusive else None
        errors = []
        for candidate in self._candidates:
            try:
                # blocksize=0 lets PortAudio use the device's native period.
                # Forcing 1536 made WASAPI shared mode stitch buffers across its
                # ~10 ms period and produced audibly muffled, garbled audio --
                # with a perfectly normal level and spectrum, which is why every
                # numeric check missed it. Callback-driven at the native period
                # is what Windows' own recorder and sd.rec do, and both are clean.
                self._stream = sd.InputStream(
                    samplerate=self.capture_rate,
                    blocksize=0,
                    channels=1,
                    dtype="int16",
                    device=candidate,
                    extra_settings=extra,
                    callback=self._callback,
                )
                self._stream.start()
            except sd.PortAudioError as exc:
                # Typically "Invalid sample rate": that host API will not
                # open at our requested rate. Try the next one.
                errors.append("[{}] {}".format(candidate, exc))
                if self._stream is not None:
                    self._stream.close()
                self._stream = None
                continue
            self.device = candidate
            name = sd.query_devices(candidate)["name"] if candidate is not None else "system default"
            log.info(
                "Microphone open: [%s] %s @ %d Hz%s%s",
                candidate, name, self.capture_rate,
                " (exclusive)" if self.exclusive else "",
                " -> {} Hz".format(self.sample_rate) if self._decim > 1 else "",
            )
            return self

        raise RuntimeError(
            "Could not open any input device at {} Hz: {}".format(
                self.capture_rate, " | ".join(errors)
            )
        )

    def __exit__(self, *exc) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
            finally:
                self._stream.close()
                self._stream = None
        # Drop the soxr handle explicitly; leaving it to interpreter shutdown
        # produced nanobind "leaked instance" warnings on exit.
        self._resampler = None
        self._queue.clear()
        self._pending = np.array([], dtype=np.int16)

    def read(self, timeout_s: float = 3.0) -> np.ndarray:
        """Block until one frame is available; append it to the ring buffer.

        Always returns exactly frame_length samples at self.sample_rate,
        resampling from the capture rate when they differ.
        """
        import time as _time

        # Logging can block on disk. Never do it on PortAudio's callback thread.
        if self.queue_overflow_count != self._reported_overflow_count:
            now = _time.monotonic()
            if self._last_overflow_warning is None or now - self._last_overflow_warning >= 5.0:
                log.warning("Microphone callback queue overflowed %d time(s); oldest audio chunks were dropped.",
                            self.queue_overflow_count)
                self._last_overflow_warning = now
                self._reported_overflow_count = self.queue_overflow_count
        deadline = _time.monotonic() + timeout_s
        # Streaming resamplers emit variable-sized chunks, including empty
        # chunks during filter startup. Preserve every sample; never pad or trim.
        while self._output_pending.size < self.frame_length:
            while self._pending.size < self._read_size:
                if self._queue:
                    stamp, data = self._queue.popleft()
                    self._last_chunk_time = stamp
                    self._pending = np.concatenate([self._pending, data])
                else:
                    if _time.monotonic() >= deadline:
                        raise TimeoutError(
                            "Microphone produced no complete audio frame for {:.1f} seconds".format(
                                timeout_s))
                    _time.sleep(0.002)
            chunk = self._pending[:self._read_size]
            self._pending = self._pending[self._read_size:]
            out = (self._resampler.resample_chunk(chunk.astype(np.float32))
                   if self._decim > 1 else chunk)
            self._output_pending = np.concatenate([self._output_pending, out])
        frame = np.clip(self._output_pending[:self.frame_length], -32768, 32767).astype(np.int16)
        self._output_pending = self._output_pending[self.frame_length:]

        if self.echo is not None:
            unread = (self._output_pending.size / self.sample_rate
                      + self._pending.size / self.capture_rate)
            try:
                self.last_clean = self.echo.process(frame, self._last_chunk_time - unread)
            except Exception as exc:
                log.warning("Echo cancellation failed; using the raw microphone: %s", exc)
                self.echo = None
                self.last_clean = None
        clean = self.last_clean
        clean_frame = None
        if (self.echo is not None and getattr(self.echo, "active", False) is True
                and isinstance(clean, np.ndarray)
                and clean.shape == frame.shape):
            # Echo may return raw audio while its speaker reference is inactive.
            # Archive only confirmed active AEC output, and own the buffer in
            # case the implementation reuses it on the next read.
            clean_frame = clean.copy()
        with self._lock:
            self._ring.append(frame)
            paired_ring = getattr(self, "_clean_ring", None)
            if paired_ring is not None:
                paired_ring.append(clean_frame)
        return frame

    def vad_frame(self, frame: np.ndarray) -> np.ndarray:
        """The frame to judge speech on: echo-cancelled when available."""
        clean = self.last_clean
        if self.echo is not None and clean is not None and clean.size == frame.size:
            return clean
        return frame

    def flush(self) -> None:
        """Discard everything captured so far.

        Called after the assistant speaks, so its own prompt never lands in the
        recording it is about to make.
        """
        self._queue.clear()
        self._pending = np.array([], dtype=np.int16)
        self._output_pending = np.array([], dtype=np.float32)
        self._last_chunk_time = 0.0
        self.last_clean = None
        with self._lock:
            self._ring.clear()
            paired_ring = getattr(self, "_clean_ring", None)
            if paired_ring is not None:
                paired_ring.clear()
        if self._resampler is not None:
            self._resampler.clear()

    @contextmanager
    def discard_capture(self):
        """Explicit half-duplex prompt playback: retain no self-speech backlog."""
        with self._lock:
            self._discard_depth += 1
        try:
            self.flush()
            yield
        finally:
            try:
                self.flush()
            finally:
                with self._lock:
                    self._discard_depth -= 1

    def preroll(self, seconds: float) -> np.ndarray:
        """The most recent `seconds` of audio already captured."""
        n_frames = int(seconds * self.sample_rate / self.frame_length)
        if n_frames <= 0:
            return np.array([], dtype=np.int16)
        with self._lock:
            frames = list(self._ring)[-n_frames:]
        return np.concatenate(frames) if frames else np.array([], dtype=np.int16)

    def preroll_pair(self, seconds: float) -> tuple:
        """Return matching raw and echo-clean frames, when all are available.

        The raw result follows ``preroll``'s frame rounding and contents. The
        processed result is None unless every selected frame has a valid,
        same-shaped echo-clean counterpart produced while AEC was active.
        """
        n_frames = int(seconds * self.sample_rate / self.frame_length)
        empty = np.array([], dtype=np.int16)
        if n_frames <= 0:
            return empty, empty.copy()
        with self._lock:
            raw_history = list(self._ring)
            clean_ring = getattr(self, "_clean_ring", None)
            raw_frames = raw_history[-n_frames:]
            raw = np.concatenate(raw_frames) if raw_frames else empty
            if clean_ring is None or len(clean_ring) != len(raw_history):
                return raw, None
            clean_frames = list(clean_ring)[-n_frames:]
            if (len(clean_frames) != len(raw_frames)
                    or not clean_frames
                    or any(clean is None for clean in clean_frames)
                    or any(not isinstance(clean, np.ndarray)
                           or clean.shape != raw_frame.shape
                           for raw_frame, clean in zip(raw_frames, clean_frames))):
                return raw, None
            processed = np.concatenate(clean_frames)
        return raw, processed

    def ring_contents(self) -> np.ndarray:
        with self._lock:
            frames = list(self._ring)
        return np.concatenate(frames) if frames else np.array([], dtype=np.int16)


class Endpointer:
    """Decides when the speaker stopped talking to the assistant.

    Silence ends the recording; a hard cap backstops it. The cap is what
    protects against walking off mid-sentence talking to someone else -- no
    acoustic method can detect that, so we bound it and explicitly report
    truncation. A partial recording must not silently become a complete task.
    """

    def __init__(
        self,
        vad: SileroVAD,
        sample_rate: int = 16000,
        frame_length: int = 512,
        silence_ms: int = 800,
        max_seconds: float = 6.0,
        speech_timeout_ms: int = 1500,
        threshold: float = 0.5,
        echo_window_ms: int = 350,
        echo_max_ms: int = 200,
        utterance_seconds: float | None = None,
    ) -> None:
        self.vad = vad
        self.sample_rate = sample_rate
        self.frame_ms = 1000.0 * frame_length / sample_rate
        self.silence_frames = int(silence_ms / self.frame_ms)
        self.max_frames = int(max_seconds * 1000 / self.frame_ms)
        self.timeout_frames = int(speech_timeout_ms / self.frame_ms)
        self.threshold = threshold
        self.utterance_frames = (None if utterance_seconds is None else
                                 max(1, int(utterance_seconds * 1000 / self.frame_ms)))
        # The Bluetooth speaker is still playing the end of our own prompt
        # when capture opens. That tail is a short burst right at the start;
        # it used to open AND close the recording before the user said a word
        # ("go ahead" -> "Okay.", "Repeat, please" -> "Please.").
        self.echo_window_frames = int(echo_window_ms / self.frame_ms)
        self.echo_max_frames = max(1, int(echo_max_ms / self.frame_ms))

    def record(self, mic: Microphone, after_prompt: bool = True) -> dict:
        """Capture one utterance.

        after_prompt=False: playback has already been discarded and settled,
        or only the 45 ms tick was used. A short opening word is the user.
        Rejecting it
        dropped one-word commands said straight after "Alexa" ("next", "pause").

        Returns PCM-relative speech offsets and duration, plus wall time spent
        waiting for the utterance in ``wait_duration_ms``.
        """
        echo_max_frames = self.echo_max_frames if after_prompt else 0
        self.vad.reset()
        frames: list = []
        buffer_origin_frame = 0
        silence_run = 0
        speech_started = False
        speech_start_frame = None
        voiced = 0
        deadline = self.timeout_frames
        keep = max(1, int(300 / self.frame_ms))
        end_reason = "capture_limit"
        capture_started = time.perf_counter()

        for i in range(self.max_frames):
            if (speech_started and self.utterance_frames is not None
                    and i - speech_start_frame >= self.utterance_frames):
                end_reason = "utterance_limit"
                break
            frame = mic.read()
            frames.append(frame)
            judged = mic.vad_frame(frame) if hasattr(mic, "vad_frame") else frame
            is_speech = self.vad(judged) >= self.threshold

            if is_speech:
                if not speech_started:
                    speech_started = True
                    speech_start_frame = i
                voiced += 1
                silence_run = 0
                continue

            silence_run += 1
            if not speech_started:
                # Nothing was ever said: a false accept. Bail early and quietly.
                if i >= deadline:
                    end_reason = "speech_timeout"
                    break
            elif (voiced <= echo_max_frames
                    and speech_start_frame < self.echo_window_frames
                    and silence_run >= min(self.silence_frames, keep)):
                # Prompt echo, not the user: forget it and keep waiting, with
                # the full reply window still ahead.
                speech_started = False
                speech_start_frame = None
                voiced = 0
                frames = frames[-keep:]
                buffer_origin_frame = i + 1 - len(frames)
                deadline = max(deadline, i + self.timeout_frames // 2)
            elif silence_run >= self.silence_frames:
                end_reason = "silence"
                break

            if not speech_started and len(frames) > keep:
                frames = frames[-keep:]
                buffer_origin_frame = i + 1 - len(frames)

        audio = np.concatenate(frames) if frames else np.array([], dtype=np.int16)
        duration_ms = int(audio.size * 1000 / self.sample_rate)
        return {
            "audio": audio,
            "speech_started": speech_started,
            "speech_start_ms": (
                int((speech_start_frame - buffer_origin_frame) * self.frame_ms)
                if speech_start_frame is not None
                else None
            ),
            "speech_end_ms": duration_ms,
            "duration_ms": duration_ms,
            "wait_duration_ms": int((time.perf_counter() - capture_started) * 1000),
            "end_reason": end_reason,
            "truncated": speech_started and end_reason in {"capture_limit", "utterance_limit"},
        }


def main() -> None:
    """`--list` prints input devices so you can pin one in config.toml."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description="Audio device helper")
    ap.add_argument("--list", action="store_true", help="List input devices")
    ap.add_argument("--vad", action="store_true", help="Live VAD meter (Ctrl-C to stop)")
    args = ap.parse_args()

    if args.list:
        import sounddevice as sd

        print("Input devices (pin one by name in config.toml -> audio.input_device):\n")
        for idx, dev in enumerate(sd.query_devices()):
            if dev["max_input_channels"] > 0:
                print("  [{}] {}".format(idx, dev["name"]))
        return

    if args.vad:
        from .config import load

        cfg = load()
        vad = SileroVAD()
        with Microphone(
            device=cfg.get("audio.input_device") or None,
            capture_rate=cfg.get("audio.capture_rate"),
            exclusive=cfg.get("audio.exclusive_mode", False),
        ) as mic:
            print("Speak. Ctrl-C to stop.")
            try:
                while True:
                    p = vad(mic.read())
                    bar = "#" * int(p * 40)
                    print("\r{:>5.2f} |{:<40}|".format(p, bar), end="", flush=True)
            except KeyboardInterrupt:
                print()


if __name__ == "__main__":
    main()
