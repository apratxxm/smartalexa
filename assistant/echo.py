"""Echo-cancelled view of the microphone, used to tell the user from the music.

The laptop mic hears the SoundDrum louder than the user (-17 dBFS of music vs
-20 to -27 dBFS of voice) and Silero VAD fired on 71% of frames of sung music,
so utterances never ended and "Alexa" was chopped into 6 s lyric chunks.

Windows can give us exactly what the speaker is being asked to play (WASAPI
loopback). WebRTC's AEC3 (via livekit, local and free) reduces that reference.
One archived microphone/reference pair showed about 14 dB less music energy;
this is not a guarantee of live suppression or preservation of user speech.

AEC3 also mangles speech enough to hurt Whisper, so its output is used ONLY to
decide when the user is speaking. Whisper and openWakeWord keep the raw mic.

Streams are paired by capture time, not by arrival order, so the pairing
survives the main loop pausing for seconds while it handles a command.
AEC3 estimates the Bluetooth delay (~370 ms here) itself.
"""
from __future__ import annotations

import collections
import logging
import threading
import time
from typing import Optional

import numpy as np

log = logging.getLogger("memoir.echo")

RATE = 16000
BLOCK = RATE // 100          # AEC3 works on exact 10 ms blocks
REF_STALE_S = 1.0            # no loopback data for this long = no reference


class SpeakerReference:
    """WASAPI loopback of the output device as timestamped 16 kHz mono."""

    def __init__(self, device_hint: str = "", max_seconds: float = 20.0) -> None:
        self.device_hint = device_hint
        self._chunks: collections.deque = collections.deque()
        self._max = int(max_seconds * RATE)
        self._held = 0
        self._lock = threading.Lock()
        self._pa = None
        self._stream = None
        self._resampler = None
        self._rate = 0
        self._channels = 0
        self.last_data_at = 0.0
        self.name = ""
        self.generation = 0

    def route_changed(self) -> bool:
        """A live fallback stream must not hide the preferred speaker's return."""
        try:
            from pycaw.pycaw import AudioUtilities
            preferred = None
            if self.device_hint:
                preferred = next((d for d in AudioUtilities.GetAllDevices()
                                  if getattr(d.state, "value", d.state) == 1 and self.device_hint.casefold()
                                  in (d.FriendlyName or "").casefold()), None)
            device = preferred or AudioUtilities.GetSpeakers()
            name = device.FriendlyName if device else ""
            return bool(name and name.casefold() not in self.name.casefold())
        except Exception as exc:
            log.debug("Could not check loopback route: %s", exc)
            return False

    def start(self) -> None:
        import pyaudiowpatch as pa
        import soxr

        self.stop()
        self._pa = pa.PyAudio()
        device = None
        if self.device_hint:
            for dev in self._pa.get_loopback_device_info_generator():
                if self.device_hint.lower() in dev["name"].lower():
                    device = dev
                    break
        if device is None:
            device = self._pa.get_default_wasapi_loopback()
            if self.device_hint:
                log.warning("Preferred echo reference %r is unavailable; using %r. "
                            "Echo rejection applies only to audio on this output.",
                            self.device_hint, device["name"])
        self._rate = int(device["defaultSampleRate"])
        self._channels = int(device["maxInputChannels"])
        self._resampler = soxr.ResampleStream(self._rate, RATE, 1, dtype="float32")
        self.name = device["name"]

        def callback(in_data, frame_count, time_info, status):
            now = time.perf_counter()
            pcm = np.frombuffer(in_data, dtype=np.int16).reshape(-1, self._channels)
            mono = pcm.astype(np.float32).mean(axis=1) / 32768.0
            out = self._resampler.resample_chunk(mono)
            if out.size:
                start = now - frame_count / self._rate
                with self._lock:
                    self._chunks.append((start, out.astype(np.float32)))
                    self._held += out.size
                    while self._held > self._max and self._chunks:
                        self._held -= self._chunks.popleft()[1].size
                self.last_data_at = now
            return (None, pa.paContinue)

        self._stream = self._pa.open(
            format=pa.paInt16, channels=self._channels, rate=self._rate,
            input=True, input_device_index=int(device["index"]),
            frames_per_buffer=int(self._rate / 100), stream_callback=callback)
        self._stream.start_stream()
        self.generation += 1
        log.info("Speaker reference open: %s @ %d Hz", self.name, self._rate)

    def take_until(self, t: float) -> np.ndarray:
        """All reference audio captured before time t (perf_counter)."""
        parts = []
        with self._lock:
            while self._chunks and self._chunks[0][0] <= t:
                start, data = self._chunks.popleft()
                # Do not feed future samples merely because their chunk starts
                # before this mic frame ends.
                count = max(0, min(data.size, int(round((t - start) * RATE))))
                self._held -= count
                if count:
                    parts.append(data[:count])
                if count < data.size:
                    self._chunks.appendleft((start + count / RATE, data[count:]))
                    break
        return np.concatenate(parts) if parts else np.zeros(0, np.float32)

    def alive(self) -> bool:
        return time.perf_counter() - self.last_data_at < REF_STALE_S

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop_stream()
            except Exception as exc:
                log.debug("Loopback stop failed: %s", exc)
            try:
                self._stream.close()
            except Exception as exc:
                log.debug("Loopback close failed: %s", exc)
            self._stream = None
        if self._pa is not None:
            try:
                self._pa.terminate()
            except Exception as exc:
                log.debug("Loopback terminate failed: %s", exc)
            finally:
                self._pa = None
        self._resampler = None
        self.last_data_at = 0.0
        with self._lock:
            self._chunks.clear()
            self._held = 0


class EchoCanceller:
    """Pairs mic frames with the speaker reference and runs WebRTC AEC3."""

    def __init__(self, reference: SpeakerReference) -> None:
        from livekit import rtc

        self._rtc = rtc
        self.reference = reference
        self._generation = getattr(reference, "generation", 0)
        self._reset_processing()

    def _reset_processing(self) -> None:
        self._apm = self._rtc.AudioProcessingModule(
            echo_cancellation=True, noise_suppression=False,
            high_pass_filter=True, auto_gain_control=False)
        self._ref_pending = np.zeros(0, np.float32)
        self._mic_pending = np.zeros(0, np.int16)
        self._out_pending = np.zeros(0, np.int16)
        self.active = False

    def _render(self, samples: np.ndarray) -> None:
        self._ref_pending = np.concatenate([self._ref_pending, samples])
        while self._ref_pending.size >= BLOCK:
            block = (np.clip(self._ref_pending[:BLOCK], -1, 1) * 32767).astype(np.int16)
            self._ref_pending = self._ref_pending[BLOCK:]
            self._apm.process_reverse_stream(
                self._rtc.AudioFrame(block.tobytes(), RATE, 1, BLOCK))

    def process(self, frame: np.ndarray, frame_end_time: float) -> np.ndarray:
        """frame: int16 mic samples ending at frame_end_time. Returns cleaned int16."""
        generation = getattr(self.reference, "generation", 0)
        if generation != self._generation:
            self._reset_processing()
            self._generation = generation
        self._render(self.reference.take_until(frame_end_time))
        self.active = self.reference.alive()
        self._mic_pending = np.concatenate([self._mic_pending, frame])
        outs = []
        while self._mic_pending.size >= BLOCK:
            block = self._mic_pending[:BLOCK]
            self._mic_pending = self._mic_pending[BLOCK:]
            audio = self._rtc.AudioFrame(block.tobytes(), RATE, 1, BLOCK)
            # Required by the APM API whenever echo processing is enabled.
            # Do not invent a fixed Bluetooth delay from a different session.
            self._apm.set_stream_delay_ms(0)
            self._apm.process_stream(audio)
            outs.append(np.frombuffer(bytes(audio.data), dtype=np.int16))
        if outs:
            self._out_pending = np.concatenate([self._out_pending] + outs)
        # Keep output frame-aligned with input; the first call pads with zeros.
        if self._out_pending.size < frame.size:
            self._out_pending = np.concatenate(
                [np.zeros(frame.size - self._out_pending.size, np.int16), self._out_pending])
        clean = self._out_pending[:frame.size]
        self._out_pending = self._out_pending[frame.size:]
        return clean if self.active else frame


def open_echo_canceller(device_hint: str = "") -> Optional[EchoCanceller]:
    """Best effort: without a loopback the assistant runs exactly as before."""
    reference = None
    try:
        reference = SpeakerReference(device_hint)
        reference.start()
        return EchoCanceller(reference)
    except Exception as exc:
        if reference is not None:
            reference.stop()
        log.warning("Echo cancellation unavailable (%s); using the raw microphone.", exc)
        return None
