"""Owned callback capture; importing this module never opens an audio device."""
from __future__ import annotations

import json
import queue
import threading
import wave
from pathlib import Path

import numpy as np

from assistant.audio import device_candidates


def resolve_devices(cfg):
    import sounddevice as sd
    import pyaudiowpatch as pa
    mic_id = device_candidates(cfg.get("audio.input_device"), cfg.get("audio.exclusive_mode", False))[0]
    mic = sd.query_devices(mic_id, kind="input")
    # Resolve the same configured output used by Memoir's audible feedback.
    output_hint = cfg.get("feedback.output_device") or cfg.get("audio.echo_reference")
    if not output_hint:
        raise ValueError("A configured output device is required for acoustic anchors")
    output = sd.query_devices(output_hint, kind="output")
    engine = pa.PyAudio()
    try:
        matches = [d for d in engine.get_loopback_device_info_generator()
                   if output["name"].casefold() in d["name"].casefold()]
        reference_hint = cfg.get("audio.echo_reference", "")
        matches = [d for d in matches if reference_hint.casefold() in d["name"].casefold()]
        if len(matches) != 1:
            raise ValueError("Cannot uniquely match configured output to configured loopback; refusing fallback")
        loopback = dict(matches[0])
    finally:
        engine.terminate()
    return {"mic_index": mic_id, "mic_name": mic["name"],
            "mic_rate": int(cfg.get("audio.capture_rate", mic["default_samplerate"])),
            "output_hint": output_hint, "output_name": output["name"],
            "output_rate": int(output["default_samplerate"]),
            "loopback_index": int(loopback["index"]), "loopback_name": loopback["name"],
            "loopback_rate": int(loopback["defaultSampleRate"]),
            "loopback_channels": int(loopback["maxInputChannels"])}


class BlockSink:
    """Keep disk IO off callbacks, preserving native-rate indices and status."""

    def __init__(self, directory: Path, name: str, rate: int):
        self.rate = rate
        self.count = 0
        self.error = None
        self.last_end_time = None
        self.pending = queue.Queue(maxsize=1024)
        self.wav = wave.open(str(directory / f"{name}.wav"), "wb")
        self.wav.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        self.sidecar = (directory / f"{name}.blocks.jsonl").open("w", encoding="utf-8")
        self.thread = threading.Thread(target=self._write, name=f"wakebench-{name}", daemon=True)
        self.thread.start()

    def push(self, data, *, status="", overflow=False, capture_time=None):
        data = np.asarray(data, dtype=np.int16).reshape(-1).copy()
        gap = (capture_time is not None and self.last_end_time is not None
               and abs(capture_time - self.last_end_time) > .005)
        row = {"start_sample": self.count, "samples": len(data), "status": str(status),
               "overflow": bool(overflow), "discontinuity": bool(status or gap),
               "capture_time": capture_time}
        if capture_time is not None:
            self.last_end_time = capture_time + len(data) / self.rate
        self.count += len(data)
        try:
            self.pending.put_nowait((data, row))
        except queue.Full:
            self.error = "callback_writer_queue_overflow"

    def _write(self):
        try:
            while True:
                item = self.pending.get()
                if item is None:
                    return
                data, row = item
                self.wav.writeframesraw(data.astype("<i2", copy=False).tobytes())
                self.sidecar.write(json.dumps(row) + "\n")
        except Exception as exc:
            self.error = f"writer_failure: {type(exc).__name__}: {exc}"

    def close(self):
        while self.thread.is_alive():
            try:
                self.pending.put(None, timeout=.2)
                break
            except queue.Full:
                continue
        self.thread.join()
        self.wav.close()
        self.sidecar.close()


class Capture:
    def __init__(self, directory, devices, cfg):
        self.directory, self.devices, self.cfg = directory, devices, cfg
        self.mic = self.reference = self.engine = self.mic_sink = self.ref_sink = None

    def __enter__(self):
        import sounddevice as sd
        import pyaudiowpatch as pa
        d = self.devices
        try:
            self.mic_sink = BlockSink(self.directory, "mic", d["mic_rate"])
            self.ref_sink = BlockSink(self.directory, "loopback", d["loopback_rate"])
            self.engine = pa.PyAudio()

            def ref_callback(data, frames, times, status):
                pcm = np.frombuffer(data, dtype=np.int16).reshape(-1, d["loopback_channels"])
                mono = np.round(pcm.astype(np.float32).mean(axis=1)).astype(np.int16)
                self.ref_sink.push(mono, status=status if status else "", overflow=bool(status & pa.paInputOverflow),
                                   capture_time=times.get("input_buffer_adc_time"))
                return None, pa.paContinue

            def mic_callback(data, frames, times, status):
                self.mic_sink.push(data[:, 0], status=str(status), overflow=status.input_overflow,
                                   capture_time=times.inputBufferAdcTime)

            self.reference = self.engine.open(format=pa.paInt16, channels=d["loopback_channels"],
                rate=d["loopback_rate"], input=True, input_device_index=d["loopback_index"],
                frames_per_buffer=d["loopback_rate"] // 100, stream_callback=ref_callback)
            self.mic = sd.InputStream(device=d["mic_index"], channels=1, dtype="int16",
                samplerate=d["mic_rate"], blocksize=0, callback=mic_callback,
                extra_settings=sd.WasapiSettings(exclusive=True) if self.cfg.get("audio.exclusive_mode", False) else None)
            self.mic.start()
            return self
        except BaseException:
            self.__exit__()
            raise

    def cue_positions(self):
        from .session import RATE
        return {"expected_mic_sample": round(self.mic_sink.count * RATE / self.mic_sink.rate),
                "expected_loopback_sample": round(self.ref_sink.count * RATE / self.ref_sink.rate)}

    def check(self):
        errors = [s.error for s in (self.mic_sink, self.ref_sink) if s and s.error]
        if errors:
            raise RuntimeError("; ".join(errors))

    def __exit__(self, *args):
        errors = []
        for stream in (self.mic, self.reference):
            if stream is not None:
                try:
                    if hasattr(stream, "stop_stream"):
                        stream.stop_stream()
                    else:
                        stream.stop()
                except Exception as exc:
                    errors.append(exc)
                finally:
                    try:
                        stream.close()
                    except Exception as exc:
                        errors.append(exc)
        if self.engine is not None:
            self.engine.terminate()
        for sink in (self.mic_sink, self.ref_sink):
            if sink is not None:
                sink.close()
        self.mic = self.reference = self.engine = None
        if errors and not (args and args[0]):
            raise RuntimeError(f"Capture restoration failed: {errors}")
