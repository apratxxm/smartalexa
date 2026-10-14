"""Record the laptop mic and the speaker feed (WASAPI loopback) together.

Offline material for measuring echo cancellation: the loopback is exactly what
the SoundDrum is being asked to play, i.e. the reference the canceller needs.
Nothing leaves the machine; files go to assistant/logs/aec/.

    python -m assistant.experimental.aec_capture --seconds 60
"""
from __future__ import annotations

import argparse
import threading
import time
import wave
from pathlib import Path

import numpy as np
import pyaudiowpatch as pa

OUT = Path(__file__).resolve().parents[1] / "logs" / "aec"


def _find_input(p: pa.PyAudio, name: str) -> dict:
    wasapi = p.get_host_api_info_by_type(pa.paWASAPI)["index"]
    for i in range(p.get_device_count()):
        d = p.get_device_info_by_index(i)
        if d["hostApi"] == wasapi and d["maxInputChannels"] > 0 and name in d["name"] \
                and not d.get("isLoopbackDevice"):
            return d
    raise RuntimeError("input device not found: " + name)


def capture(seconds: float, mic_name: str) -> Path:
    p = pa.PyAudio()
    mic = _find_input(p, mic_name)
    loop = p.get_default_wasapi_loopback()
    streams = {}
    data = {"mic": [], "ref": []}
    stamps = {"mic": [], "ref": []}

    def cb(key):
        def f(in_data, frame_count, time_info, status):
            data[key].append(in_data)
            stamps[key].append(time.perf_counter())
            return (None, pa.paContinue)
        return f

    for key, dev in (("mic", mic), ("ref", loop)):
        streams[key] = p.open(
            format=pa.paInt16, channels=int(dev["maxInputChannels"]),
            rate=int(dev["defaultSampleRate"]), input=True,
            input_device_index=int(dev["index"]), frames_per_buffer=480,
            stream_callback=cb(key))
    t0 = time.perf_counter()
    for s in streams.values():
        s.start_stream()
    time.sleep(seconds)
    for s in streams.values():
        s.stop_stream(); s.close()
    p.terminate()

    stamp = time.strftime("%Y%m%d-%H%M%S")
    folder = OUT / stamp
    folder.mkdir(parents=True, exist_ok=True)
    for key, dev in (("mic", mic), ("ref", loop)):
        ch = int(dev["maxInputChannels"])
        pcm = np.frombuffer(b"".join(data[key]), dtype=np.int16).reshape(-1, ch)
        with wave.open(str(folder / f"{key}.wav"), "wb") as wf:
            wf.setnchannels(ch); wf.setsampwidth(2)
            wf.setframerate(int(dev["defaultSampleRate"]))
            wf.writeframes(pcm.tobytes())
        first = stamps[key][0] - t0 if stamps[key] else -1
        print(f"{key}: {dev['name']} {ch}ch {int(dev['defaultSampleRate'])} Hz "
              f"{pcm.shape[0] / dev['defaultSampleRate']:.1f}s first-callback {first*1000:.0f} ms")
    print("saved", folder)
    return folder


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--mic", default="Microphone Array (Realtek(R) Audio)")
    capture(ap.parse_args().seconds, ap.parse_args().mic)
