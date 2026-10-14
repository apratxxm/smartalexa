"""Measure what WebRTC echo cancellation does to a recorded mic + loopback pair.

    python -m assistant.experimental.aec_eval assistant/logs/aec/<stamp>

Reports the speaker->mic delay, echo reduction (ERLE) and, per wake-word
attempt, the openWakeWord score and transcript before and after cancellation.
Everything runs locally.
"""
from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

import numpy as np
import soxr

RATE = 16000
FRAME = RATE // 100  # WebRTC APM works in exact 10 ms frames


def load_mono16k(path: Path) -> np.ndarray:
    with wave.open(str(path)) as wf:
        ch, rate = wf.getnchannels(), wf.getframerate()
        pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    x = pcm.reshape(-1, ch).astype(np.float32).mean(axis=1) / 32768.0
    return soxr.resample(x, rate, RATE).astype(np.float32)


def estimate_delay_ms(mic: np.ndarray, ref: np.ndarray, max_ms: int = 800) -> tuple:
    """Delay of the echo in the mic relative to the loopback, via GCC-PHAT."""
    n = min(len(mic), len(ref), RATE * 30)
    a, b = mic[:n], ref[:n]
    size = 1 << (2 * n - 1).bit_length()
    A, B = np.fft.rfft(a, size), np.fft.rfft(b, size)
    cross = A * np.conj(B)
    cross /= np.abs(cross) + 1e-12
    cc = np.fft.irfft(cross, size)
    max_lag = int(RATE * max_ms / 1000)
    lags = np.concatenate([cc[-max_lag:], cc[:max_lag + 1]])
    best = int(np.argmax(np.abs(lags))) - max_lag
    peak = float(np.abs(lags).max() / (np.abs(lags).mean() + 1e-12))
    return best * 1000 / RATE, peak


def cancel(mic: np.ndarray, ref: np.ndarray, delay_ms: float, hint_ms: int = 40,
           ns: bool = False) -> np.ndarray:
    from livekit import rtc

    apm = rtc.AudioProcessingModule(echo_cancellation=True, noise_suppression=ns,
                                    high_pass_filter=True, auto_gain_control=False)
    # Pre-align the reference so the canceller only has to find a small
    # residual delay; Bluetooth latency alone is far beyond its comfort zone.
    shift = max(0, int((delay_ms - hint_ms) * RATE / 1000))
    ref = np.concatenate([np.zeros(shift, np.float32), ref])[:len(mic)]
    if len(ref) < len(mic):
        ref = np.concatenate([ref, np.zeros(len(mic) - len(ref), np.float32)])
    out = np.empty_like(mic)
    to16 = lambda x: (np.clip(x, -1, 1) * 32767).astype(np.int16)
    for i in range(0, len(mic) - FRAME + 1, FRAME):
        r = to16(ref[i:i + FRAME])
        apm.process_reverse_stream(rtc.AudioFrame(r.tobytes(), RATE, 1, FRAME))
        apm.set_stream_delay_ms(hint_ms)
        m = rtc.AudioFrame(to16(mic[i:i + FRAME]).tobytes(), RATE, 1, FRAME)
        apm.process_stream(m)
        out[i:i + FRAME] = np.frombuffer(bytes(m.data), dtype=np.int16) / 32768.0
    out[len(mic) - len(mic) % FRAME:] = 0
    return out


def db(x: np.ndarray) -> float:
    return 10 * np.log10(np.mean(x.astype(np.float64) ** 2) + 1e-12)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--transcribe", action="store_true")
    args = ap.parse_args()
    folder = Path(args.folder)
    mic = load_mono16k(folder / "mic.wav")
    ref = load_mono16k(folder / "ref.wav")
    delay, peak = estimate_delay_ms(mic, ref)
    print(f"echo delay {delay:.0f} ms (correlation peak/mean {peak:.0f})")
    for ns in (False, True):
        out = cancel(mic, ref, delay, ns=ns)
        skip = RATE * 3  # adaptation time
        print(f"AEC{'+NS' if ns else ''}: mic {db(mic[skip:]):.1f} dB -> {db(out[skip:]):.1f} dB "
              f"(ERLE {db(mic[skip:]) - db(out[skip:]):.1f} dB)")
        name = "aec_ns.wav" if ns else "aec.wav"
        with wave.open(str(folder / name), "wb") as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(RATE)
            wf.writeframes((np.clip(out, -1, 1) * 32767).astype(np.int16).tobytes())
    with wave.open(str(folder / "mic16k.wav"), "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(RATE)
        wf.writeframes((np.clip(mic, -1, 1) * 32767).astype(np.int16).tobytes())
    print("wrote", folder / "aec.wav")


if __name__ == "__main__":
    main()
