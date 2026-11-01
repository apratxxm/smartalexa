"""Interactive continuous paired recording, with offline acoustic finalization."""
from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path

from assistant.config import ROOT, load
from .capture import Capture, resolve_devices
from .session import FRAME, RATE, chirp, finalize, write_jsonl, write_wav

COMMANDS = ("Alexa, pause", "Alexa, play Halo", "Alexa, what time is it", "Alexa, volume forty")
BASE = Path(__file__).parent / "recordings"


def _key():
    import msvcrt
    return msvcrt.getwch().lower() if msvcrt.kbhit() else None


def _wait(seconds, capture, attempt=None):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        capture.check()
        key = _key()
        if key == "q":
            raise KeyboardInterrupt
        if key == "x" and attempt is not None:
            attempt["user_invalid"] = True
            print("Attempt marked invalid.")
        if key == "p":
            # Capture continues: pause time remains measured room exposure.
            if attempt is not None:
                attempt["user_invalid"] = True
            print("Paused; press p to resume or q to abort. Recording continues.")
            while True:
                capture.check()
                key = _key()
                if key == "q":
                    raise KeyboardInterrupt
                if key == "p":
                    break
                time.sleep(.025)
            deadline = max(deadline, time.monotonic())
        time.sleep(.010)


def run(args, cfg, devices):
    import sounddevice as sd
    import soxr
    from assistant.runtime import single_instance
    session_id = uuid.uuid4().hex
    directory = args.output / session_id
    # The runtime mutex is held throughout recording, preventing either runtime
    # from starting after this check. No devices are opened before admission.
    with single_instance(ROOT):
        directory.mkdir(parents=True)
        cue = chirp()
        write_wav(directory / "cue.wav", cue)
        header = {"type": "session", "version": 1, "session_id": session_id,
            "mode": args.mode, "distance": args.distance, "music": args.music,
            "mic_wav": "mic.wav", "loopback_wav": "loopback.wav", "cue_wav": "cue.wav",
            "mic_blocks": "mic.blocks.jsonl", "loopback_blocks": "loopback.blocks.jsonl",
            "mic_rate": devices["mic_rate"], "loopback_rate": devices["loopback_rate"],
            "devices": devices, "finalized": False, "aborted": False}
        rows = [header]
        waveform = soxr.resample(cue.astype("float32") / 32768, RATE, devices["output_rate"])
        if args.mode == "negative":
            waveform *= .25
        print(f"Session {session_id}: x invalidates an attempt; p pauses; q aborts.\n{directory}")
        capture = Capture(directory, devices, cfg)
        try:
            with capture, sd.OutputStream(device=devices["output_hint"], channels=1,
                    samplerate=devices["output_rate"], dtype="float32") as output:
                def emit(index, anchor=False):
                    row = {"type": "cue", "session_id": session_id, "cue_id": index,
                           "anchor": anchor, "user_invalid": False, **capture.cue_positions()}
                    rows.append(row)
                    # Persist cue scheduling evidence before playback.
                    write_jsonl(directory / "manifest.jsonl", rows)
                    output.write(waveform.reshape(-1, 1))
                    return row

                _wait(1., capture)
                if args.mode == "positive":
                    for index in range(args.attempts):
                        print(f"{index + 1}/{args.attempts}: {COMMANDS[index % len(COMMANDS)]} — speak after the chirp")
                        row = emit(index)
                        row["command"] = COMMANDS[index % len(COMMANDS)]
                        _wait(5., capture, row)
                else:
                    end = time.monotonic() + args.minutes * 60
                    index = 0
                    while time.monotonic() < end:
                        emit(index, anchor=True)
                        index += 1
                        _wait(min(60., max(0, end - time.monotonic())), capture)
                    emit(index, anchor=True)
                    _wait(2., capture)
                capture.check()
        except KeyboardInterrupt:
            header["aborted"] = True
            print("Aborted; partial WAVs retained and excluded from benchmarking.")
        except BaseException as exc:
            header["writer_error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            errors = [s.error for s in (capture.mic_sink, capture.ref_sink) if s and s.error]
            if errors:
                header["writer_error"] = "; ".join(errors)
            write_jsonl(directory / "manifest.jsonl", rows)
        if not header["aborted"] and not header.get("writer_error"):
            result = finalize(directory)
            invalid = sum(r.get("invalid", False) for r in result if not r.get("anchor") and r["type"] == "cue")
            print(f"Finalized {directory}; invalid attempts: {invalid}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("positive", "negative"), default="positive")
    parser.add_argument("--attempts", type=int, default=30)
    parser.add_argument("--minutes", type=float, default=60)
    parser.add_argument("--distance", choices=("near", "mid", "far"))
    parser.add_argument("--music", choices=("on", "off"))
    parser.add_argument("--output", type=Path, default=BASE)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--finalize", type=Path, metavar="SESSION", help="Locate cues in saved WAVs; no devices")
    args = parser.parse_args(argv)
    if args.finalize:
        rows = finalize(args.finalize)
        print(json.dumps({"session_id": rows[0]["session_id"], "finalized": True,
                          "invalid": sum(r.get("invalid", False) for r in rows)}))
        return
    if args.attempts < 1 or not 0 < args.minutes <= 1440:
        parser.error("--attempts must be positive; --minutes must be in (0, 1440]")
    cfg = load()
    devices = resolve_devices(cfg)
    if args.dry_run:
        print(json.dumps({"plan": {"mode": args.mode, "attempts": args.attempts,
            "minutes": args.minutes, "distance": args.distance, "music": args.music,
            "frame_samples": FRAME, "sample_rate": RATE, "chirp_ms": 60,
            "window_after_cue_seconds": [.15, 3.5], "negative_anchor_seconds": 60,
            "output": str(args.output)}, "resolved_devices": devices}, indent=2))
        return
    if args.mode == "positive" and (args.distance is None or args.music is None):
        parser.error("positive recording requires --distance near|mid|far and --music on|off")
    run(args, cfg, devices)


if __name__ == "__main__":
    main()
