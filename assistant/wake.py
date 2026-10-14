"""Wake-word detection (openWakeWord).

Porcupine was the original choice and is no longer viable: Picovoice
discontinued its free tier, existing free AccessKeys stop working after
2026-06-30, there is no personal/non-commercial plan, and the SDK validates the
key online at init even for otherwise-offline use. A 7-day trial cannot underpin
an always-on daemon.

openWakeWord is Apache-licensed, keyless, fully offline, and better suited here
in two ways the plan had already anticipated:

* It exposes a real 0-1 score, so the low/high confidence tiers are native
  rather than emulated by running two engines at different sensitivities.
* It can be trained on our own archived audio -- the only path to improving
  detection using the false accepts collected from this actual room, which
  Porcupine (closed, sensitivity-adjustable only) could never consume.

Frame sizes: openWakeWord wants 1280-sample (80 ms) chunks, while Silero VAD
requires exactly 512. The microphone stays at 512 for Silero's sake and this
class buffers up to 1280 internally.
"""
from __future__ import annotations

import logging
from importlib.metadata import version
from itertools import islice
from pathlib import Path
from types import MethodType
from typing import Optional

import numpy as np

from .config import ROOT, Config

log = logging.getLogger("memoir.wake")

TIER_LOOSE = "loose"
TIER_BOTH = "both"

CHUNK = 1280          # openWakeWord's expected chunk, 80 ms at 16 kHz
SAMPLE_RATE = 16000
FRAME_LENGTH = 512    # what the mic yields; Silero requires exactly this

PRETRAINED = ("alexa", "hey_mycroft", "hey_jarvis", "hey_rhasspy")


def _stream_recent_melspectrogram(features, n_samples):
    """openWakeWord 0.6.0's exact preprocessing, without copying 10 s of audio.

    Only the last n_samples + 480 samples are used by the upstream method.
    Traverse that tail from the right, then restore chronological order.
    """
    if len(features.raw_data_buffer) < 400:
        raise ValueError("The number of input frames must be at least 400 samples @ 16khz (25 ms)!")
    recent = list(islice(reversed(features.raw_data_buffer), n_samples + 160 * 3))
    recent.reverse()
    features.melspectrogram_buffer = np.vstack((features.melspectrogram_buffer,
                                                features._get_melspectrogram(recent)))
    if features.melspectrogram_buffer.shape[0] > features.melspectrogram_max_len:
        features.melspectrogram_buffer = features.melspectrogram_buffer[-features.melspectrogram_max_len:, :]


def _optimize_preprocessor(model) -> None:
    # Instance-scoped compatibility shim for the pinned version only. Do not
    # edit site-packages or apply an old internal method to a future release.
    features = getattr(model, "preprocessor", None)
    if features is not None and version("openwakeword") == "0.6.0":
        features._streaming_melspectrogram = MethodType(_stream_recent_melspectrogram, features)


class WakeWordError(RuntimeError):
    """Startup failure. Must be loud: a silently dead wake word looks identical
    to an assistant that simply isn't hearing you."""


class WakeDetector:
    def __init__(self, cfg: Config) -> None:
        from openwakeword.model import Model
        import openwakeword

        configured = cfg.get("wake.model", "hey_jarvis")
        if isinstance(configured, str):
            names = [configured]
        elif isinstance(configured, (list, tuple)):
            names = list(configured)
        else:
            raise WakeWordError("wake.model must be a model name, path, or list of names/paths.")
        if not names:
            raise WakeWordError("No wake models are configured; set wake.model to a supported model name or model path.")
        self.threshold_low = cfg.get("wake.threshold_low", 0.5)
        self.threshold_high = cfg.get("wake.threshold_high", 0.7)

        # Several models can run at once, and ANY of them may trigger. Measured
        # per-utterance scores for a single model swing from 0.002 to 0.998 on
        # the same voice in the same room, and different models miss different
        # utterances -- so running two covers noticeably more attempts than
        # either alone. Each costs ~3 ms per 80 ms chunk.
        def required_onnx(path: Path, label: str) -> str:
            if not path.is_file():
                raise WakeWordError(
                    "Required {} ONNX asset is missing at {}. "
                    "Provision it during explicit setup; runtime startup does not download models."
                    .format(label, path)
                )
            return str(path)

        # Runtime initialization is intentionally offline. Resolve pretrained
        # aliases through the installed package metadata, but pass the exact
        # ONNX paths to Model so it never relies on name matching or TFLite files.
        models = []
        stock_model_names = {}

        def add_stock_model(path: Path, configured_name: str) -> None:
            model_key = path.stem
            if model_key in stock_model_names:
                raise WakeWordError(
                    "Configured wake models collide on openWakeWord key {!r}. "
                    "Use distinct model filenames.".format(model_key)
                )
            stock_model_names[model_key] = configured_name

        for name in names:
            if not isinstance(name, str) or not name.strip():
                raise WakeWordError("Invalid empty or non-text wake model entry: {!r}".format(name))
            if name in PRETRAINED:
                metadata = getattr(openwakeword, "MODELS", {}).get(name)
                model_path = metadata.get("model_path") if isinstance(metadata, dict) else None
                if not isinstance(model_path, str) or not model_path:
                    raise WakeWordError(
                        "Installed openWakeWord metadata has no model path for {!r}. "
                        "Install a supported openWakeWord package and provision its ONNX assets."
                        .format(name)
                    )
                onnx_path = Path(model_path).with_suffix(".onnx")
                models.append(required_onnx(onnx_path,
                                            "pretrained wake model {!r}".format(name)))
                add_stock_model(onnx_path, name)
                continue
            path = Path(name)
            if not path.is_absolute():
                path = ROOT / name
            if not path.is_file():
                raise WakeWordError(
                    "Wake word model not found at {}.\n"
                    "Either use one of {} or train a custom model "
                    "(see README).".format(path, PRETRAINED)
                )
            models.append(str(path))
            add_stock_model(path, path.stem)

        feature_models = getattr(openwakeword, "FEATURE_MODELS", {})
        feature_paths = {}
        for feature_name in ("melspectrogram", "embedding"):
            metadata = feature_models.get(feature_name)
            model_path = metadata.get("model_path") if isinstance(metadata, dict) else None
            if not isinstance(model_path, str) or not model_path:
                raise WakeWordError(
                    "Installed openWakeWord metadata has no {} feature model path. "
                    "Install a supported openWakeWord package and provision its ONNX assets."
                    .format(feature_name)
                )
            feature_paths[feature_name] = required_onnx(
                Path(model_path).with_suffix(".onnx"),
                "openWakeWord {} feature model".format(feature_name))
        # The personal model (trained on this user's voice) only PROPOSES a
        # wake: it catches ~4x more of the user's "Alexa"s than stock but
        # false-fires ~once a minute on other speech, so a text check must
        # confirm it. It is kept out of the stock score/tier logic below.
        self._personal_key = ""
        personal = cfg.get("wake.personal_model", "") or ""
        if personal:
            path = Path(personal)
            if not path.is_absolute():
                path = ROOT / personal
            if path.exists():
                if path.stem in stock_model_names:
                    raise WakeWordError(
                        "Personal wake model key {!r} collides with a configured stock model."
                        .format(path.stem)
                    )
                models.append(str(path))
                self._personal_key = path.stem
            else:
                log.warning("Personal wake model missing at %s; stock models only", path)
        self.personal_threshold = float(cfg.get("wake.personal_threshold", 0.5))
        self.personal_patience = max(1, int(cfg.get("wake.personal_patience", 2)))
        self.personal_score = 0.0
        self.personal_hit = False
        self._personal_run = 0

        try:
            self._model = Model(
                wakeword_models=models, inference_framework="onnx",
                melspec_model_path=feature_paths["melspectrogram"],
                embedding_model_path=feature_paths["embedding"])
            _optimize_preprocessor(self._model)
        except Exception as exc:
            raise WakeWordError(
                "openWakeWord failed to initialize: {}".format(exc)
            ) from exc

        self._model_keys = [k for k in self._model.models.keys() if k != self._personal_key]
        self._stock_model_names = stock_model_names
        self._keys = [stock_model_names.get(k, k) for k in self._model_keys]
        self.last_model = ""
        self._buffer = np.array([], dtype=np.int16)
        self.sample_rate = SAMPLE_RATE
        self.frame_length = FRAME_LENGTH
        self.last_score = 0.0

        log.info(
            "Wake word ready (%s), thresholds %.2f / %.2f%s",
            ", ".join(self._keys), self.threshold_low, self.threshold_high,
            "; personal {} proposes at {:.2f} x{}".format(
                self._personal_key, self.personal_threshold, self.personal_patience)
            if self._personal_key else "",
        )

    def process(self, frame: np.ndarray) -> Optional[str]:
        """Feed one 512-sample frame. Returns a confidence tier, or None.

        Frames are buffered to openWakeWord's 1280-sample chunk; most calls
        therefore do no inference at all.
        """
        self.personal_hit = False
        self._buffer = np.concatenate([self._buffer, frame])
        if self._buffer.size < CHUNK:
            return None

        chunk, self._buffer = self._buffer[:CHUNK], self._buffer[CHUNK:]
        scores = self._model.predict(chunk)
        if self._personal_key:
            self.personal_score = float(scores.get(self._personal_key, 0.0))
            self._personal_run = (self._personal_run + 1
                                  if self.personal_score >= self.personal_threshold else 0)
            # Fire once per run of consecutive chunks, not on every chunk.
            self.personal_hit = self._personal_run == self.personal_patience
        # Best of all loaded models: any one of them is enough to wake us.
        model_key = max(self._model_keys, key=lambda key: float(scores[key]))
        self.last_model = self._stock_model_names.get(model_key, model_key)
        self.last_score = float(scores[model_key])

        if self.last_score < self.threshold_low:
            return None
        return TIER_BOTH if self.last_score >= self.threshold_high else TIER_LOOSE

    def reset(self) -> None:
        """Clear internal state after a trigger, so the same utterance cannot
        re-fire on the next chunk."""
        self._buffer = np.array([], dtype=np.int16)
        self._personal_run = 0
        self.personal_hit = False
        self._model.reset()

    def close(self) -> None:
        self._buffer = np.array([], dtype=np.int16)


def main() -> None:
    """Interactive meter: input level and wake score side by side.

    Two bars, because the two failure modes look identical from the outside:
    "it isn't hearing me" (level bar stays flat) and "it hears me but doesn't
    recognise the phrase" (level bar moves, score bar doesn't).
    """
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from .audio import Microphone
    from .config import load

    ap = argparse.ArgumentParser(description="Wake word meter")
    ap.add_argument("--exclusive", dest="exclusive", action="store_true",
                    help="force WASAPI exclusive (bypasses the Windows effects chain)")
    ap.add_argument("--shared", dest="exclusive", action="store_false",
                    help="force shared mode")
    ap.add_argument("--model", help="override the wake model for this run")
    ap.set_defaults(exclusive=None)
    args = ap.parse_args()

    cfg = load()
    if args.model:
        cfg._data.setdefault("wake", {})["model"] = args.model
    exclusive = (cfg.get("audio.exclusive_mode", False)
                 if args.exclusive is None else args.exclusive)
    detector = WakeDetector(cfg)
    detections = 0
    peak_score = 0.0
    loudest = -99.0

    with Microphone(
        sample_rate=detector.sample_rate,
        frame_length=detector.frame_length,
        device=cfg.get("audio.input_device") or None,
        capture_rate=cfg.get("audio.capture_rate"),
        exclusive=exclusive,
    ) as mic:
        print("")
        print("Mode: {}".format("EXCLUSIVE (effects bypassed)" if exclusive
                                else "shared (through Windows effects)"))
        print("Say the wake word. Ctrl-C to stop.")
        print("  LEVEL should reach roughly -30 dBFS or louder when you speak.")
        print("  SCORE must cross {:.2f} to trigger.".format(detector.threshold_low))
        print("")
        try:
            while True:
                frame = mic.read()
                tier = detector.process(frame)
                rms = float(np.sqrt((frame.astype(np.float64) ** 2).mean()))
                db = 20 * np.log10(max(rms, 1.0) / 32768.0)
                loudest = max(loudest, db)
                peak_score = max(peak_score, detector.last_score)

                level_bar = "#" * int(max(0.0, (db + 60) / 60) * 24)
                score_bar = "#" * int(detector.last_score * 24)
                line = " LEVEL {:>6.1f} dB |{:<24}|  SCORE {:>5.2f} |{:<24}|".format(
                    db, level_bar, detector.last_score, score_bar
                )
                print("\r" + line, end="", flush=True)

                if tier:
                    detections += 1
                    print("\r  *** DETECTED  score={:.2f}  tier={}{}".format(
                        detector.last_score, tier, " " * 40))
                    detector.reset()
        except KeyboardInterrupt:
            print("")
            print("")
            print("summary: {} detections | loudest {:.1f} dBFS | peak score {:.3f}"
                  .format(detections, loudest, peak_score))
        finally:
            detector.close()


if __name__ == "__main__":
    main()
