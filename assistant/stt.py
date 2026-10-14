"""Local speech recognition backends.

The legacy Moonshine backend runs on CPU; faster-whisper supports GPU decoding
and optional vocabulary hints. Hints can also introduce incorrect substitutions:
their value must be measured on real recordings, not assumed from availability.
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import logging
import os
import time
from pathlib import Path

import numpy as np
from contextlib import contextmanager

log = logging.getLogger("memoir.stt")

# Whisper's conditioning degrades with a very long hint list, but silently
# truncating below the number of hints we actually have is worse: it made
# 72 fetched names into 60 used ones, chosen by arbitrary ordering. Raised
# above the real count, and truncation is now logged rather than hidden.
MAX_HOTWORDS = 120
_DLL_DIRECTORIES: dict[str, object] = {}


def _prepare_cuda_dlls() -> None:
    """Register installed Windows CUDA libraries for this process only.

    Retain add_dll_directory handles: dropping them removes the search path.
    PATH is also needed by CTranslate2's dynamically loaded CUDA libraries.
    No downloads, machine-wide environment edits, or DLLs from the cwd.
    """
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return
    candidates = []
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        candidates.append(Path(cuda_path) / "bin")
    try:
        spec = importlib.util.find_spec("ctranslate2")
        if spec is not None and spec.origin:
            candidates.append(Path(spec.origin).parent)
    except (ImportError, ValueError):
        pass
    candidates.extend(Path(p) for p in os.environ.get("PATH", "").split(os.pathsep) if p)
    for directory in candidates:
        if not directory.is_absolute() or not any(
                (directory / dll).is_file() for dll in ("cublas64_12.dll", "cudnn64_9.dll")):
            continue
        key = os.path.normcase(str(directory.resolve()))
        if key in _DLL_DIRECTORIES:
            continue
        try:
            _DLL_DIRECTORIES[key] = os.add_dll_directory(str(directory))
        except OSError as exc:
            log.warning("CUDA DLL directory unavailable: %s (%s)", directory, exc)
            continue
        entries = os.environ.get("PATH", "").split(os.pathsep)
        if not any(os.path.normcase(p) == os.path.normcase(str(directory)) for p in entries):
            os.environ["PATH"] = str(directory) + os.pathsep + os.environ.get("PATH", "")
        log.info("CUDA DLL search: %s", directory)


@contextmanager
def _hub_offline(offline: bool):
    """Scope offline mode to one block and always restore it.

    The environment variable alone is not enough: huggingface_hub reads it into
    a module constant at import time, so a later os.environ change is ignored.
    Once Moonshine had imported the hub in offline mode, faster-whisper could
    never download its weights. Both have to be set.
    """
    previous = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1" if offline else "0"
    constants = None
    prior_const = None
    try:
        import huggingface_hub.constants as constants

        prior_const = constants.HF_HUB_OFFLINE
        constants.HF_HUB_OFFLINE = offline
    except Exception:
        constants = None
    try:
        yield
    finally:
        if constants is not None:
            constants.HF_HUB_OFFLINE = prior_const
        if previous is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = previous


class _MoonshineBackend:
    """Fast, tiny, CPU-only. No decoding bias."""

    def __init__(self, model_name: str, sample_rate: int) -> None:
        import moonshine_onnx

        # Prefer the local cache so a logon-time start does not block on the
        # network. Scoped tightly: leaking HF_HUB_OFFLINE into the process once
        # stopped faster-whisper downloading its own weights.
        with _hub_offline(True):
            try:
                self._model = moonshine_onnx.MoonshineOnnxModel(model_name=model_name)
                self._tokenizer = moonshine_onnx.load_tokenizer()
                loaded = True
            except Exception as exc:
                log.info("Moonshine not cached (%s); downloading once.", exc)
                loaded = False
        if not loaded:
            with _hub_offline(False):
                self._model = moonshine_onnx.MoonshineOnnxModel(model_name=model_name)
                self._tokenizer = moonshine_onnx.load_tokenizer()
        self._model.generate(np.zeros((1, sample_rate), dtype=np.float32))

    def set_vocabulary(self, words: list) -> None:
        pass    # not supported; this is the whole reason for the other backend

    def transcribe(self, audio: np.ndarray, hotwords: str | None = None) -> str:
        tokens = self._model.generate(audio.reshape(1, -1))
        return self._tokenizer.decode_batch(tokens)[0].strip()


class _FasterWhisperBackend:
    """Slower but far more accurate, and biasable toward known names."""

    def __init__(self, model_name: str, sample_rate: int) -> None:
        _prepare_cuda_dlls()
        from faster_whisper import WhisperModel
        from huggingface_hub.errors import LocalEntryNotFoundError

        def load_model(device, compute):
            try:
                return WhisperModel(model_name, device=device, compute_type=compute,
                                    local_files_only=True)
            except LocalEntryNotFoundError:
                log.info("Whisper weights are not cached; downloading %s once", model_name)
                return WhisperModel(model_name, device=device, compute_type=compute,
                                    local_files_only=False)

        device, compute = self._pick_device()
        log.info("faster-whisper %s on %s (%s)", model_name, device, compute)
        fallback = False
        self._model = None
        with _hub_offline(False):
            try:
                self._model = load_model(device, compute)
                self._warmup(sample_rate)
            except Exception as exc:
                if device != "cuda":
                    raise
                self._release_model()
                # A host allocation failure won't be helped by loading a second
                # large model on the CPU. Let the supervisor back off instead.
                if isinstance(exc, MemoryError) or any(token in str(exc).casefold() for token in (
                        "mkl_malloc", "bad_alloc", "cannot allocate memory", "failed to allocate memory")):
                    log.error("GPU startup exhausted host memory; CPU fallback skipped")
                    raise
                log.warning("GPU init failed (%s); released failed model before CPU fallback.", exc)
                fallback = True
            # Outside the except block so the failed warmup traceback no longer
            # holds its model while the CPU replacement is being allocated.
            if fallback:
                gc.collect()
                device, compute = "cpu", "int8"
                self._model = load_model("cpu", "int8")
                self._warmup(sample_rate)
        self.device, self.compute_type = device, compute
        log.info("STT active backend: %s (%s)", self.device, self.compute_type)
        self._hotwords = ""
        self._decode_limits = {}

    def set_wake_mode(self) -> None:
        # This recognizer only identifies an invocation; it never decodes the
        # requested song. Bound vocal-music hallucinations without shrinking
        # the command model or truncating its artist/title output.
        self._decode_limits = {"max_new_tokens": 64, "no_repeat_ngram_size": 3}

    def _release_model(self) -> None:
        model, self._model = self._model, None
        native = getattr(model, "model", None)
        if native is not None:
            try:
                native.unload_model(to_cpu=False)
            except Exception as exc:
                log.warning("Failed GPU model cleanup: %s", exc)

    def _warmup(self, sample_rate: int) -> None:
        # CUDA/cuDNN errors often surface only when the lazy segment generator
        # is consumed, not when WhisperModel is constructed.
        list(self._model.transcribe(np.zeros(sample_rate, dtype=np.float32),
                                    language="en")[0])

    @staticmethod
    def _pick_device() -> tuple:
        try:
            import ctranslate2

            if ctranslate2.get_cuda_device_count() > 0:
                return "cuda", "float16"
        except Exception as exc:
            log.debug("CUDA probe failed: %s", exc)
        return "cpu", "int8"

    def set_vocabulary(self, words: list) -> None:
        """Feed the user's playlist and artist names in as decoding hints."""
        # faster-whisper truncates at max_length // 2 - 1 *tokens*, not
        # names. Keep complete names within that budget and expose what is used.
        budget = self._model.max_length // 2 - 1
        selected = []
        seen = set()
        for word in words:
            word = word.strip()
            if not word or word.casefold() in seen:
                continue
            seen.add(word.casefold())
            trial = ", ".join(selected + [word])
            tokens = self._model.hf_tokenizer.encode(" " + trial, add_special_tokens=False).ids
            if len(selected) >= MAX_HOTWORDS or len(tokens) > budget:
                continue
            selected.append(word)
        self._hotwords = ", ".join(selected)
        log.info("STT hints: %d selected from %d requested (token budget %d)",
                 len(selected), len(words), budget)
        if len(selected) < len(words):
            # Name the casualties. Knowing only that "some" were dropped is
            # useless when a command later fails on an artist: the first
            # question is always whether that name was even in the hints.
            kept = {w.casefold() for w in selected}
            dropped = [w for w in words if w.strip().casefold() not in kept]
            log.warning("STT hints dropped (%d, budget-limited): %s",
                        len(dropped), "; ".join(dropped))

    def transcribe(self, audio: np.ndarray, hotwords: str | None = None) -> str:
        # Whisper loops on non-speech input: a 2 s silent clip took 10 s to
        # decode with beam search and temperature fallback enabled. Commands
        # are short and unambiguous, so greedy decoding with no fallback is
        # both faster and safer, and the thresholds below make it give up on
        # silence rather than inventing text.
        segments, _ = self._model.transcribe(
            audio,
            language="en",
            beam_size=1,
            best_of=1,
            temperature=0.0,
            hotwords=hotwords if hotwords is not None else (self._hotwords or None),
            condition_on_previous_text=False,   # each command stands alone
            vad_filter=False,                   # Silero already gated this
            without_timestamps=True,
            no_speech_threshold=0.6,
            compression_ratio_threshold=2.4,    # kills repetition loops
            **getattr(self, "_decode_limits", {}),
        )
        return " ".join(s.text for s in segments).strip()


class Transcriber:
    def __init__(self, model_name: str = "faster-whisper/small.en",
                 sample_rate: int = 16000) -> None:
        self.model_name = model_name
        self.sample_rate = sample_rate
        if sample_rate != 16000:
            raise ValueError("STT requires 16000 Hz mono audio")
        self._vocabulary: list = []

        log.info("Loading STT model %s ...", model_name)
        t0 = time.perf_counter()
        if model_name.startswith("faster-whisper/"):
            self._backend = _FasterWhisperBackend(
                model_name.split("/", 1)[1], sample_rate)
        elif model_name.startswith("moonshine/"):
            self._backend = _MoonshineBackend(model_name, sample_rate)
        else:
            raise ValueError("Unsupported STT backend: " + model_name)
        log.info("STT ready in %.1fs", time.perf_counter() - t0)

    def set_vocabulary(self, words: list) -> None:
        """Playlist and artist names fetched from Spotify at startup.

        Proper nouns are ASR's weak spot and every 'play X' command is one.
        Only the faster-whisper backend can actually use these as decoding
        hints; for Moonshine they are still useful downstream, for fuzzy repair.
        """
        self._vocabulary = words
        self._backend.set_vocabulary(words)
        log.info("Loaded %d vocabulary hints from Spotify.", len(words))

    @property
    def vocabulary(self) -> list:
        return self._vocabulary

    def runtime_info(self) -> str:
        """Actual backend placement, not a guess from overall GPU utilization."""
        backend = self._backend
        return "{}:{}".format(getattr(backend, "device", "cpu"),
                              getattr(backend, "compute_type", "onnx"))

    def set_wake_mode(self) -> None:
        configure = getattr(self._backend, "set_wake_mode", None)
        if callable(configure):
            configure()

    # Below this, treat the clip as silence and skip the model entirely.
    # Hotword biasing has a nasty edge: given near-silence, Whisper emits the
    # hint list itself ("DHH Gym, DHH Gym, Aerosmith"). Cheap to prevent.
    SILENCE_DBFS = -55.0

    def transcribe(self, pcm: np.ndarray, *, hotwords: str | None = None) -> tuple:
        """pcm: int16 mono at 16 kHz. Returns (text, latency_ms)."""
        if pcm.ndim != 1 or pcm.dtype != np.int16:
            raise ValueError("STT requires a one-dimensional int16 PCM array")
        if pcm.size == 0:
            return "", 0
        rms = float(np.sqrt((pcm.astype(np.float64) ** 2).mean()))
        if 20 * np.log10(max(rms, 1.0) / 32768.0) < self.SILENCE_DBFS:
            return "", 0
        audio = pcm.astype(np.float32) / 32768.0
        t0 = time.perf_counter()
        text = self._backend.transcribe(audio, hotwords=hotwords)
        return text, int((time.perf_counter() - t0) * 1000)


def main() -> None:
    """Transcript sanity check, latency, and a backend A/B on the same audio."""
    import wave

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description="STT smoke test")
    ap.add_argument("--file", help="16 kHz mono WAV to transcribe")
    ap.add_argument("--record", type=float, metavar="SECONDS")
    ap.add_argument("--model", default=None)
    ap.add_argument("--compare", action="store_true",
                    help="run every backend on the same audio")
    args = ap.parse_args()

    from .config import load

    cfg = load()

    if args.record:
        from .audio import Microphone

        with Microphone(
            device=cfg.get("audio.input_device") or None,
            capture_rate=cfg.get("audio.capture_rate"),
            exclusive=cfg.get("audio.exclusive_mode", False),
        ) as mic:
            print("Recording {:.0f}s -- speak now...".format(args.record))
            frames = [mic.read() for _ in range(int(args.record * 16000 / 512))]
        pcm = np.concatenate(frames)
    elif args.file:
        with wave.open(args.file, "rb") as wf:
            if (wf.getframerate(), wf.getnchannels(), wf.getsampwidth()) != (16000, 1, 2):
                ap.error("--file must be 16000 Hz mono, 16-bit PCM WAV")
            pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    else:
        ap.error("pass --file or --record")

    models = ([args.model] if args.model else
              (["moonshine/base", "faster-whisper/small.en"] if args.compare
               else [cfg.get("stt.model", "moonshine/base")]))

    hints = []
    if cfg.get("stt.bias_from_spotify", True):
        try:
            from .spotify import Spotify

            hints = Spotify().vocabulary()
        except Exception as exc:
            log.warning("No Spotify hints: %s", exc)

    for name in models:
        stt = Transcriber(name)
        if hints:
            stt.set_vocabulary(hints)
        text, ms = stt.transcribe(pcm)
        print("  {:<28} {:>6} ms  {!r}".format(name, ms, text))


if __name__ == "__main__":
    main()
