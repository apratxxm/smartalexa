"""Backend startup and input contracts; no downloads or GPU required."""
import sys
import types
from contextlib import nullcontext

import numpy as np
import pytest

from assistant import stt


def test_lazy_cuda_failure_falls_back_to_cpu(monkeypatch):
    devices = []
    released = []

    class Model:
        def __init__(self, name, device, compute_type, local_files_only):
            assert local_files_only
            if device == "cpu":
                assert released == [False]
            devices.append(device)
            self.device = device
            self.model = types.SimpleNamespace(unload_model=lambda to_cpu: released.append(to_cpu))

        def transcribe(self, *args, **kwargs):
            def segments():
                if self.device == "cuda":
                    raise RuntimeError("missing cuDNN")
                yield types.SimpleNamespace(text="play music")
            return segments(), None

    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=Model))
    monkeypatch.setattr(stt, "_hub_offline", lambda _: nullcontext())
    monkeypatch.setattr(stt._FasterWhisperBackend, "_pick_device", staticmethod(lambda: ("cuda", "float16")))
    backend = stt._FasterWhisperBackend("small.en", 16000)
    assert devices == ["cuda", "cpu"]
    assert backend.device == "cpu" and backend.compute_type == "int8"
    assert backend.transcribe(np.ones(16000, dtype=np.float32)) == "play music"


@pytest.mark.parametrize("failure", [MemoryError(), RuntimeError("mkl_malloc: failed to allocate memory")])
def test_host_memory_failure_releases_gpu_without_second_model(monkeypatch, failure):
    attempts, released = [], []
    class Model:
        def __init__(self, name, device, compute_type, local_files_only):
            attempts.append(device)
            self.model = types.SimpleNamespace(unload_model=lambda to_cpu: released.append(to_cpu))
        def transcribe(self, *args, **kwargs):
            raise failure
    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=Model))
    monkeypatch.setattr(stt, "_hub_offline", lambda _: nullcontext())
    monkeypatch.setattr(stt._FasterWhisperBackend, "_pick_device", staticmethod(lambda: ("cuda", "float16")))
    with pytest.raises(type(failure)):
        stt._FasterWhisperBackend("small.en", 16000)
    assert attempts == ["cuda"] and released == [False]


def test_successful_gpu_keeps_original_precision_and_reports_placement(monkeypatch):
    class Model:
        def __init__(self, name, device, compute_type, local_files_only):
            assert (device, compute_type) == ("cuda", "float16")
        def transcribe(self, *args, **kwargs):
            return iter([]), None
    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=Model))
    monkeypatch.setattr(stt, "_hub_offline", lambda _: nullcontext())
    monkeypatch.setattr(stt._FasterWhisperBackend, "_pick_device", staticmethod(lambda: ("cuda", "float16")))
    transcriber = stt.Transcriber("faster-whisper/small.en")
    assert transcriber.runtime_info() == "cuda:float16"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows DLL directory API")
def test_cuda_dll_registration_is_scoped_and_idempotent(monkeypatch, tmp_path):
    cuda = tmp_path / "cuda"
    dll_dir = cuda / "bin"
    dll_dir.mkdir(parents=True)
    (dll_dir / "cublas64_12.dll").touch()
    calls = []
    monkeypatch.setenv("CUDA_PATH", str(cuda))
    monkeypatch.setenv("PATH", "")
    monkeypatch.setattr(stt, "_DLL_DIRECTORIES", {})
    monkeypatch.setattr(stt.importlib.util, "find_spec", lambda _: None)
    monkeypatch.setattr(stt.os, "add_dll_directory", lambda path: calls.append(path) or object())
    stt._prepare_cuda_dlls()
    stt._prepare_cuda_dlls()
    assert calls == [str(dll_dir)]
    assert str(dll_dir) in stt.os.environ["PATH"].split(stt.os.pathsep)
    assert len(stt._DLL_DIRECTORIES) == 1


@pytest.mark.parametrize("pcm", [np.zeros((10, 2), dtype=np.int16), np.zeros(10, dtype=np.float32)])
def test_invalid_pcm_is_rejected_before_decode(pcm):
    transcriber = stt.Transcriber.__new__(stt.Transcriber)
    with pytest.raises(ValueError, match="int16"):
        transcriber.transcribe(pcm)


def test_wrong_sample_rate_rejected_before_model_load():
    with pytest.raises(ValueError, match="16000"):
        stt.Transcriber(sample_rate=48000)


def test_one_shot_hotwords_do_not_mutate_default_hints():
    transcriber = stt.Transcriber.__new__(stt.Transcriber)
    transcriber._backend = types.SimpleNamespace(transcribe=lambda audio, hotwords=None: hotwords or "none")
    pcm = np.full(16000, 3000, dtype=np.int16)
    assert transcriber.transcribe(pcm, hotwords="Seedhe Maut")[0] == "Seedhe Maut"
    assert transcriber.transcribe(pcm)[0] == "none"


def test_unknown_backend_rejected():
    with pytest.raises(ValueError, match="Unsupported"):
        stt.Transcriber("typo/model")


def test_wake_decode_limits_are_separate_from_command_recognizer():
    calls = []
    def decode(*args, **kwargs):
        calls.append(kwargs)
        return iter([]), None
    wake = stt._FasterWhisperBackend.__new__(stt._FasterWhisperBackend)
    wake._model = types.SimpleNamespace(transcribe=decode)
    wake._hotwords = ""
    command = stt._FasterWhisperBackend.__new__(stt._FasterWhisperBackend)
    command._model = types.SimpleNamespace(transcribe=decode)
    command._hotwords = ""
    wake.set_wake_mode()
    wake.transcribe(np.ones(16000, dtype=np.float32))
    command.transcribe(np.ones(16000, dtype=np.float32))
    assert calls[0]["max_new_tokens"] == 64 and calls[0]["no_repeat_ngram_size"] == 3
    assert "max_new_tokens" not in calls[1] and "no_repeat_ngram_size" not in calls[1]


def test_hotwords_keep_whole_names_within_token_budget():
    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            return types.SimpleNamespace(ids=list(text))
    backend = stt._FasterWhisperBackend.__new__(stt._FasterWhisperBackend)
    backend._model = types.SimpleNamespace(max_length=24, hf_tokenizer=Tokenizer())
    backend.set_vocabulary(["Aerosmith", "Aerosmith", "Seedhe Maut", "", "An extremely long hint"])
    assert backend._hotwords == "Aerosmith"


def test_cache_miss_allows_one_download(monkeypatch):
    from huggingface_hub.errors import LocalEntryNotFoundError
    attempts = []
    class Model:
        def __init__(self, name, device, compute_type, local_files_only):
            attempts.append(local_files_only)
            if local_files_only:
                raise LocalEntryNotFoundError('not installed')
        def transcribe(self, *args, **kwargs):
            return iter([]), None
    monkeypatch.setitem(sys.modules, 'faster_whisper', types.SimpleNamespace(WhisperModel=Model))
    monkeypatch.setattr(stt, '_hub_offline', lambda _: nullcontext())
    monkeypatch.setattr(stt._FasterWhisperBackend, '_pick_device', staticmethod(lambda: ('cpu', 'int8')))
    stt._FasterWhisperBackend('small.en', 16000)
    assert attempts == [True, False]
