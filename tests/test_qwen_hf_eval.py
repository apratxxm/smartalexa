from __future__ import annotations

import json
import wave
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from assistant.experimental.asr_eval import run_qwen_hf as runner


def _wav(path: Path, samples: list[int]) -> None:
    import struct

    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(struct.pack("<" + "h" * len(samples), *samples))


def _manifest(tmp_path: Path) -> Path:
    command = tmp_path / "command.wav"
    negative = tmp_path / "negative.wav"
    _wav(command, [1000, 2000, 3000, 4000])
    _wav(negative, [5000, 6000, 7000, 8000])
    manifest = {
        "sample_rate": 16000,
        "command_clips": [{"id": "cmd", "audio_path": str(command),
                            "expected_transcript": "pause", "source": "explicit",
                            "offset_samples": 1, "num_samples": 2}],
        "negative_clips": [{"id": "neg", "audio_path": str(negative),
                             "source": "aec_negative", "offset_samples": 2,
                             "num_samples": 2}],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _args(tmp_path: Path, **overrides):
    values = {
        "model_path": str(tmp_path),
        "device": "cpu",
        "language": "auto",
        "max_clips": 64,
        "deadline_seconds": 30.0,
        "threads": 4,
        "max_new_tokens": 128,
        "report": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_read_audio_honors_offset_and_length(tmp_path: Path):
    manifest_path = _manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audio = runner.read_audio(manifest["command_clips"][0])
    np.testing.assert_allclose(audio, np.array([2000, 3000], dtype=np.float32) / 32768.0)


def test_one_inference_per_case_and_failures_are_not_retried(tmp_path: Path):
    manifest_path = _manifest(tmp_path)

    class FakeTorch:
        __version__ = "fake-torch"
        version = SimpleNamespace(cuda=None)
        float16 = "fp16"
        float32 = "fp32"
        cuda = SimpleNamespace(is_available=lambda: False,
                               synchronize=lambda: None)
        inference_mode = staticmethod(nullcontext)

        @staticmethod
        def set_num_threads(_count):
            pass

    class FakeProcessor:
        requests = []

        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return cls()

        def apply_transcription_request(self, *, audio, language):
            self.requests.append((audio.copy(), language))
            return {"audio": audio, "input_ids": np.array([[101, 102]])}

        @staticmethod
        def decode(tokens, **kwargs):
            assert kwargs["return_format"] == "transcription_only"
            assert "skip_special_tokens" not in kwargs
            np.testing.assert_array_equal(tokens, [[201, 202]])
            return "pause"

    class FakeModel:
        calls = 0
        device = "cpu"

        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return cls()

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            return self

        def generate(self, **_inputs):
            type(self).calls += 1
            if type(self).calls == 2:
                raise RuntimeError("synthetic inference error")
            return np.array([[101, 102, 201, 202]])

    ticks = iter(i * 0.1 for i in range(100))
    output = []
    result = runner.run(
        _args(tmp_path), manifest_path=manifest_path,
        dependency_loader=lambda: (FakeTorch, SimpleNamespace(__version__="fake-transformers"),
                                   FakeProcessor, FakeModel),
        now=lambda: next(ticks), emit=output.append,
    )

    rows = [json.loads(line) for line in output]
    cases = [row for row in rows if row.get("event") == "case"]
    assert [row["case_id"] for row in cases] == ["cmd", "neg"]
    assert [row["failure"] for row in cases] == [False, True]
    assert "synthetic inference error" in cases[1]["error"]
    assert FakeModel.calls == 2  # the failed case was not retried
    assert len(FakeProcessor.requests) == 2
    assert [len(audio) for audio, _language in FakeProcessor.requests] == [2, 2]
    assert all(language is None for _audio, language in FakeProcessor.requests)
    assert result["cases_failed"] == 1


def test_existing_report_is_refused_before_dependencies_load(tmp_path: Path):
    manifest_path = _manifest(tmp_path)
    report_path = tmp_path / "existing.jsonl"
    report_path.write_text("keep", encoding="utf-8")
    dependency_calls = []

    with pytest.raises(FileExistsError):
        runner.run(
            _args(tmp_path, report=str(report_path)), manifest_path=manifest_path,
            dependency_loader=lambda: dependency_calls.append("loaded"),
        )

    assert report_path.read_text(encoding="utf-8") == "keep"
    assert dependency_calls == []


@pytest.mark.parametrize("deadline", [float("nan"), float("inf"), 901, True])
def test_invalid_deadline_is_rejected_before_dependency_load(tmp_path, deadline):
    manifest_path = _manifest(tmp_path)
    with pytest.raises(ValueError, match="deadline-seconds"):
        runner.run(_args(tmp_path, deadline_seconds=deadline), manifest_path=manifest_path,
                   dependency_loader=lambda: pytest.fail("dependencies must not load"))


def test_features_use_model_dtype_and_prompt_tokens_are_not_decoded():
    calls = []

    class Features(dict):
        def to(self, device, dtype):
            calls.append((device, dtype))
            return self

    processor = SimpleNamespace(
        apply_transcription_request=lambda **_: Features(input_ids=np.array([[1, 2]])),
        decode=lambda tokens, **_: calls.append(tokens.tolist()) or ["pause"])
    model = SimpleNamespace(device="cuda:0", dtype="float16",
                            generate=lambda **_: np.array([[1, 2, 9]]))
    torch = SimpleNamespace(inference_mode=nullcontext)
    assert runner._infer(torch, processor, model, np.zeros(16), None, 128) == "pause"
    assert calls == [("cuda:0", "float16"), [[9]]]
