"""Offline openWakeWord asset preflight; never load inference models here."""
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from assistant.config import Config
from assistant.wake import WakeDetector, WakeWordError


class FakeModel:
    instances = []

    def __init__(self, wakeword_models, inference_framework, **kwargs):
        self.wakeword_models = list(wakeword_models)
        self.inference_framework = inference_framework
        self.kwargs = kwargs
        self.models = {Path(path).stem: None
                       for path in self.wakeword_models}
        self.scores = {}
        self.instances.append(self)

    def predict(self, _chunk):
        return dict(self.scores)

    def reset(self):
        pass


def _openwakeword(monkeypatch, tmp_path, *, stock=("alexa", "hey_mycroft"),
                  missing=(), missing_metadata=()):
    FakeModel.instances.clear()
    asset_dir = tmp_path / "package-models"
    asset_dir.mkdir(exist_ok=True)
    models = {}
    for name in stock:
        models[name] = {"model_path": str(asset_dir / (name + "_v0.1.tflite"))}
    features = {
        "melspectrogram": {"model_path": str(asset_dir / "melspectrogram.tflite")},
        "embedding": {"model_path": str(asset_dir / "embedding_model.tflite")},
    }
    for name, spec in models.items():
        if name not in missing and name not in missing_metadata:
            (asset_dir / (name + "_v0.1.onnx")).write_bytes(b"fixture")
    for feature in features:
        if feature not in missing:
            (asset_dir / (feature + ("_model" if feature == "embedding" else "") + ".onnx"))\
                .write_bytes(b"fixture")
    for name in missing_metadata:
        models.pop(name, None)
    package = types.ModuleType("openwakeword")
    package.MODELS = models
    package.FEATURE_MODELS = features
    package.utils = types.SimpleNamespace(
        download_models=lambda *_args, **_kwargs: pytest.fail("runtime download attempted"))
    model_module = types.ModuleType("openwakeword.model")
    model_module.Model = FakeModel
    monkeypatch.setitem(sys.modules, "openwakeword", package)
    monkeypatch.setitem(sys.modules, "openwakeword.model", model_module)
    return asset_dir


def _config(model, **extra):
    return Config({"wake": {"model": model, **extra}})


def test_pretrained_startup_passes_explicit_onnx_only_assets(monkeypatch, tmp_path):
    asset_dir = _openwakeword(monkeypatch, tmp_path)
    detector = WakeDetector(_config(["alexa", "hey_mycroft"],
                                    threshold_low=0.1, threshold_high=0.7))

    model = FakeModel.instances[-1]
    assert model.inference_framework == "onnx"
    assert model.wakeword_models == [
        str(asset_dir / "alexa_v0.1.onnx"), str(asset_dir / "hey_mycroft_v0.1.onnx")]
    assert model.kwargs == {
        "melspec_model_path": str(asset_dir / "melspectrogram.onnx"),
        "embedding_model_path": str(asset_dir / "embedding_model.onnx"),
    }
    assert not (asset_dir / "alexa_v0.1.tflite").exists()
    model.scores = {"alexa_v0.1": 0.8, "hey_mycroft_v0.1": 0.2}
    for _ in range(3):
        tier = detector.process(np.zeros(512, dtype=np.int16))
    assert tier == "both" and detector.last_model == "alexa"


@pytest.mark.parametrize(("missing", "message"), [
    (("alexa",), "pretrained wake model 'alexa'"),
    (("melspectrogram",), "melspectrogram feature model"),
    (("embedding",), "embedding feature model"),
])
def test_missing_required_onnx_asset_fails_before_model_init(
        monkeypatch, tmp_path, missing, message):
    _openwakeword(monkeypatch, tmp_path, missing=missing)
    with pytest.raises(WakeWordError, match=message):
        WakeDetector(_config("alexa"))
    assert FakeModel.instances == []


def test_custom_only_model_still_requires_feature_assets(monkeypatch, tmp_path):
    asset_dir = _openwakeword(monkeypatch, tmp_path, stock=(), missing=("embedding",))
    custom = tmp_path / "custom.onnx"
    custom.write_bytes(b"fixture")
    with pytest.raises(WakeWordError, match="embedding feature model"):
        WakeDetector(_config(str(custom)))
    assert FakeModel.instances == []
    assert asset_dir.exists()


def test_missing_optional_personal_model_keeps_stock_startup(monkeypatch, tmp_path, caplog):
    asset_dir = _openwakeword(monkeypatch, tmp_path, stock=("alexa",))
    missing_personal = tmp_path / "not-installed-personal.onnx"
    detector = WakeDetector(_config("alexa", personal_model=str(missing_personal)))

    assert FakeModel.instances[-1].wakeword_models == [str(asset_dir / "alexa_v0.1.onnx")]
    assert detector._personal_key == ""
    assert "Personal wake model missing" in caplog.text


@pytest.mark.parametrize("model", [[], "", ["alexa", " "]])
def test_empty_model_configuration_fails_clearly_before_model_init(monkeypatch, tmp_path, model):
    _openwakeword(monkeypatch, tmp_path)
    with pytest.raises(WakeWordError, match="No wake models|Invalid empty"):
        WakeDetector(_config(model))
    assert FakeModel.instances == []


def test_missing_pretrained_metadata_fails_clearly(monkeypatch, tmp_path):
    _openwakeword(monkeypatch, tmp_path, missing_metadata=("alexa",))
    with pytest.raises(WakeWordError, match="metadata has no model path"):
        WakeDetector(_config("alexa"))
    assert FakeModel.instances == []


def test_stock_model_key_collisions_are_rejected(monkeypatch, tmp_path):
    _openwakeword(monkeypatch, tmp_path, stock=("alexa",))
    with pytest.raises(WakeWordError, match="collide on openWakeWord key"):
        WakeDetector(_config(["alexa", "alexa"]))
    assert FakeModel.instances == []
