"""Check retained interfaces without constructing a device or loading a model."""
import ast
import inspect
import textwrap

import numpy as np
import pytest

from assistant.audio import Microphone, SileroVAD, Endpointer
from assistant.wake import WakeDetector
from assistant.stt import Transcriber
from assistant.echo import SpeakerReference, EchoCanceller
from assistant.spotify import Spotify
from assistant.feedback import Feedback

PCM = np.zeros(512, dtype=np.int16)


@pytest.mark.parametrize("cls,args,kwargs,methods,attributes", [
    (Microphone, (), dict(sample_rate=16000, frame_length=512, ring_seconds=30,
                         device="fake", capture_rate=48000, exclusive=False),
     [("__enter__", (), {}), ("__exit__", (None, None, None), {}),
      ("read", (), {"timeout_s": 3.0}), ("flush", (), {})],
     ("last_clean", "echo", "queue_overflow_count", "_callback_status_count", "intentional_discard_count")),
    (SileroVAD, (16000,), {}, [("reset", (), {}), ("__call__", (PCM,), {})], ()),
    (Endpointer, (object(),), dict(sample_rate=16000, frame_length=512, max_seconds=8),
     [("record", (object(),), {"after_prompt": False})], ()),
    (WakeDetector, (object(),), {}, [("process", (PCM,), {}), ("reset", (), {}),
                                  ("close", (), {})], ("last_score",)),
    (Transcriber, ("faster-whisper/fake", 16000), {},
     [("transcribe", (PCM,), {})], ("_backend",)),
    (SpeakerReference, ("fake",), {}, [("start", (), {}), ("stop", (), {}),
                                     ("take_until", (1.0,), {}), ("alive", (), {})], ()),
    (EchoCanceller, (object(),), {}, [("process", (PCM, 1.0), {})], ()),
    (Spotify, (), {}, [("play_track", ("Halo by Beyoncé",), {}), ("pause", (), {}),
                      ("resolve_track", ("Halo",), {}), ("play_candidate", (object(),), {}),
                      ("play_resume", (), {}), ("next", (), {})], ()),
    (Feedback, (object(),), {}, [("say", ("Paused.",), {"force": True, "blocking": True}),
                               ("tick", (), {})], ()),
])
def test_every_retained_interface_matches_actual_usage(cls, args, kwargs, methods, attributes):
    inspect.signature(cls).bind(*args, **kwargs)
    for name, args, kwargs in methods:
        method = inspect.getattr_static(cls, name)
        assert callable(method)
        inspect.signature(method).bind(None, *args, **kwargs)
    # Instance attributes are assigned in __init__, rather than on the class.
    # Inspect their declarations; never run a device constructor to discover them.
    tree = ast.parse(textwrap.dedent(inspect.getsource(cls.__init__)))
    declared = {node.attr for node in ast.walk(tree)
                if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store)
                and isinstance(node.value, ast.Name) and node.value.id == "self"}
    assert set(attributes) <= declared


def test_timestamped_command_decode_matches_the_retained_backend_interface():
    from faster_whisper import WhisperModel
    from assistant.stt import _FasterWhisperBackend
    inspect.signature(WhisperModel.transcribe).bind(
        None, PCM.astype(np.float32), language="en", beam_size=1, best_of=1,
        temperature=0.0, hotwords=None, condition_on_previous_text=False,
        vad_filter=False, word_timestamps=True, without_timestamps=False,
        no_speech_threshold=0.6, compression_ratio_threshold=2.4)
    tree = ast.parse(textwrap.dedent(inspect.getsource(_FasterWhisperBackend.__init__)))
    assert any(isinstance(node, ast.Attribute) and node.attr == "_model"
               and isinstance(node.ctx, ast.Store) for node in ast.walk(tree))
