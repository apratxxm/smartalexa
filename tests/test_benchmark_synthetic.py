import json

import numpy as np

from assistant import benchmark_synthetic as benchmark


def test_models_receive_identical_pcm(monkeypatch, capsys):
    from piper.voice import PiperVoice
    monkeypatch.setattr(PiperVoice, 'load', lambda _: object())
    monkeypatch.setattr(benchmark, 'synthesize', lambda voice, text: np.array([
        benchmark.COMMANDS.index(text)], dtype=np.int16))
    seen = {}
    class Transcriber:
        def __init__(self, model):
            self.model = model
            seen[model] = []
        def set_vocabulary(self, hints):
            assert hints == []
        def transcribe(self, pcm):
            seen[self.model].append(id(pcm))
            return benchmark.COMMANDS[int(pcm[0])], 1
    monkeypatch.setattr(benchmark, 'Transcriber', Transcriber)
    benchmark.main(['--models', 'first', 'second'])
    assert seen['first'] == seen['second']
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    summaries = [row['summary'] for row in rows if 'summary' in row]
    assert len(summaries) == 2
    assert all(s['correct_intents'] == len(benchmark.COMMANDS) for s in summaries)
