"""Controlled synthetic STT comparison; not a real-room acceptance test.

Synthesizes each command once and reuses the exact PCM for every model/hint
condition. Does not play audio, record a microphone, or issue Spotify actions.
"""
from __future__ import annotations

import argparse
import gc
import io
import json
import statistics
import wave
from contextlib import nullcontext

import numpy as np
import soxr

from .config import load
from .intent import parse
from .stt import Transcriber


COMMANDS = [
    "pause", "play music", "skip", "go back", "turn it up",
    "set volume to forty", "play Dream On by Aerosmith",
    "play RED by Seedhe Maut", "play Bohemian Rhapsody by Queen",
    "play Don't Stop Me Now by Queen", "play Next to Me by Imagine Dragons",
    "play my focus playlist",
]


def synthesize(voice, text):
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as output:
        voice.synthesize_wav(text, output)
    buffer.seek(0)
    with wave.open(buffer, 'rb') as audio:
        rate = audio.getframerate()
        pcm = np.frombuffer(audio.readframes(audio.getnframes()), dtype=np.int16)
    return np.clip(soxr.resample(pcm.astype(np.float32), rate, 16000), -32768, 32767).astype(np.int16)


def compare(models, compare_spotify_hints=False):
    from piper.voice import PiperVoice
    cfg = load()
    voice = PiperVoice.load(str(cfg.path('feedback.piper_voice')))
    clips = [(text, synthesize(voice, text), parse(text)) for text in COMMANDS]
    del voice
    conditions = [('no_hints', [])]
    if compare_spotify_hints:
        from .spotify import Spotify
        conditions.append(('spotify_hints', Spotify().vocabulary()))
    yield {'kind': 'benchmark', 'source': 'synthetic_piper',
           'clips': len(clips), 'same_pcm_across_conditions': True}
    for model in models:
        transcriber = Transcriber(model)
        runtime = getattr(getattr(getattr(transcriber, '_backend', None), '_model', None), 'model', None)
        device = getattr(runtime, 'device', 'unknown')
        for condition, hints in conditions:
            transcriber.set_vocabulary(hints)
            successes, latencies = [], []
            for spoken, pcm, expected in clips:
                heard, latency = transcriber.transcribe(pcm)
                actual = parse(heard)
                correct = (actual.action == expected.action and actual.query.casefold() == expected.query.casefold()
                           and actual.value == expected.value and actual.library_only == expected.library_only
                           and actual.volume_mode == expected.volume_mode)
                successes.append(correct)
                latencies.append(latency)
                yield {'model': model, 'condition': condition, 'spoken': spoken,
                       'heard': heard, 'correct_intent': correct, 'latency_ms': latency}
            yield {'summary': {'model': model, 'condition': condition, 'device': device,
                  'correct_intents': sum(successes), 'clips': len(clips),
                  'median_latency_ms': statistics.median(latencies), 'max_latency_ms': max(latencies)}}
        del runtime
        del transcriber
        gc.collect()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', required=True)
    parser.add_argument('--compare-spotify-hints', action='store_true')
    parser.add_argument('--output', help='New JSONL report file; never overwrites an existing report')
    args = parser.parse_args(argv)
    with (open(args.output, 'x', encoding='utf-8') if args.output else nullcontext(None)) as report:
        for result in compare(args.models, args.compare_spotify_hints):
            line = json.dumps(result, ensure_ascii=False)
            print(line, flush=True)
            if report is not None:
                report.write(line + '\n')
                report.flush()


if __name__ == '__main__':
    main()
