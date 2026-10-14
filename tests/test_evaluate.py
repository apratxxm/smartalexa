import json, wave
from array import array
from assistant.evaluate import load_manifest, evaluate

def wav(p):
    with wave.open(str(p),'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(array('h',[0]*160))

def test_offline_evaluate(tmp_path):
    p=tmp_path/'a.wav'; wav(p); m=tmp_path/'m.jsonl'; m.write_text(json.dumps({'audio_path':'a.wav','expected_action':'PAUSE','expected_query':None})+'\n')
    class T:
        def transcribe(self, pcm): return ('pause', 7)
    r=evaluate(load_manifest(str(m)),T()); assert r[0]['correct_action'] and r[0]['latency_ms']==7


def test_absolute_and_relative_volume_are_not_equivalent(tmp_path):
    p = tmp_path/'a.wav'; wav(p)
    row = {'_path': p, 'audio_path': 'a.wav', 'expected_action': 'VOLUME',
           'expected_value': 50, 'expected_volume_mode': 'relative_percent'}
    class T:
        def transcribe(self, pcm): return 'volume 50', 1
    result = evaluate([row], T())[0]
    assert result['correct_value'] is True
    assert result['correct_volume_mode'] is False and result['correct_intent'] is False

def test_manifest_rejects_bad_format(tmp_path):
    p=tmp_path/'a.wav'; wav(p); m=tmp_path/'m.jsonl'; m.write_text(json.dumps({'audio_path':'a.wav'})+'\n')
    try: load_manifest(str(m)); assert False
    except ValueError: pass


def test_cli_applies_hints_and_reports_unlabeled_query(tmp_path, monkeypatch, capsys):
    from assistant import stt
    from assistant.evaluate import main
    p = tmp_path / 'a.wav'
    wav(p)
    manifest = tmp_path / 'manifest.jsonl'
    manifest.write_text(json.dumps({'audio_path': 'a.wav', 'expected_action': 'PAUSE'}))
    applied = []
    class T:
        def __init__(self, model):
            pass
        def set_vocabulary(self, hints):
            applied.extend(hints)
        def transcribe(self, pcm):
            return 'pause', 5
    monkeypatch.setattr(stt, 'Transcriber', T)
    assert main([str(manifest), '--hints', '["Aerosmith"]']) == 0
    assert applied == ['Aerosmith']
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert rows[0]['exact_query'] is None
    assert rows[-1]['summary']['correct_actions'] == 1


def test_wrong_volume_is_not_a_correct_intent(tmp_path):
    p = tmp_path / 'a.wav'
    wav(p)
    manifest = tmp_path / 'manifest.jsonl'
    manifest.write_text(json.dumps({'audio_path': 'a.wav', 'expected_action': 'VOLUME', 'expected_value': 40}))
    class T:
        def transcribe(self, pcm):
            return 'set volume to 14', 5
    result = evaluate(load_manifest(str(manifest)), T())[0]
    assert result['correct_action']
    assert result['correct_value'] is False
    assert result['correct_intent'] is False


def test_lost_personal_playlist_scope_is_not_a_correct_intent(tmp_path):
    wav(tmp_path / 'a.wav')
    manifest = tmp_path / 'manifest.jsonl'
    manifest.write_text(json.dumps({'audio_path': 'a.wav', 'expected_action': 'PLAY_PLAYLIST',
        'expected_query': 'focus', 'expected_library_only': True}))
    class T:
        def transcribe(self, pcm):
            return 'play focus playlist', 5
    result = evaluate(load_manifest(str(manifest)), T())[0]
    assert result['correct_action'] and result['exact_query']
    assert result['correct_library_scope'] is False
    assert result['correct_intent'] is False
