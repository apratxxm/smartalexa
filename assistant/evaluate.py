"""Offline JSONL evaluation harness for recorded command clips."""
from __future__ import annotations
import argparse, json, math, statistics, sys, time, wave
from pathlib import Path
import numpy as np
from .intent import parse

def _wav(path: Path) -> np.ndarray:
    try:
        with wave.open(str(path), "rb") as w:
            if (w.getnchannels(), w.getsampwidth(), w.getframerate()) != (1, 2, 16000):
                raise ValueError("WAV must be mono, 16-bit PCM, 16000 Hz")
            data = w.readframes(w.getnframes())
            if not data: raise ValueError("WAV is empty")
            if len(data) != w.getnframes() * 2:
                raise ValueError("WAV is truncated")
            return np.frombuffer(data, dtype=np.int16)
    except (wave.Error, OSError) as e:
        raise ValueError(f"invalid WAV {path}: {e}") from e

def load_manifest(path: str) -> list[dict]:
    base = Path(path).resolve().parent; out = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip(): continue
        try: row = json.loads(line)
        except json.JSONDecodeError as e: raise ValueError(f"line {n}: invalid JSON") from e
        if not isinstance(row, dict) or not isinstance(row.get("audio_path"), str) or not isinstance(row.get("expected_action"), str) or not row.get("audio_path") or not row.get("expected_action"):
            raise ValueError(f"line {n}: audio_path and expected_action are required")
        from .intent import Action
        if row["expected_action"] not in {a.value for a in Action}:
            raise ValueError(f"line {n}: invalid expected_action {row['expected_action']!r}")
        if "expected_query" in row and row["expected_query"] is not None and not isinstance(row["expected_query"], str):
            raise ValueError(f"line {n}: expected_query must be a string")
        if row.get("expected_transcript") is not None and not isinstance(row["expected_transcript"], str):
            raise ValueError(f"line {n}: expected_transcript must be a string")
        if row.get("expected_value") is not None and type(row["expected_value"]) is not int:
            raise ValueError(f"line {n}: expected_value must be an integer")
        if row.get('expected_library_only') is not None and type(row['expected_library_only']) is not bool:
            raise ValueError(f"line {n}: expected_library_only must be a boolean")
        if row.get('expected_volume_mode') is not None and row['expected_volume_mode'] not in ('absolute', 'relative_percent', 'relative_points'):
            raise ValueError(f"line {n}: invalid expected_volume_mode")
        if row['expected_action'] in ('PLAY_TRACK', 'PLAY_PLAYLIST') and not row.get('expected_query'):
            raise ValueError(f"line {n}: content commands require expected_query")
        if row['expected_action'] == 'VOLUME' and row.get('expected_value') is None and row.get('expected_query') not in ('up', 'down'):
            raise ValueError(f"line {n}: volume commands require expected_value or up/down expected_query")
        row["_path"] = (base / row["audio_path"]).resolve()
        _wav(row["_path"])
        out.append(row)
    if not out: raise ValueError("manifest contains no clips")
    return out

def _wer(ref, hyp):
    a, b = ref.casefold().split(), hyp.casefold().split(); d=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        nd=[i]
        for j,y in enumerate(b,1): nd.append(min(nd[-1]+1,d[j]+1,d[j-1]+(x!=y)))
        d=nd
    return d[-1] / max(1,len(a))

def evaluate(manifest, transcriber):
    results=[]
    for row in manifest:
        pcm=_wav(row["_path"]); t=time.perf_counter(); got=transcriber.transcribe(pcm)
        text, latency = got if isinstance(got, tuple) else (got, int((time.perf_counter()-t)*1000))
        intent=parse(text); action=getattr(intent.action, "value", str(intent.action))
        q=row.get("expected_query"); nq=getattr(intent, "query", None)
        has_query=q is not None
        result={"audio_path":row["audio_path"],"transcript":text,"action":action,
                "expected_action":row["expected_action"],"correct_action":action==row["expected_action"],
                "exact_query": (nq or "").casefold().strip()==q.casefold().strip() if has_query else None,"latency_ms":latency}
        value = row.get('expected_value')
        result['value'] = intent.value
        result['correct_value'] = intent.value == value if value is not None else None
        personal = row.get('expected_library_only')
        result['library_only'] = intent.library_only
        result['correct_library_scope'] = intent.library_only == personal if personal is not None else None
        mode = row.get('expected_volume_mode')
        result['volume_mode'] = intent.volume_mode
        result['correct_volume_mode'] = intent.volume_mode == mode if mode is not None else None
        result['correct_intent'] = (result['correct_action'] and result['exact_query'] is not False
                                    and result['correct_value'] is not False
                                    and result['correct_library_scope'] is not False
                                    and result['correct_volume_mode'] is not False)
        if row.get("expected_transcript") is not None: result["wer"]=_wer(row["expected_transcript"],text)
        results.append(result)
    return results

def main(argv=None):
    ap=argparse.ArgumentParser(); ap.add_argument("manifest"); ap.add_argument("--model",default="faster-whisper/small.en"); ap.add_argument("--hints", help="JSON list"); ap.add_argument("--dry-run",action="store_true"); args=ap.parse_args(argv)
    try: manifest=load_manifest(args.manifest)
    except (ValueError, OSError) as e: ap.error(str(e))
    if args.dry_run: print(json.dumps({"clips":len(manifest),"valid":True})); return 0
    from .stt import Transcriber
    tr=Transcriber(args.model)
    if args.hints:
        hints=json.loads(args.hints)
        if not isinstance(hints,list) or not all(isinstance(x,str) and x.strip() for x in hints): ap.error("--hints must be a JSON list of nonempty strings")
        tr.set_vocabulary(hints)
    results=evaluate(manifest,tr)
    for r in results: print(json.dumps(r,ensure_ascii=False))
    l=[r["latency_ms"] for r in results]; action=sum(r["correct_action"] for r in results); qs=[r for r in results if r["exact_query"] is not None]; query=sum(r["exact_query"] for r in qs)
    print(json.dumps({"summary":{"clips":len(results),"correct_actions":action,"action_accuracy":action/max(1,len(results)),"correct_intents":sum(r['correct_intent'] for r in results),"exact_queries":query,"query_labeled_clips":len(qs),"query_accuracy":query/len(qs) if qs else None,"empty_transcripts":sum(not r["transcript"] for r in results),"median_latency_ms":statistics.median(l) if l else 0,"p95_latency_ms":sorted(l)[max(0,math.ceil(.95*len(l))-1)] if l else 0}}))
    return 0
if __name__ == "__main__": sys.exit(main())
