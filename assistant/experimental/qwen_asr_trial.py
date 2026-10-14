"""One-shot Qwen3-ASR benchmark for the three user-labeled command clips."""
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / "assistant" / "logs" / "real-commands-20260910.jsonl"

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-ASR-0.6B")
    ap.add_argument("--cache", default=str(ROOT / "assistant" / "models" / "qwen-cache"))
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    ap.add_argument("--output", help="Create a new JSONL report; never overwrite")
    args = ap.parse_args()
    cache = Path(args.cache); cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache)); os.environ.setdefault("TRANSFORMERS_CACHE", str(cache))
    from qwen_asr import Qwen3ASRModel
    import torch
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto": device = "cpu"
    if device == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    report = open(args.output, "x", encoding="utf-8") if args.output else None
    def emit(row):
        line = json.dumps(row, ensure_ascii=False)
        print(line, flush=True)
        if report: report.write(line + "\n"); report.flush()
    emit({"torch": torch.__version__, "cuda": torch.cuda.is_available(), "device": device, "model": args.model})
    t0 = time.perf_counter()
    model = Qwen3ASRModel.from_pretrained(args.model, dtype=dtype, device_map=device, max_inference_batch_size=1, max_new_tokens=128, cache_dir=str(cache), local_files_only=True)
    emit({"loaded_s": round(time.perf_counter()-t0, 2)})
    rows=[]
    for line in LOG.read_text(encoding="utf-8").splitlines():
        item=json.loads(line); wav=ROOT / "assistant" / "logs" / item["audio_path"]
        t=time.perf_counter(); out=model.transcribe(audio=str(wav), language=None); elapsed=time.perf_counter()-t
        text = out[0].text if isinstance(out, list) else getattr(out, "text", str(out))
        row={"audio":str(wav),"expected":item.get("expected_transcript"),"actual":text,"latency_s":round(elapsed,2)}
        rows.append(row); emit(row)
    emit({"results":rows})
    if report: report.close()
if __name__ == "__main__": main()
