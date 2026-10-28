# Wake benchmark (PowerShell, project root)
1. Use `.venv\Scripts\python`; close Memoir before recording. Keep speaker volume at your normal level.
2. Preview devices: `.venv\Scripts\python -m assistant.wakebench.record --dry-run` (opens no streams).
3. Day A: repeat all near/mid/far × music off/on combinations, setting music yourself: `.venv\Scripts\python -m assistant.wakebench.record --distance near --music off`
4. Each session has 30 prompts. Say the displayed command naturally just after its chirp; do not echo the beep.
5. Press `x` during a garbled/skipped attempt, `p` to pause/resume, or `q` to abort. Capture continues during pauses.
6. Record room negatives: `.venv\Scripts\python -m assistant.wakebench.record --mode negative --minutes 60 --music on`
7. During negatives, use the room normally without deliberately saying Alexa; quiet anchor chirps play every minute.
8. Save Day A session directory names for tuning. On another day repeat steps 3–7 for held-out Day B.
9. WAVs, per-block status sidecars, cue waveform and manifests live in `assistant/wakebench/recordings/<id>/`.
10. Finalization runs automatically; rerun offline with `python -m assistant.wakebench.record --finalize <session-dir>`.
11. Compare: `.venv\Scripts\python -m assistant.wakebench.bench --tune <A-dir> <A-negative-dir> --eval <B-dir> <B-negative-dir>`
12. Add later models with repeated `--model <path.onnx>`. Do not reuse any A session as B.
13. Read Markdown + JSON in `assistant/wakebench/reports/`; choose using held-out recall and false wakes/exposure.
14. Invalid attempts are counted and excluded; offline scores do not establish live-device acceptance.
