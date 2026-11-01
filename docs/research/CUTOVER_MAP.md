# Cutover map: legacy runtime vs core runtime (phase 5 input)

Snapshot 2026-10-27. Static AST audit plus one dynamic importtime check. Read-only over the project. The tree was still changing during the audit (new `core/adapters/*` and `wakebench/train/*`); re-run before phase 5.

## Method and roots

- Graph: AST walk of 128 `assistant.*` modules (`assistant/models/**` vendored cache excluded). Edge kinds: `import`, `from`, `from-submodule`, `importlib` literal, module-name string literal.
- Manual edges (verified by reading source, invisible to AST):
  - `core/__main__.py` selfcheck: `importlib.import_module("assistant." + name)` over `audio, echo, wake, stt, spotify, feedback, conversation_brain, core.evaluation`.
  - `computer_tools.py:132`: subprocess `-m assistant.spotify_jam`.
- Core roots (runtime=core): `assistant.core.*`, `assistant.wakebench.*`, `assistant.supervise`.
- Legacy-only edges excluded from the core closure: `supervise -> assistant.main` (`supervise.py:183`, `runtime.runtime=legacy` branch) and `supervise -> experimental.wake_train.review_server` (`supervise.py:83`, legacy branch only).
- Non-literal `importlib` calls: none found.
- Dynamic check: `A:\pisha\memoir\.venv\Scripts\python.exe -B -X importtime -c "import assistant.core.__main__, assistant.wakebench.bench, assistant.supervise"`
  - rc 0, no devices opened. 514 import records, **505 unique modules** (31 `assistant.*`).
  - `assistant.main` and `assistant.experimental.*` are not loaded at import.
- Live runtime selection: `assistant/config.toml:18` is `runtime = "legacy"`. `supervise` picks `assistant.core` or `assistant.main` from that key.

Legend: C = reached from core roots, M = reached from `assistant.main`. Chains are shortest import paths.

## A. Classification of top-level `assistant/*.py` (49 modules)

Counts: KEEP 12 · EXTRACT 13 · KEEP-FUTURE 5 · DELETE-CANDIDATE 19.

| Module | Class | Lines | Evidence |
|---|---|---|---|
| assistant (pkg init) | KEEP | 3 | package init, imported by core |
| catalogue | KEEP (transitive) | 1210 | core < spotify < catalogue; also M |
| config | KEEP | 58 | core root loader; 17 importers |
| diag | KEEP (transitive) | 162 | core < evaluation < experimental.brain_eval < brain < diag |
| ducking | KEEP (transitive) | 484 | core < feedback < ducking |
| global_commands | KEEP (transitive) | 555 | core < provider < conversation_brain < global_router < global_commands |
| global_router | KEEP (transitive) | 664 | core < provider < conversation_brain < global_router |
| model_runtime | KEEP (transitive) | 115 | core < provider < conversation_brain < model_runtime |
| process_lifetime | KEEP | 106 | imported by supervise |
| resolution | KEEP (transitive) | 255 | core < executor < spotify < resolution |
| supervise | KEEP (edit) | 223 | core root; legacy branch (L83, L183) must be removed; main imports `_ollama_ready` (main.py:1751) |
| telemetry | KEEP (open) | 1039 | core < wakebench.train.cli < experimental.wake_train.machine_proposals < review_server < telemetry |
| codex_bridge | KEEP-FUTURE | 1286 | M only (global_assistant); brief override |
| computer_tools | KEEP-FUTURE | 248 | now C: core/adapters/computer.py:13 and jam.py:4 import it; imports external_apps, intent |
| external_apps | KEEP-FUTURE | 60 | C via spotify and computer_tools |
| knowledge_retrieval | KEEP-FUTURE | 232 | no importer; tests/test_knowledge_retrieval.py only; adapters/knowledge.py does not import it |
| spotify_jam | KEEP-FUTURE | 631 | C via computer_tools subprocess (computer_tools.py:132) |
| audio | EXTRACT | 875 | core/capture: Endpointer, Microphone, SileroVAD; wakebench/capture: device_candidates |
| brain | EXTRACT | 963 | eval-only: core/evaluation < experimental/brain_eval: Brain, decide |
| conversation | EXTRACT | 313 | core/engine: ConversationCoordinator |
| conversation_brain | EXTRACT | 559 | core/provider: 5 private helpers (table B) |
| conversation_contracts | EXTRACT | 484 | core authority/engine/evaluation/executor/fallback/provider and adapters/base: 7 names (table B) |
| echo | EXTRACT | 231 | core/capture: EchoCanceller, SpeakerReference; wakebench/bench: EchoCanceller |
| feedback | EXTRACT | 319 | core/__main__.run: Feedback (pulls ducking; piper voice file) |
| intent | EXTRACT | 661 | core/fallback.py:9 `intent.parse` on live admit path; evaluation: Action; computer_tools: capability_domain |
| local_media | EXTRACT | 403 | core/playback: `_identity`, `_run`, `_verified_transport`, `status`, `snapshot` |
| runtime | EXTRACT | 278 | core/__main__ and wakebench/record: `single_instance` |
| spotify | EXTRACT | 2120 | core/executor: Spotify; core/playback: 5 names (table B) |
| stt | EXTRACT | 388 | core/transcription: Transcriber |
| wake | EXTRACT | 337 | core/capture: WakeDetector; wakebench/bench: CHUNK, WakeDetector; wakebench/session: FRAME_LENGTH, SAMPLE_RATE |
| async_listener | DELETE-CANDIDATE | 1861 | M only |
| async_runtime | DELETE-CANDIDATE | 239 | M only |
| audio_jobs | DELETE-CANDIDATE | 478 | M only |
| benchmark_intent | DELETE-CANDIDATE | 72 | orphan (no importer, no test); CLI use unverified |
| benchmark_synthetic | DELETE-CANDIDATE | 98 | orphan; tests/test_benchmark_synthetic.py only |
| capture_requests | DELETE-CANDIDATE | 474 | M only |
| concise_brain | DELETE-CANDIDATE | 318 | M only (main.py:719) |
| desktop_controls | DELETE-CANDIDATE | 107 | M only (main.py:412) |
| diagnostic_archive | DELETE-CANDIDATE | 223 | M only |
| enroll | DELETE-CANDIDATE | 169 | orphan (no importer, no test); CLI use unverified |
| evaluate | DELETE-CANDIDATE | 105 | orphan; tests/test_evaluate.py (also imports stt, EXTRACT) |
| foreground_capture | DELETE-CANDIDATE | 275 | M only |
| global_assistant | DELETE-CANDIDATE | 1055 | M only (main.py:297, 402); imports codex_bridge |
| health | DELETE-CANDIDATE | 288 | M only (main.py:21); docstring cites assistant.report |
| intent_llm | DELETE-CANDIDATE | 140 | M only (main.py:3561) |
| listener | DELETE-CANDIDATE | 735 | M only (main.py:38) |
| main | DELETE-CANDIDATE | 4255 | legacy entry; supervise selects it by string |
| power | DELETE-CANDIDATE | 253 | M only (main.py:42) |
| report | DELETE-CANDIDATE | 417 | orphan CLI (`python -m assistant.report`); reads telemetry.jsonl, memoir.log, crash.log |

## B. EXTRACT detail: what core still needs

| Source module | Core consumers (file: names) | Remainder that stays legacy |
|---|---|---|
| conversation_brain | core/provider.py:16: `_NoRedirectHandler`, `_decode_wire_decision`, `_loads_unique`, `_ollama_schema_for`, `build_messages` | ConversationBrain class and rest; drops global_router, global_commands, model_runtime from core if moved |
| conversation_contracts | core/*: `Step`, `step_to_intent`, `Decision`, `Transcript`, `ToolResult`, `validate_decision`; adapters/base.py:7: `CAPABILITIES` | Most of 22 defs |
| spotify | core/executor.py:76: `Spotify`; core/playback.py:5: `PlaybackVerificationError`, `Spotify`, `_RateLimitGuard`, `_matches_local_candidate`, `open_external_target` | 2120-line module minus those; drags catalogue, resolution, external_apps |
| intent | core/fallback.py:9 `intent.parse(text)` (module import, live path); evaluation: `Action`; computer_tools: `capability_domain` | Legacy parser internals |
| wake | core/capture: `WakeDetector`; wakebench: `CHUNK`, `WakeDetector`, `FRAME_LENGTH`, `SAMPLE_RATE` | Wake-word training/eval code (owned by wake lane) |

Second-order: `catalogue`, `resolution`, `diag`, `ducking`, `global_*`, `model_runtime` are KEEP only transitively. Re-run the graph after the EXTRACT moves; they may drop out (not verified).

## C. Experimental modules the core closure reaches (not classified, must stay until moved)

- `experimental/brain_eval.py` (core/evaluation planning eval).
- `experimental/wake_train/{active_learning, candidate_training, collection, machine_proposals, review_server, reviewed_training}.py` (imported by `wakebench/train/{data, fit, cli}.py`).
- `experimental/wake_train/review_server.py` is also spawned by legacy `supervise` (L83).
- Consequence: `experimental/` cannot be deleted wholesale in phase 5.

## D. Tests

- Tests that cover only DELETE-CANDIDATE modules (11): test_async_listener.py, test_audio_jobs.py, test_capture_requests.py, test_desktop_controls.py, test_diagnostic_archive.py, test_foreground_capture.py, test_intent_llm_holdout.py, test_listener.py, test_power.py, test_report.py, test_benchmark_synthetic.py.
- Tests importing `assistant.main` (9, plus string ref in `tests/core/test_turns.py:127`): test_async_runtime, test_conversation_flow, test_diagnostics, test_global_runtime, test_main_loop, test_radio_queue, test_resolution_diagnostics, test_volume_execution, core/test_turns. The core/test_turns entry must be edited at cutover.
- Tests touching EXTRACT modules: 43 files. They need re-pointing, not deletion.

## E. Config keys

- 102 keys parsed from `assistant/config.toml`.
- 55 keys read only by legacy-set modules (main, listener, async_*, global_assistant, intent_llm, ...): `listener.*` 19, `codex.*` 9, `capture.*` 7, `spotify.radio_*`/`catalogue_auto_refresh` 3, `ducking.*` 3, `hotkey.*` 3, `telemetry.*` 3, `audio.paused_*` 2, `intent.*` 2, `conversation.*` 2, `runtime.keep_awake_on_ac`, `wake.acoustic_activation_threshold`.
- 13 keys have no exact-string reference (`core.brain.*` read via nested get; `listener.aliases.*`). Not classified.
- Method limit: exact dotted-string match only.

## F. Data, logs, models

| Path | Size | Users | Class |
|---|---|---|---|
| assistant/logs/telemetry.jsonl | 168.0 MB | telemetry (KEEP), report (DELETE) | KEEP while telemetry is kept |
| assistant/logs/memoir.log, .1, .2, .3 | 19.0 MB | report reads; diag/telemetry write (core writer not found) | legacy-era, DELETE-CANDIDATE after zip |
| assistant/logs/crash.log | 0.38 MB | main, report | legacy-only |
| assistant/logs/supervise.log | 22 KB | supervise | KEEP |
| assistant/logs/.vocabulary.json | 1.8 KB | main.py:1374 | legacy-only |
| assistant/logs/.catalogue.json, .radio_history.json, .device.json, .track_cache.json, .spotify_backoff.json | 0.6 MB total | catalogue, spotify | KEEP with EXTRACT pieces |
| assistant/logs/.codex-last-created.json | 258 B | codex_bridge | KEEP-FUTURE |
| assistant/logs/aec/ | 16.8 MB, 8 files | experimental AEC/KWS tools only | not runtime; retention decision |
| assistant/models/en_US-lessac-medium.onnx (+ .json) | 63.2 MB | feedback.piper_voice | KEEP |
| assistant/models/silero_vad.onnx | 2.3 MB | audio.py | KEEP |
| assistant/models/wake/ | 8.9 MB | wake, wakebench | KEEP |
| assistant/models/hf-cache/ | 3.24 GB | config `[stt]` faster-whisper path | KEEP |
| assistant/models/qwen-cache/ | 3.76 GB | experimental and intent_llm, benchmark_intent | DELETE-CANDIDATE after zip (confirm) |
| assistant/models/qwen-hf-cache/ | 1.58 GB | experimental/qwen_asr_trial, tests/test_qwen_hf_eval | DELETE-CANDIDATE after zip (confirm) |
| assistant/models/torch-wheels/ | 2.75 GB | no code reference | install leftover; open decision |
| .assistant-baselines/ | 2.3 MB | no code reference | open decision |
| assistant/config.toml.live-slice-backup | 13.7 KB | no code reference | open decision |

## G. Launch and scheduled-task references

- Scheduled task "Memoir Assistant": State Disabled. Executes `A:\pisha\memoir\.venv\Scripts\pythonw.exe -m assistant.supervise`, WD project root. No direct `assistant.main` reference. Cutover switch is config `runtime.runtime` (currently legacy).
- `scripts/install_task.ps1` registers the same task.
- Startup folder: no Memoir shortcut. HKCU Run: no Memoir entry.
- `assistant.main` string references in tests and docs only (plus supervise).

## H. Zip contents (phase 5 backup must contain these to restore legacy)

Code:
- `assistant/main.py`
- `assistant/supervise.py` (pre-cutover copy)
- `assistant/config.toml` (pre-cutover, `runtime = "legacy"`)
- `assistant/__init__.py`, `assistant/config.py`, `assistant/requirements.txt`
- DELETE set (18): `assistant/{async_listener, async_runtime, audio_jobs, benchmark_intent, benchmark_synthetic, capture_requests, concise_brain, desktop_controls, diagnostic_archive, enroll, evaluate, foreground_capture, global_assistant, health, intent_llm, listener, power, report}.py`
- EXTRACT originals (full, pre-move): `assistant/{audio, brain, conversation, conversation_brain, conversation_contracts, echo, feedback, intent, local_media, runtime, spotify, stt, wake}.py`
- KEEP-set originals for diffing: `assistant/{catalogue, diag, ducking, global_commands, global_router, model_runtime, process_lifetime, resolution, telemetry, codex_bridge, computer_tools, external_apps, spotify_jam, knowledge_retrieval}.py`
- Experimental used by legacy: `assistant/experimental/wake_train/review_server.py`
- `assistant/core/**` (pre-cutover tree), `assistant/wakebench/**`, `assistant/experimental/**` (pre-cutover tree; moves are not reverted otherwise)
- Tests: `tests/**`, `pytest.ini`
- Data: `assistant/logs/{telemetry.jsonl, memoir.log, memoir.log.1, memoir.log.2, memoir.log.3, crash.log, .vocabulary.json}`
- Models only if phase 5 removes them: `assistant/models/qwen-cache/**`, `assistant/models/qwen-hf-cache/**`
- Scheduler: exported XML of task "Memoir Assistant" (Export-ScheduledTask)
- Not in zip without user decision: `.env` (secrets), `.venv*` (rebuild from requirements)

## I. Restore test (run in a scratch copy, not the project)

1. Copy the zip to a new empty scratch folder outside `A:\pisha\memoir`.
2. Extract with a fresh destination; verify every entry's SHA-256 against the zip manifest.
3. Compare restored `main.py`, `supervise.py`, `config.toml` hashes with the pre-cutover commit values.
4. Build a scratch project: copy the current tree excluding `.venv*`, `models/**` and `logs/**`; overlay the restored zip contents.
5. In the scratch project run `.venv\Scripts\python.exe -B -c "import assistant.main"` (import only; no devices).
6. Run `python -B -m pytest tests/test_main_loop.py tests/test_report.py tests/test_conversation_flow.py -q`; record pass/fail.
7. Set `runtime = "legacy"` and verify `assistant.supervise.resolve_runtime_module(load())` returns `assistant.main` (no process start).
8. Verify the duplicate-start lock is refused by the legacy path (`single_instance` guard) without touching devices.
9. Record zip hash, file count and test results in the phase 5 log.
10. Delete the scratch copy only after the user confirms.

## J. Open decisions (user or Sol gate)

1. Intent parser on the live core path (fallback.py `intent.parse`). Architecture and shared-contract change. Gate: Sol design review. Recommendation: extract `parse`, `Action`, `Intent`, `capability_domain` into core as an explicit proposal module.
2. `spotify.py` (2120 lines) is a core dependency. Extract only the needed pieces, or keep the module as KEEP. Gate: Sol design review (shared contract). Recommendation: extract the 6 names and re-run the graph.
3. Planning eval (`core/evaluation` to `experimental/brain_eval` to `brain`, `diag`). Keep the eval harness or drop it. Recommendation: drop unless eval is wanted.
4. `telemetry` stays KEEP only because `wakebench/train` imports `experimental/wake_train`, which imports `review_server`, which imports telemetry. Wake lane (`memoir-terra-wake`) to decide, since wake.py extraction also touches its files.
5. Adapters `computer.py` and `jam.py` now import legacy `computer_tools`, which imports `intent`. Keeping `computer_tools` makes `intent` a permanent dependency. Adapter lane to decide.
6. Retention: `torch-wheels` (2.75 GB), `qwen*` caches, `aec` logs, `.assistant-baselines`, `config.toml.live-slice-backup`.
7. `.env` in the zip: contains secrets. Include encrypted, exclude, or keep out of the backup.
8. Orphans `enroll`, `evaluate`, `benchmark_intent`: confirm they are not used as manual CLIs before deletion.

## K. Limitations

- Static analysis plus one importtime run. No test suite run.
- Dynamic loads are literal-only. Config keys are exact-match only.
- The tree changed during the audit. Counts reflect the state at the last run. Re-run `graph -> classify -> importtime` before phase 5.
- The restore test in section I was not executed.
