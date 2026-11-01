# SmartAlexa capability inventory

Audit snapshot: 2026-10-24. Source/config/log inspection only; no assistant, model,
microphone, Spotify, review server, or tests were run. "Live" below means an event
was recorded by the old running installation; it does not validate current config.

## State and evidence boundary

- The active checkpoint `docs/ASSISTANT_PLAN.md` says the user stopped SmartAlexa
  on 2026-10-18 and disabled its startup task. It says supervisor, listener, review
  service, and owned Ollama processes were stopped and absence verified. Thus
  **nothing runs today** according to the latest checkpoint.
- Current `assistant/config.toml` (mtime 2026-10-23 13:18) says
  `listener.async_enabled=true` (line 99) and `conversation.enabled=true` (line
  188), and `wake.model=["alexa"]` (line 78). The backup
  `assistant/config.toml.live-slice-backup` (mtime 2026-10-18 12:47) says both
  flags false and wake model `["alexa", "hey_mycroft"]`. Supplied 2026-10-23
  facts agree with the backup on all three values. This is an unresolved
  source/state conflict, not evidence the new paths ran. Available telemetry and
  logs end 2026-10-18 12:13, before the current config timestamp. Historical startup
  messages say `alexa, hey mycroft`; they do not establish the async flag value.
- Design docs describe target architecture and preferred future direction. They
  are not runtime proof. Logged outcomes are software outcomes, not user-confirmed
  success. No architecture recommendation is made here.

## Capability table

LOC are approximate source line counts. Config names are current file values.
Test names identify nearby coverage, not proof of live behavior.

| Capability | Modules (+ LOC) | Runtime path | Config flags | Last live evidence in available logs | Test files |
|---|---|---|---|---|---|
| Wake detection | `wake.py` (336), `listener.py` (734), `async_listener.py` (1860), `audio.py` (874) | Prior live path not established from retained config; current config would use async frontend | `listener.mode=stt`, async=true; current wake `model=["alexa"]`, personal model path, threshold_low=.15 | 2026-10-18 `assistant/logs/memoir.log`: capture health and rejected/near-miss records; 167 trigger-source `wake` telemetry rows are not recall denominator | `test_wake_startup.py`, `test_wake_personal.py`, `test_listener.py`, `test_async_listener.py` |
| Audio capture / VAD / echo | `audio.py` (874), `echo.py` (230), `foreground_capture.py` (274) | Microphone callback + main loop; current async config feeds cleaned VAD frames to async listener; command ASR uses raw audio | 16k model rate, 48k capture, echo cancellation=true, ring=30s | 2026-10-18 `memoir.log`: `echo_attached=True`, queue_overflows_total=0 in periodic health; historical config/path not established, no current device acceptance | `test_audio_stream.py`, `test_audio_jobs.py`, `test_prompt_echo_and_loudness.py`, `test_foreground_capture.py` |
| STT | `stt.py` (387), `audio_jobs.py` (477), `async_listener.py` | Prior listener path not established; current config would serialize listener and command jobs in async lane | small.en listener snapshot; large-v3-turbo command snapshot | Telemetry 2026-09-02–10-18: 21,823 trigger-STT timings (median 152 ms, p95 456); 437 command STT timings (497/679 ms); path/config during logging unknown | `test_stt_backend.py`, `test_stt_intent_e2e.py`, `test_async_listener.py`, `test_audio_jobs.py` |
| Intent / brain / conversation | `intent.py` (660), `brain.py` (962), `conversation.py` (312), `conversation_brain.py` (558), `concise_brain.py` (317) | Legacy rules/brain plus, if current config is launched, conversation first for ordinary turns; exact Codex controls stay separate | backend=brain, Qwen3 4B; conversation=true, interpreter=standard | No current conversation runtime event found through 2026-10-18. Historical log has legacy `ok` and `unknown_intent`; evaluation JSONL is synthetic/offline evidence | `test_intent.py`, `test_brain.py`, `test_conversation_flow.py`, `test_conversation_brain.py`, `test_concise_brain.py` |
| Coordinator / executor | `main.py` (4254), `conversation.py` (312), `conversation_contracts.py` (483), `computer_tools.py` (247) | Legacy executor; current config adds typed conversation coordinator then adapters | conversation=true; intent.backend=brain | Telemetry: 251 `ok` rows, 35 `error`, 103 `unknown_intent` across all trigger types; `ok` is not user validation | `test_main_loop.py`, `test_pipeline.py`, `test_conversation_flow.py`, `test_computer_tools.py` |
| Spotify playback | `spotify.py` (2083), `catalogue.py` (~1208), `resolution.py` | Legacy and conversation adapter use same Spotify client | Spotify target=active; radio_after_song=true; auto catalogue refresh=false | Through 2026-10-18 telemetry: 179 Spotify latency values (median 1081 ms, p95 6225); outcomes include 251 `ok`, 26 `not_found`, 7 `ambiguous` across all features | `test_spotify.py`, `test_catalogue.py`, `test_playback_verification.py`, `test_radio_queue.py` |
| Spotify Jam | `spotify_jam.py` (630), `computer_tools.py` (247) | Adapter invokes a separate bounded UI helper process; current conversation can request it | No dedicated enable flag; brain tool schema includes `start_jam` | None found in `memoir.log`/telemetry; source/test coverage only | `test_spotify_jam.py`, `test_computer_tools.py` |
| Windows volume / devices | `ducking.py` (482), `desktop_controls.py` (106), `main.py` | Legacy or coordinator adapter; volume writes Windows master, ducking changes Spotify session | duck_to_percent=10; audio input pinned to Realtek; feedback output SoundDrum | Telemetry includes volume requests but no reliable user-confirmed device/effect receipt; no specific success count isolated | `test_volume_execution.py`, `test_volume_semantics.py`, `test_desktop_controls.py`, `test_duck_sessions.py` |
| Opera / browser | `computer_tools.py` (247), `brain.py` (962) | Bounded Opera `--new-tab` launch through executor | Brain tool schema exposes explicit Opera search | None found in logs | `test_computer_tools.py`, `test_brain.py` |
| Apps / PC control | `computer_tools.py` (247), `external_apps.py` (59), `desktop_controls.py` (106) | Installed-app catalog / Explorer launch and bounded Windows media controls | No separate capability flag; exposed by brain schema/rules | None reliably attributed in logs; desktop outcomes: 2 `desktop_refused`, 2 `desktop_unknown_delivery` | `test_computer_tools.py`, `test_external_apps.py`, `test_desktop_controls.py` |
| Codex delegation | `global_router.py` (663), `global_commands.py` (554), `global_assistant.py` (1054), `codex_bridge.py` (1285) | Exact global request / confirmation path; conversation delegate calls same bridge, explicit-source gate | codex.enabled=true, model=gpt-6-luna; connection lazy | Through 2026-10-18: 2 `global_dispatched`, 1 `global_submitted`, 2 `global_submit`, 1 timeout, 1 error; not proof of completed useful job | `test_global_router.py`, `test_global_assistant.py`, `test_global_runtime.py`, `test_codex_bridge.py` |
| Spoken answers / retrieval | `knowledge_retrieval.py` (231), `conversation_brain.py`, `concise_brain.py` | Conversation response path, only when enabled; retrieval module is not imported by main runtime path | conversation=true; Qwen3 model | No logged live answer-quality evidence; conversation evals are offline | `test_knowledge_retrieval.py`, `test_conversation_brain.py`, `test_conversation_flow.py` |
| TTS / feedback | `feedback.py` (318), `main.py` | Wake beep and failure feedback; success speech disabled; TTS is Piper | beeps=true, speak_on_wake=false, failure=true, success=false | Telemetry: 46 feedback timings, median 1526 ms, p95 9664; period through 2026-10-18 | `test_feedback_cache.py`, `test_prompt_echo_and_loudness.py` |
| Hotkey / report / review page | `main.py`, `report.py` (416), `experimental/wake_train/review_server.py` | Ctrl+Alt+Space trigger, Ctrl+Alt+M miss report; supervisor starts review page | hotkey enabled; review service has no config flag | `assistant/logs/supervise.log`: review server ready on port 8767 at 2026-10-16 15:15:52; 5 `miss_reported` telemetry rows | `test_report.py`, `test_wake_review_server.py`, `test_main_loop.py` |
| Telemetry / archive | `telemetry.py` (1038), `diagnostic_archive.py` (222) | Event JSONL + rotating text log and bounded audio archive; async archive worker if async path enabled | telemetry path, archive_audio=true, 2048 MB cap, near-miss cap 256 MB | Files through 2026-10-18 12:13; 54,016 telemetry records; includes near-miss/pruned/rejected diagnostics and action events | `test_diagnostics.py`, `test_telemetry_failures.py`, `test_diagnostic_archive.py` |
| Supervisor / startup | `supervise.py` (210), `process_lifetime.py` (277), `main.py` | Task Scheduler -> supervisor -> optional Ollama and review child processes -> `assistant.main` | brain backend and conversation=true would require Ollama; review service is started by supervisor | `supervise.log` last 2026-10-18 12:06; checkpoint states startup task disabled and owned processes stopped | `test_supervise.py`, `test_process_lifetime.py`, `test_wake_startup.py` |

## Runtime paths

### What runs today

Per the 2026-10-18 checkpoint, the scheduled task is disabled and assistant,
supervisor, review service, and owned Ollama processes are stopped. So there is no
current startup-to-action path executing. The latest log lines (through 2026-10-18)
are historical runtime evidence.

### If started with the current config file

`assistant.supervise` starts/checks Ollama when brain or conversation is enabled,
ensures the local review server, then runs the assistant (`supervise.py:179-205`).
`Assistant.__init__` validates async settings, checks Spotify auth, primes Piper,
constructs wake/VAD/STT objects, creates `AsyncCaptureRuntime`, and warms one shared
Ollama model in a daemon thread (`main.py:139-267`). `run()` enters async runtime,
AC guard and microphone, starts hotkeys and optional echo canceller, then loops on
frames (`main.py:3662-3700`). Async frame processing performs acoustic proposals,
feeds the capture runtime and polls/dispatches completed events
(`main.py:_run_async_frame`, `_dispatch_async_events`). A final transcript is
offered first to `_handle_conversation`, then the global Codex handler, then legacy
intent/execution (`main.py:2803-2810`; async dispatch has equivalent handling).

With `listener.async_enabled=true`, `validate_async_config` requires STT mode and
rejects paused-input device, speech-triggered ducking, and pause-on-high-confidence
(`async_runtime.py:17-31`). `AsyncCaptureRuntime` owns serialized listener/command
ASR and diagnostic archive workers; it does not replace the main loop's microphone
or wake detector. With it false, `SttListener` consumes frames inline in the legacy
loop and no async worker/archive runtime is entered.

With `conversation.enabled=true`, main creates Brain + LocalModelRuntime at 8192
context for standard interpreter, and routes eligible final transcripts through
ConversationCoordinator before legacy intent. Exact Codex commands and active
Codex controls retain the global route. With it false, ordinary turns use legacy
intent backend (`brain` here: rules-fast paths plus local Ollama selection), with no
conversation coordinator. `intent.backend=brain` alone already initializes Ollama.

## Resource footprint

Configured/loaded on assistant startup (source does not prove current residency):

- Wake: openWakeWord stock `alexa` plus custom `assistant/models/wake/alexa_user.onnx`
  (4.30 MB); configured support/feature models are dependency-managed.
- VAD: Silero ONNX (2.33 MB). Listener STT: local faster-whisper small.en snapshot;
  command STT: large-v3-turbo `model.bin` 1.62 GB. Both transcribers are constructed
  in `main.py`; async lane serializes calls.
- TTS: Piper `en_US-lessac-medium.onnx`, 63.2 MB, primed at startup.
- Brain/interpreter: Ollama `qwen3:4b-instruct-2507-q4_K_M`; configured context
  8192 if conversation standard, otherwise 4096. Model blob size/residency not
  established by this audit. Health log once reported model loaded on GPU, but that
  is historical (`memoir.log.1`, 2026-10-15).
- Present but not runtime-configured: Qwen3-ASR-0.6B cache 1.88 GB and Qwen HF
  safetensors 1.56 GB; evaluation/training assets, not models loaded by main path.
- Project venvs: `.venv` (Python 3.10.11) is the assistant/test environment;
  `.venv-asr`, `.venv-kws`, `.venv-wake` (3.10.11) and `.venv-qwen-asr`,
  `.venv-qwen-hf` (3.12.4) are specialized eval/training environments. Exact
  script-to-venv ownership was not fully enumerated; runtime task path points at
  project Python, and no venv is started by assistant code itself.
- Supervisor may start one local `ollama serve` process and the review-server child
  (Python module); assistant itself runs Python, spawns lazy Codex CLI transport
  when explicitly used, and launches a short-lived Python helper for Spotify Jam.
  Spotify, audio drivers and Windows services are external dependencies.

## Dead, experimental, and overlapping implementations

- `assistant/experimental/` contains offline eval/training, wake review, AEC
  experiments, alternate ASR trials and candidate wake training. Main runtime does
  not import these as action handlers; the exception is supervisor's explicit
  review-server child (`assistant.experimental.wake_train.review_server`).
- Qwen ASR, Qwen HF, Parakeet/Sherpa evaluation scripts and results are not selected
  by `main.py`; configured live recognizers are faster-whisper. Their files/caches
  are not evidence of use in current runtime.
- Parallel request paths: deterministic `intent.parse`, legacy `Brain` tool choice,
  `global_router` Codex classifier, and `ConversationBrain`/`ConciseConversationBrain`
  typed plans. Current config selects standard conversation plus global control;
  legacy parser/brain remains fallback and also handles some paths. `intent_llm.py`
  is another LLM intent implementation, but main's configured `brain` branch calls
  `Brain`; no direct import from main was found.
- Parallel capture paths: `SttListener` synchronous legacy path and
  `AsyncSttListener` serialized path. `Endpointer` is reused for foreground capture
  and conversation replies. `foreground_capture.py` and `async_listener.py` have
  different ownership/use cases, not interchangeable proof of the other path.
- Spotify UI Jam is separate from Spotify Web API playback; `spotify_jam.py` is
  run as an isolated helper. No live Jam result found.

## Live outcome and timing counts

`assistant/logs/telemetry.jsonl`: 54,016 parseable records, 2026-09-02 13:27 to
2026-10-18 12:13 local time. The log mixes user command, wake rejection, near-miss,
archive pruning and global-control records; do not call its total a command count.

- `trigger_source=wake`: 167; `stt`: 291; `wake_rejected`: 8,159;
  `near_miss`: 26,750; `archive_pruned`: 18,631. Wake misses/recall cannot be
  estimated because there is no denominator of spoken wake attempts.
- Outcome counts: `ok` 251; `unknown_intent` 103; `error` 35; `false_accept` 26;
  `not_found` 26; `ambiguous` 7; `rejected` 8,159; `near_miss` 26,750;
  `pruned` 18,631. Other global/desktop events are listed in telemetry but are not
  user-confirmed successes. These categories are not mutually equivalent stages.
- Available non-null timing fields (median / nearest-rank p95): trigger STT
  21,823 (152 / 456 ms); command STT 437 (497 / 679); intent 382 (1 / 1504);
  Spotify 179 (1081 / 6225); feedback 46 (1526 / 9664); whole handler 407
  (8341 / 18615); wake-to-ack 184 (869 / 2459). Sparse fields and mixed request
  types prevent a full stage-failure funnel; 35 `error` records have no reliable
  common stage denominator.
- `memoir.log*` capture health includes frame counts, VAD/wake peaks, queue
  overflows and echo-reference status. Those periodic samples establish capture
  process activity at the time, not detection recall or audio intelligibility.

## Coverage and uncertainties

- Source/config/log/test filename mapping was inspected; test suite was not run and
  no test collection was requested. Test presence is not passing evidence.
- Current config vs supplied verified state vs stopped checkpoint conflicts on both
  flags. The logs stop before current config timestamp, so enabled async/conversation
  runtime has no live evidence here.
- Log `ok` and dispatch statuses are application outcomes, not audited external
  effects. No reliable live success evidence was found for Jam, browser, app launch,
  retrieval answers, current conversation, or current device routing.
- Startup-task disabled state is reported from the active checkpoint; this audit did
  not query Task Scheduler. Current process absence was not re-probed.
- Runtime usage mapping for all six venvs, full dependency-loaded model memory, and
  exhaustive dynamic imports were not measured. Static import/path statements above
  should be read as source reachability, not runtime tracing.
