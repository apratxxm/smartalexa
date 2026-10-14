# Memoir: architecture audit and proposed redesign

## Current Astra-reviewed implementation map — 13 October 2026

This is the lead's consolidated architecture incorporating Astra's grouped reviews, not a claim that Astra independently authored every component. The complete target below remains approved; deployment is incomplete. The user now prioritizes a visible architecture and the earliest safe live vertical slice over additional feature expansion.

```text
Microphone callback / bounded PCM queue
  -> ONE capture owner: raw + frame-paired cleaned audio, sample intervals, epoch
  -> VAD + wake-model proposals (scores are proposals, not labels or authority)
  -> bounded segmentation + immutable capture snapshots
  -> ONE serialized speech-recognition lane
       listener peeks / full turns / acoustic confirmation / verification
  -> wake receipt bound to the SAME capture, epoch and confirming interval
  -> complete command transcript; reject incomplete, failed or stale captures
  -> ONE conversational brain: whole-turn meaning, corrections, brief answers
       answer | clarify | no_request | unsupported | typed ordered plan
  -> application validation: registry, source evidence, numbers, independent authority
  -> ONE coordinator / executor -> existing capability adapters
       music | Windows volume | Opera / display | explicit Codex delegation
  -> observed effect receipts -> brief natural feedback, never invented success

Every stage -> bounded diagnostics and timings
Rejected snapshots -> off-capture archive -> human review -> grouped training data
Reviewed positive spans + independent session holdouts -> offline candidate training
Held-out wake recall / false activations + real-device acceptance -> manual promotion
```

### Frozen rules from Astra's reviews

- A late wake confirmation cannot authorize an earlier decoded snapshot. Retain boundary and continuation ownership; decode the extended capture before dispatch. The confirming interval must be contained in the final capture, with matching epoch and capture identity. Failed, stale, gapped or truncated command audio cannot execute a prefix.
- Acoustic confirmation uses the same serialized recognition lane, a fixed shared deadline and immutable snapshots. No synchronous confirmer may compete for the recognizer. One owner drains results; overload and losses must be explicit.
- Before calling a legacy handler that owns capture or recognition, stop admission and require an acknowledged, quiescent recognition lane. A timed-out handoff must not invoke the handler. First integration improves idle listening only; continuous capture during handlers requires the subsequent central capture-request API.
- The conversational model receives the entire admitted transcript. Compact its protocol rather than truncating the request. Provider evidence IDs expand to application-owned source excerpts; Codex's current-turn reference expands to the exact original transcript.
- Codex authority is checked outside the model over the whole source: allowed, revoked or ambiguous. Late cancellation vetoes dispatch. Clearly quoted/code material is data; uncertain scope blocks rather than inventing authority. Blank chat creation cannot implicitly send a task.
- Model formatting, interpretation accuracy and runtime latency are different gates. Use a bounded production-shaped startup warmup, then the ordinary strict deadline on warmed trials. Fast answers do not compensate for missing tool plans or false success claims.
- Training uses exact extractor-window arithmetic, bounded sequential extraction, deterministic disclosed session-aware selection and untouched session holdouts. Uncertain clips and detector/ASR outputs are not confirmed Alexa labels. No automatic live-model replacement.

### What exists versus what remains

| Area | Implemented in source | Still required before claiming it works live |
|---|---|---|
| General brain and coordinator | Typed decisions, capability registry, follow-up lifecycle, source-bound evidence and Codex guard | Real-model semantic acceptance, then enable the disabled conversation route |
| Responsive listening | Serialized recognition jobs, asynchronous segmentation, capture-bound acoustic reservations; disabled main-loop integration with acknowledged legacy handoff | Actual device timing and continuous handler capture acceptance |
| Rejected-wake logging | Bounded archive worker, raw/processed provenance, explicit losses and unknown labels | Connect it to the new runtime path; verify useful rejected captures on devices |
| Wake learning | Reviewed-span preparation and offline candidate path, provenance and session gates | Confirmed recent positive spans and independent session groups; extraction/training/export and held-out evaluation |
| Resources | Child process cache/concurrency limits | Installed-runtime adoption and sustained memory/latency measurements |
| Music, volume, Opera and Codex | Existing adapters plus stronger source/effect contracts | Actual audible/visible end-to-end acceptance; no claim of arbitrary app/page control |
| Capability expansion | Target design and bounded capability contracts | Retrieved spoken answers, broader verified navigation, Jam QR and interruption remain separate work |

### 14 October foreground-capture continuation

The opt-in async path now routes conversation clarification and shared music
reply capture through an application-owned `ForegroundCaptureDriver` backed by
`CaptureRequestManager`. The driver borrows the microphone/VAD only after an
acknowledged async suspension, scores each acquired frame once, checks wall
deadlines during stalls, and returns no PCM for incomplete captures. Its source
epochs/timestamps are driver-relative receipt metadata, not physical callback
timing or exact speech-onset evidence.

Codex dictation and review capture also use this helper in the opt-in path,
preserving the existing start/speech budgets. Capture, transcription and submit
boundaries fence session identity, request generation and target revision;
stale cleanup cannot erase a newer draft, including reuse of the same session
object. This extension passed 46 focused global-runtime fixture checks.

This is a cooperative foreground migration: bare-wake recorders remain legacy,
and listening does not continue during handlers. Both enablement
flags remain false. Focused acceptance on the frozen source: 78 native-diagnostic
and capture contracts, 79 async/conversation contracts, and 20 selected legacy
compatibility checks passed. None establishes microphone recall, real model
semantics or native effects.

### Earliest live rollout, in order

1. Finish the affected source checks and one bounded real-model decision comparison. Do not repeat broad mocked suites as a substitute for inference.
2. Integrate the asynchronous listener behind a disabled flag with exclusive handoff and rejected-snapshot archival. Preserve rollback and existing device behavior.
3. Enable only a slice that clears its inference and runtime gates: a brief definition, one music command, Windows volume, Opera display/search and explicitly authorized Codex routing. Unsupported work must be explicit, not silently routed elsewhere.
4. Start that slice and measure real wake-to-feedback, command completion and visible effects. If only the old route can launch, label it legacy rather than pretending the new architecture is deployed.
5. Replace domain-owned microphone reads with central capture requests, then add retrieved answers/navigation and genuine wake-data collection. Train only when the dataset gates are actually met.

### Astra's latest launch-boundary review

The newly observed Ollama 0.30 chat-format defect confounds previous model scores. Verify the separately downloaded fixed runtime's published hash/version, reproduce the exact invalid-prefix input through production `/api/chat`, then use the unchanged schema and baseline sampling for a bounded shaped warmup and strict warmed semantic evaluation. `/api/generate` and changed-penalty experiments remain diagnostics, not production acceptance. A formatting fix does not explain away an invented action.

The registry describes available capabilities; source references identify evidence. Neither proves that each proposed action was requested. Keep the central interpreter and execution contracts, but do not treat evidence references as semantic authorization, add per-phrase rules, or assume larger weights solve the problem.

If interpretation still fails on the fixed chat path, one bounded paired diagnostic can isolate capability competition: roughly twelve held-out cases using the full registry versus an oracle reduced registry containing every needed capability and at least two plausible distractors. Preserve all decision branches, policy, evidence, transcripts and sampling; measure exact action sets, omissions, inventions, kind and latency. The reduced registry is an experiment, not a new production router. Preserve an untouched final acceptance set.

Asynchronous idle integration may proceed independently behind its false flag with exclusive legacy handoff. Do not thread handlers or expand capabilities in that packet. A development decision preview with no executors is possible, but must not be presented as a working action assistant. Production actions remain gated on the exact corrected runtime's semantic and authority evidence.

As of this update the production assistant is stopped, conversation routing is disabled, and the development Ollama server has been stopped at the user's request. No model weights or recordings were deleted. The architecture map is not itself runtime evidence.

Latest runtime checkpoint: the separately verified0.31.2 runtime repaired the observed chat-format failure, including a schema-enforcement trap and the exact earlier invalid-prefix request. The current baseline model nevertheless passed only16 of28 strict warmed cases before a timeout stopped evaluation; it still generated extra actions. Production remains stopped after unloading the test runtime. Main-loop asynchronous integration, real interpretation acceptance and physical-device acceptance are distinct unfinished gates. This does not change the approved target architecture.

Status: **approved target architecture; phased implementation underway**. Audited 11 October 2026. The complete redesign is not yet implemented or accepted.

Implementation update, 12 October 2026: the user approved this direction and requested worker execution. The bounded first wave and its explicit pending gates are tracked in `ASSISTANT_PLAN.md`. The user subsequently requested stopping the assistant and Ollama while code work continues. The new conversation path remains disabled pending model and device acceptance. This document remains the target architecture; approval does not mean that all described modules or capabilities already exist.

Acceptance update, 12 October: bounded wake-free clarification capture and generation fencing are implemented behind that flag; continuous capture/interruption, retrieval and general page navigation are not. The installed 4B model's latest synthetic decision run passed 27/40 cases despite 818 passing offline integration checks. That model run retained a relaxed timeout after cold loading and does not prove the three-second deadline. Provider output grammar is only a formatting constraint, not authority or semantic correctness: explicit prompt protocol, app-derived Codex admission, exact source evidence, current-turn numeric provenance and post-generation validation remain separate layers. Volume/media readbacks strengthen effect receipts, but mocked checks do not prove native behavior. The runtime remains stopped while these gates are addressed.

This document gives you the system map, the reasons for changing it, the proposed contracts, and the order in which we can prove the replacement works. The intended product is a conversational personal assistant. Music, computer control and Codex delegation are capabilities of that assistant.

## 1. Decision in brief

**Rebuild the coordinating core incrementally. Retain and wrap the useful audio, music, Windows and Codex adapters.**

The current system can recognize a request correctly and still handle it incorrectly. It has grown from a music-command application into several overlapping request handlers. It lacks a general answer path, a shared conversation model, and a uniform definition of successful execution.

A central language model is appropriate for understanding requests and producing brief answers. It should not own microphone timing, permissions, retries, volume restoration, or the claim that an external action succeeded. Those need explicit application code and observable results.

| Direction | Benefit | Principal cost/risk | Recommendation |
|---|---|---|---|
| Keep extending current handlers | Small immediate diffs | More competing state and special cases; preserves missing answer path and listening gaps | Containment fixes only |
| Replace the coordinator behind existing adapters | Addresses the main foundation while preserving tested device work | Requires contracts, migration boundaries and replay tests | **Preferred** |
| Rewrite everything | Uniform starting point | Relearns Bluetooth, Spotify, cancellation and delivery edge cases; removes useful safeguards | Not justified by this audit |
| Replace the interaction with a hosted realtime voice service | Worth a later comparison for conversational quality | Changes privacy/cost/dependency assumptions; still needs tool safety and verification | Optional experiment, not the default design |

No model substitution or training run is approved by this document. Keep the historical local-first constraint until you choose otherwise.

## 2. Evidence and scope

The audit followed capture, wake detection, transcription, interpretation, music/desktop/Codex execution, feedback, telemetry, configuration, startup and wake-learning code. Luna collected bounded source and test evidence; the lead interpreted the results and chose the proposed design. Related Token Tracker UI code is a separate application and is not being redesigned here.

Findings below distinguish source facts, recorded observations, and hypotheses. Source inspection cannot establish microphone recall, Bluetooth audibility, real-model accuracy or the exact cause of an unrecorded missed wake. No live microphone, playback, model training, Codex submission or production restart was performed for this audit.

### Recorded observations

- The definition request “What is the meaning of virtuoso?” reached the Codex route. The route took 1,343 ms; the whole logged handler took 23,498 ms and accumulated 13,320 ms of feedback. The handler includes the confirmation workflow. These are **not** measurements of model-only latency or time to first audible answer.
- The latest 30 STT-handler records in the inspected log had a median handler duration of 8,671 ms and nearest-rank p95 of 23,801 ms. Outcomes were 22 `ok`, three `false_accept`, two `desktop_refused`, one `global_dispatched`, one `global_handled`, and one `unknown_intent`. `ok` is a program outcome, not human-verified task accuracy.
- A 24-hour window ending at the latest inspected event contained 2,796 candidate/rejection diagnostic rows. These include summaries and background detections; they are **not 2,796 user attempts or misses**. There is no honest wake-recall denominator in those logs.
- Current reviewed-label metadata contained 157 negative records, four presence-only `unsure` records, zero span-ready positives, and one retracted record. A saved consensus proposal contained 20 unsure rows and no positives. These are different label classes, not interchangeable training examples.

Log snapshot: `assistant/logs/telemetry.jsonl`, 100,317,947 bytes, last modified 2026-10-11 11:39:57.702 UTC. Later logs may differ. Test and resource-check results are recorded in section 16.

## 3. What exists today

Read this map before reading thousands of lines of implementation:

```text
Microphone callback -> bounded audio queue
    -> main listening loop -> VAD / acoustic wake / rolling wake transcription
    -> captured utterance -> command transcription
    -> existing Codex session or Codex-oriented global router
         -> create / ask / dictate / status / list / open / ignore
    -> otherwise music/desktop rules + local brain
         -> one supported tool or UNKNOWN
    -> domain execution / domain-specific clarification
    -> blocking spoken feedback
```

The microphone callback can continue while application processing blocks, but the handler explicitly discards capture in several paths. Continuous device capture is therefore not continuous conversational availability.

### Code reading map

| Responsibility | Current files | Important boundary to understand |
|---|---|---|
| Assembly and main control flow | `assistant/main.py` | About 2,954 lines; owns many unrelated lifecycles |
| Microphone and buffering | `assistant/audio.py` | Device callback versus consumer; overflow and intentional discard differ |
| Segmentation and wake acceptance | `assistant/listener.py`, `assistant/wake.py`, `SileroVAD` in `assistant/audio.py` | Speech detection, wake proposal and wake authorization are separate decisions |
| Echo/reference handling | `assistant/echo.py` | Cleaned signal helps gating; ASR and acoustic wake still receive raw microphone audio |
| Speech recognition | `assistant/stt.py` | Separate wake and command models, device placement and fallback |
| Request interpretation | `assistant/brain.py`, `assistant/intent.py` | Rules plus a music/desktop tool schema, not a general conversational brain |
| Codex routing and sessions | `assistant/global_router.py`, `assistant/global_assistant.py` | Separate classifier, held prompts, confirmation and job state |
| Codex transport | `assistant/codex_bridge.py` | Ownership, delivery uncertainty and native protocol |
| Music search and playback | `assistant/spotify.py`, `assistant/catalogue.py`, `assistant/resolution.py`, `assistant/local_media.py` | Search candidates, selection and verified playback must remain distinct |
| Bounded desktop actions | `assistant/computer_tools.py`, `assistant/desktop_controls.py`, `assistant/spotify_jam.py` | App IDs and observed UI targets; not arbitrary computer control |
| Spoken output and ducking | `assistant/feedback.py`, `assistant/ducking.py` | TTS output, media-session ducking and Windows master volume are different controls |
| Health and evidence | `assistant/health.py`, `assistant/telemetry.py` | Available resources and stage evidence, not proof of recognition accuracy |
| Configuration and lifecycle | `assistant/config.py`, `assistant/config.toml`, startup/supervisor scripts | Dependency availability, model residency and recovery |
| Learning/review | Existing wake-review and active-learning modules/scripts | Human labels, machine proposals, training readiness and promotion |

### Which model does which job today?

These are configured roles; configuration does not prove that every model is currently loaded or healthy.

| Role | Configured model/backend |
|---|---|
| Acoustic wake proposals | openWakeWord `alexa` and `hey_mycroft`, plus a configured personal candidate |
| Continuous wake transcription | faster-whisper `small.en`, pinned local snapshot |
| Command transcription | faster-whisper `large-v3-turbo`, pinned local snapshot |
| Music/desktop interpretation and shared local routing model | Ollama `qwen3:4b-instruct-2507-q4_K_M` |
| Speech synthesis | Piper `en_US-lessac-medium` |
| Delegated Codex work | Configured `gpt-6-luna`, through the Codex integration |

The Luna agents used to develop this project are not the always-listening speech recognizer. Changing the development worker alone cannot repair the live audio pipeline. Conversely, a good command transcriber does not compensate for an earlier wake gate rejecting the turn.

### One real failure, step by step

1. Speech was recognized well enough to preserve the definition question.
2. `main.py` offers the command to `_handle_global` before ordinary domain execution.
3. `global_router.routing_candidate` admits unknown non-music text as well as explicit Codex text.
4. The classifier's vocabulary contains Codex actions, not a direct factual-answer action.
5. `ask` without a selected chat can stage a new chat and prompt.
6. Confirmation protects submission, but adds a workflow the user never wanted in the first place.

The fix is an appropriate top-level decision space and explicit delegation authority. Lowering a wake threshold or adding an artist alias cannot repair this failure.

## 4. Findings ranked by architectural importance

### A. A domain router has become the assistant's general brain — confirmed

`global_router.py:109` considers unknown music parses eligible for Codex classification. `main.py:2808` offers the request to that route before ordinary parsing. The brain schema at `brain.py:108` also has no direct-answer branch.

**Effect:** ordinary questions can become coding workflows. Unknown text is being treated as a destination hint rather than a request that still needs understanding.

**Change:** one capability-neutral decision contract with answer, clarify, plan and no-request outcomes. Codex requires expressed delegation or an already authorized, matching conversation task. A mentioned word such as “code” is not itself sufficient authority.

### B. Conversation state is fragmented — confirmed

Codex dictation/review, music clarification, retry capture and ordinary listening each have their own loops. `_brain_context` at `main.py:551` supplies artists/playlists and an empty `now_playing`, not shared conversational context.

**Effect:** a pending workflow can dominate interpretation; “that”, “from the beginning”, corrections and new requests have inconsistent meanings. Timeouts and readiness cues vary by domain.

**Change:** a single conversation coordinator owns turns, pending questions, referents and cancellation. Tools never independently listen or speak.

### C. Listening gaps are an intentional part of current execution — confirmed

`_say_feedback`, `_transcribe_captured`, `_parse_intent` and `_execute` use capture-discard scopes. Codex creation can wait and drain audio for up to eight seconds. The rolling listener also shares its consumer loop with model work.

**Effect:** a wake or follow-up spoken during these intervals can be lost irrespective of the model's quality. This is a plausible class of misses; current logs cannot assign every reported miss to it.

**Change:** independent capture ownership, bounded worker queues, an explicit listening state, and stale-result cancellation. Reliable speech interruption requires echo/double-talk testing; simply removing discard would risk reacting to our own voice.

### D. Request extraction is too dependent on one substring and one tool — confirmed

The brain requests the first cohesive command, copies a contiguous substring, limits it to 400 characters, validates punctuation boundaries, and chooses one tool. Connected volume is a special-case extension. Some ambiguity checks fall back to the same rule parser.

**Effect:** the representation is awkward for “Play Red—actually don't, pause”, multiple valid actions, and longer dictated prompts. A syntactically valid tool call can still express the wrong request.

**Change:** understand the whole final turn first, including negation, correction and action order. Preserve evidence in the original transcript. Produce a bounded plan only after that interpretation. Do not execute an early phrase before the user's final correction is available.

### E. Semantic bias remains even with transcription hints disabled — confirmed source risk

`stt.bias_from_spotify` is false, but the brain prompt tells the model that a heard artist sounding like a known artist *is* that artist. Artist/playlists also occupy prompt space on unrelated requests.

**Effect:** a closed vocabulary can replace a real, unfamiliar name with a familiar one. This audit does not attribute a particular live substitution solely to that prompt.

**Change:** preserve heard entity text; let catalogue resolution return scored, attributed alternatives. Known entities are candidates, not ground truth. Use domain context only when relevant.

### F. A capability can prevent unrelated assistant startup — confirmed

`main.py:124` constructs `Spotify()` outside the guarded authentication check. `spotify.py:396` obtains required secrets; `config.secret` raises if absent.

**Effect:** missing Spotify configuration can prevent the whole assistant from starting even when the desired task is a local question or app operation.

**Change:** initialize capabilities independently, expose availability reasons, and start the core without optional account integrations. Background vocabulary refresh must not gate listening.

### G. Tool success is unevenly defined — confirmed

There is strong verification in several Spotify paths, including album restart. Other paths report request acceptance rather than observed effects; local previous-track handling is weaker than the API path. Desktop launch/search and Codex submission also have different result types.

**Effect:** “done” can mean different things. A launched process, accepted remote turn and completed user goal are not equivalent.

**Change:** a shared result envelope distinguishes accepted, completed, failed and unknown delivery, with tool-specific evidence. Preserve stronger existing verifiers.

### H. Rate limiting and account quota exhaustion are conflated — confirmed gap, historical cause unproven

The Spotify wrapper handles every 429 through `Retry-After` or a guessed 300-second delay. It does not inspect `error.reason`. Its rolling budget is process-local.

Current Spotify documentation distinguishes rolling rate limits from developer-account quota exhaustion, including `QUOTA_EXCEEDED`. New credentials are not a reliable fix for an account-scoped quota. Inspect the actual response before calling something a “ban”. [Rate limits](https://developer.spotify.com/documentation/web-api/concepts/rate-limits), [quota modes](https://developer.spotify.com/documentation/web-api/concepts/quota-modes).

**Change:** structured error classification; persisted account-aware cooldown/quota state; a single request budget owner; cache/batch where appropriate; no background retry storms. Continue verified local transport controls when remote search is unavailable.

### I. Speech UX is attached to implementation details — confirmed

Global feedback appends “Ready.”; confirmation can read back model/destination/prompt details; speech is blocking. Most feedback metadata records the last utterance although elapsed feedback may cover several.

**Effect:** irrelevant narration, long ceremonies and misleading latency diagnosis. A new voice alone will not fix these interaction costs.

**Change:** one response policy, brief answers, risk-specific confirmations and per-utterance timing. Say “Yes?” when a wake-only turn actually needs a prompt; do not insert an unnecessary wake/readiness exchange into a complete one-breath request.

### J. The learning loop cannot deliver what the review experience implies — confirmed

The simplified review UI records “Alexa heard” as presence-only uncertainty and offers no precise span controls. The reviewed-span training path remains unsupported. A separate bounded weak-label candidate trainer exists, but it has a different input contract and its outputs are not production-eligible by default.

**Effect:** reviewing lots of random/background candidates does not create the positive, representative data needed to improve recall. Detector/ASR agreement is not a human wake label.

**Change:** reconcile review labels, dataset preparation and the chosen training algorithm before another training promise. Keep presence labels, precise spans and machine proposals distinct. See section 11.

### K. Evidence, reproducibility and recovery need a baseline — confirmed

- No Git repository was available in this checkout. Versioned rollback must precede the migration.
- Root `pytest.ini` references unrelated `remitmatch` paths. A green ad hoc command is not a reproducible project check.
- Configuration is permissive dotted lookup rather than validated startup configuration.
- Enabled echo paths lazily import `pyaudiowpatch` and `livekit.rtc`; local media lazily imports `winrt.windows.media.control`. These are absent from Memoir's requirements and no separate Memoir provisioning step was found. An existing environment can work while a fresh installation loses those features. Several environment variants also need a reproducible runtime manifest; distinguish installed packages from declared requirements.
- JSONL telemetry is append-only and about 100 MB in the inspected snapshot; the text-log rotation does not rotate this file.
- Pending Codex ownership/job state is mostly in memory. A saved navigation receipt is not a durable job ledger.

**Change:** bounded diagnostics, reproducible environments, typed configuration, durable operation receipts and explicit restart reconciliation. Do not restart or replay remote work merely because local state disappeared.

## 5. Proposed architecture

Use a **modular monolith**: one application with clear modules, not a fleet of new services. Keep model serving separate where it already is, and use bounded workers for blocking native/model operations. Add processes only where measurement demonstrates isolation is needed; extra processes can duplicate RAM/VRAM.

```text
                             CAPABILITY AVAILABILITY + POLICY
                                           |
MIC -> AUDIO SERVICE -> TRANSCRIPT -> CONVERSATION COORDINATOR
       |                    |                  |
       |              source evidence         v
       |                              CENTRAL INTERPRETER
       |                                /      |       \
       |                             answer  clarify   plan
       |                                |      |       |
       |                                |      |    VALIDATOR
       |                                |      |       |
       |                                |      |    EXECUTOR -> ADAPTERS
       |                                |      |                  |
       |                                +------+------ TOOL RESULTS
       |                                               |
       +---- interruption/echo protection ---- RESPONSE + SPEECH

         Every transition -> bounded event trace / operation ledger
         Background jobs -> coordinator; never own the microphone
```

### Responsibilities and ownership

| Component | Owns | Must not own |
|---|---|---|
| Audio service | Devices, timestamps, reference audio, ring buffer, VAD, wake proposals, capture health | App intent, tool execution, indefinite queued commands |
| Transcription service | Audio-to-text, segment timing, model admission, truncation/error metadata | Artist selection or tool authorization |
| Conversation coordinator | Turn lifecycle, pending questions, context, interruption, deadlines, response ordering | Domain API implementation |
| Interpreter | Meaning of the complete turn; direct answer or proposed plan; relevant ambiguity | Permissions, success claims, arbitrary executable code |
| Capability registry | Available operations, argument/result schemas, effects, cancellation and verification rules | Treating unavailable integrations as mandatory startup dependencies |
| Validator/policy | Schema, referent validity, constraints, authorization, plan bounds | Pretending that schema validation proves semantic understanding |
| Executor | Dependency ordering, deadlines, idempotency/retry policy, receipts, partial results | Listening or speaking inside tool functions |
| Adapters | Spotify, Windows audio, apps, Opera, web retrieval, Codex, bounded UI operations | Competing top-level assistants |
| Response service | Short speech, optional screen detail, TTS lifecycle, output level | Announcing success before supporting evidence |
| Evidence/learning | Traces, retention, dataset provenance, offline evaluation | Automatic production retraining on unverified labels |

## 6. Structured contracts: the useful part of the MCP idea

Use typed contracts at the boundaries. A contract defines what information is allowed and what it means. Avoid separate free-text `intent`, `action`, `category` and `request_type` fields that can contradict one another.

MCP standardizes tool discovery and input/output schemas. It does not supply conversational understanding or authorize actions. Start with an internal registry using the same concepts; expose or consume MCP at external integration boundaries when useful. Do not build an MCP server for every internal function. [MCP tool specification](https://modelcontextprotocol.io/specification/2025-11-25/server/tools).

### Application envelopes

The application, not the model, generates IDs, timestamps and authority metadata:

- `Transcript`: schema version; session/turn ID; raw final text and timed segments; audio reference; final/truncated flags; capture/model provenance; observed quality flags. Preserve original text even when deriving normalized text.
- `ContextSnapshot`: recent relevant turns, pending question, current referents, observed app/playback state with age/source, authorized job references, and available capabilities. Do not treat stale “now playing” as current fact.
- `Decision`: one of `answer`, `clarify`, `plan`, `no_request`, or `unsupported`. Branches have their own required fields. Evidence quotes refer to input; code resolves offsets. Model-generated confidence alone is not authorization.
- `ToolResult`: operation ID; status; typed data; observed evidence and timestamp; structured error; side effects; retry/cancellation information. `accepted` and `completed` are different states.
- `SpokenResponse`: brief text, related turn/job, priority, generation, optional display detail and source links. It is derived from the decision and verified results, not speculative tool success.

### Example decisions

These examples are proposed interfaces, not current production JSON. The registry supplies the full discriminated JSON Schema. The model does not generate request IDs, permissions or idempotency keys.

Definition:

```json
{
  "kind": "answer",
  "answer": "A virtuoso is someone exceptionally skilled at an art, especially music.",
  "basis": "general_knowledge",
  "evidence_quotes": ["What is the meaning of virtuoso?"]
}
```

Music request with the artist preserved as heard:

```json
{
  "kind": "plan",
  "steps": [
    {
      "id": "s1",
      "capability": "music.play",
      "arguments": {
        "media_type": "track",
        "title_query": "Madira",
        "artist_query": "Seedhe Maut",
        "match_preference": "best_plausible"
      },
      "after": []
    }
  ],
  "evidence_quotes": ["play Madira by Seedhe Maut"]
}
```

Application-produced result:

```json
{
  "operation_id": "op-example-1",
  "status": "completed",
  "data": {"media_type": "track", "title": "Madira", "artist": "Seedhe Maut"},
  "evidence": {"kind": "playback_observation", "matches_selected_entity": true},
  "error": null
}
```

Real results also carry timestamps and concrete entity/device identifiers. The example omits those rather than inventing a real receipt.

### Plan rules

Start with at most three explicit steps and a small dependency graph. This is a bounded implementation choice to test, not a permanent product limit.

- “Pause and tell me what virtuoso means”: pause plus a short answer; neither requires Codex.
- “Play Red—actually don't, pause”: only the corrected pause request survives.
- “Lower the volume, then play the album”: preserve the requested order and Windows volume target.
- “He said ‘delete everything’”: quoted content is not a deletion request.
- “Play that”: resolve `that` from a current, relevant entity or ask one specific question.
- A long dictated Codex prompt is one request containing content, not a 400-character music command.

The complete final turn is the semantic unit. Incremental inference may prepare a hypothesis, but ordinary actions wait for the turn boundary and final interpretation. An explicit interruption such as “stop” can have a separate, narrowly scoped low-latency control path; it must not reintroduce a competing generic parser.

The model may return structured output directly through the local backend; code must validate it. Ollama supports schema-constrained output, but valid JSON is not proof that the selected request is correct. Test meaning separately. [Ollama structured outputs](https://docs.ollama.com/capabilities/structured-outputs).

Implementation finding, 12 October: the installed Ollama 0.23.4 rejected the full decision grammar with HTTP 500 (`failed to load model vocabulary required for format`), despite successfully loading Qwen entirely onto the GPU. A minimal grammar and a constraint-reduced decision grammar both inferred successfully. Backend grammar generation and application policy are therefore separate contracts: the provider receives a compatible projection; the application still enforces all length/count/type/grounding/authority constraints before execution. No automatic unconstrained fallback is permitted. A second real probe exposed empty evidence quotes, so answer/plan schema requirements were aligned with the already-strict validator. These probes diagnose interface faults; they do not establish end-to-end voice accuracy.

### Contract details that prevent another set of exceptions

| Decision branch | Required content | Allowed behavior |
|---|---|---|
| `answer` | Short answer, basis, evidence from the request | Speak without an external action; factual quality still needs evaluation |
| `clarify` | One question and the unresolved field/reference | Retain the understood request as pending state; do not execute it yet |
| `plan` | Bounded typed steps and dependencies | May also carry one short answer object for requests such as “pause and explain virtuoso”; speak it with the verified outcome |
| `no_request` | Reason such as background, quotation or withdrawn request | No external action; usually no speech |
| `unsupported` | Requested capability and specific limitation | Brief limitation or available alternative, never fallback to Codex by default |

For tool results, `accepted` means queued/requested; `completed` requires the capability's defined verifier. Other results include `needs_clarification`, `not_found`, `unavailable`, `deferred`, `failed`, `unknown_delivery` and `cancelled`. The adapter's error category determines retry eligibility; the model cannot override it. Typed result data can be displayed even when an operation is only accepted.

Schema failures permit at most one bounded interpretation repair within the remaining deadline. No tool executes before validation. Failure of the brain must not automatically run an unrelated rule parse of the same ambiguous utterance. A separately tested exact-control fallback can handle its narrow supported cases and report degraded availability.

Compound actions are not a database transaction. Report partial success explicitly. Retry only eligible unfinished steps; do not replay completed playback or submission. Undo/compensation is allowed only when the capability defines it and user intent supports it. A new conversational turn can suppress stale speech and unstarted actions, but cannot erase an external action that already happened.

## 7. Conversation, capture and latency

### One owner for conversational state

Use explicit states such as `idle`, `capturing`, `interpreting`, `awaiting_clarification`, and `speaking`, with separately tracked running jobs. A long Codex task can run while the conversation returns to `idle`; it must not lock the assistant into dictation mode.

Each new turn has a generation number. A late model result or speech item from an obsolete generation cannot talk over the new turn or execute a stale plan. Cancelling the current reply is distinct from cancelling an already submitted remote job.

### Capture and ending a turn

- The audio callback only timestamps and buffers frames; no inference, disk writes or external calls.
- A dedicated audio consumer maintains wake/capture state independently of tool calls.
- Start with VAD/silence endpointing plus explicit maximum duration and truncation metadata. Add semantic endpoint hints only if measured incomplete-turn cases justify them.
- Do not ask the LLM to judge completion on every audio frame or every partial transcript. That would add inference load and delay.
- Use shorter silence for short controls and a deliberately selected dictation mode for long prompts. Preserve both “Alexa, pause” in one breath and wake-then-command.
- A maximum capture duration is a safety limit, not a reason to wait that long after a complete utterance.
- Bound queues by time as well as count. Never execute a stale request that sat behind a slow job. Surface overload as a distinct degraded state.

### Speech and interruption

First remove blindness during non-audio work. Then add interruption of assistant speech using tested echo/reference handling. The latter requires recorded double-talk cases: user and assistant speaking together, music plus user, output-route changes, and missing/stale loopback reference.

The current echo path intentionally leaves ASR audio raw because earlier cleaned audio harmed speech. Compare raw and processed variants on held-out real recordings before changing that choice. Bluetooth delay or a bad reference is a measurement problem, not something a conversational prompt can fix.

The configured paused-speaker input name is currently empty, so the existing optional second-microphone path is not enabled. Reconcile this against the requested laptop-only-during-playback/both-when-paused behavior. If two streams are enabled, retain stream identity and deduplicate a single spoken request; do not execute twice or switch devices without verifying capture and output behavior.

If speech is unintelligible in the raw recording at the user's position, software-only guarantees are inappropriate. Establish a supported distance, speaker route and music condition from recordings; then evaluate placement/reference/input changes if that envelope is insufficient. Do not hide unsupported loud-music conditions behind aggregate accuracy.

### Response policy

- Ordinary questions: answer in one or two short sentences; elaborate on request.
- Obvious successful controls: a short acknowledgement or no redundant narration when the effect is unmistakable.
- Wake alone: one concise acknowledgement after authorization, then a clear listening window.
- Clarification: one focused question preserving already understood details.
- Failure: explain the actionable limitation briefly; offer a useful next step where possible.
- Slow operations: acknowledge once when needed, release the conversation, and deliver an outcome later.
- Never append a generic readiness phrase to every reply. Never read internal model names, tool schemas or full prompts by default.

Use deterministic wording for simple outcomes and the interpreter for natural answers. A second model call is justified for synthesizing retrieved evidence or substantial tool results, not for rephrasing every successful pause.

### Measurements

Record audio onset, last voiced frame, endpoint decision, STT start/end, interpretation start/end, tool acceptance/effect verification, TTS first sample and completion. Distinguish time to acknowledgement from time to useful answer and from task completion.

Initial **targets**, to validate on the actual laptop:

| User experience | Proposed gate |
|---|---|
| Warm, familiar short factual answer | End of speech to meaningful audio: median <=2 s, p95 <=3 s |
| Authorized wake acknowledgement | Decision to first output sample p95 <=250 ms; separately report word-end-to-decision delay |
| Short local transport command | End of speech to observed effect p95 <=1.5 s |
| Long job | Bounded acknowledgement and return to listening; completion reported separately |
| Ongoing use | No unreported capture-discard intervals; dropped/stale work is visible in diagnostics |

These are proposed acceptance budgets, not claimed results or guarantees. Tune by measured stage cost before buying complexity.

## 8. Capabilities and product behavior

### Questions and web answers

Stable general knowledge can use a direct local answer. Current facts, uncertain niche facts and requested searches need retrieval. Opening a Google search tab and retrieving evidence for a spoken answer are different capabilities.

The current small local model has not been validated as a general factual-answer backend in this audit. Compare prompt/contracts first, then evaluate the deployed model on definitions, reasoning, requests and tool selection. Model upgrades must improve that measured workload within memory and latency budgets. Neither a self-declared `general_knowledge` basis nor a confidence score proves an answer correct.

Add a bounded retrieval adapter with timeouts, source metadata and an answer-synthesis step. Treat retrieved pages as untrusted data; their text cannot change permissions or request tool execution. Choose a provider separately, including privacy, credentials and costs. The assistant's existing Opera search helper does not implement this.

### Music

Retain catalogue search, canonical entity IDs, caching and verified playback. Make them a reusable capability rather than the top-level request language.

- Title without artist: choose the best plausible result using search evidence and relevant context.
- Title with artist: preserve the supplied artist constraint. Do not silently replace it with a familiar artist.
- Approximate title: support plausible ranked matching. The user's preference for playing a likely result should reduce unnecessary rejection, but should not make unrelated first results look certain.
- If no plausible candidate exists: ask one useful clarification or offer the closest named result. Do not loop through repeated generic apologies.
- Artist, album, track, queue, display and restart remain distinct actions. “Show this album” does not start it.
- Restart the current album through observed album context and first playable track; retain the existing verification. Restarting a track seeks that track to the start.
- User volume commands control Windows master volume. Spotify's own playback volume remains at the user's chosen 100% policy; routine controls do not use Spotify volume as a substitute. Temporary ducking targets the appropriate Windows media session and restores guarded state; assistant speech level is controlled separately.
- Starting a Jam is a bounded UI operation with a visible QR success condition. No QR means do not say the invitation is ready.

A search candidate's score is matching evidence, not an authorization or a universal probability. Calibrate ambiguity on the user's recorded cases and a held-out mixed catalogue.

### Codex

Retain the existing adapter's ownership, confirmation binding, delivery uncertainty and model checks. Place it beneath explicit delegation intent.

- “Create a chat using Luna and check Gmail orchestration”: create the requested chat, submit the authorized prompt, persist the receipt and open the exact created chat when requested.
- “What is orchestration?”: answer; no chat creation.
- “Show that chat”: resolve the actual receipt or ask which chat; never fabricate a title or claim it appeared without navigation evidence.
- Long-running turns become durable jobs with accepted/running/completed/failed/cancel-requested/interrupted states.
- Reconcile on restart from receipts and observed remote state; do not resubmit an uncertain mutation.
- Cancellation request acceptance is not proof that the remote turn stopped. The current protocol uses `turn/interrupt`; wait for terminal state.
- Use the available model catalogue and explicit user selection. Do not silently upgrade to an expensive model or substitute after a limit.

Generate contract fixtures from the installed app-server version rather than relying only on remembered protocol shapes. Official app-server documentation provides schema generation and lifecycle methods. [Codex app-server](https://learn.chatgpt.com/docs/app-server).

### General computer control

Expand in layers: documented APIs/deep links, then Windows accessibility/UI automation, then an optional vision/computer-use adapter for genuinely unsupported UI tasks.

Each operation declares its goal, allowed app/window, expected visible result, timeout and cancellation behavior. Verify foreground/window state before input. Never blindly replay clicks after an uncertain timeout. Opening an app, searching a browser, and changing account settings have different risk.

The fact that this coding chat can use a computer-use tool does **not** mean the standalone assistant can call it. The runtime needs its own supported transport, authentication, available model and budget. Select that integration only after the capability contract and provider constraints are explicit.

## 9. Safety without constant user review

Routine personal-assistant use should not require approving every tool call. Use explicit, testable risk rules:

| Action class | Default behavior |
|---|---|
| Answer, inspect, navigate, play/pause, reversible volume | Execute within the request and return concise results |
| Ambiguous entity/target or contradictory request | Ask a focused clarification |
| Explicit Codex delegation within configured scope | Apply the established authorization policy; avoid full readbacks unless resolving meaningful uncertainty |
| Sending messages, purchases, deletion, account/security changes | Bind confirmation to exact target and effect |
| Uncertain external delivery | Reconcile; do not automatically retry the mutation |

Existing stronger protections stay in place during migration. Any reduced confirmation behavior needs its own reviewed policy tests and explicit product decision; the new architecture does not silently remove current guards.

A pending confirmation includes operation, arguments, target, revision, expiry and request ID. A reply to an old question cannot approve a new action. Web text, tool output and quoted speech are evidence, not fresh authority.

A recognized wake word is not identity authentication. Background speech or another person can say it. Consequential operations need the configured permission boundary regardless of apparent voice familiarity. Screen captures and retrieved/private account content also need scoped access and bounded retention; they should not be indiscriminately copied into model context.

## 10. Resource and lifecycle design

One admission controller owns heavy inference concurrency. Foreground audio/capture comes first, active-turn STT and interpretation next, background labelling/training last. Avoid simultaneous model benchmarks while evaluating live latency.

- Keep proven small real-time DSP on the CPU unless measurement favors otherwise. Moving everything to the GPU can increase contention and transfer overhead.
- Track model placement, residency, queue delay, physical RAM, commit headroom, GPU memory and actual stage latency separately.
- Compare the current two-transcriber arrangement with an alternative only through a controlled replay and warm/cold measurement. Shared weights or fewer models are hypotheses, not guaranteed wins.
- Define explicit warm-residency and idle-unload policies. Current indefinite `keep_alive` can retain local-model memory; changing it can trade memory for first-command latency.
- On an unavailable model, expose capability degradation honestly. A controlled fallback must have its own tested contract and latency budget; do not blame the microphone for an inference failure.
- Stop admitting background work before allocation headroom becomes critical. Do not silently close user apps or purge models/data.
- Initialize local conversation independently of Spotify credentials, browser availability and Codex authentication.
- Recover only owned device/process state on shutdown. Preserve guarded volume restoration and single-instance protection.

Ollama documents residency controls, CPU/GPU placement reporting and how parallel/context settings affect memory. Use measurements from the deployed runtime, not parameter count alone. [Ollama FAQ](https://docs.ollama.com/faq).

## 11. Wake quality and active learning

This is a separate evaluation track. A better central brain cannot recover an uncaptured wake. A custom false-positive verifier cannot recover examples that never crossed its first-stage proposal gate. [openWakeWord verifier documentation](https://github.com/dscripka/openWakeWord/blob/main/docs/custom_verifier_models.md).

### Repair the data contract first

1. Inventory recordings by capture path, device, session, music condition and label provenance. Keep revoked labels revoked.
2. Separate queues: likely missed wakes, likely false activations, and intentionally sampled background. Tell the reviewer which queue they are hearing and why it was sampled.
3. Preserve the simple presence buttons. For positive training, either add a short alignment-confirmation step or implement a separately validated clip-level training path. Do not silently convert “Alexa somewhere in this clip” into a precise span.
4. Use ASR/alignment or an audio-capable reviewer to **propose** labels/spans where available. Store the method and uncertainty. Do not claim to have listened to audio based only on a transcript.
5. Collect representative positive examples across independent sessions and troublesome music levels, plus negatives including vocals and the assistant's own voice. More random negatives alone do not fix recall.
6. Split by recording session before augmentation or training. Prevent related chunks from leaking across train and held-out sets.
7. Train only after the selected path's gates are satisfied. Store dataset/code/model hashes and the exact experiment configuration.
8. Evaluate stock/current/candidate on the same held-out recordings and on the complete wake pipeline, including endpoint and ASR confirmation. Promote deliberately with rollback; no silent online updates.

### Logging missed wakes honestly

Keep a bounded rolling buffer and selectively archive candidate windows under explicit retention limits. Record why audio was retained or dropped, detector scores, queue health, capture generation and reference availability. Absence of an archived clip does not mean absence of user speech.

For an actual recall estimate, use counted attempts in quiet and music conditions with independent labels. Candidate logs are useful diagnostics, not the denominator. Report false activations per monitored hour separately from miss rate and acknowledgement latency.

Raw audio stays local by default; define retention, storage cap, export consent and protection for reviewed training examples. Training should not run concurrently with live acceptance measurements.

## 12. Observability and durable state

Replace one mutable summary as the sole evidence source with append-only stage events plus a compact final summary. Record each feedback utterance separately. Use bounded queues and rotating retention so disk I/O cannot stall the audio thread.

Every trace links session, turn, decision, operation, tool result and spoken response. Logs distinguish:

- speech not captured;
- captured but wake rejected;
- recognized incorrectly;
- recognized correctly but interpreted incorrectly;
- interpreted correctly but tool unavailable;
- accepted operation without confirmed effect;
- completed operation with wrong feedback;
- stale/cancelled work intentionally suppressed.

Persist operation receipts and pending externally visible jobs in a small transactional store. Keep secrets out of traces. On restart, re-observe remote state before claiming a job is running or finished. Uncertain operations stay uncertain until reconciled.

A small status surface should show listening, processing, speaking, awaiting a specific answer, or degraded, plus running jobs when requested. This is supporting feedback, not a new widget redesign or a requirement to watch the screen.

## 13. Migration sequence, ordered by return on effort

Each phase needs a working vertical slice. Do not complete a giant framework before proving the definition question and a simple command.

| Phase | Work and retained components | Exit evidence | Rollback boundary |
|---|---|---|---|
| 0. Establish control | Versioned source baseline excluding secrets/audio/models; correct test entry point; reproducible runtime inventory; bounded traces; requirements ledger | Clean baseline manifest, scoped tests, known active process/config | Restore prior source/config without deleting user data |
| 1. Conversation-first routing | New decisions and coordinator with text input; direct answer; explicit Codex gate; retain existing execution adapters | Definition, navigation, music and correction cases; zero accidental Codex writes | Feature flag selects old route; no double execution |
| 2. Unified turn and result lifecycle | Central clarification, typed tool results, deadlines, receipts, stale-result fencing; adapters no longer speak/listen | Replays for timeout, cancellation, follow-up, unknown delivery and partial success | Per-capability wrapper fallback; one executor owner |
| 3. Responsiveness | Independent audio consumer, bounded model scheduling, per-utterance speech queue, independent job lifecycle | Stage timing, overload tests and recorded audio replay; then physical acceptance | Preserve old audio path until new path clears device gates |
| 4. Capability completeness | Retrieved answers; music policy; app/browser/display flows; Codex create/open; Jam verification; bounded compounds | End-to-end task matrix with effect evidence | Enable one capability at a time |
| 5. Wake-learning repair | Label/queue contract, representative positives, alignment or clip-level path, grouped holdout, candidate evaluation | Dataset gates and complete-pipeline comparisons; manual promotion | Production wake model untouched until acceptance |
| 6. Natural interaction | Echo-aware interruption, voice comparison, tuned follow-ups and optional alternate backend experiment | Double-talk, routing change and long-session user acceptance | Independent speech/backend flags |

Wake data collection can proceed alongside phases 1–2 without loading training models. Candidate training and heavy audio comparisons must be scheduled separately from live latency testing. Fix critical verified regressions in existing adapters when their capability is touched; do not bundle unrelated rewrites.

The authorization guard must sit outside both old and new routing paths during migration. Rolling back an experimental interpreter must not restore accidental Codex delegation. Shadow evaluation records proposed decisions without invoking tools; it cannot send a second request to compare outcomes.

### Proposed code ownership after migration

```text
assistant/
  app.py                    # composition/startup only
  contracts/                # typed envelopes and capability schemas
  conversation/             # coordinator, context, turn state, response policy
  reasoning/                # local interpreter and retrieval synthesis adapters
  execution/                # validation, policy, scheduler, job ledger
  audio/                    # existing capture/wake/STT/echo, moved gradually
  capabilities/             # wrapped music, Windows, browser, Codex, web, UI
  diagnostics/              # events, health, retention, replay support
  learning/                 # review, provenance, dataset, offline candidates
```

This is a target responsibility map, not a demand to rename every file first. Move code when its boundary changes; keep compatibility wrappers while migrating.

### Worker organization

The lead owns contracts, cross-cutting decisions, failed-test diagnosis and final acceptance. Luna implements bounded lanes after those contracts are agreed. Use one reusable validation worker for focused tests/replays; no duplicate full-suite runs or competing edits to `main.py`.

Potential independent lanes after phase 1 contracts settle: capability result wrappers, event/ledger support, and dataset/review contract repair. Audio concurrency and coordinator integration require coordinated ownership, not an indiscriminate swarm.

## 14. Acceptance matrix

The tests must measure the assistant the user wants, not only parser consistency.

| Layer | Required cases | Evidence that counts |
|---|---|---|
| Interpretation | Definition; web-needed question; music; app navigation; explicit Codex; quoted/negated/corrected speech; two connected actions; unfinished prompt | Frozen expected decisions plus real-model replay; report deterministic and model results separately |
| Conversation | Wake-alone, one-breath request, follow-up pronoun, topic change during pending clarification, cancel, timeout, delayed old result | Ordered event/state traces and no stale execution |
| Audio | Quiet/music levels, output-route changes, raw versus processed speech, speaker/mic disconnect, overflow, own speech, double-talk | Labelled recordings, device logs and physical tests |
| Execution | Missing credentials, no network, Spotify quota/rate limit, not found, uncertain delivery, target changes, repeated request | Typed failure/result plus observed side effects; no duplicate mutation |
| Music | Approximate/title-only/title+artist; album versus track; artist radio; queue; restart; show; volume and duck restore | Matching entity/context/device and correct audible/visible effect |
| Codex | Create with requested model, submit, open exact chat, status, interrupt, restart reconciliation | Real receipts and observed native app state under explicit test authorization |
| Desktop | Known app, ambiguous app, Opera search, Spotify display, unavailable window, Jam QR | Verified app/window/UI state, not just a returned keypress |
| Resources | Warm/cold start, concurrent normal apps, queue saturation, 60-minute soak, bounded log growth | RAM/commit/VRAM and latency traces; no inferred improvement from GPU label alone |
| Privacy/authority | Untrusted retrieved instructions, quoted commands, stale confirmation, cancelled/retracted labels | Explicit negative tests and no unauthorized tool invocation |

Initial release gates:

- Zero accidental Codex dispatches on the frozen ordinary-question/negative-authority corpus.
- Zero duplicate external mutations in timeout/restart/delivery-unknown tests.
- Zero master-volume changes from speech feedback or duck restoration tests.
- At least 95% correct decisions on a held-out mixed request set, reporting sample size and per-category failures. This is a target, not current measured accuracy.
- Wake target: at least 95% counted attempts accepted in each supported condition and no more than 0.5 false activations per hour. Start with at least 30 positive attempts per condition and two hours of negative listening as a **pilot**, then expand; those sample sizes alone do not establish a precise long-term guarantee.
- Latency targets from section 7 under declared warm/cold and resource conditions.
- Explicitly record unsupported device conditions rather than hiding them in an average.

No suite of mocked tests establishes real microphone recall, naturalness or correct visible app control. Keep those acceptance steps separate.

## 15. How you get control of the project

Use a short review cycle for each vertical slice:

1. State what the assistant should do for a real utterance, including a wrong interpretation to avoid.
2. Trace its input, state, decision, tool and result through the map above.
3. Agree on the contract and product behavior before implementation.
4. Review the small diff at the boundary that changed, rather than reading every unrelated file.
5. Inspect one passing and one adversarial example, with source/test/live evidence distinguished.
6. Update this design when behavior or ownership changes. Do not let architecture exist only in chat history.

Suggested first lesson/change: the definition request. Follow `main._handle_utterance` to `_handle_global`, `routing_candidate`, the Codex classifier and staging; then contrast it with `Decision(kind="answer")` and a short response. Next trace “play that album from the start” to show why context and verified tool results matter.

### Decisions proposed for your review

1. One local-first conversational coordinator; optional cloud comparison only by choice.
2. The model understands the whole turn and proposes answers/plans; code owns effects, state and authority.
3. Short, natural speech; no default full prompt readback; meaningful confirmations remain.
4. Approximate music selection is permitted when plausible, preserving an explicitly supplied artist.
5. Incremental coordinator replacement with the current adapters retained and individually audited.

These are proposals, not claims that you have approved every implementation detail. The two most useful discussion questions are: **does this division of responsibility match the assistant you want, and where would you prefer an immediate best-effort action versus one clarification?** They do not block completing this audit.

## 16. Audit validation record

### Focused checks

Windows, Python 3.10.11, pytest 8.4.2, existing project `.venv`:

```text
.venv/Scripts/python.exe -B -m pytest -p no:cacheprovider tests/test_global_router.py tests/test_global_continuation.py tests/test_main_loop.py tests/test_telemetry_failures.py tests/test_brain.py tests/test_intent.py
```

With `PYTHONDONTWRITEBYTECODE=1`: **781 passed, zero failed; pytest 17.34 seconds**, wrapper 18.055 seconds. The worker checked fixtures first: temporary storage, mocked streams/brain responses and no expected production device or network execution. Thirteen source/test files had matching before/after SHA256 fingerprints. No assertions or production behavior were changed to obtain this result.

Production subset of the tested before/after SHA256 fingerprints:

```text
assistant/global_router.py     7BF04C9DB4EE67D2B7079A735167C3E297650419E194EAE8976661B1C29594BF
assistant/global_assistant.py  BB603BA6FADA5A6DF7C7DB65181C7994D556BBB951445F3A50C1CC987F09A948
assistant/main.py              62AE9C2BEE2F58EA0DF403F440B5E09F0A0CA0D90AB2D85ECC7FC4336CC278C5
assistant/telemetry.py         9763178E5F970DE5B57700E68CD0A71BB15F64A6CE1631FF205C46C359AA7321
assistant/brain.py             FCD27F276F9F88A8ED824D983B664972A20CE3F09F581F9DEA6B53DDB84DA919
assistant/intent.py            602573BEF7A214D786C8BA7408C9C9528D92CFC538202808E37D160B30405160
```

The documentation check confirmed existing-source paths, parsed all three JSON examples, and reconciled the test/log/resource figures with the returned evidence. The proposed target tree is not represented as existing implementation.

Pure-code counterexamples:

```text
routing_candidate("What is the meaning of virtuoso?") -> True
parse("Play Red. Actually don't. Pause.") -> PLAY_TRACK, query="red"
brain.decide(the same text, stub) -> UNKNOWN, matched_rule="ambiguous_compound"
stub model calls -> 0
```

The wrapper prevented the naive parser result from executing in that probe, which is useful protection. It still did not understand the correction as pause. These are deterministic control-path observations, not tests of the real model's comprehension.

### Existing resource state, not a benchmark

Read-only snapshot at **2026-10-11 17:13:48 IST**:

- Available physical RAM: **0.97 GiB of 15.19 GiB**.
- `ollama ps`: loaded `qwen3:4b-instruct-2507-q4_K_M`, 3.5 GB model size, **100% GPU** placement, context 4096.
- RTX 4060 Laptop GPU: **6,893 / 8,188 MiB** used across the device. Per-compute-process memory was unavailable; do not attribute all of it to this model.
- Two Ollama processes reported about 6,199 and 2,218 MiB of private allocations. Private committed bytes are not the same as resident physical RAM and cannot be added to GPU usage as a before/after savings estimate.
- No Python process matching the project's `.venv` path was found in that bounded process check. That alone does not exclude another environment or launcher.

This shows existing resource pressure and a GPU-resident brain, not a reason to promise more GPU migration as the solution. No paired CPU/GPU benchmark or model-memory reduction measurement was performed.

### Limits

Passing these checks does not establish live wake recall, end-to-end response latency, correct Bluetooth playback, native Codex navigation, factual-answer quality or successful retraining. Those remain the explicit acceptance gates above. This pass changed only this architecture document and the existing request ledger.
