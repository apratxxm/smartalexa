# Historical assistant checkpoints — cold archive, not current state

Updated 2026-10-14. This ledger is for the assistant product only. It records
requirements and evidence checkpoints; planned behavior and passing checks do not
prove deployment or live acceptance.

## Product scope

SmartAlexa is a private, local-first Windows assistant for general conversation
and bounded computer actions. “Memoir” is the historical checkout/package name.
The user wants an assistant that handles whole turns, ordinary questions and
follow-ups naturally, then uses explicit typed plans for actions. The goal is not
a collection of phrase-specific command patches.

Requested behavior includes:

- Answer definitions, explanations, trivia and general questions briefly and naturally. Do not route ordinary questions to Codex or an action by default.
- Understand corrections, compound requests, references, clarification answers, cancellations and topic changes across a whole turn.
- Play, pause, resume, skip, queue and search; distinguish track, album and artist. Preserve supplied artist context; restart and “show” are distinct from playback.
- Control Windows master volume only for explicit commands. Music ducking is a Spotify-session effect and restores the prior session level.
- Open/search in the browser, show known apps/music/chats, and perform supported desktop actions through registered capabilities with observed results. No arbitrary app/page control claims.
- Dispatch Codex only when the whole current request explicitly authorizes it. Quoted text, code, ambiguity, revocation and cancellation grant no authority. Create, submit, open and report status as separate verified effects.
- Wake over music and during normal use with measured first-attempt recall, false activations and wake-to-feedback latency. Keep responses short and confirmations tied to verified results.

## Architecture and safety invariants

- One application-owned capture/VAD/ASR path and conversation owner; adapters do not listen or speak independently.
- Interpret the complete admitted turn. The model proposes a typed answer, clarification, no-request, unsupported result or ordered plan; code owns schemas, state, authority, cancellation and effects.
- A capability registry defines argument/result contracts and verifiers. Validate plans against the source turn, registry, evidence, numbers and independent policy. A citation or registry entry is not user intent.
- Bind wake confirmation and command audio to the same capture, interval and generation. Late confirmation cannot authorize a stale/partial prefix.
- One executor returns receipts distinguishing accepted, completed, failed, cancelled and unknown delivery. Verify visible/audible/native outcomes; do not repeat uncertain mutations on restart.
- Cancellation and newer turns fence stale results. Ambiguous/revoked Codex authority blocks dispatch; quoted instructions remain data.
- Unsupported work is explicit. Unknown intent never falls through to Codex, Spotify or another capability; do not add phrase-specific exceptions.
- Keep local-first processing, secrets, recordings, labels, weights, logs and personal exports out of source publication.

## Current gates

### Source and configuration

- Legacy Spotify and desktop adapters exist; source presence does not prove current end-to-end behavior.
- Typed decision and asynchronous capture/request contracts exist, but the central manager is not a complete continuous-processing path.
- `listener.async_enabled` and `conversation.enabled` are false in `assistant/config.toml`; keep both gated until their acceptance criteria pass.
- Wikipedia retrieval is not a complete spoken web-answer flow. `native_decision_eval` is diagnostic, not production acceptance.
- Synthetic scores, mocks and offline suites do not establish natural answers, microphone recall, playback, visible effects or live Codex behavior.

### Model and runtime evidence

- 2026-10-13: fixed portable Ollama 0.31.2 resolved a formatting issue from 0.30.0. The exact invalid-prefix prompt returned an accepted pause plan; this proves formatting repair on that path, not general accuracy.
- 2026-10-13: warmed strict three-second compact-protocol run passed 16/28 primary cases before stopping. Failures included invented actions, playlist scope, correction/clarification and invalid plans. Median decision latency: 1,937.5 ms. No effects ran; no promotion.
- Last observed 24-case comparison scored 11/24 for both old `qwen3:4b-instruct` and `qwen3.5:4b`; neither qualified. This is historical synthetic evidence, not a fresh result from the repository setup.
- Native experimental adapter: 39 fake/source checks. Latest native real-model probe has no durable handoff in this resumed tree, so its accuracy is unverified—not passed or failed.
- Earlier disabled async idle integration/main mocks reported 306 checks; `CaptureRequest` manager 18; review UI 89, including an actual DOM-tag correction. These do not accept continuous capture or live devices.

### Wake data and learning

- 2026-10-13 review snapshot: zero confirmed positive Alexa groups for training; four presence-only/uncertain clips. Older 59 enrollment positives are separate historical data, not new ground truth or independent-session evidence.
- Prior inventory: 149 usable recordings, four unsure excluded, zero positive train/holdout groups, 144 negative train and one negative holdout. Recheck counts and provenance before citing them as current.
- Exact human-reviewed spans are required for positives. Detector/ASR is a proposal only. Never auto-label, convert unsure to positive, or promote without independent-session holdouts and held-out evaluation.
- No physical wake-recall target (including 95%), sustained listening, device behavior or retraining readiness is established by source/mock checks.

### Live and deployed behavior

- Spotify, Windows volume, browser and Codex adapters have historical bounded checks. Current playback, volume effects, navigation, Codex actions and restart reconciliation still need acceptance.
- Source checks do not prove installed task/process state, model loading, devices, live inference, wake-over-music or deployed flags. Verify at rollout.
- Keep routing disabled until semantic/authority acceptance, safe integration handoff and a small live wake/response/effect slice pass.

## Acceptance order

1. Recheck current source/configuration and the installed process/task state without
   changing user data or unrelated applications.
2. Accept the existing disabled asynchronous idle integration in the intended
   runtime/device slice, preserving exclusive legacy handoff, stale-result
   fencing and explicit failure behavior. Its source checks already passed;
   rerun them only if their relevant files or contracts change.
3. Re-run the fixed, held-out whole-turn semantic and authority corpus on the exact
   candidate runtime; report exact decisions, invalid plans and latency. Do not
   use diagnostic ablations as production results.
4. Accept conversation lifecycle and capability receipts: clarification, follow-up,
   cancellation, timeout, partial success, unknown delivery and restart recovery.
5. Validate bounded real effects independently: a short answer, one music request,
   Windows volume, browser display/search and explicitly authorized Codex routing.
6. Run an authorized small live wake-to-feedback slice and inspect audible/visible
   outcomes. Keep unsupported cases disabled and record limits.
7. Centralize domain-owned capture, then complete retrieved spoken answers,
   navigation and genuine human-reviewed wake-data collection. Train only after the
   provenance, session and holdout gates are met.

Acceptance reporting must keep these evidence classes distinct:

| Evidence | Establishes | Does not establish |
| --- | --- | --- |
| Source review | Implemented branches and configured defaults | Correct model meaning or live effects |
| Unit/mock checks | Deterministic contracts under fixtures | Microphone, GPU inference or native apps |
| Synthetic model evaluation | Candidate behavior on that fixed prompt/corpus | Recorded/live speech or held-out generalization |
| Recorded-device probe | Behavior on the identified files/devices | Continuous listening or future-room recall |
| Live vertical slice | Observed behavior in the tested session | Broad reliability, training readiness or deployment safety |

## Private source repository checkpoint — 2026-10-14

- A private GitHub repository named `smartalexa` was created using the existing
  Git credential. This is the assistant's source destination; do not include the
  separate Token Tracker product or its history.
- The local Git repository was prepared on `main` with assistant-only exclusions
  for secrets, models, logs, labels, other-product data and personal exports.
- Assistant-only baseline commit `3938aed` was pushed to private
  `https://github.com/apratxxm/smartalexa` on `main`. Its 204 tracked files exclude
  the Token Tracker product, shared root ledger, credentials, recordings and
  model/runtime assets. Local and remote commit IDs matched after publication.
- Personal exports were relocated to ignored local storage. Branding asset hashes
  were verified. Keep those local-only boundaries intact.
- This file is the assistant-specific ledger. The repository-root
  `ASSISTANT_PLAN.md` remains the shared/mixed ledger; it is not the authoritative
  assistant-only task list.
- The user authorized coordination with the active `token tracker` chat. Its
  lead owns reconciliation and final relocation/startup changes; the prepared
  sibling checkout is not authority to overwrite that lead's newer edits. This
  chat leaves original Token Tracker files and its existing scheduled task intact.

## Active unresolved work

### Resume checkpoint — 15 October 2026

- User direction: continue the work after compacting the chat, use the current
  risk-based validation policy and preserve independent lanes. A Luna-high
  worker produced this checkpoint and the lead reviewed it. No supported
  force-compaction control or host context reset is evidenced. Earlier worker usage
  errors have cleared (current Usage reports tools allowed; 5-hour use 60%, week
  25%). Retry only after verifying availability and do not loop on a limit.
- Checkout is private `apratxxm/smartalexa`, `main` at `1053e14` (also
  `origin/main`). Working edits are uncommitted and undeployed. Preserve the
  user's changed `AGENTS.md`; Token Tracker remains separately owned.
- Capture migration in `assistant/main.py` routes all direct recorder calls
  through the guarded helper and preserves legacy `after_prompt` values. Lead
  fixed two fixture defects; nine selected capture regressions pass. This does
  not establish continuous-handler or device acceptance.
- Native evaluator changes in `assistant/experimental/native_decision_eval.py`
  and its test add incremental JSONL rows, exact `failure_category`, separate
  admission/transport timing and a stop condition. Its 54 fake checks passed;
  current source hash still matches the tested B2BDF9A8 prefix. No real native probe has run after these
  edits, so semantic results remain unknown.
- New `assistant/concise_brain.py` is not wired to production. It proposes one
  answer or action in one local call; app code owns IDs/evidence and validates
  the result. `allow_actions=False` omits tools and rejects actions. Lead fixed
  answer evidence and context handling; 30 checks passed plus one bad context fixture,
  then both corrected context profiles passed. Current-turn Codex authority and
  existing domain validators remain required. No real-model acceptance.
- New Qwen evaluator `assistant/experimental/asr_eval/run_qwen_hf.py` uses the
  fixed 48-command/16-negative corpus. Lead fixed prompt slicing, feature dtype,
  local-only loading with `trust_remote_code=False`, PCM validation, finite
  900-second deadline/30-second PCM limit, and stopping after allocation failure.
  Eight focused checks passed; no Qwen inference has run. Ignored
  `.venv-qwen-hf` metadata shows torch 2.9.1+cu128, transformers 5.13.0 and
  accelerate 1.12.0; preserve distinct `.venv-qwen-asr`. Installation completion
  and `assistant/models/qwen-hf-cache` remain unverified.
- Fresh wake collection is design-only: archived telemetry is reviewable, but
  the reviewed store cannot accept unreviewed uploads. A separate pending queue
  and source adapter are needed. No auto-positive/span-fake labels, recording
  implementation or training; require human/session provenance.
- Runtime owner is `check_runtime_model_state`, assigned one concise evaluation
  on the existing model with temporary listening pause, incremental local report
  and `finally` restoration. Do not compete for GPU or change its pause/model
  state. Last reported snapshot near 21:02 IST: scheduled assistant active,
  installed Ollama 0.23.4, original model/context 4096, routing flags false and
  pause flag absent. The 14:14 IST memory/VRAM figures are stale; refresh before
  future inference and recheck task/log health after the probe.

- Remaining ownership: `check_runtime_model_state` completes its bounded probe;
  lead owns review and corrective source checks. Resume native real-model
  acceptance only with durable per-case failure evidence and exact candidate
  identity. Wake collection contract mapping remains separate from capture
  implementation. Then pursue device-level async/continuous capture, semantic
  and authority behavior, verified tool receipts, live-device acceptance and
  publication gates. Production routing and model promotion stay disabled.
- Validation: follow `C:/Users/aprat/.codex/AGENTS.md` risk-based scope and
  cadence. Retain the selected nine capture regressions and corrected
  concise/Qwen checks as current evidence; source-only review does not warrant
  another broad run. Run focused existing checks once for a coherent behavior
  batch or when material risk/failure warrants them; preserve high-risk lifecycle
  and security coverage. Distinguish fake checks, source review, native inference
  and live-device evidence.

## Parallel continuation checkpoint — 2026-10-14

- Restored the existing review server on loopback port 8767. Page and state API
  returned 200, with exact-span review controls present. This is server/markup
  evidence, not a new human label or browser interaction acceptance.
- Fresh no-write preparation: 159 review rows, 149 usable recordings, 10
  unavailable historical recordings, four unsure excluded, 144 negative train
  and one negative holdout recording. Positive train and holdout span/session
  counts remain zero. Training did not run; representative confirmed positives
  from independent sessions are still required.
- Added a reproducible, loopback-only native-function diagnostic CLI. It records
  runtime/tag identity and experimental exact-request admission, invokes no
  effects, rejects redirects, and bounds response size/socket reads. Together
  with the managed foreground capture contracts, 78 focused fake/source checks
  passed at the frozen revision. This is not real-model semantic acceptance.
- Foreground reply integration passed 79 async/conversation checks plus 20
  selected legacy compatibility checks behind the existing disabled async path.
  Lead review corrected an invalid capture-mode enum and missing exception
  cleanup; fixtures now exercise the real driver with a fake microphone. It
  requires acknowledged ownership handoff and discards incomplete PCM. It is a
  cooperative foreground adapter, not continuous listening during handlers or
  a completed migration of every domain recorder.
- Runtime inventory found the scheduled assistant using installed Ollama
  0.23.4: merely downloading 0.31.2 did not select it. The configured executable
  override was absent and a healthy existing server bypasses binary selection.
- A controlled task stop quiesced the assistant and its owned server/runner.
  The host rejected the portable server launch. The unverified executable pin
  was removed; original task and installed 0.23.4 server were restored. Fresh
  logs then confirmed the Realtek stream, SoundDrum loopback opening and GPU STT.
  Portable-runtime inference remains untested; no model was promoted. Never
  bypass the host restriction. A separately authorized, listening-paused probe
  on the existing installed server is a compatibility diagnostic, not equivalent
  to fixed-runtime acceptance.
- Token Tracker, its active lead, source and startup remain outside this lane.
- A second accepted source packet centralizes opt-in Codex dictation/review
  capture without shrinking its speech budget. Generation, target-revision and
  session-identity fences protect capture, STT and submission; scoped cleanup
  preserves newer drafts. All 46 global-runtime fixture checks passed, with
  unchanged before/after source/test fingerprints. Physical Codex navigation
  and dispatch remain unverified; no real Codex job was created by these checks.
- Normal scheduled-task stopping later left old assistant main and Ollama
  descendants running after the supervisor exited. Cleanup used exact verified
  PID/start-time/path/ancestry, not process-name-wide kills. A Windows lifetime
  job now owns compute descendants; explicitly launched user apps/brokers opt
  out through breakaway flags. Shared pre-existing servers are not adopted.
  Fifteen lifetime and 88 external-launch fixture checks passed. A standalone
  native Windows probe confirmed internal-child cleanup, external-child survival,
  unchanged parent environment and zero remaining test processes. This is not
  acceptance of Opera/Spotify/Codex UI behavior.
- Final normal-task adoption used accepted source `ce130ab`. The old task tree
  stopped with zero forced cleanup; the new task started once. Fresh microphone
  and brain-ready markers appeared, the temporary pause flag was removed, and
  logs confirmed listening resumed at 17:34:23 followed by capture health.
  Installed Ollama remains 0.23.4 with the original model/context 4096; both
  experimental routing flags remain false. The review server still returned
  200 and no Token Tracker process/startup was changed. This proves startup and
  capture activity, not spoken wake recall, semantic accuracy or real app effects.

### Installed-server native diagnostic

At source commit `4ac0931`, installed Ollama 0.23.4 and the configured
`qwen3:4b-instruct-2507-q4_K_M` were used without effects while listening was
paused. An explicitly separate preparatory request succeeded in 5.647 seconds,
with provider-observed prompt count 1310 at context 8192. That establishes
request compatibility, not a correct decision.

The primary probe then stopped after its first case in about 3.03 seconds:
zero passed, one failed, one attempted. The execution worker's in-memory filter
used the wrong result-field name and discarded the detailed case row. The exact
failure category and whether admission or inference failed are not recoverable
from that retained output. This is **not** a 0/31 accuracy score, nor evidence
for changing model weights. No retry or held-out run was performed.

The job-created pause flag was removed and fresh logs confirmed listening
resumed. Original model residency (`keep_alive=-1`) and context 4096 were
restored. The next diagnostic must retain the actual `failure_category` row,
distinguish cold preparation/admission/inference, and avoid attributing missing
evidence to semantic quality. Production remains a one-inference path; the
experimental token-count preflight must not enter its critical path.

### Frozen next packet after grouped review

Separate semantic quality, transport compatibility and audio ownership rather
than using more protocol tests as evidence for all three. For the first useful
single-action slice, keep evidence binding, step IDs and dependencies owned by
the application; ask the model only for the outcome and necessary operation
arguments. A factual-answer-only slice may use the same completed-turn entry
point with an explicit empty action set and no executor path. It must not turn a
failed action interpretation into an answer or claim an effect occurred.

Use one production inference request, preserving whole-turn authority and all
validation gates. Keep the exact token-count preflight experimental, outside
the production critical path. Retain the real result rows and measure short
answers, negative authority, unsupported requests and single actions separately.
Multi-step actions and always-listening promotion remain gated; do not tune to
held-out failures, add per-artist repairs, or launch another model concurrently.

### Speech-model research candidate

The official [Qwen3-ASR repository](https://github.com/QwenLM/Qwen3-ASR)
documents 0.6B/1.7B multilingual models including English and Hindi, native
Transformers support, and streaming through vLLM. The 0.6B model is a candidate
for an isolated recorded-audio comparison, not a proven replacement for the
current faster-whisper command recognizer. Windows dependency compatibility,
single-request latency, memory and proper-name accuracy remain unmeasured.
Do not alter the assistant environment, download/load another resident model,
or promise the published batched-throughput numbers as laptop latency.

### October 16 continuation and restart checkpoint

- Continue assistant work while preparing the requested Windows restart. Checkpoint
  workers before restart; reopening Codex is not proof a chat resumes automatically.
- Found a stale diagnostic-owned manual pause left after a worker quota interruption.
  Removed only that identified flag; logs confirmed listening resumed at 04:43:44.
  Added bounded probe-pause lease recognition; legacy/manual flags remain indefinite,
  malformed flags fail closed, and expiry never deletes another owner's file.
  This source change is not yet loaded by the running assistant. Future diagnostics
  must use the lease schema rather than an unbounded manual marker.
- Retained real concise-brain probe: 21/31 primary cases passed, all 32 transport
  calls succeeded including warmup; median primary latency 703 ms, p95 1829 ms.
  Semantic gate failed. General contract corrections are in progress; no promotion,
  held-out evaluation or wake-model retraining has occurred.
- Six-worker activity audit reconciled to 250 execution calls over 36 started turns;
  recorded input cache reuse was 95.52%. This is not measured plan-allowance cost.
  Findings support bounded outcomes, durable cleanup and lossless result retention,
  not more workers or claiming cache misses as the cause of quota consumption.
- Pause recovery: 21 runtime checks and one main-loop transition regression passed.
  Supervisor test configuration doubles were corrected to return absent settings as
  absent, with explicit assertions for its existing one-model/one-request limit.
- Concise-brain contract corrections are source-ready: 29 injected checks passed;
  prompt now distinguishes search, show/play, track/album restart and numeric sentinels.
  This is not real-model semantic acceptance; experimental routing stays disabled.
- Restart recovery helper has ten injected safety checks passing, including endpoint
  disconnects while waiting for Spotify and during resume. Current read-only readiness
  check confirmed SoundDrum P as default at 34%, unmuted, with Spotify playing.
  Registered current-user RunOnce value `SmartAlexaRecovery-20261016` to launch the
  bounded helper after the next login. Request expires October 16 at 11:30:35 UTC.
  It activates registered Codex, restores that exact endpoint's Windows master level,
  and resumes only retained Spotify context; no Spotify volume or routing changes.
  Windows login remains manual. Postboot app/playback recovery is still unverified.
- Registered one normal non-forced restart through the current user's interactive
  task `SmartAlexa Requested Restart 20261016`; it is not a recurring reboot.
  A single thread follow-up, `resume-assistant-after-restart`, is scheduled for
  05:15 IST October 16 to inspect recovery and continue this checkpoint. Local
  execution depends on Windows sign-in and Codex being available; scheduling is
  not proof that recovery or continuation has executed.

### Orchestration consolidation and external proposal review

Scope: four relevant saved-post excerpts and six existing child-worker traces,
not the entire X bookmark collection or a controlled model comparison. The reviewed
themes were durable asynchronous work (Harrison Chase), actionable interfaces
(elvis/OpenUI), trace-based evaluation (Divya Santhanam's linked LangChain article),
and selective memory (Harkirat Singh's Oracle-memory excerpt). Promotional claims
and unread linked material are not acceptance evidence.

Applied global instructions now require independently expiring diagnostic leases,
explicit ownership transfer when outcomes change, frozen ready files, durable raw
probe evidence, reconciled result counts, and outcome-based acceptance. Existing
rules already cover bounded Luna briefs, minimal sufficient delegation, exclusive
ownership, risk-based validation, evidence reuse and bounded output. The project
instructions reference those rules instead of duplicating them. No additional
platform, memory service or agent framework was installed. Actual savings remain
unmeasured; the observed 95.52% input-cache reuse does not establish total efficiency.

October 16 pasted remediation summary is a hypothesis list, not an executable
release plan. Verified against current source: literal request-span matching and
fixed cooldowns exist; these merit bounded failure-based diagnosis. Proposed
reservation reclamation must account for existing terminal emission and release;
blind decrement risks double-release. External-app job breakaway is already
implemented. VAD reset deliberately clears model and acoustic state; do not retain
old-stream context across a reset without establishing the actual defect.

Reject unmeasured 80% accuracy/70% token/100% wake-agreement promises. Reject
flipping async/conversation flags after unit tests, overlapping audio.py ownership,
early retry before a server's Retry-After, removal of all post-playback echo tail,
or permissive response-envelope parsing without a reproduced wire failure.
Keep valid directions: scoped implementation, gradual state separation, source-bound
normalization preserving negation/uniqueness, and tests tied to actual failures.
No application behavior changed as a result of reviewing this summary.

Startup observation after the user's return: no recovery log or state exists and
the RunOnce entry is still present. The restart task reports its command completed,
but this does not prove Windows restarted or the logon helper executed. Automatic
Codex opening and audio recovery therefore remain unverified; do not reboot again
or claim a diagnosed startup cause from these observations alone.

Next continuation: inspect recovery state/log and running task first, do not request
another restart. Verify lease-support source adoption. Run one frozen primary semantic
probe using the existing local model and durable raw results, with no tool effects;
restore resource state independently of LLM availability. Assess semantic results before
held-out acceptance or any live routing promotion. Speech-model inference and positive
wake collection remain separate pending lanes, not reasons to weaken the semantic gate.

### October 16 post-return assistant progress

- Orchestration/settings changes now require the user's explicit approval of the
  proposed list. No policy or model-setting edits were made in this continuation.
  The current saved global model setting was inspected: Luna/xhigh; changing it
  to a routine medium default is proposed, not applied.
- Corrected concise-brain evaluation retained cold preparation separately from
  warm-request deadlines. Cold preparation took 2812 ms; primary attempted 26/31,
  with 17 passed and 9 failed. Stopped at the first unavailable request (3015 ms);
  five cases were unattempted. Raw transport reported InterpreterUnavailable,
  while the case evaluator recorded deadline_exceeded. No held-out run or promotion.
  Report: assistant/logs/conversation_eval_primary_corrected_20261016.jsonl.
  Negative-authority failures included plans for quoted, reported and negated
  speech. These are release blockers, not cosmetic output-format differences.
- Worker confirmed all temporary model residency was cleared after the probe.
  Root then started the existing Memoir Assistant scheduled task; scheduler
  reported Running. Fresh logs subsequently confirmed microphone open, SoundDrum
  loopback attached, brain ready and listening at 05:20:24. Read-only provider
  state confirmed the configured brain resident entirely in GPU memory. Windows
  default remained SoundDrum at 34%, unmuted. Spotify SMTC returned no session
  state, so playback resumption is not confirmed; no playback/volume change sent.
  Both experimental routing flags remain false. This verifies runtime readiness,
  not spoken-command acceptance. About 1 GiB physical RAM was free at startup;
  defer concurrent Qwen-ASR loading until a separate resource-owned offline run.
- Qwen-ASR cache has the expected model/config/tokenizer/processor files, including
  one 1,564,928,088-byte weight file; exact expected-size/hash verification and
  actual inference remain pending. Its environment dependencies are discoverable.
  All 48 command and 16 negative WAV paths in the existing test set exist.
  This is metadata readiness, not measured speech quality or latency.
- Current review store: 157 negatives, four uncertain presence-only Alexa-heard
  rows from one group/session, zero exact-span confirmed positives. Historical
  enrollment is separate. No relabeling or retraining was performed.
- Removed only the exact unused SmartAlexaRecovery-20261016 RunOnce entry so
  it cannot unexpectedly launch apps/resume playback on a later unrelated login.
  Recovery source/request/logging data was preserved; no further reboot requested.

### Approved orchestration and wake evidence repairs

User approved the proposed orchestration changes, requested marginal-value Luna parallelism,
more deterministic lead/worker responsibilities, and power-user model-selection perspectives.
Applied global evidence-reuse/approval/phase-budget rules and conditional parallelism; fleshed
out sol-lead.md intake/diagnosis/contract/lane/review/release decisions and luna-scout.md
execution/escalation/result responsibilities. Saved default effort changed xhigh to medium;
Luna worker default remains medium. Parsed both TOML files successfully. Saved configuration
is not proof existing chats reloaded it. No bundled skill or approval/security boundary changed.

Two Luna lanes ran concurrently with disjoint file ownership. Findings: the listener archives
generic rejected speech, whereas review queues do not intentionally collect positive Alexa
utterances. Low-VAD/non-segmented misses cannot be recovered retroactively if no audio was
saved; retention/rate limits also reduce available clips. Existing exact-span review controls
cannot turn that diagnostic pool into positive training ground truth.

- Review server now defaults to a wake-evidence queue: explicit acoustic-unconfirmed captures
  and user-marked misses, not generic no-trigger/short/quiet snippets. All older queues remain.
  Accepted command recordings remain excluded until reliable wake provenance is exported.
  No label or training ground truth is inferred from scores, ASR or action success.
- Telemetry now distinguishes JSONL indexing success/failure from WAV saving; returned events
  and human summaries expose indexing failures and audio suppression reasons. Fixed-category
  thread-safe counters include saved-but-unindexed WAVs. Caps/windows/privacy are unchanged.
- Telemetry fixture suite: 27 passed, plus one added direct bool-result regression passed.
  Review suite: 97 passed/2 failed initially; lead corrected accepted-source guard and an
  explicit old-queue fixture, then 12 selected checks passed. Unaffected evidence was reused.
  No real audio labels, training or live wake accuracy acceptance follows from these checks.

Still needed: intentional bounded positive collection with capture-session provenance and
human-confirmed exact spans across independent train/holdout sessions. The repaired review
selection improves relevance but does not guarantee Alexa in every recording. No retraining
has started. Live deployment of this source batch is tracked separately below.

Deployment: stopped only the Memoir scheduled task, verified its main/supervisor
processes exited, then started it once. No force kill, reboot, Spotify volume or
playback command. Fresh logs confirm brain ready, microphone and SoundDrum
loopback open, and listening at05:34:08; this is not spoken-wake acceptance.
Started the loopback review server on port8767;
read-only state returned mode=wake_evidence,100 queued candidates,no store error,
and zero positive labels. No browser interaction or new human reviews performed.

Power-user rationale, not benchmark evidence: Peter Steinberger's firsthand
December2025 post emphasizes architecture, executable feedback and total rework,
and questions unnecessary orchestration machinery (steipete.me/posts/2025/shipping-at-inference-speed).
Theo's February2026 comparison, retrieved as caption transcript via Rosetta,
warns of over-thorough blocker/test loops; its old model comparisons are not
current Sol/Luna rankings. Tibo @thsottiaux's Sol/Luna fleet post
(x.com/thsottiaux/status/2088725163923984638) was accessible through an indexed
TwStalker mirror; original X fetch returned403. Treat as practitioner/vendor
opinion, not measured quota savings. No API prices or benchmark figures imported
from these anecdotes. Approved role choices remain our task/evidence-based design.

### Current continuation: explicit collection and cross-project ledger query

Assistant implementation: build user-initiated four-second Alexa collection into
the existing review page. Three disjoint Luna lanes own private expiring pause/ack
guards, a bounded pending-sample store, and browser/HTTP controls. No microphone
recording begins without the user's click, browser permission and live assistant
pause acknowledgement. Samples remain unknown until existing playback/hash/session
and exact-span review gates pass. No new model loads or auto-positive labels.

Side query is design-only: assess a durable cross-chat work ledger for Token Tracker.
Do not edit Token Tracker, migrate storage, message its lead or change continuation
policy without implementation approval. Recommended distinctions: item type versus
work state, source coverage versus extracted items, execution permission versus
readiness, and thread activity versus a live ownership lease. Preserve uncertain
imported work and quota interruptions rather than silently closing them. Token
Tracker's separate owner would implement storage/overview; this chat owns assistant
integration only after an approved shared contract.

Collection checkpoint: runtime/manual-pause separation and expiry-bound acknowledgement
helpers are implemented. Lead corrected expiry cleanup, overflow handling, ack binding,
and cancellation-before-ack; affected helper selection passed11 and main/helper selection
passed12, reusing unaffected evidence from the initial263 passing checks. Backend12 tests
pass, including real WAV/Telemetry/iter_candidates discovery. Lead caught missing indexed
audio_path before deployment. Browser controls/HTTP checks completed in the correction
packet below; real user microphone acceptance is still pending.

Restarted only the existing assistant task to load the guard. A bounded native probe
created one private8-second lease, observed the live assistant's acknowledgement and
released its own lease; no audio was recorded, no human label or synthetic dataset row
was created, and the manual pause flag was untouched. This proves the pause handshake,
not browser microphone recording, WAV upload, live wake accuracy or training acceptance.

Browser lane initial104 HTTP/review checks passed, but lead review found deployment
blockers: suspended AudioContext not resumed, microphone lifetime extending into upload,
old async cleanup able to clobber a new attempt, unbounded fetch waits, and lazy factory
creation races. Corrected packet from Build explicit recording controls uses per-attempt
resources/generation/cancellation, stop capture before upload, bounded requests, manager
creation lock, separate collection text log, and behavioral async/HTTP regressions.
These source blockers were corrected. Eight selected collection checks passed, including
Node mocked-browser cancellation/late permission/stopped-before-upload behavior and a
concurrent factory test. Worklet-side64,000-sample cap was exercised independently of
main-thread timers. Final receipt validation rejects missing/mismatched session/save/index
data; its targeted Node lifecycle check passed. Unaffected104-case evidence was reused.

Published by restarting only the verified owned review-server process. Live GET page and
state returned200 with explicit Record Alexa controls, bounded Worklet source, no store
error and zero positive labels. Assistant task remains Running. No real microphone capture,
synthetic production recording, human label, training run or Token Tracker edit occurred.
Next user trial: reload localhost8767, create a recording session, explicitly record,
then perform existing playback/exact-span review. Real browser microphone acceptance and
independent train/holdout recording occasions remain pending.

## Related source of truth

The architecture, ownership map and acceptance matrix live in
[`../../ASSISTANT_ARCHITECTURE.md`](../../ASSISTANT_ARCHITECTURE.md). This ledger tracks
assistant requirements and checkpoints; update statuses only when a new result is
available, with the evidence type and date stated.
