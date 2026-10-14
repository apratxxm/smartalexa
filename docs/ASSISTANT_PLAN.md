# SmartAlexa assistant requirements and status

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

- Recover or repeat the native real-model acceptance with a durable result and
  exact candidate/runtime identity; no accuracy conclusion is available from the
  absent handoff.
- Validate the disabled asynchronous idle integration on actual devices, then
  integrate the central capture-request contract into domain reply/dictation
  flows. Prior mocks and manager-only checks do not establish continuous processing.
- Recheck current reviewed wake data and confirm human-span/session provenance.
- Complete semantic, authority, tool-receipt, live-device and publication gates in
  that order. Production routing and model promotion remain gated until evidence
  supports them.

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

## Related source of truth

The architecture, ownership map and acceptance matrix live in
[`../ASSISTANT_ARCHITECTURE.md`](../ASSISTANT_ARCHITECTURE.md). This ledger tracks
assistant requirements and checkpoints; update statuses only when a new result is
available, with the evidence type and date stated.
