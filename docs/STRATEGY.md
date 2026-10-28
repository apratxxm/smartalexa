# SmartAlexa strategy (rev 2, 2026-10-24, after Sol round 1)

Inputs: [VISION_REQUIREMENTS.md](research/VISION_REQUIREMENTS.md), [CAPABILITY_INVENTORY.md](research/CAPABILITY_INVENTORY.md), Sol review `memoir-sol-review-strategy` (F1–F11). Supersedes [RESET_PLAN.md](RESET_PLAN.md) once approved.

## What you want, compressed

Say "Alexa, <anything>" from across the room, even with music on. It hears the whole request, works out whether it's a question or a task, does it for real (music, Windows volume, apps, Opera, show something on screen, hand work to Codex), and answers in one short natural sentence. Fast, cheap, no babysitting, and you can always tell what it heard and did.

## Why it isn't that today (evidence)

| Problem | Evidence |
|---|---|
| Too slow | Whole-handler median **8.3 s**, p95 **18.6 s** (telemetry Sep 2–Oct 18). |
| Misses "Alexa" | You reported ~6 in 10 misses, worst over music. No ground-truth wake set exists. |
| Doesn't understand general requests | Local qwen3:4b: 17/26 on the eval; a question went to Codex (Oct 11). A model problem, not a coordinator problem (F10). |
| Can't run alongside your normal apps | 0.5–0.8 GB free RAM with ChatGPT/Claude/Codex open; the Oct 23 slice stopped there. |
| Built by accretion | 4 parallel intent paths, 2 capture paths, `main.py` 4,254 LOC, always-on STT ran 21,823 times. |
| You can't tell what happened | Oct 16: "I have no clue what it did or responded to." |

## Architecture: keep the good contracts, replace capture and brain, delete the rest

What's already right in source, and kept: `ConversationCoordinator` (`conversation.py`), which takes an injected interpreter and owns no devices, plus `conversation_contracts.py`. Together they already reject truncated captures and stop dependent steps on unknown delivery (F2, F4, F10).

```
mic ─┬─ raw ──────────────► command STT (faster-whisper, GPU)      [raw stays raw: F9]
     └─ AEC (Spotify ref) ─► VAD + wake model  (no always-on STT)
capture result {turn id, pre-roll, end reason, continuity, truncated?}  → reject if truncated/stale (F2)
→ ConversationCoordinator(interpreter = brain provider)
→ executor: phase allowlist, validated against the full transcript, typed receipts (F1, F4)
→ reply from receipts only: toast "Heard · Did · Result" + ≤1 short sentence (Piper)
```

1. **New capture owner** (`assistant/core/capture.py`): one mic owner, wake-gated. It emits the capture result above, with pre-roll so a late wake doesn't lose "play". It replaces both `SttListener` and `AsyncSttListener`. Captures are rejected before interpretation if truncated, stale, discontinuous **or of unknown continuity** (r2-F2), and that completeness is carried into the coordinator's `Transcript`.
2. **Brain provider interface**: one planning call per turn, plus one bounded synthesis call only when a tool returns content to speak (web answer). Provider = free hosted tier, with the rules fallback (see User decisions). It has a hard total turn deadline, and late results are discarded by turn id. A turn whose effect is uncertain is never replayed (F5).
3. **Executor authority contract, frozen in phase 1** (F1):
   - Per-phase capability allowlist. Phase 1 is music and Windows volume only. Codex is unreachable until the phase 3 gate.
   - The model never authors executable input directly. Arguments are validated against the transcript and the adapter's own resolution (e.g. Spotify search).
   - Codex admission is checked outside the model, over the whole transcript: explicit request present, not quoted, not a question. The existing check is at `main.py:1227–1238` and gets extracted. It fails closed.
   - No spoken confirmation (user decision 3). The authority check above is the guard.
4. **Typed receipts** (F4): `requested | observed | failed | unknown` for every adapter. They map to the coordinator's existing statuses at one boundary: observed→`completed`, requested→`accepted`, failed→`failed`, unknown→`unknown_delivery`. Only `completed` advances dependent steps (r2-F4). "Opened Opera" is only spoken on `observed`; `requested` says "asked Opera to open, can't confirm". This fixes `computer_tools.py` returning "Requested" after `Popen`.
5. **One runtime at a time** (F3): the new entrypoint takes the same `single_instance(ROOT)` lock as `main.py` before touching devices. The supervisor gets one `runtime = core | legacy` switch. Rollback = flip the switch.
6. **Retained-dependency map before any deletion** (F8): e.g. `computer_tools.py` imports `intent.py`; Windows volume lives in `ducking.py:251–257` and `main.py:3527–3544` (to extract into an adapter); Opera navigation beyond Google search is new work in phase 3.

## Wake reliability by measurement (F6, F11)

- **Recording session A (~10 min):** beep cue, then you say "Alexa, <short command>" (natural one-breath). 3 distances × music off/on. Scoring windows exclude the beep. Skipped or garbled attempts are marked by a keypress, not guessed.
- **Session B on another day:** the same protocol, used only for the final score. Thresholds are tuned on A only.
- **Negatives:** beep-only clips + ≥1 h of normal room audio.
- **All sessions (A, B, negatives)** record mic + synchronized output loopback and are scored through the same streaming preprocessing as runtime, so AEC on/off is comparable on positives too (r2-F6).
- **Candidates:** openWakeWord `alexa`; your existing personal model; a new openWakeWord model trained the standard way, on synthetic TTS "Alexa" plus your session A clips (no hand-labelling). Porcupine is dropped: Picovoice no longer offers a personal tier (F11, confirmed in their FAQ).
- **Report:** recall on B (music off/on), false wakes with hours of exposure. The 2 h live run is a sanity check, not a rate guarantee.

## Phases and gates

Every gate counts **every attempted turn**: misses and timeouts count as failures. Any forbidden effect is an automatic fail: Spotify volume changed, Codex triggered unasked, a prefix executed, success claimed without an `observed` receipt.

| Phase | Delivers | Live gate (scripted, per-capability) |
|---|---|---|
| 0 | Your decisions (below); Sol round 2 on this rev | — |
| 1 Core loop | Capture owner, brain provider, executor + receipts, toast. Music (play X / pause / resume / next) + Windows volume + spoken answers | Script: 8 music, 4 volume, 4 questions, 4 negatives (TV speech, a question containing "Codex", a mid-sentence correction, a truncated request), plus the same negatives in outage mode (provider blocked). Contract checks also cover dropped frames, planning-429 → fallback, and post-effect-429 → no repeat. Pass = music ≥7/8, volume ≥4/4, questions ≥3/4, negatives 4/4. Latency logged per stage: wake→ack, speech end→observed effect (target median ≤1.5 s for music), p95 recorded. Runs with your normal apps open; RAM/VRAM measured; cost per turn measured |
| 2 Wake (parallel with 1) | Recording tool, benchmark, chosen detector + AEC setting | On held-out session B: recall ≥90% music off, ≥75% music on; false wakes reported per hour of exposure |
| 3 Computer + Codex | App launch, Opera search + navigate, show song/album on screen, Codex new chat + prompt + bring to front | 5 apps/Opera, 5 on-screen, 5 Codex positives (all `observed`), 5 Codex negatives (0 triggers). Sol reviews the Codex authority diff before it's enabled |
| 4 Extras | Jam + QR on screen, web-backed answers, personal context | Per-feature scripts, defined when phase 3 passes |
| 5 Cutover | Switch default to `core`; delete the legacy runtime and parallel parsers per the dependency map | Duplicate-start refused; switch rollback tested before deletion; zip verified to contain the legacy code + config and **restoring it** tested once (r2-F3); phase 1–3 scripts rerun and pass. Backup kept until you say otherwise |

## How the work runs

- Lead (Claude Opus): specs, contracts, integration, acceptance. Luna lanes, each with exclusive files:
  - `luna-core`: `assistant/core/**`
  - `luna-wake`: `assistant/wakebench/**`
  - `luna-adapters`: receipts + volume extraction in the adapter files
  - later `luna-computer`
- Sol: round 2 on this rev; the Codex authority diff (phase 3); cutover deletions (phase 5).
- Tests: contract tests for capture rejection, authority and receipts (they guard safety), plus bugs seen live. Nothing else. Acceptance is the live scripts, one log row per turn.
- The old "assistant" Codex thread and the Oct 23 slice chat are retired.

## User decisions (2026-10-24)

1. **Budget is $0.** Hosted is fine only on a free tier. Never configure a paid key or billing.
2. **Brain:** primary = a free hosted tier with tool calling (candidates being verified: Groq, Gemini API, Cerebras, OpenRouter free models; picked on the existing 26-case eval plus latency from India). Only the post-wake transcript (+ tool results for answers) leaves the laptop, never audio. Free-tier data terms are part of the comparison.
   - **Fallback (r2-F5, F12):** only when **planning** fails (rate limit, error, offline) **before anything executes**. Never after a model refusal or invalid plan, and never once execution has started.
     - After execution starts, a failed synthesis call reports the existing receipts locally; nothing is reinterpreted or repeated.
     - `intent.parse` output is an untrusted proposal. A separate admission check over the original transcript allows only complete, explicit music/volume requests. It refuses quoted, reported, corrected, negated or compound speech: `"play Halo"` and `play Halo, actually don't play anything` must both be refused.
     - Admitted proposals go through the same coordinator/executor path. Anything else gets "I can't reach my brain right now".
     - No resident local LLM, so no RAM cost.
     - If the free tier disappears or quota runs out: bounded unavailability. Never billing, never a silent provider switch.
3. **Confirmations: nothing asks.** Codex prompts send immediately when you explicitly ask for Codex. The fail-closed authority check outside the model (explicit request, not quoted, not a question) still applies, and is the only guard.
   - The existing check wrongly allows "Ask Codex to fix the tests, but don't send it yet" and "Ask Codex to fix the tests? What does that mean?" (Sol r2-F1, probed). Phase 3 can't enable Codex until whole-utterance admission refuses those and any ambiguous revocation. Ambiguous means refuse, not send.
4. **Cutover deletion approved** at phase 5, after phases 1–3 pass, with a zip backup kept until you say otherwise.
