# SmartAlexa — active checkpoint

Updated 2026-10-27. Read on resume; material updates only. Plan of record: [STRATEGY.md](STRATEGY.md) (rev 2) + [CAPABILITY_CONTRACT.md](CAPABILITY_CONTRACT.md). The project is a git repo with uncommitted work (last commit 1053e14).

**Runtime state:** assistant stopped; startup task disabled since 18 Oct. `[runtime] runtime = "legacy"` default; core via `runtime = "core"` or `python -m assistant.core`. `[core] enabled_phases = ["1"]`, so only phase 1 is reachable.

## Accepted source (not live)

- **Core runtime** `assistant/core/**`: fail-closed adapter registry; allow-list whole-utterance grammar (old regex removed) with C1 exact-match for suspicious objects; typed receipts; guards before every mutation; wake-boundary trimming; free hosted brain (Cloudflare default, Groq alt) + grammar-gated fallback; outage switch `MEMOIR_CORE_PROVIDER_OUTAGE=1`; selfcheck lists capabilities. 1095+ core tests pass (lead rerun). Corpus: `tests/core/corpus/authority.yaml` (236).
- **Adapters (disabled by default):** music 1b (album/artist/playlist/queue/previous/restart), computer 3 (apps incl. .lnk→exe, Opera search, show song/album), codex 3 (needs `[core.codex] project_dir`), jam 4, knowledge 4 (Wikipedia; `[core] knowledge = false`).
- **Reviews:** Sol diff reviews on phase 1 (2 rounds). Phase 3 batch: Sol r1 (F1–F8) + r2 (F1/F3/F4/F5/F7 narrower, new F9: codex request representation) all fixed; lead verified every Sol counterexample (27 Oct). Debate closed; residual risk = free-text Codex withdrawals (denylist + first-person refusal + Codex-side preamble), checked by the live gate. Codex requests containing "I" as a word are refused by design.
- **Tests:** full `pytest tests`: only 45 failures, all out of scope (33 global_assistant/conversation_flow = delete-candidates, pre-existing; 12 token_tracker = separate product). Kept-module failures fixed 28 Oct (test-state leaks; spotify quota file read without mtime shortcut).
- **Trial harness** `python -m assistant.core.trials {phase1,phase1b,phase3,phase4} [--outage]`, `--combine`; scripts `docs/trials/*.yaml` (every line checked against the real grammar; 1b/4 thresholds provisional). Phase 1 truncated negative is now "Alexa, set the volume to"; grammar accepts "set the volume to N".
- **Phase 5 plan** `docs/research/PHASE5_PLAN.md`: Sol design review done 28 Oct (lifecycle + legacy-startup gates folded in). Two user calls, both "retain" recommended: manual CLIs/knowledge_retrieval; stored artifacts.
- **Wake:** bench + recorder; synthetic candidate `models/wake/candidates/20261027-synthtts-lessac/alexa_synth.onnx` (provisional; synthetic held-out recall 0.944 vs stock 0.778 @0.5). Not wired.
- **Cutover map** `docs/research/CUTOVER_MAP.md` (re-run before phase 5; `.env` excluded from the backup).

## Waiting on the user

1. Free keys (`CF_ACCOUNT_ID`, `CF_API_TOKEN`, optional `GROQ_API_KEY`) → `python -m assistant.core --eval`.
2. Phase 1 live trial (normal + outage) with normal apps open.
3. Wake Day A/B recordings → bench incl. the synthetic candidate.
4. Codex project dir for delegation; whether shells (cmd/PowerShell) may be opened by voice (currently allowed).

## Rules carried forward

Volume = Windows master only. $0. Only post-wake transcript leaves (plus Wikipedia for asked questions). No confirmations. Phase 5 deletion approved with a zip (no `.env`) + restore test.
