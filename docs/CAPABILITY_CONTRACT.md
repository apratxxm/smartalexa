# Capability adapter contract (draft for Sol design review, 2026-10-27)

Purpose: let phase 3–4 capabilities and more music actions be built in parallel lanes without editing core files, while keeping phase 1's guarantees (authority, receipts, deadlines, allowlist) in one place.

## Shape

- Capability names and argument schemas stay exactly as in `assistant/conversation_contracts.py::CAPABILITIES`. No new names without a contract change.
- Each adapter lives in `assistant/core/adapters/<area>.py` and exports `ADAPTER = Adapter(area, phase, handlers={capability: Handler})`.
- `Handler` (protocol):
  - `mutating: bool`
  - `license(clause: Clause) -> dict | None`: given ONE clause that core's shared authority has already found to be an un-negated, non-question, non-reported, non-quoted, non-compound imperative, return the arguments that clause licenses for this capability, or `None`. Pure and deterministic, with no I/O.
  - `execute(args: dict, guard: Callable[[], float]) -> Receipt`: `guard()` returns the remaining seconds and raises `TurnExpired` once the turn is stale. The adapter MUST call `guard()` immediately before every external mutation and bound every wait by it.
    - Returns `observed` only when the effect is verified (window/state check).
    - `requested` when dispatched but unobservable.
    - `unknown` when a mutation may have been sent but verification failed.
    - `failed` only when it's proven nothing was dispatched.
    - Never retries a possibly dispatched mutation through another transport.
  - `reply(receipt, args) -> str`: one short sentence, application-owned. Never echoes model text.
- `Clause` = `{text, verb, object, polite: bool}` produced by core's shared authority (`authority.py`).

## Authority method (lead decision, replaces the regex licensing)

- **Allow-list grammar, not deny-list regex.** After light normalisation (lowercase, Whisper punctuation and filler "please/um/uh" stripped at the ends only), the WHOLE utterance must parse as exactly one command: `[polite-prefix] VERB-PHRASE [object] [polite-suffix]`.
  - `polite-prefix` ∈ {can you, could you, would you, will you, please};
  - `polite-suffix` ∈ {please, thanks, thank you}, optionally preceded by a comma ("Pause, please.");
  - normalisation strips end punctuation (. ! ?) before parsing, so "can you play Halo?" is a polite request. Questions are refused because they don't parse ("what does pause mean", "should I open Spotify"). Only `codex_request` additionally refuses a trailing "?" (lead, 27 Oct);
  - each adapter contributes its verb phrases and an object rule (free text, number, or none).
- Anything that doesn't parse completely is refused. No partial matches, no clause splitting, no corrections. Negation, questions, reported speech and compounds are refused **because they don't parse**, not because a pattern detects them.
- Object rule for free text (song/artist/app/query): it may contain any words except a second verb phrase from any enabled capability at a word boundary preceded by a connector (and/or/then/&/+/,/;). An object of only stopwords is refused.
- **Codex has its own grammar:** `(ask|tell|get) codex to <request>`, `have codex <request>` or `(send|give) codex <request>`. (The `codex, <imperative>` form was dropped on 27 Oct as too easy to trigger by accident.)
  - The request must not end in a question mark and must not contain a revocation/hedge ("don't send", "not yet", "wait", "never mind", "or not", "maybe", "should I", "what does").
  - It must be ≥3 words.
  - The model must also have planned `codex_request`.
- The fallback uses the same grammar, restricted to phase 1 music + volume.
- The labelled corpus (positive and negative, ≥60 utterances incl. every Sol probe so far) is the acceptance test for the grammar.

## Core responsibilities (core lane only)

- `assistant/core/registry.py`:
  - discovers `assistant/core/adapters/*.py`;
  - enables a capability only if its adapter's `phase ∈ [core] enabled_phases` (config, default `[1]`) AND it's listed in `[core] enabled_capabilities` (default: phase 1 set);
  - refuses duplicate capability registrations at import.
- Engine: for each step, `authority.find_clause(transcript)` → `handler.license(clause)` → arguments must equal the step's arguments after normalisation → `execute` → coordinator mapping (observed→completed …) unchanged.
- Fallback stays phase 1 music + volume only, whatever is enabled.
- `answer`/`clarify` effect-claim guard unchanged and applied to all capabilities' effect verbs (adapters export `effect_words`).

## Special rules

- **Window-visible observation helper** `assistant/core/observe.py` (core lane): wait (bounded by guard) for a top-level window whose process image and/or title matches an expected pattern to exist and become foreground. Used for `observed` by app/Opera/Spotify-display adapters.
- **`codex_request`:**
  - `request` = the clause object (the text after the Codex verb phrase), verbatim from the wake-trimmed transcript; the model's text is ignored. (Revised 27 Oct by the lead: the object form keeps the instruction text clean and is what the adapter implements.)
  - Admission uses a stricter, separate whole-utterance check (Astra/Sol to settle the method). It must refuse revocations ("…but don't send it yet"), question suffixes, hedges and ambiguity.
  - No confirmation (user decision). Disabled until its own Sol diff review passes.
- **`open_app`:** only names in an allowlist built from the installed Start-menu catalogue (existing `external_apps.py` catalogue). Never paths, URIs or commands.
- **`search_browser`:** Opera only; the query is URL-encoded into a fixed search URL; no arbitrary navigation in this phase.
- **Answers with retrieval** (phase 4): a `knowledge` adapter is non-mutating. Retrieved text goes to one bounded synthesis call; sources are named in the toast.

## Review dispositions (lead, standing in for Sol under the Codex limit, 27 Oct)

- **C1:** a free-text object containing an enabled verb phrase, a negation or a correction word ("don't", "not", "actually", "no", "cancel", "wait", "stop"…) is licensed only if resolution finds an **exact** title/name match (case- and punctuation-insensitive) before any mutation. Otherwise refuse. ("Don't Stop Believin'" plays; "Halo actually play Hello" is refused.)
  - Conditional words count as suspicious too (lead, 27 Oct): if, unless, until, when, whenever, once, "as soon as", "in case". So "play Halo if I ask" is refused, while "When I Was Your Man" plays on an exact title.
  - Mechanism (lead, 27 Oct): core sets `Clause.suspicious`. For a suspicious clause on a mutating handler, core calls the handler's optional `exact_match(args, guard)` before `execute`. Missing method or False → refuse. Arguments are unchanged.
  - **Exemption:** `search_browser` declares `c1_exempt = True`: search only opens a tab and has no exact source. It still refuses any other mutating capability's verb phrase anywhere in the query (implemented in `computer.py`). No other handler may be exempt without a contract change.
- **C2:** core builds the verb→capability map from the adapters' `verbs` at registration. The same verb phrase on two capabilities refuses startup unless their `object_kind` or a required object keyword differs ("show the album X" vs "show X"). A clause is passed only to the handler(s) whose verb matched.
- **C3:** `observe.py`:
  - snapshot top-level windows (hwnd, pid, image, title) before dispatch;
  - `observed` requires a **new** matching window, or a foreground transition after dispatch to a window of the expected image;
  - for Opera search, the foreground Opera window title must contain the query's first ≥2 content words;
  - otherwise `requested`.
- **C4:** `open_app` launches only catalogue entries that are packaged apps (AUMID containing `!`) or shortcuts resolving to an `.exe`. Script/URL shortcuts (`.bat .cmd .ps1 .vbs .js .url`, or a non-exe target) are excluded from the catalogue.
- **C5:** Codex delegation uses the bridge's existing transport with a fixed project directory from config and never a `danger-full-access` sandbox. The sandbox is no stronger than the user's Codex default.
- **C6:** registry fails closed: an unknown capability in config, a duplicate registration, a verb conflict or an adapter import error makes `python -m assistant.core` refuse to start (selfcheck reports why).
- **C7:** adapter replies use fixed templates with only validated argument values. Helper or process messages never reach speech or the toast.
- **C8 (user):** knowledge answers send the trimmed question to Wikipedia's API as well as the brain. Assumed OK because it's explicitly requested content; the user can veto.

## Shared base

`assistant/core/adapters/base.py` (lead-owned) defines `Adapter`, `Handler`, `Clause`, `Guard`, and re-exports `Receipt`/`TurnExpired`. Adapters import it; nobody else edits it.

## Phases

- Phase 1 (enabled): play_song, pause, resume, next_track, set_volume, change_volume, answer.
- Phase 1b (music extensions, enable after live phase 1): play_artist, play_album, play_playlist, queue_song, previous_track, restart_track, restart_album.
- Phase 3: open_app, search_browser, show_track, show_album, codex_request.
- Phase 4: start_jam (+ QR on screen), knowledge answers.
