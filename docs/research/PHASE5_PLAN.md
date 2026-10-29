# Phase 5 cutover plan

**Review draft, 28 October 2026.** Plan only: no backup, restoration, switch, trial or deletion has been performed. Sol design review done 28 Oct (F1 lifecycle, F2 legacy startup proof: both folded in below; Sol judged deletion reasonable after these gates). Deletion requires phase 1–3 acceptance and backup gates in `docs/STRATEGY.md`. The checkpoint records phase 1-only enablement and outstanding live trials. Freeze and re-inventory after parallel lanes finish; do not substitute commit `1053e14` for uncommitted contents.

## Fresh static graph

The read-only graph parsed `assistant/**` and `tests/**`: **241 files, 241 module names, 608 unique edges, zero parse errors**, excluding models and bytecode. It reports immediate importers throughout the tree and the requested roots; it does not traverse those roots, so it is not a runtime reachability proof. `supervise -> main` is the legacy branch, not a core dependency. Test imports are coverage evidence, not production necessity.

Literal strings may overstate imports; constructed names, CLI entry points, config and filesystem references can be missed. `core.registry` constructs adapter names; `core.__main__` constructs selfcheck imports; `computer_tools` starts `assistant.spotify_jam` as a subprocess. Add those edges and inspect other dynamic loaders at execution. No importtime or runtime checks were performed for this plan.

Run from the project root (equivalent to the graph run, with a double-quoted PowerShell here-string):

```powershell
@"
import ast, pathlib, collections
roots=[pathlib.Path('assistant'),pathlib.Path('tests')]
files=[p for r in roots for p in r.rglob('*.py') if 'models' not in p.parts and '__pycache__' not in p.parts]
def name(p):
 s='.'.join(p.with_suffix('').parts); return s[:-9] if s.endswith('.__init__') else s
mods={name(p):p for p in files}; edges=collections.defaultdict(set); errors=[]
for mod,p in mods.items():
 try: tree=ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p))
 except (SyntaxError,UnicodeError) as e: errors.append((str(p),str(e))); continue
 package=mod if p.name=='__init__.py' else mod.rpartition('.')[0]
 for n in ast.walk(tree):
  targets=[]
  if isinstance(n,ast.Import): targets=[a.name for a in n.names]
  elif isinstance(n,ast.ImportFrom):
   base=package.split('.') if package else []
   if n.level: base=base[:max(0,len(base)-n.level+1)]; prefix='.'.join(base+([n.module] if n.module else []))
   else: prefix=n.module or ''
   targets=[prefix]+[prefix+'.'+a.name for a in n.names if a.name!='*']
  elif isinstance(n,ast.Constant) and isinstance(n.value,str) and n.value.startswith(('assistant.','tests.')): targets=[n.value]
  for target in targets:
   if target in mods and target!=mod: edges[mod].add(target)
rev=collections.defaultdict(set)
for source,targets in edges.items():
 for target in targets: rev[target].add(source)
starts=['assistant.core.__main__','assistant.supervise','assistant.main']+[m for m in mods if m.startswith('assistant.wakebench.')]
print('FILES',len(files),'MODULES',len(mods),'EDGES',sum(map(len,edges.values())),'ERRORS',errors)
for m in sorted(x for x in mods if x.startswith('assistant.') and x.count('.')==1):
 print(m,':',', '.join(sorted(rev[m])) or '(no static importer)')
print('EXPERIMENTAL:',', '.join(sorted(m for m in mods if m.startswith('assistant.experimental.') and any(s.startswith(('assistant.core.','assistant.wakebench.')) for s in rev[m]))))
print('ROOTS',len(starts),'wakebench modules',sum(m.startswith('assistant.wakebench.') for m in mods))
"@ | .\.venv\Scripts\python.exe -B -
```

## Module inventory

Names below are `assistant/<name>.py`. **Keep** retains the file. **Extract** retains the original until every retained consumer migrates and review passes. **Delete** is a gated candidate, not authorization. Importers are examples from the graph; package initialization and dynamic edges are noted separately.

| Module | Class | Importer / justification |
|---|---|---|
| `__init__` | keep | package initializer |
| `async_listener` | delete | `main`, `async_runtime` |
| `async_runtime` | delete | `main` |
| `audio` | extract | `core.capture`, `wakebench.capture` |
| `audio_jobs` | delete | `async_listener`, `async_runtime`; review `tests.test_telemetry_failures` |
| `benchmark_intent` | delete, conditional | no static importer; manual CLI unknown |
| `benchmark_synthetic` | delete, conditional | `tests.test_benchmark_synthetic`; manual CLI unknown |
| `brain` | extract | `experimental.brain_eval` via retained `core.evaluation` |
| `capture_requests` | delete | `foreground_capture` |
| `catalogue` | keep | `spotify` |
| `codex_bridge` | keep | `core.adapters.codex` |
| `computer_tools` | keep | `core.adapters.computer`, `core.adapters.jam` |
| `concise_brain` | delete | `main` |
| `config` | keep | `core.__main__`, `supervise` |
| `conversation` | extract | `core.engine` |
| `conversation_brain` | extract | `core.provider`; preserve current F9 dependencies |
| `conversation_contracts` | extract | `core.engine`, `core.authority`, `core.provider`, adapters |
| `desktop_controls` | delete | `main`, `global_assistant` |
| `diag` | keep | `brain`; retained evaluation |
| `diagnostic_archive` | delete | legacy diagnostics/tests; check dynamic references |
| `ducking` | keep | `feedback` |
| `echo` | extract | `core.capture`, `wakebench.bench` |
| `enroll` | delete, conditional | no static importer; manual CLI unknown |
| `evaluate` | delete, conditional | `tests.test_evaluate`; manual CLI unknown |
| `external_apps` | keep | `computer_tools`, `spotify` |
| `feedback` | extract | `core.__main__` |
| `foreground_capture` | delete | legacy main path and tests; verify dynamic references |
| `global_assistant` | delete | `main` |
| `global_commands` | keep | `global_router` |
| `global_router` | keep | `conversation_brain` |
| `health` | delete | `main` |
| `intent` | extract | `core.fallback`, `core.evaluation`, `computer_tools`, `conversation_contracts` |
| `intent_llm` | delete | `main`, `benchmark_intent` |
| `knowledge_retrieval` | keep, user decision | `tests.test_knowledge_retrieval`; future use unknown |
| `listener` | delete | `main`, `async_listener` |
| `local_media` | extract | `core.playback`, `spotify` |
| `main` | delete | `supervise` legacy switch; only after switch rollback window closes |
| `model_runtime` | keep | `conversation_brain` |
| `power` | delete | `main` |
| `process_lifetime` | keep | `supervise` |
| `report` | delete, conditional | `tests.test_report`; documented manual CLI |
| `resolution` | keep | `spotify`, `catalogue` |
| `runtime` | extract | `core.__main__`, `wakebench.record`, `experimental.wake_train.collection` |
| `spotify` | extract | `core.executor`, `core.playback`, `core.adapters.music` |
| `spotify_jam` | keep | `computer_tools` subprocess; test imports |
| `stt` | extract | `core.transcription` |
| `supervise` | keep/edit | startup task; legacy switch |
| `telemetry` | keep | `experimental.wake_train.collection`, `review_server` |
| `wake` | extract | `core.capture`, `wakebench.bench`, `wakebench.session` |

**Map differences:** `codex_bridge` is reached by the core Codex adapter, not future-only; `computer_tools` and `spotify_jam` are active core-path dependencies. `conversation`, `conversation_brain`, `conversation_contracts` and `spotify` remain live dependencies, including F9. Core evaluation is retained by lead decision, reversing the map recommendation. The map numbers **eight**, not seven, decisions; all are addressed below.

**Experimental boundary:** Keep every module in `assistant/experimental/**` pending future tooling decisions. Core evaluation imports `experimental.brain_eval`; wakebench training imports `experimental.wake_train.active_learning`, `candidate_training`, `machine_proposals` and transitively other wake-training modules including `collection`, `review_server` and `reviewed_training`. For remaining experimental files, manual entry points and dynamic consumers are not excluded; lack of a proven importer is not a deletion basis. Keep `core/**`, `wakebench/**` and their tests. Token Tracker is separately owned and never a deletion target.

## Ordered steps

**Process lifecycle (Sol phase 5 F1, applies to steps 2, 5 and every rollback).** The supervisor calls `load()` before each child launch, but `load()` is `lru_cache`d (`assistant/config.py`), so it keeps the runtime it first read until the supervisor process itself restarts (verified 29 Oct; Sol's F1 holds). Editing `config.toml` alone neither switches nor rolls back a running assistant, and after deletion an old supervisor would keep relaunching `assistant.main` (crash loop with backoff). Every switch, rollback and deletion therefore: (1) stops the owning supervisor and its whole child tree; (2) verifies every process has exited and the duplicate-start lock is released before any file is replaced; (3) restarts through the original launch path (the scheduled task's action, not an ad-hoc command); (4) verifies the running child is the intended module. The scheduled task's enabled state is left as the user set it. Test this lifecycle with a mocked child before step 2.


These are proposed future actions, not authorization to access devices, Spotify, network or Codex now. Before each step obtain applicable independent Sol review, freeze ownership and preserve exact pre-step contents. Stop on unexpected parallel edits. Record evidence without secrets.

1. **Backup and restore.** Files: user-approved zip and hash manifest, isolated scratch restore, task XML export. Back up actual pre-cutover `assistant/**`, config, `tests/**`, `pytest.ini`, launch scripts and any data/models later approved for removal; capture uncommitted files. Exclude `.env` and variants, credentials, `.venv*` and unapproved generated caches. **Check:** archive integrity, no secret entries, restored hashes against working-tree manifest, config and task XML; in isolated scratch with suitable dependencies verify offline legacy import, resolver and duplicate-lock refusal without devices. Missing prerequisites block the gate. **Rollback:** no production change; retain verified backup until user authorizes disposal; scratch cleanup requires user confirmation. Record restoration dependencies and secret prerequisites separately, without copying secrets.
2. **Switch default, retain both runtimes.** Files: `assistant/config.toml`, and if needed reviewed `assistant/supervise.py`, `assistant/core/__main__.py` and switch/lock tests. Account for the supervisor missing-key fallback to `legacy`, not only the TOML value. **Check:** isolated core/legacy/missing/invalid selection, duplicate start refused before device access in both cross-runtime orders, core-to-legacy switch rollback and supervisor launch wiring. Inspect task action and working directory without changing its state. **Rollback:** restore exact pre-step files and verify legacy resolution and unchanged task state.
3. **Repeat phase 1–3 trials.** Files: existing trial scripts and explicitly authorized result locations only. Freeze revisions; rerun phase 1 normal/outage, phase 2 held-out wake evaluation and phase 3 positive/negative scripts to `docs/STRATEGY.md` thresholds. The latter phases may not yet have ready harnesses; identify missing scripts or prior passes rather than inventing them. Obtain separate authority for live devices/provider/Spotify/Codex effects. **Check:** retained per-attempt results, thresholds, outage negatives, versions and restoration; all phases pass on core. **Rollback:** restore prior runtime/trial configuration, stop owned processes and restore prior device/task state. Failure or missing authorization blocks deletion.
4. **Extract in reviewed batches.** Files: extract rows, destination `core/**` or `wakebench/**`, direct consumers and corresponding tests; include preserved experimental consumers. Preserve evaluation, authority, receipts, dynamic selfcheck, adapters, lock and subprocess semantics. **Check:** refreshed graph including manual edges after each batch; focused regression and failure-path tests, offline selfcheck/eval wiring and any live acceptance invalidated by changes. Do not claim performance improvements without measurements. **Rollback:** restore complete pre-batch files/imports from an independently recorded snapshot, remove only files introduced by that batch, verify prior lock and switch behavior. Keep legacy runnable through this step.
**Legacy startup proof (Sol phase 5 F2).** Import, resolver and lock checks on the restored tree don't prove legacy can start: excluded secrets, sign-in state, model assets or lazy dependencies can be missing. Step 1 adds a restored-tree legacy startup regression with external effects mocked, plus a check that each excluded on-host prerequisite (`.env` keys, Spotify/Codex sign-in state, wake/STT model files, `.venv` packages) still exists outside the archive. Before step 5 closes the rollback window, a separately authorized legacy startup smoke check from the restored code must pass.

5. **Delete in dependency order.** Files: approved delete candidates and fully migrated extracted originals only; preserve keep rows and experimental tree. After explicit approval to close the switch-back window, remove legacy supervisor branch and `main` together, then legacy importers before dependencies; topologically order the actual refreshed graph, reviewing cycles as atomic groups. Retain ambiguous manual CLIs. Do not delete logs, caches, models or Token Tracker. **Check:** no references from retained code, tests, launch scripts or dynamic loaders; offline import/selfcheck, lock and focused regressions; rerun invalidated acceptance. **Rollback:** stop owned processes, restore full verified pre-cutover archive in a staged copy, reconcile later changes and restore in place only with authorization; verify legacy selection, lock and task state. Never overwrite newer data blindly. Switch-only rollback no longer works after this step.
6. **Prune tests last.** Files: `tests/**` only for actually deleted modules. Delete a test file only if every test exclusively covers deleted behavior; repoint or retain mixed tests including `tests/core/test_turns.py`, conversation, Spotify, wake and evaluation coverage. **Check:** map every removal to deleted-only behavior, run focused retained regressions and required repository checks, verify surviving tests have no deleted imports. **Rollback:** restore exact pre-pruning tests from the snapshot and restore required modules if needed; rerun affected checks.

## Decisions

1. **Intent — Sol review:** propose extracting `parse`, required types and `capability_domain`, proving authority and behavior equivalence; core fallback/evaluation and computer tools import it. Retain the original until every consumer moves.
2. **Spotify — Sol review:** extract only after specifying its complete transitive contract, not just the old six-name estimate; core music, executor and playback depend on it. Keep it whole if equivalence is unproven.
3. **Planning evaluation — decided:** keep it, including `experimental.brain_eval`, `brain` and `diag` until safely relocated.
4. **Telemetry/wake tooling — Sol and wake owner:** retain telemetry and wake experimental dependencies; `collection` and `review_server` import it. Removing the supervisor legacy server branch does not remove wakebench dependencies.
5. **Computer/Jam adapters — Sol review:** retain `computer_tools`, `codex_bridge`, `spotify_jam` and dependencies or extract only after proving equivalent adapter/subprocess behavior. Both adapters import `computer_tools`, which imports `intent`.
6. **Stored artifacts — user call:** recommend retaining `torch-wheels`, Qwen caches, AEC logs, `.assistant-baselines` and `config.toml.live-slice-backup`. Their use and size need rechecking; storage deletion requires a separate retention and restore decision.
7. **Backup secrets — decided:** exclude `.env` and variants. Restore requires independently available secrets and dependencies, not copied into scratch.
8. **Manual CLIs and knowledge — user call:** recommend retaining `enroll`, `evaluate`, `benchmark_intent`, `benchmark_synthetic`, `report` and `knowledge_retrieval` until the user confirms their future value. Static absence of importers does not prove a CLI is unused.

## Risks

- **Startup:** task and installer point to `assistant.supervise`; the map disabled-state observation is stale. Inspect actual task/action before cutover; do not implicitly enable or change it. Missing config currently defaults to legacy.
- **Wake/audio:** preserve models, voice assets, AEC, sample rates and shared lock; offline tests do not prove microphone or held-out wake behavior.
- **Config:** dynamic/nested reads defeat exact-key scans. Preserve unclassified keys and audit consumers before removing any; this plan does not authorize config pruning.
- **Telemetry/Token Tracker:** preserve telemetry and logs, inventory external readers; Tracker files, state and startup entries belong to a separate product and are outside scope.
- **Recovery:** an archive without secrets, environment or later user data is not a machine image. Test restore, record prerequisites, retain the archive and avoid overwriting newer data.
- **Dynamic and parallel changes:** re-run and extend the graph on the frozen execution revision and reconcile concurrent lanes. This snapshot alone is not deletion proof.
