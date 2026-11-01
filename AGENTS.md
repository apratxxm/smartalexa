# Validation delegation

## User-directed GPT-6.1 Terra medium swarm capacity

Follow global Delegation and ownership. The user's direction sets no extra fixed worker cap;
actual runtime limits still apply. Assign distinct work only when independent and non-overlapping;
reuse or queue workers at limits. Do not duplicate lanes or rename existing chats.

## Distinct and stable worker names

Follow the global naming and continuity rules; names and task identities remain stable through
continuation, restart and messages.

## Prompt-dump tracking

Use docs/ASSISTANT_PLAN.md as the compact active checkpoint (target <=400 words).
Read it once on resume or when context is missing/changed, not on every prompt,
status request or worker return. Read cold history only for a specific unresolved
detail. Update only material new requirements, decisions, blockers, accepted evidence
or interruption checkpoints; replace current item state instead of appending narration.
Keep source/mock/live evidence distinct and preserve unresolved work when compacting.
No extra model calls merely to refresh tracking. The root ASSISTANT_PLAN.md is another
product's historical ledger; Token Tracker remains separately owned. Briefly account
for multi-part requests without repeating supplied context or resolved points.

Follow global review gates, bounded reviewer packet/disposition, worker reuse/ownership and risk-based validation policy. Continue only within this project's authorized scope and checkpoint priority; keep proposed and approved work distinct. Preserve this project's proof limits:
keep source/test ownership exclusive and distinguish mocked checks from live microphone, speaker,
GPU inference, Spotify and Codex execution evidence.

# Claude Design imports

When a user provides a `claude.ai/design/p/...` project link:

- Try the configured `claude_design` MCP endpoint first: `https://api.anthropic.com/v1/design/mcp` with `/design-login` authentication.
- If that MCP is unavailable to the active Codex session or reports unknown/unsupported authentication, immediately use the already-signed-in Claude project in the in-app browser. Do not repeat generic MCP or login discovery.
- Open the supplied project URL exactly, select the requested file, read the full project file when allowed, and read every explicitly named import such as `support.js` through the project file tree/source view.
- Implement only the named design variant and preserve its interaction model, visual hierarchy, typography, spacing, and color constraints.
- Never expose the user's Claude email, organization ID, auth state, cookies, or local credentials in code or output.
- State accurately whether the source came through the MCP or through the authenticated-browser fallback.

## Declared activity and worker briefs

- Start each worker brief with its stable 3–4 word display name, the explicit overall objective, and exclusive file ownership. The worker's first visible `[tracker]` snippet repeats the stable task assignment in `task`; keep that assignment unchanged across progress updates. Use the existing snippet at each turn or phase, not once per tool call.
- For `functions.exec` and shell calls that read or change files, begin the code/command with an activity declaration comment: JavaScript `// tracker-activity: {"kind":"reading","files":["relative/path"]}` or shell `# tracker-activity: {"kind":"reading","files":["relative/path"]}`. If `functions.exec` needs its first-line `// @exec:` directive, place the declaration immediately after that directive. Replace the example kind and paths with the attempted action and bounded relative file list (at most 16 paths, 260 characters each). Do not include secrets, raw results, or private transcript text.
- Allowed `kind` values: `reading`, `editing`, `searching`, `testing`, `running_command`, `inspecting`, `coordinating`, `checking_data`, `using_tool`, `waiting`, `reading_output`, `searching_web`, and `running_code`. An annotation describes only the attempted action; recorded tool calls, results, and lifecycle determine actual activity. Never infer success, model, context, or elapsed time from the annotation.
- Leave fixed-schema tools unchanged; use their supported fields and report their actual status when no activity field exists. Do not claim universal interception. Split mixed-action batches when practical; otherwise use `running_code` and avoid a misleading single-file label. Do not add per-call LLM commentary or nested CLI bridges.

## Exact question supersession

- A normal message does not resolve a pending async question. Actual structured
  replies resolve only their matching recorded question IDs.
- If the user answers several questions in one item's reply, and the lead accepts
  those answers or determines another item is obsolete, the lead may explicitly
  supersede the remaining exact items in an assistant progress message:
  `[tracker-questions]{"superseded":[["request_user_input_async","call_id",1]]}[/tracker-questions]`
- Use the actual observed tool name, call ID and zero-based question index. Never
  emit the example ID literally, guess an ID, or supersede unrelated questions.
  Include a short normal-language explanation when relevance changes.
- This changes the tracker's attention state only. It does not cancel or remove
  Codex-owned question cards, submit a user answer, or clear other calls' items.

## Context lifecycle

Follow `C:/Users/aprat/.codex/AGENTS.md`, section “Context lifecycle and Astra escalation,”
for all root chats, subagents, and handoffs in this project. Use only supported host
compaction behavior and reviewed handoffs.
