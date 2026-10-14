# SmartAlexa (Memoir)

SmartAlexa is a local-first personal assistant for voice-driven help on a Windows PC. **Memoir** is the historical name of this checkout and its existing package paths; the Python package and local folder are still named `assistant` and `A:\pisha\memoir`.

The intended assistant can combine everyday questions with bounded capabilities such as Spotify playback, Windows volume control, browser search/display, app opening, and explicitly authorized Codex tasks. The current implementation is a transition: the legacy Spotify and desktop adapters exist, while the central whole-turn assistant path is incomplete and gated. Do not infer that every capability works end to end from source code or tests.

## Current status

- `listener.async_enabled` and `conversation.enabled` are both `false` in `assistant/config.toml`. The new asynchronous capture and typed decision contracts exist, but the central capture/request manager is not a complete continuous-processing path. The legacy listener remains the configured route.
- The new general interpreter has not passed its installed-model routing and authority gate. Recent whole-turn evaluations of both 4B model tags (`qwen3.5:4b` and `qwen3:4b-instruct-2507-q4_K_M`) were inadequate; keep the feature flags disabled. These evaluations were synthetic model tests, not microphone, speaker, visible-effect, or live assistant acceptance.
- `native_decision_eval` is diagnostic, not a production acceptance result. Wikipedia retrieval is a foundation, not a complete spoken web-answer capability.
- Existing Spotify, Windows volume, browser and Codex adapters are legacy capabilities with prior bounded checks. Their presence does not prove current live runtime behavior. Explicit volume commands target Windows master volume; Spotify ducking changes only the Spotify Windows mixer session and restores its prior session level. It does not change Spotify Connect volume or Windows master volume.
- The wake-review UI can record human judgments with exact audio bounds. One-tap Alexa labeling remains “unsure”; do not infer a positive label from a detector or transcript. The last reviewed snapshot (13 October 2026) had zero confirmed positive Alexa groups suitable for training. Recheck before making current dataset claims. No reviewed label or model promotion is implied here.

The target architecture and acceptance gates are documented in [ASSISTANT_ARCHITECTURE.md](ASSISTANT_ARCHITECTURE.md); the assistant-only work ledger is [docs/ASSISTANT_PLAN.md](docs/ASSISTANT_PLAN.md). These documents describe planned work as well as implementation, so use their status labels and dated checkpoints rather than treating the target design as shipped behavior. The old root ledger is retained locally for the other product's active agents and is not published here.

## Setup

Use Python on Windows. From the repository root:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r assistant/requirements.txt
Copy-Item .env.example .env
```

Fill `.env` with the Spotify app credentials you control. `.env` is ignored by Git; never commit it or disclose its values. The assistant and its local dependencies may also need model files under `assistant/models/`; download only the models required by your chosen configuration, and review each model's license before use or redistribution. The openWakeWord code is Apache-2.0, while its bundled pretrained models are CC BY-NC-SA; see [openWakeWord licensing](https://github.com/dscripka/openWakeWord#license). Porcupine is not the configured wake backend; verify current key and license terms before any future trial.

Authorize Spotify once:

```powershell
.venv\Scripts\python.exe -m assistant.spotify --auth
```

Refresh-token validity follows Spotify's current behavior; the assistant does not invent an expiry interval. Spotify's provider quota protection is conservative: a confirmed `QUOTA_EXCEEDED` response is treated separately from ordinary rate limits, with a persisted scope quarantine rather than a guessed reset time. Local fallbacks remain available; clearing a quarantine is an explicit operator action. This is local protection, not Spotify's official quota and not coordination across independent processes.

List audio inputs and select the intended microphone explicitly in `assistant/config.toml` (`audio.input_device`); avoid `default`, which can select a Bluetooth headset profile and degrade both capture and playback. The configured Windows media-session and echo-reference adapters are included in `assistant/requirements.txt`.

```powershell
.venv\Scripts\python.exe -m assistant.audio --list
.venv\Scripts\python.exe -m assistant.main
```

The scheduled-task installer registers startup only; create `.venv` and install requirements first:

```powershell
.\scripts\install_task.ps1
```

## Data and validation boundaries

Keep `.env`, reviewed labels and audio/WAV recordings, model weights, logs, virtual environments, `.tools`, and personal notes out of commits. A private GitHub repository does not make secrets safe to commit. The app's existing logs and local data are operational/private data, not source material for a repository migration.

Offline tests, mocked contracts, synthetic model evaluations, and historical benchmarks establish different things. None alone proves live microphone recall, speech interpretation quality, visible browser/app effects, Spotify account behavior, Windows device behavior, or sustained resource use. Do not enable either gated route or claim trained wake behavior without the corresponding current acceptance evidence.

## Repository boundaries

- This repository is the SmartAlexa assistant. Its existing internal Python package remains `assistant/`.
- Token Tracker is a separate product. An independent checkout is prepared at `A:\pisha\token-tracker`; its active lead owns reconciliation and the final startup cutover. The original folder remains temporarily for its working agents and is excluded from this repository. Token Tracker is not part of SmartAlexa's runtime or deployment.
- Keep future general-assistant work under the central assistant design rather than adding another top-level router for each capability.
