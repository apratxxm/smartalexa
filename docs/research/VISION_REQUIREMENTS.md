# SmartAlexa vision requirements

This is a deduplicated record of user-stated goals and experience requirements, with short source quotes and timestamps. Times below are UTC as recorded in the JSONL sources. “Current state” distinguishes the user’s reported experience from what the project checkpoint says is implemented or live.

## 1. One-paragraph vision

The user wants a general, personal, hands-free assistant on their laptop: it should hear a complete request, understand whether they are asking a question or asking it to do something, use the appropriate capability, and quickly give a short, natural response that reflects what actually happened. The intended range runs from music and system controls to opening apps, searching and navigating a browser, showing content on screen, answering everyday questions, and explicitly handing work to Codex. The user has described a preference for local/off-cloud operation, but also expects useful web search and low-cost model/tool choices where appropriate. The assistant should feel responsive, reliable, and understandable without requiring the user to supervise each action.

## 2. Capabilities wanted

### Wake word and listening

- **What:** Hands-free, continuous/always-ready listening for “Alexa,” including while music plays and when the user is away from the laptop. Hear the actual wake word and full command; avoid requiring a second command after an acknowledgement. Use captured requests rather than missing the first words or listening for an unnecessarily long time.
- **Evidence:** “always listens to my voice and follows commands” (2026-09-15T07:40Z); “my main hurdle here is that it is not picking up very fast” and “a lot of hits and misses” (2026-09-10T07:19Z); “not waking up on most of my alexas” (2026-10-07T19:38Z); “why is it listening for so fucking long?” (2026-10-08T16:20Z).
- **Priority:** Repeated, high frustration. On 2026-10-08 the user reported misses at roughly six of ten attempts; on 2026-10-09 the user said wake accuracy over music and capture latency remained unresolved. Asked for retraining from recordings and missed wakes repeatedly (2026-10-06, 10-08, 10-09).
- **User-reported state:** Often misses “Alexa,” especially with music/loud playback; missed-word evidence and labeling were unreliable. On 2026-10-12 the user said even the clips they labeled Alexa were hard to distinguish from background music. User later asked whether there were any positive wake examples (2026-10-12T20:58Z), indicating the training evidence was still unclear. No user message establishes that wake reliability was ultimately fixed.

### Conversation, understanding, and answers

- **What:** A general-purpose brain should distinguish questions, commands, corrections, quoted speech, and requests for Codex; identify intent and the requested action/type; route appropriately; answer ordinary questions (including trivia and meanings) in a concise, natural voice. The user proposed that all stages work toward a predictable structured representation, while explicitly calling that a tentative idea to discuss.
- **Evidence:** “I want a general-purpose brain behind the assistant” (2026-10-06T09:32Z); “It was not a Codex request. It went to Codex. I was just asking for the meaning of a board” (2026-10-11T08:36Z); “answering trivia questions I get throughout the day” (2026-10-11T12:38Z); “lets talk about it first” after proposing a central brain and expected call format (2026-10-11T11:00Z).
- **Priority:** Core direction, made urgent by misrouting and irrelevant/robotic replies. On 2026-10-11 the user explicitly said the assistant’s speech had too much irrelevant content and was robotic.
- **User-reported state:** Simple music control had been the focus, but the user said it was “nowhere near that capability” and “we’ve focused too heavily on the Spotify aspect” (2026-10-11). Answering, retrieval, and general understanding remained a goal, not a user-confirmed working capability.

### Music, Spotify, and Jam

- **What:** Play/pause/resume/skip; play a named song, artist, or album; start an album or song from its beginning; show a requested song or album in Spotify; cope helpfully when an exact search result is unavailable. The user specifically suggested playing a useful first result rather than repeatedly saying sorry. Start a Spotify Jam and display its QR code so friends can join by scanning it.
- **Evidence:** “play any song that I tell it to” and “play the album from the start” (2026-10-06T09:32Z); “I wanted to actually get the first result and play that” (2026-10-06T04:27Z); “start a jam and show the QR” (2026-10-07T21:55Z).
- **Priority:** Frequent initial use case, but the user later said it had received too much project focus relative to general-assistant behavior. Spotify search/playback errors and bans were recurring frustrations (2026-10-06T17:09Z).
- **User-reported state:** Playback recognition and exact-title resolution were unreliable; the user said the Codex interaction missed words, took too long, and failed to show the created chat on screen (2026-10-06T04:51Z). Jam QR remains a requested capability; no user source confirms it works.

### Volume and audio devices

- **What:** “Change the volume” means Windows/system output volume. Spotify’s own volume must stay at 100%; do not alter Spotify volume when the user asks for system volume. Handle requests to raise/lower/set volume and account for the selected speaker. The user also asked about using the speaker microphone when playback is paused.
- **Evidence:** “I want you to change the fucking system volume, not Spotify. Do not touch Spotify. Keep Spotify at fucking 100%.” (2026-09-23T16:19Z); “also enable the mic of speaker when it is paused” (2026-09-23T16:56Z).
- **Priority:** Explicit hard boundary after volume commands changed the wrong control; unpleasantly loud spoken playback was also reported (2026-09-26T13:02Z).
- **User-reported state:** Volume commands had not reliably changed the intended target. The Sep 23 instruction is the clearest and latest explicit target rule found; no later source reverses it.

### Browser, apps, screen, and computer control

- **What:** General computer control: open an app; open an Opera tab and search; navigate to a page; bring up a chat, song, album, or other requested material visibly on screen. The user suggested computer-use tooling or Luna as possible means, while specifying outcomes rather than a particular implementation.
- **Evidence:** “open an app, I want that app opened” and “open a tab in Opera and search this” (2026-10-06T09:32Z); “opening an app and navigating to a page” (2026-10-11T12:38Z); “I asked it to pull up the chat on screen, and that also did not happen” (2026-10-06T04:51Z).
- **Priority:** Repeated as part of the all-capable assistant vision; visible outcomes matter, not just an internal action or claim.
- **User-reported state:** The user reported an attempted Codex chat was not visibly brought up and asked for general computer control. These capabilities were not confirmed working by the user.

### Codex delegation

- **What:** Explicitly ask the assistant to create a new Codex chat, switch tabs, or use Codex hands-free by sending a prompt. It should distinguish a genuine Codex request from a normal question and make the result/chat visible so the user can tell what happened.
- **Evidence:** “so it can talk to codex to do shit and send prompts so i can then use codex completely hands free” (2026-10-03T14:18Z); “It was not a Codex request. It went to Codex” (2026-10-11T08:36Z).
- **Priority:** Named repeatedly, including a concrete failure to create/show a requested chat. Authorization is explicit in the user’s request to use Codex; this does not imply that arbitrary speech or quoted material should be sent to Codex.
- **User-reported state:** The user asked whether Codex was working and reported missed words, excess delay, and inability to see the created chat (2026-10-06). No user message confirms end-to-end success.

### Memory and personal context

- **What:** Use what the assistant knows from prior conversations and personal context to understand what the user means and help shape the assistant around their actual preferences. The user also wants to understand the code and flow so the design reflects their ideas and improves their own proficiency.
- **Evidence:** “you know all of this about me from our chats, figure it out” (2026-10-11T08:51Z); “so that all of it is mapped in my brain and we actually implement the ideas and the structure i want” (2026-10-11T08:46Z).
- **Priority:** A stated working preference and collaboration goal. The user said they wanted architecture fully written out before implementation so they could judge the direction (2026-10-11T11:36Z).
- **User-reported state:** The user felt they had not understood what the system was doing and wanted the project architecture and flow made comprehensible. This is a project/process need as well as a product preference.

### Proactive behavior and automation

- **What:** The user wants a useful, hands-free assistant and expressed interest in doing work while away from the keyboard, but the sources reviewed do not establish a clear request for unsolicited suggestions or autonomous actions outside a spoken request. Requests for retraining, diagnostics, and orchestration are requests to the development agent, not proof that the product assistant should act proactively.
- **Evidence:** “music running, im away from keyboard go on” (2026-09-10T07:06Z); the user later framed the product as a personal assistant that should handle requests without requiring them to review its work during use (2026-10-11T08:36Z).
- **Priority:** Hands-free use matters; degree of unsolicited autonomy remains unspecified.
- **User-reported state:** No clear user-confirmed proactive/automation behavior in the product.

## 3. Experience requirements

- **Fast and responsive:** Wake-to-answer/action should feel “responsive and snappy” (2026-10-06T04:51Z). Delayed pickup and long capture were recurring complaints (2026-09-10; 2026-10-08). No user-defined numeric latency threshold found in the messages reviewed.
- **Natural and concise:** Speak “much more natural” and respond “in a very short span” (2026-10-11T08:36Z); avoid irrelevant explanations and robotic wording. Keep confirmations proportionate to what happened.
- **Whole-request handling:** Capture the full command, interpret it as a whole, and do not execute a partial prefix or confuse ordinary questions with Codex instructions. The user complained that the assistant did not hear the entire command (2026-10-06T04:51Z).
- **Clear outcome feedback:** The user must be able to tell what request was heard, what the assistant did, and what it answered; a chat/action should be visibly findable when requested (2026-10-06T04:51Z; 2026-10-16T09:29Z). The user explicitly does not want to review work while using the assistant (2026-10-11T08:36Z), so ordinary use should not require debugging or labeling.
- **False wakes and misses:** Reduce both missed “Alexa” attempts and unintended/false detections. Do not treat ambiguous audio as confirmed wake data: the user found saved clips to be random snippets and could not confidently distinguish some positives from music (2026-10-10T13:13Z; 2026-10-12T12:06Z).
- **What “working” means:** The requested action happens in the real target and the user gets a short, accurate acknowledgement; for questions, they get the requested answer. A transcript, valid-looking plan, or hidden chat alone is not the user-visible outcome they requested.

## 4. Constraints the user set

- **Local/cloud:** On 2026-09-15 the user said they wanted a laptop assistant “locally, off the cloud, probably” and “always” listening. “Probably” leaves cloud use open; later they requested Google search and computer-use/Codex options, so do not treat absolute offline-only as settled.
- **Cost/model choice:** On 2026-10-03 the user asked for “very cheap models unless i ask for” something else, and asked to run work that can use the GPU on the GPU. The product should not silently use a more expensive model when a cheaper one is adequate. They also expressed concern about RAM and resource use during the project.
- **Spotify/system audio boundary:** Keep Spotify volume at 100% and change Windows/system volume for volume requests (2026-09-23T16:19Z). This is explicit and should be treated as a hard user preference unless they change it.
- **Privacy and authority:** The user prefers local/off-cloud processing but did not state a detailed recording-retention rule in these messages. Codex use should be request-driven; a question or quoted text must not be mistaken for a Codex task (misrouting incident, 2026-10-11T08:36Z).
- **No review during use:** The personal assistant should not make the user review its work during ordinary interactions (2026-10-11T08:36Z). This coexists with the user’s separate desire to inspect and understand the code/design during development.
- **No invented wake labels:** The user asked for retraining and labeling when data existed, but later described the clips as ambiguous. That is not permission to label unclear snippets as positive examples.

## 5. Recurring frustrations with the process

- **Wake reliability and delay (Sep 10–Oct 12):** Repeated complaints about misses, especially over music; frustration that “most” Alexa attempts failed and one report estimated six misses out of ten. Asked for logs and retraining, then recognized the candidate positives were hard to distinguish from music. Do not claim a model is fixed based only on source changes or unverified/random clips.
- **Confusing relabeling workflow (Oct 9–10):** Asked for a simple audio play/pause control, Alexa/not-Alexa choices, reusable prior-label categories, and a free-text fallback (2026-10-09T11:32Z). Reported broken feedback, no previous/next navigation, a disabled Not Alexa option, incorrect saved labels, and slow saving (Oct 10). This was a specific tool used to help the project, not the desired day-to-day assistant experience.
- **Random snippets and uncertain labels (Oct 10–12):** Asked why saved reports looked like random snippets and whether they were missed wake words; later said Alexa examples were hard to separate from background music. The frustration is with unreliable evidence and labeling, not a request to label every ambiguous clip as Alexa.
- **Wrong routing and unnatural verbosity (Oct 11):** A request asking for the meaning of a board went to Codex; replies were irrelevant and robotic. User emphasized this is a personal assistant, not a Spotify-only command system, and should answer quickly.
- **Over-focus on Spotify / unclear architecture (Oct 11):** Asked to talk through system design and audit the code before implementing more, to regain control and understand what the code does. They felt the product was far short of the broader plan.
- **Too many tests and slow progress (Oct 13):** Asked why so many tests were being run while also wanting the full architecture, all routing, and a working assistant soon. These goals create pressure for visible end-to-end progress without substituting test volume for user-visible capability.
- **Unclear interaction reporting (Oct 16):** Said Ctrl+Alt+M incident capture was clunky and confusing and they could not tell what the assistant heard, did, or responded to (2026-10-16T09:29Z). This reinforces the need for simple, intelligible feedback.
- **Orchestration usage concerns (Oct 18):** Asked whether orchestration principles and worker reuse were being followed because usage was going quickly (2026-10-18T06:46Z). This is a development-process constraint, not a product feature.

## 6. Contradictions and open questions for the lead

1. **Local-first vs cloud/web:** “locally, off the cloud, probably” (Sep 15) versus later expectations for Google search and possibly computer-use services (Oct 11). Ask what may leave the device and under what user-visible consent.
2. **Hands-free convenience vs approval/visibility:** The user wants a capable assistant without reviewing each action during use, while also wanting clear visible results and explicit Codex use. Clarify which actions can execute directly and which should ask first; do not infer broad authority from “hands-free.”
3. **Wake phrase and confirmation flow:** User expects a single spoken request and complained about long listening and repeated prompts, but historical interactions include “Alexa” followed by an acknowledgement and then a command. Confirm whether both one-breath commands and wake-then-command are desired; prioritize the whole request either way.
4. **Wake data and retraining:** User asked the agent to label audio, but their later judgment says some candidate Alexa clips cannot be distinguished from music. Confirm acceptable positive-data evidence and whether uncertain clips should remain unlabelled.
5. **Model cost/performance boundary:** “Very cheap models unless I ask otherwise” and GPU use are preferences, but no specific budget, model tier, or acceptable online fallback is stated.
6. **Volume wording:** Sep 23 explicitly says any volume command means system volume and Spotify stays at 100%. No later contrary request found. Confirm only if product language needs to distinguish system, speaker hardware, and Spotify volume.
7. **Proactive scope:** Away-from-keyboard development instructions do not establish permission for the product assistant to initiate actions or notifications. No clear proactive feature request was found in the sources reviewed.

## 7. Coverage

- **Read:** Main assistant thread, segments 1–4, in chronological source order. Segment 1: `rollout-2026-09-02T21-39-37-01a062e2-11a3-7223-8315-df7227b40aa9.jsonl.zst` (2026-09-02 through later continuation entries). Segment 2: `rollout-2026-10-02T20-25-07-01a062e2-11a3-7223-8315-df7227b40aa9_01a0fd1c-a133-71b2-b08d-75265bbc14f5.jsonl.zst` (2026-10-02 onward). Segment 3: `rollout-2026-10-08T03-27-35-01a062e2-11a3-7223-8315-df7227b40aa9_01a1185f-36bb-7483-836c-68b6c19c253d.jsonl.zst` (2026-10-07 onward); existing `seg3_users.txt` reused. Segment 4: `rollout-2026-10-16T04-39-30-01a062e2-11a3-7223-8315-df7227b40aa9_01a141d3-ee6f-78a1-99bc-f7aefd46fe18.jsonl` (2026-10-16 onward); existing `seg4.txt` reused and source timestamps checked.
- **Read related assistant-specific threads:** `rollout-2026-09-15T13-10-37-01a0a402-bb2e-7382-b68d-841d5895a260.jsonl.zst` (“Plan local voice assistant”; 2026-09-15); `archived_sessions/rollout-2026-10-16T17-27-03-01a14492-a5a8-7620-b1d5-357458af8e9a.jsonl.zst` (“New Realtime Voice Chat”; no additional user vision requirements found); `archived_sessions/rollout-2026-10-11T14-02-53-01a12a17-ee75-75d2-bf13-88846d580a7e.jsonl.zst` (“Memoir voice task”; user asked “What is the meaning of virtuoso?”).
- **Ignored:** User-provided ambient UI, heartbeat/automation payloads, pasted environment/AGENTS blocks, and assistant-authored claims as requirement evidence. Assistant messages were used only to resolve context where needed. Architecture/current implementation cross-check: `ASSISTANT_ARCHITECTURE.md` target section, `docs/ASSISTANT_PLAN.md`, and `docs/RESET_PLAN.md`; these are not treated as user utterances.
- **Decoder:** First `pip --target` attempt could not reach a package index (“No matching distribution found”). The existing project `.venv` already had `zstandard`, so all specified compressed sources decoded successfully without changing project environments. The requested scratch path `.cache/research-tmp` could not be used because `.cache` is an existing regular file; extraction was performed in memory, and no scratch files were created.
- **Limits:** This is a faithful synthesis of user messages in the supplied source set, not a claim that every Codex chat across the account was searched. Current-state statements are user-reported unless explicitly labeled as a project-checkpoint cross-check.
