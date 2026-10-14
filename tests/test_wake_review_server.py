"""CPU-only human-review workflow and loopback HTTP boundary checks."""
from contextlib import contextmanager
import hashlib
import http.client
import json
from pathlib import Path
import socket
import shutil
import subprocess
import threading
import wave

import pytest

from assistant.experimental.wake_train import active_learning as review
from assistant.experimental.wake_train import review_server as ui


NOW = 1791300000.0


def test_source_categories_do_not_infer_wake_presence_or_use_scores():
    acoustic = {"candidate_kind": "acoustic_unconfirmed", "reason": "wake rejected", "scores": {"wake": .99}, "path": "x", "audio_field": "audio_path", "trigger_source": "wake_rejected"}
    miss = {"candidate_kind": "short_segment", "path": "x", "audio_field": "audio_path"}
    suppressed = {"candidate_kind": "below_vad", "path": "", "audio_field": "audio_path"}
    copied = {**acoustic, "trigger_source": "stt"}
    copied_retry = {**acoustic, "trigger_source": "retry", "audio_field": "retry_audio_path"}
    other = {"candidate_kind": "retry", "path": "x", "audio_field": "audio_path", "trigger_source": "retry"}
    assert ui.source_classification(acoustic)[0] == "unconfirmed_hits"
    assert ui.source_classification(miss)[0] == "possible_misses"
    assert ui.source_classification(suppressed)[0] == "possible_misses"
    assert ui.source_classification(copied)[0] == "all"
    assert ui.source_classification(copied_retry)[0] == "all"
    assert ui.source_classification(other)[0] == "all"
    assert "not proof that Alexa was spoken" in ui.source_classification(acoustic)[1]
    assert "not a verified wake" in ui.source_classification(other)[1]


def test_mode_filter_runs_before_recent_cap(monkeypatch, tmp_path):
    rows = [{"event_id": f"other-{i}", "audio_field": "audio_path", "path": "", "candidate_kind": "command", "trigger_source": "stt", "event_ts": NOW-i, "scores": {}} for i in range(5)]
    rows += [{"event_id": f"miss-{i}", "audio_field": "audio_path", "path": "", "candidate_kind": "below_vad", "event_ts": NOW-20-i, "scores": {}} for i in range(3)]
    monkeypatch.setattr(review, "iter_candidates", lambda *args, **kwargs: iter(rows))
    selected = ui.recent_candidates(tmp_path / "telemetry", tmp_path, tmp_path, now=NOW, limit=2, selection="recent", mode="possible_misses")
    assert [row["event_id"] for row in selected] == ["miss-0", "miss-1"]


def test_default_mode_and_mode_queries_are_strict(state):
    default = ui.ReviewState(state.telemetry, state.audio_root, state.project_root, state.store, now=NOW)
    assert default.mode == "possible_misses"
    assert default.summary()["candidates"] == []
    with running(default) as server:
        for query in ("?mode=all", "?refresh=1&mode=bad", "?refresh=1&mode=all&mode=all", "?refresh=1&x=1"):
            status, _, _ = request(server, path="/api/state" + query)
            assert status == 400
        status, _, body = request(server, path="/api/state?refresh=1&mode=all")
        assert status == 200 and json.loads(body)["mode"] == "all"


def test_played_label_survives_mode_change_and_undo_is_pinned(state):
    opaque = next(iter(state.candidates))
    played(state)
    state.refresh("possible_misses")
    assert opaque in state.candidates and opaque in state.played
    assert state.summary()["candidates"] == []
    state.label(label_body(state, id=opaque))
    token = state.summary()["latest_review"]["undo_token"]
    state.retract({"undo_token": token})
    state.refresh("possible_misses")
    cards = state.summary()["candidates"]
    assert len(cards) == 1 and cards[0]["id"] == state.pinned_id
    assert "Pinned undone response; outside the selected queue" in cards[0]["source_explanation"]


def test_summary_separates_four_heard_from_span_ready(tmp_path):
    archive = tmp_path / "audio"
    archive.mkdir()
    telemetry = tmp_path / "telemetry.jsonl"
    events = []
    for index in range(4):
        clip = wav_file(archive / f"heard-{index}.wav", sample=index + 1)
        events.append({"event_id": f"heard-{index}", "ts": NOW-index,
                       "audio_path": str(clip), "candidate_kind": "short_segment",
                       "trigger_source": "near_miss", "capture_session_id": f"session-{index}"})
    telemetry.write_text("".join(json.dumps(row) + "\n" for row in events), encoding="utf-8")
    store = tmp_path / "reviews.json"
    for index in range(4):
        review.label_candidate(telemetry=telemetry, store_path=store, event_id=f"heard-{index}",
            audio_field="audio_path", presence="unsure", wake_presence="present",
            session_id=f"session-{index}", source_provenance="temporary test review",
            reviewer="test reviewer", confirm_reviewed=True, audio_root=archive, project_root=tmp_path)
    summary = ui.ReviewState(telemetry, archive, tmp_path, store, now=NOW).summary()
    assert summary["wake_heard_count"] == 4
    assert summary["span_ready_positive_count"] == 0
    assert summary["reviewed_counts"]["unsure"] == 4


def wav_file(path, *, samples=16000, sample=0, channels=1):
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(channels)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(int(sample).to_bytes(2, "little", signed=True) * samples * channels)
    return path


@pytest.fixture
def state(tmp_path):
    archive = tmp_path / "audio"
    archive.mkdir()
    clip = wav_file(archive / "recent.wav")
    telemetry = tmp_path / "telemetry.jsonl"
    telemetry.write_text(json.dumps({
        "event_id": "recent", "ts": NOW - 60, "audio_path": str(clip),
        "rejection_reason": "acoustic_unconfirmed", "wake_score": .99,
        "capture_session_id": "capture-real-session", "review_state": "unknown",
        "trigger_source": "wake_rejected",
    }) + "\n", encoding="utf-8")
    result = ui.ReviewState(telemetry, archive, tmp_path, tmp_path / "reviews.json", now=NOW)
    # Legacy workflow cases exercise the preserved all-source review contract.
    result.refresh("all")
    return result


@contextmanager
def running(state):
    server = ui.create_server(state, 0)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def request(server, method="GET", path="/api/state", body=None, headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
    try:
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()


def post_headers(server):
    return {"Origin": f"http://127.0.0.1:{server.server_port}",
            "X-Review-Nonce": server.review_state.nonce, "Content-Type": "application/json"}


def label_body(state, **changes):
    body = {"id": next(iter(state.candidates)), "presence": "negative",
            "wake_presence": "absent", "review_category": "room_sound",
            "review_note": "", "confirm_reviewed": True,
            "entire_clip_negative": True, **changes}
    return body


def historical_positive(state):
    """Seed a valid pre-simplification positive review for history checks."""
    candidate = next(iter(state.candidates.values()))
    return review.label_candidate(
        telemetry=state.telemetry, store_path=state.store, event_id=candidate["event_id"],
        audio_field=candidate["audio_field"], presence="positive",
        session_id=candidate["capture_session_id"],
        source_provenance="historical human review", reviewer="historical reviewer",
        confirm_reviewed=True, wake_start=.125, wake_end=.5,
        audio_root=state.audio_root, project_root=state.project_root)


def played(state):
    # Unit boundary only; HTTP tests establish actual server delivery below.
    opaque = next(iter(state.candidates))
    _, digest = state.audio(opaque)
    state.played[opaque] = digest


def test_browsing_never_labels_or_creates_store_and_page_has_no_fake_training(state):
    with running(state) as server:
        status, _, page = request(server, path="/")
        assert status == 200 and b"Training has not started" in page
        assert b"https://" not in page
        status, _, content = request(server)
        result = json.loads(content)
    assert status == 200
    assert result["reviewed_counts"] == {"positive": 0, "negative": 0, "unsure": 0}
    assert result["candidates"][0]["presence"] == "unknown"
    assert result["candidates"][0]["scores"] == {"wake_score": .99}
    assert result["candidates"][0]["session_seed"] == "capture-real-session"
    assert not result["readiness"]["training_ready"] and not result["training_started"]
    assert not state.store.exists() and not list(state.store.parent.glob("reviews.json*"))
    assert str(state.audio_root).encode() not in content


def test_review_page_has_simple_one_tap_labels_and_source_choices(state):
    card = state.summary()["candidates"][0]
    assert card["wav_sha256"] == hashlib.sha256((state.audio_root / "recent.wav").read_bytes()).hexdigest()
    with running(state) as server:
        status, _, page = request(server, path="/")
    assert status == 200
    for marker in (b"play", b"Alexa", b"Not Alexa", b"Song/music", b"Room sound",
                   b"Coughing", b"People talking", b"Other sound", b"textarea"):
        assert marker in page
    assert b'id="refresh"' in page and b"fetch('/api/state?refresh=1'" in page
    assert page.count(b"el('audio')") == 1
    assert b"aria-pressed','false'" in page and b"button[aria-pressed=\"true\"]" in page
    assert b"button:focus-visible" in page and b"classList.toggle('selected',selected)" in page
    assert b"Other sound (max 500 characters)" in page
    assert b'id="mode"' in page and b"Short / quiet speech" in page and b"Unconfirmed wake hits" in page
    assert b"not confirmed missed Alexas" in page
    assert "wake confirmation rejected it" in card["source_explanation"]
    assert b"el('button','Play')" in page and b"play.textContent='Pause'" in page
    assert b"Previous review" in page and b"Next review" in page
    assert b"Review ${index+1} of ${reviewable.length}" in page
    assert b"previous.disabled=index===0" in page and b"next.disabled=index===reviewable.length-1" in page
    assert b"selectedId=reviewable[index-1].id;load()" in page
    assert b"selectedId=reviewable[index+1].id;load()" in page
    assert b"audio.addEventListener('ended',maybeUnlock)" in page
    assert b"audio.addEventListener('timeupdate',maybeUnlock)" in page
    assert b"ranges=audio.played" in page and b"completePlaybackCoverage" in page
    assert b"button id=\"undo-review\"" in page and b"fetch('/api/retract'" in page
    assert b"s.latest_review.presence==='negative'?'Not Alexa':'Alexa'" in page
    assert b"Could not refresh review data; the current recording remains visible." in page
    assert b"could not be returned to the current review queue" in page
    assert b"Full recording heard. Not Alexa is now available." in page
    assert b'id="action-status"' in page and b"#action-status').textContent=status.textContent" in page
    assert b"MutationObserver" not in page and b"window.confirm" not in page
    for marker in (b"Mark start at player time", b"Mark end at player time",
                   b"Save confirmed Alexa span", b"I can clearly hear Alexa in this span",
                   b"confirm_alexa_span:true", b"start.type='number'", b"confirmation.type='checkbox'",
                   b"resource admission", b"independent sessions",
                   b"Optional span controls could not be initialized"):
        assert marker in page
    for removed in (b"#reviewer", b"confirm_session"):
        assert removed not in page


def test_review_page_skips_unavailable_and_untrusted_candidates(state):
    with running(state) as server:
        status, _, page = request(server, path="/")
    assert status == 200
    assert (b"candidates.filter(c=>c&&c.available&&typeof c.session_seed==='string'"
            b"&&c.session_seed.trim())") in page
    assert b"unavailable and ${ungrouped} missing trusted session skipped" in page
    assert b"No recordings can be reviewed right now." in page




def test_rendered_page_controls_execute_in_node_behavioral_harness():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is not installed")
    inline = ui.PAGE.replace("__NONCE__", "null").split("<script>", 1)[1].split("</script>", 1)[0]
    harness = r"""
const assert=require('node:assert/strict');
const listeners=[];
class Element{
 constructor(tag){this.tagName=tag.toUpperCase();this.localName=tag.toLowerCase();this.children=[];this.parent=null;this.attributes={};this.events={};this._text='';this.disabled=false;this.hidden=false;this.dataset={};this.value='';this.checked=false;this.paused=true;this.currentTime=0;this.duration=1;this.seeking=false;this.played={length:0,start(){return 0},end(){return 0}};this.classes=new Set();this.classList={toggle:(name,on)=>on?this.classes.add(name):this.classes.delete(name),contains:name=>this.classes.has(name)};}
 set textContent(v){this._text=String(v);this.children=[];} get textContent(){return this._text+this.children.map(c=>c.textContent).join('');}
 append(...nodes){for(const node of nodes){this.children.push(node);node.parent=this;}}
 replaceChildren(...nodes){const old=this.children;this.children=[];for(const observer of listeners)if(observer.target===this)observer.callback([{removedNodes:old}]);this.append(...nodes);}
 addEventListener(name,fn){(this.events[name]??=[]).push(fn);}
 async dispatch(name){for(const fn of this.events[name]||[])await fn({type:name,target:this});}
 async click(){if(this.disabled||this.hidden)return;await this.dispatch('click');}
 setAttribute(k,v){this.attributes[k]=String(v);}getAttribute(k){return this.attributes[k]??null;}
 querySelectorAll(sel){const out=[];const visit=node=>{for(const child of node.children){if(child.localName===sel)out.push(child);visit(child);}};visit(this);return out;}
 querySelector(sel){if(sel==='.status'){let found=null;const visit=node=>{for(const c of node.children){if((c.className||'').split(' ').includes('status')&&!found)found=c;visit(c);}};visit(this);return found;}return this.querySelectorAll(sel)[0]||null;}
}
class Audio extends Element{constructor(){super('audio');this.failNext=false;}setRanges(ranges){this.played={length:ranges.length,start:i=>ranges[i][0],end:i=>ranges[i][1]};}play(){if(this.failNext){this.failNext=false;return Promise.reject(Error('denied'));}this.paused=false;this.dispatch('play');this.dispatch('playing');return Promise.resolve();}pause(){this.paused=true;this.dispatch('pause');}}
const ids={};for(const id of ['refresh','refresh-status','summary','undo-review','undo-status','action-status','card','mode']){ids[id]=new Element(id==='card'?'main':id==='mode'?'select':'button');ids[id].id=id;}ids.mode.value='possible_misses';
const document={createElement:tag=>tag==='audio'?new Audio():new Element(tag),querySelector:s=>ids[s.slice(1)]};
global.document=document;global.window={confirm:()=>true};global.MutationObserver=class{constructor(callback){this.callback=callback;}observe(target){this.target=target;listeners.push(this);}};
const candidates=['one','two'].map((id,i)=>({id,event_id:id,audio_field:'audio_path',reason:'reason '+id+'; Short / quiet speech; may include room noise, songs or genuine missed wakes.',ts:1791300000-i,session_seed:'session',available:true,wav_sha256:'a'.repeat(64),duration_seconds:1}));
const state={candidates,reviewed_counts:{positive:0,negative:0,unsure:0},readiness:{data_eligible:false,resource_admission_checked:false,blocked_reasons:['missing train positive session groups','missing holdout positive session groups']},training_started:false,latest_review:null};
let failLabel=false;const requested=[],labelBodies=[];global.fetch=async(url,options={})=>{requested.push(url);if(url==='/api/label')labelBodies.push(JSON.parse(options.body));const response={ok:url!=='/api/label'||!failLabel,json:async()=>url==='/api/label'?(failLabel?{error:'temporary failure'}:{presence:'unsure'}):state};response.clone=()=>({json:response.json});return response;};
const source=require('node:fs').readFileSync(0,'utf8');new Function(source)();
const tick=()=>new Promise(resolve=>setTimeout(resolve,0));
async function waitFor(test){for(let i=0;i<50;i++){if(test())return;await tick();}throw Error('timed out waiting for UI render');}
const all=(root,tag)=>{const out=[];const visit=n=>{for(const c of n.children){if(c.localName===tag.toLowerCase())out.push(c);visit(c);}};visit(root);return out;};
const find=(root,text)=>all(root,'button').find(b=>b.textContent===text);
(async()=>{
 await waitFor(()=>ids.card.children.length>0);await waitFor(()=>find(ids.card,'Save confirmed Alexa span'));assert.equal(ids.card.children[0].tagName,'SECTION');assert.equal(ids.card.children[0].localName,'section');
 ids.mode.value='all';await ids.mode.dispatch('change');await waitFor(()=>requested.includes('/api/state?refresh=1&mode=all'));assert.equal(ids.mode.value,'all');
 assert.equal(ids.card.textContent.includes('Review 1 of 2'),true);
 assert.equal(ids.card.textContent.includes('Short / quiet speech; may include room noise, songs or genuine missed wakes.'),true,'source explanation is visible in the card');
 let previous=find(ids.card,'Previous review'),next=find(ids.card,'Next review');assert.equal(previous.disabled,true);assert.equal(next.disabled,false);
 const room=find(ids.card,'Room sound');await room.click();assert.equal(room.getAttribute('aria-pressed'),'true');assert.equal(room.classList.contains('selected'),true);assert.equal(all(ids.card,'label')[0].hidden,true);
 await next.click();await waitFor(()=>ids.card.textContent.includes('Review 2 of 2'));assert.equal(ids.card.textContent.includes('reason two'),true);previous=find(ids.card,'Previous review');next=find(ids.card,'Next review');assert.equal(previous.disabled,false);assert.equal(next.disabled,true);
 await previous.click();await waitFor(()=>ids.card.textContent.includes('Review 1 of 2'));await waitFor(()=>find(ids.card,'Save confirmed Alexa span'));assert.equal(ids.card.textContent.includes('reason one'),true);
 const audio=all(ids.card,'audio')[0],negative=find(ids.card,'Not Alexa'),play=find(ids.card,'Play'),spanSave=find(ids.card,'Save confirmed Alexa span');assert.equal(negative.disabled,true);assert.equal(spanSave.disabled,true);
 const spanInputs=all(ids.card,'input'),start=spanInputs[0],end=spanInputs[1],clarity=spanInputs[2];assert.equal(start.type,'number');assert.equal(start.min,'0');assert.equal(start.max,'1');assert.equal(end.max,'1');start.value='.125';end.value='.5';clarity.checked=true;await clarity.dispatch('change');assert.equal(spanSave.disabled,true,'checkbox and bounds do not bypass the served-playback gate');
 await play.click();await tick();assert.equal(play.textContent,'Pause');assert.equal(spanSave.disabled,false,'served playback, sample-aligned bounds and clear-span confirmation enable save');const markStart=find(ids.card,'Mark start at player time'),markEnd=find(ids.card,'Mark end at player time');audio.currentTime=.125;await markStart.click();assert.equal(start.value,'0.125');audio.currentTime=.5;await markEnd.click();assert.equal(end.value,'0.5');clarity.checked=false;await clarity.dispatch('change');assert.equal(spanSave.disabled,true,'clear-span confirmation is required');clarity.checked=true;await clarity.dispatch('change');audio.setRanges([[0,.3]]);audio.currentTime=.3;await audio.dispatch('timeupdate');assert.equal(negative.disabled,true,'partial playback must remain disabled');failLabel=true;await find(ids.card,'Alexa').click();assert.equal(negative.disabled,true,'failed Alexa save must not unlock Not Alexa');assert(ids.card.textContent.includes('temporary failure'));failLabel=false;
 audio.pause();assert.equal(play.textContent,'Play');await play.click();assert.equal(play.textContent,'Pause');audio.currentTime=.99;audio.setRanges([[.9,.99]]);await audio.dispatch('timeupdate');assert.equal(negative.disabled,true,'seeking near the end must not count as full playback');
 audio.setRanges([[0,.4],[.6,1]]);audio.currentTime=1;await audio.dispatch('timeupdate');assert.equal(negative.disabled,true,'a gap in played ranges must not count as full playback');
 audio.setRanges([[0,.5],[.52,1]]);audio.currentTime=1;await audio.dispatch('ended');assert.equal(negative.disabled,false,'continuous near-complete playback must unlock at ended');assert.equal(ids.card.textContent.includes('Full recording heard. Not Alexa is now available.'),true);assert.equal(play.textContent,'Play');
 audio.pause();audio.failNext=true;await play.click();await new Promise(resolve=>setImmediate(resolve));assert.equal(play.textContent,'Play');assert.equal(ids.card.textContent.includes('Playback failed.'),true);
 const alexa=find(ids.card,'Alexa');await alexa.click();assert.equal(ids['action-status'].textContent.startsWith('Saved as Alexa heard'),true,'save feedback must survive queue rerender');assert.equal(labelBodies.at(-1).presence,'unsure');assert.equal('wake_start' in labelBodies.at(-1),false,'one-tap Alexa remains presence-only');assert(ids.summary.textContent.includes('resource admission not checked'));assert(ids.summary.textContent.includes('independent sessions'));
 console.log('browser-page behavior passed');
})().catch(error=>{console.error(error.stack||error);process.exitCode=1;});
"""
    result = subprocess.run([node, "-e", harness], input=inline, text=True,
                           capture_output=True, timeout=10, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "browser-page behavior passed" in result.stdout


def test_http_refusal_names_missing_field_without_leaking_exception(state):
    played(state)
    next(iter(state.candidates.values()))["capture_session_id"] = ""
    body = label_body(state)
    with running(state) as server:
        status, _, response = request(server, "POST", "/api/label", json.dumps(body), post_headers(server))
    result = json.loads(response)
    assert status == 400 and result["error"]
    assert str(state.audio_root) not in json.dumps(result)
    assert not state.store.exists()


def test_last_48_hours_bounded_rank_is_not_ground_truth(state):
    original = json.loads(state.telemetry.read_text())
    events = [{**original, "event_id": str(i), "ts": NOW - i, "rejection_reason": "no_trigger",
               "wake_score": .99} for i in range(120)]
    events.extend([{**original, "event_id": "disagreement", "ts": NOW - 80,
                    "rejection_reason": "acoustic_unconfirmed", "wake_score": .02},
                   {**original, "event_id": "old", "ts": NOW - 48 * 3600 - 1},
                   {**original, "event_id": "future", "ts": NOW + 1},
                   {"event_id": "suppressed", "ts": NOW - 30, "candidate_kind": "below_vad",
                    "audio_reason": "audio_rate_limited"}])
    state.telemetry.write_text("\n".join(map(json.dumps, events)) + "\n", encoding="utf-8")
    rows = ui.recent_candidates(state.telemetry, state.audio_root, state.project_root, now=NOW)
    assert len(rows) == 100 and rows[0]["event_id"] == "disagreement"
    assert all(row["proposed_label"] == "uncertain" for row in rows)
    assert not {"old", "future"} & {row["event_id"] for row in rows}
    assert not state.store.exists()


def test_recent_selection_prefers_newest_while_default_keeps_score_priority(state):
    original = json.loads(state.telemetry.read_text())
    events = [{**original, "event_id": "older-high", "ts": NOW - 100,
               "wake_score": .99},
              {**original, "event_id": "newer-low", "ts": NOW - 10,
               "wake_score": .1}]
    state.telemetry.write_text("\n".join(map(json.dumps, events)) + "\n", encoding="utf-8")
    ranked = ui.recent_candidates(state.telemetry, state.audio_root, state.project_root,
                                  now=NOW, limit=1)
    recent = ui.recent_candidates(state.telemetry, state.audio_root, state.project_root,
                                  now=NOW, limit=1, selection="recent")
    assert ranked[0]["event_id"] == "older-high"
    assert recent[0]["event_id"] == "newer-low"
    with pytest.raises(ValueError, match="selection"):
        ui.recent_candidates(state.telemetry, state.audio_root, state.project_root,
                             now=NOW, selection="random")


def test_iter_candidate_filters_precede_audio_path_resolution_and_keep_duplicate_identity(state, monkeypatch):
    original = json.loads(state.telemetry.read_text(encoding="utf-8"))
    other = wav_file(state.audio_root / "other.wav", sample=3)
    duplicate = {**original, "audio_path": str(other)}
    unrelated = [{**original, "event_id": f"unrelated-{i}",
                  "audio_path": f"Z:/missing-{i}.wav"} for i in range(25)]
    state.telemetry.write_text("\n".join(map(json.dumps, [original, duplicate, *unrelated])) + "\n",
                               encoding="utf-8")
    resolved = []
    real_audio_path = review.audio_path
    def track(value, *args, **kwargs):
        resolved.append(str(value))
        return real_audio_path(value, *args, **kwargs)
    monkeypatch.setattr(review, "audio_path", track)
    rows = list(review.iter_candidates(state.telemetry, state.audio_root, state.project_root,
                                       event_id="recent"))
    assert len(rows) == 2
    assert {row["path"] for row in rows} == {str(state.audio_root / "recent.wav"), str(other)}
    assert resolved == [str(state.audio_root / "recent.wav"), str(other)]
    state.played[next(iter(state.candidates))] = review.wav_identity(state.audio_root / "recent.wav")["wav_sha256"]
    with pytest.raises(ValueError, match="candidate missing, pruned, outside archive, or ambiguous"):
        state.label(label_body(state, presence="negative", entire_clip_negative=True))
    assert not state.store.exists()


def test_recent_candidates_filter_time_and_cap_before_resolving_paths(state, monkeypatch):
    original = json.loads(state.telemetry.read_text(encoding="utf-8"))
    rows = [{**original, "event_id": f"recent-{i}", "ts": NOW - i,
             "audio_path": str(state.audio_root / "recent.wav")} for i in range(125)]
    rows.extend([{**original, "event_id": "old", "ts": NOW - 48 * 3600 - 1,
                  "audio_path": "Z:/old.wav"},
                 {**original, "event_id": "future", "ts": NOW + 1,
                  "audio_path": "Z:/future.wav"},
                 {"event_id": "no-audio-recent", "ts": NOW - 5, "candidate_kind": "miss"},
                 {"event_id": "no-audio-old", "ts": NOW - 48 * 3600 - 1, "candidate_kind": "miss"}])
    state.telemetry.write_text("\n".join(map(json.dumps, rows)) + "\n", encoding="utf-8")
    resolved = []
    real_audio_path = review.audio_path
    def track(value, *args, **kwargs):
        resolved.append(str(value))
        return real_audio_path(value, *args, **kwargs)
    monkeypatch.setattr(review, "audio_path", track)
    chosen = ui.recent_candidates(state.telemetry, state.audio_root, state.project_root, now=NOW)
    assert len(chosen) == ui.MAX_CANDIDATES
    assert len(resolved) == ui.MAX_CANDIDATES
    assert set(resolved) == {str(state.audio_root / "recent.wav")}
    assert not {"old", "future", "no-audio-old"} & {row["event_id"] for row in chosen}
    no_audio = list(review.iter_candidates(state.telemetry, state.audio_root, state.project_root,
                                          event_id="no-audio-recent", min_ts=NOW - 10, max_ts=NOW))
    assert len(no_audio) == 1 and no_audio[0]["event_ts"] == NOW - 5
    assert no_audio[0]["unavailable"] == "audio_not_archived"


def test_recent_candidates_never_trust_selected_invalid_path_and_filters_validate(state):
    original = json.loads(state.telemetry.read_text(encoding="utf-8"))
    outside = {**original, "event_id": "invalid-selected", "ts": NOW - 1,
               "audio_path": "Z:/outside.wav"}
    state.telemetry.write_text(json.dumps(outside) + "\n", encoding="utf-8")
    rows = ui.recent_candidates(state.telemetry, state.audio_root, state.project_root,
                                now=NOW, limit=1)
    assert len(rows) == 1 and rows[0]["event_id"] == "invalid-selected"
    assert rows[0]["unavailable"] and rows[0]["path"] == "Z:/outside.wav"
    for kwargs in ({"min_ts": True}, {"max_ts": float("nan")},
                   {"min_ts": NOW, "max_ts": NOW - 1}):
        with pytest.raises(ValueError):
            list(review.iter_candidates(state.telemetry, state.audio_root, state.project_root, **kwargs))


def test_explicit_refresh_adds_new_clip_preserves_played_id_and_label(state):
    opaque = next(iter(state.candidates))
    played(state)
    before = state.candidates[opaque]
    original = json.loads(state.telemetry.read_text(encoding="utf-8"))
    added = {**original, "event_id": "newly-appended", "ts": NOW - 10,
             "audio_path": str(state.audio_root / "recent.wav")}
    with state.telemetry.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(added) + "\n")
    assert "newly-appended" not in {row["event_id"] for row in state.summary()["candidates"]}
    state.refresh()
    assert state.candidates[opaque] == before
    new_id = next(key for key, row in state.candidates.items()
                  if row["event_id"] == "newly-appended")
    assert new_id != opaque
    assert state.label(label_body(state, id=opaque, presence="negative",
                                  entire_clip_negative=True))["presence"] == "negative"


def test_state_endpoint_refreshes_only_on_explicit_request(state):
    original = json.loads(state.telemetry.read_text(encoding="utf-8"))
    added = {**original, "event_id": "endpoint-new", "ts": NOW - 10}
    with state.telemetry.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(added) + "\n")
    with running(state) as server:
        status, _, body = request(server)
        assert status == 200
        assert "endpoint-new" not in {row["event_id"] for row in json.loads(body)["candidates"]}
        status, _, body = request(server, path="/api/state?refresh=1")
    assert status == 200
    assert "endpoint-new" in {row["event_id"] for row in json.loads(body)["candidates"]}


def test_refresh_excludes_reviewed_rows_before_queue_cap(state):
    original = json.loads(state.telemetry.read_text(encoding="utf-8"))
    played(state)
    state.label(label_body(state, presence="negative", entire_clip_negative=True))
    store = review.load_store(state.store)
    template = dict(store["reviews"][0])
    store["reviews"].extend({**template, "event_id": f"reviewed-{i}"}
                            for i in range(99))
    state.store.write_text(json.dumps(store), encoding="utf-8")
    rows = [{**original, "event_id": "recent", "ts": NOW - 100}]
    rows.extend({**original, "event_id": f"reviewed-{i}", "ts": NOW - 100 + i}
                for i in range(1, 100))
    rows.append({**original, "event_id": "fresh-unreviewed", "ts": NOW - 1})
    state.telemetry.write_text("\n".join([*map(json.dumps, rows), ""]), encoding="utf-8")
    state.refresh()
    assert len(state.candidates) <= ui.MAX_CANDIDATES
    assert "fresh-unreviewed" in {row["event_id"] for row in state.candidates.values()}


def test_refresh_keeps_queue_bounded_to_one_hundred(state):
    original = json.loads(state.telemetry.read_text(encoding="utf-8"))
    events = [{**original, "event_id": f"queue-{i}", "ts": NOW - i}
              for i in range(140)]
    state.telemetry.write_text("\n".join(map(json.dumps, events)) + "\n", encoding="utf-8")
    state.refresh()
    assert len(state.candidates) == ui.MAX_CANDIDATES


def test_refresh_drops_completed_played_rows_without_starving_new_clip(state):
    original = json.loads(state.telemetry.read_text(encoding="utf-8"))
    reviewed_events = [{**original, "event_id": "recent", "ts": NOW - 100}]
    reviewed_events.extend({**original, "event_id": f"reviewed-{i}", "ts": NOW - 100 + i}
                           for i in range(1, 100))
    state.telemetry.write_text("\n".join([*map(json.dumps, reviewed_events), ""]), encoding="utf-8")
    state.refresh()
    assert len(state.candidates) == ui.MAX_CANDIDATES
    wav_hash = review.wav_identity(state.audio_root / "recent.wav")["wav_sha256"]
    state.played.update({opaque: wav_hash for opaque in state.candidates})
    reviewed_id = next(opaque for opaque, candidate in state.candidates.items()
                       if candidate["event_id"] == "recent")
    state.label(label_body(state, id=reviewed_id, presence="negative",
                           entire_clip_negative=True))
    store = review.load_store(state.store)
    template = dict(store["reviews"][0])
    store["reviews"].extend({**template, "event_id": f"reviewed-{i}"}
                            for i in range(1, 100))
    state.store.write_text(json.dumps(store), encoding="utf-8")
    fresh = {**original, "event_id": "new-unreviewed", "ts": NOW - 1}
    with state.telemetry.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(fresh) + "\n")

    state.refresh()
    assert {row["event_id"] for row in state.candidates.values()} == {"new-unreviewed"}
    assert len(state.candidates) <= ui.MAX_CANDIDATES
    assert len(state.played) <= ui.MAX_CANDIDATES
    assert reviewed_id not in state.played


@pytest.mark.parametrize("change", [
    {"confirm_reviewed": False}, {"wake_presence": "present"},
    {"review_category": "made_up"}, {"review_category": None, "review_note": "  "},
    {"presence": "positive"}, {"presence": "negative", "entire_clip_negative": False},
    {"path": "outside.wav"}, {"command": "train"},
])
def test_invalid_human_labels_fail_without_creating_store(state, change):
    played(state)
    with pytest.raises((ValueError, TypeError)):
        state.label(label_body(state, **change))
    assert not state.store.exists()


def test_playback_required_and_changed_audio_cannot_reuse_review(state):
    with pytest.raises(ValueError, match="Play"):
        state.label(label_body(state))
    played(state)
    candidate = next(iter(state.candidates.values()))
    wav_file(candidate["path"], sample=1)
    with pytest.raises(ValueError, match="changed"):
        state.label(label_body(state))
    assert not state.store.exists()


@pytest.mark.parametrize("presence,wake_presence", [("unsure", "present"), ("negative", "absent")])
def test_one_tap_review_writes_metadata_and_never_trains(state, presence, wake_presence):
    played(state)
    body = label_body(state, presence=presence, wake_presence=wake_presence,
                      entire_clip_negative=presence == "negative")
    result = state.label(body)
    store = review.load_store(state.store)
    row = store["reviews"][0]
    assert result["presence"] == presence and not result["training_started"]
    assert row["human_reviewed"] and row["reviewer"] == "local-wake-reviewer"
    assert row["session_id"] == "capture-real-session" and row["capture_session_id"] == "capture-real-session"
    assert row["addressed_assistant"] == "unsure" and row["wake_presence"] == wake_presence
    assert row["wav_sha256"] == hashlib.sha256((state.audio_root / "recent.wav").read_bytes()).hexdigest()
    summary = state.summary()
    assert summary["reviewed_counts"][presence] == 1 and summary["candidates"] == []
    assert not summary["readiness"]["training_ready"] and not summary["training_started"]
    validated = review.validated_reviews(store, state.audio_root)[0]
    assert validated["wake_start_s"] is None and validated["wake_end_s"] is None


def test_confirmed_exact_alexa_span_round_trips_to_human_review_store(state):
    played(state)
    body = label_body(state, presence="positive", wake_presence="present",
                      review_category=None, review_note="", entire_clip_negative=False,
                      wake_start=.125, wake_end=.5, confirm_alexa_span=True)
    result = state.label(body)
    row = review.validated_reviews(review.load_store(state.store), state.audio_root)[0]
    assert result["presence"] == "positive" and result["training_started"] is False
    assert row["presence"] == "positive" and row["wake_presence"] is None
    assert row["wake_start_s"] == .125 and row["wake_end_s"] == .5
    assert row["wake_start_sample"] == 2000 and row["wake_end_sample"] == 8000
    assert row["addressed_assistant"] == "unsure"
    summary = state.summary()
    assert summary["span_ready_positive_count"] == 1
    assert summary["wake_heard_count"] == 1
    assert summary["readiness"]["training_ready"] is False
    assert summary["readiness"]["resource_admission_checked"] is False
    assert summary["training_started"] is False


@pytest.mark.parametrize("change", [
    {"confirm_alexa_span": False}, {"confirm_alexa_span": None},
    {"wake_start": float("nan")}, {"wake_end": float("inf")},
    {"wake_start": True}, {"wake_start": 10 ** 1000}, {"wake_start": -0.01}, {"wake_end": .125},
    {"wake_end": 1.01}, {"wake_presence": "absent"},
])
def test_invalid_confirmed_alexa_spans_fail_without_store_write(state, change):
    played(state)
    body = label_body(state, presence="positive", wake_presence="present",
                      review_category=None, review_note="", entire_clip_negative=False,
                      wake_start=.125, wake_end=.5, confirm_alexa_span=True)
    body.update(change)
    with pytest.raises((ValueError, TypeError)):
        state.label(body)
    assert not state.store.exists()


def test_review_payload_cannot_override_trusted_capture_session(state):
    played(state)
    with pytest.raises(ValueError, match="Unsupported review fields"):
        state.label(label_body(state, session_id="attacker-session"))
    assert not state.store.exists()


def test_readiness_over_100_raw_reviews_uses_full_store_not_legacy_cap(state):
    played(state)
    state.label(label_body(state, presence="negative", entire_clip_negative=True))
    store = review.load_store(state.store)
    base = dict(store["reviews"][0])
    store["reviews"].extend({**base, "event_id": f"duplicate-{index}"}
                            for index in range(100))
    state.store.write_text(json.dumps(store), encoding="utf-8")
    summary = state.summary()
    readiness = summary["readiness"]
    assert len(store["reviews"]) == 101
    assert readiness["training_ready"] is False
    assert readiness["data_eligible"] is False
    assert readiness["resource_admission_checked"] is False
    assert any("positive session groups" in reason for reason in readiness["blocked_reasons"])
    assert all("review count exceeds" not in reason for reason in readiness["blocked_reasons"])
    assert summary["training_started"] is False


def test_latest_local_review_retraction_is_audited_and_clip_returns_unlabelled(state):
    played(state)
    state.label(label_body(state, presence="unsure", wake_presence="present",
                           review_category=None, review_note="", entire_clip_negative=False))
    before = review.load_store(state.store)
    original = dict(before["reviews"][-1])
    info = state.summary()["latest_review"]
    assert info and info["presence"] == "unsure" and "undo_token" in info
    result = state.retract({"undo_token": info["undo_token"]})
    saved = review.load_store(state.store)
    assert saved["reviews"] == []
    assert saved["retracted_reviews"][-1]["review"] == original
    assert saved["retracted_reviews"][-1]["retraction_reason"]
    assert result["restored_candidate_id"]
    summary = state.summary()
    assert summary["reviewed_counts"] == {"positive": 0, "negative": 0, "unsure": 0}
    assert summary["candidates"][0]["presence"] == "unknown"
    assert summary["readiness"]["training_ready"] is False
    # A later explicit review adds to active labels without losing the audit copy.
    played(state)
    state.label(label_body(state, presence="negative", wake_presence="absent",
                           entire_clip_negative=True))
    after = review.load_store(state.store)
    assert after["retracted_reviews"] == saved["retracted_reviews"]
    assert len(after["reviews"]) == 1 and after["reviews"][0]["presence"] == "negative"


@pytest.mark.parametrize("failure_point", ["refresh", "restore"])
def test_retraction_reports_committed_success_after_unexpected_queue_error(state, monkeypatch, failure_point):
    played(state)
    state.label(label_body(state, presence="unsure", wake_presence="present",
                           review_category=None, review_note="", entire_clip_negative=False))
    token = state.summary()["latest_review"]["undo_token"]
    def fail(*_args, **_kwargs):
        raise RuntimeError("injected post-commit queue failure")
    if failure_point == "refresh":
        monkeypatch.setattr(state, "refresh", fail)
    else:
        monkeypatch.setattr(state, "_restore_unlabeled_candidate", fail)
    result = state.retract({"undo_token": token})
    saved = review.load_store(state.store)
    assert result["retracted"] is True
    if failure_point == "refresh":
        assert result["restored_candidate_id"]
    else:
        assert result["restored_candidate_id"] is None
    assert saved["reviews"] == [] and len(saved["retracted_reviews"]) == 1


def test_retraction_rejects_stale_identity_and_non_local_historical_rows(state):
    played(state)
    state.label(label_body(state, presence="unsure", wake_presence="present",
                           review_category=None, review_note="", entire_clip_negative=False))
    row = review.load_store(state.store)["reviews"][-1]
    identity = {key: row[key] for key in ("event_id", "audio_field", "reviewed_at", "wav_sha256")}
    before = state.store.read_bytes()
    identity["event_id"] = "stale-event"
    with pytest.raises(ValueError, match="latest review identity changed"):
        review.retract_latest_review(store_path=state.store, expected_identity=identity,
                                    reason="test correction", audio_root=state.audio_root,
                                    project_root=state.project_root)
    assert state.store.read_bytes() == before

    historical_positive(state)
    historical = review.load_store(state.store)["reviews"][-1]
    identity = {key: historical[key] for key in ("event_id", "audio_field", "reviewed_at", "wav_sha256")}
    before = state.store.read_bytes()
    with pytest.raises(ValueError, match="not created by the local review UI"):
        review.retract_latest_review(store_path=state.store, expected_identity=identity,
                                    reason="test correction", audio_root=state.audio_root,
                                    project_root=state.project_root)
    assert state.store.read_bytes() == before


def test_retraction_http_route_requires_current_token_and_returns_audit_result(state):
    played(state)
    state.label(label_body(state, presence="unsure", wake_presence="present",
                           review_category=None, review_note="", entire_clip_negative=False))
    token = state.summary()["latest_review"]["undo_token"]
    with running(state) as server:
        status, _, body = request(server, "POST", "/api/retract",
                                  json.dumps({"undo_token": token}), post_headers(server))
        assert status == 200 and json.loads(body)["retracted"] is True
        status, _, _ = request(server, "POST", "/api/retract",
                               json.dumps({"undo_token": "0" * 64, "event_id": "extra"}),
                               post_headers(server))
        assert status == 400
    store = review.load_store(state.store)
    assert store["reviews"] == [] and len(store["retracted_reviews"]) == 1


def test_retracted_recent_candidate_is_pinned_into_full_queue(state):
    archive = state.audio_root
    target_audio = wav_file(archive / "older-target.wav", sample=1)
    state.telemetry.write_text(json.dumps({
        "event_id": "older-target", "ts": NOW - 200, "audio_path": str(target_audio),
        "capture_session_id": "target-session", "rejection_reason": "acoustic_unconfirmed",
    }) + "\n", encoding="utf-8")
    review.label_candidate(telemetry=state.telemetry, store_path=state.store,
        event_id="older-target", audio_field="audio_path", presence="unsure",
        session_id="target-session", source_provenance="local review test",
        reviewer="local-wake-reviewer", confirm_reviewed=True, wake_presence="present",
        audio_root=archive, project_root=state.project_root)
    events = [{"event_id": "older-target", "ts": NOW - 200, "audio_path": str(target_audio),
               "capture_session_id": "target-session", "rejection_reason": "acoustic_unconfirmed"}]
    for index in range(1, 102):
        audio = wav_file(archive / f"newer-{index}.wav", sample=index)
        events.append({"event_id": f"newer-{index}", "ts": NOW - index,
                       "audio_path": str(audio), "capture_session_id": f"session-{index}",
                       "rejection_reason": "acoustic_unconfirmed"})
    state.telemetry.write_text("\n".join(map(json.dumps, events)) + "\n", encoding="utf-8")
    refreshed = ui.ReviewState(state.telemetry, archive, state.project_root, state.store, now=NOW)
    refreshed.refresh("all")
    assert len(refreshed.candidates) == ui.MAX_CANDIDATES
    assert "older-target" not in {c["event_id"] for c in refreshed.candidates.values()}
    info = refreshed.summary()["latest_review"]
    result = refreshed.retract({"undo_token": info["undo_token"]})
    assert result["restored_candidate_id"]
    assert len(refreshed.candidates) == ui.MAX_CANDIDATES
    assert next(iter(refreshed.candidates.values()))["event_id"] == "older-target"
    assert refreshed.summary()["candidates"][0]["id"] == result["restored_candidate_id"]


def test_legacy_recovery_write_preserves_retracted_audit_metadata(tmp_path):
    archive=tmp_path/'audio';archive.mkdir();clip=wav_file(archive/'legacy.wav')
    telemetry=tmp_path/'telemetry.jsonl';telemetry.write_text(json.dumps({
        'event_id':'legacy','audio_path':str(clip)})+'\n',encoding='utf-8')
    audit=[{'review':{'event_id':'prior'},'retracted_at':'2026-10-10T00:00:00+00:00',
            'retraction_reason':'user correction'}]
    store=tmp_path/'reviews.json';store.write_text(json.dumps({
        'schema':review.STORE_SCHEMA,'reviews':[],'retracted_reviews':audit,'extension_metadata':{'keep':True}}),encoding='utf-8')
    result=review.recover_confirmed_negatives(
        [{'event_id':'legacy','audio_field':'audio_path','note':'whole clip reviewed'}],
        telemetry=telemetry,store_path=store,reviewer='human',source_provenance='confirmed review',
        confirm_reviewed=True,entire_clip_negative=True,audio_root=archive,project_root=tmp_path)
    saved=review.load_store(store)
    assert result['added']==1 and saved['retracted_reviews']==audit
    assert saved['extension_metadata']=={'keep':True}


@pytest.mark.parametrize("category,note", [("song_music", ""), (None, "A distinct custom sound")])
def test_negative_source_category_or_custom_note_is_saved(state, category, note):
    played(state)
    state.label(label_body(state, review_category=category, review_note=note))
    saved = review.load_store(state.store)["reviews"][0]
    assert saved["review_category"] == category and saved["review_note"] == note


def test_historical_positive_review_still_validates_with_its_span(state):
    historical_positive(state)
    rows = review.validated_reviews(review.load_store(state.store), state.audio_root)
    assert rows[0]["presence"] == "positive"
    assert rows[0]["wake_start_sample"] == 2000 and rows[0]["wake_end_sample"] == 8000


def test_summary_hides_all_duplicate_event_cards_but_counts_unique_audio_once(state):
    played(state)
    state.label(label_body(state, presence="negative", entire_clip_negative=True))
    duplicate = json.loads(state.telemetry.read_text(encoding="utf-8").splitlines()[0])
    duplicate.update(event_id="same-audio-second-event", ts=NOW - 30)
    with state.telemetry.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(duplicate) + "\n")

    store = review.load_store(state.store)
    second = dict(store["reviews"][0], event_id="same-audio-second-event")
    store["reviews"].append(second)
    state.store.write_text(json.dumps(store), encoding="utf-8")
    refreshed = ui.ReviewState(state.telemetry, state.audio_root, state.project_root, state.store, now=NOW)

    summary = refreshed.summary()
    assert len(refreshed.candidates) == 0
    assert summary["reviewed_counts"] == {"positive": 0, "negative": 1, "unsure": 0}
    assert summary["candidates"] == []


def test_conflicting_duplicate_store_does_not_trust_any_raw_event_labels(state):
    played(state)
    historical_positive(state)
    store = review.load_store(state.store)
    conflicting = dict(store["reviews"][0], event_id="conflicting-event", presence="negative",
                       wake_start_s=None, wake_end_s=None, entire_clip_negative=True)
    store["reviews"].append(conflicting)
    state.store.write_text(json.dumps(store), encoding="utf-8")

    summary = state.summary()
    assert summary["store_error"]
    assert summary["reviewed_counts"] == {"positive": 0, "negative": 0, "unsure": 0}
    assert len(summary["candidates"]) == 1
    assert summary["candidates"][0]["presence"] == "unknown"


def test_missing_reviewed_audio_is_retained_and_does_not_block_new_label(state):
    played(state)
    historical_positive(state)
    store_before = review.load_store(state.store)
    historical = dict(store_before["reviews"][0])
    (state.audio_root / "recent.wav").unlink()

    second_clip = wav_file(state.audio_root / "second.wav", sample=12)
    second_event = json.loads(state.telemetry.read_text(encoding="utf-8"))
    second_event.update(event_id="second-event", ts=NOW - 30, audio_path=str(second_clip),
                        capture_session_id="second-real-session")
    with state.telemetry.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(second_event) + "\n")

    refreshed = ui.ReviewState(state.telemetry, state.audio_root, state.project_root,
                               state.store, now=NOW)
    refreshed.refresh("all")
    summary = refreshed.summary()
    assert summary["reviewed_counts"] == {"positive": 1, "negative": 0, "unsure": 0}
    assert summary["reviewed_audio_counts"] == {"usable": 0, "unavailable": 1}
    assert not summary["readiness"]["training_ready"]
    assert len(summary["candidates"]) == 1
    with pytest.raises(FileNotFoundError):
        review.validated_reviews(review.load_store(state.store), state.audio_root)

    played(refreshed)
    new_id = next(iter(refreshed.candidates))
    new_body = label_body(refreshed, id=new_id, presence="negative",
                          entire_clip_negative=True)
    refreshed.label(new_body)
    saved = review.load_store(state.store)["reviews"]
    assert saved[0] == historical
    assert len(saved) == 2
    after = refreshed.summary()
    assert after["reviewed_counts"] == {"positive": 1, "negative": 1, "unsure": 0}
    assert after["reviewed_audio_counts"] == {"usable": 1, "unavailable": 1}
    assert not after["readiness"]["training_ready"]


@pytest.mark.parametrize("bad_history", ["changed_audio", "malformed_metadata"])
def test_missing_audio_tolerance_does_not_allow_stale_or_malformed_history(state, bad_history):
    played(state)
    historical_positive(state)
    store = review.load_store(state.store)
    if bad_history == "changed_audio":
        wav_file(state.audio_root / "recent.wav", sample=1)
        with pytest.raises(ValueError, match="hash mismatch"):
            review.review_history(store, state.audio_root, state.project_root)
    else:
        store["reviews"][0]["addressed_assistant"] = "maybe"
        with pytest.raises(ValueError, match="addressed_assistant"):
            review.review_history(store, state.audio_root, state.project_root)


def test_missing_audio_does_not_hide_metadata_conflicts(state):
    played(state)
    historical_positive(state)
    (state.audio_root / "recent.wav").unlink()
    store = review.load_store(state.store)
    conflict = dict(store["reviews"][0], event_id="another-event", presence="negative",
                    wake_start_s=None, wake_end_s=None, entire_clip_negative=True)
    store["reviews"].append(conflict)
    with pytest.raises(ValueError, match="conflicting positive/negative"):
        review.review_history(store, state.audio_root, state.project_root)


def test_opaque_wav_http_delivery_then_explicit_label(state):
    opaque = next(iter(state.candidates))
    with running(state) as server:
        status, headers, data = request(server, path="/audio/" + opaque)
        assert status == 200 and headers["Content-Type"] == "audio/wav"
        assert headers["Cache-Control"] == "no-store" and data.startswith(b"RIFF")
        status, _, data = request(server, "POST", "/api/label", json.dumps(label_body(state)), post_headers(server))
    assert status == 200 and json.loads(data)["presence"] == "negative"
    assert len(review.load_store(state.store)["reviews"]) == 1


def test_http_playback_hash_can_save_only_confirmed_exact_span(state):
    opaque = next(iter(state.candidates))
    body = label_body(state, presence="positive", wake_presence="present",
                      review_category=None, review_note="", entire_clip_negative=False,
                      wake_start=.125, wake_end=.5, confirm_alexa_span=True)
    with running(state) as server:
        status, headers, data = request(server, path="/audio/" + opaque)
        assert status == 200 and headers["Content-Type"] == "audio/wav"
        assert hashlib.sha256(data).hexdigest() == review.wav_identity(
            state.audio_root / "recent.wav")["wav_sha256"]
        status, _, response = request(server, "POST", "/api/label", json.dumps(body), post_headers(server))
    assert status == 200 and json.loads(response)["presence"] == "positive"
    row = review.validated_reviews(review.load_store(state.store), state.audio_root)[0]
    assert row["wake_start_sample"] == 2000 and row["wake_end_sample"] == 8000


@pytest.mark.parametrize("bad", ["nonce", "origin", "host", "missing-origin", "cross-site", "non-ascii"])
def test_csrf_boundary_refuses_cross_origin_or_missing_nonce(state, bad):
    played(state)
    with running(state) as server:
        headers = post_headers(server)
        if bad == "nonce":
            headers.pop("X-Review-Nonce")
        elif bad == "origin":
            headers["Origin"] = "https://example.invalid"
        elif bad == "host":
            headers["Host"] = "example.invalid"
        elif bad == "missing-origin":
            headers.pop("Origin")
        elif bad == "cross-site":
            headers["Sec-Fetch-Site"] = "cross-site"
        else:
            headers["X-Review-Nonce"] = "\u00e9"
        status, _, _ = request(server, "POST", "/api/label", json.dumps(label_body(state)), headers)
    assert status == 403 and not state.store.exists()


@pytest.mark.parametrize("path", ["/audio/../../outside.wav", "/audio/not-a-candidate",
                                  "/audio/id?path=outside.wav", "/api/state?command=train",
                                  "http://example.invalid/api/state"])
def test_request_paths_cannot_choose_files_or_commands(state, path):
    with running(state) as server:
        status, _, _ = request(server, path=path, headers={"Host": f"127.0.0.1:{server.server_port}"})
    assert status in (400, 404) and not state.store.exists()


@pytest.mark.parametrize("header", ["Host", "Origin", "X-Review-Nonce", "Content-Length"])
def test_duplicate_sensitive_headers_are_rejected(state, header):
    with running(state) as server:
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
        try:
            connection.putrequest("POST", "/api/label", skip_host=True)
            headers = {"Host": f"127.0.0.1:{server.server_port}", **post_headers(server), "Content-Length": "2"}
            for name, value in headers.items():
                connection.putheader(name, value)
                if name == header:
                    connection.putheader(name, value)
            connection.endheaders(b"{}")
            response = connection.getresponse()
            assert response.status == (413 if header == "Content-Length" else 403)
            response.read()
        finally:
            connection.close()
    assert not state.store.exists()


def test_cross_site_get_cannot_read_page_nonce_or_recording(state):
    with running(state) as server:
        for headers in ({"Origin": "https://example.invalid"}, {"Sec-Fetch-Site": "cross-site"}):
            status, _, body = request(server, path="/", headers=headers)
            assert status == 403 and state.nonce.encode() not in body
    assert not state.store.exists()


def test_stable_port_falls_back_only_to_loopback(state):
    with running(state) as occupied:
        fallback = ui.create_server(state, occupied.server_port)
        try:
            assert fallback.server_address[0] == "127.0.0.1"
            assert occupied.server_port < fallback.server_port <= occupied.server_port + 9
        finally:
            fallback.server_close()


def test_review_listener_uses_exclusive_binding_not_address_reuse(state):
    with running(state) as server:
        assert server.allow_reuse_address is False and server.allow_reuse_port is False
        assert server.socket.getsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR) == 0
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            assert server.socket.getsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE) == 1


def test_audio_revalidates_archive_root_wav_type_and_size(state, tmp_path, monkeypatch):
    candidate = next(iter(state.candidates.values()))
    opaque = next(iter(state.candidates))
    candidate["path"] = str(wav_file(tmp_path / "outside.wav"))
    with pytest.raises(ValueError, match="archival"):
        state.audio(opaque)
    candidate["path"] = str(state.audio_root / "not_wav.txt")
    (state.audio_root / "not_wav.txt").write_bytes(b"private")
    with pytest.raises(ValueError):
        state.audio(opaque)
    candidate["path"] = str(wav_file(state.audio_root / "stereo.wav", channels=2))
    with pytest.raises(ValueError, match="mono"):
        state.audio(opaque)
    candidate["path"] = str(state.audio_root / "recent.wav")
    monkeypatch.setattr(ui, "MAX_AUDIO_BYTES", 20)
    with pytest.raises(ValueError, match="size limit"):
        state.audio(opaque)
    assert not state.played and not state.store.exists()


def test_symlink_escape_is_refused(state, tmp_path):
    outside = wav_file(tmp_path / "outside.wav")
    link = state.audio_root / "alias.wav"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("Windows symlink creation not permitted")
    next(iter(state.candidates.values()))["path"] = str(link)
    with pytest.raises(ValueError, match="archival"):
        state.audio(next(iter(state.candidates)))


@pytest.mark.parametrize("bad", ["oversize", "type", "non-finite", "chunked"])
def test_post_body_limits_and_json_semantics(state, bad):
    played(state)
    with running(state) as server:
        headers = post_headers(server)
        body = json.dumps(label_body(state))
        expected = 400
        if bad == "oversize":
            body = " " * (ui.MAX_BODY_BYTES + 1)
            expected = 413
        elif bad == "type":
            headers["Content-Type"] = "text/plain"
            expected = 415
        elif bad == "chunked":
            headers["Transfer-Encoding"] = "chunked"
        else:
            body = body.replace('"review_note": ""', '"review_note": NaN')
        status, _, _ = request(server, "POST", "/api/label", body, headers)
    assert status == expected and not state.store.exists()


def test_missing_archive_stays_unknown_and_does_not_leak_paths(state):
    (state.audio_root / "recent.wav").unlink()
    refreshed = ui.ReviewState(state.telemetry, state.audio_root, state.project_root, state.store, now=NOW)
    refreshed.refresh("all")
    card = refreshed.summary()["candidates"][0]
    assert card["presence"] == "unknown" and not card["available"]
    assert "missing, pruned or invalid" in card["unavailable"]
    assert str(state.project_root) not in json.dumps(refreshed.summary())
    assert not state.store.exists()


def test_rate_suppressed_decision_is_visible_but_not_labelable(state):
    state.telemetry.write_text(json.dumps({"event_id": "suppressed", "ts": NOW - 10,
        "candidate_kind": "below_vad", "audio_reason": "audio_rate_limited"}) + "\n", encoding="utf-8")
    refreshed = ui.ReviewState(state.telemetry, state.audio_root, state.project_root, state.store, now=NOW)
    card = refreshed.summary()["candidates"][0]
    assert not card["available"] and card["presence"] == "unknown"
    assert card["unavailable"] == "Recording unavailable: audio_rate_limited"
    with pytest.raises(ValueError, match="unavailable"):
        refreshed.audio(card["id"])
    assert not state.store.exists()


def test_corrupt_store_blocks_counts_without_loading_unsafe_manifest(state):
    state.store.write_text('{"positives": [{"label": "alexa"}]}', encoding="utf-8")
    summary = state.summary()
    assert summary["store_error"] and summary["reviewed_counts"]["positive"] == 0
    played(state)
    with pytest.raises(ValueError, match="schema"):
        state.label(label_body(state))
    assert json.loads(state.store.read_text()) == {"positives": [{"label": "alexa"}]}
