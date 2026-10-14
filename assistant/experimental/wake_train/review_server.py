#!/usr/bin/env python3
"""Loopback-only, dependency-free listening review for recent wake candidates.

No inference, automatic labels or training. The human review store is first
created only after an explicit listening review POST. Candidate paths never
come from request URLs or bodies; the server owns an opaque ID mapping.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import secrets
import socket
import threading
import time
from urllib.parse import parse_qs, urlsplit

try:
    from . import active_learning as review
except ImportError:
    import active_learning as review

MAX_BODY_BYTES = 8192
MAX_AUDIO_BYTES = 2 * 1024 * 1024
MAX_CANDIDATES = 100
REVIEW_MODES = {"possible_misses", "unconfirmed_hits", "all"}
MODE_LABELS = {"possible_misses": "Short / quiet speech", "unconfirmed_hits": "Unconfirmed wake hits", "all": "All recordings"}
DEFAULT_PORT = 8766
DEFAULT_STORE = Path(__file__).parent / "human_reviews.json"


def source_classification(candidate):
    kind = str(candidate.get("candidate_kind", ""))
    reason = str(candidate.get("reason", ""))
    source = candidate.get("trigger_source", "")
    diagnostic_kinds = {"no_trigger", "short_segment", "below_vad", "second_stt", "decode_error"}
    if (candidate.get("audio_field") != "audio_path" or source in {"stt", "wake", "command", "retry"}
            or (source not in {"near_miss", "wake_rejected", "miss", ""})
            or (not source and kind not in diagnostic_kinds)):
        return "all", "Other archived audio; not a verified wake recording."
    if kind == "acoustic_unconfirmed" or "acoustic_unconfirmed" in reason:
        return "unconfirmed_hits", "The wake model proposed a hit, but wake confirmation rejected it. This is not proof that Alexa was spoken."
    if kind in diagnostic_kinds or source in {"near_miss", "wake_rejected", "miss"}:
        return "possible_misses", "Short / quiet speech; may include room noise, songs, or genuine missed wakes."
    return "all", "Other archived audio; not a verified wake recording."


def _finite_seconds(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def recent_candidates(telemetry, audio_root, project_root, *, now=None, limit=MAX_CANDIDATES,
                      selection="ranked", excluded=(), mode="all"):
    if selection not in ("ranked", "recent"):
        raise ValueError("selection must be 'ranked' or 'recent'")
    if mode not in REVIEW_MODES:
        raise ValueError("unsupported review mode")
    cutoff = (time.time() if now is None else now) - 48 * 3600
    excluded = set(excluded)
    selected = []
    for candidate in review.iter_candidates(telemetry, audio_root, project_root,
                                            min_ts=cutoff, max_ts=cutoff + 48 * 3600,
                                            resolve_audio_paths=False):
        category, _explanation = source_classification(candidate)
        if mode != "all" and category != mode:
            continue
        if (candidate.get("event_id"), candidate.get("audio_field")) in excluded:
            continue
        stamp = candidate.get("event_ts")
        if (not isinstance(candidate.get("event_id"), str) or not candidate["event_id"]
                or not isinstance(stamp, (int, float)) or isinstance(stamp, bool)
                or not math.isfinite(stamp) or not cutoff <= stamp <= cutoff + 48 * 3600):
            continue
        score = max(candidate.get("scores", {}).values(), default=0.0)
        disagreement = any(term in str(candidate.get("reason", "")) for term in ("unconfirmed", "second_stt", "woke_early"))
        selected.append({**candidate, "event_ts": stamp, "rank": (int(disagreement), score, stamp)})
        selected.sort(key=(lambda row: row["event_ts"]) if selection == "recent"
                      else (lambda row: row["rank"]), reverse=True)
        del selected[max(1, min(int(limit), MAX_CANDIDATES)):]
    for candidate in selected:
        if not candidate.get("path"):
            continue
        try:
            candidate["path"] = str(review.audio_path(candidate["path"], audio_root, project_root))
            candidate["unavailable"] = None
        except (OSError, ValueError, TypeError) as exc:
            candidate["unavailable"] = str(exc)
    return selected


class ReviewState:
    def __init__(self, telemetry=review.TELEMETRY, audio_root=review.AUDIO_ROOT,
                 project_root=review.PROJECT_ROOT, store=DEFAULT_STORE, *, now=None):
        self.telemetry, self.audio_root, self.project_root = map(Path, (telemetry, audio_root, project_root))
        self.store = Path(store)
        self.now = now
        self.nonce = secrets.token_urlsafe(32)
        self.lock = threading.RLock()
        self.played = {}
        self.mode = "possible_misses"
        self.candidates = {}
        self.pinned_id = None
        self._id_prefix = secrets.token_hex(8)
        self._id_counter = 0
        labels = self._reviewed_keys()
        self._install_candidates(recent_candidates(
            self.telemetry, self.audio_root, self.project_root, now=now,
            selection="recent", excluded=labels, mode=self.mode), excluded=labels)

    @staticmethod
    def _identity(candidate):
        return (candidate.get("event_id"), candidate.get("audio_field"), candidate.get("path"))

    def _new_id(self):
        if self._id_counter >= 0xFFFFFFFFFFFFFFFF:
            raise RuntimeError("opaque candidate ID space exhausted")
        self._id_counter += 1
        return f"{self._id_prefix}{self._id_counter:016x}"

    def _install_candidates(self, fresh, *, excluded=()):
        excluded = set(excluded)
        old_by_identity = {self._identity(candidate): opaque
                           for opaque, candidate in self.candidates.items()}
        completed = [opaque for opaque in self.played
                     if opaque in self.candidates
                     and (self.candidates[opaque].get("event_id"),
                          self.candidates[opaque].get("audio_field")) in excluded]
        for opaque in completed:
            self.played.pop(opaque, None)
        if self.pinned_id in self.candidates:
            pinned = self.candidates[self.pinned_id]
            if (pinned.get("event_id"), pinned.get("audio_field")) in excluded:
                self.pinned_id = None
        played_ids = [opaque for opaque in self.played if opaque in self.candidates]
        retained_limit = MAX_CANDIDATES - int(self.pinned_id in self.candidates)
        retained_ids = played_ids[-retained_limit:]
        retained = {opaque: self.candidates[opaque] for opaque in retained_ids}
        for opaque in played_ids[:-retained_limit]:
            self.played.pop(opaque, None)
        new = {}
        for candidate in fresh:
            identity = self._identity(candidate)
            opaque = old_by_identity.get(identity)
            if opaque is None:
                opaque = self._new_id()
            new[opaque] = candidate
        for opaque, candidate in retained.items():
            new.setdefault(opaque, candidate)
        if self.pinned_id in self.candidates:
            pinned = self.candidates[self.pinned_id]
            if (pinned.get("event_id"), pinned.get("audio_field")) not in excluded:
                new.setdefault(self.pinned_id, pinned)
        # Played rows stay addressable for in-flight labels; trim only unplayed rows.
        if self.pinned_id in new and self.pinned_id not in retained:
            retained[self.pinned_id] = new[self.pinned_id]
        retained_set = set(retained)
        room = max(0, MAX_CANDIDATES - len(retained_set))
        fresh_rows = [(opaque, candidate) for opaque, candidate in new.items()
                      if opaque not in retained_set]
        self.candidates = {**retained, **dict(fresh_rows[:room])}

    def _reviewed_keys(self):
        if not self.store.exists():
            return set()
        try:
            store_snapshot = review.load_store(self.store)
            review.metadata_reviews(store_snapshot, self.audio_root, self.project_root)
            return {(row["event_id"], row["audio_field"])
                    for row in store_snapshot["reviews"]}
        except (OSError, ValueError, TypeError, KeyError):
            # summary reports validation failures; untrusted labels do not hide rows.
            return set()

    def refresh(self, mode=None):
        with self.lock:
            if mode is not None:
                if mode not in REVIEW_MODES:
                    raise ValueError("unsupported review mode")
                self.mode = mode
            labels = self._reviewed_keys()
            fresh = recent_candidates(self.telemetry, self.audio_root, self.project_root,
                                      now=self.now, selection="recent", excluded=labels, mode=self.mode)
            self._install_candidates(fresh, excluded=labels)

    def summary(self):
        rows, usable_rows, unavailable_rows, trusted_reviews, error = [], [], [], [], None
        readiness_error = None
        if self.store.exists():
            try:
                store_snapshot = review.load_store(self.store)
                rows, usable_rows, unavailable_rows = review.review_history(
                    store_snapshot, self.audio_root, self.project_root)
                # Metadata validation checked every row and conflicts. Keep all
                # raw event labels so duplicate event cards remain hidden too.
                trusted_reviews = store_snapshot["reviews"]
            except (OSError, ValueError, TypeError) as exc:
                error = "Review store could not be validated; labels and training are blocked."
                readiness_error = str(exc)
                for root_path in (str(self.audio_root), str(self.project_root)):
                    if root_path and root_path in readiness_error:
                        readiness_error = "Archived recording path validation failed."
        labels = {(row["event_id"], row["audio_field"]): row["presence"] for row in trusted_reviews}
        cards = []
        with self.lock:
            candidate_rows = list(self.candidates.items())
            mode, pinned_id = self.mode, self.pinned_id
        for opaque, candidate in candidate_rows:
            if (candidate["event_id"], candidate["audio_field"]) in labels:
                continue
            source_mode, source_explanation = source_classification(candidate)
            pinned_undo = opaque == pinned_id
            if mode != "all" and source_mode != mode and not pinned_undo:
                continue
            # Expose decision metadata, never archival filesystem paths or ASR prompts.
            wav_sha256 = ""
            duration_seconds = None
            if not candidate.get("unavailable") and candidate.get("path"):
                try:
                    path = review.audio_path(candidate["path"], self.audio_root, self.project_root)
                    if path.stat().st_size <= MAX_AUDIO_BYTES:
                        identity = review.wav_identity(path)
                        wav_sha256 = identity["wav_sha256"]
                        duration_seconds = identity["n_frames"] / identity["sample_rate"]
                except (OSError, ValueError, TypeError):
                    pass
            display_explanation = ("Pinned undone response; outside the selected queue. " if pinned_undo and source_mode != mode and mode != "all" else "") + source_explanation
            cards.append({"id": opaque, "event_id": candidate["event_id"],
                          "audio_field": candidate["audio_field"], "reason": display_explanation,
                          "source_classification": source_mode,
                          "source_explanation": display_explanation,
                          "wav_sha256": wav_sha256,
                          "duration_seconds": duration_seconds,
                          "scores": candidate["scores"], "ts": candidate["event_ts"],
                          "session_seed": candidate.get("capture_session_id", ""),
                          "truncated": candidate.get("recording_truncated", False),
                          "end_reason": candidate.get("end_reason", ""),
                          "available": not candidate.get("unavailable") and bool(candidate.get("path")),
                          "unavailable": self._unavailable(candidate.get("unavailable")),
                          "presence": labels.get((candidate["event_id"], candidate["audio_field"]), "unknown")})
        counts = {label: sum(row["presence"] == label for row in rows) for label in ("positive", "negative", "unsure")}
        usable_counts = {label: sum(row["presence"] == label for row in usable_rows)
                         for label in ("positive", "negative", "unsure")}
        unavailable_counts = {label: sum(row["presence"] == label for row in unavailable_rows)
                              for label in ("positive", "negative", "unsure")}
        wake_heard = sum(row.get("wake_presence") == "present" or row.get("presence") == "positive"
                         for row in trusted_reviews)
        if error:
            readiness = {"data_eligible": False, "training_ready": False,
                         "training_supported": True, "resource_admission_checked": False,
                         "feature_extraction_started": False, "training_started": False,
                         "model_trained": False,
                         "blocked_reasons": [readiness_error or "Review store validation failed; reviewed training is blocked."]}
        else:
            try:
                try:
                    from . import reviewed_training
                except ImportError:
                    import reviewed_training
                _plan, readiness = reviewed_training.prepare(self.store, audio_root=self.audio_root)
            except (OSError, ValueError, TypeError, KeyError) as exc:
                message = str(exc)
                for root_path in (str(self.audio_root), str(self.project_root)):
                    if root_path and root_path in message:
                        message = "Reviewed training preflight failed for an archived recording."
                readiness = {"data_eligible": False, "training_ready": False,
                             "training_supported": True, "resource_admission_checked": False,
                             "feature_extraction_started": False, "training_started": False,
                             "model_trained": False, "blocked_reasons": [message]}
        return {"candidates": cards, "mode": mode, "mode_label": MODE_LABELS[mode],
                "wake_heard_count": wake_heard,
                "span_ready_positive_count": sum(row.get("presence") == "positive" for row in usable_rows), "reviewed_counts": counts,
                "reviewed_audio_counts": {"usable": len(usable_rows),
                                          "unavailable": len(unavailable_rows)},
                "usable_review_counts": usable_counts,
                "unavailable_review_counts": unavailable_counts,
                "readiness": readiness,
                "latest_review": self._latest_review_info(trusted_reviews) if not error else None,
                "store_error": error, "training_started": False,
                "queue_limit": MAX_CANDIDATES, "window_hours": 48}

    def _latest_review_info(self, rows):
        if not rows:
            return None
        try:
            indexed = [(datetime.fromisoformat(row["reviewed_at"].replace("Z", "+00:00")), i, row)
                       for i, row in enumerate(rows)]
            if any(stamp.tzinfo is None for stamp, _i, _row in indexed):
                return None
            _stamp, _index, row = max(indexed, key=lambda item: (item[0], item[1]))
            if row.get("reviewer") != "local-wake-reviewer":
                return None
            identity = {name: row[name] for name in ("event_id", "audio_field", "reviewed_at", "wav_sha256")}
            token_body = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
            token = hmac.new(self.nonce.encode("ascii"), token_body, hashlib.sha256).hexdigest()
            return {**identity, "presence": row["presence"], "undo_token": token}
        except (KeyError, TypeError, ValueError):
            return None

    def retract(self, body):
        if not isinstance(body, dict) or set(body) != {"undo_token"}:
            raise ValueError("Unsupported retraction fields")
        token = body.get("undo_token")
        if (not isinstance(token, str) or len(token) != 64
                or any(char not in "0123456789abcdef" for char in token)):
            raise ValueError("Invalid retraction token")
        with self.lock:
            if not self.store.exists():
                raise ValueError("Review store does not exist")
            store = review.load_store(self.store)
            review.metadata_reviews(store, self.audio_root, self.project_root)
            info = self._latest_review_info(store["reviews"])
            if info is None or not hmac.compare_digest(token, info["undo_token"]):
                raise ValueError("Latest review changed; nothing was retracted")
            identity = {name: info[name] for name in ("event_id", "audio_field", "reviewed_at", "wav_sha256")}
            result = review.retract_latest_review(
                store_path=self.store, expected_identity=identity,
                reason="User corrected the most recent wake-review response",
                audio_root=self.audio_root, project_root=self.project_root)
            for opaque, candidate in list(self.candidates.items()):
                if (candidate.get("event_id"), candidate.get("audio_field")) == (result["event_id"], result["audio_field"]):
                    self.played.pop(opaque, None)
            try:
                self.refresh()
            except Exception:
                # The audit write is already committed; do not report it as undone.
                pass
            try:
                restored_id = self._restore_unlabeled_candidate(identity)
            except Exception:
                restored_id = None
            return {"retracted": True, "event_id": result["event_id"],
                    "audio_field": result["audio_field"], "presence": result["presence"],
                    "restored_candidate_id": restored_id}

    def _restore_unlabeled_candidate(self, identity):
        """Pin the exact retracted clip into the bounded queue when still safely reviewable."""
        now = time.time() if self.now is None else self.now
        cutoff = now - 48 * 3600
        try:
            matches = []
            for candidate in review.iter_candidates(
                    self.telemetry, self.audio_root, self.project_root,
                    event_id=identity["event_id"], min_ts=cutoff, max_ts=now):
                if (candidate.get("event_id"), candidate.get("audio_field")) != (identity["event_id"], identity["audio_field"]):
                    continue
                stamp = candidate.get("event_ts")
                if (not isinstance(stamp, (int, float)) or isinstance(stamp, bool)
                        or not math.isfinite(stamp) or not cutoff <= stamp <= now
                        or candidate.get("unavailable") or not candidate.get("capture_session_id")):
                    continue
                path = review.audio_path(candidate.get("path"), self.audio_root, self.project_root)
                if review.wav_identity(path)["wav_sha256"] != identity["wav_sha256"]:
                    continue
                matches.append({**candidate, "event_ts": stamp})
            if not matches:
                return None
            paths = {candidate.get("path") for candidate in matches}
            sessions = {candidate.get("capture_session_id") for candidate in matches}
            if len(paths) != 1 or len(sessions) != 1:
                return None
            candidate = matches[0]
        except (OSError, ValueError, TypeError):
            return None
        with self.lock:
            opaque = next((key for key, old in self.candidates.items()
                           if self._identity(old) == self._identity(candidate)), None)
            if opaque is None:
                opaque = self._new_id()
            remaining = [(key, old) for key, old in self.candidates.items() if key != opaque]
            if len(remaining) >= MAX_CANDIDATES:
                victim_index = next((i for i in range(len(remaining) - 1, -1, -1)
                                     if remaining[i][0] not in self.played), None)
                if victim_index is None:
                    return None
                remaining.pop(victim_index)
            self.candidates = {opaque: candidate, **dict(remaining)}
            self.pinned_id = opaque
            self.played.pop(opaque, None)
            return opaque

    @staticmethod
    def _unavailable(reason):
        # OS exceptions can contain private absolute paths; only expose known reasons.
        if not reason:
            return ""
        safe = {"audio_rate_limited", "audio_not_archived", "disabled", "cap_exceeded",
                "disk_error", "audio_disabled", "audio_cap_exceeded", "write_error"}
        return "Recording unavailable: " + (reason if isinstance(reason, str) and reason in safe else "missing, pruned or invalid archive")

    def audio(self, opaque):
        with self.lock:
            candidate = self.candidates.get(opaque)
            if candidate is None or candidate.get("unavailable"):
                raise ValueError("Recording unavailable")
            path = review.audio_path(candidate.get("path"), self.audio_root, self.project_root)
            if path.stat().st_size > MAX_AUDIO_BYTES:
                raise ValueError("Recording exceeds local playback size limit")
            identity = review.wav_identity(path)
            with path.open("rb") as stream:
                data = stream.read(MAX_AUDIO_BYTES + 1)
            if len(data) > MAX_AUDIO_BYTES or hashlib.sha256(data).hexdigest() != identity["wav_sha256"]:
                raise ValueError("Recording changed during playback preparation")
            return data, identity["wav_sha256"]

    def label(self, body):
        allowed = {"id", "presence", "wake_presence", "review_category", "review_note",
                   "confirm_reviewed", "entire_clip_negative", "wake_start", "wake_end",
                   "confirm_alexa_span"}
        if not isinstance(body, dict) or set(body) - allowed:
            raise ValueError("Unsupported review fields")
        opaque = body.get("id")
        if not isinstance(opaque, str):
            raise ValueError("Unknown candidate")
        if body.get("confirm_reviewed") is not True:
            raise ValueError("Listening review is required")
        with self.lock:
            candidate = self.candidates.get(opaque)
            if candidate is None:
                raise ValueError("Unknown candidate")
            category = body.get("review_category")
            note = body.get("review_note", "")
            has_span_fields = any(key in body for key in ("wake_start", "wake_end", "confirm_alexa_span"))
            if ((category is not None and category not in review.REVIEW_CATEGORIES)
                    or not isinstance(note, str) or len(note) > 500):
                raise ValueError("review_category or review_note is invalid")
            if body.get("presence") == "unsure":
                if (body.get("wake_presence") != "present" or body.get("entire_clip_negative")
                        or has_span_fields):
                    raise ValueError("Alexa presence-only labels must remain unsure")
            elif body.get("presence") == "negative":
                if (body.get("wake_presence") != "absent" or body.get("entire_clip_negative") is not True
                        or (category is None and not note.strip()) or has_span_fields):
                    raise ValueError("Not Alexa requires whole-clip listening confirmation")
            elif body.get("presence") == "positive":
                if (body.get("wake_presence") != "present" or body.get("confirm_alexa_span") is not True
                        or body.get("entire_clip_negative") or category is not None or note.strip()):
                    raise ValueError("Confirmed Alexa span requires its clarity checkbox and exact bounds")
                wake_start, wake_end = body.get("wake_start"), body.get("wake_end")
                if not all(_finite_seconds(value) for value in (wake_start, wake_end)):
                    raise ValueError("Alexa span bounds must be finite seconds")
            else:
                raise ValueError("Unsupported one-tap label")
            if not candidate.get("capture_session_id"):
                raise ValueError("This recording has no trusted capture session; it cannot be grouped safely")
            if opaque not in self.played:
                raise ValueError("Play the recording before submitting a listening review")
            path = review.audio_path(candidate.get("path"), self.audio_root, self.project_root)
            identity = review.wav_identity(path)
            if identity["wav_sha256"] != self.played[opaque]:
                raise ValueError("Recording changed since playback; play it again before reviewing")
            if body.get("presence") == "positive":
                duration = identity["n_frames"] / identity["sample_rate"]
                if not 0 <= wake_start < wake_end <= duration:
                    raise ValueError("Alexa span must be non-empty and within the recording")
            row = review.label_candidate(
                telemetry=self.telemetry, store_path=self.store, event_id=candidate["event_id"],
                audio_field=candidate["audio_field"], presence=body.get("presence"),
                session_id=candidate.get("capture_session_id"), reviewer="local-wake-reviewer",
                source_provenance="Local wake review UI; archived recording " + candidate["audio_field"], confirm_reviewed=True,
                wake_start=wake_start if body.get("presence") == "positive" else None,
                wake_end=wake_end if body.get("presence") == "positive" else None,
                entire_clip_negative=body.get("entire_clip_negative") is True,
                addressed_assistant="unsure",
                # active_learning reserves wake_presence='present' for presence-only
                # unsure labels; exact positive spans carry presence in `presence`.
                wake_presence=None if body.get("presence") == "positive" else body.get("wake_presence"),
                review_category=body.get("review_category"), review_note=body.get("review_note", ""),
                audio_root=self.audio_root, project_root=self.project_root)
        return {"presence": row["presence"], "event_id": row["event_id"], "training_started": False}


PAGE = r"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Wake review</title><style>
body{font:17px system-ui;background:#111827;color:#e5e7eb;max-width:720px;margin:32px auto;padding:0 20px}.card{background:#1f2937;border:1px solid #374151;border-radius:14px;padding:22px;margin-top:20px}button{font:inherit;padding:12px 18px;border-radius:8px;border:0;cursor:pointer;margin:8px 8px 8px 0;background:#dbeafe;color:#111827}button:disabled{opacity:.5;cursor:not-allowed}button:focus-visible{outline:3px solid #fbbf24;outline-offset:3px}button[aria-pressed="true"]{background:#14532d;color:#f0fdf4;border:2px solid #4ade80;box-shadow:0 0 0 2px #14532d;font-weight:700}.navigation{display:flex;align-items:center;gap:10px}.navigation button{margin:8px 0}.position{min-width:7em;text-align:center;color:#d1d5db}label{display:block;margin:14px 0}textarea{font:inherit;width:min(95%,600px);min-height:70px}audio{display:none}.status{min-height:1.5em}.error{color:#fca5a5}.small{color:#b9c4d5}
</style><h1>Wake review</h1><p>These are unlabelled diagnostic recordings, not confirmed missed Alexas. Music and room noise are expected. Background labels help reduce false detections.</p><label for="mode">Review queue</label><select id="mode"><option value="possible_misses">Short / quiet speech</option><option value="unconfirmed_hits">Unconfirmed wake hits</option><option value="all">All recordings</option></select><button id="refresh" type="button">Refresh recordings</button><span id="refresh-status" class="small" role="status" aria-live="polite"></span><div id="summary" class="small" aria-live="polite"></div><button id="undo-review" type="button" hidden disabled>Undo last response</button><span id="undo-status" class="small" role="status" aria-live="polite"></span><p id="action-status" class="small" role="status" aria-live="polite"></p><main id="card"></main><script>
const nonce=__NONCE__,el=(tag,text)=>{const node=document.createElement(tag);if(text!==undefined)node.textContent=text;return node;};let selectedId=null,selectedIndex=0;
document.querySelector('#mode').addEventListener('change',async event=>{try{const response=await fetch(`/api/state?refresh=1&mode=${encodeURIComponent(event.target.value)}`,{cache:'no-store'});if(!response.ok)throw Error();selectedId=null;selectedIndex=0;await load();}catch{document.querySelector('#refresh-status').textContent=' Queue change failed.';}});document.querySelector('#refresh').addEventListener('click',async()=>{const button=document.querySelector('#refresh'),status=document.querySelector('#refresh-status');button.disabled=true;status.textContent=' Refreshing…';try{const response=await fetch('/api/state?refresh=1',{cache:'no-store'});if(!response.ok)throw Error();await load();status.textContent=' Recordings refreshed.';}catch{status.textContent=' Refresh failed; the current card is unchanged.';}finally{button.disabled=false;}});
async function load(){const root=document.querySelector('#card');try{const response=await fetch('/api/state',{cache:'no-store'});if(!response.ok)throw Error('state unavailable');const s=await response.json(),candidates=Array.isArray(s.candidates)?s.candidates:[],reviewable=candidates.filter(c=>c&&c.available&&typeof c.session_seed==='string'&&c.session_seed.trim()),unavailable=candidates.filter(c=>!c||!c.available).length,ungrouped=candidates.length-reviewable.length-unavailable,skipped=unavailable+ungrouped;document.querySelector('#summary').textContent=`${reviewable.length} reviewable recordings · ${unavailable} unavailable and ${ungrouped} missing trusted session skipped · ${s.reviewed_counts.positive} span-ready positives · ${s.reviewed_counts.negative} negatives · ${s.reviewed_counts.unsure} presence-only or unsure. Training has not started.`;const undo=document.querySelector('#undo-review');undo.hidden=!s.latest_review;undo.disabled=!s.latest_review;undo.dataset.token=s.latest_review?s.latest_review.undo_token:'';undo.textContent=s.latest_review?`Undo last response (${s.latest_review.presence==='negative'?'Not Alexa':'Alexa'})`:'Undo last response';root.replaceChildren();if(!reviewable.length){selectedId=null;selectedIndex=0;root.append(el('p',skipped?`No recordings can be reviewed right now. ${unavailable} unavailable and ${ungrouped} missing trusted session recordings were skipped. Use Refresh recordings to check again.`:'No recordings are waiting. Use Refresh recordings to check again.'));return;}let index=selectedId?reviewable.findIndex(row=>row.id===selectedId):-1;if(index<0)index=Math.min(selectedIndex,reviewable.length-1);selectedIndex=index;const c=reviewable[index];selectedId=c.id;const box=el('section');box.className='card';root.append(box);const nav=el('nav');nav.className='navigation';nav.setAttribute('aria-label','Review navigation');const previous=el('button','Previous review');previous.type='button';previous.disabled=index===0;previous.addEventListener('click',()=>{selectedIndex=index-1;selectedId=reviewable[index-1].id;load();});const position=el('span',`Review ${index+1} of ${reviewable.length}`);position.className='position';position.setAttribute('aria-live','polite');const next=el('button','Next review');next.type='button';next.disabled=index===reviewable.length-1;next.addEventListener('click',()=>{selectedIndex=index+1;selectedId=reviewable[index+1].id;load();});nav.append(previous,position,next);box.append(nav);box.append(el('h2',new Date(c.ts*1000).toLocaleString()));box.append(el('p',c.reason));
const audio=el('audio');audio.preload='none';audio.src='/audio/'+c.id;box.append(audio);let played=false,finished=false,negativeButton=null;audio.addEventListener('play',()=>{played=true;});const play=el('button','Play');play.type='button';play.addEventListener('click',()=>{if(audio.paused){audio.play().catch(()=>{status.className='status error';status.textContent='Playback failed. Try again or use Refresh recordings.';play.textContent='Play';});play.textContent='Pause';}else audio.pause();});audio.addEventListener('pause',()=>{play.textContent='Play';});box.append(play);
const status=el('p');status.className='status';status.setAttribute('role','status');box.append(status);const completePlaybackCoverage=()=>{const duration=audio.duration,ranges=audio.played,tolerance=Math.min(0.08,Math.max(0.03,duration*0.005));if(!ranges||ranges.length===0)return false;let covered=0;for(let i=0;i<ranges.length;i++){const start=ranges.start(i),end=ranges.end(i);if(!Number.isFinite(start)||!Number.isFinite(end)||start<0||end<start||start>covered+tolerance)return false;covered=Math.max(covered,end);}return ranges.start(0)<=tolerance&&covered>=duration-tolerance;};const maybeUnlock=()=>{const duration=audio.duration;if(!finished&&played&&!audio.seeking&&Number.isFinite(duration)&&duration>0&&Number.isFinite(audio.currentTime)&&audio.currentTime>=duration-0.15&&completePlaybackCoverage()){finished=true;if(negativeButton)negativeButton.disabled=false;status.textContent='Full recording heard. Not Alexa is now available.';play.textContent='Play';}};audio.addEventListener('timeupdate',maybeUnlock);audio.addEventListener('ended',maybeUnlock);audio.addEventListener('error',()=>{status.className='status error';status.textContent='Recording playback failed. Refresh recordings and try another review.';play.textContent='Play';});
box.append(el('p','Listen through once to enable Not Alexa.'));box.append(el('p','For Not Alexa, choose a source if one fits. Otherwise describe the other sound.'));
const categories=[['song_music','Song/music'],['room_sound','Room sound'],['coughing','Coughing'],['people_talking','People talking']];let category=null;const picks=el('div'),otherWrap=el('label','Other sound (max 500 characters)'),note=el('textarea');note.maxLength=500;otherWrap.append(note);for(const [value,label] of categories){const button=el('button',label);button.type='button';button.setAttribute('aria-pressed','false');button.addEventListener('click',()=>{category=category===value?null:value;for(const [choiceValue] of categories){const choice=picks.querySelectorAll('button')[categories.findIndex(row=>row[0]===choiceValue)],selected=choiceValue===category;choice.setAttribute('aria-pressed',String(selected));choice.classList.toggle('selected',selected);}otherWrap.hidden=category!==null;otherWrap.setAttribute('aria-hidden',String(category!==null));});picks.append(button);}box.append(picks,otherWrap);
let busy=false;const buttons=[];async function save(label){if(busy)return;if(!played){status.className='status error';status.textContent='Play the recording before labeling it.';return;}if(label==='negative'&&!finished){status.className='status error';status.textContent='Listen through once to enable Not Alexa.';return;}if(label==='negative'&&category===null&&!note.value.trim()){status.className='status error';status.textContent='Choose a source or describe the other sound.';return;}busy=true;buttons.forEach(button=>button.disabled=true);status.textContent='Saving…';try{const result=await fetch('/api/label',{method:'POST',headers:{'Content-Type':'application/json','X-Review-Nonce':nonce},body:JSON.stringify({id:c.id,presence:label==='positive'?'unsure':'negative',wake_presence:label==='positive'?'present':'absent',review_category:label==='negative'?category:null,review_note:label==='negative'&&category===null?note.value.trim():'',confirm_reviewed:true,entire_clip_negative:label==='negative'&&finished})}),data=await result.json();if(!result.ok)throw Error(data.error||'Save refused.');status.textContent=label==='positive'?'Saved as Alexa heard; exact wake span is still required before training eligibility.':'Saved as Not Alexa after full playback confirmation.';document.querySelector('#action-status').textContent=status.textContent;await load();}catch(error){status.className='status error';status.textContent=error.message||'Save failed. Try again.';}finally{busy=false;buttons.forEach(button=>button.disabled=false);if(negativeButton)negativeButton.disabled=!finished;}}
for(const [label,text] of [['positive','Alexa'],['negative','Not Alexa']]){const button=el('button',text);button.disabled=label==='negative';if(label==='negative')negativeButton=button;button.addEventListener('click',()=>save(label));buttons.push(button);box.append(button);}if(!c.wav_sha256)box.append(el('p','No verified audio hash is available; playback remains subject to server-side identity checks.'));
}catch{document.querySelector('#summary').textContent='Could not refresh review data; the current recording remains visible.';let error=root.querySelector('.load-error');if(!error){error=el('p','Could not refresh review data; the current recording remains visible.');error.className='status error load-error';root.append(error);}}}
document.querySelector('#undo-review').addEventListener('click',async()=>{const button=document.querySelector('#undo-review'),status=document.querySelector('#undo-status');if(button.disabled||!button.dataset.token)return;button.disabled=true;status.textContent=' Undoing latest response…';try{const response=await fetch('/api/retract',{method:'POST',headers:{'Content-Type':'application/json','X-Review-Nonce':nonce},body:JSON.stringify({undo_token:button.dataset.token})}),data=await response.json();if(!response.ok)throw Error(data.error||'Undo refused.');if(data.restored_candidate_id)selectedId=data.restored_candidate_id;status.textContent=data.restored_candidate_id?' Latest response retracted; its recording is back in the queue without a label.':' Latest response retracted; its recording could not be returned to the current review queue.';await load();}catch(error){status.textContent=error.message||'Undo failed; no response was retracted.';button.disabled=false;}});
let cachedReviewState=null;
const originalStateFetch=globalThis.fetch.bind(globalThis);
globalThis.fetch=async(...args)=>{
 const response=await originalStateFetch(...args),url=typeof args[0]==='string'?args[0]:args[0].url;
 if(url==='/api/state'||url.startsWith('/api/state?')){try{cachedReviewState=await response.clone().json();}catch{}}
 return response;
};
const loadCandidates=load;
load=async function(){
 cachedReviewState=null;
 await loadCandidates();
 try{
  const state=cachedReviewState||{},readiness=state.readiness||{},reasons=Array.isArray(readiness.blocked_reasons)?readiness.blocked_reasons:[];
  const missingPositive=reasons.filter(reason=>/^missing (train|holdout) positive session groups$/.test(reason));
  const missingSides=missingPositive.map(reason=>reason.startsWith('missing train')?'train':'holdout').join(' and ');
  const prompt=missingPositive.length?`Collect clearly heard Alexa spans from independent sessions that include quiet speech and music/noise; ${missingSides} positive groups are missing. `:'';
  document.querySelector('#summary').textContent+=` · Reviewed data ${readiness.data_eligible?'eligible':'blocked'} · resource admission ${readiness.resource_admission_checked?'checked':'not checked'} · training ${state.training_started?'started':'not started'}. ${prompt}${reasons.join(' · ')}`;
  const root=document.querySelector('#card'),box=root.children[0];
  if(!box||box.localName!=='section')return;
  const audio=box.querySelector('audio');if(!audio)throw Error('audio element missing');
  const id=audio.src.split('/').pop(),card=(state.candidates||[]).find(row=>row.id===id),duration=card&&card.duration_seconds;
  const panel=el('fieldset');panel.className='span-controls';panel.setAttribute('aria-label','Optional confirmed Alexa span');
  panel.append(el('legend','Optional: save an exact Alexa span'));
  const startLabel=el('label','Wake start (seconds)'),start=el('input');start.type='number';start.min='0';start.max=String(duration??'');start.step='0.0000625';start.setAttribute('aria-label','Wake start seconds');startLabel.append(start);
  const endLabel=el('label','Wake end (seconds)'),end=el('input');end.type='number';end.min='0';end.max=String(duration??'');end.step='0.0000625';end.setAttribute('aria-label','Wake end seconds');endLabel.append(end);
  const markStart=el('button','Mark start at player time');markStart.type='button';
  const markEnd=el('button','Mark end at player time');markEnd.type='button';
  const confirmation=el('input');confirmation.type='checkbox';
  const confirmationLabel=el('label','I can clearly hear Alexa in this span');confirmationLabel.append(confirmation);
  const saveSpan=el('button','Save confirmed Alexa span');saveSpan.type='button';saveSpan.disabled=true;
  const spanStatus=el('p');spanStatus.className='small';spanStatus.setAttribute('role','status');
  panel.append(startLabel,endLabel,markStart,markEnd,confirmationLabel,saveSpan,spanStatus);box.append(panel);
  let servedPlayback=false,spanBusy=false;
  const bounds=()=>{
   if(!start.value.trim()||!end.value.trim()||!Number.isFinite(duration)||duration<=0)return null;
   const first=Number(start.value),last=Number(end.value);
   if(!Number.isFinite(first)||!Number.isFinite(last)||first<0||first>=last||last>duration)return null;
   if(Math.abs(first*16000-Math.round(first*16000))>1e-6||Math.abs(last*16000-Math.round(last*16000))>1e-6)return null;
   return [first,last];
  };
  const update=()=>{
   const sameKnownHash=!!card&&typeof card.wav_sha256==='string'&&/^[0-9a-f]{64}$/.test(card.wav_sha256);
   const valid=!!bounds()&&confirmation.checked===true;
   markStart.disabled=!servedPlayback||!Number.isFinite(audio.currentTime);
   markEnd.disabled=!servedPlayback||!Number.isFinite(audio.currentTime);
   saveSpan.disabled=spanBusy||!servedPlayback||!sameKnownHash||!valid;
  };
  audio.addEventListener('play',()=>{servedPlayback=true;update();});
  audio.addEventListener('error',()=>{servedPlayback=false;update();});
  for(const input of [start,end,confirmation]){input.addEventListener('input',update);input.addEventListener('change',update);}
  const playerTime=()=>String(Math.round(audio.currentTime*16000)/16000);
  markStart.addEventListener('click',()=>{start.value=playerTime();update();});
  markEnd.addEventListener('click',()=>{end.value=playerTime();update();});
  saveSpan.addEventListener('click',async()=>{
   const span=bounds();if(!span||!confirmation.checked||!servedPlayback||saveSpan.disabled){update();return;}
   spanBusy=true;update();spanStatus.textContent='Saving confirmed span…';
   try{
    const result=await fetch('/api/label',{method:'POST',headers:{'Content-Type':'application/json','X-Review-Nonce':nonce},body:JSON.stringify({id,presence:'positive',wake_presence:'present',wake_start:span[0],wake_end:span[1],confirm_alexa_span:true,confirm_reviewed:true})}),data=await result.json();
    if(!result.ok)throw Error(data.error||'Save refused.');
    spanStatus.textContent='Saved as a human-reviewed Alexa span.';document.querySelector('#action-status').textContent=spanStatus.textContent;await load();
   }catch(error){spanStatus.className='status error';spanStatus.textContent=error.message||'Span save failed. Try again.';spanBusy=false;update();}
  });
  update();
 }catch{
  const root=document.querySelector('#card'),failure=el('p','Optional span controls could not be initialized. Reload this page before saving a confirmed Alexa span.');
  failure.className='status error';failure.setAttribute('role','status');root.append(failure);
 }
};
load();</script></html>"""


class ReviewHandler(BaseHTTPRequestHandler):
    server_version = "MemoirWakeReview"

    def log_message(self, *_):
        pass  # URLs/nonce/body/audio provenance never enter request logs.

    def setup(self):
        super().setup()
        self.connection.settimeout(5)

    @property
    def origin(self):
        return f"http://127.0.0.1:{self.server.server_port}"

    def _host_allowed(self):
        return self.headers.get_all("Host", []) == [self.origin.removeprefix("http://")]

    def _origin_allowed(self, *, required=False):
        values = self.headers.get_all("Origin", [])
        return values == [self.origin] or (not required and not values)

    def _respond(self, code, payload, content_type="application/json; charset=utf-8"):
        data = payload.encode("utf-8") if isinstance(payload, str) else json.dumps(payload, allow_nan=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
        self.end_headers()
        self.wfile.write(data)

    @staticmethod
    def _safe_review_error(exc):
        """Return actionable validation text without exposing paths or raw exceptions."""
        message = str(exc)
        exact = {
            "Play the recording before submitting a listening review": "Play this recording before submitting.",
            "Recording changed since playback; play it again before reviewing": "The recording changed. Play it again before reviewing.",
            "Listening review and capture-session confirmation are required": "Confirm listening and the real capture session.",
            "Unsupported review fields": "The review request contains unsupported fields.",
            "Unknown candidate": "This recording is no longer in the review queue. Reload the page.",
            "candidate missing, pruned, outside archive, or ambiguous": "The archived recording is no longer available for review.",
            "invalid human review store schema": "The existing review store failed validation; no review was saved.",
            "review store is locked by another writer": "Another review save is in progress. Retry this review.",
            "presence must be positive, negative or unsure": "Choose positive, negative, or unsure.",
            "addressed_assistant must be yes, no or unsure; wake presence alone is not intent": "Choose yes, no, or unsure for assistant addressing.",
            "wake span bounds must be finite seconds": "Enter finite wake start and end times.",
            "Alexa span bounds must be finite seconds": "Enter finite Alexa span start and end times.",
            "Alexa span must be non-empty and within the recording": "Alexa span end must follow start and stay within the recording.",
            "Confirmed Alexa span requires its clarity checkbox and exact bounds": "Check that Alexa is clearly audible in this exact span, then set both bounds.",
            "wake span must align to PCM samples": "Wake bounds must align to the recording samples; set them with the player buttons.",
            "wake span must be non-empty and within the WAV": "Wake end must be after wake start and both must be within the recording.",
            "negative review requires entire-clip listening confirmation": "A negative review requires confirming you listened to the whole clip.",
            "positive review cannot assert entire-clip negative": "A positive review cannot use the whole-clip negative confirmation.",
            "wake spans are only valid for positive reviews": "Wake start and end are only used for positive reviews.",
            "WAV hash mismatch: review is stale": "The recording changed; this review is stale and was not saved.",
            "WAV frame metadata mismatch": "Recording metadata changed; reload and review the current clip.",
        }
        if message in exact:
            return exact[message]
        for field in ("reviewer", "session_id", "source_provenance"):
            if message == field + " is required":
                return {"reviewer": "Enter your reviewer name.",
                        "session_id": "Enter the real capture-session ID; do not guess it.",
                        "source_provenance": "Describe the recording you listened to."}[field]
        if "positive" in message and "wake" in message and "span" in message:
            return "Enter an exact, non-empty wake span within the clip."
        if "conflicting" in message or "overlapping reviewed wake spans" in message:
            return "This review conflicts with an existing reviewed recording/session and needs reconciliation."
        return "Review could not be saved. Check the highlighted review fields and try again; no training started."

    def do_GET(self):
        if (not self._host_allowed() or not self._origin_allowed()
                or self.headers.get("Sec-Fetch-Site") == "cross-site"):
            self._respond(403, {"error": "Loopback Host required"})
            return
        url = urlsplit(self.path)
        if url.fragment or url.scheme or url.netloc:
            self._respond(400, {"error": "Query parameters are not accepted"})
            return
        if url.path == "/":
            if url.query:
                self._respond(400, {"error": "Query parameters are not accepted"})
                return
            self._respond(200, PAGE.replace("__NONCE__", json.dumps(self.server.review_state.nonce)), "text/html; charset=utf-8")
        elif url.path == "/api/state":
            try:
                params = (parse_qs(url.query, keep_blank_values=True, strict_parsing=True)
                          if url.query else {})
            except ValueError:
                self._respond(400, {"error": "Unsupported state query"})
                return
            if any(len(values) != 1 for values in params.values()):
                self._respond(400, {"error": "Unsupported state query"})
                return
            if set(params) - {"refresh", "mode"} or ("refresh" in params and params["refresh"] != ["1"]) or ("mode" in params and (params["mode"][0] not in REVIEW_MODES or params.get("refresh") != ["1"])):
                self._respond(400, {"error": "Unsupported state query"})
                return
            if params.get("refresh") == ["1"]:
                self.server.review_state.refresh(params.get("mode", [None])[0])
            elif params:
                self._respond(400, {"error": "Unsupported state query"})
                return
            self._respond(200, self.server.review_state.summary())
        elif url.path.startswith("/audio/"):
            if url.query:
                self._respond(400, {"error": "Query parameters are not accepted"})
                return
            opaque = url.path.removeprefix("/audio/")
            try:
                with self.server.review_state.lock:
                    data, digest = self.server.review_state.audio(opaque)
                    self.send_response(200)
                    self.send_header("Content-Type", "audio/wav")
                    self.send_header("Content-Length", str(len(data)))
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("X-Content-Type-Options", "nosniff")
                    self.end_headers()
                    self.wfile.write(data)
                    self.server.review_state.played[opaque] = digest
            except (OSError, ValueError):
                self._respond(404, {"error": "Recording unavailable or invalid"})
        else:
            self._respond(404, {"error": "Unknown local endpoint"})

    def do_POST(self):
        nonces = self.headers.get_all("X-Review-Nonce", [])
        nonce = nonces[0] if len(nonces) == 1 else ""
        valid_nonce = nonce.isascii() and secrets.compare_digest(nonce, self.server.review_state.nonce)
        if (not self._host_allowed() or not self._origin_allowed(required=True)
                or not valid_nonce or self.headers.get("Sec-Fetch-Site") == "cross-site"):
            self._respond(403, {"error": "Local Origin, Host and review nonce required"})
            return
        if self.path not in ("/api/label", "/api/retract") or self.headers.get("Transfer-Encoding"):
            self._respond(400, {"error": "Unsupported review request"})
            return
        try:
            if len(self.headers.get_all("Content-Length", [])) != 1:
                raise ValueError("Exactly one content length required")
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            length = 0
        if not 0 < length <= MAX_BODY_BYTES:
            self._respond(413, {"error": "Review request size rejected"})
            return
        if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
            self._respond(415, {"error": "JSON review required"})
            return
        try:
            body = json.loads(self.rfile.read(length), parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Non-finite JSON value")))
            if self.path == "/api/retract":
                self._respond(200, self.server.review_state.retract(body))
            else:
                self._respond(200, self.server.review_state.label(body))
        except (OSError, ValueError, TypeError) as exc:
            self._respond(400, {"error": self._safe_review_error(exc)})


class ExclusiveLoopbackServer(ThreadingHTTPServer):
    # HTTPServer normally enables SO_REUSEADDR. On Windows that can permit a
    # second live listener on the same port, routing labels to the wrong nonce
    # or review store. Do not relax uniqueness merely to reuse a recent port.
    allow_reuse_address = False
    allow_reuse_port = False

    def server_bind(self):
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def create_server(state, port=DEFAULT_PORT):
    port = int(port)
    if not 0 <= port <= 65535:
        raise ValueError("Invalid loopback port")
    choices = [port] if port == 0 else range(port, min(port + 10, 65536))
    last_error = None
    for candidate in choices:
        try:
            server = ExclusiveLoopbackServer(("127.0.0.1", candidate), ReviewHandler)
            server.review_state = state
            server.daemon_threads = True
            return server
        except OSError as exc:
            last_error = exc
    raise last_error or ValueError("No free loopback port")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--telemetry", default=str(review.TELEMETRY))
    parser.add_argument("--audio-root", default=str(review.AUDIO_ROOT))
    parser.add_argument("--project-root", default=str(review.PROJECT_ROOT))
    parser.add_argument("--store", default=str(DEFAULT_STORE))
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args(argv)
    state = ReviewState(args.telemetry, args.audio_root, args.project_root, args.store)
    server = create_server(state, args.port)
    print(f"Listening review: http://127.0.0.1:{server.server_port} (training has not started)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
