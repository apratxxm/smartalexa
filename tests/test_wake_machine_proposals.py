"""Offline safety tests for machine-generated wake proposal rows."""
import builtins
import importlib.util
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
import wave
from unittest.mock import patch

from assistant.experimental.wake_train import machine_proposals as proposals


class MachineProposalTests(unittest.TestCase):
    def test_default_import_does_not_load_model_dependencies(self):
        self.assertNotIn("faster_whisper", __import__("sys").modules)

    def test_exact_token_only_and_no_asr_based_negative(self):
        self.assertEqual(proposals.proposal_presence("Alexa, stop"), "positive")
        self.assertEqual(proposals.proposal_presence("Alexis"), "unsure")
        self.assertEqual(proposals.proposal_presence(""), "unsure")
        self.assertEqual(proposals.proposal_presence(None), "unsure")

    def test_low_memory_fails_before_loader(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "model.bin").write_bytes(b"x")
            (root / "config.json").write_text("{}", encoding="utf-8")
            loaded = []
            with self.assertRaises(MemoryError):
                proposals.make_local_decoder(root, memory_gate=lambda: 1,
                                             loader=lambda *a, **k: loaded.append((a, k)))
            self.assertEqual(loaded, [])

    def test_known_reviewed_clip_is_never_proposed(self):
        with tempfile.TemporaryDirectory() as folder:
            audio = Path(folder) / "clip.wav"
            audio.write_bytes(b"fixture")
            candidate = {"event_id": "known", "audio_field": "audio_path", "path": str(audio)}
            human = {"event_id": "known", "audio_field": "audio_path", "presence": "negative"}
            with patch.object(proposals.review, "review_history", return_value=([human], [], [])), \
                    patch.object(proposals.review, "audio_path", return_value=audio), \
                    patch.object(proposals.review, "wav_identity", return_value={"wav_sha256": "abc", "n_frames": 16000}):
                report = proposals.build_proposals(telemetry="", candidates=[candidate])
            self.assertEqual(report["rows"], [])

    def test_duplicate_hash_and_asr_without_teacher_remain_unsure(self):
        with tempfile.TemporaryDirectory() as folder:
            audio = Path(folder) / "clip.wav"
            audio.write_bytes(b"fixture")
            candidates = [{"event_id": key, "audio_field": "audio_path", "path": str(audio)}
                          for key in ("one", "two")]
            empty_history = ([], [], [])
            identity = {"wav_sha256": "same", "n_frames": 16000}
            with patch.object(proposals.review, "review_history", return_value=empty_history), \
                    patch.object(proposals.review, "audio_path", return_value=audio), \
                    patch.object(proposals.review, "wav_identity", return_value=identity), \
                    patch.object(proposals, "_asr_evidence", return_value={
                        "one": {"text": "Alexa"}, "two": {"text": "silence"}}):
                report = proposals.build_proposals(telemetry="", candidates=candidates)
            self.assertEqual(len(report["rows"]), 1)
            self.assertEqual(report["rows"][0]["proposed_presence"], "unsure")
            self.assertFalse(report["rows"][0]["human_reviewed"])
            self.assertFalse(report["rows"][0]["training_eligible"])

    def test_output_refuses_existing_and_outside_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            existing = Path(folder) / "x.json"
            existing.write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                proposals._safe_output(existing)
            with self.assertRaises(ValueError):
                proposals._safe_output(proposals.OUTPUT_ROOT.parent / "outside.json")

    def test_local_model_requires_both_metadata_files(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                proposals.required_free_bytes(folder)

    def test_bool_and_float_candidate_limits_are_rejected(self):
        for value in (True, 1.0, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                proposals.build_proposals(telemetry="", candidates=[], max_candidates=value)

    def test_asr_numeric_evidence_is_finite_and_json_safe(self):
        self.assertIsNone(proposals._finite_number(float("nan")))
        self.assertIsNone(proposals._finite_number(float("inf")))
        self.assertIsNone(proposals._finite_number(True))
        self.assertEqual(proposals._finite_number(0.5), 0.5)
        self.assertEqual(proposals._safe_scores({"good": 0.4, "bad": math.inf}), {"good": 0.4})

    def test_invalid_output_preflight_does_not_load_teacher(self):
        with patch.object(proposals, "make_local_decoder") as loader:
            with self.assertRaises(SystemExit):
                proposals.main(["--output", str(Path(tempfile.gettempdir()) / "outside-proposals.json"),
                                "--teacher-model", "C:/not-a-real-model"])
        loader.assert_not_called()

    def test_cli_uses_default_human_review_store(self):
        with tempfile.TemporaryDirectory() as folder:
            store_path = Path(folder) / "reviews.json"
            store_path.write_text(json.dumps({"schema": proposals.review.STORE_SCHEMA,
                                              "reviews": []}), encoding="utf-8")
            captured = {}
            def build(**kwargs):
                captured.update(kwargs)
                return {"rows": [], "schema": proposals.PROPOSAL_SCHEMA}
            with patch.object(proposals.review_server, "DEFAULT_STORE", store_path), \
                    patch.object(proposals, "build_proposals", side_effect=build):
                self.assertEqual(proposals.main([]), 0)
            self.assertEqual(captured["store"]["schema"], proposals.review.STORE_SCHEMA)

    def test_invalid_human_metadata_fails_before_teacher_loader(self):
        cases = (
            {"schema": proposals.review.STORE_SCHEMA,
             "reviews": [{"event_id": "e", "wav_sha256": "wrong"}]},
            {"schema": proposals.review.STORE_SCHEMA,
             "reviews": [{"event_id": "e", "path": "C:/outside/archive.wav"}]},
        )
        for index, store in enumerate(cases):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as folder:
                store_path = Path(folder) / "human.json"
                store_path.write_text(json.dumps(store), encoding="utf-8")
                with patch.object(proposals, "make_local_decoder") as loader, \
                        patch.object(proposals.review, "review_history", side_effect=ValueError("invalid human metadata")):
                    with self.assertRaises(SystemExit):
                        proposals.main(["--store", str(store_path), "--teacher-model", "C:/local/model"])
                loader.assert_not_called()

    def test_trigger_stt_identity_is_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            telemetry = Path(folder) / "events.jsonl"
            telemetry.write_text(json.dumps({"event_id": "e", "trigger_stt_model": "local-trigger-v2"}) + "\n",
                                 encoding="utf-8")
            self.assertEqual(proposals._asr_evidence(telemetry, ["e"])["e"]["model_id"],
                             "local-trigger-v2")

    def test_candidate_source_requests_recent_selection(self):
        observed = {}
        def recent(telemetry, audio_root, project_root, *, limit, selection):
            observed.update(limit=limit, selection=selection)
            return []
        with patch.object(proposals.review_server, "recent_candidates", side_effect=recent), \
                patch.object(proposals.review, "review_history", return_value=([], [], [])):
            proposals.build_proposals(telemetry="", max_candidates=7)
        self.assertEqual(observed, {"limit": 7, "selection": "recent"})


if __name__ == "__main__":
    unittest.main()
