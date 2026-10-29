import json
import subprocess
import sys
import zipfile
import os
import shutil
from pathlib import Path

import pytest

from assistant.cutover.backup import MANIFEST, legacy_startup_check, make_backup, restore_to, verify_backup


@pytest.fixture
def archive(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "--allow-empty", "-qm", "initial"], check=True)
    for name in ["plain.txt", "tests/test_token_parser.py", "secret_notes.txt", ".env", ".env.local", ".env-production", ".envrc", ".env.example", ".venv/x", "venv/pyvenv.cfg", "venv/lib/a", "env/pyvenv.cfg", "python-env/pyvenv.cfg", "python-env/lib/a", ".cache", "nested/.cache/a", ".cache-user", ".spotify_cache-x", "credentials.json", "credentials-foo.json", "api_token.json", "__pycache__/a", ".pytest_cache/a", "x.pyc", "other.log", "assistant/logs/a", "thing.pem", "thing.key"]:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name, encoding="utf-8")
    zip_path = tmp_path / "backup.zip"
    manifest = tmp_path / "manifest.json"
    make_backup(root, zip_path, manifest_path=manifest)
    return root, zip_path, manifest


def test_backup_excludes_secrets_and_restores_untracked(capsys, archive, tmp_path):
    root, zip_path, manifest = archive
    with zipfile.ZipFile(zip_path) as saved:
        assert set(saved.namelist()) == {"plain.txt", ".env.example", "tests/test_token_parser.py", "secret_notes.txt", MANIFEST}
    assert "Excluded:" in capsys.readouterr().out
    assert verify_backup(zip_path, manifest)
    scratch = restore_to(zip_path, manifest, root, tmp_path)
    assert (scratch / "plain.txt").read_text() == "plain.txt"
    with pytest.raises(FileExistsError):
        make_backup(root, zip_path, manifest_path=manifest)


def test_verify_rejects_secret_even_when_manifest_matches(archive):
    _, zip_path, manifest_path = archive
    manifest = json.loads(manifest_path.read_text())
    import hashlib
    manifest["files"]["hidden-token.json"] = {"sha256": hashlib.sha256(b"secret").hexdigest(), "size": 6}
    with zipfile.ZipFile(zip_path, "w") as saved:
        saved.writestr("plain.txt", "plain.txt")
        saved.writestr(".env.example", ".env.example")
        saved.writestr("tests/test_token_parser.py", "tests/test_token_parser.py")
        saved.writestr("secret_notes.txt", "secret_notes.txt")
        saved.writestr("hidden-token.json", "secret")
        saved.writestr(MANIFEST, json.dumps(manifest))
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="excluded"):
        verify_backup(zip_path, manifest_path)


def test_tamper_and_unsafe_restore(archive, tmp_path):
    root, zip_path, manifest = archive
    with pytest.raises(ValueError):
        restore_to(zip_path, manifest, root, root)
    with zipfile.ZipFile(zip_path) as saved:
        embedded = saved.read(MANIFEST)
    with zipfile.ZipFile(zip_path, "w") as saved:
        saved.writestr("plain.txt", "tampered")
        saved.writestr(".env.example", ".env.example")
        saved.writestr("tests/test_token_parser.py", "tests/test_token_parser.py")
        saved.writestr("secret_notes.txt", "secret_notes.txt")
        saved.writestr(MANIFEST, embedded)
    with pytest.raises(ValueError):
        verify_backup(zip_path, manifest)


def test_restore_rejects_changed_archive_and_embedded_manifest(archive, tmp_path):
    import hashlib
    root, zip_path, manifest_path = archive
    with zipfile.ZipFile(zip_path) as saved:
        contents = {name: saved.read(name) for name in saved.namelist()}
    manifest = json.loads(contents[MANIFEST])
    contents["plain.txt"] = b"changed"
    manifest["files"]["plain.txt"] = {"sha256": hashlib.sha256(b"changed").hexdigest(), "size": 7}
    contents[MANIFEST] = json.dumps(manifest).encode()
    with zipfile.ZipFile(zip_path, "w") as saved:
        for name, data in contents.items():
            saved.writestr(name, data)
    with pytest.raises(ValueError, match="Embedded manifest"):
        restore_to(zip_path, manifest_path, root, tmp_path)


def test_restore_rejects_injected_symlink(archive, tmp_path, monkeypatch):
    from assistant.cutover import backup
    root, zip_path, manifest = archive
    outside = tmp_path / "outside"
    outside.mkdir()
    original = backup.tempfile.mkdtemp
    def inject(*args, **kwargs):
        directory = original(*args, **kwargs)
        try:
            os.symlink(outside, os.path.join(directory, "tests"), target_is_directory=True)
        except OSError as error:
            pytest.skip(f"Directory symlinks unavailable: {error}")
        return directory
    monkeypatch.setattr(backup.tempfile, "mkdtemp", inject)
    with pytest.raises(ValueError, match="Reparse point"):
        restore_to(zip_path, manifest, root, tmp_path)
    assert not list(outside.iterdir())


def test_legacy_cli_fails_closed(tmp_path):
    root = tmp_path / "restored"
    (root / "assistant").mkdir(parents=True)
    (root / "assistant" / "config.toml").write_text('[runtime]\nruntime = "core"\n')
    result = subprocess.run([sys.executable, "-m", "assistant.cutover", "legacy-check", "--restored-root", str(root), "--python", sys.executable, "--live-root", str(tmp_path)], capture_output=True, text=True, timeout=60)
    assert result.returncode != 0
    assert not json.loads(result.stdout)["ok"]


def test_legacy_check_in_temp_copy(tmp_path):
    root = tmp_path / "restored"
    package = root / "assistant"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "supervise.py").write_text('def load(): return {}\ndef resolve_runtime_module(config): return "assistant.main"\n')
    (package / "main.py").write_text('import socket\ntry:\n socket.create_connection(("example.invalid", 80))\nexcept RuntimeError:\n pass\nelse:\n raise AssertionError("network was not blocked")\n')
    (package / "config.toml").write_text('[wake]\nmodel = ["alexa"]\n[listener]\nstt_model = "missing/model"\n[stt]\nmodel = "missing/model"\n')
    result = legacy_startup_check(root, sys.executable, live_root=tmp_path)
    assert result["import_level"]["ok"] and result["import_level"]["runtime_module"] == "assistant.main"
    assert result["prerequisites"]["env_file"] == "missing"
    assert result["prerequisites"]["model:wake.model:alexa"] == "unverified"


def test_verify_rejects_arbitrarily_named_venv(archive):
    import hashlib
    _, zip_path, manifest_path = archive
    manifest = json.loads(manifest_path.read_text())
    with zipfile.ZipFile(zip_path) as saved:
        contents = {name: saved.read(name) for name in saved.namelist() if name != MANIFEST}
    contents["python-env/pyvenv.cfg"] = b"venv"
    manifest["files"]["python-env/pyvenv.cfg"] = {"sha256": hashlib.sha256(b"venv").hexdigest(), "size": 4}
    with zipfile.ZipFile(zip_path, "w") as saved:
        for name, data in contents.items():
            saved.writestr(name, data)
        saved.writestr(MANIFEST, json.dumps(manifest))
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="virtual environment"):
        verify_backup(zip_path, manifest_path)


def test_restored_legacy_constructs_before_mocked_main_loop(tmp_path):
    source = Path(__file__).resolve().parents[2] / "assistant"
    target = tmp_path / "assistant"
    shutil.copytree(source, target, ignore=shutil.ignore_patterns("models", "logs", "__pycache__", "*.pyc"))
    probe = '''
from unittest.mock import MagicMock, patch
import socket, subprocess
def blocked(*args, **kwargs): raise RuntimeError("external effects blocked")
with patch.object(socket.socket, "connect", blocked), patch.object(socket, "create_connection", blocked), patch.object(subprocess, "Popen", blocked):
 import assistant.main as main
 class Config:
  def get(self, name, default=None):
   return {"listener.mode": "wakeword", "conversation.enabled": False, "intent.backend": "rules", "codex.enabled": False}.get(name, default)
 with patch.object(main, "load", return_value=Config()), patch.object(main, "validate_async_config", return_value=False), patch.object(main, "Telemetry"), patch.object(main, "Feedback"), patch.object(main, "Spotify"), patch.object(main, "WakeDetector"), patch.object(main, "SileroVAD"), patch.object(main, "Transcriber"), patch.object(main, "Endpointer"), patch.object(main.ducking, "repair_orphaned_duck"), patch.object(main.ducking, "DuckLease"), patch.object(main.Assistant, "_load_vocabulary", return_value={"hints": [], "artists": []}), patch.object(main, "_record_crashes"), patch.object(main.Assistant, "start_health_monitor", return_value=None), patch.object(main.Assistant, "run") as loop, patch.object(main.Assistant, "shutdown"):
  main._run_instance()
  assert loop.call_count == 1
'''
    result = subprocess.run([sys.executable, "-B", "-c", probe], cwd=tmp_path, capture_output=True, text=True, timeout=60, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": ""})
    assert result.returncode == 0, result.stderr


def test_legacy_prerequisites_report_names_without_values(tmp_path, monkeypatch):
    from assistant.cutover import backup
    root = tmp_path / "root"
    (root / "assistant").mkdir(parents=True)
    (root / "assistant" / "config.toml").write_text('[wake]\nmodel = "alexa"\n[listener]\nstt_model = "model/a"\n[stt]\nmodel = "model/b"\n')
    (root / ".env").write_text('SPOTIPY_CLIENT_ID=private_value\nSPOTIPY_CLIENT_SECRET=private_value\n')
    (root / ".cache").write_text("private_value")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(backup.subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, stdout=json.dumps({"packages": {"numpy": True}, "imports_ok": True, "runtime_module": "assistant.main"}), stderr=""))
    result = backup.legacy_startup_check(root, sys.executable, live_root=root)
    assert result["prerequisites"]["env_key:SPOTIPY_CLIENT_ID"] == "present"
    assert result["prerequisites"]["env_key:SPOTIPY_REDIRECT_URI"] == "missing"
    assert result["prerequisites"]["codex_sign_in"] == "missing"
    assert result["prerequisites"]["spotify_token_cache"] == "present"
    assert "private_value" not in json.dumps(result)
    assert not result["ok"]
