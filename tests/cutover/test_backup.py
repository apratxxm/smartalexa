import json
import subprocess
import sys
import zipfile

import pytest

from assistant.cutover.backup import MANIFEST, legacy_startup_check, make_backup, restore_to, verify_backup


@pytest.fixture
def archive(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "--allow-empty", "-qm", "initial"], check=True)
    for name in ["plain.txt", ".env", ".env.local", ".env.example", ".venv/x", "__pycache__/a", ".pytest_cache/a", "x.pyc", "assistant/logs/a", "credentials/secret.txt", "api_token.txt", "thing.pem", "thing.key"]:
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
        assert set(saved.namelist()) == {"plain.txt", ".env.example", MANIFEST}
    assert "Excluded:" in capsys.readouterr().out
    assert verify_backup(zip_path, manifest)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    restore_to(zip_path, scratch)
    assert (scratch / "plain.txt").read_text() == "plain.txt"
    with pytest.raises(FileExistsError):
        make_backup(root, zip_path, manifest_path=manifest)


def test_verify_rejects_secret_even_when_manifest_matches(archive):
    _, zip_path, manifest_path = archive
    manifest = json.loads(manifest_path.read_text())
    import hashlib
    manifest["files"]["hidden-token.txt"] = {"sha256": hashlib.sha256(b"secret").hexdigest(), "size": 6}
    with zipfile.ZipFile(zip_path, "w") as saved:
        saved.writestr("plain.txt", "plain.txt")
        saved.writestr(".env.example", ".env.example")
        saved.writestr("hidden-token.txt", "secret")
        saved.writestr(MANIFEST, json.dumps(manifest))
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="excluded"):
        verify_backup(zip_path, manifest_path)


def test_tamper_and_unsafe_restore(archive, tmp_path):
    root, zip_path, manifest = archive
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    (scratch / "existing").touch()
    with pytest.raises(ValueError):
        restore_to(zip_path, scratch)
    empty = root / "empty"
    empty.mkdir()
    with pytest.raises(ValueError):
        restore_to(zip_path, empty)
    with zipfile.ZipFile(zip_path) as saved:
        embedded = saved.read(MANIFEST)
    with zipfile.ZipFile(zip_path, "w") as saved:
        saved.writestr("plain.txt", "tampered")
        saved.writestr(".env.example", ".env.example")
        saved.writestr(MANIFEST, embedded)
    with pytest.raises(ValueError):
        verify_backup(zip_path, manifest)


def test_legacy_check_in_temp_copy(tmp_path):
    root = tmp_path / "restored"
    package = root / "assistant"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "supervise.py").write_text('def load(): return {}\ndef resolve_runtime_module(config): return "assistant.main"\n')
    (package / "main.py").write_text('import socket\ntry:\n socket.create_connection(("example.invalid", 80))\nexcept RuntimeError:\n pass\nelse:\n raise AssertionError("network was not blocked")\n')
    (package / "config.toml").write_text('[wake]\nmodel = ["alexa"]\n[listener]\nstt_model = "missing/model"\n[stt]\nmodel = "missing/model"\n')
    result = legacy_startup_check(root, sys.executable, live_root=tmp_path)
    assert result["imports_ok"] and result["runtime_module"] == "assistant.main"
    assert not result["env_exists"]
    assert result["model_assets"]["wake.model:alexa"]["status"] == "named_model_unverified"
