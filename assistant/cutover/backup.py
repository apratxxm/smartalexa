from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


MANIFEST = ".cutover-manifest.json"


def _excluded(name: str) -> bool:
    parts = PurePosixPath(name).parts
    for index, part in enumerate(parts):
        part = part.lower()
        if part == ".env.example":
            continue
        if (part == ".env" or part.startswith(".env.") or part.startswith(".venv")
                or part in {".git", "__pycache__", ".pytest_cache"}
                or part.endswith((".pyc", ".pem", ".key"))
                or "token" in part or "secret" in part):
            return True
        if index == 1 and parts[0].lower() == "assistant" and part == "logs":
            return True
    return False


def _inside(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True,
                                   encoding="utf-8", errors="replace").strip()


def make_backup(root, dest_zip, *, manifest_path):
    root = Path(root).resolve(strict=True)
    dest = Path(dest_zip).resolve()
    manifest_file = Path(manifest_path).resolve()
    if not root.is_dir() or _inside(dest, root) or _inside(manifest_file, root):
        raise ValueError("Archive and manifest must be outside the source tree")
    if dest == manifest_file or dest.exists() or manifest_file.exists():
        raise FileExistsError("Archive or manifest already exists")
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(),
                "source_root": str(root), "git_head": _git(root, "rev-parse", "HEAD"),
                "git_status_porcelain": _git(root, "status", "--porcelain"), "files": {}}
    created_archive = False
    try:
        raw = dest.open("xb")
        created_archive = True
        with raw, zipfile.ZipFile(raw, "w", zipfile.ZIP_DEFLATED) as archive:
            for directory, dirs, files in os.walk(root, followlinks=False):
                base = Path(directory)
                for name in list(dirs):
                    path = base / name
                    relative = path.relative_to(root).as_posix()
                    if _excluded(relative):
                        print(f"Excluded: {relative}/")
                        dirs.remove(name)
                    elif path.is_symlink():
                        raise ValueError(f"Symlink not supported: {relative}")
                for name in files:
                    path = base / name
                    relative = path.relative_to(root).as_posix()
                    if _excluded(relative):
                        print(f"Excluded: {relative}")
                        continue
                    if relative == MANIFEST or path.is_symlink():
                        raise ValueError(f"Unsupported source entry: {relative}")
                    digest = hashlib.sha256()
                    size = 0
                    with path.open("rb") as source, archive.open(relative, "w") as target:
                        while chunk := source.read(1024 * 1024):
                            digest.update(chunk)
                            size += len(chunk)
                            target.write(chunk)
                    manifest["files"][relative] = {"sha256": digest.hexdigest(), "size": size}
            archive.writestr(MANIFEST, json.dumps(manifest, indent=2).encode("utf-8"))
        with manifest_file.open("x", encoding="utf-8") as output:
            json.dump(manifest, output, indent=2)
    except Exception:
        if created_archive:
            dest.unlink()
        raise
    return manifest


def _validate(archive, manifest):
    entries = archive.infolist()
    names = [entry.filename for entry in entries]
    if len(names) != len(set(names)) or names.count(MANIFEST) != 1:
        raise ValueError("Duplicate or missing manifest entry")
    if json.loads(archive.read(MANIFEST)) != manifest:
        raise ValueError("Embedded manifest differs from supplied manifest")
    expected = manifest["files"]
    if set(names) != set(expected) | {MANIFEST}:
        raise ValueError("Archive entries differ from manifest")
    for entry in entries:
        name = entry.filename
        if name == MANIFEST:
            continue
        parts = PurePosixPath(name).parts
        if (not parts or name.startswith("/") or "\\" in name or ":" in name
                or any(part in {".", ".."} for part in name.split("/"))
                or entry.is_dir() or stat.S_ISLNK(entry.external_attr >> 16)
                or _excluded(name)):
            raise ValueError(f"Unsafe or excluded archive entry: {name}")
        digest = hashlib.sha256()
        size = 0
        with archive.open(entry) as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
        if expected[name] != {"sha256": digest.hexdigest(), "size": size}:
            raise ValueError(f"Hash or size mismatch: {name}")


def verify_backup(zip_path, manifest_path):
    with Path(manifest_path).open(encoding="utf-8") as source:
        manifest = json.load(source)
    with zipfile.ZipFile(zip_path) as archive:
        _validate(archive, manifest)
    return True


def restore_to(zip_path, scratch_dir):
    scratch = Path(scratch_dir).resolve(strict=True)
    if not scratch.is_dir() or any(scratch.iterdir()):
        raise ValueError("Restore destination must be an existing empty directory")
    with zipfile.ZipFile(zip_path) as archive:
        manifest = json.loads(archive.read(MANIFEST))
        live = Path(manifest["source_root"]).resolve()
        tool_root = Path(__file__).resolve().parents[2]
        if (_inside(scratch, live) or _inside(live, scratch)
                or _inside(scratch, tool_root) or _inside(tool_root, scratch)):
            raise ValueError("Restore destination overlaps the live root")
        _validate(archive, manifest)
        for name, expected in manifest["files"].items():
            destination = scratch.joinpath(*PurePosixPath(name).parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            size = 0
            with archive.open(name) as source, destination.open("xb") as output:
                while chunk := source.read(1024 * 1024):
                    output.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
            if expected != {"sha256": digest.hexdigest(), "size": size}:
                raise ValueError(f"Restored file mismatch: {name}")
    return manifest


def legacy_startup_check(restored_root, python, *, live_root=None):
    restored = Path(restored_root).resolve(strict=True)
    live = Path(live_root).resolve(strict=True) if live_root else Path(__file__).resolve().parents[2]
    with (restored / "assistant" / "config.toml").open("rb") as source:
        config = tomllib.load(source)
    assets = {}
    for section, key in (("wake", "model"), ("listener", "stt_model"), ("stt", "model")):
        value = config.get(section, {}).get(key)
        values = value if isinstance(value, list) else [value]
        for item in values:
            label = f"{section}.{key}:{item}"
            if not isinstance(item, str) or not item:
                assets[label] = {"status": "missing_config"}
                continue
            model = item.removeprefix("faster-whisper/")
            if "/" not in model and "\\" not in model and ":" not in model:
                assets[label] = {"status": "named_model_unverified"}
                continue
            path = Path(model)
            if not path.is_absolute():
                path = live / path
            assets[label] = {"status": "present" if path.exists() else "missing", "path": str(path)}
    probe = '''
import importlib, json, socket, subprocess, sys
from unittest.mock import patch
def blocked(*args, **kwargs):
    raise RuntimeError("external effect blocked")
result = {"packages": {}}
with patch.object(socket.socket, "connect", blocked), patch.object(socket, "create_connection", blocked), patch.object(subprocess, "Popen", blocked):
    for name in ("numpy", "sounddevice", "faster_whisper", "openwakeword", "spotipy", "dotenv"):
        try:
            importlib.import_module(name)
            result["packages"][name] = True
        except Exception as exc:
            result["packages"][name] = type(exc).__name__ + ": " + str(exc)
    try:
        import assistant.supervise as supervise
        import assistant.main
        result["runtime_module"] = supervise.resolve_runtime_module(supervise.load())
        result["imports_ok"] = True
    except Exception as exc:
        result["imports_ok"] = False
        result["error"] = type(exc).__name__ + ": " + str(exc)
print(json.dumps(result))
'''
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run([str(Path(python).resolve(strict=True)), "-B", "-c", probe],
                               cwd=restored, env=environment, capture_output=True,
                               text=True, timeout=60)
    try:
        result = json.loads(completed.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        result = {"imports_ok": False, "error": "Probe returned no valid result"}
    result.update({"returncode": completed.returncode, "env_exists": (live / ".env").is_file(),
                   "model_assets": assets, "scope": "mocked imports and resolver; not a live startup"})
    return result
