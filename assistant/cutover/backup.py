from __future__ import annotations

import hashlib
import fnmatch
import json
import os
import stat
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


MANIFEST = ".cutover-manifest.json"


def _exclusion(name: str, root=None):
    parts = PurePosixPath(name).parts
    for index, part in enumerate(parts):
        part = part.lower()
        if part.startswith(".env") and part != ".env.example":
            return "environment"
        if part in {".git", "__pycache__", ".pytest_cache", ".cache"} or part.startswith((".spotify_cache", ".cache-")):
            return "metadata or cache"
        if part.endswith((".pyc", ".log", ".pem", ".key")) or fnmatch.fnmatchcase(part, "credentials*.json") or fnmatch.fnmatchcase(part, "*token*.json"):
            return "generated or credential file"
        if part.startswith(".venv") or (part in {"venv", "env"} and root is not None and (root.joinpath(*parts[:index + 1]) / "pyvenv.cfg").is_file()):
            return "virtual environment"
        if index == 1 and parts[0].lower() == "assistant" and part == "logs":
            return "assistant logs"
    return None


def _excluded(name: str) -> bool:
    return bool(_exclusion(name))


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
                "git_status_porcelain": _git(root, "status", "--porcelain"), "files": {}, "excluded": {}}
    inventory = set()
    for arguments in (("ls-files", "-z"), ("ls-files", "-z", "--others", "--exclude-standard")):
        output = subprocess.check_output(["git", "-C", str(root), *arguments])
        inventory.update(os.fsdecode(path).replace("\\", "/") for path in output.split(b"\0") if path)
    created_archive = False
    try:
        raw = dest.open("xb")
        created_archive = True
        with raw, zipfile.ZipFile(raw, "w", zipfile.ZIP_DEFLATED) as archive:
            def walk_error(error):
                raise error
            for directory, dirs, files in os.walk(root, followlinks=False, onerror=walk_error):
                base = Path(directory)
                for name in list(dirs):
                    path = base / name
                    relative = path.relative_to(root).as_posix()
                    reason = _exclusion(relative, root)
                    if reason:
                        print(f"Excluded: {relative}/")
                        manifest["excluded"][relative + "/"] = reason
                        dirs.remove(name)
                    elif path.is_symlink():
                        raise ValueError(f"Symlink not supported: {relative}")
                for name in files:
                    path = base / name
                    relative = path.relative_to(root).as_posix()
                    reason = _exclusion(relative, root)
                    if reason:
                        print(f"Excluded: {relative}")
                        manifest["excluded"][relative] = reason
                        continue
                    if relative == MANIFEST or path.is_symlink():
                        raise ValueError(f"Unsupported source entry: {relative}")
                    digest = hashlib.sha256()
                    size = 0
                    before = path.stat()
                    with path.open("rb") as source, archive.open(relative, "w") as target:
                        while chunk := source.read(1024 * 1024):
                            digest.update(chunk)
                            size += len(chunk)
                            target.write(chunk)
                    after = path.stat()
                    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
                        raise ValueError(f"Source changed during backup: {relative}")
                    manifest["files"][relative] = {"sha256": digest.hexdigest(), "size": size}
            for name in inventory:
                if name not in manifest["files"]:
                    reason = _exclusion(name, root)
                    if not reason:
                        raise ValueError(f"Inventory path not archived: {name}")
                    manifest["excluded"][name] = reason
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
    for name in names:
        parts = PurePosixPath(name).parts
        for index, part in enumerate(parts):
            if part.lower() in {"env", "venv"} and "/".join(parts[:index + 1] + ("pyvenv.cfg",)) in names:
                raise ValueError(f"Excluded virtual environment: {name}")
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


def restore_to(zip_path, manifest_path, source_root, scratch_parent):
    parent = Path(scratch_parent).resolve(strict=True)
    live = Path(source_root).resolve(strict=True)
    if not parent.is_dir():
        raise ValueError("Restore parent must be a directory")
    with Path(manifest_path).open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if Path(manifest["source_root"]).resolve() != live:
        raise ValueError("Source root differs from manifest")
    tool_root = Path(__file__).resolve().parents[2]
    if _inside(parent, live) or _inside(parent, tool_root):
        raise ValueError("Restore destination overlaps the live root")
    with zipfile.ZipFile(zip_path) as archive:
        _validate(archive, manifest)
        scratch = Path(tempfile.mkdtemp(prefix="cutover-restore-", dir=parent))
        for name, expected in manifest["files"].items():
            destination = scratch.joinpath(*PurePosixPath(name).parts)
            current = scratch
            for component in PurePosixPath(name).parts[:-1]:
                current = current / component
                if not current.exists() and not current.is_symlink():
                    current.mkdir()
                details = os.lstat(current)
                if stat.S_ISLNK(details.st_mode) or getattr(details, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0):
                    raise ValueError(f"Reparse point in restore path: {current}")
            digest = hashlib.sha256()
            size = 0
            with archive.open(name) as source, destination.open("xb") as output:
                while chunk := source.read(1024 * 1024):
                    output.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
            if expected != {"sha256": digest.hexdigest(), "size": size}:
                raise ValueError(f"Restored file mismatch: {name}")
    return scratch


def legacy_startup_check(restored_root, python, *, live_root=None):
    """Check imports and prerequisites, not actual assistant startup."""
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
                   "model_assets": assets, "scope": "import-level check; not a live startup"})
    result["unverified"] = [name for name, item in assets.items() if item["status"] == "named_model_unverified"]
    result["ok"] = (completed.returncode == 0 and result.get("imports_ok") is True
                    and result.get("runtime_module") == "assistant.main" and result["env_exists"]
                    and bool(result.get("packages")) and all(value is True for value in result["packages"].values())
                    and all(item["status"] not in {"missing", "missing_config"} for item in assets.values()))
    return result
