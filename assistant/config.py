"""Config loading. Everything tunable lives in config.toml; secrets live in .env."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

try:
    import tomllib  # py3.11+
except ModuleNotFoundError:
    import tomli as tomllib  # py3.10

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "assistant" / "config.toml"


class Config:
    """Dotted read-only access over the parsed TOML, e.g. cfg.get('wake.cooldown_seconds')."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def get(self, path: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def path(self, key: str, default: str = "") -> Path:
        """Resolve a config value that names a file, relative to the project root."""
        raw = self.get(key, default)
        if not raw:
            return Path()
        p = Path(raw)
        return p if p.is_absolute() else ROOT / p


@lru_cache(maxsize=1)
def load() -> Config:
    load_dotenv(ROOT / ".env")
    with open(CONFIG_PATH, "rb") as fh:
        return Config(tomllib.load(fh))


def secret(name: str, required: bool = True) -> str:
    """Read a secret from the environment. Fails loudly and specifically."""
    value = os.getenv(name, "")
    if required and not value:
        raise RuntimeError(
            f"{name} is not set. Copy .env.example to .env and fill it in."
        )
    return value
