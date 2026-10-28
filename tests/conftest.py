"""Keep tests independent of the developer's real local track index."""
import logging

import pytest


@pytest.fixture(autouse=True)
def _restore_memoir_logger():
    logger = logging.getLogger("memoir")
    handlers = logger.handlers[:]
    level, propagate, disabled = logger.level, logger.propagate, logger.disabled
    yield
    for handler in logger.handlers:
        if handler not in handlers:
            handler.close()
    logger.handlers = handlers
    logger.setLevel(level)
    logger.propagate = propagate
    logger.disabled = disabled


@pytest.fixture(autouse=True)
def _no_real_catalogue(tmp_path, monkeypatch):
    from assistant import catalogue
    monkeypatch.setattr(catalogue, "INDEX_PATH", tmp_path / "missing-catalogue.json")


@pytest.fixture(autouse=True)
def _no_real_spotify_backoff(tmp_path, monkeypatch):
    # A real ban recorded on this machine must not leak into tests.
    from assistant import spotify
    monkeypatch.setattr(spotify, "BACKOFF_PATH", tmp_path / "missing-backoff.json")


@pytest.fixture(autouse=True)
def _no_real_radio_history(tmp_path, monkeypatch):
    # Radio history is written after playback; tests must not touch the real file.
    from assistant import catalogue
    monkeypatch.setattr(catalogue, "RADIO_HISTORY_PATH", tmp_path / "missing-radio-history.json")
