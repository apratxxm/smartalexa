from unittest.mock import Mock

import pytest

from assistant.resolution import AmbiguousMatch
from assistant.spotify import Spotify


def playlist(name, uri="spotify:playlist:one"):
    return {"name": name, "uri": uri}


def client(saved=(), public=(), second_page=None):
    subject = Spotify.__new__(Spotify)
    subject._cfg = {}
    subject._device_id = "test-device"
    subject._device_name = "Test device"
    subject._command_device_id = None
    subject._command_device_name = ""
    subject._command_depth = 0
    subject.sp = Mock()
    subject.sp.current_user_playlists.return_value = {
        "items": list(saved), "next": "page2" if second_page else None}
    subject.sp.next.return_value = {"items": second_page, "next": None}
    subject.sp.search.return_value = {"playlists": {"items": list(public)}}
    subject._call = Mock()
    subject._verify = Mock()
    return subject


def test_personal_missing_never_searches_or_plays_public():
    subject = client(public=[playlist("focus")])
    with pytest.raises(LookupError):
        subject.play_playlist("focus", library_only=True)
    subject.sp.search.assert_not_called()
    subject._call.assert_not_called()


def test_paginated_exact_match_beats_partial_and_verifies_context():
    subject = client([playlist("focus mix")], second_page=[playlist("focus", "target")])
    assert subject.play_playlist("focus", library_only=True) == "focus"
    subject.sp.next.assert_called_once()
    subject._call.assert_called_once_with(subject.sp.start_playback, context_uri="target")
    predicate = subject._verify.call_args.args[0]
    assert predicate({"is_playing": True, "context": {"uri": "target"}})
    assert not predicate({"is_playing": True, "context": {"uri": "other"}})
    assert not predicate({"is_playing": False, "context": {"uri": "target"}})
    subject.sp.search.assert_not_called()


@pytest.mark.parametrize("rows,query", [
    ([playlist("focus", "a"), playlist("focus", "b")], "focus"),
    ([playlist("focus work", "a"), playlist("focus study", "b")], "focus"),
    ([playlist("focus")], "focuz"),
])
def test_uncertain_saved_match_never_plays_or_falls_back(rows, query):
    subject = client(rows)
    with pytest.raises(AmbiguousMatch):
        subject.play_playlist(query)
    subject._call.assert_not_called()
    subject.sp.search.assert_not_called()


def test_duplicate_uri_is_not_ambiguous():
    subject = client([playlist("focus"), playlist("focus")])
    assert subject.play_playlist("focus", library_only=True) == "focus"
    subject._call.assert_called_once()


def test_public_exact_match_not_first_result():
    subject = client(public=[playlist("unrelated", "bad"), playlist("focus", "good")])
    assert subject.play_playlist("focus") == "focus"
    subject._call.assert_called_once_with(subject.sp.start_playback, context_uri="good")


@pytest.mark.parametrize("rows,error", [
    ([playlist("unrelated")], LookupError),
    ([playlist("focus", "a"), playlist("focus", "b")], AmbiguousMatch),
    ([None, playlist("!!!")], LookupError),
])
def test_public_results_cannot_force_unsafe_selection(rows, error):
    subject = client(public=rows)
    with pytest.raises(error):
        subject.play_playlist("focus")
    subject._call.assert_not_called()


def test_empty_name_rejected_before_catalog_access():
    subject = client([playlist("focus")])
    with pytest.raises(LookupError):
        subject.play_playlist("!!!")
    subject.sp.current_user_playlists.assert_not_called()
    subject._call.assert_not_called()
