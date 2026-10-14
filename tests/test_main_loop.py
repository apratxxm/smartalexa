"""Exercise daemon loop control flow without devices, models, or Spotify."""
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import contextlib

import numpy as np
import pytest

from assistant import main
from assistant.config import Config


@pytest.fixture(autouse=True)
def _never_hold_native_power_requests(monkeypatch):
    """Loop fakes must never make real Windows sleep requests."""
    instances = []

    class NoopPowerGuard:
        def __init__(self, enabled):
            self.enabled = enabled
            self.updates = 0
            self.closed = False
            instances.append(self)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            self.closed = True
            return False

        def update(self, *, listening=True):
            self.updates += 1

    monkeypatch.setattr(main, "KeepAwakeGuard", NoopPowerGuard)
    return instances


def test_catalogue_freshness_loop_exits_when_automatic_refresh_is_off(monkeypatch):
    app = main.Assistant.__new__(main.Assistant)
    app.cfg = Config({"spotify": {"catalogue_auto_refresh": False}})
    app._stop = threading.Event()
    app.spotify = Mock()
    app.vocabulary = {"artists": ["artist"]}
    monkeypatch.setattr("assistant.catalogue.refresh_in_background",
                        Mock(side_effect=AssertionError("refresh started")))
    app._keep_catalogue_fresh()


def test_recent_nonempty_vocabulary_cache_skips_spotify_calls(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "ROOT", tmp_path)
    path = tmp_path / "assistant" / "logs" / ".vocabulary.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"hints": ["cached"], "artists": ["cached artist"]}',
                    encoding="utf-8")
    app = main.Assistant.__new__(main.Assistant)
    app.spotify = Mock()
    assert app._load_vocabulary() == {"hints": ["cached"], "artists": ["cached artist"]}
    app.spotify.vocabulary.assert_not_called()
    app.spotify.artist_names.assert_not_called()


@pytest.mark.parametrize("payload", ["[]", '"bad"', '{"hints": "bad"}',
                                      '{"artists": [null]}'])
def test_invalid_vocabulary_cache_falls_back_to_spotify(tmp_path, monkeypatch, payload):
    monkeypatch.setattr(main, "ROOT", tmp_path)
    path = tmp_path / "assistant" / "logs" / ".vocabulary.json"
    path.parent.mkdir(parents=True)
    path.write_text(payload, encoding="utf-8")
    app = main.Assistant.__new__(main.Assistant)
    app.spotify = Mock()
    app.spotify.vocabulary.return_value = ["fresh hint"]
    app.spotify.artist_names.return_value = ["fresh artist"]
    assert app._load_vocabulary() == {"hints": ["fresh hint"], "artists": ["fresh artist"]}
    app.spotify.vocabulary.assert_called_once()
    app.spotify.artist_names.assert_called_once()


def _candidate_clarification_app(answer):
    app = main.Assistant.__new__(main.Assistant)
    app.cfg = Config({})
    app.spotify = SimpleNamespace(last_resolution=None)
    app.telemetry = Mock()
    app.feedback = Mock()
    app._duck_music = lambda *_: contextlib.nullcontext()
    app._ask_and_listen = Mock(return_value=(answer, {"audio": np.zeros(512, dtype=np.int16)}, 12))
    app._execute = Mock(side_effect=lambda intent, event, **_: setattr(event, "outcome", "ok"))
    return app


@pytest.mark.parametrize("answer", ["yes", "yep", "Taaron Se", "play taaron se by Arpit Bala", "first"])
def test_known_artist_weak_title_confirms_supplied_candidate_once(answer):
    from assistant.resolution import AmbiguousMatch, Candidate
    app = _candidate_clarification_app(answer)
    candidate = Candidate("spotify:track:taaron", "Taaron Se", "Arpit Bala")
    original = main.Intent(main.Action.PLAY_TRACK, query="tarot by arpit bala", radio=False,
                           volume_after_play=20)
    exc = AmbiguousMatch(original.query, [candidate])
    exc.intent = original
    event = main.Event(intent_action=original.action.value, intent_query=original.query)
    app._handle_ambiguous_track(Mock(), event, exc)
    app._ask_and_listen.assert_called_once()
    assert app._ask_and_listen.call_args.args[1] == "Did you mean Taaron Se by Arpit Bala?"
    app._execute.assert_called_once()
    assert app._execute.call_args.args[0] is original
    assert app._execute.call_args.kwargs["confirmed_candidate"] is candidate
    assert event.outcome == "ok" and event.error == ""
    assert event.clarification_transcript == answer


@pytest.mark.parametrize("answer", ["", "no", "no Taaron Se", "pause", "perhaps", "play another song",
                                   "yes play a different song", "Taaron", "cancel"])
def test_candidate_unknown_negative_or_silent_reply_never_plays(answer):
    from assistant.resolution import AmbiguousMatch, Candidate
    app = _candidate_clarification_app(answer)
    event = main.Event(intent_action="PLAY_TRACK", intent_query="tarot by arpit bala")
    app._handle_ambiguous_track(Mock(), event, AmbiguousMatch(event.intent_query, [
        Candidate("spotify:track:taaron", "Taaron Se", "Arpit Bala")]))
    app._execute.assert_not_called()
    app._ask_and_listen.assert_called_once()
    assert event.outcome == "ambiguous"
    assert event.feedback_text == "No selection made."


@pytest.mark.parametrize("answer,plays", [("yes", True), ("no", False), ("", False)])
def test_music_confirmation_preserves_original_intent_and_never_auto_plays(answer, plays):
    from assistant.resolution import Candidate, MusicConfirmationRequired
    app = _candidate_clarification_app(answer)
    original = main.Intent(main.Action.QUEUE, query="red by artist", volume_after_play=20,
                           library_only=True, radio=False)
    candidate = Candidate("spotify:track:red", "Red", "Artist",
                          {"resolution_approximate": True})
    exc = MusicConfirmationRequired(original.query, [candidate])
    exc.intent = original
    event = main.Event(intent_action=original.action.value, intent_query=original.query,
                       intent_volume_after_play=20, intent_library_only=True)

    app._handle_music_confirmation(Mock(), event, exc)

    if plays:
        app._execute.assert_called_once()
        assert app._execute.call_args.args[0] is original
        assert app._execute.call_args.kwargs["confirmed_candidate"] is candidate
        assert original.volume_after_play == 20 and original.library_only and not original.radio
    else:
        app._execute.assert_not_called()
        assert event.outcome == "ambiguous"


def test_music_confirmation_error_precedes_lookup_fallback_and_keeps_resolution_budget():
    from contextlib import contextmanager
    from assistant.resolution import Candidate, MusicConfirmationRequired

    candidate = Candidate("spotify:track:red", "Red", "Artist",
                          {"resolution_approximate": True})
    intent = main.Intent(main.Action.PLAY_TRACK, query="red by artist", volume_after_play=20)
    observed = []

    class SpotifyStub:
        @contextmanager
        def resolution_budget(self, max_searches):
            observed.append(("budget", max_searches))
            yield {"attempts": 0}

        def cached_track(self, query):
            return None

        def resolve_track(self, query):
            observed.append(("resolve_track", query))
            raise MusicConfirmationRequired(query, [candidate])

    app = main.Assistant.__new__(main.Assistant)
    app.spotify = SpotifyStub()
    app._recover_track_from_audio = Mock(side_effect=AssertionError("no generic lookup recovery"))
    event = main.Event()
    with pytest.raises(MusicConfirmationRequired) as raised:
        app._resolve_available_track(intent, event, command_audio=np.zeros(512, dtype=np.int16),
                                     allow_album=True)
    assert raised.value.intent is intent
    assert observed == [("budget", 3), ("resolve_track", intent.query)]
    app._recover_track_from_audio.assert_not_called()


@pytest.mark.parametrize("wait_ms,expected", [(15, 15), (0, 0), (None, None),
                                               (float("nan"), None), (float("inf"), None),
                                               (float("-inf"), None)])
def test_capture_wait_duration_is_appended_to_event_without_other_capture_changes(wait_ms, expected):
    event = main.Event(capture_wait_ms=91)
    captured = {"wait_duration_ms": wait_ms, "end_reason": "timeout", "truncated": True}

    main.Assistant._capture_result(event, captured)

    assert event.capture_wait_ms == expected
    assert event.capture_end_reason == "timeout"
    assert event.capture_truncated is True


@pytest.mark.parametrize("answer,index", [("yes", None), ("Song", None), ("second", 1),
                                       ("Song Live", 1), ("Song by Artist", 0), ("third", None)])
def test_multiple_recordings_require_ordinal_or_unique_exact_label(answer, index):
    from assistant.resolution import AmbiguousMatch, Candidate
    app = _candidate_clarification_app(answer)
    choices = [Candidate("spotify:track:studio", "Song", "Artist"),
               Candidate("spotify:track:live", "Song Live", "Artist")]
    # A bare title uniquely identifies the studio candidate here.
    if answer == "Song":
        choices[1] = Candidate("spotify:track:guest", "Song", "Other Artist")
    event = main.Event(intent_action="PLAY_TRACK", intent_query="sang by artist")
    app._handle_ambiguous_track(Mock(), event, AmbiguousMatch(event.intent_query, choices))
    if index is None:
        app._execute.assert_not_called()
    else:
        assert app._execute.call_args.kwargs["confirmed_candidate"] is choices[index]
    app._ask_and_listen.assert_called_once()


def test_candidate_prompt_and_selection_distinguish_complete_collaboration_credits():
    from assistant.resolution import AmbiguousMatch, Candidate
    app = _candidate_clarification_app("Song by Primary, Guest B")
    choices = [Candidate("spotify:track:" + guest, "Song", "Primary", {
        "artists": [{"name": "Primary"}, {"name": "Guest " + guest}]}) for guest in ("A", "B")]
    event = main.Event(intent_action="PLAY_TRACK", intent_query="song by primary")
    app._handle_ambiguous_track(Mock(), event, AmbiguousMatch(event.intent_query, choices))
    prompt = app._ask_and_listen.call_args.args[1]
    assert "Guest A" in prompt and "Guest B" in prompt
    assert app._execute.call_args.kwargs["confirmed_candidate"] is choices[1]


@pytest.mark.parametrize("action", [main.Action.PLAY_TRACK, main.Action.PLAY_ALBUM])
def test_album_ambiguity_offers_typed_album_choices(action):
    from assistant.resolution import AmbiguousMatch, Candidate
    app = _candidate_clarification_app("second")
    original = main.Intent(action, query="Keychain Lalu by Arpit Bala", volume_after_play=20)
    candidates = [Candidate("spotify:album:first", "Keychain Laalu", "Arpit Bala"),
                  Candidate("spotify:album:second", "Keychain Laalu Live", "Arpit Bala")]
    exc = AmbiguousMatch(original.query, candidates)
    exc.intent = original
    event = main.Event(intent_action=action.value, intent_query=original.query)
    app._handle_ambiguous_track(Mock(), event, exc)
    assert "album Keychain Laalu" in app._ask_and_listen.call_args.args[1]
    assert app._execute.call_args.args[0] is original
    assert app._execute.call_args.kwargs["confirmed_candidate"] is candidates[1]


def test_album_candidates_cannot_answer_song_queue_ambiguity():
    from assistant.resolution import AmbiguousMatch, Candidate
    app = _candidate_clarification_app("yes")
    event = main.Event(intent_action="QUEUE", intent_query="keychain lalu by arpit bala")
    app._handle_ambiguous_track(Mock(), event, AmbiguousMatch(event.intent_query, [
        Candidate("spotify:album:release", "Keychain Laalu", "Arpit Bala")]))
    app._execute.assert_not_called()
    app._ask_and_listen.assert_not_called()


def test_generic_track_lookup_only_falls_back_to_album_after_typed_lookup_failure():
    from assistant.resolution import Candidate
    from assistant.spotify import TrackLookupError
    app = _paused_app()
    app.spotify.cached_track.return_value = None
    app.spotify.resolve_track.side_effect = TrackLookupError("song not found")
    app._recover_track_from_audio = Mock(return_value=None)
    candidate = Candidate("spotify:album:release", "Keychain Laalu", "Arpit Bala")
    app.spotify.resolve_album.return_value = candidate
    app._radio_after = Mock()
    app._set_volume = Mock()
    event = main.Event()
    app._execute(main.parse("play keychain lalu by arpit bala at 20% volume"), event)
    app.spotify.resolve_album.assert_called_once_with("keychain lalu by arpit bala")
    app.spotify.play_album.assert_called_once_with(candidate)
    app.spotify.play_candidate.assert_not_called()
    app.spotify.remember_track.assert_not_called()
    app._radio_after.assert_not_called()
    app._set_volume.assert_called_once()
    assert event.outcome == "ok" and event.spotify_action == "PLAY_ALBUM"


@pytest.mark.parametrize("error", [LookupError("other failure"), RuntimeError("device failure")])
def test_playback_failures_do_not_start_an_album_lookup(error):
    from assistant.resolution import Candidate
    app = _paused_app()
    app.spotify.cached_track.return_value = None
    app.spotify.resolve_track.return_value = Candidate("spotify:track:song", "Song", "Artist")
    app.spotify.play_candidate.side_effect = error
    if isinstance(error, LookupError):
        app._execute(main.parse("play song by artist"), main.Event())
    else:
        with pytest.raises(RuntimeError):
            app._execute(main.parse("play song by artist"), main.Event())
    app.spotify.resolve_album.assert_not_called()


def test_ambiguous_track_does_not_probe_album():
    from assistant.resolution import AmbiguousMatch, Candidate
    app = _paused_app()
    app.spotify.cached_track.return_value = None
    original = main.parse("just play tarot by arpit bala")
    error = AmbiguousMatch(original.query, [Candidate("spotify:track:taaron", "Taaron Se", "Arpit Bala")])
    app.spotify.resolve_track.side_effect = error
    approximate = Candidate("spotify:track:taaron", "Taaron Se", "Arpit Bala", {"resolution_approximate": True})
    app.spotify.resolve_best_available.return_value = approximate
    app._ask_and_listen = Mock()
    event = main.Event()
    app._execute(original, event)
    assert event.outcome == "ok"
    app.spotify.resolve_best_available.assert_called_once_with(original.query, "track", tuple(error.candidates))
    app.spotify.play_candidate.assert_called_once_with(approximate)
    app.spotify.remember_track.assert_not_called()
    app._ask_and_listen.assert_not_called()
    app.spotify.resolve_album.assert_not_called()


def test_generic_missing_song_checks_strict_album_before_best_available_song():
    from assistant.resolution import Candidate
    from assistant.spotify import TrackLookupError
    app = _paused_app()
    app.spotify.cached_track.return_value = None
    app.spotify.resolve_track.side_effect = TrackLookupError("missing song")
    app._recover_track_from_audio = Mock(return_value=None)
    candidate = Candidate("spotify:track:best", "Best Song", "Artist", {"resolution_approximate": True})
    order = []
    def missing_album(_):
        order.append("album")
        raise LookupError("missing album")
    app.spotify.resolve_album.side_effect = missing_album
    app.spotify.resolve_best_available.side_effect = lambda *_: order.append("best song") or candidate
    app._radio_after = Mock(return_value=[])
    event = main.Event()
    app._execute(main.parse("just play forgotten by artist"), event)
    assert order == ["album", "best song"]
    app.spotify.play_candidate.assert_called_once_with(candidate)
    app._radio_after.assert_called_once_with(candidate, False)
    app.spotify.remember_track.assert_not_called()
    assert event.resolved_label == "Best Song by Artist" and event.outcome == "ok"


def test_explicit_album_best_available_remains_album_context_without_track_search():
    from assistant.resolution import Candidate
    app = _paused_app()
    app.spotify.resolve_album.side_effect = LookupError("missing album")
    candidate = Candidate("spotify:album:best", "Blue Lights", "Mira Sol", {"resolution_approximate": True})
    app.spotify.resolve_best_available.return_value = candidate
    event = main.Event()
    app._execute(main.parse("play album blue light by mira sol"), event)
    app.spotify.resolve_best_available.assert_called_once_with("blue light by mira sol", "album")
    app.spotify.play_album.assert_called_once_with(candidate)
    app.spotify.resolve_track.assert_not_called()
    app.spotify.play_candidate.assert_not_called()
    app.spotify.remember_track.assert_not_called()
    assert event.spotify_action == "PLAY_ALBUM"


def test_explicit_album_known_ban_reuses_fielded_only_cached_metadata():
    from assistant.spotify import Spotify
    app = _paused_app()
    app.spotify.web_rate_limited = True
    cached = Spotify.__new__(Spotify)
    cached.sp = SimpleNamespace(search=Mock(side_effect=AssertionError("No HTTP during known ban")))
    row = {"uri": "spotify:album:release", "name": "Blue Lights",
           "artists": [{"name": "Mira Sol"}]}
    # This is the real key produced by a successful artist-qualified search.
    cached._music_search_cache = {("album", "album:blue lights artist:mira sol"): [row]}
    app.spotify.resolve_album = Mock(wraps=cached.resolve_album)
    app._set_volume = Mock()
    sequence = []
    app.spotify.play_album.side_effect = lambda _: sequence.append("album")
    app._set_volume.side_effect = lambda _: sequence.append("volume")
    event = main.Event()
    app._execute(main.parse("play album blue lights by mira sol at 20% volume"), event)
    app.spotify.resolve_album.assert_called_once_with("blue lights by mira sol", offline_only=True)
    app.spotify.resolve_best_available.assert_not_called()
    assert app.spotify.play_album.call_args.args[0].uri == "spotify:album:release"
    assert sequence == ["album", "volume"]
    cached.sp.search.assert_not_called()
    app.spotify.resolve_track.assert_not_called()
    app.spotify.play_candidate.assert_not_called()
    app.spotify.remember_track.assert_not_called()
    assert event.spotify_action == "PLAY_ALBUM" and event.outcome == "ok"


@pytest.mark.parametrize("ambiguous", [False, True])
def test_explicit_album_known_ban_fallback_stays_offline_and_album_only(ambiguous):
    from assistant.resolution import AmbiguousMatch, Candidate
    app = _paused_app()
    app.spotify.web_rate_limited = True
    query = "blue lights by mira sol"
    candidate = Candidate("spotify:album:release", "Blue Lights", "Mira Sol", {"resolution_approximate": True})
    error = AmbiguousMatch(query, [candidate]) if ambiguous else LookupError("no strict saved match")
    app.spotify.resolve_album.side_effect = error
    app.spotify.resolve_best_available.return_value = candidate
    app._execute(main.parse("play album " + query), main.Event())
    if ambiguous:
        app.spotify.resolve_best_available.assert_called_once_with(query, "album", (candidate,), offline_only=True)
    else:
        app.spotify.resolve_best_available.assert_called_once_with(query, "album", offline_only=True)
    app.spotify.play_album.assert_called_once_with(candidate)
    app.spotify.resolve_track.assert_not_called()
    app.spotify.play_candidate.assert_not_called()


@pytest.mark.parametrize("error", [main.SpotifyRateLimited("player ban"),
                                    main.SpotifyAuthError("expired authorization"), RuntimeError("transport offline")])
def test_explicit_album_known_ban_does_not_turn_transport_failures_into_lookup_fallback(error):
    app = _paused_app()
    app.spotify.web_rate_limited = True
    app.spotify.resolve_album.side_effect = error
    with pytest.raises(type(error)):
        app._execute(main.parse("play album blue lights by mira sol at 20% volume"), main.Event())
    app.spotify.resolve_best_available.assert_not_called()
    app.spotify.play_album.assert_not_called()
    app.spotify.play_candidate.assert_not_called()


def test_known_search_ban_checks_saved_strict_album_before_saved_approximate_song():
    from assistant.resolution import Candidate
    app = _paused_app()
    app.spotify.web_rate_limited = True
    app.spotify.cached_track.return_value = None
    app.spotify.resolve_local.return_value = None
    candidate = Candidate("spotify:album:release", "Keychain Laalu", "Arpit Bala")
    app.spotify.resolve_album.return_value = candidate
    event = main.Event()
    app._execute(main.parse("play keychain lalu by arpit bala"), event)
    app.spotify.resolve_album.assert_called_once_with("keychain lalu by arpit bala", offline_only=True)
    app.spotify.resolve_track.assert_not_called()
    app.spotify.resolve_best_available.assert_not_called()
    app.spotify.play_album.assert_called_once_with(candidate)


def test_known_search_ban_recovers_saved_song_without_calling_strict_web_search():
    from assistant.resolution import Candidate
    app = _paused_app()
    app.spotify.web_rate_limited = True
    app.spotify.cached_track.return_value = None
    app.spotify.resolve_local.return_value = None
    app.spotify.resolve_album.side_effect = LookupError("no saved album")
    candidate = Candidate("spotify:track:red", "RED", "Seedhe Maut", {"resolution_approximate": True})
    app.spotify.resolve_best_available.return_value = candidate
    app._radio_after = Mock(return_value=[])
    event = main.Event()
    app._execute(main.parse("just play red by sedebh"), event)
    app.spotify.resolve_best_available.assert_called_once_with("red by sedebh", "track", offline_only=True)
    app.spotify.resolve_track.assert_not_called()
    app.spotify.play_candidate.assert_called_once_with(candidate)
    app.spotify.remember_track.assert_not_called()
    assert event.resolved_label == candidate.label


def test_known_search_ban_without_grounded_saved_data_remains_temporarily_unavailable():
    from assistant.spotify import SpotifyRateLimited
    app = _paused_app()
    app.spotify.web_rate_limited = True
    app.spotify.cached_track.return_value = None
    app.spotify.resolve_local.return_value = None
    app.spotify.resolve_album.side_effect = LookupError("no saved album")
    app.spotify.resolve_best_available.side_effect = SpotifyRateLimited("Spotify search is temporarily unavailable")
    with pytest.raises(SpotifyRateLimited, match="temporarily unavailable"):
        app._execute(main.parse("play zxqv nonsensical"), main.Event())
    app.spotify.resolve_track.assert_not_called()
    app.spotify.play_candidate.assert_not_called()
    app.spotify.play_album.assert_not_called()


@pytest.mark.parametrize("error", [main.SpotifyRateLimited("new search ban"),
    main.SpotifyAuthError("expired authorization"), RuntimeError("transport offline")])
def test_new_search_rate_auth_or_transport_error_never_triggers_best_available(error):
    app = _paused_app()
    app.spotify.cached_track.return_value = None
    app.spotify.resolve_track.side_effect = error
    with pytest.raises(type(error)):
        app._execute(main.parse("play forgotten by artist"), main.Event())
    app.spotify.resolve_best_available.assert_not_called()
    app.spotify.resolve_album.assert_not_called()
    app.spotify.play_candidate.assert_not_called()


def test_explicit_album_missing_does_not_fall_through_to_song_or_change_volume():
    app = _paused_app()
    app.spotify.resolve_album.side_effect = LookupError("not found")
    app.spotify.resolve_best_available.side_effect = LookupError("no eligible album")
    app._set_volume = Mock()
    event = main.Event()
    app._execute(main.parse("play album keychain lalu by arpit bala at 20% volume"), event)
    assert event.outcome == "not_found"
    app.spotify.resolve_track.assert_not_called()
    app.spotify.play_candidate.assert_not_called()
    app.spotify.play_album.assert_not_called()
    app._set_volume.assert_not_called()


def test_confirmed_album_bypasses_search_and_plays_context_before_volume():
    from assistant.resolution import Candidate
    app = _paused_app()
    candidate = Candidate("spotify:album:release", "Keychain Laalu", "Arpit Bala")
    sequence = []
    app.spotify.play_album.side_effect = lambda _: sequence.append("album")
    app._set_volume = Mock(side_effect=lambda _: sequence.append("volume"))
    event = main.Event()
    app._execute(main.Intent(main.Action.PLAY_ALBUM, query="keychain lalu", volume_after_play=20),
                 event, confirmed_candidate=candidate)
    assert sequence == ["album", "volume"]
    app.spotify.resolve_album.assert_not_called()
    app.spotify.remember_track.assert_not_called()


def test_confirmed_song_bypasses_lookup_and_keeps_radio_opt_out():
    from assistant.resolution import Candidate
    app = _paused_app()
    candidate = Candidate("spotify:track:taaron", "Taaron Se", "Arpit Bala")
    app._radio_after = Mock(return_value=[])
    event = main.Event()
    original = main.Intent(main.Action.PLAY_TRACK, query="tarot by arpit bala", radio=False)
    app._execute(original, event, confirmed_candidate=candidate)
    app._radio_after.assert_called_once_with(candidate, False)
    app.spotify.resolve_track.assert_not_called()
    app.spotify.cached_track.assert_not_called()
    app.spotify.remember_track.assert_not_called()
    app.spotify.play_candidate.assert_called_once_with(candidate)


@pytest.mark.parametrize("spoken", ["play the song keychain lalu by arpit bala",
    "play track named keychain lalu by arpit bala", "change the track to keychain lalu by arpit bala"])
def test_explicit_track_failure_never_falls_back_to_album(spoken):
    from assistant.spotify import TrackLookupError
    app = _paused_app()
    app.spotify.cached_track.return_value = None
    app.spotify.resolve_track.side_effect = TrackLookupError("song not found")
    app.spotify.resolve_best_available.side_effect = LookupError("no eligible track")
    app._recover_track_from_audio = Mock(return_value=None)
    event = main.Event()
    app._execute(main.parse(spoken), event)
    assert event.outcome == "not_found"
    app.spotify.resolve_album.assert_not_called()
    app.spotify.play_album.assert_not_called()


def test_brain_song_with_explicit_source_type_cannot_fallback_album():
    from assistant.brain import Brain
    from assistant.spotify import TrackLookupError
    app = _paused_app()
    app.spotify.cached_track.return_value = None
    app.spotify.resolve_track.side_effect = TrackLookupError("song not found")
    app.spotify.resolve_best_available.side_effect = LookupError("no eligible track")
    app._recover_track_from_audio = Mock(return_value=None)
    intent = Brain("unused").to_intent({"tool": "play_song", "arguments": {
        "title": "Keychain Lalu", "artist": "Arpit Bala"}},
        "I'd like to hear the song Keychain Lalu by Arpit Bala")
    event = main.Event()
    app._execute(intent, event)
    assert event.outcome == "not_found"
    app.spotify.resolve_album.assert_not_called()


@pytest.mark.parametrize("original_action,uri,answer", [
    (main.Action.PLAY_TRACK, "spotify:track:x", "play the album X by Artist"),
    (main.Action.QUEUE, "spotify:track:x", "play the album X by Artist"),
    (main.Action.PLAY_ALBUM, "spotify:album:x", "play the song X by Artist"),
    (main.Action.PLAY_ALBUM, "spotify:album:x", "play track named X by Artist"),
])
def test_confirmation_cannot_change_bound_media_kind(original_action, uri, answer):
    from assistant.resolution import AmbiguousMatch, Candidate
    app = _candidate_clarification_app(answer)
    original = main.Intent(original_action, query="x by artist")
    exc = AmbiguousMatch(original.query, [Candidate(uri, "X", "Artist")])
    exc.intent = original
    event = main.Event(intent_action=original_action.value, intent_query=original.query)
    app._handle_ambiguous_track(Mock(), event, exc)
    app._execute.assert_not_called()
    assert event.outcome == "ambiguous"


@pytest.mark.parametrize("action,uri,title,answer", [
    (main.Action.PLAY_TRACK, "spotify:track:x", "The Photo Album", "play The Photo Album"),
    (main.Action.PLAY_TRACK, "spotify:track:x", "Album of the Year", "play Album of the Year"),
    (main.Action.PLAY_ALBUM, "spotify:album:x", "Song 2", "play Song 2"),
    (main.Action.PLAY_ALBUM, "spotify:album:x", "X", "play X"),
])
def test_generic_confirmation_preserves_bound_kind_and_complete_entity_titles(action, uri, title, answer):
    from assistant.resolution import AmbiguousMatch, Candidate
    app = _candidate_clarification_app(answer)
    original = main.Intent(action, query="heard by artist")
    candidate = Candidate(uri, title, "Artist")
    exc = AmbiguousMatch(original.query, [candidate])
    exc.intent = original
    event = main.Event(intent_action=action.value, intent_query=original.query)
    app._handle_ambiguous_track(Mock(), event, exc)
    app._execute.assert_called_once()
    assert app._execute.call_args.kwargs["confirmed_candidate"] is candidate


@pytest.mark.parametrize("elapsed,accepted", [(29.9, True), (30.0, False), (31.0, False)])
def test_candidate_confirmation_deadline_includes_prompt_capture_and_stt(elapsed, accepted, monkeypatch):
    from assistant.resolution import AmbiguousMatch, Candidate
    app = _candidate_clarification_app("yes")
    clock = [100.0]
    monkeypatch.setattr(main.time, "monotonic", lambda: clock[0])
    def stalled_capture(*_):
        clock[0] += elapsed
        return "yes", {"audio": np.zeros(512, dtype=np.int16)}, int(elapsed * 1000)
    app._ask_and_listen.side_effect = stalled_capture
    event = main.Event(intent_action="PLAY_TRACK", intent_query="tarot by arpit bala")
    app._handle_ambiguous_track(Mock(), event, AmbiguousMatch(event.intent_query, [
        Candidate("spotify:track:taaron", "Taaron Se", "Arpit Bala")]))
    if accepted:
        assert app._execute.call_args.kwargs["clarification_deadline"] == 130.0
    else:
        app._execute.assert_not_called()
        assert event.feedback_text == "That choice expired."


def test_missing_artist_late_reply_cannot_execute(monkeypatch):
    from assistant.resolution import AmbiguousMatch, Candidate
    app = _candidate_clarification_app("Artist")
    clock = [0.0]
    monkeypatch.setattr(main.time, "monotonic", lambda: clock[0])
    def stalled_capture(*_):
        clock[0] = 31.0
        return "Artist", {"audio": np.zeros(512, dtype=np.int16)}, 31000
    app._ask_and_listen.side_effect = stalled_capture
    event = main.Event(intent_action="PLAY_TRACK", intent_query="song")
    app._handle_ambiguous_track(Mock(), event, AmbiguousMatch("song", [
        Candidate("spotify:track:x", "Song", "Artist")]))
    app._execute.assert_not_called()
    assert event.feedback_text == "That choice expired."


def test_clarification_expiry_during_client_wait_stops_playback(monkeypatch):
    from assistant.resolution import Candidate
    app = _paused_app()
    clock = [0.0]
    monkeypatch.setattr(main.time, "monotonic", lambda: clock[0])
    app.spotify.ensure_client_running.return_value = True
    app.spotify.wait_for_client.side_effect = lambda: clock.__setitem__(0, 31.0)
    candidate = Candidate("spotify:track:x", "X", "Artist")
    with pytest.raises(main.ClarificationExpired):
        app._execute(main.Intent(main.Action.PLAY_TRACK, query="heard by artist"), main.Event(),
                     confirmed_candidate=candidate, clarification_deadline=30.0)
    app.spotify.play_candidate.assert_not_called()


def test_already_expired_clarification_cannot_launch_or_queue(monkeypatch):
    from assistant.resolution import Candidate
    app = _paused_app()
    monkeypatch.setattr(main.time, "monotonic", lambda: 31.0)
    with pytest.raises(main.ClarificationExpired):
        app._execute(main.Intent(main.Action.QUEUE, query="heard by artist"), main.Event(),
                     confirmed_candidate=Candidate("spotify:track:x", "X", "Artist"),
                     clarification_deadline=30.0)
    app.spotify.ensure_client_running.assert_not_called()
    app.spotify.queue.assert_not_called()


def test_warmup_blocks_only_semantic_global_model_routing():
    app = main.Assistant.__new__(main.Assistant)
    app.brain = SimpleNamespace(warming=True)
    app.global_assistant = Mock(session=SimpleNamespace(active=False))
    app.global_assistant.handle.return_value = None
    app._global_say = Mock()
    event = main.Event()
    assert app._handle_global("I'd like a coding assistant to inspect my project", Mock(), event) is True
    app.global_assistant.handle.assert_not_called()
    assert event.outcome == "global_warming"
    assert "still warming" in app._global_say.call_args.args[1]
    assert app._handle_global("Codex status", Mock(), main.Event()) is False
    app.global_assistant.handle.assert_called_once_with("Codex status")


def test_global_router_wrapper_cannot_queue_model_during_warmup():
    brain = SimpleNamespace(warming=True)
    router = Mock()
    wrapper = main._WarmAwareRouter(router, brain)
    with pytest.raises(RuntimeError, match="warming"):
        wrapper.route("inspect this project")
    router.route.assert_not_called()
    brain.warming = False
    assert wrapper.route("inspect this project") is router.route.return_value
    router.route.assert_called_once_with("inspect this project")


def test_global_router_runtime_gate_blocks_failed_warmup_until_retry_admission():
    from assistant.model_runtime import LocalModelRuntime

    now = [0.0]
    runtime = LocalModelRuntime("local-test-model", cooldown_seconds=30,
                                clock=lambda: now[0])
    runtime.mark_warming()
    runtime.mark_failed()
    brain = SimpleNamespace(warming=False)
    router = Mock()
    wrapper = main._WarmAwareRouter(router, brain, runtime=runtime)

    with pytest.raises(RuntimeError, match="runtime"):
        wrapper.route("inspect this project")
    router.route.assert_not_called()
    app = main.Assistant.__new__(main.Assistant)
    app.brain = brain
    app._model_runtime = runtime
    assert app._conversation_ready() is False

    now[0] = 31.0
    assert runtime.state == "degraded"  # admission is not a ready-state claim
    assert runtime.can_attempt() is True
    assert app._conversation_ready() is True
    assert wrapper.route("inspect this project") is router.route.return_value
    router.route.assert_called_once_with("inspect this project")


@pytest.mark.parametrize(("backend", "conversation_enabled", "expected_num_ctx"), [
    ("brain", False, 4096),
    ("brain", True, 8192),
    ("rules", True, 8192),
    ("rules", False, None),
])
def test_assistant_injects_one_runtime_into_brain_and_global_router(
        monkeypatch, backend, conversation_enabled, expected_num_ctx):
    values = {
        "intent.backend": backend, "intent.model": "local-test-model",
        "intent.timeout_seconds": 4.0, "conversation.enabled": conversation_enabled,
        "listener.mode": "stt", "ducking.duck_trigger": "none",
        "stt.bias_from_spotify": False, "codex.enabled": True,
    }
    cfg = SimpleNamespace(get=lambda key, default=None: values.get(key, default))

    class FakeFeedback:
        def prime(self):
            pass

    class FakeSpotify:
        def check_auth(self):
            pass

        def set_known_artists(self, artists):
            self.artists = artists

    class FakeTranscriber:
        def set_wake_mode(self):
            pass

        def transcribe(self, _pcm, **_kwargs):
            return "", 0

    class FakeGlobalAssistant:
        def __init__(self, _cfg, _root, *, router):
            self.router = router

    from assistant import global_assistant

    monkeypatch.setattr(main, "load", lambda: cfg)
    monkeypatch.setattr(main, "Telemetry", lambda _cfg: object())
    monkeypatch.setattr(main, "Feedback", lambda _cfg: FakeFeedback())
    monkeypatch.setattr(main.ducking, "repair_orphaned_duck", lambda: None)
    monkeypatch.setattr(main, "Spotify", FakeSpotify)
    monkeypatch.setattr(main, "SileroVAD", lambda _rate: object())
    monkeypatch.setattr(main, "Transcriber", lambda *_args: FakeTranscriber())
    monkeypatch.setattr(main, "SttListener", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(main, "WakeTrace", lambda *_args: object())
    monkeypatch.setattr(main, "Endpointer", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(main.Assistant, "_load_vocabulary",
                        lambda _self: {"artists": [], "hints": []})
    monkeypatch.setattr(main.threading.Thread, "start", lambda _thread: None)
    monkeypatch.setattr(global_assistant, "GlobalAssistant", FakeGlobalAssistant)

    assistant = main.Assistant()
    runtime = assistant._model_runtime
    if expected_num_ctx is None:
        assert runtime is None
        assert assistant.brain is None
        assert assistant.global_assistant.router is None
    else:
        assert runtime.model == "local-test-model"
        assert runtime.num_ctx == expected_num_ctx
        assert runtime.keep_alive == -1
        assert assistant.brain._runtime is runtime
        assert assistant.global_assistant.router.runtime is runtime
        assert assistant.global_assistant.router.router.runtime is runtime
        assert assistant.global_assistant.router.router.model == runtime.model


@pytest.mark.parametrize("reply", ["cancel", "start again"])
def test_warmup_allows_valid_held_request_local_cancellation(reply, tmp_path):
    from assistant.global_assistant import _UnsentRequest
    from test_global_assistant import make_assistant
    controller = make_assistant(tmp_path)
    now = controller.session.clock()
    controller._unsent_request = _UnsentRequest("source", "prompt", controller._selection(),
        controller._request_generation, now + 30, now + 60)
    router = Mock(side_effect=AssertionError("No model call for held cancellation"))
    controller._router = SimpleNamespace(route=router)
    app = main.Assistant.__new__(main.Assistant)
    app.brain = SimpleNamespace(warming=True)
    app.global_assistant = controller
    app._global_say = Mock(return_value=True)
    event = main.Event()
    assert app._handle_global(reply, Mock(), event) is True
    assert controller._unsent_request is None
    assert "Held request cancelled" in app._global_say.call_args.args[1]
    assert event.outcome == "global_handled"
    router.assert_not_called()


@pytest.mark.parametrize("reply", ["first", "Build Notes", "cancel"])
def test_warmup_allows_bound_named_choice_without_model_routing(reply, tmp_path):
    from assistant.global_assistant import GlobalReply, _PendingChoice
    from assistant.global_commands import GlobalCommand
    from test_global_assistant import make_assistant
    controller = make_assistant(tmp_path)
    controller._pending_choice = _PendingChoice(GlobalCommand("ask", text="inspect the logs"),
        ({"id": "thr_build", "name": "Build Notes"},), controller._selection(), controller.session.clock() + 30)
    controller._stage_resolved = Mock(return_value=GlobalReply("Choice held. Nothing sent."))
    router = Mock(side_effect=AssertionError("No model call for bound choice"))
    controller._router = SimpleNamespace(route=router)
    app = main.Assistant.__new__(main.Assistant)
    app.brain = SimpleNamespace(warming=True)
    app.global_assistant = controller
    app._global_say = Mock(return_value=True)
    event = main.Event()
    assert app._handle_global(reply, Mock(), event) is True
    assert controller._pending_choice is None
    if reply != "cancel":
        assert controller.session.target_id == "thr_build"
        controller._stage_resolved.assert_called_once()
    else:
        controller._stage_resolved.assert_not_called()
    assert event.outcome == "global_handled"
    router.assert_not_called()


def test_llm_failure_does_not_fall_back_to_guessed_action():
    app = main.Assistant.__new__(main.Assistant)
    app.cfg = Config({"intent": {"backend": "local_llm"}})
    app._intent_parser = Mock()
    app._intent_parser.parse.side_effect = TimeoutError("model busy")
    event = main.Event()
    with pytest.raises(TimeoutError): app._parse_intent("play demons", event)
    assert event.intent_latency_ms is not None


def test_exact_pause_does_not_wait_for_llm():
    app = main.Assistant.__new__(main.Assistant)
    app.cfg = Config({"intent": {"backend": "local_llm"}})
    app._intent_parser = Mock()
    assert app._parse_intent("Pause.", main.Event()).action is main.Action.PAUSE
    app._intent_parser.parse.assert_not_called()


def test_paused_mic_only_opens_when_music_is_paused(monkeypatch):
    app = main.Assistant.__new__(main.Assistant)
    app.cfg = Config({"audio": {"paused_input_device": "Headset (SoundDrum P)",
                                "paused_capture_rate": 16000}})
    app.mode = "stt"
    app._music_playing = True
    app._paused_mic = None
    app._paused_listener = None
    app._paused_mic_retry_at = 0.0
    app.listener_stt = Mock()
    secondary = Mock()
    secondary.__enter__ = Mock(return_value=secondary)
    secondary.__exit__ = Mock()
    make_mic = Mock(return_value=secondary)
    monkeypatch.setattr(main, "Microphone", make_mic)
    monkeypatch.setattr(main, "SileroVAD", Mock())
    monkeypatch.setattr(main, "SttListener", Mock())

    app._sync_paused_mic()
    make_mic.assert_not_called()
    app._music_playing = False
    app._sync_paused_mic()
    make_mic.assert_called_once()
    assert app._paused_mic is secondary
    app._music_playing = True
    app._sync_paused_mic()
    secondary.__exit__.assert_called_once()
    assert app._paused_mic is None


def test_failed_paused_mic_open_keeps_laptop_path(monkeypatch):
    app = main.Assistant.__new__(main.Assistant)
    app.cfg = Config({"audio": {"paused_input_device": "Headset (SoundDrum P)"}})
    app.mode = "stt"
    app._music_playing = False
    app._paused_mic = None
    app._paused_listener = None
    app._paused_mic_retry_at = 0.0
    monkeypatch.setattr(main, "Microphone", Mock(side_effect=RuntimeError("disconnected")))
    app._sync_paused_mic()
    assert app._paused_mic is None
    assert app._paused_mic_retry_at > 0


def test_paused_mic_startup_gap_does_not_block_or_close_it():
    app = main.Assistant.__new__(main.Assistant)
    secondary = MagicMock()
    secondary.read.side_effect = TimeoutError("Bluetooth profile starting")
    app._paused_mic = secondary
    app._paused_listener = Mock()
    app._paused_mic_last_frame_at = time.monotonic()

    assert app._read_paused_mic_frame() is None
    secondary.read.assert_called_once_with(timeout_s=0.0)
    assert app._paused_mic is secondary
    secondary.__exit__.assert_not_called()


def test_paused_mic_sustained_stall_retries_without_disabling_laptop():
    app = main.Assistant.__new__(main.Assistant)
    secondary = MagicMock()
    secondary.read.side_effect = TimeoutError("Bluetooth stream stalled")
    app._paused_mic = secondary
    app._paused_listener = Mock()
    app._paused_mic_last_frame_at = time.monotonic() - main.PAUSED_MIC_STALL_SECONDS - 1

    assert app._read_paused_mic_frame() is None
    assert app._paused_mic is None
    assert app._paused_mic_retry_at > time.monotonic()
    secondary.__exit__.assert_called_once()


def test_open_paused_mic_does_not_starve_laptop_listener(monkeypatch):
    app = main.Assistant.__new__(main.Assistant)
    app.cfg = Config({})
    app.mode = "stt"
    app._stop = threading.Event()
    app._trigger = threading.Event()
    app._start_hotkeys = Mock()
    app._sync_paused_mic = Mock()
    app._read_paused_mic_frame = Mock(return_value=None)
    app._close_paused_mic = Mock()
    app._paused_mic = Mock()
    app._paused_listener = Mock()
    app._early_lease = None
    app.wake = None
    frame = np.zeros(512, dtype=np.int16)
    laptop = MagicMock()
    laptop.__enter__.return_value = laptop
    laptop.read.return_value = frame
    monkeypatch.setattr(main, "Microphone", Mock(return_value=laptop))
    app.listener = Mock(phrases=["alexa"])
    app.listener.feed.side_effect = lambda audio, vad=None: (app._stop.set(), None)[1]

    app.run()

    app._read_paused_mic_frame.assert_called_once()
    app.listener.feed.assert_called_once()
    assert app.listener.feed.call_args.args[0] is frame


def test_play_command_closes_sounddrum_before_starting_audio():
    app = main.Assistant.__new__(main.Assistant)
    secondary = MagicMock()
    app._paused_mic = secondary
    app._paused_listener = Mock()
    app._music_playing = False
    app.spotify = Mock()
    app.feedback = Mock()
    app.spotify.play_resume.side_effect = lambda: assert_closed()

    def assert_closed():
        assert app._paused_mic is None
        assert app._music_playing is True

    app._execute(main.Intent(main.Action.RESUME), main.Event())
    secondary.__exit__.assert_called_once()
    app.spotify.play_resume.assert_called_once()


@pytest.mark.parametrize("failure_stage", ["read", "listener"])
def test_loop_failure_releases_early_duck(monkeypatch, failure_stage,
                                          _never_hold_native_power_requests):
    mic = Mock()
    mic.__enter__ = Mock(return_value=mic)
    mic.__exit__ = Mock(return_value=False)
    mic.read.side_effect = TimeoutError("disconnected") if failure_stage == "read" else [np.zeros(512, dtype=np.int16)]
    monkeypatch.setattr(main, "Microphone", lambda **kwargs: mic)
    app = main.Assistant.__new__(main.Assistant)
    app.cfg = Config({})
    app._stop = threading.Event()
    app._trigger = threading.Event()
    app.mode = "stt"
    app._start_hotkeys = Mock()
    app._early_lease = Mock()
    app._early_lease.expired.return_value = False
    app.wake = Mock()
    app.wake.process.return_value = "loose"
    app.listener = SimpleNamespace(phrases=["memoir"], flush=Mock(), feed=Mock(side_effect=TimeoutError("decode failed")))
    with pytest.raises(TimeoutError):
        app.run()
    assert app._early_lease.release.call_count == (4 if failure_stage == "read" else 1)
    if failure_stage == "listener":
        app.wake.process.assert_called_once()
        app._early_lease.acquire.assert_not_called()
    mic.__exit__.assert_called_once()
    assert len(_never_hold_native_power_requests) == 1
    assert _never_hold_native_power_requests[0].closed


@pytest.mark.parametrize("score,acoustic_handles", [(0.86, True), (0.70, False)])
def test_stt_mode_uses_strong_acoustic_wake_without_losing_text_fallback(
        monkeypatch, score, acoustic_handles):
    app = main.Assistant.__new__(main.Assistant)
    app.cfg = Config({"wake": {"acoustic_activation_threshold": 0.75}})
    app.mode = "stt"
    app._stop = threading.Event()
    app._trigger = threading.Event()
    app._start_hotkeys = Mock()
    app._sync_paused_mic = Mock()
    app._close_paused_mic = Mock()
    app._paused_mic = None
    app._early_lease = None
    app.wake = Mock(last_score=score)
    app.wake.process.return_value = main.TIER_BOTH
    app._confirm_acoustic_wake = Mock(return_value=True)
    app.handle = Mock(side_effect=lambda *args: app._stop.set())
    app.listener = Mock(phrases=["alexa"])
    app.listener.feed.side_effect = lambda _, vad=None: (app._stop.set(), None)[1]
    frame = np.zeros(512, dtype=np.int16)
    laptop = MagicMock()
    laptop.__enter__.return_value = laptop
    laptop.read.return_value = frame
    monkeypatch.setattr(main, "Microphone", Mock(return_value=laptop))

    app.run()

    if acoustic_handles:
        app.handle.assert_called_once_with(laptop, "wake", main.TIER_BOTH)
        app.wake.reset.assert_called_once()
        app.listener.feed.assert_not_called()
    else:
        app.handle.assert_not_called()
        app.listener.feed.assert_called_once()
        assert app.listener.feed.call_args.args[0] is frame


def test_acoustic_wake_reports_ambiguous_track_instead_of_generic_error():
    from contextlib import nullcontext
    from assistant.resolution import AmbiguousMatch, Candidate

    app = main.Assistant.__new__(main.Assistant)
    app.cfg = Config({})
    app.wake = Mock(last_score=0.84)
    app.spotify = Mock()
    app.spotify.playback_snapshot.return_value = (True, 100)
    app.spotify.token_age_days.return_value = None
    app.feedback = Mock()
    app.telemetry = Mock()
    app.stt = Mock(model_name="test")
    app.stt.transcribe.side_effect = [("play madeira", 1), ("", 0)]
    app.listener = None
    app.endpointer = Mock()
    audio = np.zeros(512, dtype=np.int16)
    app.endpointer.record.return_value = {
        "audio": audio, "duration_ms": 32, "speech_started": True,
        "speech_start_ms": 0, "speech_end_ms": 32,
    }
    app._acknowledge = Mock(return_value="speech")
    app._duck_music = lambda *_: nullcontext()
    app._execute = Mock(side_effect=AmbiguousMatch("madeira", [
        Candidate("spotify:track:a", "Madeira", "Artist A"),
        Candidate("spotify:track:b", "Madeira", "Artist B"),
    ]))
    mic = Mock()
    mic.preroll.return_value = audio

    app.handle(mic, "wake", main.TIER_BOTH)

    event = app.telemetry.write.call_args.args[0]
    assert event.outcome == "ambiguous"
    assert event.feedback_text == main.AMBIGUOUS_MESSAGE
    assert app.feedback.say.call_args.args[0] == event.feedback_text
    assert app.feedback.say.call_count == 2


@pytest.mark.parametrize("answer", ["Seedemoth", "by Seedemoth", "play madeira by Seedemoth"])
def test_ambiguous_title_asks_for_artist_then_executes_once(answer):
    from contextlib import nullcontext
    from assistant.resolution import AmbiguousMatch, Candidate

    app = main.Assistant.__new__(main.Assistant)
    app.cfg = Config({})
    app.telemetry = Mock()
    app.feedback = Mock()
    app._duck_music = lambda *_: nullcontext()
    audio = np.zeros(512, dtype=np.int16)
    app._ask_and_listen = Mock(return_value=(answer, {"audio": audio}, 17))
    app._execute = Mock(side_effect=lambda intent, event, **_: setattr(event, "outcome", "ok"))
    event = main.Event(intent_action="PLAY_TRACK", intent_query="madeira",
                       intent_volume_after_play=20)
    exc = AmbiguousMatch("madeira", [Candidate("a", "Madeira", "Other")])

    app._handle_ambiguous_track(Mock(), event, exc)

    app._execute.assert_called_once()
    assert app._execute.call_args.args[0].query.casefold() == "madeira by seedemoth"
    assert app._execute.call_args.args[0].volume_after_play == 20
    assert event.outcome == "ok"
    assert event.clarification_transcript == answer
    assert event.clarification_stt_latency_ms == 17
    assert event.error == ""


def test_ambiguous_title_does_not_guess_when_artist_is_unheard():
    from contextlib import nullcontext
    from assistant.resolution import AmbiguousMatch, Candidate

    app = main.Assistant.__new__(main.Assistant)
    app.cfg = Config({})
    app.telemetry = Mock()
    app.feedback = Mock()
    app._duck_music = lambda *_: nullcontext()
    app._ask_and_listen = Mock(return_value=("", {"audio": np.zeros(512, dtype=np.int16)}, 0))
    app._execute = Mock()
    event = main.Event(intent_action="PLAY_TRACK", intent_query="madeira")

    app._handle_ambiguous_track(Mock(), event, AmbiguousMatch("madeira", [
        Candidate("a", "Madeira", "Other")] ))

    app._execute.assert_not_called()
    assert event.outcome == "ambiguous"
    assert event.feedback_text == main.AMBIGUOUS_MESSAGE


@pytest.mark.parametrize('spoken,outcome', [('pause', 'ok'), ('purple elephant', 'unknown_intent')])
def test_bare_trigger_acknowledges_and_retries_unknown_once(monkeypatch, spoken, outcome):
    from contextlib import nullcontext
    from assistant.listener import SttListener
    app = main.Assistant.__new__(main.Assistant)
    app.cfg = Config({})
    app.spotify = Mock()
    app.spotify.token_age_days.return_value = None
    app.stt = Mock(model_name='test')
    app.stt.transcribe.return_value = (spoken, 1)
    app.feedback = Mock()
    app.feedback.say.return_value = main.KIND_SPEECH
    app.listener = SttListener(app.cfg, Mock(), None)
    app._early_lease = None
    app.telemetry = Mock()
    app.endpointer = Mock()
    pcm = np.zeros(512, dtype=np.int16)
    sequence = []
    app.feedback.wake.side_effect = lambda: sequence.append('acknowledge') or 'speech'
    app.endpointer.record.side_effect = lambda mic, **_: sequence.append('record') or {
        'audio': pcm, 'duration_ms': 32, 'speech_started': True}
    monkeypatch.setattr(main.ducking, 'ducked', lambda _: nullcontext((80, 20)))
    mic = Mock()
    utt = SimpleNamespace(command='', matched_phrase='memoir', transcript='memoir', latency_ms=1, audio=pcm)
    app.handle_utterance(mic, utt)
    assert sequence == (['acknowledge', 'record'] if outcome == 'ok'
                        else ['acknowledge', 'record', 'record'])
    assert app.endpointer.record.call_count == (1 if outcome == 'ok' else 2)
    assert app.feedback.stopped.call_count == (1 if outcome == 'ok' else 2)
    assert app.telemetry.write.call_args.args[0].outcome == outcome
    mic.flush.assert_called()
    if outcome == 'ok':
        app.spotify.pause.assert_called_once()
    else:
        assert app.feedback.say.call_count == 2
        assert app.feedback.say.call_args.args[0] == main.SORRY
        app.spotify.pause.assert_not_called()


def _retained_wake_handler_app(transcripts, captured):
    from assistant.listener import SttListener
    app = _paused_app()
    app.cfg = Config({"listener": {"phrases": ["alexa"]}})
    app.listener = SttListener(app.cfg, Mock(), None)
    app.listener_stt = Mock(model_name="small.en")
    app.stt = Mock(model_name="turbo")
    app.stt.transcribe.side_effect = transcripts
    app.telemetry = Mock()
    app.feedback.speaks_on_wake = False
    app.feedback.tick.return_value = "tick"
    app.endpointer = Mock()
    app.endpointer.record.return_value = captured
    app._early_lease = None
    app._duck_on_wake = Mock()
    app._duck_music = lambda *_: contextlib.nullcontext()
    return app


def test_retained_wake_only_audio_command_executes_without_followup_or_redecode():
    audio = np.arange(16000, dtype=np.int16)
    app = _retained_wake_handler_app([("Alexa pause", 18)],
                                     {"audio": np.zeros(512, dtype=np.int16),
                                      "duration_ms": 32, "speech_started": False})
    mic = Mock()
    utt = SimpleNamespace(command="", confirmed_audio_handoff=True, truncated=False,
                          matched_phrase="alexa", transcript="Alexa", latency_ms=2,
                          audio=audio)

    app.handle_utterance(mic, utt)

    app.stt.transcribe.assert_called_once_with(audio)
    app.spotify.pause.assert_called_once()
    app.endpointer.record.assert_not_called()
    app.feedback.tick.assert_not_called()
    app.feedback.wake.assert_not_called()
    assert app.telemetry.write.call_args.args[0].transcript == "Alexa pause"


def test_retained_lyrics_without_strict_wake_does_not_execute():
    audio = np.arange(16000, dtype=np.int16)
    app = _retained_wake_handler_app([("we were discussing alexa today", 18)],
                                     {"audio": np.zeros(512, dtype=np.int16),
                                      "duration_ms": 32, "speech_started": False})
    utt = SimpleNamespace(command="", confirmed_audio_handoff=True, truncated=False,
                          matched_phrase="alexa", transcript="Alexa", latency_ms=2,
                          audio=audio)

    app.handle_utterance(Mock(), utt)

    app.stt.transcribe.assert_called_once_with(audio)
    app.spotify.pause.assert_not_called()
    app.endpointer.record.assert_called_once()
    assert app.telemetry.write.call_args.args[0].outcome == "false_accept"


def test_retained_bare_wake_keeps_normal_followup_capture():
    audio = np.arange(16000, dtype=np.int16)
    followup = np.full(512, 12, dtype=np.int16)
    app = _retained_wake_handler_app([("Alexa", 18), ("pause", 20)],
                                     {"audio": followup, "duration_ms": 32,
                                      "speech_started": True})
    utt = SimpleNamespace(command="", confirmed_audio_handoff=True, truncated=False,
                          matched_phrase="alexa", transcript="Alexa", latency_ms=2,
                          audio=audio)

    app.handle_utterance(Mock(), utt)

    assert app.stt.transcribe.call_count == 2
    assert app.stt.transcribe.call_args_list[0].args[0] is audio
    assert app.stt.transcribe.call_args_list[1].args[0] is followup
    app.endpointer.record.assert_called_once()
    app.spotify.pause.assert_called_once()
    app.feedback.tick.assert_called_once()


def test_truncated_retained_wake_cannot_promote_audio_command():
    audio = np.arange(16000, dtype=np.int16)
    app = _retained_wake_handler_app([], {"audio": np.zeros(512, dtype=np.int16),
                                          "duration_ms": 32, "speech_started": False,
                                          "end_reason": "silence", "truncated": False})
    utt = SimpleNamespace(command="", confirmed_audio_handoff=True, truncated=True,
                          matched_phrase="alexa", transcript="Alexa", latency_ms=2,
                          audio=audio)

    app.handle_utterance(Mock(), utt)

    app.stt.transcribe.assert_not_called()
    app.endpointer.record.assert_called_once()
    app.spotify.pause.assert_not_called()
    event = app.telemetry.write.call_args.args[0]
    assert event.capture_truncated is False
    assert event.capture_end_reason == "silence"


def test_unknown_followup_retries_without_another_wake_word(monkeypatch):
    from contextlib import nullcontext
    from assistant.listener import SttListener
    app = main.Assistant.__new__(main.Assistant)
    app.cfg = Config({})
    app.spotify = Mock()
    app.stt = Mock(model_name='test')
    app.stt.transcribe.side_effect = [('circle', 1), ('play', 1)]
    app.feedback = Mock()
    app.feedback.say.return_value = main.KIND_SPEECH
    app.listener = SttListener(app.cfg, Mock(), None)
    app._early_lease = None
    app.telemetry = Mock()
    app.endpointer = Mock()
    pcm = np.zeros(512, dtype=np.int16)
    app.endpointer.record.return_value = {'audio': pcm, 'duration_ms': 32, 'speech_started': True}
    monkeypatch.setattr(main.ducking, 'ducked', lambda _: nullcontext((80, 20)))
    utt = SimpleNamespace(command='', matched_phrase='alexa', transcript='Alexa',
                          latency_ms=1, audio=pcm)
    app.handle_utterance(Mock(), utt)
    app.spotify.play_resume.assert_called_once()
    assert app.feedback.say.call_args_list[0].args[0] == 'Repeat, please. Ready.'
    event = app.telemetry.write.call_args.args[0]
    assert event.retry_transcript == 'play'
    assert event.outcome == 'ok'
    assert app.endpointer.record.call_count == 2


def test_retry_accepts_repeated_wake_word(monkeypatch):
    from contextlib import nullcontext
    from assistant.listener import SttListener

    app = main.Assistant.__new__(main.Assistant)
    app.cfg = Config({'listener': {'phrases': ['alexa']}})
    app.spotify = Mock()
    app.stt = Mock(model_name='test')
    app.stt.transcribe.side_effect = [('circle', 1), ('Alexa, pause.', 1)]
    app.feedback = Mock()
    app.feedback.say.return_value = main.KIND_SPEECH
    app.listener = SttListener(app.cfg, Mock(), None)
    app._early_lease = None
    app.telemetry = Mock()
    app.endpointer = Mock()
    pcm = np.zeros(512, dtype=np.int16)
    app.endpointer.record.return_value = {
        'audio': pcm, 'duration_ms': 32, 'speech_started': True}
    monkeypatch.setattr(main.ducking, 'ducked', lambda _: nullcontext((80, 20)))
    utt = SimpleNamespace(command='', matched_phrase='alexa', transcript='Alexa',
                          latency_ms=1, audio=pcm)

    app.handle_utterance(Mock(), utt)

    app.spotify.pause.assert_called_once()
    event = app.telemetry.write.call_args.args[0]
    assert event.retry_transcript == 'Alexa, pause.'
    assert event.intent_action == 'PAUSE'
    assert event.outcome == 'ok'


@pytest.mark.parametrize('larger_transcript,expected_action', [
    ('Alexa, pause!', main.Action.PAUSE),
    ('Pause.', main.Action.PAUSE),
    ('Alexa play', main.Action.RESUME),
    ('Elixir pause', main.Action.PAUSE),
])
def test_one_breath_command_uses_larger_model_on_same_audio(larger_transcript, expected_action):
    from assistant.listener import SttListener

    app = main.Assistant.__new__(main.Assistant)
    app.cfg = Config({'listener': {'phrases': ['alexa']}})
    app.spotify = Mock()
    app.spotify.token_age_days.return_value = None
    app.stt = Mock(model_name='faster-whisper/turbo')
    app.stt.transcribe.return_value = (larger_transcript, 500)
    app.listener_stt = Mock(model_name='faster-whisper/small.en')
    app.listener = SttListener(app.cfg, None, None)
    app.feedback = Mock()
    app.telemetry = Mock()
    app._early_lease = None
    audio = np.zeros(16000, dtype=np.int16)
    utt = SimpleNamespace(command='boss', matched_phrase='alexa',
                          transcript='Alexa, boss.', latency_ms=90, audio=audio)

    app.handle_utterance(Mock(), utt)

    app.stt.transcribe.assert_called_once_with(audio)
    event = app.telemetry.write.call_args.args[0]
    assert event.trigger_transcript == 'Alexa, boss.'
    assert event.transcript == larger_transcript
    assert event.trigger_stt_model == 'faster-whisper/small.en'
    assert event.trigger_audio_path == event.audio_path
    assert event.command_stt_model == 'faster-whisper/turbo'
    assert event.intent_action == expected_action.value
    assert event.outcome == 'ok'
    app.feedback.say.assert_not_called()


class _TrackedEarlyLease:
    def __init__(self):
        self.owned = False
        self.acquire_calls = 0
        self.release_calls = 0

    def acquire(self):
        self.acquire_calls += 1
        self.owned = True
        return True

    def release(self):
        self.release_calls += 1
        self.owned = False

    def expired(self):
        return False


def _ducked_handler_app(handler, transcript="pause", speech_started=True):
    app = main.Assistant.__new__(main.Assistant)
    app.cfg = Config({"capture": {"preroll_seconds": 0.1},
                      "ducking": {"duck_to_percent": 10}})
    lease = app._early_lease = _TrackedEarlyLease()
    app._can_duck_local = Mock(return_value=True)
    app._note_early_duck = Mock()
    app.wake = Mock(last_score=0.5)
    app.listener = None if handler == "wake" else Mock()
    app.listener_stt = Mock(model_name="test-gate")
    app.stt = Mock(model_name="test-command")
    pcm = np.zeros(512, dtype=np.int16)

    def transcribe(_audio):
        assert lease.owned
        return transcript, 1

    app.stt.transcribe.side_effect = transcribe
    if app.listener is not None:
        app.listener.locate_trigger.side_effect = lambda text: (
            ("alexa", "pause", False) if "pause" in text.casefold()
            else ("", "", False))
    app.spotify = Mock()
    app.spotify.token_age_days.return_value = None
    app.telemetry = Mock()
    app.telemetry.save_audio.return_value = "sample.wav"
    app._open_timeline = Mock(side_effect=lambda *_args, **_kwargs: _assert_lease_owned(lease))
    app._handle_global = Mock(return_value=False)
    app._execute = Mock(side_effect=lambda _intent, event, **_kwargs:
                        setattr(event, "outcome", "ok"))
    app._acknowledge = Mock(side_effect=lambda *_: _assert_lease_owned(lease) or "beep")
    app.feedback = SimpleNamespace(speaks_on_wake=False, stopped=Mock())
    app.endpointer = Mock()

    def record(*_args, **_kwargs):
        _assert_lease_owned(lease)
        return {"audio": pcm, "duration_ms": 32, "speech_started": speech_started,
                "speech_start_ms": 0, "speech_end_ms": 32}

    app.endpointer.record.side_effect = record

    @contextlib.contextmanager
    def duck_music(_event, to_percent):
        _assert_lease_owned(lease)
        assert to_percent == 10
        yield

    app._duck_music = duck_music
    app._ask_and_listen = Mock(side_effect=lambda *_args, **_kwargs: (
        _assert_lease_owned(lease) or ("", {"audio": pcm}, 0)))
    app._say_feedback = Mock(side_effect=lambda *_args: (
        _assert_lease_owned(lease) or "beep"))
    app._past_repeated_wake = lambda _mic, _event, text, captured, latency: (
        text, captured, latency)
    mic = Mock()
    mic.preroll.return_value = pcm
    utt = SimpleNamespace(command="pause", matched_phrase="alexa",
                          transcript="Alexa, pause", latency_ms=1, audio=pcm)
    return app, lease, mic, utt


def _assert_lease_owned(lease):
    assert lease.owned


@pytest.mark.parametrize("handler", ["wake", "utterance"])
@pytest.mark.parametrize("transcript", ["pause", "purple elephant"])
def test_accepted_handlers_hold_and_reuse_early_lease_through_transcription_and_retry(
        handler, transcript):
    app, lease, mic, utt = _ducked_handler_app(handler, transcript=transcript)
    lease_identity = id(app._early_lease)

    if handler == "wake":
        app.handle(mic, "hotkey")
    else:
        app.handle_utterance(mic, utt)

    assert app.stt.transcribe.call_count == 1
    assert lease.acquire_calls == 1
    assert lease.release_calls == 1
    assert id(app._early_lease) == lease_identity
    assert not lease.owned
    if transcript == "pause":
        app._execute.assert_called_once()
        app._ask_and_listen.assert_not_called()
    else:
        app._ask_and_listen.assert_called_once()
        app._say_feedback.assert_called_once()
        app._execute.assert_not_called()


@pytest.mark.parametrize("handler", ["wake", "utterance"])
def test_accepted_handlers_release_early_lease_after_no_speech(handler):
    app, lease, mic, utt = _ducked_handler_app(
        handler, transcript="pause", speech_started=False)
    if handler == "wake":
        app.handle(mic, "hotkey")
    else:
        utt.command = ""
        app.handle_utterance(mic, utt)

    assert lease.acquire_calls == 1
    assert lease.release_calls == 1
    assert not lease.owned
    assert app.telemetry.write.call_args.args[0].outcome == "false_accept"
    app.stt.transcribe.assert_not_called()


@pytest.mark.parametrize("handler", ["wake", "utterance"])
def test_accepted_handlers_release_early_lease_when_timeline_setup_raises(handler):
    app, lease, mic, utt = _ducked_handler_app(handler)
    app._open_timeline.side_effect = RuntimeError("timeline failed")
    if handler == "wake":
        app.handle(mic, "hotkey")
    else:
        app.handle_utterance(mic, utt)

    app._say_feedback.assert_called_once()
    assert lease.acquire_calls == 1
    assert lease.release_calls == 1
    assert not lease.owned
    assert app.telemetry.write.call_count == 1


def test_rejected_strong_acoustic_wake_never_acquires_early_lease(monkeypatch):
    app, mic = _recovery_loop(monkeypatch)
    lease = app._early_lease = _TrackedEarlyLease()
    app.cfg = Config({"wake": {"acoustic_activation_threshold": 0.75}})
    app._can_duck_local = Mock(return_value=True)
    app.wake.last_score = 0.99
    app.wake.process.return_value = main.TIER_BOTH
    app._confirm_acoustic_wake = Mock(return_value=False)
    app._record_wake_rejection = Mock()
    app._record_capture_diagnostics = Mock()
    app._speak_global_notice_if_quiet = Mock()
    app.handle = Mock()
    frame = np.zeros(512, dtype=np.int16)
    mic.read.return_value = frame
    mic.vad_frame.return_value = frame
    app.listener.feed.side_effect = lambda *_args: (app._stop.set(), None)[1]

    app.run()

    app._confirm_acoustic_wake.assert_called_once_with(mic)
    assert lease.acquire_calls == 0
    assert not lease.owned
    app.handle.assert_not_called()


def test_remote_handler_skips_lease_and_local_or_spotify_volume(monkeypatch):
    app, lease, mic, _utt = _ducked_handler_app("wake", speech_started=False)
    app.cfg = Config({"spotify": {"target": "active"},
                      "ducking": {"duck_to_percent": 10}})
    app._can_duck_local = main.Assistant._can_duck_local.__get__(app, main.Assistant)
    app._duck_music = main.Assistant._duck_music.__get__(app, main.Assistant)
    app._open_timeline = Mock()
    app._acknowledge = Mock(return_value="beep")
    app.endpointer.record = Mock(return_value={
        "audio": np.zeros(512, dtype=np.int16), "duration_ms": 32,
        "speech_started": False, "speech_start_ms": 0, "speech_end_ms": 32,
    })
    monkeypatch.setattr(main.ducking, "spotify_is_playing_local", lambda: False)
    ducked = Mock(side_effect=AssertionError("remote playback must not be ducked locally"))
    monkeypatch.setattr(main.ducking, "ducked", ducked)

    app.handle(mic, "hotkey")

    assert lease.acquire_calls == 0
    assert not lease.owned
    ducked.assert_not_called()
    app.spotify.set_volume.assert_not_called()
    assert app.telemetry.write.call_args.args[0].duck_to is None


def test_fast_gate_extra_word_does_not_turn_bare_alexa_into_retry():
    from contextlib import nullcontext
    from assistant.listener import SttListener

    app = main.Assistant.__new__(main.Assistant)
    app.cfg = Config({'listener': {'phrases': ['alexa']}})
    app.spotify = Mock()
    app.stt = Mock(model_name='faster-whisper/turbo')
    app.stt.transcribe.side_effect = [('Alexa!', 250), ('pause', 400)]
    app.listener_stt = Mock(model_name='faster-whisper/small.en')
    app.listener = SttListener(app.cfg, None, None)
    app.feedback = Mock()
    app.telemetry = Mock()
    app._early_lease = None
    app._duck_music = lambda *_: nullcontext()
    app._acknowledge = Mock(return_value='speech')
    app.endpointer = Mock()
    command_audio = np.zeros(512, dtype=np.int16)
    app.endpointer.record.return_value = {
        'audio': command_audio, 'duration_ms': 32, 'speech_started': True}
    trigger_audio = np.zeros(16000, dtype=np.int16)
    utt = SimpleNamespace(command='up.', matched_phrase='alexa',
                          transcript='Alex up.', latency_ms=90, audio=trigger_audio)

    app.handle_utterance(Mock(), utt)

    app._acknowledge.assert_called_once()
    app.endpointer.record.assert_called_once()
    app.spotify.pause.assert_called_once()
    app.feedback.say.assert_not_called()
    event = app.telemetry.write.call_args.args[0]
    assert event.trigger_audio_path
    assert event.transcript == 'Alexa! | pause'
    assert event.retry_transcript == ''
    assert event.follow_up_audio_path
    assert event.stt_latency_ms == 650
    assert event.outcome == 'ok'


def test_one_breath_does_not_fall_back_to_small_model_command():
    from assistant.listener import SttListener

    app = main.Assistant.__new__(main.Assistant)
    app.cfg = Config({'listener': {'phrases': ['alexa']}})
    app.spotify = Mock()
    app.spotify.token_age_days.return_value = None
    app.stt = Mock(model_name='faster-whisper/turbo')
    app.stt.transcribe.return_value = ('Elixir, cause', 500)
    app.listener_stt = Mock(model_name='faster-whisper/small.en')
    app.listener = SttListener(app.cfg, None, None)
    app.feedback = Mock()
    app.telemetry = Mock()
    app._early_lease = None
    app._retry_unknown = Mock()
    utt = SimpleNamespace(command='play', matched_phrase='alexa',
                          transcript='Alexa play', latency_ms=90,
                          audio=np.zeros(16000, dtype=np.int16))

    app.handle_utterance(Mock(), utt)

    app._retry_unknown.assert_called_once()
    assert app.telemetry.write.call_args.args[0].intent_action == 'UNKNOWN'
    app.spotify.play_resume.assert_not_called()


def test_spotify_device_disappearance_does_not_cancel_followup_capture(monkeypatch):
    from contextlib import contextmanager
    from assistant.spotify import PlaybackVerificationError

    @contextmanager
    def missing_target():
        raise PlaybackVerificationError("No active device")
        yield

    app = main.Assistant.__new__(main.Assistant)
    monkeypatch.setattr(main.ducking, "spotify_is_playing_local", lambda: False)
    app.cfg = Config({"spotify": {"target": "active"}})
    app.spotify = Mock()
    app.spotify.command_target.side_effect = missing_target
    event = main.Event()
    with app._duck_music(event, 20):
        captured = True
    assert captured
    assert event.duck_to is None
    app.spotify.get_volume.assert_not_called()
    app.spotify.set_volume.assert_not_called()


@pytest.mark.parametrize('command,personal', [('play my focus playlist', True), ('play focus playlist', False)])
def test_playlist_scope_reaches_spotify(command, personal):
    from assistant.intent import parse
    from assistant.telemetry import Event
    app = main.Assistant.__new__(main.Assistant)
    app.spotify = Mock()
    app.feedback = Mock()
    app._execute(parse(command), Event())
    app.spotify.play_playlist.assert_called_once()
    args, kwargs = app.spotify.play_playlist.call_args
    assert args == ('focus',) and kwargs['library_only'] is personal
    assert callable(kwargs['before_play'])


def test_restart_reaches_seek_without_skipping_track():
    from assistant.intent import parse
    from assistant.telemetry import Event
    app = main.Assistant.__new__(main.Assistant)
    app.spotify = Mock()
    app.feedback = Mock()
    app._execute(parse('restart the song'), Event())
    app.spotify.restart_track.assert_called_once()
    app.spotify.previous.assert_not_called()


def test_volume_command_opens_closed_spotify_without_touching_spotify_volume(monkeypatch):
    app = main.Assistant.__new__(main.Assistant)
    app.spotify = Mock()
    app.spotify.ensure_client_running.return_value = True
    app.feedback = Mock()
    system_volume = Mock()
    monkeypatch.setattr(main.ducking, "set_command_volume", system_volume)
    app._execute(main.parse("volume 50 percent"), main.Event())
    app.spotify.ensure_client_running.assert_called_once()
    app.spotify.wait_for_client.assert_not_called()
    app.spotify.set_volume.assert_not_called()
    system_volume.assert_called_once_with(0.5)


def test_play_command_waits_for_new_spotify_client_before_playback():
    app = main.Assistant.__new__(main.Assistant)
    app.spotify = Mock()
    app.spotify.ensure_client_running.return_value = True
    app.feedback = Mock()
    app.spotify.wait_for_client.side_effect = lambda: assert_not_played()

    def assert_not_played():
        app.spotify.play_resume.assert_not_called()

    app._execute(main.Intent(main.Action.RESUME), main.Event())
    app.spotify.wait_for_client.assert_called_once()
    app.spotify.play_resume.assert_called_once()


def test_failed_artist_query_gets_two_agreeing_targeted_decodes_before_playback():
    from assistant.listener import SttListener
    from assistant.resolution import Candidate
    app = main.Assistant.__new__(main.Assistant)
    app.spotify = Mock()
    app.spotify.ensure_client_running.return_value = False
    candidate = Candidate("spotify:track:red", "RED", "Seedhe Maut")
    app.spotify.resolve_track.side_effect = [LookupError("not found"), candidate]
    app.spotify.artist_hints_for_exact_title.return_value = ["Seedhe Maut", "Hisab"]
    app.stt = Mock(model_name="faster-whisper/test")
    app.stt.transcribe.side_effect = [
        ("Play red by Seedhe Maut", 300), ("Play RED by Seedhe Maut", 310)]
    app.listener = SttListener(Config({}), None, None)
    app.feedback = Mock()
    event = main.Event()
    app._execute(main.parse("play red by see them all"), event,
                 command_audio=np.ones(16000, dtype=np.int16))
    app.spotify.play_candidate.assert_called_once_with(candidate)
    assert event.intent_query == "red by seedhe maut"
    assert event.recovery_stt_latency_ms == 610
    assert event.outcome == "ok"


def test_targeted_decode_disagreement_never_plays_a_guess():
    from assistant.listener import SttListener
    app = main.Assistant.__new__(main.Assistant)
    app.spotify = Mock()
    app.spotify.ensure_client_running.return_value = False
    app.spotify.resolve_track.side_effect = LookupError("not found")
    app.spotify.artist_hints_for_exact_title.return_value = ["Seedhe Maut"]
    app.stt = Mock(model_name="faster-whisper/test")
    app.stt.transcribe.side_effect = [
        ("Play red by Seedhe Maut", 100), ("Play red by Aerosmith", 100)]
    app.listener = SttListener(Config({}), Mock(), None)
    app.feedback = Mock()
    event = main.Event()
    app._execute(main.parse("play red by see them all"), event,
                 command_audio=np.ones(16000, dtype=np.int16))
    app.spotify.play_candidate.assert_not_called()
    assert event.outcome == "not_found"


def _paused_app():
    app = main.Assistant.__new__(main.Assistant)
    app.cfg = Config({})
    app.spotify = Mock()
    app.spotify.ensure_client_running.return_value = False
    app.feedback = Mock()
    app._paused_mic = MagicMock()
    app._paused_listener = Mock()
    app._music_playing = False
    return app


def test_missing_song_says_sorry_once_without_readback_or_second_capture():
    app = _paused_app()
    app.spotify.resolve_track.side_effect = LookupError("unheard")
    app._ask_and_listen = Mock()
    event = main.Event()

    app._execute(main.parse("play thin cheese by the seat"), event, mic=Mock())

    app._ask_and_listen.assert_not_called()
    app.spotify.play_candidate.assert_not_called()
    app.feedback.say.assert_called_once_with(main.NOT_FOUND_MESSAGE, force=True, blocking=True)
    assert "thin cheese" not in main.NOT_FOUND_MESSAGE.casefold()
    assert event.outcome == "not_found"


def test_failed_lookup_keeps_sounddrum_mic_and_audio_path():
    app = _paused_app()
    secondary = app._paused_mic
    app.spotify.resolve_track.side_effect = LookupError("unheard")
    app._execute(main.parse("play thin cheese by the seat"), main.Event())
    secondary.__exit__.assert_not_called()
    assert app._paused_mic is secondary


def test_resolved_track_closes_sounddrum_mic_just_before_playback():
    app = _paused_app()
    secondary = app._paused_mic
    order = []
    app.spotify.resolve_track.side_effect = lambda q: order.append(("resolve", app._paused_mic)) or "cand"
    app.spotify.play_candidate.side_effect = lambda c: order.append(("play", app._paused_mic))
    app._execute(main.parse("play red by seedhe maut"), main.Event())
    assert order == [("resolve", secondary), ("play", None)]
    secondary.__exit__.assert_called_once()


def test_new_client_launch_does_not_delay_track_lookup():
    app = _paused_app()
    app.spotify.ensure_client_running.return_value = True
    calls = []
    app.spotify.resolve_track.side_effect = lambda q: calls.append("resolve") or "cand"
    app.spotify.wait_for_client.side_effect = lambda: calls.append("wait")
    app.spotify.play_candidate.side_effect = lambda c: calls.append("play")
    app._execute(main.parse("play red by seedhe maut"), main.Event())
    assert calls == ["resolve", "wait", "play"]


def test_play_with_volume_resolves_before_mutating_windows_output(monkeypatch):
    from assistant.intent import parse
    from assistant.telemetry import Event
    app = main.Assistant.__new__(main.Assistant)
    app.spotify = Mock()
    volume_write = Mock()
    monkeypatch.setattr(main.ducking, 'set_command_volume', volume_write)
    app.feedback = Mock()
    intent = parse('play red by sidaimoth at 20% volume')
    app._execute(intent, Event())
    app.spotify.resolve_track.assert_called_once_with('red by sidaimoth')
    app.spotify.play_candidate.assert_called_once_with(app.spotify.resolve_track.return_value)
    volume_write.assert_called_once_with(0.2)
    app.spotify.set_volume.assert_not_called()


def test_unresolved_song_does_not_change_volume(monkeypatch):
    from assistant.intent import parse
    from assistant.telemetry import Event
    app = main.Assistant.__new__(main.Assistant)
    app.spotify = Mock()
    app.spotify.resolve_track.side_effect = LookupError('not found')
    app.feedback = Mock()
    volume_write = Mock()
    monkeypatch.setattr(main.ducking, 'set_command_volume', volume_write)
    event = Event()
    app._execute(parse('play red by sidaimoth at 20% volume'), event)
    assert event.outcome == 'not_found'
    app.spotify.set_volume.assert_not_called()
    volume_write.assert_not_called()


def test_remote_duck_never_changes_spotify_volume(monkeypatch):

    app = main.Assistant.__new__(main.Assistant)
    app.cfg = Config({"spotify": {"target": "active"}})
    app.spotify = Mock()
    monkeypatch.setattr(main.ducking, "spotify_is_playing_local", lambda: False)
    monkeypatch.setattr(main.ducking, "ducked", lambda _: pytest.fail("remote target must not use local duck"))

    event = main.Event()
    with app._duck_music(event, 20):
        pass
    assert event.duck_to is None
    app.spotify.get_volume.assert_not_called()
    app.spotify.set_volume.assert_not_called()


def test_active_laptop_uses_local_duck(monkeypatch):
    from contextlib import nullcontext
    import socket

    app = main.Assistant.__new__(main.Assistant)
    app.cfg = Config({"spotify": {"target": "active"}})
    app.spotify = Mock()
    app.spotify.command_target.return_value = nullcontext("laptop")
    app.spotify.device_name = socket.gethostname()
    local = Mock(return_value=nullcontext((100, 20)))
    monkeypatch.setattr(main.ducking, "spotify_is_playing_local", lambda: True)
    monkeypatch.setattr(main.ducking, "ducked", local)

    event = main.Event()
    with app._duck_music(event, 20):
        assert event.duck_to == 20
    local.assert_called_once_with(20)
    app.spotify.get_volume.assert_not_called()


def test_acoustic_wake_needs_the_wake_word_in_the_transcript():
    from assistant.listener import SttListener
    app = main.Assistant.__new__(main.Assistant)
    app.cfg = Config({})
    app.listener = SttListener(Config({"listener": {"phrases": ["alexa"]}}), None, None)
    app.listener_stt = Mock()
    mic = Mock()
    mic.preroll.return_value = np.zeros(16000, dtype=np.int16)
    app.listener_stt.transcribe.return_value = ("who is fighting", 100)
    assert app._confirm_acoustic_wake(mic) is False
    app.listener_stt.transcribe.return_value = ("Alexa", 100)
    assert app._confirm_acoustic_wake(mic) is True


def test_acoustic_confirmation_preserves_history_when_small_hears_only_wake():
    from assistant.listener import SttListener
    app = main.Assistant.__new__(main.Assistant)
    app.cfg = Config({})
    app.listener = SttListener(Config({"listener": {"phrases": ["alexa"]}}), None, None)
    app.listener_stt = Mock()
    app.listener_stt.transcribe.return_value = ("Alexa", 5)
    app.wake = SimpleNamespace(personal_hit=False, last_score=0.8, last_model="alexa")
    raw = np.arange(1536, dtype=np.int16)
    clean = raw // 2
    class PairedMic:
        def preroll_pair(self, seconds):
            assert seconds == 1.6
            return raw, clean

    mic = PairedMic()
    app._duck_on_wake = Mock()

    assert app._confirm_acoustic_wake(mic)
    phrase, saved_raw, saved_clean = app._confirmed_acoustic_history
    assert phrase == "alexa"
    assert saved_raw is raw and saved_clean is clean
    assert saved_raw.size == 1536  # includes retained command audio, if any


def test_acoustic_confirmation_rejection_does_not_retain_or_seed_history():
    from assistant.listener import SttListener
    app = main.Assistant.__new__(main.Assistant)
    app.cfg = Config({})
    app.listener = SttListener(Config({"listener": {"phrases": ["alexa"]}}), None, None)
    app.listener_stt = Mock()
    app.listener_stt.transcribe.return_value = ("background speech", 5)
    app.wake = SimpleNamespace(personal_hit=False, last_score=0.8, last_model="alexa")
    app._record_near_miss = Mock()
    raw = np.arange(1536, dtype=np.int16)
    mic = Mock()
    mic.preroll.return_value = raw

    assert not app._confirm_acoustic_wake(mic)
    assert app._confirmed_acoustic_history is None
    app._record_near_miss.assert_called_once()


def test_run_seeds_real_listener_prefix_but_feeds_current_frame_once(monkeypatch):
    from assistant.listener import SttListener
    app, mic = _recovery_loop(monkeypatch)
    listener = SttListener(Config({"listener": {"phrases": ["alexa"]}}),
                           lambda _: 1.0, None)
    app.listener = listener
    app.listener_stt = Mock()
    app.listener_stt.transcribe.return_value = ("Alexa", 1)
    app.wake.process.return_value = main.TIER_BOTH
    app.wake.last_score = 0.9
    app.wake.last_model = "alexa"
    app.wake.personal_hit = False
    prefix_and_current = np.arange(1536, dtype=np.int16)
    current = prefix_and_current[-512:]
    mic.frame_length = 512
    monkeypatch.setattr(type(mic), "preroll_pair",
                        Mock(return_value=(prefix_and_current, prefix_and_current // 2)),
                        raising=False)
    mic.vad_frame.side_effect = lambda frame: frame
    reads = iter([current])

    def read_once_then_stop():
        try:
            return next(reads)
        except StopIteration:
            app._stop.set()
            raise TimeoutError("bounded test stop")

    mic.read.side_effect = read_once_then_stop
    seed = Mock(wraps=listener.confirm_wake)
    feed = Mock(wraps=listener.feed)
    listener.confirm_wake = seed
    listener.feed = feed

    app.run()

    assert seed.call_count == 1
    assert seed.call_args.args[0] == "alexa"
    np.testing.assert_array_equal(seed.call_args.args[1], prefix_and_current[:-512])
    np.testing.assert_array_equal(seed.call_args.args[2], (prefix_and_current // 2)[:-512])
    feed.assert_called_once()
    assert feed.call_args.args[0] is current
    assert listener._speech[-1] is current


def test_run_skips_redundant_acoustic_decode_when_listener_already_woke(monkeypatch):
    from assistant.listener import SttListener
    app, mic = _recovery_loop(monkeypatch)
    listener = SttListener(Config({"listener": {"phrases": ["alexa"]}}),
                           lambda _: 1.0, None)
    listener._woke = True
    app.listener = listener
    app.wake.process.return_value = main.TIER_BOTH
    app.wake.last_score = 0.9
    app.wake.last_model = "alexa"
    app.wake.personal_hit = False
    app._confirm_acoustic_wake = Mock(side_effect=AssertionError("duplicate confirmation"))
    frame = np.ones(512, dtype=np.int16)
    mic.vad_frame.side_effect = lambda value: value
    reads = iter([frame])

    def read_once_then_stop():
        try:
            return next(reads)
        except StopIteration:
            app._stop.set()
            raise TimeoutError("bounded test stop")

    mic.read.side_effect = read_once_then_stop

    app.run()

    app._confirm_acoustic_wake.assert_not_called()


@pytest.mark.parametrize("wait_ms,expected", [(0, 0), (12.8, 12), (None, None), (-1, None), (True, None)])
def test_capture_result_records_only_valid_wait_duration(wait_ms, expected):
    event = main.Event()
    captured = {"wait_duration_ms": wait_ms, "end_reason": "silence", "truncated": False}

    main.Assistant._capture_result(event, captured)

    assert event.capture_wait_ms == expected
    assert event.capture_end_reason == "silence"


def test_acoustic_refusal_decodes_raw_pair_and_archives_processed_pair_only():
    from assistant.audio import Microphone
    from assistant.listener import SttListener

    app = main.Assistant.__new__(main.Assistant)
    app.cfg = Config({})
    app.listener = SttListener(Config({"listener": {"phrases": ["alexa"]}}), None, None)
    app.listener_stt = Mock(model_name="faster-whisper/small.en")
    app.listener_stt.transcribe.return_value = ("please stop", 80)
    app.stt = Mock()
    app.stt.transcribe.return_value = ("background conversation", 170)
    app.wake = SimpleNamespace(personal_hit=True, personal_score=0.8,
                               last_score=0.0, last_model="alexa")
    app._record_near_miss = Mock()
    mic = Microphone()
    raw = np.arange(16000, dtype=np.int16)
    processed = raw // 2
    mic.preroll_pair = Mock(return_value=(raw, processed))

    assert app._confirm_acoustic_wake(mic) is False

    mic.preroll_pair.assert_called_once_with(1.6)
    assert app.listener_stt.transcribe.call_args.args[0] is raw
    assert app.stt.transcribe.call_args.args[0] is raw
    assert all(call.args[0] is not processed for call in (
        app.listener_stt.transcribe.call_args, app.stt.transcribe.call_args))
    near_miss = app._record_near_miss.call_args.args[0]
    assert near_miss["audio"] is raw
    assert near_miss["processed_audio"] is processed
    assert near_miss["processed_alignment"] == "frame_paired"
    assert near_miss["decision_source"] == "acoustic_confirmation"


@pytest.mark.parametrize("text", ["Alexa pause", "we are discussing alexa"])
def test_acoustic_shortcut_never_discards_a_command_or_wakes_on_a_mention(text):
    from assistant.listener import SttListener
    app = main.Assistant.__new__(main.Assistant)
    app.cfg = Config({})
    app.listener = SttListener(Config({"listener": {"phrases": ["alexa"]}}), None, None)
    app.listener_stt = Mock()
    app.listener_stt.transcribe.return_value = (text, 100)
    mic = Mock()
    mic.preroll.return_value = np.zeros(16000, dtype=np.int16)
    assert app._confirm_acoustic_wake(mic) is False


@pytest.mark.parametrize("turbo_text,expected", [("Alexa", True), ("hey Alexa", True),
                                               ("we discuss Alexa", False), ("Alex", False)])
def test_qualified_acoustic_hit_can_use_strict_turbo_confirmation(turbo_text, expected):
    from assistant.listener import SttListener
    app = main.Assistant.__new__(main.Assistant)
    app.cfg = Config({})
    app.listener = SttListener(Config({"listener": {"phrases": ["alexa"]}}), None, None)
    app.listener_stt = Mock()
    app.listener_stt.transcribe.return_value = ("please", 80)
    app.stt = Mock()
    app.stt.transcribe.return_value = (turbo_text, 270)
    app.wake = SimpleNamespace(personal_hit=True, personal_score=0.8, last_score=0.0, last_model="alexa")
    app._record_near_miss = Mock()
    mic = Mock()
    audio = np.ones(16000, dtype=np.int16)
    mic.preroll.return_value = audio
    assert app._confirm_acoustic_wake(mic) is expected
    assert app.stt.transcribe.call_args.args[0] is audio


def test_turbo_confirmed_one_breath_request_is_deferred_without_flushing():
    from assistant.listener import SttListener
    app = main.Assistant.__new__(main.Assistant)
    app.cfg = Config({})
    app.listener = SttListener(Config({"listener": {"phrases": ["alexa"]}}), None, None)
    app.listener_stt = Mock()
    app.listener_stt.transcribe.return_value = ("please", 80)
    app.stt = Mock()
    app.stt.transcribe.return_value = ("Alexa play Demons", 270)
    app.wake = SimpleNamespace(personal_hit=True, personal_score=0.8, last_score=0.0, last_model="alexa")
    app._duck_on_wake = Mock()
    app._record_near_miss = Mock()
    mic = Mock()
    mic.preroll.return_value = np.ones(16000, dtype=np.int16)
    assert app._confirm_acoustic_wake(mic) is False
    assert app.listener._pending_wake[0] == "alexa"
    app._duck_on_wake.assert_called_once()
    mic.flush.assert_not_called()


def test_recovery_does_not_use_broad_artist_hints_or_learn_misheard_query():
    from assistant.resolution import Candidate
    app = _paused_app()
    app.spotify.resolve_track.side_effect = LookupError("not found")
    app._recover_with_name_hints = Mock(side_effect=AssertionError("unsafe recovery"))
    app._recover_track_from_audio = Mock()
    candidate = Candidate("spotify:track:abc", "Red", "Seedhe Maut")
    event = main.Event()
    def recover(*_):
        event.recovery_transcript = "play red by Seedhe Maut"
        return candidate, "red by Seedhe Maut"
    app._recover_track_from_audio.side_effect = recover
    app._execute(main.parse("play red by see the mode"), event)
    app.spotify.play_candidate.assert_called_once_with(candidate)
    app.spotify.remember_track.assert_not_called()
    app._recover_with_name_hints.assert_not_called()


def test_brain_context_never_queries_device_on_the_command_path(monkeypatch):
    from assistant import local_media
    monkeypatch.setattr(local_media, "title", Mock(side_effect=AssertionError("blocking read")))
    app = main.Assistant.__new__(main.Assistant)
    app.vocabulary = {"artists": ["A"], "hints": ["A", "Playlist"]}
    assert app._brain_context() == {"artists": ["A"], "playlists": ["Playlist"], "now_playing": ""}


def test_mid_sentence_wake_without_command_stays_silent():
    from contextlib import nullcontext
    from assistant.listener import SttListener
    app = main.Assistant.__new__(main.Assistant)
    app.cfg = Config({})
    app.listener = SttListener(Config({"listener": {"phrases": ["alexa"]}}), None, None)
    app.listener_stt = Mock(model_name="small")
    app.stt = Mock(model_name="turbo")
    app.stt.transcribe.return_value = ("we were talking about alexa and stuff", 100)
    app.spotify = Mock()
    app.spotify.playback_snapshot.return_value = (True, None)
    app.feedback = Mock()
    app.telemetry = Mock()
    app._early_lease = None
    app._retry_unknown = Mock()
    utt = SimpleNamespace(command="and stuff", matched_phrase="alexa", mid_utterance=True,
                          transcript="we were talking about alexa and stuff", latency_ms=1,
                          audio=np.zeros(512, dtype=np.int16), completed_at=0)
    app.handle_utterance(Mock(), utt)
    app._retry_unknown.assert_not_called()
    app.feedback.say.assert_not_called()
    assert app.telemetry.write.call_args.args[0].outcome == "false_accept"


def test_text_alias_needs_recent_matching_acoustic_evidence():
    app = main.Assistant.__new__(main.Assistant)
    app.cfg = Config({"wake": {"threshold_high": 0.60}})
    assert app._authorize_text_trigger("alias", "alexa") is False
    app._last_strong_wake = (time.monotonic(), 0.70, "alexa")
    assert app._authorize_text_trigger("alias", "alexa") is True
    assert app._authorize_text_trigger("mid", "hey mycroft") is False
    app._last_strong_wake = (time.monotonic() - 3.0, 0.70, "alexa")
    assert app._authorize_text_trigger("alias", "alexa") is False


def test_rejection_counts_do_not_archive_an_unrelated_later_mic_snapshot():
    app = main.Assistant.__new__(main.Assistant)
    app.cfg = Config({"wake": {"confirm_seconds": 1.6}})
    app.wake = Mock(last_score=0.93, last_model="alexa")
    app.telemetry = Mock(archive_audio=True)
    app.telemetry.save_audio.return_value = "audio/rejected.wav"
    mic = Mock()
    mic.preroll.return_value = np.zeros(16000, dtype=np.int16)

    app._record_wake_rejection("acoustic_unconfirmed", mic=mic)
    event = app.telemetry.write.call_args.args[0]
    assert event.trigger_source == "wake_rejected"
    assert event.rejection_reason == "acoustic_unconfirmed:1"
    assert event.rejection_count == 1
    assert event.wake_model == "alexa"
    assert event.wake_score == 0.93
    assert event.audio_path == ""
    assert event.transcript == ""
    app._record_wake_rejection("acoustic_unconfirmed", mic=mic)
    app.telemetry.save_audio.assert_not_called()
    mic.preroll.assert_not_called()
    app.telemetry.write.assert_called_once()


def test_text_handled_request_resets_acoustic_state(monkeypatch):
    app = main.Assistant.__new__(main.Assistant)
    app.cfg = Config({})
    app.mode = "stt"
    app._stop = threading.Event()
    app._trigger = threading.Event()
    app._start_hotkeys = Mock()
    app._sync_paused_mic = Mock()
    app._close_paused_mic = Mock()
    app._paused_mic = None
    app._early_lease = None
    app.wake = Mock(last_score=0.70, last_model="alexa")
    app.wake.process.return_value = main.TIER_BOTH
    app._last_strong_wake = None
    app.listener = Mock(phrases=["alexa"])
    app.listener.feed.return_value = SimpleNamespace(triggered=True)
    app.handle_utterance = Mock(side_effect=lambda *args: app._stop.set())
    laptop = MagicMock()
    laptop.__enter__.return_value = laptop
    laptop.read.return_value = np.zeros(512, dtype=np.int16)
    monkeypatch.setattr(main, "Microphone", Mock(return_value=laptop))

    app.run()

    app.wake.reset.assert_called_once()
    assert app._last_strong_wake is None


def _recovery_loop(monkeypatch):
    app = main.Assistant.__new__(main.Assistant)
    app.cfg = Config({})
    app.mode = "stt"
    app._stop = threading.Event()
    app._trigger = threading.Event()
    app._start_hotkeys = Mock()
    app._sync_paused_mic = Mock()
    app._close_paused_mic = Mock()
    app._paused_mic = None
    app._early_lease = Mock()
    app._early_lease.expired.return_value = False
    app.wake = Mock()
    app.wake.process.return_value = "loose"
    app._last_strong_wake = (time.monotonic(), 0.8, "alexa")
    app.listener = Mock(phrases=["alexa"])
    app.listener.feed.return_value = None
    mic = MagicMock()
    mic.__enter__.return_value = mic
    monkeypatch.setattr(main, "Microphone", Mock(return_value=mic))
    return app, mic


def test_mic_timeout_reopens_without_reloading_models(monkeypatch):
    app, mic = _recovery_loop(monkeypatch)
    frame = np.zeros(512, dtype=np.int16)
    mic.read.side_effect = [TimeoutError("stalled"), frame]
    app.listener.feed.side_effect = lambda *_: app._stop.set()
    app.run()
    mic.reopen.assert_called_once()
    app.listener.flush.assert_called_once()
    app.wake.reset.assert_called_once()
    assert app._last_strong_wake is None
    assert app.listener.feed.call_args.args[0] is frame
    app._early_lease.acquire.assert_not_called()


def test_mic_timeout_during_shutdown_exits_without_reopen(monkeypatch):
    app, mic = _recovery_loop(monkeypatch)
    def stopped_read():
        app._stop.set()
        raise TimeoutError("stopping")
    mic.read.side_effect = stopped_read
    app.run()
    mic.reopen.assert_not_called()
    app.listener.flush.assert_not_called()


@pytest.mark.parametrize("reopen_fails", [False, True])
def test_mic_recovery_escalates_after_two_attempts_without_audio(monkeypatch, reopen_fails):
    app, mic = _recovery_loop(monkeypatch)
    mic.read.side_effect = TimeoutError("stalled")
    if reopen_fails:
        mic.reopen.side_effect = RuntimeError("device unavailable")
    with pytest.raises(RuntimeError if reopen_fails else TimeoutError):
        app.run()
    assert mic.reopen.call_count == 2
    app.listener.feed.assert_not_called()


def test_mic_recovery_budget_resets_only_after_successful_frame(monkeypatch):
    app, mic = _recovery_loop(monkeypatch)
    frame = np.zeros(512, dtype=np.int16)
    mic.read.side_effect = [TimeoutError(), TimeoutError(), frame,
                            TimeoutError(), TimeoutError(), frame]
    def stop_after_second_frame(*_):
        if app.listener.feed.call_count == 2:
            app._stop.set()
    app.listener.feed.side_effect = stop_after_second_frame
    app.run()
    assert mic.reopen.call_count == 4


def test_queue_gap_flushes_damaged_context_and_processes_only_fresh_frame(monkeypatch):
    app, mic = _recovery_loop(monkeypatch)
    damaged, gap_frame, fresh = [np.full(512, value, dtype=np.int16) for value in (1, 2, 3)]
    counts = iter([0, 2, 2])
    reads = iter([damaged, gap_frame, fresh])
    gap_rejected = []

    def read_with_count():
        mic.queue_overflow_count = next(counts)
        frame = next(reads)
        if frame is fresh:
            assert app._early_lease.release.call_count == 1
        return frame

    mic.read.side_effect = read_with_count
    app._confirmed_acoustic_history = ("alexa", damaged, damaged)
    app._last_strong_wake = (time.monotonic(), 0.9, "alexa")
    app.handle = Mock()
    def record_gap(reason, mic):
        assert reason == "capture_queue_gap"
        assert mic is not None
        assert app._early_lease.release.call_count == 1
        gap_rejected.append(True)
    app._record_wake_rejection = Mock(side_effect=record_gap)
    def stop_on_fresh(frame, *_):
        if frame is fresh:
            assert gap_rejected == [True]
            assert app._early_lease.release.call_count == 1
            app._stop.set()
    app.listener.feed.side_effect = stop_on_fresh

    app.run()

    assert app.listener.feed.call_count == 2
    assert app.listener.feed.call_args_list[0].args[0] is damaged
    assert app.listener.feed.call_args_list[1].args[0] is fresh
    assert all(call.args[0] is not gap_frame for call in app.listener.feed.call_args_list)
    mic.flush.assert_called_once()
    app.listener.flush.assert_called_once()
    app.wake.reset.assert_called_once()
    assert app._early_lease.release.call_count == 2  # gap release plus normal shutdown cleanup
    assert app._last_strong_wake is None
    assert app._confirmed_acoustic_history is None
    app._record_wake_rejection.assert_called_once_with("capture_queue_gap", mic=mic)
    app.handle.assert_not_called()


def test_stable_positive_overflow_count_does_not_repeat_gap_reset(monkeypatch):
    app, mic = _recovery_loop(monkeypatch)
    fresh = np.full(512, 3, dtype=np.int16)
    counts = iter([2, 2])
    reads = iter([np.zeros(512, dtype=np.int16), fresh])

    def read_with_count():
        mic.queue_overflow_count = next(counts)
        return next(reads)

    mic.read.side_effect = read_with_count
    app.listener.feed.side_effect = lambda *_: app._stop.set()
    app._record_wake_rejection = Mock()

    app.run()

    mic.flush.assert_called_once()
    app.listener.flush.assert_called_once()
    app.wake.reset.assert_called_once()
    app._record_wake_rejection.assert_called_once_with("capture_queue_gap", mic=mic)
    app.listener.feed.assert_called_once()
    assert app.listener.feed.call_args.args[0] is fresh


@pytest.mark.parametrize("count", [None, "2", True, 1.5])
def test_unusable_overflow_counter_does_not_reset_audio_context(monkeypatch, count):
    app, mic = _recovery_loop(monkeypatch)
    mic.queue_overflow_count = count
    frame = np.ones(512, dtype=np.int16)
    mic.read.side_effect = lambda: (app._stop.set() or frame)
    app.listener.feed.side_effect = lambda *_: app._stop.set()
    app._record_wake_rejection = Mock()

    app.run()

    mic.flush.assert_not_called()
    app.listener.flush.assert_not_called()
    app.wake.reset.assert_not_called()
    app._record_wake_rejection.assert_not_called()
    app.listener.feed.assert_called_once()


def test_echo_route_reopen_keeps_canceller_attached(monkeypatch):
    app = main.Assistant.__new__(main.Assistant)
    app.echo = Mock()
    app.echo.reference.route_changed.return_value = True
    mic = Mock(echo=app.echo)
    app._check_echo_reference()
    app.echo.reference.stop.assert_called_once()
    app.echo.reference.start.assert_called_once()
    assert mic.echo is app.echo


@pytest.mark.parametrize("transcript,mid,accepted", [
    ("alexa pause", False, True), ("hey alexa pause", False, True),
    # Terminal mentions are not standalone invocations.
    ("we discuss alexa", True, False), ("we discuss alexa today", True, False),
    ("lyrics alexa pause", True, True),
    ("remember pause", False, False), ("alexa play nonsense", False, True)])
def test_final_verifier_requires_exact_phrase_and_mid_command(transcript, mid, accepted):
    app = main.Assistant.__new__(main.Assistant)
    app.stt = Mock()
    app.stt.transcribe.return_value = (transcript, 12)
    result = app._verify_text_trigger(np.zeros(512, dtype=np.int16), "alexa", mid)
    assert bool(result[0]) is accepted


def test_verified_command_decode_is_reused_and_trigger_audio_archived():
    from assistant.listener import SttListener
    app = _paused_app()
    app.stt = Mock(model_name="turbo")
    app.stt.transcribe.side_effect = AssertionError("verified audio must not be decoded again")
    app.listener_stt = Mock(model_name="small")
    app.listener = SttListener(Config({"listener": {"phrases": ["alexa"]}}), None, None)
    app.telemetry = Mock()
    app.telemetry.save_audio.return_value = "trigger.wav"
    app._early_lease = None
    audio = np.zeros(16000, dtype=np.int16)
    utt = SimpleNamespace(command="pause", transcript="remember pause",
                          matched_phrase="alexa", latency_ms=2, audio=audio,
                          confirmation_mode="second_stt", confirmation_latency_ms=12,
                          confirmation_transcript="Alexa pause")
    app.handle_utterance(Mock(), utt)
    app.spotify.pause.assert_called_once()
    app.stt.transcribe.assert_not_called()
    event = app.telemetry.write.call_args.args[0]
    assert event.transcript == "Alexa pause"
    assert event.trigger_audio_path == "trigger.wav"
    assert event.trigger_confirmation_mode == "second_stt"
    assert app.telemetry.save_audio.call_args.args[1] is audio


def test_acoustic_shortcut_alias_still_requires_matching_evidence():
    from assistant.listener import SttListener
    app = main.Assistant.__new__(main.Assistant)
    app.cfg = Config({"listener": {"phrases": ["alexa"], "aliases": {"alex": "alexa"}}})
    app.listener = SttListener(app.cfg, None, None, authorize_trigger=app._authorize_text_trigger)
    app.listener_stt = Mock()
    app.listener_stt.transcribe.return_value = ("Alex", 1)
    mic = Mock()
    mic.preroll.return_value = np.zeros(16000, dtype=np.int16)
    assert not app._confirm_acoustic_wake(mic)
    app._last_strong_wake = (time.monotonic(), 0.8, "alexa")
    assert app._confirm_acoustic_wake(mic)


def test_failed_primary_mic_open_does_not_start_playback_watcher(monkeypatch):
    app, mic = _recovery_loop(monkeypatch)
    app.cfg = Config({"audio": {"paused_input_device": "headset"}})
    mic.__enter__.side_effect = RuntimeError("all capture backends failed")
    thread = Mock()
    monkeypatch.setattr(main.threading, "Thread", thread)
    with pytest.raises(RuntimeError, match="capture backends"):
        app.run()
    thread.assert_not_called()


def test_hotkey_request_clears_stale_acoustic_evidence(monkeypatch):
    app, mic = _recovery_loop(monkeypatch)
    mic.read.return_value = np.zeros(512, dtype=np.int16)
    app._trigger.set()
    app.handle = Mock(side_effect=lambda *_: app._stop.set())
    app.run()
    app.handle.assert_called_once_with(mic, "hotkey")
    app.wake.reset.assert_called_once()
    assert app._last_strong_wake is None


def test_wakeword_mode_unknown_retry_accepts_plain_command_without_text_listener():
    from contextlib import nullcontext
    app = _paused_app()
    app.cfg = Config({})
    app.listener = None
    app.telemetry = Mock()
    app._duck_music = lambda *_: nullcontext()
    app._ask_and_listen = Mock(return_value=("pause", {"audio": np.zeros(512, dtype=np.int16)}, 1))
    event = main.Event()
    app._retry_unknown(Mock(), event)
    app.spotify.pause.assert_called_once()
    assert event.outcome == "ok"


def test_capture_health_summary_is_bounded_and_contains_no_transcript(monkeypatch):
    app = main.Assistant.__new__(main.Assistant)
    app.cfg = Config({})
    app.wake = SimpleNamespace(last_score=0.4, last_model="alexa")
    app.listener = SimpleNamespace(segments_completed=2, discarded_short_segments=1)
    echo = SimpleNamespace(active=True, reference=SimpleNamespace(alive=Mock(return_value=True)))
    mic = SimpleNamespace(echo=echo, queue_overflow_count=3)
    log_info = Mock()
    monkeypatch.setattr(main.log, "info", log_info)
    moment = [0.0]
    monkeypatch.setattr(main.time, "monotonic", lambda: moment[0])
    raw = np.full(512, 2000, dtype=np.int16)
    clean = np.full(512, 100, dtype=np.int16)
    app._record_capture_diagnostics(mic, raw, clean, 0.7)
    moment[0] = 9.0
    app._record_capture_diagnostics(mic, raw, clean, 0.1)
    log_info.assert_not_called()
    moment[0] = 10.0
    app._record_capture_diagnostics(mic, raw, clean, 0.1)
    log_info.assert_called_once()
    values = log_info.call_args.args
    assert values[1:8] == (3, 2000.0, 2000.0, 100.0, 0.7, 1, 0.4)
    assert "transcript" not in values[0]
    echo.reference.alive.assert_called_once()
    assert app._capture_diagnostics is None
    app._record_capture_diagnostics(mic, raw, clean, 0.1)
    assert log_info.call_count == 1


def _capture_diagnostic_window(app, mic, moment, started, raw_frames, cleaned=None):
    """Feed one deterministic ten-second diagnostic window without audio I/O."""
    assert raw_frames
    for index, raw in enumerate(raw_frames):
        moment[0] = started + 9.9 * index / max(1, len(raw_frames))
        app._record_capture_diagnostics(mic, raw, cleaned, None)
    moment[0] = started + 10.0
    app._record_capture_diagnostics(mic, raw_frames[-1], cleaned, None)
    return started + 10.0


def _capture_diagnostic_app():
    app = main.Assistant.__new__(main.Assistant)
    app.cfg = Config({})
    app.wake = None
    app.listener = SimpleNamespace(segments_completed=0, discarded_short_segments=0)
    return app


def test_capture_diagnostics_warn_after_two_qualified_zero_windows(monkeypatch):
    app = _capture_diagnostic_app()
    moment = [0.0]
    monkeypatch.setattr(main.time, "monotonic", lambda: moment[0])
    zero = np.zeros(512, dtype=np.int16)
    residual = np.full(512, 50, dtype=np.int16)
    mic = SimpleNamespace(echo=None, queue_overflow_count=0)
    warning = Mock()
    info = Mock()
    monkeypatch.setattr(main.log, "warning", warning)
    monkeypatch.setattr(main.log, "info", info)

    now = _capture_diagnostic_window(app, mic, moment, 0.0, [zero] * 128, residual)
    assert app._capture_input_state == "digital_silence"
    assert app._capture_zero_windows == 1
    warning.assert_not_called()
    assert not any("signal returned" in str(call).lower() for call in info.call_args_list)

    now = _capture_diagnostic_window(app, mic, moment, now, [zero] * 128, residual)
    assert app._capture_input_state == "digital_silence"
    warning.assert_called_once()
    assert "zero-valued" in warning.call_args.args[0]
    assert "check microphone mute, permission" in warning.call_args.args[0]

    _capture_diagnostic_window(app, mic, moment, now, [zero] * 128, residual)
    warning.assert_called_once()
    assert not any("signal returned" in str(call).lower() for call in info.call_args_list)


@pytest.mark.parametrize("interruption", [
    [None] * 128,
    [np.zeros(512, dtype=np.int16)] * 3,
])
def test_capture_diagnostics_invalid_or_undersampled_window_breaks_zero_streak(
        monkeypatch, interruption):
    app = _capture_diagnostic_app()
    moment = [0.0]
    monkeypatch.setattr(main.time, "monotonic", lambda: moment[0])
    zero = np.zeros(512, dtype=np.int16)
    quiet = np.ones(512, dtype=np.int16)
    mic = SimpleNamespace(echo=None, queue_overflow_count=0)
    warning = Mock()
    info = Mock()
    monkeypatch.setattr(main.log, "warning", warning)
    monkeypatch.setattr(main.log, "info", info)

    now = _capture_diagnostic_window(app, mic, moment, 0.0, [zero] * 128)
    assert app._capture_zero_windows == 1
    now = _capture_diagnostic_window(app, mic, moment, now, interruption)
    assert app._capture_input_state == "unknown"
    assert app._capture_zero_windows == 0
    warning.assert_not_called()

    now = _capture_diagnostic_window(app, mic, moment, now, [zero] * 128)
    assert app._capture_input_state == "digital_silence"
    warning.assert_not_called()
    now = _capture_diagnostic_window(app, mic, moment, now, [zero] * 128)
    warning.assert_called_once()
    _capture_diagnostic_window(app, mic, moment, now, [quiet])
    assert app._capture_input_state == "signal_present"
    assert app._capture_zero_windows == 0
    assert any("signal returned" in str(call).lower() for call in info.call_args_list)


def test_capture_diagnostics_quiet_nonzero_raw_signal_recovers_once(monkeypatch):
    app = _capture_diagnostic_app()
    moment = [0.0]
    monkeypatch.setattr(main.time, "monotonic", lambda: moment[0])
    zero = np.zeros(512, dtype=np.int16)
    quiet = np.ones(512, dtype=np.int16)
    residual = np.full(512, 50, dtype=np.int16)
    mic = SimpleNamespace(echo=None, queue_overflow_count=0)
    warning = Mock()
    info = Mock()
    monkeypatch.setattr(main.log, "warning", warning)
    monkeypatch.setattr(main.log, "info", info)

    now = _capture_diagnostic_window(app, mic, moment, 0.0, [zero] * 128, residual)
    now = _capture_diagnostic_window(app, mic, moment, now, [zero] * 128, residual)
    assert app._capture_input_state == "digital_silence"
    warning.assert_called_once()
    assert not any("signal returned" in str(call).lower() for call in info.call_args_list)

    mixed = [quiet if index == 63 else zero for index in range(128)]
    now = _capture_diagnostic_window(app, mic, moment, now, mixed, zero)
    assert app._capture_input_state == "signal_present"
    assert app._capture_zero_windows == 0
    recoveries = [call for call in info.call_args_list if "signal returned" in str(call).lower()]
    assert len(recoveries) == 1

    _capture_diagnostic_window(app, mic, moment, now, [quiet] * 128, zero)
    assert len([call for call in info.call_args_list
                if "signal returned" in str(call).lower()]) == 1


def test_completed_untriggered_speech_records_bounded_rejection(monkeypatch):
    app, mic = _recovery_loop(monkeypatch)
    frame = np.zeros(512, dtype=np.int16)
    mic.read.return_value = frame
    mic.vad_frame.return_value = frame
    def completed(*_):
        app._stop.set()
        return SimpleNamespace(triggered=False, audio=frame)
    app.listener.feed.side_effect = completed
    app._record_wake_rejection = Mock()
    app.handle = Mock()
    app.run()
    app._record_wake_rejection.assert_called_once_with("speech_no_trigger", mic=mic, duration_ms=32)
    app.handle.assert_not_called()


def test_text_rejection_aggregates_counts_without_misrepresenting_snapshot_audio(monkeypatch):
    app = main.Assistant.__new__(main.Assistant)
    app.cfg = Config({})
    app.wake = None
    app.telemetry = Mock(archive_audio=True)
    app.telemetry.save_audio.return_value = "short.wav"
    mic = Mock()
    mic.preroll.return_value = np.zeros(16000, dtype=np.int16)
    app._active_mic = mic
    moment = [0.0]
    monkeypatch.setattr(main.time, "monotonic", lambda: moment[0])
    app._record_text_rejection("speech_too_short", 64)
    moment[0] = 10.0
    app._record_text_rejection("speech_too_short", 64)
    app.telemetry.save_audio.assert_not_called()
    assert app.telemetry.write.call_count == 1
    event = app.telemetry.write.call_args.args[0]
    assert event.rejection_reason == "speech_too_short:1"
    assert event.audio_path == ""
    assert event.transcript == ""
    moment[0] = 30.0
    app._record_text_rejection("speech_too_short", 64)
    app.telemetry.save_audio.assert_not_called()
    assert app.telemetry.write.call_args.args[0].rejection_count == 2
    app._active_mic = None
    moment[0] = 60.0
    app._record_text_rejection("speech_too_short", 64)
    app.telemetry.save_audio.assert_not_called()
    mic.preroll.assert_not_called()


def test_truncated_capture_cannot_reach_global_router_or_submit():
    app = main.Assistant.__new__(main.Assistant)
    app.global_assistant = Mock()
    app._global_say = Mock(return_value=True)
    event = main.Event()
    app._capture_result(event, {"end_reason": "utterance_limit", "truncated": True})
    assert app._handle_global("Create a new chat and", Mock(), event)
    app.global_assistant.handle.assert_not_called()
    app.global_assistant.end_session.assert_called_once()
    assert event.outcome == "capture_truncated"


def test_truncated_capture_is_refused_when_codex_is_disabled():
    app = main.Assistant.__new__(main.Assistant)
    app.global_assistant = None
    app._global_say = Mock(return_value=True)
    event = main.Event(capture_truncated=True)
    assert app._handle_global("play Red and", Mock(), event)
    assert event.outcome == "capture_truncated"
    app._global_say.assert_called_once()


def test_exact_wake_candidate_does_not_consume_activation_evidence():
    app = main.Assistant.__new__(main.Assistant)
    app.telemetry = Mock()
    app._wake_trace = Mock()
    app._wake_trace.snapshot.return_value = {"alexa": 0.41}
    raw = np.arange(512, dtype=np.int16)
    clean = raw // 2
    app._record_wake_candidate({"audio": raw, "processed_audio": clean,
                               "reason": "speech_too_short", "transcript": "",
                               "segment_id": "example", "input_device": "sounddrum",
                               "end_reason": "silence", "truncated": False})
    call = app.telemetry.record_near_miss.call_args
    assert call.args[0] is raw and call.kwargs["processed_audio"] is clean
    assert call.kwargs["input_device"] == "sounddrum"
    assert call.kwargs["scores"] == {"alexa": 0.41}
    app._wake_trace.take.assert_not_called()


def test_near_miss_forwards_paired_audio_and_decision_metadata():
    app = main.Assistant.__new__(main.Assistant)
    app.telemetry = Mock()
    app.listener_stt = SimpleNamespace(
        model_name="faster-whisper/small.en", runtime_info=lambda: "cuda")
    raw = np.ones(512, dtype=np.int16)
    processed = raw // 2

    app._record_near_miss({
        "audio": raw, "processed_audio": processed, "transcript": "please",
        "reason": "acoustic_unconfirmed", "kind": "acoustic_unconfirmed",
        "processed_alignment": "sample_aligned", "decision_source": "acoustic_confirmation",
        "relative_start_sample": 10, "relative_end_sample": 522, "sample_rate": 16000,
    })

    call = app.telemetry.record_near_miss.call_args
    assert call.args[0] is raw
    assert call.kwargs["processed_audio"] is processed
    assert call.kwargs["processed_alignment"] == "sample_aligned"
    assert call.kwargs["decision_source"] == "acoustic_confirmation"
    assert call.kwargs["relative_start_sample"] == 10
    assert call.kwargs["relative_end_sample"] == 522
    assert call.kwargs["sample_rate"] == 16000


def test_active_diagnostic_mic_clears_after_loop_failure(monkeypatch):
    app, mic = _recovery_loop(monkeypatch)
    mic.read.side_effect = RuntimeError("capture failure")
    with pytest.raises(RuntimeError, match="capture failure"):
        app.run()
    assert app._active_mic is None


def _repeat_app(transcripts):
    from assistant.listener import SttListener
    app = main.Assistant.__new__(main.Assistant)
    app.cfg = Config({})
    app.listener = SttListener(Config({"listener": {"phrases": ["alexa"]}}), None, None)
    app.feedback = Mock()
    app.endpointer = Mock()
    app.endpointer.record.return_value = {"audio": np.zeros(512, np.int16), "speech_started": True}
    app.stt = Mock()
    app.stt.transcribe.side_effect = [(t, 10) for t in transcripts]
    app._duck_music = lambda *a, **k: contextlib.nullcontext()
    return app


def test_repeated_wake_keeps_listening_for_the_command():
    app = _repeat_app(["play naksha"])
    text, _, latency = app._past_repeated_wake(Mock(), Mock(), "Alexa!", {"audio": None}, 5)
    assert text == "play naksha" and latency == 15
    app.endpointer.record.assert_called_once()


def test_wake_words_are_stripped_from_a_spoken_command():
    app = _repeat_app([])
    text, _, _ = app._past_repeated_wake(Mock(), Mock(), "Alexa, Alexa, pause", {}, 5)
    assert text == "pause"
    app.endpointer.record.assert_not_called()


def test_only_wake_words_gives_up_silently():
    app = _repeat_app(["Alexa", "Alexa"])
    text, _, _ = app._past_repeated_wake(Mock(), Mock(), "Alexa", {}, 5)
    assert text == ""


@pytest.mark.parametrize("playing,stopped", [(True, False), (False, True), (None, False)])
def test_queue_requires_positive_evidence_that_playback_is_stopped(monkeypatch, playing, stopped):
    from assistant import local_media
    app = main.Assistant.__new__(main.Assistant)
    monkeypatch.setattr(local_media, "is_playing", lambda: playing)
    assert app._nothing_playing() is stopped


def _install_computer_tools_double(monkeypatch, *, desktop_domain=True):
    import sys
    import assistant

    class ComputerControlError(RuntimeError):
        pass

    class ComputerDeliveryUnknown(ComputerControlError):
        pass

    tools = SimpleNamespace(
        ComputerControlError=ComputerControlError,
        ComputerDeliveryUnknown=ComputerDeliveryUnknown,
        open_app=Mock(return_value="Desktop request recorded."),
        search_browser=Mock(return_value="Desktop request recorded."),
        show_spotify=Mock(return_value="Desktop request recorded."),
        start_spotify_jam=Mock(return_value="Spotify Jam started."),
        desktop_domain=Mock(return_value=desktop_domain),
    )
    monkeypatch.setitem(sys.modules, "assistant.computer_tools", tools)
    monkeypatch.setattr(assistant, "computer_tools", tools, raising=False)
    return tools


@pytest.mark.parametrize(("action", "query", "browser", "method", "args"), [
    (main.Action.OPEN_APP, "Spotify", "Opera", "open_app", ("Spotify",)),
    (main.Action.SEARCH_BROWSER, "weather in Delhi", "Opera", "search_browser",
     ("weather in Delhi", "Opera")),
    (main.Action.SHOW_TRACK, "Keychain Lalu by Arpit Bala", "Opera", "show_spotify",
     ("Keychain Lalu by Arpit Bala", "track")),
    (main.Action.SHOW_ALBUM, "Keychain Lalu by Arpit Bala", "Opera", "show_spotify",
     ("Keychain Lalu by Arpit Bala", "album")),
])
def test_desktop_intents_forward_to_navigation_without_spotify(
        monkeypatch, action, query, browser, method, args):
    tools = _install_computer_tools_double(monkeypatch)
    app = _paused_app()
    event = main.Event()

    app._execute(main.Intent(action, query=query, browser=browser), event)

    getattr(tools, method).assert_called_once_with(*args)
    for name in ("open_app", "search_browser", "show_spotify"):
        if name != method:
            getattr(tools, name).assert_not_called()
    assert app.spotify.method_calls == []
    assert event.outcome == "requested"
    assert event.playback_method == "desktop_request"
    assert event.spotify_action == ""
    assert event.feedback_text == "Desktop request recorded."
    app.feedback.say.assert_called_once_with("Desktop request recorded.", force=True, blocking=False)


def test_desktop_refusal_is_spoken_without_touching_spotify(monkeypatch):
    tools = _install_computer_tools_double(monkeypatch)
    tools.open_app.side_effect = tools.ComputerControlError("I can only open supported apps.")
    app = _paused_app()
    event = main.Event()

    app._execute(main.Intent(main.Action.OPEN_APP, query="unknown tool"), event)

    assert event.outcome == "desktop_refused"
    assert event.feedback_text == "I can only open supported apps."
    app.feedback.say.assert_called_once_with(
        "I can only open supported apps.", force=True, blocking=False)
    assert app.spotify.method_calls == []


def test_start_spotify_jam_uses_desktop_tool_without_spotify_or_volume_calls(monkeypatch):
    tools = _install_computer_tools_double(monkeypatch)
    app = _paused_app()
    set_volume = Mock()
    monkeypatch.setattr(main.ducking, "set_command_volume", set_volume)
    event = main.Event()

    app._execute(main.Intent(main.Action.START_JAM), event)

    tools.start_spotify_jam.assert_called_once_with()
    for name in ("open_app", "search_browser", "show_spotify"):
        getattr(tools, name).assert_not_called()
    assert app.spotify.method_calls == []
    set_volume.assert_not_called()
    assert event.outcome == "ok"
    assert event.playback_method == "desktop_request"
    assert event.spotify_action == ""
    assert event.feedback_text == "Spotify Jam started."
    app.feedback.say.assert_called_once_with("Spotify Jam started.", force=True, blocking=False)


def test_start_spotify_jam_refusal_is_spoken_without_spotify_or_volume_calls(monkeypatch):
    tools = _install_computer_tools_double(monkeypatch)
    tools.start_spotify_jam.side_effect = tools.ComputerControlError(
        "Spotify Jam is unavailable on this device.")
    app = _paused_app()
    set_volume = Mock()
    monkeypatch.setattr(main.ducking, "set_command_volume", set_volume)
    event = main.Event()

    app._execute(main.Intent(main.Action.START_JAM), event)

    tools.start_spotify_jam.assert_called_once_with()
    assert app.spotify.method_calls == []
    set_volume.assert_not_called()
    assert event.outcome == "desktop_refused"
    assert event.playback_method == "desktop_request"
    assert event.spotify_action == ""
    assert event.feedback_text == "Spotify Jam is unavailable on this device."
    app.feedback.say.assert_called_once_with(
        "Spotify Jam is unavailable on this device.", force=True, blocking=False)


def _global_app_for_desktop_priority(controller):
    app = main.Assistant.__new__(main.Assistant)
    app.global_assistant = controller
    app.brain = SimpleNamespace(warming=False)
    app._global_say = Mock(return_value=True)
    return app


class _NavigationRetirementController:
    def __init__(self, order=None):
        self.order = order if order is not None else []
        self.session = SimpleNamespace(active=False)
        self._pending_choice = None
        self._unsent_request = None
        self.retire_calls = 0
        self.cancel_calls = 0
        self.send_calls = 0
        self.create_calls = 0
        self.handle_calls = []

    def retire_navigation(self):
        self.retire_calls += 1
        self.order.append("retire_navigation")

    def cancel(self, *_args, **_kwargs):
        self.cancel_calls += 1

    def send(self, *_args, **_kwargs):
        self.send_calls += 1

    def create(self, *_args, **_kwargs):
        self.create_calls += 1

    def handle(self, command):
        self.handle_calls.append(command)
        return None


def test_unheld_desktop_domain_bypasses_global_classifier(monkeypatch):
    tools = _install_computer_tools_double(monkeypatch)
    controller = _NavigationRetirementController()
    app = _global_app_for_desktop_priority(controller)

    assert app._handle_global("Open Calculator", Mock(), main.Event()) is False

    tools.desktop_domain.assert_called_once_with("Open Calculator")
    assert controller.handle_calls == []
    assert controller.retire_calls == 0


def test_failed_named_codex_await_is_telemetrized_without_sending_or_starting_session():
    from assistant.global_assistant import GlobalReply

    controller = Mock()
    controller.session = SimpleNamespace(active=False, target_id="", draft="")
    controller._pending_choice = None
    controller._unsent_request = None
    controller._valid_unsent = Mock(return_value=False)
    controller.handle.return_value = GlobalReply(
        "Creating the Codex task.", await_token="await-123")
    app = _global_app_for_desktop_priority(controller)
    app._await_global_creation = Mock(return_value=GlobalReply(
        "Codex could not finish creating the task. Nothing was sent.", status="failed"))
    app._run_global_session = Mock()
    event = main.Event()

    mic = Mock()
    assert app._handle_global("Codex create a task", mic, event) is True

    app._await_global_creation.assert_called_once_with(mic, event, "await-123")
    assert event.global_action == "create_ended"
    assert event.global_status == "failed"
    assert event.outcome == "global_failed"
    assert event.global_target == ""
    assert event.global_prompt_chars == 0
    controller.send.assert_not_called()
    controller.create.assert_not_called()
    controller.session.active = False
    app._run_global_session.assert_not_called()


@pytest.mark.parametrize("context", ["active_session", "pending_choice", "held_request", "codex_source"])
def test_active_or_explicit_codex_context_keeps_priority_over_desktop_domain(monkeypatch, context):
    from assistant.global_assistant import _PendingChoice, _UnsentRequest

    tools = _install_computer_tools_double(monkeypatch)
    controller = Mock()
    controller.session = SimpleNamespace(active=False)
    controller._pending_choice = None
    controller._unsent_request = None
    controller._valid_unsent = Mock(return_value=True)
    controller.handle.return_value = None
    command = "Open Calculator"
    if context == "active_session":
        controller.session.active = True
    elif context == "pending_choice":
        controller._pending_choice = _PendingChoice(None, (), (), 1.0)
    elif context == "held_request":
        controller._unsent_request = _UnsentRequest("source", "prompt", (), 0, 10.0, 20.0)
    else:
        command = "Open the Codex chat"
    app = _global_app_for_desktop_priority(controller)

    assert app._handle_global(command, Mock(), main.Event()) is False

    controller.handle.assert_called_once_with(command)
    tools.desktop_domain.assert_not_called()


def test_restart_album_stays_on_spotify_music_path():
    app = _paused_app()
    app._execute_music = Mock()
    intent = main.Intent(main.Action.RESTART_ALBUM)
    event = main.Event()

    app._execute(intent, event)

    app._execute_music.assert_called_once_with(intent, event, None, None, None, None)
    assert app.spotify.method_calls == []


@pytest.mark.parametrize("refuse", [False, True])
def test_desktop_intent_retires_navigation_before_open_and_never_mutates_codex(monkeypatch, refuse):
    order = []
    tools = _install_computer_tools_double(monkeypatch)
    controller = _NavigationRetirementController(order)

    def open_app(query):
        order.append("open_app")
        if refuse:
            raise tools.ComputerControlError("I can only open supported apps.")
        return "Desktop request recorded."

    tools.open_app.side_effect = open_app
    app = _paused_app()
    app.global_assistant = controller
    event = main.Event()

    app._execute(main.Intent(main.Action.OPEN_APP, query="Calculator"), event)

    assert order == ["retire_navigation", "open_app"]
    assert controller.retire_calls == 1
    assert controller.cancel_calls == controller.send_calls == controller.create_calls == 0
    assert controller.handle_calls == []
    assert app.spotify.method_calls == []
    assert event.outcome == ("desktop_refused" if refuse else "requested")
