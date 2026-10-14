"""The brain may repair names; it may never invent them or exceed limits."""
import json
from unittest.mock import Mock
import threading

import pytest

from assistant.brain import Brain, EXAMPLES, _resolve_request_span, decide
from assistant.intent import Action

CONTEXT = {"artists": ["Seedhe Maut", "Radiohead", "Aerosmith"],
           "playlists": ["DHH Gym", "Focus"], "now_playing": ""}


def brain():
    return Brain("test-model", context=lambda: CONTEXT)


def call(tool, **arguments):
    return {"tool": tool, "arguments": arguments}


def test_structured_request_selects_complete_span_after_sentence_noise():
    heard = "Thanks for waiting. Play Red by Seedhe Maut. Thanks."
    result = brain().to_intent({"tool": "play_song",
                                "arguments": {"title": "Red", "artist": "Seedhe Maut"},
                                "request": {"status": "ready",
                                            "text": "Play Red by Seedhe Maut"}}, heard)
    assert result.action is Action.PLAY_TRACK


def test_structured_request_keeps_supported_compound_volume_in_selected_span():
    heard = "Play Red by Seedhe Maut and set volume to 30"
    result = brain().to_intent({"tool": "play_song",
                                "arguments": {"title": "Red", "artist": "Seedhe Maut",
                                              "volume_after_play": 30},
                                "request": {"status": "ready", "text": heard}},
                               heard)
    assert result.action is Action.PLAY_TRACK and result.volume_after_play == 30


def test_structured_request_keeps_queue_and_radio_modifiers():
    queued = "queue Red by Seedhe Maut"
    result = brain().to_intent({"tool": "queue_song",
                                "arguments": {"title": "Red", "artist": "Seedhe Maut"},
                                "request": {"status": "ready", "text": queued}},
                               queued)
    assert result.action is Action.QUEUE
    radio = "play Seedhe Maut radio"
    result = brain().to_intent({"tool": "play_artist",
                                "arguments": {"artist": "Seedhe Maut", "radio": True},
                                "request": {"status": "ready", "text": radio}},
                               radio)
    assert result.action is Action.PLAY_ARTIST


@pytest.mark.parametrize("status", ["needs_more", "ambiguous", "no_command"])
def test_structured_nonready_request_fails_closed(status):
    heard = "pause"
    result = brain().to_intent({"tool": "pause", "arguments": {},
                                "request": {"status": status, "text": ""}}, heard)
    assert result.action is Action.UNKNOWN
    assert result.matched_rule == "brain_request_" + status


@pytest.mark.parametrize("heard,selected", [
    ("Don't pause the music", "pause the music"),
    ("He said pause", "pause"),
    ("Play Red by Seedhe Maut and set volume to 30", "Play Red by Seedhe Maut"),
    ("Pause not now", "Pause"),
    ("pause after this song", "pause"),
    ("play red only", "play red"),
    ("play red by Seedhe Maut", "play red by"),
])
def test_structured_request_rejects_mid_clause_or_connected_suffix(heard, selected):
    result = brain().to_intent({"tool": "pause", "arguments": {},
                                "request": {"status": "ready", "text": selected}}, heard)
    assert result.action is Action.UNKNOWN


@pytest.mark.parametrize("heard,selected", [
    ("play red. play red.", "play red"),
    ("play red by seedhe maut", "play seedhe maut by red"),
    ("play red by seedhe maut", "play the moon"),
    ("play red", ""),
])
def test_structured_request_rejects_duplicate_invented_reordered_or_empty_text(heard, selected):
    result = brain().to_intent({"tool": "play_song",
                                "arguments": {"title": "Red", "artist": ""},
                                "request": {"status": "ready", "text": selected}}, heard)
    assert result.action is Action.UNKNOWN


def test_structured_request_preserves_source_case_and_derived_offsets():
    heard = "Thanks. pLaY ReD by SeEdHe MaUt. Thanks."
    resolved = _resolve_request_span({"status": "ready", "text": "PLAY RED BY SEEDHE MAUT"},
                                     heard)
    start = heard.index("pLaY")
    end = heard.index(".", start)
    assert resolved == ("ready", start, end, heard[start:end])


@pytest.mark.parametrize("heard", [
    "Play Red. I said play Red.",
    "Pause. Play Red.",
])
def test_structured_whole_capture_with_multiple_commands_is_ambiguous(heard):
    resolved = _resolve_request_span({"status": "ready", "text": heard}, heard)
    assert resolved == ("ambiguous", None, None, "")


@pytest.mark.parametrize("heard,selected", [
    ("The window is open. Play Dream On by Aerosmith. Thank you.",
     "Play Dream On by Aerosmith"),
    ("Play Mr. Blue Sky", "Play Mr. Blue Sky"),
    ("Play Dr. Dre", "Play Dr. Dre"),
    ("Play Red Red Wine", "Play Red Red Wine"),
])
def test_sentence_guard_keeps_single_commands_abbreviations_and_repeated_title(heard, selected):
    resolved = _resolve_request_span({"status": "ready", "text": selected}, heard)
    assert resolved[0] == "ready"
    assert resolved[3] == selected


def test_sentence_guard_blocks_fast_transport_when_another_command_follows():
    model = Mock()
    intent = decide("Pause. Play Red", model)
    assert intent.action is Action.UNKNOWN
    assert intent.matched_rule == "ambiguous_compound"
    model.decide.assert_not_called()


def test_structured_request_rejects_nonstring_metadata_and_nonempty_nonready_text():
    heard = "pause"
    for request in ({"status": "ready", "text": 4},
                    {"status": "ready", "text": None},
                    {"status": "ambiguous", "text": "pause"}):
        assert brain().to_intent({"tool": "pause", "arguments": {}, "request": request},
                                 heard).action is Action.UNKNOWN


@pytest.mark.parametrize("said", [
    "Thanks for waiting. Play Red by Seedhe Maut. Thanks.",
    "Before that. Play Red by Seedhe Maut and set volume to 30. Thanks.",
    "He said pause",
    "Don't pause",
    "Play Red by",
    "Pause and lower the volume",
])
def test_messages_preserve_explicit_structured_examples(said):
    original = next(call for text, call in EXAMPLES if text == said)
    original_request = dict(original["request"])
    messages = brain()._messages(said)
    user_index = next(i for i, message in enumerate(messages[:-1])
                      if message["role"] == "user"
                      and json.loads(message["content"]) == {"heard": said})
    emitted = json.loads(messages[user_index + 1]["content"])
    assert emitted["request"] == original_request
    assert original["request"] == original_request


@pytest.mark.parametrize(("said", "example"), [
    (said, example) for said, example in EXAMPLES
    if example.get("request", {}).get("status") == "ready"
])
def test_every_explicit_ready_example_resolves_to_its_source_span(said, example):
    resolved = _resolve_request_span(example["request"], said)
    assert resolved[0] == "ready"
    assert resolved[3].casefold() == example["request"]["text"].casefold()


def test_structured_malformed_negation_metadata_fails_closed():
    intent = brain().to_intent({"tool": "pause", "arguments": {},
                                "request": {"status": "no_command", "text": "pause"}},
                               "Don't pause")
    assert intent.action is Action.UNKNOWN
    assert intent.matched_rule == "brain_request_invalid"


def test_legacy_direct_to_intent_call_keeps_full_transcript_contract():
    assert brain().to_intent(call("pause"), "pause").action is Action.PAUSE


@pytest.mark.parametrize("tool,args,heard,action,query", [
    ("open_app", {"name": "Notepad"}, "Could you pull up Notepad", Action.OPEN_APP, "Notepad"),
    ("open_app", {"name": "Notepad"}, "Can you get Notepad running", Action.OPEN_APP, "Notepad"),
    ("search_browser", {"query": "local Python logging", "browser": "Opera"},
     "Look up local Python logging in Opera", Action.SEARCH_BROWSER, "local Python logging"),
    ("search_browser", {"query": "cats and dogs?", "browser": "Opera"},
     "Could you open a tab in Opera and search for cats and dogs?", Action.SEARCH_BROWSER, "cats and dogs?"),
    ("search_browser", {"query": "cats and dogs", "browser": "Opera"},
     "Look up cats and dogs in Opera.", Action.SEARCH_BROWSER, "cats and dogs"),
    ("show_track", {"query": "Us and Them"},
     "Could you show song Us and Them on Spotify.", Action.SHOW_TRACK, "Us and Them"),
    ("show_album", {"query": "Blue Lights by Mira Sol"},
     "I'd like to see the album Blue Lights by Mira Sol", Action.SHOW_ALBUM, "Blue Lights by Mira Sol"),
    ("show_track", {"query": "Red by Seedhe Maut"},
     "Could you show me the song Red by Seedhe Maut", Action.SHOW_TRACK, "Red by Seedhe Maut"),
    ("restart_album", {}, "Restart the current album", Action.RESTART_ALBUM, ""),
    ("start_jam", {}, "Start a Spotify Jam and show the QR code", Action.START_JAM, ""),
    ("start_jam", {}, "Show the Jam QR code", Action.START_JAM, ""),
    ("start_jam", {}, "Show the QR code of the Spotify Jam", Action.START_JAM, ""),
    ("start_jam", {}, "Start a Jam on Spotify and show the QR code", Action.START_JAM, ""),
    ("start_jam", {}, "Show the Jam QR code in Spotify", Action.START_JAM, ""),
    ("start_jam", {}, "Start a Jam and show the QR of the Jam on screen", Action.START_JAM, ""),
])
def test_new_tools_ground_one_complete_request_without_entity_rewrite(tool, args, heard, action, query):
    result = brain().to_intent(call(tool, **args), heard)
    assert result.action is action and result.query == query and result.source == "llm"


@pytest.mark.parametrize("tool,args,heard", [
    ("open_app", {"name": "Spotify"}, "Open Notepad"),
    ("open_app", {"name": "Notepad", "path": "evil.exe"}, "Open Notepad"),
    ("open_app", {"name": "Notepad"}, "He said open Notepad"),
    ("open_app", {"name": "Notepad"}, "Don't open Notepad"),
    ("open_app", {"name": "Notepad"}, "Open Notepad and delete files"),
    ("search_browser", {"query": "Python", "browser": "Opera"}, "Search Python logging in Opera"),
    ("search_browser", {"query": "Python", "browser": "Opera"}, "Search Python in Chrome"),
    ("search_browser", {"query": "invented query", "browser": "Opera"},
     "Open Opera and search for cats and dogs?"),
    ("search_browser", {"query": "cats", "browser": "Opera"},
     "Open Opera and search cats and please open Notepad"),
    ("show_track", {"query": "Us and Them"},
     "Show song Us and Them and could you play Red on Spotify."),
    ("show_track", {"query": "Red"}, "Show song Red by Seedhe Maut"),
    ("show_album", {"query": "Red"}, "Show song Red"),
    ("play_song", {"title": "Red", "artist": ""}, "Show song Red"),
    ("restart_track", {}, "Restart the current album"),
    ("restart_album", {}, "Restart the current song"),
    ("start_jam", {}, "Don't start a Jam"),
    ("start_jam", {}, "He said start a Jam"),
    ("start_jam", {}, "Start a Jam and pause"),
    ("start_jam", {}, "Start a Jam in Spotify and then pause"),
    ("start_jam", {"show": True}, "Start a Jam"),
    ("play_song", {"title": "Jam", "artist": "Michael Jackson"}, "Start a Jam"),
])
def test_new_tool_arguments_and_kind_cannot_invent_or_omit_authority(tool, args, heard):
    with pytest.raises(ValueError):
        brain().to_intent(call(tool, **args), heard)


@pytest.mark.parametrize("heard", ["Open Notepad", "Search logs in Opera", "Show album Blue Lights", "Restart this album", "Start a Jam"])
def test_canonical_capability_uses_no_model(heard):
    model = brain()
    model._chat = Mock(side_effect=AssertionError("Canonical capability must be fast"))
    assert decide(heard, model).action is not Action.UNKNOWN
    model._chat.assert_not_called()


def test_explicit_album_tool_keeps_heard_entity_and_volume():
    intent = brain().to_intent(call("play_album", title="Keychain Lalu", artist="Arpit Bala",
                                   volume_after_play=20),
                               "play album keychain lalu by arpit bala and set volume to 20")
    assert (intent.action, intent.query, intent.volume_after_play, intent.radio) == (
        Action.PLAY_ALBUM, "keychain lalu by arpit bala", 20, False)


@pytest.mark.parametrize("tool,args,heard", [
    ("play_album", {"title": "Keychain Lalu", "artist": "Arpit Bala"}, "play keychain lalu by arpit bala"),
    ("play_album", {"title": "Keychain Lalu", "artist": "Other"}, "play album keychain lalu by arpit bala"),
    ("play_album", {"title": "Keychain Lalu", "artist": ""}, "play album keychain lalu by arpit bala"),
    ("play_album", {"title": "Keychain", "artist": ""}, "play album keychain lalu"),
    ("play_song", {"title": "Keychain Lalu", "artist": "Arpit Bala"}, "play album keychain lalu by arpit bala"),
    ("play_artist", {"artist": "Arpit Bala"}, "play album keychain lalu by arpit bala"),
])
def test_album_grounding_rejects_invention_omission_and_category_change(tool, args, heard):
    with pytest.raises(ValueError):
        brain().to_intent(call(tool, **args), heard)


def test_unavailable_brain_preserves_explicit_album_action():
    assert decide("play album keychain lalu by arpit bala", None).action is Action.PLAY_ALBUM


def test_novel_request_phrasing_uses_source_album_cue_without_verb_aliases():
    intent = brain().to_intent(call("play_album", title="Keychain Laalu", artist="Arpit Bala"),
                               "I'd like to hear the album Keychain Laalu by Arpit Bala")
    assert intent.action is Action.PLAY_ALBUM
    assert intent.query == "keychain laalu by arpit bala"


@pytest.mark.parametrize("heard", ["I'd like to hear Keychain Laalu by Arpit Bala",
    "queue album Keychain Laalu by Arpit Bala", "play the song called Album by Arpit Bala"])
def test_album_tool_needs_type_source_and_excludes_queue_or_explicit_track(heard):
    with pytest.raises(ValueError):
        brain().to_intent(call("play_album", title="Keychain Laalu", artist="Arpit Bala"), heard)


@pytest.mark.parametrize("heard,explicit", [
    ("play the song Keychain Lalu by Arpit Bala", True),
    ("I'd like to hear the track Keychain Lalu by Arpit Bala", True),
    ("change the track to Keychain Lalu by Arpit Bala", True),
    ("play Keychain Lalu by Arpit Bala", False),
])
def test_brain_play_song_retains_source_track_type(heard, explicit):
    intent = brain().to_intent(call("play_song", title="Keychain Lalu", artist="Arpit Bala"), heard)
    assert intent.explicit_track is explicit


@pytest.mark.parametrize("heard", ["do not play the album Blue Lights by Mira Sol",
    "don't play the album Blue Lights by Mira Sol", "she said play the album Blue Lights by Mira Sol",
    "I mentioned the album Blue Lights by Mira Sol", "I hear the album Blue Lights by Mira Sol in a cafe",
    "Yesterday I wanted to hear the album Blue Lights by Mira Sol",
    "I would like to hear about the album Blue Lights by Mira Sol",
    "play an interview about the album Blue Lights by Mira Sol"])
def test_album_entities_do_not_authorize_negated_reported_or_incidental_requests(heard):
    with pytest.raises(ValueError):
        brain().to_intent(call("play_album", title="Blue Lights", artist="Mira Sol"), heard)


@pytest.mark.parametrize("title", ["Don't Stop", "Not Today"])
def test_album_request_guard_does_not_treat_entity_words_as_negated_authority(title):
    intent = brain().to_intent(call("play_album", title=title, artist="Mira Sol"),
                               "play the album " + title + " by Mira Sol")
    assert intent.action is Action.PLAY_ALBUM
    assert intent.query == title.casefold() + " by mira sol"


@pytest.mark.parametrize("prefix", ["just play the", "please, I'd like to hear the",
    "I want to listen to the", "can you listen to the", "only play the",
    "I would like to hear the", "I'd like to listen to the"])
def test_album_positive_request_framing_is_grounded_without_parser_verb_aliases(prefix):
    intent = brain().to_intent(call("play_album", title="Blue Lights", artist="Mira Sol"),
                               prefix + " album Blue Lights by Mira Sol")
    assert intent.action is Action.PLAY_ALBUM


def test_canonical_title_before_album_keeps_parser_grounding_policy():
    intent = brain().to_intent(call("play_album", title="Blue Lights", artist="Mira Sol"),
                               "play Blue Lights album by Mira Sol")
    assert intent.action is Action.PLAY_ALBUM
    assert intent.query == "blue lights by mira sol"


def test_heard_name_repaired_to_known_artist():
    intent = brain().to_intent(call("play_song", title="Madira", artist="Seedhe Maut"),
                               "play madira by see the mode")
    assert (intent.action, intent.query) == (Action.PLAY_TRACK, "madira by seedhe maut")


def test_known_artist_that_was_not_said_is_rejected():
    with pytest.raises(ValueError):
        brain().to_intent(call("play_song", title="Madira", artist="Aerosmith"),
                          "play madira by see the mode")


def test_invented_title_is_rejected():
    with pytest.raises(ValueError):
        brain().to_intent(call("play_song", title="Bohemian Rhapsody", artist=""), "play demons")


@pytest.mark.parametrize("title,heard", [
    ("Blue Red", "play red blue"),
    ("Red Red", "play red blue"),
])
def test_title_words_must_be_contiguous_and_in_order(title, heard):
    with pytest.raises(ValueError, match="Title"):
        brain().to_intent(call("play_song", title=title, artist=""), heard)


def test_song_title_cannot_supply_an_unspoken_artist():
    with pytest.raises(ValueError, match="Artist"):
        brain().to_intent(call("play_song", title="Seedhe Maut", artist="Seedhe Maut"),
                          "play seedhe maut")


def test_artist_repair_uses_only_explicit_artist_suffix():
    intent = brain().to_intent(call("play_song", title="Madira", artist="Seedhe Maut"),
                               "play madira by see the mode")
    assert intent.query == "madira by seedhe maut"
    with pytest.raises(ValueError, match="Artist"):
        brain().to_intent(call("play_song", title="Red", artist="Seedhe Maut"),
                          "play red by aerosmith")


@pytest.mark.parametrize("level", [150, -5])
def test_volume_out_of_range_is_rejected(level):
    with pytest.raises(ValueError):
        brain().to_intent(call("set_volume", level=level), "set volume to whatever")


def test_ignore_maps_to_unknown():
    assert brain().to_intent(call("ignore", reason="chat"), "how are you").action is Action.UNKNOWN


def test_unspecified_volume_change_uses_default_step():
    intent = brain().to_intent(call("change_volume", direction="down", amount=0, unit="points"),
                               "turn it down a bit")
    assert (intent.action, intent.query, intent.value) == (Action.VOLUME, "down", None)


def test_play_artist():
    intent = brain().to_intent(call("play_artist", artist="Seedhe Maut"), "put on some seedhe mod")
    assert (intent.action, intent.query) == (Action.PLAY_ARTIST, "Seedhe Maut")


def test_plain_transport_never_waits_for_the_model():
    b = Mock()
    assert decide("pause", b).action is Action.PAUSE
    assert decide("volume 40", b).value == 40
    b.decide.assert_not_called()


def test_model_failure_falls_back_to_rules():
    b = Mock()
    b.decide.side_effect = TimeoutError("slow")
    intent = decide("play red", b)
    assert (intent.action, intent.query) == (Action.PLAY_TRACK, "red")
    b.decide.assert_called_once()


@pytest.mark.parametrize("heard,expected", [
    ("play red by seedhe maut", Action.PLAY_TRACK),
    ("play album keychain lalu by arpit bala", Action.PLAY_ALBUM),
    ("play the playlist Focus", Action.PLAY_PLAYLIST),
])
def test_explicit_parser_grounded_music_requests_skip_brain_model(heard, expected, monkeypatch):
    import assistant.brain as brain_module

    model = brain()
    model._chat = Mock(side_effect=AssertionError("Explicit rule request must be fast"))
    annotated = Mock()
    monkeypatch.setattr(brain_module, "annotate_event", annotated)

    result = decide(heard, model)

    assert result.action is expected and result.source == "rule"
    model._chat.assert_not_called()
    annotated.assert_called_with(brain_outcome="rules_fast_explicit_playback")


@pytest.mark.parametrize("heard", [
    "He said play red by seedhe maut",
    "Do not play red by seedhe maut",
    "Tell Codex play red by seedhe maut",
])
def test_explicit_playback_fast_path_rejects_nonpresent_or_global_text(heard):
    model = brain()
    model._chat = Mock(return_value=call("ignore", reason="not a direct music request"))

    assert decide(heard, model).action is Action.UNKNOWN
    model._chat.assert_called_once()


@pytest.mark.parametrize("heard", [
    "playing Red by Seedhe Maut",
    "played Red by Seedhe Maut",
    "plays Red by Seedhe Maut",
])
def test_inflected_play_narration_reaches_brain_not_explicit_fast_path(heard, monkeypatch):
    import assistant.brain as brain_module

    model = brain()
    model._chat = Mock(return_value=call("ignore", reason="not a direct music request"))
    annotated = Mock()
    monkeypatch.setattr(brain_module, "annotate_event", annotated)

    result = decide(heard, model)

    assert result.action is Action.UNKNOWN
    model._chat.assert_called_once()
    assert not any(entry.kwargs.get("brain_outcome") == "rules_fast_explicit_playback"
                   for entry in annotated.call_args_list)


def test_explicit_playback_fast_path_keeps_dynamic_compounds_on_brain():
    model = brain()
    model._chat = Mock(return_value=call("play_song", title="Red", artist="Seedhe Maut",
                                        volume_after_play=20))

    result = decide("play red by seedhe maut and set volume to 20", model)

    assert result.action is Action.PLAY_TRACK and result.volume_after_play == 20
    model._chat.assert_called_once()


def test_jam_in_track_title_is_not_a_jam_control_shortcut():
    model = brain()
    model._chat = Mock(return_value=call("play_song", title="Jam", artist="Michael Jackson"))

    result = decide("play Jam by Michael Jackson", model)

    assert result.action is Action.PLAY_TRACK
    assert result.query == "jam by michael jackson"


def test_warmup_timeout_does_not_leak_to_concurrent_command(monkeypatch):
    import assistant.brain as brain_module

    b = brain()
    entered = threading.Event()
    release = threading.Event()
    timeouts = []

    def fake_chat(messages, keep_alive=-1, num_predict=120, timeout=None):
        timeouts.append((num_predict, timeout if timeout is not None else b.timeout))
        if timeout == 120.0:
            entered.set()
            assert release.wait(2)
        return call("ignore", reason="not a request")

    monkeypatch.setattr(b, "_chat", fake_chat)
    worker = threading.Thread(target=b.warm_up)
    worker.start()
    assert entered.wait(2)
    try:
        assert brain_module.decide("how are you", b).action is Action.UNKNOWN
    finally:
        release.set()
        worker.join(2)

    assert not worker.is_alive()
    assert timeouts == [(120, 120.0)]
    assert b.warming is False
    assert b.timeout == 4.0


def test_warmup_uses_complete_production_messages_and_does_not_change_command_timeout():
    b = Brain("test-model", timeout=3.0, context=lambda: CONTEXT, warmup_pending=True)
    expected = b._messages("pause")
    b._chat = Mock(return_value=call("pause"))
    b.warm_up()
    b._chat.assert_called_once_with(expected, num_predict=120, timeout=120.0)
    assert b.timeout == 3.0 and b.warming is False
    assert expected[0]["role"] == "system" and "play_album" in expected[0]["content"]
    assert any("Blue Lights" in message["content"] for message in expected)
    assert len(expected) > 3


def test_pending_startup_warmup_returns_rules_without_queueing_model(monkeypatch):
    import assistant.brain as brain_module
    b = Brain("test-model", timeout=3.0, warmup_pending=True)
    b._chat = Mock(side_effect=AssertionError("No command inference during warmup"))
    annotated = Mock()
    monkeypatch.setattr(brain_module, "annotate_event", annotated)
    assert b.decide("play the song Blue Lights by Mira Sol").explicit_track is True
    b._chat.assert_not_called()
    annotated.assert_called_with(brain_outcome="warming_rules")


def test_low_level_command_cannot_queue_http_while_warmup_is_pending():
    from assistant.brain import BrainWarming
    b = Brain("test-model", warmup_pending=True)
    b._http = Mock()
    with pytest.raises(BrainWarming):
        b._chat(b._messages("pause"))
    b._http.open.assert_not_called()


def test_warmup_gate_race_during_message_construction_returns_rules(monkeypatch):
    b = brain()
    original = b._messages
    def messages(transcript):
        result = original(transcript)
        b._warmup_pending = True
        return result
    monkeypatch.setattr(b, "_messages", messages)
    b._http = Mock()
    assert b.decide("play Blue Lights by Mira Sol").action is Action.PLAY_TRACK
    b._http.open.assert_not_called()


def test_failed_warmup_releases_gate_and_model_retries_after_existing_cooldown(monkeypatch):
    import assistant.brain as brain_module
    clock = [0.0]
    monkeypatch.setattr(brain_module.time, "monotonic", lambda: clock[0])
    b = Brain("test-model", warmup_pending=True)
    b._chat = Mock(side_effect=[ConnectionError("offline"),
        call("play_song", title="Madira", artist="Seedhe Maut")])
    with pytest.raises(ConnectionError):
        b.warm_up()
    assert b.warming is False
    assert b.decide("would you play Madira by Seedhe Maut").action is Action.UNKNOWN
    assert b._chat.call_count == 1
    clock[0] = 31.0
    assert b.decide("would you play Madira by Seedhe Maut").source == "llm"
    assert b._chat.call_count == 2


def test_shared_runtime_owns_brain_http_request_policy_and_lock_shape():
    from assistant.model_runtime import LocalModelRuntime

    runtime = LocalModelRuntime("test-model", num_ctx=8192, keep_alive=-1)
    b = Brain("test-model", runtime=runtime)
    captured = {}

    def envelope(body, timeout):
        captured.update(body=body, timeout=timeout)
        return {"message": {"content": json.dumps(call("ignore", reason="test"))}}

    b._request_envelope = envelope
    assert b._chat(b._messages("would you play Red"))["tool"] == "ignore"
    assert captured["body"]["model"] == "test-model"
    assert captured["body"]["options"]["num_ctx"] == 8192
    assert captured["body"]["keep_alive"] == -1
    assert captured["timeout"] == b.timeout
    assert runtime.state == "ready"


def test_shared_runtime_failed_warmup_blocks_until_cooldown(monkeypatch):
    import assistant.brain as brain_module
    from assistant.model_runtime import LocalModelRuntime

    now = [0.0]
    monkeypatch.setattr(brain_module.time, "monotonic", lambda: now[0])
    runtime = LocalModelRuntime("test-model", cooldown_seconds=30,
                                clock=lambda: now[0])
    b = Brain("test-model", runtime=runtime, warmup_pending=True)
    b._request_envelope = Mock(side_effect=[ConnectionError("offline"), {
        "message": {"content": json.dumps(call("play_song", title="Madira",
                                                artist="Seedhe Maut"))}}])
    with pytest.raises(ConnectionError):
        b.warm_up()
    assert runtime.state == "degraded"
    assert b.warming is False
    assert b.decide("would you play Madira by Seedhe Maut").action is Action.UNKNOWN
    b._request_envelope.assert_called_once()

    now[0] = 31.0
    assert b.decide("would you play Madira by Seedhe Maut").source == "llm"
    assert b._request_envelope.call_count == 2
    assert runtime.state == "ready"


def test_startup_pending_marks_shared_runtime_warming_synchronously():
    from assistant.model_runtime import LocalModelRuntime

    runtime = LocalModelRuntime("test-model")
    Brain("test-model", runtime=runtime, warmup_pending=True)
    assert runtime.state == "warming"
    assert runtime.can_attempt() is False


def test_connection_failure_is_suppressed_during_cooldown():
    b = brain()
    b._chat = Mock(side_effect=ConnectionError("Ollama is down"))

    for _ in range(3):
        intent = decide("would you play red by seedhe maut", b)
        assert intent.action is Action.UNKNOWN

    b._chat.assert_called_once()


def test_brain_retries_after_unavailable_cooldown(monkeypatch):
    import assistant.brain as brain_module

    now = [10.0]
    monkeypatch.setattr(brain_module.time, "monotonic", lambda: now[0])
    b = brain()
    b._chat = Mock(side_effect=[ConnectionError("Ollama is down"),
                                call("ignore", reason="not a request")])

    assert decide("would you play red by seedhe maut", b).action is Action.UNKNOWN
    assert decide("would you play blue by seedhe maut", b).action is Action.UNKNOWN
    b._chat.assert_called_once()

    now[0] += 31
    assert decide("would you play green by seedhe maut", b).action is Action.UNKNOWN
    assert b._chat.call_count == 2


def test_explicit_model_refusal_stays_unknown_without_rule_fallback():
    b = brain()
    b._chat = Mock(return_value=call("ignore", reason="unclear"))

    intent = decide("would you play red by seedhe maut", b)

    assert intent.action is Action.UNKNOWN
    assert intent.source == "llm"
    b._chat.assert_called_once()


def test_empty_utterance_does_not_call_model():
    b = brain()
    b._chat = Mock()

    assert b.decide("   ").action is Action.UNKNOWN
    b._chat.assert_not_called()


def test_evaluator_marks_same_action_wrong_target_mismatch(monkeypatch, capsys):
    from assistant.experimental import brain_eval
    from assistant.intent import Intent

    monkeypatch.setattr(brain_eval, "VOCAB", Mock(read_text=Mock(return_value='{"artists": []}')))
    monkeypatch.setattr(brain_eval, "Brain", Mock())
    monkeypatch.setattr(brain_eval.sys, "argv", ["brain_eval", "test-model"])
    monkeypatch.setattr(brain_eval, "CASES", [("play madira", Action.PLAY_TRACK, ["madira"])])
    monkeypatch.setattr(brain_eval, "decide", lambda *_: Intent(Action.PLAY_TRACK, query="demons"))

    brain_eval.main()

    output = capsys.readouterr().out
    assert "TARGET" in output
    assert "1 actionable intent mismatches" in output


@pytest.mark.parametrize("tool,args,heard,query", [
    ("play_song", {"title": "Madira", "artist": "Seedhe Maut"},
     "play madira by see the mode and set volume to 20", "madira by seedhe maut"),
    ("play_artist", {"artist": "Seedhe Maut"},
     "play some seedhe mod then volume 20", "Seedhe Maut"),
    ("play_playlist", {"name": "Focus", "mine": True},
     "play my focus playlist and turn it down to twenty", "focus"),
])
def test_compound_play_preserves_requested_absolute_volume(tool, args, heard, query):
    for include_argument in (False, True):
        arguments = dict(args)
        if include_argument:
            arguments["volume_after_play"] = 20
        intent = brain().to_intent(call(tool, **arguments), heard)
        assert intent.query == query
        assert intent.volume_after_play == 20


def test_play_volume_tail_is_not_part_of_artist_grounding():
    intent = brain().to_intent(call("play_song", title="Madira", artist="Seedhe Maut"),
                               "play madira by see the mode at full volume")
    assert intent.query == "madira by seedhe maut"
    assert intent.volume_after_play == 100


@pytest.mark.parametrize("heard,proposed", [
    ("play madira", 20),
    ("play madira and volume 20", 30),
    ("play madira and volume 20", True),
    ("play madira and volume 20", 150),
])
def test_invented_post_play_volume_is_rejected(heard, proposed):
    with pytest.raises(ValueError, match="volume"):
        brain().to_intent(call("play_song", title="Madira", artist="", volume_after_play=proposed), heard)


@pytest.mark.parametrize("tail", ["turn it down", "make it louder", "reduce volume by 20 percent",
                                   "set volume to 150", "pause"])
def test_unsupported_play_compound_refuses_before_model_and_fallback(tail):
    b = Mock()
    intent = decide("play madira and " + tail, b)
    assert intent.action is Action.UNKNOWN
    b.decide.assert_not_called()


def test_compound_play_volume_survives_model_failure():
    b = Mock()
    b.decide.side_effect = TimeoutError("slow")
    intent = decide("play madira and set volume to 20", b)
    assert intent.action is Action.PLAY_TRACK
    assert intent.query == "madira"
    assert intent.volume_after_play == 20


def test_compound_play_cannot_be_reduced_to_only_volume():
    b = brain()
    b._chat = Mock(return_value=call("set_volume", level=20))
    intent = decide("play madira and set volume to 20", b)
    assert intent.action is Action.PLAY_TRACK
    assert intent.query == "madira"
    assert intent.volume_after_play == 20


@pytest.mark.parametrize("heard", [
    "pause and lower the volume",
    "pause and set volume to 30",
])
def test_unsupported_transport_compounds_fail_before_fastpath_or_fallback(heard):
    model = Mock()
    intent = decide(heard, model)
    assert intent.action is Action.UNKNOWN
    assert intent.matched_rule == "unsupported_compound"
    model.decide.assert_not_called()


def test_next_then_open_does_not_fast_execute_as_next():
    model = Mock()
    model.decide.return_value = brain().to_intent(
        call("next_track"), "next and open Notepad")
    intent = decide("next and open Notepad", model)
    assert intent.action is Action.UNKNOWN


def test_noisy_supported_compound_reaches_brain_for_span_selection():
    heard = "Thanks. Play Red by Seedhe Maut and set volume to 30. Thanks."
    selected = "Play Red by Seedhe Maut and set volume to 30"
    model = brain()
    model._chat = Mock(return_value={
        "tool": "play_song",
        "arguments": {"title": "Red", "artist": "Seedhe Maut", "volume_after_play": 30},
        "request": {"status": "ready", "text": selected}})
    intent = decide(heard, model)
    assert intent.action is Action.PLAY_TRACK
    assert intent.volume_after_play == 30
    model._chat.assert_called_once()


@pytest.mark.parametrize(("heard", "raises"), [
    ("pause and lower the volume", False),
    ("pause and set volume to 30", True),
    ("next and open Notepad", False),
])
def test_brain_rejects_unsupported_compound_even_if_model_selects_transport(heard, raises):
    tool = "next_track" if heard.startswith("next") else "pause"
    if raises:
        with pytest.raises(ValueError, match="Compound playback requires a play tool"):
            brain().to_intent(call(tool), heard)
        return
    intent = brain().to_intent(call(tool), heard)
    assert intent.action is Action.UNKNOWN
    assert intent.matched_rule == "brain_unsupported_compound"


def test_valid_jam_and_qr_request_is_not_blocked_as_unsupported_compound():
    model = Mock()
    intent = decide("Start a Jam and show the QR of the Jam on screen", model)
    assert intent.action is Action.START_JAM
    model.decide.assert_not_called()


def test_song_title_conjunction_is_not_mistaken_for_compound_command():
    intent = brain().to_intent(call("play_song", title="Roses and Rain", artist=""),
                               "play Roses and Rain")
    assert intent.action is Action.PLAY_TRACK
    assert intent.query == "roses and rain"


@pytest.mark.parametrize("tool,args", [
    ("pause", {}), ("resume", {}), ("next_track", {}),
    ("previous_track", {}), ("restart_track", {}),
    ("set_volume", {"level": 100}),
    ("change_volume", {"direction": "up", "amount": 100, "unit": "percent"}),
])
def test_model_control_tool_on_chatter_is_refused(tool, args):
    assert brain().to_intent(call(tool, **args), "what is the weather tomorrow").action is Action.UNKNOWN


@pytest.mark.parametrize("heard,direction", [
    ("make it louder", "up"), ("it's too loud", "down"), ("turn it down a bit", "down"),
])
def test_natural_volume_direction_is_grounded(heard, direction):
    intent = brain().to_intent(call("change_volume", direction=direction, amount=0, unit="points"), heard)
    assert intent.action is Action.VOLUME
    assert intent.query == direction


def test_natural_restart_is_grounded_without_turning_title_into_control():
    assert brain().to_intent(call("restart_track"), "play that again from the start").action is Action.RESTART
    assert brain().to_intent(call("restart_track"), "what is the weather tomorrow").action is Action.UNKNOWN
    assert brain().to_intent(call("restart_track"), "play beginning").action is Action.UNKNOWN
    assert brain().to_intent(call("restart_track"), "play again").action is Action.UNKNOWN


@pytest.mark.parametrize("tool,args,heard", [
    ("set_volume", {"level": 100}, "set volume to 20"),
    ("change_volume", {"direction": "up", "amount": 0, "unit": "points"}, "it's too loud"),
    ("change_volume", {"direction": "down", "amount": 80, "unit": "points"}, "make it quieter"),
])
def test_model_cannot_invent_volume_level_or_direction(tool, args, heard):
    assert brain().to_intent(call(tool, **args), heard).action is Action.UNKNOWN


def test_similarly_qualifying_known_artists_defer_to_catalogue(monkeypatch):
    import assistant.brain as brain_module
    b = Brain("test-model", context=lambda: {"artists": ["Artist One", "Artist Two"]})
    monkeypatch.setattr(brain_module, "sound_score", lambda *_: 0.75)
    with pytest.raises(ValueError, match="Ambiguous"):
        b.to_intent(call("play_song", title="Red", artist="Artist One"), "play red by ardis won")
    b._chat = Mock(return_value=call("play_song", title="Red", artist="Artist One"))
    # The brain must not pick either artist; the words go to Spotify as heard,
    # where only an artist who really has that title can match.
    intent = decide("play red by ardis won", b)
    assert intent.action is Action.PLAY_TRACK
    assert intent.query == "red by ardis won"


def test_evaluator_requires_compound_volume_slot(monkeypatch, capsys):
    from assistant.experimental import brain_eval
    from assistant.intent import Intent

    monkeypatch.setattr(brain_eval, "VOCAB", Mock(read_text=Mock(return_value='{"artists": []}')))
    monkeypatch.setattr(brain_eval, "Brain", Mock())
    monkeypatch.setattr(brain_eval.sys, "argv", ["brain_eval", "test-model"])
    monkeypatch.setattr(brain_eval, "CASES", [
        ("play madira and volume 20", Action.PLAY_TRACK, ["madira"], {"volume_after_play": 20})])
    monkeypatch.setattr(brain_eval, "decide", lambda *_: Intent(Action.PLAY_TRACK, query="madira"))
    brain_eval.main()
    assert "1 actionable intent mismatches" in capsys.readouterr().out


@pytest.mark.parametrize("heard", ["play my new playlist", "play my playlist new",
                                   "play my new playlist and set volume to 20"])
def test_uncached_personal_playlist_preserves_library_constraint(heard):
    intent = brain().to_intent(call("play_playlist", name="new", mine=False), heard)
    assert intent.action is Action.PLAY_PLAYLIST
    assert intent.query == "new"
    assert intent.library_only is True


def test_playlist_request_words_cannot_become_the_playlist_name():
    b = brain()
    with pytest.raises(ValueError, match="Playlist"):
        b.to_intent(call("play_playlist", name="my", mine=True), "play my new playlist")
    b._chat = Mock(return_value=call("play_playlist", name="my", mine=True))
    intent = decide("play my new playlist", b)
    assert intent.action is Action.PLAY_PLAYLIST
    assert intent.query == "new"
    assert intent.library_only is True


def test_uncached_playlist_name_cannot_drop_entity_words():
    b = brain()
    with pytest.raises(ValueError, match="omitted"):
        b.to_intent(call("play_playlist", name="evening", mine=True),
                    "play my evening focus playlist")
    intent = b.to_intent(call("play_playlist", name="evening focus", mine=False),
                         "play my evening focus playlist")
    assert intent.query == "evening focus"
    assert intent.library_only is True


def test_cached_playlist_name_cannot_drop_literal_entity_words():
    with pytest.raises(ValueError, match="omitted"):
        brain().to_intent(call("play_playlist", name="Focus", mine=True),
                          "play my evening focus playlist")
