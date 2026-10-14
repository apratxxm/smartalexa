import pytest

from assistant.intent import Action, parse


@pytest.mark.parametrize("text,value", [("increase volume to 50%", 50), ("raise volume to fifty", 50)])
def test_increase_to_is_absolute(text, value):
    intent = parse(text)
    assert (intent.action, intent.value, intent.query, intent.volume_mode) == (
        Action.VOLUME, value, "", "absolute"
    )


@pytest.mark.parametrize("text,value", [("decrease volume by 50%", 50), ("lower volume by fifty percent", 50)])
def test_decrease_by_is_relative_percent(text, value):
    intent = parse(text)
    assert (intent.action, intent.value, intent.query, intent.volume_mode) == (
        Action.VOLUME, value, "down", "relative_percent"
    )


def test_volume_points_are_relative_points():
    intent = parse("turn volume down by 10 points")
    assert (intent.action, intent.value, intent.query, intent.volume_mode) == (
        Action.VOLUME, 10, "down", "relative_points"
    )


@pytest.mark.parametrize("text", ["volume 101", "volume -1", "decrease volume by 101%", "turn volume down by ten bananas"])
def test_invalid_volume_amounts_are_unknown(text):
    assert parse(text).action is Action.UNKNOWN


def test_title_words_do_not_trigger_volume():
    intent = parse("play decrease volume by 50")
    assert intent.action is Action.PLAY_TRACK
    assert intent.query == "decrease volume by 50"


def test_retry_interjection_does_not_hide_volume_command():
    intent = parse("Oh! Set volume to 45%")
    assert (intent.action, intent.value, intent.volume_mode) == (
        Action.VOLUME, 45, "absolute")
