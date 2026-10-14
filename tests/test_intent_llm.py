import pytest

from assistant.intent import Action
from assistant.intent_llm import LocalIntentParser, validate_result, validate_grounding


def row(**kwargs):
    return dict(action="PAUSE", query="", value=None, volume_mode="absolute", library_only=False, **kwargs)


def test_transport_schema():
    assert validate_result(row()).action is Action.PAUSE


def test_content_names_must_come_from_transcript():
    data = row(); data.update(action="PLAY_TRACK", query="demons by imagine dragons")
    with pytest.raises(ValueError, match="invented"):
        validate_grounding(validate_result(data), "play demons")
    assert validate_grounding(validate_result(data), "please play demons by imagine dragons").query == data['query']


def test_playlist_scope_cannot_be_invented():
    data = row(); data.update(action="PLAY_PLAYLIST", query="focus", library_only=True)
    with pytest.raises(ValueError, match="scope"):
        validate_grounding(validate_result(data), "play the focus playlist")


@pytest.mark.parametrize("field,value", [
    ("action", "SHELL"), ("value", True), ("value", 101),
    ("query", "spotify:track:invented"), ("library_only", True),
    ("volume_mode", "relative_percent"),
])
def test_invalid_combinations_rejected(field, value):
    data = row(); data[field] = value
    with pytest.raises(ValueError): validate_result(data)


def test_volume_schema():
    data = row(); data.update(action="VOLUME", query="down", value=50, volume_mode="relative_percent")
    result = validate_result(data)
    assert result.query == "down" and result.value == 50 and result.source == "llm"


def test_no_extra_fields():
    data = row(); data["url"] = "https://example.com"
    with pytest.raises(ValueError): validate_result(data)


@pytest.mark.parametrize("tag", ["model:cloud", "model:120b-cloud", "org/model"])
def test_cloud_tag_rejected(tag):
    with pytest.raises(ValueError): LocalIntentParser(tag)


def test_response_truncation_is_rejected():
    import io, json
    from unittest.mock import Mock
    parser = LocalIntentParser()
    parser._http = Mock()
    parser._http.open.return_value = io.BytesIO(json.dumps({
        "done": True, "done_reason": "length", "message": {"content": json.dumps(row())}
    }).encode())
    with pytest.raises(ValueError, match="Incomplete"): parser.parse("pause")


def test_local_request_is_schema_bound():
    import io, json
    from unittest.mock import Mock
    parser = LocalIntentParser()
    parser._http = Mock()
    parser._http.open.return_value = io.BytesIO(json.dumps({
        "done": True, "message": {"content": json.dumps(row())}
    }).encode())
    assert parser.parse("pause").action is Action.PAUSE
    request = parser._http.open.call_args.args[0]
    assert request.full_url == "http://127.0.0.1:11434/api/chat"
    payload = json.loads(request.data)
    assert payload["think"] is False and payload["stream"] is False
    assert all(branch["additionalProperties"] is False for branch in payload["format"]["anyOf"])
