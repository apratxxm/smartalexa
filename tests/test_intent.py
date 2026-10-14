"""Intent table tests, written against messy real ASR output rather than clean text."""
import pytest

from assistant.intent import Action, parse


@pytest.mark.parametrize("spoken", ["play album keychain lalu by arpit bala",
    "play the album keychain lalu by arpit bala", "start album keychain lalu by arpit bala",
    "put on keychain lalu album by arpit bala", "play keychain lalu by arpit bala album"])
def test_explicit_album_has_its_own_action(spoken):
    result = parse(spoken)
    assert result.action is Action.PLAY_ALBUM
    assert result.query == "keychain lalu by arpit bala"
    assert result.radio is False


def test_album_volume_is_separate_and_queue_stays_song():
    result = parse("play album keychain lalu by arpit bala at 20% volume")
    assert result.action is Action.PLAY_ALBUM
    assert result.volume_after_play == 20
    assert result.query == "keychain lalu by arpit bala"
    assert parse("queue keychain lalu by arpit bala").action is Action.QUEUE


@pytest.mark.parametrize("spoken,query", [
    ("play the song called keychain lalu by arpit bala", "keychain lalu by arpit bala"),
    ("play track named keychain lalu by arpit bala", "keychain lalu by arpit bala"),
    ("start the track named keychain lalu by arpit bala", "keychain lalu by arpit bala"),
    ("change the track to keychain lalu by arpit bala", "keychain lalu by arpit bala"),
    ("play Song 2 by Blur", "song 2 by blur"),
    ("play Track of My Tears", "track of my tears"),
    ("play the song of the sea", "the song of the sea"),
    ("start the song of the sea", "the song of the sea"),
])
def test_explicit_track_source_preserves_type_without_stripping_ordinary_titles(spoken, query):
    result = parse(spoken)
    assert result.action is Action.PLAY_TRACK
    assert result.explicit_track is True
    assert result.query == query


def test_unspecified_play_has_no_explicit_track_constraint():
    assert parse("play keychain lalu by arpit bala").explicit_track is False


@pytest.mark.parametrize('spoken,direction', [
    ('increase volume', 'up'),
    ('increase the volume', 'up'),
    ('decrease volume', 'down'),
    ('lower the volume', 'down'),
])
def test_bare_volume_change_uses_default_step(spoken, direction):
    result = parse(spoken)
    assert result.action is Action.VOLUME
    assert result.query == direction
    assert result.value is None


@pytest.mark.parametrize('spoken', [
    'volume 50 percent', 'set volume 50 percent',
    'sir volume 50 percent',
])
def test_spoken_percent_volume_is_absolute(spoken):
    result = parse(spoken)
    assert result.action is Action.VOLUME
    assert result.value == 50
    assert result.volume_mode == 'absolute'


@pytest.mark.parametrize('spoken', [
    'restart the song', 'restart this track', 'replay', 'start over',
])
def test_restart_is_distinct_from_previous(spoken):
    assert parse(spoken).action is Action.RESTART


def test_playback_volume_is_not_part_of_song_or_artist():
    result = parse('Play Red by Sidaimoth at 20% volume.')
    assert result.action is Action.PLAY_TRACK
    assert result.query == 'red by sidaimoth'
    assert result.volume_after_play == 20


@pytest.mark.parametrize('spoken', [
    'Change song to Champagne Poetry by Drake',
    'Switch the track to Champagne Poetry by Drake',
])
def test_named_change_song_preserves_artist(spoken):
    result = parse(spoken)
    assert result.action is Action.PLAY_TRACK
    assert result.query == 'champagne poetry by drake'


def test_change_song_without_destination_does_not_guess():
    assert parse('change song').action is Action.UNKNOWN


def test_invalid_playback_volume_does_not_play():
    assert parse('play red at 120% volume').action is Action.UNKNOWN


@pytest.mark.parametrize('text', ['do not pause', "don't pause", 'please do not stop',
                                 'never skip', 'pause then skip', 'pause and play demons',
                                 'I said pause', 'not paws'])
def test_negation_and_embedded_transport_do_not_execute(text):
    assert parse(text).action is Action.UNKNOWN


@pytest.mark.parametrize('title', ['louder', 'volume 2', 'turn it up', 'softer'])
def test_volume_words_inside_track_title_do_not_change_volume(title):
    result = parse('play ' + title)
    assert result.action is Action.PLAY_TRACK
    assert result.query == title


@pytest.mark.parametrize('spoken,value', [('fourteen', 14), ('forty five', 45),
    ('ninety-nine', 99), ('one hundred', 100), ('seven', 7)])
def test_spoken_volume_numbers_are_composed(spoken, value):
    result = parse('set volume to ' + spoken)
    assert result.action is Action.VOLUME
    assert result.value == value


@pytest.mark.parametrize('spoken', ['forty five hundred', '1000'])
def test_invalid_volume_number_is_not_silently_truncated(spoken):
    assert parse('volume ' + spoken).action is Action.UNKNOWN


@pytest.mark.parametrize('spoken,query,personal', [
    ('play my focus playlist', 'focus', True),
    ('play my playlist focus', 'focus', True),
    ('play focus playlist', 'focus', False),
    ('play playlist focus', 'focus', False),
    ('play mythology playlist', 'mythology', False),
    ('play by focus playlist', 'focus', True),
])
def test_playlist_ownership_survives_parsing(spoken, query, personal):
    result = parse(spoken)
    assert result.action is Action.PLAY_PLAYLIST
    assert result.query == query
    assert result.library_only is personal

TRANSPORT = [
    ("pause", Action.PAUSE),
    ("hey memoir pause", Action.PAUSE),
    ("stop", Action.PAUSE),
    ("shut up", Action.PAUSE),
    ("next", Action.NEXT),
    ("skip this one", Action.NEXT),
    ("go back", Action.PREV),
    ("previous song", Action.PREV),
    ("resume", Action.RESUME),
    ("continue", Action.RESUME),
]

RESUME_SHAPED = [
    ("play", Action.RESUME),
    ("play music", Action.RESUME),
    ("hey memoir play something", Action.RESUME),
    ("uh play anything", Action.RESUME),
]


@pytest.mark.parametrize("text,expected", TRANSPORT + RESUME_SHAPED)
def test_closed_vocabulary(text, expected):
    assert parse(text).action is expected


@pytest.mark.parametrize(
    "text,query",
    [
        ("play bohemian rhapsody", "bohemian rhapsody"),
        ("play uh bohemian rhapsody", "bohemian rhapsody"),
        ("hey memoir play sabrina carpenter", "sabrina carpenter"),
        ("play bohemian rhapsody on spotify", "bohemian rhapsody"),
        ("can you play espresso for me", "espresso"),
    ],
)
def test_play_track(text, query):
    intent = parse(text)
    assert intent.action is Action.PLAY_TRACK
    assert intent.query == query


@pytest.mark.parametrize(
    "text,query",
    [
        ("play my focus playlist", "focus"),
        ("play playlist deep focus", "deep focus"),
        ("play the playlist late night", "late night"),
        ("play my gym mix", "gym mix"),
    ],
)
def test_play_playlist(text, query):
    intent = parse(text)
    assert intent.action is Action.PLAY_PLAYLIST
    assert intent.query == query


@pytest.mark.parametrize(
    "text,value",
    [
        ("volume 40", 40),
        ("set volume to 65", 65),
        ("volume to fifty", 50),
        ("volume max", 100),
    ],
)
def test_volume_absolute(text, value):
    intent = parse(text)
    assert intent.action is Action.VOLUME
    assert intent.value == value


@pytest.mark.parametrize(
    "text,direction",
    [
        ("volume up", "up"),
        ("turn it up", "up"),
        ("louder", "up"),
        ("volume down", "down"),
        ("turn it down", "down"),
        ("quieter", "down"),
    ],
)
def test_volume_relative(text, direction):
    intent = parse(text)
    assert intent.action is Action.VOLUME
    assert intent.query == direction
    assert intent.value is None


class TestTrailingConversation:
    """The roommate problem: the parser, not the microphone, ends the command."""

    def test_query_truncated_at_conversational_marker(self):
        intent = parse("play something chill yeah no i told him tuesday")
        assert intent.action is Action.PLAY_TRACK
        assert intent.query == "something chill"

    def test_trailing_speech_after_track_name(self):
        intent = parse("play bohemian rhapsody actually wait he said monday")
        assert intent.query == "bohemian rhapsody"

    def test_trailing_speech_after_playlist(self):
        intent = parse("play my focus playlist anyway we should go")
        assert intent.action is Action.PLAY_PLAYLIST
        assert intent.query == "focus"

    def test_transport_ignores_trailing_speech_entirely(self):
        assert parse("pause yeah i'll be right there").action is Action.PAUSE


class TestUnknown:
    """UNKNOWN must be silent and logged, never a guess."""

    @pytest.mark.parametrize(
        "text", ["", "   ", "what time is it", "hey memoir", "the weather tomorrow"]
    )
    def test_unknown(self, text):
        assert parse(text).action is Action.UNKNOWN


class TestFuzzyTransport:
    """Real Moonshine misfires on single-word commands, collected from actual
    ASR output: "pause" -> "polls", "play" -> "pain"/"pay".

    Only ever applied to the tiny closed transport vocabulary -- never to song
    or playlist queries, where a near-miss would play the wrong thing.
    """

    @pytest.mark.parametrize(
        "heard,expected",
        [
            ("pay memoir polls", Action.PAUSE),
            ("hey memoir paws", Action.PAUSE),
            ("a memoir skips", Action.NEXT),
            ("hey memoir resumed", Action.RESUME),
        ],
    )
    def test_near_miss_still_resolves(self, heard, expected):
        assert parse(heard).action is expected

    @pytest.mark.parametrize(
        "heard", ["what time is it", "the weather tomorrow", "tell him i said hi"]
    )
    def test_unrelated_speech_stays_unknown(self, heard):
        """The cutoff must not turn ambient conversation into commands."""
        assert parse(heard).action is Action.UNKNOWN

    def test_long_utterance_is_not_fuzzy_matched(self):
        assert parse("i was talking to paul about the post office").action is Action.UNKNOWN


class TestRealMishearings:
    """Transcripts taken verbatim from telemetry.jsonl during live use.

    These are not hypothetical: an English-only STT reliably turns "my" into
    "by" and mangles short or non-English playlist names, and the first version
    of the rule table sent them to a track search instead of the playlist.
    """

    @pytest.mark.parametrize(
        "heard,query",
        [
            ("play by beyond playlist", "beyond"),
            ("play my beyond playlist", "beyond"),
            ("play by gym playlist", "gym"),
            ("play me dil playlist", "dil"),
            # Article dropped for PLAYLIST lookups: the library has no playlist
            # literally named "the focus".
            ("play by the focus playlist", "focus"),
        ],
    )
    def test_my_misheard_as_by_still_finds_the_playlist(self, heard, query):
        intent = parse(heard)
        assert intent.action is Action.PLAY_PLAYLIST
        assert intent.query == query

    def test_play_my_x_without_the_word_playlist(self):
        intent = parse("play by gym mix")
        assert intent.action is Action.PLAY_PLAYLIST
        assert intent.query == "gym mix"


class TestTrackByArtist:
    """"play <track> by <artist>" must stay a track search.

    This is the counter-case to the "my"->"by" rescue above: "by" is a real word
    in music queries. The rescue only fires when the possessive is the first
    word after "play", or when the phrase ends in "playlist", so these are safe.
    """

    @pytest.mark.parametrize(
        "text,query",
        [
            ("play red by seedhe maut", "red by seedhe maut"),
            ("play bohemian rhapsody by queen", "bohemian rhapsody by queen"),
            ("play espresso by sabrina carpenter", "espresso by sabrina carpenter"),
            ("hey memoir play nazar by dhanji", "nazar by dhanji"),
        ],
    )
    def test_track_by_artist_stays_a_track(self, text, query):
        intent = parse(text)
        assert intent.action is Action.PLAY_TRACK
        assert intent.query == query


class TestQueryWordsAreNotCommands:
    """Transport vocabulary inside a requested title is opaque query text."""

    @pytest.mark.parametrize(
        "text,query",
        [
            ("play don't stop me now", "don't stop me now"),
            ("play next to me by new west", "next to me by new west"),
            ("play pause by beyonce", "pause by beyonce"),
        ],
    )
    def test_transport_words_after_play_stay_in_query(self, text, query):
        intent = parse(text)
        assert intent.action is Action.PLAY_TRACK
        assert intent.query == query

    def test_filler_inside_title_is_preserved(self):
        intent = parse("play something like this")
        assert intent.action is Action.PLAY_TRACK
        assert intent.query == "something like this"


@pytest.mark.parametrize("text,value,mode,direction", [
    ("turn down volume to 20%", 20, "absolute", ""),
    ("turn it up to 80", 80, "absolute", ""),
    ("bring the volume down to thirty", 30, "absolute", ""),
    ("turn down the volume by 20%", 20, "relative_percent", "down"),
    ("turn the volume down by 20%", 20, "relative_percent", "down"),
])
def test_spoken_volume_targets(text, value, mode, direction):
    intent = parse(text)
    assert (intent.value, intent.volume_mode, intent.query) == (value, mode, direction)


@pytest.mark.parametrize("text", ["blame madhira by seedhe maut", "flame madira by seedhe maut"])
def test_misheard_play_verb(text):
    assert parse(text).action is Action.PLAY_TRACK
@pytest.mark.parametrize("heard,action,query", [
    ("Open Notepad", Action.OPEN_APP, "Notepad"),
    ("Launch the application Spotify", Action.OPEN_APP, "Spotify"),
    ("Start the Notepad app", Action.OPEN_APP, "Notepad"),
    ("Search C++ & logging in Opera", Action.SEARCH_BROWSER, "C++ & logging"),
    ("In Opera search Python classes", Action.SEARCH_BROWSER, "Python classes"),
    ("Open a tab in Opera and search for cats and dogs?", Action.SEARCH_BROWSER, "cats and dogs?"),
    ("Open Opera and search C++ and Python!", Action.SEARCH_BROWSER, "C++ and Python!"),
    ("Search cats and dogs in Opera", Action.SEARCH_BROWSER, "cats and dogs"),
    ("Search cats and dogs in Opera.", Action.SEARCH_BROWSER, "cats and dogs"),
    ("Search C++? and Python! in Opera GX browser?", Action.SEARCH_BROWSER, "C++? and Python!"),
    ('Search "pause and play" in Opera', Action.SEARCH_BROWSER, '"pause and play"'),
    ("Show song Us and Them", Action.SHOW_TRACK, "Us and Them"),
    ("Show song Us and Them on Spotify.", Action.SHOW_TRACK, "Us and Them"),
    ("Show song Why? on Spotify!", Action.SHOW_TRACK, "Why?"),
    ("Show album What Now?", Action.SHOW_ALBUM, "What Now?"),
    ('Open Opera and search "cats and please open Notepad"', Action.SEARCH_BROWSER, '"cats and please open Notepad"'),
    ("Show the song Red by Seedhe Maut on Spotify", Action.SHOW_TRACK, "Red by Seedhe Maut"),
    ("Open the album Blue Lights by Mira Sol", Action.SHOW_ALBUM, "Blue Lights by Mira Sol"),
    ("Restart the current album", Action.RESTART_ALBUM, ""),
    ("Start this album from the beginning", Action.RESTART_ALBUM, ""),
    ("Start a Jam", Action.START_JAM, ""),
    ("Start a Spotify Jam and show the QR code", Action.START_JAM, ""),
    ("Create the Spotify Jam and show me the QR on screen", Action.START_JAM, ""),
    ("Host a Jam", Action.START_JAM, ""),
    ("Show the Jam QR code", Action.START_JAM, ""),
    ("Open Spotify Jam invite code", Action.START_JAM, ""),
    ("Show the QR code of the Spotify Jam", Action.START_JAM, ""),
    ("Show the QR for the Jam", Action.START_JAM, ""),
    ("Start a Jam in Spotify and show the QR code", Action.START_JAM, ""),
    ("Start a Jam on Spotify and show the QR code", Action.START_JAM, ""),
    ("Show the Jam QR code in Spotify", Action.START_JAM, ""),
    ("Open the Jam QR code on Spotify", Action.START_JAM, ""),
    ("Start a Jam and show the QR of the Jam on screen", Action.START_JAM, ""),
    ("Play Jam by Michael Jackson", Action.PLAY_TRACK, "jam by michael jackson"),
])
def test_bounded_capabilities_keep_complete_source_queries(heard, action, query):
    result = parse(heard)
    assert result.action is action and result.query == query


@pytest.mark.parametrize("heard", [
    "Don't open Notepad", "He said open Notepad", "Yesterday I opened Notepad",
    "Open Notepad and delete my files", "Show album Blue then play Red",
    "Search logs in Opera and open Notepad", "Open Notepad; launch Spotify",
    "Start the Notepad app and delete files",
    "Search logs in an invented browser", "Show the song Red and set volume to 20",
    "Open Opera and search cats and then play Red",
    "Open a tab in Opera and search for cats and open Notepad",
    "Open Opera and search for cats; then delete files",
    "Search cats and dogs in Chrome",
    "Open Opera and search cats and please open Notepad",
    "Open a tab in Opera and search cats and also can you open Notepad",
    "Search cats and then please close Opera in Opera",
    "Show song Us and Them and could you play Red on Spotify.",
    "Don't start a Jam", "He said start a Jam", "My friend asked me to show the Jam QR code",
    "Start a Jam and pause", "Start a Spotify Jam then open Notepad",
    "Show the Jam QR code and play Red",
    "Start a Jam on Spotify and then pause", "Show the QR code of the Jam with a song",
])
def test_bounded_capabilities_do_not_cut_compounds_or_authorize_reported_requests(heard):
    assert parse(heard).action is Action.UNKNOWN
