"""Spotify Jam flow tests use snapshots only; no native app is launched."""
import pytest

from assistant.spotify_jam import (
    Control,
    INVITE_INSTRUCTIONS,
    INVITE_PROMPT,
    Snapshot,
    SpotifyJamError,
    SUCCESS,
    _input_types,
    drive_spotify_jam,
    drive_spotify_search,
)


def control(name, kind="button", enabled=True, offscreen=False):
    return Control(name, kind, enabled, offscreen)


def invitation(*, copy_enabled=True):
    return Snapshot(
        controls=(control("Copy link", enabled=copy_enabled),),
        text=(INVITE_PROMPT, INVITE_INSTRUCTIONS),
    )


class FakeBackend:
    def __init__(self, snapshots, *, click_error=None):
        self.snapshots = list(snapshots)
        self.click_error = click_error
        self.clicks = []

    def snapshot(self):
        if len(self.snapshots) > 1:
            return self.snapshots.pop(0)
        return self.snapshots[0]

    def click(self, name):
        if self.click_error:
            raise self.click_error
        self.clicks.append(name)


class FakeTime:
    def __init__(self):
        self.value = 0.0

    def now(self):
        return self.value

    def sleep(self, seconds):
        self.value += max(seconds, 0.01)


def drive(backend, fake_time=None):
    fake_time = fake_time or FakeTime()
    return drive_spotify_jam(backend, timeout=0.5, clock=fake_time.now, sleep=fake_time.sleep)


def test_create_flow_opens_connect_then_starts_and_verifies_invite():
    backend = FakeBackend([
        Snapshot((control("Connect to a device", "toggle"),)),
        Snapshot((control("Start a Jam"),)),
        invitation(),
    ])
    assert drive(backend) == SUCCESS
    assert backend.clicks == ["Connect to a device", "Start a Jam"]


def test_existing_jam_uses_invite_and_verifies_the_invitation_panel():
    backend = FakeBackend([
        Snapshot((control("Alex's Jam", "heading"), control("Invite"))),
        invitation(),
    ])
    assert drive(backend) == SUCCESS
    assert backend.clicks == ["Invite"]


def test_existing_jam_footer_can_open_jam_panel_before_invite():
    backend = FakeBackend([
        Snapshot((control("Alex's Jam", "text"), control("Alex's Jam", "button"))),
        Snapshot((control("Alex's Jam", "heading"), control("Invite"))),
        invitation(),
    ])
    assert drive(backend) == SUCCESS
    assert backend.clicks == ["Alex's Jam", "Invite"]


def test_already_open_qr_invitation_is_success_without_interaction():
    backend = FakeBackend([invitation()])
    assert drive(backend) == SUCCESS
    assert backend.clicks == []


def test_disabled_start_button_fails_without_interaction():
    backend = FakeBackend([Snapshot((control("Start a Jam", enabled=False),))])
    with pytest.raises(SpotifyJamError, match="disabled"):
        drive(backend)
    assert backend.clicks == []


def test_missing_controls_fail_with_actionable_state_error():
    backend = FakeBackend([Snapshot(())])
    with pytest.raises(SpotifyJamError, match="did not show a verified Jam invitation"):
        drive(backend)
    assert backend.clicks == []


def test_generic_invite_without_jam_heading_is_never_clicked():
    backend = FakeBackend([Snapshot((control("Invite"),))])
    with pytest.raises(SpotifyJamError, match="without an identifiable Jam heading"):
        drive(backend)
    assert backend.clicks == []


def test_multiple_start_buttons_are_ambiguous_and_never_clicked():
    backend = FakeBackend([Snapshot((control("Start a Jam"), control("Start a Jam")))])
    with pytest.raises(SpotifyJamError, match="multiple Start a Jam"):
        drive(backend)
    assert backend.clicks == []


def test_focus_loss_aborts_without_retrying_the_interaction():
    backend = FakeBackend(
        [Snapshot((control("Connect to a device", "toggle"),))],
        click_error=SpotifyJamError("Spotify focus changed. No Jam control was clicked."),
    )
    with pytest.raises(SpotifyJamError, match="focus changed"):
        drive(backend)
    assert backend.clicks == []


def test_connect_panel_toggle_is_not_repeated_if_state_does_not_change():
    connect = Snapshot((control("Connect to a device", "toggle"),))
    backend = FakeBackend([connect])
    with pytest.raises(SpotifyJamError, match="did not show a verified Jam invitation"):
        drive(backend)
    assert backend.clicks == ["Connect to a device"]


def test_partial_invitation_or_generic_copy_link_never_reports_success():
    partial = Snapshot((control("Copy link"),), text=(INVITE_PROMPT,))
    with pytest.raises(SpotifyJamError, match="invitation panel is incomplete"):
        drive(FakeBackend([partial]))
    generic = Snapshot((control("Copy link"),), text=("Share this item",))
    with pytest.raises(SpotifyJamError, match="did not show a verified Jam invitation"):
        drive(FakeBackend([generic]))


def test_offscreen_controls_are_not_used():
    backend = FakeBackend([Snapshot((control("Start a Jam", offscreen=True),))])
    with pytest.raises(SpotifyJamError, match="did not show a verified Jam invitation"):
        drive(backend)
    assert backend.clicks == []


def test_collapsed_existing_jam_opens_its_panel_without_creating_another():
    backend = FakeBackend([
        Snapshot((control("Owner’s Jam"), control("Connect to a device", "toggle"))),
        Snapshot((control("Owner’s Jam", "text"), control("Invite"))),
        invitation(),
    ])
    assert drive(backend) == SUCCESS
    assert backend.clicks == ["Owner’s Jam", "Invite"]


def test_native_focus_checks_do_not_reactivate_after_user_switches_window():
    import ctypes
    from ctypes import wintypes
    from types import SimpleNamespace
    from assistant.spotify_jam import WindowsSpotifyBackend
    backend = WindowsSpotifyBackend.__new__(WindowsSpotifyBackend)
    backend.ctypes, backend.wintypes = ctypes, wintypes
    backend.hwnd, backend.pid, backend._activated = 123, 456, True
    backend._ensure_window = lambda: None
    activations = []
    backend.user32 = SimpleNamespace(
        GetForegroundWindow=lambda: 999,
        SetForegroundWindow=lambda hwnd: activations.append(hwnd),
        IsIconic=lambda hwnd: False,
    )
    with pytest.raises(SpotifyJamError, match="focus changed"):
        backend._focus()
    assert activations == []


def test_sendinput_layout_matches_32_and_64_bit_win32_abi_without_native_input():
    import ctypes

    key32, union32, input32 = _input_types(ctypes, ctypes.c_uint32)
    key64, union64, input64 = _input_types(ctypes, ctypes.c_uint64)
    assert ctypes.sizeof(key32) == 16
    assert ctypes.sizeof(union32) == 24
    assert ctypes.sizeof(input32) == 28
    assert input32.payload.offset == 4
    assert ctypes.sizeof(key64) == 24
    assert ctypes.sizeof(union64) == 32
    assert ctypes.sizeof(input64) == 40
    assert input64.payload.offset == 8


def search_results(query_value, result_names=("RED", "Seedhe Maut")):
    controls = [control("What do you want to play?", "combobox")]
    controls[0] = Control(controls[0].name, "combobox", True, False, query_value)
    controls.extend(control(name, "button") for name in ("All", "Songs", "Playlists", "Albums"))
    controls.extend(control(name, "text") for name in result_names)
    return Snapshot(tuple(controls))


class FakeSearchBackend:
    def __init__(self, snapshots=(), *, submit_error=None):
        self.snapshots = list(snapshots)
        self.submit_error = submit_error
        self.submissions = []

    def submit_search(self, query):
        if self.submit_error:
            raise self.submit_error
        self.submissions.append(query)

    def snapshot(self):
        if len(self.snapshots) > 1:
            return self.snapshots.pop(0)
        return self.snapshots[0]


def drive_search(backend, query="Red Seedhe Maut", kind="track", fake_time=None):
    fake_time = fake_time or FakeTime()
    return drive_spotify_search(
        backend, query, kind, timeout=0.5,
        clock=fake_time.now, sleep=fake_time.sleep)


def test_spotify_search_reports_success_only_for_matching_native_results():
    backend = FakeSearchBackend([search_results("Red Seedhe Maut")])
    assert drive_search(backend) == "Spotify desktop search results are visible for that query."
    assert backend.submissions == ["Red Seedhe Maut"]


def test_spotify_search_waits_through_stale_value_before_accepting_fresh_results():
    backend = FakeSearchBackend([
        search_results("Old Search"),
        search_results("Red Seedhe Maut"),
    ])
    assert drive_search(backend) == "Spotify desktop search results are visible for that query."
    assert backend.submissions == ["Red Seedhe Maut"]


def test_spotify_search_propagates_focus_loss_without_retrying_submission():
    backend = FakeSearchBackend(submit_error=SpotifyJamError(
        "Spotify's search focus changed before submission. No key was sent."))
    with pytest.raises(SpotifyJamError, match="focus changed"):
        drive_search(backend)
    assert backend.submissions == []


@pytest.mark.parametrize("error", [
    "Spotify shows multiple matching search boxes.",
    "Spotify's named search box is unavailable.",
])
def test_spotify_search_refuses_ambiguous_or_missing_search_control(error):
    backend = FakeSearchBackend(submit_error=SpotifyJamError(error))
    with pytest.raises(SpotifyJamError, match="search box"):
        drive_search(backend)
    assert backend.submissions == []


def test_spotify_search_timeout_does_not_resubmit_after_stale_results():
    stale = search_results("Old Search")
    backend = FakeSearchBackend([stale])
    with pytest.raises(SpotifyJamError, match="may have been submitted"):
        drive_search(backend)
    assert backend.submissions == ["Red Seedhe Maut"]


@pytest.mark.parametrize("query,kind", [
    ("", "track"), (None, "track"), ("x\nReturn", "track"), ("x\x00", "track"),
    ("x" * 401, "album"), ("Red", "playlist"),
])
def test_spotify_search_rejects_invalid_queries_or_kinds_before_submission(query, kind):
    backend = FakeSearchBackend([search_results("Red")])
    with pytest.raises(SpotifyJamError):
        drive_search(backend, query=query, kind=kind)
    assert backend.submissions == []
