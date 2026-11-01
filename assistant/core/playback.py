"""Per-turn guard hooks around retained Spotify mutation and wait boundaries."""
from contextlib import contextmanager
import threading
import time
from types import SimpleNamespace
from ..spotify import Spotify, _RateLimitGuard, PlaybackVerificationError, _matches_local_candidate, open_external_target


class TurnExpired(RuntimeError):
    pass


class NativePreDispatchRefusal(RuntimeError):
    """No native mutation was attempted."""


def remaining(guard):
    budget = guard()
    if budget <= 0:
        raise TurnExpired("Turn expired before mutation")
    return budget


@contextmanager
def guarded_playback(spotify, guard):
    if not isinstance(spotify, Spotify):
        yield
        return
    originals = {}
    def replace(name, method):
        originals[name] = (name in vars(spotify), vars(spotify).get(name))
        setattr(spotify, name, method)
    native, verify = spotify._now_playing, spotify._verify
    session = spotify.sp._client._session if isinstance(spotify.sp, _RateLimitGuard) else None
    send = session.send if session is not None else None
    owned_send = session is not None and "send" in vars(session)
    saved_send = vars(session).get("send") if session is not None else None
    if session is not None:
        def bounded_send(request, **kwargs):
            budget = remaining(guard)
            deadline = getattr(spotify, "_idle_uri_deadline", None)
            if deadline is not None:
                budget = min(budget, deadline - time.monotonic())
                if budget <= 0:
                    raise TurnExpired("Idle playback verification deadline expired")
            timeout = kwargs.get("timeout")
            kwargs["timeout"] = (tuple(min(v or budget, budget) for v in timeout)
                                 if isinstance(timeout, tuple) else min(timeout or budget, budget))
            return send(request, **kwargs)
        session.send = bounded_send

    def transport(name):
        from .. import local_media
        ready = False
        def build(before):
            nonlocal ready
            if before is None:
                return None
            identity = local_media._identity(before)
            if name == "try_skip_next_async" and not all(identity):
                return None
            remaining(guard)  # Runs after session discovery and metadata waits.
            ready = True
            if name == "try_skip_next_async":
                return lambda after: all(local_media._identity(after)) and local_media._identity(after) != identity
            return lambda after: after.playing == (name == "try_play_async")
        try:
            observed = local_media._run(local_media._verified_transport(name, build),
                                        timeout=min(3.0, remaining(guard)))
        except Exception as exc:
            cls = PlaybackVerificationError if ready else NativePreDispatchRefusal
            raise cls("Native transport could not establish completion") from exc
        if not observed:
            cls = PlaybackVerificationError if ready else NativePreDispatchRefusal
            raise cls("Native transport could not establish completion")
        return True
    controls = SimpleNamespace(play=lambda: transport("try_play_async"),
                               pause=lambda: transport("try_pause_async"),
                               next_track=lambda: transport("try_skip_next_async"))

    def call(fn, *args, **kwargs):
        target = spotify.device()  # Device discovery can consume the budget.
        remaining(guard)
        return fn(*args, device_id=target, **kwargs)

    def local(action):
        from .. import local_media
        if not spotify._local_enabled() or local_media.status() is None:
            return False
        remaining(guard)
        try:
            observed = action(controls)
        except NativePreDispatchRefusal:
            return False
        if not observed:
            raise PlaybackVerificationError("Local playback was not observed")
        spotify.last_playback = {"method": "local_media"}
        return True

    def desktop(candidate):
        from .. import local_media
        if not spotify._local_enabled() or not candidate.uri.startswith("spotify:track:"):
            return False
        current = local_media.snapshot()
        if current is None and not spotify._client_running():
            return False
        if _matches_local_candidate(candidate, current, require_playing=False):
            if current.playing:
                return True
            remaining(guard)
            try:
                observed = controls.play()
            except NativePreDispatchRefusal:
                pass
            else:
                try:
                    if observed is True and spotify._now_playing(candidate, timeout=1.5):
                        return True
                except Exception as exc:
                    raise PlaybackVerificationError("Native play verification failed after possible dispatch") from exc
                raise PlaybackVerificationError("Native play may have dispatched; no further mutation allowed")
        remaining(guard)
        controls.pause()
        remaining(guard)
        with local_media.InvisibleSpotify() as window:
            remaining(guard)
            open_external_target(candidate.uri)
            threading.Thread(target=window.return_focus, daemon=True).start()
            return bool(spotify._now_playing(candidate, timeout=8.0))

    def open_idle_uri(uri):
        remaining(guard)
        open_external_target(uri)

    def idle_budget():
        budget = remaining(guard)
        deadline = getattr(spotify, "_idle_uri_deadline", None)
        if deadline is not None:
            budget = min(budget, deadline - time.monotonic())
            if budget <= 0:
                raise TurnExpired("Idle playback verification deadline expired")
        return budget

    replace("_idle_uri_budget", idle_budget)
    replace("_open_idle_uri", open_idle_uri)
    replace("_call", call)
    replace("_local", local)
    replace("_play_uri_locally", desktop)
    replace("_now_playing", lambda candidate, timeout, **kwargs:
            native(candidate, timeout=min(timeout, remaining(guard)), **kwargs))
    replace("_verify", lambda predicate, description, horizon=None:
            verify(predicate, description, horizon=min(
                0.75 if horizon is None else horizon, remaining(guard))))
    try:
        yield
    finally:
        if session is not None:
            if owned_send:
                session.send = saved_send
            else:
                vars(session).pop("send", None)
        for name, (owned, value) in originals.items():
            if owned:
                setattr(spotify, name, value)
            else:
                vars(spotify).pop(name, None)
