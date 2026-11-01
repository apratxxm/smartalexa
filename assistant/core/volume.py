"""Windows master-volume writes; never a Spotify or mixer-session operation."""
from __future__ import annotations
import math
from numbers import Real
from .playback import remaining


class SystemVolumeVerificationError(RuntimeError):
    """A write was attempted and its effect cannot be established."""


def _endpoint():
    from pycaw.pycaw import AudioUtilities
    device = AudioUtilities.GetSpeakers()
    if device is None or device.EndpointVolume is None:
        raise RuntimeError("No Windows output endpoint is available")
    return device.EndpointVolume


def get_system_volume():
    return float(_endpoint().GetMasterVolumeLevelScalar())


def set_system_volume(level, *, guard=lambda: float("inf")):
    if isinstance(level, bool) or not isinstance(level, Real) or not math.isfinite(float(level)):
        raise ValueError("System volume must be a finite real number")
    target = min(1.0, max(0.0, float(level)))
    endpoint = _endpoint()  # Failure here precedes the write.
    remaining(guard)
    try:
        endpoint.SetMasterVolumeLevelScalar(target, None)
        actual = endpoint.GetMasterVolumeLevelScalar()
        if isinstance(actual, bool) or not isinstance(actual, Real):
            raise ValueError("invalid endpoint readback")
        actual = float(actual)
        delta = abs(actual - target)
        if not math.isfinite(actual) or not 0 <= actual <= 1 or (
                delta > 0.01 and not math.isclose(delta, 0.01, rel_tol=1e-12, abs_tol=1e-12)):
            raise ValueError("endpoint readback did not match")
    except Exception as exc:
        raise SystemVolumeVerificationError("Windows volume write could not be verified") from exc
