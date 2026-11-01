"""Phase 4 Spotify Jam adapter: shows the Jam invite QR through the isolated helper."""
from __future__ import annotations

import time

from ... import computer_tools
from ...computer_tools import ComputerControlError, ComputerDeliveryUnknown
from .base import Adapter, Clause, Guard, Receipt, TurnExpired

_VERBS = ("start a jam", "start jam", "show the jam code", "show the jam qr")

_OBSERVED_REPLY = "Jam started; the QR code is on screen."
_UNKNOWN_REPLY = "I asked Spotify to start a Jam but can't confirm it."
_FAILED_REPLY = "I couldn't start the Spotify Jam."


class StartJamHandler:
    mutating = True
    verbs = _VERBS
    object_kind = "none"
    effect_words = ("started", "starting", "showing", "jam")

    def license(self, clause: Clause) -> dict | None:
        if clause.verb in self.verbs and clause.object == "":
            return {}
        return None

    def execute(self, args: dict, guard: Guard) -> Receipt:
        if args != {}:
            return Receipt("failed", error="start_jam takes no arguments")
        try:
            budget = guard()
        except TurnExpired:
            return Receipt("failed", error="turn expired before dispatch")
        if budget <= 0:
            return Receipt("failed", error="turn expired before dispatch")
        # Last guard point before the helper launches; nothing above has dispatched.
        try:
            computer_tools.start_spotify_jam(deadline=time.monotonic() + budget)
        except ComputerDeliveryUnknown:
            # Helper ran but did not verify the invite UI (or returned a not-ok payload).
            return Receipt("unknown", error="jam invite not verified")
        except ComputerControlError:
            # The helper process could not be started, so nothing was dispatched.
            return Receipt("failed", error="helper could not be started")
        except Exception:
            # Any other failure after launch may still have delivered the action.
            return Receipt("unknown", error="jam invite state unverified")
        return Receipt("observed", detail="jam invite verified")

    def reply(self, receipt: Receipt, args: dict) -> str:
        if receipt.state == "observed":
            return _OBSERVED_REPLY
        if receipt.state == "failed":
            return _FAILED_REPLY
        return _UNKNOWN_REPLY


START_JAM = StartJamHandler()

ADAPTER = Adapter("jam", "4", handlers={"start_jam": START_JAM})

__all__ = ["ADAPTER", "START_JAM", "StartJamHandler"]
