"""Audible feedback, including cached spoken cues for each listening state."""
from __future__ import annotations

import logging
import threading

from .config import Config

log = logging.getLogger("memoir.feedback")

# (frequency Hz, duration ms) -- distinguishable without being musical.
# Two-tone pair so the listening window is unmistakable: rising = I am
# recording, falling = I have stopped. The complaint this solves is not knowing
# which step you are at.
BEEP_WAKE = (880, 90)
BEEP_TICK = (990, 45)
BEEP_LISTEN = (1050, 110)
BEEP_STOP = (700, 90)
BEEP_OK = (1180, 70)
BEEP_ERR = (330, 180)

KIND_WAKE = "beep_wake"
KIND_OK = "beep_ok"
KIND_ERR = "beep_err"
KIND_SPEECH = "speech"

# Synthesised phrases are few and repeat constantly; synthesising "Sorry." on
# every failure added a few hundred milliseconds of silence before it.
_SPEECH_CACHE_LIMIT = 32
_SPEECH_CACHE_BYTES = 4 * 1024 * 1024
_SPEECH_CACHE_ITEM_BYTES = 1024 * 1024
_SPEECH_CACHE_TEXT_LIMIT = 512


class Feedback:
    def __init__(self, cfg: Config) -> None:
        self.wake_phrase = cfg.get("feedback.wake_phrase", "")
        self.speak_on_wake = cfg.get("feedback.speak_on_wake", False)
        self._cues = {}
        self._speech_cache = {}
        self._speech_cache_lock = threading.Lock()
        self.beeps_enabled = cfg.get("feedback.beeps_enabled", True)
        self.speak_on_failure = cfg.get("feedback.speak_on_failure", True)
        self.speak_on_success = cfg.get("feedback.speak_on_success", False)
        self.output_device = cfg.get("feedback.output_device") or None
        self.voice_path = cfg.path("feedback.piper_voice")
        # Ceiling on how loud a prompt may be, as a Windows master-volume
        # equivalent. At 100% Windows volume with Spotify closed there is no
        # duck, and full-scale speech into the SoundDrum was painfully loud.
        # Only our own PCM is scaled: the Windows and Spotify levels the user
        # chose are never touched.
        self.max_output_level = max(0.0, min(1.0, float(
            cfg.get("feedback.max_output_level", 0.35))))
        self._piper = None
        self._piper_failed = False

    # ---------- output ----------

    @staticmethod
    def _device_level(device) -> float:
        """Read the volume of the endpoint actually selected for the cue."""
        from pycaw.pycaw import AudioUtilities
        name = str(device).casefold()
        default = AudioUtilities.GetSpeakers()
        devices = [default] if default is not None else []
        if not devices or not name.startswith((default.FriendlyName or "").casefold()):
            devices = AudioUtilities.GetAllDevices()
        for endpoint in devices:
            friendly = endpoint.FriendlyName or ""
            state = getattr(endpoint.state, "value", endpoint.state)
            if friendly and state == 1 and name.startswith(friendly.casefold()):
                return float(endpoint.EndpointVolume.GetMasterVolumeLevelScalar())
        raise RuntimeError("Selected feedback endpoint volume unavailable")

    def _output_gain(self, device=None) -> float:
        try:
            from . import ducking
            level = self._device_level(device) if device is not None else ducking.get_system_volume()
        except Exception as exc:
            log.debug("Output level unknown (%s); assuming full volume", exc)
            level = 1.0
        return min(1.0, self.max_output_level / max(level, 0.01))

    def _output(self):
        import sounddevice as sd
        try:
            sd.query_devices(self.output_device, "output")
            return self.output_device
        except ValueError:
            log.warning("Configured speaker unavailable; using Windows default output.")
            return None

    def _play(self, pcm, rate: int, blocking: bool = True) -> bool:
        """Play int16 or float PCM, loudness-capped. Never raises."""
        try:
            import numpy as np
            import sounddevice as sd

            audio = np.asarray(pcm, dtype=np.float32)
            if np.issubdtype(np.asarray(pcm).dtype, np.integer):
                audio = audio / 32768.0
            device = self._output()
            audio = np.clip(audio * self._output_gain(device), -1.0, 1.0)
            sd.play(audio, rate, device=device)
            if blocking:
                sd.wait()
            return True
        except Exception as exc:
            log.warning("Audio feedback failed: %s", exc)
            return False

    # ---------- beeps ----------

    def _beep(self, freq: int, ms: int, blocking: bool = True) -> None:
        if not self.beeps_enabled:
            return
        import numpy as np

        rate = 16000
        samples = max(1, int(rate * ms / 1000))
        t = np.arange(samples, dtype=np.float32) / rate
        fade = min(samples // 4, int(rate * 0.01))
        envelope = np.ones(samples, dtype=np.float32)
        if fade:
            envelope[:fade] = np.linspace(0, 1, fade, dtype=np.float32)
            envelope[-fade:] = np.linspace(1, 0, fade, dtype=np.float32)
        self._play(0.15 * np.sin(2 * np.pi * freq * t) * envelope, rate, blocking)

    # ---------- speech ----------

    def _load_piper(self):
        if self._piper is not None or self._piper_failed:
            return self._piper
        try:
            from piper.voice import PiperVoice

            if not self.voice_path.exists():
                raise FileNotFoundError(
                    "Piper voice not found at {}. Download a voice from "
                    "https://huggingface.co/rhasspy/piper-voices".format(self.voice_path)
                )
            self._piper = PiperVoice.load(str(self.voice_path))
            log.info("Piper voice loaded: %s", self.voice_path.name)
        except Exception as exc:
            # Never fatal: losing speech should degrade to beeps, not stop the
            # assistant from controlling music.
            log.warning("TTS unavailable, falling back to beeps only: %s", exc)
            self._piper_failed = True
        return self._piper

    def _synth(self, text: str):
        """(pcm, rate) for full text; cache only small cues, None without a voice."""
        with self._speech_cache_lock:
            cached = self._speech_cache.get(text)
        if cached is not None:
            return cached
        voice = self._load_piper()
        if voice is None:
            return None
        import io
        import wave

        import numpy as np

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            voice.synthesize_wav(text, wf)
        buf.seek(0)
        with wave.open(buf, "rb") as wf:
            rate = wf.getframerate()
            pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
        # Piper leaves a quiet tail; keep consonants but not the wait.
        window = max(1, int(rate * 0.02))
        levels = [np.sqrt(np.mean(pcm[i:i + window].astype(np.float32) ** 2))
                  for i in range(0, len(pcm), window)]
        audible = [i for i, level in enumerate(levels) if level > 350]
        if audible:
            start = max(0, (audible[0] - 1) * window)
            end = min(len(pcm), (audible[-1] + 5) * window)
            pcm = pcm[start:end]
        if (len(text) <= _SPEECH_CACHE_TEXT_LIMIT
                and pcm.nbytes <= _SPEECH_CACHE_ITEM_BYTES):
            # A trimmed view retains the entire WAV buffer. Copy only cues
            # eligible for caching so the byte budget reflects retained PCM.
            pcm = pcm.copy()
            with self._speech_cache_lock:
                self._speech_cache.pop(text, None)
                cache_bytes = sum(cue[0].nbytes for cue in self._speech_cache.values())
                while self._speech_cache and (
                    len(self._speech_cache) >= _SPEECH_CACHE_LIMIT
                    or cache_bytes + pcm.nbytes > _SPEECH_CACHE_BYTES
                ):
                    oldest = self._speech_cache.pop(next(iter(self._speech_cache)))
                    cache_bytes -= oldest[0].nbytes
                self._speech_cache[text] = (pcm, rate)
        return pcm, rate

    def prime(self) -> None:
        """Synthesise the fixed phrases once, before accepting commands."""
        phrases = [("retry", "Repeat, please."), ("ok", "Done.")]
        if self.speak_on_wake and self.wake_phrase:
            phrases = [("wake", self.wake_phrase), ("stopped", "Got it.")] + phrases
        try:
            for name, phrase in phrases:
                cue = self._synth(phrase)
                if cue is None:
                    return
                self._cues[name] = cue
            for phrase in ("Sorry.", "Sorry, I couldn't find that."):
                self._synth(phrase)
            if "wake" in self._cues:
                log.info("Wake prompt ready: %r", self.wake_phrase)
            else:
                log.info("No spoken wake prompt: music ducks and capture starts at once")
        except Exception as exc:
            log.warning("Could not prepare spoken cues (%s); using beeps.", exc)
            self._cues.pop("wake", None)

    def _play_cue(self, name: str, blocking: bool = True) -> bool:
        cue = self._cues.get(name)
        if cue is None:
            return False
        return self._play(*cue, blocking=blocking)

    def wake(self) -> str:
        """Acknowledge the wake word, BLOCKING until the prompt has finished.

        Blocking matters: if we returned while still speaking, the recorder
        would capture the assistant's own voice and try to obey it.
        """
        if self._play_cue("wake"):
            return KIND_SPEECH
        self._beep(*BEEP_WAKE)
        return KIND_WAKE

    @property
    def speaks_on_wake(self) -> bool:
        return self._cues.get("wake") is not None

    def tick(self) -> str:
        """A short, soft, non-blocking wake cue; never delays capture."""
        self._beep(*BEEP_TICK, blocking=False)
        return KIND_WAKE

    def listening(self) -> str:
        """The preceding spoken prompt already says when to speak."""
        if self._cues.get("wake") is not None:
            return KIND_SPEECH
        self._beep(*BEEP_LISTEN)
        return "beep_listen"

    def stopped(self) -> str:
        """Confirm that recording closed. Never delays transcription."""
        if self._play_cue("stopped", blocking=False):
            return KIND_SPEECH
        self._beep(*BEEP_STOP, blocking=False)
        return "beep_stop"

    def ok(self) -> str:
        # The music starting is the confirmation; a spoken "Done." on every
        # command only added delay. speak_on_success restores it.
        if self.speak_on_success and self._play_cue("ok", blocking=False):
            return KIND_SPEECH
        self._beep(*BEEP_OK, blocking=False)
        return KIND_OK

    def error(self) -> str:
        self._beep(*BEEP_ERR)
        return KIND_ERR

    def say(self, text: str, force: bool = False, blocking: bool = False) -> str:
        """Speak a short message. Used for failures and questions only."""
        if not (force or self.speak_on_failure):
            return self.error()
        if text == "Repeat, please." and self._play_cue("retry", blocking=blocking):
            return KIND_SPEECH

        def _speak() -> bool:
            try:
                cue = self._synth(text)
            except Exception as exc:
                log.warning("Speech synthesis failed: %s", exc)
                cue = None
            if cue is None:
                self.error()
                return False
            return self._play(*cue, blocking=True)

        if self._load_piper() is None:
            return self.error()
        if blocking:
            # Used before listening for a reply: returning early would put our
            # own question into the recording we are about to make.
            return KIND_SPEECH if _speak() else KIND_ERR
        else:
            threading.Thread(target=_speak, daemon=True).start()
        return KIND_SPEECH


def main() -> None:
    """Smoke test: play each beep, then a spoken failure message."""
    import time

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from .config import load

    fb = Feedback(load())
    print("output gain at current Windows volume: %.2f" % fb._output_gain())
    for name, fn in (("wake", fb.wake), ("ok", fb.ok), ("error", fb.error)):
        print("beep:", name)
        fn()
        time.sleep(0.6)
    print("speech: failure message")
    fb.say("Sorry.", force=True, blocking=True)


if __name__ == "__main__":
    main()
