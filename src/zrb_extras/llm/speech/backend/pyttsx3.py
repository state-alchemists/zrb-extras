from __future__ import annotations

import importlib.util

from zrb.llm.speech import AnySpeechBackend, Utterance


def speak_with_pyttsx3(
    text: str,
    voice_name: str | None = None,
    rate: int | None = None,
    volume: float | None = None,
) -> None:
    """Say *text* with pyttsx3, blocking until it is said."""
    import pyttsx3

    engine = pyttsx3.init()
    if voice_name:
        engine.setProperty("voice", voice_name)
    if rate:
        engine.setProperty("rate", rate)
    if volume is not None:
        engine.setProperty("volume", volume)
    engine.say(text)
    engine.runAndWait()


def list_available_voices() -> list[dict[str, str | list[str]]]:
    """The installed pyttsx3 voices, as ``id``, ``name`` and ``languages``."""
    import pyttsx3

    return [
        {
            "id": voice.id,
            "name": voice.name,
            "languages": list(voice.languages or []),
        }
        for voice in pyttsx3.init().getProperty("voices")
    ]


class Pyttsx3SpeechBackend(AnySpeechBackend):
    """Offline text-to-speech through pyttsx3 (SAPI5, NSSpeechSynthesizer or
    espeak), in process. *rate* is words per minute, *volume* 0.0 to 1.0."""

    def __init__(
        self,
        voice_name: str | None = None,
        rate: int | None = None,
        volume: float | None = None,
    ) -> None:
        self._voice_name = voice_name
        self._rate = rate
        self._volume = volume

    @property
    def name(self) -> str:
        return "pyttsx3"

    def create_utterance(self, text: str) -> Utterance:
        if importlib.util.find_spec("pyttsx3") is None:
            raise RuntimeError("pyttsx3 is not installed: pip install 'zrb-extras[pyttsx3]'")
        return Pyttsx3Utterance(text, self._voice_name, self._rate, self._volume)


class Pyttsx3Utterance(Utterance):
    """Speech pyttsx3 plays itself; there is no player command."""

    def __init__(
        self,
        text: str,
        voice_name: str | None,
        rate: int | None,
        volume: float | None,
    ) -> None:
        super().__init__([])
        self.text = text
        self._voice_name = voice_name
        self._rate = rate
        self._volume = volume

    def play(self, timeout: float | None) -> None:
        # ponytail: timeout ignored; pyttsx3's runAndWait cannot be interrupted
        # from outside, so a stuck engine holds the speech thread until it returns.
        speak_with_pyttsx3(self.text, self._voice_name, self._rate, self._volume)
