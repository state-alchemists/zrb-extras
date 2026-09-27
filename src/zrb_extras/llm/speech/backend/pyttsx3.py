from __future__ import annotations

import importlib.util

from zrb.llm.speech import AnySpeechBackend, Utterance

from zrb_extras.llm.tool.pyttsx3.speak import speak_with_pyttsx3


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
