"""Speech backends for zrb's built-in speech (`zrb.llm.presentation.speech.enable_speech`
in zrb 4.0, `zrb.llm.speech.enable_speech` before it)."""

from zrb_extras.llm.speech.backend import Pyttsx3SpeechBackend

__all__ = ["Pyttsx3SpeechBackend"]
