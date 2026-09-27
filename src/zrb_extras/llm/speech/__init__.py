"""Speech backends for zrb's built-in speech (`zrb.llm.speech.enable_speech`)."""

from zrb_extras.llm.speech.backend import Pyttsx3SpeechBackend, TermuxSpeechBackend

__all__ = ["Pyttsx3SpeechBackend", "TermuxSpeechBackend"]
