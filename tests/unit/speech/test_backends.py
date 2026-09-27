import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from zrb_extras.llm.speech import Pyttsx3SpeechBackend, TermuxSpeechBackend


def test_termux_utterance_runs_termux_tts_speak_with_the_options():
    backend = TermuxSpeechBackend(language="id", voice_name="f1", rate=1.2)
    with patch("shutil.which", return_value="/usr/bin/termux-tts-speak"):
        utterance = backend.create_utterance("halo")

    assert utterance.argv == [
        "termux-tts-speak", "-l", "id", "-v", "f1", "-r", "1.2", "halo"
    ]
    assert backend.name == "termux"


def test_termux_without_termux_api_raises_so_zrb_falls_back():
    with patch("shutil.which", return_value=None):
        with pytest.raises(RuntimeError, match="termux-api"):
            TermuxSpeechBackend().create_utterance("halo")


def test_pyttsx3_utterance_speaks_in_process_with_its_settings():
    engine = MagicMock()
    fake = SimpleNamespace(init=lambda: engine)
    backend = Pyttsx3SpeechBackend(voice_name="english-us+m3", rate=150, volume=0.5)
    with patch.dict(sys.modules, {"pyttsx3": fake}), patch(
        "importlib.util.find_spec", return_value=object()
    ):
        backend.create_utterance("hello").play(timeout=10)

    engine.setProperty.assert_any_call("voice", "english-us+m3")
    engine.setProperty.assert_any_call("rate", 150)
    engine.setProperty.assert_any_call("volume", 0.5)
    engine.say.assert_called_once_with("hello")
    engine.runAndWait.assert_called_once()
    assert backend.name == "pyttsx3"


def test_pyttsx3_missing_raises_so_zrb_falls_back():
    with patch("importlib.util.find_spec", return_value=None):
        with pytest.raises(RuntimeError, match="zrb-extras\\[pyttsx3\\]"):
            Pyttsx3SpeechBackend().create_utterance("hello")


def test_backends_plug_into_zrb_speech():
    from zrb.llm.speech import SpeechConfig
    from zrb.llm.speech.backend import get_speech_backend

    backend = TermuxSpeechBackend()
    assert get_speech_backend(backend, SpeechConfig().resolve()) is backend
