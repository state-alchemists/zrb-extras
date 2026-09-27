import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from zrb_extras.llm.speech import Pyttsx3SpeechBackend


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


def test_pyttsx3_plugs_into_zrb_speech():
    from zrb.llm.speech import SpeechConfig
    from zrb.llm.speech.backend import get_speech_backend

    backend = Pyttsx3SpeechBackend()
    assert get_speech_backend(backend, SpeechConfig().resolve()) is backend


def test_termux_speak_tool_builds_zrb_termux_command():
    import asyncio

    from zrb_extras.llm.tool.termux.speak import create_speak_tool

    speak = create_speak_tool(language="id")
    with patch("shutil.which", return_value="/usr/bin/termux-tts-speak"), patch(
        "subprocess.run"
    ) as run:
        assert asyncio.run(speak("halo")) is True

    assert run.call_args.args[0] == ["termux-tts-speak", "-l", "id", "halo"]
