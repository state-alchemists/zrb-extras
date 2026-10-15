from unittest.mock import MagicMock, patch

import pytest
from zrb.llm.speech import SpeechConfig

from zrb_extras.llm.tool import create_speak_tool
from zrb_extras.llm.tool.speak import MAX_SPEAKERS


async def test_speak_uses_the_configured_voice_by_default():
    with patch("zrb_extras.llm.tool.speak.Speaker") as speaker_class:
        speak = create_speak_tool(SpeechConfig(backend="openai", voice="coral"))
        assert await speak("hello") is True

    config = speaker_class.call_args.args[0]
    assert (config.backend, config.voice) == ("openai", "coral")
    speaker_class.return_value.speak.assert_called_once_with("hello")


async def test_speak_overrides_voice_and_instruction_per_call():
    with patch("zrb_extras.llm.tool.speak.Speaker") as speaker_class:
        speak = create_speak_tool(SpeechConfig(backend="openai", voice="coral"))
        await speak("hi", voice_name="onyx", instruction="whisper")

    config = speaker_class.call_args.args[0]
    assert (config.voice, config.style) == ("onyx", "whisper")


async def test_speak_reuses_one_speaker_per_voice_and_instruction():
    with patch("zrb_extras.llm.tool.speak.Speaker") as speaker_class:
        speak = create_speak_tool(SpeechConfig(backend="openai"))
        await speak("one", voice_name="onyx")
        await speak("two", voice_name="onyx")
        await speak("three", voice_name="nova")

    assert speaker_class.call_count == 2


async def test_speak_closes_the_least_recently_used_speaker_past_the_limit():
    with patch("zrb_extras.llm.tool.speak.Speaker") as speaker_class:
        speakers = [MagicMock() for _ in range(MAX_SPEAKERS + 1)]
        speaker_class.side_effect = speakers
        speak = create_speak_tool(SpeechConfig(backend="openai"))
        for index in range(MAX_SPEAKERS + 1):
            await speak("hi", instruction=f"style {index}")

    speakers[0].close.assert_called_once()
    assert not any(speaker.close.called for speaker in speakers[1:])


async def test_speak_refuses_blank_text():
    with patch("zrb_extras.llm.tool.speak.Speaker") as speaker_class:
        speak = create_speak_tool()
        with pytest.raises(ValueError, match="blank"):
            await speak("   ")

    speaker_class.assert_not_called()


async def test_speak_raises_when_nothing_could_be_said():
    with patch("zrb_extras.llm.tool.speak.Speaker") as speaker_class:
        speaker_class.return_value.speak.side_effect = RuntimeError(
            "No speech backend could speak (openai: 401)"
        )
        speak = create_speak_tool(SpeechConfig(backend="openai"))
        with pytest.raises(RuntimeError, match="openai: 401"):
            await speak("hello")


def test_speak_tool_can_be_renamed():
    speak = create_speak_tool(tool_name="say", tool_description="Talk.")

    assert (speak.__name__, speak.__doc__) == ("say", "Talk.")
