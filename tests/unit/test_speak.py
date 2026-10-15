from unittest.mock import patch

from zrb.llm.speech import SpeechConfig

from zrb_extras.llm.tool import create_speak_tool


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


def test_speak_tool_can_be_renamed():
    speak = create_speak_tool(tool_name="say", tool_description="Talk.")

    assert (speak.__name__, speak.__doc__) == ("say", "Talk.")
