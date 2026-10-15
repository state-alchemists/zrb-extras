import asyncio
import threading
from collections import OrderedDict
from dataclasses import replace
from typing import Any, Callable, Coroutine

from zrb.llm.speech import Speaker, SpeechConfig

MAX_SPEAKERS = 8


def create_speak_tool(
    config: SpeechConfig | None = None,
    tool_name: str | None = None,
    tool_description: str | None = None,
) -> Callable[[str, str | None, str | None], Coroutine[Any, Any, bool]]:
    """A `speak` tool that says text aloud through zrb's speech backends.

    *config* picks the backend and its defaults exactly as for `enable_speech`
    (``backend="openai"``, ``"gemini"``, ``"termux"``, a Pipecat voice such as
    ``"kokoro"``, or an `AnySpeechBackend` like `Pyttsx3SpeechBackend`); a field
    left ``None`` is read from ``ZRB_LLM_SPEECH_*``. A backend passed as an
    object ignores the per-call voice and instruction.
    """
    base = config or SpeechConfig()
    # One utterance at a time: parallel tool calls would otherwise talk over each other.
    speaking = threading.Lock()

    # A Speaker per voice/instruction pair keeps its backend (and any Pipecat
    # pipeline) warm; the least recently used is closed past MAX_SPEAKERS.
    speakers: OrderedDict[tuple[str | None, str | None], Speaker] = OrderedDict()

    def get_speaker(voice: str | None, style: str | None) -> Speaker:
        key = (voice, style)
        if key in speakers:
            speakers.move_to_end(key)
            return speakers[key]
        speakers[key] = Speaker(
            replace(base, voice=voice or base.voice, style=style or base.style).resolve()
        )
        if len(speakers) > MAX_SPEAKERS:
            speakers.popitem(last=False)[1].close()
        return speakers[key]

    def say(text: str, voice: str | None, style: str | None) -> None:
        with speaking:
            get_speaker(voice, style).speak(text)

    async def speak(
        text: str,
        voice_name: str | None = None,
        instruction: str | None = None,
    ) -> bool:
        """Say *text* aloud to the user and return once it has been said.

        Keep it to a sentence or two: the user waits while it plays.

        Args:
          text: What to say.
          voice_name: The voice to use (e.g. "alloy" or "coral" on OpenAI,
            "Sulafat" or "Puck" on Gemini). Leave empty for the default.
          instruction: How to say it (e.g. "cheerful", "whisper slowly").
            Honoured by OpenAI's gpt-4o-mini-tts and by Gemini.
        """
        await asyncio.to_thread(say, text, voice_name, instruction)
        return True

    if tool_name is not None:
        speak.__name__ = tool_name
    if tool_description is not None:
        speak.__doc__ = tool_description
    return speak
