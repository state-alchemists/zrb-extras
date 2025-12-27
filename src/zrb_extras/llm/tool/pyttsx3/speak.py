import asyncio
from typing import Any, Callable, Coroutine

import pyttsx3
from zrb import AnyContext


def create_speak_tool(
    voice_id: str | None = None,
    rate: int | None = None,
    volume: float | None = None,
    tool_name: str | None = None,
    tool_description: str | None = None,
) -> Callable[[AnyContext, str], Coroutine[Any, Any, bool]]:
    """
    Factory to create a speak tool using pyttsx3 (offline TTS).
    """

    async def speak(ctx: AnyContext, text: str) -> bool:
        """Converts text to speech using pyttsx3."""
        # pyttsx3 engine initialization might need to be thread-local or carefully managed
        # initializing it inside the function is safest for infrequent use,
        # though less efficient than reusing.

        def _speak_sync():
            try:
                engine = pyttsx3.init()
                if voice_id:
                    engine.setProperty("voice", voice_id)
                if rate:
                    engine.setProperty("rate", rate)
                if volume is not None:
                    engine.setProperty("volume", volume)

                ctx.print(f"Speaking (pyttsx3): {text}", plain=True)
                engine.say(text)
                engine.runAndWait()
                return True
            except Exception as e:
                ctx.print(f"Error in pyttsx3: {e}", plain=True)
                return False

        return await asyncio.to_thread(_speak_sync)

    if tool_name is not None:
        speak.__name__ = tool_name
    if tool_description is not None:
        speak.__doc__ = tool_description
    return speak
