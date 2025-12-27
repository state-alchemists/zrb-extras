import asyncio
from typing import Any, Callable, Coroutine

from zrb import AnyContext


def create_speak_tool(
    voice_name: str | None = None,  # Renamed from voice_id to match Google
    rate: int | None = None,
    volume: float | None = None,
    tool_name: str | None = None,
    tool_description: str | None = None,
) -> Callable[[AnyContext, str, str | None], Coroutine[Any, Any, bool]]:
    """
    Factory to create a speak tool using pyttsx3 (offline TTS).
    """

    async def speak(
        ctx: AnyContext, text: str, voice_name: str | None = voice_name
    ) -> bool:
        """Converts text to speech using pyttsx3."""
        # Capture closure variables to avoid confusion with local args
        factory_rate = rate
        factory_volume = volume

        def _speak_sync():
            try:
                import pyttsx3
            except ImportError:
                raise ImportError(
                    "pyttsx3 is not installed. Please install zrb-extras[tts] or zrb-extras[all]."
                )

            try:
                engine = pyttsx3.init()
                final_voice = voice_name
                if final_voice:
                    engine.setProperty("voice", final_voice)
                if factory_rate:
                    engine.setProperty("rate", factory_rate)
                if factory_volume is not None:
                    engine.setProperty("volume", factory_volume)
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
