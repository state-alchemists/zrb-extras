import shutil
import subprocess
from typing import Any, Callable, Coroutine

from zrb import AnyContext


def create_speak_tool(
    tool_name: str | None = None,
    tool_description: str | None = None,
) -> Callable[[AnyContext, str], Coroutine[Any, Any, bool]]:
    """
    Factory to create a speak tool using Termux API.
    """

    async def speak(ctx: AnyContext, text: str) -> bool:
        """Converts text to speech using Termux native TTS."""
        if not shutil.which("termux-tts-speak"):
            raise RuntimeError("termux-tts-speak not found. Is Termux API installed?")

        ctx.print(f"Speaking: {text}", plain=True)
        try:
            subprocess.run(["termux-tts-speak", text], check=True)
            return True
        except subprocess.CalledProcessError as e:
            ctx.print(f"Error calling Termux TTS: {e}", plain=True)
            return False

    if tool_name is not None:
        speak.__name__ = tool_name
    if tool_description is not None:
        speak.__doc__ = tool_description
    return speak
