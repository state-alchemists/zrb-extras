import shutil
import subprocess
from typing import Any, Callable, Coroutine

from zrb import AnyContext


def create_speak_tool(
    language: str | None = None,
    voice_name: str | None = None,  # Maps to -v (variant)
    engine: str | None = None,  # Maps to -e
    region: str | None = None,  # Maps to -n
    rate: float | None = None,
    pitch: float | None = None,
    stream: str | None = None,
    tool_name: str | None = None,
    tool_description: str | None = None,
) -> Callable[[AnyContext, str, str | None], Coroutine[Any, Any, bool]]:
    """
    Factory to create a speak tool using Termux API.
    """

    async def speak(
        ctx: AnyContext, text: str, voice_name: str | None = voice_name
    ) -> bool:
        """Converts text to speech using Termux native TTS."""
        if not shutil.which("termux-tts-speak"):
            raise RuntimeError("termux-tts-speak not found. Is Termux API installed?")

        ctx.print(f"Speaking: {text}", plain=True)

        cmd = ["termux-tts-speak"]
        if language:
            cmd.extend(["-l", language])
        if engine:
            cmd.extend(["-e", engine])
        if region:
            cmd.extend(["-n", region])

        # Determine voice config.
        # voice_name overrides factory default if provided
        final_voice = voice_name
        if final_voice:
            cmd.extend(["-v", final_voice])

        if rate:
            cmd.extend(["-r", str(rate)])
        if pitch:
            cmd.extend(["-p", str(pitch)])
        if stream:
            cmd.extend(["-s", stream])

        cmd.append(text)

        try:
            subprocess.run(cmd, check=True)
            return True
        except subprocess.CalledProcessError as e:
            ctx.print(f"Error calling Termux TTS: {e}", plain=True)
            return False

    if tool_name is not None:
        speak.__name__ = tool_name
    if tool_description is not None:
        speak.__doc__ = tool_description
    return speak
