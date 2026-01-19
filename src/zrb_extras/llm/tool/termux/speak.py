import shutil
import subprocess
from typing import Any, Callable, Coroutine

from .default_config import (
    TTS_ENGINE,
    TTS_LANGUAGE,
    TTS_PITCH,
    TTS_RATE,
    TTS_REGION,
    TTS_STREAM,
    TTS_VOICE_NAME,
)


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
) -> Callable[[str, str | None], Coroutine[Any, Any, bool]]:
    """
    Factory to create a speak tool using Termux API.
    """
    language = language or TTS_LANGUAGE
    voice_name = voice_name or TTS_VOICE_NAME
    engine = engine or TTS_ENGINE
    region = region or TTS_REGION
    rate = rate if rate is not None else TTS_RATE
    pitch = pitch if pitch is not None else TTS_PITCH
    stream = stream or TTS_STREAM

    async def speak(
        text: str, voice_name: str | None = voice_name
    ) -> bool:
        """Converts text to speech using Termux native TTS."""
        if not shutil.which("termux-tts-speak"):
            raise RuntimeError("termux-tts-speak not found. Is Termux API installed?")

        print(f"Speaking: {text}")

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
            print(f"Error calling Termux TTS: {e}")
            return False

    if tool_name is not None:
        speak.__name__ = tool_name
    if tool_description is not None:
        speak.__doc__ = tool_description
    return speak
