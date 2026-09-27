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
        cmd = create_termux_tts_command(
            text,
            language=language,
            voice_name=voice_name,
            engine=engine,
            region=region,
            rate=rate,
            pitch=pitch,
            stream=stream,
        )

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


def create_termux_tts_command(
    text: str,
    language: str | None = None,
    voice_name: str | None = None,
    engine: str | None = None,
    region: str | None = None,
    rate: float | None = None,
    pitch: float | None = None,
    stream: str | None = None,
) -> list[str]:
    """The `termux-tts-speak` argv that says *text*."""
    cmd = ["termux-tts-speak"]
    for flag, value in (
        ("-l", language),
        ("-e", engine),
        ("-n", region),
        ("-v", voice_name),
        ("-r", rate),
        ("-p", pitch),
        ("-s", stream),
    ):
        if value:
            cmd.extend([flag, str(value)])
    cmd.append(text)
    return cmd
