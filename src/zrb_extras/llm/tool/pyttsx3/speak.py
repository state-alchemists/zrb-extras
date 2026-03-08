import asyncio
import os
from typing import Any, Callable, Coroutine


# Environment variable names for configuration
ENV_VOICE_NAME = "PYTTSX3_VOICE_NAME"
ENV_VOICE_RATE = "PYTTSX3_VOICE_RATE"
ENV_VOICE_VOLUME = "PYTTSX3_VOICE_VOLUME"


def create_speak_tool(
    voice_name: str | None = None,  # Renamed from voice_id to match Google
    rate: int | None = None,
    volume: float | None = None,
    tool_name: str | None = None,
    tool_description: str | None = None,
) -> Callable[[str, str | None], Coroutine[Any, Any, bool]]:
    """
    Factory to create a speak tool using pyttsx3 (offline TTS).
    
    Configuration can be done via:
    - Function parameters (highest priority)
    - Environment variables: PYTTSX3_VOICE_NAME, PYTTSX3_VOICE_RATE, PYTTSX3_VOICE_VOLUME
    
    To see available voices, run:
        python -c "import pyttsx3; e=pyttsx3.init(); print([v.id for v in e.getProperty('voices')])"
    
    On Linux, you may need to install espeak-ng for better voice quality:
        sudo apt install espeak-ng
    
    Common voice IDs on Linux (espeak-ng):
        - english-us (default)
        - english-us+m1 (male)
        - english-us+f1 (female)
    """

    async def speak(
        text: str, voice_name: str | None = voice_name
    ) -> bool:
        """Converts text to speech using pyttsx3."""
        import sys
        if sys.platform == "darwin":
            print(f"Speaking (say): {text}")
            cmd = ["say", text]
            if voice_name:
                cmd.extend(["-v", voice_name])
            if rate:
                # pyttsx3 rate 200 is roughly normal.
                # say rate 175 is roughly normal.
                # mapping: say_rate = (rate / 200) * 175
                say_rate = int((rate / 200) * 175)
                cmd.extend(["-r", str(say_rate)])
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL
            )
            await process.wait()
            return True

        # Capture closure variables to avoid confusion with local args
        # Also check environment variables for configuration
        factory_rate = rate if rate is not None else _get_env_int(ENV_VOICE_RATE, 150)
        factory_volume = volume if volume is not None else _get_env_float(ENV_VOICE_VOLUME, 1.0)

        def _speak_sync():
            try:
                import pyttsx3
            except ImportError:
                raise ImportError(
                    "pyttsx3 is not installed. Please install zrb-extras[vosk] or zrb-extras[all]."
                )

            try:
                engine = pyttsx3.init()
                
                # Resolve voice: parameter > env var > None (system default)
                final_voice = voice_name or os.environ.get(ENV_VOICE_NAME)
                if final_voice:
                    engine.setProperty("voice", final_voice)
                
                # Set rate (words per minute, 150 is a good default for clarity)
                if factory_rate:
                    engine.setProperty("rate", factory_rate)
                
                # Set volume (0.0 to 1.0)
                if factory_volume is not None:
                    engine.setProperty("volume", factory_volume)
                
                print(f"Speaking (pyttsx3): {text}")
                engine.say(text)
                engine.runAndWait()
                return True
            except Exception as e:
                print(f"Error in pyttsx3: {e}")
                return False

        return await asyncio.to_thread(_speak_sync)

    if tool_name is not None:
        speak.__name__ = tool_name
    if tool_description is not None:
        speak.__doc__ = tool_description
    return speak


def _get_env_int(name: str, default: int) -> int:
    """Get integer from environment variable with default."""
    try:
        val = os.environ.get(name)
        if val is not None:
            return int(val)
    except ValueError:
        pass
    return default


def _get_env_float(name: str, default: float) -> float:
    """Get float from environment variable with default."""
    try:
        val = os.environ.get(name)
        if val is not None:
            return float(val)
    except ValueError:
        pass
    return default


def list_available_voices() -> list[dict[str, str]]:
    """
    List all available pyttsx3 voices.
    
    Returns a list of dicts with 'id', 'name', and 'languages' keys.
    """
    try:
        import pyttsx3
    except ImportError:
        raise ImportError(
            "pyttsx3 is not installed. Please install zrb-extras[vosk] or zrb-extras[all]."
        )
    
    engine = pyttsx3.init()
    voices = engine.getProperty("voices")
    result = []
    for voice in voices:
        result.append({
            "id": voice.id,
            "name": voice.name,
            "languages": list(voice.languages) if voice.languages else [],
        })
    return result
