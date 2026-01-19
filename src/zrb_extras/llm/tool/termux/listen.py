import shutil
import subprocess
from typing import Any, Callable, Coroutine

from zrb_extras.llm.tool.vad import record_until_silence

from .default_config import (
    CHANNELS,
    MAX_SILENCE,
    SAMPLE_RATE,
    SILENCE_THRESHOLD,
)


def create_listen_tool(
    sample_rate: int | None = None,
    channels: int | None = None,
    silence_threshold: float | None = None,
    max_silence: float | None = None,
    text_processor: Callable[[str], str] | None = None,
    tool_name: str | None = None,
    tool_description: str | None = None,
) -> Callable[[], Coroutine[Any, Any, str]]:
    """
    Factory to create a listen tool using Termux API.
    Requires 'termux-speech-to-text' to be installed and available in path.
    """
    sample_rate = sample_rate if sample_rate is not None else SAMPLE_RATE
    channels = channels if channels is not None else CHANNELS
    silence_threshold = (
        silence_threshold if silence_threshold is not None else SILENCE_THRESHOLD
    )
    max_silence = max_silence if max_silence is not None else MAX_SILENCE

    async def listen() -> str:
        """Listens for and transcribes the user's verbal response using Termux native UI.

        Returns:
            The transcribed text from the user's speech.
        """
        # Try to use VAD if dependencies are available to 'wait' for speech
        # before popping up the Termux STT dialog.
        # This helps in noisy environments where you don't want the dialog
        # to stay open or trigger prematurely.
        try:
            # We just use VAD to detect the START of speech.
            # Once speech is detected, we invoke the native Termux STT which has its own recording.
            # This is a bit of a hybrid approach because we can't easily feed our audio to termux-speech-to-text.
            print("Monitoring for speech (VAD)...")
            await record_until_silence(
                sample_rate=sample_rate,
                channels=channels,
                silence_threshold=silence_threshold,
                max_silence=0.1,  # Very short max_silence because we just want to detect the start
            )
        except (ImportError, Exception):
            # Fallback to direct call if sounddevice/numpy missing or fails
            pass

        if not shutil.which("termux-speech-to-text"):
            raise RuntimeError(
                "termux-speech-to-text not found. Is Termux API installed?"
            )

        print("Listening (Termux UI)...")
        # Run blocking subprocess as it invokes a UI
        try:
            result = subprocess.run(
                ["termux-speech-to-text"], capture_output=True, text=True, check=True
            )
            text = result.stdout.strip()
            print(f"Heard: {text}")

            if text_processor:
                return text_processor(text)
            return text
        except subprocess.CalledProcessError as e:
            print(f"Error calling Termux STT: {e}")
            return ""

    if tool_name is not None:
        listen.__name__ = tool_name
    if tool_description is not None:
        listen.__doc__ = tool_description
    return listen
