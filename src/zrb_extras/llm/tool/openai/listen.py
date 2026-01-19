import io
from typing import TYPE_CHECKING, Any, Callable, Coroutine

from zrb_extras.llm.tool.openai.client import get_client
from zrb_extras.llm.tool.openai.default_config import (
    CHANNELS,
    MAX_SILENCE,
    SAMPLE_RATE,
    SILENCE_THRESHOLD,
    STT_MODEL,
)
from zrb_extras.llm.tool.vad import record_until_silence

if TYPE_CHECKING:
    from openai import AsyncOpenAI


def create_listen_tool(
    client: "AsyncOpenAI | None" = None,
    api_key: str | None = None,
    base_url: str | None = None,
    stt_model: str | None = None,
    sample_rate: int | None = None,
    channels: int | None = None,
    silence_threshold: float | None = None,
    max_silence: float | None = None,
    text_processor: None | Callable[[str], str] = None,
    tool_name: str | None = None,
    tool_description: str | None = None,
) -> Callable[[], Coroutine[Any, Any, str]]:
    """
    Factory to create a configurable listen tool using OpenAI.
    """
    stt_model = stt_model or STT_MODEL
    sample_rate = sample_rate if sample_rate is not None else SAMPLE_RATE
    channels = channels if channels is not None else CHANNELS
    silence_threshold = (
        silence_threshold if silence_threshold is not None else SILENCE_THRESHOLD
    )
    max_silence = max_silence if max_silence is not None else MAX_SILENCE

    async def listen() -> str:
        """Listens for and transcribes the user's verbal response.

        Use this tool to capture speech from the user.
        The tool records audio from the microphone, automatically detects when the user
        has finished speaking, and returns the transcribed text.

        Returns:
            The transcribed text from the user's speech.
        """
        try:
            import numpy as np
            import sounddevice as sd
            import soundfile as sf
        except ImportError:
            raise ImportError(
                "openai dependencies are not installed. "
                "Please install zrb-extras[openai] or zrb-extras[all]."
            )

        # Warm up the sound device to prevent ALSA timeout
        with sd.Stream(samplerate=sample_rate, channels=channels):
            pass

        # Record audio
        audio_data = await record_until_silence(
            sample_rate=sample_rate,
            channels=channels,
            silence_threshold=silence_threshold,
            max_silence=max_silence,
        )
        # Normalize and write to memory buffer
        audio_data = audio_data / np.max(np.abs(audio_data))
        buf = io.BytesIO()
        sf.write(buf, audio_data, sample_rate, format="WAV", subtype="PCM_16")
        audio_bytes = buf.getvalue()

        # Transcribe
        transcribed_text = await _transcribe_audio_bytes(
            client=get_client(client, api_key, base_url),
            audio_bytes=audio_bytes,
            stt_model=stt_model,
        )
        if text_processor is None:
            return transcribed_text
        return text_processor(transcribed_text)

    if tool_name is not None:
        listen.__name__ = tool_name
    if tool_description is not None:
        listen.__doc__ = tool_description
    return listen


async def _transcribe_audio_bytes(
    client: "AsyncOpenAI",
    audio_bytes: bytes,
    stt_model: str,
) -> str:
    # Ask model to transcribe
    print("Requesting transcription...")

    # OpenAI requires a filename for the file-like object
    file_obj = io.BytesIO(audio_bytes)
    file_obj.name = "audio.wav"

    resp = await client.audio.transcriptions.create(
        model=stt_model,
        file=file_obj,
    )

    text = (resp.text or "").strip()
    print("Transcription result:", repr(text))
    return text
