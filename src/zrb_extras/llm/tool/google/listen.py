import io
from typing import TYPE_CHECKING, Any, Callable, Coroutine

from typing_extensions import TypedDict
from zrb import AnyContext

from zrb_extras.llm.tool.google.client import get_client
from zrb_extras.llm.tool.google.default_config import (
    CHANNELS,
    MAX_SILENCE,
    SAMPLE_RATE,
    SILENCE_THRESHOLD,
    STT_MODEL,
)
from zrb_extras.llm.tool.vad import record_until_silence

if TYPE_CHECKING:
    from google import genai
    from google.genai import types


class MultiSpeakerVoice(TypedDict):
    speaker: str
    voice: str


def create_listen_tool(
    client: "genai.Client | None" = None,
    api_key: str | None = None,
    stt_model: str | None = None,
    sample_rate: int | None = None,
    channels: int | None = None,
    silence_threshold: float | None = None,
    max_silence: float | None = None,
    text_processor: None | Callable[[str], str] = None,
    safety_settings: "list[types.SafetySetting] | None" = None,
    tool_name: str | None = None,
    tool_description: str | None = None,
) -> Callable[[AnyContext], Coroutine[Any, Any, str]]:
    """
    Factory to create a configurable listen tool.
    """
    stt_model = stt_model or STT_MODEL
    sample_rate = sample_rate if sample_rate is not None else SAMPLE_RATE
    channels = channels if channels is not None else CHANNELS
    silence_threshold = (
        silence_threshold
        if silence_threshold is not None else SILENCE_THRESHOLD
    )
    max_silence = max_silence if max_silence is not None else MAX_SILENCE

    async def listen(ctx: AnyContext) -> str:
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
                "google-genai dependencies are not installed. "
                "Please install zrb-extras[google-genai] or zrb-extras[all]."
            )

        # Warm up the sound device to prevent ALSA timeout
        with sd.Stream(samplerate=sample_rate, channels=channels):
            pass

        # Record audio
        audio_data = await record_until_silence(
            ctx,
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
        transcribed_text = _transcribe_audio_bytes(
            ctx,
            client=get_client(client, api_key),
            audio_bytes=audio_bytes,
            stt_model=stt_model,
            safety_settings=safety_settings,
        )
        if text_processor is None:
            return transcribed_text
        return text_processor(transcribed_text)

    if tool_name is not None:
        listen.__name__ = tool_name
    if tool_description is not None:
        listen.__doc__ = tool_description
    return listen


def _transcribe_audio_bytes(
    ctx: AnyContext,
    client: "genai.Client",
    audio_bytes: bytes,
    stt_model: str,
    safety_settings: "list[types.SafetySetting] | None" = None,
) -> str:
    try:
        from google.genai import types
    except ImportError:
        raise ImportError(
            "google-genai is not installed. "
            "Please install zrb-extras[google-genai] or zrb-extras[all]."
        )

    # Ask model to transcribe (inline audio + instruction style)
    ctx.print("Requesting transcription...", plain=True)
    resp = client.models.generate_content(
        model=stt_model,
        contents=[
            types.Part(inline_data=types.Blob(mime_type="audio/wav", data=audio_bytes)),
            "Please transcribe the audio exactly.",
        ],
        config=types.GenerateContentConfig(safety_settings=safety_settings),
    )
    # response.text is the canonical convenience property for text outputs
    text = (resp.text or "").strip()
    ctx.print("Transcription result:", repr(text), plain=True)
    return text
