import asyncio
import io
from typing import TYPE_CHECKING, Any, Callable, Coroutine

from zrb import AnyContext

from zrb_extras.llm.tool.openai.client import get_client
from zrb_extras.llm.tool.openai.default_config import TTS_MODEL, VOICE_NAME

if TYPE_CHECKING:
    from openai import AsyncOpenAI


def create_speak_tool(
    client: "AsyncOpenAI | None" = None,
    api_key: str | None = None,
    base_url: str | None = None,
    tts_model: str | None = None,
    voice_name: str | None = None,
    sample_rate_out: int | None = None,
    tool_name: str | None = None,
    tool_description: str | None = None,
) -> Callable[[AnyContext, str, str | None], Coroutine[Any, Any, bool]]:
    tts_model = tts_model or TTS_MODEL
    voice_name = voice_name or VOICE_NAME
    sample_rate_out = sample_rate_out if sample_rate_out is not None else 24000

    async def speak(
        ctx: AnyContext,
        text: str,
        instruction: str | None = None,
        voice_name: str | None = voice_name,
    ) -> bool:
        """Converts text to speech and plays it aloud.

        Use this tool to verbally communicate with the user.
        The system will play the generated audio and return `True` upon completion.
        Keep the text concise (a sentence or two) for a faster response.

        Args:
          text: The text to be spoken.
          instruction: Additional instruction to control the voice of your generated audio.
          voice_name: The voice to use (e.g., "alloy", "echo", "fable", "onyx", "nova", "shimmer").

        Returns:
          True if speech was successfully generated and played, False otherwise.
        """
        return await _synthesize_and_play(
            ctx,
            client=get_client(client, api_key, base_url),
            text=text,
            instruction=instruction,
            tts_model=tts_model,
            voice_name=voice_name,
            sample_rate_out=sample_rate_out,
        )

    if tool_name is not None:
        speak.__name__ = tool_name
    if tool_description is not None:
        speak.__doc__ = tool_description
    return speak


async def _synthesize_and_play(
    ctx: AnyContext,
    client: "AsyncOpenAI",
    text: str,
    instruction: str | None,
    tts_model: str,
    voice_name: str | None = None,
    sample_rate_out: int = 24000,
):
    try:
        import sounddevice as sd
        import soundfile as sf
        from openai import Omit, omit
    except ImportError:
        raise ImportError(
            "openai, sounddevice, or soundfile is not installed. Please install zrb-extras[openai] or zrb-extras[all]."
        )
    if not text:
        text = "I have nothing to say."

    # Resolve voice name
    selected_voice = voice_name or VOICE_NAME

    ctx.print("Requesting TTS...", plain=True)

    # OpenAI TTS request
    response = await client.audio.speech.create(
        model=tts_model,
        voice=selected_voice,
        input=text,
        instructions=omit if instruction is None else instruction,
        response_format="wav",  # Get WAV directly
    )

    # Extract audio content
    ctx.print("Extracting audio...", plain=True)
    audio_content = b""
    for chunk in response.iter_bytes():
        audio_content += chunk

    # Prepare an in-memory WAV buffer
    ctx.print("Preparing audio buffer...", plain=True)
    buf = io.BytesIO(audio_content)

    # Now decode and play directly from buffer
    data, sr = sf.read(buf)

    # If sample rate needs adjustment or verify?
    # Usually soundfile/sounddevice handles it, but let's check if we need to force it.
    # The default response format 'wav' from OpenAI should be playable.

    ctx.print("Playing audio...", plain=True)
    sd.play(data, sr)
    await asyncio.to_thread(sd.wait)
    return True
