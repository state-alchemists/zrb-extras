from __future__ import annotations

import asyncio
import io
import time
from collections import deque
from typing import TYPE_CHECKING, Any, Callable, Coroutine

from zrb import AnyContext

from zrb_extras.llm.tool.openai.client import get_client
from zrb_extras.llm.tool.openai.default_config import (
    CHANNELS,
    MAX_SILENCE,
    SAMPLE_RATE,
    SILENCE_THRESHOLD,
    STT_MODEL,
)

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
) -> Callable[[AnyContext], Coroutine[Any, Any, str]]:
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

    async def listen(ctx: AnyContext) -> str:
        """Listens for and transcribes the user's verbal response.

        Use this tool to capture speech from the user.
        The tool records audio from the microphone, automatically detects when the user
        has finished speaking, and returns the transcribed text.

        Returns:
            The transcribed text from the user's speech.
        """
        import numpy as np
        import sounddevice as sd
        import soundfile as sf

        # Warm up the sound device to prevent ALSA timeout
        with sd.Stream(samplerate=sample_rate, channels=channels):
            pass

        # Record audio
        audio_data = await _record_until_silence(
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
        transcribed_text = await _transcribe_audio_bytes(
            ctx,
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


async def _record_until_silence(
    ctx: AnyContext,
    sample_rate: int,
    channels: int,
    silence_threshold: float,
    max_silence: float,
):
    """Wait for speech to start, record, then stop after silence."""
    import numpy as np
    import sounddevice as sd

    q = asyncio.Queue()
    rec_data = []
    PRE_BUFFER_DURATION = 0.5  # seconds
    pre_buffer_size = int(PRE_BUFFER_DURATION * sample_rate / 1024)  # in blocks
    pre_buffer = deque(maxlen=pre_buffer_size)

    def callback(indata, frames, time_info, status):
        q.put_nowait(indata.copy())

    ctx.print("Waiting for speech...", plain=True)
    with sd.InputStream(samplerate=sample_rate, channels=channels, callback=callback):
        # First detect speech
        while True:
            block = await q.get()
            pre_buffer.append(block)
            volume_norm = np.linalg.norm(block) / len(block)
            if volume_norm > silence_threshold:
                ctx.print("Speech detected, recording...", plain=True)
                rec_data.extend(pre_buffer)
                rec_data.append(block)
                break

        # Record until silence for max_silence seconds
        silence_start = None
        while True:
            block = await q.get()
            rec_data.append(block)
            volume_norm = np.linalg.norm(block) / len(block)
            ctx.print(f"Volume: {volume_norm:.4f}", end="\r", plain=True)
            if volume_norm < silence_threshold:
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start > max_silence:
                    ctx.print("\nSilence detected, stop recording.", plain=True)
                    break
            else:
                silence_start = None

    # Combine into a single numpy array
    audio_data = np.concatenate(rec_data, axis=0)
    # convert to 16-bit integers
    audio_data = (audio_data * 32767).astype(np.int16)
    return audio_data


async def _transcribe_audio_bytes(
    ctx: AnyContext,
    client: AsyncOpenAI,
    audio_bytes: bytes,
    stt_model: str,
) -> str:
    # Ask model to transcribe
    ctx.print("Requesting transcription...", plain=True)

    # OpenAI requires a filename for the file-like object
    file_obj = io.BytesIO(audio_bytes)
    file_obj.name = "audio.wav"

    resp = await client.audio.transcriptions.create(
        model=stt_model,
        file=file_obj,
    )

    text = (resp.text or "").strip()
    ctx.print("Transcription result:", repr(text), plain=True)
    return text
