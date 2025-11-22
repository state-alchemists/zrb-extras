import asyncio
import tempfile
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, Coroutine

import numpy as np
import sounddevice as sd
import soundfile as sf
from google import genai
from google.genai import types
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


class MultiSpeakerVoice(TypedDict):
    speaker: str
    voice: str


def create_listen_tool(
    client: genai.Client | None = None,
    api_key: str | None = None,
    stt_model: str = STT_MODEL,
    sample_rate: int = SAMPLE_RATE,
    channels: int = CHANNELS,
    silence_threshold: float = SILENCE_THRESHOLD,
    max_silence: float = MAX_SILENCE,
    text_processor: None | Callable[[str], str] = None,
    safety_settings: list[types.SafetySetting] | None = None,
    tool_name: str | None = None,
    tool_description: str | None = None,
) -> Callable[[AnyContext], Coroutine[Any, Any, str]]:
    """
    Factory to create a configurable listen tool.
    """

    async def listen(ctx: AnyContext) -> str:
        """Listens for and transcribes the user's verbal response.

        Use this tool to capture speech from the user.
        The tool records audio from the microphone, automatically detects when the user
        has finished speaking, and returns the transcribed text.

        Returns:
            The transcribed text from the user's speech.
        """
        # Warm up the sound device to prevent ALSA timeout
        with sd.Stream(samplerate=sample_rate, channels=channels):
            pass
        tmpdir = Path(tempfile.mkdtemp(prefix="gemini_stt_tts_"))
        in_wav = tmpdir / "input.wav"
        # Record audio
        audio_data = await _record_until_silence(
            ctx,
            sample_rate=sample_rate,
            channels=channels,
            silence_threshold=silence_threshold,
            max_silence=max_silence,
        )
        # Normalize and write to file
        audio_data = audio_data / np.max(np.abs(audio_data))
        sf.write(str(in_wav), audio_data, sample_rate, subtype="PCM_16")
        # Transcribe
        transcribed_text = _transcribe_file(
            ctx,
            client=get_client(client, api_key),
            wav_path=str(in_wav),
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


async def _record_until_silence(
    ctx: AnyContext,
    sample_rate: int,
    channels: int,
    silence_threshold: float,
    max_silence: float,
):
    """Wait for speech to start, record, then stop after silence."""
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


def _transcribe_file(
    ctx: AnyContext,
    client: genai.Client,
    wav_path: str,
    stt_model: str,
    safety_settings: list[types.SafetySetting] | None = None,
) -> str:
    # Upload file to Gemini Files API
    ctx.print("Uploading for transcription...", plain=True)
    uploaded = client.files.upload(file=wav_path)
    # Ask model to transcribe (upload + instruction style)
    ctx.print("Requesting transcription...", plain=True)
    resp = client.models.generate_content(
        model=stt_model,
        contents=[uploaded, "Please transcribe the uploaded audio exactly."],
        config=types.GenerateContentConfig(safety_settings=safety_settings),
    )
    # response.text is the canonical convenience property for text outputs
    text = (resp.text or "").strip()
    ctx.print("Transcription result:", repr(text), plain=True)
    return text
