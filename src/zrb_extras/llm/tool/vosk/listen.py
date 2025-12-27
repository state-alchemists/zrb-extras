import asyncio
import json
import time
from collections import deque
from typing import TYPE_CHECKING, Any, Callable, Coroutine

from zrb import AnyContext

from .default_config import MODEL_LANG, SAMPLE_RATE

if TYPE_CHECKING:
    import numpy as np


def create_listen_tool(
    model_lang: str | None = None,
    model_path: str | None = None,
    model_name: str | None = None,
    sample_rate: int | None = None,
    channels: int | None = None,
    silence_threshold: float | None = None,
    max_silence: float | None = None,
    text_processor: Callable[[str], str] | None = None,
    tool_name: str | None = None,
    tool_description: str | None = None,
) -> Callable[[AnyContext], Coroutine[Any, Any, str]]:
    """
    Factory to create a listen tool using Vosk (offline STT).
    """
    model_lang = model_lang or MODEL_LANG
    sample_rate = sample_rate if sample_rate is not None else SAMPLE_RATE
    channels = channels if channels is not None else 1
    silence_threshold = silence_threshold if silence_threshold is not None else 0.01
    max_silence = max_silence if max_silence is not None else 1.5

    # Lazy load the model to avoid overhead on every call
    _model = None

    def get_model():
        try:
            from vosk import Model
        except ImportError:
            # This should be caught by the outer listen try/except or propagate
            raise ImportError(
                "vosk is not installed. Please install zrb-extras[vosk] or zrb-extras[all]."
            )

        nonlocal _model
        if _model is None:
            # Prioritize path, then name, then lang (which defaults to en-us)
            _model = Model(
                model_path=model_path, model_name=model_name, lang=model_lang
            )
        return _model

    async def listen(ctx: AnyContext) -> str:
        """Listens for and transcribes speech using Vosk.

        The tool records audio from the microphone, automatically detects when the user
        has finished speaking, and returns the transcribed text.
        """
        try:
            import sounddevice as sd
            from vosk import KaldiRecognizer
        except ImportError:
            raise ImportError(
                "vosk or sounddevice is not installed. Please install zrb-extras[vosk] or zrb-extras[all]."
            )

        # Get model (this might trigger download on first run, which blocks)
        # Ideally this should be run in a thread if it blocks for a long time,
        # but Model() constructor is C-extension based.
        # Let's wrap it in asyncio.to_thread just in case to avoid blocking the loop
        model = await asyncio.to_thread(get_model)

        rec = KaldiRecognizer(model, sample_rate)

        # Warm up
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

        # Transcribe
        data_bytes = audio_data.tobytes()

        ctx.print("Transcribing...", plain=True)
        result_text = ""
        if rec.AcceptWaveform(data_bytes):
            res = json.loads(rec.Result())
            result_text = res.get("text", "")
        else:
            res = json.loads(rec.FinalResult())
            result_text = res.get("text", "")

        ctx.print(f"Vosk Heard: {result_text}", plain=True)

        if text_processor:
            return text_processor(result_text)
        return result_text

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
) -> "np.ndarray":
    """Wait for speech to start, record, then stop after silence."""
    try:
        import numpy as np
        import sounddevice as sd
    except ImportError:
        raise ImportError(
            "numpy or sounddevice is not installed. Please install zrb-extras[vosk] or zrb-extras[all]."
        )

    q = asyncio.Queue()
    rec_data = []
    PRE_BUFFER_DURATION = 0.5  # seconds
    pre_buffer_size = int(PRE_BUFFER_DURATION * sample_rate / 1024)  # in blocks
    pre_buffer = deque(maxlen=pre_buffer_size)

    def callback(indata, frames, time_info, status):
        q.put_nowait(indata.copy())

    ctx.print("Waiting for speech (Vosk)...", plain=True)
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
