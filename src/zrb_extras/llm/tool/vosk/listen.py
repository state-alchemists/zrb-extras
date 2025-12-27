import asyncio
import json
import queue
import time
from collections import deque
from typing import Any, Callable, Coroutine

import numpy as np
import sounddevice as sd
from vosk import KaldiRecognizer, Model
from zrb import AnyContext

from .default_config import MODEL_LANG, SAMPLE_RATE


def create_listen_tool(
    model_lang: str = MODEL_LANG,
    model_path: str | None = None,
    sample_rate: int = SAMPLE_RATE,
    tool_name: str | None = None,
    tool_description: str | None = None,
) -> Callable[[AnyContext], Coroutine[Any, Any, str]]:
    """
    Factory to create a listen tool using Vosk (offline STT).
    """
    # Load model once if possible, or lazy load.
    # For a factory, we might want to load it lazily inside the function
    # or let the user manage it.
    # To keep it simple and safe for multiprocess/thread, we'll load inside the closure or lazily.
    # However, loading model takes time.

    _model = None

    def get_model():
        nonlocal _model
        if _model is None:
            if model_path:
                _model = Model(model_path)
            else:
                _model = Model(lang=model_lang)
        return _model

    async def listen(ctx: AnyContext) -> str:
        """Listens for and transcribes speech using Vosk."""
        model = get_model()
        rec = KaldiRecognizer(model, sample_rate)

        q = queue.Queue()

        def callback(indata, frames, time_info, status):
            if status:
                print(status, flush=True)
            q.put(bytes(indata))

        ctx.print("Listening (Vosk)...", plain=True)

        # Configs for VAD
        SILENCE_THRESHOLD = 0.01
        MAX_SILENCE = 1.5
        CHANNELS = 1

        # Queue for audio blocks
        audio_q = asyncio.Queue()

        def audio_callback(indata, frames, time_info, status):
            audio_q.put_nowait(indata.copy())

        # Buffer for pre-speech audio
        PRE_BUFFER_DURATION = 0.5
        pre_buffer_size = int(PRE_BUFFER_DURATION * sample_rate / 1024)
        pre_buffer = deque(maxlen=pre_buffer_size)

        recorded_audio = []

        with sd.InputStream(
            samplerate=sample_rate, channels=CHANNELS, callback=audio_callback
        ):
            # 1. Wait for speech
            ctx.print("Waiting for speech...", plain=True)
            while True:
                block = await audio_q.get()
                pre_buffer.append(block)
                volume_norm = np.linalg.norm(block) / len(block)
                if volume_norm > SILENCE_THRESHOLD:
                    ctx.print("Speech detected...", plain=True)
                    recorded_audio.extend(pre_buffer)
                    recorded_audio.append(block)
                    break

            # 2. Record until silence
            silence_start = None
            while True:
                block = await audio_q.get()
                recorded_audio.append(block)
                volume_norm = np.linalg.norm(block) / len(block)

                if volume_norm < SILENCE_THRESHOLD:
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start > MAX_SILENCE:
                        ctx.print("End of speech.", plain=True)
                        break
                else:
                    silence_start = None

        # 3. Transcribe
        # Concatenate audio
        data_np = np.concatenate(recorded_audio, axis=0)
        # Convert to int16 for Vosk
        data_int16 = (data_np * 32767).astype(np.int16)
        data_bytes = data_int16.tobytes()

        if rec.AcceptWaveform(data_bytes):
            res = json.loads(rec.Result())
            return res.get("text", "")
        else:
            res = json.loads(rec.FinalResult())
            return res.get("text", "")

    if tool_name is not None:
        listen.__name__ = tool_name
    if tool_description is not None:
        listen.__doc__ = tool_description
    return listen
