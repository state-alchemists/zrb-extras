import asyncio
import json
from typing import Any, Callable, Coroutine

from zrb import AnyContext

from zrb_extras.llm.tool.vad import record_until_silence

from .default_config import MODEL_LANG, SAMPLE_RATE


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
                "vosk or sounddevice is not installed. "
                "Please install zrb-extras[vosk] or zrb-extras[all]."
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
        audio_data = await record_until_silence(
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
