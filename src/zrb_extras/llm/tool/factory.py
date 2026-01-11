from typing import TYPE_CHECKING, Any, Callable, Coroutine, Literal

from zrb import AnyContext

from zrb_extras.llm.tool.google import create_listen_tool as create_google_listen_tool  # noqa
from zrb_extras.llm.tool.google import create_speak_tool as create_google_speak_tool  # noqa
from zrb_extras.llm.tool.openai import create_listen_tool as create_openai_listen_tool  # noqa
from zrb_extras.llm.tool.openai import create_speak_tool as create_openai_speak_tool  # noqa
from zrb_extras.llm.tool.pyttsx3 import create_speak_tool as create_pyttsx3_speak_tool  # noqa
from zrb_extras.llm.tool.termux import create_listen_tool as create_termux_listen_tool  # noqa
from zrb_extras.llm.tool.termux import create_speak_tool as create_termux_speak_tool  # noqa
from zrb_extras.llm.tool.vosk import create_listen_tool as create_vosk_listen_tool  # noqa

if TYPE_CHECKING:
    from google import genai
    from google.genai import types
    from openai import AsyncOpenAI

    from zrb_extras.llm.tool.google.speak import (
        MultiSpeakerVoice as GoogleMultiSpeakerVoice,
    )

Mode = Literal["google", "openai", "termux", "vosk"]


def create_listen_tool(
    mode: Mode = "vosk",
    # Common
    tool_name: str | None = None,
    tool_description: str | None = None,
    text_processor: Callable[[str], str] | None = None,
    # Audio Params (Common)
    sample_rate: int | None = None,
    channels: int | None = None,
    silence_threshold: float | None = None,
    max_silence: float | None = None,
    # Google (GenAI)
    genai_client: "genai.Client | None" = None,
    genai_api_key: str | None = None,
    genai_stt_model: str | None = None,
    genai_safety_settings: "list[types.SafetySetting] | None" = None,
    # OpenAI
    openai_client: "AsyncOpenAI | None" = None,
    openai_api_key: str | None = None,
    openai_base_url: str | None = None,
    openai_stt_model: str | None = None,
    # Vosk
    vosk_model_lang: str | None = None,
    vosk_model_path: str | None = None,
    vosk_model_name: str | None = None,
) -> Callable[[AnyContext], Coroutine[Any, Any, str]]:
    """
    Generic factory to create a listen tool for different backends.
    """
    if mode == "google":
        return create_google_listen_tool(
            client=genai_client,
            api_key=genai_api_key,
            stt_model=genai_stt_model,
            safety_settings=genai_safety_settings,
            sample_rate=sample_rate,
            channels=channels,
            silence_threshold=silence_threshold,
            max_silence=max_silence,
            text_processor=text_processor,
            tool_name=tool_name,
            tool_description=tool_description,
        )
    if mode == "openai":
        return create_openai_listen_tool(
            client=openai_client,
            api_key=openai_api_key,
            base_url=openai_base_url,
            stt_model=openai_stt_model,
            sample_rate=sample_rate,
            channels=channels,
            silence_threshold=silence_threshold,
            max_silence=max_silence,
            text_processor=text_processor,
            tool_name=tool_name,
            tool_description=tool_description,
        )
    if mode == "termux":
        return create_termux_listen_tool(
            sample_rate=sample_rate,
            channels=channels,
            silence_threshold=silence_threshold,
            max_silence=max_silence,
            text_processor=text_processor,
            tool_name=tool_name,
            tool_description=tool_description,
        )
    if mode == "vosk":
        return create_vosk_listen_tool(
            model_lang=vosk_model_lang,
            model_path=vosk_model_path,
            model_name=vosk_model_name,
            sample_rate=sample_rate,
            channels=channels,
            silence_threshold=silence_threshold,
            max_silence=max_silence,
            text_processor=text_processor,
            tool_name=tool_name,
            tool_description=tool_description,
        )
    raise ValueError(f"Unknown mode: {mode}")


def create_speak_tool(
    mode: Mode = "vosk",
    # Common
    tool_name: str | None = None,
    tool_description: str | None = None,
    sample_rate_out: int | None = None,
    # Google (GenAI)
    genai_client: "genai.Client | None" = None,
    genai_api_key: str | None = None,
    genai_tts_model: str | None = None,
    genai_voice_name: "str | list[GoogleMultiSpeakerVoice] | None" = None,
    genai_safety_settings: "list[types.SafetySetting] | None" = None,
    # OpenAI
    openai_client: "AsyncOpenAI | None" = None,
    openai_api_key: str | None = None,
    openai_base_url: str | None = None,
    openai_tts_model: str | None = None,
    openai_voice_name: str | None = None,
    # Termux
    termux_language: str | None = None,
    termux_voice_name: str | None = None,
    termux_engine: str | None = None,
    termux_region: str | None = None,
    termux_rate: float | None = None,
    termux_pitch: float | None = None,
    termux_stream: str | None = None,
    # Pyttsx3 (Vosk mode)
    pyttsx3_voice_name: str | None = None,
    pyttsx3_rate: int | None = None,
    pyttsx3_volume: float | None = None,
) -> Callable[[AnyContext, str, Any], Coroutine[Any, Any, bool]]:
    """
    Generic factory to create a speak tool for different backends.
    """
    if mode == "google":
        return create_google_speak_tool(
            client=genai_client,
            api_key=genai_api_key,
            tts_model=genai_tts_model,
            voice_name=genai_voice_name,
            sample_rate_out=sample_rate_out,
            safety_settings=genai_safety_settings,
            tool_name=tool_name,
            tool_description=tool_description,
        )
    if mode == "openai":
        return create_openai_speak_tool(
            client=openai_client,
            api_key=openai_api_key,
            base_url=openai_base_url,
            tts_model=openai_tts_model,
            voice_name=openai_voice_name,
            sample_rate_out=sample_rate_out,
            tool_name=tool_name,
            tool_description=tool_description,
        )
    if mode == "termux":
        return create_termux_speak_tool(
            language=termux_language,
            voice_name=termux_voice_name,
            engine=termux_engine,
            region=termux_region,
            rate=termux_rate,
            pitch=termux_pitch,
            stream=termux_stream,
            tool_name=tool_name,
            tool_description=tool_description,
        )
    if mode == "vosk":
        # Vosk mode uses Pyttsx3 for speaking
        return create_pyttsx3_speak_tool(
            voice_name=pyttsx3_voice_name,
            rate=pyttsx3_rate,
            volume=pyttsx3_volume,
            tool_name=tool_name,
            tool_description=tool_description,
        )
    raise ValueError(f"Unknown mode: {mode}")
