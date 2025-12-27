import os
from typing import Callable

from zrb import llm_config
from zrb.builtin import llm_ask, llm_chat_trigger

from zrb_extras.llm.tool import (
    create_google_listen_tool,
    create_google_speak_tool,
    create_openai_listen_tool,
    create_openai_speak_tool,
    create_pyttsx3_speak_tool,
    create_vosk_listen_tool,
    fetch_youtube_transcript,
)

GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", ""))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
VOICE_MODE = os.getenv("VOICE_MODE", "").strip().lower()


def get_voice_tools() -> tuple[Callable, Callable]:
    if VOICE_MODE == "google":
        listen = create_google_listen_tool(
            api_key=GOOGLE_API_KEY,
            stt_model="gemini-2.5-flash",
            sample_rate=16000,
            channels=1,
            silence_threshold=0.01,
            max_silence=1.5,
        )
        speak = create_google_speak_tool(
            api_key=GOOGLE_API_KEY,
            tts_model="gemini-2.5-flash-preview-tts",
            voice_name="sulafat",
            sample_rate_out=24000,
        )
        return listen, speak
    if VOICE_MODE == "openai":
        listen = create_openai_listen_tool(
            api_key=OPENAI_API_KEY,
            stt_model="whisper-1",
            sample_rate=16000,
            channels=1,
            silence_threshold=0.01,
            max_silence=1.5,
        )
        speak = create_openai_speak_tool(
            api_key=OPENAI_API_KEY,
            tts_model="tts-1",
            voice_name="alloy",
        )
        return listen, speak
    listen = create_vosk_listen_tool()
    speak = create_pyttsx3_speak_tool()
    return listen, speak


listen, speak = get_voice_tools()
llm_chat_trigger.add_trigger(listen)
llm_ask.add_tool(speak, fetch_youtube_transcript)

# Optional: allow LLM to speak or listen without asking for user approval
if not llm_config.default_yolo_mode:
    llm_config.set_default_yolo_mode(["speak"])
