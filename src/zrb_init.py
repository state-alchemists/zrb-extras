import os

from zrb import llm_config
from zrb.builtin import llm_chat

from zrb_extras.llm.tool import (
    create_listen_tool,
    create_speak_tool,
    fetch_youtube_transcript,
)

GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", ""))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
VOICE_MODE = os.getenv("VOICE_MODE", "vosk").strip().lower()
if VOICE_MODE not in ("google", "openai", "termux", "vosk"):
    VOICE_MODE = "vosk"

listen = create_listen_tool(
    mode=VOICE_MODE,
    genai_api_key=GOOGLE_API_KEY,
    genai_stt_model="gemini-2.5-flash",
    openai_api_key=OPENAI_API_KEY,
    openai_stt_model="whisper-1",
    sample_rate=16000,
    channels=1,
    silence_threshold=0.01,
    max_silence=1.5,
    text_processor=lambda txt: (
            f"> Note: Respond to the following user message with with speak tool :\n{txt}"
    ),
    use_sound_classifier=True,
    classification_system_prompt=(
        "You are a sound classifier. Analyze the provided transcript "
        "and determine if it contains speech that should be handled "
        "(i.e., user mention your name, Zaruba). "
        "Consider background noise, non-speech sounds, and unclear speech. "
        "If unsure, assume it's speech to be safe."
    ),
)
speak = create_speak_tool(
    mode=VOICE_MODE,
    genai_api_key=GOOGLE_API_KEY,
    genai_tts_model="gemini-2.5-flash-preview-tts",
    genai_voice_name="sulafat",
    openai_api_key=OPENAI_API_KEY,
    openai_tts_model="tts-1",
    openai_voice_name="alloy",
    sample_rate_out=24000,
)

# llm_chat.add_trigger(listen)
llm_chat.add_tool(speak, fetch_youtube_transcript)
