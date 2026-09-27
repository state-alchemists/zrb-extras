"""A chat that talks back with the `speak` tool.

Dictation (`/voice`, `/handsfree`) is built into zrb; this adds a `speak`
tool the agent calls to answer aloud, and the YouTube transcript tool.
"""

import os

from zrb.builtin import llm_chat
from zrb.llm.tool_call.tool_policy.auto_approve import auto_approve

from zrb_extras.llm.tool import create_speak_tool, fetch_youtube_transcript

GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", ""))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
VOICE_MODE = os.getenv("VOICE_MODE", "pyttsx3").strip().lower()
if VOICE_MODE not in ("google", "openai", "termux", "pyttsx3"):
    VOICE_MODE = "pyttsx3"

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

llm_chat.append_tool(speak, fetch_youtube_transcript)
llm_chat.prepend_tool_policy(auto_approve("speak"))
