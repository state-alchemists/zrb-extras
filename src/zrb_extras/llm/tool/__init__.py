from zrb_extras.llm.tool.google import create_listen_tool as create_google_listen_tool
from zrb_extras.llm.tool.google import create_speak_tool as create_google_speak_tool
from zrb_extras.llm.tool.openai import create_listen_tool as create_openai_listen_tool
from zrb_extras.llm.tool.openai import create_speak_tool as create_openai_speak_tool
from zrb_extras.llm.tool.pyttsx3 import create_speak_tool as create_pyttsx3_speak_tool
from zrb_extras.llm.tool.termux import create_listen_tool as create_termux_listen_tool
from zrb_extras.llm.tool.termux import create_speak_tool as create_termux_speak_tool
from zrb_extras.llm.tool.vosk import create_listen_tool as create_vosk_listen_tool
from zrb_extras.llm.tool.youtube.transcript import fetch_youtube_transcript

create_listen_tool = create_vosk_listen_tool
create_speak_tool = create_pyttsx3_speak_tool

__all__ = [
    "create_listen_tool",
    "create_speak_tool",
    "create_google_listen_tool",
    "create_google_speak_tool",
    "create_openai_listen_tool",
    "create_openai_speak_tool",
    "create_termux_listen_tool",
    "create_termux_speak_tool",
    "create_vosk_listen_tool",
    "create_pyttsx3_speak_tool",
    "fetch_youtube_transcript",
]
