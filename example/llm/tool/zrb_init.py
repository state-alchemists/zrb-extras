import os

from zrb import AnyContext, llm_config
from zrb.builtin import llm_ask, llm_chat_trigger

from zrb_extras.llm.tool import create_listen_tool, create_speak_tool

API_KEY = os.getenv("GOOGLE_API_KEY", "")

llm_ask.add_tool(
    create_speak_tool(
        api_key=API_KEY,
        tts_model="gemini-2.5-flash-preview-tts",
        voice_name="sulafat",
        sample_rate_out=24000,
    )
)


# listen = create_listen_tool(
#     api_key=API_KEY,
#     stt_model="gemini-2.5-flash",
#     sample_rate=16000,
#     channels=1,
#     silence_threshold=0.01,
#     max_silence=4.0,
# )
# llm_ask.add_tool(listen)

listen_as_chat_trigger = create_listen_tool(
    api_key=API_KEY,
    stt_model="gemini-2.5-flash",
    sample_rate=16000,
    channels=1,
    silence_threshold=0.01,
    max_silence=4.0,
    as_chat_trigger=True,
)
llm_chat_trigger.add_trigger(listen_as_chat_trigger)


# Optional: allow LLM to speak or listen without asking for user approval
if not llm_config.default_yolo_mode:
    llm_config.set_default_yolo_mode(["speak"])
