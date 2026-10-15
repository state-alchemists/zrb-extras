"""A chat that talks back with the `speak` tool.

Dictation (`/voice`, `/handsfree`) is built into zrb; this adds a `speak`
tool the agent calls to answer aloud, and the YouTube transcript tool.

The backend is zrb's speech backend: set ``ZRB_LLM_SPEECH_BACKEND`` (``auto``,
``openai``, ``gemini``, ``termux``, ``kokoro``, ...) and ``ZRB_LLM_SPEECH_VOICE``.
"""

from zrb.builtin import llm_chat
from zrb.llm.tool_call.tool_policy.auto_approve import auto_approve

from zrb_extras.llm.tool import create_speak_tool, fetch_youtube_transcript

llm_chat.append_tool(create_speak_tool(), fetch_youtube_transcript)
llm_chat.prepend_tool_policy(auto_approve("speak"))
