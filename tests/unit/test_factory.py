from unittest.mock import patch

import pytest

from zrb_extras.llm.tool.factory import create_speak_tool


@pytest.mark.parametrize("mode", ["google", "openai", "termux", "pyttsx3"])
def test_each_mode_builds_its_backend_speak_tool(mode):
    with patch(f"zrb_extras.llm.tool.factory.create_{mode}_speak_tool") as create:
        tool = create_speak_tool(mode=mode, tool_name="say")

    assert tool is create.return_value
    assert create.call_args.kwargs["tool_name"] == "say"


def test_pyttsx3_is_the_default_mode():
    with patch("zrb_extras.llm.tool.factory.create_pyttsx3_speak_tool") as create:
        create_speak_tool(pyttsx3_rate=120)

    assert create.call_args.kwargs["rate"] == 120


def test_an_unknown_mode_is_refused():
    with pytest.raises(ValueError, match="Unknown mode: vosk"):
        create_speak_tool(mode="vosk")  # type: ignore[arg-type]
