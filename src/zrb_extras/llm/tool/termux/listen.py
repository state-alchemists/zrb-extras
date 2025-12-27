import shutil
import subprocess
from typing import Any, Callable, Coroutine

from zrb import AnyContext


def create_listen_tool(
    tool_name: str | None = None,
    tool_description: str | None = None,
) -> Callable[[AnyContext], Coroutine[Any, Any, str]]:
    """
    Factory to create a listen tool using Termux API.
    Requires 'termux-speech-to-text' to be installed and available in path.
    """

    async def listen(ctx: AnyContext) -> str:
        """Listens for and transcribes the user's verbal response using Termux native UI.

        Returns:
            The transcribed text from the user's speech.
        """
        if not shutil.which("termux-speech-to-text"):
            raise RuntimeError(
                "termux-speech-to-text not found. Is Termux API installed?"
            )

        ctx.print("Listening (Termux UI)...", plain=True)
        # Run blocking subprocess as it invokes a UI
        try:
            result = subprocess.run(
                ["termux-speech-to-text"], capture_output=True, text=True, check=True
            )
            text = result.stdout.strip()
            ctx.print(f"Heard: {text}", plain=True)
            return text
        except subprocess.CalledProcessError as e:
            ctx.print(f"Error calling Termux STT: {e}", plain=True)
            return ""

    if tool_name is not None:
        listen.__name__ = tool_name
    if tool_description is not None:
        listen.__doc__ = tool_description
    return listen
