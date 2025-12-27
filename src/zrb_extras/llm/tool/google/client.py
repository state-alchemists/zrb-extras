from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from google import genai


def get_client(
    client: genai.Client | None = None,
    api_key: str | None = None,
) -> genai.Client:
    if client is not None:
        return client

    from google import genai

    if api_key is None:
        api_key = os.environ.get("GEMINI_API_KEY", os.environ.get("GOOGLE_API_KEY"))
    return genai.Client(api_key=api_key)
