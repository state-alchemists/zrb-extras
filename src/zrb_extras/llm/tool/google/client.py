import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from google import genai


def get_client(
    client: "genai.Client | None" = None,
    api_key: str | None = None,
) -> "genai.Client":
    if client is not None:
        return client

    try:
        from google import genai
    except ImportError:
        raise ImportError(
            "google-genai is not installed. Please install zrb-extras[google-genai] or zrb-extras[all]."
        )

    if api_key is None:
        api_key = os.environ.get("GEMINI_API_KEY", os.environ.get("GOOGLE_API_KEY"))
    return genai.Client(api_key=api_key)
