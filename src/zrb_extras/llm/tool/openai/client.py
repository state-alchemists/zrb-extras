import os

from openai import AsyncOpenAI


def get_client(
    client: AsyncOpenAI | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> AsyncOpenAI:
    if client is not None:
        return client

    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY")

    return AsyncOpenAI(api_key=api_key, base_url=base_url)
