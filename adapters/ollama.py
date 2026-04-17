import json
from typing import AsyncGenerator

import httpx

from config import OLLAMA_BASE, OLLAMA_MODEL


async def chat(messages: list[dict], system: str | None = None, model: str = OLLAMA_MODEL, base_url: str = OLLAMA_BASE) -> str:
    """
    Send a conversation to Ollama. Accepts OpenAI-style messages list.
    """
    full_messages = _build_messages(messages, system)

    async with httpx.AsyncClient(timeout=300) as client:
        response = await client.post(
            f"{base_url}/api/chat",
            json={"model": model, "messages": full_messages, "stream": False},
        )
        response.raise_for_status()
        return response.json()["message"]["content"]


async def stream(messages: list[dict], system: str | None = None, model: str = OLLAMA_MODEL, base_url: str = OLLAMA_BASE) -> AsyncGenerator[str, None]:
    """
    Streaming version for Ollama.
    """
    full_messages = _build_messages(messages, system)

    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream(
            "POST",
            f"{base_url}/api/chat",
            json={"model": model, "messages": full_messages, "stream": True},
        ) as response:
            async for line in response.aiter_lines():
                if line:
                    data = json.loads(line)
                    if not data.get("done"):
                        yield data["message"]["content"]


def _build_messages(messages: list[dict], system: str | None) -> list[dict]:
    """
    Prepend system message if provided and not already in messages list.
    """
    has_system = any(m.get("role") == "system" for m in messages)
    if system and not has_system:
        return [{"role": "system", "content": system}] + messages
    return messages
