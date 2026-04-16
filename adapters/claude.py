import asyncio
import json
import shutil
from typing import AsyncGenerator

from config import CLAUDE_MODEL

CLAUDE_BIN = shutil.which("claude") or "/home/erikp/.local/bin/claude"


async def chat(messages: list[dict], system: str | None = None, model: str = CLAUDE_MODEL) -> str:
    """
    Send a conversation to Claude via the Claude Code CLI subprocess.
    Uses the existing `claude auth login` session — no API key needed.
    Accepts OpenAI-style messages list.
    """
    # Build prompt from messages (collapse to single string for -p flag)
    prompt = _messages_to_prompt(messages)

    args = [
        CLAUDE_BIN,
        "-p", prompt,
        "--output-format", "json",
        "--model", model,
    ]
    if system:
        args += ["--system", system]

    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()

    if proc.returncode != 0:
        raise RuntimeError(f"Claude CLI error: {stderr.decode().strip()}")

    # Claude Code outputs JSONL — find the result entry
    for line in reversed(stdout.decode().strip().splitlines()):
        try:
            data = json.loads(line)
            if data.get("type") == "result":
                return data.get("result", "")
        except json.JSONDecodeError:
            continue

    return stdout.decode().strip()


async def stream(messages: list[dict], system: str | None = None, model: str = CLAUDE_MODEL) -> AsyncGenerator[str, None]:
    """
    Streaming version — yields text chunks as Claude produces them.
    """
    prompt = _messages_to_prompt(messages)

    args = [
        CLAUDE_BIN,
        "-p", prompt,
        "--output-format", "stream-json",
        "--model", model,
    ]
    if system:
        args += ["--system", system]

    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    async for raw in proc.stdout:
        try:
            data = json.loads(raw.decode().strip())
            if data.get("type") == "assistant":
                for block in data.get("message", {}).get("content", []):
                    if block.get("type") == "text":
                        yield block["text"]
        except (json.JSONDecodeError, KeyError):
            continue

    await proc.wait()


def _messages_to_prompt(messages: list[dict]) -> str:
    """
    Convert OpenAI-style messages to a plain-text prompt string.
    System messages are excluded here (handled via --system flag).
    """
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            continue
        parts.append(f"{role.capitalize()}: {content}" if len(messages) > 1 else content)
    return "\n".join(parts)
