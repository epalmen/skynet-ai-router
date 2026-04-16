import asyncio
import base64
import json
import mimetypes
import shutil
from pathlib import Path
from typing import AsyncGenerator

from config import CLAUDE_MODEL

CLAUDE_BIN = shutil.which("claude") or "/home/erikp/.local/bin/claude"


def _build_input(messages: list[dict], system: str | None, image_path: str | None) -> tuple[list[str], str | None]:
    """
    Returns (args, stdin_data).
    When an image is present we use stream-json input so we can send
    multimodal content blocks. Otherwise we use the simpler -p text mode.
    """
    if image_path:
        # Build a stream-json user message with image + text content blocks
        text = _messages_to_prompt(messages)
        raw = Path(image_path).read_bytes()
        media_type = mimetypes.guess_type(image_path)[0] or "image/jpeg"
        content = [
            {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": base64.standard_b64encode(raw).decode()}},
            {"type": "text", "text": text},
        ]
        stdin_data = json.dumps({"type": "user", "message": {"role": "user", "content": content}})
        args = [CLAUDE_BIN, "-p", "--input-format", "stream-json", "--output-format", "stream-json", "--verbose", "--model", CLAUDE_MODEL]
        if system:
            args += ["--system-prompt", system]
        return args, stdin_data
    else:
        prompt = _messages_to_prompt(messages)
        args = [CLAUDE_BIN, "-p", prompt, "--output-format", "json", "--model", CLAUDE_MODEL]
        if system:
            args += ["--system", system]
        return args, None


async def chat(messages: list[dict], system: str | None = None, model: str = CLAUDE_MODEL, image_path: str | None = None) -> str:
    """
    Send a conversation to Claude via the Claude Code CLI subprocess.
    Supports optional image attachment via base64 stream-json input.
    """
    args, stdin_data = _build_input(messages, system, image_path)

    proc = await asyncio.create_subprocess_exec(
        *args,
        stdin=asyncio.subprocess.PIPE if stdin_data else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate(input=stdin_data.encode() if stdin_data else None)

    if proc.returncode != 0:
        raise RuntimeError(f"Claude CLI error: {stderr.decode().strip() or stdout.decode().strip()}")

    for line in reversed(stdout.decode().strip().splitlines()):
        try:
            data = json.loads(line)
            if data.get("type") == "result":
                return data.get("result", "")
        except json.JSONDecodeError:
            continue

    return stdout.decode().strip()


async def stream(messages: list[dict], system: str | None = None, model: str = CLAUDE_MODEL, image_path: str | None = None) -> AsyncGenerator[str, None]:
    """
    Streaming version — yields text chunks as Claude produces them.
    """
    args, stdin_data = _build_input(messages, system, image_path)
    # Force stream-json output for streaming
    if "--output-format" in args:
        idx = args.index("--output-format")
        args[idx + 1] = "stream-json"
    if "--verbose" not in args:
        args.append("--verbose")

    proc = await asyncio.create_subprocess_exec(
        *args,
        stdin=asyncio.subprocess.PIPE if stdin_data else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    if stdin_data:
        proc.stdin.write(stdin_data.encode())
        await proc.stdin.drain()
        proc.stdin.close()

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
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            continue
        parts.append(f"{role.capitalize()}: {content}" if len(messages) > 1 else content)
    return "\n".join(parts)
