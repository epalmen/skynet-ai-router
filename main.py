"""
Skynet AI Router
================
A lightweight FastAPI router that forwards requests to either a local Ollama
instance or the Claude Code CLI, based on configuration.

Backend selection (in priority order):
  1. Per-request  — set `backend` field in the request body
  2. Per-app      — the calling app reads its own .env and passes `backend`
  3. Router default — DEFAULT_BACKEND env var (falls back to "local")
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import time

from adapters import claude, ollama
from config import Backend, DEFAULT_BACKEND, OLLAMA_MODEL, CLAUDE_MODEL, REMOTE_LLM_BASE, REMOTE_LLM_MODEL, ROUTER_HOST, ROUTER_PORT

app = FastAPI(
    title="Skynet AI Router",
    description="Routes AI requests to Ollama (local) or Claude CLI based on configuration.",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class Message(BaseModel):
    role: str           # "system" | "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]
    backend: Optional[Backend] = None   # None → use DEFAULT_BACKEND
    model: Optional[str] = None         # override model for chosen backend
    stream: bool = False
    image_path: Optional[str] = None    # local file path to attach to the request


class ChatResponse(BaseModel):
    result: str
    backend_used: Backend
    model_used: str


# OpenAI-compatible request shape so apps using the OpenAI SDK need no changes
class OpenAIChatRequest(BaseModel):
    model: Optional[str] = None
    messages: list[Message]
    stream: bool = False
    # x-backend is a custom extension field for backend selection
    x_backend: Optional[Backend] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_backend(requested: Optional[Backend]) -> Backend:
    return requested if requested is not None else DEFAULT_BACKEND


def _resolve_model(backend: Backend, override: Optional[str]) -> str:
    if override:
        return override
    if backend == Backend.CLAUDE:
        return CLAUDE_MODEL
    if backend == Backend.REMOTE:
        return REMOTE_LLM_MODEL
    return OLLAMA_MODEL


def _messages_as_dicts(messages: list[Message]) -> list[dict]:
    return [m.model_dump() for m in messages]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    backend = _resolve_backend(req.backend)
    model = _resolve_model(backend, req.model)
    msgs = _messages_as_dicts(req.messages)

    try:
        if backend == Backend.CLAUDE:
            result = await claude.chat(msgs, model=model, image_path=req.image_path)
        elif backend == Backend.REMOTE:
            result = await ollama.chat(msgs, model=model, base_url=REMOTE_LLM_BASE)
        else:
            result = await ollama.chat(msgs, model=model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return ChatResponse(result=result, backend_used=backend, model_used=model)


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    backend = _resolve_backend(req.backend)
    model = _resolve_model(backend, req.model)
    msgs = _messages_as_dicts(req.messages)

    async def generate():
        try:
            if backend == Backend.CLAUDE:
                async for chunk in claude.stream(msgs, model=model, image_path=req.image_path):
                    yield chunk
            elif backend == Backend.REMOTE:
                async for chunk in ollama.stream(msgs, model=model, base_url=REMOTE_LLM_BASE):
                    yield chunk
            else:
                async for chunk in ollama.stream(msgs, model=model):
                    yield chunk
        except Exception as e:
            yield f"\n[Router error: {e}]"

    return StreamingResponse(generate(), media_type="text/plain")


@app.post("/v1/chat/completions")
async def openai_chat(req: OpenAIChatRequest):
    """
    OpenAI-compatible endpoint. Apps using the OpenAI SDK can point their
    base_url at this router without any code changes.

    Set x_backend in the request body to choose a backend, or rely on the
    router's DEFAULT_BACKEND env var.
    """
    backend = _resolve_backend(req.x_backend)
    model = _resolve_model(backend, req.model)
    msgs = _messages_as_dicts(req.messages)

    try:
        if backend == Backend.CLAUDE:
            result = await claude.chat(msgs, model=model)
        else:
            result = await ollama.chat(msgs, model=model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Return OpenAI-compatible response shape
    return {
        "id": f"chatcmpl-skynet",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result},
                "finish_reason": "stop",
            }
        ],
        "x_backend_used": backend,
    }


@app.get("/health")
async def health():
    return {"status": "ok", "default_backend": DEFAULT_BACKEND}


@app.get("/backends")
async def backends():
    return {
        "available": [b.value for b in Backend],
        "default": DEFAULT_BACKEND,
    }


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=ROUTER_HOST, port=ROUTER_PORT)
