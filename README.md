# Skynet AI Router

A lightweight FastAPI router that transparently routes AI requests to either a **local Ollama instance** (GPU-accelerated) or **Claude via the Claude Code CLI** (Pro subscription, no API key required).

## Architecture

```
App A (DEFAULT_BACKEND=local)   ──┐
App B (DEFAULT_BACKEND=claude)  ──┤──> http://skynet:3000 ──> Ollama / Claude CLI
App C (no .env, uses router default) ┘
```

Each app sets its preferred backend in its own `.env`. The router itself also has a fallback default.

## Backend selection (priority order)

1. **Per-request** — set `backend` field in the request body (`"local"` or `"claude"`)
2. **Per-app** — app reads its own `.env` and passes `backend` in the request
3. **Router default** — `DEFAULT_BACKEND` env var on the router (falls back to `local`)

## Quick start

```bash
cp .env.example .env
# Edit .env — set DEFAULT_BACKEND, OLLAMA_MODEL, etc.

pip install -r requirements.txt
python main.py
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/chat` | Native chat endpoint |
| POST | `/chat/stream` | Streaming version |
| POST | `/v1/chat/completions` | OpenAI-compatible endpoint |
| GET | `/health` | Health check + active default backend |
| GET | `/backends` | List available backends |

## Native `/chat` usage

```python
import httpx

ROUTER_URL = "http://skynet:3000"

async def ask_ai(prompt: str, backend: str = None) -> str:
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            f"{ROUTER_URL}/chat",
            json={
                "messages": [{"role": "user", "content": prompt}],
                "backend": backend,   # "claude", "local", or None for default
            }
        )
        return response.json()["result"]

# Examples:
# await ask_ai("what is 2+2")                     -> uses DEFAULT_BACKEND
# await ask_ai("analyse this", backend="claude")  -> always Claude
# await ask_ai("quick question", backend="local") -> always local
```

## OpenAI-compatible usage

Apps already using the OpenAI SDK need only change `base_url`:

```python
from openai import AsyncOpenAI

client = AsyncOpenAI(
    api_key="not-needed",
    base_url="http://skynet:3000/v1",
)

response = await client.chat.completions.create(
    model="qwen2.5:7b",          # passed through to Ollama
    messages=[{"role": "user", "content": "hello"}],
    extra_body={"x_backend": "local"},  # optional override
)
```

## App-level backend selection via `.env`

In each app that calls the router:

```env
# app/.env
AI_ROUTER_URL=http://skynet:3000
AI_BACKEND=claude   # or: local
```

```python
import os
from dotenv import load_dotenv
import httpx

load_dotenv()

ROUTER_URL = os.getenv("AI_ROUTER_URL", "http://localhost:3000")
BACKEND = os.getenv("AI_BACKEND")   # None = use router's default

async def ask(prompt: str) -> str:
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(f"{ROUTER_URL}/chat", json={
            "messages": [{"role": "user", "content": prompt}],
            "backend": BACKEND,
        })
        return r.json()["result"]
```

## Systemd service

```bash
sudo cp systemd/skynet-router.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now skynet-router
sudo systemctl status skynet-router
```

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DEFAULT_BACKEND` | `local` | Fallback backend when not specified per-request |
| `OLLAMA_BASE` | `http://localhost:11434` | Ollama API base URL |
| `OLLAMA_MODEL` | `qwen2.5:7b` | Default Ollama model |
| `CLAUDE_MODEL` | `claude-sonnet-4-6` | Claude model to use |
| `ROUTER_HOST` | `0.0.0.0` | Bind address |
| `ROUTER_PORT` | `3000` | Bind port |
