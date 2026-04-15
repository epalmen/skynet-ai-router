# Skynet AI Router — Bouwplan

Een persoonlijke AI router op skynet die je apps transparant laat kiezen
tussen de lokale LLM (Ollama + RX 6600 XT) en Claude via de Claude Code CLI
(Pro subscription, geen API key nodig).

---

## Architectuur

```
Telegram bot / andere apps
        |
        v
  FastAPI Router          <- http://skynet:3000
  /mnt/workspace/router
        |           |
        v           v
  claude CLI     Ollama
  (Pro sub)      (lokaal)
  subprocess     RX 6600 XT
```

---

## Directory structuur

```
/mnt/workspace/router/
├── main.py              # FastAPI router server
├── adapters/
│   ├── __init__.py
│   ├── claude.py        # Claude Code CLI adapter (Paperclip approach)
│   └── ollama.py        # Ollama lokale adapter
├── config.py            # Routing configuratie
├── requirements.txt
└── systemd/
    └── skynet-router.service
```

---

## Stap 1 — Dependencies installeren

```bash
cd /mnt/workspace
mkdir router && cd router

pip install fastapi uvicorn httpx pydantic --break-system-packages
```

---

## Stap 2 — Claude adapter (Paperclip approach)

`adapters/claude.py`

```python
import subprocess
import json
import asyncio
from typing import AsyncGenerator

async def chat(prompt: str, system: str = None) -> str:
    """
    Spawn de claude CLI als subprocess — precies zoals Paperclip dit doet.
    Gebruikt de bestaande auth van: claude auth login
    Geen API key nodig.
    """
    args = [
        'claude',
        '-p', prompt,
        '--output-format', 'json',
        '--model', 'claude-sonnet-4-6',
    ]

    if system:
        args += ['--system', system]

    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    stdout, stderr = await proc.communicate()

    if proc.returncode != 0:
        raise RuntimeError(f"Claude CLI fout: {stderr.decode()}")

    # Claude Code output is JSONL — parse de laatste result entry
    lines = stdout.decode().strip().split('\n')
    for line in reversed(lines):
        try:
            data = json.loads(line)
            if data.get('type') == 'result':
                return data.get('result', '')
        except json.JSONDecodeError:
            continue

    return stdout.decode().strip()


async def stream(prompt: str, system: str = None) -> AsyncGenerator[str, None]:
    """
    Stream versie — stuurt chunks terug terwijl claude bezig is.
    """
    args = [
        'claude',
        '-p', prompt,
        '--output-format', 'stream-json',
        '--model', 'claude-sonnet-4-6',
    ]

    if system:
        args += ['--system', system]

    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    async for line in proc.stdout:
        try:
            data = json.loads(line.decode().strip())
            if data.get('type') == 'assistant':
                for block in data.get('message', {}).get('content', []):
                    if block.get('type') == 'text':
                        yield block['text']
        except (json.JSONDecodeError, KeyError):
            continue

    await proc.wait()
```

---

## Stap 3 — Ollama adapter

`adapters/ollama.py`

```python
import httpx
from typing import AsyncGenerator

OLLAMA_BASE = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:7b"  # pas aan naar jouw model


async def chat(prompt: str, system: str = None, model: str = DEFAULT_MODEL) -> str:
    """
    Roep Ollama aan via HTTP — werkt met GPU acceleratie via RX 6600 XT.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            f"{OLLAMA_BASE}/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": False,
            }
        )
        response.raise_for_status()
        return response.json()['message']['content']


async def stream(prompt: str, system: str = None, model: str = DEFAULT_MODEL) -> AsyncGenerator[str, None]:
    """
    Stream versie voor Ollama.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream(
            "POST",
            f"{OLLAMA_BASE}/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": True,
            }
        ) as response:
            async for line in response.aiter_lines():
                if line:
                    import json
                    data = json.loads(line)
                    if not data.get('done'):
                        yield data['message']['content']
```

---

## Stap 4 — Config

`config.py`

```python
from enum import Enum

class Backend(str, Enum):
    LOCAL = "local"       # Ollama + RX 6600 XT
    CLAUDE = "claude"     # Claude Code CLI via Pro subscription

# Routing regels — pas aan naar jouw voorkeur
ROUTING_RULES = {
    # Gebruik lokaal voor:
    "local_triggers": [
        "samenvatting", "vertaal", "briefing", "notitie",
        "kort", "snel", "simpel",
    ],
    # Default backend
    "default": Backend.LOCAL,
}

# Ollama model
OLLAMA_MODEL = "qwen2.5:7b"
```

---

## Stap 5 — FastAPI Router

`main.py`

```python
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import asyncio

from adapters import claude, ollama
from config import Backend, ROUTING_RULES, OLLAMA_MODEL

app = FastAPI(title="Skynet AI Router")


class ChatRequest(BaseModel):
    prompt: str
    system: Optional[str] = None
    backend: Optional[Backend] = None   # None = auto-route
    model: Optional[str] = None         # alleen voor ollama
    stream: bool = False


def auto_route(prompt: str) -> Backend:
    """
    Simpele routing logica op basis van keywords.
    Uitbreidbaar naar complexere logica.
    """
    prompt_lower = prompt.lower()
    for trigger in ROUTING_RULES["local_triggers"]:
        if trigger in prompt_lower:
            return Backend.LOCAL
    return ROUTING_RULES["default"]


@app.post("/chat")
async def chat(req: ChatRequest):
    backend = req.backend or auto_route(req.prompt)

    try:
        if backend == Backend.CLAUDE:
            result = await claude.chat(req.prompt, req.system)
        else:
            result = await ollama.chat(
                req.prompt,
                req.system,
                req.model or OLLAMA_MODEL
            )
        return {
            "result": result,
            "backend_used": backend,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    backend = req.backend or auto_route(req.prompt)

    async def generate():
        if backend == Backend.CLAUDE:
            async for chunk in claude.stream(req.prompt, req.system):
                yield chunk
        else:
            async for chunk in ollama.stream(
                req.prompt,
                req.system,
                req.model or OLLAMA_MODEL
            ):
                yield chunk

    return StreamingResponse(generate(), media_type="text/plain")


@app.get("/health")
async def health():
    return {"status": "ok", "backends": ["claude", "local"]}


@app.get("/backends")
async def backends():
    return {
        "available": [b.value for b in Backend],
        "default": ROUTING_RULES["default"],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)
```

---

## Stap 6 — Systemd service

`systemd/skynet-router.service`

```ini
[Unit]
Description=Skynet AI Router
After=network.target ollama.service
Wants=ollama.service

[Service]
Type=simple
User=erikp
WorkingDirectory=/mnt/workspace/router
ExecStart=/usr/bin/python3 -m uvicorn main:app --host 0.0.0.0 --port 3000
Restart=always
RestartSec=5
Environment=PATH=/home/erikp/.local/bin:/usr/local/bin:/usr/bin:/bin

[Install]
WantedBy=multi-user.target
```

Installeren:

```bash
sudo cp systemd/skynet-router.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable skynet-router
sudo systemctl start skynet-router
sudo systemctl status skynet-router
```

---

## Stap 7 — Telegram bot aanpassen

In je bestaande Telegram bot, vervang de directe Claude aanroep door:

```python
import httpx

ROUTER_URL = "http://localhost:3000"

async def ask_ai(prompt: str, backend: str = None) -> str:
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            f"{ROUTER_URL}/chat",
            json={
                "prompt": prompt,
                "backend": backend,  # "claude", "local", of None voor auto
            }
        )
        data = response.json()
        return data["result"]

# Gebruik:
# await ask_ai("maak een samenvatting")           -> auto-route (lokaal)
# await ask_ai("analyseer dit", backend="claude") -> altijd Claude
# await ask_ai("snelle vraag", backend="local")   -> altijd lokaal
```

---

## Testen

```bash
# Health check
curl http://localhost:3000/health

# Test lokaal
curl -X POST http://localhost:3000/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "wat is 2+2", "backend": "local"}'

# Test Claude (via Pro sub)
curl -X POST http://localhost:3000/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "leg recursie uit", "backend": "claude"}'

# Auto-routing
curl -X POST http://localhost:3000/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "maak een korte samenvatting van mijn dag"}'
```

---

## Volgorde van bouwen

1. ✅ GPU setup (RX 6600 XT + PCIe adapter + custom kernel)
2. ✅ Ollama installeren met GPU support
3. 🔲 Router bouwen en testen met alleen Ollama
4. 🔲 Claude adapter toevoegen en testen
5. 🔲 Systemd service activeren
6. 🔲 Telegram bot aanpassen
7. 🔲 Andere apps aansluiten

---

## Uitbreidingsideeën

- **Automatische fallback**: als Ollama traag is of faalt → automatisch naar Claude
- **Per-app routing**: Telegram bot gebruikt lokaal, SAP tools gebruiken Claude
- **Kosten tracking**: log welk backend je gebruikt
- **Model selector in Telegram**: inline keyboard met "🏠 Lokaal" / "☁️ Claude" knop
- **Context persistence**: sessie IDs bijhouden zoals Paperclip dat doet
