import os
from enum import Enum


class Backend(str, Enum):
    LOCAL = "local"
    CLAUDE = "claude"


DEFAULT_BACKEND = Backend(os.getenv("DEFAULT_BACKEND", "local"))
OLLAMA_BASE = os.getenv("OLLAMA_BASE", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
ROUTER_HOST = os.getenv("ROUTER_HOST", "0.0.0.0")
ROUTER_PORT = int(os.getenv("ROUTER_PORT", "3000"))
