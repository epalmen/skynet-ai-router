import os
from enum import Enum


class Backend(str, Enum):
    LOCAL = "local"
    CLAUDE = "claude"
    REMOTE = "remote"


DEFAULT_BACKEND = Backend(os.getenv("DEFAULT_BACKEND", "local"))
OLLAMA_BASE = os.getenv("OLLAMA_BASE", "http://localhost:11434")
REMOTE_LLM_BASE = os.getenv("REMOTE_LLM_BASE", "http://192.168.68.141:11434")
REMOTE_LLM_MODEL = os.getenv("REMOTE_LLM_MODEL", "qwen2.5:7b")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
ROUTER_HOST = os.getenv("ROUTER_HOST", "0.0.0.0")
ROUTER_PORT = int(os.getenv("ROUTER_PORT", "3000"))
