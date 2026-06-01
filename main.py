"""Entry point for the response-adapter proxy server.

Loads configuration from ``.env`` (or environment variables), initializes
back-end HTTP clients, creates the FastAPI application, and starts uvicorn.
"""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        with env_path.open() as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip())

import uvicorn
from src.proxy.server import create_app
from src.proxy.client import ChatClient, AnthropicClient


def main():
    """Parse environment, build clients & app, then start the uvicorn server."""

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    default_target = os.environ.get("TARGET_BACKEND", "chat").lower().strip()
    if default_target not in ("chat", "anthropic"):
        print(f"WARNING: invalid TARGET_BACKEND={default_target!r}, falling back to 'chat'")
        default_target = "chat"

    chat_api_key = os.environ.get("OPENAI_API_KEY", "")
    chat_base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    anthropic_base_url = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1")

    chat_client = ChatClient(api_key=chat_api_key, base_url=chat_base_url) if chat_api_key else None
    anthropic_client = AnthropicClient(api_key=anthropic_api_key, base_url=anthropic_base_url) if anthropic_api_key else None

    if not chat_client:
        print("WARNING: OPENAI_API_KEY not set — Chat backend unavailable")
    if not anthropic_client:
        print("WARNING: ANTHROPIC_API_KEY not set — Anthropic backend unavailable")

    app = create_app(
        default_target=default_target,
        chat_client=chat_client,
        anthropic_client=anthropic_client,
    )

    print(f"Starting response-adapter proxy on {host}:{port}")
    print(f"  Default backend: {default_target}")
    if chat_client:
        print(f"  Chat BASE:      {chat_base_url}")
    if anthropic_client:
        print(f"  Anthropic BASE: {anthropic_base_url}")
    print(f"  WebUI:          http://localhost:{port}/")
    print(f"  API (chat):     POST /v1/responses (X-Target-Backend: chat)")
    print(f"  API (anthropic): POST /v1/responses (X-Target-Backend: anthropic)")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
