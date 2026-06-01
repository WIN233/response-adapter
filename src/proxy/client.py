"""HTTP clients for Chat Completions and Anthropic Messages APIs.

Provides async clients for both non-streaming and streaming requests,
with configurable API keys, base URLs, and timeouts from environment
variables or constructor arguments.
"""

from __future__ import annotations

import json
import os
from typing import AsyncIterator

import httpx


class ChatClient:
    """Async HTTP client for OpenAI-compatible Chat Completions API.

    Supports both ``create_completion`` (single response) and
    ``stream_completion`` (SSE-based streaming via ``aiter_lines``).
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 120.0,
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.timeout = timeout

    async def create_completion(self, payload: dict) -> dict:
        """Send a non-streaming chat completion request."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=payload,
            )
            resp.raise_for_status()
            return resp.json()

    async def stream_completion(self, payload: dict) -> AsyncIterator[dict]:
        """Stream a chat completion via SSE.

        Yields parsed JSON chunks from ``data: ...`` lines until
        ``data: [DONE]`` or the stream ends.
        """
        payload["stream"] = True
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=payload,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break
                    if not data_str:
                        continue
                    try:
                        yield json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }


class AnthropicClient:
    """Async HTTP client for Anthropic Messages API.

    Supports both ``create_message`` (single response) and
    ``stream_message`` (SSE-based streaming).
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 120.0,
    ):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.base_url = (base_url or os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1")).rstrip("/")
        self.timeout = timeout

    async def create_message(self, payload: dict) -> dict:
        """Send a non-streaming message request."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/messages",
                headers=self._headers(),
                json=payload,
            )
            resp.raise_for_status()
            return resp.json()

    async def stream_message(self, payload: dict) -> AsyncIterator[dict]:
        """Stream a message via SSE.

        Anthropic SSE format uses ``event: ...`` and ``data: ...`` lines.
        Yields parsed JSON from data lines, skipping event lines.
        """
        payload["stream"] = True
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/messages",
                headers=self._headers(),
                json=payload,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if not data_str:
                            continue
                        try:
                            yield json.loads(data_str)
                        except json.JSONDecodeError:
                            continue
                    elif line.startswith("event: "):
                        pass

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
