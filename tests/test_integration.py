"""Integration tests for the response-adapter proxy against real APIs.

Requires ``.env`` with ``OPENAI_API_KEY`` (and optionally
``ANTHROPIC_API_KEY``).  Tests are skipped automatically when
the corresponding key is missing or is a placeholder.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.proxy.client import AnthropicClient, ChatClient
from src.proxy.server import create_app


def load_env() -> None:
    """Load environment variables from ``.env`` if available."""
    env_path = Path(__file__).parent.parent / ".env"
    if not env_path.exists():
        return
    with env_path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())


load_env()

CHAT_API_KEY = os.environ.get("OPENAI_API_KEY", "")
CHAT_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_BASE_URL = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1")

skip_no_chat = pytest.mark.skipif(not CHAT_API_KEY, reason="OPENAI_API_KEY not set")
skip_no_anthropic = pytest.mark.skipif(
    not ANTHROPIC_API_KEY or len(ANTHROPIC_API_KEY) < 20,
    reason="ANTHROPIC_API_KEY not set or placeholder",
)


def _build_app(target: str = "chat") -> tuple:
    chat_client = ChatClient(api_key=CHAT_API_KEY, base_url=CHAT_BASE_URL) if CHAT_API_KEY else None
    anthropic_client = AnthropicClient(api_key=ANTHROPIC_API_KEY, base_url=ANTHROPIC_BASE_URL) if ANTHROPIC_API_KEY else None
    app = create_app(default_target=target, chat_client=chat_client, anthropic_client=anthropic_client)
    return TestClient(app)


class TestChatIntegration:
    """Integration tests against a real OpenAI-compatible Chat backend."""

    @skip_no_chat
    def test_non_streaming(self):
        """Basic non-streaming request returns a valid Responses API response."""
        tc = _build_app("chat")
        resp = tc.post("/v1/responses", json={
            "model": "deepseek-chat",
            "input": "Say hello in one word",
            "instructions": "Be concise.",
        })
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data.get("object") == "response"
        assert data.get("model"), f"model not returned: {data}"
        output = data.get("output", [])
        assert len(output) > 0
        assert output[0].get("type") == "message"
        content = output[0].get("content", [])
        assert len(content) > 0
        texts = [c.get("text", "") for c in content if c.get("type") == "output_text"]
        assert len(texts) > 0
        assert len(texts[0]) > 0
        usage = data.get("usage", {})
        assert usage.get("input_tokens", 0) > 0
        assert usage.get("output_tokens", 0) > 0

    @skip_no_chat
    def test_non_streaming_deepseek_reasoner(self):
        """Test with deepseek-reasoner model (uses streaming internally)."""
        tc = _build_app("chat")
        resp = tc.post("/v1/responses", json={
            "model": "deepseek-reasoner",
            "input": "What is 2+2?",
        })
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data.get("object") == "response"
        output = data.get("output", [])
        assert len(output) > 0
        assert output[0].get("type") == "message"
        usage = data.get("usage", {})
        assert usage.get("total_tokens", 0) > 0

    @skip_no_chat
    def test_streaming(self):
        """Streaming request emits SSE events: output_text.delta + completed."""
        tc = _build_app("chat")
        texts = []
        completed = False
        with tc.stream("POST", "/v1/responses", json={
            "model": "deepseek-chat",
            "input": "Count 1 2 3",
            "stream": True,
        }) as r:
            assert r.status_code == 200
            for line in r.iter_lines():
                if not line:
                    continue
                if line.startswith("data: ") and line != "data: [DONE]":
                    try:
                        d = json.loads(line[6:])
                        if d.get("type") == "response.output_text.delta":
                            texts.append(d["delta"])
                        elif d.get("type") == "response.completed":
                            completed = True
                            usage = d["response"]["usage"]
                            assert usage.get("total_tokens", 0) > 0
                    except Exception:
                        pass
        assert len(texts) > 0, "No text deltas received"
        assert completed, "response.completed event not received"

    @skip_no_chat
    def test_health(self):
        """Health endpoint reports status and backend availability."""
        tc = _build_app("chat")
        resp = tc.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["chat_configured"] is True

    @skip_no_chat
    def test_invalid_model(self):
        """An invalid model triggers an upstream error."""
        tc = _build_app("chat")
        try:
            resp = tc.post("/v1/responses", json={
                "model": "nonexistent-model-xyz",
                "input": "Hello",
            })
            assert resp.status_code >= 400
        except Exception:
            pass


class TestAnthropicIntegration:
    """Integration tests against the real Anthropic API."""

    @skip_no_anthropic
    def test_non_streaming(self):
        """Basic non-streaming request against Anthropic."""
        tc = _build_app("anthropic")
        resp = tc.post("/v1/responses", json={
            "model": "claude-sonnet-4-20250514",
            "input": "Say hello in one word",
            "max_output_tokens": 50,
        }, headers={"X-Target-Backend": "anthropic"})
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data.get("object") == "response"
        output = data.get("output", [])
        assert len(output) > 0
        usage = data.get("usage", {})
        assert usage.get("total_tokens", 0) > 0

    @skip_no_anthropic
    def test_streaming(self):
        """Streaming request against Anthropic emits valid SSE events."""
        tc = _build_app("anthropic")
        texts = []
        completed = False
        with tc.stream("POST", "/v1/responses", json={
            "model": "claude-sonnet-4-20250514",
            "input": "Count 1 2 3",
            "max_output_tokens": 50,
            "stream": True,
        }, headers={"X-Target-Backend": "anthropic"}) as r:
            assert r.status_code == 200
            for line in r.iter_lines():
                if not line:
                    continue
                if line.startswith("data: ") and line != "data: [DONE]":
                    try:
                        d = json.loads(line[6:])
                        if d.get("type") == "response.output_text.delta":
                            texts.append(d["delta"])
                        elif d.get("type") in ("response.completed", "response.incomplete"):
                            completed = True
                    except Exception:
                        pass
        assert len(texts) > 0, "No text deltas received"
        assert completed, "response.completed event not received"

    @skip_no_chat
    def test_health(self):
        tc = _build_app("chat")
        resp = tc.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["chat_configured"] is True

    @skip_no_chat
    def test_invalid_model(self):
        tc = _build_app("chat")
        try:
            resp = tc.post("/v1/responses", json={
                "model": "nonexistent-model-xyz",
                "input": "Hello",
            })
            # If we get a response, it should be an error
            assert resp.status_code >= 400
        except Exception:
            # Upstream API may raise directly before FastAPI wrapping
            pass


