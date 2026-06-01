"""Pydantic models for OpenAI Chat Completions API.

Defines request/response schemas for the /v1/chat/completions endpoint,
including messages, tool definitions, choices, and token usage.
"""

from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel


class ChatMessage(BaseModel):
    """A message in a chat conversation with role and optional tool calls."""
    role: Literal["system", "user", "assistant", "tool", "developer"]
    content: str | list[dict[str, Any]] | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    name: str | None = None


class FunctionDefinition(BaseModel):
    """A function definition for tool calling."""
    name: str
    description: str = ""
    parameters: dict[str, Any] = {}


class ToolDefinition(BaseModel):
    """A tool wrapping a function definition."""
    type: Literal["function"] = "function"
    function: FunctionDefinition


class ChatRequest(BaseModel):
    """Request payload for POST /v1/chat/completions."""
    model: str
    messages: list[ChatMessage]
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stream: bool = False
    stop: str | list[str] | None = None
    tools: list[ToolDefinition] | None = None
    tool_choice: str | dict[str, Any] | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    seed: int | None = None
    user: str | None = None
    metadata: dict[str, str] | None = None
    response_format: dict[str, Any] | None = None
    n: int | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = None
    parallel_tool_calls: bool | None = None


class ChatChoice(BaseModel):
    """A single completion choice with message and finish reason."""
    index: int
    message: ChatMessage
    finish_reason: str | None = None
    logprobs: dict[str, Any] | None = None


class Usage(BaseModel):
    """Token usage for a chat completion."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatResponse(BaseModel):
    """Response payload from POST /v1/chat/completions."""
    id: str
    object: str = "chat.completion"
    created: int = 0
    model: str
    choices: list[ChatChoice]
    usage: Usage | None = None
    system_fingerprint: str | None = None
