"""Pydantic models for Anthropic Messages API.

Defines request/response schemas for the Anthropic /v1/messages endpoint,
including content blocks (text, image, tool_use, tool_result), message
format, tool definitions, and usage statistics.
"""

from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel


# --- Content block types ---

class AnthropicTextContent(BaseModel):
    """A text content block."""
    type: Literal["text"] = "text"
    text: str


class AnthropicImageSource(BaseModel):
    """Base64-encoded image source."""
    type: Literal["base64"]
    media_type: str
    data: str


class AnthropicImageContent(BaseModel):
    """An image content block."""
    type: Literal["image"] = "image"
    source: AnthropicImageSource


class AnthropicToolUseContent(BaseModel):
    """A tool use content block (model requesting to call a function)."""
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, Any]


class AnthropicToolResultContent(BaseModel):
    """A tool result content block (response from a tool call)."""
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: str | list[dict[str, Any]]


AnthropicContentBlock = (
    AnthropicTextContent
    | AnthropicImageContent
    | AnthropicToolUseContent
    | AnthropicToolResultContent
)


# --- Message & Tools ---

class AnthropicMessage(BaseModel):
    """A message in the conversation with role and content."""
    role: Literal["user", "assistant"]
    content: str | list[AnthropicContentBlock]


class AnthropicToolInputSchema(BaseModel):
    """JSON Schema for a tool's input parameters."""
    type: Literal["object"] = "object"
    properties: dict[str, Any] = {}
    required: list[str] = []


class AnthropicToolDefinition(BaseModel):
    """A tool definition for the Anthropic API."""
    name: str
    description: str = ""
    input_schema: AnthropicToolInputSchema


# --- Request & Response ---

class AnthropicRequest(BaseModel):
    """Request payload for POST /v1/messages."""
    model: str
    messages: list[AnthropicMessage]
    max_tokens: int
    system: str | list[dict[str, Any]] | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    stream: bool = False
    stop_sequences: list[str] | None = None
    tools: list[AnthropicToolDefinition] | None = None
    tool_choice: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None


class AnthropicUsage(BaseModel):
    """Token usage returned by the Anthropic API."""
    input_tokens: int = 0
    output_tokens: int = 0


class AnthropicResponse(BaseModel):
    """Response payload from POST /v1/messages."""
    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: list[AnthropicTextContent | AnthropicToolUseContent]
    model: str
    stop_reason: str | None = None
    stop_sequence: str | None = None
    usage: AnthropicUsage
