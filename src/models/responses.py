"""Pydantic models for OpenAI Responses API request/response schemas.

These models define the data structures used for the Responses API endpoint,
including input items (text, image, file, messages), function tools, and
the unified response format with output items and usage statistics.
"""

from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel


# --- Input item types ---

class ResponseInputText(BaseModel):
    """A text input item."""
    type: Literal["input_text"] = "input_text"
    text: str


class ResponseInputImage(BaseModel):
    """An image input item, referenced by URL or file ID."""
    type: Literal["input_image"] = "input_image"
    image_url: str | None = None
    file_id: str | None = None
    detail: str | None = None


class ResponseInputFile(BaseModel):
    """A file input item (binary data, file ID, or URL)."""
    type: Literal["input_file"] = "input_file"
    file_data: str | None = None
    file_id: str | None = None
    file_url: str | None = None
    filename: str | None = None
    detail: str | None = None


class EasyInputMessage(BaseModel):
    """A conversational input message with role and content."""
    type: Literal["message"] = "message"
    role: Literal["user", "assistant", "system", "developer"]
    content: str | list[ResponseInputText | ResponseInputImage | ResponseInputFile]


ResponseInputItem = ResponseInputText | ResponseInputImage | ResponseInputFile | EasyInputMessage


# --- Tool definitions ---

class FunctionTool(BaseModel):
    """A function tool definition for the model to call."""
    type: Literal["function"] = "function"
    name: str
    description: str = ""
    parameters: dict[str, Any] = {}
    strict: bool = False


# --- Request ---

class ResponseTextConfig(BaseModel):
    """Configuration for text output formatting."""
    format: dict[str, Any] | None = None


class ResponsesRequest(BaseModel):
    """Top-level request model for POST /v1/responses."""
    model: str
    input: str | list[ResponseInputItem]
    instructions: str | None = None
    max_output_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stream: bool = False
    tools: list[FunctionTool] | None = None
    tool_choice: str | dict[str, Any] | None = None
    text: ResponseTextConfig | None = None
    metadata: dict[str, str] | None = None
    store: bool | None = None
    previous_response_id: str | None = None
    user: str | None = None


# --- Output item types ---

class ResponseOutputText(BaseModel):
    """A text content block within a response output message."""
    type: Literal["output_text"] = "output_text"
    text: str
    annotations: list[dict[str, Any]] = []


class ResponseOutputMessage(BaseModel):
    """An assistant message in the response output."""
    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: list[ResponseOutputText]
    status: str = "completed"


class ResponseFunctionToolCall(BaseModel):
    """A function call request in the response output."""
    id: str
    type: Literal["function_call"] = "function_call"
    name: str
    arguments: str
    status: str = "completed"


ResponseOutputItem = ResponseOutputMessage | ResponseFunctionToolCall


# --- Usage & Response ---

class ResponseUsage(BaseModel):
    """Token usage statistics for a response."""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class ResponsesResponse(BaseModel):
    """Top-level response model returned by POST /v1/responses."""
    id: str
    object: str = "response"
    model: str
    output: list[ResponseOutputItem]
    usage: ResponseUsage | None = None
    status: str = "completed"
