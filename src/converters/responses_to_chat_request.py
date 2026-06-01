"""Convert OpenAI Responses API requests to Chat Completions API format.

Maps the unified Responses input format (text, image, file, message items)
into the standard Chat Completions messages array, and converts tool
definitions and response format settings.
"""

from __future__ import annotations

from src.models.responses import (
    ResponsesRequest,
    ResponseInputItem,
    EasyInputMessage,
    ResponseInputText,
    ResponseInputImage,
    ResponseInputFile,
    FunctionTool,
)
from src.models.chat import ChatRequest, ChatMessage, ToolDefinition, FunctionDefinition


def responses_to_chat_request(resp_req: ResponsesRequest) -> ChatRequest:
    """Convert a Responses API request to a Chat Completions request.

    * Instructions become a system message.
    * String input becomes a single user message.
    * Array input items are converted individually.
    * Tools, temperature, top_p, and other params are mapped directly.
    * ``text.format`` becomes ``response_format``.
    """
    messages: list[ChatMessage] = []

    if resp_req.instructions:
        messages.append(
            ChatMessage(role="system", content=resp_req.instructions)
        )

    inp = resp_req.input
    if isinstance(inp, str):
        messages.append(ChatMessage(role="user", content=inp))
    else:
        for item in inp:
            msg = _convert_input_item(item)
            if msg:
                messages.append(msg)

    tools: list[ToolDefinition] | None = None
    if resp_req.tools:
        tools = [
            ToolDefinition(
                type="function",
                function=FunctionDefinition(
                    name=t.name,
                    description=t.description,
                    parameters=t.parameters,
                ),
            )
            for t in resp_req.tools
        ]

    response_format = None
    if resp_req.text and resp_req.text.format:
        response_format = resp_req.text.format

    return ChatRequest(
        model=resp_req.model,
        messages=messages,
        max_tokens=resp_req.max_output_tokens,
        temperature=resp_req.temperature,
        top_p=resp_req.top_p,
        stream=resp_req.stream,
        tools=tools or None,
        tool_choice=resp_req.tool_choice,
        response_format=response_format,
        metadata=resp_req.metadata,
        user=resp_req.user,
    )


def _convert_input_item(item: ResponseInputItem) -> ChatMessage | None:
    """Convert a single Responses input item to a ChatMessage."""
    if isinstance(item, EasyInputMessage):
        content = _convert_easy_content(item.content)
        role = item.role
        if role == "developer":
            role = "system"
        return ChatMessage(role=role, content=content)

    if isinstance(item, ResponseInputText):
        return ChatMessage(role="user", content=item.text)

    if isinstance(item, ResponseInputImage):
        return ChatMessage(
            role="user",
            content="[Image: {}]".format(item.image_url or item.file_id or "unknown"),
        )

    if isinstance(item, ResponseInputFile):
        return ChatMessage(
            role="user",
            content="[File: {}]".format(item.filename or item.file_id or "unknown"),
        )

    return None


def _convert_easy_content(
    content: str | list,
) -> str | list[dict]:
    """Convert EasyInputMessage content (simplified) into Chat API content parts.

    Handles text, image_url, and file parts.
    """
    if isinstance(content, str):
        return content
    parts: list[dict] = []
    for c in content:
        if isinstance(c, ResponseInputText):
            parts.append({"type": "text", "text": c.text})
        elif isinstance(c, ResponseInputImage):
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": c.image_url or "", "detail": c.detail or "auto"},
                }
            )
        elif isinstance(c, ResponseInputFile):
            parts.append(
                {
                    "type": "file",
                    "file": {
                        "file_id": c.file_id or "",
                        "filename": c.filename or "",
                    },
                }
            )
    return parts if parts else ""
