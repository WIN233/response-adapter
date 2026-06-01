"""Convert OpenAI Chat Completions responses to Responses API format.

Transforms the chat completion output (choices with messages, tool calls)
into the unified Responses format with output messages and function calls.
"""

from __future__ import annotations

from src.models.chat import ChatResponse
from src.models.responses import (
    ResponsesResponse,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseFunctionToolCall,
    ResponseUsage,
)


def chat_response_to_responses(chat_resp: ChatResponse) -> ResponsesResponse:
    """Convert a ChatCompletions response to a unified Responses response.

    Each choice becomes a ResponseOutputMessage. Tool calls in a choice are
    appended as ResponseFunctionToolCall items. Usage is mapped across.
    """
    output: list = []
    for choice in chat_resp.choices:
        msg = choice.message

        if msg.content or msg.content is None:
            text_parts: list[str] = []
            if msg.content:
                if isinstance(msg.content, str):
                    text_parts.append(msg.content)
                elif isinstance(msg.content, list):
                    for part in msg.content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text_parts.append(part.get("text", ""))

            text = "\n".join(text_parts)
            output.append(
                ResponseOutputMessage(
                    id=f"msg_{chat_resp.id}_{choice.index}",
                    content=[ResponseOutputText(text=text)] if text else [],
                )
            )

        if msg.tool_calls:
            for tc in msg.tool_calls:
                func = tc.get("function", {})
                output.append(
                    ResponseFunctionToolCall(
                        id=tc.get("id", f"fc_{choice.index}"),
                        name=func.get("name", ""),
                        arguments=func.get("arguments", "{}"),
                    )
                )

    usage = None
    if chat_resp.usage:
        usage = ResponseUsage(
            input_tokens=chat_resp.usage.prompt_tokens,
            output_tokens=chat_resp.usage.completion_tokens,
            total_tokens=chat_resp.usage.total_tokens,
        )

    return ResponsesResponse(
        id=chat_resp.id,
        model=chat_resp.model,
        output=output,
        usage=usage,
    )
