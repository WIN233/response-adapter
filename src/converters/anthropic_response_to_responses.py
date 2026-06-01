"""Convert Anthropic Messages API responses to Responses API format.

Transforms Anthropic's message response (content blocks with text and
tool_use) into the unified Responses format with output messages and
function call items.
"""

from __future__ import annotations

from src.models.anthropic import AnthropicResponse
from src.models.responses import (
    ResponsesResponse,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseFunctionToolCall,
    ResponseUsage,
)


def anthropic_response_to_responses(ant_resp: AnthropicResponse) -> ResponsesResponse:
    """Convert an Anthropic Messages response to a unified Responses response.

    * Text content blocks are joined into a single ResponseOutputMessage.
    * Tool_use blocks become ResponseFunctionToolCall items.
    * Usage is mapped (total = input + output).
    """
    output: list = []
    text_parts: list[str] = []

    for block in ant_resp.content:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "tool_use":
            import json
            output.append(
                ResponseFunctionToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=json.dumps(block.input) if not isinstance(block.input, str) else block.input,
                )
            )

    text = "\n".join(text_parts)
    output.insert(
        0,
        ResponseOutputMessage(
            id=f"msg_{ant_resp.id}",
            content=[ResponseOutputText(text=text)] if text else [],
        ),
    )

    usage = None
    if ant_resp.usage:
        usage = ResponseUsage(
            input_tokens=ant_resp.usage.input_tokens,
            output_tokens=ant_resp.usage.output_tokens,
            total_tokens=ant_resp.usage.input_tokens + ant_resp.usage.output_tokens,
        )

    return ResponsesResponse(
        id=ant_resp.id,
        model=ant_resp.model,
        output=output,
        usage=usage,
    )
