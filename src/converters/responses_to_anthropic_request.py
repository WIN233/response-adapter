"""Convert OpenAI Responses API requests to Anthropic Messages API format.

Maps the unified Responses input format (text, image, file, message items
including XML-wrapped tool calls/results) into Anthropic's message content
blocks. Also translates tool definitions and tool_choice settings.
"""

from __future__ import annotations

import json
import re

from src.models.responses import (
    ResponsesRequest,
    ResponseInputItem,
    EasyInputMessage,
    ResponseInputText,
    ResponseInputImage,
    ResponseInputFile,
    FunctionTool,
)
from src.models.anthropic import (
    AnthropicRequest,
    AnthropicMessage,
    AnthropicTextContent,
    AnthropicImageContent,
    AnthropicImageSource,
    AnthropicToolUseContent,
    AnthropicToolResultContent,
    AnthropicToolDefinition,
    AnthropicToolInputSchema,
)


def responses_to_anthropic_request(resp_req: ResponsesRequest) -> AnthropicRequest:
    """Convert a Responses API request to an Anthropic Messages request.

    Handles:
    * Instructions → system prompt
    * Input items → AnthropicMessage with content blocks
    * XML-wrapped tool calls/results (``<tool_call>``, ``<tool_result>``)
    * Function tools → AnthropicToolDefinition
    * Tool choice mapping (required→any, none→none, function→tool)
    """
    messages: list[AnthropicMessage] = []
    tool_use_accumulators: dict[str, str] = {}

    inp = resp_req.input
    if isinstance(inp, str):
        messages.append(AnthropicMessage(role="user", content=inp))
    else:
        for item in inp:
            result = _convert_input_item_to_anthropic(item, tool_use_accumulators)
            if result:
                messages.append(result)

    system: str | None = resp_req.instructions

    tools: list[AnthropicToolDefinition] | None = None
    if resp_req.tools:
        tools = [
            AnthropicToolDefinition(
                name=t.name,
                description=t.description,
                input_schema=AnthropicToolInputSchema(
                    type="object",
                    properties=t.parameters.get("properties", {}),
                    required=t.parameters.get("required", []),
                ),
            )
            for t in resp_req.tools
        ]

    tool_choice = None
    if resp_req.tool_choice:
        if isinstance(resp_req.tool_choice, str):
            if resp_req.tool_choice == "required":
                tool_choice = {"type": "any"}
            elif resp_req.tool_choice == "none":
                tool_choice = {"type": "none"}
            else:
                tool_choice = {"type": "auto"}
        elif isinstance(resp_req.tool_choice, dict):
            tc = resp_req.tool_choice
            if tc.get("type") == "function":
                tool_choice = {"type": "tool", "name": tc.get("name", "")}
            else:
                tool_choice = {"type": "auto"}

    max_tokens = resp_req.max_output_tokens or 4096

    return AnthropicRequest(
        model=resp_req.model,
        messages=messages,
        max_tokens=max_tokens,
        system=system,
        temperature=resp_req.temperature,
        top_p=resp_req.top_p,
        stream=resp_req.stream,
        tools=tools or None,
        tool_choice=tool_choice,
    )


def _convert_input_item_to_anthropic(
    item: ResponseInputItem,
    tool_use_accumulators: dict[str, str],
) -> AnthropicMessage | None:
    """Convert a single Responses input item to an AnthropicMessage."""
    if isinstance(item, EasyInputMessage):
        blocks = _convert_easy_content_to_anthropic_blocks(
            item.content, item.role, tool_use_accumulators
        )
        role = "assistant" if item.role == "assistant" else "user"
        if not blocks:
            return None
        if len(blocks) == 1 and isinstance(blocks[0], AnthropicTextContent):
            return AnthropicMessage(role=role, content=blocks[0].text)
        return AnthropicMessage(role=role, content=blocks)

    if isinstance(item, ResponseInputText):
        return AnthropicMessage(role="user", content=item.text)

    if isinstance(item, ResponseInputImage):
        return AnthropicMessage(
            role="user",
            content=[
                AnthropicImageContent(
                    source=AnthropicImageSource(
                        type="base64",
                        media_type="image/jpeg",
                        data=item.image_url or "",
                    )
                )
            ],
        )

    if isinstance(item, ResponseInputFile):
        return AnthropicMessage(
            role="user",
            content="[File: {}]".format(item.filename or item.file_id or "unknown"),
        )

    return None


def _convert_easy_content_to_anthropic_blocks(
    content: str | list,
    role: str,
    tool_use_accumulators: dict[str, str],
) -> list:
    """Convert simplified EasyInputMessage content to Anthropic content blocks.

    Detects and accumulates ``<tool_call>`` / ``<tool_result>`` XML fragments
    across multiple content parts and emits them as tool_use blocks.
    """
    if isinstance(content, str):
        return [AnthropicTextContent(text=content)]

    blocks: list = []
    for c in content:
        if isinstance(c, ResponseInputText):
            text = c.text
            if text.startswith("<tool_call"):
                _accumulate_tool_call(text, tool_use_accumulators)
                continue
            if text.startswith("<tool_result"):
                _accumulate_tool_result(text, tool_use_accumulators)
                continue
            blocks.append(AnthropicTextContent(text=text))

        elif isinstance(c, ResponseInputImage):
            blocks.append(
                AnthropicImageContent(
                    source=AnthropicImageSource(
                        type="base64",
                        media_type="image/jpeg",
                        data=c.image_url or "",
                    )
                )
            )

    if tool_use_accumulators.get("pending_call"):
        blocks.append(
            AnthropicToolUseContent(
                type="tool_use",
                id=tool_use_accumulators.get("call_id", ""),
                name=tool_use_accumulators.get("call_name", ""),
                input=json.loads(tool_use_accumulators["pending_call"]) if tool_use_accumulators.get("pending_call") else {},
            )
        )
        tool_use_accumulators.clear()

    return blocks


def _accumulate_tool_call(text: str, acc: dict[str, str]) -> None:
    """Parse a ``<tool_call>`` XML fragment and accumulate for conversion."""
    name_match = re.search(r'name="([^"]+)"', text)
    if name_match:
        acc["call_name"] = name_match.group(1)
    content_start = text.find(">") + 1
    content_end = text.rfind("</tool_call>")
    if content_start > 0 and content_end > content_start:
        acc["pending_call"] = text[content_start:content_end]
    acc["call_id"] = f"tu_{hash(text) % 10000000:07d}"


def _accumulate_tool_result(text: str, acc: dict[str, str]) -> None:
    """Parse a ``<tool_result>`` XML fragment and accumulate for conversion."""
    content_start = text.find(">") + 1
    content_end = text.rfind("</tool_result>")
    if content_start > 0 and content_end > content_start:
        result_content = text[content_start:content_end]
        acc["pending_result"] = result_content
