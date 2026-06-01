# toresponses Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a proxy server that converts OpenAI Chat Completions API and Anthropic Messages API requests/responses to/from the OpenAI Responses API format, enabling clients using older formats to seamlessly use the new Responses API.

**Architecture:** FastAPI server with two proxy endpoints (`/v1/chat/completions` and `/v1/messages`). Each endpoint converts incoming requests to OpenAI Responses API format, forwards to OpenAI's `/v1/responses` endpoint via httpx, then converts Responses API output back to the original format (including streaming SSE events).

**Tech Stack:** Python 3.14+, FastAPI, uvicorn, httpx, Pydantic

---

### Task 1: Project structure and dependencies

**Files:**
- Create: `src/__init__.py`
- Create: `src/models/__init__.py`
- Create: `src/converters/__init__.py`
- Create: `src/proxy/__init__.py`
- Create: `src/streaming/__init__.py`
- Modify: `pyproject.toml`

- [x] **Step 1: Create directory structure**

```bash
mkdir -p src/models src/converters src/proxy src/streaming docs/superpowers/plans
touch src/__init__.py src/models/__init__.py src/converters/__init__.py src/proxy/__init__.py src/streaming/__init__.py
```

- [x] **Step 2: Update pyproject.toml with dependencies**

```toml
[project]
name = "toresponses"
version = "0.1.0"
description = "Proxy server converting Chat/Anthropic API formats to OpenAI Responses API"
readme = "README.md"
requires-python = ">=3.14"
dependencies = [
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.34.0",
    "httpx>=0.28.0",
    "pydantic>=2.0.0",
]

[project.scripts]
toresponses = "main:main"
```

### Task 2: Data models

**Files:**
- Create: `src/models/chat.py` — Chat Completions request/response types
- Create: `src/models/anthropic.py` — Anthropic Messages request/response types  
- Create: `src/models/responses.py` — OpenAI Responses API types

- [ ] **Step 1: Create Chat Completions models**

```python
from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool", "developer"]
    content: str | list[dict[str, Any]] | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    name: str | None = None


class FunctionDefinition(BaseModel):
    name: str
    description: str = ""
    parameters: dict[str, Any] = {}


class ToolDefinition(BaseModel):
    type: Literal["function"] = "function"
    function: FunctionDefinition


class ChatRequest(BaseModel):
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
    index: int
    message: ChatMessage
    finish_reason: str | None = None
    logprobs: dict[str, Any] | None = None


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int = 0
    model: str
    choices: list[ChatChoice]
    usage: Usage | None = None
    system_fingerprint: str | None = None
```

- [ ] **Step 2: Create Anthropic Messages models**

```python
from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel


class AnthropicTextContent(BaseModel):
    type: Literal["text"] = "text"
    text: str


class AnthropicImageSource(BaseModel):
    type: Literal["base64"]
    media_type: str
    data: str


class AnthropicImageContent(BaseModel):
    type: Literal["image"] = "image"
    source: AnthropicImageSource


class AnthropicToolUseContent(BaseModel):
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, Any]


class AnthropicToolResultContent(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: str | list[dict[str, Any]]


AnthropicContentBlock = AnthropicTextContent | AnthropicImageContent | AnthropicToolUseContent | AnthropicToolResultContent


class AnthropicMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str | list[AnthropicContentBlock]


class AnthropicToolInputSchema(BaseModel):
    type: Literal["object"] = "object"
    properties: dict[str, Any] = {}
    required: list[str] = []


class AnthropicToolDefinition(BaseModel):
    name: str
    description: str = ""
    input_schema: AnthropicToolInputSchema


class AnthropicRequest(BaseModel):
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
    input_tokens: int = 0
    output_tokens: int = 0


class AnthropicResponse(BaseModel):
    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: list[AnthropicTextContent | AnthropicToolUseContent]
    model: str
    stop_reason: str | None = None
    stop_sequence: str | None = None
    usage: AnthropicUsage
```

- [ ] **Step 3: Create Responses API models**

```python
from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel


# ---- Input items ----

class ResponseInputText(BaseModel):
    type: Literal["input_text"] = "input_text"
    text: str


class ResponseInputImage(BaseModel):
    type: Literal["input_image"] = "input_image"
    image_url: str | None = None
    file_id: str | None = None
    detail: str | None = None


class ResponseInputFile(BaseModel):
    type: Literal["input_file"] = "input_file"
    file_data: str | None = None
    file_id: str | None = None
    file_url: str | None = None
    filename: str | None = None
    detail: str | None = None


class EasyInputMessage(BaseModel):
    type: Literal["message"] = "message"
    role: Literal["user", "assistant", "system", "developer"]
    content: str | list[ResponseInputText | ResponseInputImage | ResponseInputFile]


ResponseInputItem = ResponseInputText | ResponseInputImage | ResponseInputFile | EasyInputMessage


# ---- Function tool ----

class FunctionTool(BaseModel):
    type: Literal["function"] = "function"
    name: str
    description: str = ""
    parameters: dict[str, Any] = {}
    strict: bool = False


# ---- Request ----

class ResponseTextConfig(BaseModel):
    format: dict[str, Any] | None = None


class ResponsesRequest(BaseModel):
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


# ---- Output items ----

class ResponseOutputText(BaseModel):
    type: Literal["output_text"] = "output_text"
    text: str
    annotations: list[dict[str, Any]] = []


class ResponseOutputMessage(BaseModel):
    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: list[ResponseOutputText]
    status: str = "completed"


class ResponseFunctionToolCall(BaseModel):
    id: str
    type: Literal["function_call"] = "function_call"
    name: str
    arguments: str
    status: str = "completed"


ResponseOutputItem = ResponseOutputMessage | ResponseFunctionToolCall


class ResponseUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class ResponsesResponse(BaseModel):
    id: str
    object: str = "response"
    model: str
    output: list[ResponseOutputItem]
    usage: ResponseUsage | None = None
    status: str = "completed"
```

### Task 3: Chat Completions → Responses converter

**Files:**
- Create: `src/converters/chat_to_responses.py`

- [ ] **Step 1: Implement converter**

```python
from __future__ import annotations

from src.models.chat import ChatRequest, ChatMessage
from src.models.responses import (
    ResponsesRequest, ResponseInputItem, EasyInputMessage,
    ResponseInputText, FunctionTool, ResponseTextConfig,
)


def convert_chat_to_responses(chat_req: ChatRequest) -> ResponsesRequest:
    instructions: str | None = None
    input_items: list[ResponseInputItem] = []

    for msg in chat_req.messages:
        if msg.role in ("system", "developer"):
            if instructions is None:
                instructions = _extract_text(msg.content)
            else:
                instructions = instructions + "\n" + _extract_text(msg.content)
            continue

        if msg.role == "tool":
            input_items.append(
                ResponseInputText(
                    type="input_text",
                    text=f"<tool_result>{_extract_text(msg.content)}</tool_result>"
                )
            )
            continue

        role = "assistant" if msg.role == "assistant" else "user"
        content = _convert_content(msg.content, msg.tool_calls)
        input_items.append(
            EasyInputMessage(
                type="message",
                role=role,
                content=content,
            )
        )

    tools: list[FunctionTool] | None = None
    if chat_req.tools:
        tools = [
            FunctionTool(
                type="function",
                name=t.function.name,
                description=t.function.description,
                parameters=t.function.parameters,
            )
            for t in chat_req.tools
        ]

    text_config: ResponseTextConfig | None = None
    if chat_req.response_format:
        text_config = ResponseTextConfig(format=chat_req.response_format)

    return ResponsesRequest(
        model=chat_req.model,
        input=input_items if len(input_items) != 1 else _simplify_input(input_items),
        instructions=instructions,
        max_output_tokens=chat_req.max_tokens,
        temperature=chat_req.temperature,
        top_p=chat_req.top_p,
        stream=chat_req.stream,
        tools=tools or None,
        tool_choice=chat_req.tool_choice,
        text=text_config,
        metadata=chat_req.metadata,
        user=chat_req.user,
    )


def _extract_text(content: str | list | None) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            parts.append(item.get("text", ""))
    return " ".join(parts)


def _convert_content(
    content: str | list | None,
    tool_calls: list[dict] | None,
) -> str | list[ResponseInputItem]:
    if tool_calls:
        return _tool_calls_to_text(tool_calls)
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    items: list[ResponseInputItem] = []
    for part in content:
        if isinstance(part, dict):
            if part.get("type") == "text":
                items.append(ResponseInputText(text=part["text"]))
    return items if items else ""


def _tool_calls_to_text(tool_calls: list[dict]) -> str:
    parts = []
    for tc in tool_calls:
        func = tc.get("function", {})
        parts.append(
            f'<tool_call name="{func.get("name", "")}">'
            f'{func.get("arguments", "{}")}'
            f"</tool_call>"
        )
    return "\n".join(parts)


def _simplify_input(items: list[ResponseInputItem]) -> str | list[ResponseInputItem]:
    if len(items) == 1 and isinstance(items[0], EasyInputMessage):
        content = items[0].content
        if isinstance(content, str):
            return content
    return items
```

### Task 4: Anthropic Messages → Responses converter

**Files:**
- Create: `src/converters/anthropic_to_responses.py`

- [ ] **Step 1: Implement converter**

```python
from __future__ import annotations

from src.models.anthropic import AnthropicRequest, AnthropicMessage
from src.models.responses import (
    ResponsesRequest, ResponseInputItem, EasyInputMessage,
    ResponseInputText, FunctionTool,
)


def convert_anthropic_to_responses(ant_req: AnthropicRequest) -> ResponsesRequest:
    instructions: str | None = None
    if ant_req.system:
        if isinstance(ant_req.system, str):
            instructions = ant_req.system
        else:
            parts = [
                b.get("text", "") for b in ant_req.system
                if isinstance(b, dict) and b.get("type") == "text"
            ]
            instructions = "\n".join(parts) if parts else None

    input_items: list[ResponseInputItem] = []
    for msg in ant_req.messages:
        role = "user" if msg.role == "user" else "assistant"
        content = _convert_anthropic_content(msg.content)
        input_items.append(
            EasyInputMessage(type="message", role=role, content=content)
        )

    tools: list[FunctionTool] | None = None
    if ant_req.tools:
        tools = [
            FunctionTool(
                type="function",
                name=t.name,
                description=t.description,
                parameters=t.input_schema.model_dump() if hasattr(t.input_schema, 'model_dump') else {},
            )
            for t in ant_req.tools
        ]

    tool_choice = None
    if ant_req.tool_choice:
        tc = ant_req.tool_choice
        if tc.get("type") == "any":
            tool_choice = "auto"
        elif tc.get("type") == "tool":
            tool_choice = {"type": "function", "name": tc.get("name", "")}
        elif tc.get("type") == "auto":
            tool_choice = "auto"

    return ResponsesRequest(
        model=ant_req.model,
        input=input_items if len(input_items) != 1 else input_items[0].content if isinstance(input_items[0].content, str) else input_items,
        instructions=instructions,
        max_output_tokens=ant_req.max_tokens,
        temperature=ant_req.temperature,
        top_p=ant_req.top_p,
        stream=ant_req.stream,
        tools=tools or None,
        tool_choice=tool_choice,
    )


def _convert_anthropic_content(content: str | list) -> str | list[ResponseInputItem]:
    if isinstance(content, str):
        return content
    items: list[ResponseInputItem] = []
    for block in content:
        if isinstance(block, dict):
            btype = block.get("type")
            if btype == "text":
                items.append(ResponseInputText(text=block.get("text", "")))
            elif btype == "image":
                source = block.get("source", {})
                items.append(
                    ResponseInputText(
                        text=f"[Image: {source.get('media_type', '')} (base64)]"
                    )
                )
            elif btype == "tool_use":
                items.append(
                    ResponseInputText(
                        text=f'<tool_call name="{block.get("name", "")}">{block.get("input", {})}</tool_call>'
                    )
                )
            elif btype == "tool_result":
                items.append(
                    ResponseInputText(
                        text=f"<tool_result>{block.get('content', '')}</tool_result>"
                    )
                )
    return items if items else ""
```

### Task 5: Responses → Chat Completions converter (reverse)

**Files:**
- Create: `src/converters/responses_to_chat.py`

- [ ] **Step 1: Implement reverse converter**

```python
from __future__ import annotations

import time
import json
from src.models.responses import ResponsesResponse, ResponseOutputItem, ResponseOutputMessage, ResponseFunctionToolCall
from src.models.chat import ChatResponse, ChatChoice, ChatMessage, Usage


def convert_responses_to_chat(
    resp: ResponsesResponse,
    original_model: str,
) -> ChatResponse:
    message = ChatMessage(role="assistant", content=None)
    tool_calls: list[dict] = []
    finish_reason: str | None = "stop"

    for item in resp.output:
        if isinstance(item, ResponseOutputMessage):
            text_parts: list[str] = []
            for content_block in item.content:
                text_parts.append(content_block.text)
            message.content = "\n".join(text_parts) if text_parts else None

        elif isinstance(item, ResponseFunctionToolCall):
            tool_calls.append({
                "id": item.id,
                "type": "function",
                "function": {
                    "name": item.name,
                    "arguments": item.arguments,
                },
            })
            finish_reason = "tool_calls"

    if tool_calls:
        message.tool_calls = tool_calls

    usage = None
    if resp.usage:
        usage = Usage(
            prompt_tokens=resp.usage.input_tokens,
            completion_tokens=resp.usage.output_tokens,
            total_tokens=resp.usage.total_tokens,
        )

    return ChatResponse(
        id=resp.id,
        created=int(time.time()),
        model=original_model,
        choices=[
            ChatChoice(
                index=0,
                message=message,
                finish_reason=finish_reason,
            )
        ],
        usage=usage,
    )
```

### Task 6: Responses → Anthropic Messages converter (reverse)

**Files:**
- Create: `src/converters/responses_to_anthropic.py`

- [ ] **Step 1: Implement reverse converter**

```python
from __future__ import annotations

from src.models.responses import ResponsesResponse, ResponseOutputItem, ResponseOutputMessage, ResponseFunctionToolCall
from src.models.anthropic import (
    AnthropicResponse, AnthropicTextContent, AnthropicToolUseContent,
    AnthropicUsage,
)


def convert_responses_to_anthropic(
    resp: ResponsesResponse,
    original_model: str,
) -> AnthropicResponse:
    content: list[AnthropicTextContent | AnthropicToolUseContent] = []
    stop_reason: str | None = "end_turn"

    for item in resp.output:
        if isinstance(item, ResponseOutputMessage):
            for content_block in item.content:
                content.append(
                    AnthropicTextContent(type="text", text=content_block.text)
                )

        elif isinstance(item, ResponseFunctionToolCall):
            content.append(
                AnthropicToolUseContent(
                    type="tool_use",
                    id=item.id,
                    name=item.name,
                    input=json.loads(item.arguments) if isinstance(item.arguments, str) else item.arguments,
                )
            )
            stop_reason = "tool_use"

    if not content:
        content.append(AnthropicTextContent(type="text", text=""))

    usage = AnthropicUsage()
    if resp.usage:
        usage = AnthropicUsage(
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
        )

    return AnthropicResponse(
        id=resp.id,
        model=original_model,
        content=content,
        stop_reason=stop_reason,
        usage=usage,
    )
```

### Task 7: Proxy client

**Files:**
- Create: `src/proxy/client.py`

- [ ] **Step 1: Implement OpenAI Responses API client**

```python
from __future__ import annotations

import os
import json
from typing import AsyncIterator
import httpx
from src.models.responses import ResponsesRequest, ResponsesResponse
from src.streaming.events import StreamEvent


class ResponsesClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 120.0,
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def create_response(
        self, request: ResponsesRequest
    ) -> ResponsesResponse:
        payload = request.model_dump(exclude_none=True)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/responses",
                headers=self._headers(),
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
        return ResponsesResponse(**data)

    async def create_response_stream(
        self, request: ResponsesRequest,
    ) -> AsyncIterator[StreamEvent]:
        payload = request.model_dump(exclude_none=True)
        payload["stream"] = True
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/responses",
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
                        event_data = json.loads(data_str)
                        yield StreamEvent(
                            type=event_data.get("type", ""),
                            data=event_data,
                        )
                    except json.JSONDecodeError:
                        continue

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
```

- [ ] **Step 2: Create streaming event types**

```python
# src/streaming/events.py
from __future__ import annotations

from typing import Any
from pydantic import BaseModel


class StreamEvent(BaseModel):
    type: str
    data: dict[str, Any]
```

### Task 8: FastAPI server routes

**Files:**
- Create: `src/proxy/server.py`

- [ ] **Step 1: Implement FastAPI app with both proxy routes**

```python
from __future__ import annotations

import json
import time
import uuid
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse

from src.models.chat import ChatRequest, ChatResponse, ChatChoice, ChatMessage, Usage
from src.models.anthropic import AnthropicRequest, AnthropicResponse
from src.models.responses import ResponsesRequest
from src.converters.chat_to_responses import convert_chat_to_responses
from src.converters.anthropic_to_responses import convert_anthropic_to_responses
from src.converters.responses_to_chat import convert_responses_to_chat
from src.converters.responses_to_anthropic import convert_responses_to_anthropic
from src.proxy.client import ResponsesClient


def create_app(client: ResponsesClient) -> FastAPI:
    app = FastAPI(title="toresponses")

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        chat_req = ChatRequest(**body)

        responses_req = convert_chat_to_responses(chat_req)

        if chat_req.stream:
            return StreamingResponse(
                _stream_chat(client, responses_req, chat_req.model),
                media_type="text/event-stream",
            )

        responses_resp = await client.create_response(responses_req)
        chat_resp = convert_responses_to_chat(responses_resp, chat_req.model)
        return JSONResponse(content=chat_resp.model_dump(exclude_none=True))

    @app.post("/v1/messages")
    async def anthropic_messages(request: Request):
        body = await request.json()
        ant_req = AnthropicRequest(**body)

        responses_req = convert_anthropic_to_responses(ant_req)

        if ant_req.stream:
            return StreamingResponse(
                _stream_anthropic(client, responses_req, ant_req.model),
                media_type="text/event-stream",
            )

        responses_resp = await client.create_response(responses_req)
        ant_resp = convert_responses_to_anthropic(responses_resp, ant_req.model)
        return JSONResponse(content=ant_resp.model_dump(exclude_none=True))

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app


async def _stream_chat(
    client: ResponsesClient,
    req: ResponsesRequest,
    model: str,
) -> AsyncIterator[str]:
    response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    full_content = ""
    finish_reason: str | None = None

    yield f"data: {json.dumps({'id': response_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model, 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': None}, 'logprobs': None, 'finish_reason': None}]})}\n\n"

    async for event in client.create_response_stream(req):
        etype = event.type
        data = event.data

        if etype == "response.output_item.added":
            pass
        elif etype == "response.content_part.added":
            pass
        elif etype == "response.text.delta":
            delta = data.get("delta", "")
            full_content += delta
            yield f"data: {json.dumps({'id': response_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model, 'choices': [{'index': 0, 'delta': {'content': delta}, 'finish_reason': None}]})}\n\n"
        elif etype == "response.output_item.done":
            item = data.get("item", {})
            if item.get("type") == "function_call":
                fc = item
                yield f"data: {json.dumps({'id': response_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model, 'choices': [{'index': 0, 'delta': {'tool_calls': [{'index': 0, 'id': fc.get('id'), 'type': 'function', 'function': {'name': fc.get('name'), 'arguments': fc.get('arguments', '')}}]}, 'finish_reason': 'tool_calls'}]})}\n\n"
                finish_reason = "tool_calls"
        elif etype == "response.completed":
            finish_reason = finish_reason or "stop"

    yield f"data: {json.dumps({'id': response_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': finish_reason or 'stop'}]})}\n\n"
    yield "data: [DONE]\n\n"


async def _stream_anthropic(
    client: ResponsesClient,
    req: ResponsesRequest,
    model: str,
) -> AsyncIterator[str]:
    response_id = f"msg_{uuid.uuid4().hex[:24]}"
    full_content = ""
    stop_reason: str | None = None

    yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': response_id, 'type': 'message', 'role': 'assistant', 'content': [], 'model': model, 'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': 0, 'output_tokens': 0}}})}\n\n"

    yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"

    async for event in client.create_response_stream(req):
        etype = event.type
        data = event.data

        if etype == "response.text.delta":
            delta = data.get("delta", "")
            full_content += delta
            yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': delta}})}\n\n"

        elif etype == "response.completed":
            resp_data = data.get("response", data)
            usage = resp_data.get("usage", {})
            stop_reason = "end_turn"

    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"

    yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': stop_reason, 'stop_sequence': None}, 'usage': {'output_tokens': 0}})}\n\n"

    yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
```

### Task 9: Main entry point

**Files:**
- Modify: `main.py`

- [ ] **Step 1: Wire up entry point**

```python
from __future__ import annotations

import os
import uvicorn
from src.proxy.server import create_app
from src.proxy.client import ResponsesClient


def main():
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

    if not api_key:
        print("WARNING: OPENAI_API_KEY environment variable not set")

    client = ResponsesClient(api_key=api_key, base_url=base_url)
    app = create_app(client)

    print(f"Starting toresponses proxy on {host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
```

### Task 10: Verification

- [ ] **Step 1: Run syntax/type checks**

```bash
python -c "from src.models.chat import ChatRequest; print('Chat models OK')"
python -c "from src.models.anthropic import AnthropicRequest; print('Anthropic models OK')"
python -c "from src.models.responses import ResponsesRequest; print('Responses models OK')"
python -c "from src.converters.chat_to_responses import convert_chat_to_responses; print('Chat converter OK')"
python -c "from src.converters.anthropic_to_responses import convert_anthropic_to_responses; print('Anthropic converter OK')"
python -c "from src.converters.responses_to_chat import convert_responses_to_chat; print('Chat reverse OK')"
python -c "from src.converters.responses_to_anthropic import convert_responses_to_anthropic; print('Anthropic reverse OK')"
```

- [ ] **Step 2: Run the server and test with curl**

```bash
# Terminal 1
OPENAI_API_KEY=sk-... python main.py

# Terminal 2 - Chat endpoint test
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o","messages":[{"role":"user","content":"Hello"}],"max_tokens":50}'

# Terminal 3 - Anthropic endpoint test
curl -X POST http://localhost:8000/v1/messages \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o","messages":[{"role":"user","content":"Hello"}],"max_tokens":50}'
```
