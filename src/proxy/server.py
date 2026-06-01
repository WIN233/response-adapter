"""FastAPI proxy server that translates between OpenAI Responses API and backend APIs.

Accepts Requests API format at ``POST /v1/responses``, converts to either
Chat Completions or Anthropic Messages format, calls the backend, and
returns the result in Responses API format.  Supports streaming (SSE) and
a simple WebUI at ``GET /`` for manual testing.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse

from src.models.responses import ResponsesRequest
from src.models.chat import ChatResponse
from src.models.anthropic import AnthropicResponse
from src.converters.responses_to_chat_request import responses_to_chat_request
from src.converters.chat_response_to_responses import chat_response_to_responses
from src.converters.responses_to_anthropic_request import responses_to_anthropic_request
from src.converters.anthropic_response_to_responses import anthropic_response_to_responses
from src.proxy.client import ChatClient, AnthropicClient


def create_app(
    default_target: str = "chat",
    chat_client: ChatClient | None = None,
    anthropic_client: AnthropicClient | None = None,
) -> FastAPI:
    app = FastAPI(title="response-adapter")

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        return JSONResponse(
            status_code=500,
            content={"error": {"message": str(exc), "type": type(exc).__name__}},
        )

    webui_html = _build_webui()

    @app.get("/")
    async def webui():
        return HTMLResponse(content=webui_html)

    @app.post("/v1/responses")
    async def responses_proxy(request: Request):
        body = await request.json()
        resp_req = ResponsesRequest(**body)

        target = request.headers.get("x-target-backend", default_target)

        if target == "anthropic":
            return await _proxy_to_anthropic(resp_req, anthropic_client)
        return await _proxy_to_chat(resp_req, chat_client)

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "default_target": default_target,
            "chat_configured": chat_client is not None,
            "anthropic_configured": anthropic_client is not None,
        }

    return app


async def _proxy_to_chat(
    resp_req: ResponsesRequest,
    client: ChatClient | None,
) -> JSONResponse | StreamingResponse:
    if not client:
        return JSONResponse(
            status_code=500,
            content={"error": "ChatClient not configured (set OPENAI_API_KEY)"},
        )

    chat_req = responses_to_chat_request(resp_req)

    if resp_req.stream:
        return StreamingResponse(
            _stream_chat_to_responses(client, chat_req.model_dump(exclude_none=True), resp_req.model),
            media_type="text/event-stream",
        )

    payload = chat_req.model_dump(exclude_none=True)
    chat_resp_data = await client.create_completion(payload)
    chat_resp = ChatResponse(**chat_resp_data)
    responses_resp = chat_response_to_responses(chat_resp)
    return JSONResponse(content=responses_resp.model_dump(exclude_none=True))


async def _proxy_to_anthropic(
    resp_req: ResponsesRequest,
    client: AnthropicClient | None,
) -> JSONResponse | StreamingResponse:
    if not client:
        return JSONResponse(
            status_code=500,
            content={"error": "AnthropicClient not configured (set ANTHROPIC_API_KEY)"},
        )

    ant_req = responses_to_anthropic_request(resp_req)

    if resp_req.stream:
        return StreamingResponse(
            _stream_anthropic_to_responses(client, ant_req.model_dump(exclude_none=True), resp_req.model),
            media_type="text/event-stream",
        )

    payload = ant_req.model_dump(exclude_none=True)
    ant_resp_data = await client.create_message(payload)
    ant_resp = AnthropicResponse(**ant_resp_data)
    responses_resp = anthropic_response_to_responses(ant_resp)
    return JSONResponse(content=responses_resp.model_dump(exclude_none=True))


async def _stream_chat_to_responses(
    client: ChatClient,
    payload: dict,
    model: str,
) -> AsyncIterator[str]:
    response_id = f"resp_{uuid.uuid4().hex[:16]}"
    msg_id = f"msg_{response_id}"
    now = int(time.time())
    output_items: list[dict] = []
    current_text = ""
    finish_reason: str | None = None

    yield _sse_event("response.created", {"response": {"id": response_id, "model": model, "output": [], "created_at": now}})

    last_usage: dict = {}
    item_added = False

    async for chunk in client.stream_completion(payload):
        choices = chunk.get("choices", [])
        usage = chunk.get("usage")
        if usage:
            last_usage = usage

        for choice in choices:
            delta = choice.get("delta", {})
            fr = choice.get("finish_reason")

            # Emit output_item.added once when assistant role chunk is received
            if delta.get("role") == "assistant" and not item_added:
                item_added = True
                yield _sse_event("response.output_item.added", {
                    "output_index": 0,
                    "item": {"id": msg_id, "type": "message", "role": "assistant", "content": []},
                })

            # Stream text deltas as they arrive from the backend
            content = delta.get("content")
            if content:
                current_text += content
                yield _sse_event("response.output_text.delta", {
                    "delta": content,
                    "item_id": msg_id,
                    "output_index": 0,
                })

            # Handle tool calls (function calling) — each tool call index
            tool_calls = delta.get("tool_calls", [])
            for tc in tool_calls:
                tc_index = tc.get("index", 0)
                func = tc.get("function", {})
                tc_id = tc.get("id") or f"fc_{tc_index}"

                yield _sse_event("response.output_item.added", {
                    "output_index": len(output_items) + 1,
                    "item": {"id": tc_id, "type": "function_call", "name": func.get("name", ""), "arguments": ""},
                })

                if func.get("arguments"):
                    yield _sse_event("response.function_call_arguments.delta", {
                        "delta": func["arguments"],
                        "item_id": tc_id,
                        "output_index": len(output_items) + 1,
                    })

                output_items.append({
                    "id": tc_id,
                    "type": "function_call",
                    "name": func.get("name", ""),
                    "arguments": func.get("arguments", ""),
                })
                finish_reason = "tool_calls"

            if fr:
                finish_reason = fr

    # Finalize the text output item
    if item_added:
        yield _sse_event("response.output_item.done", {
            "output_index": 0,
            "item": {"id": msg_id, "type": "message", "role": "assistant", "content": [{"type": "text", "text": current_text}]},
        })

    # Final completed event with full response and usage info
    yield _sse_event("response.completed", {
        "response": {
            "id": response_id,
            "model": model,
            "created_at": now,
            "output": [
                {"id": msg_id, "type": "message", "role": "assistant", "content": [{"type": "text", "text": current_text}]},
                *output_items,
            ],
            "usage": {
                "input_tokens": last_usage.get("prompt_tokens", 0),
                "output_tokens": last_usage.get("completion_tokens", 0),
                "total_tokens": last_usage.get("total_tokens", 0),
            },
        }
    })

    yield "data: [DONE]\n\n"


async def _stream_anthropic_to_responses(
    client: AnthropicClient,
    payload: dict,
    model: str,
) -> AsyncIterator[str]:
    response_id = f"resp_{uuid.uuid4().hex[:16]}"
    msg_id = f"msg_{response_id}"
    now = int(time.time())
    current_text = ""
    current_tool_calls: list[dict] = []

    yield _sse_event("response.created", {"response": {"id": response_id, "model": model, "output": [], "created_at": now}})

    yield _sse_event("response.output_item.added", {
        "output_index": 0,
        "item": {"id": msg_id, "type": "message", "role": "assistant", "content": []},
    })

    last_usage: dict = {}

    async for chunk in client.stream_message(payload):
        ctype = chunk.get("type")

        if ctype == "content_block_delta":
            delta = chunk.get("delta", {})
            dtype = delta.get("type")
            if dtype == "text_delta":
                text = delta.get("text", "")
                current_text += text
                yield _sse_event("response.output_text.delta", {
                    "delta": text,
                    "item_id": msg_id,
                    "output_index": 0,
                })
            elif dtype == "input_json_delta" and current_tool_calls:
                # Accumulate streaming JSON arguments for the latest tool call
                partial = delta.get("partial_json", "")
                tc = current_tool_calls[-1]
                tc["arguments"] += partial
                yield _sse_event("response.function_call_arguments.delta", {
                    "delta": partial,
                    "item_id": tc["id"],
                    "output_index": len(current_tool_calls),
                })

        elif ctype == "content_block_start":
            block = chunk.get("content_block", {})
            if block.get("type") == "tool_use":
                tc_id = block.get("id", f"fc_{len(current_tool_calls)}")
                current_tool_calls.append({"id": tc_id, "name": block.get("name", ""), "arguments": ""})
                yield _sse_event("response.output_item.added", {
                    "output_index": len(current_tool_calls),
                    "item": {"id": tc_id, "type": "function_call", "name": block.get("name", ""), "arguments": ""},
                })

        elif ctype == "message_delta":
            usage = chunk.get("usage", {})
            if usage:
                last_usage = usage

    yield _sse_event("response.output_item.done", {
        "output_index": 0,
        "item": {"id": msg_id, "type": "message", "role": "assistant", "content": [{"type": "text", "text": current_text}]},
    })

    yield _sse_event("response.completed", {
        "response": {
            "id": response_id,
            "model": model,
            "created_at": now,
            "output": [
                {"id": msg_id, "type": "message", "role": "assistant", "content": [{"type": "text", "text": current_text}]},
                *[{"id": tc["id"], "type": "function_call", "name": tc["name"], "arguments": tc["arguments"]} for tc in current_tool_calls],
            ],
            "usage": {
                "input_tokens": last_usage.get("input_tokens", 0),
                "output_tokens": last_usage.get("output_tokens", 0),
                "total_tokens": last_usage.get("input_tokens", 0) + last_usage.get("output_tokens", 0),
            },
        }
    })

    yield "data: [DONE]\n\n"


def _sse_event(event_type: str, data: dict) -> str:
    """Build an SSE-formatted string with event type and JSON data payload.

    The ``type`` field is injected into the data payload to match the
    OpenAI Responses API SSE specification, enabling client-side Zod
    schema validation (e.g. in Cherry Studio).
    """
    data = {"type": event_type, **data}
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


def _build_webui() -> str:
    return r"""<!DOCTYPE html>
<html lang="zh-cn">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>response-adapter - Responses API Proxy</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#f5f5f5;color:#333;padding:20px}
.container{max-width:900px;margin:0 auto}
h1{font-size:1.5rem;margin-bottom:4px}
.sub{color:#666;font-size:.85rem;margin-bottom:16px}
.badge{display:inline-block;background:#6366f1;color:#fff;font-size:.75rem;padding:2px 10px;border-radius:10px;margin-left:8px}
.badge-anthropic{background:#d97706}
.layout{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:700px){.layout{grid-template-columns:1fr}}
.card{background:#fff;border-radius:8px;padding:14px;box-shadow:0 1px 3px rgba(0,0,0,.1)}
.card h2{font-size:.95rem;margin-bottom:8px;color:#444}
textarea{width:100%;min-height:220px;font-family:'SF Mono','Fira Code','Cascadia Code',monospace;font-size:.8rem;padding:10px;border:1px solid #d4d4d8;border-radius:6px;resize:vertical;tab-size:2}
textarea:focus{outline:none;border-color:#6366f1;box-shadow:0 0 0 2px rgba(99,102,241,.2)}
.actions{display:flex;gap:8px;margin-top:10px;flex-wrap:wrap}
.btn{padding:8px 18px;border:none;border-radius:6px;font-size:.85rem;cursor:pointer;transition:opacity .2s}
.btn:hover{opacity:.85}
.btn:disabled{opacity:.5;cursor:not-allowed}
.btn-primary{background:#6366f1;color:#fff}
.btn-outline{background:transparent;border:1px solid #d4d4d8;color:#333}
select{padding:8px 12px;border:1px solid #d4d4d8;border-radius:6px;font-size:.85rem;background:#fff;cursor:pointer}
#output{background:#1e1e2e;color:#cdd6f4;min-height:220px;white-space:pre-wrap;font-family:'SF Mono','Fira Code','Cascadia Code',monospace;font-size:.8rem;padding:10px;border-radius:6px;overflow:auto;line-height:1.5}
#status{margin-top:10px;font-size:.82rem;color:#666}
.loading{opacity:.6;pointer-events:none}
.presets{margin-bottom:8px;display:flex;gap:4px;flex-wrap:wrap}
.presets button{font-size:.75rem;padding:3px 10px;border:1px solid #d4d4d8;border-radius:4px;background:#fafafa;cursor:pointer;transition:all .15s}
.presets button:hover{border-color:#6366f1;color:#6366f1}
</style>
</head>
<body>
<div class="container">
  <h1>response-adapter <span id="badge" class="badge">chat</span></h1>
  <p class="sub">输入 OpenAI Responses API 格式的请求，代理到后端并返回 Responses 格式响应</p>

  <div class="layout">
    <div class="card">
      <h2>请求 (Responses API)</h2>
      <div class="presets">
        <button onclick='setPreset("simple")'>简单文本</button>
        <button onclick='setPreset("chat")'>多轮对话</button>
        <button onclick='setPreset("tools")'>工具调用</button>
        <button onclick='setPreset("stream")'>流式</button>
      </div>
      <textarea id="request" placeholder='{"model":"gpt-4o","input":"Hello","instructions":"Be helpful."}'>{&quot;model&quot;:&quot;gpt-4o&quot;,&quot;input&quot;:&quot;Hello&quot;,&quot;instructions&quot;:&quot;Be helpful.&quot;}</textarea>
      <div class="actions">
        <select id="targetSelect" onchange="updateBadge()">
          <option value="chat">OpenAI Chat</option>
          <option value="anthropic">Anthropic</option>
        </select>
        <button class="btn btn-primary" onclick="sendRequest()" id="sendBtn">发送</button>
        <button class="btn btn-outline" onclick="clearOutput()">清空</button>
      </div>
      <div id="status">就绪</div>
    </div>

    <div class="card">
      <h2>响应 (Responses API)</h2>
      <div id="output">等待发送请求...</div>
    </div>
  </div>
</div>

<script>
const presets = {
  simple: JSON.stringify({"model":"gpt-4o","input":"Hello","instructions":"Be helpful."}, null, 2),
  chat: JSON.stringify({"model":"gpt-4o","input":[{"type":"message","role":"user","content":"Who wrote Romance of the Three Kingdoms?"},{"type":"message","role":"assistant","content":"Luo Guanzhong wrote Romance of the Three Kingdoms."},{"type":"message","role":"user","content":"When was he born?"}]}, null, 2),
  tools: JSON.stringify({"model":"gpt-4o","input":"What is the weather in NYC?","tools":[{"type":"function","name":"get_weather","description":"Get weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}],"tool_choice":"auto"}, null, 2),
  stream: JSON.stringify({"model":"gpt-4o","input":"Tell me a short story","stream":true}, null, 2),
};

function setPreset(name) {
  document.getElementById('request').value = presets[name];
}

function getTarget() {
  return document.getElementById('targetSelect').value;
}

function updateBadge() {
  const t = getTarget();
  const badge = document.getElementById('badge');
  badge.textContent = t;
  badge.className = 'badge' + (t === 'anthropic' ? ' badge-anthropic' : '');
}

async function sendRequest() {
  const btn = document.getElementById('sendBtn');
  const status = document.getElementById('status');
  const output = document.getElementById('output');
  const target = getTarget();
  btn.disabled = true;
  document.querySelector('.card:last-child').classList.add('loading');

  try {
    const reqText = document.getElementById('request').value;
    const body = JSON.parse(reqText);
    const isStream = body.stream === true;

    const headers = {'Content-Type': 'application/json', 'X-Target-Backend': target};

    if (isStream) {
      output.textContent = '';
      status.textContent = '流式接收中...';

      const resp = await fetch('/v1/responses', {method: 'POST', headers, body: JSON.stringify(body)});

      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const {done, value} = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, {stream: true});
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data: ') && line !== 'data: [DONE]') {
            try {
              const data = JSON.parse(line.slice(6));
              if (data.type === 'response.output_text.delta') {
                output.textContent += data.delta;
                output.scrollTop = output.scrollHeight;
              } else if (data.type === 'response.completed') {
                output.textContent = JSON.stringify(data.response, null, 2);
              }
            } catch {}
          }
        }
      }
      status.textContent = '流式完成 \u2713';
    } else {
      status.textContent = '发送中...';
      const resp = await fetch('/v1/responses', {method: 'POST', headers, body: JSON.stringify(body)});

      if (!resp.ok) {
        const err = await resp.json();
        output.textContent = JSON.stringify(err, null, 2);
        status.textContent = '请求失败';
        return;
      }

      const data = await resp.json();
      output.textContent = JSON.stringify(data, null, 2);
      status.textContent = '完成 \u2713';
    }
  } catch (e) {
    document.getElementById('output').textContent = '错误: ' + e.message;
    status.textContent = '出错';
  } finally {
    btn.disabled = false;
    document.querySelector('.card:last-child').classList.remove('loading');
  }
}

function clearOutput() {
  document.getElementById('output').textContent = '等待发送请求...';
  document.getElementById('status').textContent = '就绪';
}
</script>
</body>
</html>"""
