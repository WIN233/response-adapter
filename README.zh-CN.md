# response-adapter

[English](README.md)

将 OpenAI Responses API 格式的请求透明转换成 Chat Completions 或 Anthropic Messages 格式，并代理到对应后端。

## 安装

```bash
pip install -e .
```

## 配置（代理服务器模式）

复制环境变量模板并填入 API Key：

```bash
cp .env.example .env
```

```ini
# .env
TARGET_BACKEND=chat              # 默认: chat 或 anthropic
OPENAI_API_KEY=sk-...            # Chat 后端 API key
OPENAI_BASE_URL=https://api.deepseek.com/v1  # 任意 OpenAI 兼容端点
ANTHROPIC_API_KEY=sk-ant-...     # Anthropic 后端 API key
ANTHROPIC_BASE_URL=https://api.anthropic.com/v1
HOST=0.0.0.0
PORT=8008
```

两个 API Key 可以同时配置，运行时通过 `X-Target-Backend` 请求头切换后端。

## 启动服务器

```bash
python main.py
```

打开 `http://localhost:8008` 访问 WebUI（可通过下拉菜单切换后端）。

### curl 示例

```bash
# 默认后端（来自 .env 的 TARGET_BACKEND）
curl -X POST 'http://localhost:8008/v1/responses' \
  -H 'Content-Type: application/json' \
  -d '{"model":"gpt-4o","input":"Hello"}'

# 通过请求头指定后端
curl -X POST 'http://localhost:8008/v1/responses' \
  -H 'Content-Type: application/json' \
  -H 'X-Target-Backend: anthropic' \
  -d '{"model":"claude-opus-4-8","input":"Hello","max_output_tokens":100}'

# 流式输出
curl -N -X POST 'http://localhost:8008/v1/responses' \
  -H 'Content-Type: application/json' \
  -H 'X-Target-Backend: chat' \
  -d '{"model":"gpt-4o","input":"Tell me a story","stream":true}'
```

## 作为库使用

```python
from src.models.responses import ResponsesRequest
from src.converters.responses_to_chat_request import responses_to_chat_request
from src.converters.chat_response_to_responses import chat_response_to_responses
from src.models.chat import ChatResponse

# Responses → Chat 请求
resp_req = ResponsesRequest(model="gpt-4o", input="Hello", instructions="Be helpful.")
chat_req = responses_to_chat_request(resp_req)
# chat_req.model_dump() → POST /v1/chat/completions

# Chat 响应 → Responses 响应
chat_resp = ChatResponse(**api_response)
responses_resp = chat_response_to_responses(chat_resp)
```

```python
from src.converters.responses_to_anthropic_request import responses_to_anthropic_request
from src.converters.anthropic_response_to_responses import anthropic_response_to_responses
from src.models.anthropic import AnthropicResponse

# Responses → Anthropic 请求
ant_req = responses_to_anthropic_request(resp_req)
# ant_req.model_dump() → POST /v1/messages

# Anthropic 响应 → Responses 响应
ant_resp = AnthropicResponse(**api_response)
responses_resp = anthropic_response_to_responses(ant_resp)
```

## API

| 路由 | 方法 | 说明 |
|------|------|------|
| `/` | GET | WebUI 交互界面 |
| `/v1/responses` | POST | 代理到 `.env` 配置的后端 |
| `/health` | GET | 健康检查 |

## 参数映射

| Responses 参数 | Chat 映射 | Anthropic 映射 |
|---|---|---|
| `input` | `messages[].content` | `messages[].content` |
| `instructions` | system message | `system` |
| `max_output_tokens` | `max_tokens` | `max_tokens` |
| `temperature` / `top_p` | 同左 | 同左 |
| `tools` (function) | `tools` (function) | `tools` |
| `tool_choice` | `tool_choice` | `tool_choice` |
| `text.format` | `response_format` | `output_config` |
| `stream` | SSE 流式 | SSE 流式 |

## 项目结构

```
src/
  models/              # Pydantic 数据模型
    chat.py            # Chat Completions
    anthropic.py       # Anthropic Messages
    responses.py       # OpenAI Responses API
  converters/          # 格式转换器
    responses_to_chat_request.py
    chat_response_to_responses.py
    responses_to_anthropic_request.py
    anthropic_response_to_responses.py
  proxy/
    client.py          # HTTP 客户端
    server.py          # FastAPI 服务器 + WebUI + SSE 流式
main.py                # 入口
.env.example           # 配置模板
```
