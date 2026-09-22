"""Text-only adapter for a short Keural Cortex demonstration.

This program deliberately does not expose model-management endpoints, tools, or
caller-supplied system prompts.  It is a plain-chat demonstration layer, not a
general-purpose agent gateway.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json
import os
import time
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

UPSTREAM = os.getenv("VLLM_UPSTREAM", "http://127.0.0.1:8000").rstrip("/")
MODEL = os.getenv("SERVED_MODEL_NAME", "Keural-cortex-v2")
MAX_BODY_BYTES = 512 * 1024
MAX_OUTPUT_TOKENS = 1024
SYSTEM_PROMPT = (
    "You are Keural, a helpful Korean-English text assistant. Answer the latest user "
    "request directly and match the user's language. Respond to greetings with a short "
    "greeting. You have no tools, web access, private files, or ability to perform "
    "external actions. Do not claim to have used them. Do not emit tool calls, tool "
    "markup, or interactive selection forms. Draft requested text directly, using a "
    "placeholder only when an essential detail is unavailable."
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.slots = asyncio.Semaphore(2)
    async with httpx.AsyncClient(timeout=httpx.Timeout(90, connect=5), trust_env=False) as client:
        app.state.client = client
        yield


app = FastAPI(title="Cortex demo adapter", docs_url=None, redoc_url=None,
              openapi_url=None, lifespan=lifespan)


def api_error(message: str, status: int = 400) -> JSONResponse:
    return JSONResponse({"error": {"message": message, "type": "demo_adapter_error"}}, status_code=status)


def build_request(payload: dict[str, Any]) -> dict[str, Any]:
    """Allow only text chat history. Agent instructions and tools are discarded."""
    if payload.get("model") != MODEL:
        raise ValueError(f"Use model ID {MODEL}.")
    if not isinstance(payload.get("stream", False), bool):
        raise ValueError("stream must be a boolean.")
    source = payload.get("messages")
    if not isinstance(source, list) or not source:
        raise ValueError("messages must be a nonempty list.")
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for row in source:
        if not isinstance(row, dict):
            raise ValueError("Every message must be an object.")
        role = row.get("role")
        if role in ("system", "developer"):
            continue
        if role in ("tool", "function") or row.get("tool_calls") or row.get("function_call"):
            raise ValueError("Tool history is not supported. Start a new plain-chat conversation.")
        if role not in ("user", "assistant") or not isinstance(row.get("content"), str):
            raise ValueError("Only user and assistant text messages are supported.")
        messages.append({"role": role, "content": row["content"]})
    if len(messages) < 2 or messages[-1]["role"] != "user":
        raise ValueError("A new user message is required.")
    if len(messages) > 65:
        raise ValueError("This demo accepts at most 64 conversation messages. Start a new chat.")
    requested = payload.get("max_completion_tokens", payload.get("max_tokens", MAX_OUTPUT_TOKENS))
    if isinstance(requested, bool) or not isinstance(requested, int) or requested < 1:
        raise ValueError("max_tokens must be a positive integer.")
    return {
        "model": MODEL, "messages": messages, "stream": False, "temperature": 0,
        "top_p": 1, "max_tokens": min(requested, MAX_OUTPUT_TOKENS),
        "chat_template_kwargs": {"enable_thinking": False},
    }


def sse(completion: dict[str, Any], include_usage: bool) -> bytes:
    """Return a completed answer as valid, buffered OpenAI SSE frames."""
    base = {key: completion[key] for key in ("id", "created", "model")}
    base["object"] = "chat.completion.chunk"
    answer = completion["choices"][0]
    chunks: list[dict[str, Any]] = [
        {**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]},
        {**base, "choices": [{"index": 0, "delta": {"content": answer["message"]["content"]}, "finish_reason": None}]},
        {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": answer["finish_reason"]}]},
    ]
    if include_usage:
        chunks.append({**base, "choices": [], "usage": completion.get("usage")})
    return ("".join("data: " + json.dumps(x, ensure_ascii=False) + "\n\n" for x in chunks) + "data: [DONE]\n\n").encode()


@app.get("/health")
async def health(request: Request):
    try:
        response = await request.app.state.client.get(f"{UPSTREAM}/health", timeout=5)
    except httpx.HTTPError:
        return api_error("vLLM is unavailable.", 503)
    return JSONResponse({"status": "ok", "mode": "plain_chat_demo", "streaming": "buffered"},
                        status_code=200 if response.status_code == 200 else 503)


@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [{"id": MODEL, "object": "model", "created": int(time.time()),
                                             "owned_by": "cortex-demo"}]}


@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = await request.body()
    if len(body) > MAX_BODY_BYTES:
        return api_error("Request exceeds 512 KiB.", 413)
    try:
        incoming = json.loads(body)
        if not isinstance(incoming, dict):
            raise ValueError("Request must be a JSON object.")
        clean = build_request(incoming)
    except (ValueError, json.JSONDecodeError) as exc:
        return api_error(str(exc))
    if request.app.state.slots.locked():
        return api_error("Demo is busy. Try again shortly.", 429)
    async with request.app.state.slots:
        try:
            upstream = await request.app.state.client.post(f"{UPSTREAM}/v1/chat/completions", json=clean)
        except httpx.TimeoutException:
            return api_error("Generation exceeded the 90-second demo timeout.", 504)
        except httpx.HTTPError:
            return api_error("vLLM is unavailable.", 502)
    if upstream.status_code != 200:
        return api_error("vLLM rejected the request.", 502)
    try:
        raw = upstream.json()
        choice = raw["choices"][0]
        content = choice["message"].get("content")
        if not isinstance(content, str) or not content.strip() or choice["message"].get("tool_calls"):
            raise ValueError
        completion = {"id": raw["id"], "object": "chat.completion", "created": raw["created"],
                      "model": MODEL, "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                      "finish_reason": choice["finish_reason"]}], "usage": raw.get("usage")}
    except (KeyError, IndexError, TypeError, ValueError):
        return api_error("The model did not return a valid plain-text answer.", 502)
    headers = {"Cache-Control": "no-store", "X-Cortex-Demo": "plain-chat"}
    if incoming.get("stream"):
        use = isinstance(incoming.get("stream_options"), dict) and incoming["stream_options"].get("include_usage") is True
        return Response(sse(completion, use), media_type="text/event-stream", headers=headers)
    return JSONResponse(completion, headers=headers)
