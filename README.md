# Cortex Demo

A self-contained **plain-chat demonstration gateway** for the public Hugging Face model [`mkd-hossain/Keural-Cortex-8B-v2`](https://huggingface.co/mkd-hossain/Keural-Cortex-8B-v2).

It starts vLLM locally and places a small adapter in front of it. The adapter removes Keural agent prompts and tool definitions, so the model receives only text-chat history. This exists for a reliable executive demo while tool-routing compatibility is evaluated separately.

## What this demo does

- Korean/English plain chat.
- OpenAI-compatible `/v1/chat/completions` and `/v1/models` endpoints.
- JSON responses and buffered SSE compatibility for clients that request streaming.
- No model weights, data, API keys, cloud credentials, or fixed tunnel URL are included.

## What this demo does not do

- It does **not** support `ask_clarification`, web search, URL fetching, document retrieval, files, or other tool calls.
- It disables thinking for predictable demonstrations.
- Buffered SSE sends the answer after generation completes; it is not token-by-token streaming.
- A Cloudflare Quick Tunnel is temporary and unauthenticated. Do not use it for confidential data or production.

## Run locally

Create a Python environment that includes vLLM, FastAPI, httpx, and uvicorn. Then set `MODEL_PATH` to an existing local export, for example:

```bash
git clone git@github.com:mkd-hossain/cortex-demo.git
cd cortex-demo
export MODEL_PATH=/absolute/path/to/Keural-Cortex-8B-v2
export VLLM_BIN=/absolute/path/to/vllm
export PYTHON_BIN=/absolute/path/to/python
bash run-demo.sh
```

The local gateway URL is:

```text
http://127.0.0.1:8081/v1
```

## Run a temporary public demo

With `cloudflared` installed:

```bash
PUBLIC_TUNNEL=1 bash run-demo.sh
```

Cloudflare prints a temporary HTTPS URL. Append `/v1` and use that as the Keural gateway Base URL. The URL changes after the tunnel restarts.

For this demo gateway use:

| Gateway field | Value |
|---|---|
| Model ID | `Keural-cortex-v2` |
| Supplier connection | No connection / no API key (demo only) |
| Base URL | The temporary tunnel URL followed by `/v1` |

## Test

```bash
curl http://127.0.0.1:8081/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Keural-cortex-v2","messages":[{"role":"user","content":"Explain briefly why the sky is blue."}],"stream":false}'
```

Press `Ctrl+C` in the launcher terminal when the demo ends. It stops vLLM, the adapter, and the optional tunnel.

## Safety and status

This repository is a demo wrapper, not a claim that the underlying checkpoint is production-ready. The model card documents known agent/tool-use limitations. Keep tool execution disabled and stop public tunnels after demonstrations.
