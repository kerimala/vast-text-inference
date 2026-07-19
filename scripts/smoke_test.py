#!/usr/bin/env python3
"""Smoke-test a tunneled OpenAI-compatible vLLM endpoint."""

from __future__ import annotations

import argparse
import json
import urllib.request


def request_json(url: str, payload: dict | None = None) -> dict:
    body = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.load(response)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="aeon-qwen3.6-27b-bf16")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    models = request_json(f"{base}/v1/models")
    model_ids = [item["id"] for item in models.get("data", [])]
    if args.model not in model_ids:
        raise RuntimeError(f"Expected model {args.model!r}; endpoint returned {model_ids!r}")

    chat = request_json(
        f"{base}/v1/chat/completions",
        {
            "model": args.model,
            "messages": [{"role": "user", "content": "Antworte nur mit dem Wort bereit."}],
            "temperature": 0,
            "max_tokens": 512,
        },
    )
    content = chat["choices"][0]["message"].get("content") or ""
    if not content.strip():
        raise RuntimeError("Basic chat completion returned no text.")

    tool = request_json(
        f"{base}/v1/chat/completions",
        {
            "model": args.model,
            "messages": [{"role": "user", "content": "Wie ist das Wetter in Aachen? Nutze das Werkzeug."}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Return current weather for a city",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                }
            ],
            "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
            "temperature": 0,
            "max_tokens": 1024,
        },
    )
    tool_calls = tool["choices"][0]["message"].get("tool_calls") or []
    if not tool_calls or tool_calls[0].get("function", {}).get("name") != "get_weather":
        raise RuntimeError(f"Forced tool-call parsing failed: {tool_calls!r}")

    print(json.dumps({"models": model_ids, "chat": content, "tool_call": tool_calls[0]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
