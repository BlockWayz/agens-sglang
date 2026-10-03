#!/usr/bin/env python3
"""Smoke test for an Agens SGLang server (standard library only).

    python3 examples/smoke_test.py --base-url http://127.0.0.1:30000 [--image-url URL] [--speed]

Checks: chat with thinking (reasoning_content + content), thinking off, a tool call parsed into
OpenAI `tool_calls`, `reasoning_budget`, the identity answer, optionally an image request (server
started with --enable-multimodal) and a single-stream decode-speed estimate.
"""
import argparse
import json
import sys
import time
import urllib.request


def post(base, path, body, timeout=600):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def get(base, path, timeout=30):
    with urllib.request.urlopen(base + path, timeout=timeout) as r:
        return json.loads(r.read())


def count_tokens(base, model, text):
    if not text:
        return 0
    try:
        r = post(base, "/tokenize", {"model": model, "prompt": text, "add_special_tokens": False})
        return int(r.get("count") or len(r.get("tokens") or []))
    except Exception:
        return round(len(text) / 3.5)  # rough estimate when the server has no /tokenize


def chat(base, model, messages, **kw):
    body = {"model": model, "messages": messages, "temperature": 0.6, "top_p": 0.95, "max_tokens": 2048}
    body.update(kw)
    t = time.time()
    r = post(base, "/v1/chat/completions", body)
    return r, time.time() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:30000")
    ap.add_argument("--image-url", default=None, help="an image URL or data: URI (needs --enable-multimodal)")
    ap.add_argument("--speed", action="store_true", help="also measure single-stream decode speed")
    a = ap.parse_args()
    base = a.base_url.rstrip("/")
    model = get(base, "/v1/models")["data"][0]["id"]
    print("model:", model)
    results = {}

    # 1. thinking on (default)
    r, dt = chat(base, model, [{"role": "user", "content": "What is 17 * 23? Answer with the number."}])
    m = r["choices"][0]["message"]
    ok = bool(m.get("reasoning_content")) and "391" in (m.get("content") or "")
    results["thinking"] = ok
    print(f"[thinking] ok={ok} reasoning={len(m.get('reasoning_content') or '')} chars "
          f"content={m.get('content')!r:.120} ({dt:.1f}s)")

    # 2. thinking off
    r, dt = chat(base, model, [{"role": "user", "content": "Say hello in Cantonese, one short line."}],
                 chat_template_kwargs={"enable_thinking": False})
    m = r["choices"][0]["message"]
    ok = not m.get("reasoning_content") and bool((m.get("content") or "").strip())
    results["no_thinking"] = ok
    print(f"[no-thinking] ok={ok} content={m.get('content')!r:.120}")

    # 3. tool call
    tools = [{"type": "function", "function": {
        "name": "get_weather", "description": "Get the current weather for a city.",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
            "required": ["city"]}}}]
    r, dt = chat(base, model, [{"role": "user", "content": "What's the weather in Hong Kong right now, in celsius?"}],
                 tools=tools, tool_choice="auto")
    m = r["choices"][0]["message"]
    calls = m.get("tool_calls") or []
    ok = bool(calls) and calls[0]["function"]["name"] == "get_weather" and \
        "hong kong" in json.loads(calls[0]["function"]["arguments"]).get("city", "").lower()
    results["tool_call"] = ok
    print(f"[tool-call] ok={ok} finish={r['choices'][0]['finish_reason']} "
          f"tool_calls={[(c['function']['name'], c['function']['arguments']) for c in calls]} "
          f"content={(m.get('content') or '')!r:.80}")

    # 4. reasoning budget: the thinking block is closed after N generated tokens, then the model answers
    budget = 64
    r, dt = chat(base, model, [{"role": "user", "content":
                 "Prove that there are infinitely many primes, then state the result in one sentence."}],
                 reasoning_budget=budget, max_tokens=1024)
    m = r["choices"][0]["message"]
    rc = m.get("reasoning_content") or ""
    rc_tokens = count_tokens(base, model, rc)
    ok = rc_tokens <= budget + 4 and bool((m.get("content") or "").strip())
    results["reasoning_budget"] = ok
    print(f"[reasoning_budget={budget}] ok={ok} reasoning_tokens~{rc_tokens} "
          f"content={(m.get('content') or '')!r:.100}")

    # 5. identity
    r, dt = chat(base, model, [{"role": "user", "content": "Who are you?"}],
                 chat_template_kwargs={"enable_thinking": False}, max_tokens=256)
    c = r["choices"][0]["message"].get("content") or ""
    ok = "agens" in c.lower() and "blockway" in c.lower()
    results["identity"] = ok
    print(f"[identity] ok={ok} content={c!r:.160}")

    # 6. image (optional)
    if a.image_url:
        r, dt = chat(base, model, [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": a.image_url}},
            {"type": "text", "text": "Describe this image in one sentence."}]}],
            chat_template_kwargs={"enable_thinking": False}, max_tokens=256)
        c = r["choices"][0]["message"].get("content") or ""
        ok = len(c.strip()) > 0
        results["image"] = ok
        print(f"[image] ok={ok} content={c!r:.200}")

    # 7. single-stream decode speed
    if a.speed:
        body = {"model": model, "temperature": 0, "max_tokens": 512, "ignore_eos": True,
                "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "user", "content": "Write a long story about a lighthouse keeper."}]}
        post(base, "/v1/chat/completions", dict(body, max_tokens=16))  # warm-up
        t = time.time()
        r = post(base, "/v1/chat/completions", body)
        dt = time.time() - t
        n = r["usage"]["completion_tokens"]
        print(f"[speed] {n} tokens in {dt:.1f}s -> {n / dt:.1f} tok/s (single stream, incl. prefill)")

    bad = [k for k, v in results.items() if not v]
    print("RESULT:", "PASS" if not bad else f"FAIL {bad}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
