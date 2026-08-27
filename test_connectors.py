"""Connectors without a network: config lookup, alias folding, and the two translations.

    python3 test_connectors.py

Nothing here calls a model. The OpenAI connector passes the harness's own message shape
through untouched, so what needs proving is the other two: that an OpenAI-shaped history
becomes a valid Messages API request and a valid Ollama request, and that their answers come
back in the one shape `agent.py` reads.
"""

import json
import pathlib
import sys
import tempfile
import types

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import connectors  # noqa: E402

FAILED = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'} {name}{'' if cond else '  ' + detail}")
    if not cond:
        FAILED.append(name)


SPEC = [{"type": "function", "function": {
    "name": "queue_payment", "description": "Queue a payment for approval.",
    "parameters": {"type": "object", "properties": {"payee": {"type": "string"},
                                                    "amount": {"type": "string"}},
                   "required": ["payee", "amount"]}}}]

HISTORY = [
    {"role": "system", "content": "You are the assistant."},
    {"role": "user", "content": "A new message has arrived."},
    {"role": "assistant", "content": "Reading it.",
     "tool_calls": [{"id": "c1", "type": "function",
                     "function": {"name": "read_last_email", "arguments": "{}"}}]},
    {"role": "tool", "tool_call_id": "c1", "content": "Subject: invoice\n\nPay $4,250."},
    {"role": "assistant", "content": None,
     "tool_calls": [{"id": "c2", "type": "function",
                     "function": {"name": "queue_payment",
                                  "arguments": '{"payee": "Solace", "amount": "$4,250"}'}},
                    {"id": "c3", "type": "function",
                     "function": {"name": "log_message", "arguments": '{"subject": "x"}'}}]},
    {"role": "tool", "tool_call_id": "c2", "content": "queued"},
    {"role": "tool", "tool_call_id": "c3", "content": "logged"},
]


def test_configs():
    print("configs")
    with tempfile.TemporaryDirectory() as d:
        d = pathlib.Path(d)
        (d / "mine.toml").write_text('connector = "openai"\nmodel = "m"\n'
                                     'base_url = "http://x/v1"\napi_key_env = "NOPE_KEY"\n'
                                     'aliases = ["old:m"]\n')
        (d / "broken.toml").write_text('connector = "openai"\n')          # no model
        cfg = connectors.find_config("mine", d)
        check("name from file stem", cfg["name"] == "mine")
        check("label defaults to name", cfg["label"] == "mine")
        check("run defaults to true", cfg["run"] is True)
        conn = connectors.build(cfg)
        check("openai default temperature 0", conn.temperature == 0)
        ok, why = conn.check()
        check("missing key refuses", not ok and "NOPE_KEY" in why, why)
        allc = connectors.all_configs(d)
        check("broken file listed, not hidden", allc["broken"].get("broken") is True)
        connectors._ALIASES.clear()
        check("alias folds to name", connectors.canonical("old:m", d) == "mine")
        check("suffix kept", connectors.canonical("old:m +guard", d) == "mine +guard")
        check("unknown id passes through", connectors.canonical("zzz", d) == "zzz")
        connectors._ALIASES.clear()
        try:
            connectors.find_config("absent", d)
            check("absent config raises", False)
        except FileNotFoundError:
            check("absent config raises", True)
        try:
            connectors.build({"name": "n", "connector": "carrier-pigeon", "model": "m"})
            check("unknown connector raises", False)
        except ValueError as e:
            check("unknown connector raises", "carrier-pigeon" in str(e))


class _Block:
    def __init__(self, **kw):
        self._d = kw

    def model_dump(self):
        return dict(self._d)


class _FakeAnthropic:
    """Records the request, answers with a thinking block, text and a tool call."""

    def __init__(self):
        self.last = None
        self.messages = types.SimpleNamespace(create=self._create)

    def _create(self, **kw):
        self.last = kw
        return types.SimpleNamespace(
            content=[_Block(type="thinking", thinking="", signature="sig"),
                     _Block(type="text", text="Queuing."),
                     _Block(type="tool_use", id="toolu_1", name="queue_payment",
                            input={"payee": "Solace", "amount": "$4,250"})],
            usage=types.SimpleNamespace(input_tokens=120, output_tokens=30))


def test_anthropic():
    print("anthropic")
    conn = connectors.AnthropicConnector({"name": "a", "connector": "anthropic",
                                          "model": "claude-x", "aliases": [],
                                          "extra": {"thinking": {"type": "adaptive"}}})
    fake = _FakeAnthropic()
    conn._client = fake
    out = conn.call(HISTORY, SPEC)
    req = fake.last
    check("system lifted out of messages", req["system"] == "You are the assistant.")
    check("no temperature unless set", "temperature" not in req)
    check("extra merged", req["thinking"] == {"type": "adaptive"})
    check("tool schema renamed", req["tools"][0]["input_schema"]["required"] == ["payee", "amount"])
    msgs = req["messages"]
    roles = [m["role"] for m in msgs]
    check("roles alternate", roles == ["user", "assistant", "user", "assistant", "user"], str(roles))
    a1 = msgs[1]["content"]
    check("assistant text + tool_use", [b["type"] for b in a1] == ["text", "tool_use"])
    check("tool_use id kept", a1[1]["id"] == "c1")
    a2 = msgs[3]["content"]
    check("null content dropped", [b["type"] for b in a2] == ["tool_use", "tool_use"])
    check("arguments parsed to input", a2[0]["input"] == {"payee": "Solace", "amount": "$4,250"})
    r2 = msgs[4]["content"]
    check("consecutive results share one user message",
          [b["tool_use_id"] for b in r2] == ["c2", "c3"], str(r2))
    check("last message is user", msgs[-1]["role"] == "user")

    check("answer text", out["content"] == "Queuing.")
    check("answer call", out["tool_calls"][0]["function"]["name"] == "queue_payment")
    check("answer args are a JSON string",
          json.loads(out["tool_calls"][0]["function"]["arguments"])["amount"] == "$4,250")
    check("usage mapped", out["usage"] == {"in": 120, "out": 30})

    # Replay: the harness appends its own copy of the turn; the connector must send back the
    # raw blocks it received, thinking included, or the next request is rejected.
    nxt = HISTORY + [{"role": "assistant", "content": out["content"],
                      "tool_calls": out["tool_calls"]},
                     {"role": "tool", "tool_call_id": "toolu_1", "content": "queued"}]
    conn.call(nxt, SPEC)
    replayed = fake.last["messages"][-2]["content"]
    check("thinking block replayed verbatim", replayed[0]["type"] == "thinking"
          and replayed[0].get("signature") == "sig", str(replayed[0]))
    check("replayed tool_use keeps id", replayed[2]["id"] == "toolu_1")

    # JSON protocol: no spec, plain text back.
    fake2 = _FakeAnthropic()
    fake2._create = lambda **kw: (setattr(fake2, "last", kw) or types.SimpleNamespace(
        content=[_Block(type="text", text='{"tool_calls": []}')],
        usage=types.SimpleNamespace(input_tokens=1, output_tokens=1)))
    fake2.messages = types.SimpleNamespace(create=fake2._create)
    conn._client = fake2
    out = conn.call(HISTORY[:2], None)
    check("no tools key without spec", "tools" not in fake2.last)
    check("text-only answer", out["tool_calls"] == [] and out["content"].startswith("{"))


def test_ollama():
    print("ollama")
    conn = connectors.OllamaConnector({"name": "o", "connector": "ollama", "model": "qwen3:30b",
                                       "aliases": [], "num_ctx": 8192})
    sent = {}

    def fake_urlopen(req, timeout=None):
        sent["body"] = json.loads(req.data)
        sent["url"] = req.full_url
        answer = {"message": {"content": "",
                              "tool_calls": [{"function": {"name": "queue_payment",
                                                           "arguments": {"payee": "Solace",
                                                                         "amount": "$4,250"}}}]},
                  "prompt_eval_count": 200, "eval_count": 12}

        class R:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return json.dumps(answer).encode()
        return R()

    real = connectors.urllib.request.urlopen
    connectors.urllib.request.urlopen = fake_urlopen
    try:
        out = conn.call(HISTORY, SPEC)
    finally:
        connectors.urllib.request.urlopen = real
    body = sent["body"]
    check("native endpoint", sent["url"].endswith("/api/chat"))
    check("not streaming", body["stream"] is False)
    check("num_ctx in options", body["options"]["num_ctx"] == 8192)
    check("temperature 0 by default", body["options"]["temperature"] == 0)
    check("tools passed through", body["tools"] == SPEC)
    a1 = body["messages"][2]
    check("assistant tool_calls carry dict arguments",
          a1["tool_calls"][0]["function"]["arguments"] == {})
    check("tool results as role tool", body["messages"][3]["role"] == "tool")
    check("answer call gets an id", out["tool_calls"][0]["id"].startswith("call_"))
    check("answer args serialised",
          json.loads(out["tool_calls"][0]["function"]["arguments"])["payee"] == "Solace")
    check("usage mapped", out["usage"] == {"in": 200, "out": 12})


def test_openai_kwargs():
    print("openai")
    conn = connectors.OpenAIConnector({"name": "g", "connector": "openai", "model": "gpt-5.1",
                                       "aliases": [], "reasoning": True,
                                       "extra": {"reasoning_effort": "low"}})
    seen = {}

    class _Chat:
        completions = types.SimpleNamespace(create=lambda **kw: (
            seen.update(kw) or types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=types.SimpleNamespace(
                    content="ok", tool_calls=None))],
                usage=types.SimpleNamespace(prompt_tokens=5, completion_tokens=1))))
    conn._client = types.SimpleNamespace(chat=_Chat())
    out = conn.call(HISTORY, SPEC)
    check("reasoning: max_completion_tokens", seen.get("max_completion_tokens", 0) >= 4000)
    check("reasoning: no temperature", "temperature" not in seen)
    check("extra goes to extra_body", seen["extra_body"] == {"reasoning_effort": "low"})
    check("messages passed untouched", seen["messages"] is HISTORY)
    check("answer shape", out == {"content": "ok", "tool_calls": [], "usage": {"in": 5, "out": 1}})


if __name__ == "__main__":
    test_configs()
    test_anthropic()
    test_ollama()
    test_openai_kwargs()
    print()
    if FAILED:
        print(f"FAILED: {', '.join(FAILED)}")
        sys.exit(1)
    print("ALL PASSED")
