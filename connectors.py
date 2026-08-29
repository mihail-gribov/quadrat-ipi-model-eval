"""How a model is reached: one class per wire format, one config file per model.

The bench needs exactly one thing from a model: given a message list and a tool spec, return
what it said and which tools it called. Everything about HOW that happens -- the endpoint, the
key, the wire format, the token cap, the sampling -- is a property of the connection, not of
the harness, so it lives in a config file and never in code. A row then differs from another
row by the config and by nothing else, and a reader can see the whole difference in one file.

    models/gpt-4o-mini.toml
        connector = "openai"                 # which class below
        model = "gpt-4o-mini"                # the id the endpoint knows
        base_url = "https://api.openai.com/v1"
        api_key_env = "OPENAI_API_KEY"
        label = "gpt-4o-mini"                # the name tables use; defaults to the file name
        run = true                           # money_all.sh sweeps the configs with run = true

A run names the config, not the model: `MODEL=gpt-4o-mini ./money.sh` reads
`models/gpt-4o-mini.toml`. Episodes record the config name as `model`, plus `model_id` and
`connector`, so a log always says what was called and how.

Three connectors, three wire formats:

    openai     -- OpenAI chat completions with tools. Also every OpenAI-compatible endpoint:
                  Nebius, Mistral, Google's compatibility surface, aggregators, and a vLLM /
                  TGI / llama.cpp / Ollama server on your own machine (`base_url`).
    anthropic  -- the Messages API, through the `anthropic` SDK. Tool calls and results are
                  translated to and from content blocks; thinking blocks are replayed verbatim.
    ollama     -- Ollama's native `/api/chat`. Its OpenAI shim works too (`openai` with
                  base_url http://localhost:11434/v1), but the native route lets the config set
                  `num_ctx`, which the shim silently leaves at the model's small default.

Every connector returns the same shape from `call(messages, spec)`:

    {"content": str, "tool_calls": [{"id", "type": "function", "function": {"name",
     "arguments": <json str>}}], "usage": {"in": int, "out": int}}

and takes the same OpenAI-shaped message list from `agent.py`. That contract is the only thing
the rest of the harness knows about a model.

Config keys common to every connector:

    connector, model, label, run, note, aliases (old ids of the same model in older logs),
    order (sweep position for money_all.sh; lower runs first, default 50),
    api_key_env (name of the env var / .env line holding the key), base_url, or
    base_url_env when the URL itself is private,
    max_tokens, temperature, timeout (seconds), extra (a table merged into every request body
    as-is: reasoning effort, thinking config, whatever the endpoint takes).

Temperature: `openai` and `ollama` send 0 unless the config says otherwise; `anthropic` sends
none. `temperature = "none"` makes any connector leave it out -- the newest models reject the
parameter.

Configs are read from `models/` beside this file. `QUADRAT_MODELS` names another directory,
searched first, so a project that imports this module keeps its own configs -- the ones with
private routes -- outside the public list; `QUADRAT_ENV` points it at its own `.env`.

    python3 connectors.py list            # every config: name, connector, model id, run, key
    python3 connectors.py check NAME...   # exit 1 if any key is missing; no model is called
"""

import collections
import json
import os
import pathlib
import re
import sys
import threading
import tomllib
import urllib.error
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_DIR = HERE / "models"
ENV_FILE = HERE / ".env"
STOP = threading.Event()        # set by the runner on Ctrl-C; a backoff in progress then gives up

# Throttling and provider hiccups are waited out here rather than surfaced: a 429 is not a
# measurement of the model. The client libraries' own two retries span seconds, which is
# nothing against a per-minute cap hit from eight workers -- one free tier turned nearly a
# whole sweep into 429 receipts before this existed. After the last wait the call is made once
# more and its error, if any, is the episode's error.
WAITS = (2, 4, 8, 16, 32, 60)


def env_files():
    """`.env` candidates in order: `QUADRAT_ENV` if set, the working directory, this
    module's directory. A project that imports this module points the variable at its own."""
    out = []
    if os.environ.get("QUADRAT_ENV"):
        out.append(pathlib.Path(os.environ["QUADRAT_ENV"]))
    out += [pathlib.Path.cwd() / ".env", ENV_FILE]
    return out


def env_val(key):
    """A key: the environment first, then `.env` in the working directory, then `.env` beside
    this module. Keys never sit in a model config -- a config names the variable, `.env`
    holds the value, and only `.env` is git-ignored."""
    v = os.environ.get(key)
    if v:
        return v
    for env_file in env_files():
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                m = _ENV_LINE.match(line)
                if m and m.group(1) == key:
                    return _unquote(m.group(2))
    return None


_ENV_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$")


def _unquote(v):
    """The right-hand side of a `.env` line: quotes and a trailing comment stripped."""
    v = v.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
        return v[1:-1]
    return v.split(" #", 1)[0].strip()


# ----------------------------------------------------------------------------- config files

def model_dirs(extra=None):
    """Where configs are looked up, first match wins."""
    dirs = []
    if extra:
        dirs.append(pathlib.Path(extra))
    env = os.environ.get("QUADRAT_MODELS")
    if env:
        dirs.append(pathlib.Path(env))
    dirs.append(DEFAULT_DIR)
    return [d for d in dirs if d.is_dir()]


def read_config(path):
    path = pathlib.Path(path)
    with path.open("rb") as f:
        cfg = tomllib.load(f)
    cfg.setdefault("name", path.stem)
    cfg.setdefault("label", cfg["name"])
    cfg.setdefault("run", True)
    cfg.setdefault("aliases", [])
    cfg.setdefault("order", 50)
    if "connector" not in cfg or "model" not in cfg:
        raise ValueError(f"{path}: a config needs `connector` and `model`")
    if cfg["connector"] not in CONNECTORS:
        raise ValueError(f"{path}: unknown connector {cfg['connector']!r}; "
                         f"known: {', '.join(CONNECTORS)}")
    cfg["path"] = str(path)
    return cfg


def find_config(name, models_dir=None):
    """`name` is a config name (`gpt-4o-mini`) or a path to a .toml file."""
    p = pathlib.Path(name)
    if p.suffix == ".toml" and p.exists():
        return read_config(p)
    for d in model_dirs(models_dir):
        cand = d / f"{name}.toml"
        if cand.exists():
            return read_config(cand)
    raise FileNotFoundError(
        f"no model config named {name!r} in " + ", ".join(str(d) for d in model_dirs(models_dir)))


def all_configs(models_dir=None):
    """Every config, by name, in sweep order (`order`, then name); a directory earlier in the
    search order shadows a later one."""
    out = {}
    for d in model_dirs(models_dir):
        for p in sorted(d.glob("*.toml")):
            if p.stem not in out:
                try:
                    out[p.stem] = read_config(p)
                except Exception as e:                      # a broken file is reported, not hidden
                    out[p.stem] = {"name": p.stem, "label": p.stem, "run": False,
                                   "connector": "?", "model": "?", "aliases": [], "order": 50,
                                   "note": f"unreadable: {e}", "path": str(p), "broken": True}
    return dict(sorted(out.items(), key=lambda kv: (kv[1]["order"], kv[0].lower())))


def canonical(model_field, models_dir=None):
    """Map what a log line calls its model to a config name.

    Logs written before configs existed carry routing ids (`oai:gpt-4o-mini`); a config lists
    those under `aliases`, and the tables then read old and new sweeps of one model as one
    model. Suffixes the scorer appends (` +guard`) are kept.
    """
    base, sep, suffix = model_field.partition(" +")
    name = _alias_map(models_dir).get(base, base)
    return name + (sep + suffix if sep else "")


_ALIASES = {}


def forget_configs():
    """Drop the cached alias map (tests that write their own config directory)."""
    _ALIASES.clear()


def _alias_map(models_dir=None):
    """alias or name -> name, read once per directory set: `canonical` runs per log line."""
    key = (str(models_dir or ""), os.environ.get("QUADRAT_MODELS", ""))
    if key not in _ALIASES:
        m = {}
        for cfg in all_configs(models_dir).values():
            m[cfg["name"]] = cfg["name"]
            for a in cfg["aliases"]:
                m.setdefault(a, cfg["name"])
        _ALIASES[key] = m
    return _ALIASES[key]


def labels(models_dir=None):
    """config name -> label, plus every alias -> label, for the tables."""
    out = {}
    for cfg in all_configs(models_dir).values():
        out[cfg["name"]] = cfg["label"]
        for a in cfg["aliases"]:
            out[a] = cfg["label"]
    return out


# ----------------------------------------------------------------------------- connectors

class Connector:
    """One wire format. Subclasses implement `_call`; this class owns config, keys, waiting."""

    kind = "?"
    needs_key = True

    def __init__(self, cfg):
        self.cfg = cfg
        self.name = cfg["name"]
        self.model = cfg["model"]
        self.max_tokens = int(cfg.get("max_tokens", 1600))
        t = cfg.get("temperature")
        self.temperature = None if t in (None, "none") else float(t)
        self.temperature_set = "temperature" in cfg     # a subclass may default it otherwise
        self.timeout = float(cfg.get("timeout", 300))
        self.extra = dict(cfg.get("extra") or {})
        self.key_env = cfg.get("api_key_env")
        # A URL may be private too (an aggregator behind a contract): `base_url_env` names
        # the variable that holds it, the same way `api_key_env` names the key.
        self.base_url = cfg.get("base_url") or (
            env_val(cfg["base_url_env"]) if cfg.get("base_url_env") else None)
        self._client = None

    # -- keys ---------------------------------------------------------------------------
    def key(self):
        return env_val(self.key_env) if self.key_env else None

    def check(self):
        """(ok, why): can this connector be used at all? No model is called (Ollama pings its
        server to see that the model is pulled)."""
        if self.needs_key and not self.key():
            return False, f"no key: {self.key_env or 'api_key_env not set'}"
        if self.cfg.get("base_url_env") and not self.base_url:
            return False, f"no url: {self.cfg['base_url_env']}"
        return True, ""

    # -- the contract -------------------------------------------------------------------
    def call(self, messages, spec):
        """One turn of the model. `spec` is the OpenAI tool list or None (JSON protocol)."""
        for wait in WAITS:
            try:
                return self._call(messages, spec)
            except Exception as e:
                if STOP.is_set() or not self.retryable(e):
                    raise
                if STOP.wait(wait):            # Ctrl-C during a wait: give up now
                    raise
        return self._call(messages, spec)

    def _call(self, messages, spec):
        raise NotImplementedError

    def retryable(self, exc):
        return False

    def describe(self):
        return f"{self.name}: {self.kind} {self.model}"


class OpenAIConnector(Connector):
    """OpenAI chat completions and everything that speaks it.

    Extra keys: `reasoning = true` for models that reject `temperature` and count their
    thinking against the completion budget (the cap is then sent as `max_completion_tokens`
    and raised to at least 4000).

    Tool calls are returned as the endpoint sent them, extra fields included, and `agent.py`
    replays them verbatim on the next hop. Some endpoints attach state to a tool call that the
    next request must carry back (Gemini's `extra_content.thought_signature`); stripping a call
    down to id/name/arguments loses it and the second hop is rejected.
    """

    kind = "openai"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.base_url = self.base_url or "https://api.openai.com/v1"
        self.reasoning = bool(cfg.get("reasoning", False))
        if not self.temperature_set and not self.reasoning:
            self.temperature = 0

    def client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(base_url=self.base_url, api_key=self.key() or "none",
                                  timeout=self.timeout)
        return self._client

    def retryable(self, exc):
        import openai
        return isinstance(exc, (openai.RateLimitError, openai.APIConnectionError,
                                openai.InternalServerError))

    def _call(self, messages, spec):
        kw = {"model": self.model, "messages": messages}
        if self.reasoning:
            kw["max_completion_tokens"] = max(self.max_tokens, 4000)
        else:
            kw["max_tokens"] = self.max_tokens
        if self.temperature is not None:        # for a reasoning model only when the config says
            kw["temperature"] = self.temperature
        if spec:
            kw["tools"] = spec
            kw["tool_choice"] = "auto"
        if self.extra:
            kw["extra_body"] = self.extra
        r = self.client().chat.completions.create(**kw)
        m = r.choices[0].message
        calls = []
        for c in (m.tool_calls or []):
            d = c.model_dump(exclude_none=True) if hasattr(c, "model_dump") else dict(c)
            d.setdefault("id", getattr(c, "id", ""))
            d["type"] = "function"
            fn = d.get("function") or {}
            d["function"] = {**fn, "name": fn.get("name", ""),
                             "arguments": fn.get("arguments") or "{}"}
            calls.append(d)
        u = getattr(r, "usage", None)
        usage = {"in": getattr(u, "prompt_tokens", 0) or 0,
                 "out": getattr(u, "completion_tokens", 0) or 0} if u else {"in": 0, "out": 0}
        return {"content": m.content or "", "tool_calls": calls, "usage": usage}


class AnthropicConnector(Connector):
    """The Messages API through the `anthropic` SDK.

    The harness speaks OpenAI shapes, so each request translates: the system message becomes
    `system`, assistant tool calls become `tool_use` blocks, tool results become `tool_result`
    blocks in one user message (consecutive results must share a message). The response
    translates back: `tool_use` blocks become tool calls, `input` is re-serialised as the JSON
    string the rest of the harness expects.

    Thinking. Models that think return `thinking` blocks that must be sent back unchanged with
    the tool call they precede, or the next request is rejected. The harness does not keep
    them -- its assistant message is `content` + `tool_calls` -- so the connector keeps the raw
    content of every turn it returned, keyed by the ids of the tool calls in it, and replays
    that instead of rebuilding the turn from the harness's copy.

    Extra keys are passed as-is, so a config can set `[extra] thinking = {type = "adaptive"}`
    or `output_config = {effort = "low"}` for models that take them. `temperature` is only
    sent when set: the newest models reject it.
    """

    kind = "anthropic"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.key_env = cfg.get("api_key_env", "ANTHROPIC_API_KEY")
        self._raw = collections.OrderedDict()      # tool_call id -> raw assistant content
        self._raw_lock = threading.Lock()

    def client(self):
        if self._client is None:
            import anthropic
            kw = {"api_key": self.key(), "timeout": self.timeout, "max_retries": 0}
            if self.base_url:
                kw["base_url"] = self.base_url
            self._client = anthropic.Anthropic(**kw)
        return self._client

    def retryable(self, exc):
        import anthropic
        if isinstance(exc, (anthropic.RateLimitError, anthropic.APIConnectionError)):
            return True
        return isinstance(exc, anthropic.APIStatusError) and exc.status_code >= 500

    # -- translation ---------------------------------------------------------------------
    @staticmethod
    def tools(spec):
        return [{"name": t["function"]["name"],
                 "description": t["function"].get("description", ""),
                 "input_schema": t["function"].get("parameters")
                 or {"type": "object", "properties": {}}}
                for t in (spec or [])]

    def _remember(self, blocks, ids):
        with self._raw_lock:
            for i in ids:
                self._raw[i] = blocks
            while len(self._raw) > 5000:
                self._raw.popitem(last=False)

    def _assistant(self, m):
        calls = m.get("tool_calls") or []
        if calls:
            with self._raw_lock:
                raw = self._raw.get(calls[0].get("id"))
            if raw is not None:
                return raw
        blocks = []
        if m.get("content"):
            blocks.append({"type": "text", "text": m["content"]})
        for c in calls:
            fn = c["function"]
            try:
                inp = json.loads(fn.get("arguments") or "{}")
            except ValueError:
                inp = {"_raw": fn.get("arguments")}
            blocks.append({"type": "tool_use", "id": c["id"], "name": fn["name"],
                           "input": inp if isinstance(inp, dict) else {"_raw": inp}})
        return blocks or [{"type": "text", "text": ""}]

    def translate(self, messages):
        """(system, messages) in Messages API shape."""
        system, out = None, []
        for m in messages:
            role = m["role"]
            if role == "system":
                system = m["content"] if system is None else system + "\n\n" + m["content"]
            elif role == "user":
                out.append({"role": "user", "content": m["content"]})
            elif role == "assistant":
                out.append({"role": "assistant", "content": self._assistant(m)})
            elif role == "tool":
                block = {"type": "tool_result", "tool_use_id": m.get("tool_call_id", ""),
                         "content": m.get("content") or ""}
                if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                    out[-1]["content"].append(block)
                else:
                    out.append({"role": "user", "content": [block]})
        return system, out

    def _call(self, messages, spec):
        system, msgs = self.translate(messages)
        kw = {"model": self.model, "max_tokens": self.max_tokens, "messages": msgs}
        if system:
            kw["system"] = system
        if self.temperature is not None:
            kw["temperature"] = self.temperature
        if spec:
            kw["tools"] = self.tools(spec)
        kw.update(self.extra)
        r = self.client().messages.create(**kw)
        text, calls, raw = [], [], []
        for b in r.content:
            d = b.model_dump() if hasattr(b, "model_dump") else dict(b)
            raw.append(d)
            if d.get("type") == "text":
                text.append(d.get("text") or "")
            elif d.get("type") == "tool_use":
                calls.append({"id": d["id"], "type": "function",
                              "function": {"name": d["name"],
                                           "arguments": json.dumps(d.get("input") or {})}})
        if calls:
            self._remember(raw, [c["id"] for c in calls])
        u = getattr(r, "usage", None)
        usage = {"in": getattr(u, "input_tokens", 0) or 0,
                 "out": getattr(u, "output_tokens", 0) or 0} if u else {"in": 0, "out": 0}
        return {"content": "\n".join(text), "tool_calls": calls, "usage": usage}


class OllamaConnector(Connector):
    """Ollama's native chat API, for a model on this machine. No key, no SDK: plain HTTP.

    Extra keys: `num_ctx` (context length; Ollama's default is small enough to cut a
    19-tool manifest), `keep_alive`. Anything under `[extra]` goes into `options`.
    """

    kind = "ollama"
    needs_key = False

    def __init__(self, cfg):
        super().__init__(cfg)
        self.base_url = (self.base_url or "http://localhost:11434").rstrip("/")
        self.num_ctx = cfg.get("num_ctx")
        self.keep_alive = cfg.get("keep_alive")
        if not self.temperature_set:
            self.temperature = 0
        self._seq = 0
        self._seq_lock = threading.Lock()

    def check(self):
        try:
            with urllib.request.urlopen(self.base_url + "/api/tags", timeout=5) as r:
                names = {m.get("name") for m in json.load(r).get("models", [])}
        except Exception as e:
            return False, f"ollama not reachable at {self.base_url}: {e}"
        if self.model not in names and self.model + ":latest" not in names:
            return False, f"model {self.model} not pulled (ollama pull {self.model})"
        return True, ""

    def retryable(self, exc):
        # HTTPError is a URLError: test it first, or a 400 would be retried through every wait.
        if isinstance(exc, urllib.error.HTTPError):
            return exc.code in (429, 500, 502, 503, 504)
        return isinstance(exc, urllib.error.URLError)

    def _messages(self, messages):
        out = []
        for m in messages:
            if m["role"] == "assistant":
                d = {"role": "assistant", "content": m.get("content") or ""}
                if m.get("tool_calls"):
                    d["tool_calls"] = []
                    for c in m["tool_calls"]:
                        try:
                            args = json.loads(c["function"].get("arguments") or "{}")
                        except ValueError:
                            args = {}
                        d["tool_calls"].append({"function": {"name": c["function"]["name"],
                                                             "arguments": args}})
                out.append(d)
            elif m["role"] == "tool":
                out.append({"role": "tool", "content": m.get("content") or ""})
            else:
                out.append({"role": m["role"], "content": m["content"]})
        return out

    def _call(self, messages, spec):
        options = {"num_predict": self.max_tokens}
        if self.temperature is not None:
            options["temperature"] = self.temperature
        if self.num_ctx:
            options["num_ctx"] = int(self.num_ctx)
        options.update(self.extra)
        body = {"model": self.model, "messages": self._messages(messages), "stream": False,
                "options": options}
        if spec:
            body["tools"] = spec
        if self.keep_alive is not None:
            body["keep_alive"] = self.keep_alive
        req = urllib.request.Request(self.base_url + "/api/chat",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            d = json.load(r)
        msg = d.get("message") or {}
        calls = []
        for c in msg.get("tool_calls") or []:
            fn = c.get("function") or {}
            with self._seq_lock:
                self._seq += 1
                cid = f"call_{self._seq}"
            calls.append({"id": cid, "type": "function",
                          "function": {"name": fn.get("name", ""),
                                       "arguments": json.dumps(fn.get("arguments") or {})}})
        usage = {"in": d.get("prompt_eval_count", 0) or 0, "out": d.get("eval_count", 0) or 0}
        return {"content": msg.get("content") or "", "tool_calls": calls, "usage": usage}


CONNECTORS = {c.kind: c for c in (OpenAIConnector, AnthropicConnector, OllamaConnector)}


def build(cfg):
    cls = CONNECTORS.get(cfg["connector"])
    if cls is None:
        raise ValueError(f"{cfg.get('path', cfg['name'])}: unknown connector "
                         f"{cfg['connector']!r}; known: {', '.join(CONNECTORS)}")
    cfg.setdefault("name", cfg.get("label", "?"))
    cfg.setdefault("aliases", [])
    return cls(cfg)


def load(name, models_dir=None):
    """The connector for a config name, a .toml path, or an alias an older script still uses."""
    try:
        return build(find_config(name, models_dir))
    except FileNotFoundError:
        alias = canonical(name, models_dir)
        if alias == name:
            raise
        return build(find_config(alias, models_dir))


# ----------------------------------------------------------------------------- CLI

def main(argv):
    cmd = argv[0] if argv else "list"
    if cmd == "list":
        # TSV, one line per config, for shell loops: RUN|SKIP, name, label, connector,
        # model id, note. SKIP carries the reason (run = false, a missing key, a broken file).
        for cfg in all_configs().values():
            if cfg.get("broken"):
                print("SKIP", cfg["name"], cfg["label"], "?", "?", cfg["note"], sep="\t")
                continue
            conn = build(cfg)
            ok, why = (True, "") if not cfg["run"] else conn.check()
            kind = "RUN" if cfg["run"] and ok else "SKIP"
            note = cfg.get("note", "") if cfg["run"] else "run = false: " + cfg.get("note", "")
            if cfg["run"] and not ok:
                note = why
            print(kind, cfg["name"], cfg["label"], cfg["connector"], cfg["model"], note,
                  sep="\t")
        return 0
    if cmd == "check":
        bad = 0
        for name in argv[1:]:
            try:
                conn = load(name)
                ok, why = conn.check()
            except Exception as e:
                ok, why = False, str(e)
            print(f"{'ok  ' if ok else 'FAIL'} {name}{'' if ok else ' -- ' + why}")
            bad += not ok
        return 1 if bad else 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
