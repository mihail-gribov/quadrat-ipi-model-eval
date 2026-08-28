"""The agent loop without a model: a scripted caller stands in for the connector.

    python3 test_agent.py        # or: pytest test_agent.py

What is proved: the JSON protocol's parser survives what models actually emit; an unknown tool
is refused and logged, not raised; `finish` ends the episode; the hop cap and a transport error
are recorded as what they are; a malformed connector answer becomes `stop = error`, not a crash.
"""
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import agent     # noqa: E402
import world     # noqa: E402

FAILED = []
MANIFEST = ["read_last_email", "log_message", "queue_payment", "reply_to_user", "finish"]


def check(name, got, want):
    ok = got == want
    print(f"  {'ok  ' if ok else 'FAIL'} {name:<56} {got!r}" + ("" if ok else f" != {want!r}"))
    if not ok:
        FAILED.append(name)


def scripted(*turns):
    """A caller that answers each hop with the next scripted message, then stops calling."""
    turns = list(turns)

    def call(messages, spec):
        if turns:
            t = turns.pop(0)
            if isinstance(t, Exception):
                raise t
            return t
        return {"content": "Done.", "tool_calls": [], "usage": {"in": 1, "out": 1}}
    return call


def tool_call(cid, name, args):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": args}}


def fresh():
    w = world.World()
    w.mail["msg-1"] = {"raw": "Subject: Hello\n\nBody."}
    return w


def main():
    print("\nJSON PROTOCOL PARSER")
    p = agent.parse_json_calls
    check("plain object", p('{"tool_calls": [{"tool": "finish", "args": {"answer": "x"}}]}'),
          [("finish", {"answer": "x"})])
    check("fenced", p('```json\n{"tool_calls": [{"tool": "finish"}]}\n```'), [("finish", {})])
    check("prose before the object", p('Sure. {"tool_calls": [{"tool": "finish"}]} done'),
          [("finish", {})])
    check("a brace inside a string does not cut the object",
          p('{"tool_calls": [{"tool": "log_message", "args": {"subject": "a } b"}}]}'),
          [("log_message", {"subject": "a } b"})])
    check("no object", p("I would rather not."), [])
    check("not a dict", p("[1, 2]"), [])
    check("args that are not a dict are dropped", p('{"tool_calls": [{"tool": "f", "args": 3}]}'),
          [("f", {})])

    print("\nNATIVE LOOP")
    w = fresh()
    res = agent.run(scripted(
        {"content": "", "tool_calls": [tool_call("c1", "read_last_email", "{}")],
         "usage": {"in": 10, "out": 2}},
        {"content": "Logging.", "tool_calls": [tool_call("c2", "log_message",
                                                          '{"subject": "Hello"}'),
                                               tool_call("c3", "finish", '{"answer": "ok"}')],
         "usage": {"in": 20, "out": 3}},
    ), w, "sys", "user", MANIFEST)
    check("stops on finish", res["stop"], "finish")
    check("every call logged", [c["tool"] for c in w.log],
          ["read_last_email", "log_message", "finish"])
    check("the journal has the subject", w.journal, [{"subject": "Hello"}])
    check("usage summed over hops", res["usage"], {"in": 30, "out": 5})
    check("what was said is kept", res["said"], "Logging.")
    check("tool results went back as tool messages",
          [m["role"] for m in res["messages"]][2:], ["assistant", "tool", "assistant", "tool",
                                                     "tool"])

    w = fresh()
    res = agent.run(scripted(
        {"content": "", "tool_calls": [tool_call("c1", "pay_invoice", '{"amount": 5}')],
         "usage": {}},
    ), w, "sys", "user", MANIFEST)
    check("unknown tool: refused, logged, not raised",
          (w.log[0]["ok"], w.log[0]["result"].startswith("No such tool")), (False, True))
    check("then the model stopped calling", res["stop"], "no_call")

    w = fresh()
    res = agent.run(scripted(
        {"content": "", "tool_calls": [tool_call("c1", "log_message", "not json")], "usage": {}},
    ), w, "sys", "user", MANIFEST)
    check("unparseable arguments are kept raw", w.log[0]["args"], {"_raw": "not json"})

    w = fresh()
    res = agent.run(scripted(*[{"content": "", "tool_calls": [tool_call(f"c{i}",
                                                                          "read_last_email", "{}")],
                                "usage": {}} for i in range(20)]),
                    w, "sys", "user", MANIFEST, max_hops=3)
    check("hop cap recorded as max_hops", (res["stop"], len(w.log)), ("max_hops", 3))

    w = fresh()
    res = agent.run(scripted(RuntimeError("connection reset")), w, "sys", "user", MANIFEST)
    check("transport error recorded, not raised",
          (res["stop"], "connection reset" in (res["error"] or "")), ("error", True))

    w = fresh()
    res = agent.run(scripted({"content": "", "tool_calls": [{"bogus": 1}], "usage": {}}),
                    w, "sys", "user", MANIFEST)
    check("malformed connector answer -> error", res["stop"], "error")

    print("\nJSON LOOP")
    w = fresh()
    res = agent.run(scripted(
        {"content": '{"tool_calls": [{"tool": "read_last_email"}]}', "usage": {}},
        {"content": '{"tool_calls": [{"tool": "log_message", "args": {"subject": "Hello"}}, '
                    '{"tool": "finish", "args": {"answer": "ok"}}]}', "usage": {}},
    ), w, "sys", "user", MANIFEST, protocol="json")
    check("same outcome through the JSON protocol",
          (res["stop"], [c["tool"] for c in w.log]),
          ("finish", ["read_last_email", "log_message", "finish"]))
    check("results came back as a user turn", res["messages"][3]["role"], "user")
    check("the manifest was rendered into the system prompt",
          "queue_payment(" in res["messages"][0]["content"], True)

    print(f"\n{'ALL PASSED' if not FAILED else 'FAILED: ' + ', '.join(FAILED)}")
    return 1 if FAILED else 0


def test_agent():
    """For pytest: the whole file is one test, the printed table is its detail."""
    assert main() == 0, FAILED


if __name__ == "__main__":
    sys.exit(main())
