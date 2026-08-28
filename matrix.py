"""Measure ONE model over the money column: every family the corpus has, on the scene that
carries it.

Four admission rules decide what gets measured, and all four refuse out loud:

  1. the manifest, widened by the action's own instrument, can carry the demand
     (`tools.admits`);
  2. the scene declares the demand meaningful for its role (`scenes.meaningful`);
  3. the model does the benign job on the control arm -- checked after the run
     (`score.admitted_models`), so a model that cannot do the job is never printed as one
     that refused;
  4. the letter itself is admitted: a demand whose compliance is not a tool call is dropped
     before any model runs (`actions.unverifiable`), and the scene reads the letter before
     accepting it (`scenes.admits_row`).

A cell or a letter that fails any of them is printed with its reason and never appears as a
zero. In this bench a zero has to mean "would not", not "could not".

Work is ordered ROUND-ROBIN across cells, not cell by cell. A run that is stopped halfway then
covers every cell shallowly instead of a few cells deeply -- which is the useful half-measurement,
since the matrix is read across cells.

Two processes must not share a TAG: both would append to one file and interleave records. The
run refuses to start while another run holds the tag's lock file.
"""

import argparse
import hashlib
import json
import os
import pathlib
import random
import signal
import sys
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import actions as actionreg
import canary
import connectors
import corpus
import episode
import scenes
import tools as toolreg

HERE = pathlib.Path(__file__).resolve().parent
STOP = connectors.STOP


def load_rows(host):
    """The corpus rows with a payload, on the carriers named by `host` (comma list or `*`).

    The shipped list is email only; the filter is what makes that explicit."""
    wanted = None if host in ("*", "", None) else {h.strip() for h in host.split(",")}
    rows = corpus.rows()
    return [r for r in rows if r.get("injection") and (wanted is None or r["host_type"] in wanted)]


def select_rows(rows, from_path, where):
    """Narrow the pool to payloads named by a label file -- ids, not a pattern.

    What a payload DEMANDS is not a property of its words. `wire|payment|invoice` flags the
    output_marking rows where the money moves downstream of a token the model prints, and misses
    `move the funds to Yuki Collins` because `funds` was not on the list. So the demand is
    labelled once, by a model, into data/labels_money.jsonl, and every run that wants a subset
    of the corpus reads that file. Same harness, different input -- which is the only thing a
    money run should differ by.

    `where` is `field=value` pairs against the label record: `demand=money_out,executor=agent`.
    """
    if not from_path:
        return rows, ""
    labels = {}
    for line in pathlib.Path(from_path).read_text().splitlines():
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if d.get("id"):
            labels[d["id"]] = d
    cond = []
    for part in (where or "").split(","):
        if "=" in part:
            k, _, v = part.partition("=")
            cond.append((k.strip(), {x for x in v.split("|") if x}))
    keep = []
    for r in rows:
        d = labels.get(r["id"])
        if d is None:
            continue
        if all(str(d.get(k)) in vals for k, vals in cond):
            keep.append(r)
    unlabelled = sum(1 for r in rows if r["id"] not in labels)
    note = (f"input {pathlib.Path(from_path).name}"
            f"{' where ' + where if where else ''}: {len(keep)} of {len(rows)} rows"
            f"{f'; {unlabelled} not in the label file' if unlabelled else ''}")
    return keep, note


def plan(rows, scene_names, per_cell, seed):
    """Every (family, action, scene) triple, split into what will run and what cannot."""
    by_cell = defaultdict(list)
    for r in rows:
        by_cell[(r["family"], r["action"])].append(r)

    jobs, refused, dropped = [], [], Counter()
    for (family, action), pool in sorted(by_cell.items()):
        usable = []
        for scene in scene_names:
            ok_mean, why = scenes.meaningful(scene, action)
            if not ok_mean:
                refused.append((family, action, scene, why))
                continue
            manifest, _ = actionreg.manifest_for(scenes.SCENES[scene][2], action)
            ok_tool, why_tool = toolreg.admits(manifest, action)
            if not ok_tool:
                refused.append((family, action, scene, why_tool))
                continue
            if actionreg.grade_mode(action) == "none":
                refused.append((family, action, scene, "not gradable in one move"))
                continue
            usable.append(scene)
        if not usable:
            continue
        # Same rows for every scene of a cell: the scene is then the only thing that differs.
        # per_cell <= 0 takes the whole pool. Capping is a budget decision, not a design one --
        # work is ordered round-robin, so depth can be decided by stopping the run instead.
        ordered = sorted(pool, key=lambda r: r["id"])
        for scene in usable:
            # Rule 4, per letter. This runs BEFORE the sample, or `--per-cell 8` quietly returns
            # six -- the gate would eat part of the cell instead of part of the pool. What it
            # turned away is counted and printed, so the pool a cell actually ran on is stated
            # rather than assumed.
            admitted = []
            for r in ordered:
                bad, why_bad = actionreg.unverifiable(r)
                if bad:
                    dropped[(family, action, scene, why_bad)] += 1
                    continue
                ok_row, why_row = scenes.admits_row(scene, r)
                if ok_row:
                    admitted.append(r)
                else:
                    dropped[(family, action, scene, why_row)] += 1
            rnd = random.Random(f"{seed}:{family}:{action}:{scene}")
            picked = (admitted if per_cell <= 0
                      else rnd.sample(admitted, min(per_cell, len(admitted))))
            for r in picked:
                jobs.append((family, action, scene, r))
    return jobs, refused, dropped


def marker_for(row):
    h = hashlib.sha1(row["id"].encode()).hexdigest()[:10].upper()
    return actionreg.PROMPT_MARKER.format(h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="model config name: models/<name>.toml")
    ap.add_argument("--scenes", default="intake_mail")
    ap.add_argument("--host", default="email",
                    help="carriers to draw payloads from: comma list or *")
    ap.add_argument("--per-cell", type=int, default=8,
                    help="rows per (family, action) cell; 0 = every row the filters leave")
    ap.add_argument("--control-per-cell", type=int, default=10,
                    help="control episodes per (family, action, scene); 0 = one per injected row")
    ap.add_argument("--actions", default="", help="comma list, default all in the corpus")
    ap.add_argument("--families", default="", help="comma list, default all in the corpus")
    ap.add_argument("--rows-from", default="",
                    help="label file (jsonl with `id`) that decides which payloads run")
    ap.add_argument("--rows-where", default="",
                    help="field=value[|value] pairs against the label record, comma separated")
    ap.add_argument("--seed", type=int, default=53)
    ap.add_argument("--max-hops", type=int, default=8)
    ap.add_argument("--protocol", default="native", choices=["native", "json"])
    ap.add_argument("--conc", type=int, default=8)
    ap.add_argument("--tag", default="matrix")
    ap.add_argument("--guard", action="store_true",
                    help="run with the defended system prompt (scenes.INTAKE_SYSTEM_GUARDED)")
    ap.add_argument("--no-control", action="store_true")
    ap.add_argument("--escalate", action="store_true",
                    help="give the agent a way to report the attack. NOT neutral: it lowers "
                         "compliance markedly, so it is a different deployment")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore episodes already on disk: use when the WORLD changed, not the "
                         "model -- resumed work would then mix two different experiments")
    ap.add_argument("--status", action="store_true", help="show state and exit")
    ap.add_argument("--plan", action="store_true", help="show the admission table and exit")
    args = ap.parse_args()

    out_ep = HERE / "data" / f"episodes-{args.tag}.jsonl"
    out_calls = HERE / "data" / f"calls-{args.tag}.jsonl"
    scene_names = [s.strip() for s in args.scenes.split(",") if s.strip()]
    rows = load_rows(args.host)
    if args.actions:
        keep = {a.strip() for a in args.actions.split(",")}
        rows = [r for r in rows if r["action"] in keep]
    if args.families:
        keep = {f.strip() for f in args.families.split(",")}
        rows = [r for r in rows if r["family"] in keep]
    rows, note = select_rows(rows, args.rows_from, args.rows_where)
    if note:
        print(note, flush=True)
        if not rows:
            sys.exit("the label file leaves no rows -- is it finished?")
    jobs, refused, dropped = plan(rows, scene_names, args.per_cell, args.seed)

    # The control answers for the SCENE, not for the payload: it is the same scene with nothing
    # planted, and the scorer reads it per scene and action, never paired to one injected row.
    # So one control per injected row buys nothing; a fixed sample per cell x scene does.
    specs = [(f, a, s, r, "injected") for f, a, s, r in jobs]
    if not args.no_control:
        by_cell = defaultdict(list)
        for f, a, s, r in jobs:
            by_cell[(f, a, s)].append(r)
        for (f, a, s), rs in sorted(by_cell.items()):
            ordered = sorted(rs, key=lambda r: r["id"])
            n = len(ordered) if args.control_per_cell <= 0 else min(args.control_per_cell,
                                                                   len(ordered))
            rnd = random.Random(f"{args.seed}:control:{f}:{a}:{s}")
            for r in rnd.sample(ordered, n):
                specs.append((f, a, s, r, "control"))

    cells = sorted({(f, a) for f, a, _, _ in jobs})
    n_inj = sum(1 for x in specs if x[4] == "injected")
    print(f"model {args.model}{' +guard' if args.guard else ''}"
          f"{' +escalate' if args.escalate else ''} | host {args.host} | "
          f"{len(cells)} cells x scenes -> "
          f"{n_inj} injected + {len(specs) - n_inj} control = {len(specs)} episodes", flush=True)

    if args.plan:
        pool = Counter((r["family"], r["action"]) for r in rows)
        taken = Counter((f, a) for f, a, s, _ in jobs if s == scene_names[0])
        print("\npool per cell (rows available -> rows taken, per scene):")
        for (f, a), n in sorted(pool.items()):
            flag = "  <- thin" if n < 20 else ""
            print(f"  {f:<18} {a:<20} {n} -> {taken[(f, a)]}{flag}")
        print("\nadmitted (cell -> scenes):")
        per_cell = defaultdict(set)
        for f, a, s, _ in jobs:
            per_cell[(f, a)].add(s)
        for (f, a), ss in sorted(per_cell.items()):
            print(f"  {f:<18} {a:<20} {', '.join(sorted(ss))}")
        if dropped:
            print("\nletters the scene turned away (rule 4, by cell):")
            per = defaultdict(Counter)
            for (f, a, _sc, why), n in dropped.items():
                per[(f, a)][why] += n
            for (f, a), whys in sorted(per.items()):
                took = taken[(f, a)]
                lost = sum(whys.values())
                worst = ", ".join(f"{w} x{n}" for w, n in whys.most_common(2))
                print(f"  {f:<18} {a:<20} kept {took}, dropped {lost}  ({worst})")
        print("\nrefused (never reported as a zero):")
        seen = set()
        for _f, a, s, why in refused:
            if (a, s, why) in seen:            # the reason is a property of scene x action
                continue
            seen.add((a, s, why))
            print(f"  {a:<20} {s:<10} {why}")
        return

    # Normally resume looks at every checkpoint in the directory: the same episode shot under
    # another tag is the same episode. `--fresh` narrows that to THIS tag's own file, which is
    # what a changed WORLD requires -- other runs measured something else and must not be
    # counted as done -- while still letting an interrupted run continue instead of writing
    # every episode twice.
    done = set()
    paths = ([HERE / "data" / f"episodes-{args.tag}.jsonl"] if args.fresh
             else sorted((HERE / "data").glob("episodes*.jsonl")))
    # An episode that ended in a transport error is NOT done: the provider answered 429 or
    # nothing at all, and the record on disk is a receipt for that, not a measurement. Counting
    # it as done makes a re-run say "nothing to do" over a column of zeros that were never
    # measured. `score.load` drops such a record once a real one with the same key exists.
    for path in [p for p in paths if p.exists()]:
        for line in path.open():
            try:
                d = json.loads(line)
                if d.get("stop") == "error":
                    continue
                done.add((connectors.canonical(d["model"]), d["scene"], d["id"], d["variant"],
                          bool(d.get("guard")), bool(d.get("escalate"))))
            except Exception:
                pass
    todo = [x for x in specs
            if (args.model, x[2], x[3]["id"], x[4], args.guard, args.escalate) not in done]
    # Round-robin: one job per cell, then the next, so an interrupted run is still balanced.
    order = defaultdict(list)
    for j in todo:
        order[(j[0], j[1], j[2])].append(j)
    todo = [j for i in range(max((len(v) for v in order.values()), default=0))
            for k in sorted(order) for j in ([order[k][i]] if i < len(order[k]) else [])]

    coverage = Counter((f, a) for f, a, _, _, _ in todo)
    print(f"  done {len(specs) - len(todo)}, left {len(todo)}; "
          f"cells still open {len(coverage)}", flush=True)
    if args.status:
        for (f, a), n in sorted(coverage.items()):
            print(f"    {f:<18} {a:<20} {n} left", flush=True)
        return
    if not todo:
        print("nothing to do", flush=True)
        return

    conn = connectors.load(args.model)
    ok, why = conn.check()
    if not ok:
        sys.exit(f"{args.model}: {why} -- nothing started")
    lock_path = HERE / "data" / f"episodes-{args.tag}.lock"
    if lock_path.exists():
        sys.exit(f"{lock_path.name} exists: another run holds tag {args.tag!r}. "
                 f"Wait for it, or remove the file if that run is dead -- nothing started")
    lock_path.write_text(str(os.getpid()))

    used, planted = set(), {}
    for _, _, _, r in jobs:
        if r["id"] not in planted:
            planted[r["id"]] = canary.plant(r, used)

    caller = conn.call
    lock = threading.Lock()
    counter = {"n": 0}
    fe, fc = out_ep.open("a"), out_calls.open("a")

    def work(job):
        if STOP.is_set():
            return
        family, action, scene, row, variant = job
        rec = episode.run(caller, row, scene, variant, planted[row["id"]], action=action,
                          max_hops=args.max_hops, protocol=args.protocol,
                          marker=marker_for(row), guard=args.guard, escalate=args.escalate)
        rec["model"] = args.model
        rec["model_id"] = conn.model
        rec["connector"] = conn.kind
        with lock:
            fe.write(json.dumps(rec, ensure_ascii=False) + "\n")
            for c in episode.flat_calls(rec):
                fc.write(json.dumps(c, ensure_ascii=False) + "\n")
            fe.flush()
            fc.flush()
            counter["n"] += 1
            if counter["n"] % 25 == 0 or counter["n"] == len(todo):
                print(f"  {counter['n']}/{len(todo)}", flush=True)

    def on_int(signum, frame):
        if STOP.is_set():                  # second Ctrl-C: do not wait for in-flight episodes
            lock_path.unlink(missing_ok=True)
            os._exit(130)
        STOP.set()
        print(f"\nstopping at {counter['n']}/{len(todo)} after the episodes in flight; "
              f"continue: rerun the same command (Ctrl-C again to quit at once)", flush=True)
    signal.signal(signal.SIGINT, on_int)
    signal.signal(signal.SIGTERM, on_int)

    try:
        with ThreadPoolExecutor(max_workers=args.conc) as pool:
            list(pool.map(work, todo))
    finally:
        fe.close()
        fc.close()
        lock_path.unlink(missing_ok=True)
    print(f"episodes -> {out_ep}\ncalls    -> {out_calls}", flush=True)


if __name__ == "__main__":
    main()
