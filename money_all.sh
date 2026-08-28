#!/bin/sh
# The money run across every model config in models/, one after another.
#
# Nothing here measures anything of its own: it is `money.sh` in a loop, so one model's number
# differs from another's by the model and by nothing else. The list is the config directory
# (`python3 connectors.py list`): a model that could not be reached is a file with `run = false`
# and a note, instead of a name quietly missing from a script.
#
# Sequential on purpose: the providers rate-limit per account, and two runs at once would trade
# wall-clock for a throttling pattern that lands differently on each model.
#
# Run it in your own tmux window; the log is the monitor. Ctrl-C stops cleanly at the episode
# boundary, and re-running continues where it stopped -- done work is counted from the episodes
# on disk, never from this log.
#
#   ./money_all.sh --status               state per model, starts nothing
#   ./money_all.sh --plan                 the admission table per model, starts nothing
#   ./money_all.sh                        run / continue
#   ./money_all.sh --protocol json        extra arguments go to every money.sh
#   ONLY="gpt-5.1 gemini-3.7-flash" ./money_all.sh    just these configs (by name or label)
#   PARALLEL=1 ./money_all.sh             one chain per API key, chains side by side
#   AGENT_BENCH_MODELS=/path/to/configs ./money_all.sh   another config directory, searched first
#
# Order: configs run by their `order` key (default 50), then by name; put a slow or dear model
# last with `order = 90`. TAG, CTL_PER_CELL, GUARD, ESCALATE, FRESH pass through to money.sh.
set -e
cd "$(dirname "$0")"

CTL_PER_CELL="${CTL_PER_CELL:-20}"
export CTL_PER_CELL
# GUARD is passed through untouched: the defended run is the same sweep over the same config with
# one line added to the system prompt, and `money.sh` already gives it its own file and its own
# resume key. Nothing here needs to know what the flag means.
export GUARD
# Passed through for the same reason as GUARD: the sweep does not need to know what it means.
export FRESH
export ESCALATE
export AGENT_BENCH_MODELS

# The config directory is the source of truth for what runs and what is skipped WITH A REASON.
# `connectors.py list` prints one TSV line per config: RUN|SKIP, name, label, connector, model
# id, note. A missing key is a SKIP with the key's name, found before anything is spent.
LIST=$(ONLY="$ONLY" python3 - <<'PY'
import os, subprocess, sys
only = {x for x in os.environ.get("ONLY", "").split() if x}
out = subprocess.run([sys.executable, "connectors.py", "list"], capture_output=True, text=True)
if out.returncode != 0:
    sys.exit(out.stderr or "connectors.py list failed")
for line in out.stdout.splitlines():
    kind, name, label, conn, model, note = (line.split("\t") + [""] * 6)[:6]
    if only and name not in only and label not in only:
        continue
    # ONLY deliberately overrides `run = false` -- that is how a disabled model gets re-tested.
    # A missing key is never overridden: nothing can start without it.
    if kind == "SKIP" and (not only or note.startswith("no key") or note.startswith("unreadable")):
        print("SKIP", name, note, sep="\t")
        continue
    print("RUN", name, label, sep="\t")
PY
)

echo "$LIST" | while IFS="$(printf '\t')" read -r kind a b; do
  if [ "$kind" = "SKIP" ]; then echo "skipped: $a -- $b"; fi
done

RUNNING=$(echo "$LIST" | awk -F'\t' '$1=="RUN"{print $2}')
[ -n "$RUNNING" ] || { echo "nothing to run"; exit 1; }

# Preconditions before anything is spent: a missing key found half way through a six-model run
# is found after five models have already paid for themselves. `check` calls nothing.
python3 connectors.py check $RUNNING || { echo "nothing started"; exit 1; }

if [ "$1" = "--status" ]; then
  for m in $RUNNING; do
    echo "=== $m"
    MODEL="$m" ./money.sh --status 2>&1 | sed -n '2p;/done /p'
  done
  exit 0
fi
if [ "$1" = "--plan" ]; then
  for m in $RUNNING; do
    echo "=== $m"
    MODEL="$m" ./money.sh --plan
  done
  exit 0
fi

trap 'echo; echo "stopped. continue: ./money_all.sh"; kill 0 2>/dev/null; exit 130' INT TERM

total=$(echo "$RUNNING" | wc -w)

if [ -n "$PARALLEL" ]; then
  # One chain per KEY, chains side by side, models inside a chain still one at a time. That is
  # where the real constraint sits: rate limits are per account, so two models on the same key
  # would only trade wall-clock for a throttling pattern that lands unevenly -- while two models
  # on different accounts do not touch each other at all.
  #
  # Each chain also gets its OWN checkpoint file. Two processes appending to one jsonl will
  # eventually interleave a long line at a buffer boundary and corrupt it; `run.py` says so in
  # its header, and the tag is what keeps that from happening. Read them back with a glob:
  #   python3 report.py --only money --tag 'money7*,<TAG>*' 
  KEYS=$(python3 - $RUNNING <<'PY'
import sys, connectors
for n in sys.argv[1:]:
    print(n, connectors.find_config(n).get("api_key_env") or "nokey")
PY
)
  for key in $(echo "$KEYS" | awk '{print $2}' | sort -u); do
    (
      for m in $(echo "$KEYS" | awk -v k="$key" '$2==k{print $1}'); do
        echo "[$key] === $m${GUARD:+  [defended prompt]}"
        MODEL="$m" TAG="${TAG:-money}${GUARD:+-guard}-$key" ./money.sh "$@" 2>&1 | sed "s/^/[$key] /" \
          || echo "[$key] !! $m failed or was interrupted -- continuing with the rest"
      done
      echo "[$key] chain done"
    ) &
  done
  wait
else
  n=0
  for m in $RUNNING; do
    n=$((n + 1))
    echo
    echo "======================================================================"
    echo "  model $n/$total: $m${GUARD:+  [defended prompt]}"
    echo "======================================================================"
    MODEL="$m" ./money.sh "$@" || echo "  !! $m failed or was interrupted -- continuing with the rest"
  done
fi

echo
echo "all models done. read them together:"
echo "  python3 report.py --only money --tag 'money7*,${TAG:-money}*'"
