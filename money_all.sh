#!/bin/sh
# The money run across every model in models.jsonl, one after another.
#
# Nothing here measures anything of its own: it is `money.sh` in a loop, so one model's number
# differs from another's by the model and by nothing else. The list lives in `models.jsonl`
# rather than in this file -- a model that could not be reached is then a line with `run: false`
# and a reason, instead of a name quietly missing from a script.
#
# Sequential on purpose: the providers rate-limit per account, and two runs at once would trade
# wall-clock for a throttling pattern that lands differently on each model.
#
# Run it in your own tmux window; the log is the monitor. Ctrl-C stops cleanly at the episode
# boundary, and re-running continues where it stopped -- done work is counted from the episodes
# on disk, never from this log.
#
#   ./money_all.sh --status               state per model, starts nothing
#   ./money_all.sh                        run / continue
#   ONLY="gpt-5.1 gemini-3.6-flash" ./money_all.sh    just these labels
#   MODELS_FILE=other.jsonl ./money_all.sh
set -e
cd "$(dirname "$0")"

MODELS_FILE="${MODELS_FILE:-models.jsonl}"
CTL_PER_CELL="${CTL_PER_CELL:-20}"
export CTL_PER_CELL
# GUARD is passed through untouched: the defended run is the same sweep over the same config with
# one line added to the system prompt, and `money.sh` already gives it its own file and its own
# resume key. Nothing here needs to know what the flag means.
export GUARD
# Passed through for the same reason as GUARD: the sweep does not need to know what it means.
export FRESH
export PERMISSIVE
export ESCALATE

[ -f "$MODELS_FILE" ] || { echo "no model config: $MODELS_FILE"; exit 1; }

# The config is the source of truth for what runs and what is skipped WITH A REASON.
LIST=$(ONLY="$ONLY" python3 - "$MODELS_FILE" <<'PY'
import json, os, sys
only = {x for x in os.environ.get("ONLY", "").split() if x}
for line in open(sys.argv[1]):
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    d = json.loads(line)
    if only and d["label"] not in only and d["model"] not in only:
        continue
    # ONLY deliberately overrides `run: false` -- that is how a disabled model gets re-tested.
    # It therefore needs labels to be unique, or asking for one model silently starts two: a
    # direct route and an aggregator route once shared the label `gemini-3.7-flash`.
    if not only and not d.get("run", True):
        print("SKIP", d["label"], d.get("note", "run: false"), sep="\t")
        continue
    print("RUN", d["model"], d["label"], sep="\t")
PY
)

# `|| true` is not decoration: the loop's last command is the SKIP test, and when the config ends
# on a RUN line it returns 1, which `set -e` reads as a failed pipeline and kills the script --
# silently, before anything is printed. It only showed up once ONLY= made a RUN line the last one.
echo "$LIST" | while IFS="$(printf '\t')" read -r kind a b; do
  [ "$kind" = "SKIP" ] && echo "skipped: $a -- $b"
  true
done || true

RUNNING=$(echo "$LIST" | awk -F'\t' '$1=="RUN"{print $2}')
[ -n "$RUNNING" ] || { echo "nothing to run"; exit 1; }

# Preconditions before anything is spent: a missing key found half way through a six-model run
# is found after five models have already paid for themselves.
for m in $RUNNING; do
  case "$m" in
    oai:*) key=OPENAI_API_KEY ;;
    neb:*) key=NEBIUS_API_KEY ;;
    gem:*) key=GOOGLE_API_KEY ;;
    oac:*) key=OAC_API_KEY ;;
    mis:*) key=MISTRAL_API_KEY ;;
    *) echo "unknown provider prefix in $m"; exit 1 ;;
  esac
  eval "v=\$$key"
  [ -n "$v" ] || v=$(grep -s "^$key=" .env | head -1 | cut -d= -f2-)
  [ -n "$v" ] || { echo "no $key for $m -- nothing started"; exit 1; }
done

if [ "$1" = "--status" ] || [ "$1" = "--plan" ]; then
  for m in $RUNNING; do
    echo "=== $m"
    MODEL="$m" ./money.sh "$1" 2>&1 | sed -n '2p;/done /p'
  done
  exit 0
fi

trap 'echo; echo "stopped. continue: ./money_all.sh"; kill 0 2>/dev/null; exit 130' INT TERM

total=$(echo "$RUNNING" | wc -w)

if [ -n "$PARALLEL" ]; then
  # One chain per PROVIDER, chains side by side, models inside a chain still one at a time.
  # That is where the real constraint sits: rate limits are per account, so two models on the
  # same provider would only trade wall-clock for a throttling pattern that lands unevenly --
  # while two models on DIFFERENT providers do not touch each other at all.
  #
  # Each chain also gets its OWN checkpoint file. Two processes appending to one jsonl will
  # eventually interleave a long line at a buffer boundary and corrupt it; `run.py` says so in
  # its header, and the tag is what keeps that from happening. Read them back with a glob:
  #   python3 score.py --tag 'money*' ...
  for prov in $(echo "$RUNNING" | tr ' ' '\n' | sed 's/:.*//' | sort -u); do
    (
      for m in $RUNNING; do
        case "$m" in "$prov":*) ;; *) continue ;; esac
        echo "[$prov] === $m${GUARD:+  [defended prompt]}"
        MODEL="$m" TAG="${TAG:-money}${GUARD:+-guard}-$prov" ./money.sh 2>&1 | sed "s/^/[$prov] /" \
          || echo "[$prov] !! $m failed or was interrupted -- continuing with the rest"
      done
      echo "[$prov] chain done"
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
    MODEL="$m" ./money.sh || echo "  !! $m failed or was interrupted -- continuing with the rest"
  done
fi

echo
echo "all models done. read them together:"
echo "  python3 score.py --tag 'money*' --family all \\"
echo "    --labels data/labels_money.jsonl --labels-where demand=money_out"
