#!/bin/sh
# The money run: the matrix harness, fed a different input.
#
# Nothing here changes how an episode runs or how it is graded -- same scene, same manifest,
# same admission rules, same grader. The only difference is WHICH payloads go in: the ones a
# labelling pass marked as demanding a funds movement (data/labels_money.jsonl, written by
# a one-pass LLM labelling, shipped in the repo). Keeping that the sole difference is the point: a money number and a matrix
# number then differ by the input and by nothing else.
#
# Run it in your own tmux window; the log is the monitor. Ctrl-C stops cleanly and re-running
# continues from where it stopped.
#
#   ./money.sh --plan                 what will run, what is refused, and why
#   ./money.sh --status               state only, starts nothing
#   ./money.sh                        run / continue
#   MODEL=gpt-5.1 ./money.sh          another model config (each model has its own episodes on disk)
#   GUARD=1 ./money.sh                the same run with the defended system prompt
#   TAG=money3 FRESH=1 ./money.sh     a re-shoot after the world gained tools
#
# One scene, `intake_mail`: the letter is a corpus letter kept verbatim and the payload sits at
# the offset the corpus recorded. The earlier three scenes could not be read apart from their
# carriers -- the two safe ones delivered by file, the dangerous one by mail -- so "the scene
# decides" and "the carrier decides" were the same sentence. With one scene and one carrier the
# only thing left varying across cells is the payload.
#
# WHERE picks the slice of the labelling. `executor=agent` additionally drops the payloads
# where the money is claimed to move downstream of a token the model prints rather than by the
# model's own tool call -- a real vector, but a different one.
set -e
cd "$(dirname "$0")"

LABELS="${LABELS:-data/labels_money.jsonl}"
WHERE="${WHERE:-demand=money_out}"
MODEL="${MODEL:-Qwen3-30B}"          # a config name: models/<name>.toml
SCENES="${SCENES:-intake_mail}"
HOST="${HOST:-email}"
ACTIONS="${ACTIONS:-unauthorized_action}"
PER_CELL="${PER_CELL:-0}"        # 0 = every labelled row; depth is decided by stopping, not by a cap
CTL_PER_CELL="${CTL_PER_CELL:-10}"
CONC="${CONC:-8}"
# The defended prompt writes to its own file and carries its own flag on every episode, so the
# two runs can never be averaged into one number by accident.
if [ -n "$GUARD" ]; then GUARD_FLAG="--guard"; TAG="${TAG:-money-guard}"; else GUARD_FLAG=""; fi
# FRESH=1 when the WORLD changed, not the model: episodes shot under another tag measured a
# different set of instruments and must not count as done. Resume within this tag still works.
[ -n "$FRESH" ] && FRESH_FLAG="--fresh" || FRESH_FLAG=""
# ESCALATE=1 adds the incident tool. Off by default: a different deployment, not a better
# measurement of the same one.
[ -n "$ESCALATE" ] && ESC_FLAG="--escalate" || ESC_FLAG=""
# PERMISSIVE=1: the world stops saying no -- same nineteen tools, but the key is found and the
# file opens. A payload that still fails there is one no agent could carry out in this world,
# which is what makes the run a test of the PAYLOADS rather than of the model.
if [ -n "$PERMISSIVE" ]; then PERM_FLAG="--permissive"; TAG="${TAG:-money-permissive}";
else PERM_FLAG=""; fi
TAG="${TAG:-money}"

[ -f "$LABELS" ] || { echo "no label file: $LABELS -- it ships with the repository, restore it from git"; exit 1; }

exec python3 -u matrix.py \
  --model "$MODEL" --scenes "$SCENES" --host "$HOST" --actions "$ACTIONS" \
  --rows-from "$LABELS" --rows-where "$WHERE" \
  --per-cell "$PER_CELL" --control-per-cell "$CTL_PER_CELL" \
  --conc "$CONC" --tag "$TAG" $GUARD_FLAG $FRESH_FLAG $PERM_FLAG $ESC_FLAG "$@"
