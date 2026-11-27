#!/bin/sh
#$ -cwd
# Run via qsub -g GROUP -l RESOURCE=1 -l h_rt=HH:MM:SS -t 1-N script.
# Set RIYU_TASK_FILE, RIYU_DB, RIYU_OUTPUT_ROOT, RIYU_AUTHORIZATION_LOCAL,
# RIYU_WORKERS, and optionally RIYU_LAKE_BIN / RIYU_SELECTION_PLAN.
set -eu
umask 077
: "${SGE_TASK_ID:?array task ID is required}"
: "${RIYU_TASK_FILE:?approved offset/length TSV required}"
: "${RIYU_DB:?read-only archive path required}"
: "${RIYU_OUTPUT_ROOT:?private output root required}"
: "${RIYU_AUTHORIZATION_LOCAL:?authorization covering this environment is required}"
case "$SGE_TASK_ID" in
    ''|*[!0-9]*) exit 2 ;;
esac
range=$(awk -v task="$SGE_TASK_ID" 'NR == task && NF == 2 && $1 ~ /^[0-9]+$/ && $2 ~ /^[0-9]+$/ { print $1, $2 }' "$RIYU_TASK_FILE")
test -n "$range"
offset=${range% *}
limit=${range#* }
set -- "${RIYU_LAKE_BIN:-target/release/riyu-data-lake}" "$RIYU_DB" "$RIYU_OUTPUT_ROOT/task-$SGE_TASK_ID"
set -- "$@" --offset "$offset" --limit "$limit" --workers "${RIYU_WORKERS:-4}"
set -- "$@" --authorization-reference "$RIYU_AUTHORIZATION_LOCAL" --format parquet --profile C --stop-on-replay-error
if test -n "${RIYU_SELECTION_PLAN:-}"; then
    set -- "$@" --plan "$RIYU_SELECTION_PLAN"
fi
exec "$@"
