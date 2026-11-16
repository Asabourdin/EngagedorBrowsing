#!/usr/bin/env bash
# Rerun the whole project from start to end, without manual intervention.
#
# Usage (from any folder, with the environment of requirements.txt activated):
#   bash run_all.sh              # uses data/sessions.csv already in the repo
#   bash run_all.sh --download   # first rebuilds data/sessions.csv from Yambda (large download)
#
# The notebooks are run in order, each in a fresh kernel, and saved with their outputs.
# Later layers read the results of earlier ones, so the script stops at the first error.

set -euo pipefail

# The notebooks read and write data/, results/ and figures/ relative to code/
cd "$(dirname "$0")/code"

if [[ "${1:-}" == "--download" ]]; then
    echo "== Building data/sessions.csv from Yambda"
    python prepare_yambda.py
elif [[ ! -f data/sessions.csv ]]; then
    echo "data/sessions.csv is missing: run 'bash run_all.sh --download'" >&2
    exit 1
fi

start_all=$SECONDS
for notebook in layer1_baseline.ipynb layer2_session_types.ipynb layer3_mood_switching.ipynb; do
    echo "== Running $notebook"
    start=$SECONDS
    # timeout=-1: no time limit per cell (some Layer 3 cells take several minutes)
    python -m jupyter nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=-1 "$notebook"
    echo "   done in $(( (SECONDS - start) / 60 )) min $(( (SECONDS - start) % 60 )) s"
done
echo "== All notebooks done in $(( (SECONDS - start_all) / 60 )) min"
