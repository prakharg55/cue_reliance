#!/usr/bin/env bash
# Regenerates every saved output in results/ and every figure in figures/ from the
# run logs in logs/. CPU only. Run from anywhere: bash reproduce.sh
set -euo pipefail
cd "$(dirname "$0")"
export SOURCE_DATE_EPOCH=0    # fixed PDF timestamps, so figures/ regenerates byte for byte
PYTHON=${PYTHON:-python}
mkdir -p results figures

for name in canonical_numbers fixed_effects primary_analysis supporting_analyses \
            validity_controls equivalence_bounds robustness_and_epochs \
            repair_analysis repair_targeting repair_allocation public_classifiers; do
    echo "analysis/$name.py  ->  results/$name.txt"
    "$PYTHON" "analysis/$name.py" > "results/$name.txt"
done

for script in analysis/fig*.py; do
    echo "$script"
    "$PYTHON" "$script" > /dev/null
done
echo "Done. git diff results/ should show no changes."
