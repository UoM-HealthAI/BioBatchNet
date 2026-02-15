#!/bin/bash

DATASETS=(pancreas macaque lung mousebrain)
OUTFILE="results/time_rna.json"

mkdir -p results

echo "{" > "$OUTFILE"

first=true
for dataset in "${DATASETS[@]}"; do
    echo "=== $dataset ==="

    SECONDS=0
    python -m src.train --config "$dataset" --run_name "time_test" --trainer.epochs 1 --seed 42
    T1=$SECONDS

    SECONDS=0
    python -m src.train --config "$dataset" --run_name "time_test" --trainer.epochs 3 --seed 42
    T3=$SECONDS

    per_epoch=$(( (T3 - T1) / 2 ))
    total=$(( per_epoch * 50 ))
    total_min=$(awk "BEGIN {printf \"%.1f\", $total / 60}")

    echo "$dataset: ~${per_epoch}s/epoch, 50 epochs ≈ ${total}s (~${total_min}min)"

    if $first; then
        first=false
    else
        echo "," >> "$OUTFILE"
    fi
    printf '  "%s": {"per_epoch_sec": %d, "total_50ep_sec": %d, "total_50ep_min": %s}' \
        "$dataset" "$per_epoch" "$total" "$total_min" >> "$OUTFILE"
done

echo "" >> "$OUTFILE"
echo "}" >> "$OUTFILE"

echo ""
echo "Results written to $OUTFILE"
cat "$OUTFILE"
