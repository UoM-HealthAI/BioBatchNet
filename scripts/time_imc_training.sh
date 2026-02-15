#!/bin/bash

DATASETS=(damond hoch immucan)

for dataset in "${DATASETS[@]}"; do
    echo "=== $dataset ==="

    SECONDS=0
    python -m src.train --config "$dataset" --run_name "time_test" --trainer.epochs 1 --seed 42
    T1=$SECONDS

    SECONDS=0
    python -m src.train --config "$dataset" --run_name "time_test" --trainer.epochs 3 --seed 42
    T3=$SECONDS

    per_epoch=$(( (T3 - T1) / 2 ))
    total=$(( per_epoch * 30 ))

    echo "$dataset: ~${per_epoch}s/epoch, 30 epochs ≈ ${total}s (~$((total/60))min)"
done
