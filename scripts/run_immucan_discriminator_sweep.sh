#!/bin/bash

# Discriminator strength sweep for IMMUcan dataset
# Values: 0.05, 0.1, 0.3

DISCRIMINATOR_VALUES=(0.05 0.1 0.3)

for disc in "${DISCRIMINATOR_VALUES[@]}"; do
    echo "Running with discriminator=$disc"
    python -m src.train \
        --config immucan \
        --loss.discriminator "$disc" \
        --run_name "disc_${disc}" \
        --seed 42
done

echo "All runs completed."
