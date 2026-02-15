#!/bin/bash
export BIOBATCHNET_RUN_TS=$(date +%Y%m%d_%H%M%S)

KL_BIO_VALUES=(1e-4 1e-3 5e-3)
DISC_VALUES=(0.05 0.1 0.3 0.5)

for kl in "${KL_BIO_VALUES[@]}"; do
    for disc in "${DISC_VALUES[@]}"; do
        echo "Running kl_bio=$kl discriminator=$disc"
        python -m src.train \
            --config immucan \
            --loss.kl_bio "$kl" \
            --loss.discriminator "$disc" \
            --run_name "kl_bio${kl}_discriminator${disc}" \
            --seed 42
    done
done

echo "All 16 runs completed."
