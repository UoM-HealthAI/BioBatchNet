#!/bin/bash
CONFIG=${1:-pancreas}
LOG_DIR="logs/${CONFIG}"
mkdir -p "$LOG_DIR"

export BIOBATCHNET_RUN_TS=$(date +%Y%m%d_%H%M%S)
echo "Run timestamp: $BIOBATCHNET_RUN_TS"

SEEDS=(42 52 62 72 82)

for SEED in "${SEEDS[@]}"; do

echo "=== Running seed=$SEED ==="

CUDA_VISIBLE_DEVICES=0 python -u -m src.train \
  --config "$CONFIG" --seed "$SEED" --run_name baseline \
  --devices 1 --bs 128 \
  > "${LOG_DIR}/baseline_seed${SEED}.log" 2>&1 &
echo "baseline seed=$SEED pid=$!"

CUDA_VISIBLE_DEVICES=1 python -u -m src.train \
  --config "$CONFIG" --seed "$SEED" --run_name no_discriminator --loss.discriminator 0 \
  --devices 1 --bs 128 \
  > "${LOG_DIR}/no_discriminator_seed${SEED}.log" 2>&1 &
echo "no_discriminator seed=$SEED pid=$!"

CUDA_VISIBLE_DEVICES=2 python -u -m src.train \
  --config "$CONFIG" --seed "$SEED" --run_name no_classifier --loss.classifier 0 \
  --devices 1 --bs 128 \
  > "${LOG_DIR}/no_classifier_seed${SEED}.log" 2>&1 &
echo "no_classifier seed=$SEED pid=$!"

CUDA_VISIBLE_DEVICES=3 python -u -m src.train \
  --config "$CONFIG" --seed "$SEED" --run_name no_ortho --loss.ortho 0 \
  --devices 1 --bs 128 \
  > "${LOG_DIR}/no_ortho_seed${SEED}.log" 2>&1 &
echo "no_ortho seed=$SEED pid=$!"

wait
echo "Seed $SEED done!"

done

echo "All ablation experiments done!"
