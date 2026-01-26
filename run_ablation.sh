#!/bin/bash
CONFIG=${1:-immuncan}
mkdir -p logs

SEEDS=(42 52 62 72 82)

for SEED in "${SEEDS[@]}"; do

CUDA_VISIBLE_DEVICES=0 python -u -m biobatchnet.train \
  --config "$CONFIG" --seed "$SEED" --run_name baseline \
  > "logs/baseline_seed${SEED}.log" 2>&1 &
echo "baseline pid=$!"

CUDA_VISIBLE_DEVICES=1 python -u -m biobatchnet.train \
  --config "$CONFIG" --seed "$SEED" --run_name no_discriminator --loss.discriminator 0 \
  > "logs/no_discriminator_seed${SEED}.log" 2>&1 &
echo "no_discriminator pid=$!"

CUDA_VISIBLE_DEVICES=2 python -u -m biobatchnet.train \
  --config "$CONFIG" --seed "$SEED" --run_name no_classifier --loss.classifier 0 \
  > "logs/no_classifier_seed${SEED}.log" 2>&1 &
echo "no_classifier pid=$!"

CUDA_VISIBLE_DEVICES=3 python -u -m biobatchnet.train \
  --config "$CONFIG" --seed "$SEED" --run_name no_ortho --loss.ortho 0 \
  > "logs/no_ortho_seed${SEED}.log" 2>&1 &
echo "no_ortho pid=$!"

wait

# Run "nobatch" as a separate round (after the 4 ablations finish)
CUDA_VISIBLE_DEVICES=0 python -u -m biobatchnet.train_nobatch \
  --config "$CONFIG" --seed "$SEED" --run_name nobatch \
  > "logs/nobatch_seed${SEED}.log" 2>&1

echo "Seed ${SEED} done!"

done

echo "All ablation experiments done!"
