#!/usr/bin/env bash
set -e

LOSS="weighted_mse"
LOSS_TAG="weighted"

MODELS=(
  "flat_linear"
  "flat_mlp"
  "grouped_parallel"
  "grouped_sequential"
)

SEEDS=(
  "123"
  "456"
)

for seed in "${SEEDS[@]}"; do
  for model in "${MODELS[@]}"; do

    head_run="${model}_head_${LOSS_TAG}_seed${seed}"
    ft_run="${model}_ft_${LOSS_TAG}_seed${seed}"

    echo "============================================================"
    echo "Head training: ${head_run}"
    echo "============================================================"

    python -m src.train \
      --run "${head_run}" \
      --model_type "${model}" \
      --loss "${LOSS}" \
      --stage head \
      --seed "${seed}"

    echo "============================================================"
    echo "Fine-tuning: ${ft_run}"
    echo "============================================================"

    python -m src.train \
      --run "${ft_run}" \
      --model_type "${model}" \
      --loss "${LOSS}" \
      --stage finetune \
      --init_checkpoint "/workspace/data/runs/${head_run}/best_model.pt" \
      --seed "${seed}"

  done
done