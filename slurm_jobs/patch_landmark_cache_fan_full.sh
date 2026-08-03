#!/bin/bash

#SBATCH --job-name=patch_landmark_cache_fan_full
#SBATCH --output=/home/u6kf/sk3925.u6kf/sk3925-project/msc-project/slurm_jobs/output/patch_landmark_cache_fan_full_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00

# Single node, 8-way sharing of 1 GPU (same reasoning as
# prewarm_landmark_cache.sh's per-GPU sharing, scaled down): this only covers
# the DEV split of 3 datasets (~3300 clips total: how2sign 1739, csl_daily
# 1077, phoenix2014t 519 dev clips - see training/eval_loaders.py), not a full
# multi-dataset train-split prewarm, so it doesn't need that job's 4-node/
# 72-shard scale.

PROJECT_DIR=/home/u6kf/sk3925.u6kf/sk3925-project/msc-project
DATALOADER_CONFIG=dataset_processing/config/dataloader.yaml
NUM_SHARDS=8

cd "$PROJECT_DIR"

srun --ntasks="$NUM_SHARDS" --gres=gpu:1 bash -c '
  export CUDA_VISIBLE_DEVICES=0
  .venv/bin/python scripts/patch_landmark_cache_fan_full.py \
    --dataloader_config '"$DATALOADER_CONFIG"' \
    --dataset how2sign,phoenix2014t,csl_daily --split dev --device cuda \
    --num_shards '"$NUM_SHARDS"' --shard_index $SLURM_PROCID
'

echo "Bash script done!"
