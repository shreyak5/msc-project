#!/bin/bash

#SBATCH --job-name=prewarm_landmark_cache_smoke_test
#SBATCH --output=/home/u6ga/sk3925.u6ga/sk3925_project/msc-project/slurm_jobs/output/prewarm_landmark_cache_smoke_test_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:4
#SBATCH --mem=32G
#SBATCH --time=00:10:00

# Small-scale rehearsal of prewarm_landmark_cache.sh's srun/gres shape (tasks
# share GPUs via CUDA_VISIBLE_DEVICES=$((SLURM_LOCALID % 4))), scaled down to
# 8 tasks over 4 GPUs (2-way sharing, same pattern as the real job's 72-over-4
# just smaller) and scoped to the smallest 2D dataset (ffhq, image-only - no
# video decode) with a short time limit, so the srun --gres fix and the
# script's own up-front CUDA-availability guard can be verified quickly and
# cheaply before committing to the full multi-hour rerun.

PROJECT_DIR=/home/u6ga/sk3925.u6ga/sk3925_project/msc-project
DATALOADER_CONFIG=dataset_processing/config/dataloader.yaml
NUM_SHARDS=8

cd "$PROJECT_DIR"

srun --ntasks="$NUM_SHARDS" --gres=gpu:4 bash -c '
  export CUDA_VISIBLE_DEVICES=$((SLURM_LOCALID % 4))
  .venv/bin/python scripts/prewarm_landmark_cache.py \
    --dataloader_config '"$DATALOADER_CONFIG"' \
    --dataset ffhq --split all --device cuda \
    --num_shards '"$NUM_SHARDS"' --shard_index $SLURM_PROCID
'

echo "Bash script done!"
