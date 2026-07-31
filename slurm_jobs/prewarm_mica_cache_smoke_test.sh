#!/bin/bash

#SBATCH --job-name=prewarm_mica_cache_smoke_test
#SBATCH --output=/home/u6kf/sk3925.u6kf/sk3925-project/msc-project/slurm_jobs/output/prewarm_mica_cache_smoke_test_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --mem=32G
#SBATCH --time=00:10:00

# Small-scale rehearsal of prewarm_mica_cache.sh's srun/gres shape (one task
# per GPU via CUDA_VISIBLE_DEVICES=$SLURM_LOCALID), scaled to a single node
# (4 tasks, 4 GPUs) and scoped to the smallest 2D dataset (ffhq, image-only -
# no video decode) with a short time limit, so the srun --gres fix and the
# script's own up-front CUDA-availability guard can be verified quickly and
# cheaply before committing to the full multi-hour, 16-GPU rerun.

PROJECT_DIR=/home/u6kf/sk3925.u6kf/sk3925-project/msc-project
DATALOADER_CONFIG=dataset_processing/config/dataloader.yaml
NUM_SHARDS=4

cd "$PROJECT_DIR"

srun --ntasks="$NUM_SHARDS" --gres=gpu:4 bash -c '
  export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
  .venv/bin/python scripts/prewarm_mica_cache.py \
    --dataloader_config '"$DATALOADER_CONFIG"' \
    --dataset ffhq --split all --device cuda \
    --num_shards '"$NUM_SHARDS"' --shard_index $SLURM_PROCID
'

echo "Bash script done!"
