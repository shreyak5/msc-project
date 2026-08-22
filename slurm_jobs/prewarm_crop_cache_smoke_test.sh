#!/bin/bash

#SBATCH --job-name=prewarm_crop_cache_smoke_test
#SBATCH --output=/home/u6ga/sk3925.u6ga/sk3925_project/msc-project/slurm_jobs/output/prewarm_crop_cache_smoke_test_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --mem=32G
#SBATCH --time=00:10:00

# Small-scale rehearsal of prewarm_crop_cache.sh's CPU-only srun shape, scaled
# to a single node (4 tasks) and scoped to the smallest 2D dataset (ffhq,
# image-only - no video decode) with a short time limit, so the new
# hash-bucket sharding + bucket-container writes (utils/cache_utils.py) can be
# verified quickly and cheaply before committing to the full multi-hour,
# 16-task rerun.

PROJECT_DIR=/home/u6ga/sk3925.u6ga/sk3925_project/msc-project
DATALOADER_CONFIG=dataset_processing/config/dataloader.yaml
NUM_SHARDS=4

cd "$PROJECT_DIR"

srun --ntasks="$NUM_SHARDS" bash -c '
  .venv/bin/python scripts/prewarm_crop_cache.py \
    --dataloader_config '"$DATALOADER_CONFIG"' \
    --dataset ffhq --split all --device cpu \
    --num_shards '"$NUM_SHARDS"' --shard_index $SLURM_PROCID
'

echo "Bash script done!"
