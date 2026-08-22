#!/bin/bash

#SBATCH --job-name=prewarm_face_parsing_cache_smoke_test
#SBATCH --output=/home/u6ga/sk3925.u6ga/sk3925_project/msc-project/slurm_jobs/output/prewarm_face_parsing_cache_smoke_test_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --mem=32G
#SBATCH --time=00:10:00

# Small-scale rehearsal of prewarm_face_parsing_cache.sh's CPU-only srun
# shape, scaled down to 8 tasks and scoped to the smallest 2D dataset (ffhq,
# image-only - no video decode) with a short time limit, so the new
# hash-bucket sharding + bucket-container writes (utils/cache_utils.py) can be
# verified quickly and cheaply before committing to the full multi-hour,
# 72-task rerun. Also the first real (non-synthetic) rehearsal of the
# quota-exhaustion fix this whole restructuring exists for.

PROJECT_DIR=/home/u6ga/sk3925.u6ga/sk3925_project/msc-project
DATALOADER_CONFIG=dataset_processing/config/dataloader.yaml
NUM_SHARDS=8

cd "$PROJECT_DIR"

srun --ntasks="$NUM_SHARDS" bash -c '
  .venv/bin/python scripts/prewarm_face_parsing_cache.py \
    --dataloader_config '"$DATALOADER_CONFIG"' \
    --dataset ffhq --split all --device cpu \
    --num_shards '"$NUM_SHARDS"' --shard_index $SLURM_PROCID
'

echo "Bash script done!"
