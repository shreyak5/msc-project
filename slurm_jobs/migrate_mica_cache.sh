#!/bin/bash

#SBATCH --job-name=migrate_mica_cache
#SBATCH --output=/home/u6kf/sk3925.u6kf/sk3925-project/msc-project/slurm_jobs/output/migrate_mica_cache_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --ntasks-per-node=64
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# CPU-only, pure file I/O: repackages mica_cache's existing one-file-per-frame
# layout (partial coverage - only 2D datasets, and only whatever the earlier
# quota-limited prewarm runs actually completed, so this is a smaller job than
# crop_cache's) into the new bucket-container format
# (scripts/migrate_cache_to_buckets.py) - no MICA model, nothing recomputed,
# just read+validate+repack+delete on existing files. No --gres: this job
# never touches a GPU. Same shape as migrate_crop_cache.sh - see that script's
# comment for the full reasoning on node/task/time sizing; mica_cache has
# less data than crop_cache (real sampled bucket counts put it at roughly a
# fifth of crop_cache's total files), so the 2h ceiling has ample margin.

PROJECT_DIR=/home/u6kf/sk3925.u6kf/sk3925-project/msc-project
DATALOADER_CONFIG=dataset_processing/config/dataloader.yaml
NUM_SHARDS=64

cd "$PROJECT_DIR"

MICA_CACHE_ROOT=$(.venv/bin/python -c "
import yaml
print(yaml.safe_load(open('$DATALOADER_CONFIG'))['mica_cache_root'])
")

srun --ntasks="$NUM_SHARDS" bash -c '
  .venv/bin/python scripts/migrate_cache_to_buckets.py \
    --cache_root '"$MICA_CACHE_ROOT"' \
    --extension npy \
    --num_shards '"$NUM_SHARDS"' --shard_index $SLURM_PROCID
'

echo "Bash script done!"
