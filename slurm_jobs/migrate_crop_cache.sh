#!/bin/bash

#SBATCH --job-name=migrate_crop_cache
#SBATCH --output=/home/u6kf/sk3925.u6kf/sk3925-project/msc-project/slurm_jobs/output/migrate_crop_cache_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --ntasks-per-node=64
#SBATCH --mem=32G
#SBATCH --time=05:00:00

# CPU-only, pure file I/O: repackages crop_cache's already-fully-prewarmed
# one-file-per-frame layout into the new bucket-container format
# (scripts/migrate_cache_to_buckets.py) - no detector, no model, nothing
# recomputed, just read+validate+repack+delete on ~14.5M existing files. No
# --gres: this job never touches a GPU. Single node, 64 tasks (matching
# prewarm_face_parsing_cache.sh's precedent of running many CPU-only tasks
# on one node, given nodes here have up to 288 CPUs) - keeps the same total
# parallelism as a 4-node/16-tasks-per-node split without needing multiple
# nodes; this workload is small-file/metadata-bound rather than
# network-bandwidth-bound, so consolidating onto one node shouldn't
# meaningfully change Lustre-side throughput. 5h is a generous ceiling for a
# repackaging pass at this file count (~226K files/task, comfortably under
# budget even at a conservative ~25ms/file); safely resumable if it were
# ever to run out of time - a bucket already migrated on a rerun simply
# won't show up as work again (its old per-frame directory no longer exists).

PROJECT_DIR=/home/u6kf/sk3925.u6kf/sk3925-project/msc-project
DATALOADER_CONFIG=dataset_processing/config/dataloader.yaml
NUM_SHARDS=64

cd "$PROJECT_DIR"

CROP_CACHE_ROOT=$(.venv/bin/python -c "
import yaml
print(yaml.safe_load(open('$DATALOADER_CONFIG'))['crop_cache_root'])
")

srun --ntasks="$NUM_SHARDS" bash -c '
  .venv/bin/python scripts/migrate_cache_to_buckets.py \
    --cache_root '"$CROP_CACHE_ROOT"' \
    --extension png \
    --num_shards '"$NUM_SHARDS"' --shard_index $SLURM_PROCID
'

echo "Bash script done!"
