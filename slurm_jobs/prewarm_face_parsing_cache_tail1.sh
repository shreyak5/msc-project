#!/bin/bash

#SBATCH --job-name=prewarm_face_parsing_cache_tail1
#SBATCH --output=/home/u6kf/sk3925.u6kf/sk3925-project/msc-project/slurm_jobs/output/prewarm_face_parsing_cache_tail1_%j.out
#SBATCH --nodes=4
#SBATCH --ntasks=576
#SBATCH --ntasks-per-node=144
#SBATCH --mem=288G
#SBATCH --time=24:00:00

# One of 3 parallel jobs (tail1/tail2/tail3) covering the datasets the main
# prewarm_face_parsing_cache.sh job hasn't reached yet - as of submission it
# was still working through mead (5th of 13, in datasets.yaml's fixed
# processing order: celeba, ffhq, bupt_cbface12, afew_va, mead, dad_3dheads,
# coma, vocaset, famos, headspace, csl_daily, phoenix2014t, how2sign). The 8
# remaining datasets are round-robin balanced across the 3 tail jobs (one big
# video dataset each - how2sign/phoenix2014t/csl_daily - so all 3 jobs finish
# around the same time instead of one job carrying all the expensive work).
#
# Boosted from the original 1 node x 72 tasks: that figure was inherited from
# the old GPU-sharing design (18 tasks/GPU x 4 GPUs), which no longer applies
# now that this job is CPU-only - there's no reason to stay capped at 72.
# 4 nodes x 144 tasks/node = 576 total tasks, ~8x the original parallelism,
# while staying well under each node's 460GB RAM (144 tasks x ~2GB/task
# observed usage =~ 288GB, real margin left - packing all 288 cores/node
# instead would risk the memory ceiling).
#
# Safe to run concurrently with the main job (and, if it's still running
# under the old 72-shard config, with itself under a different NUM_SHARDS):
# cache writes are bucketed per dataset (bucket_container_path includes
# entry.name), so jobs on different datasets never touch the same bucket
# file. Sharding is hash-based on sample_id (shard_of), not positional
# striping - stable per dataset regardless of which other datasets a given
# invocation covers, and race-safe across different NUM_SHARDS values too:
# utils/cache_utils.py's write_bucket_entries takes a flock and re-reads the
# container fresh before merging, so even if two differently-sharded runs
# both touch the same bucket, it's just occasionally redundant compute,
# never data loss. The dataset-suffixed log filename (shard_{i}_{dataset}.csv)
# also avoids any log-file collision.

PROJECT_DIR=/home/u6kf/sk3925.u6kf/sk3925-project/msc-project
DATALOADER_CONFIG=dataset_processing/config/dataloader.yaml
NUM_SHARDS=576

cd "$PROJECT_DIR"

for DATASET in how2sign headspace coma; do
  srun --nodes=4 --ntasks-per-node=144 --ntasks="$NUM_SHARDS" bash -c '
    .venv/bin/python scripts/prewarm_face_parsing_cache.py \
      --dataloader_config '"$DATALOADER_CONFIG"' \
      --dataset '"$DATASET"' --split all --device cpu \
      --num_shards '"$NUM_SHARDS"' --shard_index $SLURM_PROCID
  '
done

FACE_PARSING_CACHE_ROOT=$(.venv/bin/python -c "
import yaml
print(yaml.safe_load(open('$DATALOADER_CONFIG'))['face_parsing_cache_root'])
")
.venv/bin/python scripts/merge_prewarm_logs.py --face_parsing_cache_root "$FACE_PARSING_CACHE_ROOT" --num_shards "$NUM_SHARDS"

echo "DONE!"

# 5789213
