#!/bin/bash

#SBATCH --job-name=prewarm_face_parsing_cache_tail2
#SBATCH --output=/home/u6ga/sk3925.u6ga/sk3925_project/msc-project/slurm_jobs/output/prewarm_face_parsing_cache_tail2_%j.out
#SBATCH --nodes=4
#SBATCH --ntasks=576
#SBATCH --ntasks-per-node=144
#SBATCH --mem=288G
#SBATCH --time=05:00:00

# One of 3 parallel jobs (tail1/tail2/tail3) covering the datasets the main
# prewarm_face_parsing_cache.sh job hasn't reached yet - see tail1.sh's
# comment for the full reasoning (round-robin dataset balance, boosted
# 4-node/576-task compute, bucket-safe concurrency across different
# NUM_SHARDS values, per-dataset log filenames).

PROJECT_DIR=/home/u6ga/sk3925.u6ga/sk3925_project/msc-project
DATALOADER_CONFIG=dataset_processing/config/dataloader.yaml
NUM_SHARDS=576

cd "$PROJECT_DIR"

for DATASET in phoenix2014t famos dad_3dheads; do
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

# 5789214
