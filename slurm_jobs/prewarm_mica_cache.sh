#!/bin/bash

#SBATCH --job-name=prewarm_mica_cache
#SBATCH --output=/home/u6kf/sk3925.u6kf/sk3925-project/msc-project/slurm_jobs/output/prewarm_mica_cache_%j.out
#SBATCH --nodes=4
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00

PROJECT_DIR=/home/u6kf/sk3925.u6kf/sk3925-project/msc-project
DATALOADER_CONFIG=dataset_processing/config/dataloader.yaml
NUM_SHARDS=16

cd "$PROJECT_DIR"

srun --ntasks="$NUM_SHARDS" bash -c '
  export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
  .venv/bin/python scripts/prewarm_mica_cache.py \
    --dataloader_config '"$DATALOADER_CONFIG"' \
    --dataset all --split all --device cuda \
    --num_shards '"$NUM_SHARDS"' --shard_index $SLURM_PROCID
'

MICA_CACHE_ROOT=$(.venv/bin/python -c "
import yaml
print(yaml.safe_load(open('$DATALOADER_CONFIG'))['mica_cache_root'])
")
.venv/bin/python scripts/merge_prewarm_logs.py --mica_cache_root "$MICA_CACHE_ROOT" --num_shards "$NUM_SHARDS"

echo "DONE!"
