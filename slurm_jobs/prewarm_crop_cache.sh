#!/bin/bash

#SBATCH --job-name=prewarm_crop_cache
#SBATCH --output=/home/u6ga/sk3925.u6ga/sk3925_project/msc-project/slurm_jobs/output/prewarm_crop_cache_%j.out
#SBATCH --nodes=4
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00

# CPU-only: the only model this job runs is the RetinaFace detector, which has
# no GPU work worth reserving a GPU for here - see dataloader.yaml's
# detector.device default (cpu) for the same choice at training time. Not
# requesting --gres means these tasks still land on this cluster's ordinary
# (GPU-equipped) nodes, but leave the GPUs free for other jobs to use.

PROJECT_DIR=/home/u6ga/sk3925.u6ga/sk3925_project/msc-project
DATALOADER_CONFIG=dataset_processing/config/dataloader.yaml
NUM_SHARDS=16

cd "$PROJECT_DIR"

srun --ntasks="$NUM_SHARDS" bash -c '
  .venv/bin/python scripts/prewarm_crop_cache.py \
    --dataloader_config '"$DATALOADER_CONFIG"' \
    --dataset all --split all --device cpu \
    --num_shards '"$NUM_SHARDS"' --shard_index $SLURM_PROCID
'

CROP_CACHE_ROOT=$(.venv/bin/python -c "
import yaml
print(yaml.safe_load(open('$DATALOADER_CONFIG'))['crop_cache_root'])
")
.venv/bin/python scripts/merge_prewarm_logs.py --crop_cache_root "$CROP_CACHE_ROOT" --num_shards "$NUM_SHARDS"

echo "Bash script done!"
