#!/bin/bash

#SBATCH --job-name=prewarm_landmark_cache
#SBATCH --output=/home/u6kf/sk3925.u6kf/sk3925-project/msc-project/slurm_jobs/output/prewarm_landmark_cache_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --ntasks-per-node=72
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00

# 72 tasks, not 4: the node has 288 CPUs but this job only ever had cgroup
# access to 1 per task. Detection + landmark inference are lightweight GPU
# work (fine to share 18-way per GPU), but video-frame decoding (OpenCV/
# ffmpeg) is CPU-bound and was the real bottleneck at 4/288 cores.

PROJECT_DIR=/home/u6kf/sk3925.u6kf/sk3925-project/msc-project
DATALOADER_CONFIG=dataset_processing/config/dataloader.yaml
NUM_SHARDS=72

cd "$PROJECT_DIR"

srun --ntasks="$NUM_SHARDS" bash -c '
  export CUDA_VISIBLE_DEVICES=$((SLURM_LOCALID % 4))
  .venv/bin/python scripts/prewarm_landmark_cache.py \
    --dataloader_config '"$DATALOADER_CONFIG"' \
    --dataset all --split all --device cuda \
    --num_shards '"$NUM_SHARDS"' --shard_index $SLURM_PROCID
'

LANDMARK_CACHE_ROOT=$(.venv/bin/python -c "
import yaml
print(yaml.safe_load(open('$DATALOADER_CONFIG'))['landmark_cache_root'])
")
.venv/bin/python scripts/merge_prewarm_logs.py --landmark_cache_root "$LANDMARK_CACHE_ROOT" --num_shards "$NUM_SHARDS"

echo "DONE!"
