#!/bin/bash

#SBATCH --job-name=prewarm_landmark_cache
#SBATCH --output=/home/u6ga/sk3925.u6ga/sk3925_project/msc-project/slurm_jobs/output/prewarm_landmark_cache_%j.out
#SBATCH --nodes=4
#SBATCH --ntasks=72
#SBATCH --ntasks-per-node=18
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=24:00:00

# 72 tasks total, not 4: the node has 288 CPUs but this job only ever had
# cgroup access to 1 per task. Detection + landmark inference are lightweight
# GPU work (fine to share 18-way per GPU), but video-frame decoding (OpenCV/
# ffmpeg) is CPU-bound and was the real bottleneck at 4/288 cores.
#
# 4 nodes x 18 tasks/node x 1 GPU/node (still 18-way sharing per GPU, same
# ratio as before, and still 4 GPUs total across the job - not asking for
# more GPU than before), not 1 node x 72 tasks x 4 GPUs: each task
# independently imports torch, initializes its own CUDA context, and loads
# its own FAN + MediaPipe models - measured at ~360MB of host RAM just for
# those imports, before any CUDA context or model weights, which are real
# additional per-process cost on top. Cramming 72 such processes onto one
# node's shared RAM pool OOM'd this job in 3.5 minutes even at --mem=128G
# (~1.78GB/task, single node). --mem is per-node in SLURM, not split across
# --nodes - so keeping --mem=128G with --nodes=4 gives each node its own full
# 128G (512G total across the job), now shared by only 18 tasks/node instead
# of 72, i.e. ~7.1GB/task, on 4 separate physical RAM pools rather than one.

PROJECT_DIR=/home/u6ga/sk3925.u6ga/sk3925_project/msc-project
DATALOADER_CONFIG=dataset_processing/config/dataloader.yaml
NUM_SHARDS=72

cd "$PROJECT_DIR"

srun --nodes=4 --ntasks-per-node=18 --ntasks="$NUM_SHARDS" --gres=gpu:1 bash -c '
  export CUDA_VISIBLE_DEVICES=0
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

echo "Bash script done!"
