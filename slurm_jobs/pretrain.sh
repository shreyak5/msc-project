#!/bin/bash

#SBATCH --job-name=pretrain
#SBATCH --output=/home/u6ga/sk3925.u6ga/sk3925_project/msc-project/slurm_jobs/output/pretrain_%j.out
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00

# One srun task per NODE (not per GPU): srun (called with no --ntasks override
# below) inherits --ntasks=4/--ntasks-per-node=1 from this job's allocation, so
# it launches exactly 4 processes, one per node. Each of those runs torchrun
# --nproc_per_node=4 itself, which is what spawns the 4 per-GPU worker
# processes on that node - a different topology from the prewarm jobs'
# --ntasks-per-node=4, where srun launched one independent process per GPU
# directly, with no torchrun layer.
#
# --gres=gpu:4 on srun itself (not just #SBATCH above) is required: on this
# cluster, GRES from the job's own allocation does not automatically propagate
# to an srun step - each step must request its own share. Confirmed via sacct
# after a real failed run: the job's overall AllocTRES showed gres/gpu=16 (all
# 4 nodes x 4 GPUs, correctly reserved), but the srun step itself only got
# gres/gpu=1 total without this flag, leaving most of torchrun's spawned ranks
# with no (or the wrong) GPU visible - "ProcessGroupNCCL ... no GPUs found" /
# "CUDA error: invalid device ordinal". The prewarm jobs never hit this because
# their one-task-per-GPU topology (--ntasks-per-node=4) happens to line up with
# Slurm's default per-task GRES round-robin; this job's one-task-per-node
# topology does not.

PROJECT_DIR=/home/u6ga/sk3925.u6ga/sk3925_project/msc-project
CONFIG=training/config/pretrain.yaml
GPUS_PER_NODE=4
MASTER_PORT=29500

cd "$PROJECT_DIR"

export WANDB_MODE=offline

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

srun --gres=gpu:4 bash -c '
  .venv/bin/torchrun \
    --nnodes='"$SLURM_NNODES"' \
    --nproc_per_node='"$GPUS_PER_NODE"' \
    --rdzv_id='"$SLURM_JOB_ID"' \
    --rdzv_backend=c10d \
    --rdzv_endpoint='"$MASTER_ADDR:$MASTER_PORT"' \
    -m training.pretrain --config '"$CONFIG"'
'

echo "DONE!"
