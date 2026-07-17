#!/bin/bash

#SBATCH --job-name=stage2
#SBATCH --output=/home/u6kf/sk3925.u6kf/sk3925-project/msc-project/slurm_jobs/output/stage2_%j.out
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00

# Same topology as slurm_jobs/pretrain.sh: one srun task per NODE (not per
# GPU) - srun (no --ntasks override below) inherits --ntasks=4/
# --ntasks-per-node=1 from this job's allocation, launching exactly 4
# processes, one per node. Each runs torchrun --nproc_per_node=4 itself,
# which spawns the 4 per-GPU worker processes on that node.

PROJECT_DIR=/home/u6kf/sk3925.u6kf/sk3925-project/msc-project
CONFIG=training/config/stage2.yaml
GPUS_PER_NODE=4
MASTER_PORT=29500

cd "$PROJECT_DIR"

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

srun bash -c '
  .venv/bin/torchrun \
    --nnodes='"$SLURM_NNODES"' \
    --nproc_per_node='"$GPUS_PER_NODE"' \
    --rdzv_id='"$SLURM_JOB_ID"' \
    --rdzv_backend=c10d \
    --rdzv_endpoint='"$MASTER_ADDR:$MASTER_PORT"' \
    -m training.stage2 --config '"$CONFIG"'
'

echo "DONE!"
