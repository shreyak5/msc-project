#!/bin/bash

#SBATCH --job-name=stage2_AB
#SBATCH --output=/home/u6ga/sk3925.u6ga/sk3925_project/msc-project/slurm_jobs/output/stage2_AB_%j.out
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=07:00:00

# Same topology as slurm_jobs/pretrain.sh: one srun task per NODE (not per
# GPU) - srun (no --ntasks override below) inherits --ntasks=4/
# --ntasks-per-node=1 from this job's allocation, launching exactly 4
# processes, one per node. Each runs torchrun --nproc_per_node=4 itself,
# which spawns the 4 per-GPU worker processes on that node.
#
# --gres=gpu:4 on srun itself is required (see pretrain.sh's own comment for
# the full explanation): GRES from the job's #SBATCH allocation does not
# automatically propagate to an srun step on this cluster - confirmed via a
# real failed pretrain.sh run, where sacct showed the job's overall
# AllocTRES had gres/gpu=16 but the srun step itself only got gres/gpu=1
# without this flag.

PROJECT_DIR=/home/u6ga/sk3925.u6ga/sk3925_project/msc-project
CONFIG=training/config/stage2_AB.yaml
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
    -m training.stage2 --config '"$CONFIG"'
'

echo "DONE!"
