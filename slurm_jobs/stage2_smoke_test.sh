#!/bin/bash

#SBATCH --job-name=stage2_smoke_test
#SBATCH --output=/home/u6kf/sk3925.u6kf/sk3925-project/msc-project/slurm_jobs/output/stage2_smoke_test_%j.out
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --time=00:15:00

# Cheap, fast-turnaround validation of the exact same multi-node SLURM/torchrun/
# DDP launch mechanics as stage2.sh, without needing the full 4-node/16-GPU
# allocation - see slurm_jobs/pretrain_smoke_test.sh's own comment for the full
# reasoning behind the 2-node/2-GPU-per-node topology (minimum needed to
# exercise both cross-node NCCL rendezvous and the CUDA_VISIBLE_DEVICES-per-
# rank fix, which only ever broke local_rank!=0 processes).
#
# Additionally exercises Stage 2's own added complexity beyond Stage 1: the
# three-pass round-robin scheduler, its two independently-cycling loaders
# (frame_pool for Pass A/B, clip for Pass C), DDP wrapping of four modules
# (svit/heads/unet/tt) with find_unused_parameters=True, and the one-time
# Stage-1-checkpoint seed load - training/config/stage2_smoke.yaml's
# num_steps=6 gives exactly 2 full cycles of pass_pattern [A, B, C].
#
# Same --gres=gpu:2 requirement on srun itself as pretrain_smoke_test.sh (GRES
# doesn't auto-propagate from the job's #SBATCH allocation down to an srun
# step on this cluster).

PROJECT_DIR=/home/u6kf/sk3925.u6kf/sk3925-project/msc-project
CONFIG=training/config/stage2_smoke.yaml
GPUS_PER_NODE=2
MASTER_PORT=29500

cd "$PROJECT_DIR"

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

srun --gres=gpu:2 bash -c '
  .venv/bin/torchrun \
    --nnodes='"$SLURM_NNODES"' \
    --nproc_per_node='"$GPUS_PER_NODE"' \
    --rdzv_id='"$SLURM_JOB_ID"' \
    --rdzv_backend=c10d \
    --rdzv_endpoint='"$MASTER_ADDR:$MASTER_PORT"' \
    -m training.stage2 --config '"$CONFIG"'
'

echo "Bash script done!"
