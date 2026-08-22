#!/bin/bash

#SBATCH --job-name=pretrain_smoke_test
#SBATCH --output=/home/u6ga/sk3925.u6ga/sk3925_project/msc-project/slurm_jobs/output/pretrain_smoke_test_%j.out
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --time=00:15:00

# Cheap, fast-turnaround validation of the exact same multi-node SLURM/torchrun/
# DDP launch mechanics as pretrain.sh, without needing the full 4-node/16-GPU
# allocation (which can sit in queue for hours) just to find out in the first 2
# minutes whether the launch mechanics themselves are broken.
#
# 2 nodes x 2 GPUs (not 1 GPU/node) is the minimum topology that actually
# exercises the bugs this was built to catch: 2 nodes are needed to prove
# cross-node NCCL rendezvous works at all, and 2 GPUs per node are needed so
# both local_rank=0 AND local_rank=1 get exercised on each node - a 1-GPU/node
# test would never re-trigger the CUDA_VISIBLE_DEVICES-per-rank bug (training/
# distributed.py's setup_distributed), since every rank would trivially be
# local_rank=0 and the bug only ever broke local_rank!=0 processes.
#
# Same --gres=gpu:2 requirement on srun itself as pretrain.sh (see its own
# comment for the full reasoning - GRES doesn't auto-propagate from the job's
# #SBATCH allocation down to an srun step on this cluster).
#
# Points at training/config/pretrain_smoke.yaml (num_steps=5, log_interval_
# steps=1) instead of the real pretrain.yaml, so a successful run prints all 5
# steps and exits in well under a minute once actually scheduled - nothing to
# babysit.

PROJECT_DIR=/home/u6ga/sk3925.u6ga/sk3925_project/msc-project
CONFIG=training/config/pretrain_smoke.yaml
GPUS_PER_NODE=2
MASTER_PORT=29500

cd "$PROJECT_DIR"

export WANDB_MODE=offline

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

srun --gres=gpu:2 bash -c '
  .venv/bin/torchrun \
    --nnodes='"$SLURM_NNODES"' \
    --nproc_per_node='"$GPUS_PER_NODE"' \
    --rdzv_id='"$SLURM_JOB_ID"' \
    --rdzv_backend=c10d \
    --rdzv_endpoint='"$MASTER_ADDR:$MASTER_PORT"' \
    -m training.pretrain --config '"$CONFIG"'
'

echo "DONE!"
