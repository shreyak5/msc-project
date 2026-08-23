#!/bin/bash

#SBATCH --job-name=pretrain_50_smoke_test
#SBATCH --output=/home/u6ga/sk3925.u6ga/sk3925_project/msc-project/slurm_jobs/output/pretrain_50_smoke_test_%j.out
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --time=00:15:00

# Identical launch mechanics to pretrain_smoke_test.sh (see its own comments for
# the full reasoning on topology/GRES) - only the config differs, pointing at
# pretrain_50_smoke.yaml (num_expression_params=50) instead of pretrain_smoke.yaml,
# so this also exercises the configurable-expression-dim code path itself before
# the real pretrain_50.sh job is submitted.

PROJECT_DIR=/home/u6ga/sk3925.u6ga/sk3925_project/msc-project
CONFIG=training/config/pretrain_50_smoke.yaml
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
