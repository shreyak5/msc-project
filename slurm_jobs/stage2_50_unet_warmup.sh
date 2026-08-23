#!/bin/bash

#SBATCH --job-name=stage2_50_unet_warmup
#SBATCH --output=/home/u6ga/sk3925.u6ga/sk3925_project/msc-project/slurm_jobs/output/stage2_50_unet_warmup_%j.out
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00

# Identical launch mechanics to stage2_unet_warmup.sh (see its own comments for
# the full nodes/GRES-propagation reasoning) - only the config differs, pointing
# at stage2_50_unet_warmup.yaml (num_expression_params=50, seeded from
# pretrain_50's checkpoint) instead of the 100-dim stage2_unet_warmup.yaml.
# --time scaled down from stage2_unet_warmup.sh's 5:00:00/100000-steps ratio
# for this run's shorter 8000-step budget, with margin for startup/rendezvous
# and periodic eval overhead.

PROJECT_DIR=/home/u6ga/sk3925.u6ga/sk3925_project/msc-project
CONFIG=training/config/stage2_50_unet_warmup.yaml
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
