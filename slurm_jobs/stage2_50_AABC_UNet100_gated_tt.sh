#!/bin/bash

#SBATCH --job-name=stage2_50_AABC_UNet100_gated_tt
#SBATCH --output=/home/u6ga/sk3925.u6ga/sk3925_project/msc-project/slurm_jobs/output/stage2_50_AABC_UNet100_gated_tt_%j.out
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=10:00:00

# Identical launch mechanics to stage2_50_AB_unet100.sh (see its own comments
# for the full nodes/GRES-propagation reasoning) - only the config differs,
# pointing at stage2_50_AABC_UNet100_gated_tt.yaml (pass_pattern extended to
# [A, A, B, C] with gated TT, seeded from stage2_50_AAB_UNet100_v3's
# checkpoint with a fresh optimizer).

PROJECT_DIR=/home/u6ga/sk3925.u6ga/sk3925_project/msc-project
CONFIG=training/config/stage2_50_AABC_UNet100_gated_tt.yaml
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
