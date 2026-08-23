#!/bin/bash

#SBATCH --job-name=stage2_passC_occlusion_subset
#SBATCH --output=/home/u6ga/sk3925.u6ga/sk3925_project/msc-project/slurm_jobs/output/stage2_passC_occlusion_subset_%j.out
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00

# Exact copy of stage2_passC_id_pooling_gate_reshape.sh, except CONFIG points at
# stage2_passC_occlusion_subset.yaml, which adds pass_c_occlusion_subset_index_dir
# on top of that same identity-pooling + gate-reshape run - see that yaml's own
# header comment. Requires scripts/build_occlusion_index.py to have already been
# run (populates dataset_processing/occlusion_clip_index/).

PROJECT_DIR=/home/u6ga/sk3925.u6ga/sk3925_project/msc-project
CONFIG=training/config/stage2_passC_occlusion_subset.yaml
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

