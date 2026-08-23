#!/bin/bash

#SBATCH --job-name=stage2_passC_smoothness
#SBATCH --output=/home/u6ga/sk3925.u6ga/sk3925_project/msc-project/slurm_jobs/output/stage2_passC_smoothness_%j.out
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00

# Shared launcher for both smoothness-boost experiments
# (stage2_passC_occlusion_subset_smoothness_10k.yaml and _15k.yaml) - same
# job/GPU footprint as every other Pass C experiment this session, but the
# config path is a script ARGUMENT rather than hardcoded, so one script
# covers both runs instead of two near-duplicate .sh files. The %j-suffixed
# output filename (job id, not config name) is what distinguishes the two
# runs' logs after submission.
#
# Usage:
#   sbatch stage2_passC_occlusion_subset_smoothness.sh training/config/stage2_passC_occlusion_subset_smoothness_10k.yaml
#   sbatch stage2_passC_occlusion_subset_smoothness.sh training/config/stage2_passC_occlusion_subset_smoothness_15k.yaml

if [ -z "$1" ]; then
  echo "Usage: sbatch $0 <config_path>" >&2
  exit 1
fi

PROJECT_DIR=/home/u6ga/sk3925.u6ga/sk3925_project/msc-project
CONFIG="$1"
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

