#!/bin/bash

#SBATCH --job-name=stage2_passC_occlusion_exp
#SBATCH --output=/home/u6ga/sk3925.u6ga/sk3925_project/msc-project/slurm_jobs/output/stage2_passC_occlusion_exp_%j.out
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00

# Same topology as stage2.sh (4 nodes x 4 GPUs = 16 GPUs) - occlusion-
# experiment1.md's Pass-C-only ablation, run at the same scale as other real
# (non-smoke) Stage 2 runs. Time limit trimmed to 2h from stage2.sh's 7h:
# num_steps here is 10000 (training/config/stage2_passC_occlusion_exp.yaml),
# a tenth of stage2.sh's 100000, and only Pass C (a single pass per step, no
# A/B alternation) runs each step - adjust if real wall-clock/step turns out
# higher than expected.
#
# --gres=gpu:4 on srun itself is required (see pretrain.sh's own comment for
# the full explanation): GRES from the job's #SBATCH allocation does not
# automatically propagate to an srun step on this cluster.

PROJECT_DIR=/home/u6ga/sk3925.u6ga/sk3925_project/msc-project
CONFIG=training/config/stage2_passC_occlusion_exp.yaml
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
