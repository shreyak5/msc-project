#!/bin/bash

#SBATCH --job-name=stage2_passC_occlusion_exp_simple_tt
#SBATCH --output=/home/u6ga/sk3925.u6ga/sk3925_project/msc-project/slurm_jobs/output/stage2_passC_occlusion_exp_simple_tt_%j.out
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00

# Exact copy of stage2_passC_occlusion_exp.sh, except CONFIG points at
# stage2_passC_occlusion_exp_simple_tt.yaml, which swaps in
# SimpleTemporalTransformer (model/temporal.py) via tt_variant: simple instead
# of the original TemporalTransformer - see that yaml's own header comment for
# what does/doesn't carry over (svit/heads/unet still warm-start from stage2_A;
# tt itself starts fresh since its architecture changed).

PROJECT_DIR=/home/u6ga/sk3925.u6ga/sk3925_project/msc-project
CONFIG=training/config/stage2_passC_occlusion_exp_simple_tt.yaml
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

