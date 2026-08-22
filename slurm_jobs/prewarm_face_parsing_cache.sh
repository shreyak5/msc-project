#!/bin/bash

#SBATCH --job-name=prewarm_face_parsing_cache
#SBATCH --output=/home/u6ga/sk3925.u6ga/sk3925_project/msc-project/slurm_jobs/output/prewarm_face_parsing_cache_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --ntasks-per-node=72
#SBATCH --mem=128G
#SBATCH --time=24:00:00

# CPU-only: XSeg's parsing step is CPU-only regardless of GPU count (no
# onnxruntime-gpu wheel for this cluster's aarch64 nodes, see dataloader.yaml's
# xseg_device comment), and the RetinaFace detector is not worth reserving a
# GPU for either - see dataloader.yaml's detector.device default (cpu) for the
# same choice at training time. Not requesting --gres means these tasks still
# land on this cluster's ordinary (GPU-equipped) nodes, but leave the GPUs
# free for other jobs to use.
#
# 72 tasks, not 4: the node has 288 CPUs but this job only ever had cgroup
# access to 1 per task before. Now that neither model touches a GPU, 72 tasks
# just gives the CPU-bound detection + XSeg work more parallelism (1 core each
# out of 288), unconstrained by any GPU count.

PROJECT_DIR=/home/u6ga/sk3925.u6ga/sk3925_project/msc-project
DATALOADER_CONFIG=dataset_processing/config/dataloader.yaml
NUM_SHARDS=72

cd "$PROJECT_DIR"

srun --ntasks="$NUM_SHARDS" bash -c '
  .venv/bin/python scripts/prewarm_face_parsing_cache.py \
    --dataloader_config '"$DATALOADER_CONFIG"' \
    --dataset all --split all --device cpu \
    --num_shards '"$NUM_SHARDS"' --shard_index $SLURM_PROCID
'

FACE_PARSING_CACHE_ROOT=$(.venv/bin/python -c "
import yaml
print(yaml.safe_load(open('$DATALOADER_CONFIG'))['face_parsing_cache_root'])
")
.venv/bin/python scripts/merge_prewarm_logs.py --face_parsing_cache_root "$FACE_PARSING_CACHE_ROOT" --num_shards "$NUM_SHARDS"

echo "Bash script done!"
