#!/bin/bash

#SBATCH --job-name=prewarm_face_parsing_cache
#SBATCH --output=/home/u6kf/sk3925.u6kf/sk3925-project/msc-project/slurm_jobs/output/prewarm_face_parsing_cache_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00

# Same scaled-down topology as prewarm_landmark_cache.sh - XSeg's parsing step
# is CPU-only regardless of GPU count (no onnxruntime-gpu wheel for this
# cluster's aarch64 nodes, see dataloader.yaml's xseg_device comment), so only
# the RetinaFace detection half of this job actually benefits from more GPUs.

PROJECT_DIR=/home/u6kf/sk3925.u6kf/sk3925-project/msc-project
DATALOADER_CONFIG=dataset_processing/config/dataloader.yaml
NUM_SHARDS=4

cd "$PROJECT_DIR"

srun --ntasks="$NUM_SHARDS" bash -c '
  export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
  .venv/bin/python scripts/prewarm_face_parsing_cache.py \
    --dataloader_config '"$DATALOADER_CONFIG"' \
    --dataset all --split all --device cuda \
    --num_shards '"$NUM_SHARDS"' --shard_index $SLURM_PROCID
'

FACE_PARSING_CACHE_ROOT=$(.venv/bin/python -c "
import yaml
print(yaml.safe_load(open('$DATALOADER_CONFIG'))['face_parsing_cache_root'])
")
.venv/bin/python scripts/merge_prewarm_logs.py --face_parsing_cache_root "$FACE_PARSING_CACHE_ROOT" --num_shards "$NUM_SHARDS"

echo "DONE!"
