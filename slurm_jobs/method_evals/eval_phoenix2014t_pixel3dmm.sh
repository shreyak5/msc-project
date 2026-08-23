#!/bin/bash

#SBATCH --job-name=eval_phoenix2014t_pixel3dmm
#SBATCH --output=/home/u6ga/sk3925.u6ga/sk3925_project/msc-project/slurm_jobs/output/eval_phoenix2014t_pixel3dmm_%j.out
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00

# Pixel3DMM is ~60-100x slower per clip than smirk/ours_* (~8 min/clip vs seconds), so
# the full test set needs more shards than eval_phoenix2014t.sh's single 4-GPU node -
# topology copied from slurm_jobs/prewarm_mica_cache.sh (proven in production: see its
# slurm_jobs/output/prewarm_mica_cache_5656216.out, 16 shards completed cleanly, no
# CUDA/device errors). --gres=gpu:4 must be passed to srun itself, not just #SBATCH - see
# slurm_jobs/pretrain.sh's comment on a real prior failure from omitting it.
# --time=12:00:00 (not 24h like the other two eval_*_pixel3dmm.sh scripts): phoenix2014t
# is the smallest test set (642 clips) - at 8 shards the ~8.3min/clip average estimates
# ~11h, comfortable under 12h without requesting the full 24h grant unnecessarily.
#
# run_evaluation_dataset.py resumes automatically if this script is resubmitted after a
# timeout: it detects each shard's already-written results.csv and skips those clips
# rather than redoing them, so a partial run's progress isn't lost.

PROJECT_DIR=/home/u6ga/sk3925.u6ga/sk3925_project/msc-project
OUTPUT_DIR=evaluation/output_dataset_pixel3dmm/phoenix2014t
NUM_SHARDS=8

cd "$PROJECT_DIR"

srun --ntasks="$NUM_SHARDS" --gres=gpu:4 bash -c '
  export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
  .venv/bin/python evaluation/run_evaluation_dataset.py \
    --input_dir /projects/u6ga/sk3925_datasets/sign_datasets/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px/test \
    --image_seq --fps 25 \
    --method pixel3dmm --crop_size 512 \
    --output_dir "'"$OUTPUT_DIR"'" \
    --dataset_name phoenix2014t \
    --num_shards $SLURM_NTASKS --shard_index $SLURM_PROCID
'

.venv/bin/python evaluation/merge_dataset_shards.py --output_dir "$OUTPUT_DIR" --num_shards "$NUM_SHARDS" --method pixel3dmm

