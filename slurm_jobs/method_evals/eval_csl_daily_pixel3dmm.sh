#!/bin/bash

#SBATCH --job-name=eval_csl_daily_pixel3dmm
#SBATCH --output=/home/u6ga/sk3925.u6ga/sk3925_project/msc-project/slurm_jobs/output/eval_csl_daily_pixel3dmm_%j.out
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00

# Pixel3DMM is ~60-100x slower per clip than smirk/ours_* (~8 min/clip vs seconds), so
# the full test set needs many more shards than eval_csl_daily.sh's single 4-GPU node to
# fit under this cluster's 24h wall-time cap - topology copied from
# slurm_jobs/prewarm_mica_cache.sh (proven in production: see its
# slurm_jobs/output/prewarm_mica_cache_5656216.out, 16 shards completed cleanly, no
# CUDA/device errors). --gres=gpu:4 must be passed to srun itself, not just #SBATCH - see
# slurm_jobs/pretrain.sh's comment on a real prior failure from omitting it.
#
# run_evaluation_dataset.py resumes automatically if this script is resubmitted after a
# timeout: it detects each shard's already-written results.csv and skips those clips
# rather than redoing them, so a partial run's progress isn't lost.

PROJECT_DIR=/home/u6ga/sk3925.u6ga/sk3925_project/msc-project
OUTPUT_DIR=evaluation/output_dataset_pixel3dmm/csl_daily
NUM_SHARDS=8

cd "$PROJECT_DIR"

srun --ntasks="$NUM_SHARDS" --gres=gpu:4 bash -c '
  export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
  .venv/bin/python evaluation/run_evaluation_dataset.py \
    --input_dir /projects/u6ga/sk3925_datasets/sign_datasets/csl-daily/test \
    --image_seq --fps 30 \
    --method pixel3dmm --crop_size 512 \
    --output_dir "'"$OUTPUT_DIR"'" \
    --dataset_name csl_daily \
    --num_shards $SLURM_NTASKS --shard_index $SLURM_PROCID
'

.venv/bin/python evaluation/merge_dataset_shards.py --output_dir "$OUTPUT_DIR" --num_shards "$NUM_SHARDS" --method pixel3dmm

