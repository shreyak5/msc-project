#!/bin/bash

#SBATCH --job-name=eval_csl_daily_ours_noTT
#SBATCH --output=/home/u6ga/sk3925.u6ga/sk3925_project/msc-project/slurm_jobs/output/eval_csl_daily_ours_noTT_%j.out
#SBATCH --nodes=1
#SBATCH --gpus=4
# Bumped from 6h: evaluate_clip now writes to the crop/GT-landmark caches
# (evaluation/gt_cache.py) if this run finds them cold, on top of the same
# live detection it always did - extra margin for that write overhead.
#SBATCH --time=09:00:00

PROJECT_DIR=/home/u6ga/sk3925.u6ga/sk3925_project/msc-project
NUM_SHARDS=4
OUTPUT_DIR=evaluation/output_dataset_ours_noTT/csl_daily

cd "$PROJECT_DIR"

for ((i = 0; i < NUM_SHARDS; i++)); do
  CUDA_VISIBLE_DEVICES=$i .venv/bin/python evaluation/run_evaluation_dataset.py \
    --input_dir /projects/u6ga/sk3925_datasets/sign_datasets/csl-daily/test \
    --image_seq --fps 30 \
    --output_dir "$OUTPUT_DIR" \
    --method ours_no_temporal \
    --checkpoint /projects/u6ga/sk3925_misc/checkpoints/stage2_50_AAB_UNet100_v3/step_00025999.pt \
    --dataset_name csl_daily \
    --num_shards "$NUM_SHARDS" --shard_index $i &
done
wait

.venv/bin/python evaluation/merge_dataset_shards.py --output_dir "$OUTPUT_DIR" --num_shards "$NUM_SHARDS" --method ours_no_temporal
