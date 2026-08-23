#!/bin/bash

#SBATCH --job-name=eval_phoenix2014t_emica
#SBATCH --output=/home/u6ga/sk3925.u6ga/sk3925_project/msc-project/slurm_jobs/output/eval_phoenix2014t_emica_%j.out
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=04:00:00

# EMICA is a single feed-forward pass per clip (~48.9s/clip average measured during Phase
# B validation, including its own per-clip model-reload overhead), ~10x faster than
# Pixel3DMM's ~5000-iteration per-clip optimization - fits comfortably in the same
# single-node 4-GPU pattern eval_phoenix2014t.sh already uses for smirk/ours_no_temporal,
# no multi-node complexity needed (unlike
# slurm_jobs/method_evals/eval_phoenix2014t_pixel3dmm.sh).
#
# run_evaluation_dataset.py resumes automatically if this script is resubmitted after a
# timeout: it detects each shard's already-written results.csv and skips those clips
# rather than redoing them.

PROJECT_DIR=/home/u6ga/sk3925.u6ga/sk3925_project/msc-project
NUM_SHARDS=4
OUTPUT_DIR=evaluation/output_dataset_emica/phoenix2014t

cd "$PROJECT_DIR"

for ((i = 0; i < NUM_SHARDS; i++)); do
  CUDA_VISIBLE_DEVICES=$i .venv/bin/python evaluation/run_evaluation_dataset.py \
    --input_dir /projects/u6ga/sk3925_datasets/sign_datasets/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px/test \
    --image_seq --fps 25 \
    --method emica --crop_size 224 \
    --output_dir "$OUTPUT_DIR" \
    --dataset_name phoenix2014t \
    --num_shards "$NUM_SHARDS" --shard_index $i &
done
wait

.venv/bin/python evaluation/merge_dataset_shards.py --output_dir "$OUTPUT_DIR" --num_shards "$NUM_SHARDS" --method emica

